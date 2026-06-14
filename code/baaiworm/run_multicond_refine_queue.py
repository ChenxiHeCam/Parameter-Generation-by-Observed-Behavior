"""Multi-condition refinement for recovered BAAIWorm parameters.

This queue continues from an existing recovered result, but evaluates each
candidate on several food locations and start/orientation conditions. It is
intended as the next-stage calibration target after single-food rescue.
"""
import argparse
import copy
import glob
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")

import run_extra_queue as base  # noqa: E402
from recovery.methods.gradient_descent.trajectory_loss import TrajectoryLoss  # noqa: E402
from recovery.methods.spsa_optimizer import SPSARecoverer  # noqa: E402
from recovery.simulation.closed_loop_simulator import ClosedLoopSimulator  # noqa: E402
from recovery.utils.io_utils import load_pickle, save_json, save_pickle  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("multicond_refine")

PROJECT_ROOT = "/root/BAAIWorm-main"
BUILD_DIR = "/root/BAAIWorm-main/build_headless/build"
PHASE1_DIR = "/root/BAAIWorm-main/recovery/output/phase1"
PHASE2_DIR = "/root/BAAIWorm-main/recovery/output/phase2"
BASELINE_DIR = "/root/BAAIWorm-main/recovery/output/multicond_baseline"

ABS_CIRCUIT_PATH = os.path.join(
    PROJECT_ROOT,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_abscircuit.pkl",
)
WOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_wout.pkl",
)

DEFAULT_START = np.array([0.43650287, -1.0677443, -1.7485859], dtype=np.float32)

SOURCES = [
    ("combo_syn_gj_spsa_scale25_seed1", ["syn", "gj"]),
    ("combo_ion_channels_passive_params_spsa_scale25_seed1", ["ion_channels", "passive_params"]),
    ("combo_gj_wout_spsa_scale25_seed1", ["gj", "wout"]),
    ("combo_gj_wout_spsa_scale30_seed0", ["gj", "wout"]),
    ("combo_gj_wout_spsa_scale40_seed0", ["gj", "wout"]),
    ("combo_syn_gj_wout_spsa_scale25_seed1", ["syn", "gj", "wout"]),
    ("combo_syn_wout_spsa_scale25_seed1", ["syn", "wout"]),
]


def food_grid(n_foods):
    default = np.array([1.8275, -0.0276, -0.3082], dtype=np.float32)
    foods = [default]
    anchors = [
        [1.5, -0.5, -0.8],
        [2.0, 0.3, 0.2],
        [1.0, -0.2, -1.5],
        [2.5, 0.0, 0.5],
        [1.25, 0.45, -1.1],
    ]
    foods.extend(np.array(x, dtype=np.float32) for x in anchors)
    return foods[:n_foods]


def start_grid(n_starts):
    starts = [
        (DEFAULT_START, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        (
            DEFAULT_START + np.array([0.20, 0.00, 0.10], dtype=np.float32),
            np.array([0.25, 0.0, 0.97], dtype=np.float32),
        ),
        (
            DEFAULT_START + np.array([-0.20, 0.00, -0.10], dtype=np.float32),
            np.array([-0.25, 0.0, 0.97], dtype=np.float32),
        ),
        (
            DEFAULT_START + np.array([0.00, 0.00, 0.25], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
    ]
    out = []
    for start, orient in starts[:n_starts]:
        norm = np.linalg.norm(orient)
        out.append((start.astype(np.float32), (orient / max(norm, 1e-8)).astype(np.float32)))
    return out


def make_conditions(n_foods, n_starts):
    conditions = []
    for food_idx, food in enumerate(food_grid(n_foods)):
        for start_idx, (start, orient) in enumerate(start_grid(n_starts)):
            conditions.append(
                {
                    "food_idx": food_idx,
                    "start_idx": start_idx,
                    "food": food,
                    "start": start,
                    "orientation": orient,
                }
            )
    return conditions


def make_simulator(cond, n_steps):
    return ClosedLoopSimulator(
        PROJECT_ROOT,
        BUILD_DIR,
        ABS_CIRCUIT_PATH,
        WOUT_PATH,
        n_init_steps=30,
        n_sim_steps=n_steps,
        food_location=cond["food"],
        start_location=[cond["start"]],
        orientation=[cond["orientation"]],
    )


def baseline_path(cond, n_steps):
    return os.path.join(
        BASELINE_DIR,
        f"food{cond['food_idx']:02d}_start{cond['start_idx']:02d}_steps{n_steps}.pkl",
    )


def load_or_run_baseline(cond, normal_weights, n_steps):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    path = baseline_path(cond, n_steps)
    if os.path.exists(path):
        return load_pickle(path)
    logger.info(
        "Running normal baseline food=%d start=%d",
        cond["food_idx"],
        cond["start_idx"],
    )
    sim = make_simulator(cond, n_steps)
    traj = sim.run_with_custom_weights(normal_weights, ["syn"], n_steps=n_steps)
    payload = {
        "condition": {
            "food_idx": cond["food_idx"],
            "start_idx": cond["start_idx"],
            "food": cond["food"].tolist(),
            "start": cond["start"].tolist(),
            "orientation": cond["orientation"].tolist(),
        },
        "trajectory": traj,
    }
    save_pickle(payload, path)
    return payload


class MultiConditionSPSARecoverer(SPSARecoverer):
    def __init__(
        self,
        target_data,
        perturbed_weight_dict,
        param_types,
        conditions,
        target_payloads,
        config=None,
    ):
        super().__init__(
            target_data=target_data,
            perturbed_weight_dict=perturbed_weight_dict,
            param_types=param_types,
            simulator=None,
            config=config or {},
        )
        self.conditions = conditions
        self.target_payloads = target_payloads
        self.loss_fns = [
            TrajectoryLoss(payload["trajectory"])
            for payload in target_payloads
        ]

    def _evaluate(self, w):
        weight_dict = self._decode_weights(w)
        losses = []
        for cond, loss_fn in zip(self.conditions, self.loss_fns):
            try:
                sim = make_simulator(cond, self.sim_steps)
                sim_result = sim.run_with_custom_weights(
                    weight_dict,
                    self.param_types,
                    n_steps=self.sim_steps,
                )
                loss, _ = loss_fn.compute(sim_result)
            except Exception as exc:
                logger.warning(
                    "Simulation failed food=%s start=%s: %s",
                    cond["food_idx"],
                    cond["start_idx"],
                    exc,
                )
                loss = 1e6
            losses.append(float(loss))
        return float(np.mean(losses))


def load_source(exp_name):
    path = os.path.join(PHASE2_DIR, exp_name, "result.pkl")
    if not os.path.exists(path):
        logger.warning("Missing source %s", path)
        return None
    result = load_pickle(path)
    weights = result.get("recovered_weights")
    if not weights:
        logger.warning("Source %s has no recovered_weights", exp_name)
        return None
    return result


def all_tasks(pool, n_foods, n_starts):
    tasks = []
    for source, param_types in SOURCES:
        if pool != "all" and pool not in source:
            continue
        src = load_source(source)
        if src is None:
            continue
        name = f"{source}_stage_mf{n_foods}s{n_starts}_spsa"
        tasks.append((source, src.get("best_loss"), param_types, src["recovered_weights"], name))
    return sorted(tasks, key=lambda x: x[-1])


def evaluate_final(weights, param_types, conditions, targets, n_steps):
    rows = []
    for cond, target in zip(conditions, targets):
        sim = make_simulator(cond, n_steps)
        traj = sim.run_with_custom_weights(weights, param_types, n_steps=n_steps)
        loss, details = TrajectoryLoss(target["trajectory"]).compute(traj)
        rows.append(
            {
                "food_idx": cond["food_idx"],
                "start_idx": cond["start_idx"],
                "loss": float(loss),
                "details": details,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="all")
    parser.add_argument("--n-foods", type=int, default=5)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    args = parser.parse_args()

    normal_weights = load_pickle(os.path.join(PHASE1_DIR, "normal_weights.pkl"))
    conditions = make_conditions(args.n_foods, args.n_starts)
    logger.info("Preparing %d multi-condition targets", len(conditions))
    targets = [
        load_or_run_baseline(cond, normal_weights, args.n_steps)
        for cond in conditions
    ]

    tasks = [
        task for i, task in enumerate(all_tasks(args.pool, args.n_foods, args.n_starts))
        if i % args.n_shards == args.shard_id
    ]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    logger.info("Multi-condition shard %d/%d: %d tasks", args.shard_id, args.n_shards, len(tasks))
    for source, source_loss, param_types, _, name in tasks:
        logger.info("  %s source_loss=%s params=%s -> %s", source, source_loss, "+".join(param_types), name)

    for source, source_loss, param_types, weights, name in tasks:
        out_dir = os.path.join(PHASE2_DIR, name)
        result_path = os.path.join(out_dir, "result.pkl")
        if os.path.exists(result_path):
            logger.info("%s already done; skipping", name)
            continue
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        rec = MultiConditionSPSARecoverer(
            target_data=targets[0]["trajectory"],
            perturbed_weight_dict=copy.deepcopy(weights),
            param_types=param_types,
            conditions=conditions,
            target_payloads=targets,
            config={
                "a0": 0.006,
                "c0": 0.025,
                "A": 20,
                "sim_steps": args.n_steps,
                "seed": 20260506 + args.shard_id,
            },
        )
        result = rec.optimize(n_iterations=args.iterations)
        result["time_seconds"] = time.time() - t0
        result["method"] = f"multicond_spsa_mf{args.n_foods}s{args.n_starts}"
        result["param_type"] = "+".join(param_types)
        result["stage_source"] = source
        result["stage_source_loss"] = source_loss
        result["n_foods"] = args.n_foods
        result["n_starts"] = args.n_starts
        result["n_conditions"] = len(conditions)
        result["n_steps"] = args.n_steps
        result["condition_summary"] = {
            "foods": [c["food"].tolist() for c in conditions if c["start_idx"] == 0],
            "starts": [s.tolist() for s, _ in start_grid(args.n_starts)],
        }
        if result.get("recovered_weights"):
            rows = evaluate_final(
                result["recovered_weights"],
                param_types,
                conditions,
                targets,
                args.n_steps,
            )
            result["multicond_eval_rows"] = rows
            result["multicond_eval_loss_mean"] = float(np.mean([r["loss"] for r in rows]))
            result["multicond_eval_loss_std"] = float(np.std([r["loss"] for r in rows]))
            result["eval_metrics"] = base.evaluate_recovered(result["recovered_weights"], param_types)
            try:
                result["weight_error"] = base.WeightRecoveryEvaluator.compute_weight_error(
                    result["recovered_weights"], base.normal_weights, param_types
                )
                result["weight_error"].update(base.param_corr(result["recovered_weights"], param_types))
            except Exception:
                logger.exception("Failed weight diagnostics")

        save_pickle(result, result_path)
        save_json(
            {k: v for k, v in result.items() if k not in ("recovered_weights", "loss_history")},
            os.path.join(out_dir, "result.json"),
        )
        logger.info(
            "%s done: best_loss=%.6f multicond_eval=%.6f",
            name,
            result.get("best_loss", 999),
            result.get("multicond_eval_loss_mean", 999),
        )


if __name__ == "__main__":
    main()
