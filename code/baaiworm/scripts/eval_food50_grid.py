"""Evaluate completed recovered parameter sets on a deterministic 50-food grid.

This is validation only: no optimization is performed. For every completed
phase2 experiment with recovered weights, it simulates the recovered circuit and
the normal unperturbed circuit under the same food location, then stores feature
deltas for downstream summaries.
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")

from recovery.evaluation.behavior_metrics import (  # noqa: E402
    compute_body_wave,
    compute_forward_speed,
    compute_muscle_alternation,
    detect_zigzag,
)
from recovery.simulation.closed_loop_simulator import ClosedLoopSimulator  # noqa: E402
from recovery.utils.io_utils import load_pickle, save_json  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("eval_food50")

PROJECT_ROOT = "/root/BAAIWorm-main"
BUILD_DIR = "/root/BAAIWorm-main/build_headless/build"
PHASE1_DIR = "/root/BAAIWorm-main/recovery/output/phase1"
PHASE2_DIR = "/root/BAAIWorm-main/recovery/output/phase2"
OUTPUT_DIR = "/root/BAAIWorm-main/recovery/output/eval_food50"

ABS_CIRCUIT_PATH = os.path.join(
    PROJECT_ROOT,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_abscircuit.pkl",
)
WOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_wout.pkl",
)


def food_grid(n_foods, food_seed=20260504):
    """Return deterministic food locations, with the canonical food first."""
    default = np.array([1.8275, -0.0276, -0.3082], dtype=np.float32)
    foods = [default]
    rng = np.random.default_rng(food_seed)
    anchors = [
        [1.5, -0.5, -0.8],
        [2.0, 0.3, 0.2],
        [1.0, -0.2, -1.5],
        [2.5, 0.0, 0.5],
        [1.25, 0.45, -1.1],
        [2.35, -0.55, 0.0],
    ]
    foods.extend(np.array(x, dtype=np.float32) for x in anchors)
    while len(foods) < n_foods:
        x = rng.uniform(0.9, 2.7)
        y = rng.uniform(-0.65, 0.55)
        z = rng.uniform(-1.6, 0.65)
        foods.append(np.array([x, y, z], dtype=np.float32))
    return foods[:n_foods]


def normalize_param_types(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    return str(value).replace(",", "+").split("+")


def discover_experiments(limit=0, include_errors=False, experiment_prefix="", experiment_contains=""):
    rows = []
    for result_path in sorted(glob.glob(os.path.join(PHASE2_DIR, "*", "result.pkl"))):
        exp_name = os.path.basename(os.path.dirname(result_path))
        if experiment_prefix and not exp_name.startswith(experiment_prefix):
            continue
        if experiment_contains and experiment_contains not in exp_name:
            continue
        try:
            result = load_pickle(result_path)
        except Exception as exc:
            logger.warning("Cannot load %s: %s", result_path, exc)
            continue
        if not result.get("recovered_weights") and not include_errors:
            continue
        param_types = normalize_param_types(result.get("param_type"))
        if not param_types:
            logger.warning("Skipping %s: missing param_type", exp_name)
            continue
        rows.append(
            {
                "experiment": exp_name,
                "result_path": result_path,
                "param_types": param_types,
                "source_loss": result.get("best_loss"),
                "method": result.get("method"),
            }
        )
    return rows[:limit] if limit else rows


def make_simulator(food, n_steps):
    return ClosedLoopSimulator(
        PROJECT_ROOT,
        BUILD_DIR,
        ABS_CIRCUIT_PATH,
        WOUT_PATH,
        n_init_steps=30,
        n_sim_steps=n_steps,
        food_location=np.array(food, dtype=np.float32),
    )


def corr_flat(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = min(len(a), len(b))
    if n < 2:
        return None
    a = a[:n]
    b = b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics_from_traj(traj, baseline_muscle=None):
    zz = detect_zigzag(traj["rel_x"], traj["rel_z"])
    speed = compute_forward_speed(traj["rel_x"], traj["rel_z"])
    wave = compute_body_wave(traj["rel_x"], traj["rel_z"])
    metrics = {
        "zigzag_score": float(zz.get("zigzag_score", 0.0)),
        "n_reversals": int(zz.get("n_reversals", 0)),
        "mean_speed": float(speed.get("mean_speed", 0.0)),
        "speed_std": float(speed.get("speed_std", 0.0)),
        "mean_amplitude": float(zz.get("mean_amplitude", 0.0)),
        "body_wave_freq": float(wave.get("dominant_freq", 0.0)),
    }
    muscle = traj.get("muscle_activation")
    if muscle is not None:
        muscle = np.asarray(muscle)
        alt = compute_muscle_alternation(muscle)
        metrics["dorsal_ventral_corr"] = float(alt.get("dorsal_ventral_corr", 0.0))
        if baseline_muscle is not None:
            metrics["muscle_signal_corr"] = corr_flat(muscle, baseline_muscle)
    return metrics


def load_or_run_baseline(food_idx, food, normal_weights, n_steps):
    os.makedirs(os.path.join(OUTPUT_DIR, "baseline"), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "baseline", f"food{food_idx:02d}.json")
    muscle_path = os.path.join(OUTPUT_DIR, "baseline", f"food{food_idx:02d}_muscle.npy")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", payload)
        muscle = np.load(muscle_path) if os.path.exists(muscle_path) else None
        return metrics, muscle

    sim = make_simulator(food, n_steps)
    traj = sim.run_with_custom_weights(normal_weights, ["syn"], n_steps=n_steps)
    muscle = traj.get("muscle_activation")
    metrics = metrics_from_traj(traj)
    payload = {"food_idx": food_idx, "food_location": food.tolist(), "metrics": metrics}
    save_json(payload, out_path)
    if muscle is not None:
        np.save(muscle_path, np.asarray(muscle))
    return metrics, np.asarray(muscle) if muscle is not None else None


def summarize_rows(rows):
    by_exp = {}
    for row in rows:
        if "error" in row:
            continue
        by_exp.setdefault(row["experiment"], []).append(row)
    summary = {}
    for exp, exp_rows in by_exp.items():
        item = {
            "n_foods": len(exp_rows),
            "param_types": exp_rows[0]["param_types"],
            "source_loss": exp_rows[0].get("source_loss"),
            "method": exp_rows[0].get("method"),
        }
        for key in [
            "delta_zigzag",
            "delta_speed",
            "delta_amplitude",
            "delta_body_wave_freq",
            "muscle_signal_corr",
        ]:
            vals = [r.get(key) for r in exp_rows if r.get(key) is not None]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                item[f"{key}_mean"] = float(np.mean(vals))
                item[f"{key}_std"] = float(np.std(vals))
        summary[exp] = item
    return summary


def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-foods", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--food-seed", type=int, default=20260504)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit-experiments", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--experiment-prefix", default="")
    parser.add_argument("--experiment-contains", default="")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    normal_weights = load_pickle(os.path.join(PHASE1_DIR, "normal_weights.pkl"))
    foods = food_grid(args.n_foods, food_seed=args.food_seed)
    experiments = discover_experiments(
        limit=args.limit_experiments,
        experiment_prefix=args.experiment_prefix,
        experiment_contains=args.experiment_contains,
    )

    tasks = []
    for exp in experiments:
        for food_idx, food in enumerate(foods):
            tasks.append((exp, food_idx, food))
    tasks = [
        task for i, task in enumerate(tasks)
        if i % args.n_shards == args.shard_id
    ]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    logger.info(
        "Food50 shard %d/%d: %d experiments, %d tasks",
        args.shard_id,
        args.n_shards,
        len(experiments),
        len(tasks),
    )

    rows = []
    for exp, food_idx, food in tasks:
        exp_dir = os.path.join(OUTPUT_DIR, exp["experiment"])
        os.makedirs(exp_dir, exist_ok=True)
        out_path = os.path.join(exp_dir, f"food{food_idx:02d}.json")
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                rows.append(json.load(f))
            continue

        t0 = time.time()
        try:
            baseline_metrics, baseline_muscle = load_or_run_baseline(
                food_idx, food, normal_weights, args.n_steps
            )
            result = load_pickle(exp["result_path"])
            sim = make_simulator(food, args.n_steps)
            traj = sim.run_with_custom_weights(
                result["recovered_weights"],
                exp["param_types"],
                n_steps=args.n_steps,
            )
            rec_metrics = metrics_from_traj(traj, baseline_muscle)
            row = {
                "experiment": exp["experiment"],
                "param_types": exp["param_types"],
                "method": exp.get("method"),
                "source_loss": exp.get("source_loss"),
                "food_idx": food_idx,
                "food_location": food.tolist(),
                "baseline": baseline_metrics,
                "recovered": rec_metrics,
                "delta_zigzag": rec_metrics["zigzag_score"] - baseline_metrics["zigzag_score"],
                "delta_speed": rec_metrics["mean_speed"] - baseline_metrics["mean_speed"],
                "delta_amplitude": rec_metrics["mean_amplitude"] - baseline_metrics["mean_amplitude"],
                "delta_body_wave_freq": rec_metrics["body_wave_freq"] - baseline_metrics["body_wave_freq"],
                "muscle_signal_corr": rec_metrics.get("muscle_signal_corr"),
                "time_seconds": time.time() - t0,
            }
        except Exception as exc:
            logger.exception("Failed %s food%d", exp["experiment"], food_idx)
            row = {
                "experiment": exp["experiment"],
                "param_types": exp["param_types"],
                "method": exp.get("method"),
                "source_loss": exp.get("source_loss"),
                "food_idx": food_idx,
                "food_location": food.tolist(),
                "error": str(exc),
                "time_seconds": time.time() - t0,
            }
        save_json(row, out_path)
        rows.append(row)

    shard_path = os.path.join(
        OUTPUT_DIR,
        f"rows_shard{args.shard_id:03d}_of_{args.n_shards:03d}.json",
    )
    save_json(rows, shard_path)

    all_rows = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*", "food*.json")):
        if os.path.basename(os.path.dirname(path)) == "baseline":
            continue
        try:
            with open(path, "r") as f:
                all_rows.append(json.load(f))
        except Exception:
            pass
    save_json(summarize_rows(all_rows), os.path.join(OUTPUT_DIR, "summary.json"))
    logger.info("Done shard %d/%d", args.shard_id, args.n_shards)


if __name__ == "__main__":
    main()
