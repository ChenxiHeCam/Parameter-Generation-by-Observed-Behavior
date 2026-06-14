"""Evaluate recovered weights across held-out food/start/orientation conditions.

This is a validation-only experiment: it does not optimize weights. It replays
completed phase2 recoveries under several closed-loop conditions and compares
behavior metrics against the normal circuit under the same condition.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")

from recovery.evaluation.behavior_metrics import (  # noqa: E402
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
logger = logging.getLogger("eval_conditions")

PROJECT_ROOT = "/root/BAAIWorm-main"
BUILD_DIR = "/root/BAAIWorm-main/build_headless/build"
PHASE1_DIR = "/root/BAAIWorm-main/recovery/output/phase1"
PHASE2_DIR = "/root/BAAIWorm-main/recovery/output/phase2"
OUTPUT_DIR = "/root/BAAIWorm-main/recovery/output/eval_conditions"

ABS_CIRCUIT_PATH = os.path.join(
    PROJECT_ROOT,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_abscircuit.pkl",
)
WOUT_PATH = os.path.join(
    PROJECT_ROOT,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_wout.pkl",
)


EXPERIMENTS = {
    "syn_es": ["syn"],
    "wout_es": ["wout"],
    "gj_es": ["gj"],
    "ion_channels_es": ["ion_channels"],
    "passive_params_es": ["passive_params"],
    "combo_syn_gj": ["syn", "gj"],
    "combo_syn_wout": ["syn", "wout"],
    "combo_syn_polarity": ["syn", "polarity"],
    "combo_syn_gj_wout": ["syn", "gj", "wout"],
    "combo_gj_wout": ["gj", "wout"],
    "combo_ion_passive": ["ion_channels", "passive_params"],
    "combo_4param": ["syn", "gj", "wout", "polarity"],
}


CONDITIONS = [
    {
        "name": "default",
        "food": [1.8275, -0.0276, -0.3082],
        "start": [0.43650287, -1.0677443, -1.7485859],
        "orientation": [0.0, 0.0, 1.0],
    },
    {
        "name": "food_left",
        "food": [1.5, -0.5, -0.8],
        "start": [0.43650287, -1.0677443, -1.7485859],
        "orientation": [0.0, 0.0, 1.0],
    },
    {
        "name": "food_right",
        "food": [2.0, 0.3, 0.2],
        "start": [0.43650287, -1.0677443, -1.7485859],
        "orientation": [0.0, 0.0, 1.0],
    },
    {
        "name": "start_shift",
        "food": [1.8275, -0.0276, -0.3082],
        "start": [0.15, -1.05, -1.55],
        "orientation": [0.0, 0.0, 1.0],
    },
    {
        "name": "yaw_left",
        "food": [1.8275, -0.0276, -0.3082],
        "start": [0.43650287, -1.0677443, -1.7485859],
        "orientation": [0.25, 0.0, 0.97],
    },
    {
        "name": "yaw_right",
        "food": [1.8275, -0.0276, -0.3082],
        "start": [0.43650287, -1.0677443, -1.7485859],
        "orientation": [-0.25, 0.0, 0.97],
    },
]


def arr(values):
    return np.array(values, dtype=np.float32)


def as_condition_kwargs(condition):
    orientation = arr(condition["orientation"])
    orientation = orientation / np.linalg.norm(orientation)
    return {
        "food_location": arr(condition["food"]),
        "start_location": [arr(condition["start"])],
        "orientation": [orientation],
    }


def metrics_from_traj(traj, baseline_muscle=None):
    zz = detect_zigzag(traj["rel_x"], traj["rel_z"])
    speed = compute_forward_speed(traj["rel_x"], traj["rel_z"])
    metrics = {
        "zigzag_score": float(zz.get("zigzag_score", 0.0)),
        "n_reversals": int(zz.get("n_reversals", 0)),
        "mean_speed": float(speed["mean_speed"]),
        "speed_std": float(speed.get("speed_std", 0.0)),
        "mean_amplitude": float(zz.get("mean_amplitude", 0.0)),
    }

    muscle = traj.get("muscle_activation")
    if muscle is not None:
        muscle = np.asarray(muscle)
        alt = compute_muscle_alternation(muscle)
        metrics["dorsal_ventral_corr"] = float(alt["dorsal_ventral_corr"])
        if baseline_muscle is not None:
            baseline_muscle = np.asarray(baseline_muscle)
            n = min(len(baseline_muscle), len(muscle))
            if n > 1:
                metrics["muscle_signal_corr"] = float(
                    np.corrcoef(
                        baseline_muscle[:n].flatten(),
                        muscle[:n].flatten(),
                    )[0, 1]
                )
    return metrics


def run_condition(weight_dict, param_types, condition, n_steps):
    simulator = ClosedLoopSimulator(
        PROJECT_ROOT,
        BUILD_DIR,
        ABS_CIRCUIT_PATH,
        WOUT_PATH,
        n_init_steps=30,
        n_sim_steps=n_steps,
        **as_condition_kwargs(condition),
    )
    return simulator.run_with_custom_weights(weight_dict, param_types, n_steps=n_steps)


def load_result(exp_name):
    result_path = os.path.join(PHASE2_DIR, exp_name, "result.pkl")
    if not os.path.exists(result_path):
        return None
    return load_pickle(result_path)


def summarize(values, key):
    xs = [v[key] for v in values if key in v and np.isfinite(v[key])]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    normal_weights = load_pickle(os.path.join(PHASE1_DIR, "normal_weights.pkl"))
    conditions = CONDITIONS[: args.limit] if args.limit else CONDITIONS

    baseline_cache = {}
    rows = []
    summary = {}

    for ci, condition in enumerate(conditions):
        logger.info("Baseline condition %s", condition["name"])
        t0 = time.time()
        traj = run_condition(normal_weights, ["syn"], condition, args.n_steps)
        baseline_cache[condition["name"]] = {
            "metrics": metrics_from_traj(traj),
            "muscle": traj.get("muscle_activation"),
            "time_seconds": time.time() - t0,
        }
        save_json(
            baseline_cache[condition["name"]]["metrics"],
            os.path.join(OUTPUT_DIR, f"baseline_{ci}_{condition['name']}.json"),
        )

    for exp_name, param_types in EXPERIMENTS.items():
        result = load_result(exp_name)
        if result is None or not result.get("recovered_weights"):
            logger.warning("Skipping %s: no recovered result", exp_name)
            continue

        exp_rows = []
        for ci, condition in enumerate(conditions):
            cname = condition["name"]
            out_path = os.path.join(OUTPUT_DIR, f"{exp_name}_{ci}_{cname}.json")
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    row = json.load(f)
                rows.append(row)
                exp_rows.append(row)
                continue

            logger.info("Evaluating %s on %s", exp_name, cname)
            t0 = time.time()
            try:
                traj = run_condition(
                    result["recovered_weights"], param_types, condition, args.n_steps
                )
                rec_metrics = metrics_from_traj(
                    traj, baseline_cache[cname].get("muscle")
                )
                base_metrics = baseline_cache[cname]["metrics"]
                row = {
                    "experiment": exp_name,
                    "param_types": param_types,
                    "source_loss": result.get("best_loss"),
                    "condition": cname,
                    "condition_spec": condition,
                    "baseline": base_metrics,
                    "recovered": rec_metrics,
                    "delta_zigzag": rec_metrics["zigzag_score"]
                    - base_metrics["zigzag_score"],
                    "delta_speed": rec_metrics["mean_speed"]
                    - base_metrics["mean_speed"],
                    "time_seconds": time.time() - t0,
                }
            except Exception as exc:
                logger.exception("Failed %s on %s", exp_name, cname)
                row = {
                    "experiment": exp_name,
                    "param_types": param_types,
                    "source_loss": result.get("best_loss"),
                    "condition": cname,
                    "condition_spec": condition,
                    "error": str(exc),
                    "time_seconds": time.time() - t0,
                }

            save_json(row, out_path)
            rows.append(row)
            exp_rows.append(row)

        valid = [r for r in exp_rows if "error" not in r]
        if valid:
            dz_mean, dz_std = summarize(valid, "delta_zigzag")
            ds_mean, ds_std = summarize(valid, "delta_speed")
            rec_zz_mean, rec_zz_std = summarize(
                [r["recovered"] for r in valid], "zigzag_score"
            )
            rec_sp_mean, rec_sp_std = summarize(
                [r["recovered"] for r in valid], "mean_speed"
            )
            summary[exp_name] = {
                "param_types": param_types,
                "source_loss": result.get("best_loss"),
                "n_conditions": len(valid),
                "delta_zigzag_mean": dz_mean,
                "delta_zigzag_std": dz_std,
                "delta_speed_mean": ds_mean,
                "delta_speed_std": ds_std,
                "recovered_zigzag_mean": rec_zz_mean,
                "recovered_zigzag_std": rec_zz_std,
                "recovered_speed_mean": rec_sp_mean,
                "recovered_speed_std": rec_sp_std,
            }
            logger.info(
                "%s summary: dzz=%.4f +/- %.4f, dsp=%.6f +/- %.6f",
                exp_name,
                dz_mean or 0.0,
                dz_std or 0.0,
                ds_mean or 0.0,
                ds_std or 0.0,
            )

    save_json(rows, os.path.join(OUTPUT_DIR, "condition_rows.json"))
    save_json(summary, os.path.join(OUTPUT_DIR, "summary.json"))
    logger.info("Done. Wrote %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
