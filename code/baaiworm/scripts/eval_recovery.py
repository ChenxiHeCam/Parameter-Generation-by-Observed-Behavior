"""评估恢复结果：用恢复权重跑仿真，检查行为和肌肉模式"""
import sys
import os
import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main")

from recovery.utils.io_utils import load_pickle
from recovery.evaluation.behavior_metrics import (
    detect_zigzag, compute_forward_speed, compute_muscle_alternation
)
from recovery.simulation.closed_loop_simulator import ClosedLoopSimulator

phase1_dir = "/root/BAAIWorm-main/recovery/output/phase1"
phase2_dir = "/root/BAAIWorm-main/recovery/output/phase2"
project_root = "/root/BAAIWorm-main"
build_dir = "/root/BAAIWorm-main/build_headless/build"
abs_circuit_path = os.path.join(
    project_root,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_abscircuit.pkl")
wout_path = os.path.join(
    project_root,
    "eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_wout.pkl")

# Baseline metrics
baseline = load_pickle(os.path.join(phase1_dir, "baseline", "traj_0.pkl"))
rx = baseline["rel_x"]
rz = baseline["rel_z"]
zz = detect_zigzag(rx, rz)
sp = compute_forward_speed(rx, rz)

print("=== BASELINE (normal) ===")
print("  zigzag_score: %.4f" % zz.get("zigzag_score", 0))
print("  n_reversals:  %d" % zz.get("n_reversals", 0))
print("  mean_speed:   %.5f" % sp["mean_speed"])
print("  mean_amplitude: %.2f" % zz.get("mean_amplitude", 0))

if baseline.get("muscle_activation") is not None:
    muscle = np.array(baseline["muscle_activation"])
    ma = compute_muscle_alternation(muscle)
    print("  dorsal_ventral_corr: %.4f" % ma["dorsal_ventral_corr"])
    print("  muscle_mean: %.4f  muscle_std: %.4f" % (np.mean(muscle), np.std(muscle)))

# Init simulator
simulator = ClosedLoopSimulator(
    project_root, build_dir, abs_circuit_path, wout_path,
    n_init_steps=30, n_sim_steps=100)

normal_weights = load_pickle(os.path.join(phase1_dir, "normal_weights.pkl"))

# Evaluate each recovered param type
for param in ["syn", "wout", "gj"]:
    print()
    print("=== %s RECOVERED ===" % param.upper())

    # Find best result
    final_path = os.path.join(phase2_dir, "%s_3stage" % param, "final_result.pkl")
    if not os.path.exists(final_path):
        print("  No result found")
        continue

    final = load_pickle(final_path)
    print("  best_loss: %.4f (s1=%.4f s2=%.4f s3=%.4f)" % (
        final["best_loss"], final["stage1_loss"],
        final["stage2_loss"], final["stage3_loss"]))
    print("  weight_error:", final.get("weight_error", {}))

    # Find the stage with best loss and load its recovered weights
    best_stage = "stage1"
    best_loss = final["stage1_loss"]
    for s in ["stage2", "stage3"]:
        if final[s + "_loss"] < best_loss:
            best_loss = final[s + "_loss"]
            best_stage = s

    stage_path = os.path.join(phase2_dir, "%s_3stage" % param, "%s_result.pkl" % best_stage)
    stage_result = load_pickle(stage_path)
    recovered_weights = stage_result["recovered_weights"]

    # Run simulation with recovered weights
    print("  Running simulation with recovered weights...")
    try:
        traj = simulator.run_with_custom_weights(
            recovered_weights, [param], n_steps=100)

        # Behavior metrics
        rx2 = traj["rel_x"]
        rz2 = traj["rel_z"]
        zz2 = detect_zigzag(rx2, rz2)
        sp2 = compute_forward_speed(rx2, rz2)

        print("  RECOVERED BEHAVIOR:")
        print("    zigzag_score: %.4f (baseline: %.4f)" % (
            zz2.get("zigzag_score", 0), zz.get("zigzag_score", 0)))
        print("    n_reversals:  %d (baseline: %d)" % (
            zz2.get("n_reversals", 0), zz.get("n_reversals", 0)))
        print("    mean_speed:   %.5f (baseline: %.5f)" % (
            sp2["mean_speed"], sp["mean_speed"]))
        print("    mean_amplitude: %.2f (baseline: %.2f)" % (
            zz2.get("mean_amplitude", 0), zz.get("mean_amplitude", 0)))

        # Muscle check (validation, not used in loss)
        if traj.get("muscle_activation") is not None:
            muscle2 = np.array(traj["muscle_activation"])
            ma2 = compute_muscle_alternation(muscle2)
            print("  MUSCLE VALIDATION (not in loss):")
            print("    dorsal_ventral_corr: %.4f (baseline: %.4f)" % (
                ma2["dorsal_ventral_corr"],
                ma["dorsal_ventral_corr"] if baseline.get("muscle_activation") is not None else 0))
            print("    muscle_mean: %.4f  muscle_std: %.4f" % (
                np.mean(muscle2), np.std(muscle2)))

            # Muscle signal correlation with baseline
            if baseline.get("muscle_activation") is not None:
                bm = np.array(baseline["muscle_activation"])
                T = min(len(bm), len(muscle2))
                muscle_corr = np.corrcoef(bm[:T].flatten(), muscle2[:T].flatten())[0, 1]
                print("    muscle_signal_corr with baseline: %.4f" % muscle_corr)

    except Exception as e:
        print("  Simulation failed: %s" % e)
        import traceback
        traceback.print_exc()

print()
print("=== DONE ===")
