"""SBC rank-statistic test for BAAIWorm 5d (scale-vector) PGOB recovery.

Reviewer #12: SBC in main text, both worms. modWorm SBC is the cheap high-N anchor;
this BAAIWorm shard validates the SAME calibration property in the heavy closed-loop
simulator at smaller N (each recovery runs the real BAAIWorm sim).

Generative model matches the claim-4 ablation harness: 5d multiplicative scale theta
on FULL5 weight groups (syn, gj, wout, ion_channels, passive_params). The "observed"
behaviour is the trajectory produced by clean SOTA weights scaled by theta_true.
Recovery = SPSA on 5d to match the observed trajectory (pure TrajectoryLoss; no
muscle/neural internal signal). Held-out cond for the calibration statistic.

Per draw:
  theta_true ~ prior (multiplicative jitter)
  observed traj  = sim(base_w scaled by theta_true) on TRAIN cond
  ensemble: L SPSA recoveries from independent inits
  rank = #{ensemble member held-out loss < theta_true held-out loss}
A calibrated procedure -> ranks Uniform{0..L}.
"""
from __future__ import annotations
import os, sys, json, copy, time, argparse, logging
import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")

import run_multicond_refine_queue as mcr
from save_trajectory_dataset import food_grid, start_grid
from recovery.methods.gradient_descent.trajectory_loss import TrajectoryLoss
from recovery.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("SBCbaai")

SOTA = "/root/BAAIWorm_clone_ready/BAAIWorm-main/recovery/output/phase2/full5_merge_sota_combo_ionpass_T8/result.pkl"
BASELINE_DIR = "/root/BAAIWorm-main/recovery/output/multicond_baseline"
FULL5 = ["syn", "gj", "wout", "ion_channels", "passive_params"]
N_STEPS = 100
PRIOR_SCALE = 0.20
# Train cond = food50; held-out eval cond = food51 (train/test split).
TRAIN_FOOD = 50
EVAL_FOOD = 51

# SPSA hyperparams (match ablation harness)
N_ITER = 10
INIT_PERTURB = 0.3
A_SPSA, C_SPSA, A_SPALL, ALPHA, GAMMA = 0.05, 0.05, 5, 0.602, 0.101


def scale_weights(base_w, theta5):
    w = copy.deepcopy(base_w)
    s_syn, s_gj, s_wout, s_ion, s_pas = theta5
    w["syn_weights"] = w["syn_weights"] * s_syn
    w["gj_weights"] = w["gj_weights"] * s_gj
    w["wout"] = w["wout"] * s_wout
    ic = w["ion_channels"]
    if isinstance(ic, dict):
        w["ion_channels"] = {k: (v * s_ion if isinstance(v, np.ndarray) else v) for k, v in ic.items()}
    pp = w["passive_params"]
    if isinstance(pp, dict):
        w["passive_params"] = {k: (v * s_pas if isinstance(v, np.ndarray) else v) for k, v in pp.items()}
    return w


def cond_for(food_idx):
    """Build a sim cond dict matching the claim-4 ablation harness exactly
    (food_idx/start_idx + unpacked start + orientation from the baseline pkl)."""
    foods = food_grid(95)
    starts = start_grid(8)
    fi = food_idx
    si = 0
    s, o = starts[si]
    cond = {"food_idx": fi, "start_idx": si, "food": foods[fi],
            "start": s, "orientation": o}
    return cond


def traj_of(base_w, theta5, cond):
    w = scale_weights(base_w, theta5)
    sim = mcr.make_simulator(cond, N_STEPS)
    return sim.run_with_custom_weights(w, FULL5, n_steps=N_STEPS)


def loss_to_target(base_w, theta5, target_traj, cond):
    try:
        tr = traj_of(base_w, theta5, cond)
        lfn = TrajectoryLoss(target_traj)
        tot, _ = lfn.compute(tr)
        return float(tot)
    except Exception as ex:
        log.warning("sim fail: %s", ex)
        return 1.0


def spsa_recover(base_w, target_traj, train_cond, seed):
    rng = np.random.RandomState(seed)
    theta = 1.0 + rng.uniform(-INIT_PERTURB, INIT_PERTURB, size=5)
    theta = np.maximum(theta, 0.01)
    best_theta, best_L = theta.copy(), float("inf")
    for k in range(N_ITER):
        a_k = A_SPSA / (k + 1 + A_SPALL) ** ALPHA
        c_k = C_SPSA / (k + 1) ** GAMMA
        delta = rng.choice([-1, 1], size=5).astype(np.float64)
        tp = np.maximum(theta + c_k * delta, 0.01)
        tm = np.maximum(theta - c_k * delta, 0.01)
        Lp = loss_to_target(base_w, tp, target_traj, train_cond)
        Lm = loss_to_target(base_w, tm, target_traj, train_cond)
        g = (Lp - Lm) / (2.0 * c_k * delta)
        gn = np.linalg.norm(g)
        if gn > 10.0:
            g = g * 10.0 / gn
        theta = np.maximum(theta - a_k * g, 0.01)
        for (th, L) in [(tp, Lp), (tm, Lm)]:
            if L < best_L:
                best_L, best_theta = L, th.copy()
    return best_theta, best_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    res = load_pickle(SOTA)
    base_w = res["recovered_weights"]
    train_cond = cond_for(TRAIN_FOOD)
    eval_cond = cond_for(EVAL_FOOD)

    rng = np.random.RandomState(90000 + args.draw)
    theta_true = np.maximum(1.0 + PRIOR_SCALE * rng.randn(5), 0.05)

    # observed behaviour (train + held-out eval) from theta_true
    observed_train = traj_of(base_w, theta_true, train_cond)
    observed_eval = traj_of(base_w, theta_true, eval_cond)

    # held-out statistic for ground truth: loss of theta_true vs its own eval traj is ~0,
    # so we instead measure each candidate's eval-cond loss against the OBSERVED eval traj.
    L_true_eval = loss_to_target(base_w, theta_true, observed_eval, eval_cond)

    ens_eval_losses, ens_thetas = [], []
    for j in range(args.ensemble):
        seed = 600000 + args.draw * 1000 + j
        th, _ = spsa_recover(base_w, observed_train, train_cond, seed)
        Lj = loss_to_target(base_w, th, observed_eval, eval_cond)
        ens_eval_losses.append(Lj)
        ens_thetas.append(th.tolist())

    rank = int(np.sum(np.array(ens_eval_losses) < L_true_eval))
    ens_arr = np.array(ens_thetas)
    per_param_rank = [int(np.sum(ens_arr[:, d] < theta_true[d])) for d in range(5)]

    out = {
        "sim": "BAAIWorm", "draw": args.draw, "ensemble": args.ensemble,
        "L_true_eval": L_true_eval, "ensemble_eval_losses": ens_eval_losses,
        "rank_loss": rank, "per_param_rank": per_param_rank,
        "theta_true": theta_true.tolist(),
        "elapsed_s": time.time() - t0,
        "config": {"prior_scale": PRIOR_SCALE, "n_iter": N_ITER,
                   "train_food": TRAIN_FOOD, "eval_food": EVAL_FOOD},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sbc-baai draw {args.draw}] rank={rank}/{args.ensemble} "
          f"L_true={L_true_eval:.4f} t={out['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
