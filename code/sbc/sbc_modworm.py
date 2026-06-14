"""SBC (Simulation-Based Calibration) rank-statistic test for modWorm 30d recovery.

Reviewer #12: elevate SBC to main text. Tests whether the PGOB recovery procedure
is well-calibrated: for many draws theta* ~ prior, simulate behaviour, recover an
ensemble of posterior samples, and compute the rank of the ground-truth theta*
within the recovered ensemble. If the inference is calibrated, ranks are uniform.

Pure-trajectory: loss is MSE on the rollout bend trajectory only (no muscle/neural
internal signal). Deterministic: every rollout uses a fixed seed.

Protocol (per draw i):
  1. theta_true ~ prior  (prior = lognormal-ish multiplicative jitter on THETA_GT)
  2. targets = rollout(theta_true, seed in TRAIN_SEEDS)   # the "observed" behaviour
  3. recover an ensemble of L posterior samples by running CMA-ES from L different
     init perturbations, taking the final best of each (held-out EVAL seeds for the
     calibration loss so fit is train/test split).
  4. rank = # of ensemble members whose held-out loss is < held-out loss(theta_true)
            (equivalently, where theta_true falls in the ordered ensemble by an
             unbiased posterior-predictive statistic). We use the standard SBC
             definition: rank of the prior-draw within the posterior ensemble under
             a 1-D test statistic (here: held-out trajectory loss to the observed).

A well-calibrated procedure yields ranks ~ Uniform{0..L}. We report the histogram,
chi-square uniformity p-value, and the ECDF max-deviation.

This shard runs ONE draw and writes a per-draw json so draws parallelise across
cores with xargs -P. Merge afterwards.
"""
from __future__ import annotations
import sys, json, time, argparse
import numpy as np

sys.path.insert(0, "/root")
import modworm_recovery_30d as m

THETA_GT = m.THETA_GT
DIM = len(THETA_GT)
TRAIN_SEEDS = list(range(0, 8))     # seeds defining the "observed" behaviour
EVAL_SEEDS = list(range(100, 108))  # held-out seeds for calibration loss (train/test split)

# Prior: multiplicative jitter on the reference parameters. SBC draws theta_true
# from EXACTLY this prior; recovery must invert the same generative model.
PRIOR_SCALE = 0.15


def prior_sample(rng):
    return THETA_GT * (1.0 + PRIOR_SCALE * rng.randn(DIM))


def targets_from(theta, seeds):
    return [m.rollout(theta, seed=s) for s in seeds]


def held_out_loss(theta, eval_targets):
    """MSE of theta's rollout vs the observed eval-seed targets. Pure trajectory."""
    vals = []
    for s, tg in zip(EVAL_SEEDS, eval_targets):
        if tg is None:
            continue
        tr = m.rollout(theta, seed=s)
        if tr is None or not np.all(np.isfinite(tr)):
            continue
        vals.append(float(np.mean((tr - tg) ** 2)))
    return float(np.mean(vals)) if vals else 1.0


def cma_recover(train_targets, init_seed, n_iter, popsize, sigma0):
    """One posterior sample: CMA-ES fit to TRAIN targets, return best theta."""
    import cma
    rng = np.random.RandomState(init_seed)
    theta0 = THETA_GT * (1.0 + PRIOR_SCALE * rng.randn(DIM))
    scale_vec = np.abs(THETA_GT)
    x0 = (theta0 - THETA_GT) / scale_vec

    def fit_loss(theta):
        vals = []
        for s, tg in zip(TRAIN_SEEDS, train_targets):
            if tg is None:
                continue
            tr = m.rollout(theta, seed=s)
            if tr is None or not np.all(np.isfinite(tr)):
                continue
            vals.append(float(np.mean((tr - tg) ** 2)))
        return float(np.mean(vals)) if vals else 1.0

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        "popsize": popsize, "maxiter": n_iter, "verbose": -9,
        "seed": init_seed + 1, "CMA_diagonal": True})
    gen = 0
    while not es.stop() and gen < n_iter:
        xs = es.ask()
        losses = [fit_loss(THETA_GT + np.array(xi) * scale_vec) for xi in xs]
        es.tell(xs, losses)
        gen += 1
    xbest = np.array(es.result.xbest)
    return THETA_GT + xbest * scale_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--ensemble", type=int, default=12)   # L posterior samples
    ap.add_argument("--n_iter", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--sigma0", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.RandomState(70000 + args.draw)
    theta_true = prior_sample(rng)

    train_targets = targets_from(theta_true, TRAIN_SEEDS)
    eval_targets = targets_from(theta_true, EVAL_SEEDS)

    # held-out statistic for the ground truth
    L_true = held_out_loss(theta_true, eval_targets)

    # recover ensemble (each member: independent CMA init seed)
    ens_losses = []
    ens_thetas = []
    for j in range(args.ensemble):
        init_seed = 800000 + args.draw * 1000 + j
        th = cma_recover(train_targets, init_seed, args.n_iter, args.popsize, args.sigma0)
        Lj = held_out_loss(th, eval_targets)
        ens_losses.append(Lj)
        ens_thetas.append(th.tolist())

    # SBC rank: number of ensemble members with statistic < statistic(theta_true).
    # Statistic = held-out trajectory loss (lower = closer to observed).
    rank = int(np.sum(np.array(ens_losses) < L_true))

    # Also per-parameter rank (componentwise) for a richer multi-dim SBC view.
    ens_arr = np.array(ens_thetas)  # (L, DIM)
    per_param_rank = [int(np.sum(ens_arr[:, d] < theta_true[d])) for d in range(DIM)]

    res = {
        "draw": args.draw,
        "ensemble": args.ensemble,
        "L_true_heldout": L_true,
        "ensemble_heldout_losses": ens_losses,
        "rank_loss": rank,            # in 0..ensemble
        "per_param_rank": per_param_rank,  # each in 0..ensemble
        "theta_true": theta_true.tolist(),
        "elapsed_s": time.time() - t0,
        "config": {"prior_scale": PRIOR_SCALE, "n_iter": args.n_iter,
                   "popsize": args.popsize, "sigma0": args.sigma0,
                   "train_seeds": TRAIN_SEEDS, "eval_seeds": EVAL_SEEDS},
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[sbc-mw draw {args.draw}] rank_loss={rank}/{args.ensemble} "
          f"L_true={L_true:.4e} t={res['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
