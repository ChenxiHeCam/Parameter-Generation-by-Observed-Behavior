"""Merge SBC per-draw shards and compute calibration diagnostics:
  - rank histogram (loss-statistic rank in 0..L)
  - chi-square uniformity test p-value
  - per-parameter rank histograms (multi-dim SBC)
Run on the remote where shards live (py38 has scipy). Pure post-hoc analysis.
"""
import json, glob, sys, os
import numpy as np
from scipy import stats

def analyze(shard_glob, tag):
    files = sorted(glob.glob(shard_glob))
    files = [f for f in files if "smoke" not in f]
    draws = []
    for f in files:
        try:
            draws.append(json.load(open(f)))
        except Exception:
            pass
    if not draws:
        print(f"[{tag}] no shards at {shard_glob}")
        return None
    L = draws[0]["ensemble"]
    ranks = np.array([d["rank_loss"] for d in draws])
    N = len(ranks)
    # rank histogram over 0..L  (L+1 bins)
    hist = np.bincount(ranks, minlength=L + 1)
    expected = N / (L + 1)
    chi2 = float(np.sum((hist - expected) ** 2 / expected))
    dof = L
    p_chi2 = float(1 - stats.chi2.cdf(chi2, dof))
    # per-parameter SBC
    pp = np.array([d["per_param_rank"] for d in draws])  # (N, DIM)
    dim = pp.shape[1]
    pp_chi2_p = []
    for d in range(dim):
        h = np.bincount(pp[:, d], minlength=L + 1)
        c = float(np.sum((h - expected) ** 2 / expected))
        pp_chi2_p.append(round(float(1 - stats.chi2.cdf(c, dof)), 4))
    out = {
        "tag": tag, "N_draws": N, "ensemble_L": L,
        "rank_hist_loss": hist.tolist(),
        "rank_mean": float(ranks.mean()), "rank_expected": L / 2.0,
        "chi2_loss": round(chi2, 3), "chi2_dof": dof,
        "p_uniform_loss": round(p_chi2, 5),
        "per_param_p_uniform": pp_chi2_p,
        "interpretation": (
            "p_uniform>0.05 => ranks consistent with Uniform => calibrated. "
            "Strong left/right skew (low p, rank_mean far from L/2) => mis-calibration: "
            "rank_mean<<L/2 means recovery systematically over-fits (posterior too tight / "
            "ensemble beats ground truth on held-out), rank_mean>>L/2 means under-fits.")
    }
    print(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    res = {}
    r1 = analyze("/root/autodl-tmp/sbc_mw/draw*.json", "modWorm_30d")
    if r1: res["modWorm"] = r1
    r2 = analyze("/root/autodl-tmp/sbc_baai/draw*.json", "BAAIWorm_5d")
    if r2: res["BAAIWorm"] = r2
    with open("/root/autodl-tmp/sbc_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE /root/autodl-tmp/sbc_summary.json")
