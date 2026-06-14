"""Proper SBC: per-parameter rank histograms (the valid statistic). The loss-rank
column is degenerate (optimized ensemble always beats the un-optimized prior draw),
so we report parameter-space ranks: rank of theta_true[d] among the L recovered
ensemble values for dim d. Calibrated => Uniform{0..L} per dim.
Aggregate across the 30 (modWorm) dims gives a pooled histogram with high power.
"""
import json, glob
import numpy as np
from scipy import stats

files = [f for f in sorted(glob.glob("/root/autodl-tmp/sbc_mw/draw*.json")) if "smoke" not in f]
draws = [json.load(open(f)) for f in files]
L = draws[0]["ensemble"]
pp = np.array([d["per_param_rank"] for d in draws])  # (N, DIM)
N, DIM = pp.shape

# pooled histogram over all dims (N*DIM samples), bins 0..L
pooled = pp.reshape(-1)
hist = np.bincount(pooled, minlength=L + 1)
exp = pooled.size / (L + 1)
chi2 = float(np.sum((hist - exp) ** 2 / exp))
p = float(1 - stats.chi2.cdf(chi2, L))

# direction: mean rank vs L/2
mean_rank = float(pooled.mean())

# how many of the DIM per-dim tests are individually uniform at 0.05
perdim_p = []
for d in range(DIM):
    h = np.bincount(pp[:, d], minlength=L + 1)
    c = float(np.sum((h - (N/(L+1)))**2 / (N/(L+1))))
    perdim_p.append(1 - stats.chi2.cdf(c, L))
n_uniform = int(np.sum(np.array(perdim_p) > 0.05))

out = {
    "statistic": "per_parameter_rank (valid SBC)",
    "N_draws": N, "DIM": DIM, "ensemble_L": L,
    "pooled_rank_hist": hist.tolist(),
    "pooled_mean_rank": round(mean_rank, 3), "expected_mean_rank": L / 2.0,
    "pooled_chi2": round(chi2, 2), "pooled_p_uniform": round(p, 6),
    "n_dims_uniform_p>0.05": n_uniform, "of_total_dims": DIM,
    "shape": ("U-shaped/edge-heavy => posterior over-dispersed; "
              "peaked-center => over-confident; "
              "monotone => biased estimator"),
}
print(json.dumps(out, indent=2))
json.dump(out, open("/root/autodl-tmp/sbc_perparam_summary.json", "w"), indent=2)
print("WROTE /root/autodl-tmp/sbc_perparam_summary.json")
