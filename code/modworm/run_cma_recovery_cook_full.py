#!/usr/bin/env python3
"""modWorm sim-to-sim parameter recovery via CMA-ES, FULL per-edge 279 connectome (Varshney).

Upgrade vs run_cma_recovery_real.py:
  - Uses Cook (not Varshney) adjustment matrices: conn_gap_adjust_Varshney.npy, conn_syn_adjust_Varshney.npy
  - Parameter vector is ~565-d:
      [0..2]   global gains (ggap, gsyn, gmuscle)
      [3..6]   intrinsic global scales (cap, gleak, tau_ion, eleak_shift)
      [7..285] per-neuron gleak multiplier (279 entries)
      [286..564] per-neuron cap multiplier (279 entries)

  This captures heterogeneity across the Cook 279 connectome rather than only
  7 global knobs. CMA-ES still tractable in sep mode (diagonal cov).
"""
import os, sys, time, json, traceback
import numpy as np

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
from julia.api import Julia
jl = Julia(compiled_modules=False)

os.chdir("/root/modWorm")
sys.path.insert(0, "/root/modWorm")

from modWorm import sys_paths
from modWorm import utils
from modWorm import predefined_classes_nv as pcn
from modWorm import predefined_classes_mb as pcm
from modWorm import proprioception_simulation as p_sim
import cma

print("modWorm Cook full loaded ok", flush=True)

data_dir = "/root/autodl-tmp/modWorm/modWorm/data"
conn_gap = np.load(f"{data_dir}/conn_gap_adjust_Varshney.npy")
conn_syn = np.load(f"{data_dir}/conn_syn_adjust_Varshney.npy")
N_NEURON = conn_gap.shape[0]
assert N_NEURON == 279, f"expected 279, got {N_NEURON}"
print(f"[cook] conn_gap {conn_gap.shape} nnz={int((conn_gap!=0).sum())}, "
      f"conn_syn {conn_syn.shape} nnz={int((conn_syn!=0).sum())}", flush=True)

rng_mm = np.random.default_rng(42)
_MM_REAL = "/root/autodl-tmp/modWorm/modWorm/muscle_maps/muscle_map_real.npy"
if os.path.exists(_MM_REAL):
    muscle_map = np.load(_MM_REAL).astype(float)
else:
    muscle_map = (rng_mm.random((95, N_NEURON)) < 0.05).astype(float) * rng_mm.uniform(0.5, 1.5, (95, N_NEURON))
print(f"[muscle_map] shape={muscle_map.shape}", flush=True)

stim_full = np.load("/root/autodl-tmp/modWorm/modWorm/presets_input/input_mat_gentle_post_touch.npy")
T_STEPS = 300
stim = stim_full[:T_STEPS, :N_NEURON].copy() if stim_full.shape[1] >= N_NEURON else np.pad(
    stim_full[:T_STEPS], ((0,0),(0, N_NEURON - stim_full.shape[1])))
print(f"[stim] shape {stim.shape}", flush=True)

# Cook-edge-aware parameterisation:
# DIM = 3 (global gains: ggap, gsyn, gmuscle) + N_GAP_NNZ + N_SYN_NNZ multipliers
GAP_NZ = np.argwhere(conn_gap != 0)  # (K_gap, 2)
SYN_NZ = np.argwhere(conn_syn != 0)
N_GAP_NNZ = len(GAP_NZ)
N_SYN_NNZ = len(SYN_NZ)
N_GLOBAL = 3
DIM = N_GLOBAL + N_GAP_NNZ + N_SYN_NNZ
print(f"[dim] full-cook param dim = {DIM} (3 global + {N_GAP_NNZ} gap-edges + {N_SYN_NNZ} syn-edges)", flush=True)


def build_nv_mb(p):
    g = np.asarray(p, dtype=float)
    cg = conn_gap.copy()
    cs = conn_syn.copy()
    # per-edge multipliers
    gap_mul = g[N_GLOBAL : N_GLOBAL + N_GAP_NNZ]
    syn_mul = g[N_GLOBAL + N_GAP_NNZ : N_GLOBAL + N_GAP_NNZ + N_SYN_NNZ]
    cg[GAP_NZ[:,0], GAP_NZ[:,1]] = conn_gap[GAP_NZ[:,0], GAP_NZ[:,1]] * gap_mul
    cs[SYN_NZ[:,0], SYN_NZ[:,1]] = conn_syn[SYN_NZ[:,0], SYN_NZ[:,1]] * syn_mul
    # global gains on top
    cg = cg * g[0]
    cs = cs * g[1]
    mm = muscle_map * g[2]
    nv = pcn.CelegansWorm_NervousSystem_PPC_Julia(cg, cs)
    mb = pcm.CelegansWorm_MuscleBody_PPC_Julia(mm)
    return nv, mb


def rollout(p):
    nv, mb = build_nv_mb(p)
    sol = p_sim.run_network_julia(nv, mb, stim)
    return np.stack([sol["x_solution"], sol["y_solution"]], -1)


SCALE = float(os.environ.get("RECOVERY_SCALE", "0.10"))
SEED = int(os.environ.get("RECOVERY_SEED", "0"))
N_ITER = int(os.environ.get("RECOVERY_N_ITER", "30"))
POPSIZE = int(os.environ.get("RECOVERY_POPSIZE", "20"))
OUT_DIR = os.environ.get("RECOVERY_OUT", f"/root/cook_full_recovery_seed{SEED}")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
# target: perturb full vector by scale around 1.0
p_true = 1.0 + rng.uniform(-SCALE, SCALE, size=DIM)
print(f"[target] scale={SCALE} dim={DIM}", flush=True)

t0 = time.time()
traj_target = rollout(p_true)
print(f"[target] rollout ok in {time.time()-t0:.1f}s, shape={traj_target.shape}", flush=True)
np.save(os.path.join(OUT_DIR, "traj_target.npy"), traj_target)
np.save(os.path.join(OUT_DIR, "p_true.npy"), p_true)


def loss(p):
    try:
        tr = rollout(p)
    except Exception as e:
        return 1.0
    if tr.shape != traj_target.shape or not np.all(np.isfinite(tr)):
        return 1.0
    return float(np.mean((tr - traj_target) ** 2))


# CMA in centered relative coords around 1.0
x0 = np.zeros(DIM)
es = cma.CMAEvolutionStrategy(x0, SCALE * 0.5,
    {"popsize": POPSIZE, "maxiter": N_ITER, "CMA_diagonal": True,
     "verbose": -9, "seed": SEED + 1, "tolx": 1e-7, "tolfun": 1e-9})

hist = []
gen = 0
L0 = loss(np.ones(DIM))
print(f"[init] L0(at theta=1)={L0:.4e}", flush=True)
while not es.stop() and gen < N_ITER:
    xs = es.ask()
    fits = []
    for xi in xs:
        p_i = 1.0 + xi
        fits.append(loss(p_i))
    es.tell(xs, fits)
    gen += 1
    rec = dict(gen=gen, best=float(min(fits)), median=float(np.median(fits)),
               sigma=float(es.sigma))
    hist.append(rec)
    print(f"[gen {gen:02d}] best={rec['best']:.4e} med={rec['median']:.4e} sigma={rec['sigma']:.3e}", flush=True)
    with open(os.path.join(OUT_DIR, "cma.log"), "a") as f:
        f.write(json.dumps(rec) + "\n")

xbest = es.result.xbest if es.result.xbest is not None else es.mean
p_final = 1.0 + xbest
L_final = loss(p_final)
rel0 = float(np.linalg.norm(np.ones(DIM) - p_true) / np.linalg.norm(p_true))
rel_final = float(np.linalg.norm(p_final - p_true) / np.linalg.norm(p_true))
print(f"[done] L0={L0:.4e}  L_final={L_final:.4e}  rel {rel0:.4f}->{rel_final:.4f}", flush=True)
np.save(os.path.join(OUT_DIR, "p_final.npy"), p_final)
json.dump(dict(L0=L0, L_final=L_final, rel0=rel0, rel_final=rel_final,
               hist=hist, scale=SCALE, dim=DIM, n_neuron=N_NEURON,
               connectome="Cook", seed=SEED),
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
print("FINISH", flush=True)
