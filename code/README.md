# code/

Recovery, evaluation, and Simulation-Based Calibration drivers for PGOB. The framework
itself is a single behaviour-only pipeline (SPSA / CMA-ES over multi-condition rollouts
plus a universal behaviour-feature evaluator); each subdirectory wires that pipeline to
one published simulator.

```
code/
├── baaiworm/   BAAIWorm (C. elegans, NEURON) full5 recovery + behaviour evaluator
│   ├── run_full5_multicond_gan.py     behaviour-only multi-condition recovery driver
│   ├── run_multicond_refine_queue.py  multi-condition rollout/refine queue
│   ├── default_config.yaml            optimiser + loss configuration
│   ├── methods/                       SPSA / CMA-ES optimisers (base + two backends)
│   ├── evaluation/                    behaviour-feature metrics + trajectory comparison
│   ├── experiments/                   single / combination recovery + degeneracy analysis
│   └── scripts/                       held-out-condition evaluation harnesses
├── modworm/    modWorm (C. elegans, Julia ODE)
│   ├── run_cma_recovery_cook_full.py  Cook-ODE per-edge CMA-ES recovery
│   ├── modworm_cook_scale10_prior.py  same-species soft-prior transfer
│   └── p1_modworm_eigenworm.py        eigenworm (Stephens 2008) posture encoding
└── sbc/        Simulation-Based Calibration
    ├── sbc_baaiworm.py / sbc_modworm.py   per-simulator SBC drivers
    ├── sbc_perparam.py                    per-parameter rank statistics
    └── sbc_analyze.py                     rank-histogram analysis
```

## Simulators (install from upstream)

The drivers call into the published simulators; install the one(s) you want to reproduce
against from their upstream projects:

- **BAAIWorm** — *C. elegans* whole-organism simulator (NEURON backend).
- **modWorm** — *C. elegans* connectome ODE simulator (Julia backend).
- **flybody** — *Drosophila* MuJoCo body model.
- **FlyGym / NeuroMechFly** — *Drosophila* MuJoCo neuromechanical model.

Core PGOB dependencies (optimiser + evaluator): `numpy`, `scipy`, `cma`, `matplotlib`.
Each simulator additionally pulls in its own heavy dependency (NEURON, Julia, MuJoCo).

## Expected host layout

For reproducibility the drivers reference the simulator clones at their install
locations on the run host (e.g. a BAAIWorm clone added to `sys.path`, a modWorm working
directory). Adjust the `sys.path.insert(...)` / `os.chdir(...)` lines at the top of each
driver, and the output paths, to match where you installed each simulator. These are the
verbatim scripts that produced the paper's numbers; the only host-specific edits needed
are those paths.

## Determinism and splits

The simulators are bit-exact deterministic: a fixed seed reproduces the same trajectory,
so multi-start and cross-mode experiments record their `train_seeds` / `test_seeds`
explicitly in their result files. Any fit/match experiment uses a held-out train/test
split and reports the train number, the test number, and the gap.
