# PGOB — Parameter Generation by Observed Behavior

Generate working parameters for whole-organism biophysical simulators of *C. elegans*
and *Drosophila* **from observed behaviour alone** — a single laboratory's tracking
video, with no patch clamp, no two-photon imaging, and no EMG.

Repository: <https://github.com/ChenxiHeCam/Parameter-Generation-by-Observed-Behavior>

---

## What PGOB does

Whole-organism simulators (BAAIWorm, modWorm, flybody, FlyGym) carry thousands of
biophysical parameters that are normally pinned by intracellular electrophysiology most
behaviour laboratories cannot acquire. PGOB replaces that calibration step with ordinary
tracking video:

- **Behaviour-only parameter generation.** The input is a kinematic trajectory; from a
  randomised start the procedure *generates* a working parameter set, together with a
  Jacobian sensitivity map that attributes each remaining behavioural residual to a
  specific simulator mechanism.
- **Inference-engine-agnostic.** The optimisation back-end is pluggable (instantiated
  here with SPSA and CMA-ES); the contribution is the behaviour-only formulation, not the
  optimiser.
- **One pipeline, four simulators, two species.** The framework runs unchanged across
  BAAIWorm, modWorm, flybody, and FlyGym. The only dimension-adaptive setting is the
  initialisation (random for low-dimensional worm models; a constrained same-species
  prior for high-dimensional fly models).

### Headline findings

- **Cross-simulator behaviour recovery (primary statistic).** Applied unchanged to all
  four simulators, PGOB moves **26 / 36 behavioural observables toward real biology**
  (7/10 BAAIWorm, 2/6 modWorm, 9/10 flybody, 8/10 FlyGym). We report this
  baseline-independent count as the primary cross-simulator statistic because the
  deposited-default magnitudes are not commensurable across simulators.
- **Generation, not re-calibration.** Every recovery is placed on a single anchored axis
  running from a randomised start, through the generated parameters, to the
  expert-calibrated default, to real biology. On a *Drosophila* (FlyGym) walker that
  begins from a fully randomised, frequently non-walking controller, the generated
  parameters close **94.9 %** of the random-to-expert distance to real Janelia kinematics
  (98.7 % on forward speed, 90.4 % on lateral speed). On the BAAIWorm connectome the same
  procedure reconstructs the full synaptic, gap-junction, motor-output, ion-channel, and
  passive-membrane parameter set from a fully scrambled start, up to the simulator's
  published behaviour band.
- **Recovered parameters are biology, not fitted digits.** The recovered parameter
  *distribution* transfers as a prior across a sibling simulator of the **same** species
  (BAAIWorm→modWorm, 13/16 held-out starts improved, *p* = 0.025; a FlyGym→flybody
  self-prior gives a 9.3× optimisation gain), whereas cross-species priors do not
  transfer — so the learned structure is species-specific.
- **Prospective biological test.** From the N2-recovered Jacobian, behavioural deviation
  is predicted on **43 held-out *C. elegans* mutant strains** never seen during fitting
  (leave-one-strain-out mean Spearman ρ̄ = 0.684 vs a strain-shuffle null 0.527; 41/43
  strains ρ > 0).
- **Generalises across laboratories and to neural data.** Pattern-level observables
  generalise to a second worm laboratory (median cross/within W₁ = 2.80×), and
  population-level neural statistics from independent calcium imaging agree with the
  recovered simulator without any neural supervision.
- **Debugging by attribution.** Where a simulator's substrate cannot express a target
  behaviour, the residual is a property of the deposited model — and the Jacobian
  attribution map says, axis by axis, which mechanism is responsible.

---

## Repository layout

```
.
├── README.md              This file
├── LICENSE                MIT
├── CITATION.cff           How to cite
├── paper/                 Manuscript + supplementary (PDF and LaTeX source)
│   ├── PGOB_main.pdf / .tex
│   ├── PGOB_supplementary.pdf / .tex
│   ├── references.bib
│   └── figures/           Figures referenced by the manuscript and supplementary
├── code/                  Per-simulator recovery, evaluation, and SBC drivers
│   ├── README.md          Environment setup + per-simulator entry points
│   ├── baaiworm/          full5 recovery + behaviour evaluator + recovery experiments
│   ├── modworm/           Cook-ODE CMA-ES recovery + eigenworm encoding + prior transfer
│   └── sbc/               Simulation-Based Calibration drivers + analysis
└── data/
    └── README.md          Pointers to public real-animal datasets (data not vendored)
```

---

## Installation

PGOB drives each published simulator in its own environment. Clone this repository, then
install the simulator(s) you want to reproduce against (links in `code/README.md`):

```bash
git clone https://github.com/ChenxiHeCam/Parameter-Generation-by-Observed-Behavior
cd Parameter-Generation-by-Observed-Behavior
python -m pip install numpy scipy cma matplotlib   # core optimiser/eval dependencies
```

The recovery drivers expect the corresponding simulator to be installed on the host (see
`code/README.md` for per-simulator install instructions and the expected layout). Each
simulator pins its own heavy dependencies (NEURON for BAAIWorm, Julia for modWorm,
MuJoCo for flybody / FlyGym); install those from the upstream projects.

---

## Reproducing key results

All drivers are deterministic given a fixed seed (the simulators are bit-exact), so a
fixed seed reproduces the same trajectory. Representative entry points:

| Result | Command |
|---|---|
| BAAIWorm full5 behaviour-only recovery | `python code/baaiworm/run_full5_multicond_gan.py` |
| BAAIWorm recovery evaluation (held-out conditions) | `python code/baaiworm/scripts/eval_recovery.py` |
| BAAIWorm single / combination recovery | `python code/baaiworm/experiments/phase2_single_recovery.py` |
| modWorm Cook-ODE CMA-ES recovery | `python code/modworm/run_cma_recovery_cook_full.py` |
| modWorm same-species prior transfer | `python code/modworm/modworm_cook_scale10_prior.py` |
| modWorm eigenworm posture encoding | `python code/modworm/p1_modworm_eigenworm.py` |
| Simulation-Based Calibration (per-simulator) | `python code/sbc/sbc_baaiworm.py` · `python code/sbc/sbc_modworm.py` |

See `code/README.md` for arguments, expected host paths, and the train/test split
conventions used by each driver.

---

## Data

Large binaries (raw trajectories, checkpoints) are **not** vendored in this repository.
`data/README.md` lists the public sources for the real-animal references used in the
paper (OpenWorm Movement Database / Yemini 2013, the Brown/Tierpsy cross-laboratory
cohort, and the Janelia walking deposit) and the derived split definitions. The
derived-data archive will be deposited on Zenodo and linked here when the manuscript is
accepted.

---

## Citing

If you use PGOB, please cite the accompanying paper (see `CITATION.cff`). The archival
Zenodo DOI will be added here on acceptance.

## License

MIT — see `LICENSE`. The four simulators and the public datasets carry their own upstream
licenses; consult the respective projects.
