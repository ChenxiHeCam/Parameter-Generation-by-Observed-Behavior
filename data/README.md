# data/

This directory does **not** vendor large binaries (raw trajectories, simulator
rollouts, checkpoints). It documents where the real-animal reference data come from
and how the derived data are organised.

## Public real-animal sources

PGOB's real-biology comparisons and the prospective mutant-strain test use publicly
available datasets, obtained from their original repositories:

- **OpenWorm Movement Database (OWMD) / Yemini 2013** — *C. elegans* N2 and mutant-strain
  tracking. Used for the real-worm reference (N2) and the held-out 43-strain prospective
  Jacobian test. Source: the OpenWorm Movement Database / Yemini *et al.* 2013 deposit.
- **Brown / Tierpsy cross-laboratory worm tracking** — independent *C. elegans* tracking
  cohort used for the leave-one-lab-out cross-laboratory test. Source: Zenodo deposit
  `10.5281/zenodo.3837679`.
- **Janelia walking-imitation deposit** — *Drosophila* walking trajectories, used as the
  real-fly reference for the fly simulators. Source: Janelia figshare deposit
  (DOI `10.25378/janelia.25309105`).

Download these from their upstream repositories; their licenses are set by the
respective data providers.

## Derived data and reproduction archive

The derived split definitions, simulator rollouts, and recovered-parameter snapshots
needed to recompute every main-text number will be deposited as an archival release on
Zenodo and linked here when the manuscript is accepted. (No DOI is minted yet.)

## Simulators

The four simulators are obtained from their upstream projects (see `code/README.md`):
BAAIWorm, modWorm, flybody, and FlyGym.
