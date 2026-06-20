**Chenxi He**
Cavendish Laboratory, University of Cambridge
Cambridge CB3 0HE, United Kingdom
ch2067@cam.ac.uk

20 June 2026

To the Editors, *Nature Computational Science*

Dear Editors,

I am pleased to submit my manuscript, **"Generating behaviour-equivalent parameters for whole-organism simulators from tracking video,"** for consideration in *Nature Computational Science*. I am the sole author.

Recovering the hundreds-to-thousands of hidden parameters of a whole-organism biophysical simulator from observable outputs is the classical inverse problem of systems identification. In current practice those parameters are pinned by invasive physiology — intracellular electrophysiology, two-photon imaging, EMG — that most laboratories cannot perform and that for many species cannot be performed at all; this calibration step, not the simulators themselves, locks most groups and most species out of using them. I present a computational method, PGOB (Parameter Generation by Observed-Behaviour), that turns calibration into a problem solvable from tracking video alone: from the externally observed kinematics of a behaving animal, and from a start drawn independently of the optimum, it generates a working set of simulator parameters and returns a Jacobian sensitivity map attributing any residual mismatch to a specific simulator mechanism. The inference engine is swappable — PGOB ships with black-box optimisers (SPSA, CMA-ES) and an amortised simulation-based-inference engine (NPE-MAF via the `sbi` toolkit) — and recovers the same simulators with either.

**Why *Nature Computational Science*.** The manuscript pairs a general, engine-agnostic computational method with cross-simulator generality and a reproducibility-first protocol, and it is careful to claim only what the data support. The same formulation runs *unchanged*, with no per-simulator re-engineering, across four published simulators spanning two species, generating behaviour-equivalent parameters from an uninformative start (modWorm ground-truth parameters to relative error 0.009; a *Drosophila* walker closing 94.9% of the random-to-expert distance; the BAAIWorm connectome rebuilt from a fully scrambled start to held-out behaviour distance 0.1199, within its published band). Where a simulator carries its own target, the recovery is tight: the generated worm reaches its simulator's own target more closely than two real worms differ from each other (recovery-to-target 0.71 vs. real-to-real spread 1.98, in 1000/1000 bootstrap resamples) — a statement about recovery tightness, not about beating real biology. The deployment claim I make is deliberately a **parity** claim, not a superiority one: from behaviour alone, cold-started, with the stimuli and drives that produced the animal's path entirely unknown, PGOB reaches the *same* closeness to real biology as the simulator's own physiologically-calibrated default (recovered-to-real ≈ default-to-real, e.g. 21.8 vs. 21.8 on the worm; 3.29 vs. 2.69 on modWorm) — that is, behaviour-only inference *matches* physiological calibration without performing any physiology, rather than surpassing it. That the recovered parameters are biologically meaningful, not an arbitrary curve fit, is shown by a falsifiability test: generation *fails* across species — a worm simulator cannot be driven onto fly walking, nor a fly simulator onto worm chemotaxis (6 of 8 cross-species pairs reach none of their target conditions, at 87–93% residual deviation, an order of magnitude worse than within-phylum fitting) — and by the generated parameters transferring as a same-species prior and prospectively predicting 43 held-out mutant strains (leave-one-strain-out ρ̄ = 0.684 vs. 0.527).

Methodologically, every load-bearing claim is controlled under a single whole-paper Holm–Šidák family-wise error budget; confidence intervals use B = 1000 bootstrap resamples; and all runs use deterministic seeding. As a reproducibility check, all four simulators were rebuilt from public source on commodity CPU and the headline behaviour-recovery results reproduced live — including a headless CPU inference engine compiled from the BAAIWorm source. All code is MIT-licensed on GitHub, third-party data are public under their original licences, and the manuscript, SI and per-figure reproduction commands are archived on Zenodo (DOI 10.5281/zenodo.20691877).

**Outlook.** Because the formulation is simulator- and engine-agnostic and consumes only behavioural video, it can be pointed at future mechanistic simulators to generate their parameters at scale without an accompanying physiology study. The reach I find most valuable — stated as an outlook, not a claim settled here — is to species and preparations that current calibration cannot reach: organisms with no historical electrophysiology, endangered animals, and preparations that cannot be patched or imaged, yet whose movement can still be filmed.

The work is original, unpublished, and not under consideration elsewhere; it is submitted to *Nature Computational Science* alone. I declare no competing interests and am happy to suggest or exclude reviewers.

Yours sincerely,

Chenxi He
University of Cambridge
