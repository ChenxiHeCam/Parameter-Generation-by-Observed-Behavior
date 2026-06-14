"""Full5 multi-condition SPSA recovery with GAN-Discriminator loss.

loss = -D(traj)   (D high = looks like baseline)
Optional hybrid: alpha * v1_behavioral + (1 - alpha) * (-D(traj))
"""
import argparse, logging, multiprocessing as mp, os, random as pyrandom, sys, time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")
sys.path.insert(0, "/root/gan_pipeline")

import numpy as np
import run_multicond_refine_queue as mcr
from save_trajectory_dataset import food_grid, start_grid
from recovery.methods.spsa_optimizer import SPSARecoverer
from recovery.methods.gradient_descent.trajectory_loss import TrajectoryLoss
from recovery.utils.io_utils import load_pickle, save_pickle, save_json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("full5_multicond_gan")

BASELINE_DIR = "/root/BAAIWorm-main/recovery/output/multicond_baseline"
FULL5 = ["syn", "gj", "wout", "ion_channels", "passive_params"]


def build_conditions(food_lo, food_hi, n_starts, n_steps, n_foods_total, n_starts_total):
    foods = food_grid(n_foods_total); starts = start_grid(n_starts_total)
    out = []
    for fi in range(food_lo, food_hi):
        for si in range(n_starts):
            path = os.path.join(BASELINE_DIR, f"food{fi:02d}_start{si:02d}_steps{n_steps}.pkl")
            if not os.path.exists(path):
                continue
            payload = load_pickle(path)
            s, o = starts[si]
            cond = {"food_idx": fi, "start_idx": si, "food": foods[fi], "start": s, "orientation": o}
            out.append((cond, payload))
    return out


def _run_sim_for_cond(cond, weight_dict, param_types, n_steps):
    sim = mcr.make_simulator(cond, n_steps)
    return sim.run_with_custom_weights(weight_dict, param_types, n_steps=n_steps)


def _worker_init(d_ckpt_path):
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_k] = "1"
    for p in ("/root/BAAIWorm-main/build_headless/build", "/root/BAAIWorm-main",
              "/root/BAAIWorm-main/recovery/scripts", "/root/gan_pipeline"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import loss_gan
    loss_gan.load_d(d_ckpt_path, device="cpu")


def _worker_one(args):
    (cond, target_traj, food_loc, weight_dict, param_types, n_steps, alpha, d_ckpt) = args
    try:
        import loss_gan
        traj = _run_sim_for_cond(cond, weight_dict, param_types, n_steps)
        d_val = loss_gan.d_score(traj, food=food_loc, ckpt_path=d_ckpt, device="cpu")
        gan_l = -d_val
        if alpha > 0.0:
            v1_fn = TrajectoryLoss(target_traj)
            v1_l, _ = v1_fn.compute(traj)
            l = alpha * float(v1_l) + (1.0 - alpha) * gan_l
            return float(l), {"v1": float(v1_l), "d": float(d_val)}, None
        return float(gan_l), {"d": float(d_val)}, None
    except Exception as exc:
        return 1e6, None, f"{type(exc).__name__}: {exc}"


_POOL = None
def get_pool(n_workers, d_ckpt):
    global _POOL
    if _POOL is None and n_workers > 1:
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(d_ckpt,))
        logger.info("Spawned worker pool n=%d, D=%s", n_workers, d_ckpt)
    return _POOL


class GANMiniBatchSPSA(SPSARecoverer):
    def __init__(self, init_weights, train_pairs, val_pairs, param_types,
                 batch_size, seed, config, alpha, d_ckpt, n_workers):
        super().__init__(target_data=train_pairs[0][1]["trajectory"],
                         perturbed_weight_dict=init_weights, param_types=param_types,
                         simulator=None, config=config)
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.alpha = alpha
        self.d_ckpt = d_ckpt
        self.n_workers = n_workers
        self.batch_size = min(batch_size, len(train_pairs))
        self.rng = pyrandom.Random(seed)
        self._iter = 0

    def _evaluate(self, w):
        wd = self._decode_weights(w)
        idxs = self.rng.sample(range(len(self.train_pairs)), self.batch_size)
        pool = get_pool(self.n_workers, self.d_ckpt)
        tasks = []
        for i in idxs:
            cond, payload = self.train_pairs[i]
            tasks.append((cond, payload["trajectory"], cond["food"], wd,
                          self.param_types, self.sim_steps, self.alpha, self.d_ckpt))
        if pool is not None:
            outs = pool.map(_worker_one, tasks)
        else:
            outs = [_worker_one(t) for t in tasks]
        losses = []
        d_vals = []
        v1_vals = []
        for l, det, err in outs:
            if err:
                logger.warning("sim fail: %s", err)
            losses.append(l)
            if det:
                if "d" in det: d_vals.append(det["d"])
                if "v1" in det: v1_vals.append(det["v1"])
        avg = float(np.mean(losses))
        self._iter += 1
        if self._iter % 2 == 0:
            logger.info("iter %d loss=%.5f D=%.4f v1=%.4f", self._iter, avg,
                        float(np.mean(d_vals)) if d_vals else 0.0,
                        float(np.mean(v1_vals)) if v1_vals else 0.0)
        return avg

    def final_val(self):
        if not self.val_pairs or self.best_weights is None:
            return None
        wd = self.best_weights
        v1s, ds = [], []
        import loss_gan
        loss_gan.load_d(self.d_ckpt, device="cpu")
        for cond, payload in self.val_pairs:
            try:
                traj = _run_sim_for_cond(cond, wd, self.param_types, self.sim_steps)
                v1, _ = TrajectoryLoss(payload["trajectory"]).compute(traj)
                d = loss_gan.d_score(traj, food=cond["food"])
                v1s.append(float(v1)); ds.append(float(d))
            except Exception as e:
                logger.warning("val fail: %s", e)
        if not v1s: return None
        return {"v1_mean": float(np.mean(v1s)), "v1_std": float(np.std(v1s)),
                "d_mean": float(np.mean(ds)), "d_std": float(np.std(ds)),
                "n": len(v1s)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", required=True)
    p.add_argument("--d-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.0, help="0=pure D, 0.5=hybrid")
    p.add_argument("--train-foods", type=int, default=20)
    p.add_argument("--train-starts", type=int, default=4)
    p.add_argument("--val-foods", type=int, default=15)
    p.add_argument("--val-starts", type=int, default=2)
    p.add_argument("--train-food-lo", type=int, default=70)
    p.add_argument("--val-food-lo", type=int, default=50)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--spsa-a0", type=float, default=0.005)
    p.add_argument("--spsa-c0", type=float, default=0.03)
    p.add_argument("--n-workers", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prev = load_pickle(args.init_from)
    init_w = prev["recovered_weights"]
    prev_loss = prev.get("best_loss", "?")
    logger.info("Warm from %s prev_loss=%s D=%s alpha=%.2f",
                args.init_from, prev_loss, args.d_ckpt, args.alpha)

    n_foods_total = args.train_foods + args.val_foods + 5
    n_starts_total = max(args.train_starts, args.val_starts)
    train_pairs = build_conditions(args.train_food_lo,
                                   args.train_food_lo + args.train_foods,
                                   args.train_starts, args.n_steps,
                                   max(n_foods_total, 95), max(n_starts_total, 8))
    val_pairs = build_conditions(args.val_food_lo,
                                 args.val_food_lo + args.val_foods,
                                 args.val_starts, args.n_steps,
                                 max(n_foods_total, 95), max(n_starts_total, 8))
    logger.info("Train=%d Val=%d", len(train_pairs), len(val_pairs))

    opt = GANMiniBatchSPSA(
        init_weights=init_w, train_pairs=train_pairs, val_pairs=val_pairs,
        param_types=FULL5, batch_size=args.batch_size, seed=args.seed,
        config={"a0": args.spsa_a0, "c0": args.spsa_c0,
                "A": max(args.iters // 3, 3),
                "sim_steps": args.n_steps, "seed": args.seed},
        alpha=args.alpha, d_ckpt=args.d_ckpt, n_workers=args.n_workers)
    t0 = time.time()
    result = opt.optimize(n_iterations=args.iters)
    elapsed = time.time() - t0
    val_summary = opt.final_val()
    if val_summary:
        logger.info("FINAL VAL v1=%.4f+/-%.4f D=%.4f n=%d",
                    val_summary["v1_mean"], val_summary["v1_std"],
                    val_summary["d_mean"], val_summary["n"])
    result["seed"] = args.seed
    result["method"] = f"multicond_spsa_gan_alpha{args.alpha}"
    result["param_types"] = FULL5
    result["full5"] = True
    result["init_from"] = args.init_from
    result["init_from_loss"] = prev_loss
    result["d_ckpt"] = args.d_ckpt
    result["alpha"] = args.alpha
    result["n_train_conds"] = len(train_pairs)
    result["n_val_conds"] = len(val_pairs)
    result["val_summary"] = val_summary
    result["time_seconds"] = elapsed
    save_pickle(result, os.path.join(args.out_dir, "result.pkl"))
    save_json({k: v for k, v in result.items()
               if k not in ("recovered_weights", "loss_history")},
              os.path.join(args.out_dir, "result.json"))
    logger.info("DONE in %.1f min, saved %s", elapsed / 60, args.out_dir)


if __name__ == "__main__":
    main()
