"""
CMA-ES（进化策略）权重恢复器。

路线 A 的主力方法：无需梯度，直接在权重空间搜索使轨迹 loss 最小的权重。
天然处理不可微的物理引擎。
"""
import os
import copy
import numpy as np
import logging

from recovery.methods.base_optimizer import BaseWeightRecoverer

logger = logging.getLogger(__name__)

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class CMAESRecoverer(BaseWeightRecoverer):
    """
    CMA-ES 权重恢复器。

    每代采样 N 个候选权重 → 每个跑完整闭环仿真 → 计算轨迹 loss → 更新分布。
    """

    def __init__(self, target_data, perturbed_weight_dict, param_types,
                 simulator, config=None):
        """
        Args:
            simulator: 仿真器实例（HeadlessSimulator 或带物理引擎的）
            config: dict, 包含：
                - population_size: int, 每代候选数（默认 20）
                - sigma0: float, 初始步长（默认 0.1）
                - pca_dim: int, PCA 降维维度（默认 50，0=不降维）
                - max_fevals: int, 最大函数评估次数（默认 5000）
                - sim_steps: int, 每次仿真步数（默认 300）
                - loss_route: str, 'trajectory' 或 'muscle'（默认 'trajectory'）
        """
        if not HAS_CMA:
            raise ImportError("CMAESRecoverer requires `cma` package: pip install cma")

        super().__init__(target_data, perturbed_weight_dict, param_types, config)
        self.simulator = simulator

        self.population_size = config.get('population_size', 20)
        self.sigma0 = config.get('sigma0', 0.1)
        self.pca_dim = config.get('pca_dim', 50)
        self.max_fevals = config.get('max_fevals', 5000)
        self.sim_steps = config.get('sim_steps', 300)
        self.loss_route = config.get('loss_route', 'trajectory')

        self._setup_weight_space()
        self._setup_loss()

    def _setup_weight_space(self):
        """提取要优化的权重向量，可选 PCA 降维"""
        raw_vectors = []
        self.param_slices = {}
        offset = 0

        for pt in self.param_types:
            if pt == 'syn':
                vec = self.perturbed_weight_dict['syn_weights'].copy()
            elif pt == 'gj':
                vec = self.perturbed_weight_dict['gj_weights'].copy()
            elif pt == 'polarity':
                vec = self.perturbed_weight_dict['polarity'].copy()
            elif pt == 'wout':
                vec = self.perturbed_weight_dict['wout'].flatten().copy()
            elif pt == 'input_gain':
                vec = self.perturbed_weight_dict.get('input_gain', np.array([])).copy()
            elif pt == 'ion_channels':
                ic = self.perturbed_weight_dict.get('ion_channels', {})
                vec = np.array(ic.get('matrix', [])).flatten().copy() if isinstance(ic, dict) else np.array([])
            elif pt == 'passive_params':
                pp = self.perturbed_weight_dict.get('passive_params', {})
                vec = np.array(pp.get('matrix', [])).flatten().copy() if isinstance(pp, dict) else np.array([])
            else:
                continue
            if len(vec) == 0:
                continue
            self.param_slices[pt] = (offset, offset + len(vec))
            raw_vectors.append(vec)
            offset += len(vec)

        self.raw_dim = offset
        self.x0_raw = np.concatenate(raw_vectors)

        # PCA 降维
        if HAS_SKLEARN and self.pca_dim > 0 and self.raw_dim > self.pca_dim:
            rng = np.random.RandomState(42)
            samples = np.array([
                self.x0_raw + rng.normal(0, np.std(self.x0_raw) * 0.05, self.raw_dim)
                for _ in range(max(self.pca_dim * 3, 200))
            ])
            self.pca = PCA(n_components=self.pca_dim)
            self.pca.fit(samples)
            self.x0 = self.pca.transform(self.x0_raw.reshape(1, -1))[0]
            self.use_pca = True
            logger.info(f"PCA: {self.raw_dim}D → {self.pca_dim}D, "
                        f"explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        else:
            self.x0 = self.x0_raw.copy()
            self.use_pca = False
            self.pca_dim = self.raw_dim

    def _setup_loss(self):
        """初始化 loss 函数"""
        if self.loss_route == 'trajectory':
            from recovery.methods.gradient_descent.trajectory_loss import TrajectoryLoss
            self.loss_fn = TrajectoryLoss(self.target_data)
        else:
            from recovery.methods.gradient_descent.trajectory_loss import MuscleLoss
            self.loss_fn = MuscleLoss(
                np.array(self.target_data['muscle_activation']),
                self.perturbed_weight_dict.get('wout',
                    np.zeros((80, 96)))
            )

    def _decode_weights(self, x):
        """从优化向量解码回 weight_dict"""
        if self.use_pca:
            raw = self.pca.inverse_transform(x.reshape(1, -1))[0]
        else:
            raw = x.copy()

        result = copy.deepcopy(self.perturbed_weight_dict)
        for pt in self.param_types:
            if pt not in self.param_slices:
                continue
            start, end = self.param_slices[pt]
            vec = raw[start:end]
            if pt == 'syn':
                result['syn_weights'] = vec
            elif pt == 'gj':
                result['gj_weights'] = np.maximum(vec, 1e-9)
            elif pt == 'polarity':
                result['polarity'] = np.sign(vec)
                result['polarity'][result['polarity'] == 0] = 1.0
                result['syn_weights'] = np.abs(result['syn_weights']) * result['polarity']
            elif pt == 'wout':
                result['wout'] = vec.reshape(
                    self.perturbed_weight_dict['wout'].shape)
            elif pt == 'input_gain':
                result['input_gain'] = vec
            elif pt == 'ion_channels':
                ic = self.perturbed_weight_dict.get('ion_channels', {})
                if isinstance(ic, dict) and 'matrix' in ic:
                    orig_shape = np.array(ic['matrix']).shape
                    clamped = np.maximum(vec, 0.0)
                    result['ion_channels'] = dict(ic)
                    result['ion_channels']['matrix'] = clamped.reshape(orig_shape).tolist()
            elif pt == 'passive_params':
                pp = self.perturbed_weight_dict.get('passive_params', {})
                if isinstance(pp, dict) and 'matrix' in pp:
                    orig_shape = np.array(pp['matrix']).shape
                    reshaped = vec.reshape(orig_shape)
                    reshaped[:, 0] = np.maximum(reshaped[:, 0], 1.0)
                    reshaped[:, 1] = np.maximum(reshaped[:, 1], 1e-4)
                    reshaped[:, 2] = np.maximum(reshaped[:, 2], 1e-9)
                    result['passive_params'] = dict(pp)
                    result['passive_params']['matrix'] = reshaped.tolist()
        return result

    def _evaluate(self, x):
        """评估单个候选权重的 loss"""
        weight_dict = self._decode_weights(x)

        try:
            sim_result = self.simulator.run_with_custom_weights(
                weight_dict, self.param_types, n_steps=self.sim_steps)
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return 1e6

        if self.loss_route == 'trajectory':
            loss, _ = self.loss_fn.compute(sim_result)
        else:
            mnv = sim_result.get('motor_neuron_voltage', np.zeros((1, 80)))
            loss = self.loss_fn.compute_scalar(mnv)

        return loss

    def optimize(self, n_iterations=None):
        """
        运行 CMA-ES 优化。

        Args:
            n_iterations: int, 最大迭代次数（覆盖 max_fevals）

        Returns:
            dict
        """
        max_fevals = n_iterations * self.population_size if n_iterations else self.max_fevals

        opts = {
            'popsize': self.population_size,
            'maxfevals': max_fevals,
            'verb_disp': 100,
            'verb_log': 0,
            'seed': self.config.get('seed', 42),
        }

        es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, opts)

        generation = 0
        while not es.stop():
            candidates = es.ask()
            losses = [self._evaluate(x) for x in candidates]
            es.tell(candidates, losses)

            best_loss = min(losses)
            self.loss_history.append(best_loss)

            if best_loss < self.best_loss:
                self.best_loss = best_loss
                best_idx = np.argmin(losses)
                self.best_weights = self._decode_weights(candidates[best_idx])

            generation += 1
            if generation % 10 == 0:
                logger.info(f"Gen {generation}: best_loss={self.best_loss:.6f}, "
                            f"mean_loss={np.mean(losses):.6f}")

        logger.info(f"CMA-ES finished: {generation} generations, "
                    f"best_loss={self.best_loss:.6f}")

        return {
            'recovered_weights': self.best_weights,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'n_iterations': generation,
            'stop_reason': str(es.stop()),
        }
