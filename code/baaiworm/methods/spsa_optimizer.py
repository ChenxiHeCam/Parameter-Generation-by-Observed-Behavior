"""
SPSA（Simultaneous Perturbation Stochastic Approximation）权重恢复器。

核心优势：每次梯度估计只需 2 次仿真，不管参数维度多高。
适合 CMA-ES 找到大致区域后做精细调优，或作为独立的轻量级优化器。

参考：Spall (1992), Wang et al. (2025)
"""
import numpy as np
import copy
import logging

from recovery.methods.base_optimizer import BaseWeightRecoverer

logger = logging.getLogger(__name__)


class SPSARecoverer(BaseWeightRecoverer):
    """
    SPSA 权重恢复器。

    每步：
    1. 生成随机扰动向量 delta（Rademacher 分布）
    2. 评估 f(w + c*delta) 和 f(w - c*delta)
    3. 估计梯度 g = (f+ - f-) / (2*c*delta)
    4. 更新 w = w - a * g
    """

    def __init__(self, target_data, perturbed_weight_dict, param_types,
                 simulator, config=None):
        super().__init__(target_data, perturbed_weight_dict, param_types, config)
        self.simulator = simulator

        # SPSA 超参数（Spall 推荐值）
        self.a0 = config.get('a0', 0.01)       # 初始步长
        self.c0 = config.get('c0', 0.05)        # 初始扰动幅度
        self.A = config.get('A', 50)             # 步长衰减参数
        self.alpha = config.get('alpha', 0.602)  # 步长衰减指数
        self.gamma = config.get('gamma', 0.101)  # 扰动衰减指数
        self.sim_steps = config.get('sim_steps', 200)

        self._setup_weight_space()
        self._setup_loss()

    def _setup_weight_space(self):
        """提取要优化的权重向量"""
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

        self.dim = offset
        self.w = np.concatenate(raw_vectors)
        logger.info("SPSA: optimizing %d parameters", self.dim)

    def _setup_loss(self):
        from recovery.methods.gradient_descent.trajectory_loss import TrajectoryLoss
        self.loss_fn = TrajectoryLoss(self.target_data)

    def _decode_weights(self, w):
        """从优化向量解码回 weight_dict"""
        result = copy.deepcopy(self.perturbed_weight_dict)
        for pt in self.param_types:
            if pt not in self.param_slices:
                continue
            start, end = self.param_slices[pt]
            vec = w[start:end]
            if pt == 'syn':
                result['syn_weights'] = vec
            elif pt == 'gj':
                result['gj_weights'] = np.maximum(vec, 1e-9)
            elif pt == 'polarity':
                result['polarity'] = np.sign(vec)
                result['polarity'][result['polarity'] == 0] = 1.0
                result['syn_weights'] = np.abs(result['syn_weights']) * result['polarity']
            elif pt == 'wout':
                result['wout'] = vec.reshape(self.perturbed_weight_dict['wout'].shape)
            elif pt == 'input_gain':
                result['input_gain'] = vec
            elif pt == 'ion_channels':
                ic = self.perturbed_weight_dict.get('ion_channels', {})
                if isinstance(ic, dict) and 'matrix' in ic:
                    orig_shape = np.array(ic['matrix']).shape
                    # conductance 必须 >= 0
                    clamped = np.maximum(vec, 0.0)
                    result['ion_channels'] = dict(ic)
                    result['ion_channels']['matrix'] = clamped.reshape(orig_shape).tolist()
            elif pt == 'passive_params':
                pp = self.perturbed_weight_dict.get('passive_params', {})
                if isinstance(pp, dict) and 'matrix' in pp:
                    orig_shape = np.array(pp['matrix']).shape
                    reshaped = vec.reshape(orig_shape)
                    # Ra(col0), cm(col1), gpas(col2) 必须 > 0; epas(col3) 不限
                    reshaped[:, 0] = np.maximum(reshaped[:, 0], 1.0)    # Ra > 1
                    reshaped[:, 1] = np.maximum(reshaped[:, 1], 1e-4)   # cm > 0
                    reshaped[:, 2] = np.maximum(reshaped[:, 2], 1e-9)   # gpas > 0
                    result['passive_params'] = dict(pp)
                    result['passive_params']['matrix'] = reshaped.tolist()
                    result['passive_params'] = dict(pp)
                    result['passive_params']['matrix'] = vec.reshape(orig_shape).tolist()
        return result

    def _evaluate(self, w):
        """评估单个权重向量的 loss"""
        weight_dict = self._decode_weights(w)
        try:
            sim_result = self.simulator.run_with_custom_weights(
                weight_dict, self.param_types, n_steps=self.sim_steps)
        except Exception as e:
            logger.warning("Simulation failed: %s", e)
            return 1e6
        loss, _ = self.loss_fn.compute(sim_result)
        return loss

    def optimize(self, n_iterations=200):
        """
        运行 SPSA 优化。

        每步只需 2 次仿真（不管参数维度）。

        Returns:
            dict
        """
        rng = np.random.RandomState(self.config.get('seed', 42))

        for k in range(n_iterations):
            # 衰减调度
            a_k = self.a0 / (k + 1 + self.A) ** self.alpha
            c_k = self.c0 / (k + 1) ** self.gamma

            # Rademacher 随机扰动（±1）
            delta = rng.choice([-1, 1], size=self.dim).astype(np.float64)

            # 两次仿真评估
            w_plus = self.w + c_k * delta
            w_minus = self.w - c_k * delta

            loss_plus = self._evaluate(w_plus)
            loss_minus = self._evaluate(w_minus)

            # 梯度估计
            g_hat = (loss_plus - loss_minus) / (2.0 * c_k * delta)

            # 梯度裁剪（防止爆炸）
            g_norm = np.linalg.norm(g_hat)
            max_norm = self.config.get('max_grad_norm', 10.0)
            if g_norm > max_norm:
                g_hat = g_hat * max_norm / g_norm

            # 更新
            self.w = self.w - a_k * g_hat

            # 记录
            current_loss = min(loss_plus, loss_minus)
            self.loss_history.append(current_loss)

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights = self._decode_weights(
                    w_plus if loss_plus < loss_minus else w_minus)

            if k % 20 == 0:
                logger.info("SPSA iter %d: loss=%.6f (best=%.6f), a=%.6f, c=%.6f",
                           k, current_loss, self.best_loss, a_k, c_k)

        logger.info("SPSA finished: %d iterations, best_loss=%.6f",
                   n_iterations, self.best_loss)

        return {
            'recovered_weights': self.best_weights,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'n_iterations': n_iterations,
        }
