"""
三种恢复方法的统一抽象接口。

核心约束：恢复过程中只能使用运动轨迹数据，不能访问原始权重。
原始权重的对比评估由独立的 WeightRecoveryEvaluator 在恢复完成后进行。
"""
import os
import numpy as np
import logging
from abc import ABC, abstractmethod
from recovery.utils.io_utils import save_pickle, save_npy

logger = logging.getLogger(__name__)


class BaseWeightRecoverer(ABC):
    """
    权重恢复的抽象基类。

    所有恢复方法（梯度下降、RL、IRL）都实现这个接口。
    注意：优化过程中不接触原始权重，只使用目标轨迹数据。
    """

    def __init__(self, target_data, perturbed_weight_dict, param_types, config=None):
        """
        Args:
            target_data: dict, 目标数据（正常轨迹/肌肉信号等）
                - 'muscle_activation': np.ndarray, shape (T, 96)
                - 'motor_neuron_voltage': np.ndarray, shape (T, n_output)
                - 'rel_x', 'rel_z': np.ndarray, shape (T, 17) (如果有完整轨迹)
            perturbed_weight_dict: dict, 打乱后的权重（优化起点）
            param_types: list of str, 要恢复的参数类型 ['syn', 'gj', 'polarity', 'wout']
            config: dict, 方法特定的配置
        """
        self.target_data = target_data
        self.perturbed_weight_dict = perturbed_weight_dict
        self.param_types = param_types
        self.config = config or {}

        self.loss_history = []
        self.best_loss = float('inf')
        self.best_weights = None

    @abstractmethod
    def optimize(self, n_iterations):
        """
        执行优化。

        Args:
            n_iterations: int, 迭代次数

        Returns:
            dict: {
                'recovered_weights': dict, 恢复的权重
                'loss_history': list of float,
                'best_loss': float,
                'n_iterations': int,
            }
        """
        pass

    def evaluate_weights(self, weight_dict):
        """
        用给定权重运行仿真并评估。

        子类可以覆盖此方法以使用不同的仿真器。

        Args:
            weight_dict: dict, 要评估的权重

        Returns:
            dict: 仿真结果 + 评估指标
        """
        raise NotImplementedError("子类需要实现 evaluate_weights")

    def save_checkpoint(self, save_dir, iteration):
        """保存检查点"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'iteration': iteration,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'best_weights': self.best_weights,
        }
        save_pickle(checkpoint, os.path.join(save_dir, f"checkpoint_{iteration}.pkl"))

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        from recovery.utils.io_utils import load_pickle
        checkpoint = load_pickle(checkpoint_path)
        self.loss_history = checkpoint['loss_history']
        self.best_loss = checkpoint['best_loss']
        self.best_weights = checkpoint['best_weights']
        return checkpoint['iteration']


class WeightRecoveryEvaluator:
    """
    恢复后的权重评估器。

    独立于优化过程，仅在恢复完成后用于对比原始权重。
    这样保证优化过程中不可能接触到原始权重。
    """

    @staticmethod
    def compute_weight_error(recovered_weights, original_weights, param_types):
        """
        计算恢复权重与原始权重的误差。

        Args:
            recovered_weights: dict, 恢复的权重
            original_weights: dict, 原始权重
            param_types: list of str

        Returns:
            dict: 各类参数的误差
        """
        errors = {}
        for pt in param_types:
            if pt == 'syn' and 'syn_weights' in recovered_weights:
                orig = original_weights['syn_weights']
                recov = recovered_weights['syn_weights']
                errors['syn_mse'] = float(np.mean((orig - recov) ** 2))
                errors['syn_corr'] = float(np.corrcoef(orig, recov)[0, 1])
            elif pt == 'gj' and 'gj_weights' in recovered_weights:
                orig = original_weights['gj_weights']
                recov = recovered_weights['gj_weights']
                errors['gj_mse'] = float(np.mean((orig - recov) ** 2))
                errors['gj_corr'] = float(np.corrcoef(orig, recov)[0, 1])
            elif pt == 'polarity' and 'polarity' in recovered_weights:
                orig = original_weights['polarity']
                recov = recovered_weights['polarity']
                errors['polarity_accuracy'] = float(np.mean(orig == recov))
            elif pt == 'wout' and 'wout' in recovered_weights:
                orig = original_weights['wout']
                recov = recovered_weights['wout']
                errors['wout_mse'] = float(np.mean((orig - recov) ** 2))
                errors['wout_corr'] = float(
                    np.corrcoef(orig.flatten(), recov.flatten())[0, 1])
        return errors
