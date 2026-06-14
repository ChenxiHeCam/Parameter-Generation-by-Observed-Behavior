"""
Phase 2: 单参数恢复实验。

对每种参数类型，用三种方法分别恢复。
"""
import os
import sys
import yaml
import logging
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recovery.utils.io_utils import load_pickle, save_pickle, save_json
from recovery.utils.logging_utils import setup_logger
from recovery.evaluation.behavior_metrics import compute_all_metrics
from recovery.evaluation.trajectory_comparison import compute_all_distances
from recovery.evaluation.visualization import (
    plot_trajectory_comparison, plot_weight_comparison,
    plot_recovery_loss, plot_metric_dashboard
)
from recovery.simulation.batch_runner import ExperimentRunner

logger = logging.getLogger(__name__)

DEFAULT_INPUT_NEURONS = [
    "AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR",
    "ALNL", "ALNR", "PLML", "PHAL", "PHAR",
    "URYDL", "URYDR", "URYVL", "URYVR"
]


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_recovery(param_type, method, seed, config, phase1_dir, output_dir):
    """
    运行单个恢复实验。

    Args:
        param_type: str, 'syn' / 'gj' / 'polarity' / 'wout'
        method: str, 'gd' / 'rl' / 'irl'
        seed: int, 使用哪个打乱种子
        config: dict
        phase1_dir: str, Phase 1 输出目录
        output_dir: str, 本次实验输出目录

    Returns:
        dict, 实验结果
    """
    os.makedirs(output_dir, exist_ok=True)
    project_root = config['project_root']

    # 加载数据
    normal_weights = load_pickle(os.path.join(phase1_dir, 'normal_weights.pkl'))
    normal_traj = load_pickle(os.path.join(phase1_dir, 'normal_trajectory.pkl'))
    perturbed_weights = load_pickle(
        os.path.join(phase1_dir, f'perturbed_{param_type}', f'weights_seed{seed}.pkl'))

    # 初始化仿真器
    from recovery.simulation.headless_sim import HeadlessSimulator
    abs_circuit_path = os.path.join(project_root, config['data']['abs_circuit'])
    wout_path = os.path.join(project_root, config['data']['wout'])
    config_json_path = os.path.join(project_root, config['data']['config_json'])

    simulator = HeadlessSimulator(
        abs_circuit_path=abs_circuit_path,
        wout_path=wout_path,
        config_json_path=config_json_path,
    )

    # 选择恢复方法
    if method == 'gd' and param_type == 'polarity':
        logger.warning(f"Skipping GD for polarity (discrete parameter, not differentiable)")
        return {'param_type': param_type, 'method': method, 'seed': seed,
                'best_loss': None, 'skipped': True}

    if method == 'gd' and param_type == 'wout':
        from recovery.methods.gradient_descent.wout_gd_optimizer import WoutGDRecoverer
        recoverer = WoutGDRecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            simulator=simulator,
            config=config.get('wout_gd', {}),
        )
        n_iter = config.get('wout_gd', {}).get('n_iterations', 200)

    elif method == 'gd':
        from recovery.methods.gradient_descent.gd_optimizer import GradientDescentRecoverer
        abs_circuit_perturbed = os.path.join(
            phase1_dir, f'perturbed_{param_type}', f'circuit_seed{seed}.pkl')
        recoverer = GradientDescentRecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            config={**config.get('gradient_descent', {}), 'checkpoint_dir': output_dir},
            abs_circuit_path=abs_circuit_perturbed,
            input_names=DEFAULT_INPUT_NEURONS,
        )
        n_iter = config.get('gradient_descent', {}).get('n_iterations', 100)

    elif method == 'rl':
        from recovery.methods.reinforcement_learning.ppo_trainer import PPOWeightRecoverer
        recoverer = PPOWeightRecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            simulator=simulator,
            config=config.get('reinforcement_learning', {}),
        )
        n_iter = config.get('reinforcement_learning', {}).get('n_iterations', 100000)

    elif method == 'irl':
        from recovery.methods.inverse_rl.irl_optimizer import IRLWeightRecoverer
        recoverer = IRLWeightRecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            simulator=simulator,
            config=config.get('inverse_rl', {}),
        )
        n_iter = config.get('inverse_rl', {}).get('n_iterations', 50000)

    elif method == 'cmaes':
        from recovery.methods.cmaes_optimizer import CMAESRecoverer
        recoverer = CMAESRecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            simulator=simulator,
            config=config.get('cmaes', {}),
        )
        n_iter = config.get('cmaes', {}).get('max_generations', 200)

    elif method == 'spsa':
        from recovery.methods.spsa_optimizer import SPSARecoverer
        recoverer = SPSARecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            simulator=simulator,
            config=config.get('spsa', {'sim_steps': 200}),
        )
        n_iter = config.get('spsa', {}).get('n_iterations', 200)

    elif method == 'sbi':
        from recovery.methods.sbi_optimizer import SBIRecoverer
        recoverer = SBIRecoverer(
            target_data=normal_traj,
            perturbed_weight_dict=perturbed_weights,
            param_types=[param_type],
            simulator=simulator,
            config=config.get('sbi', {'n_simulations': 500, 'n_rounds': 3, 'sim_steps': 200}),
        )
        n_iter = None  # SBI 内部管理迭代

    else:
        raise ValueError(f"Unknown method: {method}")

    # 执行恢复
    logger.info(f"Running {method} recovery for {param_type} (seed={seed})...")
    result = recoverer.optimize(n_iterations=n_iter)

    # 保存结果
    save_pickle(result, os.path.join(output_dir, 'recovery_result.pkl'))

    # 恢复后评估（独立于优化过程，此时才加载原始权重对比）
    from recovery.methods.base_optimizer import WeightRecoveryEvaluator
    weight_error = WeightRecoveryEvaluator.compute_weight_error(
        result['recovered_weights'], normal_weights, [param_type])
    save_json(weight_error, os.path.join(output_dir, 'weight_error.json'))

    # 可视化
    if result.get('loss_history'):
        plot_recovery_loss(result['loss_history'], f"{method}_{param_type}",
                           os.path.join(output_dir, 'loss_curve.png'))

    logger.info(f"Recovery complete: best_loss={result.get('best_loss', 'N/A')}, "
                f"weight_error={weight_error}")

    return {
        'param_type': param_type,
        'method': method,
        'seed': seed,
        'best_loss': result.get('best_loss'),
        'weight_error': weight_error,
    }


def run_phase2(config_path=None):
    """Phase 2 主函数"""
    config = load_config(config_path)
    output_dir = os.path.join(config['output_dir'], 'phase2')
    phase1_dir = os.path.join(config['output_dir'], 'phase1')

    setup_logger('phase2', output_dir)
    logger.info("=" * 60)
    logger.info("Phase 2: Single Parameter Recovery")
    logger.info("=" * 60)

    runner = ExperimentRunner(output_dir)

    param_types = ['syn', 'gj', 'polarity', 'wout']
    methods = ['cmaes', 'spsa', 'sbi', 'gd', 'rl']

    experiments = []
    for param_type in param_types:
        for method in methods:
            for seed in [0]:  # 先用 seed=0 验证
                exp_id = f"{param_type}_{method}_seed{seed}"
                exp_output = os.path.join(output_dir, exp_id)

                def make_run_fn(pt=param_type, m=method, s=seed, eo=exp_output):
                    return lambda: run_single_recovery(
                        pt, m, s, config, phase1_dir, eo)

                experiments.append((exp_id, make_run_fn()))

    runner.run_batch(experiments)
    runner.summary()

    logger.info("Phase 2 complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    run_phase2(args.config)
