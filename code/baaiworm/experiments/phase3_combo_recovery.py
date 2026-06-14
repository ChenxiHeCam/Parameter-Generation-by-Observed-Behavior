"""
Phase 3: 组合参数恢复实验。

同时打乱多类参数，然后用渐进式策略恢复。
策略：按重要性排序（wout > syn > gj > polarity），逐个恢复并固定。
"""
import os
import sys
import copy
import yaml
import logging
import numpy as np
from itertools import combinations

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recovery.perturbation.weight_perturber import apply_multi_perturbation
from recovery.perturbation.perturbed_circuit_builder import build_perturbed_circuit
from recovery.utils.io_utils import load_pickle, save_pickle, save_json, load_json
from recovery.utils.logging_utils import setup_logger
from recovery.simulation.batch_runner import ExperimentRunner
from recovery.methods.base_optimizer import WeightRecoveryEvaluator

logger = logging.getLogger(__name__)

PARAM_TYPES = ['syn', 'gj', 'polarity', 'wout']
RECOVERY_PRIORITY = ['wout', 'syn', 'gj', 'polarity']
METHOD_FOR_PARAM = {
    'syn': 'gd',
    'gj': 'gd',
    'polarity': 'rl',
    'wout': 'gd',
}


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_combo_experiments():
    """生成所有组合实验配置"""
    combos = []
    for r in range(2, 5):
        combos.extend(list(combinations(PARAM_TYPES, r)))
    return combos


def run_combo_recovery(combo, config, phase1_dir, output_dir):
    """
    渐进式组合恢复：按优先级逐个恢复参数。

    Args:
        combo: tuple of str, 要同时打乱的参数类型
        config: dict
        phase1_dir: str
        output_dir: str

    Returns:
        dict
    """
    os.makedirs(output_dir, exist_ok=True)
    project_root = config['project_root']

    normal_weights = load_pickle(os.path.join(phase1_dir, 'normal_weights.pkl'))
    normal_traj = load_pickle(os.path.join(phase1_dir, 'normal_trajectory.pkl'))
    thresholds = load_json(os.path.join(phase1_dir, 'thresholds.json'))

    # 组合打乱
    reduction = config['perturbation']['combo_reduction']
    param_configs = []
    for pt in combo:
        threshold = thresholds.get(pt, 0.25)
        noise_level = threshold * reduction
        param_configs.append((pt, noise_level))

    logger.info(f"Combo perturbation: {param_configs}")
    perturbed_weights = apply_multi_perturbation(
        normal_weights, param_configs, seed=42)
    save_pickle(perturbed_weights, os.path.join(output_dir, 'perturbed_weights.pkl'))

    # 渐进恢复：按优先级排序
    ordered_params = [pt for pt in RECOVERY_PRIORITY if pt in combo]
    current_weights = copy.deepcopy(perturbed_weights)
    results = {}

    from recovery.simulation.headless_sim import HeadlessSimulator
    abs_circuit_path = os.path.join(project_root, config['data']['abs_circuit'])
    wout_path = os.path.join(project_root, config['data']['wout'])
    config_json_path = os.path.join(project_root, config['data']['config_json'])

    simulator = HeadlessSimulator(
        abs_circuit_path=abs_circuit_path,
        wout_path=wout_path,
        config_json_path=config_json_path,
    )

    for pt in ordered_params:
        method = METHOD_FOR_PARAM[pt]
        logger.info(f"  Recovering {pt} with {method}...")
        step_dir = os.path.join(output_dir, f'step_{pt}')
        os.makedirs(step_dir, exist_ok=True)

        try:
            if method == 'gd' and pt == 'wout':
                from recovery.methods.gradient_descent.wout_gd_optimizer import WoutGDRecoverer
                recoverer = WoutGDRecoverer(
                    target_data=normal_traj,
                    perturbed_weight_dict=current_weights,
                    param_types=[pt],
                    config=config.get('wout_gd', {}),
                )
                n_iter = config.get('wout_gd', {}).get('n_iterations', 2000)

            elif method == 'gd':
                from recovery.methods.gradient_descent.gd_optimizer import GradientDescentRecoverer
                recoverer = GradientDescentRecoverer(
                    target_data=normal_traj,
                    perturbed_weight_dict=current_weights,
                    param_types=[pt],
                    config={**config.get('gradient_descent', {}), 'checkpoint_dir': step_dir},
                )
                n_iter = config.get('gradient_descent', {}).get('n_iterations', 100)

            elif method == 'rl':
                from recovery.methods.reinforcement_learning.ppo_trainer import PPOWeightRecoverer
                recoverer = PPOWeightRecoverer(
                    target_data=normal_traj,
                    perturbed_weight_dict=current_weights,
                    param_types=[pt],
                    simulator=simulator,
                    config=config.get('reinforcement_learning', {}),
                )
                n_iter = config.get('reinforcement_learning', {}).get('n_iterations', 100000)
            else:
                logger.warning(f"  Skipping {pt} (no suitable method)")
                results[pt] = {'status': 'skipped'}
                continue

            result = recoverer.optimize(n_iterations=n_iter)
            recovered = result.get('recovered_weights', current_weights)

            # 将恢复的参数合并到 current_weights（固定已恢复的，继续恢复下一个）
            if pt == 'syn' and 'syn_weights' in recovered:
                current_weights['syn_weights'] = recovered['syn_weights']
            elif pt == 'gj' and 'gj_weights' in recovered:
                current_weights['gj_weights'] = recovered['gj_weights']
            elif pt == 'wout' and 'wout' in recovered:
                current_weights['wout'] = recovered['wout']
            elif pt == 'polarity' and 'polarity' in recovered:
                current_weights['polarity'] = recovered['polarity']

            save_pickle(result, os.path.join(step_dir, 'recovery_result.pkl'))
            results[pt] = {
                'status': 'completed',
                'method': method,
                'best_loss': result.get('best_loss'),
            }
            logger.info(f"  {pt} recovered: loss={result.get('best_loss', 'N/A')}")

        except Exception as e:
            logger.error(f"  {pt} recovery failed: {e}")
            results[pt] = {'status': 'failed', 'error': str(e)}

    # 最终评估
    weight_error = WeightRecoveryEvaluator.compute_weight_error(
        current_weights, normal_weights, list(combo))

    final_result = {
        'combo': list(combo),
        'param_configs': [(pt, float(nl)) for pt, nl in param_configs],
        'per_param_results': results,
        'weight_error': weight_error,
    }
    save_json(final_result, os.path.join(output_dir, 'combo_result.json'))
    save_pickle(current_weights, os.path.join(output_dir, 'recovered_weights.pkl'))

    return final_result


def run_phase3(config_path=None):
    """Phase 3 主函数"""
    config = load_config(config_path)
    output_dir = os.path.join(config['output_dir'], 'phase3')
    phase1_dir = os.path.join(config['output_dir'], 'phase1')

    setup_logger('phase3', output_dir)
    logger.info("=" * 60)
    logger.info("Phase 3: Combination Recovery")
    logger.info("=" * 60)

    combos = generate_combo_experiments()
    logger.info(f"Total combinations: {len(combos)}")
    for i, combo in enumerate(combos):
        logger.info(f"  {i+1}. {'+'.join(combo)}")

    runner = ExperimentRunner(output_dir)
    experiments = []

    for combo in combos:
        combo_name = '+'.join(combo)
        exp_id = combo_name
        exp_output = os.path.join(output_dir, exp_id)

        def make_run_fn(c=combo, eo=exp_output):
            return lambda: run_combo_recovery(c, config, phase1_dir, eo)

        experiments.append((exp_id, make_run_fn()))

    runner.run_batch(experiments)
    runner.summary()

    logger.info("Phase 3 complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    run_phase3(args.config)
