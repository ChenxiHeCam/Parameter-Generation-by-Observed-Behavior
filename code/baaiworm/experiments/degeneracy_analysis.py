"""
论文 2：退化性分析。

对同一打乱权重，用多个随机种子运行恢复，
分析解空间几何和差异脆弱性。
"""
import os
import sys
import json
import numpy as np
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recovery.utils.io_utils import save_pickle, load_pickle, save_json
from recovery.utils.logging_utils import setup_logger
from recovery.evaluation.behavior_metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def collect_degenerate_solutions(phase2_dir, param_type, n_seeds=20,
                                 behavior_threshold=0.5):
    """
    收集多组通过行为阈值的恢复权重。

    Args:
        phase2_dir: Phase 2 输出目录
        param_type: 参数类型
        n_seeds: 恢复种子数
        behavior_threshold: zigzag_score 阈值

    Returns:
        list of dict: 每个元素是一组恢复权重 + 指标
    """
    solutions = []
    for seed in range(n_seeds):
        result_path = os.path.join(
            phase2_dir, f'{param_type}_cmaes_seed{seed}', 'recovery_result.pkl')
        if not os.path.exists(result_path):
            continue
        result = load_pickle(result_path)
        if result.get('best_loss', float('inf')) > 1.0:
            continue
        solutions.append({
            'seed': seed,
            'weights': result['recovered_weights'],
            'loss': result['best_loss'],
        })
    logger.info("Collected %d solutions for %s", len(solutions), param_type)
    return solutions


def analyze_solution_space(solutions, param_type):
    """
    分析解空间几何。

    Returns:
        dict: PCA 结果、权重统计、退化维度
    """
    from sklearn.decomposition import PCA

    # 提取权重向量
    key_map = {'syn': 'syn_weights', 'gj': 'gj_weights',
               'wout': 'wout', 'polarity': 'polarity'}
    key = key_map.get(param_type, param_type + '_weights')

    vectors = []
    for sol in solutions:
        w = sol['weights'].get(key)
        if w is None:
            continue
        vectors.append(w.flatten())

    if len(vectors) < 3:
        return {'error': 'too few solutions', 'n_solutions': len(vectors)}

    W = np.array(vectors)
    n_solutions, n_params = W.shape

    # 基本统计
    mean_w = np.mean(W, axis=0)
    std_w = np.std(W, axis=0)
    cv = std_w / (np.abs(mean_w) + 1e-10)  # 变异系数

    # PCA
    n_components = min(n_solutions - 1, 20, n_params)
    pca = PCA(n_components=n_components)
    W_pca = pca.fit_transform(W)

    # 退化维度：方差解释比 < 1% 的维度
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_effective = int(np.searchsorted(cumvar, 0.95)) + 1

    # 权重间相关性
    if n_params < 5000:
        corr_matrix = np.corrcoef(W.T)
        high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.8) - n_params
        high_corr_pairs //= 2
    else:
        corr_matrix = None
        high_corr_pairs = -1

    # 哪些参数被精确恢复（低变异系数）
    well_recovered = np.sum(cv < 0.1)
    degenerate = np.sum(cv > 0.5)

    results = {
        'n_solutions': n_solutions,
        'n_params': n_params,
        'n_effective_dims': n_effective,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': cumvar.tolist(),
        'well_recovered_params': int(well_recovered),
        'degenerate_params': int(degenerate),
        'high_corr_pairs': int(high_corr_pairs),
        'mean_cv': float(np.mean(cv)),
        'pca_coords': W_pca.tolist(),
        'param_cv': cv.tolist(),
    }
    return results


def test_differential_vulnerability(solutions, param_type, simulator,
                                     perturbation_levels=[0.05, 0.10, 0.15, 0.20],
                                     n_sim_steps=200):
    """
    差异脆弱性测试：对每组恢复权重施加相同扰动，测量哪些先失去行为。

    Returns:
        list of dict: 每个解在每个扰动水平下的行为指标
    """
    from recovery.perturbation.weight_perturber import apply_perturbation

    results = []
    for i, sol in enumerate(solutions):
        sol_results = {'seed': sol['seed'], 'levels': []}
        for level in perturbation_levels:
            perturbed = apply_perturbation(
                sol['weights'], param_type, level, seed=999)

            try:
                traj = simulator.run_with_custom_weights(
                    perturbed, [param_type], n_steps=n_sim_steps)
                metrics = compute_all_metrics(traj)
            except Exception as e:
                metrics = {'error': str(e), 'zigzag_score': 0}

            sol_results['levels'].append({
                'noise_level': level,
                'zigzag_score': metrics.get('zigzag_score', 0),
                'mean_speed': metrics.get('mean_speed', 0),
                'chemotaxis_index': metrics.get('chemotaxis_index', 0),
            })
        results.append(sol_results)
        logger.info("Vulnerability test: solution %d/%d done", i + 1, len(solutions))

    return results


def run_degeneracy_analysis(config_path=None, param_type='syn'):
    """主函数"""
    from recovery.experiments.phase1_data_generation import load_config

    config = load_config(config_path)
    phase2_dir = os.path.join(config['output_dir'], 'phase2')
    output_dir = os.path.join(config['output_dir'], 'degeneracy', param_type)
    os.makedirs(output_dir, exist_ok=True)
    setup_logger('degeneracy', output_dir)

    logger.info("=" * 60)
    logger.info("Degeneracy Analysis for %s", param_type)
    logger.info("=" * 60)

    # Step 1: 收集解
    solutions = collect_degenerate_solutions(phase2_dir, param_type)
    if len(solutions) < 3:
        logger.error("Need at least 3 solutions, got %d", len(solutions))
        return

    # Step 2: 解空间分析
    logger.info("Analyzing solution space...")
    space_analysis = analyze_solution_space(solutions, param_type)
    save_json(space_analysis, os.path.join(output_dir, 'solution_space.json'))
    logger.info("Effective dims: %d/%d, well-recovered: %d, degenerate: %d",
               space_analysis.get('n_effective_dims', 0),
               space_analysis.get('n_params', 0),
               space_analysis.get('well_recovered_params', 0),
               space_analysis.get('degenerate_params', 0))

    # Step 3: 差异脆弱性（需要仿真器）
    logger.info("Differential vulnerability test...")
    # TODO: 初始化仿真器后运行
    # vulnerability = test_differential_vulnerability(solutions, param_type, simulator)
    # save_json(vulnerability, os.path.join(output_dir, 'vulnerability.json'))

    logger.info("Degeneracy analysis complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--param', type=str, default='syn')
    args = parser.parse_args()
    run_degeneracy_analysis(args.config, args.param)
