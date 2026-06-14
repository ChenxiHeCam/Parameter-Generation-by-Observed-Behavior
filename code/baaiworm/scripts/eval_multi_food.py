"""
用已恢复的参数 + 不同食物位置跑仿真，获取 error bars。
不重新优化，只评估。每次仿真 ~5分钟，总共 ~2小时。
"""
import sys
import os
import json
import logging
import numpy as np
import pickle
import time

sys.path.insert(0, '/root/BAAIWorm-main/build_headless/build')
sys.path.insert(0, '/root/BAAIWorm-main')

from recovery.utils.io_utils import load_pickle, save_json
from recovery.evaluation.behavior_metrics import (
    detect_zigzag, compute_forward_speed, compute_muscle_alternation
)
from recovery.simulation.closed_loop_simulator import ClosedLoopSimulator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('eval_food')

project_root = '/root/BAAIWorm-main'
build_dir = '/root/BAAIWorm-main/build_headless/build'
phase1_dir = '/root/BAAIWorm-main/recovery/output/phase1'
phase2_dir = '/root/BAAIWorm-main/recovery/output/phase2'
output_dir = '/root/BAAIWorm-main/recovery/output/eval_food'
os.makedirs(output_dir, exist_ok=True)

abs_circuit_path = os.path.join(
    project_root,
    'eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_abscircuit.pkl')
wout_path = os.path.join(
    project_root,
    'eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_wout.pkl')

normal_weights = load_pickle(os.path.join(phase1_dir, 'normal_weights.pkl'))

# 5个不同食物位置
food_locations = [
    np.array([1.8275, -0.0276, -0.3082], dtype=np.float32),   # 默认
    np.array([1.5, -0.5, -0.8], dtype=np.float32),
    np.array([2.0, 0.3, 0.2], dtype=np.float32),
    np.array([1.0, -0.2, -1.5], dtype=np.float32),
    np.array([2.5, 0.0, 0.5], dtype=np.float32),
]

# 每个参数类型的最佳恢复结果
best_experiments = {
    'syn': 'syn_es',
    'wout': 'wout_es',
    'gj': 'gj_es',
    'ion_channels': 'ion_channels_es',
    'passive_params': 'passive_params_es',
}

# 也需要手动计算 ion_channels 和 passive_params 的权重相关
def compute_weight_corr(recovered_weights, param_type):
    if param_type in ['ion_channels', 'passive_params']:
        key = param_type
        rec = recovered_weights.get(key, {})
        norm = normal_weights.get(key, {})
        if isinstance(rec, dict) and 'matrix' in rec and isinstance(norm, dict) and 'matrix' in norm:
            rec_m = np.array(rec['matrix']).flatten()
            norm_m = np.array(norm['matrix']).flatten()
            n = min(len(rec_m), len(norm_m))
            if n > 0:
                return float(np.corrcoef(rec_m[:n], norm_m[:n])[0, 1])
    return None


def eval_one(param_type, recovered_weights, food_loc, food_idx):
    """用恢复的参数 + 指定食物位置跑一次仿真"""
    simulator = ClosedLoopSimulator(
        project_root, build_dir, abs_circuit_path, wout_path,
        n_init_steps=30, n_sim_steps=100,
        food_location=food_loc)

    metrics = {}

    # baseline（正常权重 + 这个食物位置）
    try:
        bl_traj = simulator.run_with_custom_weights(
            normal_weights, [param_type], n_steps=100)
        bl_zz = detect_zigzag(bl_traj['rel_x'], bl_traj['rel_z'])
        bl_sp = compute_forward_speed(bl_traj['rel_x'], bl_traj['rel_z'])
        metrics['baseline_zigzag'] = bl_zz['zigzag_score']
        metrics['baseline_speed'] = bl_sp['mean_speed']
        metrics['baseline_reversals'] = bl_zz['n_reversals']
    except Exception as e:
        logger.error('  Baseline failed: %s', e)
        return {'error': f'baseline: {e}'}

    # 恢复后的参数
    try:
        rec_traj = simulator.run_with_custom_weights(
            recovered_weights, [param_type], n_steps=100)
        rec_zz = detect_zigzag(rec_traj['rel_x'], rec_traj['rel_z'])
        rec_sp = compute_forward_speed(rec_traj['rel_x'], rec_traj['rel_z'])
        metrics['recovered_zigzag'] = rec_zz['zigzag_score']
        metrics['recovered_speed'] = rec_sp['mean_speed']
        metrics['recovered_reversals'] = rec_zz['n_reversals']

        if rec_traj.get('muscle_activation') is not None:
            muscle = np.array(rec_traj['muscle_activation'])
            ma = compute_muscle_alternation(muscle)
            metrics['recovered_dv_corr'] = ma['dorsal_ventral_corr']

            if bl_traj.get('muscle_activation') is not None:
                bl_muscle = np.array(bl_traj['muscle_activation'])
                T = min(len(bl_muscle), len(muscle))
                metrics['muscle_signal_corr'] = float(
                    np.corrcoef(bl_muscle[:T].flatten(), muscle[:T].flatten())[0, 1])
    except Exception as e:
        logger.error('  Recovery eval failed: %s', e)
        metrics['error'] = str(e)

    # 扰动后的参数（未恢复）
    try:
        pw = load_pickle(os.path.join(phase1_dir, f'perturbed_{param_type}', 'weights_seed0.pkl'))
        pert_traj = simulator.run_with_custom_weights(
            pw, [param_type], n_steps=100)
        pert_zz = detect_zigzag(pert_traj['rel_x'], pert_traj['rel_z'])
        pert_sp = compute_forward_speed(pert_traj['rel_x'], pert_traj['rel_z'])
        metrics['perturbed_zigzag'] = pert_zz['zigzag_score']
        metrics['perturbed_speed'] = pert_sp['mean_speed']
    except Exception as e:
        logger.warning('  Perturbed eval failed: %s', e)

    return metrics


# 主循环
all_results = {}

for param_type, exp_name in best_experiments.items():
    # 加载已恢复的权重
    result_path = os.path.join(phase2_dir, exp_name, 'result.pkl')
    if not os.path.exists(result_path):
        logger.warning('%s: no result at %s', param_type, result_path)
        continue

    result = load_pickle(result_path)
    recovered_weights = result.get('recovered_weights')
    if not recovered_weights:
        logger.warning('%s: no recovered weights', param_type)
        continue

    # 计算权重相关（补 ion_channels 和 passive_params）
    wt_corr = compute_weight_corr(recovered_weights, param_type)

    logger.info('=' * 60)
    logger.info('Evaluating %s (from %s, loss=%.4f) across %d food locations',
                param_type, exp_name, result.get('best_loss', 999), len(food_locations))

    food_results = []
    for fi, food_loc in enumerate(food_locations):
        logger.info('--- %s / food%d ---', param_type, fi)
        t0 = time.time()
        m = eval_one(param_type, recovered_weights, food_loc, fi)
        m['food_idx'] = fi
        m['food_location'] = food_loc.tolist()
        m['time_seconds'] = time.time() - t0
        food_results.append(m)

        if 'error' not in m:
            logger.info('  BL: zz=%.4f spd=%.5f | REC: zz=%.4f spd=%.5f | PERT: zz=%.4f',
                        m.get('baseline_zigzag', 0), m.get('baseline_speed', 0),
                        m.get('recovered_zigzag', 0), m.get('recovered_speed', 0),
                        m.get('perturbed_zigzag', 0))

    # 汇总统计
    valid = [m for m in food_results if 'error' not in m]
    if valid:
        stats = {
            'param_type': param_type,
            'source_experiment': exp_name,
            'source_loss': result.get('best_loss', 999),
            'weight_corr': wt_corr,
            'n_foods': len(valid),
            'baseline_zigzag_mean': float(np.mean([m['baseline_zigzag'] for m in valid])),
            'baseline_zigzag_std': float(np.std([m['baseline_zigzag'] for m in valid])),
            'baseline_speed_mean': float(np.mean([m['baseline_speed'] for m in valid])),
            'baseline_speed_std': float(np.std([m['baseline_speed'] for m in valid])),
            'recovered_zigzag_mean': float(np.mean([m['recovered_zigzag'] for m in valid])),
            'recovered_zigzag_std': float(np.std([m['recovered_zigzag'] for m in valid])),
            'recovered_speed_mean': float(np.mean([m['recovered_speed'] for m in valid])),
            'recovered_speed_std': float(np.std([m['recovered_speed'] for m in valid])),
        }
        if any('perturbed_zigzag' in m for m in valid):
            pert_valid = [m for m in valid if 'perturbed_zigzag' in m]
            stats['perturbed_zigzag_mean'] = float(np.mean([m['perturbed_zigzag'] for m in pert_valid]))
            stats['perturbed_zigzag_std'] = float(np.std([m['perturbed_zigzag'] for m in pert_valid]))
            stats['perturbed_speed_mean'] = float(np.mean([m['perturbed_speed'] for m in pert_valid]))
            stats['perturbed_speed_std'] = float(np.std([m['perturbed_speed'] for m in pert_valid]))
        if any('recovered_dv_corr' in m for m in valid):
            stats['recovered_dv_mean'] = float(np.mean([m['recovered_dv_corr'] for m in valid if 'recovered_dv_corr' in m]))
            stats['recovered_dv_std'] = float(np.std([m['recovered_dv_corr'] for m in valid if 'recovered_dv_corr' in m]))

        all_results[param_type] = stats
        logger.info('STATS %s: rec_zz=%.4f+/-%.4f rec_spd=%.5f+/-%.5f',
                    param_type,
                    stats['recovered_zigzag_mean'], stats['recovered_zigzag_std'],
                    stats['recovered_speed_mean'], stats['recovered_speed_std'])

    save_json(food_results, os.path.join(output_dir, f'{param_type}_food_results.json'))

save_json(all_results, os.path.join(output_dir, 'summary_with_errorbars.json'))
logger.info('Done! Summary saved.')
