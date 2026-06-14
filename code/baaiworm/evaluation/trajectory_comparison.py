"""
轨迹比较：距离度量和统计检验。
"""
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def trajectory_mse(traj_a, traj_b):
    """
    两条轨迹的均方误差。

    Args:
        traj_a, traj_b: dict with 'rel_x', 'rel_z', shape (T, 17)

    Returns:
        float
    """
    T = min(traj_a['rel_x'].shape[0], traj_b['rel_x'].shape[0])
    diff_x = traj_a['rel_x'][:T] - traj_b['rel_x'][:T]
    diff_z = traj_a['rel_z'][:T] - traj_b['rel_z'][:T]
    return float(np.mean(diff_x**2 + diff_z**2))


def dtw_distance(seq_a, seq_b):
    """
    Dynamic Time Warping 距离（简化版，用于 1D 或 2D 序列）。

    Args:
        seq_a, seq_b: np.ndarray, shape (T_a, D) 和 (T_b, D)

    Returns:
        float, DTW 距离
    """
    n, m = len(seq_a), len(seq_b)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i-1] - seq_b[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    return float(dtw_matrix[n, m])


def frechet_distance_approx(curve_a, curve_b):
    """
    Frechet 距离的近似（使用离散 Frechet）。

    Args:
        curve_a, curve_b: np.ndarray, shape (T, 2), 质心轨迹

    Returns:
        float
    """
    n, m = len(curve_a), len(curve_b)
    ca = np.full((n, m), -1.0)

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = np.linalg.norm(curve_a[i] - curve_b[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j-1), d)
        else:
            ca[i, j] = max(min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)), d)
        return ca[i, j]

    return float(_c(n-1, m-1))


def muscle_signal_correlation(muscle_a, muscle_b):
    """
    两组肌肉信号的 Pearson 相关系数（逐肌肉取平均）。

    Args:
        muscle_a, muscle_b: np.ndarray, shape (T, 96)

    Returns:
        float, 平均相关系数
    """
    T = min(len(muscle_a), len(muscle_b))
    corrs = []
    for i in range(muscle_a.shape[1]):
        a = muscle_a[:T, i]
        b = muscle_b[:T, i]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        corrs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0


def compute_all_distances(traj_a, traj_b):
    """
    计算两条轨迹之间的所有距离度量。

    Args:
        traj_a, traj_b: dict with 'rel_x', 'rel_z', optionally 'muscle_activation'

    Returns:
        dict
    """
    result = {}
    result['mse'] = trajectory_mse(traj_a, traj_b)

    # 质心轨迹
    T = min(traj_a['rel_x'].shape[0], traj_b['rel_x'].shape[0])
    center_a = np.column_stack([
        np.mean(traj_a['rel_x'][:T], axis=1),
        np.mean(traj_a['rel_z'][:T], axis=1)
    ])
    center_b = np.column_stack([
        np.mean(traj_b['rel_x'][:T], axis=1),
        np.mean(traj_b['rel_z'][:T], axis=1)
    ])

    # DTW（对长序列用下采样加速）
    step = max(1, T // 200)
    result['dtw'] = dtw_distance(center_a[::step], center_b[::step])
    result['frechet'] = frechet_distance_approx(center_a[::step], center_b[::step])
    result['hausdorff'] = float(max(
        directed_hausdorff(center_a, center_b)[0],
        directed_hausdorff(center_b, center_a)[0]
    ))

    if 'muscle_activation' in traj_a and 'muscle_activation' in traj_b:
        result['muscle_corr'] = muscle_signal_correlation(
            np.array(traj_a['muscle_activation']),
            np.array(traj_b['muscle_activation'])
        )

    return result


def permutation_test(
    normal_metrics_list,
    recovered_metrics_list,
    metric_key='zigzag_score',
    n_permutations=10000,
    seed=42
):
    """
    置换检验：检验恢复轨迹与正常轨迹是否有显著差异。

    H0: 两组来自同一分布（无显著差异 = 恢复成功）

    Args:
        normal_metrics_list: list of dict, 多次正常运行的指标
        recovered_metrics_list: list of dict, 多次恢复运行的指标
        metric_key: str, 要检验的指标
        n_permutations: int
        seed: int

    Returns:
        dict: {'p_value': float, 'effect_size': float, 'pass': bool}
    """
    rng = np.random.RandomState(seed)

    normal_vals = np.array([m[metric_key] for m in normal_metrics_list])
    recovered_vals = np.array([m[metric_key] for m in recovered_metrics_list])

    observed_diff = abs(np.mean(normal_vals) - np.mean(recovered_vals))

    combined = np.concatenate([normal_vals, recovered_vals])
    n_normal = len(normal_vals)
    count = 0

    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = abs(np.mean(perm[:n_normal]) - np.mean(perm[n_normal:]))
        if perm_diff >= observed_diff:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    # Cohen's d
    pooled_std = np.sqrt((np.var(normal_vals) + np.var(recovered_vals)) / 2)
    effect_size = observed_diff / (pooled_std + 1e-10)

    return {
        'p_value': p_value,
        'effect_size': effect_size,
        'pass': p_value > 0.05,
    }
