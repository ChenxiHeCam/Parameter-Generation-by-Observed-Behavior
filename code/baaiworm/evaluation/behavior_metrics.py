"""
行为评估指标：zigzag 检测、速度、体波频率等。

BAAIWorm 输出的轨迹数据格式：
- rel_x, rel_y, rel_z: shape (T, 17), 17 段身体在 body frame 中的位置
- rel_vx, rel_vy, rel_vz: shape (T, 17), 速度
- 肌肉信号: shape (T, 96), 96 块肌肉的激活值
"""
import numpy as np
from scipy import signal as scipy_signal


def compute_heading_angle(trajectory_xz):
    """
    计算航向角序列。

    Args:
        trajectory_xz: np.ndarray, shape (T, 2), 质心的 x-z 坐标

    Returns:
        np.ndarray, shape (T-1,), 航向角（弧度）
    """
    dx = np.diff(trajectory_xz[:, 0])
    dz = np.diff(trajectory_xz[:, 1])
    heading = np.arctan2(dz, dx)
    return heading


def compute_angular_velocity(heading):
    """计算角速度（航向角的变化率）"""
    dtheta = np.diff(heading)
    # 处理 -pi/pi 跳变
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    return dtheta


def detect_zigzag(rel_x, rel_z, min_period=5, max_period=50):
    """
    检测 zigzag 运动模式。

    Args:
        rel_x: np.ndarray, shape (T, 17), x 坐标
        rel_z: np.ndarray, shape (T, 17), z 坐标
        min_period: int, 最小 zigzag 周期（时间步）
        max_period: int, 最大 zigzag 周期

    Returns:
        dict: {
            'has_zigzag': bool,
            'n_reversals': int,
            'mean_period': float,
            'period_std': float,
            'mean_amplitude': float,
            'zigzag_score': float (0~1, 越高越像 zigzag),
        }
    """
    # 用头部（第0段）的轨迹
    head_x = rel_x[:, 0]
    head_z = rel_z[:, 0]
    trajectory_xz = np.column_stack([head_x, head_z])

    if len(trajectory_xz) < 30:
        return _empty_zigzag_result()

    heading = compute_heading_angle(trajectory_xz)
    if len(heading) < 10:
        return _empty_zigzag_result()

    angular_vel = compute_angular_velocity(heading)

    # 低通滤波去噪
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5
        cutoff = 0.15
        b, a = butter(3, cutoff / nyq, btype='low')
        if len(angular_vel) > 12:
            angular_vel_filtered = filtfilt(b, a, angular_vel)
        else:
            angular_vel_filtered = angular_vel
    except (ImportError, ValueError):
        angular_vel_filtered = angular_vel

    # 检测零交叉（方向反转）
    zero_crossings = np.where(np.diff(np.sign(angular_vel_filtered)))[0]
    n_reversals = len(zero_crossings)

    if n_reversals < 2:
        return _empty_zigzag_result()

    # 计算反转间隔
    intervals = np.diff(zero_crossings)
    mean_period = float(np.mean(intervals))
    period_std = float(np.std(intervals))

    # 计算振幅：相邻反转点之间的航向角变化
    amplitudes = []
    for i in range(len(zero_crossings) - 1):
        start = zero_crossings[i]
        end = zero_crossings[i + 1]
        if end < len(heading):
            amp = abs(heading[end] - heading[start])
            amplitudes.append(amp)
    mean_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0

    # 频率域验证：FFT 检查是否有主频
    fft_score = 0.0
    if len(angular_vel_filtered) >= 20:
        freqs = np.fft.rfftfreq(len(angular_vel_filtered))
        fft_vals = np.abs(np.fft.rfft(angular_vel_filtered - np.mean(angular_vel_filtered)))
        if len(fft_vals) > 1:
            power = fft_vals[1:] ** 2
            total_power = np.sum(power) + 1e-10
            peak_power = np.max(power)
            fft_score = float(peak_power / total_power)

    # 周期质量（连续函数，理想周期 ~15 步）
    ideal_period = (min_period + max_period) / 2.0
    period_quality = np.exp(-((mean_period - ideal_period) / (max_period / 3.0)) ** 2)
    if mean_period < min_period or mean_period > max_period:
        period_quality *= 0.3

    # 振幅质量（连续函数，0.1 rad 以下快速衰减）
    amplitude_quality = 1.0 - np.exp(-mean_amplitude / 0.15)

    # 周期规律性
    regularity = 1.0 - min(period_std / (mean_period + 1e-6), 1.0)

    # 综合评分（全部连续）
    zigzag_score = (
        0.30 * amplitude_quality +
        0.25 * period_quality +
        0.20 * regularity +
        0.15 * min(n_reversals / 20.0, 1.0) +
        0.10 * fft_score
    )

    return {
        'has_zigzag': zigzag_score > 0.5,
        'n_reversals': n_reversals,
        'mean_period': mean_period,
        'period_std': period_std,
        'mean_amplitude': np.degrees(mean_amplitude),
        'zigzag_score': zigzag_score,
    }


def _empty_zigzag_result():
    return {
        'has_zigzag': False,
        'n_reversals': 0,
        'mean_period': 0.0,
        'period_std': 0.0,
        'mean_amplitude': 0.0,
        'zigzag_score': 0.0,
    }


def compute_forward_speed(rel_x, rel_z, dt=1.0):
    """
    计算质心前进速度。

    Args:
        rel_x, rel_z: shape (T, 17)
        dt: 时间步长

    Returns:
        dict: {'mean_speed': float, 'speed_std': float, 'speed_trace': np.ndarray}
    """
    # 用所有 17 段的平均位置作为质心
    cx = np.mean(rel_x, axis=1)
    cz = np.mean(rel_z, axis=1)
    dx = np.diff(cx)
    dz = np.diff(cz)
    speed = np.sqrt(dx**2 + dz**2) / dt
    return {
        'mean_speed': float(np.mean(speed)),
        'speed_std': float(np.std(speed)),
        'speed_trace': speed,
    }


def compute_body_wave(rel_x, rel_z, dt=1.0):
    """
    计算体波频率和波长。

    通过 17 段身体的曲率时间序列的 FFT 分析。

    Args:
        rel_x, rel_z: shape (T, 17)
        dt: 时间步长

    Returns:
        dict: {'dominant_freq': float, 'power_spectrum': np.ndarray, 'freqs': np.ndarray}
    """
    T, n_seg = rel_x.shape

    # 计算每个时间步的身体曲率
    curvatures = np.zeros((T, n_seg - 2))
    for t in range(T):
        for s in range(1, n_seg - 1):
            v1 = np.array([rel_x[t, s] - rel_x[t, s-1], rel_z[t, s] - rel_z[t, s-1]])
            v2 = np.array([rel_x[t, s+1] - rel_x[t, s], rel_z[t, s+1] - rel_z[t, s]])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            norm1 = np.linalg.norm(v1) + 1e-10
            norm2 = np.linalg.norm(v2) + 1e-10
            curvatures[t, s-1] = cross / (norm1 * norm2)

    # 对中间段的曲率做 FFT
    mid_curvature = curvatures[:, n_seg // 2 - 1]
    if len(mid_curvature) < 10:
        return {'dominant_freq': 0.0, 'power_spectrum': np.array([]), 'freqs': np.array([])}

    freqs = np.fft.rfftfreq(len(mid_curvature), d=dt)
    fft_vals = np.abs(np.fft.rfft(mid_curvature - np.mean(mid_curvature)))
    power = fft_vals ** 2

    # 排除直流分量
    if len(power) > 1:
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = float(freqs[dominant_idx])
    else:
        dominant_freq = 0.0

    return {
        'dominant_freq': dominant_freq,
        'power_spectrum': power,
        'freqs': freqs,
    }


def compute_muscle_alternation(muscle_signals):
    """
    计算背腹肌肉交替模式。

    BAAIWorm 肌肉排列: DR(0-23), VR(24-47), DL(48-71), VL(72-95)
    正常运动时背侧和腹侧肌肉应该交替收缩。

    Args:
        muscle_signals: np.ndarray, shape (T, 96)

    Returns:
        dict: {'alternation_score': float, 'dorsal_ventral_corr': float}
    """
    if muscle_signals.shape[0] < 10:
        return {'alternation_score': 0.0, 'dorsal_ventral_corr': 0.0}

    # 背侧平均 (DR + DL)
    dorsal = np.mean(muscle_signals[:, :24], axis=1) + np.mean(muscle_signals[:, 48:72], axis=1)
    dorsal /= 2.0
    # 腹侧平均 (VR + VL)
    ventral = np.mean(muscle_signals[:, 24:48], axis=1) + np.mean(muscle_signals[:, 72:96], axis=1)
    ventral /= 2.0

    # 背腹相关性（正常应该是负相关，即交替）
    if np.std(dorsal) < 1e-10 or np.std(ventral) < 1e-10:
        corr = 0.0
    else:
        corr = float(np.corrcoef(dorsal, ventral)[0, 1])

    # 交替评分：负相关越强越好
    alternation_score = max(0.0, -corr)

    return {
        'alternation_score': alternation_score,
        'dorsal_ventral_corr': corr,
    }


def behavior_destroyed(metrics, reference_metrics, threshold=0.3):
    """
    判断行为是否被破坏。

    Args:
        metrics: dict, 当前轨迹的指标
        reference_metrics: dict, 正常轨迹的指标
        threshold: float, 综合评分低于此值认为行为被破坏

    Returns:
        bool
    """
    score = 0.0
    total_weight = 0.0

    # zigzag 是否存在（最重要）
    if reference_metrics.get('has_zigzag', False):
        weight = 0.4
        total_weight += weight
        if metrics.get('has_zigzag', False):
            score += weight * (metrics['zigzag_score'] / max(reference_metrics['zigzag_score'], 0.01))

    # 速度是否接近
    ref_speed = reference_metrics.get('mean_speed', 0)
    if ref_speed > 0:
        weight = 0.2
        total_weight += weight
        cur_speed = metrics.get('mean_speed', 0)
        speed_ratio = min(cur_speed / ref_speed, ref_speed / max(cur_speed, 1e-10))
        score += weight * speed_ratio

    # 体波频率是否接近
    ref_freq = reference_metrics.get('dominant_freq', 0)
    if ref_freq > 0:
        weight = 0.2
        total_weight += weight
        cur_freq = metrics.get('dominant_freq', 0)
        freq_ratio = min(cur_freq / ref_freq, ref_freq / max(cur_freq, 1e-10))
        score += weight * freq_ratio

    # 肌肉交替模式
    ref_alt = reference_metrics.get('alternation_score', 0)
    if ref_alt > 0:
        weight = 0.2
        total_weight += weight
        cur_alt = metrics.get('alternation_score', 0)
        score += weight * min(cur_alt / ref_alt, 1.0)

    if total_weight > 0:
        normalized_score = score / total_weight
    else:
        normalized_score = 0.0

    return normalized_score < threshold


# ============================================================
# 趋化行为指标（论文 Fig. 5，功能性验证）
# ============================================================

def compute_chemotaxis_index(world_head_locations, food_location):
    """
    趋化指数：衡量线虫是否在向食物移动。

    CI = (d_start - d_end) / total_path_length
    范围 [-1, 1]：1=完美趋化，0=随机游走，-1=远离食物

    Args:
        world_head_locations: list of np.ndarray(3,), 头部世界坐标序列
        food_location: np.ndarray(3,), 食物位置

    Returns:
        float, chemotaxis index
    """
    if len(world_head_locations) < 2:
        return 0.0
    locs = np.array(world_head_locations)
    food = np.array(food_location)

    d_start = np.linalg.norm(locs[0] - food)
    d_end = np.linalg.norm(locs[-1] - food)

    path_lengths = np.linalg.norm(np.diff(locs, axis=0), axis=1)
    total_path = np.sum(path_lengths)

    if total_path < 1e-10:
        return 0.0
    return float((d_start - d_end) / total_path)


def compute_distance_to_food_curve(world_head_locations, food_location):
    """
    离食物距离随时间的变化曲线。

    Args:
        world_head_locations: list of np.ndarray(3,)
        food_location: np.ndarray(3,)

    Returns:
        np.ndarray, shape (T,), 每个时间步到食物的距离
    """
    locs = np.array(world_head_locations)
    food = np.array(food_location)
    return np.linalg.norm(locs - food, axis=1)


def compute_steering_angle(world_head_locations, food_location):
    """
    转向角：速度方向与食物方向的夹角。

    夹角越小说明越朝食物方向运动。

    Args:
        world_head_locations: list of np.ndarray(3,)
        food_location: np.ndarray(3,)

    Returns:
        dict: {
            'mean_angle': float (度),
            'angle_trace': np.ndarray (T-1,) (度),
        }
    """
    locs = np.array(world_head_locations)
    food = np.array(food_location)

    if len(locs) < 3:
        return {'mean_angle': 90.0, 'angle_trace': np.array([])}

    velocities = np.diff(locs, axis=0)
    to_food = food - locs[:-1]

    angles = []
    for v, tf in zip(velocities, to_food):
        v_norm = np.linalg.norm(v)
        tf_norm = np.linalg.norm(tf)
        if v_norm < 1e-10 or tf_norm < 1e-10:
            angles.append(90.0)
            continue
        cos_angle = np.clip(np.dot(v, tf) / (v_norm * tf_norm), -1, 1)
        angles.append(np.degrees(np.arccos(cos_angle)))

    angle_trace = np.array(angles)
    return {
        'mean_angle': float(np.mean(angle_trace)),
        'angle_trace': angle_trace,
    }


def compute_turning_efficiency(world_head_locations, food_location, window=10):
    """
    转向效率：浓度下降时转向频率是否增加。

    正常线虫在远离食物时会增加转向（pirouette），接近时直行。

    Args:
        world_head_locations: list of np.ndarray(3,)
        food_location: np.ndarray(3,)
        window: int, 滑动窗口大小

    Returns:
        dict: {
            'efficiency': float (0~1, 越高越好),
            'turn_rate_approaching': float,
            'turn_rate_leaving': float,
        }
    """
    locs = np.array(world_head_locations)
    food = np.array(food_location)

    if len(locs) < window * 3:
        return {'efficiency': 0.0, 'turn_rate_approaching': 0.0,
                'turn_rate_leaving': 0.0}

    distances = np.linalg.norm(locs - food, axis=1)
    d_change = np.diff(distances)

    velocities = np.diff(locs, axis=0)
    if len(velocities) < 3:
        return {'efficiency': 0.0, 'turn_rate_approaching': 0.0,
                'turn_rate_leaving': 0.0}

    heading = np.arctan2(velocities[:, 2], velocities[:, 0])
    angular_vel = np.abs(np.diff(heading))
    angular_vel = np.arctan2(np.sin(angular_vel), np.cos(angular_vel))

    T = min(len(d_change), len(angular_vel))
    d_change = d_change[:T]
    angular_vel = angular_vel[:T]

    approaching = d_change < 0
    leaving = d_change > 0

    turn_approaching = float(np.mean(angular_vel[approaching])) if np.any(approaching) else 0.0
    turn_leaving = float(np.mean(angular_vel[leaving])) if np.any(leaving) else 0.0

    if turn_leaving < 1e-10:
        efficiency = 0.0
    else:
        efficiency = float(np.clip(turn_leaving / (turn_approaching + 1e-10) - 1.0, 0, 1))

    return {
        'efficiency': efficiency,
        'turn_rate_approaching': turn_approaching,
        'turn_rate_leaving': turn_leaving,
    }


# ============================================================
# 神经动力学指标（论文 Fig. 6 第一列）
# ============================================================

def compute_pc_matrix(neural_voltages):
    """
    Pearson Correlation matrix of neuron membrane potentials.

    论文 Fig. 6 第一列，用于评估神经网络动力学是否正常。

    Args:
        neural_voltages: np.ndarray, shape (T, N_neurons) 或 (N_neurons, T)

    Returns:
        np.ndarray, shape (N, N), Pearson 相关矩阵
    """
    if neural_voltages.ndim != 2:
        return np.array([[]])
    if neural_voltages.shape[0] > neural_voltages.shape[1]:
        neural_voltages = neural_voltages.T
    pc = np.corrcoef(neural_voltages)
    pc[np.isnan(pc)] = 0.0
    return pc


def compare_pc_matrices(pc_a, pc_b):
    """
    比较两个 PC matrix 的相似度。

    Args:
        pc_a, pc_b: np.ndarray, shape (N, N)

    Returns:
        dict: {'correlation': float, 'mse': float}
    """
    if pc_a.size == 0 or pc_b.size == 0:
        return {'correlation': 0.0, 'mse': 1.0}
    n = min(pc_a.shape[0], pc_b.shape[0])
    a = pc_a[:n, :n].flatten()
    b = pc_b[:n, :n].flatten()
    mask = ~(np.isnan(a) | np.isnan(b))
    if np.sum(mask) < 2:
        return {'correlation': 0.0, 'mse': 1.0}
    return {
        'correlation': float(np.corrcoef(a[mask], b[mask])[0, 1]),
        'mse': float(np.mean((a[mask] - b[mask]) ** 2)),
    }


# ============================================================
# 综合指标计算（更新版，包含趋化指标）
# ============================================================

def compute_all_metrics(trajectory_data):
    """
    计算所有行为指标（运动学 + 趋化 + 神经动力学）。

    Args:
        trajectory_data: dict with keys:
            'rel_x', 'rel_y', 'rel_z': shape (T, 17)
            'rel_vx', 'rel_vy', 'rel_vz': shape (T, 17)
            'muscle_activation': shape (T, 96) (可选)
            'neural_voltage': list or array (可选)
            'world_head_location': list of (3,) (可选)
            'food_location': (3,) (可选)

    Returns:
        dict: 所有指标的汇总
    """
    metrics = {}

    # 运动学指标
    if 'rel_x' in trajectory_data and 'rel_z' in trajectory_data:
        rel_x = trajectory_data['rel_x']
        rel_z = trajectory_data['rel_z']
        metrics.update(detect_zigzag(rel_x, rel_z))
        metrics.update(compute_forward_speed(rel_x, rel_z))
        metrics.update(compute_body_wave(rel_x, rel_z))

    # 肌肉指标
    if 'muscle_activation' in trajectory_data:
        muscle = np.array(trajectory_data['muscle_activation'])
        metrics.update(compute_muscle_alternation(muscle))

    # 趋化指标
    if 'world_head_location' in trajectory_data:
        head_locs = trajectory_data['world_head_location']
        food_loc = trajectory_data.get('food_location',
                                        np.array([0, 0, 0], dtype=np.float32))
        metrics['chemotaxis_index'] = compute_chemotaxis_index(head_locs, food_loc)
        metrics['distance_to_food_final'] = float(
            np.linalg.norm(np.array(head_locs[-1]) - food_loc))
        steering = compute_steering_angle(head_locs, food_loc)
        metrics['mean_steering_angle'] = steering['mean_angle']
        turning = compute_turning_efficiency(head_locs, food_loc)
        metrics['turning_efficiency'] = turning['efficiency']

    # 神经动力学指标
    if 'neural_voltage' in trajectory_data:
        nv = np.array(trajectory_data['neural_voltage'])
        pc = compute_pc_matrix(nv)
        metrics['pc_matrix'] = pc

    return metrics
