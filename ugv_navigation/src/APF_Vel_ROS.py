import numpy as np
import math

EPS = 1e-6


# 计算目标点的吸引势
def attractive_potential(target_location, current_position, k_att=1.0, d_goal_threshold=2.0):
    target_location = np.asarray(target_location, dtype=np.float32)
    current_position = np.asarray(current_position, dtype=np.float32)
    delta = target_location - current_position
    distance = np.linalg.norm(delta)  # 计算当前点到目标点的欧式距离
    if distance < EPS:
        return np.zeros(2, dtype=np.float32)
    if distance <= d_goal_threshold:
        att = k_att * delta
    else:
        att = d_goal_threshold * k_att * delta / distance
    return att


# 计算障碍物的排斥势
def repulsive_potential(obs_pos, curr_pos, obs_radius, k_rep=1.0):
    curr_pos = np.asarray(curr_pos, dtype=np.float32)
    obs_pos = np.asarray(obs_pos, dtype=np.float32)
    rep = np.zeros(2, dtype=np.float32)
    if obs_pos.size == 0:
        return rep

    obs_pos = np.atleast_2d(obs_pos)
    for obs in obs_pos:
        offset = curr_pos - obs
        dist = np.linalg.norm(offset)
        if dist < EPS:
            dist = EPS
        if dist < obs_radius:
            rep += 0.5 * k_rep * (1.0 / dist - 1.0 / obs_radius) * (1.0 / dist) ** 2 * offset / dist
    return rep


# 计算世界坐标系下的期望速度方向
def vel_control(target_location, current_position, obs_pos, mass=1.48, obs_radius=2.0, **kwargs):
    if "obs_distance_threshold" in kwargs:
        obs_radius = kwargs.pop("obs_distance_threshold")
    k_att = kwargs.pop("k_att", 1.0)
    k_rep = kwargs.pop("k_rep", 1.0)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if mass <= EPS:
        raise ValueError("mass must be positive.")

    att = attractive_potential(target_location, current_position, k_att=k_att)
    rep = repulsive_potential(obs_pos, current_position, obs_radius, k_rep=k_rep)
    total_pot = att + rep
    total_norm = np.linalg.norm(total_pot)
    if total_norm > 1:
        total_pot = total_pot / total_norm
    vx_world = total_pot[0] / mass
    vy_world = total_pot[1] / mass
    return att, rep, vx_world, vy_world


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def convert_to_ugv_frame(vx_world, vz_world, yaw):
    # Conversion to body coordinate system
    vx_ugv = 1.8 * (vz_world * math.sin(yaw) + vx_world * math.cos(yaw))
    vz_ugv = 1.8 * (vz_world * math.cos(yaw) - vx_world * math.sin(yaw))
    return vx_ugv, vz_ugv


def vector_to_ugv_controls(vx_world, vy_world, yaw, linear_gain=1.8, angular_gain=2.0 / math.pi):
    """Convert a desired planar motion vector into UGV linear/angular commands."""
    speed = math.hypot(vx_world, vy_world)
    if speed < EPS:
        return 0.0, 0.0

    desired_heading = math.atan2(vy_world, vx_world)
    heading_error = normalize_angle(desired_heading - yaw)
    linear_cmd = linear_gain * speed * math.cos(heading_error)
    angular_cmd = angular_gain * heading_error
    return linear_cmd, angular_cmd


def triangular_membership(x, a, b, c):
    """Triangular membership function."""
    if a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0.0


def fuzzy_map_v_triangular(v_scaled, action_space, strategy="max"):
    """Map scaled v to DQN's discrete action space using triangular membership functions."""
    action_space = np.asarray(action_space, dtype=np.float32)
    # If v_scaled exceeds the endpoint value then select the corresponding endpoint value
    if v_scaled >= max(action_space):
        return len(action_space) - 1  # Returns the index of the maximum action value
    elif v_scaled <= min(action_space):
        return 0  # Returns the index of the minimum action value

    # Define the triangular membership functions for each action
    if len(action_space) > 1:
        width = float(np.min(np.diff(action_space)))
    else:
        width = 1.0
    memberships = np.array(
        [triangular_membership(v_scaled, action - width / 2, action, action + width / 2) for action in action_space],
        dtype=np.float32,
    )

    # Find all actions with the highest membership value
    max_indices = np.argwhere(memberships == memberships.max()).flatten()

    # If there is only one maximum affiliation value, it is returned directly
    if len(max_indices) == 1:
        return int(max_indices[0])

    # Selection of actions with larger or smaller absolute values depending on the strategy
    if strategy == "max":
        selected_action_index = max(max_indices, key=lambda index: abs(action_space[index]))
    elif strategy == "min":
        selected_action_index = min(max_indices, key=lambda index: abs(action_space[index]))
    else:
        raise ValueError("Strategy must be either 'max' or 'min'.")

    return int(selected_action_index)
