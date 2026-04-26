#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Validation script for pure APF baseline
Tests the APF controller with the same 8 evaluation metrics as learned policies
"""
from __future__ import absolute_import
from __future__ import print_function
import csv
import time
from pathlib import Path

import numpy as np
import rospy

import env
import APF_Vel_ROS

max_step_per_episode = 100
max_episode = 100
success_count = 0
collision_count = 0
timeout_count = 0
step_count = 0
total_trajectory_length = 0.0
total_energy_consumption = 0.0
total_posture_stability = 0.0
total_execution_time = 0.0

ugv_mass = 1.48  # 小车总质量 kg
ugv_inertia = 2.4
obs_radius = 5.0  # 障碍物影响范围半径 m

validate_dir = Path("./Validate/APF")
csv_path = validate_dir / "APF_velocity.csv"
tra_path = validate_dir / "APF_trajectory.txt"
metrics_path = validate_dir / "metrics.csv"
validate_dir.mkdir(parents=True, exist_ok=True)

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)


def write_dict_csv(path, fieldnames, row):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_velocity_csv(path, linear_x_values, angular_z_values):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["linear_x", "angular_z"])
        writer.writerows(zip(linear_x_values, angular_z_values))


print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
print("=" * 50)
print("Validating Pure APF Baseline")
print("=" * 50)


# --------------------------Path Finding with APF--------------------
for i_episode in range(max_episode):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    print(f"Episode {i_episode + 1}/{max_episode}: dist_normalized = {dist_normalized:.3f}")
    rospy.sleep(0.1)

    episode_start_time = time.time()
    episode_trajectory_length = 0.0
    episode_energy_consumption = 0.0
    episode_roll_sq = 0.0
    episode_pitch_sq = 0.0
    prev_ke = 0.0
    prev_pos = np.array(gazebo_ugv.self_state[0:3])
    linear_x_values = []
    angular_z_values = []
    ugv_pos_list = []

    for t in range(max_step_per_episode + 1):
        curr_pos_3d = np.array(gazebo_ugv.self_state[0:3])
        ugv_pos_list.append(curr_pos_3d)
        dist = np.linalg.norm(curr_pos_3d - prev_pos)
        episode_trajectory_length += dist
        prev_pos = curr_pos_3d

        goal = np.array(gazebo_ugv.goal, dtype=np.float32)
        curr_pos = np.array(gazebo_ugv.self_state[0:2], dtype=np.float32)
        obs_pos = np.array(gazebo_ugv.cylinder_pos, dtype=np.float32)
        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz

        att, rep, vx_world, vy_world = APF_Vel_ROS.vel_control(
            target_location=goal,
            current_position=curr_pos,
            obs_pos=obs_pos,
            mass=ugv_mass,
            obs_radius=obs_radius,
        )
        yaw = gazebo_ugv.self_state[3]
        linear_cmd, angular_cmd = APF_Vel_ROS.vector_to_ugv_controls(vx_world, vy_world, yaw)
        output_vx_index = APF_Vel_ROS.fuzzy_map_v_triangular(linear_cmd, action_space_vx, strategy="min")
        output_vz_index = APF_Vel_ROS.fuzzy_map_v_triangular(angular_cmd, action_space_vz, strategy="max")

        gazebo_ugv.execute_linear_velocity(output_vx_index, output_vz_index)

        vx = action_space_vx[output_vx_index]
        omega = action_space_vz[output_vz_index]
        current_ke = 0.5 * ugv_mass * vx**2 + 0.5 * ugv_inertia * omega**2
        episode_energy_consumption += current_ke - prev_ke
        prev_ke = current_ke

        roll = gazebo_ugv.self_state[4]
        pitch = gazebo_ugv.self_state[5]
        episode_roll_sq += roll**2
        episode_pitch_sq += pitch**2
        linear_x_values.append(vx)
        angular_z_values.append(omega)

        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        if terminal:
            if termination_state == "arrival":
                np.savetxt(str(tra_path), ugv_pos_list)
                write_velocity_csv(csv_path, linear_x_values, angular_z_values)
                success_count += 1
                step_count += t + 1
                total_trajectory_length += episode_trajectory_length
                total_energy_consumption += episode_energy_consumption
                rms_roll = np.sqrt(episode_roll_sq / (t + 1))
                rms_pitch = np.sqrt(episode_pitch_sq / (t + 1))
                total_posture_stability += rms_roll + rms_pitch
                episode_execution_time = time.time() - episode_start_time
                total_execution_time += episode_execution_time
                print(f"  ✓ Success! Steps={t + 1}, Time={episode_execution_time:.2f}s")
            elif termination_state == "collision":
                collision_count += 1
                print("  ✗ Collision!")
            elif termination_state == "timeout":
                timeout_count += 1
                print("  ⏱ Timeout!")
            elif termination_state == "out":
                print("  ✗ Out of bounds!")
            break
        state1 = next_state1
        state2 = next_state2

success_rate = success_count / max_episode
collision_rate = collision_count / max_episode
timeout_rate = timeout_count / max_episode

if success_count > 0:
    average_step = step_count / success_count
    average_trajectory_length = total_trajectory_length / success_count
    average_energy_consumption = total_energy_consumption / success_count
    average_posture_stability = total_posture_stability / success_count
    average_execution_time = total_execution_time / success_count
else:
    average_step = 0
    average_trajectory_length = 0
    average_energy_consumption = 0
    average_posture_stability = 0
    average_execution_time = 0
    print("Warning: No successful episodes!")

print()
print("=" * 50)
print("APF Validation Results")
print("=" * 50)
print(f"Success Rate: {success_rate * 100:.2f}%")
print(f"Collision Rate: {collision_rate * 100:.2f}%")
print(f"Timeout Rate: {timeout_rate * 100:.2f}%")
print(f"Average Time Step: {average_step:.0f}")
print(f"Average Trajectory Length: {average_trajectory_length:.2f} m")
print(f"Average Energy Consumption: {average_energy_consumption:.2f} J")
print(f"Average Posture Stability: {average_posture_stability:.4f} rad")
print(f"Average Execution Time: {average_execution_time:.3f} s")
print("=" * 50)

write_dict_csv(
    metrics_path,
    [
        "algorithm",
        "success_rate_pct",
        "collision_rate_pct",
        "timeout_rate_pct",
        "average_step",
        "average_trajectory_length_m",
        "average_energy_consumption_j",
        "average_posture_stability_rad",
        "average_execution_time_s",
        "max_episode",
        "max_step_per_episode",
    ],
    {
        "algorithm": "APF",
        "success_rate_pct": success_rate * 100,
        "collision_rate_pct": collision_rate * 100,
        "timeout_rate_pct": timeout_rate * 100,
        "average_step": average_step,
        "average_trajectory_length_m": average_trajectory_length,
        "average_energy_consumption_j": average_energy_consumption,
        "average_posture_stability_rad": average_posture_stability,
        "average_execution_time_s": average_execution_time,
        "max_episode": max_episode,
        "max_step_per_episode": max_step_per_episode,
    },
)
print(f"Metrics saved to: {metrics_path}")
