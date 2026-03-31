#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Validation script for BC (Behavior Cloning) algorithm
Tests the trained BC model with all 8 evaluation metrics
"""
import rospy
import numpy as np
import pandas as pd
import time
import env
import bc
import onnxruntime as ort

max_step_per_episode = 400
max_episode = 100
success_count = 0
collision_count = 0
timeout_count = 0
step_count = 0
total_trajectory_length = 0.0
total_energy_consumption = 0.0
total_posture_stability = 0.0
total_execution_time = 0.0

ugv_mass = 1.48  # kg
ugv_inertia = 2.4  # kg·m²

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = bc.BC(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    learning_rate=1e-3,
    network="Duel",
)

## ONNX Model
model = "./Model_BC/model_bc.onnx"
csv_path = "./BC_velocity.csv"
tra_path = "./BC_trajectory.txt"

try:
    sess = ort.InferenceSession(model)
    obs_img = sess.get_inputs()[0].name
    obs_pos_onnx = sess.get_inputs()[1].name
except:
    print("Warning: ONNX model not found. Using PyTorch model for inference.")
    sess = None

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
print("=" * 50)
print("Validating BC Agent (Behavior Cloning)")
print("=" * 50)

linear_x_values = []
angular_z_values = []
ugv_pos_list = []

for i in range(max_episode):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    print(f"Episode {i+1}/{max_episode}: dist_normalized = {dist_normalized:.3f}")
    rospy.sleep(0.1)
    episode_start_time = time.time()
    episode_trajectory_length = 0.0
    episode_energy_consumption = 0.0
    episode_roll_sq = 0.0
    episode_pitch_sq = 0.0
    prev_ke = 0.0
    prev_pos = np.array(gazebo_ugv.self_state[0:3])

    for t in range(max_step_per_episode + 1):
        curr_pos = np.array(gazebo_ugv.self_state[0:3])
        ugv_pos_list.append(curr_pos)
        dist = np.linalg.norm(curr_pos - prev_pos)
        episode_trajectory_length += dist
        prev_pos = curr_pos

        if sess is not None:
            # Use ONNX model
            output_velocity = sess.run(
                None,
                {
                    obs_img: np.array(np.expand_dims(state1, axis=0), dtype=np.float32),
                    obs_pos_onnx: np.array(state2, dtype=np.float32).reshape(1, -1),
                },
            )
            output_vx_index = np.argmax(output_velocity[0])
            output_vz_index = np.argmax(output_velocity[1])
        else:
            # Use PyTorch model
            output_vx_index, output_vz_index = agent.get_action(state1, state2, dist_normalized)

        gazebo_ugv.execute_linear_velocity(output_vx_index, output_vz_index)

        vx = gazebo_ugv.action_space_vx[output_vx_index]
        omega = gazebo_ugv.action_space_vz[output_vz_index]
        current_ke = 0.5 * ugv_mass * vx**2 + 0.5 * ugv_inertia * omega**2
        episode_energy_consumption += current_ke - prev_ke
        prev_ke = current_ke

        # Posture stability
        roll = gazebo_ugv.self_state[4]
        pitch = gazebo_ugv.self_state[5]
        episode_roll_sq += roll**2
        episode_pitch_sq += pitch**2

        linear_x_values.append(vx)
        angular_z_values.append(omega)

        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)

        if terminal:
            if termination_state == "arrival":
                np.savetxt(tra_path, ugv_pos_list)
                df_velocity = pd.DataFrame({"linear_x": linear_x_values, "angular_z": angular_z_values})
                df_velocity.to_csv(csv_path, index=False)
                success_count += 1
                step_count += t + 1
                total_trajectory_length += episode_trajectory_length
                total_energy_consumption += episode_energy_consumption
                rms_roll = np.sqrt(episode_roll_sq / (t + 1))
                rms_pitch = np.sqrt(episode_pitch_sq / (t + 1))
                total_posture_stability += rms_roll + rms_pitch
                episode_execution_time = time.time() - episode_start_time
                total_execution_time += episode_execution_time
                print(f"  ✓ Success! Steps={t+1}, Time={episode_execution_time:.2f}s")
                linear_x_values.clear()
                angular_z_values.clear()
                ugv_pos_list.clear()
            elif termination_state == "collision":
                collision_count += 1
                print(f"  ✗ Collision!")
            elif termination_state == "timeout":
                timeout_count += 1
                print(f"  ⏱ Timeout!")
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
print("BC Validation Results")
print("=" * 50)
print(f"Success Rate: {success_rate*100:.2f}%")
print(f"Collision Rate: {collision_rate*100:.2f}%")
print(f"Timeout Rate: {timeout_rate*100:.2f}%")
print(f"Average Time Step: {average_step:.0f}")
print(f"Average Trajectory Length: {average_trajectory_length:.2f} m")
print(f"Average Energy Consumption: {average_energy_consumption:.2f} J")
print(f"Average Posture Stability: {average_posture_stability:.4f} rad")
print(f"Average Execution Time: {average_execution_time:.3f} s")
print("=" * 50)
