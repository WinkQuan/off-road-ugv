#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Training script for DAgger algorithm
Policy rollouts are relabeled online by the APF expert and aggregated into the dataset
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time

import numpy as np
import rospy
import torch
import wandb

import APF_Vel_ROS
import dagger
import env


random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4)

wandb.login()
wandb.init(project="OFF-ROAD-UGV", name="DAgger_" + time.strftime("%Y-%m-%d %H:%M:%S"))

total_episode = 2000
max_step_per_episode = 100
ugv_mass = 1.48

model_path = "Model/DAgger/"
pth_path = model_path + "model.pth"
onnx_path = model_path + "model_dagger.onnx"

os.makedirs(model_path, exist_ok=True)

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = dagger.DAgger(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    learning_rate=1e-3,
    network="Duel",
)

ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
print("=" * 50)
print("Training DAgger Agent")
print("Policy acts in the environment, APF expert labels visited states")
print("=" * 50)

for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0
    episode_losses = []

    for t in range(max_step_per_episode):
        target_location = np.array(gazebo_ugv.goal, dtype=np.float32)
        current_position = np.array(gazebo_ugv.self_state[0:2], dtype=np.float32)
        obs_pos = np.array(gazebo_ugv.cylinder_pos, dtype=np.float32)

        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz

        _, _, vx_world, vy_world = APF_Vel_ROS.vel_control(
            target_location=target_location,
            current_position=current_position,
            obs_pos=obs_pos,
            mass=ugv_mass,
            obs_radius=5.0,
        )
        yaw = gazebo_ugv.self_state[3]
        linear_cmd, angular_cmd = APF_Vel_ROS.vector_to_ugv_controls(vx_world, vy_world, yaw)
        expert_vx_index = APF_Vel_ROS.fuzzy_map_v_triangular(linear_cmd, action_space_vx, strategy="min")
        expert_vz_index = APF_Vel_ROS.fuzzy_map_v_triangular(angular_cmd, action_space_vz, strategy="max")

        action_vx_index, action_vz_index = agent.get_action(state1, state2, dist_normalized)
        print("action{}:{} {}".format(t + 1, action_space_vx[action_vx_index], action_space_vz[action_vz_index]))

        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)
        rospy.sleep(0.1)
        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        current_episode_reward += reward

        agent.replay_buffer.add(state1, state2, expert_vx_index, expert_vz_index)

        if len(agent.replay_buffer.memory) >= agent.batch_size:
            loss_dagger, _ = agent.learn()
            episode_losses.append(loss_dagger)

        if terminal:
            if termination_state == "arrival":
                ep_success_list.append(1)
            else:
                ep_success_list.append(0)
            break

        state1 = next_state1
        state2 = next_state2

    ep_reward_list.append(current_episode_reward)
    ep_step_list.append(t + 1)

    if len(ep_success_list) >= 50:
        recent_results = ep_success_list[-50:]
        success_rate = sum(recent_results) / 50
    else:
        success_rate = sum(ep_success_list) / len(ep_success_list)

    mean_loss_dagger = float(np.mean(episode_losses)) if episode_losses else 0.0

    print(
        "Episode:{} \t step:{} \t current_episode_reward:{:.2f} \t loss_dagger:{:.4f}".format(
            i_episode, t + 1, current_episode_reward, mean_loss_dagger
        )
    )

    wandb.log({"Reward": current_episode_reward}, step=i_episode)
    wandb.log({"Step": t + 1}, step=i_episode)
    wandb.log({"Success Rate": success_rate}, step=i_episode)
    wandb.log({"Loss_DAgger": mean_loss_dagger}, step=i_episode)

    reward_file_path = model_path + "Reward"
    step_file_path = model_path + "Step"
    success_rate_file_path = model_path + "Success_Rate"

    mode = "a" if os.path.exists(success_rate_file_path) else "w"
    with open(success_rate_file_path, mode) as f:
        f.write(f"{success_rate}\n")

    if (i_episode + 1) % 500 == 0:
        mode = "a" if os.path.exists(reward_file_path) else "w"
        with open(reward_file_path, mode) as f:
            for reward_value in ep_reward_list:
                f.write(f"{reward_value}\n")
        ep_reward_list = []

        mode = "a" if os.path.exists(step_file_path) else "w"
        with open(step_file_path, mode) as f:
            for step_value in ep_step_list:
                f.write(f"{step_value}\n")
        ep_step_list = []

        agent.save_model(pth_path)
        agent.save_onnx_model(onnx_path)
        onnx_file_name = "Model_" + time.strftime("%Y-%m-%d") + f"_{i_episode + 1}.onnx"
        agent.save_onnx_model(model_path + onnx_file_name)
