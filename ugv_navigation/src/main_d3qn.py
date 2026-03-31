#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Training script for D3QN algorithm
Minimal changes from the original main.py - only change the agent import and initialization
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import env
import d3qn  # Changed from ddqn to d3qn
import numpy as np
import random
import time
import torch
import rospy
import wandb
import APF_Vel_ROS
import helper_functions

# 设置随机数种子
random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4)

# 可视化数据工具
wandb.login()
wandb.init(project="OFF-ROAD-UGV", name="D3QN_" + time.strftime("%Y-%m-%d %H:%M:%S"))

# 设置训练总轮数、最大步长和车重
total_episode = 5000
max_step_per_episode = 80
ugv_mass = 1.48

# 设置模型保存路径和模型文件名
model_path = "Model_D3QN/"
pth_path = model_path + "model_d3qn.pth"

if not os.path.exists(model_path):
    os.makedirs(model_path)

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = d3qn.D3QN(  # Changed from DQN to D3QN
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    target_update=4,
    gamma=0.99,
    learning_rate=1e-4,
    epsilon=0.95,
    epsilon_min=0.1,
    epsilon_period=5000,
    network="Duel",
)

# 存储每个episode的奖励、每个episode的step数、和每个episode是成功还是失败
ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
print("=" * 50)
print("Training D3QN Agent")
print("=" * 50)

# 开始训练
for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0
    episode_success = False

    for t in range(max_step_per_episode):
        target_location = np.array(gazebo_ugv.goal)
        current_position = np.array(gazebo_ugv.self_state[0:2])
        obs_pos = np.array(gazebo_ugv.cylinder_pos)

        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz

        att, rep, vx_world, vz_world = APF_Vel_ROS.vel_control(
            target_location=target_location,
            current_position=current_position,
            obs_pos=obs_pos,
            obs_distance_threshold=3.5,
            k_att=1.0,
            k_rep=40,
        )

        apf_vx_index, apf_vz_index = helper_functions.choose_action(
            vx_world, vz_world, action_space_vx, action_space_vz
        )

        if np.random.uniform() < agent.epsilon:
            action_vx_index = np.random.randint(0, len(action_space_vx))
            action_vz_index = np.random.randint(0, len(action_space_vz))
        else:
            action_vx_index, action_vz_index = agent.get_action(state1, state2, dist_normalized)

        agent.epsilon = max(agent.epsilon_min, agent.epsilon - (agent.epsilon - agent.epsilon_min) / agent.epsilon_period)

        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)

        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)

        action_index = [action_vx_index, action_vz_index]
        apf_index = [apf_vx_index, apf_vz_index]
        agent.replay_buffer.add(state1, state2, action_index, apf_index, reward, next_state1, next_state2, terminal)
        current_episode_reward += reward

        if len(agent.replay_buffer.memory) >= agent.batch_size:
            loss_imitation, loss_dqn = agent.learn()

        if terminal:
            if termination_state == "arrival":
                episode_success = True
            break

        state1 = next_state1
        state2 = next_state2

    ep_reward_list.append(current_episode_reward)
    ep_step_list.append(t + 1)
    ep_success_list.append(1 if episode_success else 0)

    if (i_episode + 1) % 100 == 0:
        print(
            f"Episode {i_episode + 1}: Reward = {current_episode_reward:.2f}, Steps = {t+1}, Success = {episode_success}"
        )
        agent.save_model(pth_path)

    wandb.log(
        {
            "episode": i_episode,
            "reward": current_episode_reward,
            "steps": t + 1,
            "success": 1 if episode_success else 0,
            "epsilon": agent.epsilon,
        }
    )

print("Training finished!")
agent.save_model(pth_path)
agent.save_onnx_model(model_path + "model_d3qn.onnx")
print(f"Model saved to {pth_path}")
