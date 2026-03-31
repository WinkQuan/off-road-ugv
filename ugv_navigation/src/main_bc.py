#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Training script for BC (Behavior Cloning) algorithm
Pure imitation learning from expert (APF) demonstrations
Minimal changes from the original main.py - only agent import and initialization
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import env
import bc  # Changed from ddqn to bc
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
wandb.init(project="OFF-ROAD-UGV", name="BC_" + time.strftime("%Y-%m-%d %H:%M:%S"))

# 设置训练总轮数、最大步长和车重
total_episode = 5000
max_step_per_episode = 80
ugv_mass = 1.48

# 设置模型保存路径和模型文件名
model_path = "Model_BC/"
pth_path = model_path + "model_bc.pth"

if not os.path.exists(model_path):
    os.makedirs(model_path)

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = bc.BC(  # Changed from DQN to BC
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    learning_rate=1e-3,  # BC typically uses higher learning rate
    network="Duel",
)

# 存储每个episode的奖励、每个episode的step数、和每个episode是成功还是失败
ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
print("=" * 50)
print("Training BC Agent (Behavior Cloning)")
print("Pure imitation learning from APF demonstrations")
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

        # Get expert action (APF)
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

        # BC always follows expert or uses policy network (no epsilon-greedy)
        action_vx_index, action_vz_index = agent.get_action(state1, state2, dist_normalized)

        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)

        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)

        action_index = [action_vx_index, action_vz_index]
        apf_index = [apf_vx_index, apf_vz_index]
        # Store transition with expert label
        agent.replay_buffer.add(state1, state2, action_index, apf_index, reward, next_state1, next_state2, terminal)
        current_episode_reward += reward

        # BC learns to imitate expert demonstrations
        if len(agent.replay_buffer.memory) >= agent.batch_size:
            loss_bc, _ = agent.learn()

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
        }
    )

print("Training finished!")
agent.save_model(pth_path)
agent.save_onnx_model(model_path + "model_bc.onnx")
print(f"Model saved to {pth_path}")
