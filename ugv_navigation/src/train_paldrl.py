#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import env
import paldrl
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
wandb.init(project="OFF-ROAD-UGV", name="PAL-DRL_" + time.strftime("%Y-%m-%d %H:%M:%S"))

# 设置训练总轮数、最大步长和车重
total_episode = 2000
max_step_per_episode = 100
# ugv_mass = 62.01455
ugv_mass = 1.48

# 设置模型保存路径和模型文件名
model_path = "Model/PAL-DRL/"
pth_path = model_path + "model.pth"
onnx_path = model_path + "model_pal_drl.onnx"

os.makedirs(model_path, exist_ok=True)

# 初始化环境和智能体
gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = paldrl.DQN(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    target_update=4,
    gamma=0.99,
    learning_rate=1e-4,
    epsilon=0.1,
    epsilon_min=0.1,
    epsilon_period=50,
    network="Duel",
)  # Change the network parameter if you want to train on different network

# 存储每个episode的奖励、每个episode的step数、和每个episode是成功还是失败
ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)

# 开始训练，训练的总轮数为total_episode
for i_episode in range(total_episode + 1):
    # 没用到dist_normalized
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0
    # 每一步的操作
    for t in range(max_step_per_episode):
        target_location = np.array(gazebo_ugv.goal)
        current_position = np.array(gazebo_ugv.self_state[0:2])
        # obs_pos = np.array(gazebo_ugv.cylinder_pos)
        # 只获取当前可视障碍物位置
        visible_obs_pos = np.array(gazebo_ugv.get_visible_obstacles())
        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz
        # att和rep似乎不需要返回？
        att, rep, vx_world, vy_world = APF_Vel_ROS.vel_control(
            target_location=target_location,
            current_position=current_position,
            obs_pos=visible_obs_pos,
            mass=ugv_mass,
            obs_radius=5.0,
        )
        yaw = gazebo_ugv.self_state[3]
        # 将世界坐标系下的APF引导向量转换为UGV的线速度/角速度命令
        linear_cmd, angular_cmd = APF_Vel_ROS.vector_to_ugv_controls(vx_world, vy_world, yaw)
        vx_ugv_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(linear_cmd, action_space_vx, strategy="min")
        vz_ugv_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(angular_cmd, action_space_vz, strategy="max")
        # 开始训练，获取动作 (DQN action selection)
        action_vx_index, action_vz_index = agent.get_action(state1, state2, dist_normalized)
        print(
            "action{}:{} {}".format(t + 1, action_space_vx[action_vx_index], action_space_vz[action_vz_index])
        )  # 输出选择的动作
        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)
        if len(agent.replay_buffer.memory) > 64:
            loss_imitation, loss_dqn = agent.learn()
            wandb.log({"Loss_Imi": loss_imitation}, step=i_episode)
            wandb.log({"Loss_DQN": loss_dqn}, step=i_episode)
        rospy.sleep(0.1)
        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        current_episode_reward += reward
        action_index = [action_vx_index, action_vz_index]
        apf_index = [vx_ugv_mapped, vz_ugv_mapped]
        agent.replay_buffer.add(state1, state2, action_index, apf_index, reward, next_state1, next_state2, terminal)
        if terminal:
            if termination_state == "arrival":
                ep_success_list.append(1)
            else:
                ep_success_list.append(0)
            break
        state1 = next_state1
        state2 = next_state2
    if i_episode % agent.epsilon_period == 0:
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - 0.05)
    ep_reward_list.append(current_episode_reward)
    ep_step_list.append(t + 1)

    if len(ep_success_list) >= 50:
        # Use the latest N results to calculate the success rate.
        recent_results = ep_success_list[-50:]
        success_rate = sum(recent_results) / 50
    else:
        success_rate = sum(ep_success_list) / len(ep_success_list)

    print(
        "Episode:{} \t\t step:{} \t\t current_episode_reward:{:.2f} \t\t epsilon:{:.2f}".format(
            i_episode, t + 1, current_episode_reward, agent.epsilon
        )
    )
    wandb.log({"Reward": current_episode_reward}, step=i_episode)
    wandb.log({"Step": t + 1}, step=i_episode)
    wandb.log({"Success Rate": success_rate}, step=i_episode)
    reward_file_path = model_path + "Reward"
    step_file_path = model_path + "Step"
    success_rate_file_path = model_path + "Success_Rate"
    # If the file exists, it is opened in append mode; otherwise, a new file is created
    mode = "a" if os.path.exists(success_rate_file_path) else "w"
    with open(success_rate_file_path, mode) as f:
        f.write(f"{success_rate}\n")
    if (i_episode + 1) % 500 == 0:
        mode = "a" if os.path.exists(reward_file_path) else "w"
        with open(reward_file_path, mode) as f:
            for reward in ep_reward_list:
                f.write(f"{reward}\n")
        ep_reward_list = []
        mode = "a" if os.path.exists(step_file_path) else "w"
        with open(step_file_path, mode) as f:
            for step in ep_step_list:
                f.write(f"{step}\n")
        ep_step_list = []
        # Save the model and ONNX file
        agent.save_model(pth_path)
        agent.save_onnx_model(onnx_path)
        onnx_file_name = "Model_" + time.strftime("%Y-%m-%d") + f"_{i_episode + 1}.onnx"
        agent.save_onnx_model(model_path + onnx_file_name)
