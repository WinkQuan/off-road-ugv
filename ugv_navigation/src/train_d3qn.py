#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import env
import d3qn  # 导入新的纯净版 D3QN
import numpy as np
import random
import time
import torch
import rospy
import wandb

# 设置随机数种子
random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4)

# 可视化数据工具 - 命名为 D3QN Baseline 以作区分
wandb.login()
wandb.init(project="OFF-ROAD-UGV", name="D3QN_" + time.strftime("%Y-%m-%d %H:%M:%S"))

# 设置训练总轮数、最大步长
total_episode = 2000
max_step_per_episode = 100

# 设置模型保存路径和模型文件名
model_path = "Model/D3QN/"
pth_path = model_path + "model.pth"
onnx_path = model_path + "model_d3qn.onnx"

os.makedirs(model_path, exist_ok=True)

# 初始化环境和智能体
gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = d3qn.D3QN(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    target_update=4,
    gamma=0.99,
    learning_rate=1e-4,
    epsilon=1.0,  # RL 初期需要较高的探索率
    epsilon_min=0.1,
    epsilon_period=50,
    network="Duel",  # 维持 Dueling 网络结构构成 D3QN
)

# 存储每个episode的奖励、步数、成功状态
ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)

# 开始训练
for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0

    # 每一步的操作
    for t in range(max_step_per_episode):
        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz

        # 仅通过网络获取动作，无需APF融合计算
        action_vx_index, action_vz_index = agent.get_action(state1, state2)
        print("action{}:{} {}".format(t + 1, action_space_vx[action_vx_index], action_space_vz[action_vz_index]))

        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)

        # 满足 batch 条件后进行学习
        if len(agent.replay_buffer.memory) >= agent.batch_size:
            loss_dqn = agent.learn()
            wandb.log({"Loss_DQN": loss_dqn}, step=i_episode)

        rospy.sleep(0.1)
        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        current_episode_reward += reward

        # 存入纯粹的 RL 经验
        action_index = [action_vx_index, action_vz_index]
        agent.replay_buffer.add(state1, state2, action_index, reward, next_state1, next_state2, terminal)

        if terminal:
            if termination_state == "arrival":
                ep_success_list.append(1)
            else:
                ep_success_list.append(0)
            break

        state1 = next_state1
        state2 = next_state2

    # Epsilon 衰减逻辑
    if i_episode % agent.epsilon_period == 0:
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - 0.05)

    ep_reward_list.append(current_episode_reward)
    ep_step_list.append(t + 1)

    if len(ep_success_list) >= 50:
        recent_results = ep_success_list[-50:]
        success_rate = sum(recent_results) / 50
    else:
        success_rate = sum(ep_success_list) / len(ep_success_list)

    print(
        "Episode:{} \t step:{} \t current_episode_reward:{:.2f} \t epsilon:{:.2f}".format(
            i_episode, t + 1, current_episode_reward, agent.epsilon
        )
    )

    # WandB 记录
    wandb.log({"Reward": current_episode_reward}, step=i_episode)
    wandb.log({"Step": t + 1}, step=i_episode)
    wandb.log({"Success Rate": success_rate}, step=i_episode)

    # 结果持久化保存
    reward_file_path = model_path + "Reward"
    step_file_path = model_path + "Step"
    success_rate_file_path = model_path + "Success_Rate"

    mode = "a" if os.path.exists(success_rate_file_path) else "w"
    with open(success_rate_file_path, mode) as f:
        f.write(f"{success_rate}\n")

    if (i_episode + 1) % 500 == 0:
        mode = "a" if os.path.exists(reward_file_path) else "w"
        with open(reward_file_path, mode) as f:
            for r in ep_reward_list:
                f.write(f"{r}\n")
        ep_reward_list = []

        mode = "a" if os.path.exists(step_file_path) else "w"
        with open(step_file_path, mode) as f:
            for s in ep_step_list:
                f.write(f"{s}\n")
        ep_step_list = []

        # Save model and ONNX file
        agent.save_model(pth_path)
        agent.save_onnx_model(onnx_path)
        onnx_file_name = "Model_" + time.strftime("%Y-%m-%d") + f"_{i_episode + 1}.onnx"
        agent.save_onnx_model(model_path + onnx_file_name)
