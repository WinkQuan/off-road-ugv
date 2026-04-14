#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Training script for BC (Behavior Cloning) algorithm.
BC is trained on APF expert demonstrations and evaluated periodically with its own policy.
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
import bc
import env


random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4)

wandb.login()
wandb.init(project="OFF-ROAD-UGV", name="BC_" + time.strftime("%Y-%m-%d %H:%M:%S"))

total_episode = 2000
max_step_per_episode = 100
ugv_mass = 1.48
eval_interval = 100
eval_episodes = 5
save_interval = 500

model_path = "Model/BC/"
pth_path = model_path + "model_bc.pth"
onnx_path = model_path + "model_bc.onnx"

os.makedirs(model_path, exist_ok=True)

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

eval_reward_buffer = []
eval_step_buffer = []
eval_success_rate_buffer = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
print("=" * 50)
print("Training BC Agent (Behavior Cloning)")
print("APF executes demonstrations; BC policy is evaluated every 100 episodes")
print("=" * 50)


def append_metrics(file_path, values):
    if not values:
        return
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode) as file_obj:
        for value in values:
            file_obj.write(f"{value}\n")


def evaluate_bc_policy(agent_obj, env_obj, num_episodes):
    success_count = 0
    total_reward = 0.0
    total_steps = 0.0

    for _ in range(num_episodes):
        state1, state2, dist_normalized = env_obj.reset()
        episode_reward = 0.0

        for t in range(max_step_per_episode):
            action_vx_index, action_vz_index = agent_obj.get_action(state1, state2, dist_normalized)
            env_obj.execute_linear_velocity(action_vx_index, action_vz_index)
            rospy.sleep(0.1)

            next_state1, next_state2, terminal, reward, termination_state = env_obj.step(time_step=t + 1)
            episode_reward += reward

            if terminal:
                if termination_state == "arrival":
                    success_count += 1
                break

            state1 = next_state1
            state2 = next_state2

        total_reward += episode_reward
        total_steps += t + 1

    eval_success_rate = success_count / num_episodes
    eval_average_reward = total_reward / num_episodes
    eval_average_step = total_steps / num_episodes
    return eval_success_rate, eval_average_reward, eval_average_step


for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0.0
    episode_success = False
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
        apf_vx_index = APF_Vel_ROS.fuzzy_map_v_triangular(linear_cmd, action_space_vx, strategy="min")
        apf_vz_index = APF_Vel_ROS.fuzzy_map_v_triangular(angular_cmd, action_space_vz, strategy="max")

        agent.replay_buffer.add(state1, state2, apf_vx_index, apf_vz_index)
        gazebo_ugv.execute_linear_velocity(apf_vx_index, apf_vz_index)
        rospy.sleep(0.1)

        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        current_episode_reward += reward

        if len(agent.replay_buffer.memory) >= agent.batch_size:
            loss_bc, _ = agent.learn()
            episode_losses.append(loss_bc)

        if terminal:
            if termination_state == "arrival":
                episode_success = True
            break

        state1 = next_state1
        state2 = next_state2

    mean_loss_bc = float(np.mean(episode_losses)) if episode_losses else 0.0
    expert_step = t + 1
    expert_success = 1 if episode_success else 0

    print(
        "Episode:{} \t expert_step:{} \t expert_reward:{:.2f} \t loss_bc:{:.4f}".format(
            i_episode, expert_step, current_episode_reward, mean_loss_bc
        )
    )

    log_data = {
        "Expert Reward": current_episode_reward,
        "Expert Step": expert_step,
        "Expert Success": expert_success,
        "Loss_BC": mean_loss_bc,
    }

    if (i_episode + 1) % eval_interval == 0:
        eval_success_rate, eval_average_reward, eval_average_step = evaluate_bc_policy(agent, gazebo_ugv, eval_episodes)
        eval_reward_buffer.append(eval_average_reward)
        eval_step_buffer.append(eval_average_step)
        eval_success_rate_buffer.append(eval_success_rate)

        print(
            "Evaluation:{} \t reward:{:.2f} \t step:{:.2f} \t success_rate:{:.2f}".format(
                i_episode + 1, eval_average_reward, eval_average_step, eval_success_rate
            )
        )

        log_data.update(
            {
                "Reward": eval_average_reward,
                "Step": eval_average_step,
                "Success Rate": eval_success_rate,
            }
        )

    wandb.log(log_data, step=i_episode)

    if (i_episode + 1) % save_interval == 0:
        append_metrics(model_path + "Reward", eval_reward_buffer)
        append_metrics(model_path + "Step", eval_step_buffer)
        append_metrics(model_path + "Success_Rate", eval_success_rate_buffer)
        eval_reward_buffer = []
        eval_step_buffer = []
        eval_success_rate_buffer = []

        agent.save_model(pth_path)
        agent.save_onnx_model(onnx_path)
        onnx_file_name = "Model_" + time.strftime("%Y-%m-%d") + f"_{i_episode + 1}.onnx"
        agent.save_onnx_model(model_path + onnx_file_name)
