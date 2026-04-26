#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import time

import rospy
import wandb

import env
import paldrl_ablation


TOTAL_EPISODE = 2000
MAX_STEP_PER_EPISODE = 100
UGV_MASS = 1.48


def parse_args():
    parser = argparse.ArgumentParser(description="Train PAL-DRL ablation variants.")
    parser.add_argument("--variant", required=True, choices=sorted(paldrl_ablation.ABLATION_VARIANTS.keys()))
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def append_lines(file_path, values):
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode) as file_obj:
        for value in values:
            file_obj.write(f"{value}\n")


def main():
    args = parse_args()
    variant_config = paldrl_ablation.ABLATION_VARIANTS[args.variant]
    paldrl_ablation.set_random_seed(args.seed)

    wandb.login()
    wandb.init(
        project="OFF-ROAD-UGV",
        name=f"PALDRL-Ablation_{args.variant}_seed{args.seed}_{time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    model_path = os.path.join("Model", "PAL-DRL-Ablation", args.variant, f"seed_{args.seed}")
    pth_path = os.path.join(model_path, "model.pth")
    onnx_path = os.path.join(model_path, "model_pal_drl_ablation.onnx")
    os.makedirs(model_path, exist_ok=True)

    gazebo_ugv = env.GazeboUGV(max_step=MAX_STEP_PER_EPISODE)
    agent = paldrl_ablation.AblationPALDRLAgent(
        gazebo_ugv,
        gazebo_ugv.action_space_vx,
        gazebo_ugv.action_space_vz,
        use_dropout=variant_config["use_dropout"],
        batch_size=64,
        memory_size=10000,
        target_update=4,
        gamma=0.99,
        learning_rate=1e-4,
        epsilon=variant_config["epsilon"],
        network="Duel",
    )

    ep_reward_list = []
    ep_step_list = []
    ep_success_list = []

    print(f"Variant: {args.variant} -> {variant_config['display_name']}")
    print(f"Seed: {args.seed}")
    print("Action Space_vx = ", gazebo_ugv.action_space_vx)
    print("Action Space_vz = ", gazebo_ugv.action_space_vz)

    for i_episode in range(TOTAL_EPISODE + 1):
        state1, state2, dist_normalized = gazebo_ugv.reset()
        current_episode_reward = 0.0
        episode_imitation_losses = []
        episode_dqn_losses = []

        for t in range(MAX_STEP_PER_EPISODE):
            apf_vx_index, apf_vz_index = paldrl_ablation.compute_apf_teacher_indices(
                gazebo_ugv, variant_config["teacher_mode"], UGV_MASS
            )

            action_vx_index, action_vz_index = agent.get_action(state1, state2, dist_normalized)
            print(
                "action{}:{} {}".format(
                    t + 1,
                    gazebo_ugv.action_space_vx[action_vx_index],
                    gazebo_ugv.action_space_vz[action_vz_index],
                )
            )

            gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)
            rospy.sleep(0.1)

            next_state1, next_state2, terminal, reward, termination_state = paldrl_ablation.step_with_reward_mode(
                gazebo_ugv, t + 1, variant_config["reward_mode"]
            )
            current_episode_reward += reward

            action_index = [action_vx_index, action_vz_index]
            apf_index = [apf_vx_index, apf_vz_index]
            agent.replay_buffer.add(state1, state2, action_index, apf_index, reward, next_state1, next_state2, terminal)

            if len(agent.replay_buffer.memory) >= agent.batch_size:
                loss_imitation, loss_dqn = agent.learn()
                episode_imitation_losses.append(loss_imitation)
                episode_dqn_losses.append(loss_dqn)

            if terminal:
                ep_success_list.append(1 if termination_state == "arrival" else 0)
                break

            state1 = next_state1
            state2 = next_state2

        ep_reward_list.append(current_episode_reward)
        ep_step_list.append(t + 1)

        if len(ep_success_list) >= 50:
            success_rate = sum(ep_success_list[-50:]) / 50.0
        else:
            success_rate = sum(ep_success_list) / len(ep_success_list)

        mean_loss_imi = (
            sum(episode_imitation_losses) / len(episode_imitation_losses) if episode_imitation_losses else 0.0
        )
        mean_loss_dqn = sum(episode_dqn_losses) / len(episode_dqn_losses) if episode_dqn_losses else 0.0

        print(
            "Episode:{} \t step:{} \t reward:{:.2f} \t success_rate:{:.2f} \t loss_imi:{:.4f} \t loss_dqn:{:.4f}".format(
                i_episode, t + 1, current_episode_reward, success_rate, mean_loss_imi, mean_loss_dqn
            )
        )

        wandb.log(
            {
                "Reward": current_episode_reward,
                "Step": t + 1,
                "Success Rate": success_rate,
                "Loss_Imi": mean_loss_imi,
                "Loss_DQN": mean_loss_dqn,
                "Variant": args.variant,
                "Seed": args.seed,
            },
            step=i_episode,
        )

        success_rate_file_path = os.path.join(model_path, "Success_Rate")
        mode = "a" if os.path.exists(success_rate_file_path) else "w"
        with open(success_rate_file_path, mode) as file_obj:
            file_obj.write(f"{success_rate}\n")

        if (i_episode + 1) % 500 == 0:
            append_lines(os.path.join(model_path, "Reward"), ep_reward_list)
            append_lines(os.path.join(model_path, "Step"), ep_step_list)
            ep_reward_list = []
            ep_step_list = []

            agent.save_model(pth_path)
            agent.save_onnx_model(onnx_path)
            onnx_file_name = f"Model_{time.strftime('%Y-%m-%d')}_{i_episode + 1}.onnx"
            agent.save_onnx_model(os.path.join(model_path, onnx_file_name))


if __name__ == "__main__":
    main()
