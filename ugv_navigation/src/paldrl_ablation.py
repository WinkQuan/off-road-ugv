#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import random
from collections import deque

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim

import APF_Vel_ROS


ABLATION_VARIANTS = {
    "full": {
        "display_name": "PAL-DRL",
        "use_dropout": True,
        "teacher_mode": "local",
        "reward_mode": "posture_aware",
        "epsilon": 0.1,
    },
    "wo_dropout": {
        "display_name": "PAL-DRL w/o Dropout",
        "use_dropout": False,
        "teacher_mode": "local",
        "reward_mode": "posture_aware",
        "epsilon": 0.1,
    },
    "wo_local_teacher": {
        "display_name": "PAL-DRL w/o Local APF Teacher",
        "use_dropout": True,
        "teacher_mode": "full",
        "reward_mode": "posture_aware",
        "epsilon": 0.1,
    },
    "wo_posture_reward": {
        "display_name": "PAL-DRL w/o Posture-Aware Reward",
        "use_dropout": True,
        "teacher_mode": "local",
        "reward_mode": "base",
        "epsilon": 0.1,
    },
    "wo_exploration": {
        "display_name": "PAL-DRL w/o Exploration",
        "use_dropout": True,
        "teacher_mode": "local",
        "reward_mode": "posture_aware",
        "epsilon": 0.0,
    },
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AblationReplayBuffer:
    def __init__(self, max_size=100000):
        super(AblationReplayBuffer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    def add(self, state1, state2, action_index, apf_index, reward, next_state1, next_state2, done):
        self.memory.append((state1, state2, action_index, apf_index, reward, next_state1, next_state2, done))

    def sample_and_process(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states1, states2, action_indices, apf_indices, rewards, next_states1, next_states2, dones = zip(*batch)

        states1 = torch.FloatTensor(np.stack(states1)).to(self.device)
        states2 = torch.FloatTensor(np.stack(states2)).to(self.device)
        action_indices = torch.LongTensor(np.stack(action_indices)).to(self.device)
        action_index_vx = action_indices[:, 0].long().view(-1, 1)
        action_index_vz = action_indices[:, 1].long().view(-1, 1)
        apf_indices = torch.LongTensor(np.stack(apf_indices)).to(self.device)
        apf_index_vx = apf_indices[:, 0].long().view(-1, 1)
        apf_index_vz = apf_indices[:, 1].long().view(-1, 1)
        rewards = torch.FloatTensor(np.stack(rewards)).to(self.device)
        next_states1 = torch.FloatTensor(np.stack(next_states1)).to(self.device)
        next_states2 = torch.FloatTensor(np.stack(next_states2)).to(self.device)
        dones = torch.FloatTensor(np.stack(dones)).to(self.device)

        return (
            states1,
            states2,
            action_index_vx,
            action_index_vz,
            apf_index_vx,
            apf_index_vz,
            rewards,
            next_states1,
            next_states2,
            dones,
        )


class AblationPALDRLNet(nn.Module):
    def __init__(self, network, action_space_vx, action_space_vz, use_dropout=True):
        super(AblationPALDRLNet, self).__init__()
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.network = network
        self.use_dropout = use_dropout

        cnn_layers = [
            nn.Conv2d(12, 32, kernel_size=(4, 3), stride=4),
            nn.ReLU(),
        ]
        if self.use_dropout:
            cnn_layers.append(nn.Dropout(0.1))
        cnn_layers.extend(
            [
                nn.Conv2d(32, 64, kernel_size=(4, 3), stride=3),
                nn.ReLU(),
            ]
        )
        if self.use_dropout:
            cnn_layers.append(nn.Dropout(0.1))

        self.fc_target = nn.Linear(2, 64)
        self.cnn_a = nn.Sequential(*cnn_layers)
        self.fc_1 = nn.Linear(64 * 18 * 18 + 64, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.output_vx = nn.Linear(256, len(self.action_space_vx))
        self.output_vz = nn.Linear(256, len(self.action_space_vz))

        self.advantage_vx = nn.Linear(128, len(self.action_space_vx))
        self.value_vx = nn.Linear(128, 1)
        self.advantage_vz = nn.Linear(128, len(self.action_space_vz))
        self.value_vz = nn.Linear(128, 1)

    def forward(self, state1, state2):
        batch_size = state1.size(0)
        img = state1 / 255.0
        x3 = self.cnn_a(img.transpose(1, 3))

        x_target = F.relu(self.fc_target(state2))
        x_merge = torch.cat((x3.view(batch_size, -1), x_target), axis=1)
        fc_1 = F.relu(self.fc_1(x_merge))
        fc_2 = F.relu(self.fc_2(fc_1))

        if self.network == "Duel":
            advantage_vx, value_vx = torch.split(fc_2, 128, dim=1)
            advantage_vx = self.advantage_vx(advantage_vx)
            value_vx = self.value_vx(value_vx)
            vx_output = value_vx + advantage_vx - torch.mean(advantage_vx, dim=1, keepdim=True)

            advantage_vz, value_vz = torch.split(fc_2, 128, dim=1)
            advantage_vz = self.advantage_vz(advantage_vz)
            value_vz = self.value_vz(value_vz)
            vz_output = value_vz + advantage_vz - torch.mean(advantage_vz, dim=1, keepdim=True)
        else:
            vx_output = self.output_vx(fc_2)
            vz_output = self.output_vz(fc_2)

        return vx_output, vz_output


class AblationPALDRLAgent:
    def __init__(
        self,
        env,
        action_space_vx,
        action_space_vz,
        use_dropout,
        memory_size=50000,
        learning_rate=1e-4,
        batch_size=64,
        target_update=4,
        gamma=0.99,
        epsilon=0.1,
        network="Duel",
    ):
        super(AblationPALDRLAgent, self).__init__()
        self.env = env
        self.network = network
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.use_dropout = use_dropout

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(
            "Using Device:",
            torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
        )

        self.predict_net = AblationPALDRLNet(
            network=self.network,
            action_space_vx=self.action_space_vx,
            action_space_vz=self.action_space_vz,
            use_dropout=self.use_dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.target_net = AblationPALDRLNet(
            network=self.network,
            action_space_vx=self.action_space_vx,
            action_space_vz=self.action_space_vz,
            use_dropout=self.use_dropout,
        ).to(self.device)
        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.target_net.eval()

        self.target_update = target_update
        self.update_count = 0
        self.replay_buffer = AblationReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = 0.1
        self.decay_counter = 0

        print(
            f"Ablation PAL-DRL initialized: network={self.network}, dropout={self.use_dropout}, epsilon={self.epsilon}"
        )

    def get_action(self, state1, state2, dist_normalized=None):
        self.predict_net.eval()
        with torch.no_grad():
            state1 = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2 = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            q_values_vx, q_values_vz = self.predict_net(state1, state2)

            if np.random.random() < self.epsilon:
                action_vx_index = np.random.randint(0, len(self.action_space_vx))
                action_vz_index = np.random.randint(0, len(self.action_space_vz))
            else:
                action_vx_index = int(np.argmax(q_values_vx.cpu().detach().numpy()))
                action_vz_index = int(np.argmax(q_values_vz.cpu().detach().numpy()))
        return action_vx_index, action_vz_index

    def learn(self):
        self.predict_net.train()
        (
            states1,
            states2,
            action_index_vx,
            action_index_vz,
            apf_index_vx,
            apf_index_vz,
            rewards,
            next_states1,
            next_states2,
            dones,
        ) = self.replay_buffer.sample_and_process(self.batch_size)

        q_values_vx, q_values_vz = self.predict_net(states1, states2)
        loss_imitation_vx = F.cross_entropy(q_values_vx.squeeze(1), apf_index_vx.squeeze(1))
        loss_imitation_vz = F.cross_entropy(q_values_vz.squeeze(1), apf_index_vz.squeeze(1))
        loss_imitation = loss_imitation_vx + loss_imitation_vz
        predict_values_vx = q_values_vx.gather(1, action_index_vx.view(-1, 1))
        predict_values_vz = q_values_vz.gather(1, action_index_vz.view(-1, 1))

        if self.network == "Duel" or self.network == "Double":
            with torch.no_grad():
                q_values_vx_pred, q_values_vz_pred = self.predict_net(next_states1, next_states2)
                _, actions_prime_vx = torch.max(q_values_vx_pred, 1)
                _, actions_prime_vz = torch.max(q_values_vz_pred, 1)

                target_q_values_vx, target_q_values_vz = self.target_net(next_states1, next_states2)
                q_target_value_vx = target_q_values_vx.gather(1, actions_prime_vx.view(-1, 1))
                q_target_value_vz = target_q_values_vz.gather(1, actions_prime_vz.view(-1, 1))

                target_values_vx = rewards.view(-1, 1) + self.gamma * q_target_value_vx * (1 - dones).view(-1, 1)
                target_values_vz = rewards.view(-1, 1) + self.gamma * q_target_value_vz * (1 - dones).view(-1, 1)
        else:
            with torch.no_grad():
                q_values_target_vx, q_values_target_vz = self.target_net(next_states1, next_states2)
                target_values_vx = (rewards + self.gamma * torch.max(q_values_target_vx, 1)[0] * (1 - dones)).view(
                    -1, 1
                )
                target_values_vz = (rewards + self.gamma * torch.max(q_values_target_vz, 1)[0] * (1 - dones)).view(
                    -1, 1
                )

        loss_dqn_vx = self.loss_fn(predict_values_vx, target_values_vx)
        loss_dqn_vz = self.loss_fn(predict_values_vz, target_values_vz)
        loss_dqn = loss_dqn_vx + loss_dqn_vz
        loss = self.alpha * loss_dqn + (1 - self.alpha) * loss_imitation

        self.decay_counter += 1
        if self.decay_counter % 500 == 0:
            self.alpha += 0.05
            self.alpha = min(self.alpha, 0.9)
            print(f"Weight of the DQN Loss is set to {self.alpha}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count >= self.target_update:
            self.target_net.load_state_dict(self.predict_net.state_dict())
            self.update_count = 0

        return loss_imitation.item(), loss_dqn.item()

    def save_model(self, path):
        checkpoint = {
            "model_states": self.predict_net.state_dict(),
            "target_model_states": self.target_net.state_dict(),
            "optimizer_states": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def save_onnx_model(self, param_path_onnx):
        self.predict_net.eval()
        dummy_state1 = torch.randn(64, 224, 224, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(
            self.predict_net,
            (dummy_state1, dummy_state2),
            param_path_onnx,
            input_names=["dummy_state1", "dummy_state2"],
            output_names=["output_velocity_x", "output_velocity_y"],
            dynamic_axes={"dummy_state1": {0: "batch_size"}, "dummy_state2": {0: "batch_size"}},
        )

    def load_model(self, filename, device):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.predict_net.load_state_dict(checkpoint["model_states"])
            target_states = checkpoint.get("target_model_states", checkpoint["model_states"])
            self.target_net.load_state_dict(target_states)
            self.optimizer.load_state_dict(checkpoint["optimizer_states"])
            self.predict_net.eval()
            self.target_net.eval()
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")

    @staticmethod
    def load_onnx_model(onnx_file_path):
        return ort.InferenceSession(onnx_file_path)


def compute_apf_teacher_indices(gazebo_ugv, teacher_mode, ugv_mass):
    target_location = np.array(gazebo_ugv.goal)
    current_position = np.array(gazebo_ugv.self_state[0:2])

    if teacher_mode == "local":
        obs_pos = np.array(gazebo_ugv.get_visible_obstacles(), dtype=np.float32)
    elif teacher_mode == "full":
        obs_pos = np.array(gazebo_ugv.cylinder_pos, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported teacher_mode: {teacher_mode}")

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
    vx_ugv_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(linear_cmd, action_space_vx, strategy="min")
    vz_ugv_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(angular_cmd, action_space_vz, strategy="max")
    return vx_ugv_mapped, vz_ugv_mapped


def step_with_reward_mode(gazebo_ugv, time_step, reward_mode):
    d1, alpha1 = gazebo_ugv.goal2robot()
    gazebo_ugv.position = [d1, alpha1]
    gazebo_ugv.dist = d1

    if reward_mode == "posture_aware":
        terminal, reward, termination_state = gazebo_ugv.posture_aware_reward(time_step)
    elif reward_mode == "base":
        terminal, reward, termination_state = gazebo_ugv.get_reward_and_terminate(time_step)
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    gazebo_ugv.reward = reward
    gazebo_ugv.dist_init = gazebo_ugv.dist
    img, pos = gazebo_ugv.get_states()
    return img, pos, terminal, reward, termination_state
