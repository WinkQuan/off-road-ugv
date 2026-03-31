#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
D3QN (Dueling Double Deep Q-Network) Algorithm Implementation
Extends the existing DQN with Triple network architecture for improved stability
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import onnxruntime as ort
from ddqn import DQNNet, ReplayBuffer


class D3QN:
    """
    D3QN Agent - Triple DQN with improved target network update strategy
    Uses two target networks for more stable Q-value estimation
    """

    def __init__(
        self,
        env,
        action_space_vx,
        action_space_vz,
        memory_size=50000,
        learning_rate=4e-5,
        batch_size=32,
        target_update=1000,
        gamma=0.95,
        epsilon=0.95,
        epsilon_min=0.1,
        epsilon_period=2000,
        network="Duel",
    ):
        super(D3QN, self).__init__()
        self.env = env
        self.network = network
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz

        # Torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # Primary network
        self.predict_net = DQNNet(
            network=self.network, action_space_vx=self.action_space_vx, action_space_vz=self.action_space_vz
        ).to(self.device)
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Two target networks for D3QN
        self.target_net_1 = DQNNet(
            network=self.network, action_space_vx=self.action_space_vx, action_space_vz=self.action_space_vz
        ).to(self.device)
        self.target_net_1.load_state_dict(self.predict_net.state_dict())
        self.target_net_1.eval()

        self.target_net_2 = DQNNet(
            network=self.network, action_space_vx=self.action_space_vx, action_space_vz=self.action_space_vz
        ).to(self.device)
        self.target_net_2.load_state_dict(self.predict_net.state_dict())
        self.target_net_2.eval()

        self.target_update = target_update
        self.update_count = 0
        self.target_net_2_update_count = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        # Learning setting
        self.gamma = gamma

        # Exploration setting
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_period = epsilon_period
        self.alpha = 0.1
        self.decay_counter = 0

        print("D3QN (Dueling Double Deep Q-Network) Agent initialized")

    def get_action(self, state1, state2, dist_normalized):
        self.predict_net.eval()
        with torch.no_grad():
            state1 = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2 = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            q_values_vx, q_values_vz = self.predict_net(state1, state2)

            action_vx_index = np.argmax(q_values_vx.cpu().detach().numpy())
            action_vz_index = np.argmax(q_values_vz.cpu().detach().numpy())

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

        # D3QN: Use minimum of two target networks
        q_values_vx_pred, q_values_vz_pred = self.predict_net(next_states1, next_states2)
        _, actions_prime_vx = torch.max(q_values_vx_pred, 1)
        _, actions_prime_vz = torch.max(q_values_vz_pred, 1)

        q_target_value_vx_1 = self.target_net_1(next_states1, next_states2)[0].gather(1, actions_prime_vx.view(-1, 1))
        q_target_value_vz_1 = self.target_net_1(next_states1, next_states2)[1].gather(1, actions_prime_vz.view(-1, 1))

        q_target_value_vx_2 = self.target_net_2(next_states1, next_states2)[0].gather(1, actions_prime_vx.view(-1, 1))
        q_target_value_vz_2 = self.target_net_2(next_states1, next_states2)[1].gather(1, actions_prime_vz.view(-1, 1))

        # Take minimum of two targets for stability
        q_target_value_vx = torch.min(q_target_value_vx_1, q_target_value_vx_2)
        q_target_value_vz = torch.min(q_target_value_vz_1, q_target_value_vz_2)

        target_values_vx = rewards.view(-1, 1) + self.gamma * q_target_value_vx * (1 - dones).view(-1, 1)
        target_values_vz = rewards.view(-1, 1) + self.gamma * q_target_value_vz * (1 - dones).view(-1, 1)

        predict_values_vx = self.predict_net(states1, states2)[0].gather(1, action_index_vx.view(-1, 1))
        predict_values_vz = self.predict_net(states1, states2)[1].gather(1, action_index_vz.view(-1, 1))

        # Calculate loss
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

        # Update networks
        self.update_count += 1
        if self.update_count == self.target_update:
            self.target_net_1.load_state_dict(self.predict_net.state_dict())
            self.update_count = 0

        # Update second target network less frequently
        self.target_net_2_update_count += 1
        if self.target_net_2_update_count == self.target_update * 2:
            self.target_net_2.load_state_dict(self.predict_net.state_dict())
            self.target_net_2_update_count = 0

        return loss_imitation.item(), loss_dqn.item()

    def save_model(self, path):
        checkpoint = {"model_states": self.predict_net.state_dict(), "optimizer_states": self.optimizer.state_dict()}
        torch.save(checkpoint, path)

    def save_onnx_model(self, param_path_onnx):
        self.predict_net.eval()
        dummy_state1 = torch.randn(64, 480, 640, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(
            self.predict_net,
            (dummy_state1, dummy_state2),
            param_path_onnx,
            input_names=["dummy_state1", "dummy_state2"],
            output_names=["output_velocity_x", "output_velocity_y"],
            dynamic_axes={
                "dummy_state1": {0: "batch_size"},
                "dummy_state2": {0: "batch_size"},
            },
        )

    def load_model(self, filename, device):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.predict_net.load_state_dict(checkpoint["model_states"])
            self.optimizer.load_state_dict(checkpoint["optimizer_states"])
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")

    @staticmethod
    def load_onnx_model(onnx_file_path):
        sess = ort.InferenceSession(onnx_file_path)
        return sess
