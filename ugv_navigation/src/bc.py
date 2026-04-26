#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
BC (Behavior Cloning) Algorithm Implementation
Pure imitation learning from expert demonstrations
"""

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


class ReplayBuffer:
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    def add(self, state1, state2, expert_index_vx, expert_index_vz):
        self.memory.append((state1, state2, expert_index_vx, expert_index_vz))

    def sample_and_process(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states1, states2, expert_index_vx, expert_index_vz = zip(*batch)

        states1 = torch.FloatTensor(np.stack(states1)).to(self.device)
        states2 = torch.FloatTensor(np.stack(states2)).to(self.device)
        expert_index_vx = torch.LongTensor(np.array(expert_index_vx)).to(self.device).view(-1, 1)
        expert_index_vz = torch.LongTensor(np.array(expert_index_vz)).to(self.device).view(-1, 1)

        return states1, states2, expert_index_vx, expert_index_vz


class DQNNet(nn.Module):
    def __init__(self, network, action_space_vx, action_space_vz):
        super(DQNNet, self).__init__()
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.network = network

        self.fc_target = nn.Linear(2, 64)
        self.cnn_a = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(4, 3), stride=4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=3),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
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
        img = state1 / 255
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


class BC:
    """
    Behavior Cloning Agent - Pure imitation learning from expert demonstrations
    Learns directly from APF (Artificial Potential Field) actions without Q-learning
    """

    def __init__(
        self,
        env,
        action_space_vx,
        action_space_vz,
        memory_size=50000,
        learning_rate=1e-3,
        batch_size=32,
        network="Duel",
    ):
        super(BC, self).__init__()
        self.env = env
        self.network = network
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.learning_rate = learning_rate

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("Using Device: CPU")

        self.policy_net = DQNNet(
            network=self.network, action_space_vx=self.action_space_vx, action_space_vz=self.action_space_vz
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        print("BC (Behavior Cloning) Agent initialized - Pure imitation learning")

    def get_action(self, state1, state2, dist_normalized=None):
        self.policy_net.eval()
        with torch.no_grad():
            state1 = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2 = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            q_values_vx, q_values_vz = self.policy_net(state1, state2)

            action_vx_index = np.argmax(q_values_vx.cpu().detach().numpy())
            action_vz_index = np.argmax(q_values_vz.cpu().detach().numpy())

        return action_vx_index, action_vz_index

    def learn(self):
        """
        BC learning: minimize the difference between policy and expert (APF) actions
        No Q-learning, only imitation loss
        """
        self.policy_net.train()
        states1, states2, expert_index_vx, expert_index_vz = self.replay_buffer.sample_and_process(self.batch_size)

        q_values_vx, q_values_vz = self.policy_net(states1, states2)

        loss_bc_vx = self.loss_fn(q_values_vx, expert_index_vx.squeeze(1))
        loss_bc_vz = self.loss_fn(q_values_vz, expert_index_vz.squeeze(1))
        loss_bc = loss_bc_vx + loss_bc_vz

        self.optimizer.zero_grad()
        loss_bc.backward()
        self.optimizer.step()

        return loss_bc.item(), 0.0

    def save_model(self, path):
        checkpoint = {"model_states": self.policy_net.state_dict(), "optimizer_states": self.optimizer.state_dict()}
        torch.save(checkpoint, path)

    def save_onnx_model(self, param_path_onnx):
        self.policy_net.eval()
        dummy_state1 = torch.randn(64, 224, 224, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(
            self.policy_net,
            (dummy_state1, dummy_state2),
            param_path_onnx,
            input_names=["dummy_state1", "dummy_state2"],
            output_names=["output_velocity_x", "output_velocity_y"],
            dynamic_axes={"dummy_state1": {0: "batch_size"}, "dummy_state2": {0: "batch_size"}},
        )

    def load_model(self, filename, device):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint["model_states"])
            self.optimizer.load_state_dict(checkpoint["optimizer_states"])
            self.policy_net.eval()
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")

    @staticmethod
    def load_onnx_model(onnx_file_path):
        sess = ort.InferenceSession(onnx_file_path)
        return sess
