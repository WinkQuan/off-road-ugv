#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
BC (Behavior Cloning) Algorithm Implementation
Pure imitation learning from expert demonstrations
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

        # Torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

        # Network for behavior cloning
        self.policy_net = DQNNet(
            network=self.network, action_space_vx=self.action_space_vx, action_space_vz=self.action_space_vz
        ).to(self.device)

        # Use only classification loss (CrossEntropyLoss)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        # Replay buffer to store experiences
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        print("BC (Behavior Cloning) Agent initialized - Pure imitation learning")

    def get_action(self, state1, state2, dist_normalized):
        self.policy_net.eval()
        with torch.no_grad():
            state1 = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2 = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            q_values_vx, q_values_vz = self.policy_net(state1, state2)

            # In BC, we directly use the policy network outputs
            action_vx_index = np.argmax(q_values_vx.cpu().detach().numpy())
            action_vz_index = np.argmax(q_values_vz.cpu().detach().numpy())

        return action_vx_index, action_vz_index

    def learn(self):
        """
        BC learning: minimize the difference between policy and expert (APF) actions
        No Q-learning, only imitation loss
        """
        self.policy_net.train()

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

        # Get policy outputs
        q_values_vx, q_values_vz = self.policy_net(states1, states2)

        # BC Loss: Cross entropy between policy and expert demonstrations
        loss_bc_vx = self.loss_fn(q_values_vx.squeeze(1), apf_index_vx.squeeze(1))
        loss_bc_vz = self.loss_fn(q_values_vz.squeeze(1), apf_index_vz.squeeze(1))
        loss_bc = loss_bc_vx + loss_bc_vz

        # Optimize
        self.optimizer.zero_grad()
        loss_bc.backward()
        self.optimizer.step()

        # BC only returns imitation loss (no DQN loss)
        return loss_bc.item(), 0.0

    def save_model(self, path):
        checkpoint = {"model_states": self.policy_net.state_dict(), "optimizer_states": self.optimizer.state_dict()}
        torch.save(checkpoint, path)

    def save_onnx_model(self, param_path_onnx):
        self.policy_net.eval()
        dummy_state1 = torch.randn(64, 480, 640, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(
            self.policy_net,
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
            self.policy_net.load_state_dict(checkpoint["model_states"])
            self.optimizer.load_state_dict(checkpoint["optimizer_states"])
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")

    @staticmethod
    def load_onnx_model(onnx_file_path):
        sess = ort.InferenceSession(onnx_file_path)
        return sess
