# -*- coding: utf-8 -*-
"""
ModularRL project

Copyright (c) 2023 horrible

This software includes the following third-party libraries:
PyTorch  (BSD-Style License): https://pytorch.org/ - Copyright (c) Facebook.

"""
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
