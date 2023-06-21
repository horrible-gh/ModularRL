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
        '''
        This code defines a class ActorCriticNetwork that implements an actor-critic network using PyTorch.
        The actor-critic algorithm is a popular reinforcement learning algorithm that is used to learn how to maximize rewards in an environment by having an agent interact with the environment.
        This network takes as input the state and action dimensions, and the number of neurons in the hidden layers of both the actor and critic networks.
        It then predicts state-value and action-value for the given input.

        :param state_dim: (int) The dimensionality of the state space.
        :param action_dim: (int) The dimensionality of the action space.
        :param hidden_size: (int) The number of neurons in the hidden layers of both the actor and critic networks.
        '''

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
        '''
        The forward method of this class takes a state tensor of shape (batch_size, state_dim) as input and returns the action probabilities and state value.
        The :param comments provide additional information about the inputs to the functions.

        :param state: (tensor) The input state tensor of shape (batch_size, state_dim).
        '''
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
