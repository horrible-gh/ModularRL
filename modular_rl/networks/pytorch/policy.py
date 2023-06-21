# -*- coding: utf-8 -*-
"""
ModularRL project

Copyright (c) 2023 horrible

PolicyNetwork is a neural network that takes in the current state of the environment as input and outputs a probability distribution over the available actions. 
The purpose of this network is to determine the best action to take in the current state.

ValueNetwork, on the other hand, 
is a neural network that takes in the current state of the environment as input and outputs a value that represents the expected future reward that can be obtained from that state. 
The purpose of this network is to estimate the value of a given state so that the agent can make decisions that lead to the most reward in the long term.

Both networks are important components of the Proximal Policy Optimization (PPO) algorithm, 
which is a type of reinforcement learning algorithm used for training agents to perform tasks in an environment.

This software includes the following third-party libraries:
PyTorch  (BSD-Style License): https://pytorch.org/ - Copyright (c) Facebook.

"""
import torch.nn as nn
import torch


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_option='medium', num_layers=2):
        """
        Initialize the PolicyNetwork class with the specified input dimension, output dimension, and network settings.

        :param input_dim: The dimension of the input layer.
        :type input_dim: int
        :param output_dim: The dimension of the output layer.
        :type output_dim: int
        :param hidden_option: The size of the hidden layers (options are 'small', 'medium', 'large', 'huge', or 'exhuge').
        :type hidden_option: str
        :param num_layers: The number of hidden layers.
        :type num_layers: int
        """

        super(PolicyNetwork, self).__init__()

        hidden_dim_map = {
            'small': 32,
            'medium': 64,
            'large': 128,
            'huge': 256,
            'exhuge': 512
        }

        hidden_dim = hidden_dim_map[hidden_option]

        # Create each layer
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim * i, hidden_dim * (i + 1)))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Compute a forward pass of the policy network.

        :param x: The input tensor.
        :type x: torch.Tensor
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        x = self.net(x)
        x = torch.softmax(x, dim=-1)
        return x
