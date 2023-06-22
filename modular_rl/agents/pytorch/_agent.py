# -*- coding: utf-8 -*-
"""
ModularRL project

Copyright (c) 2023 horrible-gh

ModularRL is a Python library for creating and training reinforcement learning agents using various algorithms.
The library is designed to be easily customizable and modular,
allowing users to quickly set up and train agents for various environments without being limited to a specific algorithm.

This software includes the following third-party libraries:
Gym (MIT License): https://github.com/openai/gym - Copyright (c) OpenAI.
NumPy (BSD License): https://numpy.org - Copyright (c) NumPy Developers.
PyTorch  (BSD-Style License): https://pytorch.org/ - Copyright (c) Facebook.
"""

import gym
from LogAssist.log import Logger
import torch
import torch.optim as optim
from modular_rl.networks.pytorch.policy import PyTorchPolicyNetwork
from modular_rl.networks.pytorch.value import PyTorchValueNetwork
from modular_rl.networks.pytorch.actor_critic import PyTorchActorCriticNetwork
from modular_rl.agents._common._agents import CommonAgents


class Agent(CommonAgents):
    def __init__(self, env, setting):
        super().__init__(env, setting)

    def reset(self):
        super().reset()

    def _check_state(self, state):
        return super()._check_state(state)

    def learn_reset(self):
        return super().learn_reset()

    def learn_close(self):
        super().learn_close

    def learn_check(self):
        super().learn_check()

    def update_step(self, state, action, reward, done, next_state):
        super().update_step(state, action, reward, done, next_state)

    def step_unpack(self, step_output):
        return super().step_unpack(step_output)

    def update_reward(self, reward):
        super().update_reward(reward)

    def update_episode(self):
        super().update_episode()

    def init_policy_value(self):
        """
        Initializes policy and value networks, and their respective optimizers.
        """

        # Create neural network instances and optimizer
        networks_size = self.setting.get('networks', 'middle')
        Logger.verb('_agent:init_policy_value',
                    f'Initialize policy and value networks to {networks_size}')
        self.policy_net = PyTorchPolicyNetwork(
            self.state_dim, self.action_dim, networks_size)
        self.value_net = PyTorchValueNetwork(
            self.state_dim, networks_size)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.setting.get('optimizer_speed', 3e-4))
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.setting.get('optimizer_speed', 3e-4))

    def init_actor_critic(self):
        """
        Initializes the actor-critic network and its optimizer.
        """

        # Neural Network
        self.actor_critic_net = PyTorchActorCriticNetwork(
            self.state_dim, self.action_dim)
        self.actor_critic_optimizer = optim.Adam(
            self.actor_critic_net.parameters(), lr=self.setting.get('optimizer_speed', 3e-4))

    def select_action(self, state):
        '''
        These functions are placeholders and must be implemented by the child class that extends this Agent class.

        select_action() function is a placeholder that needs to be implemented in the child class that extends the Agent class. This function takes the current state of the environment and returns the selected action for the agent to take.

        :param state: The current state of the environment.
        :return: The selected action for the agent to take.
        '''
        pass

    def update(self):
        '''
        This function is a placeholder and must be implemented by the child class that extends this Agent class.

        update() function is a placeholder that needs to be implemented in the child class that extends the Agent class. This function is responsible for updating the agent's state, action, and policy based on the new state and reward received from the environment.

        No parameters are passed into this function and it does not return anything.
        '''
        pass

    def save_policy_value(self, file_name):
        """
        Save the policy and value networks and their optimizer states in a file.

        :param file_name: The name of the file to save the networks and optimizer states.
        :type file_name: str
        """

        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, file_name)

    def load_policy_value(self, file_name):
        """
        Load the policy and value networks and their optimizer states from a file.

        :param file_name: The name of the file to load the networks and optimizer states from.
        :type file_name: str
        """

        checkpoint = torch.load(file_name)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(
            checkpoint['value_optimizer_state_dict'])

    def save_actor_critic(self, file_name):
        """
        Save the actor-critic network and its optimizer state in a file.

        :param file_name: The name of the file to save the network and optimizer state.
        :type file_name: str
        """

        torch.save({
            'actor_critic_net_state_dict': self.actor_critic_net.state_dict(),
            'optimizer_state_dict': self.actor_critic_optimizer.state_dict(),
        }, file_name)

    def load_actor_critic(self, file_name):
        """
        Load the actor-critic network and its optimizer state from a file.

        :param file_name: The name of the file to load the network and optimizer state from.
        :type file_name: str
        """

        checkpoint = torch.load(file_name)
        self.actor_critic_net.load_state_dict(
            checkpoint['actor_critic_net_state_dict'])
        self.actor_critic_optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])

    def convert_float_to_double(self, input_tensor):
        if input_tensor.dtype == torch.float32:
            return input_tensor.double()
        else:
            return input_tensor

    def ensure_float(self, input_tensor):
        if input_tensor.dtype == torch.float64:
            return input_tensor.float()
        else:
            return input_tensor
