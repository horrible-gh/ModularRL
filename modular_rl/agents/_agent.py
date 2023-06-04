# -*- coding: utf-8 -*-
"""
ModularRL project

Copyright (c) 2023 horrible-gh

Class AgentPPO is an implementation of the Proximal Policy Optimization (PPO) algorithm.
It takes an environment and a setting configuration as inputs, initializes neural network instances and optimizers,
and sets various learning parameters.
It has methods to predict an action given a state, perform a learning step, update the neural network parameters,
save and load a checkpoint, and reset learning parameters.
The class also has instance variables to keep track of episode and total rewards, previous reward, and average reward.

This software includes the following third-party libraries:
Gym (MIT License): https://github.com/openai/gym - Copyright (c) OpenAI.
NumPy (BSD License): https://numpy.org - Copyright (c) NumPy Developers.
PyTorch  (BSD-Style License): https://pytorch.org/ - Copyright (c) Facebook.
"""

import gym
from LogAssist.log import Logger
import torch
import torch.optim as optim
from modular_rl.networks.policy import PolicyNetwork
from modular_rl.networks.value import ValueNetwork
from modular_rl.networks.actor_critic import ActorCriticNetwork


class Agent:
    def __init__(self, env, setting):
        '''
        :param env: The environment for the agent to interact with, if not provided, CartPole-v0 will be used.
        :param setting: A dictionary containing the settings and hyperparameters for the agent's training process.
        '''

        # Environment preparation
        self.env = env if env else gym.make('CartPole-v0')
        self.setting = setting
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Defination networks and optimizers
        self.policy_net = None
        self.value_net = None
        self.policy_optimizer = None
        self.value_optimizer = None

        self.actor_critic_net = None
        self.actor_critic_optimizer = None

        # Training parameters(Common)
        self.max_episodes = setting.get('max_episodes', 30)
        self.max_timesteps = setting.get('max_timesteps', 100)
        self.update_timestep = setting.get('update_timestep', 200)
        self.gamma = setting.get('gamma', 0.99)
        self.early_stop_threshold = setting.get('early_stop_threshold', -1)
        self.done_loop_end = setting.get('done_loop_end', False)

        # Set learn episode parameters
        self.episode_reward = 0
        self.total_reward = 0
        self.prev_reward = 0
        self.episode = 0
        self.avg_reward = 0

        # Set learn parameters (If necessary)
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.reset()

        # Logger initialize
        self.log_level = setting.get('log_level', 'debug')
        self.log_init_pass = setting.get('log_init_pass', False)
        if self.log_init_pass == False:
            Logger.init(
                dir_name=None,
                file_name=None,
                log_level=self.log_level,
                out_console=True,
                out_file=None,
                prev_log_remove=None
            )

    def reset(self):
        """
        Reset the lists that contain information about the states, actions, rewards, and other values for the agent.
        """

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def init_policy_value(self):
        """
        Initializes policy and value networks, and their respective optimizers.
        """

        # Create neural network instances and optimizer
        self.policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, self.setting.get('networks', 'middle'))
        self.value_net = ValueNetwork(
            self.state_dim, self.setting.get('networks', 'middle'))
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.setting.get('optimizer_speed', 3e-4))
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.setting.get('optimizer_speed', 3e-4))

    def init_actor_critic(self):
        """
        Initializes the actor-critic network and its optimizer.
        """

        # Neural Network
        self.actor_critic_net = ActorCriticNetwork(
            self.state_dim, self.action_dim)
        self.actor_critic_optimizer = optim.Adam(
            self.actor_critic_net.parameters(), lr=self.setting.get('optimizer_speed', 3e-4))

    def _check_state(self, state):
        '''
        This function takes a state parameter and returns the first element of a tuple if state has a length of 2, otherwise it simply returns the state parameter.

        :param state: The state data to be checked.
        :return: The checked state data.
        '''

        state_num = len(state)
        if state_num == 2:
            state, _ = state  # Unpack the tuple
        return state


    def learn_reset(self):
        """
        Reset the agent's state and episode reward.
        """

        self.state = self.env.reset()
        return self._check_state(self.state)

    def learn_close(self):
        """
        Close the environment and reset the agent's total reward, episode count, and episode reward.
        """

        self.env.close()
        self.total_reward = 0
        self.episode = 0
        self.episode_reward = 0

    def learn_check(self):
        """
        Print the episode count, previous reward, episode reward, total reward, and average episode reward.
        """

        avg_reward = self.total_reward / (self.episode + 1)
        Logger.debug(
            f'Episode: {self.episode}, Previous Reward: {self.prev_reward},  Episode Reward: {self.episode_reward}, Total Reward: {self.total_reward}, Average Episode Reward: {avg_reward}')

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

    def step_unpack(self, step_output):
        step_output_num = len(step_output)
        if step_output_num == 4:
            next_state, reward, is_done, _ = step_output
        elif step_output_num == 5:
            next_state, reward, is_done, _, _ = step_output
        return next_state, reward, is_done

    def update_reward(self, reward) :
        self.episode_reward += reward
        self.total_reward += reward
        self.prev_reward = reward

    def update_episode(self):
        self.episode += 1
        self.episode_reward = 0
