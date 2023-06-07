
# -*- coding: utf-8 -*-
"""
ModularRL project

Copyright (c) 2023 horrible-gh

Class AgentMCTS is an implementation of the Monte Carlo Tree Search (MCTS) algorithm.
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

import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
from modular_rl.util.node import Node
from modular_rl.agents._agent import Agent
from LogAssist.log import Logger


class AgentMCTS(Agent):
    def __init__(self, env, setting):
        """
        Initialize the AgentMCTS class with the specified environment and settings.

        :param env: The environment to use for training.
        :type env: gym.Env or None
        :param setting: The settings for the MCTS algorithm.
        :type setting: AgentSettings
        """

        super().__init__(env, setting)
        super().init_actor_critic()

        # MCTS parameters
        self.num_simulations = setting.get('num_simulations', 800)
        self.cpuct = setting.get('cpuct', 1.0)
        self.temperature = setting.get('temperature', 1.0)
        self.gamma = 0.95
        self.total_value = 0

        self.device = setting.get('device', None)
        if self.device == None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        self.state = None
        self.action = None
        self.reward = None
        self.done = None
        self.total_reward = 0

    def select_action(self, state):
        """
        Select an action using MCTS.

        :param state: The current state.
        :type state: numpy.ndarray
        :return: The selected action.
        :rtype: int
        """

        state_tensor = self.check_tensor(self._check_state(state))
        action_probs, _ = self.actor_critic_net(state_tensor)
        action_probs = action_probs.detach()
        m = Categorical(action_probs)
        chosen_action = m.sample().item()

        # Use uniform prior for root node.
        root = Node(state, [1 / self.env.action_space.n]
                    * self.env.action_space.n)

        for _ in range(self.num_simulations):
            node, search_path = root, [root]

            while node.expanded():
                action, node = node.select_child(self.cpuct)
                if action is None:
                    action = chosen_action
                search_path.append(node)

            parent, action = (search_path[-2], search_path[-1].action) if len(
                search_path) > 1 else (search_path[0], None)
            step_output = self.env.step(action) if action is not None else (
                parent.state, 0, False, None)
            state, reward, done, *_ = step_output
            Logger.verb('mcts:select_action', f"Step Output Reward: {reward}")

            if not done:
                state_tensor = self.check_tensor(state).to(self.device)
                action_probs, _ = self.actor_critic_net(state_tensor)
                node.expand(self.env.action_space.n,
                            action_probs.detach().cpu().numpy().flatten(), False)

            self.backpropagate(search_path, reward, done)

        chosen_action_node = root.children[chosen_action]

        self.state = chosen_action_node.state
        self.action = chosen_action
        self.reward = chosen_action_node.total_value
        self.done = self.reward != 0

        self.total_reward += self.reward

        return self.state, self.action, self.reward, self.done

    def backpropagate(self, search_path, reward, done):
        """
        Backpropagate the value estimates back to the root node.

        :param search_path: The nodes visited during the search.
        :type search_path: list of Node
        :param reward: The reward obtained after the search.
        :type reward: float
        :param done: Whether the episode has ended.
        :type done: bool
        """
        for node in reversed(search_path):
            # Logger.verb('mcts:backpropagate',
            #            f"Node Reward({reward}) after Update Stats: {node.total_value}")
            node.update_stats(reward)

    def learn(self):
        """
        Train the agent.
        """
        self.train()

    def train(self):
        """
        Train the agent.
        """
        self.total_reward = 0
        for episode in range(self.max_episodes):
            self.episode = episode
            state = self.learn_reset()
            # state = self.env.reset()

            for t in range(self.max_timesteps):
                state = self._check_state(state)
                self.state_tensor = self.check_tensor(state).squeeze(0)
                self.next_state, self.action, self.reward, self.done = self.select_action(
                    self.state_tensor)

                self.learn_check()

                if self.done:
                    # self.update()
                    break

                self.update()

                state = self.next_state

            self.episode += 1
            # print(f"Episode: {self.episode}, Reward: {self.total_reward}")

    def compute_loss(self, state, action, reward, done):
        '''
        This function computes the actor and critic loss using the provided state, action, reward, next_state, and done variables.
        The actor loss is computed based on the policy gradient algorithm,
        and the critic loss is computed as the mean squared error between the estimated value of the current state and the target value of the next state.

        compute_loss() function computes the actor and critic loss values for the provided state, action, reward, next_state, and done variables.

        The state parameter is the current state of the environment.
        The action parameter is the action taken in the current state.
        The reward parameter is the reward received for taking the action in the current state.
        The next_state parameter is the state resulting from taking the action in the current state.
        The done parameter is a flag indicating whether the episode has ended.

        :param state: The current state of the environment.
        :param action: The action taken in the current state.
        :param reward: The reward received for taking the action in the current state.
        :param next_state: The state resulting from taking the action in the current state.
        :param done: A flag indicating whether the episode has ended.
        :return: The computed actor and critic loss values.
        '''
        # Predict action probabilities and values
        action_probs, values = self.actor_critic_net(state)

        # Compute the value loss
        # Convert reward to tensor
        reward = torch.tensor(reward, device=self.device)
        # Convert total_value to tensor
        total_value = torch.tensor(self.total_value, device=self.device)
        target_values = reward + self.gamma * total_value * (1 - done)
        target_values = target_values.unsqueeze(0)
        critic_loss = F.mse_loss(values, target_values.detach())

        # If action is not a list or tuple or its length is zero, initialize it with zeros
        if not isinstance(action, (list, tuple)) or len(action) == 0:
            action = torch.zeros_like(action_probs)

        # Compute the policy loss
        m = Categorical(action_probs)
        logprobs = m.log_prob(self.check_tensor(action))
        actor_loss = -logprobs * (target_values - values).detach()

        # Average the actor and critic loss
        loss = (actor_loss.mean() + critic_loss.mean()) / 2

        return loss

    def update(self):
        '''
        This function updates the network parameters using the optimizer and computed loss values.

        update() function updates the network parameters using the optimizer and computed loss values.
        It uses the compute_loss() function to compute the loss and the optimizer object to perform the optimization step.

        This function does not take any parameters and does not return anything.

        :return: None
        '''

        # Update the network
        # Convert state to tensor
        state_tensor = torch.tensor(self.state, device=self.device)

        # Update the network
        self.actor_critic_optimizer.zero_grad()
        loss = self.compute_loss(
            state_tensor, self.action, self.reward, self.done)
        Logger.verb('mcts:update',
                    f"Loss: {loss.item()}, Reward: {self.reward}")
        loss.backward()
        self.actor_critic_optimizer.step()

        self.reset()

    def check_tensor(self, obj):
        '''
        This function checks if the provided object is a PyTorch tensor, and if not, converts it to a tensor.

        check_tensor() function checks if the provided obj parameter is a PyTorch tensor.
        If it is not a tensor, it converts it to a tensor using torch.FloatTensor().
        If it is already a tensor, it simply returns the tensor.

        The obj parameter is the object to check/convert to a PyTorch tensor.

        The function returns the input object as a PyTorch tensor.

        :param obj: The object to check/convert to a PyTorch tensor.
        :return: The input object as a PyTorch tensor.
        '''

        if not torch.is_tensor(obj):
            obj_tensor = torch.FloatTensor(obj)
        else:
            obj_tensor = obj

        Logger.verb('mcts:check_tensor', f"Before remapping: {obj_tensor}")
        obj_tensor = obj_tensor % self.env.action_space.n
        obj_tensor = torch.where(
            obj_tensor >= self.env.action_space.n, self.env.action_space.n - 1, obj_tensor)
        obj_tensor = torch.round(obj_tensor)
        Logger.verb('mcts:check_tensor', f"After remapping: {obj_tensor}")
        return obj_tensor

    def save_model(self, file_name):
        """
        This function saves the model to the specified file.

        :param file_name: The name of the file to save the model to.
        :return: None
        """
        self.save(file_name)

    def save(self, file_name):
        """
        This function saves the actor critic network to the specified file.

        :param file_name: The name of the file to save the actor critic network to.
        :return: None
        """

        self.save_actor_critic(file_name)

    def load_model(self, file_name):
        """
        This function loads the model from the specified file.

        :param file_name: The name of the file to load the model from.
        :return: None
        """
        self.load(file_name)

    def load(self, file_name):
        """
        This function loads the actor critic network from the specified file.

        :param file_name: The name of the file to load the actor critic network from.
        :return: None
        """

        self.load_actor_critic(file_name)

    def update_step(self, state, action, reward, done, next_state):
        """
        In MCTS, update_step is not necessary as the agent is updated
        after every simulation (or a batch of simulations), each of which
        represents an entire episode. Thus, this method can be left blank or removed.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state
        self.update_reward(reward)
        self.update_episode()
