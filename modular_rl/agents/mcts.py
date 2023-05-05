
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

import gym
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from modular_rl.networks.actor_critic import ActorCriticNetwork
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

        # Neural Network
        self.network = ActorCriticNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=setting.get('optimizer_speed', 3e-4))

        # MCTS parameters
        self.num_simulations = setting.get('num_simulations', 800)
        self.cpuct = setting.get('cpuct', 1.0)
        self.temperature = setting.get('temperature', 1.0)

        self.device = setting.get('device', None)
        if self.device == None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        # Save selected learning data separately
        # self.state_tensor

    def select_action(self, state):
        """
        Select an action using MCTS.

        :param state: The current state.
        :type state: numpy.ndarray
        :return: The selected action.
        :rtype: int
        """

        state_tensor = self.check_tensor(self._check_state(state))
        action_probs, _ = self.network(state_tensor)
        action_probs = action_probs.detach().numpy().flatten()
        root = Node(state, action_probs)
        Logger.verb(
            'agents:mcts:select_action:self.num_simulations', self.num_simulations)
        Logger.verb(
            'agents:mcts:select_action:root', root)
        Logger.verb(
            'agents:mcts:select_action:action_probs', action_probs)
        Logger.verb('agents:mcts:select_action:node', f'Root node: {root}')
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            # Logger.verb(
            #    'agents:mcts:select_action:search_path', search_path)
            # Selection
            while node.expanded():
                action, node = node.select_child(self.cpuct)
                search_path.append(node)
            Logger.verb('agents:mcts:select_action:After selection',
                        f' {search_path[-1]}')

            # Expansion
            if len(search_path) > 1:
                parent, action = search_path[-2], search_path[-1].action
            else:
                # or some other suitable defaults
                parent, action = search_path[0], None

            # Logger.verb('mcts:select_action:action', action)
            step_output = self.env.step(action) if action is not None else (
                parent.state, 0, False, None)
            step_output_num = len(step_output)

            if step_output_num == 4:
                state, reward, done, _ = step_output
            elif step_output_num == 5:
                state, reward, done, _, _ = step_output

            if not done:
                # Logger.verb(
                #    'agents:mcts:select_action:state', state)
                if not torch.is_tensor(state):
                    state_tensor = torch.from_numpy(
                        state).float().to(self.device)
                else:
                    state_tensor = state.to(self.device)

                action_probs, value = self.network(state_tensor)
                action_space = self.env.action_space.n
                node.expand(action_space, action_probs)

            # Backpropagation
            self.backpropagate(search_path, reward, done)

        chosen_action = root.select_action(self.temperature)
        chosen_action_reward = root.children[chosen_action].total_value
        chosen_action_state = root.children[chosen_action].state
        # Assuming that a non-zero reward indicates a terminal state
        done = (chosen_action_reward != 0)

        # Save selected learning data separately
        self.state = chosen_action_state
        self.action = chosen_action
        self.reward = chosen_action_reward
        self.done = done

        self.total_reward += reward
        self.prev_reward = reward

        return chosen_action_state, chosen_action, chosen_action_reward, done

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
            node.update_stats(reward)
            if not done:
                _, reward = self.network(node.state)
                reward = reward.item()

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

    def compute_loss(self, state, action, reward, next_state, done):
        # Predict action probabilities and values
        action_probs, values = self.network(state)

        # Compute the value loss
        target_values = reward + self.gamma * \
            self.network(next_state)[1] * (1 - done)
        critic_loss = F.mse_loss(values, target_values.detach())

        # Compute the policy loss
        m = Categorical(action_probs)
        logprobs = m.log_prob(self.check_tensor(action))
        actor_loss = -logprobs * (target_values - values).detach()

        return actor_loss, critic_loss

    def update(self):
        # Update the network
        self.optimizer.zero_grad()
        actor_loss, critic_loss = self.compute_loss(
            self.state_tensor, self.action, self.reward, self.next_state, self.done)
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

    def check_tensor(self, obj):
        if not torch.is_tensor(obj):
            obj_tensor = torch.FloatTensor(obj)
        else:
            obj_tensor = obj
        return obj_tensor
