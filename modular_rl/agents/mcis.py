
"""
ModularRL project

Copyright (c) 2023 horrible-gh

Class AgentMCIS is an implementation of the Monte Carlo Importance Sampling (MCIS) algorithm.
The algorithm is used for solving problems of sequential decision making under uncertainty.
It takes an environment and a setting configuration as inputs, initializes neural network instances and optimizers,
and sets various learning parameters.
This class has methods to predict an action given a state, perform a learning step, update the neural network parameters,
save and load a checkpoint, and reset learning parameters.
The class also has instance variables to keep track of episode and total rewards, previous reward, and average reward.

Importance Sampling is used to estimate the properties of a particular target distribution, given some observed data and a proposal distribution.
This implementation makes use of Monte Carlo methods, which rely on repeated random sampling to obtain numerical results.

This software includes the following third-party libraries:
Gym (MIT License): https://github.com/openai/gym - Copyright (c) OpenAI.
NumPy (BSD License): https://numpy.org - Copyright (c) NumPy Developers.
PyTorch (BSD-Style License): https://pytorch.org/ - Copyright (c) Facebook.
"""

import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from modular_rl.util.node import Node
from modular_rl.agents._agent import Agent
import numpy as np


class AgentMCIS(Agent):
    def __init__(self, env, setting):
        """
        Initialize the AgentMCIS class with the specified environment and settings.

        :param env: The environment to use for training.
        :type env: gym.Env or None
        :param setting: The settings for the MCIS algorithm.
        :type setting: AgentSettings
        """

        super().__init__(env, setting)
        super().init_actor_critic()

        # mcis parameters
        self.num_simulations = setting.get('num_simulations', 800)
        self.cpuct = setting.get('cpuct', 1.0)
        self.temperature = setting.get('temperature', 1.0)

        self.device = setting.get('device', None)
        if self.device == None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        """
        Select an action using mcis.

        :param state: The current state.
        :type state: numpy.ndarray
        :return: The selected action.
        :rtype: int
        """

        state_tensor = self.check_tensor(state).to(self.device)
        action_probs, _ = self.actor_critic_net(state_tensor)
        action_probs = action_probs.detach().numpy().flatten()
        root = Node(state, action_probs)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            while node.expanded():
                action, node = node.select_child(self.cpuct)
                search_path.append(node)
            if len(search_path) > 1:
                parent, action = search_path[-2], search_path[-1].action
            else:
                parent, action = search_path[0], None
            if True not in self.dones:  # Check if the game is not over
                state_tensor = self.check_tensor(node.state).to(self.device)
                action_probs, value = self.actor_critic_net(state_tensor)
                action_space = self.env.action_space.n
                node.expand(action_space, action_probs, False)
            state_tensor = self.check_tensor(node.state).to(self.device)
            _, value = self.actor_critic_net(state_tensor)
            self.backpropagate(search_path, value.item(), node.done)
        root_state_tensor = self.check_tensor(
            root.state).to(self.device)  # This line is added
        action_probs, _ = self.actor_critic_net(root_state_tensor)
        chosen_action = np.random.choice(
            range(len(action_probs)), p=action_probs.detach().numpy())
        return chosen_action

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
                if isinstance(node.state, np.ndarray):
                    state_tensor = torch.from_numpy(
                        node.state).float().to(self.device)
                elif torch.is_tensor(node.state):
                    state_tensor = node.state.float().to(self.device)
                else:
                    raise ValueError(
                        "node.state must be a numpy array or torch tensor")
                _, reward = self.actor_critic_net(state_tensor)
                reward = reward.item()

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

            for t in range(self.max_timesteps):
                state = self._check_state(state)
                self.state_tensor = self.check_tensor(state).squeeze(0)

                # Select an action
                action = self.select_action(self.state_tensor)

                # Take a step in the environment
                step_out = self.env.step(action)
                next_state, reward, done = self.step_unpack(step_out)

                # Update the agent's experience
                self.update_step(state, action, reward, done, next_state)

                # Update the network parameters at the end of the episode
                if done:
                    self.update()
                    self.learn_check()
                    break

                state = next_state

            self.episode += 1

    def compute_loss(self, state, action, reward, next_state, done):
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

        # Logger.verb('agents:mcis:compute_loss',f'{self.actor_critic_net(next_state)}')

        # Compute the value loss
        actor_output, critic_output = self.actor_critic_net(next_state)
        target_values = reward + self.gamma * \
            torch.mean(critic_output) * (1 - done)
        target_values = target_values.unsqueeze(1)
        critic_loss = F.mse_loss(values, target_values.detach())

        # Compute the policy loss
        m = Categorical(action_probs)
        logprobs = m.log_prob(self.check_tensor(action))
        actor_loss = -logprobs * (target_values - values).detach()

        # Take mean of actor_loss and critic_loss
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()

        return actor_loss, critic_loss

    def update(self):
        """
        This function updates the network parameters using the optimizer and computed loss values.
        """

        if not self.states:  # Check if the states list is empty
            return

        # Logger.verb('agents:mcis:update',f'states={self.states},actions={self.actions},rewards={self.rewards},dones={self.dones}')

        # Prepare data
        states_tensor = torch.stack(
            [self.check_tensor(state) for state in self.states]).to(self.device)
        actions_tensor = torch.Tensor(self.actions).to(self.device)
        rewards_tensor = torch.Tensor(self.rewards).to(self.device)
        next_states_tensor = torch.stack(
            [self.check_tensor(state) for state in self.next_states]).to(self.device)
        dones_tensor = torch.Tensor(self.dones).to(self.device)

        # Compute loss
        actor_loss, critic_loss = self.compute_loss(
            states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)

        # Compute gradients and update network parameters
        self.actor_critic_optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.actor_critic_optimizer.step()

        # Reset the lists for the next episode
        self.reset()
        self.update_episode()

    def check_tensor(self, state):
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

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        return state

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
