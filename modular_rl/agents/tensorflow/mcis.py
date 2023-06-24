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
TensorFlow (Apache License 2.0): https://www.tensorflow.org/ - Copyright (c) TensorFlow Developers.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from LogAssist.log import Logger
from modular_rl.agents.tensorflow._agent import Agent
from modular_rl.util.node import Node


class TensorFlowAgentMCIS(Agent):
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
            # TensorFlow automatically uses the GPU if one is available, otherwise it uses the CPU.
            # Therefore, there is no need to manually set the device as in PyTorch.
            pass

    def update_step(self, state, action, reward, done, next_state):
        """
        Updates the provided state, action, reward, done, and next_state.

        :param state: The current state of the environment.
        :type state: numpy.ndarray
        :param action: The action taken by the agent.
        :type action: int
        :param reward: The reward for the current step.
        :type reward: float
        :param done: Flag to mark if the episode is done or not.
        :type done: bool
        :param next_state: The next state after the current action.
        :type next_state: numpy.ndarray
        """

        self.update_reward(reward)

        self.states.append(tf.squeeze(state))  # change here
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(tf.squeeze(next_state))  # change here

        if done:
            self.update()

    def select_action(self, state):
        state_tensor = tf.convert_to_tensor(
            self.check_tensor(state), dtype=tf.float32)
        action_probs, _ = self.actor_critic_net(state_tensor)
        action_probs = action_probs.numpy().flatten()
        root = Node(state, action_probs)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            while node.expanded():
                best_score = -np.inf
                best_action = -1
                for action, child in node.children.items():
                    uct_score = child.get_score(self.cpuct)
                    if np.any(uct_score > best_score):
                        best_score = uct_score
                        best_action = action

                if best_action != -1:
                    action = best_action
                    node = node.children[action]
                    search_path.append(node)
                else:
                    break

            if len(search_path) > 1:
                parent, action = search_path[-2], search_path[-1].action
            else:
                parent, action = search_path[0], None
            if True not in self.dones:  # Check if the game is not over
                state_tensor = tf.convert_to_tensor(
                    self.check_tensor(node.state), dtype=tf.float32)
                action_probs, value = self.actor_critic_net(state_tensor)
                action_space = self.env.action_space.n
                node.expand(action_space, action_probs, False)
            state_tensor = tf.convert_to_tensor(
                self.check_tensor(node.state), dtype=tf.float32)
            _, value = self.actor_critic_net(state_tensor)
            self.backpropagate(search_path, value.numpy(), node.done)
        root_state_tensor = tf.convert_to_tensor(
            self.check_tensor(root.state), dtype=tf.float32)
        action_probs, _ = self.actor_critic_net(root_state_tensor)
        action_probs = action_probs.numpy().flatten()  # make sure it is a flat array
        Logger.verb('mcis:select_action:action_probs', f'{action_probs}')
        chosen_action = np.random.choice(
            range(len(action_probs)), p=action_probs)  # choosing from the range of all possible actions
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
                    state_tensor = tf.convert_to_tensor(
                        node.state, dtype=tf.float32)
                elif tf.is_tensor(node.state):
                    state_tensor = node.state
                else:
                    raise ValueError(
                        "node.state must be a numpy array or tensorflow tensor")
                _, reward = self.actor_critic_net(state_tensor)
                reward = reward.numpy()

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

        # Compute the value loss
        actor_output, critic_output = self.actor_critic_net(next_state)
        target_values = reward + self.gamma * \
            tf.reduce_mean(critic_output) * (1 - done)
        target_values = tf.expand_dims(target_values, 1)
        critic_loss = tf.keras.losses.MSE(
            values, tf.stop_gradient(target_values))

        # Compute the policy loss
        m = tfp.distributions.Categorical(probs=action_probs)
        logprobs = m.log_prob(self.check_tensor(action))
        actor_loss = -logprobs * tf.stop_gradient(target_values - values)

        # Take mean of actor_loss and critic_loss
        actor_loss = tf.reduce_mean(actor_loss)
        critic_loss = tf.reduce_mean(critic_loss)

        return actor_loss, critic_loss

    def update(self):
        """
        This function updates the network parameters using the optimizer and computed loss values.
        """

        if not self.states:  # Check if the states list is empty
            return

        # Prepare data
        states_tensor = tf.stack(
            [self.check_tensor(state) for state in self.states])
        actions_tensor = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        next_states_tensor = tf.stack(
            [self.check_tensor(state) for state in self.next_states])
        dones_tensor = tf.convert_to_tensor(self.dones, dtype=tf.float32)

        # Compute gradients and update network parameters
        with tf.GradientTape() as tape:
            # Compute loss
            actor_loss, critic_loss = self.compute_loss(
                states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
            loss = actor_loss + critic_loss

        # Calculate gradients
        gradients = tape.gradient(
            loss, self.actor_critic_net.trainable_variables)

        # Apply gradients
        self.actor_critic_optimizer.apply_gradients(
            zip(gradients, self.actor_critic_net.trainable_variables))

        # Reset the lists for the next episode
        self.reset()
        self.update_episode()

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
                Logger.verb('mcis:train', f'{state.shape}')
                self.state_tensor = self.check_tensor(state).numpy()

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

    def check_tensor(self, state):
        '''
        This function checks if the provided object is a TensorFlow tensor, and if not, converts it to a tensor.

        :param state: The object to check/convert to a TensorFlow tensor.
        :return: The input object as a TensorFlow tensor.
        '''
        Logger.verb('_agent:check_tensor', f'state={state}')
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
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
