
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
TensorFlow (Apache License 2.0): https://www.tensorflow.org/ - Copyright (c) TensorFlow Developers.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from LogAssist.log import Logger
from modular_rl.agents.tensorflow._agent import Agent


class TensorFlowAgentPPO(Agent):
    def __init__(self, env, setting):
        super(TensorFlowAgentPPO, self).__init__(env, setting)
        super(TensorFlowAgentPPO, self).init_policy_value()

        # Set learning parameters
        self.ppo_epochs = setting.get('ppo_epochs', 4)
        self.mini_batch_size = setting.get('mini_batch_size', 64)
        self.lam = setting.get('lam', 0.95)
        self.clip_param = setting.get('clip_param', 0.2)

        # Set learn modular parameters
        self.state = None
        self.dist = None

        # Create optimizers
        self.value_optimizer = tf.keras.optimizers.Adam()

    def learn_reset(self):
        return super(TensorFlowAgentPPO, self).learn_reset()

    def reset(self):
        super(TensorFlowAgentPPO, self).reset()

    def check_tensor(self, state):
        '''
        This function checks if the provided object is a TensorFlow tensor, and if not, converts it to a tensor.

        :param state: The object to check/convert to a TensorFlow tensor.
        :return: The input object as a TensorFlow tensor.
        '''
        Logger.verb('_agent:check_tensor', f'state={state}')
        if not isinstance(state[0], tf.Tensor):
            state = (tf.convert_to_tensor(
                state[0], dtype=tf.float32), state[1])
        return state

    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * \
                values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * \
                last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        return advantages

    def select_action(self, state):
        state = self._check_state(self.state)
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = self.policy_net(state_tensor)
        dist = tfp.distributions.Categorical(probs=action_probs)
        self.dist = dist
        action = dist.sample()
        return action, dist

    def learn_step(self, state, timestep):
        action, dist = self.select_action(state)
        return self.update_step(state, dist, action, timestep)

    def update_step(self, state, dist, action, timestep, auto_step=True, is_done=False, reward=0, next_state=None):
        if dist is None and self.dist:
            dist = self.dist
        if auto_step:
            step_output = self.env.step(int(action.numpy()[0]))
            next_state, reward, is_done = self.step_unpack(step_output)
        else:
            if next_state is None:
                next_state = state
        self.update_reward(reward)
        state = self._check_state(state)
        # Note: we use Python lists here, as TensorFlow has a better support for Pythonic operations
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(is_done)
        self.log_probs.append(dist.log_prob(action))
        self.state = next_state
        if timestep > 0:
            timestep += 1
        if self.update_timestep > 0 and timestep > 0 and (timestep % self.update_timestep == 0):
            self.update()
        return action, reward, is_done, timestep

    def update(self):
        # convert lists to tensors
        states_tensor = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(
            self.next_states, dtype=tf.float32)
        done_tensor = tf.convert_to_tensor(self.dones, dtype=tf.float32)
        log_probs_tensor = tf.stack(self.log_probs)

        values = self.value_net(states_tensor).numpy().squeeze(1)
        next_values = self.value_net(next_states_tensor).numpy().squeeze(1)

        if len(next_states_tensor) > 0:
            last_value = next_values[-1]
        else:
            last_value = 0

        advantages = self.compute_advantages(rewards_tensor.numpy(), np.append(
            values, last_value), done_tensor.numpy(), self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-5)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)

        if len(advantages) > 1:
            returns = np.add(advantages[:-1], values[:-1])
        else:
            returns = np.add(advantages, values)

        self.ppo_update(self.ppo_epochs, self.mini_batch_size, states_tensor,
                        actions_tensor, log_probs_tensor, returns, advantages_tensor)

        self.reset()
        self.update_episode()

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        batch_size = states.shape[0]
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        for _ in range(ppo_epochs):
            for idx in range(batch_size // mini_batch_size):
                # Extract mini-batch
                minibatch_indices = tf.range(
                    idx * mini_batch_size, (idx + 1) * mini_batch_size)
                states_minibatch = tf.gather(states, minibatch_indices)
                actions_minibatch = tf.gather(actions, minibatch_indices)
                old_log_probs_minibatch = tf.gather(
                    log_probs, minibatch_indices)
                returns_minibatch = tf.gather(returns, minibatch_indices)
                advantages_minibatch = tf.gather(advantages, minibatch_indices)

                # Calculate policy loss
                with tf.GradientTape() as tape:
                    new_probs = self.policy_net(states_minibatch)
                    new_dist = tfp.distributions.Categorical(probs=new_probs)
                    new_log_probs = new_dist.log_prob(actions_minibatch)
                    ratio = tf.exp(new_log_probs - old_log_probs_minibatch)
                    surr1 = ratio * advantages_minibatch
                    surr2 = tf.clip_by_value(
                        ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_minibatch
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                # Calculate gradients
                policy_gradients = tape.gradient(
                    policy_loss, self.policy_net.trainable_variables)
                # Apply gradients
                self.policy_optimizer.apply_gradients(
                    zip(policy_gradients, self.policy_net.trainable_variables))

                # Calculate value loss
                with tf.GradientTape() as tape:
                    current_value_estimate = self.value_net(states_minibatch)
                    value_loss = tf.reduce_mean(
                        (returns_minibatch - current_value_estimate) ** 2)

                # Calculate gradients
                value_gradients = tape.gradient(
                    value_loss, self.value_net.trainable_variables)
                # Apply gradients
                self.value_optimizer.apply_gradients(
                    zip(value_gradients, self.value_net.trainable_variables))

    def train(self):
        """
        Execute the learning loop, where the PPO algorithm is used to train the agent on the specified environment.
        """
        self.learn()

    def learn(self):
        """
        Execute the learning loop, where the PPO algorithm is used to train the agent on the specified environment.
        """

        timestep = 0
        test = 0
        self.total_reward = 0

        if self.max_episodes > 0 and self.max_timesteps > 0:
            is_done = False
            for episode in range(self.max_episodes):
                self.episode = episode
                self.reset()
                self.learn_reset()
                self.episode_reward = 0
                reward = 0

                for t in range(self.max_timesteps):
                    state_tensor = self.check_tensor(self.state)
                    action, reward, is_done, timestep = self.learn_step(
                        state_tensor, timestep)
                    if is_done:
                        break

                self.learn_check()
                avg_reward = self.total_reward / (self.episode + 1)

                if avg_reward >= self.early_stop_threshold > 0:
                    Logger.info(
                        f'Early stopping: Average reward has reached the threshold ({self.early_stop_threshold}) at episode {self.episode}')
                    break
                if is_done and self.done_loop_end:
                    break

                if self.episode + 1 >= self.max_episodes:
                    break

            self.env.close()
        else:
            self.reset()

    def learn_next(self):
        """
        Perform a single learning step and update the episode and total rewards.

        :return: The action taken, the resulting reward, and whether the episode is done or not.
        :rtype: tuple(torch.Tensor, float, bool)
        """

        action, reward, is_done, timestep = self.learn_step(self.state, -1)

        return action, reward, is_done, timestep

    def learn_close(self):
        """
        Close the environment and reset the agent's total reward, episode count, and episode reward.
        """

        super().learn_close()

    def learn_check(self):
        """
        Print the episode count, previous reward, episode reward, total reward, and average episode reward.
        """

        super().learn_check()

    def save_model(self, file_name):
        """
        This function saves the model to the specified file.

        :param file_name: The name of the file to save the model to.
        :return: None
        """
        self.save(file_name)

    def save(self, file_name):
        """
        This function saves the policy and value networks to the specified file.

        :param file_name: The name of the file to save the policy and value networks to.
        :return: None
        """

        self.save_policy_value(file_name)

    def load_model(self, file_name):
        """
        This function loads the model from the specified file.

        :param file_name: The name of the file to load the model from.
        :return: None
        """
        self.load(file_name)

    def load(self, file_name):
        """
        This function loads the policy and value networks from the specified file.

        :param file_name: The name of the file to load the policy and value networks from.
        :return: None
        """

        self.load_policy_value(file_name)
