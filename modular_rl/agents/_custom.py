# -*- coding: utf-8 -*-
"""
ModularRL project

Copyright (c) 2023 horrible-gh

ModularRL is a Python library for creating and training reinforcement learning agents using various algorithms.
The library is designed to be easily customizable and modular,
allowing users to quickly set up and train agents for various environments without being limited to a specific algorithm.

"""


from LogAssist.log import Logger
from modular_rl.envs._custom import CustomEnv


class AgentCustom:
    def __init__(self, env, setting):
        """
        :param env: The environment for the agent to interact with. It should be an instance of a compatible environment class. If None is provided, a default CustomEnv environment will be used.
        :type env: object
        :param setting: The setting configuration for the agent, specifying various learning parameters and settings.
        :type setting: dict
        :return: None
        """

        self.env = env if env else CustomEnv()
        self.setting = setting

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

        # Training parameters(Common)
        self.max_episodes = setting.get('max_episodes', 30)
        self.max_timesteps = setting.get('max_timesteps', 100)

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

    def step_unpack(self, step_output):
        step_output_num = len(step_output)
        if step_output_num == 4:
            next_state, reward, is_done, _ = step_output
        elif step_output_num == 5:
            next_state, reward, is_done, _, _ = step_output
        return next_state, reward, is_done

    def update_reward(self, reward):
        self.episode_reward += reward
        self.total_reward += reward
        self.prev_reward = reward

    def update_episode(self):
        self.episode += 1
        self.episode_reward = 0
