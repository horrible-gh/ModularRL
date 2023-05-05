import gym
from LogAssist.log import Logger


class Agent:
    def __init__(self, env, setting):
        # Environment preparation
        self.env = env if env else gym.make('CartPole-v0')
        self.setting = setting
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Training parameters(Common)
        self.max_episodes = setting.get('max_episodes', 30)
        self.max_timesteps = setting.get('max_timesteps', 1000)
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

        # Logger initialize
        self.log_level = setting.get('log_level', 'debug')
        Logger.init(self.log_level, None, None, None, True, False)

    def _check_state(self, state):
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
        pass

    def update(self):
        pass
