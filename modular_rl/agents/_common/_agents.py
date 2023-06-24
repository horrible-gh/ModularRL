import gym
from LogAssist.log import Logger


class CommonAgents:
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

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

        if done:
            self.update()

    def step_unpack(self, step_output):
        """
        Unpack the step output tuple and extract the next state, reward, and done flag.

        :param step_output: The step output tuple containing next state, reward, and done flag.
        :type step_output: tuple
        :return: The unpacked next state, reward, and done flag.
        :rtype: tuple
        """

        step_output_num = len(step_output)
        if step_output_num == 4:
            next_state, reward, is_done, _ = step_output
        elif step_output_num == 5:
            next_state, reward, is_done, _, _ = step_output
        return next_state, reward, is_done

    def update_reward(self, reward):
        """
        Update the reward-related variables with the given reward value.

        :param reward: The reward value to update the variables with.
        :type reward: float
        """

        self.episode_reward += reward
        self.total_reward += reward
        self.prev_reward = reward

    def update_episode(self):
        """
        Update the episode-related variables to indicate the start of a new episode.
        """

        self.episode += 1
        self.episode_reward = 0

    def convert_float_to_double(self, input_tensor):
        """
        Convert the input tensor from float32 to double precision if necessary.

        :param input_tensor: The input tensor to be converted.
        :type input_tensor: torch.Tensor
        :return: The converted tensor.
        :rtype: torch.Tensor
        """

        if input_tensor.dtype == torch.float32:
            return input_tensor.double()
        else:
            return input_tensor

    def ensure_float(self, input_tensor):
        """
        Ensure that the input tensor is of float precision by converting it from float64 if necessary.

        :param input_tensor: The input tensor to be converted.
        :type input_tensor: torch.Tensor
        :return: The converted tensor.
        :rtype: torch.Tensor
        """

        if input_tensor.dtype == torch.float64:
            return input_tensor.float()
        else:
            return input_tensor

    def update(self):
        '''
        This function is a placeholder and must be implemented by the child class that extends this Agent class.

        update() function is a placeholder that needs to be implemented in the child class that extends the Agent class. This function is responsible for updating the agent's state, action, and policy based on the new state and reward received from the environment.

        No parameters are passed into this function and it does not return anything.
        '''
        pass
