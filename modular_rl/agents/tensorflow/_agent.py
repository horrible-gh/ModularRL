import tensorflow as tf
from tensorflow.keras.models import load_model
from LogAssist.log import Logger
from modular_rl.networks.tensorflow.policy import PolicyNetwork
from modular_rl.networks.tensorflow.value import ValueNetwork
from modular_rl.networks.tensorflow.actor_critic import ActorCriticNetwork
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
        self.policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, networks_size)
        self.value_net = ValueNetwork(
            self.state_dim, networks_size)
        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.setting.get('optimizer_speed', 3e-4))
        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.setting.get('optimizer_speed', 3e-4))

    def init_actor_critic(self):
        """
        Initializes the actor-critic network and its optimizer.
        """

        # Neural Network
        self.actor_critic_net = ActorCriticNetwork(
            self.state_dim, self.action_dim)
        self.actor_critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.setting.get('optimizer_speed', 3e-4))

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
        Save the policy and value networks.

        :param file_name: The base name of the files to save the networks.
        :type file_name: str
        """

        self.policy_net.save(file_name + '_policy.h5')
        self.value_net.save(file_name + '_value.h5')

    def load_policy_value(self, file_name):
        """
        Load the policy and value networks.

        :param file_name: The base name of the files to load the networks.
        :type file_name: str
        """

        self.policy_net = load_model(file_name + '_policy.h5')
        self.value_net = load_model(file_name + '_value.h5')

    def save_actor_critic(self, file_name):
        """
        Save the actor-critic network.

        :param file_name: The name of the file to save the network.
        :type file_name: str
        """

        self.actor_critic_net.save(file_name + '_actor_critic.h5')

    def load_actor_critic(self, file_name):
        """
        Load the actor-critic network.

        :param file_name: The name of the file to load the network.
        :type file_name: str
        """

        self.actor_critic_net = load_model(file_name + '_actor_critic.h5')
