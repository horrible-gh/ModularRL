import tensorflow as tf
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
