import tensorflow as tf
from tensorflow.keras import layers


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()

        # Actor network
        self.actor = tf.keras.Sequential([
            layers.Dense(hidden_size, activation='relu',
                         input_shape=(state_dim,)),
            layers.Dense(action_dim, activation='softmax')
        ])

        # Critic network
        self.critic = tf.keras.Sequential([
            layers.Dense(hidden_size, activation='relu',
                         input_shape=(state_dim,)),
            layers.Dense(1)
        ])

    def call(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
