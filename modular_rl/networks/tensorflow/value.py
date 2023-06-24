import tensorflow as tf
from tensorflow.keras import layers


class ValueNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_option='medium', num_layers=2):
        super(ValueNetwork, self).__init__()

        hidden_dim_map = {
            'small': 32,
            'medium': 64,
            'large': 128,
            'huge': 256,
            'exhuge': 512
        }

        hidden_dim = hidden_dim_map[hidden_option]

        self.layers_list = []
        for i in range(num_layers):
            if i == 0:
                self.layers_list.append(layers.Dense(
                    hidden_dim, activation='relu'))
            elif i == num_layers - 1:
                self.layers_list.append(layers.Dense(1))
            else:
                self.layers_list.append(layers.Dense(
                    hidden_dim * (i + 1), activation='relu'))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x
