import tensorflow as tf
from tensorflow.keras import layers


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_option='medium', num_layers=2):
        super(PolicyNetwork, self).__init__()

        hidden_dim_map = {
            'small': 32,
            'medium': 64,
            'large': 128,
            'huge': 256,
            'exhuge': 512
        }

        hidden_dim = hidden_dim_map[hidden_option]

        # Create each layer
        self.layers_list = []
        for i in range(num_layers):
            if i == 0:
                self.layers_list.append(layers.Dense(
                    hidden_dim, activation='relu', input_shape=(input_dim,)))
            elif i == num_layers - 1:
                self.layers_list.append(layers.Dense(
                    output_dim, activation='softmax'))
            else:
                self.layers_list.append(layers.Dense(
                    hidden_dim * (i + 1), activation='relu'))

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x
