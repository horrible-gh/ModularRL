# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
class PolicyNetwork(nn.Module):
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

        # 각 계층을 만듦
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim * i, hidden_dim * (i + 1)))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = torch.softmax(x, dim=-1)
        return x

class ValueNetwork(nn.Module):
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

        # 각 계층을 만듦
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, 1))
            else:
                layers.append(nn.Linear(hidden_dim * i, hidden_dim * (i + 1)))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
