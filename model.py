# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

import numpy as np
import torch
from torch import nn

class ManeuvorNetwork(nn.Module):
    def __init__(self, action_space, maneuveur_capacity):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = action_space,
            hidden_size = maneuveur_capacity,
            num_layers = 2
        )

        self.linear = nn.Sequential(
            nn.Linear(maneuveur_capacity, 100),
            nn.ReLU(),
            nn.Linear(100, maneuveur_capacity), # Output category
            nn.Softmax()
        )

        self.states = None

    def forward(self, x):
        # Convert batch to single a single time sequence
        x = x[:, None, :] # [Time, Batch, *]
        lstm_out, self.states = self.lstm(x, self.states)
        lstm_out_flat = lstm_out.view(-1, lstm_out.size(2))
        maneuver = self.linear(lstm_out_flat)

        return maneuver

    def reset_state(self):
        ''' Resets the LSTM state, used when a new episode starts'''
        self.states = None


class SymbolNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self._build_cnn()

    def _build_cnn(self):
        ''' builds the principle CNN for vision'''
        filter_cfg = [[64, 64, ], [128, 128]]
        kernel = (3, 3)

        layers = []
        prev_filter_size = 3  # Initial image has 3 channels
        for module in filter_cfg:
            for f_size in module:
                layers.append(nn.Conv2d(prev_filter_size, f_size, kernel))  # Conv2d
                layers.append(nn.ReLU())
                prev_filter_size = f_size

            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # Remove last pooling layer and ReLU, replace with Softmax
        layers = layers[:-2]

        # NOTE: Maybe not the best to add ReLU
        layers.append(nn.ReLU())

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class SimulatorNet(nn.Module):
    def __init__(self, maneuveur_capacity, symbol_space, symbol_capacity):
        super().__init__()
        input_size = maneuveur_capacity + symbol_space * symbol_space * symbol_capacity # C_m + x * y * C_o
        symbol_space_full = symbol_space * symbol_space * symbol_capacity
        self.network = nn.Sequential(
            nn.Linear(input_size,1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.fc_reward = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Tanh(),
        )

        self.fc_omap = nn.Sequential(
            nn.Linear(1024, symbol_space_full),
            nn.Softmax(),
        )


    def forward(self,  o, m):
        ''' o = output of symbol net, m = output of maneuver network'''

        o_flat = o.view(o.size(0), -1)
        x = torch.cat((o_flat, m), 1)
        out = self.network(x)

        # predicted symbol space
        o_out_flat = self.fc_omap(out)
        o_out = o_out_flat.view_as(o)

        # reward output
        reward_out = self.fc_reward(out)

        return o_out, reward_out




