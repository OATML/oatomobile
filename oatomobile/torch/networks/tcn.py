"""
Source code obtained from: https://github.com/locuslab/TCN
Citation:
    @article{BaiTCN2018,
        author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
        title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
        journal   = {arXiv:1803.01271},
        year      = {2018},
    }
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from typing import List

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: int=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int=2, dropout: int=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    """
    This generic TCN model converts the image + modalities representation to a sequence of grid points.

    Args:
        input_channels: length of input sequence, and in our case, this value is 1 (1 image)
        num_input_features: number of features in input sequence, and in our case, this value is 64 
        num_output_features: number of features in output sequence, and in our case, this value is 2 (2D grid points)
        num_channels: list of size of hidden layers
        kernel_size: kernel size of 1D dilated convolution
        dropout: dropout rate after every 1D dilated convolution

    Usage:
        In our setting, we initialize our TCN as:
            TCN(
            input_channels=1, 
            num_input_features=64,
            num_output_features=2,
            num_channels=[30, 30, 30, 30, 30, 30, 30, 30, 4],
            kernel_size=7,
            dropout=0.0
        )
        This converts the (batch_size, 1, num_input_features) inputs to (batch_size, num_channels[-1], num_output_features)
        In our setting, this converts (1, 1, 64) inputs to (1, 4, 2) outputs.
    """
    def __init__(self, input_channels, num_input_features, num_output_features, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size=kernel_size, dropout=dropout)
        # going from 64 image features to 2 grid points
        self.linear = nn.Linear(num_input_features, num_output_features)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1)
