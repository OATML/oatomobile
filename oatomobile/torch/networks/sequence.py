# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sequence generation."""

from typing import Tuple, List

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from oatomobile.torch import types
from oatomobile.torch.networks.mlp import MLP


class AutoregressiveFlow(nn.Module):
  """An autoregressive flow-based sequence generator."""

  def __init__(
      self,
      output_shape: types.Shape = (4, 2),
      hidden_size: int = 64,
  ):
    """Constructs a simple autoregressiv flow-based sequence generator.

    Args:
      output_shape: The shape of the base and
        data distribution (a.k.a. event_shape).
      hidden_size: The dimensionality of the GRU hidden state.
    """
    super(AutoregressiveFlow, self).__init__()
    self._output_shape = output_shape

    # Initialises the base distribution.
    self._base_dist = D.MultivariateNormal(
        loc=torch.zeros(self._output_shape[-2] * self._output_shape[-1]),  # pylint: disable=no-member
        scale_tril=torch.eye(self._output_shape[-2] * self._output_shape[-1]),  # pylint: disable=no-member
    )

    # The decoder recurrent network used for the sequence generation.
    self._decoder = nn.GRUCell(
        input_size=self._output_shape[-1],
        hidden_size=hidden_size,
    )

    # The output head.
    self._locscale = MLP(
        input_size=hidden_size,
        output_sizes=[32, self._output_shape[0]],
        activation_fn=nn.ReLU,
        dropout_rate=None,
        activate_final=False,
    )

  def to(self, *args, **kwargs):
    """Handles non-parameter tensors when moved to a new device."""
    self = super().to(*args, **kwargs)
    self._base_dist = D.MultivariateNormal(
        loc=self._base_dist.mean.to(*args, **kwargs),
        scale_tril=self._base_dist.scale_tril.to(*args, **kwargs),
    )
    return self

  def forward(
      self,
      z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-pass, stochastic generation of a sequence.

    Args:
      z: The contextual parameters of the conditional density estimator, with
        shape `[B, K]`.

    Returns:
      The sampels from the push-forward distribution, with shape `[B, D]`.
    """
    # Samples from the base distribution.
    x = self._base_dist.sample_n(n=z.shape[0])
    x = x.reshape(-1, *self._output_shape)

    return self._forward(x, z)[0]

  def _forward(
      self,
      x: torch.Tensor,
      z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transforms samples from the base distribution to the data distribution.

    Args:
      x: Samples from the base distribution, with shape `[B, D]`.
      z: The contextual parameters of the conditional density estimator, with
        shape `[B, K]`.

    Returns:
      y: The sampels from the push-forward distribution,
        with shape `[B, D]`.
      logabsdet: The log absolute determinant of the Jacobian,
        with shape `[B]`.
    """

    # Output containers.
    y = list()
    scales = list()

    # Initial input variable.
    y_tm1 = torch.zeros(  # pylint: disable=no-member
        size=(z.shape[0], self._output_shape[-1]),
        dtype=z.dtype,
    ).to(z.device)

    for t in range(x.shape[-2]):
      x_t = x[:, t, :]

      # Unrolls the GRU.
      z = self._decoder(y_tm1, z)

      # Predicts the location and scale of the MVN distribution.
      dloc_scale = self._locscale(z)
      dloc = dloc_scale[..., :2]
      scale = F.softplus(dloc_scale[..., 2:]) + 1e-3

      # Data distribution corresponding sample.
      y_t = (y_tm1 + dloc) + scale * x_t

      # Update containers.
      y.append(y_t)
      scales.append(scale)
      y_tm1 = y_t

    # Prepare tensors, reshape to [B, T, 2].
    y = torch.stack(y, dim=-2)  # pylint: disable=no-member
    scales = torch.stack(scales, dim=-2)  # pylint: disable=no-member

    # Log absolute determinant of Jacobian.
    logabsdet = torch.log(torch.abs(torch.prod(scales, dim=-2)))  # pylint: disable=no-member
    logabsdet = torch.sum(logabsdet, dim=-1)  # pylint: disable=no-member

    return y, logabsdet

  def _inverse(
      self,
      y: torch.Tensor,
      z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transforms samples from the data distribution to the base distribution.

    Args:
      y: Samples from the data distribution, with shape `[B, D]`.
      z: The contextual parameters of the conditional density estimator, with shape
        `[B, K]`.

    Returns:
      x: The sampels from the base distribution,
        with shape `[B, D]`.
      log_prob: The log-likelihood of the samples under
        the base distibution probability, with shape `[B]`.
      logabsdet: The log absolute determinant of the Jacobian,
        with shape `[B]`.
    """

    # Output containers.
    x = list()
    scales = list()

    # Initial input variable.
    y_tm1 = torch.zeros(  # pylint: disable=no-member
        size=(z.shape[0], self._output_shape[-1]),
        dtype=z.dtype,
    ).to(z.device)

    for t in range(y.shape[-2]):
      y_t = y[:, t, :]

      # Unrolls the GRU.
      z = self._decoder(y_tm1, z)

      # Predicts the location and scale of the MVN distribution.
      dloc_scale = self._locscale(z)
      dloc = dloc_scale[..., :2]
      scale = F.softplus(dloc_scale[..., 2:]) + 1e-3

      # Base distribution corresponding sample.
      x_t = (y_t - (y_tm1 + dloc)) / scale

      # Update containers.
      x.append(x_t)
      scales.append(scale)
      y_tm1 = y_t

    # Prepare tensors, reshape to [B, T, 2].
    x = torch.stack(x, dim=-2)  # pylint: disable=no-member
    scales = torch.stack(scales, dim=-2)  # pylint: disable=no-member

    # Log likelihood under base distribution.
    log_prob = self._base_dist.log_prob(x.view(x.shape[0], -1))

    # Log absolute determinant of Jacobian.
    logabsdet = torch.log(  # pylint: disable=no-member
        torch.abs(torch.prod(  # pylint: disable=no-member
            scales, dim=-1)))  # determinant == product over xy-coordinates
    logabsdet = torch.sum(logabsdet, dim=-1)  # sum over T dimension # pylint: disable=no-member

    return x, log_prob, logabsdet

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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
