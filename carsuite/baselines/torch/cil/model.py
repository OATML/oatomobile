# Copyright 2020 The CARSUITE Authors. All Rights Reserved.
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
"""Defines a discriminative model for the conditional imitation learner."""

from typing import Mapping
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from carsuite.baselines.torch import transforms
from carsuite.baselines.torch.models import MLP
from carsuite.baselines.torch.models import MobileNetV2
from carsuite.baselines.torch.typing import ArrayLike


class BehaviouralModel(nn.Module):
  """A `PyTorch` implementation of a behavioural cloning model."""

  def __init__(
      self,
      output_shape: Tuple[int, int] = (40, 2),
  ) -> None:
    """Constructs a simple behavioural cloning model.

    Args:
      output_shape: The shape of the base and
        data distribution (a.k.a. event_shape).
    """
    super(BehaviouralModel, self).__init__()
    self._output_shape = output_shape

    # The convolutional encoder model.
    self._encoder = MobileNetV2(num_classes=128, in_channels=2)

    # Merges the encoded features and the vector inputs.
    self._merger = MLP(
        input_size=128 + 3 + 1 + 1 + 1,
        output_sizes=[64, 64, 64],
        activation_fn=nn.ReLU,
        dropout_rate=None,
        activate_final=True,
    )

    # The decoder recurrent network used for the sequence generation.
    self._decoder = nn.GRUCell(input_size=2, hidden_size=64)

    # The output head.
    self._output = nn.Linear(
        in_features=64,
        out_features=self._output_shape[-1],
    )

  def forward(self, **context: torch.Tensor) -> torch.Tensor:
    """Returns the expert plan."""

    # Parses context variables.
    if not "visual_features" in context:
      raise ValueError("Missing `visual_features` keyword argument.")
    visual_features = context.get("visual_features")
    if not "velocity" in context:
      raise ValueError("Missing `velocity` keyword argument.")
    velocity = context.get("velocity")
    if not "is_at_traffic_light" in context:
      raise ValueError("Missing `is_at_traffic_light` keyword argument.")
    is_at_traffic_light = context.get("is_at_traffic_light")
    if not "traffic_light_state" in context:
      raise ValueError("Missing `traffic_light_state` keyword argument.")
    traffic_light_state = context.get("traffic_light_state")
    if not "mode" in context:
      raise ValueError("Missing `mode` keyword argument.")
    mode = context.get("mode")

    # Encodes the visual input.
    visual_features = self._encoder(visual_features)

    # Merges visual input logits and vector inputs.
    z = torch.cat(  # pylint: disable=no-member
        tensors=[
            visual_features,
            velocity,
            is_at_traffic_light,
            traffic_light_state,
            mode,
        ],
        dim=-1,
    )

    # The decoders initial state.
    z = self._merger(z)

    # Output container.
    y = list()

    # Initial input variable.
    x = torch.zeros(  # pylint: disable=no-member
        size=(z.shape[0], self._output_shape[-1]),
        dtype=z.dtype,
    ).to(z.device)

    # Autoregressive generation of plan.
    for _ in range(self._output_shape[0]):
      # Unrolls the GRU.
      z = self._decoder(x, z)

      # Predicts the displacement (residual).
      dx = self._output(z)
      x = dx + x

      # Updates containers.
      y.append(x)

    return torch.stack(y, dim=1)  # pylint: disable=no-member

  def transform(
      self,
      sample: Mapping[str, ArrayLike],
  ) -> Mapping[str, torch.Tensor]:
    """Prepares variables for the interface of the model.

    Args:
      sample: (keyword arguments) The raw sample variables.

    Returns:
      The processed sample.
    """

    # Preprocesses the target variables.
    if "player_future" in sample:
      sample["player_future"] = transforms.downsample_target(
          player_future=sample["player_future"],
          num_timesteps_to_keep=self._output_shape[-2],
      )

    # Renames `lidar` to `visual_features`.
    if "lidar" in sample:
      sample["visual_features"] = sample.pop("lidar")

    # Preprocesses the visual features.
    if "visual_features" in sample:
      sample["visual_features"] = transforms.transpose_visual_features(
          transforms.downsample_visual_features(
              visual_features=sample["visual_features"],
              output_shape=(100, 100),
          ))

    # Removes the "STOP" command to avoid causal confusion with traffic lights.
    if "mode" in sample:
      sample["mode"][sample["mode"] == 1.0] = 0.0

    return sample
