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
"""Defines a discriminative model for the conditional imitation learner."""

from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from oatomobile.baselines.torch import transforms
from oatomobile.baselines.torch.models import MLP
from oatomobile.baselines.torch.models import MobileNetV2
from oatomobile.baselines.torch.typing import ArrayLike
from oatomobile.core.typing import ShapeLike


class ImitativeModel(nn.Module):
  """A `PyTorch` implementation of an imitative model."""

  def __init__(
      self,
      output_shape: ShapeLike = (4, 2),
  ) -> None:
    """Constructs a simple behavioural cloning model.

    Args:
      output_shape: The shape of the base and
        data distribution (a.k.a. event_shape).
    """
    super(ImitativeModel, self).__init__()
    self._output_shape = output_shape

    # Initialises the base distribution.
    self._base_dist = D.MultivariateNormal(
        loc=torch.zeros(self._output_shape[-2] * self._output_shape[-1]),  # pylint: disable=no-member
        scale_tril=torch.eye(self._output_shape[-2] * self._output_shape[-1]),  # pylint: disable=no-member
    )

    # The convolutional encoder model.
    self._encoder = MobileNetV2(num_classes=128, in_channels=2)

    # Merges the encoded features and the vector inputs.
    self._merger = MLP(
        input_size=128 + 3 + 1 + 1,
        output_sizes=[64, 64, 64],
        activation_fn=nn.ReLU,
        dropout_rate=None,
        activate_final=True,
    )

    # The decoder recurrent network used for the sequence generation.
    self._decoder = nn.GRUCell(input_size=2, hidden_size=64)

    # The output head.
    self._locscale = MLP(
        input_size=64,
        output_sizes=[32, 4],
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
      num_steps: int,
      goal: Optional[torch.Tensor] = None,
      lr: float = 1e-1,
      epsilon: float = 1.0,
      **context: torch.Tensor) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """Returns a local mode from the posterior.

    Args:
      num_steps: The number of gradient-descent steps for finding the mode.
      goal: The locations of the the goals.
      epsilon: The tolerance parameter for the goal.
      context: (keyword arguments) The conditioning
        variables used for the conditional flow.

    Returns:
      A mode from the posterior, with shape `[D, 2]`.
    """
    if not "visual_features" in context:
      raise ValueError("Missing `visual_features` keyword argument.")
    batch_size = context["visual_features"].shape[0]

    # Sets initial sample to base distribution's mean.
    x = self._base_dist.sample().clone().detach().repeat(batch_size, 1).view(
        batch_size,
        *self._output_shape,
    )
    x.requires_grad = True

    # The contextual parameters, caches for efficiency.
    z = self._params(**context)

    # Initialises a gradient-based optimiser.
    optimizer = optim.Adam(params=[x], lr=lr)

    # Stores the best values.
    x_best = x.clone()
    loss_best = torch.ones(()).to(x.device) * 1000.0  # pylint: disable=no-member

    for _ in range(num_steps):
      # Resets optimizer's gradients.
      optimizer.zero_grad()
      # Operate on `y`-space.
      y, _ = self._forward(x=x, z=z)
      # Calculates imitation prior.
      _, log_prob, logabsdet = self._inverse(y=y, z=z)
      imitation_prior = torch.mean(log_prob - logabsdet)  # pylint: disable=no-member
      # Calculates goal likelihodd.
      goal_likelihood = 0.0
      if goal is not None:
        goal_likelihood = self._goal_likelihood(y=y, goal=goal, epsilon=epsilon)
      loss = -(imitation_prior + goal_likelihood)
      # Backward pass.
      loss.backward(retain_graph=True)
      # Performs a gradient descent step.
      optimizer.step()
      # Book-keeping
      if loss < loss_best:
        x_best = x.clone()
        loss_best = loss.clone()

    y, _ = self._forward(x=x_best, z=z)

    return y

  def _goal_likelihood(self, y: torch.Tensor, goal: torch.Tensor,
                       **hyperparams) -> torch.Tensor:
    """Returns the goal-likelihood of a plan `y`, given `goal`.

    Args:
      y: A plan under evaluation, with shape `[B, T, 2]`.
      goal: The goal locations, with shape `[B, K, 2]`.
      hyperparams: (keyword arguments) The goal-likelihood hyperparameters.

    Returns:
      The log-likelihodd of the plan `y` under the `goal` distribution.
    """
    # Parses tensor dimensions.
    B, K, _ = goal.shape

    # Fetches goal-likelihood hyperparameters.
    epsilon = hyperparams.get("epsilon", 1.0)

    # TODO(filangel): implement other goal likelihoods from the DIM paper
    # Initializes the goal distribution.
    goal_distribution = D.MixtureSameFamily(
        mixture_distribution=D.Categorical(
            probs=torch.ones((B, K)).to(goal.device)),  # pylint: disable=no-member
        component_distribution=D.Independent(
            D.Normal(loc=goal, scale=torch.ones_like(goal) * epsilon),  # pylint: disable=no-member
            reinterpreted_batch_ndims=1,
        ))

    return torch.mean(goal_distribution.log_prob(y[:, -1, :]), dim=0)  # pylint: disable=no-member

  def _params(self, **context: torch.Tensor) -> torch.Tensor:
    """Returns the contextual parameters of the conditional density estimator.

    Args:
      visual_features: The visual input, with shape `[B, H, W, 3]`.
      velocity: The vehicle's current velocity, with shape `[B, 3]`.
      is_at_traffic_light: The flag for being close to a traffic light, with
        shape `[B, 1]`.
      traffic_light_state: The state of the nearest traffic light, with shape
        `[B, 1]`.

    Returns:
      The contextual parameters of the conditional density estimator.
    """

    # Parses context variables.
    if not "visual_features" in context:
      raise ValueError("Missing `visual_features` keyword argument.")
    if not "velocity" in context:
      raise ValueError("Missing `velocity` keyword argument.")
    if not "is_at_traffic_light" in context:
      raise ValueError("Missing `is_at_traffic_light` keyword argument.")
    if not "traffic_light_state" in context:
      raise ValueError("Missing `traffic_light_state` keyword argument.")
    visual_features = context.get("visual_features")
    velocity = context.get("velocity")
    is_at_traffic_light = context.get("is_at_traffic_light")
    traffic_light_state = context.get("traffic_light_state")

    # Encodes the visual input.
    visual_features = self._encoder(visual_features)

    # Merges visual input logits and vector inputs.
    visual_features = torch.cat(  # pylint: disable=no-member
        tensors=[
            visual_features,
            velocity,
            is_at_traffic_light,
            traffic_light_state,
        ],
        dim=-1,
    )

    # The decoders initial state.
    visual_features = self._merger(visual_features)

    return visual_features

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

    return sample
