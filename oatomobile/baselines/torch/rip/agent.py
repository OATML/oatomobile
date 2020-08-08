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
"""Implements the robust imitative planning agent."""

from typing import Mapping
from typing import Sequence

import numpy as np
import scipy.interpolate
import torch
import torch.optim as optim

import oatomobile
from oatomobile.baselines.base import SetPointAgent
from oatomobile.baselines.torch.dim.model import ImitativeModel


class RIPAgent(SetPointAgent):
  """The robust imitative planning agent."""

  def __init__(self, environment: oatomobile.Env, *, algorithm: str,
               models: Sequence[ImitativeModel], **kwargs) -> None:
    """Constructs a robust imitative planning agent.

    Args:
      environment: The navigation environment to spawn the agent.
      algorithm: The RIP variant used, one of {"WCM", "MA", "BCM"}.
      models: The deep imitative models.
    """
    # Specifices the RIP variant.
    assert algorithm in ("WCM", "MA", "BCM")
    self._algorithm = algorithm

    super(RIPAgent, self).__init__(environment=environment, **kwargs)

    # Determines device, accelerator.
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
    self._models = [model.to(self._device) for model in models]

  def __call__(
      self,
      observation: Mapping[str, np.ndarray],
  ) -> np.ndarray:
    """Returns the imitative prior."""

    # Prepares observation for the neural-network.
    observation["overhead_features"] = observation[
        "bird_view_camera_cityscapes"]
    for attr in observation:
      if not isinstance(observation[attr], np.ndarray):
        observation[attr] = np.atleast_1d(observation[attr])
      observation[attr] = observation[attr][None, ...].astype(np.float32)

    # Makes `goal` 2D.
    observation["goal"] = observation["goal"][..., :2]
    # Convert image to CHW.
    observation["lidar"] = np.transpose(observation["lidar"], (0, 3, 1, 2))
    # Processes observations for the `ImitativeModel`.
    observation = {
        key: torch.from_numpy(tensor).to(self._device)  # pylint: disable=no-member
        for (key, tensor) in observation.items()
    }
    observation = self._models[0].transform(observation)

    # TODO(filangel) move this in `ImitativeModel.imitation_posterior`.
    lr = 1e-1
    epsilon = 1.0
    num_steps = 10
    ######
    batch_size = observation["visual_features"].shape[0]

    # Sets initial sample to base distribution's mean.
    x = self._models[0]._base_dist.mean.clone().detach().repeat(
        batch_size, 1).view(
            batch_size,
            *self._models[0]._output_shape,
        )
    x.requires_grad = True

    # The contextual parameters, caches for efficiency.
    zs = [model._params(**observation) for model in self._models]

    # Initialises a gradient-based optimiser.
    optimizer = optim.Adam(params=[x], lr=lr)

    # Stores the best values.
    x_best = x.clone()
    loss_best = torch.ones(()).to(x.device) * 1000.0  # pylint: disable=no-member

    for _ in range(num_steps):
      # Resets optimizer's gradients.
      optimizer.zero_grad()
      # Operate on `y`-space.
      y, _ = self._models[0]._forward(x=x, z=zs[0])
      # Iterates over the `K` models and calculates the imitation posterior.
      imitation_posteriors = list()
      for model, z in zip(self._models, zs):
        # Calculates imitation prior.
        _, log_prob, logabsdet = model._inverse(y=y, z=z)
        imitation_prior = torch.mean(log_prob - logabsdet)  # pylint: disable=no-member
        # Calculates goal likelihodd.
        goal_likelihood = model._goal_likelihood(
            y=y,
            goal=observation["goal"],
            epsilon=epsilon,
        )
        imitation_posteriors.append(imitation_prior + goal_likelihood)
      # Aggregate scores from the `K` models.
      imitation_posteriors = torch.stack(imitation_posteriors, dim=0)  # pylint: disable=no-member
      if self._algorithm == "WCM":
        loss, _ = torch.min(-imitation_posteriors, dim=0)  # pylint: disable=no-member
      elif self._algorithm == "BCM":
        loss, _ = torch.max(-imitation_posteriors, dim=0)  # pylint: disable=no-member
      else:
        loss = torch.mean(-imitation_posteriors, dim=0)  # pylint: disable=no-member
      # Backward pass.
      loss.backward(retain_graph=True)
      # Performs a gradient descent step.
      optimizer.step()
      # Book-keeping
      if loss < loss_best:
        x_best = x.clone()
        loss_best = loss.clone()

    plan, _ = self._models[0]._forward(x=x_best, z=zs[0])
    ######
    plan = plan.detach().cpu().numpy()[0]  # [T, 2]

    # TODO(filangel): clean API.
    # Interpolates plan.
    player_future_length = 40
    increments = player_future_length // plan.shape[0]
    time_index = list(range(0, player_future_length, increments))  # [T]
    plan_interp = scipy.interpolate.interp1d(x=time_index, y=plan, axis=0)
    xy = plan_interp(np.arange(0, time_index[-1]))

    # Appends z dimension.
    z = np.zeros(shape=(xy.shape[0], 1))
    return np.c_[xy, z]
