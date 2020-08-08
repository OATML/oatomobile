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
"""Implements the deep imitative model-based agent."""

from typing import Mapping

import numpy as np
import scipy.interpolate
import torch

import oatomobile
from oatomobile.baselines.base import SetPointAgent
from oatomobile.baselines.torch.dim.model import ImitativeModel


class DIMAgent(SetPointAgent):
  """The deep imitative model agent."""

  def __init__(self, environment: oatomobile.Env, *, model: ImitativeModel,
               **kwargs) -> None:
    """Constructs a deep imitation model agent.

    Args:
      environment: The navigation environment to spawn the agent.
      model: The deep imitative model.
    """
    super(DIMAgent, self).__init__(environment=environment, **kwargs)

    # Determines device, accelerator.
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
    self._model = model.to(self._device)

  def __call__(self, observation: Mapping[str, np.ndarray],
               **kwargs) -> np.ndarray:
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
    observation = self._model.transform(observation)

    # Queries model.
    plan = self._model(num_steps=kwargs.get("num_steps", 20),
                       epsilon=kwargs.get("epsilon", 1.0),
                       lr=kwargs.get("lr", 5e-2),
                       **observation).detach().cpu().numpy()[0]  # [T, 2]

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
