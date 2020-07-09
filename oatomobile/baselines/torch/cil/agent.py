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
"""Implements the conditional imitation learning agent."""

from typing import Mapping

import numpy as np
import scipy.interpolate
import torch

import oatomobile
from oatomobile.baselines.base import SetPointAgent
from oatomobile.baselines.torch.cil.model import BehaviouralModel


class CILAgent(SetPointAgent):
  """The conditional imitation learning agent."""

  def __init__(self, environment: oatomobile.envs.CARLAEnv, *,
               model: BehaviouralModel, **kwargs) -> None:
    """Constructs a conditional imitation learning agent.

    Args:
      environment: The navigation environment to spawn the agent.
      model: The deep behavioural cloned model.
    """
    super(CILAgent, self).__init__(environment=environment, **kwargs)

    # Determines device, accelerator.
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member
    self._model = model.to(self._device)

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
    # Calculates `command` -- borrowed from `CARLADataset.mode`
    _x_T, _y_T = observation["goal"][0, -1, :2]
    _norm = np.linalg.norm([_x_T, _y_T])
    _theta = np.degrees(np.arccos(_x_T / (_norm + 1e-3)))
    if _norm < 3:  # STOP
      observation["mode"] = 1
    elif _theta > 15:  # LEFT
      observation["mode"] = 2
    elif _theta <= 15:  # RIGHT
      observation["mode"] = 3
    else:  # FORWARD
      observation["mode"] = 0
    observation["mode"] = np.atleast_2d(observation["mode"])
    observation["mode"] = observation["mode"].astype(observation["goal"].dtype)
    # Processes observations for the `BehaviouralModel`.
    observation = {
        key: torch.from_numpy(tensor).to(self._device)  # pylint: disable=no-member
        for (key, tensor) in observation.items()
    }
    observation = self._model.transform(observation)

    # Queries model.
    plan = self._model(**observation).detach().cpu().numpy()[0]  # [T, 2]

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
