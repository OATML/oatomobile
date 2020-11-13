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
"""Defines a simple PID-controller-based autopilot."""

import math
import random
from typing import Any
from typing import Optional
from typing import Tuple

from absl import logging

import carla
import oatomobile
from oatomobile import envs
from oatomobile.simulators.carla import defaults
from oatomobile.utils import carla as cutil

try:
  from agents.navigation.behavior_agent import BehaviorAgent
except ImportError:
  raise ImportError("Missing CARLA installation, "
                    "make sure the environment variable CARLA_ROOT is provided "
                    "and that the PythonAPI is `easy_install`ed")


class AutopilotAgent(oatomobile.Agent):
  """An autopilot agent, based on the official implementation of
  `carla.PythonAPI.agents.navigation.basic_agent.BasicAgent`"""

  def __init__(self,
               environment: envs.CARLAEnv,
               *,
               noise: float = 0.0) -> None:  # TODO(farzad) why should we have noise=0.1 for autopilot?
    """Constructs an autopilot agent.

    Args:
      environment: The navigation environment to spawn the agent.
      noise: The percentage of random actions.
    """
    super(AutopilotAgent, self).__init__(environment=environment)

    # References to the CARLA objects.
    self._vehicle = self._environment.simulator.hero
    self._world = self._vehicle.get_world()
    self._map = self._world.get_map()
    self._agent = BehaviorAgent(self._vehicle, behavior='normal')
    self._noise = noise

    spawn_points = self._map.get_spawn_points()
    random.shuffle(spawn_points)

    if spawn_points[0].location != self._agent.vehicle.get_location():
      destination = spawn_points[0].location
    else:
      destination = spawn_points[1].location

    self._agent.set_destination(self._agent.vehicle.get_location(), destination, clean=True)

  def act(
      self,
      observation: oatomobile.Observations,
  ) -> oatomobile.Action:
    """Takes in an observation, samples from agent's policy, returns an
    action."""
    # Remove unused arguments.
    del observation

    # Random action branch.
    if random.random() < self._noise:
      return carla.VehicleControl(  # pylint: disable=no-member
          **{
              k: float(v)
              for (k, v) in self._environment.action_space.sample().items()
          })
    # Normal autopilot action.
    else:
      return self._run_step()

  def _run_step(
      self,
      debug: bool = False,
  ) -> carla.VehicleControl:  # pylint: disable=no-member
    """Executes one step of navigation."""

    self._agent.update_information()

    speed_limit = self._vehicle.get_speed_limit()
    self._agent.get_local_planner().set_speed(speed_limit)

    control = self._agent.run_step()

    return control

