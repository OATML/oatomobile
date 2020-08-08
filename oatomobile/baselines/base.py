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
"""Defines the interfaces for agents."""

import abc
import copy
import math
from collections import deque
from typing import Any
from typing import Mapping
from typing import Optional

import numpy as np
from absl import logging

import oatomobile

# Default PID controllers configuration.
SIMULATOR_FPS = 20
LATERAL_PID_CONTROLLER_CONFIG = {
    'K_P': 1.95,
    'K_D': 0.01,
    'K_I': 1.4,
    'dt': 1.0 / SIMULATOR_FPS,
}
LONGITUDINAL_PID_CONTROLLER_CONFIG = {
    'K_P': 1.0,
    'K_D': 0,
    'K_I': 1.0,
    'dt': 1.0 / SIMULATOR_FPS,
}


class SetPointAgent(oatomobile.Agent):
  """An agent that predicts setpoints and consumes them with a PID
  controller."""

  def __init__(
      self,
      environment: oatomobile.Env,
      *,
      setpoint_index: int = 5,
      replan_every_steps: int = 1,
      lateral_control_dict: Mapping[str, Any] = LATERAL_PID_CONTROLLER_CONFIG,
      longitudinal_control_dict: Mapping[
          str, Any] = LONGITUDINAL_PID_CONTROLLER_CONFIG,
      fixed_delta_seconds_between_setpoints: Optional[int] = None) -> None:
    """Constructs a setpoint-based agent.

    Args:
      environment: The navigation environment to spawn the agent.
      setpoint_index: The index of the point to cut-off the plan.
      replan_every_steps: The number of steps between subsequent call on the model.
      lateral_control_dict: The lateral PID controller's config.
      longitudinal_control_dict: The longitudinal PID controller's config.
      fixed_delta_seconds_between_setpoints: The time difference (in seconds)
        between the setpoints. It defaults to the fps of the simulator.
    """
    try:
      from agents.navigation.controller import VehiclePIDController  # pylint: disable=import-error
    except ImportError:
      raise ImportError(
          "Missing CARLA installation, "
          "make sure the environment variable CARLA_ROOT is provided "
          "and that the PythonAPI is `easy_install`ed")

    super(SetPointAgent, self).__init__(environment=environment)

    # References to the CARLA objects.
    self._vehicle = self._environment.simulator.hero
    self._world = self._vehicle.get_world()
    self._map = self._world.get_map()

    # Sets up PID controllers.
    dt = self._vehicle.get_world().get_settings().fixed_delta_seconds
    lateral_control_dict = lateral_control_dict.copy()
    lateral_control_dict.update({"dt": dt})
    logging.debug(
        "Lateral PID controller config {}".format(lateral_control_dict))
    longitudinal_control_dict = longitudinal_control_dict.copy()
    longitudinal_control_dict.update({"dt": dt})
    logging.debug("Longitudinal PID controller config {}".format(
        longitudinal_control_dict))
    self._vehicle_controller = VehiclePIDController(
        vehicle=self._vehicle,
        args_lateral=lateral_control_dict,
        args_longitudinal=longitudinal_control_dict,
    )

    # Sets agent's hyperparameters.
    self._setpoint_index = setpoint_index
    self._replan_every_steps = replan_every_steps
    self._fixed_delta_seconds_between_setpoints = fixed_delta_seconds_between_setpoints or dt

    # Inits agent's buffer of setpoints.
    self._setpoints_buffer = None
    self._steps_counter = 0

  @abc.abstractmethod
  def __call__(self, observation: oatomobile.Observations, *args,
               **kwargs) -> np.ndarray:
    """Returns the predicted plan in ego-coordinates, based on a model."""

  def act(self, observation: oatomobile.Observations, *args,
          **kwargs) -> oatomobile.Action:
    """Takes in an observation, samples from agent's policy, returns an
    action."""
    from oatomobile.util import carla as cutil

    # Current measurements used for local2world2local transformations.
    current_location = observation["location"]
    current_rotation = observation["rotation"]

    if self._setpoints_buffer is None or self._steps_counter % self._replan_every_steps == 0:
      # Get agent predictions.
      predicted_plan_ego = self(copy.deepcopy(observation), *args,
                                **kwargs)  # [T, 3]
      # Transform plan to world coordinates
      predicted_plan_world = cutil.local2world(
          current_location=current_location,
          current_rotation=current_rotation,
          local_locations=predicted_plan_ego,
      )

      # Refreshes buffer.
      self._setpoints_buffer = predicted_plan_world

    else:
      # Pops first setpoint from the buffer.
      self._setpoints_buffer = self._setpoints_buffer[1:]

    # Registers setpoints for rendering.
    self._environment.unwrapped.simulator.sensor_suite.get(
        "predictions").predictions = cutil.world2local(
            current_location=current_location,
            current_rotation=current_rotation,
            world_locations=self._setpoints_buffer,
        )

    # Increments counter.
    self._steps_counter += 1

    # Calculates target speed by averaging speed in the `setpoint_index` window.
    target_speed = np.linalg.norm(
        np.diff(self._setpoints_buffer[:self._setpoint_index], axis=0),
        axis=1,
    ).mean() / (self._fixed_delta_seconds_between_setpoints)

    # Converts plan to PID controller setpoint.
    setpoint = self._map.get_waypoint(
        cutil.ndarray_to_location(self._setpoints_buffer[self._setpoint_index]))

    # Avoids getting stuck when spawned.
    if self._steps_counter <= 100:
      target_speed = 20.0 / 3.6

    # Run PID step.
    control = self._vehicle_controller.run_step(
        target_speed=target_speed *
        3.6,  # PID controller expects speed in km/h!
        waypoint=setpoint,
    )

    return control
