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
"""OpenAI Gym wrapper of the CARLA simulator."""

import copy
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

import carla
import gym
import numpy as np
from absl import logging

from carsuite.core.registry import registry
from carsuite.core.rl import Env
from carsuite.core.rl import Metric
from carsuite.core.rl import Transition
from carsuite.core.simulator import Observations
from carsuite.simulators.carla import defaults
from carsuite.simulators.carla.simulator import CARLAAction
from carsuite.simulators.carla.simulator import CARLASimulator
from carsuite.util import carla as cutil


class CARLAEnv(Env):
  """A CARLA simulator-based OpenAI gym-compatible environment."""

  def __init__(
      self,
      *,
      town: str,
      spawn_point: Optional[Union[int, carla.Location]] = None,  # pylint: disable=no-member
      destination: Optional[Union[int, carla.Location]] = None,  # pylint: disable=no-member
      fps: int = defaults.SIMULATOR_FPS,
      sensors: Sequence[str] = defaults.CARLA_SENSORS,
      num_vehicles: int = 0,
      num_pedestrians: int = 0) -> None:
    """Constructs a CARLA simulator-based OpenAI gym-compatible environment.

    Args:
      town: The `CARLA` town identifier.
      spawn_point: The hero vehicle spawn point. If an int is
        provided then the index of the spawn point is used.
        If None, then randomly selects a spawn point every time
        from the available spawn points of each map.
      destination: The final destination. If an int is
        provided then the index of the spawn point is used.
        If None, then randomly selects a spawn point every time
        from the available spawn points of each map.
      fps: The frequency (in Hz) of the simulation.
      sensors: The set of sensors registered on the ego vehicle.
      num_vehicles: The number of vehicles to spawn.
      num_pedestrians: The number of pedestrians to spawn.
    """
    # Makes sure main sensors are registered and that passed are registered.
    _sensors = set([
        "collision",
        "lane_invasion",
        "location",
        "rotation",
        "control",
        "predictions",
    ])
    for sensor in sensors:
      if registry.get_sensor(sensor) is not None:
        _sensors.add(sensor)
    _sensors = list(set(_sensors))

    # Core simulator used to interface with CARLA server.
    super(CARLAEnv, self).__init__(
        sim_fn=CARLASimulator,
        # The keyword arguments passed in `CARLASimulator`.
        town=town,
        sensors=_sensors,
        fps=fps,
        spawn_point=spawn_point,
        destination=destination,
        num_vehicles=num_vehicles,
        num_pedestrians=num_pedestrians,
    )

  @property
  def action_space(self) -> gym.spaces.Dict:
    """Returns the expected action passed to the `step` method."""
    return gym.spaces.Dict(
        throttle=gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        ),
        steer=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        ),
        brake=gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        ),
    )


class CARLANavEnv(CARLAEnv):
  """CARLA simulator-based navigation environment."""

  def __init__(
      self,
      *,
      town: str,
      origin: Union[int, carla.Location],  # pylint: disable=no-member
      destination: Union[int, carla.Location],  # pylint: disable=no-member
      fps: int = defaults.SIMULATOR_FPS,
      sensors: Sequence[str] = defaults.CARLA_SENSORS,
      num_vehicles: int = 0,
      num_pedestrians: int = 0,
      proximity_destination_threshold: float = 7.5) -> None:
    """Constructs a CARLA simulator-based OpenAI gym-compatible environment.

    Args:
      town: The `CARLA` town identifier.
      fps: The frequency (in Hz) of the simulation.
      sensors: The set of sensors registered on the ego vehicle.
      spawn_point: The hero vehicle spawn point. If an int is
        provided then the index of the spawn point is used.
        If None, then randomly selects a spawn point every time
        from the available spawn points of each map.
      destination: The final destination. If an int is
        provided then the index of the spawn point is used.
        If None, then randomly selects a spawn point every time
        from the available spawn points of each map.
      num_vehicles: The number of vehicles to spawn.
      num_pedestrians: The number of pedestrians to spawn.
      proximity_destination_threshold: Distance from destination
        to successfully reach the goal.
    """
    super(CARLANavEnv, self).__init__(
        town=town,
        spawn_point=origin,
        destination=destination,
        fps=fps,
        sensors=sensors,
        num_vehicles=num_vehicles,
        num_pedestrians=num_pedestrians,
    )
    # Internalize hyperparameters.
    self._proximity_destination_threshold = proximity_destination_threshold

  def step(self, action: CARLAAction) -> Transition:
    """Makes a step in the simulator, provided an action."""
    observation, reward, done, info = super(CARLANavEnv, self).step(action)

    # Get distance from destination.
    if not done:
      destination = self.simulator._destination
      current_location = observation["location"]
      destination_location = np.asarray(
          [
              destination.location.x,
              destination.location.y,
              destination.location.z,
          ],
          dtype=np.float32,
      )
      distance_to_go = np.linalg.norm(current_location - destination_location)
      done = distance_to_go < self._proximity_destination_threshold
      reward = float(done)

    return observation, reward, done, info


class LaneInvasionsMetric(Metric):
  """Records the number of lane invasions in an episode."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes the lane invasion counter."""
    super(LaneInvasionsMetric, self).__init__(initial_value=0)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the metric."""
    return "lane_invasions"

  def update(self, observations: Observations, action: CARLAAction,
             reward: float, new_observations: Observations, *args: Any,
             **kwargs: Any) -> None:
    """Records transition and update evaluation."""
    if new_observations["lane_invasion"] > 0:
      self.value += 1


class TerminateOnLaneInvasionWrapper(gym.Wrapper):
  """Terminates episode on lane invasion."""

  def __init__(self, env: gym.Env) -> None:
    """Constructs a gym wrapper to terminate execution on lane invasion."""
    super(TerminateOnLaneInvasionWrapper, self).__init__(env=env)

  def step(self, action: CARLAAction, *args: Any, **kwargs: Any) -> Transition:
    """Steps the wrapped environment and terminates if any lane is invaded."""
    observation, reward, done, info = self.env.step(action)
    if observation["lane_invasion"] > 0:
      logging.debug("A lane was invaded")
      done = True
      reward = -1.0
    return observation, reward, done, info


class CollisionsMetric(Metric):
  """Records the number of collisions in an episode."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes the lane invasion counter."""
    super(CollisionsMetric, self).__init__(initial_value=0)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the metric."""
    return "collisions"

  def update(self, observations: Observations, action: CARLAAction,
             reward: float, new_observations: Observations, *args: Any,
             **kwargs: Any) -> None:
    """Records transition and update evaluation."""
    if new_observations["collision"] > 0:
      self.value += 1


class TerminateOnCollisionWrapper(gym.Wrapper):
  """Terminates episode on collision."""

  def __init__(self, env: gym.Env) -> None:
    """Constructs a gym wrapper to terminate execution on collision."""
    super(TerminateOnCollisionWrapper, self).__init__(env=env)

  def step(self, action: CARLAAction, *args: Any, **kwargs: Any) -> Transition:
    """Steps the wrapped environment and terminates if any collision occurs."""
    observation, reward, done, info = self.env.step(action)
    if observation["collision"] > 0:
      logging.debug("A collision occured")
      done = True
      reward = -1.0
    return observation, reward, done, info


class DistanceMetric(Metric):
  """Records the travelled distance (in meters) in an episode."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes the distance counter."""
    super(DistanceMetric, self).__init__(initial_value=0.0)
    self._past_location = None

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the metric."""
    return "distance"

  def update(self, observations: Observations, action: CARLAAction,
             reward: float, new_observations: Observations, *args: Any,
             **kwargs: Any) -> None:
    """Records transition and update evaluation."""
    self.value += np.linalg.norm(  # Euclidean distance in meters
        x=new_observations["location"] - observations["location"],
        ord=2,
    )
