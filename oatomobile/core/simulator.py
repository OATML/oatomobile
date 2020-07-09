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
"""Defines the core APIs to interface with simulators."""

import abc
from enum import Enum
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Union

import gym
import numpy as np
from absl import logging

# All agents are expected to return the same action type.
Action = Any

# Enumeration of types of sensors.
SensorTypes = Enum


class Sensor(abc.ABC):
  """A sensor consists of a fetching mechanism for observations."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Constructs a sensor."""
    self.uuid = self._get_uuid(*args, **kwargs)
    self.sensor_type = self._get_sensor_type(*args, **kwargs)

  @abc.abstractmethod
  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""

  @abc.abstractmethod
  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
    """Returns the the type of the sensor."""

  @property
  @abc.abstractproperty
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Space:
    """Returns the observation spec of the sensor."""

  @abc.abstractmethod
  def get_observation(self, *args: Any, **kwargs: Any) -> Any:
    """Retrieves the observation from the sensor."""

  @abc.abstractmethod
  def close(self, *args: Any, **kwargs: Any) -> None:
    """Destroys the sensor and any connections to the server."""

  @classmethod
  @abc.abstractclassmethod
  def default(cls, *args: Any, **kwargs: Any) -> "Sensor":
    """Returns the default sensor instance."""


class Observations(dict):
  """Dictionary containing sensor observations."""

  def __init__(self, sensors: Mapping[str, Sensor], *args: Any,
               **kwargs: Any) -> None:
    """Constructors a dictionary of observations from sensors.

    Args:
      sensors: list of sensors whose observations are fetched and packaged.
    """
    data = [(uuid, sensor.get_observation(*args, **kwargs))
            for uuid, sensor in sensors.items()]
    super().__init__(data)


class SensorSuite:
  """Represents a set of sensors, with each sensor being identified through a
  unique id."""

  def __init__(
      self,
      sensors: Iterable[Sensor],
  ) -> None:
    """Constructs a sensor suite.

    Args:
      sensors: The set of all the sensors to be bundled.
    """
    self.sensors = dict()
    self._observation_space = dict()
    for sensor in sensors:
      if sensor.uuid in self.sensors:
        raise KeyError("{} is duplicated sensor uuid".format(sensor.uuid))
      self.sensors[sensor.uuid] = sensor
      self._observation_space[sensor.uuid] = sensor.observation_space

  def get(self, uuid: str) -> Sensor:
    """Returns a pointer to the sensor with uuid.

    Args:
      uuid: The universal unique identifier of the sensor.

    Returns:
      A reference to the sensor with a specific uuid.
    """
    return self.sensors.get(uuid)

  def get_observations(self, *args: Any, **kwargs: Any) -> Observations:
    """Collects data from all sensors and returns it packaged."""
    return Observations(self.sensors, *args, **kwargs)

  @property
  def observation_space(self) -> gym.spaces.Dict:
    """Returns the observation spec of the sensors."""
    return gym.spaces.Dict({
        sensor.uuid: sensor.observation_space
        for sensor in self.sensors.values()
    })

  def close(self) -> None:
    """Closes all the sensors."""
    for name, sensor in self.sensors.items():
      logging.debug("Destroys {} sensor".format(name))
      sensor.close()


class Simulator(abc.ABC):
  """Basic simulator class for `oatomobile`."""

  @property
  @abc.abstractmethod
  def sensor_suite(self) -> SensorSuite:
    """Returns a refernce to the suite of sensors."""

  @abc.abstractmethod
  def action_space(self) -> Any:
    """Returns the specification of the actions expected by the simulator."""

  @property
  def observation_space(self) -> gym.spaces.Dict:
    """Returns the specification of the observations returned by the
    simulator."""
    return self.sensor_suite.observation_space

  @abc.abstractmethod
  def seed(self, seed: int) -> None:
    """Fixes the random number generator state."""

  @abc.abstractmethod
  def reset(self, *args: Any, **kwargs: Any) -> Observations:
    """Resets the state of the simulation to the initial state."""

  @abc.abstractmethod
  def step(self, action: Action, *args: Any, **kwargs: Any) -> Observations:
    """Makes a step in the simulator, provided an action."""

  @abc.abstractmethod
  def render(self, mode: str = "rgb_array", *args: Any, **kwargs: Any) -> Any:
    """Renders current state of the simulator."""

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the simulator down and controls connection to the server."""
