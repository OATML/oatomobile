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
"""Basic `Sensor` interface."""

import abc
from typing import Any
from typing import Iterable
from typing import Mapping

from absl import logging
from acme.types import NestedSpec


class Sensor(abc.ABC):
  """A sensor consists of a fetching mechanism for observations."""

  def __init__(self, uuid: str) -> None:
    """Initialize a `Sensor`.

    Args:
      uuid: String unique identifier.
    """
    self._uuid = uuid

  @property
  def uuid(self) -> str:
    """Returns the sensor's unique identifier."""
    return self._uuid

  @abc.abstractmethod
  def observation_spec(self) -> NestedSpec:
    """Returns the observation spec of the sensor."""

  @abc.abstractmethod
  def get_observation(self, *args: Any, **kwargs: Any) -> Any:
    """Retrieves the observation from the sensor."""

  def close(self) -> None:
    """Destroys the sensor and any (potential) connections to the server."""

  @classmethod
  @abc.abstractclassmethod
  def default(cls, *args: Any, **kwargs: Any) -> "Sensor":
    """Returns the default sensor instance."""


class Observations(dict):
  """Dictionary containing sensor observations."""

  def __init__(
      self,
      sensors: Mapping[str, Sensor],
      *args: Any,
      **kwargs: Any,
  ) -> None:
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
    self._sensors = dict()
    for sensor in sensors:
      if sensor.uuid in self._sensors:
        raise KeyError("{} is duplicated sensor uuid".format(sensor.uuid))
      self._sensors[sensor.uuid] = sensor

  @property
  def sensors(self) -> Mapping[str, Sensor]:
    """Returns a mapping of sensor uuid and sensor object."""
    return self._sensors

  def get(self, uuid: str) -> Sensor:
    """Returns a pointer to the sensor with uuid.

    Args:
      uuid: The universal unique identifier of the sensor.

    Returns:
      A reference to the sensor with a specific uuid.
    """
    return self.sensors.get(uuid)

  def get_observations(self, *args, **kwargs) -> Observations:
    """Collects data from all sensors and returns it packaged."""
    return Observations(self.sensors, *args, **kwargs)

  def observation_spec(self) -> NestedSpec:
    """Returns the observation spec of the sensors."""
    return {
        sensor.uuid: sensor.observation_spec()
        for sensor in self.sensors.values()
    }

  def close(self) -> None:
    """Closes all the sensors."""
    for name, sensor in self.sensors.items():
      logging.debug("Destroys {} sensor".format(name))
      sensor.close()