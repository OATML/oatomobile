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
"""CARLA LIDAR sensor."""

import queue
from typing import Any
from typing import Mapping

import carla  # pylint: disable=import-error
import dm_env
import numpy as np
from absl import logging

from oatomobile.carla import constants
from oatomobile.carla import utils as cutils
from oatomobile.carla.sensors import base
from oatomobile.carla.sensors import registry

# Default LIDAR sensor configuration.
_LIDAR_SENSOR_CONFIG = {
    "attributes": {
        "range": "5000",
        "points_per_second": str(constants.SIMULATOR_FPS * 10000),
        "rotation_frequency": str(constants.SIMULATOR_FPS),
        "upper_fov": "10",
        "lower_fov": "-30",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 2.5,
        },
    },
}


@registry.add(name="lidar")
class LIDARSensor(base.Sensor):
  """LIDAR overhead sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
  ) -> None:
    """Initializes a `LIDARSensor`.

    Args:
      hero: The CARLA actor where the sensor gets mounted.
      config: The sensor configuration file.
    """
    super().__init__(uuid="lidar")

    # Internalize components.
    self._hero = hero
    self._config = config

    # Initialize server-side sensor.
    self._sensor = self._spawn_sensor(self._hero, self._config)
    self._queue = queue.Queue()
    self._sensor.listen(self._queue.put)

  def observation_spec(self) -> dm_env.specs.BoundedArray:
    """Returns the observation spec of the sensor."""
    return dm_env.specs.BoundedArray(
        minimum=0.0,
        maximum=1.0,
        shape=(200, 200, 2),
        dtype=np.float32,
    )

  def get_observation(
      self,
      frame: int,
      timeout: float,
      *args: Any,
      **kwargs: Any,
  ) -> np.ndarray:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.
      timeout: The interval waited before stopping search
        and raising a TimeoutError.

    Returns:
      An array representation of the LIDAR point cloud.
    """
    try:
      while True:
        data = self._queue.get(timeout=timeout)
        # Confirms synced frames.
        if data.frame == frame:
          break
      # Processes the raw sensor data to a RGB array.
      return cutils.carla_lidar_measurement_to_ndarray(data)
    except queue.Empty:
      logging.debug(
          "The queue of {} sensor was empty, returns a random observation".
          format("LIDARSensor"))
      return self.observation_spec().generate_value()

  def close(self) -> None:
    """Destroys the LIDAR sensor from the CARLA server."""
    self._sensor.destroy()

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs,
  ) -> "LIDARSensor":
    """Returns the default sensor.

    Args:
      hero: The agent to attach the LIDAR on.
    """
    return cls(hero=hero, config=_LIDAR_SENSOR_CONFIG)

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
  ) -> carla.ServerSideSensor:  # pylint: disable=no-member
    """Spawns LIDAR overhead sensor on `hero`.

    Args:
      hero: The agent to attach the sensor on.
      config: The attribute-value pairs for the configuration
        of the sensor.

    Returns:
      The spawned LIDAR overhead sensor.
    """
    return cutils.spawn_lidar(hero, config)