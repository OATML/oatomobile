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
"""Core API to interface with the CARLA simulator."""

import abc
import atexit
import os
import queue
import random
import signal
import time
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import gym
import numpy as np
import pygame
from absl import logging

import carla
from oatomobile.core import simulator
from oatomobile.core.registry import registry
from oatomobile.simulators.carla import defaults
from oatomobile.utils import carla as cutil
from oatomobile.utils import graphics as gutil

# All agents are expected to return the same action type.
CARLAAction = carla.VehicleControl  # pylint: disable=no-member


class CARLASensorTypes(simulator.SensorTypes):
  """Enumeration of types of sensors."""

  FRONT_CAMERA_RGB = 0
  BIRD_VIEW_CAMERA_RGB = 1
  LIDAR = 2
  CONTROL = 3
  LOCATION = 4
  ROTATION = 5
  VELOCITY = 6
  ACCELERATION = 7
  ORIENTATION = 8
  ANGULAR_VELOCITY = 9
  SPEED_LIMIT = 10
  IS_AT_TRAFFIC_LIGHT = 11
  TRAFFIC_LIGHT_STATE = 12
  COLLISION = 13
  LANE_INVASION = 14
  GAME_STATE = 15
  REAR_CAMERA_RGB = 16
  LEFT_CAMERA_RGB = 17
  RIGHT_CAMERA_RGB = 18
  BIRD_VIEW_CAMERA_CITYSCAPES = 19
  RED_LIGHT_INVASION = 20
  ACTORS_TRACKER = 21
  GOAL = 22
  PREDICTIONS = 23


class CameraSensor(simulator.Sensor, abc.ABC):
  """Abstract class for CARLA camera sensors."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
      *args,
      **kwargs) -> None:
    """Constructs a camera sensor with a dedicated queue."""
    super().__init__(*args, **kwargs)
    self.config = config
    self.sensor = self._spawn_sensor(hero, self.config)  # pylint: disable=no-member
    self.queue = queue.Queue()
    self.sensor.listen(self.put_to_queue)

  def put_to_queue(self, item):
    # logging.warning(f"frame {item.frame} added.")
    self.queue.put(item)

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(
            int(self.config["attributes"].get("image_size_y")),
            int(self.config["attributes"].get("image_size_x")),
            3,
        ),
        dtype=np.float32,
    )

  def close(self) -> None:
    """Destroys the RGB camera sensor from the CARLA server."""
    self.sensor.destroy()


class CameraRGBSensor(CameraSensor):
  """Abstract class for CARLA RGB camera sensors."""

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Optional[Mapping[str, str]] = None,
  ) -> carla.ServerSideSensor:  # pylint: disable=no-member
    """Spawns a camera on `hero`.

    Args:
      hero: The agent to attach the camera on.
      config: The attribute-value pairs for the configuration
        of the sensor.

    Returns:
      The spawned a camera sensor.
    """
    return cutil.spawn_camera(hero, config, camera_type="rgb")

  def get_observation(
      self,
      frame: int,
      timeout: float,
  ) -> np.ndarray:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.
      timeout: The interval waited before stopping search
        and raising a TimeoutError.

    Returns:
      A representation of the camera view.
    """
    try:
      while True:
        data = self.queue.get(timeout=timeout)
        # logging.debug(f"frame {data.frame} removed from queue. frame {frame} requested.")

        # Confirms synced frames.
        if data.frame == frame:
          # logging.debug(f"received/removed synced frame {data.frame} from queue.")
          break
      # Processes the raw sensor data to a RGB array.
      return cutil.carla_rgb_image_to_ndarray(data)
    except queue.Empty:
      logging.debug(
          "The queue of {} sensor was empty, returns a random observation".
          format(self.uuid))
      return self.observation_space.sample()


class CameraCityScapesSensor(CameraSensor):
  """Abstract class for CARLA CityScapes camera sensors."""

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Optional[Mapping[str, str]] = None,
  ) -> carla.ServerSideSensor:  # pylint: disable=no-member
    """Spawns a camera on `hero`.

    Args:
      hero: The agent to attach the camera on.
      config: The attribute-value pairs for the configuration
        of the sensor.

    Returns:
      The spawned a camera sensor.
    """
    return cutil.spawn_camera(hero, config, camera_type="semantic_segmentation")

  def get_observation(
      self,
      frame: int,
      timeout: float,
  ) -> np.ndarray:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.
      timeout: The interval waited before stopping search
        and raising a TimeoutError.

    Returns:
      A representation of the camera view.
    """
    try:
      while True:
        data = self.queue.get(timeout=timeout)
        # Confirms synced frames.
        if data.frame == frame:
          break
      # Processes the raw sensor data to a RGB array.
      return cutil.carla_cityscapes_image_to_ndarray(data)
    except queue.Empty:
      logging.debug(
          "The queue of {} sensor was empty, returns a random observation".
          format(self.uuid))
      return self.observation_space.sample()


@registry.register_sensor(name="front_camera_rgb")
class FrontCameraRGBSensor(CameraRGBSensor):
  """Front camera view sensor."""

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "front_camera_rgb"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.FRONT_CAMERA_RGB

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "FrontCameraRGBSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.FRONT_CAMERA_RGB_SENSOR_CONFIG)


@registry.register_sensor(name="rear_camera_rgb")
class RearCameraRGBSensor(CameraRGBSensor):
  """Rear camera view sensor."""

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "rear_camera_rgb"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.REAR_CAMERA_RGB

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "RearCameraRGBSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.REAR_CAMERA_RGB_SENSOR_CONFIG)


@registry.register_sensor(name="left_camera_rgb")
class LeftCameraRGBSensor(CameraRGBSensor):
  """Left camera view sensor."""

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "left_camera_rgb"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.LEFT_CAMERA_RGB

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "LeftCameraRGBSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.LEFT_CAMERA_RGB_SENSOR_CONFIG)


@registry.register_sensor(name="right_camera_rgb")
class RightCameraRGBSensor(CameraRGBSensor):
  """Right camera view sensor."""

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "right_camera_rgb"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.RIGHT_CAMERA_RGB

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "RightCameraRGBSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.RIGHT_CAMERA_RGB_SENSOR_CONFIG)


@registry.register_sensor(name="bird_view_camera_rgb")
class BirdViewCameraRGBSensor(CameraRGBSensor):
  """Bird-view camera sensor."""

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "bird_view_camera_rgb"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.BIRD_VIEW_CAMERA_RGB

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "BirdViewCameraRGBSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.BIRD_VIEW_CAMERA_RGB_SENSOR_CONFIG)


@registry.register_sensor(name="bird_view_camera_cityscapes")
class BirdViewCameraCityScapesSensor(CameraCityScapesSensor):
  """Bird-view CityScapes camera sensor."""

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "bird_view_camera_cityscapes"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.BIRD_VIEW_CAMERA_CITYSCAPES

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "BirdViewCameraCityScapesSensor":
    """Returns the default sensor."""
    return cls(
        hero=hero,
        config=defaults.BIRD_VIEW_CAMERA_CITYSCAPES_SENSOR_CONFIG,
    )


@registry.register_sensor(name="lidar")
class LIDARSensor(simulator.Sensor):
  """LIDAR, overhead sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
      *args,
      **kwargs) -> None:
    """Constructs an overhead LIDAR sensor with a dedicated queue."""
    super().__init__(*args, **kwargs)
    self.config = config
    self.sensor = self._spawn_sensor(hero, self.config)
    self.queue = queue.Queue()
    self.sensor.listen(self.queue.put)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "lidar"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.LIDAR

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(200, 200, 2),
        dtype=np.float32,
    )

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
  ) -> carla.ServerSideSensor:  # pylint: disable=no-member
    """Spawns LIDAR overhead sensor on `hero`.

    Args:
      hero: The agent to attach the LIDAR on.
      config: The attribute-value pairs for the configuration
        of the sensor.

    Returns:
      The spawned LIDAR overhead sensor.
    """
    return cutil.spawn_lidar(hero, config)

  def get_observation(
      self,
      frame: int,
      timeout: float,
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
        data = self.queue.get(timeout=timeout)
        # Confirms synced frames.
        if data.frame == frame:
          break
      # Processes the raw sensor data to a RGB array.
      return cutil.carla_lidar_measurement_to_ndarray(data)
    except queue.Empty:
      logging.debug(
          "The queue of {} sensor was empty, returns a random observation".
          format(self.uuid))
      return self.observation_space.sample()

  def close(self) -> None:
    """Destroys the LIDAR sensor from the CARLA server."""
    self.sensor.destroy()

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "LIDARSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.LIDAR_SENSOR_CONFIG)


@registry.register_sensor(name="control")
class ControlSensor(simulator.Sensor):
  """CARLA vehicle control sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a control sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "control"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.CONTROL

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the control of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle control.
    """
    control = self._hero.get_control()
    return cutil.carla_control_to_ndarray(control)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "ControlSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="location")
class LocationSensor(simulator.Sensor):
  """CARLA vehicle location sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a location sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "location"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.LOCATION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the location of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle location.
    """
    location = self._hero.get_transform().location
    return cutil.carla_xyz_to_ndarray(location)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "LocationSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="rotation")
class RotationSensor(simulator.Sensor):
  """CARLA vehicle rotation sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a rotation sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "rotation"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.ROTATION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=0.0,
        high=360.0,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the rotation of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle rotation.
    """
    rotation = self._hero.get_transform().rotation
    return cutil.carla_rotation_to_ndarray(rotation)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "RotationSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="velocity")
class VelocitySensor(simulator.Sensor):
  """CARLA vehicle velocity sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a velocity sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "velocity"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.VELOCITY

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the velocity of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle velocity.
    """
    velocity = self._hero.get_velocity()
    return cutil.carla_xyz_to_ndarray(velocity)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "VelocitySensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="acceleration")
class AccelerationSensor(simulator.Sensor):
  """CARLA vehicle acceleration sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a acceleration sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "acceleration"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.ACCELERATION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the acceleration of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle acceleration.
    """
    acceleration = self._hero.get_acceleration()
    return cutil.carla_xyz_to_ndarray(acceleration)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "AccelerationSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="orientation")
class OrientationSensor(simulator.Sensor):
  """CARLA vehicle orientation sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a orientation sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "orientation"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.ORIENTATION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the orientation of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle orientation.
    """
    orientation = self._hero.get_transform().get_forward_vector()
    return cutil.carla_xyz_to_ndarray(orientation)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "OrientationSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="angular_velocity")
class AngularVelocitySensor(simulator.Sensor):
  """CARLA vehicle angular velocity sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a velocity sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "angular_velocity"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.ANGULAR_VELOCITY

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the velocity of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle angular velocity.
    """
    angular_velocity = self._hero.get_angular_velocity()
    return cutil.carla_xyz_to_ndarray(angular_velocity)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "AngularVelocitySensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="speed_limit")
class SpeedLimitSensor(simulator.Sensor):
  """CARLA vehicle speed limit sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a speed limit sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "speed_limit"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.SPEED_LIMIT

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(3,),
        dtype=np.float32,
    )

  def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
    """Collects the speed limit of the ego vehicle.

    Returns:
      An array representation of the CARLA vehicle speed limit.
    """
    speed_limit = self._hero.get_speed_limit()
    return np.asarray(
        speed_limit,
        dtype=np.float32,
    )

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "SpeedLimitSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="is_at_traffic_light")
class IsAtTrafficLightSensor(simulator.Sensor):
  """CARLA sensor for detecting proximity from a traffic light."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a speed limit sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "is_at_traffic_light"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.IS_AT_TRAFFIC_LIGHT

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Discrete:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Discrete(n=2)

  def get_observation(self, *args: Any, **kwargs: Any) -> int:
    """Returns True if close to a traffic light.

    Returns:
      An array representation of the proximity to a traffic light.
    """
    is_at_traffic_light = self._hero.is_at_traffic_light()
    return int(is_at_traffic_light)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "IsAtTrafficLightSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="traffic_light_state")
class TrafficLightStateSensor(simulator.Sensor):
  """CARLA sensor for parsing the traffic light state."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a traffic light state sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "traffic_light_state"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.TRAFFIC_LIGHT_STATE

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Discrete:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Discrete(n=4)

  def get_observation(self, *args: Any, **kwargs: Any) -> int:
    """Returns True if close to a traffic light.

    Returns:
      An array representation of the proximity to a traffic light.
    """
    traffic_light_state = self._hero.get_traffic_light_state().conjugate()
    return int(traffic_light_state)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "TrafficLightStateSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="collision")
class CollisionSensor(simulator.Sensor):
  """Collision sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a collision sensor."""
    super().__init__(*args, **kwargs)
    self.sensor = self._spawn_sensor(hero)
    self.queue = queue.Queue()
    self.sensor.listen(self.queue.put)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "collision"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.COLLISION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Discrete(
        n=4)  # {0: no-collision, 1: vehicle, 2: pedestrian, 3: other}

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
  ) -> carla.ServerSideSensor:  # pylint: disable=no-member
    """Spawns collision sensor on `hero`.

    Args:
      hero: The agent to attach the collision sensor.

    Returns:
      The spawned collision sensor.
    """
    return cutil.spawn_collision(hero)

  def get_observation(self, frame: int, **kwargs) -> int:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.

    Returns:
      The collision type:
        0: no collision.
        1: collision with vehicle.
        2: collision with pedestrian.
        3: collision with other actor.
    """
    try:
      for event in self.queue.queue:
        # Confirms synced frames.
        if event.frame == frame:
          if "vehicle" in event.other_actor.type_id:
            return 1
          elif "walker" in event.other_actor.type_id:
            return 2
          else:
            return 3
      # Default return value.
      return 0
    except queue.Empty:
      logging.debug(
          "The queue of {} sensor was empty, returns a random observation".
          format(self.uuid))
      return self.observation_space.sample()

  def close(self) -> None:
    """Destroys the LIDAR sensor from the CARLA server."""
    self.sensor.destroy()

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "CollisionSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="lane_invasion")
class LaneInvasionSensor(simulator.Sensor):
  """Lane invasion sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a lane invasion sensor."""
    super().__init__(*args, **kwargs)
    self.sensor = self._spawn_sensor(hero)
    self.queue = queue.Queue()
    self.sensor.listen(self.queue.put)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "lane_invasion"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.LANE_INVASION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Discrete:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Discrete(n=2)  # {0: no-invasion, 1: lane-invasion}

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
  ) -> carla.ServerSideSensor:  # pylint: disable=no-member
    """Spawns lane invasion sensor on `hero`.

    Args:
      hero: The agent to attach the lane invasion sensor.

    Returns:
      The spawned lane invasion sensor.
    """
    return cutil.spawn_lane_invasion(hero)

  def get_observation(self, frame: int, **kwargs) -> int:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.

    Returns:
      The collision type:
        0: no invasion.
        1: lane invasion.
    """
    try:
      for event in self.queue.queue:
        # Confirms synced frames.
        if event.frame == frame:
          return 1
      # Default return value.
      return 0
    except queue.Empty:
      logging.debug(
          "The queue of {} sensor was empty, returns a random observation".
          format(self.uuid))
      return self.observation_space.sample()

  def close(self) -> None:
    """Destroys the lane invasion sensor from the CARLA server."""
    self.sensor.destroy()

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "LaneInvasionSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="actors_tracker")
class ActorsTrackerSensor(simulator.Sensor):
  """Lane invasion sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a lane invasion sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "actors_tracker"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.ACTORS_TRACKER

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Dict:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Dict(
        vehicles=gym.spaces.Dict(),  # id<->location
        pedestrians=gym.spaces.Dict(),  # id<->location
    )

  def get_observation(self, frame: int,
                      **kwargs) -> Mapping[str, Mapping[str, np.ndarray]]:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.

    Returns:
      The locations of the vehicles and pedestrians.
    """
    del frame  # Unused arg

    # Fetches all the vehicles and pedestrians in the map.
    world = self._hero.get_world()
    actors = world.get_actors()
    vehicles = actors.filter("vehicle.*")
    pedestrians = actors.filter("walker.pedestrian.*")

    # Output container.
    observation = dict(vehicles={}, pedestrians={})
    for actor_type, actors in [
        ("vehicles", vehicles),
        ("pedestrians", pedestrians),
    ]:
      for actor in actors:
        if actor.attributes["role_name"] is not "hero":
          observation[actor_type][actor.id] = cutil.carla_xyz_to_ndarray(
              actor.get_location())

    return observation

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "ActorsTrackerSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="goal")
class GoalSensor(simulator.Sensor):
  """Goal sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
      destination: carla.Waypoint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a goal sensor."""
    super().__init__(*args, **kwargs)
    self.config = config
    self.destination = destination
    self._hero = hero

    # Parses hyperparameters.
    self._num_goals = self.config["num_goals"]
    self._sampling_radius = self.config["sampling_radius"]
    self._replan_every_steps = self.config["replan_every_steps"]

    # Stored goal.
    self._goal = None
    self._num_steps = 0

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "goal"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.GOAL

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(20, 3),
        dtype=np.float32,
    )

  def get_observation(self, frame: int,
                      **kwargs) -> Mapping[str, Mapping[str, np.ndarray]]:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.

    Returns:
      The ego-locations of the goal(s).
    """
    del frame  # Unused arg

    # Fetches hero measurements for the coordinate transformations.
    hero_transform = self._hero.get_transform()

    if self._goal is None or self._num_steps % int(self._replan_every_steps) == 0:
      # References to CARLA objects.
      carla_world = self._hero.get_world()
      carla_map = carla_world.get_map()

      # Fetches start and end waypoints for the A* planner.
      origin = hero_transform.location
      start_waypoint = carla_map.get_waypoint(origin)
      end_waypoint = carla_map.get_waypoint(self.destination.location)

      # Caclulates global plan.
      waypoints, _, _ = cutil.global_plan(
          world=carla_world,
          origin=start_waypoint.transform.location,
          destination=end_waypoint.transform.location,
      )

      # Samples goals.
      goals_world = [waypoints[0]]
      for _ in range(int(self._num_goals) - 1):
        goals_world.append(goals_world[-1].next(self._sampling_radius)[0])

      # Converts goals to `NumPy` arrays.
      self._goal = np.asarray([
          cutil.carla_xyz_to_ndarray(waypoint.transform.location)
          for waypoint in goals_world
      ])

    # Converts goals to ego coordinates.
    current_location = cutil.carla_xyz_to_ndarray(hero_transform.location)
    current_rotation = cutil.carla_rotation_to_ndarray(hero_transform.rotation)
    goals_local = cutil.world2local(
        current_location=current_location,
        current_rotation=current_rotation,
        world_locations=self._goal,
    )

    # Increments counter, bookkeping.
    self._num_steps += 1

    return goals_local.astype(np.float32)

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "GoalSensor":
    """Returns the default sensor."""
    if "destination" not in kwargs:
      raise ValueError("missing `destination` argument")
    return cls(
        hero=hero,
        config=defaults.GOAL_SENSOR_CONFIG,
        destination=kwargs.get("destination"),
    )


@registry.register_sensor(name="predictions")
class PredictionsSensor(simulator.Sensor):
  """Goal sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a goal sensor."""
    super().__init__(*args, **kwargs)
    self._hero = hero

    # Stored predictions.
    self._predictions = None

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "predictions"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.PREDICTIONS

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4, 2),
        dtype=np.float32,
    )

  @property
  def predictions(self) -> np.ndarray:
    """Returns predictions from PREVIOUS timestep."""
    return self._predictions

  @predictions.setter
  def predictions(self, value: np.ndarray) -> None:
    """Records the predictions for the sensor."""
    self._predictions = value

  def get_observation(self, frame: int,
                      **kwargs) -> Mapping[str, Mapping[str, np.ndarray]]:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.

    Returns:
      The ego-locations of the goal(s).
    """
    del frame  # Unused arg

    return self.predictions

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "PredictionsSensor":
    """Returns the default sensor."""
    return cls(hero=hero)


@registry.register_sensor(name="red_light_invasion")
class RedLightInvasion(simulator.Sensor):
  """Red light invasion sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> None:
    """Constructs a lane invasion sensor."""
    super().__init__(*args, **kwargs)
    self._hero = self._spawn_sensor(hero)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "red_light_invasion"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.RED_LIGHT_INVASION

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Discrete(n=2)  # {0: no-invasion, 1: red-light-invasion}

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
  ) -> None:  # pylint: disable=no-member
    """Dummy call, to satisfy interface."""
    return hero

  def get_observation(self, frame: int, **kwargs) -> int:
    """Finds the data point that matches the current `frame` id.

    Args:
      frame: The synchronous simulation time step ID.

    Returns:
      The invasion type:
        0: no invasion.
        1: red traffi light invasion.
    """
    del frame

    # Skips calculations if not necessary.
    if not self._hero.is_at_traffic_light():
      return 0

    raise NotImplementedError

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "RedLightInvasion":
    """Returns the default sensor."""
    return cls(hero=hero, *args, **kwargs)


@registry.register_sensor(name="game_state")
class GameStateSensor(simulator.Sensor):
  """Game state sensor."""

  def __init__(
      self,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
      *args,
      **kwargs) -> None:
    """Constructs a lane invasion sensor."""
    super().__init__(*args, **kwargs)
    self.config = config
    self.hero = hero
    # Static surfaces.
    self.road_mask, self.lane_boundaries_mask = self._spawn_sensor(
        self.hero,
        self.config,
    )

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the sensor."""
    return "game_state"

  def _get_sensor_type(self, *args: Any, **kwargs: Any) -> CARLASensorTypes:
    """Returns the the type of the sensor."""
    return CARLASensorTypes.GAME_STATE

  @property
  def observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Box:
    """Returns the observation spec of the sensor."""
    return gym.spaces.Box(
        low=0,
        high=1,
        shape=(*self.road_mask.shape, 8),
        dtype=int,
    )

  @staticmethod
  def _spawn_sensor(
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      config: Mapping[str, str],
  ) -> Tuple[np.ndarray, np.ndarray]:  # pylint: disable=no-member
    """Spawns lane invasion sensor on `hero`.

    Args:
      hero: The ego vehicle agent.
      config: The attribute-value pairs for the configuration
        of the sensor.

    Returns:
      road_mask: The road game state.
      lane_boundaries_mask: The lane boundaries of the game state.
    """
    # Fetch CARLA world from ego vehicle.
    world = hero.get_world()

    # Fetch static surfaces.
    road_surface = gutil.get_road_surface(world, **config)
    lane_boundaries_surface = gutil.get_lane_boundaries_surface(world, **config)

    # Convert `PyGame` surfaces to binary tensors.
    road_ndarray = gutil.pygame_surface_to_ndarray(road_surface)
    road_mask = gutil.rgb_to_binary_mask(road_ndarray)
    lane_boundaries_ndarray = gutil.pygame_surface_to_ndarray(
        lane_boundaries_surface)
    lane_boundaries_mask = gutil.rgb_to_binary_mask(lane_boundaries_ndarray)

    return road_mask, lane_boundaries_mask

  def get_observation(self, **kwargs) -> np.ndarray:
    """Gets a snapshot of the game state.

    Returns:
      A multi-channel mask for the game state.
    """
    # Fetch CARLA world from ego vehicle.
    world = self.hero.get_world()

    # Fetch dynamic surfaces.
    vehicles_surface = gutil.get_vehicles_surface(world=world, **self.config)
    pedestrians_surface = gutil.get_pedestrians_surface(world=world,
                                                        **self.config)
    green_surface, yellow_surface, red_surface = gutil.get_traffic_lights_surface(
        world=world, **self.config)
    hero_surface = gutil.get_hero_surface(world=world,
                                          hero=self.hero,
                                          **self.config)

    # Convert `PyGame` surfaces to binary tensors.
    vehicles_ndarray = gutil.pygame_surface_to_ndarray(vehicles_surface)
    vehicles_mask = gutil.rgb_to_binary_mask(vehicles_ndarray)
    pedestrians_ndarray = gutil.pygame_surface_to_ndarray(pedestrians_surface)
    pedestrians_mask = gutil.rgb_to_binary_mask(pedestrians_ndarray)
    green_ndarray = gutil.pygame_surface_to_ndarray(green_surface)
    green_mask = gutil.rgb_to_binary_mask(green_ndarray)
    yellow_ndarray = gutil.pygame_surface_to_ndarray(yellow_surface)
    yellow_mask = gutil.rgb_to_binary_mask(yellow_ndarray)
    red_ndarray = gutil.pygame_surface_to_ndarray(red_surface)
    red_mask = gutil.rgb_to_binary_mask(red_ndarray)
    hero_ndarray = gutil.pygame_surface_to_ndarray(hero_surface)
    hero_mask = gutil.rgb_to_binary_mask(hero_ndarray)

    return np.c_[self.road_mask, self.lane_boundaries_mask, vehicles_mask,
                 pedestrians_mask, green_mask, yellow_mask, red_mask, hero_mask]

  def close(self) -> None:
    """Dummy call, to satisfy interface."""
    pass

  @classmethod
  def default(
      cls,
      hero: carla.ActorBlueprint,  # pylint: disable=no-member
      *args,
      **kwargs) -> "GameStateSensor":
    """Returns the default sensor."""
    return cls(hero=hero, config=defaults.GAME_STATE_CONFIG)


@registry.register_simulator(name="carla")
class CARLASimulator(simulator.Simulator):
  """A thin CARLA simulator wrapper."""

  def __init__(
      self,
      town: str,
      sensors: Sequence[str] = defaults.CARLA_SENSORS,
      spawn_point: Optional[Union[int, carla.Location]] = None,  # pylint: disable=no-member
      destination: Optional[Union[int, carla.Location]] = None,  # pylint: disable=no-member
      num_vehicles: int = 0,
      num_pedestrians: int = 0,
      fps: int = defaults.SIMULATOR_FPS,
      client_timeout: float = defaults.CARLA_CLIENT_TIMEOUT,
  ) -> None:
    """Constructs a CARLA simulator wrapper.

    Args:
      town: The `CARLA` town identifier.
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
      fps: The frequency (in Hz) of the simulation.
      client_timeout: The time interval before stopping
        the search for the carla server.
    """
    # Configuration variables.
    self._town = town
    self._sensors = sensors
    self._fps = fps
    self._client_timeout = client_timeout
    self._num_vehicles = num_vehicles
    self._num_pedestrians = num_pedestrians

    # CARLA objects (lazy initialization).
    self._client = None
    self._world = None
    self._frame = None
    self._server = None
    self._traffic_manager = None
    self._frame0 = None
    self._dt = None
    self._vehicles = None
    self._pedestrians = None
    self._sensor_suite = None
    self._hero = None
    self._destination = destination
    self._spawn_point = spawn_point

    # Randomness controller.
    self._np_random = np.random.RandomState(None)  # pylint: disable=no-member

    # State of the game.
    self._observations = None
    self._time_elapsed = 0.0

    # Graphics setup.
    self._display = None
    self._clock = None

  @property
  def hero(self) -> carla.Vehicle:  # pylint: disable=no-member
    """Returns a reference to the ego car."""
    return self._hero

  @property
  def spawn_point(self) -> carla.Waypoint:  # pylint: disable=no-member
    """Returns a reference to the spawn point."""
    if self._world is None:
      raise ValueError("Make sure the environment is reset first.")
    return cutil.get_spawn_point(self._world, self._spawn_point)

  @property
  def destination(self) -> carla.Waypoint:  # pylint: disable=no-member
    """Returns a reference to the destination."""
    if self._world is None:
      raise ValueError("Make sure the environment is reset first.")
    return cutil.get_spawn_point(self._world, self._destination)

  @property
  def sensor_suite(self) -> simulator.SensorSuite:
    """Returns a refernce to the suite of sensors."""
    return self._sensor_suite

  @property
  def action_space(self) -> Any:
    """Returns the specification of the actions expected by the simulator."""
    return Optional[simulator.Action]

  @property
  def observation_space(self) -> Optional[gym.spaces.Dict]:
    """Returns the specification of the observations returned by the
    simulator."""
    if self.sensor_suite is None:
      return None
    else:
      return self.sensor_suite.observation_space

  def seed(self, seed: int) -> None:
    """Fixes the random number generator state."""
    random.seed(seed)
    self._np_random = np.random.RandomState(seed)  # pylint: disable=no-member

  def reset(self) -> simulator.Observations:
    """Resets the state of the simulation to the initial state.

    Returns:
      The initial observations.
    """
    # CARLA setup.
    self._client, self._world, self._frame, self._server, self._traffic_manager = cutil.setup(
        town=self._town,
        fps=self._fps,
        client_timeout=self._client_timeout,
    )
    self._frame0 = int(self._frame)
    self._dt = self._world.get_settings().fixed_delta_seconds

    # Initializes hero agent.
    self._hero = cutil.spawn_hero(
        world=self._world,
        spawn_point=self.spawn_point,
        vehicle_id="vehicle.tesla.model3",
    )
    # Initializes the other vehicles.
    self._vehicles, self._pedestrians = cutil.spawn_vehicles_and_pedestrians(
        world=self._world,
        client=self._client,
        traffic_manager=self._traffic_manager,
        num_vehicles=self._num_vehicles,
        num_pedestrians=self._num_pedestrians
    )
    # Registers the sensors.
    self._sensor_suite = simulator.SensorSuite([
        registry.get_sensor(sensor).default(
            hero=self.hero,
            destination=self.destination,
        ) for sensor in self._sensors
    ])

    # HACK(filangel): due to the bug with the lifted vehicle and
    # the LocalPlanner, perform K=50 steps in the simulator.
    for _ in range(50):
      obs = self.step(action=None)

    return obs

  def step(self, action: simulator.Action) -> simulator.Observations:
    """Makes a step in the simulator, provided an action.

    Args:
      action: The hero vehicle's actions.

    Returns:
      The current set of observations from the sensors
      after the step from the `action`.
    """
    # Perform an `action` in the world.
    if action is not None:
      if not isinstance(action, CARLAAction):
        action = CARLAAction(**{k: float(v) for (k, v) in action.items()})
      self.hero.apply_control(action)

    # Advance the simulation by a time step.
    self._frame = self._world.tick()
    self._time_elapsed = np.asarray(
        (self._frame - self._frame0) * self._dt,
        dtype=np.float32,
    )
    if self._clock is not None:
      self._clock.tick()

    # Retrieves observations from registered sensors.
    self._observations = self._sensor_suite.get_observations(
        frame=self._frame,
        timeout=defaults.QUEUE_TIMEOUT,
    )

    return self._observations

  def render(self, mode: str = "human", **kwargs) -> Any:
    """Renders current state of the simulator."""
    if mode not in ("human", "rgb_array"):
      raise ValueError("Unrecognised mode value {} passed.".format(mode))

    if self._display is None or self._clock is None:
      # TODO(filangel): clean this up
      width = 0
      if "left_camera_rgb" in self._observations:
        width = width + 320
        height = 180
      if "front_camera_rgb" in self._observations:
        width = width + 320
        height = 180
      if "rear_camera_rgb" in self._observations:
        width = width + 320
        height = 180
      if "right_camera_rgb" in self._observations:
        width = width + 320
        height = 180
      if "lidar" in self._observations:
        width = width + 200
        height = 200
      if "bird_view_camera_rgb" in self._observations:
        width = width + 200
        height = 200
      if "bird_view_camera_cityscapes" in self._observations:
        width = width + 200
        height = 200
      self._display, self._clock, self._font = gutil.setup(
          width=width,
          height=height,
          render=mode == "human",
      )
    # It mutates the `self._display` `PyGame` object.
    gutil.make_dashboard(
        display=self._display,
        font=self._font,
        clock=self._clock,
        **{
            **self._observations,
            **kwargs
        },
    )

    if mode == "human":
      # Update window display.
      pygame.display.flip()
    elif mode == "rgb_array":
      # Converts surface to an RGB tensor.
      return gutil.pygame_surface_to_ndarray(self._display)

  def close(self) -> None:
    """Closes the simulator down and controls connection to CARLA server."""
    if self.sensor_suite is not None:
      self.sensor_suite.close()
      self._sensor_suite = None
    settings = self._world.get_settings()
    settings.synchronous_mode = False
    self._world.apply_settings(settings)
    logging.debug("Closes the CARLA server with process PID {}".format(
        self._server.pid))
    os.killpg(self._server.pid, signal.SIGKILL)
    atexit.unregister(lambda: os.killpg(self._server.pid, signal.SIGKILL))
