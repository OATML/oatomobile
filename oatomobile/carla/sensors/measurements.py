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
"""CARLA vehicle measurements sensors."""

from typing import Any
from typing import Callable

import carla  # pylint: disable=import-error
import dm_env
import numpy as np

from oatomobile.carla import utils as cutils
from oatomobile.carla.sensors import base
from oatomobile.carla.sensors import registry

Measurement = Any


def make_measurement_sensor(
    name: str,
    measurement_fn: Callable[[carla.Actor], Measurement],
    transform_fn: Callable[[Measurement], np.ndarray],
    observation_spec: dm_env.specs.Array,
):
  """Generates CARLA measurement sensors."""

  @registry.add(name=name)
  class MeasurementSensor(base.Sensor):
    """CARLA measurement sensor."""

    def __init__(
        self,
        hero: carla.ActorBlueprint,  # pylint: disable=no-member
    ) -> None:
      """Initializes a `MeasurementSensor`.

      Args:
        hero: The agent to attach the sensor on.
      """
      super().__init__(uuid=name)

      # Internalize components.
      self._hero = hero

    def observation_spec(self) -> dm_env.specs.BoundedArray:
      """Returns the observation spec of the sensor."""
      return observation_spec

    def get_observation(self, *args: Any, **kwargs: Any) -> np.ndarray:
      """Collects the measurement of the ego vehicle.

      Returns:
        An array representation of the CARLA vehicle measurement.
      """
      measurement = measurement_fn(self._hero)
      return transform_fn(measurement)

    @classmethod
    def default(
        cls,
        hero: carla.ActorBlueprint,  # pylint: disable=no-member
        *args,
        **kwargs,
    ) -> "MeasurementSensor":
      """Returns the default sensor.

      Args:
        hero: The agent to attach the sensor on.
      """
      return cls(hero=hero)

  return MeasurementSensor


# CARLA vehicle control sensor.
ControlSensor = make_measurement_sensor(
    name="control",
    measurement_fn=lambda hero: hero.get_control(),
    transform_fn=cutils.carla_control_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=0.0,
        maximum=1.0,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle location sensor.
LocationSensor = make_measurement_sensor(
    name="location",
    measurement_fn=lambda hero: hero.get_transform().location,
    transform_fn=cutils.carla_xyz_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=-np.inf,
        maximum=np.inf,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle rotation sensor.
RotationSensor = make_measurement_sensor(
    name="rotation",
    measurement_fn=lambda hero: hero.get_transform().rotation,
    transform_fn=cutils.carla_rotation_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=0.0,
        maximum=360.0,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle velocity sensor.
VelocitySensor = make_measurement_sensor(
    name="velocity",
    measurement_fn=lambda hero: hero.get_velocity(),
    transform_fn=cutils.carla_xyz_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=-np.inf,
        maximum=np.inf,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle acceleration sensor.
AccelerationSensor = make_measurement_sensor(
    name="acceleration",
    measurement_fn=lambda hero: hero.get_acceleration(),
    transform_fn=cutils.carla_xyz_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=-np.inf,
        maximum=np.inf,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle orientation sensor.
OrientationSensor = make_measurement_sensor(
    name="orientation",
    measurement_fn=lambda hero: hero.get_transform().get_forward_vector(),
    transform_fn=cutils.carla_xyz_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=-np.inf,
        maximum=np.inf,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle angular velocity sensor.
AngularVelocitySensor = make_measurement_sensor(
    name="angular_velocity",
    measurement_fn=lambda hero: hero.get_angular_velocity(),
    transform_fn=cutils.carla_xyz_to_ndarray,
    observation_spec=dm_env.specs.BoundedArray(
        minimum=-np.inf,
        maximum=np.inf,
        shape=(3,),
        dtype=np.float32,
    ),
)

# CARLA vehicle speed limit sensor.
SpeedLimitSensor = make_measurement_sensor(
    name="speed_limit",
    measurement_fn=lambda hero: hero.get_speed_limit(),
    transform_fn=lambda speed_limit: np.asarray(speed_limit, dtype=np.float32),
    observation_spec=dm_env.specs.BoundedArray(
        minimum=0.0,
        maximum=np.inf,
        shape=(1,),
        dtype=np.float32,
    ),
)

# CARLA vehicle is at traffic light sensor.
IsAtTrafficLightSensor = make_measurement_sensor(
    name="is_at_traffic_light",
    measurement_fn=lambda hero: hero.is_at_traffic_light(),
    transform_fn=np.int32,
    observation_spec=dm_env.specs.DiscreteArray(num_values=2, dtype=np.int32),
)

# CARLA vehicle traffic light state sensor.
TrafficLightStateSensor = make_measurement_sensor(
    name="traffic_light_state",
    measurement_fn=lambda hero: hero.get_traffic_light_state(),
    transform_fn=np.int32,
    observation_spec=dm_env.specs.DiscreteArray(num_values=4, dtype=np.int32),
)
