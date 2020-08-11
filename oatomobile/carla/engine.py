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
"""CARLA `Simulator` implementation and helper classes."""

import atexit
import os
import signal
import subprocess
from typing import Any
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union

import carla  # pylint: disable=import-error
import dm_env
import numpy as np
from absl import logging
from acme.types import NestedSpec
from acme.types import NestedTensor

from oatomobile import core
from oatomobile.carla import constants
from oatomobile.carla import types
from oatomobile.carla import utils as cutils
from oatomobile.carla.sensors import registry
from oatomobile.carla.sensors.base import SensorSuite


class Actors(NamedTuple):
  """CARLA actors, i.e., hero, others and pedestrians."""

  hero: carla.Vehicle  # pylint: disable=no-member
  others: Sequence[carla.Vehicle]  # pylint: disable=no-member
  pedestrians: Sequence[carla.Walker]  # pylint: disable=no-member


class Data(NamedTuple):
  """CARLA data, i.e., actors, client, world, server."""

  client: carla.Client  # pylint: disable=no-member
  world: carla.World  # pylint: disable=no-member
  server_process: subprocess.Popen
  actors: Actors
  frame: int
  frame0: int


class Simulator(core.Simulator):
  """Encapsulates a CARLA simulator."""

  def __init__(
      self,
      fps: int = constants.SIMULATOR_FPS,
      client_timeout: float = constants.CLIENT_TIMEOUT,
  ):
    """Constructs a CARLA `Simulator`.

    Args:
      fps: The frequency (in Hz) of the simulation.
      client_timeout: The time interval before stopping
        the search for the carla server.
    """

    # Internalize components.
    self._fps = fps
    self._client_timeout = client_timeout

    # CARLA objects (lazy initialization).
    self._data = Data(
        client=None,
        server_process=None,
        world=None,
        actors=None,
        frame0=None,
        frame=None,
    )

  def step(self, num_sub_steps: int = 1) -> None:
    """Updates the simulation state.

    Args:
      num_sub_steps: Optional number of times to repeatedly update the simulation
        state. Defaults to 1.
    """
    for _ in range(num_sub_steps):
      # Advance the simulation by a time step.
      self._frame = self._data.world.tick()

  def time(self) -> float:
    """Returns the elapsed simulation time in seconds."""
    if self._frame is not None:
      return np.float32((self._frame - self._data.frame0) * self.timestep())

  def timestep(self) -> float:
    """Returns the simulation timestep."""
    if self._data.world is not None:
      return np.float32(self._data.world.get_snapshot().delta_seconds)
    else:
      return np.float32(0.0)

  def set_control(self, control: types.Action) -> None:
    """Sets the control signal for the vehicle."""
    if self._data.actors is not None:
      self._data.actors.hero.apply_control(control)
    else:
      logging.warn("`set_control` was applied on a `None` hero.")

  def reset(self) -> None:
    """Resets internal variables of the simulator simulation."""

    # Close any connection before reset.
    self.close()

    # Initialize CARLA client/server pair.
    client, server_process = cutils.setup(
        fps=self._fps,
        client_timeout=self._client_timeout,
    )
    self._data = self._data._replace(
        client=client,
        server_process=server_process,
        world=client.get_world(),
    )

  def after_reset(self):
    """Runs after resetting internal variables of the physics simulation."""

    # HACK(filangel): due to the bug with the lifted vehicle and
    # the LocalPlanner, perform K=50 steps in the simulator.
    for _ in range(50):
      self._data.world.tick()

    # Initial timestamp used to calculate elapsed simulation time.
    self._data = self._data._replace(
        frame0=int(self._data.world.get_snapshot().frame))
    self._data = self._data._replace(frame=self._data.frame0)

  def close(self) -> None:
    """Closes the simulator down and controls connection to the server."""
    if self._data.server_process is not None:
      # Disable CARLA synchronous mode.
      settings = self._data.world.get_settings()
      settings.synchronous_mode = False
      self._data.world.apply_settings(settings)
      logging.debug("Closes the CARLA server with process PID {}".format(
          self._data.server_process.pid))
      # Shut down the server.
      os.killpg(self._data.server_process.pid, signal.SIGKILL)
      atexit.unregister(
          lambda: os.killpg(self._data.server_process.pid, signal.SIGKILL))

    # Reset CARLA object pointers.
    self._data = Data(
        client=None,
        server_process=None,
        world=None,
        actors=None,
        frame0=None,
        frame=None,
    )

  @property
  def data(self) -> Data:
    """Returns a reference to the CARLA data."""
    return self._data

  def set_actors(self, actors: Actors) -> None:
    """Sets the CARLA actors."""
    self._data = self._data._replace(actors=actors)

  def render(self, mode: str = "rgb_array") -> Any:
    """Renders current state of the simulator."""
    raise NotImplementedError()


class Task(core.Task):
  """Base class for tasks in CARLA."""

  def __init__(
      self,
      town: str,
      sensors: Sequence[str] = constants.SENSORS,
      spawn_point: Optional[Union[int, carla.Location]] = None,  # pylint: disable=no-member
      destination: Optional[Union[int, carla.Location]] = None,  # pylint: disable=no-member
      num_vehicles: int = 0,
      num_pedestrians: int = 0,
  ) -> None:
    """Initializes a CARLA `Task`.

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
    """

    # Internalize components.
    self._town = town
    self._sensors = tuple(set(sensors))
    self._destination = destination
    self._spawn_point = spawn_point
    self._num_vehicles = num_vehicles
    self._num_pedestrians = num_pedestrians

    # CARLA objects (lazy initialization).
    self._actors = Actors(hero=None, others=None, pedestrians=None)
    self._sensor_suite = None

  @property
  def actors(self) -> Actors:
    """Returns a refernce to the CARLA actors."""
    return self._actors

  @property
  def sensor_suite(self) -> Optional[SensorSuite]:
    """Returns a refernce to the suite of sensors."""
    return self._sensor_suite

  def initialize_episode(self, simulator: Simulator) -> None:
    """Sets the state of the environment at the start of each episode.

    Called by `control.Environment` at the start of each episode *within*
    `simulator.reset_context()` (see the documentation for `base.Simulator`).

    Args:
      simulator: Instance of `Simulator`.
    """

    # Convert spawn and destination points to CARLA friendly objects.
    self._spawn_point = cutils.get_spawn_point(
        simulator.data.world,
        self._spawn_point,
    )
    self._destination = cutils.get_spawn_point(
        simulator.data.world,
        self._destination,
    )

    # Spawn hero/ego vehicle, others and pedestrians.
    hero = cutils.spawn_hero(
        world=simulator.data.world,
        spawn_point=self._spawn_point,
        vehicle_id="vehicle.ford.mustang",
    )
    others = cutils.spawn_vehicles(
        world=simulator.data.world,
        num_vehicles=self._num_vehicles,
    )
    pedestrians = cutils.spawn_pedestrians(
        world=simulator.data.world,
        num_pedestrians=self._num_pedestrians,
    )

    # Spawn sensors.
    self._sensor_suite = SensorSuite([
        registry.get_sensor(sensor).default(
            hero=hero,
            destination=self._destination,
        ) for sensor in self._sensors
    ])

    # Register actors in `Task` and `Simulator`.
    self._actors = Actors(hero=hero, others=others, pedestrians=pedestrians)
    simulator.set_actors(self._actors)

  def before_step(self, action: types.Action, simulator: Simulator) -> None:
    """Updates the task from the provided action.

    Called by `control.Environment` before stepping the simulator engine.

    Args:
      action: The hero vehicle's actions. Should conform to the specification
        returned by `self.action_spec(simulator)`.
      simulator: Instance of `Simulator`.
    """

    # Perform an `action` in the simulator.
    if action is not None:
      if not isinstance(action, types.Action):
        action = types.Action(**{k: float(v) for (k, v) in action.items()})
      simulator.set_control(action)

  def action_spec(self, simulator: Simulator) -> NestedSpec:
    """Returns a specification describing the valid actions for this task.

    Args:
      simulator: Instance of `Simulator`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the action array(s) passed to `self.step`.
    """
    return dict(
        throttle=dm_env.specs.BoundedArray(
            minimum=0.0,
            maximum=1.0,
            shape=(),
            dtype=np.float32,
        ),
        steer=dm_env.specs.BoundedArray(
            minimum=-1.0,
            maximum=1.0,
            shape=(),
            dtype=np.float32,
        ),
        brake=dm_env.specs.BoundedArray(
            minimum=0.0,
            maximum=1.0,
            shape=(),
            dtype=np.float32,
        ),
    )

  def get_observation(self, simulator: Simulator) -> NestedTensor:
    """Returns an observation from the environment.

    Args:
      simulator: Instance of `Simulator`.
    """
    return self.sensor_suite.get_observations(
        frame=simulator.data.frame,
        timeout=constants.QUEUE_TIMEOUT,
    )

  def get_reward(self, simulator: Simulator) -> float:
    """Returns a reward from the environment.

    Args:
      simulator: Instance of `Simulator`.
    """
    return 0.0

  def get_termination(self, simulator: Simulator) -> bool:
    """If the episode should end, returns True, otherwise False."""
    return False

  def observation_spec(
      self,
      simulator: Simulator,
  ) -> NestedSpec:
    """Returns the observation spec.

    Args:
      simulator: Instance of `Simulator`.

    Returns:
      A dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    if self.sensor_suite is None:
      return None
    else:
      return self.sensor_suite.observation_spec()
