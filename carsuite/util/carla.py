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
"""CARLA utility functions and wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import collections
import os
import random
import signal
import subprocess
import sys
import time
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import carla
import numpy as np
import transforms3d.euler
from absl import logging


def setup(
    town: str,
    fps: int = 20,
    server_timestop: float = 20.0,
    client_timeout: float = 0.0,
    num_max_restarts: int = 5,
) -> Tuple[carla.Client, carla.World, int, subprocess.Popen]:  # pylint: disable=no-member
  """Returns the `CARLA` `server`, `client` and `world`.

  Args:
    town: The `CARLA` town identifier.
    fps: The frequency (in Hz) of the simulation.
    server_timestop: The time interval between spawing the server
      and resuming program.
    client_timeout: The time interval before stopping
      the search for the carla server.
    num_max_restarts: Number of attempts to connect to the server.

  Returns:
    client: The `CARLA` client.
    world: The `CARLA` world.
    frame: The synchronous simulation time step ID.
    server: The `CARLA` server.
  """
  assert town in ("Town01", "Town02", "Town03", "Town04", "Town05")

  # The attempts counter.
  attempts = 0

  while attempts < num_max_restarts:
    logging.debug("{} out of {} attempts to setup the CARLA simulator".format(
        attempts + 1, num_max_restarts))

    # Random assignment of port.
    port = np.random.randint(2000, 3000)

    # Start CARLA server.
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "offscreen"
    env["SDL_HINT_CUDA_DEVICE"] = "0"
    logging.debug("Inits a CARLA server at port={}".format(port))
    server = subprocess.Popen(
        [
            os.path.join(os.environ.get("CARLA_ROOT"), "CarlaUE4.sh"),
            "-carla-rpc-port={}".format(port),
            "-quality-level=Epic",
        ],
        stdout=None,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    atexit.register(os.killpg, server.pid, signal.SIGKILL)
    time.sleep(server_timestop)

    # Connect client.
    logging.debug("Connects a CARLA client at port={}".format(port))
    try:
      client = carla.Client("localhost", port)  # pylint: disable=no-member
      client.set_timeout(client_timeout)
      client.load_world(map_name=town)
      world = client.get_world()
      world.set_weather(carla.WeatherParameters.ClearNoon)  # pylint: disable=no-member
      frame = world.apply_settings(
          carla.WorldSettings(  # pylint: disable=no-member
              no_rendering_mode=False,
              synchronous_mode=True,
              fixed_delta_seconds=1.0 / fps,
          ))
      logging.debug("Server version: {}".format(client.get_server_version()))
      logging.debug("Client version: {}".format(client.get_client_version()))
      return client, world, frame, server
    except RuntimeError as msg:
      logging.debug(msg)
      attempts += 1
      logging.debug("Stopping CARLA server at port={}".format(port))
      os.killpg(server.pid, signal.SIGKILL)
      atexit.unregister(lambda: os.killpg(server.pid, signal.SIGKILL))

  logging.debug(
      "Failed to connect to CARLA after {} attempts".format(num_max_restarts))
  sys.exit()


def carla_rgb_image_to_ndarray(image: carla.Image) -> np.ndarray:  # pylint: disable=no-member
  """Returns a `NumPy` array from a `CARLA` RGB image.

  Args:
    image: The `CARLA` RGB image.

  Returns:
    A `NumPy` array representation of the image.
  """
  image.convert(carla.ColorConverter.Raw)  # pylint: disable=no-member
  array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  array = array.astype(np.float32) / 255
  array = np.reshape(array, (image.height, image.width, 4))
  array = array[:, :, :3]
  array = array[:, :, ::-1]
  return array


def carla_cityscapes_image_to_ndarray(image: carla.Image) -> np.ndarray:  # pylint: disable=no-member
  """Returns a `NumPy` array from a `CARLA` semantic segmentation image.

  Args:
    image: The `CARLA` semantic segmented image.

  Returns:
    A `NumPy` array representation of the image.
  """
  image.convert(carla.ColorConverter.CityScapesPalette)  # pylint: disable=no-member
  array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  array = array.astype(np.float32) / 255
  array = np.reshape(array, (image.height, image.width, 4))
  array = array[:, :, :3]
  array = array[:, :, ::-1]
  return array


def carla_lidar_measurement_to_ndarray(
    lidar_measurement: carla.LidarMeasurement,  # pylint: disable=no-member
    pixels_per_meter: int = 2,
    hist_max_per_pixel: int = 5,
    meters_max: int = 50,
) -> np.ndarray:
  """Returns a `NumPy` array from a `CARLA` LIDAR point cloud.

  Args:
    lidar_measurement: The `CARLA` LIDAR point cloud.

  Returns:
    A `NumPy` array representation of the point cloud.
  """

  def splat_points(
      point_cloud,
      pixels_per_meter: int,
      hist_max_per_pixel: int,
      meters_max: int,
  ):
    """Converts point cloud to 2D histograms."""
    # Allocate 2D histogram bins.
    ymeters_max = meters_max
    xbins = np.linspace(
        -meters_max,
        meters_max + 1,
        meters_max * 2 * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        -meters_max,
        ymeters_max + 1,
        ymeters_max * 2 * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat

  # Serialise and parse to `NumPy` tensor.
  points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype("f4"))
  points = np.reshape(points, (int(points.shape[0] / 3), 3))

  # Split observations in the Z dimension (height).
  below = points[points[..., 2] <= -2.5]
  above = points[points[..., 2] >= -2.5]
  # Convert point clouds to 2D histograms.
  features = list()
  features.append(
      splat_points(
          below,
          pixels_per_meter,
          hist_max_per_pixel,
          meters_max,
      ))
  features.append(
      splat_points(
          above,
          pixels_per_meter,
          hist_max_per_pixel,
          meters_max,
      ))
  features = np.stack(features, axis=-1)

  return features.astype(np.float32)


def spawn_hero(
    world: carla.World,  # pylint: disable=no-member
    spawn_point: carla.Transform,  # pylint: disable=no-member
    vehicle_id: Optional[str] = None,
) -> carla.Vehicle:  # pylint: disable=no-member
  """Spawns `hero` in `spawn_point`.

  Args:
    world: The world object associated with the simulation.
    spawn_point: The point to spawn the hero actor.
    vehicle_id: An (optional) valid id for the vehicle object.

  Returns:
    The actor (vehicle) object.
  """
  # Blueprints library.
  bl = world.get_blueprint_library()
  if vehicle_id is not None:
    # Get the specific vehicle from the library.
    hero_bp = bl.find(vehicle_id)
  else:
    # Randomly choose a vehicle from the list.
    hero_bp = random.choice(bl.filter("vehicle.*"))
  # Rename the actor to `hero`.
  hero_bp.set_attribute("role_name", "hero")
  logging.debug("Spawns hero actor at {}".format(
      carla_xyz_to_ndarray(spawn_point.location)))
  hero = world.spawn_actor(hero_bp, spawn_point)
  return hero


def spawn_vehicles(
    world: carla.World,  # pylint: disable=no-member
    num_vehicles: int,
) -> Sequence[carla.Vehicle]:  # pylint: disable=no-member
  """Spawns `vehicles` randomly in spawn points.

  Args:
    world: The world object associated with the simulation.
    num_vehicles: The number of vehicles to spawn.

  Returns:
    The list of vehicles actors.
  """
  # Blueprints library.
  bl = world.get_blueprint_library()
  # List of spawn points.
  spawn_points = world.get_map().get_spawn_points()
  # Output container
  actors = list()
  for _ in range(num_vehicles):
    # Fetch random blueprint.
    vehicle_bp = random.choice(bl.filter("vehicle.*"))
    # Attempt to spawn vehicle in random location.
    actor = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if actor is not None:
      # Enable autopilot.
      actor.set_autopilot(True)
      # Append actor to the list.
      actors.append(actor)
  logging.debug("Spawned {} other vehicles".format(len(actors)))
  return actors


def spawn_pedestrians(
    world: carla.World,  # pylint: disable=no-member
    num_pedestrians: int,
    speeds: Sequence[float] = (1.0, 1.5, 2.0),
) -> Sequence[carla.Vehicle]:  # pylint: disable=no-member
  """Spawns `pedestrians` in random locations.

  Args:
    world: The world object associated with the simulation.
    num_pedestrians: The number of pedestrians to spawn.
    speeds: The valid set of speeds for the pedestrians.

  Returns:
    The list of pedestrians actors.
  """
  # Blueprints library.
  bl = world.get_blueprint_library()
  # Output container
  actors = list()
  for n in range(num_pedestrians):
    # Fetch random blueprint.
    pedestrian_bp = random.choice(bl.filter("walker.pedestrian.*"))
    # Make pedestrian invicible.
    pedestrian_bp.set_attribute("is_invincible", "true")
    while len(actors) != n:
      # Get random location.
      spawn_point = carla.Transform()  # pylint: disable=no-member
      spawn_point.location = world.get_random_location_from_navigation()
      if spawn_point.location is None:
        continue
      # Attempt to spawn vehicle in random location.
      actor = world.try_spawn_actor(pedestrian_bp, spawn_point)
      if actor is not None:
        actors.append(actor)
  logging.debug("Spawned {} pedestrians".format(len(actors)))
  return actors


def spawn_camera(
    hero: carla.ActorBlueprint,  # pylint: disable=no-member
    config: Mapping[str, Any],
    camera_type: str,
) -> carla.ServerSideSensor:  # pylint: disable=no-member
  """Spawns a camera on `hero`.

  Args:
    hero: The agent to attach the camera on.
    config: The attribute-value pairs for the configuration
      of the sensor.
    camera_type: Camera type, one of ("rgb", "semantic_segmentation").

  Returns:
    The spawned  camera sensor.
  """
  assert camera_type in ("rgb", "semantic_segmentation")

  # Get hero's world.
  world = hero.get_world()
  # Blueprints library.
  bl = world.get_blueprint_library()
  # Configure blueprint.
  camera_bp = bl.find("sensor.camera.{}".format(camera_type))
  for attribute, value in config["attributes"].items():
    camera_bp.set_attribute(attribute, value)
  logging.debug("Spawns a {} camera".format(camera_type))
  return world.spawn_actor(
      camera_bp,
      carla.Transform(  # pylint: disable=no-member
          carla.Location(**config["actor"]["location"]),  # pylint: disable=no-member
          carla.Rotation(**config["actor"]["rotation"]),  # pylint: disable=no-member
      ),
      attach_to=hero,
  )


def spawn_lidar(
    hero: carla.ActorBlueprint,  # pylint: disable=no-member
    config: Mapping[str, Any],
) -> carla.ServerSideSensor:  # pylint: disable=no-member
  """Spawns LIDAR sensor on `hero`.

  Args:
    hero: The agent to attach the LIDAR sensor on.
    config: The attribute-value pairs for the configuration
      of the sensor.

  Returns:
    The spawned LIDAR sensor.
  """
  # Get hero's world.
  world = hero.get_world()
  # Blueprints library.
  bl = world.get_blueprint_library()
  # Configure blueprint.
  lidar_bp = bl.find("sensor.lidar.ray_cast")
  for attribute, value in config["attributes"].items():
    lidar_bp.set_attribute(attribute, value)
  logging.debug("Spawns a LIDAR sensor")
  return world.spawn_actor(
      lidar_bp,
      carla.Transform(  # pylint: disable=no-member
          carla.Location(**config["actor"]["location"]),  # pylint: disable=no-member
          carla.Rotation(),  # pylint: disable=no-member
      ),
      attach_to=hero,
  )


def spawn_collision(
    hero: carla.ActorBlueprint,  # pylint: disable=no-member
) -> carla.ServerSideSensor:  # pylint: disable=no-member
  """Spawns collision sensor on `hero`.

  Args:
    hero: The agent to attach the collision sensor on.

  Returns:
    The spawned collision sensor.
  """
  # Get hero's world.
  world = hero.get_world()
  # Blueprints library.
  bl = world.get_blueprint_library()
  # Configure blueprint.
  collision_bp = bl.find("sensor.other.collision")
  logging.debug("Spawns a collision sensor")
  return world.spawn_actor(
      collision_bp,
      carla.Transform(),  # pylint: disable=no-member
      attach_to=hero,
  )


def spawn_lane_invasion(
    hero: carla.ActorBlueprint,  # pylint: disable=no-member
) -> carla.ServerSideSensor:  # pylint: disable=no-member
  """Spawns lane invasion sensor on `hero`.

  Args:
    hero: The agent to attach the collision sensor on.

  Returns:
    The spawned lane invasion sensor.
  """
  # Get hero's world.
  world = hero.get_world()
  # Blueprints library.
  bl = world.get_blueprint_library()
  # Configure blueprint.
  collision_bp = bl.find("sensor.other.lane_invasion")
  logging.debug("Spawns a lane invasion sensor")
  return world.spawn_actor(
      collision_bp,
      carla.Transform(),  # pylint: disable=no-member
      attach_to=hero,
  )


def get_spawn_point(
    world: carla.World,  # pylint: disable=no-member
    spawn_point: Optional[Union[int, carla.Transform]]  # pylint: disable=no-member
) -> carla.Location:  # pylint: disable=no-member
  """Parses and returns a CARLA spawn points."""
  if isinstance(spawn_point, carla.Transform):  # pylint: disable=no-member
    _spawn_point = spawn_point
  elif isinstance(spawn_point, int):
    _spawn_point = world.get_map().get_spawn_points()[spawn_point]
  else:
    _spawn_point = random.choice(world.get_map().get_spawn_points())
  return _spawn_point


def get_actors(
    world: carla.World,  # pylint: disable=no-member
    spawn_point: Optional[Union[int, carla.Location]],  # pylint: disable=no-member
    num_vehicles: int,
    num_pedestrians: int,
) -> Tuple[carla.Vehicle, Sequence[Optional[carla.Vehicle]],  # pylint: disable=no-member
           Sequence[Optional[carla.Walker]]]:  # pylint: disable=no-member
  """Spawns and returns the `hero`, the `vehicles` and the `pedestrians`.

  Args:
    world: The world object associated with the simulation.
    spawn_point: The hero vehicle spawn point. If an int is
      provided then the index of the spawn point is used.
      If None, then randomly selects a spawn point every time
      from the available spawn points of each map.
    num_vehicles: The number of vehicles to spawn.
    num_pedestrians: The number of pedestrians to spawn.

  Returns:
    hero: The spawned ego vehicle agent object.
    vehicles: The spawned vehicles agent objcets.
    pedestrians: The spawned walker agent objects.
  """
  # HERO agent.
  _spawn_point = get_spawn_point(world, spawn_point)
  hero = spawn_hero(
      world=world,
      spawn_point=_spawn_point,
      vehicle_id="vehicle.ford.mustang",
  )
  # Other vehicles.
  vehicles = spawn_vehicles(
      world=world,
      num_vehicles=num_vehicles,
  )
  # Other pedestrians.
  pedestrians = spawn_pedestrians(
      world=world,
      num_pedestrians=num_pedestrians,
  )
  return hero, vehicles, pedestrians


def vehicle_to_carla_measurements(
    vehicle: carla.Vehicle,  # pylint: disable=no-member
) -> Mapping[str, Any]:
  """Wraps all the `get_` calls from the `CARLA` interface."""
  control = vehicle.get_control()
  _transform = vehicle.get_transform()
  location = _transform.location
  rotation = _transform.rotation
  velocity = vehicle.get_velocity()
  acceleration = vehicle.get_acceleration()
  orientation = _transform.get_forward_vector()
  angular_velocity = vehicle.get_angular_velocity()
  speed_limit = vehicle.get_speed_limit()
  is_at_traffic_light = vehicle.is_at_traffic_light()
  traffic_light_state = vehicle.get_traffic_light_state().conjugate()
  return dict(
      control=control,
      location=location,
      rotation=rotation,
      velocity=velocity,
      acceleration=acceleration,
      orientation=orientation,
      angular_velocity=angular_velocity,
      speed_limit=speed_limit,
      is_at_traffic_light=is_at_traffic_light,
      traffic_light_state=traffic_light_state,
  )


def carla_xyz_to_ndarray(xyz: Any) -> np.ndarray:
  """Converts a `CARLA` measurement with attributes `x`, `y` and `z` to neural
  network friendly tensor."""
  return np.asarray(
      [xyz.x, xyz.y, xyz.z],
      dtype=np.float32,
  )


def carla_rotation_to_ndarray(
    rotation: carla.VehicleControl  # pylint: disable=no-member
) -> np.ndarray:
  """Converts a `CARLA` rotation to a neural network friendly tensor."""
  return np.asarray(
      [rotation.pitch, rotation.yaw, rotation.roll],
      dtype=np.float32,
  )


def carla_control_to_ndarray(
    control: carla.VehicleControl  # pylint: disable=no-member
) -> np.ndarray:
  """Converts a `CARLA` vehicle control to a neural network friendly tensor."""
  return np.asarray(
      [control.throttle, control.steer, control.brake],
      dtype=np.float32,
  )


def carla_measurements_to_ndarrays(
    measurements: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
  """Converts the `CARLA` measurements to neural network friendly tensors."""
  control = measurements["control"]
  location = measurements["location"]
  rotation = measurements["rotation"]
  velocity = measurements["velocity"]
  acceleration = measurements["acceleration"]
  orientation = measurements["orientation"]
  angular_velocity = measurements["angular_velocity"]
  speed_limit = measurements["speed_limit"]
  is_at_traffic_light = measurements["is_at_traffic_light"]
  traffic_light_state = measurements["traffic_light_state"]
  return dict(
      control=carla_control_to_ndarray(control),
      location=carla_xyz_to_ndarray(location),
      rotation=carla_rotation_to_ndarray(rotation),
      velocity=carla_xyz_to_ndarray(velocity),
      acceleration=carla_xyz_to_ndarray(acceleration),
      orientation=carla_xyz_to_ndarray(orientation),
      angular_velocity=carla_xyz_to_ndarray(angular_velocity),
      speed_limit=np.asarray(
          speed_limit,
          dtype=np.float32,
      ),
      is_at_traffic_light=int(is_at_traffic_light),
      traffic_light_state=int(traffic_light_state),
  )


def ndarray_to_location(array: np.ndarray) -> carla.Location:  # pylint: disable=no-member
  """Converts neural network friendly tensor back to `carla.Location`."""
  return carla.Location(*list(map(float, array)))  # pylint: disable=no-member


def ndarray_to_rotation(array: np.ndarray) -> carla.Rotation:  # pylint: disable=no-member
  """Converts neural network friendly tensor back to `carla.Rotation`."""
  return carla.Rotation(*list(map(float, array)))  # pylint: disable=no-member


def ndarray_to_vector3d(array: np.ndarray) -> carla.Vector3D:  # pylint: disable=no-member
  """Converts neural network friendly tensor back to `carla.Vector3D`."""
  return carla.Vector3D(*list(map(float, array)))  # pylint: disable=no-member


def ndarray_to_control(array: np.ndarray) -> carla.VehicleControl:  # pylint: disable=no-member
  """Converts neural network friendly tensor back to `carla.VehicleControl`."""
  return carla.VehicleControl(*list(map(float, array)))  # pylint: disable=no-member


def ndarrays_to_vehicle_measurements(
    observation: Mapping[str, np.ndarray],  # pylint: disable=no-member
) -> Mapping[str, Any]:
  """Converts neural network friendly tensors back to `CARLA` objects."""
  return dict(
      control=carla.VehicleControl(*list(map(float, observation["control"]))),  # pylint: disable=no-member
      location=ndarray_to_location(observation["location"]),
      rotation=ndarray_to_rotation(observation["rotation"]),
      velocity=ndarray_to_vector3d(observation["velocity"]),
      acceleration=ndarray_to_vector3d(observation["acceleration"]),
      orientation=ndarray_to_vector3d(observation["orientation"]),
      angular_velocity=ndarray_to_vector3d(observation["angular_velocity"]),
      speed_limit=float(observation["speed_limit"]),
      is_at_traffic_light=bool(observation["is_at_traffic_light"]),
      traffic_light_state=carla.TrafficLightState.values[int(  # pylint: disable=no-member
          observation["traffic_light_state"])],
  )


def rot2mat(rotation: np.ndarray) -> np.ndarray:
  """Returns the rotation matrix (3x3) given rotation in degrees."""
  rotation_radians = ndarray_to_rotation(rotation)
  pitch = np.deg2rad(rotation_radians.pitch)
  roll = np.deg2rad(rotation_radians.roll)
  yaw = np.deg2rad(rotation_radians.yaw)
  return transforms3d.euler.euler2mat(roll, pitch, yaw).T


def world2local(*, current_location: np.ndarray, current_rotation: np.ndarray,
                world_locations: np.ndarray) -> np.ndarray:
  """Converts `world_locations` to local coordinates.

  Args:
    current_location: The ego-vehicle location, with shape `[3]`.
    current_rotation: The ego-vehicle rotation, with shape `[3]`.
    world_locations: The locations to be transformed, with shape `[..., 3]`.

  Returns:
    The local coordinates, with shape `[..., 3]`.
  """
  # Prepares interfaces.
  assert current_location.shape == (3,)
  assert current_rotation.shape == (3,)
  assert len(world_locations.shape) < 3
  world_locations = np.atleast_2d(world_locations)

  # Builds the rotation matrix.
  R = rot2mat(current_rotation)
  # Transforms world coordinates to local coordinates.
  local_locations = np.dot(a=R, b=(world_locations - current_location).T).T

  return np.squeeze(local_locations)


def local2world(*, current_location: np.ndarray, current_rotation: np.ndarray,
                local_locations: np.ndarray) -> np.ndarray:
  """Converts `local_locations` to global coordinates.

  Args:
    current_location: The ego-vehicle location, with shape `[3]`.
    current_rotation: The ego-vehicle rotation, with shape `[3]`.
    local_locations: The locations to be transformed, with shape `[..., 3]`.

  Returns:
    The global coordinates, with shape `[..., 3]`.
  """
  # Prepares interfaces.
  assert current_location.shape == (3,)
  assert current_rotation.shape == (3,)
  assert len(local_locations.shape) < 3
  local_locations = np.atleast_2d(local_locations)

  # Builds the inverse rotation matrix.
  R_inv = np.linalg.inv(rot2mat(current_rotation))
  # Transforms local coordinates to world coordinates.
  global_locations = np.dot(a=R_inv, b=local_locations.T).T + current_location

  return global_locations


def global_plan(
    world: carla.World,  # pylint: disable=no-member
    origin: carla.Location,  # pylint: disable=no-member
    destination: carla.Location,  # pylint: disable=no-member
) -> Tuple[Sequence[carla.Waypoint], Sequence[Any], float]:  # pylint: disable=no-member
  """Generates the optimal plan between two location, respecting the topology.

  Args:
    world: The `CARLA` world.
    origin: The starting location.
    destination: The final destination.

  Returns:
    waypoints: A sequence of waypoints.
    roadoptions: A sequence of commands to navigate at each waypoint.
    distances: The distance per pair of waypoints of the plan.
  """
  from agents.navigation.global_route_planner import GlobalRoutePlanner  # pylint: disable=import-error
  from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # pylint: disable=import-error

  # Setup global planner.
  grp_dao = GlobalRoutePlannerDAO(wmap=world.get_map(), sampling_resolution=1)
  grp = GlobalRoutePlanner(grp_dao)
  grp.setup()
  # Generate plan.
  waypoints, roadoptions = zip(*grp.trace_route(origin, destination))
  # Accummulate pairwise distance.
  distances = [0.0]
  for i in range(1, len(waypoints)):
    loc_tm1 = waypoints[i - 1].transform.location
    loc_tm1 = np.asarray([loc_tm1.x, loc_tm1.y, loc_tm1.z])
    loc_t = waypoints[i].transform.location
    loc_t = np.asarray([loc_t.x, loc_t.y, loc_t.z])
    distances.append(np.linalg.norm(loc_tm1 - loc_t))

  return waypoints, roadoptions, distances
