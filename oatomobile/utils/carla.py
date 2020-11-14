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

import numpy as np
import transforms3d.euler
from absl import logging

import carla


def load_and_wait_for_world(client: carla.Client,
                            town: str,
                            fps: int,
                            traffic_manager: carla.TrafficManager
                            ) -> Tuple[carla.World, int]:
    """
    Load a new CARLA world
    """

    client.load_world(town)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    frame = world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)

    world.tick()

    return world, frame


def setup(
        town: str,
        off_screen: bool = False,
        server_timestop: float = 20.0
) -> Tuple[subprocess.Popen, int]:  # pylint: disable=no-member
    """Returns the `CARLA` `server`, `client` and `world`.

    Args:
      town: The `CARLA` town identifier.
      server_timestop: The time interval between spawning the server
        and resuming program.
    Returns:
      server: The `CARLA` server.
    """
    assert town in ("Town01", "Town02", "Town03", "Town04", "Town05")

    # Random assignment of port.
    port = np.random.randint(2000, 3000)

    # Start CARLA server.
    env = os.environ.copy()
    params = [
        os.path.join(os.environ.get("CARLA_ROOT"), "CarlaUE4.sh"),
        "-carla-rpc-port={}".format(port),
        "-quality-level=Epic"
    ]
    if off_screen:
        env["DISPLAY"] = ""
        params.append("-opengl")

    logging.debug("Inits a CARLA server at port={}".format(port))
    server = subprocess.Popen(params,
                              stdout=None,
                              stderr=subprocess.STDOUT,
                              preexec_fn=os.setsid,
                              env=env)
    atexit.register(os.killpg, server.pid, signal.SIGKILL)
    time.sleep(server_timestop)

    return server, port

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
  points = np.reshape(points, (int(points.shape[0] / 4), 4))

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
  hero = world.try_spawn_actor(hero_bp, spawn_point)
  return hero


def spawn_vehicles_and_pedestrians(
    world: carla.World,  # pylint: disable=no-member
    client: carla.Client,
    traffic_manager: carla.TrafficManager,
    num_vehicles: int,
    num_pedestrians: int,
) -> Tuple[Sequence[str], Sequence[str]]:  # pylint: disable=no-member
  """Spawns `vehicles` randomly in spawn points.

  Args:
    world: The world object associated with the simulation.
    num_vehicles: The number of vehicles to spawn.

  Returns:
    The list of vehicles actors.
  """
  vehicles_list = []
  walkers_list = []
  all_id = []

  # Assume synchronous mode
  synchronous_master = True
  try:


      blueprints = world.get_blueprint_library().filter('vehicle.*')
      blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

      # avoid spawning vehicles prone to accidents
      blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
      blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
      blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
      blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
      blueprints = [x for x in blueprints if not x.id.endswith('t2')]

      blueprints = sorted(blueprints, key=lambda bp: bp.id)

      spawn_points = world.get_map().get_spawn_points()
      number_of_spawn_points = len(spawn_points)

      if num_vehicles < number_of_spawn_points:
          random.shuffle(spawn_points)
      elif num_vehicles > number_of_spawn_points:
          msg = 'requested %d vehicles, but could only find %d spawn points'
          logging.warning(msg, num_vehicles, number_of_spawn_points)
          num_vehicles = number_of_spawn_points

      # @todo cannot import these directly.
      SpawnActor = carla.command.SpawnActor
      SetAutopilot = carla.command.SetAutopilot
      SetVehicleLightState = carla.command.SetVehicleLightState
      FutureActor = carla.command.FutureActor

      # --------------
      # Spawn vehicles
      # --------------
      batch = []
      for n, transform in enumerate(spawn_points):
          if n >= num_vehicles:
              break
          blueprint = random.choice(blueprints)
          if blueprint.has_attribute('color'):
              color = random.choice(blueprint.get_attribute('color').recommended_values)
              blueprint.set_attribute('color', color)
          if blueprint.has_attribute('driver_id'):
              driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
              blueprint.set_attribute('driver_id', driver_id)
          blueprint.set_attribute('role_name', 'autopilot')

          # prepare the light state of the cars to spawn
          light_state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam  # (farzad) Does not work!
          # light_state = carla.VehicleLightState.NONE

          # spawn the cars and set their autopilot and light state all together
          batch.append(SpawnActor(blueprint, transform)
                       .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                       .then(SetVehicleLightState(FutureActor, light_state)))

      for response in client.apply_batch_sync(batch, synchronous_master):
          if response.error:
              logging.error(response.error)
          else:
              vehicles_list.append(response.actor_id)

      # -------------
      # Spawn Walkers
      # -------------
      # some settings
      percentagePedestriansRunning = 0.0  # how many pedestrians will run
      percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
      # 1. take all the random locations to spawn
      spawn_points = []
      for i in range(num_pedestrians):
          spawn_point = carla.Transform()
          loc = world.get_random_location_from_navigation()
          if (loc != None):
              spawn_point.location = loc
              spawn_points.append(spawn_point)
      # 2. we spawn the walker object
      batch = []
      walker_speed = []
      for spawn_point in spawn_points:
          walker_bp = random.choice(blueprintsWalkers)
          # set as not invincible
          if walker_bp.has_attribute('is_invincible'):
              walker_bp.set_attribute('is_invincible', 'false')
          # set the max speed
          if walker_bp.has_attribute('speed'):
              if (random.random() > percentagePedestriansRunning):
                  # walking
                  walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
              else:
                  # running
                  walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
          else:
              print("Walker has no speed")
              walker_speed.append(0.0)
          batch.append(SpawnActor(walker_bp, spawn_point))
      results = client.apply_batch_sync(batch, True)
      walker_speed2 = []
      for i in range(len(results)):
          if results[i].error:
              logging.error(results[i].error)
          else:
              walkers_list.append({"id": results[i].actor_id})
              walker_speed2.append(walker_speed[i])
      walker_speed = walker_speed2
      # 3. we spawn the walker controller
      batch = []
      walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
      for i in range(len(walkers_list)):
          batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
      results = client.apply_batch_sync(batch, True)
      for i in range(len(results)):
          if results[i].error:
              logging.error(results[i].error)
          else:
              walkers_list[i]["con"] = results[i].actor_id
      # 4. we put altogether the walkers and controllers id to get the objects from their id
      for i in range(len(walkers_list)):
          all_id.append(walkers_list[i]["con"])
          all_id.append(walkers_list[i]["id"])
      all_actors = world.get_actors(all_id)

      # wait for a tick to ensure client receives the last transform of the walkers we have just created
      if not synchronous_master:
          world.wait_for_tick()
      else:
          world.tick()

      # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
      # set how many pedestrians can cross the road
      world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
      for i in range(0, len(all_id), 2):
          # start walker
          all_actors[i].start()
          # set walk to random point
          all_actors[i].go_to_location(world.get_random_location_from_navigation())
          # max speed
          all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))
  except:
      if synchronous_master:
          settings = world.get_settings()
          settings.synchronous_mode = False
          traffic_manager.set_synchronous_mode(False)
          settings.fixed_delta_seconds = None
          world.apply_settings(settings)

      logging.debug(('\ndestroying %d vehicles' % len(vehicles_list)))
      client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

      # stop walker controllers (list is [controller, actor, controller, actor ...])
      for i in range(0, len(all_id), 2):
          all_actors[i].stop()

      logging.debug('\ndestroying %d walkers' % len(walkers_list))
      client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

      time.sleep(0.5)

  logging.debug("Spawned {} vehicles and {} pedestrians".format(len(vehicles_list), len(walkers_list)))
  return vehicles_list, [w['id'] for w in walkers_list]


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
      attachment_type=carla.AttachmentType.Rigid
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
  # vehicles = spawn_vehicles(
  #     world=world,
  #     num_vehicles=num_vehicles,
  # )
  # Other pedestrians.
  # pedestrians = spawn_pedestrians(
  #     world=world,
  #     num_pedestrians=num_pedestrians,
  # )
  # return hero, vehicles, pedestrians


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
    rotation: carla.Rotation  # pylint: disable=no-member
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
  try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # pylint: disable=import-error
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # pylint: disable=import-error
  except ImportError:
    raise ImportError(
        "Missing CARLA installation, "
        "make sure the environment variable CARLA_ROOT is provided "
        "and that the PythonAPI is `easy_install`ed")

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
