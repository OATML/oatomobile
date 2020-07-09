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
"""Defines a simple PID-controller-based autopilot."""

import math
import random
from typing import Any
from typing import Optional
from typing import Tuple

import carla
from absl import logging

import oatomobile
from oatomobile.simulators.carla import defaults
from oatomobile.util import carla as cutil


class AutopilotAgent(oatomobile.Agent):
  """An autopilot agent, based on the official implementation of
  `carla.PythonAPI.agents.navigation.basic_agent.BasicAgent`"""

  def __init__(self,
               environment: oatomobile.envs.CARLAEnv,
               *,
               proximity_tlight_threshold: float = 5.0,
               proximity_vehicle_threshold: float = 10.0,
               noise: float = 0.1) -> None:
    """Constructs an autopilot agent.

    Args:
      environment: The navigation environment to spawn the agent.
      proximity_tlight_threshold: The threshold (in metres) to
        the traffic light.
      proximity_vehicle_threshold: The threshold (in metres) to
        the front vehicle.
      noise: The percentage of random actions.
    """
    super(AutopilotAgent, self).__init__(environment=environment)

    # References to the CARLA objects.
    self._vehicle = self._environment.simulator.hero
    self._world = self._vehicle.get_world()
    self._map = self._world.get_map()

    # Agent hyperparametres.
    self._proximity_tlight_threshold = proximity_tlight_threshold
    self._proximity_vehicle_threshold = proximity_vehicle_threshold
    self._hop_resolution = 2.0
    self._path_seperation_hop = 2
    self._path_seperation_threshold = 0.5
    self._target_speed = defaults.TARGET_SPEED
    self._noise = noise

    # The internal state of the agent.
    self._last_traffic_light = None

    # Local planner, including the PID controllers.
    dt = self._vehicle.get_world().get_settings().fixed_delta_seconds
    lateral_control_dict = defaults.LATERAL_PID_CONTROLLER_CONFIG.copy()
    lateral_control_dict.update({"dt": dt})
    # TODO(filangel): tune the parameters for FPS != 20
    self._local_planner = oatomobile.LocalPlanner(
        self._vehicle,
        opt_dict=dict(
            target_speed=self._target_speed,
            dt=dt,
        ),
    )

    # Set agent's dsestination.
    if hasattr(self._environment.unwrapped.simulator, "destination"):
      self._set_destination(
          self._environment.unwrapped.simulator.destination.location)

  def act(
      self,
      observation: oatomobile.Observations,
  ) -> oatomobile.Action:
    """Takes in an observation, samples from agent's policy, returns an
    action."""
    # Remove unused arguments.
    del observation

    # Random action branch.
    if random.random() < self._noise:
      return carla.VehicleControl(  # pylint: disable=no-member
          **{
              k: float(v)
              for (k, v) in self._environment.action_space.sample().items()
          })
    # Normal autopilot action.
    else:
      return self._run_step()

  def _run_step(
      self,
      debug: bool = False,
  ) -> carla.VehicleControl:  # pylint: disable=no-member
    """Executes one step of navigation."""

    # is there an obstacle in front of us?
    hazard_detected = False

    # retrieve relevant elements for safe navigation, i.e.: traffic lights
    # and other vehicles
    actor_list = self._world.get_actors()
    vehicle_list = actor_list.filter("*vehicle*")
    lights_list = actor_list.filter("*traffic_light*")

    # check possible obstacles
    vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
    if vehicle_state:
      if debug:
        logging.debug('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

      hazard_detected = True

    # check for the state of the traffic lights
    light_state, traffic_light = self._is_light_red(lights_list)
    if light_state:
      if debug:
        logging.debug('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

      hazard_detected = True

    if hazard_detected:
      control = carla.VehicleControl()  # pylint: disable=no-member
      control.steer = 0.0
      control.throttle = 0.0
      control.brake = 1.0
      control.hand_brake = False
    else:
      # standard local planner behavior
      control = self._local_planner.run_step(debug=debug)

    return control

  def _set_destination(
      self,
      destination: carla.Location,  # pylint: disable=no-member
  ) -> None:
    """Generates the global plan for the agent.

    Args:
      destination: The location of the new destination.
    """
    # Set vehicle's current location as start for the plan.
    origin = self._vehicle.get_location()
    start_waypoint = self._map.get_waypoint(origin).transform.location
    end_waypoint = self._map.get_waypoint(destination).transform.location
    # Calculate the plan.
    waypoints, roadoptions, _ = cutil.global_plan(
        world=self._world,
        origin=start_waypoint,
        destination=end_waypoint,
    )
    # Mutate the local planner's global plan.
    self._local_planner.set_global_plan(list(zip(waypoints, roadoptions)))

  def _is_vehicle_hazard(
      self,
      vehicle_list,
  ) -> Tuple[bool, Optional[carla.Vehicle]]:  # pylint: disable=no-member
    """It detects if a vehicle in the scene can be dangerous for the ego
    vehicle's current plan.

    Args:
      vehicle_list: List of potential vehicles (obstancles) to check.

    Returns:
      vehicle_ahead: If True a vehicle ahead blocking us and False otherwise.
      vehicle: The blocker vehicle itself.
    """

    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    for target_vehicle in vehicle_list:
      # do not account for the ego vehicle.
      if target_vehicle.id == self._vehicle.id:
        continue

      # if the object is not in our lane it's not an obstacle.
      target_vehicle_waypoint = self._map.get_waypoint(
          target_vehicle.get_location())
      if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
              target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
        continue

      loc = target_vehicle.get_location()
      if oatomobile.is_within_distance_ahead(
          loc,
          ego_vehicle_location,
          self._vehicle.get_transform().rotation.yaw,
          self._proximity_vehicle_threshold,
      ):
        return (True, target_vehicle)

    return (False, None)

  def _is_light_red(
      self,
      lights_list,
  ) -> Tuple[bool, Any]:  # pylint: disable=no-member
    """It detects if the light in the scene is red.

    Args:
      lights_list: The list containing TrafficLight objects.

    Returns:
      light_ahead: If True a traffic light ahead is read and False otherwise.
      traffic_light: The traffic light object ahead itself.
    """
    if self._map.name == 'Town01' or self._map.name == 'Town02':
      return self._is_light_red_europe_style(lights_list)
    else:
      return self._is_light_red_us_style(lights_list)

  def _is_light_red_europe_style(self, lights_list):
    """This method is specialized to check European style traffic lights."""
    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    for traffic_light in lights_list:
      object_waypoint = self._map.get_waypoint(traffic_light.get_location())
      if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
              object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
        continue

      loc = traffic_light.get_location()
      if oatomobile.is_within_distance_ahead(
          loc,
          ego_vehicle_location,
          self._vehicle.get_transform().rotation.yaw,
          self._proximity_tlight_threshold,
      ):
        if traffic_light.state == carla.TrafficLightState.Red:  # pylint: disable=no-member
          return (True, traffic_light)

    return (False, None)

  def _is_light_red_us_style(self, lights_list, debug=False):
    """This method is specialized to check US style traffic lights."""
    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    if ego_vehicle_waypoint.is_junction:
      # It is too late. Do not block the intersection! Keep going!
      return (False, None)

    if self._local_planner.target_waypoint is not None:
      if self._local_planner.target_waypoint.is_junction:
        min_angle = 180.0
        sel_magnitude = 0.0
        sel_traffic_light = None
        for traffic_light in lights_list:
          loc = traffic_light.get_location()
          magnitude, angle = oatomobile.compute_magnitude_angle(
              loc, ego_vehicle_location,
              self._vehicle.get_transform().rotation.yaw)
          if magnitude < 60.0 and angle < min(25.0, min_angle):
            sel_magnitude = magnitude
            sel_traffic_light = traffic_light
            min_angle = angle

        if sel_traffic_light is not None:
          if debug:
            logging.debug('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                sel_magnitude, min_angle, sel_traffic_light.id))

          if self._last_traffic_light is None:
            self._last_traffic_light = sel_traffic_light

          if self._last_traffic_light.state == carla.TrafficLightState.Red:  # pylint: disable=no-member
            return (True, self._last_traffic_light)
        else:
          self._last_traffic_light = None

    return (False, None)

  def _get_trafficlight_trigger_location(
      self,
      traffic_light,
  ) -> carla.Location:  # pylint: disable=no-member
    """Calculates the yaw of the waypoint that represents the trigger volume of
    the traffic light."""

    def rotate_point(point, radians):
      """Rotates a given point by a given angle."""
      rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
      rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

      return carla.Vector3D(rotated_x, rotated_y, point.z)  # pylint: disable=no-member

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(
        carla.Vector3D(0, 0, area_ext.z),  # pylint: disable=no-member
        math.radians(base_rot),
    )
    point_location = area_loc + carla.Location(x=point.x, y=point.y)  # pylint: disable=no-member

    return carla.Location(point_location.x, point_location.y, point_location.z)  # pylint: disable=no-member
