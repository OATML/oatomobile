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
"""Helper functions for `PyGame` and `CARLA` rendering."""

import glob
import os
import tempfile
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import carla  # pylint: disable=import-error
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pygame
import tqdm
from absl import logging
from skimage import transform


def setup(
    width: int = 400,
    height: int = 300,
    render: bool = True,
) -> Tuple[pygame.Surface, pygame.time.Clock, pygame.font.Font]:
  """Returns the `display`, `clock` and for a `PyGame` app.

  Args:
    width: The width (in pixels) of the app window.
    height: The height (in pixels) of the app window.
    render: If True it renders a window, it keeps the
      frame buffer on the memory otherwise.

  Returns:
    display: The main app window or frame buffer object.
    clock: The main app clock.
    font: The font object used for generating text.
  """
  # PyGame setup.
  pygame.init()  # pylint: disable=no-member
  pygame.display.set_caption("OATomobile")
  if render:
    logging.debug("PyGame initializes a window display")
    display = pygame.display.set_mode(  # pylint: disable=no-member
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF,  # pylint: disable=no-member
    )
  else:
    logging.debug("PyGame initializes a headless display")
    display = pygame.Surface((width, height))  # pylint: disable=too-many-function-args
  clock = pygame.time.Clock()
  font = pygame.font.SysFont("dejavusansmono", 14)
  return display, clock, font


def rgb_to_binary_mask(array: np.ndarray) -> np.ndarray:
  """Flattens the `array` in the channels dimension.

  Expects black and white.
  """
  # Removes the channel dimension.
  mask = np.sum(array, axis=-1)
  # Converts array to boolean.
  return mask.astype(bool).astype(int)[..., np.newaxis]


def ndarray_to_pygame_surface(
    array: np.ndarray,
    swapaxes: bool,
) -> pygame.Surface:
  """Returns a `PyGame` surface from a `NumPy` array (image).

  Args:
    array: The `NumPy` representation of the image to be converted to `PyGame`.

  Returns:
    A `PyGame` surface.
  """
  # Make sure its in 0-255 range.
  array = 255 * (array / array.max())
  if swapaxes:
    array = array.swapaxes(0, 1)
  return pygame.surfarray.make_surface(array)


def pygame_surface_to_ndarray(surface: pygame.Surface) -> np.ndarray:
  """Returns a `NumPy` array (image) from `PyGame` surface.

  Args:
    surface: The `PyGame` surface to be converted.

  Returns:
    A `NumPy` representation of the image converted from `PyGame`.
  """
  return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def mpl_figure_to_rgba(figure: plt.Figure) -> np.ndarray:
  """Returns a `NumPy` array (RGBA image) from a `matplotlib` figure.

  Args:
    figure: The `matplotlib` figure to be converted to an RGBA image.

  Returns:
    The RGBA image.
  """
  # Stores figure temporarily.
  with tempfile.NamedTemporaryFile(delete=True) as tmp:
    figure.savefig(
        "{}".format(tmp.name),
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    # Reads figure on memory.
    image = imageio.imread("{}".format(tmp.name))

  # Processes the image.
  image = image.astype(np.float32) / 255.0

  return image


def lidar_2darray_to_rgb(array: np.ndarray) -> np.ndarray:
  """Returns a `NumPy` array (image) from a 4 channel LIDAR point cloud.

  Args:
    array: The original LIDAR point cloud array.

  Returns:
    The `PyGame`-friendly image to be visualized.
  """
  # Get array shapes.
  W, H, C = array.shape
  assert C == 2

  # Select channel.
  img = np.c_[array, np.zeros(shape=(W, H, 1))]
  # Convert to 8-bit image.
  img = 255 * (img / img.max())
  return img


def world_to_pixel(
    location: carla.Location,  # pylint: disable=no-member
    scale: float,
    offset: Tuple[float, float],
    pixels_per_meter: int,
) -> Tuple[int, int]:
  """Converts to CARLA world coordinates to pixel coordinates.

  Args:
    world_coords: The xyz absolute (world) coordinates.
    scale: The scale factor of the coordinates.
    offset: The offset of the location.
    pixels_per_meter: The number of pixels per meter.

  Returns:
    The transformed pixel coordinates.
  """
  x = scale * pixels_per_meter * (location.x - offset[0])
  y = scale * pixels_per_meter * (location.y - offset[1])
  return (int(x - offset[0]), int(y - offset[1]))


def lateral_shift(
    transform: carla.Transform,  # pylint: disable=no-member
    shift: float,
) -> carla.Location:  # pylint: disable=no-member
  """Makes a lateral shift to the transform.

  Args:
    transform: The original transform.
    shift: The coordinate shift.

  Returns:
    The laterally shifted location.
  """
  transform.rotation.yaw += 90
  return transform.location + shift * transform.get_forward_vector()


def make_dashboard(display: pygame.Surface, font: pygame.font.Font,
                   clock: Optional[pygame.time.Clock],
                   **observations: np.ndarray) -> None:
  """Generates the dashboard used for visualizing the agent.

  Args:
    display: The `PyGame` renderable surface.
    observations: The aggregated observation object.
    font: The font object used for generating text.
    clock: The PyGame (client) clock.
  """
  # Clear dashboard.
  display.fill(COLORS["BLACK"])

  # Adaptive width.
  ada_width = 0

  if "lidar" in observations:
    # Render overhead LIDAR reading.
    ob_lidar_rgb = ndarray_to_pygame_surface(
        lidar_2darray_to_rgb(array=observations.get("lidar")),
        swapaxes=False,
    )
    display.blit(ob_lidar_rgb, (ada_width, 0))
    ada_width = ada_width + ob_lidar_rgb.get_width()

  if "left_camera_rgb" in observations:
    # Render left camera view.
    ob_left_camera_rgb_rgb = ndarray_to_pygame_surface(
        array=observations.get("left_camera_rgb"),
        swapaxes=True,
    )
    display.blit(ob_left_camera_rgb_rgb, (ada_width, 0))
    ada_width = ada_width + ob_left_camera_rgb_rgb.get_width()

  if "front_camera_rgb" in observations:
    # Render front camera view.
    ob_front_camera_rgb_rgb = ndarray_to_pygame_surface(
        array=observations.get("front_camera_rgb"),
        swapaxes=True,
    )
    display.blit(ob_front_camera_rgb_rgb, (ada_width, 0))
    ada_width = ada_width + ob_front_camera_rgb_rgb.get_width()

  if "rear_camera_rgb" in observations:
    # Render rear camera view.
    ob_rear_camera_rgb_rgb = ndarray_to_pygame_surface(
        array=observations.get("rear_camera_rgb"),
        swapaxes=True,
    )
    display.blit(ob_rear_camera_rgb_rgb, (ada_width, 0))
    ada_width = ada_width + ob_rear_camera_rgb_rgb.get_width()

  if "right_camera_rgb" in observations:
    # Render left camera view.
    ob_right_camera_rgb_rgb = ndarray_to_pygame_surface(
        array=observations.get("right_camera_rgb"),
        swapaxes=True,
    )
    display.blit(ob_right_camera_rgb_rgb, (ada_width, 0))
    ada_width = ada_width + ob_right_camera_rgb_rgb.get_width()

  for overhead_features in [
      "bird_view_camera_rgb",
      "bird_view_camera_cityscapes",
  ]:
    if overhead_features in observations:
      overhead_features_ndarray = observations.get(overhead_features)
      # Render observation.
      bev_meters = 25.0
      height, width, _ = overhead_features_ndarray.shape
      fig, ax = plt.subplots(clear=True)
      ax.imshow(
          overhead_features_ndarray,
          extent=(-bev_meters, bev_meters, bev_meters, -bev_meters),
      )
      ncol = 0
      if "goal" in observations:
        goal = observations.get("goal")
        ax.plot(
            goal[..., 1],
            -goal[..., 0],
            marker="D",
            markersize=4,
            color="#ecb01f",
            linestyle="None",
            alpha=0.25,
            label=r"$\mathcal{G}$",
        )
        ncol = ncol + 1
      if "predictions" in observations:
        predictions = observations.get("predictions")
        if predictions is not None:
          ax.plot(
              predictions[..., 1],
              -predictions[..., 0],
              marker="x",
              markersize=4,
              color="#d85218",
              alpha=0.5,
              label="agent",
          )
          ncol = ncol + 1
      if ncol > 0:
        ax.legend(
            ncol=ncol,
            loc="lower center",
            fancybox=False,
            prop={'size': 6},
        )
      ax.set(
          xlim=(-bev_meters, bev_meters),
          ylim=(bev_meters, -bev_meters),
          frame_on=False,
      )
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      overhead_features_frame = mpl_figure_to_rgba(fig)[..., :3]
      plt.close(fig)
      overhead_features_frame = transform.resize(
          overhead_features_frame,
          output_shape=(height, width),
      )
      overhead_features_surface = ndarray_to_pygame_surface(
          overhead_features_frame,
          swapaxes=True,
      )
      display.blit(overhead_features_surface, (ada_width, 0))
      ada_width = ada_width + overhead_features_surface.get_width()

  if "bird_view_camera_cityscapes" in observations:
    # Render bird-view camera observation.
    ob_bird_view_camera_cityscapes_rgb = ndarray_to_pygame_surface(
        array=observations.get("bird_view_camera_cityscapes"),
        swapaxes=True,
    )
    display.blit(ob_bird_view_camera_cityscapes_rgb, (ada_width, 0))
    ada_width = ada_width + ob_bird_view_camera_cityscapes_rgb.get_width()

  if "control" in observations:
    # Render control bars.
    control = observations.get("control")
    background_rect = pygame.Rect(0, 180, 64, 20)
    pygame.draw.rect(display, COLORS["APPLE SPACE GREY"], background_rect, 0)
    throttle_rect = pygame.Rect(10, 190, 4, -8 * control[0])
    pygame.draw.rect(display, COLORS["GOOGLE GREEN"], throttle_rect, 0)
    steer_rect = pygame.Rect(30, 188, 8 * control[1], 4)
    pygame.draw.rect(display, COLORS["GOOGLE BLUE"], steer_rect, 0)
    break_rect = pygame.Rect(50, 190, 4, 8 * control[2])
    pygame.draw.rect(display, COLORS["GOOGLE RED"], break_rect, 0)

  # if clock is not None:
  #   text = font.render(
  #       "FPS={0:.1f}".format(clock.get_fps()),
  #       True,
  #       COLORS["WHITE"],
  #   )
  #   display.blit(text, (80, 180))

  if "velocity" in observations:
    text = font.render(
        "v={0:.1f}km/h".format(3.6 * np.linalg.norm(observations["velocity"])),
        True,
        COLORS["WHITE"],
    )
    display.blit(text, (200, 180))

  if "acceleration" in observations:
    text = font.render(
        "v={0:.1f}m/s^2".format(np.linalg.norm(observations["acceleration"])),
        True,
        COLORS["WHITE"],
    )
    display.blit(text, (300, 180))

  if "is_at_traffic_light" in observations and "traffic_light_state" in observations:
    if observations["is_at_traffic_light"] == 1 and observations[
        "traffic_light_state"] == 0:
      pygame.draw.circle(display, COLORS["GOOGLE RED"], (80, 190), 7)


def draw_settings(
    carla_map: carla.Map,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> Tuple[float, pygame.Surface]:
  """Calculates the `PyGame` surfaces settings.

  Args:
    carla_map: The `CARLA` town map.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns:
    offset: The world offset.
    surface: The empty `PyGame` surface.
  """
  # The graph representing the CARLA map.
  waypoints = carla_map.generate_waypoints(1)

  # Calculate the width of the surface.
  max_x = max(
      waypoints,
      key=lambda x: x.transform.location.x,
  ).transform.location.x + margin
  max_y = max(
      waypoints,
      key=lambda x: x.transform.location.y,
  ).transform.location.y + margin
  min_x = min(
      waypoints,
      key=lambda x: x.transform.location.x,
  ).transform.location.x - margin
  min_y = min(
      waypoints,
      key=lambda x: x.transform.location.y,
  ).transform.location.y - margin
  world_offset = (min_x, min_y)
  width = max(max_x - min_x, max_y - min_y)
  width_in_pixels = int(pixels_per_meter * width)

  return world_offset, pygame.Surface((width_in_pixels, width_in_pixels))  # pylint: disable=too-many-function-args


def get_road_surface(
    world: carla.World,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> pygame.Surface:
  """Generates a `PyGame` surface of a CARLA town.

  Heavily inspired by the official CARLA `no_rendering_mode.py` example.

  Args:
    world: The `CARLA` world.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The topology of a CARLA town as a `PyGame` surface.
  """
  # Fetch CARLA map.
  carla_map = world.get_map()

  # Setups the `PyGame` surface and offsets.
  world_offset, surface = draw_settings(
      carla_map=carla_map,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )

  # Set background black
  surface.fill(COLORS["BLACK"])
  precision = 0.05

  # Parse OpenDrive topology.
  topology = [x[0] for x in carla_map.get_topology()]
  topology = sorted(topology, key=lambda w: w.transform.location.z)
  set_waypoints = []
  for waypoint in topology:
    waypoints = [waypoint]

    nxt = waypoint.next(precision)
    if len(nxt) > 0:
      nxt = nxt[0]
      while nxt.road_id == waypoint.road_id:
        waypoints.append(nxt)
        nxt = nxt.next(precision)
        if len(nxt) > 0:
          nxt = nxt[0]
        else:
          break
    set_waypoints.append(waypoints)

  # Draw roads.
  for waypoints in set_waypoints:
    waypoint = waypoints[0]
    road_left_side = [
        lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints
    ]
    road_right_side = [
        lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints
    ]

    polygon = road_left_side + [x for x in reversed(road_right_side)]
    polygon = [
        world_to_pixel(
            x,
            scale=scale,
            offset=world_offset,
            pixels_per_meter=pixels_per_meter,
        ) for x in polygon
    ]

    if len(polygon) > 2:
      pygame.draw.polygon(surface, COLORS["WHITE"], polygon, 5)
      pygame.draw.polygon(surface, COLORS["WHITE"], polygon)

  return surface


def get_lane_boundaries_surface(
    world: carla.World,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> pygame.Surface:
  """Generates a `PyGame` surface of a CARLA town lane boundaries.

  Heavily inspired by the official CARLA `no_rendering_mode.py` example.

  Args:
    world: The `CARLA` world.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The lane boundaries of a CARLA town as a `PyGame` surface.
  """
  # Fetch CARLA map.
  carla_map = world.get_map()

  # Setups the `PyGame` surface and offsets.
  world_offset, surface = draw_settings(
      carla_map=carla_map,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )

  def get_lane_markings(lane_marking_type, waypoints, sign):
    margin = 0.25
    marking_1 = [
        world_to_pixel(
            lateral_shift(w.transform, sign * w.lane_width * 0.5),
            scale,
            world_offset,
            pixels_per_meter,
        ) for w in waypoints
    ]
    if lane_marking_type == carla.LaneMarkingType.Broken or (  # pylint: disable=no-member
        lane_marking_type == carla.LaneMarkingType.Solid):  # pylint: disable=no-member
      return [(lane_marking_type, marking_1)]
    else:
      marking_2 = [
          world_to_pixel(
              lateral_shift(w.transform,
                            sign * (w.lane_width * 0.5 + margin * 2)),
              scale,
              world_offset,
              pixels_per_meter,
          ) for w in waypoints
      ]
      if lane_marking_type == carla.LaneMarkingType.SolidBroken:  # pylint: disable=no-member
        return [
            (carla.LaneMarkingType.Broken, marking_1),  # pylint: disable=no-member
            (carla.LaneMarkingType.Solid, marking_2),  # pylint: disable=no-member
        ]
      elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:  # pylint: disable=no-member
        return [
            (carla.LaneMarkingType.Solid, marking_1),  # pylint: disable=no-member
            (carla.LaneMarkingType.Broken, marking_2),  # pylint: disable=no-member
        ]
      elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:  # pylint: disable=no-member
        return [
            (carla.LaneMarkingType.Broken, marking_1),  # pylint: disable=no-member
            (carla.LaneMarkingType.Broken, marking_2),  # pylint: disable=no-member
        ]

      elif lane_marking_type == carla.LaneMarkingType.SolidSolid:  # pylint: disable=no-member
        return [
            (carla.LaneMarkingType.Solid, marking_1),  # pylint: disable=no-member
            (carla.LaneMarkingType.Solid, marking_2),  # pylint: disable=no-member
        ]

    return [
        (carla.LaneMarkingType.NONE, []),  # pylint: disable=no-member
    ]

  def draw_solid_line(surface, color, closed, points, width):
    if len(points) >= 2:
      pygame.draw.lines(surface, color, closed, points, width)

  def draw_broken_line(surface, color, closed, points, width):
    broken_lines = [
        x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0
    ]
    for line in broken_lines:
      pygame.draw.lines(surface, color, closed, line, width)

  def draw_lane_marking_single_side(surface, waypoints, sign):
    lane_marking = None

    marking_type = carla.LaneMarkingType.NONE  # pylint: disable=no-member
    previous_marking_type = carla.LaneMarkingType.NONE  # pylint: disable=no-member

    markings_list = []
    temp_waypoints = []
    current_lane_marking = carla.LaneMarkingType.NONE  # pylint: disable=no-member
    for sample in waypoints:
      lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

      if lane_marking is None:
        continue

      marking_type = lane_marking.type

      if current_lane_marking != marking_type:
        markings = get_lane_markings(
            previous_marking_type,
            temp_waypoints,
            sign,
        )
        current_lane_marking = marking_type

        for marking in markings:
          markings_list.append(marking)

        temp_waypoints = temp_waypoints[-1:]

      else:
        temp_waypoints.append((sample))
        previous_marking_type = marking_type

    # Add last marking.
    last_markings = get_lane_markings(
        previous_marking_type,
        temp_waypoints,
        sign,
    )
    for marking in last_markings:
      markings_list.append(marking)

    for markings in markings_list:
      if markings[0] == carla.LaneMarkingType.Solid:  # pylint: disable=no-member
        draw_solid_line(
            surface,
            COLORS["WHITE"],
            False,
            markings[1],
            2,
        )
      elif markings[0] == carla.LaneMarkingType.Broken:  # pylint: disable=no-member
        draw_broken_line(
            surface,
            COLORS["WHITE"],
            False,
            markings[1],
            2,
        )

  # Set background black
  surface.fill(COLORS["BLACK"])
  precision = 0.05

  # Parse OpenDrive topology.
  topology = [x[0] for x in carla_map.get_topology()]
  topology = sorted(topology, key=lambda w: w.transform.location.z)
  set_waypoints = []
  for waypoint in topology:
    waypoints = [waypoint]

    nxt = waypoint.next(precision)
    if len(nxt) > 0:
      nxt = nxt[0]
      while nxt.road_id == waypoint.road_id:
        waypoints.append(nxt)
        nxt = nxt.next(precision)
        if len(nxt) > 0:
          nxt = nxt[0]
        else:
          break
    set_waypoints.append(waypoints)

  # Draw roads.
  for waypoints in set_waypoints:
    waypoint = waypoints[0]
    road_left_side = [
        lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints
    ]
    road_right_side = [
        lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints
    ]

    polygon = road_left_side + [x for x in reversed(road_right_side)]
    polygon = [
        world_to_pixel(
            x,
            scale=scale,
            offset=world_offset,
            pixels_per_meter=pixels_per_meter,
        ) for x in polygon
    ]

    # Draw Lane Markings
    if not waypoint.is_junction:
      # Left Side
      draw_lane_marking_single_side(surface, waypoints, -1)
      # Right Side
      draw_lane_marking_single_side(surface, waypoints, 1)

  return surface


def get_vehicles_surface(
    world: carla.World,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> pygame.Surface:
  """Generates a `PyGame` surface of other vehicles.

  Args:
    world: The `CARLA` world.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The vehicles rendered as a `PyGame` surface.
  """
  # Fetch CARLA map.
  carla_map = world.get_map()

  # Fetch all the vehicles.
  vehicles = [
      actor for actor in world.get_actors() if "vehicle" in actor.type_id
  ]

  # Setups the `PyGame` surface and offsets.
  world_offset, surface = draw_settings(
      carla_map=carla_map,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )

  # Set background black
  surface.fill(COLORS["BLACK"])

  # Iterate over vehicles.
  for vehicle in vehicles:
    # Draw pedestrian as a rectangular.
    corners = actor_to_corners(vehicle)
    # Convert to pixel coordinates.
    corners = [
        world_to_pixel(p, scale, world_offset, pixels_per_meter)
        for p in corners
    ]
    pygame.draw.polygon(surface, COLORS["WHITE"], corners)

  return surface


def get_pedestrians_surface(
    world: carla.World,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> pygame.Surface:
  """Generates a `PyGame` surface of pedestrians.

  Args:
    world: The `CARLA` world.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The pedestrians rendered as a `PyGame` surface.
  """
  # Fetch CARLA map.
  carla_map = world.get_map()

  # Fetch all the pedestrians.
  pedestrians = [
      actor for actor in world.get_actors()
      if "walker.pedestrian" in actor.type_id
  ]

  # Setups the `PyGame` surface and offsets.
  world_offset, surface = draw_settings(
      carla_map=carla_map,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )

  # Set background black
  surface.fill(COLORS["BLACK"])

  # Iterate over pedestrians.
  for pedestrian in pedestrians:
    # Draw pedestrian as a rectangular.
    corners = actor_to_corners(pedestrian)
    # Convert to pixel coordinates.
    corners = [
        world_to_pixel(p, scale, world_offset, pixels_per_meter)
        for p in corners
    ]
    pygame.draw.polygon(surface, COLORS["WHITE"], corners)

  return surface


def get_traffic_lights_surface(
    world: carla.World,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> Tuple[pygame.Surface, pygame.Surface, pygame.Surface]:
  """Generates three `PyGame` surfaces of traffic lights (Green, Yellow, Red).

  Args:
    world: The `CARLA` world.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The traffic lights rendered as `PyGame` surface (Green, Yellow, Red).
  """
  # Fetch CARLA map.
  carla_map = world.get_map()

  # Fetch all the pedestrians.
  traffic_lights = [
      actor for actor in world.get_actors()
      if "traffic.traffic_light" in actor.type_id
  ]

  # Setups the `PyGame` surface and offsets.
  world_offset, green_surface = draw_settings(
      carla_map=carla_map,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )
  width = green_surface.get_width()
  height = green_surface.get_height()
  yellow_surface = pygame.Surface((width, height))  # pylint: disable=too-many-function-args
  red_surface = pygame.Surface((width, height))  # pylint: disable=too-many-function-args

  # Set background black
  green_surface.fill(COLORS["BLACK"])
  yellow_surface.fill(COLORS["BLACK"])
  red_surface.fill(COLORS["BLACK"])

  # Iterate over pedestrians.
  for traffic_light in traffic_lights:
    # Identify state of the traffic light.
    if traffic_light.state.name == "Green":
      surface = green_surface
    elif traffic_light.state.name == "Yellow":
      surface = yellow_surface
    elif traffic_light.state.name == "Red":
      surface = red_surface
    else:
      continue
    # Convert to pixel coordinates.
    center = world_to_pixel(
        traffic_light.get_transform().location,
        scale,
        world_offset,
        pixels_per_meter,
    )
    pygame.draw.circle(surface, COLORS["WHITE"], center, 10)

  return green_surface, yellow_surface, red_surface


def get_hero_surface(
    world: carla.World,  # pylint: disable=no-member
    hero: carla.Vehicle,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> pygame.Surface:
  """Generates a `PyGame` surface of pedestrians.

  Args:
    world: The `CARLA` world.
    hero: The ego vehicle actor.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The ego vehicle rendered as a `PyGame` surface.
  """
  # Fetch CARLA map.
  carla_map = world.get_map()

  # Setups the `PyGame` surface and offsets.
  world_offset, surface = draw_settings(
      carla_map=carla_map,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )

  # Set background black
  surface.fill(COLORS["BLACK"])

  # Draw hero as a rectangular.
  corners = actor_to_corners(hero)
  # Convert to pixel coordinates.
  corners = [
      world_to_pixel(p, scale, world_offset, pixels_per_meter) for p in corners
  ]
  pygame.draw.polygon(surface, COLORS["WHITE"], corners)

  return surface


def actor_to_corners(actor: carla.Actor) -> Sequence[Tuple[float, float]]:  # pylint: disable=no-member
  """Draws a rectangular for the bounding box of `actor`."""
  bb = actor.bounding_box.extent
  # Draw actor as a rectangular.
  corners = [
      carla.Location(x=-bb.x, y=-bb.y),  # pylint: disable=no-member
      carla.Location(x=-bb.x, y=+bb.y),  # pylint: disable=no-member
      carla.Location(x=+bb.x, y=+bb.y),  # pylint: disable=no-member
      carla.Location(x=+bb.x, y=-bb.y),  # pylint: disable=no-member
  ]
  # Convert to global coordinates.
  actor.get_transform().transform(corners)

  return corners


def draw_game_state(
    world: carla.World,  # pylint: disable=no-member
    hero: Optional[carla.Vehicle] = None,  # pylint: disable=no-member
    pixels_per_meter: int = 5,
    scale: float = 1.0,
    margin: int = 150,
) -> pygame.Surface:
  """Generates a `PyGame` surface of the CARLA game state.

  Args:
    world: The `CARLA` world.
    hero: The ego vehicle actor.
    pixels_per_meter: The number of pixels rendered per meter.
    scale: The scaling factor of the rendered map.
    margin: The number of pixels used for margin.

  Returns
    The lane boundaries of a CARLA town as a `PyGame` surface.
  """
  # Draw individual surfaces.
  road_surface = get_road_surface(
      world=world,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )
  lane_boundaries_surface = get_lane_boundaries_surface(
      world=world,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )
  vehicles_surface = get_vehicles_surface(
      world=world,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )
  pedestrians_surface = get_pedestrians_surface(
      world=world,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )
  green_surface, yellow_surface, red_surface = get_traffic_lights_surface(
      world=world,
      pixels_per_meter=pixels_per_meter,
      scale=scale,
      margin=margin,
  )
  if hero is not None:
    hero_surface = get_hero_surface(
        world=world,
        hero=hero,
        pixels_per_meter=pixels_per_meter,
        scale=scale,
        margin=margin,
    )

  # Repaint surfaces.
  road_surface = repaint_surface(
      surface=road_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["SILVER"],
  )
  lane_boundaries_surface = repaint_surface(
      surface=lane_boundaries_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["GOOGLE YELLOW"],
  )
  vehicles_surface = repaint_surface(
      surface=vehicles_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["GOOGLE BLUE"],
  )
  pedestrians_surface = repaint_surface(
      surface=pedestrians_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["SLACK AUBERGINE"],
  )
  green_surface = repaint_surface(
      surface=green_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["GOOGLE GREEN"],
  )
  yellow_surface = repaint_surface(
      surface=yellow_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["GOOGLE YELLOW"],
  )
  red_surface = repaint_surface(
      surface=red_surface,
      old_color=COLORS["WHITE"],
      new_color=COLORS["GOOGLE RED"],
  )
  if hero is not None:
    hero_surface = repaint_surface(
        surface=hero_surface,
        old_color=COLORS["WHITE"],
        new_color=COLORS["SLACK BLUE"],
    )

  # Set z-ordering.
  surfaces = [
      road_surface,
      lane_boundaries_surface,
      green_surface,
      yellow_surface,
      red_surface,
      vehicles_surface,
      pedestrians_surface,
  ]

  # Append hero agent.
  if hero is not None:
    surfaces.append(hero_surface)

  # Merge surfaces.
  return merge_surfaces(
      surfaces=surfaces,
      background_color=COLORS["BLACK"],
  )


def repaint_surface(
    surface: pygame.Surface,
    old_color: pygame.Color,
    new_color: pygame.Color,
) -> pygame.Surface:
  """Replaces paints `surface` with `color`."""
  # Parses the boolean surface to a tensor.
  surf = pygame_surface_to_ndarray(surface)
  R = surf[..., 0]
  G = surf[..., 1]
  B = surf[..., 2]
  mask = (R == old_color.r) & (G == old_color.g) & (B == old_color.b)

  # Temprorary tensor.
  tmp = np.zeros(mask.shape + (3,), dtype=np.uint8)
  # Set R, G and B channels.
  tmp[mask, :3] = [new_color.r, new_color.g, new_color.b]

  return ndarray_to_pygame_surface(tmp, swapaxes=False)


def merge_surfaces(
    surfaces: Sequence[pygame.Surface],
    background_color: pygame.Color,
) -> pygame.Surface:
  """Merges all the surfaces, assuming same dimensions and z-ordering."""
  # Parse background color.
  R_B = background_color.r
  G_B = background_color.g
  B_B = background_color.b

  # Temporary tensor, used for merging.
  tmp = np.copy(pygame_surface_to_ndarray(surfaces[0]))

  # Iterate over planes.
  for surface in surfaces[1:]:
    # Parse image.
    image = pygame_surface_to_ndarray(surface)
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]
    # Mask of non-background pixels.
    mask = (R != R_B) + (G != G_B) + (B != B_B)
    # Overlay image.
    tmp[mask] = image[mask]

  return ndarray_to_pygame_surface(tmp, swapaxes=False)


def pngs_to_gif(dirname: str, output_filename: str) -> None:
  """Converts a list of PNGs found in `dirname` and sorted by name to a GIF."""
  pngs = sorted(glob.glob(os.path.join(dirname, "*.png")))
  with imageio.get_writer(output_filename, mode="I") as gif:
    for filename in tqdm.tqdm(pngs):
      gif.append_data(imageio.imread(filename))


def downsample(image: np.ndarray, *, factor: int) -> np.ndarray:
  """Downsamples an `image` by a `factor`."""
  return image[0::factor, 0::factor]


# Color palette, the RGB values found at https://brandpalettes.com/.
COLORS = {
    # Default palette.
    "WHITE": pygame.Color(255, 255, 255),
    "BLACK": pygame.Color(0, 0, 0),
    "RED": pygame.Color(255, 0, 0),
    "GREEN": pygame.Color(0, 255, 0),
    "BLUE": pygame.Color(0, 0, 255),
    "SILVER": pygame.Color(195, 195, 195),
    # Google palette.
    "GOOGLE BLUE": pygame.Color(66, 133, 244),
    "GOOGLE RED": pygame.Color(219, 68, 55),
    "GOOGLE YELLOW": pygame.Color(244, 160, 0),
    "GOOGLE GREEN": pygame.Color(15, 157, 88),
    # Apple palettte.
    "APPLE MIDNIGHT GREEN": pygame.Color(78, 88, 81),
    "APPLE SPACE GREY": pygame.Color(83, 81, 80),
    "APPLE ROSE GOLD": pygame.Color(250, 215, 189),
    "APPLE LIGHT PURPLE": pygame.Color(209, 205, 218),
    "APPLE LIGHT YELLOW": pygame.Color(255, 230, 129),
    "APPLE LIGHT GREEN": pygame.Color(255, 230, 129),
    "APPLE SILVER": pygame.Color(163, 170, 174),
    "APPLE BLACK": pygame.Color(31, 32, 32),
    "APPLE WHITE": pygame.Color(249, 246, 239),
    "APPLE RED": pygame.Color(165, 40, 44),
    "APPLE GOLD": pygame.Color(245, 221, 197),
    # Slack palette.
    "SLACK AUBERGINE": pygame.Color(74, 21, 75),
    "SLACK BLUE": pygame.Color(54, 197, 240),
    # Other palettes.
    "AMAZON ORANGE": pygame.Color(255, 153, 0),
    "FACEBOOK BLUE": pygame.Color(66, 103, 178),
    "AIRBNB CORAL": pygame.Color(255, 88, 93),
    "DR.PEPPER MAROON": pygame.Color(113, 31, 37),
}
