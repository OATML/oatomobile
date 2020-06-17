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
"""Implementation of CARNOVEL [1], the autonomous car novel-scene benchmark to
evaluate the robustness of driving agents to a suite of tasks involving
distribution shift. CARNOVEL is based on the CARLA simulator.

#### References

[1]: Angelos Filos*, Panagiotis Tigkas*, Rowan McAllister, Nicholas Rhinehart, Sergey Levine, Yarin Gal
     Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?
"""

import functools
import glob
import json
import os
from typing import Callable
from typing import Mapping
from typing import Sequence

from carsuite.core.benchmark import Benchmark
from carsuite.core.rl import Metric
from carsuite.core.rl import ReturnsMetric
from carsuite.core.rl import SaveToDiskWrapper
from carsuite.core.rl import StepsMetric
from carsuite.envs.carla import CARLANavEnv
from carsuite.envs.carla import CollisionsMetric
from carsuite.envs.carla import DistanceMetric
from carsuite.envs.carla import LaneInvasionsMetric
from carsuite.envs.carla import TerminateOnCollisionWrapper

_configs = glob.glob(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "*.json",
    ))
_TASKS = dict()
for _config in _configs:
  _task_id = os.path.basename(_config).replace(".json", "")
  with open(_config, "r") as _dict:
    _TASKS[_task_id] = json.load(_dict)


class CARNOVEL(Benchmark):
  """The CARNOVEL benchmark."""

  def load(
      self,
      task_id: str,
  ) -> CARLANavEnv:
    """Loads a CARNOVEL task.

    Args:
      task_id: The unique identifier of a task.
      max_episode_steps: The number of steps before termination, default `inf`.

    Returns:
      A task from the benchmark with `task_id`.
    """
    env = super(CARNOVEL, self).load(task_id, max_episode_steps=1500)

    # Terminate on collision.
    env = TerminateOnCollisionWrapper(env)

    return env

  @property
  def tasks(self) -> Mapping[str, Callable[..., CARLANavEnv]]:
    """Returns the list of tasks associated with the benchmark."""
    return {
        task_id: functools.partial(CARLANavEnv, **config)
        for (task_id, config) in _TASKS.items()
    }

  @property
  def metrics(self) -> Sequence[Metric]:
    """Returns the list of metrics associated with the benchmark."""
    return [
        StepsMetric(),
        CollisionsMetric(),
        LaneInvasionsMetric(),
        DistanceMetric(),
        ReturnsMetric(),
    ]

  def plot_benchmark(
      self,
      output_dir: str,
  ) -> None:
    """Visualizes all the tasks in a benchmark (A -> B).

    Args:
      output_dir: The full path to the output directory.
    """
    import signal
    import matplotlib.pyplot as plt
    import numpy as np
    import tqdm
    from carsuite.util import carla as cutil

    def world_to_pixel(
        location: np.ndarray,
        town: str,
        scale: float = 12.0,
    ) -> np.ndarray:
      """Converts CARLA world coordinates to pixel coordinates."""
      assert town in [
          "Town01",
          "Town02",
          "Town03",
          "Town04",
          "Town05",
      ]
      offset = {
          "Town01": (-52.059906005859375, -52.04995942115784),
          "Town02": (-57.459808349609375, 55.3907470703125),
          "Town03": (-207.43186950683594, -259.27125549316406),
          "Town04": (-565.26904296875, -446.1461181640625),
          "Town05": (-326.0448913574219, -257.8750915527344)
      }[town]
      return (location - offset) * scale

    # Creates the necessary output directory.
    os.makedirs(output_dir, exist_ok=True)

    for task_id in tqdm.tqdm(self.tasks):
      try:
        # Initialize environment and fetches origin->destination.
        env = self.load(task_id)
        town = env.simulator._town
        world = env.simulator.world
        origin = env.simulator.hero.get_transform()
        destination = env.unwrapped.simulator._destination

        # Gets global plan.
        waypoints, _, distances = cutil.global_plan(
            world,
            origin.location,
            destination.location,
        )

        # Converts locations to ego coordinates.
        pixel_coordinates = list()
        for waypoint in waypoints:
          coordinates = cutil.carla_xyz_to_ndarray(
              waypoint.transform.location)[:2]
          pixel_coordinates.append(world_to_pixel(coordinates, town=town))
        pixel_coordinates = np.asarray(pixel_coordinates)

        # Visualizes optimal task on CARLA map.
        fig, ax = plt.subplots(figsize=(15.0, 15.0))
        ax.imshow(
            plt.imread(
                os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    os.pardir,
                    os.pardir,
                    os.pardir,
                    "assets",
                    "maps",
                    "{}.png".format(town),
                )))
        cb = ax.scatter(
            pixel_coordinates[..., 0],
            pixel_coordinates[..., 1],
            cmap="RdYlBu_r",
            linewidth=0.1,
            marker=".",
            s=300,
            c=np.linspace(0, 1, len(pixel_coordinates)),
        )
        # Box around the task.
        center = np.mean(pixel_coordinates, axis=0)
        ax.set(
            title="{} | distance: {:.2f}".format(
                task_id,
                sum(distances),
            ),
            frame_on=False,
            xlim=[center[0] - 1000, center[0] + 1000],
            ylim=[center[1] - 1000, center[1] + 1000],
        )
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # fig.colorbar(cb, ax=ax)
        fig.savefig(
            os.path.join(output_dir, "{}.png".format(task_id)),
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )

      finally:
        env.close()


carnovel = CARNOVEL()
