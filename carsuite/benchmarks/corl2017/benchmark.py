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
from typing import Text

from carsuite.core.benchmark import Benchmark
from carsuite.core.rl import Metric
from carsuite.core.rl import SaveToDiskWrapper
from carsuite.core.rl import StepsMetric
from carsuite.envs.carla import CARLANavEnv
from carsuite.envs.carla import CollisionsMetric
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


class CORL2017(Benchmark):
  """The CORL2017 benchmark."""

  def load(
      self,
      task_id: Text,
  ) -> CARLANavEnv:
    """Loads a CORL2017 task.

    Args:
      task_id: The unique identifier of a task.
      max_episode_steps: The number of steps before termination, default `inf`.

    Returns:
      A task from the benchmark with `task_id`.
    """
    # TODO(filangel): figure out the correct horizon.
    env = super(CORL2017, self).load(task_id, max_episode_steps=1500)

    # Terminate on collision.
    env = TerminateOnCollisionWrapper(env)

    return env

  @property
  def tasks(self) -> Mapping[Text, Callable[..., CARLANavEnv]]:
    """Returns the list of tasks associated with the benchmark."""
    return {
        task_id: functools.partial(CARLANavEnv, **config)
        for (task_id, config) in _TASKS.items()
    }

  @property
  def metrics(self) -> Sequence[Metric]:
    """Returns the list of metrics associated with the benchmark."""
    return [StepsMetric(), CollisionsMetric(), LaneInvasionsMetric()]


corl2017 = CORL2017()
