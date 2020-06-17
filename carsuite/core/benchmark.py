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
"""Defines the core API for benchmarks within `carsuite`."""

import abc
import os
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd
import tqdm
from absl import logging

from carsuite.core.agent import Agent
from carsuite.core.rl import Env
from carsuite.core.rl import FiniteHorizonWrapper
from carsuite.core.rl import Metric
from carsuite.core.rl import MonitorWrapper
from carsuite.core.typing import Scalar


class Benchmark(abc.ABC):
  """An abstract class for a benchmark in `carsuite`."""

  @abc.abstractproperty
  def metrics(self) -> Sequence[Metric]:
    """Returns the list of metrics associated with the benchmark."""

  @abc.abstractproperty
  def tasks(self) -> Mapping[str, Callable[..., Env]]:
    """Returns the list of tasks associated with the benchmark."""

  def load(self,
           task_id: str,
           max_episode_steps: Optional[int] = None,
           *args: Any,
           **kwargs: Any) -> Env:
    """Loads a task.

    Args:
      task_id: The unique identifier of a task.
      max_episode_steps: The number of steps before termination, default `inf`.

    Returns:
      A task from the benchmark with `task_id`.
    """
    if task_id not in self.tasks:
      raise ValueError("Unrecognised task with id {}".format(task_id))

    # Initializes environment.
    env = self.tasks[task_id](*args, **kwargs)

    if max_episode_steps is not None:
      env = FiniteHorizonWrapper(env, max_episode_steps=max_episode_steps)

    return env

  def evaluate(self,
               agent_fn: Callable[..., Agent],
               log_dir: str,
               render: bool = False,
               monitor: bool = False,
               subtasks_id: Optional[str] = None,
               *args: Any,
               **kwargs: Any) -> None:
    """Runs a full evaluation of an agent on the benchmark.

    Args:
      agent_fn: The agent's construction function that receives each task.
      log_dir: The full path to the directory where all the logs are kept.
      render: If True, it renders the display.
      monitor: If True, it stores the videos on the screen.
      subtasks_id: The subset of tasks to run, matches regex.
    """
    # Makes sure the output directory exists.
    os.makedirs(log_dir, exist_ok=True)

    # Keep only the tasks that have `subtasks`.
    tasks = self.tasks if subtasks_id is None else [
        task for task in self.tasks if subtasks_id in task
    ]

    # Evaluate on tasks, sequentially -- could be run on parallel too.
    for task_id in tqdm.tqdm(tasks):
      logging.debug("Start evaluation on task {}".format(task_id))
      task_dir = os.path.join(log_dir, task_id)
      os.makedirs(task_dir, exist_ok=True)

      # Load environment.
      env = self.load(task_id)
      if monitor:
        video_fname = os.path.join(task_dir, "video.gif")
        env = MonitorWrapper(env, output_fname=video_fname)

      # Initialize agent.
      agent = agent_fn(environment=env, *args, **kwargs)

      # Run episode and record metrics.
      results = self.run_episode(
          agent=agent,
          environment=env,
          metrics=self.metrics,
          render_mode="human" if render else "none",
      )

      # Dumps results in a CSV file.
      results = {uuid: [value] for (uuid, value) in results.items()}
      pd.DataFrame(results).to_csv(
          os.path.join(task_dir, "metrics.csv"),
          header=True,
          index=False,
      )

  @staticmethod
  def run_episode(*,
                  agent: Agent,
                  environment: Env,
                  metrics: Optional[Sequence[Metric]] = None,
                  render_mode: str = "none") -> Optional[Mapping[str, Scalar]]:
    """Runs an `agent` on an `environment` for a single episode.

    Args:
      agent: The agent under evaluation.
      environment: The environment to evaluate on.
      metrics: Set of metrics to record during episode.
      render_mode: The render mode, one of {"none", "human", "rgb_array"}.

    Returns:
      If `metrics` are provided, it returns their final values in a dictionary format.
    """
    assert render_mode in ("none", "human", "rgb_array")

    if metrics is not None:
      # Reset metrics to default values.
      for metric in metrics:
        metric.reset()

    try:
      # Initializes environment.
      done = False
      observation = environment.reset()
      if render_mode is not "none":
        environment.render(mode=render_mode)

      # Episode loop.
      while not done:
        # Get vehicle control.
        action = agent.act(observation)

        # Progresses the simulation.
        new_observation, reward, done, _ = environment.step(action)
        if render_mode is not "none":
          environment.render(mode=render_mode)

        # Updates the agent belief.
        agent.update(observation, action, new_observation)

        # Update metrics.
        if metrics is not None:
          for metric in metrics:
            metric.update(observation, action, reward, new_observation)

        # Book-keeping.
        observation = new_observation

    except Exception as msg:
      logging.error(msg)

    finally:
      # Garbage collector.
      try:
        environment.close()
      except NameError:
        pass

      # Returns the recorded metrics.
      if metrics is not None:
        return {metric.uuid: metric.value for metric in metrics}
      else:
        return None
