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
"""Defines the core API for `gym.Env` and driving simulators."""

import abc
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Tuple

import gym
import imageio
import numpy as np
import tqdm

from oatomobile.core.dataset import Episode
from oatomobile.core.dataset import tokens
from oatomobile.core.simulator import Action
from oatomobile.core.simulator import Observations
from oatomobile.core.simulator import Simulator

# OpenAI Gym transition.
Transition = Tuple[Observations, float, bool, Mapping[str, Any]]


class Env(gym.Env):
  """Fundamental environment class for `oatomobile` that implements the OpenAI
  Gym interface, wrapping a driving simulator."""

  def __init__(self, sim_fn: Callable[..., Simulator], *args: Any,
               **kwargs: Any) -> None:
    """Constructs an environment, that wraps a driving simulator."""
    self._sim = sim_fn(*args, **kwargs)
    self._reset_next_step = True

  @property
  def simulator(self):
    """Returns a reference to the simulator object."""
    return self._sim

  @property
  def observation_space(self) -> gym.spaces.Dict:
    """Returns the expected observation specification returned by `step`
    method."""
    return self.simulator.observation_space

  def reset(self, *args: Any, **kwargs: Any) -> Observations:
    """Resets the state of the simulation to the initial state.

    Returns:
      The initial transition.
    """
    self._reset_next_step = False
    return self.simulator.reset(*args, **kwargs)

  def step(self, action: Action, *args: Any, **kwargs: Any) -> Transition:
    """Makes a step in the simulator, provided an action.

    Args:
      action: The hero vehicle's actions.

    Returns:
      The current timestep.
    """
    if self._reset_next_step:
      return self.reset()

    # Step the simulator.
    observation = self.simulator.step(action, *args, **kwargs)

    # Dummy return values.
    reward = 0.0
    done = False
    info = dict()

    return observation, reward, done, info

  def render(self, mode: str = "human", *args: Any, **kwargs: Any) -> Any:
    """Renders current state of the simulator."""
    return self.simulator.render(mode=mode, *args, **kwargs)

  def close(self) -> None:
    """Closes the simulator down and controls connection to CARLA server."""
    self.simulator.close()


class Metric(abc.ABC):
  """Abstract class used for metrics in `oatomobile`."""

  def __init__(self, initial_value: float, *args: Any, **kwargs: Any) -> None:
    """Initializes a stateful metric.

    Args:
      initial_value: The default/initial value of the metric.
    """
    self._initial_value = initial_value

    self.value = self._initial_value
    self.uuid = self._get_uuid(*args, **kwargs)

  def __repr__(self):
    """The universal string representation of all metrics."""
    return "{}: {}".format(self.uuid, self.value)

  def reset(self) -> None:
    """Resets `value` to initial value."""
    self.value = self._initial_value

  @abc.abstractmethod
  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the metric."""

  @abc.abstractclassmethod
  def update(self, observations: Observations, action: Action, reward: float,
             new_observations: Observations, *args: Any, **kwargs: Any) -> None:
    """Records transition and update evaluation."""


class StepsMetric(Metric):
  """Counts the number of steps in an environment."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes the step counter."""
    super(StepsMetric, self).__init__(initial_value=0)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the metric."""
    return "steps"

  def update(self, observations: Observations, action: Action, reward: float,
             new_observations: Observations, *args: Any, **kwargs: Any) -> None:
    """Records transition and update evaluation."""
    self.value += 1


class ReturnsMetric(Metric):
  """Counts the cumulative undiscounted rewards in an episode."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes the returns counter."""
    super(ReturnsMetric, self).__init__(initial_value=0.0)

  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the metric."""
    return "returns"

  def update(self, observations: Observations, action: Action, reward: float,
             new_observations: Observations, *args: Any, **kwargs: Any) -> None:
    """Records transition and update evaluation."""
    self.value += reward


class FiniteHorizonWrapper(gym.Wrapper):
  """Terminates simulation after specified number of steps."""

  def __init__(self, env: gym.Env, *, max_episode_steps: int) -> None:
    """Constructs a gym wrapper to terminate execution after
    `max_episode_steps` steps."""
    super(FiniteHorizonWrapper, self).__init__(env=env)

    # Number of steps before termination and the counter.
    self._max_episode_steps = max_episode_steps
    self._episode_step = 0
    self._pbar = tqdm.tqdm(total=self._max_episode_steps)

  def reset(self, *args: Any, **kwargs: Any) -> Observations:
    """Resets the wrapped environment and sets the counter to 0."""
    self._episode_step = 0
    self._pbar.reset()
    return self.env.reset(*args, **kwargs)

  def step(self, action: Action, *args: Any, **kwargs: Any) -> Transition:
    """Steps the wrapped environment, increments the counter and terminates if
    the maximum number of steps is reached."""
    observation, reward, done, info = self.env.step(action)
    self._episode_step += 1
    self._pbar.update(n=1)
    if self._episode_step >= self._max_episode_steps:
      done = True
    return observation, reward, done, info


class SaveToDiskWrapper(gym.Wrapper):
  """Stores observations to the disk."""

  def __init__(self, env: gym.Env, *, output_dir: str) -> None:
    """Constructs a gym wrapper to store observations to the disk."""
    super(SaveToDiskWrapper, self).__init__(env=env)

    # The parent directory to store the observations.
    self._output_dir = output_dir
    self._episode = None

  def reset(self, *args: Any, **kwargs: Any) -> Observations:
    """Resets the wrapped environment and initializes a new episode."""
    # Initializes a new episode.
    self._episode = Episode(self._output_dir, next(tokens))
    observation = self.env.reset(*args, **kwargs)
    self._episode.append(**observation)
    return observation

  def step(self, action: Action, *args: Any, **kwargs: Any) -> Transition:
    """Steps the wrapped environment, increments the counter and terminates if
    the maximum number of steps is reached."""
    observation, reward, done, info = self.env.step(action)
    self._episode.append(**observation)
    return observation, reward, done, info


class MonitorWrapper(gym.Wrapper):
  """Records a video of the episode."""

  def __init__(self,
               env: gym.Env,
               *,
               output_fname: str,
               downsample_factor: int = 1) -> None:
    """Constructs a gym wrapper to record a video of the episode."""
    super(MonitorWrapper, self).__init__(env=env)

    # The parent directory to store the video.
    self._output_fname = output_fname
    self._downsample_factor = downsample_factor
    self._recorder = imageio.get_writer(self._output_fname, mode="I")

  def reset(self, *args: Any, **kwargs: Any) -> Observations:
    """Resets the wrapped environment and initializes a new episode."""
    observation = self.env.reset(*args, **kwargs)
    self._record_frame()
    return observation

  def step(self, action: Action, *args: Any, **kwargs: Any) -> Transition:
    """Steps the wrapped environment, increments the counter and terminates if
    the maximum number of steps is reached."""
    observation, reward, done, info = self.env.step(action)
    self._record_frame()
    return observation, reward, done, info

  def _record_frame(self):
    """Appends a `frame` in the video."""
    from oatomobile.util import graphics as gutil
    frame = gutil.downsample(
        image=self.render(mode="rgb_array"),
        factor=self._downsample_factor,
    )
    self._recorder.append_data(frame)
