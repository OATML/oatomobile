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
"""Core data structures for OATomobile."""

import abc
import collections
import contextlib
import os
import uuid
from typing import Any
from typing import Generator
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import dm_env
import numpy as np
from acme.types import NestedSpec

# Generic type definition for actions.
Action = Any


def _unique_token_generator() -> Generator[str, None, None]:
  """Generates unique (random) tokens, access by `next(gen)`."""
  while True:
    yield str(uuid.uuid4()).replace("-", "")


# OATomobile's unique identifier generator.
tokens = _unique_token_generator()


class Simulator(abc.ABC):
  """Simulates a physical environment."""

  @abc.abstractmethod
  def step(self, num_sub_steps: int = 1) -> None:
    """Updates the simulation state.

    Args:
      num_sub_steps: Optional number of times to repeatedly update the simulation
        state. Defaults to 1.
    """

  @abc.abstractmethod
  def time(self) -> float:
    """Returns the elapsed simulation time in seconds."""

  @abc.abstractmethod
  def timestep(self) -> float:
    """Returns the simulation timestep."""

  def set_control(self, control):
    """Sets the control signal for the vehicle."""
    raise NotImplementedError("set_control is not supported.")

  @contextlib.contextmanager
  def reset_context(self):
    """Context manager for resetting the simulation state.

    Sets the internal simulation to a default state when entering the block.

    ```python
    with simulator.reset_context():
      # Set joint and object positions.

    simulator.step()
    ```

    Yields:
      The `Simulator` instance.
    """
    try:
      self.reset()
    except SimulatorError:
      pass
    yield self
    self.after_reset()

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets internal variables of the simulator simulation."""

  @abc.abstractmethod
  def after_reset(self):
    """Runs after resetting internal variables of the physics simulation."""

  def close(self) -> None:
    """Closes the simulator down and controls connection to the server."""

  def render(self, mode: str = "rgb_array") -> Any:
    """Renders current state of the simulator."""
    raise NotImplementedError()


class SimulatorError(RuntimeError):
  """Raised if the state of the simulator simulation becomes divergent."""


class Task(abc.ABC):
  """Defines a task in a `control.Environment`."""

  @abc.abstractmethod
  def initialize_episode(self, simulator: Simulator) -> None:
    """Sets the state of the environment at the start of each episode.

    Called by `control.Environment` at the start of each episode *within*
    `simulator.reset_context()` (see the documentation for `base.Simulator`).

    Args:
      simulator: Instance of `Simulator`.
    """

  @abc.abstractmethod
  def before_step(self, action: Action, simulator: Simulator) -> None:
    """Updates the task from the provided action.

    Called by `control.Environment` before stepping the simulator engine.

    Args:
      action: numpy array or array-like action values, or a nested structure of
        such arrays. Should conform to the specification returned by
        `self.action_spec(simulator)`.
      simulator: Instance of `Simulator`.
    """

  def after_step(self, simulator: Simulator) -> None:
    """Optional method to update the task after the simulator engine has
    stepped.

    Called by `control.Environment` after stepping the simulator engine and before
    `control.Environment` calls `get_observation, `get_reward` and
    `get_termination`.

    The default implementation is a no-op.

    Args:
      simulator: Instance of `Simulator`.
    """

  @abc.abstractmethod
  def action_spec(self, simulator) -> NestedSpec:
    """Returns a specification describing the valid actions for this task.

    Args:
      simulator: Instance of `Simulator`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the action array(s) passed to `self.step`.
    """

  @abc.abstractmethod
  def get_observation(self, simulator: Simulator):
    """Returns an observation from the environment.

    Args:
      simulator: Instance of `Simulator`.
    """

  @abc.abstractmethod
  def get_reward(self, simulator: Simulator) -> float:
    """Returns a reward from the environment.

    Args:
      simulator: Instance of `Simulator`.
    """

  def get_termination(self, simulator: Simulator) -> bool:
    """If the episode should end, returns True, otherwise False."""
    return False

  @abc.abstractmethod
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


class Environment(dm_env.Environment):
  """Class for driving simulator-based reinforcement learning environments."""

  def __init__(
      self,
      simulator: Simulator,
      task: Task,
      time_limit: float = float("inf"),
      num_sub_steps: int = 1,
  ) -> None:
    """Initializes a new `Environment`.

    Args:
      simulator: Instance of `Simulator`.
      task: Instance of `Task`.
      time_limit: Optional `int`, maximum time for each episode in seconds. By
        default this is set to infinite.
      num_sub_steps: Optional number of physical time-steps in one control
        time-step.
    """

    # Internalize components.
    self._simulator = simulator
    self._task = task
    self._time_limit = time_limit
    self._num_sub_steps = num_sub_steps

    # Calculated properties.
    if self._time_limit == float("inf"):
      self._step_limit = float("inf")
    else:
      self._step_limit = self._time_limit / (self._simulator.timestep() *
                                             self._num_sub_steps)
    self._step_count = 0
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    """Starts a new episode and returns the first `TimeStep`."""

    # Book-keeping.
    self._reset_next_step = False
    self._step_count = 0

    # Reset simulator to a state.
    with self._simulator.reset_context():
      self._task.initialize_episode(self._simulator)

    # Fetch observation.
    observation = self._task.get_observation(self._simulator)

    return dm_env.restart(observation=observation)

  def step(self, action: Action) -> dm_env.TimeStep:
    """Updates the environment using the action and returns a `TimeStep`."""

    # Force reset.
    if self._reset_next_step:
      return self.reset()

    self._task.before_step(action, self._simulator)
    for _ in range(self._num_sub_steps):
      self._simulator.step()
    self._task.after_step(self._simulator)

    reward = self._task.get_reward(self._simulator)
    observation = self._task.get_observation(self._simulator)

    self._step_count += 1
    if self._step_count >= self._step_limit:
      done = True
    else:
      done = self._task.get_termination(self._simulator)

    if done:
      self._reset_next_step = True
      return dm_env.termination(observation=observation, reward=reward)
    else:
      return dm_env.transition(
          observation=observation,
          reward=reward,
          discount=1.0,
      )

  def action_spec(self) -> NestedSpec:
    """Returns the action specification for this environment."""
    return self._task.action_spec(self._simulator)

  def observation_spec(self) -> NestedSpec:
    """Returns the observation specification for this environment.

    Returns:
      An dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    return self._task.observation_spec(self._simulator)

  @property
  def simulator(self) -> Simulator:
    return self._simulator

  @property
  def task(self) -> Task:
    return self._task

  def control_timestep(self) -> float:
    """Returns the interval between agent actions in seconds."""
    return self.simulator.timestep() * self._num_sub_steps


class Episode:
  """The abstract class for a `OATomobile` episode."""

  def __init__(
      self,
      parent_dir: str,
      token: str,
  ) -> None:
    """Constructs an episode with a unique identifier."""
    self._parent_dir = parent_dir
    self._token = token

    # The path where the samples are stored/recovered.
    self._episode_dir = os.path.join(self._parent_dir, self._token)
    os.makedirs(self._episode_dir, exist_ok=True)

    # The episode's metadata.
    self._metadata_fname = os.path.join(self._episode_dir, "metadata")

  def append(self, **observations: np.ndarray) -> None:
    """Appends `observations` to the episode."""
    # Samples a random token.
    sample_token = next(tokens)

    # Stores the `NumPy` tensors on the disk.
    np.savez_compressed(
        os.path.join(
            self._episode_dir,
            "{}.npz".format(sample_token),
        ), **observations)

    # Checks if metadata exist, else inits.
    if not os.path.exists(self._metadata_fname):
      with open(self._metadata_fname, "w") as _:
        pass
    with open(self._metadata_fname, "a") as metadata:
      metadata.write("{}\n".format(sample_token))

  def fetch(self) -> Sequence[str]:
    """Returns all the sample tokens in order."""
    with open(self._metadata_fname, "r") as metadata:
      samples = metadata.read()
    # Removes trailing newlines.
    return list(filter(None, samples.split("\n")))

  def read_sample(
      self,
      sample_token: str,
      attr: Optional[str] = None,
  ) -> Union[Mapping[str, np.ndarray], np.ndarray]:
    """Loads and parses an observation or a single attribute.

    Args:
      sample_token: The sample id of the observations.
      attr: If `None` all the attributes are loaded, else
        only the specified key is parsed and returned.

    Returns:
      Either a complete observation or a single attribute.
    """
    with np.load(
        os.path.join(
            self._episode_dir,
            "{}.npz".format(sample_token),
        ),
        allow_pickle=True,
    ) as npz_file:
      if attr is not None:
        # Returns a single attribute.
        out = npz_file[attr]
      else:
        observation = dict()
        for _attr in npz_file:
          observation[_attr] = npz_file[_attr]
        # Returns all attributes.
        out = observation

    return out


class Dataset(abc.ABC):
  """The abstract class for a `OATomobile` dataset."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Constructs a dataset."""
    self.uuid = self._get_uuid(*args, **kwargs)

  @abc.abstractmethod
  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the dataset."""

  @abc.abstractproperty
  def info(self) -> Mapping[str, Any]:
    """The dataset description."""

  @abc.abstractproperty
  def url(self) -> str:
    """The URL where the dataset is hosted."""

  @abc.abstractmethod
  def download_and_prepare(
      self,
      output_dir: str,
      *args: Any,
      **kwargs: Any,
  ) -> None:
    """Downloads and prepares the dataset from the host URL.

    Args:
      output_dir: The absolute path where the prepared dataset is stored.
    """

  @abc.abstractstaticmethod
  def load_datum(
      fname: str,
      *args: Any,
      **kwargs: Any,
  ) -> Any:
    """Loads a datum from the dataset.

    Args:
      fname: The absolute path to the datum.

    Returns:
      The parsed datum, in a Python-friendly format.
    """

  @abc.abstractstaticmethod
  def plot_datum(
      fname: str,
      output_dir: str,
      *args: Any,
      **kwargs: Any,
  ) -> None:
    """Visualizes a datum from the dataset.

    Args:
      fname: The absolute path to the datum.
      output_dir: The full path to the output directory.
    """


_NAME_ALREADY_EXISTS = (
    "A function named {name!r} already exists in the container and "
    "`allow_overriding_keys` is False.")


class Registry(collections.Mapping):
  """Maps object names to their corresponding factory functions.

  To store a function in a `Registry` container, we can use its `.add`
  decorator:

  ```python
  registry = Registry()

  @registry.add("lidar")
  def make_lidar_sensor():
    ...
    return sensor

  sensor_factory = registry["lidar"]
  ```
  """

  def __init__(self, allow_overriding_keys=False):
    """Initializes a new `Registry` container.

    Args:
      allow_overriding_keys: Boolean, whether `add` can override existing keys
        within the container. If False (default), calling `add` multiple times
        with the same function name will result in a `ValueError`.
    """
    self._functions = collections.OrderedDict()
    self.allow_overriding_keys = allow_overriding_keys

  def add(self, name):
    """Decorator that adds a factory function to the container with name.

    Args:
      name: String specifying the name for this function.

    Returns:
      The same function.

    Raises:
      ValueError: if a function with the same name already exists within the
        container and `allow_overriding_keys` is False.
    """

    def wrap(factory_func):
      if name in self and not self.allow_overriding_keys:
        raise ValueError(_NAME_ALREADY_EXISTS.format(name=name))
      self._functions[name] = factory_func
      return factory_func

    return wrap

  def __getitem__(self, k):
    return self._functions[k]

  def __iter__(self):
    return iter(self._functions)

  def __len__(self):
    return len(self._functions)

  def __repr__(self):
    return "{}({})".format(self.__class__.__name__, str(self._functions))