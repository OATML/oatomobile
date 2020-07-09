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
"""Loop definitions, inspired by DeepMind Acme."""

from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence

from absl import logging

from oatomobile.core.agent import Agent
from oatomobile.core.rl import Env
from oatomobile.core.rl import Metric
from oatomobile.core.typing import Scalar


class EnvironmentLoop:
  """A simple RL-like environment loop.

  This takes `Env` and `Agent` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, agent_fn)
    loop.run()
  """

  def __init__(
      self,
      agent_fn: Callable[..., Agent],
      environment: Env,
      metrics: Optional[Sequence[Metric]] = None,
      render_mode: str = "none",
  ) -> None:
    """Constructs an environment loop.

    Args:
      agent_fn: The agent's construction function that receives each task.
      environment: The environment to evaluate on.
      metrics: Set of metrics to record during episode.
      render_mode: The render mode, one of {"none", "human", "rgb_array"}.
    """
    assert render_mode in ("none", "human", "rgb_array")

    # Internalizes agent and environment.
    self._agent_fn = agent_fn
    self._environment = environment
    self._metrics = metrics
    self._render_mode = render_mode

  def run(self) -> Optional[Mapping[str, Scalar]]:
    """Perform the run loop.

    Returns:
      If `metrics` are provided, it returns their final values in a dictionary
      format.
    """

    try:
      # Initializes environment.
      done = False
      observation = self._environment.reset()
      if self._render_mode is not "none":
        self._environment.render(mode=self._render_mode)
      # Initializes agent.
      agent = self._agent_fn(environment=self._environment)

      # Episode loop.
      while not done:
        # Get vehicle control.
        action = agent.act(observation)

        # Progresses the simulation.
        new_observation, reward, done, _ = self._environment.step(action)
        if self._render_mode is not "none":
          self._environment.render(mode=self._render_mode)

        # Updates the agent belief.
        agent.update(observation, action, new_observation)

        # Update metrics.
        if self._metrics is not None:
          for metric in self._metrics:
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
      if self._metrics is not None:
        return {metric.uuid: metric.value for metric in self._metrics}
      else:
        return None
