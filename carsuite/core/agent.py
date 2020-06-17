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
"""Base implementation of agent inside `carsuite`."""

import abc
from typing import Any

from carsuite.core.rl import Env
from carsuite.core.simulator import Action
from carsuite.core.simulator import Observations


class Agent(abc.ABC):
  """An agent consists of an action-selection mechanism and an update rule."""

  def __init__(self, environment: Env, *args: Any, **kwargs: Any) -> None:
    """Constructs an agent."""
    self._environment = environment

  @abc.abstractmethod
  def act(
      self,
      observations: Observations,
  ) -> Action:
    """Samples an action from agent's policy, given observations."""

  def update(
      self,
      observations: Observations,
      action: Action,
      new_observations: Observations,
  ) -> None:
    """Updates the agent given a transition."""
    del observations
    del action
    del new_observations
