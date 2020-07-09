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
"""Defines a simple blind agent that follows the goals."""

import numpy as np

import oatomobile
from oatomobile.baselines.base import SetPointAgent


class BlindAgent(SetPointAgent):
  """A simple blind agent that follows the goals."""

  def __call__(self, observation: oatomobile.Observations) -> np.ndarray:
    """Returns a trajectory that connects current location and next goal."""

    return np.asarray(observation["goal"])
