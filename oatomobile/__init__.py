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
"""Public API for `oatomobile`."""

import os
import sys

from absl import logging

# Make __version__ accessible.
from oatomobile._metadata import __version__

# Core API.
from oatomobile import types
from oatomobile.core.agent import Agent
from oatomobile.core.benchmark import Benchmark
from oatomobile.core.dataset import Dataset
from oatomobile.core.dataset import Episode
from oatomobile.core.dataset import tokens
from oatomobile.core.loop import EnvironmentLoop
from oatomobile.core.registry import registry
from oatomobile.core.rl import Env
from oatomobile.core.rl import FiniteHorizonWrapper
from oatomobile.core.rl import Metric
from oatomobile.core.rl import MonitorWrapper
from oatomobile.core.rl import ReturnsMetric
from oatomobile.core.rl import SaveToDiskWrapper
from oatomobile.core.rl import StepsMetric
from oatomobile.core.simulator import Action
from oatomobile.core.simulator import Observations
from oatomobile.core.simulator import Sensor
from oatomobile.core.simulator import SensorSuite
from oatomobile.core.simulator import SensorTypes
from oatomobile.core.simulator import Simulator

###############
# CARLA SETUP #
###############

# HACK(filangel): resolves https://github.com/carla-simulator/carla/issues/2132.
try:
  import torch
except ImportError:
  pass
try:
  import sonnet as snt
except ImportError:
  pass

# Enable CARLA PythonAPI to be accessed from `oatomobile`.
carla_path = os.getenv("CARLA_ROOT")
if carla_path is None:
  logging.warn(
      "Missing environment variable CARLA_ROOT, "
      "if you want to use CARLA specify it before importing oatomobile")

logging.debug("CARLA_ROOT={}".format(carla_path))
carla_python_api = os.path.join(
    carla_path or "$CARLA_ROOT",
    "PythonAPI",
    "carla",
)
if not os.path.exists(carla_python_api):
  logging.warn("Missing CARLA installation at {}".format(carla_python_api))
else:
  sys.path.append(carla_python_api)

###################
# Matplotlib Hack #
###################

# Remove before release.
import matplotlib
matplotlib.use("Agg")

###################

# Public API.
__all__ = (
    # OATomobile core API
    "Agent",
    "Benchmark",
    "Dataset",
    "EnvironmentLoop",
    "Episode",
    "tokens",
    "registry",
    "Env",
    "FiniteHorizonWrapper",
    "Metric",
    "MonitorWrapper",
    "ReturnsMetric",
    "StepsMetric",
    "SaveToDiskWrapper",
    "Action",
    "Observations",
    "Sensor",
    "SensorSuite",
    "Simulator",
)

#  ASCII art borrowed from dm-haiku.
#  ____________________________________________
# / Please don't use these symbols they        \
# \ are not part of the OATomobile public API. /
#  --------------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
del os
del sys
del logging
del carla_path
del carla_python_api
try:
  del snt
except NameError:
  pass
try:
  del torch
except NameError:
  pass
