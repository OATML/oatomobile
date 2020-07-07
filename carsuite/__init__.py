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
"""Public API for `carsuite`."""

import os
import sys

from absl import logging

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

# Enable CARLA PythonAPI to be accessed from `carsuite`.
carla_path = os.getenv("CARLA_ROOT")
if carla_path is None:
  raise EnvironmentError(
      "Missing environment variable CARLA_ROOT, specify it before importing carsuite"
  )

logging.debug("CARLA_ROOT={}".format(carla_path))
carla_python_api = os.path.join(
    carla_path,
    "PythonAPI",
    "carla",
)
if not os.path.exists(carla_python_api):
  raise ImportError("Missing CARLA installation at {}".format(carla_python_api))
sys.path.append(carla_python_api)

from agents.navigation.controller import VehiclePIDController  # pylint: disable=import-error
from agents.navigation.local_planner import \
    LocalPlanner  # pylint: disable=import-error
from agents.tools.misc import \
    compute_magnitude_angle  # pylint: disable=import-error
from agents.tools.misc import draw_waypoints  # pylint: disable=import-error
from agents.tools.misc import get_speed  # pylint: disable=import-error
from agents.tools.misc import \
    is_within_distance_ahead  # pylint: disable=import-error

###############

# HACK(filangel): matplotlib setup - remove before release.
import matplotlib
matplotlib.use("Agg")

# Benchmarks API.
from carsuite.benchmarks.carnovel.benchmark import carnovel
# Core API.
from carsuite.core.agent import Agent
from carsuite.core.benchmark import Benchmark
from carsuite.core.dataset import Dataset
from carsuite.core.dataset import Episode
from carsuite.core.dataset import tokens
from carsuite.core.loop import EnvironmentLoop
from carsuite.core.registry import registry
from carsuite.core.rl import Env
from carsuite.core.rl import FiniteHorizonWrapper
from carsuite.core.rl import Metric
from carsuite.core.rl import MonitorWrapper
from carsuite.core.rl import ReturnsMetric
from carsuite.core.rl import SaveToDiskWrapper
from carsuite.core.rl import StepsMetric
from carsuite.core.simulator import Action
from carsuite.core.simulator import Observations
from carsuite.core.simulator import Sensor
from carsuite.core.simulator import SensorSuite
from carsuite.core.simulator import SensorTypes
from carsuite.core.simulator import Simulator

# Public API.
__all__ = (
    # CARLA Python API
    "compute_magnitude_angle",
    "draw_waypoints",
    "is_within_distance_ahead",
    "LocalPlanner",
    "VehiclePIDController",
    # CARSUITE core API
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
#  __________________________________________
# / Please don't use these symbols they      \
# \ are not part of the CARSUITE public API. /
#  ------------------------------------------
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
