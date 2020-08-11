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
  logging.warn("Missing CARLA PythonAPI at {}".format(carla_python_api))
else:
  sys.path.append(carla_python_api)
carla_python_wheel = os.path.join(
    carla_path or "$CARLA_ROOT",
    "PythonAPI",
    "carla",
    "dist",
    "carla-0.9.6-py3.5-linux-x86_64.egg",
)
if not os.path.exists(carla_python_wheel):
  logging.warn("Missing CARLA wheel at {}".format(carla_python_wheel))
else:
  sys.path.append(carla_python_wheel)

###################
# Matplotlib Hack #
###################

# Remove before release.
import matplotlib

matplotlib.use("Agg")

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
del carla_python_wheel
try:
  del matplotlib
except NameError:
  pass
try:
  del snt
except NameError:
  pass
try:
  del torch
except NameError:
  pass
