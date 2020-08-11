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
"""Constant, default value definitions."""

# CARLA simulator frames per second (FPS).
SIMULATOR_FPS = 20

# CARLA default sensor suite.
SENSORS = (
    "lidar",
    "location",
    "rotation",
    "velocity",
    "is_at_traffic_light",
    "traffic_light_state",
)

# The time interval before stopping the search for the carla server.
CLIENT_TIMEOUT = 20.0

# The time interval before stopping the search to the queue.
QUEUE_TIMEOUT = 2.0