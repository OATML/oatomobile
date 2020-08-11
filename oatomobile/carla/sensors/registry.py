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
"""A global registry of CARLA sensors."""

from oatomobile import core

_ALL_CARLA_SENSORS = core.Registry(allow_overriding_keys=False)

add = _ALL_CARLA_SENSORS.add
get_sensor = _ALL_CARLA_SENSORS.__getitem__
get_all_names = _ALL_CARLA_SENSORS.keys