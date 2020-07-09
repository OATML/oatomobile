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
"""Registry is central source of truth in `oatomobile` (borrowed from
Habitat)."""

import collections
from typing import Optional

from absl import logging

from oatomobile.core.typing import Singleton


class Registry(metaclass=Singleton):
  """The `oatomobile` singleton registry object definition."""

  mapping = collections.defaultdict(dict)

  @classmethod
  def _register_impl(cls, _type, to_register, name, assert_type=None):

    def wrap(to_register):
      if assert_type is not None:
        assert issubclass(
            to_register,
            assert_type,
        ), "{} must be a subclass of {}".format(
            to_register,
            assert_type,
        )
      register_name = to_register.__name__ if name is None else name
      logging.debug("Registers {} at {}".format(register_name, _type))

      cls.mapping[_type][register_name] = to_register
      return to_register

    if to_register is None:
      return wrap
    else:
      return wrap(to_register)

  @classmethod
  def _get_impl(cls, _type, name):
    logging.debug("Fetches {} from {}".format(name, _type))
    return cls.mapping[_type].get(name, None)

  @classmethod
  def register_simulator(
      cls,
      to_register=None,
      name: Optional[str] = None,
  ):
    """Register a simulator to registry with key `name`."""
    from oatomobile.core.simulator import Simulator
    return cls._register_impl(
        "simulators",
        to_register,
        name,
        assert_type=Simulator,
    )

  @classmethod
  def register_sensor(
      cls,
      to_register=None,
      name: Optional[str] = None,
  ):
    """Register a sensor to registry with key `name`."""
    from oatomobile.core.simulator import Sensor
    return cls._register_impl(
        "sensors",
        to_register,
        name,
        assert_type=Sensor,
    )

  @classmethod
  def register_env(
      cls,
      to_register=None,
      name: Optional[str] = None,
  ):
    """Register a environemnt to registry with key `name`."""
    from oatomobile.core.rl import Env
    return cls._register_impl(
        "envs",
        to_register,
        name,
        assert_type=Env,
    )

  @classmethod
  def get_simulator(cls, name: str):
    """Fetches a registered simulator.

    Args:
      name: The uuid of the registered simulator.

    Returns:
      The `name`d registered simulator.
    """
    return cls._get_impl("simulators", name)

  @classmethod
  def get_sensor(cls, name: str):
    """Fetches a registered sensor.

    Args:
      name: The uuid of the registered sensor.

    Returns:
      The `name`d registered sensor.
    """
    return cls._get_impl("sensors", name)

  @classmethod
  def get_env(cls, name: str):
    """Fetches a registered environment.

    Args:
      name: The uuid of the registered environment.

    Returns:
      The `name`d registered environment.
    """
    return cls._get_impl("envs", name)


# Initializes the singleton registry for `oatomobile`.
registry = Registry()
