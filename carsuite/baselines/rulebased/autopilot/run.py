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
"""Runs the autopilot on CARLA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import carsuite
from carsuite.baselines.rulebased.autopilot.agent import AutopilotAgent

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_enum(
    name="town",
    default="Town01",
    enum_values=[
        "Town01",
        "Town02",
        "Town03",
        "Town04",
        "Town05",
    ],
    help="The CARLA town id.",
)
flags.DEFINE_list(
    name="sensors",
    default=[
        "velocity",
        "bird_view_camera_cityscapes",
        "bird_view_camera_rgb",
        "front_camera_rgb",
        "lidar",
    ],
    help="The list of recorded sensors.",
)
flags.DEFINE_integer(
    name="max_episode_steps",
    default=None,
    help="The number of steps in the simulator.",
)
flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="The full path to the output directory.",
)
flags.DEFINE_bool(
    name="render",
    default=False,
    help="If True it spawn the `PyGame` display.",
)


def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  town = FLAGS.town
  sensors = FLAGS.sensors
  max_episode_steps = FLAGS.max_episode_steps
  output_dir = FLAGS.output_dir
  render = FLAGS.render

  try:
    # Setups the environment.
    env = carsuite.envs.CARLAEnv(
        town=town,
        fps=20,
        sensors=sensors,
    )
    if max_episode_steps is not None:
      env = carsuite.FiniteHorizonWrapper(
          env,
          max_episode_steps=max_episode_steps,
      )
    if output_dir is not None:
      env = carsuite.SaveToDiskWrapper(env, output_dir=output_dir)
    env = carsuite.MonitorWrapper(env, output_fname="tmp/yoo.gif")

    # Initializes the agent.
    agent = AutopilotAgent(environment=env)
    carsuite.EnvironmentLoop(
        agent=agent,
        environment=env,
        render_mode="human" if render else "none",
    ).run()

  finally:
    # Garbage collector.
    try:
      env.close()
    except NameError:
      pass


if __name__ == "__main__":
  app.run(main)
