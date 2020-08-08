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
"""Default values for some parameters of the core API."""

import copy
import os

# CARLA simulator frames per second (FPS).
SIMULATOR_FPS = 20

# The front RGB camera configuration.
FRONT_CAMERA_RGB_SENSOR_CONFIG = {
    "attributes": {
        "image_size_x": "320",
        "image_size_y": "180",
        "fov": "90",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 2.3,
        },
        "rotation": {
            "pitch": 0,
        }
    },
}
# The rear RGB camera configuration.
REAR_CAMERA_RGB_SENSOR_CONFIG = {
    "attributes": {
        "image_size_x": "320",
        "image_size_y": "180",
        "fov": "90",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 2.3,
        },
        "rotation": {
            "pitch": 0,
            "yaw": 180,
        }
    },
}
# The left RGB camera configuration.
LEFT_CAMERA_RGB_SENSOR_CONFIG = {
    "attributes": {
        "image_size_x": "320",
        "image_size_y": "180",
        "fov": "90",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 2.3,
        },
        "rotation": {
            "pitch": 0,
            "yaw": 270,
        }
    },
}
# The right RGB camera configuration.
RIGHT_CAMERA_RGB_SENSOR_CONFIG = {
    "attributes": {
        "image_size_x": "320",
        "image_size_y": "180",
        "fov": "90",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 2.3,
        },
        "rotation": {
            "pitch": 0,
            "yaw": 90,
        }
    },
}

# The bird-view RGB/CityScapes camera configuration.
BIRD_VIEW_CAMERA_RGB_SENSOR_CONFIG = {
    "attributes": {
        "image_size_x": "200",
        "image_size_y": "200",
        "fov": "90",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 25.0,
        },
        "rotation": {
            "pitch": 270,
        }
    },
}
BIRD_VIEW_CAMERA_CITYSCAPES_SENSOR_CONFIG = copy.deepcopy(
    BIRD_VIEW_CAMERA_RGB_SENSOR_CONFIG)

# The LIDAR configuration.
LIDAR_SENSOR_CONFIG = {
    "attributes": {
        "range": "5000",
        "points_per_second": str(SIMULATOR_FPS * 10000),
        "rotation_frequency": str(SIMULATOR_FPS),
        "upper_fov": "10",
        "lower_fov": "-30",
    },
    "actor": {
        "location": {
            "x": 0.0,
            "y": 0.0,
            "z": 2.5,
        },
    },
}

# The goal sensor configuration.
GOAL_SENSOR_CONFIG = {
    "num_goals": 10,
    "sampling_radius": 2.0,
    "replan_every_steps": 5,
}

# The game state configuration.
GAME_STATE_CONFIG = {
    "margin": 150,
    "scale": 1.0,
    "pixels_per_meter": 5,
}

# Default sensors.
CARLA_SENSORS = (
    "goal",
    "front_camera_rgb",
    "lidar",
    "bird_view_camera_cityscapes",
    "bird_view_camera_rgb",
    "control",
    "location",
    "rotation",
    "velocity",
    "collision",
    "lane_invasion",
    "is_at_traffic_light",
    "traffic_light_state",
)

# The time interval before stopping the search for the carla server.
CARLA_CLIENT_TIMEOUT = 20.0

# The time interval before stopping the search to the queue.
QUEUE_TIMEOUT = 2.0

# Available CARLA towns.
AVAILABLE_CARLA_TOWNS = (
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
)

# Speed configuration of autopilot.
TARGET_SPEED = 20.0

# The number of simulator steps before termination.
MAX_EPISODE_STEPS = int(1e4)
