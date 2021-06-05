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
"""Utilities for logging to Weights & Biases."""

import wandb
from absl import flags
from ray.tune.integration.wandb import WandbLogger

from oatomobile.utils.loggers import base

wandb.init(project="oatomobile", config=flags.FLAGS)


class WandBLogger(base.Logger, WandbLogger):
  """Logs to a `wandb` dashboard."""

  def write(self, values: base.LoggingData) -> None:
    wandb.log(values)
