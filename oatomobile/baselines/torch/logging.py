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
"""Helper functions for logging, saving and loading models in `PyTorch`."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from oatomobile.baselines.torch.typing import ArrayLike

COLORS = [
    "#0071bc",
    "#d85218",
    "#ecb01f",
    "#7d2e8d",
    "#76ab2f",
    "#4cbded",
    "#a1132e",
]

# Determines device, accelerator.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member


class Checkpointer:
  """A simple `PyTorch` model load/save wrapper."""

  def __init__(
      self,
      model: nn.Module,
      ckpt_dir: str,
  ) -> None:
    """Constructs a simple load/save checkpointer."""
    self._model = model
    self._ckpt_dir = ckpt_dir
    os.makedirs(self._ckpt_dir, exist_ok=True)

  def save(
      self,
      epoch: int,
  ) -> str:
    """Saves the model to the `ckpt_dir/epoch/model.pt` file."""
    ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
    torch.save(self._model.state_dict(), ckpt_path)
    return ckpt_path

  def load(
      self,
      epoch: int,
  ) -> nn.Module:
    """Loads the model from the `ckpt_dir/epoch/model.pt` file."""
    ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
    self._model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return self._model


class TensorBoardWriter:
  """A simple `Pytorch`-friendly `TensorBoard` wrapper."""

  def __init__(
      self,
      log_dir: str,
  ) -> None:
    """Constructs a simgple `TensorBoard` wrapper."""
    # Makes sure output directories exist.
    log_dir_train = os.path.join(log_dir, "train")
    log_dir_val = os.path.join(log_dir, "val")
    os.makedirs(log_dir_train, exist_ok=True)
    os.makedirs(log_dir_val, exist_ok=True)

    # Initialises the `TensorBoard` writters.
    self._summary_writter_train = SummaryWriter(log_dir=log_dir_train)
    self._summary_writter_val = SummaryWriter(log_dir=log_dir_val)

  def log(
      self,
      split: str,
      loss: float,
      global_step: int,
      overhead_features: Optional[ArrayLike] = None,
      predictions: Optional[ArrayLike] = None,
      ground_truth: Optional[ArrayLike] = None,
  ) -> None:
    """Logs the scalar loss and visualizes predictions for qualitative
    inspection."""

    if split == "train":
      summary_writter = self._summary_writter_train
    elif split == "val":
      summary_writter = self._summary_writter_val
    else:
      raise ValueError("Unrecognised split={} was passed".format(split))

    # Logs the training loss.
    summary_writter.add_scalar(
        tag="loss",
        scalar_value=loss,
        global_step=global_step,
    )

    if overhead_features is not None:
      # Visualizes the predictions.
      overhead_features = np.transpose(overhead_features,
                                       (0, 2, 3, 1))  # to NHWC
      raw = list()
      for _, (
          o_t,
          p_t,
          g_t,
      ) in enumerate(zip(
          overhead_features,
          predictions,
          ground_truth,
      )):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        bev_meters = 25.0
        # Overhead features.
        ax.imshow(
            o_t.squeeze()[..., 0],
            extent=(-bev_meters, bev_meters, bev_meters, -bev_meters),
            cmap="gray",
        )
        # Ground truth.
        ax.plot(
            g_t[..., 1],
            -g_t[..., 0],
            marker="o",
            markersize=4,
            color=COLORS[1],
            alpha=1.0,
            label="ground truth",
        )
        # Model prediction.
        ax.plot(
            p_t[..., 1],
            -p_t[..., 0],
            marker="o",
            markersize=4,
            color=COLORS[0],
            alpha=0.75,
            label="predictions",
        )
        ax.legend()
        ax.set(frame_on=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Convert `matplotlib` canvas to `NumPy` RGB.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        raw.append(buf)
        plt.close(fig)
      raw = np.reshape(np.asarray(raw), (-1, w, h, 4))
      summary_writter.add_images(
          tag="examples",
          img_tensor=raw[..., :3],
          global_step=global_step,
          dataformats="NHWC",
      )
