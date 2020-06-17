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
"""Defines the core API for a dataset."""

import abc
import os
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from carsuite.util.uuid import unique_token_generator

tokens = unique_token_generator()


class Episode:
  """The abstract class for a `carsuite` episode."""

  def __init__(
      self,
      parent_dir: str,
      token: str,
  ) -> None:
    """Constructs an episode with a unique identifier."""
    self._parent_dir = parent_dir
    self._token = token

    # The path where the samples are stored/recovered.
    self._episode_dir = os.path.join(self._parent_dir, self._token)
    os.makedirs(self._episode_dir, exist_ok=True)

    # The episode's metadata.
    self._metadata_fname = os.path.join(self._episode_dir, "metadata")

  def append(self, **observations: np.ndarray) -> None:
    """Appends `observations` to the episode."""
    # Samples a random token.
    sample_token = next(tokens)

    # Stores the `NumPy` tensors on the disk.
    np.savez_compressed(
        os.path.join(
            self._episode_dir,
            "{}.npz".format(sample_token),
        ), **observations)

    # Checks if metadata exist, else inits.
    if not os.path.exists(self._metadata_fname):
      with open(self._metadata_fname, "w") as _:
        pass
    with open(self._metadata_fname, "a") as metadata:
      metadata.write("{}\n".format(sample_token))

  def fetch(self) -> Sequence[str]:
    """Returns all the sample tokens in order."""
    with open(self._metadata_fname, "r") as metadata:
      samples = metadata.read()
    # Removes trailing newlines.
    return list(filter(None, samples.split("\n")))

  def read_sample(
      self,
      sample_token: str,
      attr: Optional[str] = None,
  ) -> Union[Mapping[str, np.ndarray], np.ndarray]:
    """Loads and parses an observation or a single attribute.

    Args:
      sample_token: The sample id of the observations.
      attr: If `None` all the attributes are loaded, else
        only the specified key is parsed and returned.

    Returns:
      Either a complete observation or a single attribute.
    """
    with np.load(
        os.path.join(
            self._episode_dir,
            "{}.npz".format(sample_token),
        ),
        allow_pickle=True,
    ) as npz_file:
      if attr is not None:
        # Returns a single attribute.
        out = npz_file[attr]
      else:
        observation = dict()
        for _attr in npz_file:
          observation[_attr] = npz_file[_attr]
        # Returns all attributes.
        out = observation

    return out


class Dataset(abc.ABC):
  """The abstract class for a `carsuite` dataset."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Constructs a dataset."""
    self.uuid = self._get_uuid(*args, **kwargs)

  @abc.abstractmethod
  def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
    """Returns the universal unique identifier of the dataset."""

  @abc.abstractproperty
  def info(self) -> Mapping[str, Any]:
    """The dataset description."""

  @abc.abstractproperty
  def url(self) -> str:
    """The URL where the dataset is hosted."""

  @abc.abstractmethod
  def download_and_prepare(self, output_dir: str, *args: Any,
                           **kwargs: Any) -> None:
    """Downloads and prepares the dataset from the host URL.

    Args:
      output_dir: The absolute path where the prepared dataset is stored.
    """

  @abc.abstractstaticmethod
  def load_datum(fname: str, *args: Any, **kwargs: Any) -> Any:
    """Loads a datum from the dataset.

    Args:
      fname: The absolute path to the datum.

    Returns:
      The parsed datum, in a Python-friendly format.
    """

  @abc.abstractstaticmethod
  def plot_datum(fname: str, output_dir: str, *args: Any,
                 **kwargs: Any) -> None:
    """Visualizes a datum from the dataset.

    Args:
      fname: The absolute path to the datum.
      output_dir: The full path to the output directory.
    """
