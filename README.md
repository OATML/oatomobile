# CARSUITE: A research framework for autonomous driving

  **[Overview](#overview)**
| **[Installation](#installation)**
| **[Baselines]**
| **[Paper]**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/carsuite)
![PyPI version](https://badge.fury.io/py/carsuite.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2006.14911-b31b1b.svg)](https://arxiv.org/abs/2006.14911)


CARSUITE is a library for autonomous driving research.
CARSUITE strives to expose simple, efficient, well-tuned and readable agents, that serve both as reference implementations of popular algorithms and as strong baselines, while still providing enough flexibility to do novel research.

## Overview

If you just want to get started using CARSUITE quickly, the first thing to know about the framework is that we wrap [CARLA] towns and scenarios in OpenAI [gym]s:

```python
import carsuite

# Initializes a CARLA environment.
environment = carsuite.envs.CARLAEnv(town="Town01")

# Makes an initial observation.
observation = environment.reset()
done = False

while not done:
  # Selects a random action.
  action = environment.action_space.sample()
  observation, reward, done, info = environment.step(action)
  
  # Renders interactive display.
  environment.render(mode="human")

# Book-keeping: closes 
environment.close()
```

[Baselines] can also be used out-of-the-box:

```python
# Rule-based agents.
import carsuite.baselines.rulebased

agent = carsuite.baselines.rulebased.AutopilotAgent(environment)
action = agent.act(observation)

# Imitation-learners.
import torch
import carsuite.baselines.torch

models = [carsuite.baselines.torch.ImitativeModel() for _ in range(4)]
ckpts = ... # Paths to the model checkpoints.
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
agent = carsuite.baselines.torch.RIPAgent(
  environment=environment,
  models=models,
  algorithm="WCM",
)
action = agent.act(observation)
```

## Installation

We have tested CARSUITE on Python 3.5.

1.  To install the core libraries (including [CARLA], the backend simulator):

    ```bash
    # The path to download CARLA 0.9.6.
    export CARLA_ROOT=...
    mkdir -p $CARLA_ROOT

    # Downloads hosted binaries.
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz

    # CARLA 0.9.6 installation.
    tar -xvzf CARLA_0.9.6.tar.gz -C $CARLA_ROOT

    # Installs CARLA 0.9.6 Python API.
    easy_install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
    ```

1.  To install the CARSUITE core API:

    ```bash
    pip install --upgrade pip setuptools
    pip install carsuite
    ```

1.  To install dependencies for our [PyTorch]- or [TensorFlow]-based agents:

    ```bash
    pip install carsuite[torch]
    # and/or
    pip install carsuite[tf]
    ```

## Citing CARSUITE

If you use CARSUITE in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@inproceedings{filos2020can,
    title={Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?},
    author={Filos, Angelos and
            Tigas, Panagiotis and
            McAllister, Rowan and
            Rhinehart, Nicholas and
            Levine, Sergey and
            Gal, Yarin},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2020}
}
```

[Baselines]: carsuite/baselines/
[Examples]: examples/
[CARLA]: https://carla.readthedocs.io/
[Paper]: https://arxiv.org/abs/2006.14911
[TensorFlow]: https://tensorflow.org
[PyTorch]: http://pytorch.org
[gym]: https://github.com/openai/gym
