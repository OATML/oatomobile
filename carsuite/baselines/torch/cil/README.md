# Conditional Imitation Learning (CIL)

This folder contains an implementation of the CIL agent ([Codevilla et al., 2017]).

Script                   | Description
-------------------------| -----------------------------------------------------------
[`agent.py`](./agent.py) | defines the Conditional Imitation Learner (`CILAgent`)
[`model.py`](./model.py) | defines the behavioural cloning model (`BehaviouralModel`)
[`train.py`](./train.py) | trains a `BehaviouralModel` on expert demonstrations

[Codevilla et al., 2017]: https://arxiv.org/abs/1710.02410