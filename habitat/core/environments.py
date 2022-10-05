#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@habitat.registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type

import numpy as np

import habitat
from habitat import Config, Dataset


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return habitat.registry.get_env(env_name)


@habitat.registry.register_env(name="RLTaskEnv")
class RLTaskEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.TASK.REWARD_MEASURE  # 'move_obj_reward'
        self._success_measure_name = self.config.TASK.SUCCESS_MEASURE   # 'composite_success'
        assert (
            self._reward_measure_name is not None
        ), "The key TASK.REWARD_MEASURE cannot be None"
        assert (
            self._success_measure_name is not None
        ), "The key TASK.SUCCESS_MEASURE cannot be None"

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self.config.TASK.SLACK_REWARD  # -0.01

        reward += current_measure

        if self._episode_success():
            reward += self.config.TASK.SUCCESS_REWARD  # 100

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if self.config.TASK.END_ON_SUCCESS and self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
