from abc import ABCMeta, abstractmethod
from copy import deepcopy

import torch as th

from omnigibson.utils.python_utils import Registerable, classproperty

REGISTERED_REWARD_FUNCTIONS = dict()


class BaseRewardFunction(Registerable, metaclass=ABCMeta):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """

    def __init__(self):
        # Store internal vars that will be filled in at runtime
        self._num_envs = None
        self._reward = None
        self._info = None

    @abstractmethod
    def _step(self, task, env, action, env_indices=None):
        """
        Step the reward function and compute the reward at the current timestep. Overwritten by subclasses.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): One action row per selected scene
            env_indices (sequence of int, optional): Scene indices in action and result order

        Returns:
            2-tuple:
                - th.Tensor: (num_selected,) computed reward per env
                - list[dict]: reward-related information per env
        """
        raise NotImplementedError()

    def step(self, task, env, action, env_indices=None):
        """
        Step the reward function and compute the reward at the current timestep.

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            action (n-array): One action row per selected scene
            env_indices (sequence of int, optional): Scene indices in action and result order

        Returns:
            2-tuple:
                - th.Tensor: (num_selected,) computed reward per env
                - list[dict]: any reward-related information for this specific reward per env
        """
        if env_indices is None:
            rewards, self._info = self._step(task=task, env=env, action=action)
            self._reward = rewards.clone()
            return rewards, deepcopy(self._info)

        rewards, infos = self._step(task=task, env=env, action=action, env_indices=env_indices)
        self._reward[env_indices] = rewards
        for idx, info in zip(env_indices, infos):
            self._info[idx] = info
        return rewards, deepcopy(infos)

    def reset(self, task, env, env_indices):
        """
        Reward function-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
            env_indices (th.Tensor): Indices of environments to reset
        """
        if self._num_envs is None:
            self._num_envs = env.num_envs
            self._reward = th.zeros(self._num_envs, dtype=th.float32)
            self._info = [dict() for _ in range(self._num_envs)]
        self._reward[env_indices] = 0.0
        for idx in env_indices:
            self._info[idx] = dict()

    @property
    def reward(self):
        """
        Returns:
            th.Tensor: (num_envs,) current reward for this reward function
        """
        assert self._reward is not None, "At least one step() must occur before reward can be calculated!"
        return self._reward

    @property
    def info(self):
        """
        Returns:
            list[dict]: Current info for this reward function per env
        """
        assert self._info is not None, "At least one step() must occur before info can be calculated!"
        return self._info

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRewardFunction")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_REWARD_FUNCTIONS
        return REGISTERED_REWARD_FUNCTIONS
