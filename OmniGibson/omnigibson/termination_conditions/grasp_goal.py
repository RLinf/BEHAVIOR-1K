import torch as th

from omnigibson.termination_conditions.termination_condition_base import SuccessCondition


class GraspGoal(SuccessCondition):
    """
    GraspGoal (success condition)
    """

    def __init__(self, obj_name):
        self.obj_name = obj_name
        self._objs = None

        # Run super init
        super().__init__()

    def _step(self, task, env, action, env_indices=None):
        if self._objs is None:
            self._objs = [env.scenes[i].object_registry("name", self.obj_name) for i in range(env.num_envs)]

        indices = range(env.num_envs) if env_indices is None else env_indices
        results = th.zeros(len(indices), dtype=th.bool)
        for row, env_idx in enumerate(indices):
            obj = self._objs[env_idx]
            robot = env.scenes[env_idx].robots[0]
            obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
            results[row] = obj_in_hand == obj
        return results
