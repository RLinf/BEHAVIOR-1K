from omnigibson.termination_conditions.termination_condition_base import FailureCondition


class Timeout(FailureCondition):
    """
    Timeout (failure condition)
    Episode terminates if max_step steps have passed

    Args:
        max_steps (int): Maximum number of episode steps before timeout occurs
    """

    def __init__(self, max_steps=500):
        # Store internal vars
        self._max_steps = max_steps

        # Run super
        super().__init__()

    def _step(self, task, env, action, env_indices=None):
        # Terminate if number of steps passed exceeds threshold
        # env.episode_steps is (num_envs,) tensor
        steps = env.episode_steps if env_indices is None else env.episode_steps[env_indices]
        return steps >= self._max_steps
