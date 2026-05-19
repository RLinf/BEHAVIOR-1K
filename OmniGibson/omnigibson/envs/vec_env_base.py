import copy

from tqdm import trange

import omnigibson as og


class VectorEnvironment:
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        if og.sim is not None:
            og.sim.stop()

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        self.envs = [
            og.Environment(configs=copy.deepcopy(config), in_vec_env=True)
            for _ in trange(num_envs, desc="Loading environments")
        ]

        # Play, and finish loading all the envs
        og.sim.play()
        for env in self.envs:
            env.post_play_load()

    def step(self, actions, env_indices=None, get_obs=True, render=True):
        # When stepping a subset, we scope ``og.sim._scenes`` to the
        # requested envs so the simulator only advances those scenes.
        scoped = env_indices is not None
        if scoped:
            from omnigibson.utils.usd_utils import ControllableObjectViewAPI
            view_dict_before = ControllableObjectViewAPI._VIEWS_BY_PATTERN
            all_scenes = og.sim._scenes
            og.sim._scenes = [all_scenes[idx] for idx in env_indices]
            indices = env_indices
        else:
            indices = range(self.num_envs)

        try:
            for idx, action in zip(indices, actions):
                self.envs[idx]._pre_step(action)
            with og.sim.render_on_step(render):
                og.sim.step()
            observations, rewards, terminates, truncates, infos = [], [], [], [], []
            for idx, action in zip(indices, actions):
                obs, reward, terminated, truncated, info = self.envs[idx]._post_step(
                    action, get_obs=get_obs
                )
                observations.append(obs)
                rewards.append(reward)
                terminates.append(terminated)
                truncates.append(truncated)
                infos.append(info)
        finally:
            if scoped:
                og.sim._scenes = all_scenes
                if ControllableObjectViewAPI._VIEWS_BY_PATTERN is not view_dict_before:
                    og.sim.update_handles()
        return observations, rewards, terminates, truncates, infos

    def reset(self, env_indices=None, get_obs=True, **kwargs):
        indices = range(self.num_envs) if env_indices is None else env_indices
        if get_obs:
            observations, infos = [], []
            for idx in indices:
                obs, info = self.envs[idx].reset(get_obs=get_obs, **kwargs)
                observations.append(obs)
                infos.append(info)
            return observations, infos
        else:
            for idx in indices:
                self.envs[idx].reset(get_obs=get_obs, **kwargs)

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
