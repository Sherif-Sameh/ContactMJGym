from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from contact_gym.wrappers.controllers import ALL_CONTROLLERS, CONTROLLER_TO_CLS

if TYPE_CHECKING:
    from examples.sb3.common.config import EnvCfg, VecNormalizeCfg


def make_vec_env(env_cfg: EnvCfg, monitor_dir: str | None = None) -> VecEnv:
    """Build a SB3 :class:`DummyVecEnv` or :class:`SubprocVecEnv` with optional
    :class:`Monitor` and task-space controller wrappers according to `env_cfg`.

    If no controller is selected, the raw unscaled action space is rescaled to [-1, 1].

    Does *NOT* apply `VecNormalize`, see :func:`wrap_vec_normalize` for that, since
    the train and eval envs need different `training=` values.
    """
    env_fns = [
        make_env_fn(env_cfg, rank, monitor_dir=monitor_dir) for rank in range(env_cfg.n_envs)
    ]
    assert env_cfg.vec_env_type in ["dummy", "subproc"], (
        f"Unknown vec_env_type {env_cfg.vec_env_type} for env {env_cfg.env_id}. "
        "Expected 'dummy' or 'subproc'."
    )
    vec_env = DummyVecEnv(env_fns) if env_cfg.vec_env_type == "dummy" else SubprocVecEnv(env_fns)
    vec_env.seed(env_cfg.seed)
    return vec_env


def wrap_vec_normalize(
    vec_env: VecEnv, vecnormalize_cfg: VecNormalizeCfg | None, *, training: bool
) -> VecEnv:
    """Apply :class:`VecNormalize` wrapper to `vec_env` according to `vecnormalize_cfg`."""
    if vecnormalize_cfg is None or not vecnormalize_cfg.enabled:
        return vec_env
    return VecNormalize(
        vec_env,
        training=training,
        norm_obs=vecnormalize_cfg.norm_obs,
        norm_reward=vecnormalize_cfg.norm_reward,
        clip_obs=vecnormalize_cfg.clip_obs,
        clip_reward=vecnormalize_cfg.clip_reward,
        gamma=vecnormalize_cfg.gamma,
        epsilon=vecnormalize_cfg.epsilon,
        norm_obs_keys=vecnormalize_cfg.norm_obs_keys,
    )


def make_env_fn(
    env_cfg: EnvCfg, rank: int, monitor: bool = True, monitor_dir: str | None = None
) -> Callable[[], gym.Env]:
    """Create factory function for single gymnasium environments with optional
    :class:`Monitor` and task-space controller wrappers.

    If no controller is selected, the raw unscaled action space is rescaled to [-1, 1].
    """

    def _init() -> gym.Env:
        env = gym.make(env_cfg.env_id, **env_cfg.env_kwargs)
        monitor_path = None
        if monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
            monitor_path = os.path.join(monitor_dir, f"{rank}")
        if monitor:
            env = Monitor(
                env, filename=monitor_path, info_keywords=tuple(env_cfg.monitor_info_keywords)
            )
        if env_cfg.tscontroller is not None and env_cfg.tscontroller.enabled:
            ctrl_cfg = env_cfg.tscontroller
            assert ctrl_cfg.controller in ALL_CONTROLLERS, (
                f"Unknown task-space controller type {ctrl_cfg.controller}."
                f"Must be one of {ALL_CONTROLLERS}."
            )
            env = CONTROLLER_TO_CLS[ctrl_cfg.controller](env, ctrl_cfg.get_cfg())
        else:  # rescale raw action space
            env = RescaleAction(env, min_action=-1, max_action=1)
        return env

    return _init
