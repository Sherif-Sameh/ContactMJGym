from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class TaskSpaceControllerCfg:
    """Task-space controller wrapper configuration.

    Used to initialize one of :class:`~contact_gym.controllers.MocapControllerAction`
    or :class:`~contact_gym.controllers.MinkControllerAction` or None (joint-space control).
    """

    enabled: bool = True
    controller: Literal["mocap", "mink"] = "mocap"
    max_tstep: float = 0.05
    max_rstep: float = 0.05 * np.pi
    fltr_acts_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VecNormalizeCfg:
    """SB3 :class:`stable_baselines3.common.vec_env.VecNormalize` wrapper configuration."""

    enabled: bool = True
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    gamma: float = 0.99
    epsilon: float = 1e-8
    norm_obs_keys: list[str] | None = None


@dataclass
class EnvCfg:
    """Environment configuration for both training and evaluation."""

    env_id: str
    n_envs: int = 1
    vec_env_type: Literal["dummy", "subproc"] = "dummy"
    seed: int = 0
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    tscontroller: TaskSpaceControllerCfg | None = None
    vecnormalize: VecNormalizeCfg | None = None
    monitor_info_keywords: list[str] = field(default_factory=list)


@dataclass
class HERCfg:
    """Hindsight Experience Replay (HER) buffer configuration."""

    enabled: bool = False
    n_sampled_goal: int = 4
    goal_selection_strategy: Literal["future", "final", "episode"] = "future"
    copy_info_dict: bool = False


@dataclass
class LoggingCfg:
    """TensorBoard logging configuration."""

    tensorboard_log: str = "runs"
    hparam_metrics: list[str] | None = None
    log_interval: int = 4  # in total episodes
    eval_freq: int = 10_000  # in total env steps
    n_eval_episodes: int = 20
    save_freq: int = 50_000  # in total env steps
    verbose: int = 1


@dataclass
class TrainingCfg:
    """Model training configuration."""

    total_timesteps: int = 1_000_000
    progress_bar: bool = True
