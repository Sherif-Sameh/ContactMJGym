from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from contact_gym.wrappers.controllers import (  # noqa: TC001
    MinkControllerCfg,
    MocapControllerCfg,
    OscControllerCfg,
)


@dataclass
class TaskSpaceControllerWrapperCfg:
    """Task-space controller wrapper configuration.

    Used to initialize one of the task-space controllers from
    :class:`contact_gym.controllers.ALL_CONTROLLERS` or None (joint-space control).
    """

    enabled: bool = True
    controller: Literal["mink", "mocap", "osc"] = "mocap"
    mink_cfg: MinkControllerCfg | None = None
    mocap_cfg: MocapControllerCfg | None = None
    osc_cfg: OscControllerCfg | None = None

    def get_cfg(self):
        """Get the configuration of the active set controller if any."""
        if not self.enabled:
            return None
        match self.controller:
            case "mink":
                return self.mink_cfg
            case "mocap":
                return self.mocap_cfg
            case "osc":
                return self.osc_cfg
            case _:
                raise ValueError(f"Invalid task-space controller {self.controller}.")


@dataclass
class ActionHistoryWrapperCfg:
    """Action history wrapper configuration.

    Used to add an :class:`ActionHistoryWrapper` to the environment to append previous
    action history onto the environment's observations.
    """

    enabled: bool = True
    n_stack: int = 1
    reset_value: float = 0.0


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
    tscontroller: TaskSpaceControllerWrapperCfg | None = None
    actionhistory: ActionHistoryWrapperCfg | None = None
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
