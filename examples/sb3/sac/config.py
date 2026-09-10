from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from examples.sb3.common.config import DemoCfg, EnvCfg, HERCfg, LoggingCfg, TrainingCfg


@dataclass
class SACAlgorithmCfg:
    """SB3 :class:`stable_baselines3.sac.SAC` algorithm configuration."""

    policy: Literal["MlpPolicy", "CnnPolicy", "MultiInputPolicy"] = "MultiInputPolicy"
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: Any | None = None
    ent_coef: str | float = "auto"
    target_update_interval: int = 1
    target_entropy: str | float = "auto"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    device: str = "auto"


@dataclass
class SACExperimentCfg:
    """SAC experiment configuration."""

    name: str
    train_env: EnvCfg
    eval_env: EnvCfg
    algorithm: SACAlgorithmCfg
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    her: HERCfg = field(default_factory=HERCfg)
    demo: DemoCfg = field(default_factory=DemoCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
