from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from examples.sb3.common.config import EnvCfg, HERCfg, LoggingCfg, TrainingCfg

if TYPE_CHECKING:
    from stable_baselines3.common.noise import ActionNoise


@dataclass
class TD3AlgorithmCfg:
    """SB3 :class:`stable_baselines3.td3.TD3` algorithm configuration."""

    policy: str = "MultiInputPolicy"
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: ActionNoise | None = None
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    device: str = "auto"


@dataclass
class TD3ExperimentCfg:
    """TD3 experiment configuration."""

    name: str
    train_env: EnvCfg
    eval_env: EnvCfg
    algorithm: TD3AlgorithmCfg
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    her: HERCfg = field(default_factory=HERCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
