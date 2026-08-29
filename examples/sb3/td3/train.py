from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import fire

with open(os.devnull, "w") as fnull, contextlib.redirect_stderr(fnull):
    import gymnasium_robotics  # noqa: F401
import tomllib
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her import HerReplayBuffer

from examples.common.config_utils import dict_to_dataclass
from examples.sb3.common.callbacks import (
    EvalWithStatsCallback,
    HParamCallback,
    RolloutWithStatsCallback,
)
from examples.sb3.common.env_utils import make_vec_env, wrap_vec_normalize
from examples.sb3.td3.config import TD3ExperimentCfg


def _build_env(cfg: TD3ExperimentCfg, run_dir: Path, *, split: Literal["train", "eval"]) -> VecEnv:
    env_cfg = cfg.train_env if split == "train" else cfg.eval_env
    vec_env = make_vec_env(env_cfg, monitor_dir=str(run_dir / f"monitor_{split}"))
    vec_env = wrap_vec_normalize(vec_env, env_cfg.vecnormalize, training=(split == "train"))
    return vec_env


def train(config: str) -> None:
    """Train TD3 (optionally with HER) using Stable-Baselines3 from a TOML config.

    Args:
        config: Path to a TD3 experiment TOML config, see `config/*.toml` for examples).

    Usage:
        python -m examples.sb3.td3.train \
            --config examples/sb3/td3/config/fetch_reach_dense.toml

        python -m examples.sb3.td3.train \
            --config examples/sb3/td3/config/fetch_pick_and_place_sparse_her.toml
    """
    with open(config, "rb") as fh:
        cfg_dict = tomllib.load(fh)
    cfg = dict_to_dataclass(cfg_dict, TD3ExperimentCfg)

    run_id = f"{cfg.name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(cfg.logging.tensorboard_log) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.name}] logging to {run_dir}")

    train_env = _build_env(cfg, run_dir, split="train")
    eval_env = _build_env(cfg, run_dir, split="eval")

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if cfg.her.enabled:
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=cfg.her.n_sampled_goal,
            goal_selection_strategy=cfg.her.goal_selection_strategy,
            copy_info_dict=cfg.her.copy_info_dict,
        )
        print(
            f"[{cfg.name}] HER enabled:\n\tn_sampled_goal={cfg.her.n_sampled_goal}"
            f"\n\tgoal_selection_strategy={cfg.her.goal_selection_strategy!r}"
        )

    algo_cfg = cfg.algorithm
    model = TD3(
        policy=algo_cfg.policy,
        env=train_env,
        learning_rate=algo_cfg.learning_rate,
        buffer_size=algo_cfg.buffer_size,
        learning_starts=algo_cfg.learning_starts,
        batch_size=algo_cfg.batch_size,
        tau=algo_cfg.tau,
        gamma=algo_cfg.gamma,
        train_freq=algo_cfg.train_freq,
        gradient_steps=algo_cfg.gradient_steps,
        action_noise=algo_cfg.action_noise,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_delay=algo_cfg.policy_delay,
        target_policy_noise=algo_cfg.target_policy_noise,
        target_noise_clip=algo_cfg.target_noise_clip,
        policy_kwargs=algo_cfg.policy_kwargs,
        seed=algo_cfg.seed,
        device=algo_cfg.device,
        tensorboard_log=str(run_dir),
        verbose=cfg.logging.verbose,
    )

    # SB3 callbacks count vectorized steps, so freqs given in the config as
    # total env step need dividing by n_envs
    eval_freq = max(cfg.logging.eval_freq // cfg.train_env.n_envs, 1)
    save_freq = max(cfg.logging.save_freq // cfg.train_env.n_envs, 1)
    callbacks = CallbackList(
        [
            HParamCallback(
                config=cfg_dict, metrics=cfg.logging.hparam_metrics, verbose=cfg.logging.verbose
            ),
            RolloutWithStatsCallback(cfg.logging.log_interval),
            EvalWithStatsCallback(
                eval_env=eval_env,
                n_eval_episodes=cfg.logging.n_eval_episodes,
                eval_freq=eval_freq,
                best_model_save_path=str(run_dir / "best_model"),
                verbose=cfg.logging.verbose,
            ),
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=cfg.name,
                save_vecnormalize=True,
                verbose=cfg.logging.verbose,
            ),
        ]
    )

    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        log_interval=cfg.logging.log_interval,
        progress_bar=cfg.training.progress_bar,
        tb_log_name="td3",
    )

    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir / "model"))
    if isinstance(train_env, VecNormalize):
        train_env.save(str(final_dir / "vecnormalize.pkl"))
    print(f"[{cfg.name}] saved final model + VecNormalize state (if any) to {final_dir}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    fire.Fire(train)
