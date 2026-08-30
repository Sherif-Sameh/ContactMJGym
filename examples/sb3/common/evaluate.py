from __future__ import annotations

import time

import fire
import numpy as np
import tomllib
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import contact_gym  # noqa: F401
from examples.common.config_utils import dict_to_dataclass, import_from_path
from examples.sb3.common.config import EnvCfg
from examples.sb3.common.env_utils import make_env_fn


def _make_render_env_fn(env_cfg: EnvCfg, render: bool):
    env_cfg.env_kwargs = dict(env_cfg.env_kwargs)
    if render:
        env_cfg.env_kwargs.setdefault("render_mode", "human")
    return make_env_fn(env_cfg, 0, monitor=False)


def evaluate(
    config: str,
    model_path: str,
    algo: str,
    vecnormalize_path: str | None = None,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = True,
    seed: int = 0,
) -> None:
    """Evaluate a trained SB3 policy, with optional rendering.

    Args:
        config: Path to the TOML experiment config used for training; only its
            `[eval_env]` entry is read.
        model_path: Path to the SB3 saved model `.zip`.
        algo: Dotted import path to the SB3 algorithm class that produced
            `model_path` (e.g. "stable_baselines3.TD3", "stable_baselines3.SAC").
        vecnormalize_path: Optional path to a saved `VecNormalize` `.pkl`. Should be
            passed whenever `VecNormalize` was used during training.
        n_episodes: Number of episodes to roll out.
        deterministic: Use the policy's deterministic action.
        render: Open the MuJoCo viewer and step in real time.
        seed: Seed for the evaluation environment.

    Usage:
        python -m examples.sb3.common.evaluate \
            --config examples/sb3/td3/config/fetch_reach_dense.toml \
            --model_path runs/td3_fetch_reach_dense/final_model/model.zip \
            --algo stable_baselines3.TD3 \
            --n_episodes 10 \
            --seed 10250
    """
    with open(config, "rb") as fh:
        cfg_dict = tomllib.load(fh)
    assert "eval_env" in cfg_dict, f"'{config}' has no [eval_env] entry."
    env_cfg = dict_to_dataclass(cfg_dict["eval_env"], EnvCfg)

    algo_cls = import_from_path(algo)

    vec_env = DummyVecEnv([_make_render_env_fn(env_cfg, render)])
    vec_env.seed(seed=seed)

    if vecnormalize_path is not None:
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    elif env_cfg.vecnormalize is not None and env_cfg.vecnormalize.enabled:
        print(
            "Warning: [eval_env] declares a VecNormalize wrapper but no checkpoint  "
            f"found at {vecnormalize_path}; evaluating with raw (unnormalized) "
            "observations instead."
        )

    model = algo_cls.load(model_path, env=vec_env, device="cpu")

    frame_dt = 1 / vec_env.unwrapped.metadata["render_fps"]
    next_frame = time.perf_counter()

    episode_rewards, episode_lengths, episode_successes = [], [], []
    for episode in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward, ep_len, ep_success = 0.0, 0, 0.0
        while not done:
            # Sample and apply action
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, info = vec_env.step(action)
            done = bool(done_arr[0])
            ep_reward += float(reward[0])
            ep_len += 1
            if done and "is_success" in info[0]:
                ep_success = float(info[0]["is_success"])
            # Rate limit loop
            next_frame += frame_dt
            if not render:
                continue
            sleep_time = next_frame - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame = time.perf_counter()

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        episode_successes.append(ep_success)
        print(
            f"Episode {episode + 1}/{n_episodes}:"
            f"\n\tReward = {ep_reward:.3f}\n\tLength = {ep_len}\n\tSuccess = {ep_success:.0f}\n"
        )
    vec_env.close()

    episode_rewards_arr = np.asarray(episode_rewards)
    episode_lengths_arr = np.asarray(episode_lengths)
    print("\n=== Summary ===")
    print(
        f"\nReward:\n\tMean = {episode_rewards_arr.mean():.3f}"
        f"\n\tMin = {episode_rewards_arr.min():.3f}\n\tMax = {episode_rewards_arr.max():.3f}"
    )
    print(
        f"\nLength:\n\tMean = {episode_lengths_arr.mean():.1f}"
        f"\n\tMin = {episode_lengths_arr.min()}\n\tMax = {episode_lengths_arr.max()}"
    )
    if episode_successes:
        successes_arr = np.asarray(episode_successes)
        print(
            f"\nSuccess:\n\tMean = {successes_arr.mean():.3f} "
            f"\n\tMin = {successes_arr.min():.0f}\n\tMax = {successes_arr.max():.0f}"
        )


if __name__ == "__main__":
    fire.Fire(evaluate)
