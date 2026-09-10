from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from numpy.typing import NDArray
    from stable_baselines3.common.buffers import DictReplayBuffer
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.her import HerReplayBuffer

    from contact_gym.envs.mujoco_base import ActType, InfoType, ObsType

    TransitionType: TypeAlias = tuple[ObsType, ObsType, ActType, float, bool, bool, InfoType]


def load_episodes(paths: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load and concatenate episodes from one or more demo pkl files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    episodes: list[dict[str, Any]] = []
    for p in paths:
        with open(p, "rb") as f:
            payload = pickle.load(f)
        ep_list = payload["episodes"] if isinstance(payload, dict) else payload
        episodes.extend(ep_list)
    return episodes


def preload_replay_buffer(
    replay_buffer: DictReplayBuffer | HerReplayBuffer,
    episodes: list[dict[str, Any]],
    n_envs: int,
    max_transitions: int | None = None,
    verbose: int = 0,
) -> int:
    """Warm-start an SB3 (Dict/Her)ReplayBuffer with demo episodes.

    Distributes episodes across `n_envs` buffer columns (load-balanced, cycling short
    columns) and replays them through the buffer's public `.add()` API tick-by-tick, so
    all internal book-keeping stays correct.

    Args:
        replay_buffer: `model.replay_buffer`, already constructed with the target
            `n_envs`.
        episodes: Output of :func:`load_episodes`.
        n_envs: Number of parallel training envs (`model.n_envs`).
        max_transitions: Optional cap on total transitions added.
        verbose: Verbose level. If > 0, a short summary is printed.

    Returns:
        Number of transitions added (== n_ticks * n_envs).
    """
    if not episodes:
        return 0
    columns, loads = _assign_columns(episodes, n_envs)
    streams = [_column_stream(col) for col in columns]

    n_ticks = max(loads) if any(loads) else 0
    if max_transitions is not None:
        n_ticks = min(n_ticks, max(1, max_transitions // n_envs))
    for _ in range(n_ticks):
        obs_b, next_obs_b, act_b, rew_b, done_b, info_b = [], [], [], [], [], []
        for stream in streams:
            obs, next_obs, action, reward, terminated, truncated, info = next(stream)
            obs_b.append(obs)
            next_obs_b.append(next_obs)
            act_b.append(np.asarray(action, dtype=np.float32))
            rew_b.append(reward)
            done_b.append(terminated or truncated)
            info = dict(info)
            info["TimeLimit.truncated"] = bool(truncated and not terminated)
            info_b.append(info)

        replay_buffer.add(
            _stack_obs(obs_b),
            _stack_obs(next_obs_b),
            np.stack(act_b),
            np.asarray(rew_b, dtype=np.float32),
            np.asarray(done_b, dtype=bool),
            info_b,
        )

    n_added = n_ticks * n_envs
    if verbose > 0:
        n_total = sum(len(ep["actions"]) for ep in episodes)
        print(
            f"Preloaded {n_added} transitions from {len(episodes)} demo episodes "
            f"({n_total} raw transitions, balanced across {n_envs} envs) into "
            f"replay buffer (pos={replay_buffer.pos}, size={replay_buffer.size()})."
        )
    return n_added


def warm_start_vecnormalize(
    vec_normalize: VecEnv | VecNormalize, episodes: list[dict[str, Any]], verbose: int = 0
) -> None:
    """Seed a VecNormalize wrapper's running obs/reward stats from demos.

    Feeds the VecNormalize wrapper through its RunningMeanStd.update()` method, the same batch-merge (Chan et al.)
    formula VecNormalize itself calls incrementally online, so seeding once
    with the full demo batch is mathematically equivalent to having replayed
    the demos step-by-step through a live wrapper. Because the running count
    blends smoothly with subsequent online updates, the demo stats' influence
    naturally fades as real training data accumulates — no manual annealing
    needed.

    No-op if `vec_normalize` is not an instance of `VecNormalize`.

    Args:
        vec_normalize: The VecEnv or VecNormalize-wrapped training env.
        episodes: Output of :func:`load_episodes`.
        verbose: Verbose level. If > 0, a short summary is printed.
    """
    if not isinstance(vec_normalize, VecNormalize) or not episodes:
        return
    gamma = vec_normalize.gamma
    # Feed observations
    if isinstance(vec_normalize.obs_rms, dict):
        for key, rms in vec_normalize.obs_rms.items():
            all_obs = np.concatenate([ep["observations"][key] for ep in episodes], axis=0)
            rms.update(all_obs.astype(np.float64))
    else:
        all_obs = np.concatenate([ep["observations"]["observation"] for ep in episodes], axis=0)
        vec_normalize.obs_rms.update(all_obs.astype(np.float64))
    # Feed rewards (discounted returns)
    if vec_normalize.norm_reward:
        all_returns = []
        for ep in episodes:
            ret = 0.0
            ep_returns = np.empty(len(ep["rewards"]), dtype=np.float64)
            for t, r in enumerate(ep["rewards"]):
                ret = ret * gamma + r
                ep_returns[t] = ret
            all_returns.append(ep_returns)
        vec_normalize.ret_rms.update(np.concatenate(all_returns))

    if verbose:
        print(
            f"Warm-started VecNormalize from {len(episodes)} demo episodes "
            f"(gamma={gamma}, norm_reward={vec_normalize.norm_reward})."
        )


# region Helpers


def _assign_columns(episodes: list[dict], n_envs: int) -> tuple[list[list[dict]], list[int]]:
    """Greedily load-balance episodes across n_envs columns by transition count."""
    columns: list[list[dict]] = [[] for _ in range(n_envs)]
    loads = [0] * n_envs
    for ep in sorted(episodes, key=lambda e: len(e["actions"]), reverse=True):
        i = min(range(n_envs), key=lambda k: loads[k])
        columns[i].append(ep)
        loads[i] += len(ep["actions"])
    # Guarantee every column has something to stream, even with n_envs > n_episodes.
    if any(loads):
        fullest = loads.index(max(loads))
        for i in range(n_envs):
            if not columns[i]:
                columns[i] = columns[fullest]
    return columns, loads


def _column_stream(episodes_for_col: list[dict]) -> Generator[TransitionType, None, None]:
    """Infinite generator cycling through a column's assigned episodes."""
    for ep in itertools.cycle(episodes_for_col):
        for t in range(len(ep["actions"])):
            yield _get_transition_at(ep, t)


def _get_transition_at(ep: dict[str, Any], t: int) -> TransitionType:
    """Get episode transition at step `ŧ`."""
    obs = {k: v[t] for k, v in ep["observations"].items()}
    next_obs = {k: v[t] for k, v in ep["next_observations"].items()}
    action = ep["actions"][t]
    reward = float(ep["rewards"][t])
    terminated = bool(ep["terminations"][t])
    truncated = bool(ep["truncations"][t])
    info = dict(ep["infos"][t]) or {}
    return obs, next_obs, action, reward, terminated, truncated, info


def _stack_obs(obs_list: list[ObsType]) -> dict[str, NDArray[np.float32]]:
    """Stack list of dict observations into a dict of batched observation keys."""
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list]).astype(np.float32) for k in keys}
