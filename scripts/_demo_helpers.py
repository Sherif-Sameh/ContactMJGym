"""Helpers for collecting, processing and storing demos from environments."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import contact_gym.dr as dr
from contact_gym.utils.noise import GaussianSampler, Noise

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from contact_gym.envs.mujoco_base import ActType, InfoType, ObsType


@dataclass
class EpisodeBuffer:
    """Temporary buffer for stroing a single episode's transitions."""

    observations: list[ObsType] = field(default_factory=list)
    next_observations: list[ObsType] = field(default_factory=list)
    actions: list[ActType] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminations: list[bool] = field(default_factory=list)
    truncations: list[bool] = field(default_factory=list)
    infos: list[InfoType] = field(default_factory=list)

    def add(
        self,
        obs: ObsType,
        next_obs: ObsType,
        action: ActType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: InfoType,
    ) -> None:
        """Add transition to episode buffer."""
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.terminations.append(bool(terminated))
        self.truncations.append(bool(truncated))
        self.infos.append(info)

    def __len__(self) -> int:
        return len(self.actions)

    def to_episode_dict(self) -> dict[str, Any]:
        """Stack stored transitions into NumPy arrays."""

        def stack_obs(obs_list: list[ObsType]) -> dict[str, NDArray[np.float32]]:
            keys = obs_list[0].keys()
            return {k: np.stack([o[k] for o in obs_list]).astype(np.float32) for k in keys}

        return {
            "observations": stack_obs(self.observations),
            "next_observations": stack_obs(self.next_observations),
            "actions": np.stack(self.actions),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "terminations": np.asarray(self.terminations, dtype=bool),
            "truncations": np.asarray(self.truncations, dtype=bool),
            "infos": self.infos,  # not converted
        }

    def reset(self) -> None:
        """Clear episode buffer."""
        self.__init__()


def get_object_state_randomizer() -> dr.DataStateRandomizer:
    """Get the default object state randomizer."""
    rand = dr.DataStateRandomizer(
        dr.state.qpos_state_cfg(
            Noise(
                GaussianSampler(std=[0.1, 0.1, 0, 0, 0, 0.3]),
                operation="add_se3",
                scalar_first=True,
            ),
            entry_sel=slice(-7, None),
        )
    )
    return rand


def save_demos(
    episodes: list[dict[str, Any]], output_path: str, env_name: str, extra_meta: dict | None = None
) -> None:
    """Save episodic demonstrations to an output path via Pickle."""
    payload = {
        "env_name": env_name,
        "n_episodes": len(episodes),
        "meta": extra_meta or {},
        "episodes": episodes,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
