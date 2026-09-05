from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray: TypeAlias = NDArray[np.floating]
    ActType: TypeAlias = FloatArray
    BoxObsType: TypeAlias = FloatArray
    DictObsType: TypeAlias = dict[str, FloatArray]
    ObsType: TypeAlias = BoxObsType | DictObsType
    InfoType: TypeAlias = dict[str, Any]

# region Wrapper


class ActionHistoryWrapper(gym.Wrapper):
    """Appends the last `n_stack` agent actions to a flat observation.

    Supports both Box or GoalEnv-like Dict observation spaces. Only Box action spaces are
    supported.

    Args:
        env: The environment to wrap.
        n_stack: Number of past actions to store and append onto observations
            (1 = just the previous action). Default value is 1.
        reset_value: Placeholder value used to fill history at the start of an episode,
            before `n_stack` actions have been taken. Default value is 0.
    """

    def __init__(self, env: gym.Env, n_stack: int = 1, reset_value: float = 0.0):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box), (
            f"Action history wrapper expects a Box action space. Got {type(env.action_space)}."
        )
        assert isinstance(env.observation_space, (spaces.Box, spaces.Dict)), (
            "Action history wrapper expects a Box or Dict observation space. "
            f"Got {type(env.observation_space)}."
        )
        if isinstance(env.observation_space, spaces.Dict):
            for key in ["observation", "achieved_goal", "desired_goal"]:
                assert key in env.observation_space.keys(), (
                    "Action history wrapper expects GoalEnv-like Dict observation space. "
                    f"Did not find {key} key in observation space."
                )
            observation_space = env.observation_space["observation"]
            assert isinstance(observation_space, spaces.Box), (
                "Action history wrapper expects a Box observation sub-space with Dict "
                f"observation spaces. Got {type(observation_space)}."
            )
        else:
            observation_space = env.observation_space
        assert len(observation_space.shape) == 1, (
            "Action history wrapper expects flat observations. "
            f"Got {len(observation_space.shape)}D observations."
        )
        self.n_stack = n_stack
        self.reset_value = reset_value
        # Setup action history buffer
        self._act_dim = int(np.prod(env.action_space.shape))
        self._history = deque(maxlen=n_stack)
        # Setup observation space
        act_low = np.tile(env.action_space.low, n_stack)
        act_high = np.tile(env.action_space.high, n_stack)
        self._dtype = observation_space.dtype
        observation_space_wrapper = spaces.Box(
            low=np.concatenate([observation_space.low, act_low], dtype=self._dtype),
            high=np.concatenate([observation_space.high, act_high], dtype=self._dtype),
            dtype=self._dtype,
        )
        if isinstance(env.observation_space, spaces.Box):
            self.observation_space = observation_space_wrapper
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": observation_space_wrapper,
                    "achieved_goal": env.observation_space["achieved_goal"],
                    "desired_goal": env.observation_space["desired_goal"],
                }
            )
        # Build observation helper function
        self.augment_observation = self._build_augment_observation()

    # region Env API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_history()
        return self.augment_observation(obs), info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(action.copy())
        return self.augment_observation(obs), reward, terminated, truncated, info

    # region Helpers

    def _build_augment_observation(
        self,
    ) -> Callable[[BoxObsType], BoxObsType] | Callable[[DictObsType], DictObsType]:
        """Build the function"""

        def augment_observation_box(obs: BoxObsType) -> BoxObsType:
            stacked = np.concatenate(list(self._history))
            return np.concatenate([obs, stacked])

        def augment_observation_dict(obs: DictObsType) -> DictObsType:
            obs["observation"] = augment_observation_box(obs["observation"])
            return obs

        if isinstance(self.env.observation_space, spaces.Box):
            return augment_observation_box
        return augment_observation_dict

    def _reset_history(self) -> None:
        """Reset action history and fill buffer with placeholder values."""
        self._history.clear()
        pad = np.full(self._act_dim, self.reset_value, dtype=self._dtype)
        for _ in range(self.n_stack):
            self._history.append(pad)
