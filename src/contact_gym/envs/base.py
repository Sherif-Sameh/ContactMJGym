from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ActType = ObsType = NDArray[np.float64]
    InfoType = dict[str, Any]
    RGBType = NDArray[np.uint8]


class MujocoBaseEnv(ABC, gym.Env):
    """Base MuJoCo-based environment.

    Defines common environment `action_space`, `reset`, `step` and `render` logic. Extending
    classes must provide `observation_space`, `_reset_data`, `_get_obs`, `_get_info` and
    `_compute_reward` logic.
    """

    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    def __init__(
        self,
        spec: mujoco.MjSpec,
        frame_skip: int = 10,
        render_mode: str | None = None,
    ):
        assert frame_skip >= 1, f"Frame skip must be >= 1. Got {frame_skip}."
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Invalid rendering mode {render_mode}. Must be in {self.metadata['render_modes']}."
        )
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self._renderer = None
        # Setup action space
        n_act = self._model.nu
        ctrl_low = self._model.actuator_ctrlrange[:, 0]
        ctrl_high = self._model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)
        self._last_action = np.zeros(n_act, dtype=np.float32)

    # region Core API

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self._reset_data()
        mujoco.mj_forward(self._model, self._data)
        self._last_action[:] = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        # Apply action in environment
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data, nstep=self.frame_skip)
        self._last_action = action
        reward, terminated = self._compute_reward(action)
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self) -> RGBType:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=240, width=320)
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    # region Helpers

    @abstractmethod
    def _reset_data(self) -> None:
        """Apply any additional resets to self._data after `mujoco.mj_resetData`.

        Called before `mujoco.mj_forward`.
        """

    @abstractmethod
    def _get_obs(self) -> ObsType:
        """Get the latest observations."""

    @abstractmethod
    def _get_info(self) -> InfoType:
        """Get the latest info dict."""

    @abstractmethod
    def _compute_reward(self, action: ActType) -> tuple[float, bool]:
        """Compute the reward and termination signal.

        Args:
            action: Latest action. Useful for action penalties.

        Returns:
            tuple containing the reward and termination signal.
        """
