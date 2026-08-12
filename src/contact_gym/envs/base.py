from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ActType = ObsType = NDArray[np.float32]
    InfoType = dict[str, Any]
    RGBType = NDArray[np.uint8]


class MujocoBaseEnv(ABC, gym.Env):
    """Base MuJoCo-based environment.

    Defines common environment `action_space`, `reset`, `step` and `render` logic. Extending
    classes must provide `observation_space`, `_reset_data`, `_get_obs`, `_get_info` and
    `_compute_reward` logic.
    """

    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    def __init__(self, spec: mujoco.MjSpec, frame_skip: int = 10, render_mode: str | None = None):
        assert frame_skip >= 1, f"Frame skip must be >= 1. Got {frame_skip}."
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Invalid rendering mode {render_mode}. Must be in {self.metadata['render_modes']}."
        )
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        self._frame_skip = frame_skip
        self.render_mode = render_mode
        self._renderer = None
        self._home_key_id = self._set_home_key()
        # Setup action space
        ctrl_low = self._model.actuator_ctrlrange[:, 0]
        ctrl_high = self._model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)

    # region Core API

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed, options=options)
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_key_id)
        self._reset_data()
        self._apply_options(options)
        mujoco.mj_forward(self._model, self._data)
        return self._get_obs(), self._get_info()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        # Apply action in environment
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._data.ctrl[:] = action
        # Stepping logic follows the step2 -> step1 pattern used in dm_control for updated fields
        # https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L147
        if self._model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
            mujoco.mj_step2(self._model, self._data)
            if self._frame_skip > 1:
                mujoco.mj_step(self._model, self._data, self._frame_skip - 1)
        else:
            mujoco.mj_step(self._model, self._data, self._frame_skip)
        mujoco.mj_step1(self._model, self._data)
        obs = self._get_obs()
        reward, terminated = self._compute_reward(obs, action)
        return obs, reward, terminated, False, self._get_info()

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
    def _compute_reward(self, obs: ObsType, act: ActType) -> tuple[float, bool]:
        """Compute the reward and termination signal.

        Args:
            obs: Latest observation.
            act: Latest action.

        Returns:
            tuple containing the reward and termination signals.
        """

    def _set_home_key(self) -> int:
        """Set the home keyframe for free objects in the scene."""
        assert self._model.key("home") is not None, "Scene does not have a home key."
        home_key_id = self._model.key("home").id
        for jnt_id in range(self._model.njnt):
            jnt_type = self._model.jnt_type[jnt_id]
            if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            qvel_adr = self._model.jnt_dofadr[jnt_id]
            initial_qpos = self._data.qpos[qpos_adr : qpos_adr + 7].copy()
            initial_qvel = self._data.qvel[qvel_adr : qvel_adr + 6].copy()
            self._model.key_qpos[home_key_id, qpos_adr : qpos_adr + 7] = initial_qpos
            self._model.key_qvel[home_key_id, qvel_adr : qvel_adr + 6] = initial_qvel
        return home_key_id

    def _apply_options(self, options: dict[str, Any] | None) -> None:
        """Apply state overrides in options dict."""
        if not options:
            return
        qpos = options.get("qpos")
        qvel = options.get("qvel")
        ctrl = options.get("ctrl")
        if qpos is not None:
            self._data.qpos[:] = qpos
        if qvel is not None:
            self._data.qvel[:] = qvel
        if ctrl is not None:
            self._data.ctrl[:] = ctrl
