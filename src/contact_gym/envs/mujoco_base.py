from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import ChainMap
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..curriculum import CurriculumTerm
    from ..dr import DomainRandomizer

    FloatArray: TypeAlias = NDArray[np.float32]
    BoolArray: TypeAlias = NDArray[np.bool_]
    ActType: TypeAlias = FloatArray
    GoalType: TypeAlias = FloatArray
    ObsType: TypeAlias = dict[str, FloatArray]
    InfoType: TypeAlias = dict[str, Any]
    RGBType: TypeAlias = NDArray[np.uint8]

logger = logging.getLogger(__name__)


class RewardType(StrEnum):
    """Environment reward type. Options include 'sparse' and 'dense' only."""

    SPARSE = "sparse"
    DENSE = "dense"


class MujocoBaseEnv(ABC, gym.Env):
    """Base MuJoCo-based environment.

    Follows the **goal** environment API from Gymnasium Robotics. Defines common
    environment `action_space`, domain randomization and curriculum learning logic.

    Provides `reset`, `step`, `render`, `close`, and `compute_truncated` methods.
    Extending classes must provide the goal-based `compute_reward` and
    `compute_terminated` implementations, in additon to the environment-specific
    `observation_space` as well as `_reset_data`, `_sample_goal`, `_get_obs` and
    `_get_info` methods.

    Args:
        spec: MuJoCo scene spec (MjSpec) to build model from.
        frame_skip: Number of sim steps per env step. Default value is 20.
        reward_type: Reward type, one of ["dense", "sparse"]. Default value is sparse.
        domain_randomizers: Sequence of domain randomizers to apply during environment
            reset. See :class:`DomainRandomizer` for details. Default value is empty.
        curriculum_terms: Sequence of curriculum terms to call during environment reset.
            See :class:`CurriculumTerm` for details. Default value is empty.
        render_mode: Environment rendering mode. Default value is None.
        renderer_kwargs: Optional kwargs to pass to :class:`mujoco.Renderer` for rendering.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}  # noqa: RUF012

    def __init__(
        self,
        spec: mujoco.MjSpec,
        frame_skip: int = 20,
        reward_type: str | RewardType = "sparse",
        domain_randomizers: Sequence[DomainRandomizer] = (),
        curriculum_terms: Sequence[CurriculumTerm] = (),
        render_mode: str | None = None,
        renderer_kwargs: dict[str, Any] = {},
    ):
        assert frame_skip >= 1, f"Frame skip must be >= 1. Got {frame_skip}."
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Invalid rendering mode {render_mode}. Must be in {self.metadata['render_modes']}."
        )
        self.step_count = 0
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.reward_type = RewardType(reward_type)
        self.rng = np.random.default_rng()
        self.domain_randomizers = domain_randomizers
        self.curriculum_terms = curriculum_terms
        self.render_mode = render_mode
        self._renderer = (
            mujoco.Renderer(self.model, **renderer_kwargs) if render_mode == "rgb_array" else None
        )
        self._viewer = None
        self.metadata["render_fps"] = int(
            np.round(1.0 / (self.model.opt.timestep * self.frame_skip))
        )
        self._home_key_id = self._set_home_key()
        # Setup action space
        ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)

    @property
    def viewer_is_running(self) -> bool:
        """True if a human-mode viewer window is open and hasn't been closed by the user."""
        return self._viewer is not None and self._viewer.is_running()

    # region Core API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            self.action_space.seed(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        param_dicts = [c_term(self, self.step_count) for c_term in self.curriculum_terms]
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Step: {self.step_count}\nCurriculum Parameters: {dict(ChainMap(*param_dicts))}"
            )
        for domain_randomizer in self.domain_randomizers:
            domain_randomizer(self.model, self.data, self.rng)
        self._reset_data()
        self._apply_options(options)
        self._sample_goal()
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self._get_info(obs)

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoType]:
        self.step_count += 1
        # Apply action in environment
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        # Stepping logic follows the step2 -> step1 pattern used in dm_control for updated fields
        # https://github.com/google-deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L147
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
            mujoco.mj_step2(self.model, self.data)
            if self.frame_skip > 1:
                mujoco.mj_step(self.model, self.data, self.frame_skip - 1)
        else:
            mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_step1(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info(obs)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        terminated = self.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), False, info

    def render(self, *, camera: mujoco.mjvCamera | str | int = -1) -> RGBType | None:
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # region Goal API

    @abstractmethod
    def compute_reward(
        self, achieved_goal: GoalType, desired_goal: GoalType, info: InfoType
    ) -> np.float32 | FloatArray:
        """Compute task reward for acheived and desired goals. Must support batched inputs."""

    @abstractmethod
    def compute_terminated(
        self, achieved_goal: GoalType, desired_goal: GoalType, info: InfoType
    ) -> np.bool_ | BoolArray:
        """Compute terminated signal for acheived and desired goals. Must support batched inputs."""

    def compute_truncated(
        self, achieved_goal: GoalType, desired_goal: GoalType, info: InfoType
    ) -> np.bool_ | BoolArray:
        """Compute truncated signal for acheived and desired goals. Must support batched inputs.

        Returns:
            NumPy array with False values. Zero-dimensional for single inputs.
        """
        return np.zeros(achieved_goal.shape[:-1], dtype=np.bool_)

    # region Env API

    @abstractmethod
    def _reset_data(self) -> None:
        """Apply any additional resets to self.data after environment reset.

        Called after `mujoco.mj_resetData` and domain randomization, before `mujoco.mj_forward`.
        """

    @abstractmethod
    def _sample_goal(self) -> None:
        """Sample a new goal for the environment."""

    @abstractmethod
    def _get_obs(self) -> ObsType:
        """Get the latest observations."""

    @abstractmethod
    def _get_info(self, obs: ObsType) -> InfoType:
        """Get the latest info dict."""

    # region Helpers

    def _set_home_key(self) -> int:
        """Set the home keyframe for free objects in the scene."""
        assert self.model.key("home") is not None, "Scene does not have a home key."
        home_key_id = self.model.key("home").id
        for jnt_id in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt_id]
            if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
                continue
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            qvel_adr = self.model.jnt_dofadr[jnt_id]
            initial_qpos = self.data.qpos[qpos_adr : qpos_adr + 7].copy()
            initial_qvel = self.data.qvel[qvel_adr : qvel_adr + 6].copy()
            self.model.key_qpos[home_key_id, qpos_adr : qpos_adr + 7] = initial_qpos
            self.model.key_qvel[home_key_id, qvel_adr : qvel_adr + 6] = initial_qvel
        return home_key_id

    def _apply_options(self, options: dict[str, Any] | None) -> None:
        """Apply state overrides in options dict."""
        if not options:
            return
        qpos = options.get("qpos")
        qvel = options.get("qvel")
        ctrl = options.get("ctrl")
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
