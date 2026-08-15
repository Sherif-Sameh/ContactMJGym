from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers.utils import rescale_box

from ..envs.base import MujocoBaseEnv
from ..utils.mj_utils import filter_actuators

if TYPE_CHECKING:
    from ..envs.base import ActType

    WrapperActType = ActType


class TaskSpaceControllerAction(ABC, gym.ActionWrapper):
    """Base task-space action wrapper for MuJoCo manipulation environments.

    Each action specifies, per robot, a delta pose relative to the end-effector
    site's current pose:
    - A delta position offset, expressed in the world frame
    - A delta rotation, expressed as a rotation vector in the tangent space of the site's
        current orientation.

    Before applying an action, translation and rotation offsets are each clipped to a
    maximum step size. Actions are expected in a normalized [-1, 1] range and are
    internally unscaled before being converted into world-frame pose targets and
    applied to the underlying environment.

    **Note**: this wrapper assumes *no other* action wrappers have been already applied
    to the environment.

    Args:
        env: The MuJoCo-based manipulation environment to wrap.
        nrobot: Number of robots/poses to convert to the task-space.
        max_tstep: Maximum translation step size (Euclidean norm, in meters)
            applied per action. Default value is 0.05.
        max_rstep: Maximum rotation step size (norm of the rotation vector,
            in radians) applied per action. Defaults value is 0.05 * pi.
        fltr_acts_kwargs: Kwargs for filtering for gripper actuators. For details, see
            :func:`filter_actuators`. If empty, we rely on a simple heuristic by filtering
            for actuators whose `trntype` is `mujoco.mjtTrn.mjTRN_TENDON`. Default value
            is an empty dict.
    """

    def __init__(
        self,
        env: MujocoBaseEnv,
        nrobot: int,
        max_tstep: float = 0.05,
        max_rstep: float = 0.05 * np.pi,
        fltr_acts_kwargs: dict[str, Any] = {},
    ):
        super().__init__(env)
        assert isinstance(env.unwrapped, MujocoBaseEnv), (
            f"Unsupported env type {env.unwrapped.__class__.__name__}. "
            f"Must be a subclass of {MujocoBaseEnv.__name__}."
        )
        assert self.env.unwrapped.model.nu == self.env.unwrapped.model.nactuator, (
            "Wrapper assumes all actuators are SISO."
        )
        assert self.env.unwrapped.model.nu == env.action_space.shape[0]
        self.data = self.env.unwrapped.data
        # Find gripper actuator ids
        gri_idxs = self._get_gripper_indices(self.env.unwrapped.model, fltr_acts_kwargs)
        if len(gri_idxs) == 1:
            gri_idxs = slice(gri_idxs[0], gri_idxs[0] + 1)
        # Setup action buffer
        home = self.env.unwrapped.model.key("home")
        self.action_buffer = (
            np.zeros(self.env.unwrapped.model.nu) if home is None else home.ctrl.copy()
        ).astype(dtype=self.env.action_space.dtype)
        # Setup action space
        action_space_unscaled = self._get_unscaled_action_space(
            nrobot, max_tstep, max_rstep, gri_idxs
        )
        self.action_space, _, self.unscale_action = rescale_box(
            action_space_unscaled, new_min=-1, new_max=1
        )
        # Build action function for gripper actions
        self.gripper_action = self._build_gripper_action(gri_idxs)

    @abstractmethod
    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """

    # region Helpers

    @staticmethod
    def _get_gripper_indices(model: mujoco.MjModel, fltr_kwargs: dict[str, Any]) -> list[int]:
        """Get the indices that correspond to gripper actuators in ctrl."""
        if fltr_kwargs:  # rely on user filters
            return filter_actuators(model, **fltr_kwargs)
        # Fall back to simple trntype heuristic
        return filter_actuators(model, trntype=mujoco.mjtTrn.mjTRN_TENDON)

    def _get_unscaled_action_space(
        self, nrobot: int, max_tstep: float, max_rstep: float, gripper_indices: list[int] | slice
    ) -> spaces.Box:
        """Get unscaled robot (task-space) + gripper (unchanged) box action space."""
        dtype = self.env.action_space.dtype
        low_pose = np.array(([-max_tstep] * 3 + [-max_rstep] * 3) * nrobot, dtype=dtype)
        high_pose = np.array(([max_tstep] * 3 + [max_rstep] * 3) * nrobot, dtype=dtype)
        if gripper_indices:
            low_gri = self.env.action_space.low[gripper_indices]
            high_gri = self.env.action_space.high[gripper_indices]
            low_new = np.concatenate([low_pose, low_gri])
            high_new = np.concatenate([high_pose, high_gri])
        else:
            low_new = low_pose
            high_new = high_new
        return spaces.Box(low=low_new, high=high_new)

    def _build_gripper_action(
        self, gripper_indices: list[int] | slice
    ) -> Callable[[WrapperActType], None]:
        """Build the gripper action function to update gripper ctrls given the current action."""
        # No gripper actuators -> do nothing
        if isinstance(gripper_indices, list) and not gripper_indices:

            def empty_action(action: WrapperActType) -> None: ...

            return empty_action

        ngripper_acts = 1 if isinstance(gripper_indices, slice) else len(gripper_indices)

        def gripper_action(action: WrapperActType) -> None:
            # Write gripper actions into action buffer
            self.action_buffer[gripper_indices] = action[-ngripper_acts:]

        return gripper_action
