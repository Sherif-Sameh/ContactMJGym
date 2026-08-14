from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers.utils import rescale_box

from ..envs.base import MujocoBaseEnv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..envs.base import ActType

    WrapperActType = ActType


class MocapControllerAction(gym.ActionWrapper):
    """Task-space action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through mocap bodies welded to gripper sites, following the
    approach used by Gymnasium-Robotics' Fetch environments. At initialization, this wrapper
    disables all of the robot(s)' actuators (gripper actuators are left untouched) and enables
    the weld equality constraints between each mocap body's site and its corresponding gripper
    site. The wrapper supports any number of robots/grippers present in the model, each
    controlled through its own mocap-weld pair.

    Each action specifies, per gripper, a delta pose relative to the gripper
    site's *current* pose:
    - A delta position offset, expressed in the world frame
    - A delta rotation, expressed as a rotation vector in the tangent space of the site's
        current orientation.

    Before applying an action, the mocap body is reset to coincide with its welded site's current
    pose, and the requested offset is then added on top to obtain the new mocap target. Translation
    and rotation offsets are each clipped to a maximum step size.

    Actions are expected in a normalized [-1, 1] range and are internally unscaled before being
    converted into world-frame pose targets and applied to the underlying environment.

    **Note**: this wrapper assumes weld equality constraints are defined between
    *sites* (mocap site <-> gripper site), not bodies.

    Args:
        env: The MuJoCo-based manipulation environment to wrap. Must define
            at least one gripper, with mocap bodies welded to sites via
            site-to-site equality constraints.
        max_tstep: Maximum translation step size (Euclidean norm, in meters)
            applied per action. Default value is 0.05.
        max_rstep: Maximum rotation step size (norm of the rotation vector,
            in radians) applied per action. Defaults to 0.05 * pi.
    """

    ROBOT_ACTUATOR_GROUP = 1

    def __init__(
        self, env: MujocoBaseEnv, max_tstep: float = 0.05, max_rstep: float = 0.05 * np.pi
    ):
        super().__init__(env)
        assert issubclass(env.unwrapped.__class__, MujocoBaseEnv), (
            f"Unsupported env type {env.unwrapped.__class__.__name__}. "
            f"Must be a subclass of {MujocoBaseEnv.__name__}."
        )
        assert self.env.unwrapped.model.nu == self.env.unwrapped.model.nactuator, (
            "Wrapper assumes all actuators are SISO."
        )
        assert self.env.unwrapped.model.nu == env.action_space.shape[0]
        assert self.env.unwrapped.model.nmocap > 0
        self.data = self.env.unwrapped.data
        self.nmocap = self.env.unwrapped.model.nmocap
        self.action_buffer = np.zeros(
            self.env.action_space.shape[0], dtype=self.env.action_space.dtype
        )
        self._gri_idxs = tuple(self._get_gripper_indices(self.env.unwrapped.model))
        self._gri_idxs = (
            slice(self._gri_idxs[0], self._gri_idxs[0] + 1)
            if len(self._gri_idxs) == 1
            else self._gri_idxs
        )
        self._disable_robot_actuators(self.env.unwrapped.model)
        # Setup action space
        action_space_unscaled = self._get_unscaled_action_space(max_tstep, max_rstep)
        self.action_space, _, self._func = rescale_box(action_space_unscaled, new_min=-1, new_max=1)
        # Enable mocap weld constraints and get mocap -> site id mapping
        mocap_siteid = self._setup_mocap_bodies(self.env.unwrapped.model)
        # Build action function for mocap bodies
        self._mocap_action = self._build_mocap_action(mocap_siteid, max_tstep, max_rstep)

    def action(self, action: ActType) -> WrapperActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        # Unscale actions
        action = self._func(action)
        # Extract and apply mocap actions to mocap bodies
        action_mocap = action[: 6 * self.nmocap].reshape(self.nmocap, 6)
        self._mocap_action(action_mocap)
        # Write gripper actions into action buffer
        action_gri = action[6 * self.nmocap :]
        self.action_buffer[self._gri_idxs] = action_gri
        return self.action_buffer

    # region Helpers

    @staticmethod
    def _get_gripper_indices(model: mujoco.MjModel) -> list[int]:
        """Get the indices that correspond to gripper actuators in ctrl."""
        # TODO: Find a more robust way of evaluating robot vs gripper actuators
        gripper_indices = [
            i
            for i in range(model.nactuator)
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON
        ]
        assert gripper_indices, "No gripper actuators. Wrapper assumes every robot has a gripper."
        return gripper_indices

    @classmethod
    def _disable_robot_actuators(cls, model: mujoco.MjModel) -> None:
        """Move all robot actuators to a free group then disable the actuator group."""
        # Find a free actuator group to disable
        active_groups = set(model.actuator_group)
        while cls.ROBOT_ACTUATOR_GROUP in active_groups:
            cls.ROBOT_ACTUATOR_GROUP += 1
        # Move all robot actuators to the new empty group
        # TODO: Find a more robust way of evaluating robot vs gripper actuators
        robot_act_ids = [
            i
            for i in range(model.nactuator)
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON
        ]
        model.actuator_group[robot_act_ids] = cls.ROBOT_ACTUATOR_GROUP
        # Disable group
        model.opt.disableactuator |= 1 << cls.ROBOT_ACTUATOR_GROUP

    def _get_unscaled_action_space(self, max_tstep: float, max_rstep: float) -> spaces.Box:
        """Get unscaled robot (task-space) + gripper (unchanged) box action space."""
        dtype = self.env.action_space.dtype
        low_gri = self.env.action_space.low[self._gri_idxs]
        high_gri = self.env.action_space.high[self._gri_idxs]
        low_mocap = np.array(([-max_tstep] * 3 + [-max_rstep] * 3) * self.nmocap, dtype=dtype)
        high_mocap = np.array(([max_tstep] * 3 + [max_rstep] * 3) * self.nmocap, dtype=dtype)
        low_new = np.concatenate([low_mocap, low_gri])
        high_new = np.concatenate([high_mocap, high_gri])
        return spaces.Box(low=low_new, high=high_new)

    def _setup_mocap_bodies(self, model: mujoco.MjModel) -> list[int]:
        """Enable weld constraints involving mocap bodies and return mocap -> site id map."""
        # Enable weld constraints and establish mocap -> site id map
        mocap_siteid = []
        for i in range(model.neq):
            if (
                model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD
                or model.eq_objtype[i] != mujoco.mjtObj.mjOBJ_SITE
            ):
                continue
            mocap1id = model.body_mocapid[model.site_bodyid[model.eq_obj1id[i]]]
            mocap2id = model.body_mocapid[model.site_bodyid[model.eq_obj2id[i]]]
            if mocap1id >= 0 or mocap2id >= 0:
                model.eq_active0[i] = 1
                self.data.eq_active[i] = 1
                if mocap1id >= 0:  # obj1 is the mocap site
                    mocapid = mocap1id
                    siteid = model.eq_obj2id[i]
                else:  # obj2 is the mocap site
                    mocapid = mocap2id
                    siteid = model.eq_obj1id[i]
                mocap_siteid.append((mocapid, siteid))
        assert len(mocap_siteid) == model.nmocap, (
            "Expected a weld constraint for every mocap body. "
            f"Got {len(mocap_siteid)}/{model.nmocap} constrained mocaps."
        )
        # Order mocap -> site id map by mocap id and keep only site ids
        mocap_siteid = [siteid for _, siteid in sorted(mocap_siteid, key=lambda x: x[0])]
        return mocap_siteid

    def _build_mocap_action(
        self, mocap_siteid: list[int], max_tstep: float, max_rstep: float
    ) -> Callable[[NDArray], None]:
        """Build the mocap action function to update the mocaps poses given the current action."""
        nmocap = len(mocap_siteid)
        limits = np.array([max_tstep, max_rstep]).reshape(1, 2)

        def mocap_action(action: NDArray) -> None:
            # Limit the action norms
            action = action.reshape(nmocap, 2, 3)
            step = np.sqrt(np.sum(action * action, axis=2)) + 1e-12
            action *= np.minimum(1.0, limits / step)[:, :, None]
            # Add pose offsets to site poses
            site_xpos = self.data.site_xpos.take(mocap_siteid, axis=0)
            site_xmat = self.data.site_xmat.take(mocap_siteid, axis=0)
            self.data.mocap_pos[:] = site_xpos + action[:, 0]
            quat = self.data.mocap_quat
            for i in range(nmocap):
                mujoco.mju_mat2Quat(quat[i], site_xmat[i])
                mujoco.mju_quatIntegrate(quat[i], action[i, 1], 1.0)

        return mocap_action
