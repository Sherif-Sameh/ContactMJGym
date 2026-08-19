from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from ..utils.mj_utils import disable_actuators
from .task_space import TaskSpaceControllerAction

if TYPE_CHECKING:
    from ..envs.mujoco_base import ActType, InfoType, MujocoBaseEnv, ObsType
    from .task_space import WrapperActType


class MocapControllerAction(TaskSpaceControllerAction):
    """Mocap task-space action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through mocap bodies welded to end-effector sites, following
    the approach used by Gymnasium-Robotics' Fetch environments. At initialization, this
    wrapper enables the weld equality constraints between each mocap body's site and its
    corresponding end-effector site. Optionally, the robot(s)' joint controllers can be
    retained to regulate it towards its initial configuration, which is updated at every
    environment reset. Otherwise, the robot(s)' actuators are disabled, relying only on
    equality constraints for control. The wrapper supports any number of robots/grippers
    present in the model, each controlled through its own mocap-weld pair.

    For action convention, see :class:`TaskSpaceControllerAction`.

    **Note**: this wrapper assumes weld equality constraints are defined between
    *sites* (mocap site <-> gripper site), not bodies, and that *no other* action wrappers
    have been already applied to the environment.

    Args:
        env: The MuJoCo-based manipulation environment to wrap. Must define mocap bodies
            welded to sites via site-to-site equality constraints.
        max_tstep: Maximum translation step size (Euclidean norm, in meters)
            applied per action. Default value is 0.05.
        max_rstep: Maximum rotation step size (norm of the rotation vector,
            in radians) applied per action. Defaults value is 0.05 * pi.
        fltr_acts_kwargs: Kwargs for filtering for gripper actuators. For details, see
            :func:`~..utils.mj_utils.filter_actuators`. If empty, we rely on a simple
            heuristic by filtering for actuators whose `trntype` is
            `mujoco.mjtTrn.mjTRN_TENDON`. Default value is an empty dict.
        disable_acts: If True, the robot(s)' actuators are disabled (gripper actuators are
            unaffected). Default value is False.
    """

    def __init__(
        self,
        env: MujocoBaseEnv,
        max_tstep: float = 0.05,
        max_rstep: float = 0.05 * np.pi,
        fltr_acts_kwargs: dict[str, Any] = {},
        disable_acts: bool = False,
    ):
        nrobot = 1 if not hasattr(env.unwrapped, "model") else env.unwrapped.model.nmocap
        super().__init__(env, nrobot, max_tstep, max_rstep, fltr_acts_kwargs)
        assert env.unwrapped.model.nmocap > 0
        # Enable mocap weld constraints and get mocap -> site id mapping
        self._mocap_siteid = self._setup_mocap_bodies(env.unwrapped.model)
        # Disable actuators if requested
        if disable_acts:
            nactuator = env.unwrapped.model.nactuator
            gri_acts = self._get_gripper_actuators(env.unwrapped.model, fltr_acts_kwargs)
            rbt_acts = [i for i in range(nactuator) if i not in gri_acts]
            disable_actuators(env.unwrapped.model, rbt_acts)
        # Build action function for mocap bodies
        self.mocap_action = self._build_mocap_action(max_tstep, max_rstep)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.
        """
        obs, info = super().reset(seed=seed, options=options)
        # Reset mocap bodies to corresponding sites
        site_xpos = self.data.site_xpos.take(self._mocap_siteid, axis=0)
        site_xmat = self.data.site_xmat.take(self._mocap_siteid, axis=0)
        self.data.mocap_pos[:] = site_xpos
        for quat, xmat in zip(self.data.mocap_quat, site_xmat):
            mujoco.mju_mat2Quat(quat, xmat)
        # Reset action buffer
        self.action_buffer[:] = self.data.ctrl
        return obs, info

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        action = self.unscale_action(action)
        self.mocap_action(action)
        self.gripper_action(action)
        return self.action_buffer

    # region Helpers

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
        self, max_tstep: float, max_rstep: float
    ) -> Callable[[WrapperActType], None]:
        """Build the mocap action function to update mocap poses given the current action."""
        nmocap = len(self._mocap_siteid)
        limits = np.array([max_tstep, max_rstep]).reshape(1, 2)

        def mocap_action(action: WrapperActType) -> None:
            action = action[: 6 * nmocap].reshape(nmocap, 2, 3)
            # Limit the action norms
            step = np.sqrt(np.sum(action * action, axis=2)) + 1e-12
            action *= np.minimum(1.0, limits / step)[:, :, None]
            # Add pose offsets to mocap poses in place
            self.data.mocap_pos += action[:, 0]
            quat = self.data.mocap_quat
            for i in range(nmocap):
                mujoco.mju_quatIntegrate(quat[i], action[i, 1], 1.0)

        return mocap_action
