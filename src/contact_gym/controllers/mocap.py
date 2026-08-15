from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from ..utils.mj_utils import disable_actuators
from .task_space import TaskSpaceControllerAction

if TYPE_CHECKING:
    from ..envs.base import ActType, MujocoBaseEnv
    from .task_space import WrapperActType


class MocapControllerAction(TaskSpaceControllerAction):
    """Mocap task-space action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through mocap bodies welded to end-effector sites, following
    the approach used by Gymnasium-Robotics' Fetch environments. At initialization, this
    wrapper enables the weld equality constraints between each mocap body's site and its
    corresponding end-effector site. If a `home` key exists, the default robot joint ctrl is
    adopted and joint controllers are retained to regulate the robot towards its home
    configuration in the presence of redundancies. Otherwise, the robot(s)' actuators are
    disabled, relying only on equality constraints for control. The wrapper supports any
    number of robots/grippers present in the model, each controlled through its own
    mocap-weld pair.

    Before applying an action, the mocap body is reset to coincide with its welded site's current
    pose, and the requested offset is then added on top to obtain the new mocap target. Translation
    and rotation offsets are each clipped to a maximum step size.

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
    """

    def __init__(
        self,
        env: MujocoBaseEnv,
        max_tstep: float = 0.05,
        max_rstep: float = 0.05 * np.pi,
        fltr_acts_kwargs: dict[str, Any] = {},
    ):
        nrobot = 1 if not hasattr(env.unwrapped, "model") else env.unwrapped.model.nmocap
        super().__init__(env, nrobot, max_tstep, max_rstep, fltr_acts_kwargs)
        assert self.env.unwrapped.model.nmocap > 0
        self.data = self.env.unwrapped.data
        # Enable mocap weld constraints and get mocap -> site id mapping
        mocap_siteid = self._setup_mocap_bodies(self.env.unwrapped.model)
        # Disable actuators if no home key exists
        if self.env.unwrapped.model.key("home") is None:
            nactuator = self.env.unwrapped.model.nactuator
            gri_idxs = self._get_gripper_indices(self.env.unwrapped.model, fltr_acts_kwargs)
            robot_idxs = [i for i in range(nactuator) if i not in gri_idxs]
            disable_actuators(self.env.unwrapped.model, robot_idxs)
        # Build action function for mocap bodies
        self.mocap_action = self._build_mocap_action(mocap_siteid, max_tstep, max_rstep)

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
        self, mocap_siteid: list[int], max_tstep: float, max_rstep: float
    ) -> Callable[[WrapperActType], None]:
        """Build the mocap action function to update mocap poses given the current action."""
        nmocap = len(mocap_siteid)
        limits = np.array([max_tstep, max_rstep]).reshape(1, 2)

        def mocap_action(action: WrapperActType) -> None:
            action = action[: 6 * nmocap].reshape(nmocap, 2, 3)
            # Limit the action norms
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
