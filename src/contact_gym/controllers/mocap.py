from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from ..utils.mj_utils import MJTJOINT_TO_QPOS_DIM
from .task_space import TaskSpaceControllerAction, TaskSpaceControllerCfg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..envs.mujoco_base import InfoType, MujocoBaseEnv, ObsType
    from .task_space import FloatArray


# region Controller


class MocapControllerAction(TaskSpaceControllerAction):
    """Mocap task-space action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through mocap bodies welded to end-effector sites, following
    the approach used by Gymnasium-Robotics' Fetch environments. At initialization, this
    wrapper enables the weld equality constraints between each mocap body's site and its
    corresponding end-effector site, optionally overriding its default solver parameters.

    Supports applying a null-space projection to the computed joint accelerations towards
    a home configuration for regularization without affecting the main task-space
    tracking task. Otherwise, the same regularization will be applied, using the
    configured or input (if variable) motion control parameters without the null-space
    projection.

    For a detailed configuration description and action convention, see
    :class:`TaskSpaceControllerAction`.

    **Notes**:
    - See the common task-space notes defined in :class:`TaskSpaceControllerAction`.
    - Wrapper assumes weld equality constraints are defined between *sites*
        (mocap site <-> gripper site), not bodies.
    - Since the main end-effector driving force comes from the weld equality constraint,
        which is not available when computing controls, a large fraction of the
        motion-inducing force is not multiplied by the generalized mass matrix or exposed
        to the controller. Hence, `mass_mult` will not be physically very accurate and
        friction over-compensation due to `fric_mult` is > 1 will also not be accurate.

    Args:
        env: The MuJoCo-based manipulation environment to wrap. Must define mocap bodies
            welded to sites via site-to-site equality constraints.
        cfg: Configuration for task-space actions, compensation terms and motion control
            parameters. See :class:`TaskSpaceControllerCfg`.
        solimp: Optional solver impedence parameters for overriding weld equality
            constraint parameters. Default value is None.
        solref: Optional solver reference parameters for overriding weld equality
            constraint parameters. Default value is None.
        null_project: If True, apply null-space projection to regularization joint
            accelerations. Default value is False.
        sigma_damp: Damping factor for singular values when computing Moore-Penrose
            pseudoinverse of the Jacobian via the SVD. Default value is 1e-3.
        sigma_thr: Singular value threshold for applying damping when computing
            Moore-Penrose pseudoinverse of the Jacobian via the SVD. Default value is 1e-5.
    """

    def __init__(
        self,
        env: MujocoBaseEnv,
        cfg: TaskSpaceControllerCfg = TaskSpaceControllerCfg(),
        solimp: Sequence[float] | None = None,
        solref: Sequence[float] | None = None,
        null_project: bool = False,
        sigma_damp: float = 1e-3,
        sigma_thr: float = 1e-5,
    ):
        super().__init__(env, cfg=cfg)
        assert cfg.param_cfg.param_space == "joint", (
            "Mocap controller only supports joint-space parameters."
        )
        assert env.unwrapped.model.nmocap >= cfg.nrobot
        self.null_project = null_project
        self.sigma_damp = sigma_damp
        self.sigma_thr = sigma_thr
        # Store robot qpos range
        rbt_acts, _ = self._split_model_actuators(self.model, cfg.fltr_acts_kwargs)
        self._rbt_qpos_range, rbt_qpos_len = self._get_actuator_qpos_range(self.model, rbt_acts)
        # Store home configuration for regularization
        if (homekey := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")) >= 0:
            self._qpos_home = self.model.key_qpos[homekey, self._rbt_qpos_range].copy()
        else:
            self._qpos_home = np.zeros(rbt_qpos_len, dtype=np.float64)
        # Allocate buffers for Jacobian
        self._jac = np.zeros((6, self.model.nv), dtype=np.float64)
        self._jac_robot = np.zeros_like(self._jac)
        # Enable mocap weld constraints and get mocap -> site id mapping
        self._mocapid, self._mocap_siteid = self._setup_mocap_bodies(
            self.model, self.data, cfg.nrobot, solimp, solref
        )
        # Build action function for mocap bodies
        self.mocap_action = self._build_mocap_action()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # Reset mocap bodies to corresponding sites
        site_xpos = self.data.site_xpos.take(self._mocap_siteid, axis=0)
        self.data.mocap_pos[self._mocapid] = site_xpos
        for mid, sid in zip(self._mocapid, self._mocap_siteid):
            mujoco.mju_mat2Quat(self.data.mocap_quat[mid], self.data.site_xmat[sid])
        return obs, info

    def get_robot_ctrl(self, ts_action: FloatArray, kp: FloatArray, kv: FloatArray) -> FloatArray:
        """Compute the latest robot actuator ctrl signal.

        Args:
            ts_action: Task-space control action for position and orientation.
                Shape is (6 * `nrobot`).
            kp: Positional gain (stiffness) for motion control. Shape is (`ncontrol`,).
            kv: Velocity gain (computed from `kp` and damping ratio) for motion control.
                Shape is (`ncontrol`,).

        Returns:
            Robot control signal computed from task-space action and motion control gains.
        """
        self.mocap_action(ts_action)
        ctrl_reg = (
            kp * (self._qpos_home - self.data.qpos[self._rbt_qpos_range])
            - kv * self.data.qvel[self._rbt_dof_range]
        )
        if self.null_project:
            # Compute full Jacobian matrix
            mujoco.mj_jacSite(
                self.model, self.data, self._jac[:3], self._jac[3:], self._mocap_siteid[0]
            )
            for siteid in self._mocap_siteid[1:]:  # robots assumed independent
                mujoco.mj_jacSite(
                    self.model, self.data, self._jac_robot[:3], self._jac_robot[3:], siteid
                )
                self._jac += self._jac_robot
            # Apply null-space projection
            null_projector = self._get_nullspace_projector(
                self._jac[self._rbt_dof_range, self._rbt_dof_range]
            )
            ctrl_reg = null_projector @ ctrl_reg
        return ctrl_reg

    # region Helpers

    @staticmethod
    def _get_actuator_qpos_range(
        model: mujoco.MjModel, actuators: list[int]
    ) -> tuple[slice | tuple[int, ...], int]:
        """Get the range (slice or indices) that correspond to the given actuators in qpos."""
        qpos_indices = []
        for act in actuators:
            jnt_id = model.actuator_trnid[act, 0]
            qposadr = model.jnt_qposadr[jnt_id]
            qposdim = MJTJOINT_TO_QPOS_DIM[model.jnt_type[jnt_id]]
            qpos_indices.extend(list(range(qposadr, qposadr + qposdim)))
        return MocapControllerAction._indices_to_slice(qpos_indices), len(qpos_indices)

    @staticmethod
    def _setup_mocap_bodies(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        nrobot: int,
        solimp: Sequence[float] | None,
        solref: Sequence[float] | None,
    ) -> tuple[slice | tuple[int, ...], slice | tuple[int, ...]]:
        """Enable weld constraints involving mocap bodies and return mocap -> site id map."""
        # Enable weld constraints and establish mocap -> site id map
        body_mocapid, mocap_siteid = [], []
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
                data.eq_active[i] = 1
                if solimp is not None:
                    model.eq_solimp[i] = solimp
                if solref is not None:
                    model.eq_solref[i] = solref
                if mocap1id >= 0:  # obj1 is the mocap site
                    mocapid = mocap1id
                    siteid = model.eq_obj2id[i]
                else:  # obj2 is the mocap site
                    mocapid = mocap2id
                    siteid = model.eq_obj1id[i]
                body_mocapid.append(mocapid)
                mocap_siteid.append(siteid)
        assert len(mocap_siteid) == nrobot, (
            f"Found only {len(mocap_siteid)}/{nrobot} mocap weld constraint."
        )
        body_mocapid = MocapControllerAction._indices_to_slice(body_mocapid)
        mocap_siteid = MocapControllerAction._indices_to_slice(mocap_siteid)
        return body_mocapid, mocap_siteid

    # region Action Helpers

    def _build_mocap_action(self) -> Callable[[FloatArray], None]:
        """Build the mocap action function to update mocap poses given the current
        task-space action."""
        nmocap = len(self._mocap_siteid)
        limits = np.array([self.cfg.max_tstep, self.cfg.max_rstep]).reshape(1, 2)

        def mocap_action(ts_action: FloatArray) -> None:
            ts_action = ts_action.reshape(nmocap, 2, 3)
            # Limit the action norms
            step = np.sqrt(np.sum(ts_action * ts_action, axis=2)) + 1e-12
            ts_action *= np.minimum(1.0, limits / step)[:, :, None]
            # Add pose offsets to mocap poses in place
            self.data.mocap_pos[self._mocapid] += ts_action[:, 0]
            quat = self.data.mocap_quat
            for i, mid in enumerate(self._mocapid):
                mujoco.mju_quatIntegrate(quat[mid], ts_action[i, 1], 1.0)

        return mocap_action

    def _get_nullspace_projector(self, jac: FloatArray) -> FloatArray:
        """Compute the damped Moore-Penrose pseudoinverse of the Jacobian using its SVD."""
        U, S, Vt = np.linalg.svd(jac, full_matrices=False)
        # Apply variable damping to small singular values
        lambda_sqr = np.where(
            S < self.sigma_thr, self.sigma_damp**2 * (1.0 - (S / self.sigma_thr) ** 2), 0.0
        )
        S_damped = S / (S**2 + lambda_sqr)
        jac_pinv = (Vt.T * S_damped) @ U.T
        null_projector = np.eye(jac_pinv.shape[1]) - jac_pinv @ jac
        return null_projector
