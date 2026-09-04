from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from .task_space import TaskSpaceControllerAction, TaskSpaceControllerCfg

if TYPE_CHECKING:
    from ..envs.mujoco_base import InfoType, MujocoBaseEnv, ObsType
    from .task_space import FloatArray


# region Config


@dataclass(slots=True)
class MocapControllerCfg(TaskSpaceControllerCfg):
    """Mocap task-space controller action wrapper configuration."""

    eq_solimp: tuple[float, ...] | None = None
    """Optional solver impedence parameters for overriding weld equality constraint
    parameters. Default value is None."""

    eq_solref: tuple[float, ...] | None = None
    """Optional solver reference parameters for overriding weld equality constraint
    parameters. Default value is None."""

    null_project: bool = False
    """If True, apply null-space projection to regularization joint accelerations.
    Default value is False."""

    sigma_damp: float = 1e-2
    """Damping factor for singular values when computing Moore-Penrose pseudoinverse
    of the Jacobian via the SVD. Default value is 1e-2."""

    sigma_thr: float = 1e-2
    """Singular value threshold for applying damping when computing Moore-Penrose
    pseudoinverse of the Jacobian via the SVD. Default value is 1e-2."""


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
    - Override default control gains and weld equality parameters according to robot and
        controller configuration for optimal/smooth performance.
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
        cfg: Configuration for task-space actions, compensation terms, motion control
            parameters and null-space projection. See :class:`MocapControllerCfg`.
            Default value is None.
    """

    @dataclass(slots=True)
    class MjcbControlData(TaskSpaceControllerAction.MjcbControlData):
        """Stores pre-computed data required for `mjcb_control` callback."""

        null_projector: FloatArray | None = None

    def __init__(self, env: MujocoBaseEnv, cfg: MocapControllerCfg | None = None):
        cfg = MocapControllerCfg() if cfg is None else cfg
        super().__init__(env, cfg=cfg)
        self._check_cfg(self.model, cfg)
        self.cfg = cfg  # for type-hints
        self._mjcb_data = self.MjcbControlData()
        # Store robot qpos range
        rbt_acts, _ = self._split_model_actuators(self.model, cfg.fltr_acts_kwargs)
        self._rbt_qpos_range, rbt_qpos_len = self._get_actuator_qpos_range(self.model, rbt_acts)
        # Store home configuration for regularization
        if (homekey := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")) >= 0:
            self._qpos_home = self.model.key_qpos[homekey, self._rbt_qpos_range].copy()
        else:
            self._qpos_home = np.zeros(rbt_qpos_len, dtype=np.float64)
        # Allocate buffers for null-space projection
        self._jac_full = np.zeros((6, self.model.nv), dtype=np.float64)
        self._jac_robot = np.zeros_like(self._jac_full)
        self._eye = np.eye(rbt_qpos_len, dtype=np.float64)
        # Enable mocap weld constraints and get mocap -> site id mapping
        self._mocapid, self._siteid = self._setup_mocap_bodies(
            self.model, self.data, cfg.nrobot, cfg.eq_solimp, cfg.eq_solref
        )
        # Build action function for mocap bodies
        self.mocap_action = self._build_mocap_action()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.
        """
        obs, info = super().reset(seed=seed, options=options)
        # Reset mocap bodies to corresponding sites
        site_xpos = self.data.site_xpos.take(self._siteid, axis=0)
        self.data.mocap_pos[self._mocapid] = site_xpos
        for mid, sid in zip(self._mocapid, self._siteid):
            mujoco.mju_mat2Quat(self.data.mocap_quat[mid], self.data.site_xmat[sid])
        return obs, info

    def precompute_data(self, ts_action: FloatArray) -> None:
        """Pre-compute data needed for ctrl computation during mjcb_control callback.

        Called at the same rate as the environment stepping rate (sim_freq // frame_skip).

        Args:
            ts_action: Task-space control action for position and orientation.
                Shape is (6 * `nrobot`).
        """
        # Update target mocap poses in-place
        self.mocap_action(self.data, ts_action)
        # Compute Jacobian and null projector
        if self.cfg.null_project:
            mujoco.mj_jacSite(
                self.model, self.data, self._jac_full[:3], self._jac_full[3:], self._siteid[0]
            )
            for siteid in self._siteid[1:]:  # robots assumed independent
                mujoco.mj_jacSite(
                    self.model, self.data, self._jac_robot[:3], self._jac_robot[3:], siteid
                )
                self._jac_full += self._jac_robot
            jac = self._jac_full[:, self._rbt_dof_range]
            jac_pinv = self._get_damped_inverse(jac)
            self._mjcb_data.null_projector = self._eye - jac_pinv @ jac

    def get_robot_ctrl(self, model: mujoco.MjModel, data: mujoco.MjData) -> FloatArray:
        """Compute the latest robot actuator ctrl signal."""
        # Update configuration regularization ctrl signal
        qpos = data.qpos[self._rbt_qpos_range]
        qvel = data.qvel[self._rbt_dof_range]
        ctrl = self._mjcb_data.kp * (self._qpos_home - qpos) - self._mjcb_data.kv * qvel
        # Apply null-space projection
        if self.cfg.null_project:
            ctrl = self._mjcb_data.null_projector @ ctrl
        return ctrl

    # region Helpers

    @staticmethod
    def _check_cfg(model: mujoco.MjModel, cfg: MocapControllerCfg) -> None:
        """Perform validity checks on controller configuration."""
        assert cfg.param_cfg.param_space == "joint", (
            "Mocap controller only supports joint-space parameters."
        )
        assert model.nmocap >= cfg.nrobot

    @staticmethod
    def _setup_mocap_bodies(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        nrobot: int,
        solimp: tuple[float, ...] | None,
        solref: tuple[float, ...] | None,
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
        return body_mocapid, mocap_siteid

    # region Action Helpers

    def _build_mocap_action(self) -> Callable[[mujoco.MjData, FloatArray], None]:
        """Build the mocap action function to update mocap poses based-on their current
        poses and the given task-space action."""
        nrobot = self.cfg.nrobot
        limits = np.array([self.cfg.max_tstep, self.cfg.max_rstep]).reshape(1, 2, 1)

        def mocap_action(data: mujoco.MjData, ts_action: FloatArray) -> None:
            ts_action = ts_action.reshape(nrobot, 2, 3)
            # Limit the action norms
            step = np.sqrt(np.sum(ts_action * ts_action, axis=2, keepdims=True)) + 1e-12
            ts_action *= np.minimum(1.0, limits / step)
            # Add pose offsets to mocap poses in-place
            data.mocap_pos[self._mocapid] += ts_action[:, 0]
            quat = data.mocap_quat
            for i, mid in enumerate(self._mocapid):
                mujoco.mju_quatIntegrate(quat[mid], ts_action[i, 1], 1.0)

        return mocap_action

    def _get_damped_inverse(self, mat: FloatArray) -> FloatArray:
        """Compute the damped inverse or Moore-Penrose pseudoinverse of the input
        matrix using its SVD."""
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        # Apply variable damping to small singular values
        lambda_sqr = np.where(
            S < self.cfg.sigma_thr,
            self.cfg.sigma_damp**2 * (1.0 - (S / self.cfg.sigma_thr) ** 2),
            0.0,
        )
        S_damped = S / (S**2 + lambda_sqr)
        mat_inv = (Vt.T * S_damped) @ U.T
        return mat_inv
