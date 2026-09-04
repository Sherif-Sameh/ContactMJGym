from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from .task_space import TaskSpaceControllerAction, TaskSpaceControllerCfg

if TYPE_CHECKING:
    from ..envs.mujoco_base import InfoType, MujocoBaseEnv, ObsType
    from .task_space import FloatArray

_ParameterCfg = TaskSpaceControllerCfg.ParameterCfg

# region Config


@dataclass(slots=True)
class OscControllerCfg(TaskSpaceControllerCfg):
    """Operational Space Controller action wrapper configuration."""

    sites: tuple[str, ...] = ("gripper-tcp",)
    """Names of end-effector sites to control. A single site is expected per robot.
    Default value is ("gripper-tcp",)."""

    mocaps: tuple[str, ...] = ("mocap",)
    """Names of mocap bodies for target visualization. A single mocap is expected per
    robot. Default value is ("mocap",)."""

    linvel_sensors: tuple[str, ...] = ("tcp_linvel",)
    """Names of linear velocity sensors for end-effector sites to control. Must be
    referenced to the global frame. Default value is ("tcp_linvel",)."""

    angvel_sensors: tuple[str, ...] = ("tcp_angvel",)
    """Names of linear velocity sensors for end-effector sites to control. Must be
    referenced to the global frame. Default value is ("tcp_linvel",)."""

    dynamically_consistent: bool = False
    """If True, use the dynamically consistent Jacobian and null-space projection. Else,
    use the kinematic Jacobian and null-space projection. Default value is False."""

    decouple_dynamics: bool = True
    """If True and `dynamically_consistent` is True, only the upper and lower 3x3 blocks
    of the operational space mass matrix are computed, ignoring the coupling between
    translational and rotational dynamics. Must be True if `dynamically_consistent` and
    `null_project` are True for an accurate null-space projection. Default value is True."""

    sigma_damp: float = 1e-2
    """Damping factor for singular values when computing the operational space mass matrix
    via the SVD. Used only if `dynamically_consistent` is True. Default value is 1e-2."""

    sigma_thr: float = 1e-2
    """Singular value threshold for applying damping when computing the operational space
    mass matrix via the SVD. Used only if `dynamically_consistent` is True. Default value
    is 1e-2."""

    param_cfg: _ParameterCfg = _ParameterCfg(param_space="task")
    """Operational space stiffness and damping configuration (default override)."""

    null_project: bool = False
    """If True, add joint configuration regularization to joint accelerations via the
    null-space projection. Default value is False."""

    null_param_cfg: _ParameterCfg = _ParameterCfg()
    """Null-space stiffness and damping configuration. Parameters are always fixed."""


# region Controller


class OscControllerAction(TaskSpaceControllerAction):
    """Operational Space Controller action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through explicit operational space control, with motion
    control parameters expressed in the task-space.

    Supports both the dynamically consistant Jacobian and null-space projection, or their
    simpler kinematic variants. Additionally, supports joint configuration regularization
    towards a home configuration through the null-space projection.

    For a detailed configuration description and action convention, see
    :class:`TaskSpaceControllerAction`.

    **Notes**:
    - See the common task-space notes defined in :class:`TaskSpaceControllerAction`.
    - Site linear and angular velocity sensors must be referenced to the global frame.
    - If a dynamically-consistent null-space projection is required, `decouple_dynamics`
        must be turned off.
    - If the null-space projection is enabled, its stiffness and damping parameters are
        always fixed. Variable configs in `null_param_cfg` are ignored.

    Args:
        env: The MuJoCo-based manipulation environment to wrap. Must define mocap bodies
            welded to sites via site-to-site equality constraints.
        cfg: Configuration for task-space actions, compensation terms, motion control
            parameters and operational space parameters. See :class:`OscControllerCfg`.
    """

    @dataclass(slots=True)
    class MjcbControlData(TaskSpaceControllerAction.MjcbControlData):
        """Stores pre-computed data required for `mjcb_control` callback."""

        kp_null: FloatArray | None = None
        kv_null: FloatArray | None = None
        ctrl_projector: FloatArray | None = None
        null_projector: FloatArray | None = None

    def __init__(self, env: MujocoBaseEnv, cfg: OscControllerCfg = OscControllerCfg()):
        super().__init__(env, cfg=cfg)
        self._check_cfg(self.model, cfg)
        self.cfg = cfg  # for type-hints
        self._mjcb_data = self.MjcbControlData()
        # Store site, mocap and sensor ids
        self._siteid = [self.model.site(site).id for site in cfg.sites]
        self._mocapid = [self.model.body_mocapid[self.model.body(mocap).id] for mocap in cfg.mocaps]
        lv_snsradr = [self.model.sensor(sensor).adr[0] for sensor in cfg.linvel_sensors]
        av_snsradr = [self.model.sensor(sensor).adr[0] for sensor in cfg.angvel_sensors]
        self._lv_range = self._indices_to_slice(
            sum([list(range(adr, adr + 3)) for adr in lv_snsradr], start=[])
        )
        self._av_range = self._indices_to_slice(
            sum([list(range(adr, adr + 3)) for adr in av_snsradr], start=[])
        )
        # Store robot qpos/dof start addresses and dimension
        rbt_acts, _ = self._split_model_actuators(self.model, cfg.fltr_acts_kwargs)
        self._rbt_qpos_range, rbt_qpos_len = self._get_actuator_qpos_range(self.model, rbt_acts)
        self._per_rbt_dof_dim = rbt_qpos_len // cfg.nrobot
        if isinstance(self._rbt_qpos_range, slice):
            slc, dim = self._rbt_qpos_range, self._per_rbt_dof_dim
            self._rbt_qpos_adr = [slc.start + i * slc.step * dim for i in range(cfg.nrobot)]
        else:
            dim = self._per_rbt_dof_dim
            self._rbt_qpos_adr = [self._rbt_qpos_range[i * dim] for i in range(cfg.nrobot)]
        # Store home configuration for regularization
        if (homekey := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")) >= 0:
            self._qpos_home = (
                self.model.key_qpos[homekey, self._rbt_qpos_range].reshape(cfg.nrobot, -1).copy()
            )
        else:
            self._qpos_home = np.zeros((cfg.nrobot, self._per_rbt_dof_dim), dtype=np.float64)
        # Setup null-space control parameters
        self._mjcb_data.kp_null = (
            np.broadcast_to(cfg.null_param_cfg.kp, rbt_qpos_len).reshape((cfg.nrobot, -1)).copy()
        )
        kp_null_sqrt = np.sqrt(self._mjcb_data.kp_null)
        damping_null = (
            np.broadcast_to(cfg.null_param_cfg.damping, rbt_qpos_len)
            .reshape((cfg.nrobot, -1))
            .copy()
        )
        self._mjcb_data.kv_null = 2 * kp_null_sqrt * damping_null
        # Disable base-class mass multiplication if dynamically consistent is set
        # to avoid unnecessary M @ M_inv matrix multiplication (compensated in get_robot_ctrl)
        if cfg.dynamically_consistent:
            self._comp_mass_mult = cfg.comp_cfg.mass_mult
            self.cfg.comp_cfg.mass_mult = 0
        # Allocate buffers for OSC and null-space projection
        self._jac_full = np.zeros((cfg.nrobot, 6, self.model.nv), dtype=np.float64)
        self._jac_robot = np.zeros((cfg.nrobot, 6, self._per_rbt_dof_dim), dtype=np.float64)
        self._eye = np.eye(self._per_rbt_dof_dim, dtype=np.float64)
        self._os_mass_matrix = np.zeros((cfg.nrobot, 6, 6), dtype=np.float64)
        # Build functions to update target poses and compute pose errors
        self.update_targets = self._build_update_targets()
        self.get_pose_error = self._build_get_pose_error()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
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
        nrobot = self.cfg.nrobot
        rbt_dim = self._per_rbt_dof_dim
        self._mjcb_data.kp = self._mjcb_data.kp.reshape((nrobot, 6))
        self._mjcb_data.kv = self._mjcb_data.kv.reshape((nrobot, 6))
        # Update target mocap poses in-place
        self.update_targets(ts_action)
        # Compute full Jacobian matrix
        for i, (siteid, jac, qpos_adr) in enumerate(
            zip(self._siteid, self._jac_full, self._rbt_qpos_adr)
        ):
            mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], siteid)
            jac[3:] = self.data.site_xmat[siteid].reshape(3, 3).mT @ jac[3:]
            self._jac_robot[i] = jac[:, qpos_adr : qpos_adr + rbt_dim]
        # Compute control projector
        if self.cfg.dynamically_consistent and self._comp_mass_mult > 0:
            # Get per-robot inverse of the scaled generalized mass matrix
            mujoco.mj_fullM(self.model, self.data, self._mass_matrix)
            mass_matrix = (
                self._get_diagonal_blocks(
                    self._mass_matrix[self._rbt_dof_range, self._rbt_dof_range],
                    nblock=nrobot,
                    blocksize=rbt_dim,
                )
                * self._comp_mass_mult
            )  # (nrobot, rbt_dim, rbt_dim)
            mass_matrix_inv = np.linalg.inv(mass_matrix)
            # Compute the operational space mass matrix
            if self.cfg.decouple_dynamics:
                self._os_mass_matrix[:, :3, :3] = self._get_damped_inverse(
                    self._jac_robot[:, :3] @ mass_matrix_inv @ self._jac_robot[:, :3].mT
                )
                self._os_mass_matrix[:, 3:6, 3:6] = self._get_damped_inverse(
                    self._jac_robot[:, 3:6] @ mass_matrix_inv @ self._jac_robot[:, 3:6].mT
                )
            else:
                self._os_mass_matrix[:] = self._get_damped_inverse(
                    self._jac_robot @ mass_matrix_inv @ self._jac_robot.mT
                )
            # Compute the dynamically-consistent control projection
            self._mjcb_data.ctrl_projector = self._jac_robot.mT @ self._os_mass_matrix
        else:
            # Store the kinematic control projection
            self._mjcb_data.ctrl_projector = self._jac_robot.mT
        # Compute null space projector
        if self.cfg.null_project:
            if self.cfg.dynamically_consistent and self._comp_mass_mult > 0:
                null_projector_mult = mass_matrix
                jac_pinv_transpose = self._os_mass_matrix @ self._jac_robot @ mass_matrix_inv
            else:
                null_projector_mult = self._eye
                jac_pinv_transpose = self._get_damped_inverse(self._jac_robot).mT
            null_projector_transpose = self._eye - self._jac_robot.mT @ jac_pinv_transpose
            self._mjcb_data.null_projector = null_projector_transpose @ null_projector_mult

    def get_robot_ctrl(self, model: mujoco.MjModel, data: mujoco.MjData) -> FloatArray:
        """Compute the latest robot actuator ctrl signal."""
        nrobot = self.cfg.nrobot
        rbt_dim = self._per_rbt_dof_dim
        pose_err = self.get_pose_error()
        # Compute and project reference acceleration for main task
        linvel = data.sensordata[self._lv_range].reshape((nrobot, 3))
        site_xmat = data.site_xmat.take(self._siteid, axis=0).reshape((nrobot, 3, 3))
        angvel = data.sensordata[self._av_range].reshape((nrobot, 3, 1))
        angvel_local = (site_xmat.mT @ angvel)[:, :, 0]
        site_vel = np.concatenate([linvel, angvel_local], axis=-1)
        ctrl = self._mjcb_data.kp * pose_err - self._mjcb_data.kv * site_vel  # (nrobot, 6)
        ctrl = (self._mjcb_data.ctrl_projector @ ctrl[:, :, None])[:, :, 0]  # (nrobot, rbt_dim)
        # Compute and project reference acceleration for joint configuration regularization
        if self.cfg.null_project:
            qpos = data.qpos[self._rbt_qpos_range].reshape(nrobot, rbt_dim)
            qvel = data.qvel[self._rbt_dof_range].reshape(nrobot, rbt_dim)
            ctrl_reg = (
                self._mjcb_data.kp_null * (self._qpos_home - qpos) - self._mjcb_data.kv_null * qvel
            )
            ctrl += (self._mjcb_data.null_projector @ ctrl_reg[:, :, None])[:, :, 0]
        return ctrl.ravel()

    # region Helpers

    @staticmethod
    def _check_cfg(model: mujoco.MjModel, cfg: OscControllerCfg) -> None:
        """Perform validity checks on controller configuration."""
        assert (
            cfg.nrobot
            == len(cfg.sites)
            == len(cfg.mocaps)
            == len(cfg.linvel_sensors)
            == len(cfg.angvel_sensors)
        ), "Inconsistent nrobot, sites, mocaps, linvel_sensors and angvel_sensors configuration."
        assert cfg.param_cfg.param_space == "task", (
            "OSC controller only supports task-space parameters."
        )
        assert cfg.null_param_cfg.param_space == "joint", (
            "OSC controller null-space only supports joint-space parameters."
        )
        if cfg.dynamically_consistent and cfg.null_project:
            assert not cfg.decouple_dynamics, (
                "Decoupling dynamics cannot be enabled for a dynamically-consistent "
                "null-space projection."
            )
        assert all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0 for site in cfg.sites
        ), "Not all end-effector site names are valid. Check controller configuration."
        assert all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap) >= 0 for mocap in cfg.mocaps
        ), "Not all end-effector mocap body names are valid. Check controller configuration."
        assert all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor) >= 0
            for sensor in (cfg.linvel_sensors + cfg.angvel_sensors)
        ), "Not all end-effector velocity sensor names are valid. Check controller configuration."
        assert all(
            model.sensor_refid[model.sensor(sensor).id] == -1
            for sensor in (cfg.linvel_sensors + cfg.angvel_sensors)
        ), (
            "Not all end-effector velocity sensors are referenced to the global frame. "
            "Check controller configuration."
        )

    # region Action Helpers

    @staticmethod
    def _get_diagonal_blocks(mat: FloatArray, nblock: int, blocksize: int) -> FloatArray:
        """Extract readonly view of fixed-size blocks along diagonal of input square matrix."""
        stride_row, stride_col = mat.strides
        new_shape = (nblock, blocksize, blocksize)
        # New strides:
        # 1. Moving 1 block along the diagonal shifts down blocksize rows and right blocksize columns
        # 2. Moving 1 row inside a block shifts by 1 row
        # 3. Moving 1 col inside a block shifts by 1 column
        new_strides = (blocksize * stride_row + blocksize * stride_col, stride_row, stride_col)
        return np.lib.stride_tricks.as_strided(
            mat, shape=new_shape, strides=new_strides, writeable=False
        )

    def _build_update_targets(self) -> Callable[[FloatArray], None]:
        """Build the function to update mocap poses based-on the current site poses
        and the given task-space action."""
        nrobot = self.cfg.nrobot
        limits = np.array([self.cfg.max_tstep, self.cfg.max_rstep]).reshape(1, 2)

        def update_targets(ts_action: FloatArray) -> None:
            ts_action = ts_action.reshape(nrobot, 2, 3)
            # Limit the action norms
            step = np.sqrt(np.sum(ts_action * ts_action, axis=2)) + 1e-12
            ts_action *= np.minimum(1.0, limits / step)[:, :, None]
            # Update mocap position
            site_xpos = self.data.site_xpos.take(self._siteid, axis=0)
            self.data.mocap_pos[self._mocapid] = site_xpos + ts_action[:, 0]
            # Update mocap orientation
            quat, xmat = self.data.mocap_quat, self.data.site_xmat
            for i, (mid, sid) in enumerate(zip(self._mocapid, self._siteid)):
                mujoco.mju_mat2Quat(quat[mid], xmat[sid])
                mujoco.mju_quatIntegrate(quat[mid], ts_action[i, 1], 1.0)

        return update_targets

    def _build_get_pose_error(self) -> Callable[[], FloatArray]:
        """Compute the pose error between the current mocap targets and site poses."""
        nrobot = self.cfg.nrobot
        site_quat = np.zeros(4, dtype=np.float64)
        ori_err = np.zeros((nrobot, 3), dtype=np.float64)

        def get_pose_error() -> FloatArray:
            # Compute position error
            site_xpos = self.data.site_xpos.take(self._siteid, axis=0)
            mocap_pos = self.data.mocap_pos.take(self._mocapid, axis=0)
            pos_err = mocap_pos - site_xpos
            # Compute orienation error
            quat, xmat = self.data.mocap_quat, self.data.site_xmat
            for i, (mid, sid) in enumerate(zip(self._mocapid, self._siteid)):
                mujoco.mju_mat2Quat(site_quat, xmat[sid])
                mujoco.mju_subQuat(ori_err[i], quat[mid], site_quat)
            return np.concatenate([pos_err, ori_err], axis=1)

        return get_pose_error

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
        mat_inv = (Vt.mT * S_damped) @ U.mT
        return mat_inv
