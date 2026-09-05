from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

try:
    import mink
except ImportError:
    pass  # don't force onto users of other controllers

from .task_space import TaskSpaceControllerAction, TaskSpaceControllerCfg

if TYPE_CHECKING:
    from ...envs.mujoco_base import InfoType, MujocoBaseEnv, ObsType
    from .task_space import FloatArray

_ParameterCfg = TaskSpaceControllerCfg.ParameterCfg

# region Config


@dataclass(slots=True)
class MinkControllerCfg(TaskSpaceControllerCfg):
    """Mink task-space controller action wrapper configuration."""

    @dataclass(frozen=True, slots=True)
    class MinkCfg:
        """Configuration for the mink tasks, IK solver, thresholds, etc."""

        sites: tuple[str, ...] = ("gripper-tcp",)
        """Names of end-effector sites to control. A single site is expected per robot.
        Default value is ("gripper-tcp",)."""

        mocaps: tuple[str, ...] = ("mocap",)
        """Names of mocap bodies for target visualization. A single mocap is expected per
        robot. Default value is ("mocap",)."""

        max_iters: int = 5
        """Maximum number of iterations for solving IK per action. Default value is 5."""

        dt: float = 5e-3
        """Integration timestep for IK solver in seconds. If < 0, the sim's dt is used
        instead. Default value is 5e-3."""

        solver: str = "daqp"
        """Backend QP solver used to solve IK. Default value is daqp."""

        damping: float = 1e-5
        """LM damping applied to all tasks when solving IK. Default value is 1e-5."""

        pos_thr: float = 1e-3
        """Position error threshold in meters for early termination.
        Default value is 1e-3."""

        ori_thr: float = 1.5e-3
        """Orientation error threshold in radians for early termination.
        Default value is 1.5e-3."""

        @dataclass(frozen=True)
        class FrameTaskCfg:
            """Configuration for the main :class:`mink.FrameTask` pose tracking tasks."""

            position_cost: float = 1.0
            orientation_cost: float = 1.0
            lm_damping: float = 1.0

        frame_task_cfg: FrameTaskCfg = FrameTaskCfg()
        """Configuration for the main :class:`mink.FrameTask` pose tracking tasks."""

        @dataclass(frozen=True)
        class PostureTaskCfg:
            """Configuration for the :class:`mink.PostureTask` regularization task."""

            cost: float = 0.05
            gain: float = 1.0
            lm_damping: float = 0.0

        posture_task_cfg: PostureTaskCfg = PostureTaskCfg()
        """Configuration for the :class:`mink.PostureTask` regularization task."""

        @dataclass(frozen=True)
        class ConfigurationLimitCfg:
            """Configuration for the :class:`mink.ConfigurationLimit` limit."""

            gain: float = 0.95
            min_distance_from_limits: float = 0

        configuration_limit_cfg: ConfigurationLimitCfg = ConfigurationLimitCfg()
        """Configuration for the :class:`mink.ConfigurationLimit` limit."""

    mink_cfg: MinkCfg = MinkCfg()
    """Configuration for the mink tasks, IK solver, thresholds, etc. See :class:`MinkCfg`."""

    aux_limits: list[mink.Limit] = field(default_factory=list)
    """Auxiliary limits to add to the default :class:`mink.ConfigurationLimit`.
    Default value is an empty list."""

    param_cfg: _ParameterCfg = _ParameterCfg(kp=2000, damping=1.3)
    """Stiffness and damping parameters configuration (default override)."""


# region Controller


class MinkControllerAction(TaskSpaceControllerAction):
    """Mink task-space action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through `mink`'s IK solver, integrating posture
    regularization tasks and hard limits. At initialization, a :class:`mink.FrameTask`
    is added for each end-effector site, in addition to a single :class:`mink.PostureTask`
    for regularization towards a home configuration. Additionally, a
    :class:`mink.ConfigurationLimit` is used to avoid exceeding joint limits.

    For a detailed configuration description and action convention, see
    :class:`TaskSpaceControllerAction`.

    **Notes**:
    - See the common task-space notes defined in :class:`TaskSpaceControllerAction`.
    - Override default control gains according to robot and controller configuration for
        optimal/smooth performance.
    - Wrapper relies on `mink`'s IK solver to compute target qpos, integrating frame and
        regularization tasks, then joint accelerations are computed via the configured or
        input (if variable) motion control parameters.

    Args:
        env: The MuJoCo-based manipulation environment to wrap. Must define matching
            mocap bodies for every end-effector site.
        cfg: Configuration for task-space actions, compensation terms, motion control
            parameters and `mink`. See :class:`MinkControllerCfg`. Default value is None.
    """

    @dataclass(slots=True)
    class MjcbControlData(TaskSpaceControllerAction.MjcbControlData):
        """Stores pre-computed data required for `mjcb_control` callback."""

        qpos_target: FloatArray | None = None

    def __init__(self, env: MujocoBaseEnv, cfg: MinkControllerCfg | None = None) -> None:
        import mink  # ensure mink is installed

        cfg = MinkControllerCfg() if cfg is None else cfg
        super().__init__(env, cfg=cfg)
        self._check_cfg(self.model, cfg)
        self.cfg = cfg  # for type-hints
        self._mjcb_data = self.MjcbControlData()
        # Store target site and visualization mocap ids
        self._siteid = [self.model.site(site).id for site in cfg.mink_cfg.sites]
        self._mocapid = [
            self.model.body_mocapid[self.model.body(mocap).id] for mocap in cfg.mink_cfg.mocaps
        ]
        # Setup mink configuration, tasks, and limits
        self._configuration = mink.Configuration(self.model)
        self._tasks = [
            mink.FrameTask(site, "site", **asdict(cfg.mink_cfg.frame_task_cfg))
            for site in cfg.mink_cfg.sites
        ] + [mink.PostureTask(self.model, **asdict(cfg.mink_cfg.posture_task_cfg))]
        self._frame_tasks = self._tasks[:-1]
        self._limits = [
            mink.ConfigurationLimit(self.model, **asdict(cfg.mink_cfg.configuration_limit_cfg))
        ] + cfg.aux_limits
        # Store robot qpos range
        rbt_acts, _ = self._split_model_actuators(self.model, cfg.fltr_acts_kwargs)
        self._rbt_qpos_range, _ = self._get_actuator_qpos_range(self.model, rbt_acts)
        # Setup home configuration for posture task
        if (homekey := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")) >= 0:
            qpos_home = self.model.key_qpos[homekey].copy()
        else:
            qpos_home = np.zeros(self.model.nq, dtype=np.float64)
        self._configuration.update(qpos_home)
        self._tasks[-1].set_target_from_configuration(self._configuration)
        # Build action function to compute target qpos via mink
        self.mink_action = self._build_mink_action()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.
        """
        obs, info = super().reset(seed=seed, options=options)
        # Reset frame tasks and mocap
        for task, mid, sid in zip(self._frame_tasks, self._mocapid, self._siteid):
            site_xpos, site_xmat = self.data.site_xpos[sid], self.data.site_xmat[sid]
            mocap_pos, mocap_quat = self.data.mocap_pos[mid], self.data.mocap_quat[mid]
            mocap_pos[:] = site_xpos
            mujoco.mju_mat2Quat(mocap_quat, site_xmat)
            task.set_target(mink.SE3(wxyz_xyz=np.concatenate([mocap_quat, mocap_pos])))
        return obs, info

    def precompute_data(self, ts_action: FloatArray) -> None:
        """Pre-compute data needed for ctrl computation during mjcb_control callback.

        Called at the same rate as the environment stepping rate (sim_freq // frame_skip).

        Args:
            ts_action: Task-space control action for position and orientation.
                Shape is (6 * `nrobot`).
        """
        # Update target qpos via mink
        self._mjcb_data.qpos_target = self.mink_action(self.data, ts_action)

    def get_robot_ctrl(self, model: mujoco.MjModel, data: mujoco.MjData) -> FloatArray:
        """Compute the latest robot actuator ctrl signal."""
        qpos = data.qpos[self._rbt_qpos_range]
        qvel = data.qvel[self._rbt_dof_range]
        ctrl = self._mjcb_data.kp * (self._mjcb_data.qpos_target - qpos) - self._mjcb_data.kv * qvel
        return ctrl

    # region Helpers

    @staticmethod
    def _check_cfg(model: mujoco.MjModel, cfg: MinkControllerCfg) -> None:
        """Perform validity checks on controller configuration."""
        assert cfg.param_cfg.param_space == "joint", (
            "Mink controller only supports joint-space parameters."
        )
        assert cfg.nrobot == len(cfg.mink_cfg.sites), (
            "Number of robots in config does not match number of target sites. "
            f"Got {cfg.nrobot} robots and {len(cfg.mink_cfg.sites)} sites."
        )
        assert len(cfg.mink_cfg.sites) == len(cfg.mink_cfg.mocaps), (
            "Expected matching end-effector sites and mocap bodies. "
            f"Got {len(cfg.mink_cfg.sites)} sites and {len(cfg.mink_cfg.mocaps)} mocaps."
        )
        assert all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0
            for site in cfg.mink_cfg.sites
        ), "Not all end-effector site names are valid. Check mink configuration."
        assert all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap) >= 0
            for mocap in cfg.mink_cfg.mocaps
        ), "Not all end-effector mocap body names are valid. Check mink configuration."

    # region Action Helpers

    def _build_mink_action(self) -> Callable[[mujoco.MjData, FloatArray], FloatArray]:
        """Build the action function to solve IK via mink and compute target qpos given
        the current task-space action."""
        mink_cfg = self.cfg.mink_cfg
        dt = self.model.opt.timestep if mink_cfg.dt <= 0 else mink_cfg.dt
        pos_thr_sqr, ori_thr_sqr = mink_cfg.pos_thr**2, mink_cfg.ori_thr**2
        limits = np.array([self.cfg.max_tstep, self.cfg.max_rstep]).reshape(1, 2, 1)

        def mink_action(data: mujoco.MjData, ts_action: FloatArray) -> FloatArray:
            self._configuration.update(data.qpos)
            self._update_frame_targets(data, ts_action, limits)
            for _ in range(mink_cfg.max_iters):
                vel = mink.solve_ik(
                    self._configuration,
                    self._tasks,
                    dt,
                    mink_cfg.solver,
                    damping=mink_cfg.damping,
                    limits=self._limits,
                )
                self._configuration.integrate_inplace(vel, dt)
                if self._has_converged(pos_thr_sqr, ori_thr_sqr):
                    break
            return self._configuration.q[self._rbt_qpos_range]

        return mink_action

    def _update_frame_targets(
        self, data: mujoco.MjData, ts_action: FloatArray, limits: FloatArray
    ) -> None:
        """Update frame tasks targets using their current poses and task-space actions."""
        ts_action = ts_action.reshape(self.cfg.nrobot, 2, 3)
        # Limit the action norms
        step = np.sqrt(np.sum(ts_action * ts_action, axis=2, keepdims=True)) + 1e-12
        ts_action *= np.minimum(1.0, limits / step)
        # Add pose offsets to mocap poses in-place and update targets
        for i, (task, mid) in enumerate(zip(self._frame_tasks, self._mocapid)):
            mocap_pos, mocap_quat = data.mocap_pos[mid], data.mocap_quat[mid]
            mocap_pos += ts_action[i, 0]
            mujoco.mju_quatIntegrate(mocap_quat, ts_action[i, 1], 1.0)
            target = task.transform_target_to_world.wxyz_xyz
            target[:4], target[4:] = mocap_quat, mocap_pos

    def _has_converged(self, pos_threshold_sqr: float, ori_threshold_sqr: float) -> bool:
        """Check whether all frame tasks have converged according to the set error thresholds."""
        for task in self._frame_tasks:
            err = task.compute_error(self._configuration)
            err_sqr = err * err
            if np.sum(err_sqr[:3]) > pos_threshold_sqr or np.sum(err_sqr[3:]) > ori_threshold_sqr:
                return False
        return True
