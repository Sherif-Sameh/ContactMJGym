from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

try:
    import mink
except ImportError:
    pass  # don't force onto users of other controllers

from .task_space import TaskSpaceControllerAction

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..envs.mujoco_base import ActType, InfoType, MujocoBaseEnv, ObsType
    from .task_space import WrapperActType


class MinkControllerAction(TaskSpaceControllerAction):
    """Mink task-space action wrapper for MuJoCo manipulation environments.

    Exposes end-effector control through `mink`'s IK solver, integrating posture
    regularization tasks and hard limits. At initialization, a :class:`mink.FrameTask`
    is added for each end-effector site, as well as a :class:`mink.PostureTask` for
    regularization towards the initial configuration, which is updated at every
    environment reset. Additionally, a :class:`mink.ConfigurationLimit` is used to avoid
    exceeding joint limits. The wrapper supports any number of robots/grippers present in
    the model, each driven by a separate :class:`mink.FrameTask`.

    For action convention, see :class:`TaskSpaceControllerAction`.

    **Note**: this wrapper assumes that *no other* action wrappers have been already
    applied to the environment.

    Args:
        env: The MuJoCo-based manipulation environment to wrap. Must define matching
            mocap bodies for every end-effector site.
        max_tstep: Maximum translation step size (Euclidean norm, in meters)
            applied per action. Default value is 0.02.
        max_rstep: Maximum rotation step size (norm of the rotation vector,
            in radians) applied per action. Defaults value is 0.04 * pi.
        fltr_acts_kwargs: Kwargs for filtering for gripper actuators. For details, see
            :func:`~..utils.mj_utils.filter_actuators`. If empty, we rely on a simple
            heuristic by filtering for actuators whose `trntype` is
            `mujoco.mjtTrn.mjTRN_TENDON`. Default value is an empty dict.
        mink_cfg: Mink configuration. Determines sites, mocaps, solver config, error
            thresholds, and task + limit configs. If None, default values are used.
            Default value is None.
        aux_limits: Auxiliary limits to add to the default :class:`mink.ConfigurationLimit`.
            Default value is [].
    """

    def __init__(
        self,
        env: MujocoBaseEnv,
        max_tstep: float = 0.02,
        max_rstep: float = 0.04 * np.pi,
        fltr_acts_kwargs: dict[str, Any] = {},
        mink_cfg: MinkCfg | None = None,
        aux_limits: list[mink.Limit] = [],
    ) -> None:
        import mink  # ensure mink is installed

        mink_cfg = MinkCfg() if mink_cfg is None else mink_cfg
        nrobot = len(mink_cfg.sites)
        super().__init__(env, nrobot, max_tstep, max_rstep, fltr_acts_kwargs)
        assert len(mink_cfg.sites) == len(mink_cfg.mocaps), (
            "Expected matching end-effector sites and mocap bodies. "
            f"Got {len(mink_cfg.sites)} sites and {len(mink_cfg.mocaps)} mocaps."
        )
        assert not any(env.unwrapped.model.site(site) is None for site in mink_cfg.sites)
        assert not any(env.unwrapped.model.body(mocap) is None for mocap in mink_cfg.mocaps)
        self.siteid = [env.unwrapped.model.site(site).id for site in mink_cfg.sites]
        self.mocapid = [
            env.unwrapped.model.body_mocapid[env.unwrapped.model.body(mocap).id]
            for mocap in mink_cfg.mocaps
        ]
        # Setup mink configuration, tasks, and limits
        self._configuration = mink.Configuration(env.unwrapped.model)
        self._tasks = [
            mink.FrameTask(site, "site", **asdict(mink_cfg.frame_task_cfg))
            for site in mink_cfg.sites
        ] + [mink.PostureTask(env.unwrapped.model, **asdict(mink_cfg.posture_task_cfg))]
        self._frame_tasks = self._tasks[:-1]
        self._limits = [
            mink.ConfigurationLimit(env.unwrapped.model, **asdict(mink_cfg.configuration_limit_cfg))
        ] + aux_limits
        # Setup robot qpos and ctrl ids
        gri_acts = self._get_gripper_actuators(env.unwrapped.model, fltr_acts_kwargs)
        rbt_acts = [i for i in range(env.unwrapped.model.nactuator) if i not in gri_acts]
        gri_idxs = self._get_gripper_indices(env.unwrapped.model, gri_acts)
        ctrl_idxs = [i for i in range(env.unwrapped.model.nu) if i not in gri_idxs]
        jnt_idxs = self._get_qpos_indices(env.unwrapped.model, rbt_acts)
        # Build action function for robot control via mink
        self.mink_action = self._build_mink_action(
            mink_cfg, ctrl_idxs, jnt_idxs, max_tstep, max_rstep
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, InfoType]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # Reset frame tasks and mocap
        for i, task in enumerate(self._frame_tasks):
            sid, mid = self.siteid[i], self.mocapid[i]
            site_xpos, site_xmat = self.data.site_xpos[sid], self.data.site_xmat[sid]
            mocap_pos, mocap_quat = self.data.mocap_pos[mid], self.data.mocap_quat[mid]
            site_quat = np.empty(4)
            mujoco.mju_mat2Quat(site_quat, site_xmat)
            task.set_target(mink.SE3(wxyz_xyz=np.concatenate([site_quat, site_xpos])))
            mocap_pos[:], mocap_quat[:] = site_xpos, site_quat
        # Reset posture task
        self._configuration.update(self.data.qpos)
        self._tasks[-1].set_target_from_configuration(self._configuration)
        return obs, info

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        action = self.unscale_action(action)
        self.mink_action(action)
        self.gripper_action(action)
        return self.action_buffer

    # region Helpers

    @staticmethod
    def _get_qpos_indices(model: mujoco.MjModel, actuator_indices: list[int]) -> list[int]:
        """Get the qpos indices that correspond to the joints of the robot's actuators."""
        qpos_indices = []
        for act in actuator_indices:
            assert int(model.actuator_trntype[act]) in [
                mujoco.mjtTrn.mjTRN_JOINT,
                mujoco.mjtTrn.mjTRN_JOINTINPARENT,
            ]
            jnt_id = model.actuator_trnid[act, 0]
            qpos_indices.append(int(model.jnt_qposadr[jnt_id]))
        return qpos_indices

    def _build_mink_action(
        self,
        mink_cfg: MinkCfg,
        ctrl_indices: list[int],
        qpos_indices: list[int],
        max_tstep: float,
        max_rstep: float,
    ) -> Callable[[WrapperActType], None]:
        dt = self.env.unwrapped.model.opt.timestep if mink_cfg.dt <= 0 else mink_cfg.dt
        pos_thr_sqr, ori_thr_sqr = mink_cfg.pos_thr**2, mink_cfg.ori_thr**2
        nrobot = len(self.siteid)
        limits = np.array([max_tstep, max_rstep]).reshape(1, 2)

        def mink_action(action: WrapperActType) -> None:
            self._configuration.update(self.data.qpos)
            self._update_frame_targets(action, nrobot, limits)
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
            self.action_buffer[ctrl_indices] = self._configuration.q[qpos_indices]

        return mink_action

    def _update_frame_targets(self, action: WrapperActType, nrobot: int, limits: NDArray) -> None:
        """Update frame tasks targets using current targets and action."""
        action = action[: 6 * nrobot].reshape(nrobot, 2, 3)
        # Limit the action norms
        step = np.sqrt(np.sum(action * action, axis=2)) + 1e-12
        action *= np.minimum(1.0, limits / step)[:, :, None]
        # Add pose offsets to target poses in place and update mocaps
        for i, task in enumerate(self._frame_tasks):
            mid = self.mocapid[i]
            mocap_pos, mocap_quat = self.data.mocap_pos[mid], self.data.mocap_quat[mid]
            target = task.transform_target_to_world
            tgt_quat, tgt_pos = target.wxyz_xyz[:4], target.wxyz_xyz[4:]
            tgt_pos += action[i, 0]
            mujoco.mju_quatIntegrate(tgt_quat, action[i, 1], 1.0)
            mocap_pos[:], mocap_quat[:] = tgt_pos, tgt_quat

    def _has_converged(self, pos_threshold_sqr: float, ori_threshold_sqr: float) -> bool:
        """Check whether all frame tasks have converged according to the set error thresholds."""
        for task in self._frame_tasks:
            err = task.compute_error(self._configuration)
            err_sqr = err * err
            if np.sum(err_sqr[:3]) > pos_threshold_sqr or np.sum(err_sqr[3:]) > ori_threshold_sqr:
                return False
        return True


# region MinkCfg


@dataclass(frozen=True, slots=True)
class MinkCfg:
    """Configuration for the mink tasks, IK solver, thresholds, etc."""

    sites: tuple[str, ...] = ("gripper-tcp",)
    """Names of end-effector sites to control. A single site is expected per robot.
    Default value is ("gripper-tcp",)."""

    mocaps: tuple[str, ...] = ("mocap",)
    """Names of mocap bodies for target visualization. A single mocap is expected per robot.
    Default value is ("mocap",)."""

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
    """Position error threshold in meters for early termination. Default value is 1e-3."""

    ori_thr: float = 1.5e-3
    """Orientation error threshold in radians for early termination. Default value is 1.5e-3."""

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
