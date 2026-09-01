from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
from gymnasium import spaces

from ..robots import get_qpos_dim
from ..scenes.builder import build_edge_grasp
from ..utils.mj_utils import get_dof_dim_from_joints
from .mujoco_base import MujocoBaseEnv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..curriculum import CurriculumTerm
    from ..dr import DomainRandomizer
    from .mujoco_base import BoolArray, FloatArray, GoalType, InfoType, ObsType

# region RewardCfg


@dataclass(slots=True)
class EdgeGraspEnvCfg:
    """EdgeGrasp environment configuration for scene, task and reward."""

    @dataclass(frozen=True)
    class SceneCfg:
        """Environment scene/spec configuration."""

        robot: str = "panda"
        """Robot manipulator, see :func:`~..robots.ALL_ROBOTS` for options.
        Default value is panda."""

        gripper: str = "panda_hand"
        """Robot gripper, see :func:`~..robots.ALL_GRIPPERS` for options.
        Default value is panda_hand."""

        object: str = "block"
        """Object to grasp, see :func:`~..objects.ALL_OBJECTS` for options.
        Default is block."""

    scene_cfg: SceneCfg = SceneCfg()

    @dataclass(slots=True)
    class TaskCfg:
        """Environment task configuration."""

        high_goal_prob: float = 0.5
        """Probability of sampling high goals above the table [0, 1]. Default value is 0.5."""

        height_max: float = 0.35
        """Maximum height for high goals above the table. Default value is 0.35."""

        height_min: float = 0.15
        """Minimum height for high goals above the table. Default value is 0.15."""

        goal_tol: float = 0.05
        """Tolerance for object goal position. Default value is 0.05."""

        lift_tol: float = 0.03
        """Object-table distance dense reward term is swapped for object-target distance if
        object is lifted above `lift_tol`. Default value is 0.03."""

        col_tol: float = 30.0
        """Termination due to robot collision is triggered if the max collision force exceeds
        `col_tol`. Default value is 30."""

        fall_tol: float = 0.05
        """Termination due to object falling is triggered if object falls below the table by
        more than `fall_tol`. Defautl value is 0.05."""

        dist_mult: float = 2.5
        """Multiplier for object-table distance before applying tanh() for dense reward.
        Default value is 2.5."""

    task_cfg: TaskCfg = field(default_factory=TaskCfg)

    @dataclass(slots=True)
    class Weights:
        """Weights for the individual dense reward terms."""

        tgt_dist: float = -0.4
        """Weight for the object-target distance reward term. Default value is -0.4."""

        tbl_dist: float = 0.35
        """Weight for the object-table center distance reward term. Default value is 0.35."""

        tcp_dist: float = -0.2
        """Weight for the tcp-object distance reward term. Default value is -0.2."""

        con: float = 0.05
        """Weight for the gripper-object contact reward term. Default value is 0.05."""

        qvel_l2: float = -0.1
        """Weight for the joint velocity L2 norm reward term. Default value is -0.1."""

        qacc_l2: float = -5e-4
        """Weight for the joint acceleration L2 norm reward term. Default value is -5e-4."""

        survive: float = 1.0
        """Weight for survival (not termination) reward term. Default value is 1."""

        fail: float = -3.0
        """Weight for failure/termination reward term. Default value is -3."""

    weights: Weights = field(default_factory=Weights)


class EdgeGraspEnv(MujocoBaseEnv):
    """MuJoCo-based edge grasp environment.

    The environment is setup with an object that cannot be picked up when laying
    completely flush on the table. Therefore, for goals that require lifting the object,
    the robot must first push it towards the edge of the table, then pick it up.

    **Observations**

    A vector of length 54 + 2 * `gripper_dof`, built by concatenating the following
    blocks in order. Positions/linear velocities are world-frame; rotation matrices are
    row-major flattened (3, 3) -> (9,); angular velocities are local to the frame of the
    body they describe (TCP-local for the TCP, object-local for the object and for the
    relative twist).

    - TCP pose: `tcp_pos` (3), `tcp_rmat` (9)
    - Object pose: `obj_pos` (3), `obj_rmat` (9)
    - Object pose relative to TCP: `tcp_obj_pos` (3), `tcp_obj_rmat` (9)
    - Gripper positions: `gri_pos` (`gripper_dof`)
    - TCP twist: `tcp_vel` (3), `tcp_omega` (3)
    - Object twist: `obj_vel` (3), `obj_omega` (3)
    - Object twist relative to TCP: `tcp_obj_vel` (3), `tcp_obj_omega` (3)
    - Gripper velocities: `gri_vel` (`gripper_dof`)

    **Goals**

    A vector of length 4, containing the object's position in the world frame with an
    additional flag indicating whether the object is laying on the table (0) or lifted
    above it (1). Goals that require lifting are sampled with a probability `high_goal_prob`.

    **Reward**

    The reward is a weighted sum of a goal guidance term, two smoothness/effort
    regularization terms, and two sparse survival/failure terms. In the sparse case, the
    guidance term is determined only by the error in the object's position. In the dense
    scenario, the guidance term is itself made up of four separate terms, so that the
    full dense reward is:

    1. Object-to-target distance (minimized). Activated for targets that require lifting
       after their height exceeds `lift_tol` above the table.
    2. Planar object distance from the table center (maximized). Disabled for table
       targets and once the object lifts off the table by more than `lift_tol`.
    3. TCP-to-object distance (minimized). Encourages the gripper to approach the object.
    4. Gripper-object contact (maximized). Encourages establishing and maintaining contact.
    5. L2 norm of robot joint velocities (minimized). Penalizes jerky motion.
    6. L2 norm of robot joint acceleration (minimized). Penalizes jerky motion.
    7. Survival reward (maximized). Encourages avoiding early termination.
    8. Failure penalty. A sparse penalty triggered by heavy robot/gripper collision
       forces exceeding `col_tol` or the object falling off the table by `fall_tol`;
       also terminates the episode.

    Terms 5-8 are independent of the goal and are applied identically regardless of
    `reward_type`; only the guidance term (1-4 when dense, or the single distance
    threshold when sparse) changes between reward types. Joint-based terms (5, 6) apply
    to the robot arm joints only, excluding the gripper.

    Args:
        cfg: Configuration for scene, task and reward, see :class:`EdgeGraspEnvCfg`.
        frame_skip: Number of sim steps per env step. Default value is 20.
        reward_type: Reward type, one of ["dense", "sparse"]. Default value is sparse.
        domain_randomizers: Sequence of domain randomizers to apply during environment
            reset. See :class:`DomainRandomizer` for details. Default value is empty.
        curriculum_terms: Sequence of curriculum terms to call during environment reset.
            See :class:`CurriculumTerm` for details. Default value is empty.
        render_mode: Environment rendering mode. Default value is None.
        renderer_kwargs: Optional kwargs to pass to :class:`mujoco.Renderer` for rendering.
    """

    @dataclass(frozen=True, slots=True)
    class ModelData:
        """MuJoCo model fixed attributes needed by the environment."""

        gri_qpos_adr: int
        obj_qpos_adr: int
        gri_dof_adr: int
        obj_dof_adr: int
        table_body_id: int
        goal_mocap_id: int
        tcp_site_id: int
        tcp_vel_snsr_adr: int
        gri_con_snsr_adr: int
        rbt_con_snsr_adr: int
        table_height: float
        table_extent: float
        obj_spawn_min: NDArray[np.float64]
        obj_spawn_max: NDArray[np.float64]

        def __post_init__(self) -> None:
            for f in fields(self):
                if f.type is not int:
                    continue
                assert getattr(self, f.name) >= 0, f"{f.name} is invalid."

    def __init__(
        self,
        cfg: EdgeGraspEnvCfg = EdgeGraspEnvCfg(),
        frame_skip: int = 20,
        reward_type: Literal["dense", "sparse"] = "sparse",
        domain_randomizers: Sequence[DomainRandomizer] = (),
        curriculum_terms: Sequence[CurriculumTerm] = (),
        render_mode: str | None = None,
        renderer_kwargs: dict[str, Any] = {},
    ):
        spec = build_edge_grasp(**asdict(cfg.scene_cfg))
        super().__init__(
            spec=spec,
            frame_skip=frame_skip,
            reward_type=reward_type,
            domain_randomizers=domain_randomizers,
            curriculum_terms=curriculum_terms,
            render_mode=render_mode,
            renderer_kwargs=renderer_kwargs,
        )
        self.cfg = cfg
        self._mdata = self._setup_model_data(cfg.scene_cfg.robot, cfg.scene_cfg.gripper)
        self._desired_goal = np.zeros(4, dtype=np.float32)
        self._qvel_prev = np.zeros(self._mdata.gri_dof_adr)
        # Setup observation space
        nobs = 12 * 3 + 6 * 3 + get_qpos_dim(cfg.scene_cfg.gripper) * 2
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(nobs,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            }
        )

    # region Goal API

    def compute_reward(
        self, achieved_goal: GoalType, desired_goal: GoalType, info: InfoType
    ) -> np.float32 | FloatArray:
        """Compute task reward for achieved and desired goals. Must support batched inputs."""
        tgt_dist = self._norm(achieved_goal[..., :3] - desired_goal[..., :3])
        if self.reward_type == "sparse":
            guidance_rew = -(tgt_dist > self.cfg.task_cfg.goal_tol).astype(np.float32)
        else:
            tgt_flag = desired_goal[..., 3]
            obj_height_raw = achieved_goal[..., 2] - self._mdata.table_height
            tgt_rew = self._get_target_dist_reward(tgt_dist, tgt_flag, obj_height_raw)
            tbl_rew = self._get_table_dist_reward(achieved_goal[..., :3], tgt_flag, obj_height_raw)
            guidance_rew = (
                self.cfg.weights.tgt_dist * tgt_rew
                + self.cfg.weights.tbl_dist * tbl_rew
                + info["state_rew"]
            )
        return guidance_rew + info["reg_rew"]

    def compute_terminated(
        self, achieved_goal: GoalType, desired_goal: GoalType, info: InfoType
    ) -> np.bool_ | BoolArray:
        """Compute terminated signal for acheived and desired goals. Must support batched inputs."""
        return info["terminated"]  # termination is goal independent

    # region Env API

    def _reset_data(self) -> None:
        """Apply any additional resets to self.data after environment reset.

        Called after `mujoco.mj_resetData` and domain randomization, before `mujoco.mj_forward`.
        """
        self._qvel_prev[:] = 0.0
        if not self.domain_randomizers:
            return  # initial state guaranteed to be valid
        spawn_min, spawn_max = self._get_world_spawn_range()
        # Ensure object spawns on the table and with a unit quaternion
        # Note: assumes object is flat on the table (i.e. yaw randomization only)
        adr = self._mdata.obj_qpos_adr
        self.data.qpos[adr : adr + 3] = self.data.qpos[adr : adr + 3].clip(spawn_min, spawn_max)
        mujoco.mju_normalize4(self.data.qpos[adr + 3 : adr + 7])

    def _sample_goal(self) -> None:
        """Sample a new goal for the environment."""
        spawn_min, spawn_max = self._get_world_spawn_range()
        # Sample XY and height uniformly
        self._desired_goal[:3] = self.rng.uniform(spawn_min, spawn_max)
        if self.rng.random() < self.cfg.task_cfg.high_goal_prob:
            self._desired_goal[2] += self.rng.uniform(
                self.cfg.task_cfg.height_min, self.cfg.task_cfg.height_max
            )
            self._desired_goal[3] = 1.0
        else:
            self._desired_goal[3] = 0.0
        self.data.mocap_pos[self._mdata.goal_mocap_id] = self._desired_goal[:3]

    def _get_obs(self) -> ObsType:
        """Get the latest observations."""
        tcp_pos, tcp_rmat, obj_pos, obj_rmat, tcp_obj_pos, tcp_obj_rmat = self._get_poses()
        tcp_vel, tcp_omega, obj_vel, obj_omega, tcp_obj_vel, tcp_obj_omega = self._get_twists(
            tcp_rmat, obj_rmat
        )
        gri_pos, gri_vel = self._get_gripper_state()
        observation = np.concatenate(
            [
                tcp_pos,
                tcp_rmat.ravel(),
                obj_pos,
                obj_rmat.ravel(),
                tcp_obj_pos,
                tcp_obj_rmat.ravel(),
                gri_pos,
                tcp_vel,
                tcp_omega,
                obj_vel,
                obj_omega,
                tcp_obj_vel,
                tcp_obj_omega,
                gri_vel,
            ],
            dtype=np.float32,
        )

        obj_height_raw = obj_pos[2:] - self._mdata.table_height
        achieved_goal = np.concatenate(
            [obj_pos, obj_height_raw > self.cfg.task_cfg.lift_tol], dtype=np.float32
        )
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self._desired_goal.copy(),
        }

    def _get_info(self, obs: ObsType) -> InfoType:
        """Get the latest info dict.

        Precomputes and caches reward terms + termination that are goal-independent.
        """
        obj_height_raw = obs["achieved_goal"][2] - self._mdata.table_height
        terminated = self._get_terminated(obj_height_raw)
        is_success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info = {
            "terminated": terminated,
            "is_success": is_success,
            "reg_rew": self._get_reg_reward(terminated),
        }
        if self.reward_type == "dense":
            info["state_rew"] = self._get_dense_state_reward(obs["observation"])
        return info

    # region Helpers

    def _setup_model_data(self, robot: str, gripper: str) -> EdgeGraspEnv.ModelData:
        assert self.model.sensor("gripper_object_contact").dim == 1
        assert self.model.sensor("robot_contact").dim == 3
        rbt_qpos_dim = get_qpos_dim(robot)
        gri_qpos_dim = get_qpos_dim(gripper)
        rbt_dof_dim = get_dof_dim_from_joints(self.model, 0, rbt_qpos_dim)
        gri_dof_dim = get_dof_dim_from_joints(self.model, rbt_qpos_dim, gri_qpos_dim)
        table_geom = self.model.geom("table-tabletop")
        table_height = float(table_geom.pos[2] + table_geom.size[2])
        obj_geom_size = self.model.geom("object-geom").size
        obj_geom_extent = max(obj_geom_size[:-1])
        spawn_range_xy = [s - obj_geom_extent for s in table_geom.size[:2]]
        return EdgeGraspEnv.ModelData(
            gri_qpos_adr=rbt_qpos_dim,
            obj_qpos_adr=rbt_qpos_dim + gri_qpos_dim,
            gri_dof_adr=rbt_dof_dim,
            obj_dof_adr=rbt_dof_dim + gri_dof_dim,
            table_body_id=self.model.body("table-table").id,
            goal_mocap_id=self.model.body("mocap_goal").mocapid,
            tcp_site_id=self.model.site("gripper-tcp").id,
            tcp_vel_snsr_adr=self.model.sensor("tcp_linvel").adr[0],
            gri_con_snsr_adr=self.model.sensor("gripper_object_contact").adr[0],
            rbt_con_snsr_adr=self.model.sensor("robot_contact").adr[0],
            table_height=table_height,
            table_extent=float(max(table_geom.size[:2])),
            obj_spawn_min=np.array(
                [-spawn_range_xy[0], -spawn_range_xy[1], table_height + obj_geom_size[-1] + 1e-3]
            ),
            obj_spawn_max=np.array(
                [spawn_range_xy[0], spawn_range_xy[1], table_height + obj_geom_size[-1] + 1e-3]
            ),
        )

    def _get_world_spawn_range(self) -> tuple[NDArray, NDArray]:
        """Get the object spawn range in the world frame"""
        table_pos = self.model.body_pos[self._mdata.table_body_id]
        table_quat = self.model.body_quat[self._mdata.table_body_id]
        spawn_min, spawn_max = np.split(np.empty(6, dtype=np.float64), 2)
        mujoco.mju_rotVecQuat(spawn_min, self._mdata.obj_spawn_min, table_quat)
        mujoco.mju_rotVecQuat(spawn_max, self._mdata.obj_spawn_max, table_quat)
        return table_pos + spawn_min, table_pos + spawn_max

    # region Obs Helpers

    def _get_poses(self) -> tuple[NDArray, ...]:
        """Get the TCP, object, object-TCP poses as position vectors and rotation matrices."""
        # TCP pose relative to the world frame
        tcp_pos = self.data.site_xpos[self._mdata.tcp_site_id]
        tcp_rmat = self.data.site_xmat[self._mdata.tcp_site_id].reshape(3, 3)
        # Object pose relative to the world frame
        obj_pos = self.data.qpos[self._mdata.obj_qpos_adr : self._mdata.obj_qpos_adr + 3]
        obj_quat = self.data.qpos[self._mdata.obj_qpos_adr + 3 : self._mdata.obj_qpos_adr + 7]
        obj_rmat = np.empty(9)
        mujoco.mju_quat2Mat(obj_rmat, obj_quat)
        obj_rmat = obj_rmat.reshape(3, 3)
        # Object pose relative to the TCP frame
        tcp_obj_pos = obj_pos - tcp_pos
        tcp_obj_rmat = tcp_rmat.T @ obj_rmat
        return tcp_pos, tcp_rmat, obj_pos, obj_rmat, tcp_obj_pos, tcp_obj_rmat

    def _get_twists(self, tcp_rmat: NDArray, obj_rmat: NDArray) -> tuple[NDArray, ...]:
        """Get TCP, object, and object-TCP twists as linear and angular velocity vectors.

        Linear velocities are world-frame. Angular velocities are local to the frame of the body
        they describe: TCP-local for the TCP, object-local for the object and for the relative twist.
        """
        # TCP twist
        adr = self._mdata.tcp_vel_snsr_adr
        tcp_vel = self.data.sensordata[adr : adr + 3]
        tcp_omega_world = self.data.sensordata[adr + 3 : adr + 6]
        tcp_omega = tcp_rmat.T @ tcp_omega_world
        # Object twist
        adr = self._mdata.obj_dof_adr
        obj_vel = self.data.qvel[adr : adr + 3]
        obj_omega = self.data.qvel[adr + 3 : adr + 6]
        # Object relative twist
        tcp_obj_vel = obj_vel - tcp_vel
        tcp_obj_omega = obj_omega - obj_rmat.T @ tcp_omega_world
        return tcp_vel, tcp_omega, obj_vel, obj_omega, tcp_obj_vel, tcp_obj_omega

    def _get_gripper_state(self) -> tuple[NDArray, NDArray]:
        """Get gripper finger positions and velocities."""
        gri_pos = self.data.qpos[self._mdata.gri_qpos_adr : self._mdata.obj_qpos_adr]
        gri_vel = self.data.qvel[self._mdata.gri_dof_adr : self._mdata.obj_dof_adr]
        return gri_pos, gri_vel

    # region Rew Helpers

    @staticmethod
    def _norm(x: FloatArray) -> np.float64 | FloatArray:
        """L2 norm along the last axis with np.sqrt(np.sum(x * x))"""
        return np.sqrt(np.sum(x * x, axis=-1))

    def _get_target_dist_reward(
        self,
        tgt_dist: FloatArray,
        tgt_flag: np.float32 | FloatArray,
        obj_height_raw: np.float32 | FloatArray,
    ) -> np.float32 | FloatArray:
        """Get the target-object distance reward term (guidance term for sparse rewards)."""
        tgt_dist_rew = np.tanh(tgt_dist)
        mask = np.logical_and(tgt_flag > 0, obj_height_raw < self.cfg.task_cfg.lift_tol)
        return np.where(mask, 1.0, tgt_dist_rew)

    def _get_table_dist_reward(
        self,
        obj_pos: FloatArray,
        tgt_flag: np.float32 | FloatArray,
        obj_height_raw: np.float32 | FloatArray,
    ) -> np.float32 | FloatArray:
        """Get the object's planar distance from table center reward term."""
        table_pos = self.data.xpos[self._mdata.table_body_id]
        obj_dist = self._norm(obj_pos[..., :2] - table_pos[:2]) / self._mdata.table_extent
        tbl_dist_rew = np.tanh(self.cfg.task_cfg.dist_mult * obj_dist) - 1
        mask = np.logical_or(tgt_flag == 0, obj_height_raw > self.cfg.task_cfg.lift_tol)
        return np.where(mask, 0.0, tbl_dist_rew)

    def _get_dense_state_reward(self, observation: FloatArray) -> float:
        """Get state, goal-independent dense reward terms."""
        tcp_obj_pos = observation[24:27]
        tcp_dist_rew = float(np.tanh(self._norm(tcp_obj_pos)))
        con_rew = float(self.data.sensordata[self._mdata.gri_con_snsr_adr] > 0) - 1
        return self.cfg.weights.tcp_dist * tcp_dist_rew + self.cfg.weights.con * con_rew

    def _get_reg_reward(self, terminated: bool) -> float:
        """Get the regularization and failure reward terms."""
        qvel = self.data.qvel[: self._mdata.gri_dof_adr]
        qacc = (qvel - self._qvel_prev) / (self.model.opt.timestep * self.frame_skip)
        self._qvel_prev[:] = qvel
        qvel_l2 = float(self._norm(qvel))
        qacc_l2 = float(self._norm(qacc))
        return (
            self.cfg.weights.qvel_l2 * qvel_l2
            + self.cfg.weights.qacc_l2 * qacc_l2
            + self.cfg.weights.survive * float(not terminated)
            + self.cfg.weights.fail * float(terminated)
        )

    def _get_terminated(self, obj_height_raw: float) -> bool:
        """Get the terminated signal due to heavy robot/gripper collisions or the object falling."""
        adr = self._mdata.rbt_con_snsr_adr
        con_frc = self.data.sensordata[adr : adr + 3]
        con_frc_term = self._norm(con_frc) > self.cfg.task_cfg.col_tol
        obj_fall_term = obj_height_raw < -self.cfg.task_cfg.fall_tol
        return bool(con_frc_term or obj_fall_term)

    def _is_success(self, achieved_goal: GoalType, desired_goal: GoalType) -> float:
        """Get task success status."""
        return float(self._norm(achieved_goal - desired_goal) < self.cfg.task_cfg.goal_tol)
