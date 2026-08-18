from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from gymnasium import spaces

from ..robots import get_qpos_dim
from ..scenes.builder import build_edge_grasp
from ..utils.mj_utils import get_dof_dim_from_joints
from .mujoco_base import MujocoBaseEnv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .mujoco_base import ActType, InfoType, ObsType

    TupleNDArray6 = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]


class EdgeGraspEnv(MujocoBaseEnv):
    """MuJoCo-based edge grasp environment.

    The environment is setup with an object that cannot be picked up when laying
    completely flush on the table. Therefore, to grasp and lift it, the robot must first
    push it towards the edge of the table, then pick it up.

    **Observation space**

    A single vector of length `54 + 2 * `gripper_dof`, built by concatenating the
    following blocks in order. Positions/linear velocities are world-frame; rotation
    matrices are row-major flattened (3, 3) -> (9,); angular velocities are local to the
    frame of the body they describe (TCP-local for the TCP, object-local for the object
    and for the relative twist).

    - TCP pose: `tcp_pos` (3), `tcp_rmat` (9)
    - Object pose: `obj_pos` (3), `obj_rmat` (9)
    - Object pose relative to TCP: `tcp_obj_pos` (3), `tcp_obj_rmat` (9)
    - Gripper positions: `gri_pos` (`gripper_dof`)
    - TCP twist: `tcp_vel` (3), `tcp_omega` (3)
    - Object twist: `obj_vel` (3), `obj_omega` (3)
    - Object twist relative to TCP: `tcp_obj_vel` (3), `tcp_obj_omega` (3)
    - Gripper velocities: `gri_vel` (`gripper_dof`)

    **Reward**

    The reward is a weighted sum of four dense guidance terms, two smoothness/effort
    regularization terms, and one sparse failure penalty:

    1. Planar object distance from the table center (maximized while the object is still
       flush on the table). Encourages pushing the object towards an edge so it can later
       be grasped. Disabled once the object lifts off the table more than `lift_tol`.
    2. TCP-to-object distance (minimized). Encourages the gripper to approach the object.
    3. Gripper-object contact (maximized). Encourages establishing and maintaining contact.
    4. Object height above the table (maximized, up to `tgt_height`). Encourages lifting
       the object once it is graspable.
    5. Squared L2 norm of robot joint velocities (minimized). Penalizes jerky motion.
    6. Squared L2 norm of robot joint torques (minimized). Penalizes excessive actuation effort.
    7. Failure penalty. A sparse penalty triggered by heavy robot/gripper collision
       forces exceeding `col_tol` or the object falling off the table by `fall_tol`;
       also terminates the episode.

    Joint-based terms (5, 6) apply to the robot arm joints only, excluding the gripper.

    Args:
        robot: Choice of robot manipulator, see :func:`~..robots.ALL_ROBOTS` for options.
            Default is panda.
        gripper: Choice of gripper, see :func:`~..robots.ALL_GRIPPERS` for options.
            Default is panda_hand.
        object: Choice of object to grasp, see :func:`~..objects.ALL_OBJECTS` for options.
            Default is block.
        frame_skip: Number of sim steps per env step. Default value is 10.
        render_mode: Environment rendering mode. Default value is None.
        renderer_kwargs: Optional kwargs to pass to :class:`mujoco.Renderer` for rendering.
        rew_cfg: Reward function configuration. Determines thresholds, multipliers and reward
            weights. If None, default values are used. Default value is None.
        debug_info: If True, the info dict contains the raw values of each reward term. Otherwise,
            an empty dict is returned for info. Default value is False.
    """

    @dataclass(frozen=True, slots=True)
    class ModelData:
        """MuJoCo model fixed attributes needed by the environment."""

        gri_qpos_adr: int
        obj_qpos_adr: int
        gri_dof_adr: int
        obj_dof_adr: int
        tcp_site_id: int
        table_site_id: int
        tcp_vel_snsr_adr: int
        gri_con_snsr_adr: int
        rbt_con_snsr_adr: int
        table_height: float
        table_extent: float

        def __post_init__(self) -> None:
            for f in fields(self):
                if f.type is float:
                    continue
                assert getattr(self, f.name) >= 0, f"{f.name} is invalid."

    @dataclass(slots=True)
    class RewardTerms:
        """Individual reward terms computed by the environment."""

        obj_dist: float = 0.0
        tcp_dist: float = 0.0
        height: float = 0.0
        con: float = 0.0
        qvel_l2: float = 0.0
        qfrc_l2: float = 0.0
        fail: float = 0.0

    def __init__(
        self,
        robot: str = "panda",
        gripper: str = "panda_hand",
        object: str = "block",
        frame_skip: int = 10,
        render_mode: str | None = None,
        renderer_kwargs: dict[str, Any] = {},
        rew_cfg: EdgeGraspRewardCfg | None = None,
        debug_info: bool = False,
    ):
        spec = build_edge_grasp(robot=robot, gripper=gripper, object=object)
        super().__init__(spec, frame_skip, render_mode, renderer_kwargs)
        self._mdata = self._setup_model_data(robot, gripper)
        self._rcfg = EdgeGraspRewardCfg() if rew_cfg is None else rew_cfg
        self._rterms = EdgeGraspEnv.RewardTerms()
        # Setup observation space
        nobs = 12 * 3 + 6 * 3 + get_qpos_dim(gripper) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(nobs,), dtype=np.float32
        )
        # Setup info function
        self._get_info_fn = self._get_debug_info if debug_info else (lambda: {})

    # region Env API

    def _reset_data(self) -> None:
        """Apply any additional resets to self.data after `mujoco.mj_resetData`.

        Called before `mujoco.mj_forward`.
        """
        # TODO: Update once domain randomization is implemented
        self._rterms = EdgeGraspEnv.RewardTerms()

    def _get_obs(self) -> ObsType:
        """Get the latest observations."""
        tcp_pos, tcp_rmat, obj_pos, obj_rmat, tcp_obj_pos, tcp_obj_rmat = self._get_poses()
        tcp_vel, tcp_omega, obj_vel, obj_omega, tcp_obj_vel, tcp_obj_omega = self._get_twists(
            tcp_rmat, obj_rmat
        )
        gri_pos, gri_vel = self._get_gripper_state()
        return np.concatenate(
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

    def _get_info(self) -> InfoType:
        """Get the latest info dict."""
        return self._get_info_fn()

    def _compute_reward(self, obs: ObsType, _: ActType) -> tuple[float, bool]:
        """Compute the reward and termination signal.

        Args:
            obs: Latest observation.
            _: Latest action (unused).

        Returns:
            tuple containing the reward and termination signals.
        """
        obj_pos, tcp_obj_pos = obs[12:15], obs[24:27]
        table_pos = self.data.site_xpos[self._mdata.table_site_id]
        obj_height_raw = obj_pos[2] - table_pos[2]

        self._rterms.obj_dist = self._get_planar_dist_reward(obj_pos, table_pos, obj_height_raw)
        self._rterms.tcp_dist = self._get_tcp_dist_reward(tcp_obj_pos)
        self._rterms.height = self._get_height_reward(obj_height_raw)
        self._rterms.con = self._get_contact_reward()
        self._rterms.qvel_l2, self._rterms.qfrc_l2 = self._get_joint_reward()

        terminated = self._get_terminated(obj_height_raw)
        self._rterms.fail = float(terminated)

        reward = (
            self._rcfg.weights.obj_dist * self._rterms.obj_dist
            + self._rcfg.weights.tcp_dist * self._rterms.tcp_dist
            + self._rcfg.weights.height * self._rterms.height
            + self._rcfg.weights.con * self._rterms.con
            + self._rcfg.weights.qvel_l2 * self._rterms.qvel_l2
            + self._rcfg.weights.qfrc_l2 * self._rterms.qfrc_l2
            + self._rcfg.weights.fail * self._rterms.fail
        )
        return reward, terminated

    # region Helpers

    def _setup_model_data(self, robot: str, gripper: str) -> EdgeGraspEnv.ModelData:
        assert self.model.sensor("gripper_object_contact").dim == 1
        assert self.model.sensor("robot_contact").dim == 3
        rbt_qpos_dim = get_qpos_dim(robot)
        gri_qpos_dim = get_qpos_dim(gripper)
        rbt_dof_dim = get_dof_dim_from_joints(self.model, 0, rbt_qpos_dim)
        gri_dof_dim = get_dof_dim_from_joints(self.model, rbt_qpos_dim, gri_qpos_dim)
        table_geom = self.model.geom("table-tabletop")
        return EdgeGraspEnv.ModelData(
            gri_qpos_adr=rbt_qpos_dim,
            obj_qpos_adr=rbt_qpos_dim + gri_qpos_dim,
            gri_dof_adr=rbt_dof_dim,
            obj_dof_adr=rbt_dof_dim + gri_dof_dim,
            tcp_site_id=self.model.site("gripper-tcp").id,
            table_site_id=self.model.site("table-topcenter").id,
            tcp_vel_snsr_adr=self.model.sensor("tcp_linvel").adr[0],
            gri_con_snsr_adr=self.model.sensor("gripper_object_contact").adr[0],
            rbt_con_snsr_adr=self.model.sensor("robot_contact").adr[0],
            table_height=float(table_geom.pos[2] + table_geom.size[2]),
            table_extent=float(max(table_geom.size[:2])),
        )

    def _get_debug_info(self) -> InfoType:
        """Return the dict of individual reward terms to values."""
        return {f.name: getattr(self._rterms, f.name) for f in fields(self._rterms)}

    # region Obs Helpers

    def _get_poses(self) -> TupleNDArray6:
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

    def _get_twists(self, tcp_rmat: NDArray, obj_rmat: NDArray) -> TupleNDArray6:
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

    def _get_planar_dist_reward(
        self, obj_pos: NDArray, table_pos: NDArray, obj_height_raw: float
    ) -> float:
        """Get the object's planar distance from table center reward term."""
        if obj_height_raw > self._rcfg.lift_tol:
            return 0.0
        tt_obj_pos_xy = obj_pos[:2] - table_pos[:2]
        obj_dist = np.sqrt(np.sum(tt_obj_pos_xy * tt_obj_pos_xy))
        obj_dist /= self._mdata.table_extent
        return float(np.tanh(self._rcfg.dist_mult * obj_dist) - 1)

    def _get_tcp_dist_reward(self, tcp_obj_pos: NDArray) -> float:
        """Get the TCP-object distance reward term."""
        tcp_dist = np.sqrt(np.sum(tcp_obj_pos * tcp_obj_pos))
        return float(np.tanh(tcp_dist))

    def _get_height_reward(self, obj_height_raw: float) -> float:
        """Get the object height-off-table reward term."""
        height_norm = np.clip(obj_height_raw / self._rcfg.tgt_height, 0, 1)
        return float(np.tanh(self._rcfg.height_mult * height_norm) - 1)

    def _get_contact_reward(self) -> float:
        """Get the gripper-object contact reward term."""
        con_fnd = self.data.sensordata[self._mdata.gri_con_snsr_adr]
        return float(con_fnd > 0) - 1

    def _get_joint_reward(self) -> tuple[float, float]:
        """Get the robot joint velocity and force squared L2 reward terms."""
        qvel = self.data.qvel[: self._mdata.gri_dof_adr]
        qfrc = self.data.qfrc_actuator[: self._mdata.gri_dof_adr]
        qvel_l2_term = float((qvel * qvel).sum())
        qfrc_l2_term = float((qfrc * qfrc).sum())
        return qvel_l2_term, qfrc_l2_term

    def _get_terminated(self, obj_height_raw: float) -> bool:
        """Get the terminated signal due to heavy robot/gripper collisions or object falling."""
        adr = self._mdata.rbt_con_snsr_adr
        con_frc = self.data.sensordata[adr : adr + 3]
        con_frc_norm = np.sqrt(np.sum(con_frc * con_frc))
        con_frc_term = con_frc_norm > self._rcfg.col_tol
        obj_fall_term = obj_height_raw < -self._rcfg.fall_tol
        return bool(con_frc_term or obj_fall_term)


# region RewardCfg


@dataclass(frozen=True, slots=True)
class EdgeGraspRewardCfg:
    """Reward function configuration for the EdgeGrasp environment."""

    tgt_height: float = 0.5
    """Target height for lifting object above the table. Default value is 0.5."""

    lift_tol: float = 0.025
    """Object-table distance reward is disabled if object is lifted above `lift_tol` to allow its
    free movement. Default value is 0.025."""

    col_tol: float = 10.0
    """Failure due to robot collision is triggered if the max collision force exceeds `col_tol`.
    Default value is 10."""

    fall_tol: float = 0.02
    """Failure due to object falling is triggered if object falls below the table by more
    than `fall_tol`. Defautl value is 0.02."""

    dist_mult: float = 2.5
    """Multiplier for object distance before applying tanh() in reward term. Default value is 2.5."""

    height_mult: float = 2.0
    """Multiplier for object height before applying tanh() for reward term. Default value is 2."""

    @dataclass(frozen=True, slots=True)
    class Weights:
        """Weights for the individual reward terms."""

        obj_dist: float = 0.2
        """Weight for the object-table center distance reward term. Default value is 0.2."""

        tcp_dist: float = -0.15
        """Weight for the tcp-object distance reward term. Default value is -0.15."""

        height: float = 0.55
        """Weight for the object height reward term. Default value is 0.55."""

        con: float = 0.1
        """Weight for the gripper-object contact reward term. Default value is 0.1."""

        qvel_l2: float = -5e-3
        """Weight for the joint velocity squared L2 reward term. Default value is 5e-3."""

        qfrc_l2: float = -5e-3
        """Weight for the joint force squared L2 reward term. Default value is 5e-3."""

        fail: float = -3.0
        """Weight for failure/termination reward term. Default value is -3."""

    weights: Weights = Weights()
