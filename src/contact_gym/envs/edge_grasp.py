from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from gymnasium import spaces

from ..scenes.builder import build_edge_grasp
from .base import MujocoBaseEnv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .base import ActType, InfoType, ObsType


class MujocoEdgeGraspEnv(MujocoBaseEnv):
    """MuJoCo-based edge grasp environment.

    The environment is setup with an object that cannot be picked up when laying completely flush on
    the table. Therefore, to grasp and lift it, the robot must first push it towards the edge of the
    table, then pick it up.

    **Observation space**

    A single float32 vector of length `54 + 2 * GRIPPER_DOFS`, built by concatenating the following
    blocks in order. Positions/linear velocities are world-frame; rotation matrices are row-major
    flattened (3, 3) -> (9,); angular velocities are local to the frame of the body they describe
    (TCP-local for the TCP, object-local for the object and for the relative twist).

    - TCP pose: `tcp_pos` (3), `tcp_rmat` (9)
    - Object pose: `obj_pos` (3), `obj_rmat` (9)
    - Object pose relative to TCP: `tcp_obj_pos` (3), `tcp_obj_rmat` (9)
    - Gripper positions: `gri_pos` (`GRIPPER_DOFS`)
    - TCP twist: `tcp_vel` (3), `tcp_omega` (3)
    - Object twist: `obj_vel` (3), `obj_omega` (3)
    - Object twist relative to TCP: `tcp_obj_vel` (3), `tcp_obj_omega` (3)
    - Gripper velocities: `gri_vel` (`GRIPPER_DOFS`)

    **Reward**

    The reward is a weighted sum of four dense guidance terms, two smoothness/effort
    regularization terms, and one sparse failure penalty:

    1. Planar object distance from the tabletop center (maximized while the object is still
       flush on the table). Encourages pushing the object towards an edge so it can later be
       grasped. Disabled once the object lifts off the table more than `lift_tol`.
    2. TCP-to-object distance (minimized). Encourages the gripper to approach the object.
    3. Gripper-object contact (maximized). Encourages establishing and maintaining contact.
    4. Object height above the tabletop (maximized, up to `tgt_height`). Encourages lifting
       the object once it is graspable.
    5. Squared L2 norm of robot joint velocities (minimized). Penalizes jerky motion.
    6. Squared L2 norm of robot joint torques (minimized). Penalizes excessive actuation effort.
    7. Failure penalty. A sparse penalty triggered by heavy robot/gripper collision forces
       exceeding `col_tol` or the object falling off the table by `fall_tol`; also terminates the
       episode.

    Joint-based terms (5, 6) apply to the robot arm joints only, excluding the gripper.
    """

    GRIPPER_DOFS = 2
    GRIPPER_CONTYPE, OBJECT_CONTYPE = 2, 8

    @dataclass(frozen=True)
    class ModelData:
        """MuJoCo model fixed attributes needed by the environment."""

        tcp_site_id: int
        con_snsr_adr: int
        obj_qpos_adr: int
        obj_qvel_adr: int
        gri_qpos_addr: int
        gri_qvel_addr: int
        tabletop_site_id: int
        tabletop_height: float
        tabletop_extent: float

        def __post_init__(self) -> None:
            for f in fields(self):
                if f.type is float:
                    continue
                assert getattr(self, f.name) >= 0, f"{f.name} is invalid."

    @dataclass
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
        object: str = "block",
        frame_skip: int = 10,
        render_mode: str | None = None,
        rew_cfg: EdgeGraspRewardCfg | None = None,
    ):
        spec = build_edge_grasp(robot=robot, object=object)
        super().__init__(spec, frame_skip=frame_skip, render_mode=render_mode)
        self._mdata = self._setup_model_data()
        self._rcfg = EdgeGraspRewardCfg() if rew_cfg is None else rew_cfg
        self._rterms = MujocoEdgeGraspEnv.RewardTerms()
        # Setup observation space
        nobs = 12 * 3 + 6 * 3 + self.GRIPPER_DOFS * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(nobs,), dtype=np.float32
        )

    # region Env API

    def _reset_data(self) -> None:
        """Apply any additional resets to self.data after `mujoco.mj_resetData`.

        Called before `mujoco.mj_forward`.
        """
        # TODO: Update once domain randomization is implemented
        self._rterms = MujocoEdgeGraspEnv.RewardTerms()

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
        return self._rterms.__dict__.copy()

    def _compute_reward(self, obs: ObsType, _: ActType) -> tuple[float, bool]:
        """Compute the reward and termination signal.

        Args:
            obs: Latest observation.
            _: Latest action (unused).

        Returns:
            tuple containing the reward and termination signals.
        """
        obj_pos, tcp_obj_pos = obs[12:15], obs[24:27]
        tabletop_pos = self.data.site_xpos[self._mdata.tabletop_site_id]
        obj_height_raw = obj_pos[2] - tabletop_pos[2]

        self._rterms.obj_dist = self._get_planar_dist_reward(obj_pos, tabletop_pos, obj_height_raw)
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

    def _setup_model_data(self) -> MujocoEdgeGraspEnv.ModelData:
        obj_jnt_id = self.model.joint("object-joint").id
        assert self.model.sensor("impact").dim == 3
        finger_jnt_id = self.model.joint("hand-finger_joint1").id
        geom = self.model.geom("table-tabletop")
        tabletop_pos, tabletop_size = geom.pos, geom.size
        return MujocoEdgeGraspEnv.ModelData(
            tcp_site_id=self.model.site("hand-tcp").id,
            con_snsr_adr=self.model.sensor("impact").adr,
            obj_qpos_adr=self.model.jnt_qposadr[obj_jnt_id],
            obj_qvel_adr=self.model.jnt_dofadr[obj_jnt_id],
            gri_qpos_adr=self.model.jnt_qposadr[finger_jnt_id],
            gri_qvel_adr=self.model.jnt_dofadr[finger_jnt_id],
            tabletop_site_id=self.model.site("table-tabletop_center").id,
            tabletop_height=tabletop_pos[2] + tabletop_size[2],
            tabletop_extent=max(tabletop_size[:2]),
        )

    # region Obs Helpers

    def _get_poses(self) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
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

    def _get_twists(
        self, tcp_rmat: NDArray, obj_rmat: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Get TCP, object, and object-TCP twists as linear and angular velocity vectors.

        Linear velocities are world-frame. Angular velocities are local to the frame of the body
        they describe: TCP-local for the TCP, object-local for the object and for the relative twist.
        """
        # TCP twist
        tcp_twist = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self._mdata.tcp_site_id, tcp_twist, 0
        )
        tcp_omega_world, tcp_vel = tcp_twist[:3], tcp_twist[3:]
        tcp_omega = tcp_rmat.T @ tcp_omega_world
        # Object twist
        adr = self._mdata.obj_qvel_adr
        obj_vel = self.data.qvel[adr : adr + 3]
        obj_omega = self.data.qvel[adr + 3 : adr + 6]
        # Object relative twist
        tcp_obj_vel = obj_vel - tcp_vel
        tcp_obj_omega = obj_omega - obj_rmat.T @ tcp_omega_world
        return tcp_vel, tcp_omega, obj_vel, obj_omega, tcp_obj_vel, tcp_obj_omega

    def _get_gripper_state(self) -> tuple[NDArray, NDArray]:
        """Get gripper finger positions and velocities."""
        qpos_adr, qvel_adr = self._mdata.gri_qpos_addr, self._mdata.gri_qvel_addr
        gri_pos = self.data.qpos[qpos_adr : qpos_adr + self.GRIPPER_DOFS]
        gri_vel = self.data.qvel[qvel_adr : qvel_adr + self.GRIPPER_DOFS]
        return gri_pos, gri_vel

    # region Rew Helpers

    def _get_planar_dist_reward(
        self, obj_pos: NDArray, tabletop_pos: NDArray, obj_height_raw: float
    ) -> float:
        """Get the object's planar distance from table center reward term."""
        if obj_height_raw >= self._rcfg.lift_tol:
            return 0.0
        obj_dist = np.linalg.vector_norm(obj_pos[:2] - tabletop_pos[:2])
        obj_dist /= self._mdata.tabletop_extent
        return float(np.tanh(self._rcfg.dist_mult * obj_dist) - 1)

    def _get_tcp_dist_reward(self, tcp_obj_pos: NDArray) -> float:
        """Get the TCP-object distance reward term."""
        tcp_dist = np.linalg.vector_norm(tcp_obj_pos)
        return float(np.tanh(tcp_dist))

    def _get_height_reward(self, obj_height_raw: float) -> float:
        """Get the object height-off-tabletop reward term."""
        height_norm = np.clip(obj_height_raw / self._rcfg.tgt_height, 0, 1)
        return float(np.tanh(self._rcfg.height_mult * height_norm) - 1)

    def _get_contact_reward(self) -> float:
        """Get the gripper-object contact reward term."""
        g1_type = self.model.geom_contype[self.data.contact.geom1[: self.data.ncon]]
        g2_type = self.model.geom_contype[self.data.contact.geom2[: self.data.ncon]]
        gripper_object = ((g1_type == self.GRIPPER_CONTYPE) & (g2_type == self.OBJECT_CONTYPE)) | (
            (g1_type == self.OBJECT_CONTYPE) & (g2_type == self.GRIPPER_CONTYPE)
        )
        return float(np.any(gripper_object)) - 1

    def _get_joint_reward(self) -> tuple[float, float]:
        """Get the robot joint velocity and force squared L2 reward terms."""
        qvel = self.data.qvel[: self._mdata.gri_qvel_addr]
        qfrc = self.data.qfrc_actuator[: self._mdata.gri_qvel_addr]
        qvel_l2_term = float(np.square(qvel).sum())
        qfrc_l2_term = float(np.square(qfrc).sum())
        return qvel_l2_term, qfrc_l2_term

    def _get_terminated(self, obj_height_raw: float) -> bool:
        """Get the terminated signal due to heavy robot/gripper collisions or object falling."""
        con_frc = self.data.sensordata[self._mdata.con_snsr_adr : self._mdata.con_snsr_adr + 3]
        con_frc_norm = np.linalg.vector_norm(con_frc)
        con_frc_term = con_frc_norm > self._rcfg.col_tol
        obj_fall_term = obj_height_raw < -self._rcfg.fall_tol
        return bool(con_frc_term or obj_fall_term)


# region RewardCfg


@dataclass(frozen=True)
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
    """Failure due to object falling is triggered if object falls below the tabletop by more
    than `fall_tol`. Defautl value is 0.02."""

    dist_mult: float = 2.5
    """Multiplier for object distance before applying tanh() in reward term. Default value is 2.5."""

    height_mult: float = 2.0
    """Multiplier for object height before applying tanh() for reward term. Default value is 2."""

    @dataclass(frozen=True)
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
