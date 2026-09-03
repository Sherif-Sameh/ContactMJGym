from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers.utils import rescale_box

from ..envs.mujoco_base import MujocoBaseEnv
from ..utils.mj_utils import filter_actuators, mjtjoint_to_dof_dim, mjtjoint_to_qpos_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..envs.mujoco_base import ActType, FloatArray

    WrapperActType: TypeAlias = ActType


class ParamSpace(StrEnum):
    """Control parameters space. Options include 'joint' and 'task' only."""

    JOINT = "joint"
    TASK = "task"


class ParamType(StrEnum):
    """Control parameter type. Options include 'fixed' and 'variable' only."""

    FIXED = "fixed"
    VARIABLE = "variable"


# region Config


@dataclass(slots=True)
class TaskSpaceControllerCfg:
    """Task-space controller action wrapper configuration."""

    nrobot: int = 1
    """Number of robot manipulators to control. Default value is 1."""

    max_tstep: float = 0.02
    """Maximum translation step size (Euclidean norm, in meters) applied per action.
    Default value is 0.02."""

    max_rstep: float = 0.04 * np.pi
    """Maximum rotation step size (rotation vector norm, in radians) applied per action.
    Defaults value is 0.04 * pi."""

    fltr_acts_kwargs: dict[str, Any] = field(default_factory=dict)
    """Kwargs for filtering for gripper actuators. For details, see :func:`filter_actuators`.
    If empty, we rely on a simple heuristic by filtering for actuators whose `trntype` is
    `mujoco.mjtTrn.mjTRN_TENDON`. Default value is an empty dict."""

    @dataclass(slots=True)
    class CompensationCfg:
        """Dynamics and force compensation configuration."""

        mass_mult: float = 0.0
        bias_mult: float = 0.0
        grav_mult: float = 0.0
        damp_mult: float = 0.0
        fric_mult: float = 0.0

        def __post_init__(self) -> None:
            for f in fields(self):
                f_value = getattr(self, f.name)
                assert f_value >= 0, f"{f.name} field must be non-negative. Got {f_value}."
            if self.bias_mult > 0 and self.grav_mult > 0:
                warnings.warn(
                    "Both bias and gravity compensation are active. This is probably not "
                    "intended behavior, since gravity will be compensated twice"
                )

    comp_cfg: CompensationCfg = field(default_factory=CompensationCfg)
    """Dynamics and force compensation configuration. See :class:`CompensationCfg`."""

    @dataclass(frozen=True, slots=True)
    class ParameterCfg:
        """Stiffness and damping parameters configuration."""

        param_space: ParamSpace = "joint"
        """Space where parameters are defined. Default value is 'joint'."""

        kp: float | Sequence[float] = 100.0
        """Positional gain (stiffness) for motion control. Space according to
        `param_space`. If a sequence is given, its length must match the expected number
        of DOFs. If `kp_type` is variable, value is interpreted as the maximum value.
        Default value is 100."""

        kp_type: ParamType = "fixed"
        """Positional gain (stiffness) type. Default value is 'fixed'."""

        damping: float | Sequence[float] = 1.0
        """Damping ratio for motion control. Space according to `param_space`. If a
        sequence is given, its length must match the expected number of DOFs. If
        `damping_type` is variable, value is interpreted as the maximum value.
        Default value is 1."""

        damping_type: ParamType = "fixed"
        """Damping ratio type. Default value is 'fixed'."""

        def __post_init__(self) -> None:
            assert np.all(np.asarray(self.kp) > 0), f"Kp must be positive. Got {self.kp}."
            assert np.all(np.asarray(self.damping) >= 0), (
                f"Damping must be non-negative. Got {self.damping}."
            )

    param_cfg: ParameterCfg = ParameterCfg()
    """Stiffness and damping parameters configuration."""


# region Controller


class TaskSpaceControllerAction(ABC, gym.ActionWrapper):
    """Base task-space action wrapper for MuJoCo manipulation environments.

    Each action specifies, per robot, a **delta pose** relative to the end-effector
    site's current pose:
    - A delta position offset, expressed in the world frame
    - A delta rotation, expressed as a rotation vector in the tangent space of the
        site's current orientation.

    Allows for **configurable compensation** of robot inertia, bias
    (Coriolis + centrifugal + gravity), gravity only, viscous damping, or dry friction.
    Additonally, supports both **fixed or variable stiffness and damping ratio** motion
    control parameters, defined in either the joint- or task-spaces.

    Before applying an action, translation and rotation offsets are each clipped to a
    maximum step size. Actions are ordered as (delta pose, stiffness, damping ratio,
    gripper controls) if the full action space with variable motion parameters is used.
    Actions are expected in a normalized [-1, 1] range and are internally unscaled, split
    into their individual parts and used accordingly.

    **Notes**:
    - Wrapper assumes *no other* action wrappers have been already applied to the
        environment.
    - To allow force control, wrapper converts all non-user-defined actuators into motor
        actuators with a unit gain. Force ranges are used to update ctrl ranges.
    - By default, no compensation takes place and motion control parameters are fixed.
    - If variable stiffness or damping ratio are used, their values in the config are
        interpreted as their max values and used to scale the actions space accordingly.
    - If `mass_mult` is zero, joint accelerations (in this case direct torques) will not
        be multiplied by the generalized mass matrix.
    - If `bias_mult` is non-zero, `grav_mult` should be zero to prevent double gravity
        compensation.
    - If `fric_mult` is > 1, a friction over-compensation torque is applied whose
        magnitude is `frictionloss` * (fric_mult - 1) and sign is derived from the vector
        of joint torques that drive the robot's motion.

    Args:
        env: The MuJoCo-based manipulation environment to wrap.
        cfg: Configuration for task-space actions, compensation terms and motion control
            parameters. See :class:`TaskSpaceControllerCfg`.
    """

    def __init__(self, env: MujocoBaseEnv, cfg: TaskSpaceControllerCfg = TaskSpaceControllerCfg()):
        super().__init__(env)
        assert isinstance(env.unwrapped, MujocoBaseEnv), (
            f"Unsupported env type {env.unwrapped.__class__.__name__}. "
            f"Must be a subclass of {MujocoBaseEnv.__name__}."
        )
        assert env.unwrapped.model.nu == env.action_space.shape[0]
        self.cfg = cfg
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        # Setup action and nv buffers
        self.action_buffer = np.zeros(self.model.nu, dtype=env.action_space.dtype)
        self._nv_buffer = np.zeros(self.model.nv, dtype=np.float64)
        self._nv_M_buffer = np.zeros_like(self._nv_buffer)
        # Find separate robot and gripper actuator, ctrl and dof ids
        rbt_acts, gri_acts = self._split_model_actuators(self.model, cfg.fltr_acts_kwargs)
        self._rbt_dof_range, rbt_dof_len = self._get_actuator_dof_range(self.model, rbt_acts)
        self._rbt_ctrl_range, rbt_ctrl_len = self._get_actuators_ctrl_range(self.model, rbt_acts)
        self._gri_ctrl_range, _ = self._get_actuators_ctrl_range(self.model, gri_acts)
        assert rbt_ctrl_len == rbt_dof_len, (
            f"Robot nu must be equal to nv. Got {rbt_ctrl_len} and {rbt_dof_len}."
        )
        assert all(
            int(self.model.actuator_trntype[act])
            in [mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT]
            for act in rbt_acts
        ), "Robot actuator transmission types must be joints."
        # Setup frictionloss and gravcomp joint/body parameters
        friction_residual = self._setup_frictionloss_and_gravcomp(
            self.model, rbt_acts, cfg.comp_cfg.fric_mult, cfg.comp_cfg.grav_mult
        )
        self._friction_residual = np.array(friction_residual, dtype=np.float64)
        # Convert robot actuators to torque-driven motor actuators
        self._to_motor_actuators(self.model, rbt_acts)
        dtype = env.action_space.dtype
        self.env.action_space.low = self.model.actuator_ctrlrange[:, 0].astype(dtype)
        self.env.action_space.high = self.model.actuator_ctrlrange[:, 1].astype(dtype)
        # Setup motion control parameters
        ncontrol = 6 * cfg.nrobot if cfg.param_cfg.param_space == "task" else len(rbt_acts)
        kp = np.broadcast_to(cfg.param_cfg.kp, ncontrol).copy()
        damping = np.broadcast_to(cfg.param_cfg.damping, ncontrol).copy()
        # Setup action space
        action_space_unscaled = self._get_unscaled_action_space(kp, damping)
        self.action_space, _, self.unscale_action = rescale_box(
            action_space_unscaled, new_min=-1, new_max=1
        )
        # Build function for splitting composite action
        self.split_action = self._build_split_action(kp, damping)

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        # Unscale and split action
        action = self.unscale_action(action.clip(-1, 1))
        ts_action, kp, kp_sqrt, damping, gri_action = self.split_action(action)
        kv = 2 * kp_sqrt * damping
        # Update robot ctrls and scale them by generalized mass matrix
        self._nv_buffer[self._rbt_dof_range] = self.get_robot_ctrl(ts_action, kp, kv)
        if self.cfg.comp_cfg.mass_mult > 0:
            mujoco.mj_mulM(self.model, self.data, self._nv_M_buffer, self._nv_buffer)
            rbt_ctrl = self.cfg.comp_cfg.mass_mult * self._nv_M_buffer[self._rbt_dof_range]
        else:
            rbt_ctrl = self._nv_buffer[self._rbt_dof_range]
        # Set robot ctrl and remaining compensation terms
        self.action_buffer[self._rbt_ctrl_range] = (
            rbt_ctrl
            + self.cfg.comp_cfg.bias_mult * self.data.qfrc_bias[self._rbt_dof_range]
            - self.cfg.comp_cfg.damp_mult * self.data.qfrc_damper[self._rbt_dof_range]
            + np.sign(rbt_ctrl) * self._friction_residual
        )
        # Set gripper action
        if self._gri_ctrl_range:
            self.action_buffer[self._gri_ctrl_range] = gri_action
        return self.action_buffer

    @abstractmethod
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
        pass

    # region Helpers

    @staticmethod
    def _indices_to_slice(indices: Sequence[int]) -> slice | tuple[int, ...]:
        """Convert a sequence of non-negative unique indices into a slice if possible."""
        seq = tuple(indices)
        if not seq:
            return seq
        if len(seq) == 1:
            return slice(seq[0], seq[0] + 1, 1)
        start, step = seq[0], seq[1] - seq[0]
        # Check if step size is constant
        if all(seq[i] - seq[i - 1] == step for i in range(1, len(seq))):
            stop = seq[-1] + step
            return slice(start, stop, step)
        return seq

    @staticmethod
    def _split_model_actuators(
        model: mujoco.MjModel, fltr_kwargs: dict[str, Any]
    ) -> tuple[list[int], list[int]]:
        """Split the model actuator IDs into separate robot and gripper actuators."""
        if fltr_kwargs:  # rely on user filters
            gripper_actuators = filter_actuators(model, **fltr_kwargs)
        else:  # fall back to simple trntype heuristic
            gripper_actuators = filter_actuators(
                model, trntype=mujoco.mjtTrn.mjTRN_TENDON, flags=(True,)
            )
        robot_actuators = [i for i in range(model.nactuator) if i not in gripper_actuators]
        return robot_actuators, gripper_actuators

    @staticmethod
    def _get_actuator_qpos_range(
        model: mujoco.MjModel, actuators: list[int]
    ) -> tuple[slice | tuple[int, ...], int]:
        """Get the range (slice or indices) that correspond to the given actuators in qpos."""
        qpos_indices = []
        for act in actuators:
            jnt_id = model.actuator_trnid[act, 0]
            qposadr = model.jnt_qposadr[jnt_id]
            qposdim = mjtjoint_to_qpos_dim(model.jnt_type[jnt_id])
            qpos_indices.extend(list(range(qposadr, qposadr + qposdim)))
        return TaskSpaceControllerAction._indices_to_slice(qpos_indices), len(qpos_indices)

    @staticmethod
    def _get_actuator_dof_range(
        model: mujoco.MjModel, actuators: list[int]
    ) -> tuple[slice | tuple[int, ...], int]:
        """Get the range (slice or indices) that correspond to the given actuators in dof arrays."""
        dof_indices = []
        for act in actuators:
            jnt_id = model.actuator_trnid[act, 0]
            dofadr = model.jnt_dofadr[jnt_id]
            dofdim = mjtjoint_to_dof_dim(model.jnt_type[jnt_id])
            dof_indices.extend(list(range(dofadr, dofadr + dofdim)))
        return TaskSpaceControllerAction._indices_to_slice(dof_indices), len(dof_indices)

    @staticmethod
    def _get_actuators_ctrl_range(
        model: mujoco.MjModel, actuators: list[int]
    ) -> tuple[slice | tuple[int, ...], int]:
        """Get the range (slice or indices) that correspond to the given actuators in ctrl."""
        ctrl_indices = []
        for act in actuators:
            ctrladr = model.actuator_ctrladr[act]
            ctrlnum = model.actuator_ctrlnum[act]
            ctrl_indices.extend(list(range(ctrladr, ctrladr + ctrlnum)))
        return TaskSpaceControllerAction._indices_to_slice(ctrl_indices), len(ctrl_indices)

    @staticmethod
    def _setup_frictionloss_and_gravcomp(
        model: mujoco.MjModel, actuators: list[int], fric_mult: float, grav_mult: float
    ) -> tuple[float, ...]:
        """Setup actuator-associated joint frictionloss and body gravcomp parameters."""
        residual = []
        for act in actuators:
            jnt_id = model.actuator_trnid[act, 0]
            dof_adr = model.jnt_dofadr[jnt_id]
            body_id = model.jnt_bodyid[jnt_id]
            residual.append(model.dof_frictionloss[dof_adr] * max(fric_mult - 1, 0))
            model.dof_frictionloss[dof_adr] *= max(1 - fric_mult, 0)
            model.jnt_actgravcomp[jnt_id] = True
            model.body_gravcomp[body_id] = grav_mult
        return tuple(residual)

    @staticmethod
    def _to_motor_actuators(model: mujoco.MjModel, actuators: list[int]) -> None:
        """Convert non-user actuators in `actuators` to motor actuators with a unit gain."""
        for act in actuators:
            if (
                model.actuator_dyntype[act] == mujoco.mjtDyn.mjDYN_USER
                or model.actuator_gaintype[act] == mujoco.mjtGain.mjGAIN_USER
                or model.actuator_biastype[act] == mujoco.mjtBias.mjBIAS_USER
            ):
                warnings.warn(
                    f"Skipping conversion of user-defined actuator {act}, "
                    f"named {model.actuator(act).name}, to a direct motor."
                )
                continue
            # Reset dyn, gain, bias and gear parameters to motor configuration
            model.actuator_dyntype[act] = mujoco.mjtDyn.mjDYN_NONE
            model.actuator_gaintype[act] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_NONE
            model.actuator_gainprm[act, 0] = 1.0
            model.actuator_gainprm[act, 1:] = 0.0
            model.actuator_biasprm[act] = 0.0
            model.actuator_gear[act] = [1.0] + [0.0] * 5
            # Update ctrl range from force range
            model.actuator_ctrllimited[act] = True
            if model.actuator_forcelimited[act]:
                model.actuator_ctrlrange[act] = model.actuator_forcerange[act]
            else:  # fall back to actuation force limits on corresponding joint
                jnt_id = model.actuator_trnid[act, 0]
                assert model.jnt_actfrclimited[jnt_id], (
                    f"Actuator {act} doesn't have force limits. Ctrls must be limited "
                    "through actuator 'forcerange' or joint 'actuatorfrcrange'."
                )
                model.actuator_ctrlrange[act] = model.jnt_actfrcrange[jnt_id]

    def _get_unscaled_action_space(self, kp: FloatArray, damping: FloatArray) -> spaces.Box:
        """Get unscaled robot (task-space) + parameter (if variable) + gripper (unchanged)
        box action space."""
        cfg = self.cfg
        dtype = self.env.action_space.dtype
        low_new = np.array(([-cfg.max_tstep] * 3 + [-cfg.max_rstep] * 3) * cfg.nrobot, dtype=dtype)
        high_new = np.array(([cfg.max_tstep] * 3 + [cfg.max_rstep] * 3) * cfg.nrobot, dtype=dtype)
        if cfg.param_cfg.kp_type == "variable":
            low_new = np.concatenate([low_new, np.zeros_like(kp).astype(dtype=dtype)])
            high_new = np.concatenate([high_new, kp.astype(dtype=dtype)])
        if cfg.param_cfg.damping_type == "variable":
            low_new = np.concatenate([low_new, np.zeros_like(damping).astype(dtype=dtype)])
            high_new = np.concatenate([high_new, damping.astype(dtype=dtype)])
        if self._gri_ctrl_range:
            low_new = np.concatenate([low_new, self.env.action_space.low[self._gri_ctrl_range]])
            high_new = np.concatenate([high_new, self.env.action_space.high[self._gri_ctrl_range]])
        return spaces.Box(low=low_new, high=high_new)

    # region Action Helpers

    def _build_split_action(
        self, kp: FloatArray, damping: FloatArray
    ) -> Callable[[WrapperActType], tuple[FloatArray, ...]]:
        """Build a function for splitting actions into a
        (task-space, kp, kp_sqrt, damping, gripper) action tuple."""
        # Pre-compute sqrt(kp) and parameters for slicing
        kp_sqrt = np.sqrt(kp)
        nrobot, ncontrol = self.cfg.nrobot, kp.shape[0]
        tsdim = 6 * nrobot
        tsdim_plus_ncontrol = tsdim + ncontrol
        tsdim_plus_2ncontrol = tsdim + 2 * ncontrol
        # Create split function according to kp and damping types
        kp_type = self.cfg.param_cfg.kp_type
        damping_type = self.cfg.param_cfg.damping_type
        if kp_type == "variable" and damping_type == "variable":
            return lambda action: (
                action[:tsdim],
                action[tsdim:tsdim_plus_ncontrol],
                np.sqrt(action[tsdim:tsdim_plus_ncontrol]),
                action[tsdim_plus_ncontrol:tsdim_plus_2ncontrol],
                action[tsdim_plus_2ncontrol:],
            )
        if kp_type == "variable":  # damping_type == fixed
            return lambda action: (
                action[:tsdim],
                action[tsdim:tsdim_plus_ncontrol],
                np.sqrt(action[tsdim:tsdim_plus_ncontrol]),
                damping,
                action[tsdim_plus_ncontrol:],
            )
        if damping_type == "variable":  # kp_type == fixed
            return lambda action: (
                action[:tsdim],
                kp,
                kp_sqrt,
                action[tsdim:tsdim_plus_ncontrol],
                action[tsdim_plus_ncontrol:],
            )
        return lambda action: (action[:tsdim], kp, kp_sqrt, damping, action[tsdim:])
