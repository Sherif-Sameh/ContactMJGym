from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ModelParamRandomizerCfg, jnt_sel_to_dof_sel

if TYPE_CHECKING:
    import mujoco

    from ..utils.noise import NoiseModel
    from .base import ModelPostProc, SelectorType


# region jnt_ attributes


def joint_linear_stiffness_cfg(
    noise: NoiseModel, jnt_sel: SelectorType = slice(None), post_proc: ModelPostProc | None = None
) -> ModelParamRandomizerCfg:
    """Factory for joint linear stiffness randomizer configuration.

    `jnt_sel` maps directly to `inst_sel` since attribute is defined per joint as `jnt_*`
    arrays.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="jnt_stiffness", inst_sel=jnt_sel, post_proc=post_proc
    )


# region dof_ attributes


def joint_dof_param_cfg(
    model: mujoco.MjModel,
    attr: str,
    noise: NoiseModel,
    jnt_sel: SelectorType = slice(None),
    attr_sel: SelectorType | None = None,
    post_proc: ModelPostProc | None = None,
) -> ModelParamRandomizerCfg:
    """Factory for joint per-DOF generic parameter randomizer configuration.

    `model` is required to map `jnt_sel` to `dof_sel` internally since attribute is
    defined per DOF as `dof_*` arrays. `dof_sel` is passed as `inst_sel` to randomizer
    configuration. `noise` should be configured with this conversion in mind for
    multi-DOF joints.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    dof_sel = jnt_sel_to_dof_sel(model, jnt_sel)
    return ModelParamRandomizerCfg(
        noise, attr=attr, inst_sel=dof_sel, attr_sel=attr_sel, post_proc=post_proc
    )


def joint_frictionloss_cfg(
    model: mujoco.MjModel,
    noise: NoiseModel,
    jnt_sel: SelectorType = slice(None),
    post_proc: ModelPostProc | None = None,
) -> ModelParamRandomizerCfg:
    """Factory for joint friction-loss randomizer configuration.

    **Important**: non-scalar `noise` should be configured with the joints' `dof`
    dimension in mind, not the number of joints.

    See :func:`joint_dof_param_cfg` for argument descriptions.
    """
    return joint_dof_param_cfg(model, "dof_frictionloss", noise, jnt_sel, post_proc=post_proc)


def joint_armature_cfg(
    model: mujoco.MjModel,
    noise: NoiseModel,
    jnt_sel: SelectorType = slice(None),
    post_proc: ModelPostProc | None = None,
) -> ModelParamRandomizerCfg:
    """Factory for joint armature randomizer configuration.

    **Important**: non-scalar `noise` should be configured with the joints' `dof`
    dimension in mind, not the number of joints.

    See :func:`joint_dof_param_cfg` for argument descriptions.
    """
    return joint_dof_param_cfg(model, "dof_armature", noise, jnt_sel, post_proc=post_proc)


def joint_linear_damping_cfg(
    model: mujoco.MjModel,
    noise: NoiseModel,
    jnt_sel: SelectorType = slice(None),
    post_proc: ModelPostProc | None = None,
) -> ModelParamRandomizerCfg:
    """Factory for joint linear damping randomizer configuration.

    **Important**: non-scalar `noise` should be configured with the joints' `dof`
    dimension in mind, not the number of joints.

    See :func:`joint_dof_param_cfg` for argument descriptions.
    """
    return joint_dof_param_cfg(model, "dof_damping", noise, jnt_sel, post_proc=post_proc)
