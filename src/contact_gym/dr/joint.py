from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ModelParamRandomizerCfg, jnt_sel_to_dof_sel

if TYPE_CHECKING:
    import mujoco

    from ..utils.noise import NoiseModel
    from .base import SelectorType


# region jnt_ attributes


def joint_linear_stiffness_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for joint linear stiffness randomizer configuration.

    `inst_sel` maps directly to `jnt_sel` since attribute is defined per joint as `jnt_*`
    arrays.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="jnt_stiffness", inst_sel=inst_sel)


# region dof_ attributes


def joint_dof_param_cfg(
    attr: str,
    noise: NoiseModel,
    inst_sel: SelectorType = slice(None),
    attr_sel: SelectorType | None = None,
    model: mujoco.MjModel | None = None,
) -> ModelParamRandomizerCfg:
    """Factory for joint per-DOF generic parameter randomizer configuration.

    If `model` is given, `inst_sel` is assumed to correspond to `jnt_sel` and is mapped
    to `dof_sel` internally, since attribute is defined per DOF as `dof_*` arrays. If not
    given, then `inst_sel` is assumed to correspond to `dof_sel` directly. `noise` should
    be configured with this conversion in mind for multi-DOF joints.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    if model is not None:
        inst_sel = jnt_sel_to_dof_sel(model, inst_sel)
    return ModelParamRandomizerCfg(noise, attr=attr, inst_sel=inst_sel, attr_sel=attr_sel)


def joint_frictionloss_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), model: mujoco.MjModel | None = None
) -> ModelParamRandomizerCfg:
    """Factory for joint friction-loss randomizer configuration.

    **Important**: non-scalar `noise` should be configured with the joints' `dof`
    dimension in mind, not the number of joints.

    See :func:`joint_dof_param_cfg` for argument descriptions.
    """
    return joint_dof_param_cfg("dof_frictionloss", noise, inst_sel, model=model)


def joint_armature_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), model: mujoco.MjModel | None = None
) -> ModelParamRandomizerCfg:
    """Factory for joint armature randomizer configuration.

    **Important**: non-scalar `noise` should be configured with the joints' `dof`
    dimension in mind, not the number of joints.

    See :func:`joint_dof_param_cfg` for argument descriptions.
    """
    return joint_dof_param_cfg("dof_armature", noise, inst_sel, model=model)


def joint_linear_damping_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), model: mujoco.MjModel | None = None
) -> ModelParamRandomizerCfg:
    """Factory for joint linear damping randomizer configuration.

    **Important**: non-scalar `noise` should be configured with the joints' `dof`
    dimension in mind, not the number of joints.

    See :func:`joint_dof_param_cfg` for argument descriptions.
    """
    return joint_dof_param_cfg("dof_damping", noise, inst_sel, model=model)
