from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DataStateRandomizerCfg, jnt_sel_to_dof_sel, jnt_sel_to_qpos_sel

if TYPE_CHECKING:
    import mujoco

    from ..utils.noise import NoiseModel
    from .base import DataPostProc, SelectorType


def qpos_state_cfg(
    noise: NoiseModel, entry_sel: SelectorType = slice(None), post_proc: DataPostProc | None = None
) -> DataStateRandomizerCfg:
    """Factory for qpos state randomizer configuration.

    See :class:`DataStateRandomizerCfg` for argument descriptions.
    """
    return DataStateRandomizerCfg(noise, attr="qpos", entry_sel=entry_sel, post_proc=post_proc)


def qpos_state_cfg_from_jnt_sel(
    model: mujoco.MjModel,
    noise: NoiseModel,
    jnt_sel: SelectorType = slice(None),
    post_proc: DataPostProc | None = None,
) -> DataStateRandomizerCfg:
    """Factory for qpos state randomizer configuration.

    `model` is required to map `jnt_sel` to `entry_sel` for qpos internally. `noise`
    should be configured with this conversion in mind for multi-DOF joints.

    See :class:`DataStateRandomizerCfg` for argument descriptions.
    """
    return qpos_state_cfg(noise, entry_sel=jnt_sel_to_qpos_sel(model, jnt_sel), post_proc=post_proc)


def qvel_state_cfg(
    noise: NoiseModel, entry_sel: SelectorType = slice(None), post_proc: DataPostProc | None = None
) -> DataStateRandomizerCfg:
    """Factory for qvel state randomizer configuration.

    See :class:`DataStateRandomizerCfg` for argument descriptions.
    """
    return DataStateRandomizerCfg(noise, attr="qvel", entry_sel=entry_sel, post_proc=post_proc)


def qvel_state_cfg_from_jnt_sel(
    model: mujoco.MjModel,
    noise: NoiseModel,
    jnt_sel: SelectorType = slice(None),
    post_proc: DataPostProc | None = None,
) -> DataStateRandomizerCfg:
    """Factory for qvel state randomizer configuration.

    `model` is required to map `jnt_sel` to `entry_sel` for qvel internally. `noise`
    should be configured with this conversion in mind for multi-DOF joints.

    See :class:`DataStateRandomizerCfg` for argument descriptions.
    """
    return qvel_state_cfg(noise, entry_sel=jnt_sel_to_dof_sel(model, jnt_sel), post_proc=post_proc)
