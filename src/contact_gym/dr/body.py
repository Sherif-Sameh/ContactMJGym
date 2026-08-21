from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ModelParamRandomizerCfg

if TYPE_CHECKING:
    from ..utils.noise import NoiseModel
    from .base import ModelPostProc, SelectorType


def body_pos_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), post_proc: ModelPostProc | None = None
) -> ModelParamRandomizerCfg:
    """Factory for body position (rel. to parent) randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="body_pos", inst_sel=inst_sel, attr_sel=slice(3), post_proc=post_proc
    )


def body_quat_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), post_proc: ModelPostProc | None = None
) -> ModelParamRandomizerCfg:
    """Factory for body orientation (rel. to parent) randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="body_quat", inst_sel=inst_sel, attr_sel=slice(4), post_proc=post_proc
    )


def body_mass_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), post_proc: ModelPostProc | None = None
) -> ModelParamRandomizerCfg:
    """Factory for body mass randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="body_mass", inst_sel=inst_sel, post_proc=post_proc)


def body_inertia_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), post_proc: ModelPostProc | None = None
) -> ModelParamRandomizerCfg:
    """Factory for body diagonal inertia randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="body_inertia", inst_sel=inst_sel, attr_sel=slice(3), post_proc=post_proc
    )


def body_gravitycomp_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None), post_proc: ModelPostProc | None = None
) -> ModelParamRandomizerCfg:
    """Factory for body gravity compensation parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="body_gravcomp", inst_sel=inst_sel, post_proc=post_proc
    )
