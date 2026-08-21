from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ModelParamRandomizerCfg

if TYPE_CHECKING:
    import mujoco

    from ..utils.noise import NoiseModel
    from .base import SelectorType


def body_pos_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for body position (rel. to parent) randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="body_pos", inst_sel=inst_sel, attr_sel=slice(3))


def body_quat_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for body orientation (rel. to parent) randomizer configuration.

    Ensures unit-norm quaternions after randomization.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """

    def normalize_quat(model: mujoco.MjModel) -> None:
        norm = np.linalg.vector_norm(model.body_quat[inst_sel], axis=1, keepdims=True)
        model.body_quat[inst_sel] /= norm + 1e-8

    return ModelParamRandomizerCfg(
        noise, attr="body_quat", inst_sel=inst_sel, attr_sel=slice(4), post_proc=normalize_quat
    )


def body_mass_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for body mass randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="body_mass", inst_sel=inst_sel)


def body_inertia_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for body diagonal inertia randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="body_inertia", inst_sel=inst_sel, attr_sel=slice(3))


def body_gravitycomp_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for body gravity compensation parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="body_gravcomp", inst_sel=inst_sel)
