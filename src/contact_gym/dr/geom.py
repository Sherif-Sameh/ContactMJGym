from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ModelParamRandomizerCfg

if TYPE_CHECKING:
    from ..utils.noise import NoiseModel
    from .base import SelectorType


def geom_solref_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom `solref` parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="geom_solref", inst_sel=inst_sel, attr_sel=slice(None)
    )


def geom_solimp_impedance_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom `solimp` impedance (`d0` and `dwidth`) parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="geom_solimp", inst_sel=inst_sel, attr_sel=slice(2))


def geom_sliding_friction_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom sliding friction parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="geom_friction", inst_sel=inst_sel, attr_sel=0)


def geom_sliding_torsional_friction_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom sliding and torsional friction parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="geom_friction", inst_sel=inst_sel, attr_sel=slice(2)
    )


def geom_friction_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for full friction parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="geom_friction", inst_sel=inst_sel, attr_sel=slice(3)
    )


def geom_surface_linvel_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom surface linear velocity parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="geom_surfacevel", inst_sel=inst_sel, attr_sel=slice(3)
    )


def geom_surface_angvel_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom surface angular velocity parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(
        noise, attr="geom_surfacevel", inst_sel=inst_sel, attr_sel=slice(3, 6)
    )


def geom_adhesion_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom adhesion parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="geom_adhesion", inst_sel=inst_sel)


def geom_rgb_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom color parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="geom_rgba", inst_sel=inst_sel, attr_sel=slice(3))


def geom_rgba_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for geom color + opacity parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="geom_rgba", inst_sel=inst_sel, attr_sel=slice(4))
