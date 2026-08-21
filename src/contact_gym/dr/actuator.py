from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ModelParamRandomizerCfg

if TYPE_CHECKING:
    import mujoco

    from ..utils.noise import NoiseModel
    from .base import SelectorType


def actuator_motor_gain_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for motor actuator `gainprm[0]` parameter randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="actuator_gainprm", inst_sel=inst_sel, attr_sel=0)


def actuator_position_kp_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for position actuator `kp` parameter randomizer configuration.

    Ensures consistent `kp` values in both `gainprm` and `biasprm` after randomization.
    For `kv` randomization, see :func:`actuator_position_neg_kv_cfg`.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """

    def copy_kp_into_biasprm(model: mujoco.MjModel) -> None:
        model.actuator_biasprm[inst_sel, 1] = -model.actuator_gainprm[inst_sel, 0]

    return ModelParamRandomizerCfg(
        noise,
        attr="actuator_gainprm",
        inst_sel=inst_sel,
        attr_sel=0,
        post_proc=copy_kp_into_biasprm,
    )


def actuator_position_neg_kv_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for position actuator `-kv` parameter randomizer configuration.

    **Important**: Randomization is setup for **NEGATIVE** `kv`, since the model has no
    attribute that corresponds to `kv` only. Therefore, `noise` should setup with that in
    mind. For `kp` randomization, see :func:`actuator_position_kp_cfg`.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="actuator_biasprm", inst_sel=inst_sel, attr_sel=2)


def actuator_linear_damping_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for actuator linear damping randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="actuator_damping", inst_sel=inst_sel)


def actuator_armature_cfg(
    noise: NoiseModel, inst_sel: SelectorType = slice(None)
) -> ModelParamRandomizerCfg:
    """Factory for actuator armature randomizer configuration.

    See :class:`ModelParamRandomizerCfg` for argument descriptions.
    """
    return ModelParamRandomizerCfg(noise, attr="actuator_armature", inst_sel=inst_sel)
