from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco
    import numpy as np
    from numpy.typing import NDArray

    from ..utils.noise import NoiseModel
    from .base import SelectorType

# region Scalar Params


@dataclass(slots=True)
class ActuatorScalarRandomizerCfg:
    """Configuration for actuator scalar parameter domain randomizers."""

    noise: NoiseModel
    act_idxs: SelectorType


class ActuatorArmatureRandomizer:
    """Actuator armature parameter domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`ActuatorScalarRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ActuatorScalarRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        model.actuator_armature[self.cfg.act_idxs] = self.cfg.noise.sample(self.nominal, rng)

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray[np.floating]:
        return model.actuator_armature[self.cfg.act_idxs].copy()


class ActuatorLinearDampingRandomizer:
    """Actuator linear damping parameter domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`ActuatorScalarRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ActuatorScalarRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        model.actuator_damping[self.cfg.act_idxs] = self.cfg.noise.sample(self.nominal, rng)

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray[np.floating]:
        return model.actuator_damping[self.cfg.act_idxs].copy()


class ActuatorKpRandomizer:
    """Actuator `kp` parameter domain randomizer for position actuators.

    Ensures consistent `kp` values in both `gainprm` and `biasprm` after randomization.
    For `kv` randomization, configure the :class:`ActuatorBiasRandomizer` accordingly.

    Args:
        cfg: Randomizer configuration, see :class:`ActuatorScalarRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ActuatorScalarRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        kp_gain = self.cfg.noise.sample(self.nominal, rng)
        model.actuator_gainprm[self.cfg.act_idxs, 0] = kp_gain
        model.actuator_biasprm[self.cfg.act_idxs, 1] = -kp_gain

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray[np.floating]:
        assert (
            model.actuator_gainprm[self.cfg.act_idxs, 0]
            == -model.actuator_biasprm[self.cfg.act_idxs, 1]
        ), "Inconsistent kp gains detected in initial model state."
        return model.actuator_gainprm[self.cfg.act_idxs, 0].copy()


# region Array Params


@dataclass(slots=True)
class ActuatorArrayRandomizerCfg:
    """Configuration for actuator array parameter domain randomizers."""

    noise: NoiseModel
    act_idxs: SelectorType
    arr_idxs: SelectorType


class ActuatorDynamicsRandomizer:
    """Actuator dynamics parameters `dynprm` domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`ActuatorArrayRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ActuatorArrayRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        model.actuator_dynprm[self.cfg.act_idxs, self.cfg.arr_idxs] = self.cfg.noise.sample(
            self.nominal, rng
        )

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray[np.floating]:
        return model.actuator_dynprm[self.cfg.act_idxs, self.cfg.arr_idxs].copy()


class ActuatorGainRandomizer:
    """Actuator gain parameters `gainprm` domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`ActuatorArrayRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ActuatorArrayRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        model.actuator_gainprm[self.cfg.act_idxs, self.cfg.arr_idxs] = self.cfg.noise.sample(
            self.nominal, rng
        )

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray[np.floating]:
        return model.actuator_gainprm[self.cfg.act_idxs, self.cfg.arr_idxs].copy()


class ActuatorBiasRandomizer:
    """Actuator bias parameters `biasprm` domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`ActuatorArrayRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ActuatorArrayRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        model.actuator_biasprm[self.cfg.act_idxs, self.cfg.arr_idxs] = self.cfg.noise.sample(
            self.nominal, rng
        )

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray[np.floating]:
        return model.actuator_biasprm[self.cfg.act_idxs, self.cfg.arr_idxs].copy()
