from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mujoco
    import numpy as np
    from numpy.typing import NDArray

    from ..utils.noise import NoiseModel

    SelectorType: TypeAlias = int | slice | Sequence[int] | NDArray[np.integer]
    ModelPostProc: TypeAlias = Callable[[mujoco.MjModel], None]


class DomainRandomizer(Protocol):
    """Protocol implemented by MuJoCo domain randomization models."""

    def __call__(
        self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
    ) -> None:
        """Apply randomization to `model` or `data` parameters in place."""
        ...


# region ModelParam


@dataclass(slots=True)
class ModelParamRandomizerCfg:
    """Configuration for MjModel parameter domain randomizer."""

    noise: NoiseModel
    """Noise model for sampling parameter values, see :class:`NoiseModel`."""

    attr: str
    """Name of attribute of MjModel to randomize. (e.g., `actuator_dynprm`)."""

    inst_sel: SelectorType = slice(None)
    """Selector for instances of type (e.g., actuators, joints, etc.) whose parameters
    should be randomized. Defaults to `slice(None)` (all instances).
    """

    attr_sel: SelectorType | None = None
    """Selector for individual attributes on array parameters (e.g., `actuator_gainprm`).
    Not required for scalar attributes (e.g., `actuator_damping`). Default value is None.
    """

    post_proc: ModelPostProc | None = None
    """Optional post-processor for model after parameter updates. Default value is None."""

    def __post_init__(self):
        if self.post_proc is None:

            def do_nothing(_: mujoco.MjModel): ...

            self.post_proc = do_nothing


class ModelParamRandomizer:
    """MjModel parameter domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`ModelParamRandomizerCfg`.
        model: Optional MuJoCo model. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: ModelParamRandomizerCfg, model: mujoco.MjModel | None = None):
        self.cfg = cfg
        self.set_view = self._build_set_view()
        self.nominal = None if model is None else self._get_nominal(model)

    def __call__(self, model: mujoco.MjModel, _: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(model)
        self.set_view(model, self.cfg.noise.sample(self.nominal, rng))
        self.cfg.post_proc(model)

    def _build_set_view(self) -> Callable[[mujoco.MjModel], NDArray]:
        def setter_scalar(model: mujoco.MjModel, values: NDArray) -> None:
            getattr(model, self.cfg.attr)[self.cfg.inst_sel] = values

        def setter_array(model: mujoco.MjModel, values: NDArray) -> None:
            getattr(model, self.cfg.attr)[self.cfg.inst_sel, self.cfg.attr_sel] = values

        if self.cfg.attr_sel is None:
            return setter_scalar
        return setter_array

    def _get_nominal(self, model: mujoco.MjModel) -> NDArray:
        assert hasattr(model, self.cfg.attr), f"Model has no attribute named {self.cfg.attr}."
        nominal = getattr(model, self.cfg.attr).copy()
        if nominal.ndim == 1:
            assert self.cfg.attr_sel is None, (
                f"Attribute selector must be None for scalar parameter {self.cfg.attr}. Got {self.cfg.attr_sel}."
            )
            return nominal[self.cfg.inst_sel]

        assert self.cfg.attr_sel is not None, (
            f"Attribute selector must be specified for array parameter {self.cfg.attr}. Got None."
        )
        return nominal[self.cfg.inst_sel, self.cfg.attr_sel]
