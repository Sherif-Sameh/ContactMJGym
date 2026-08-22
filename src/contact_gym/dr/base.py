from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, TypeAlias

import mujoco

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ..utils.noise import NoiseModel

    SelectorType: TypeAlias = int | slice | Sequence[int] | NDArray[np.integer]
    ModelPostProc: TypeAlias = Callable[[mujoco.MjModel], None]
    DataPostProc: TypeAlias = Callable[[mujoco.MjData], None]


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
    should be randomized. Default value is `slice(None)` (all instances).
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


# region DataState


@dataclass(slots=True)
class DataStateRandomizerCfg:
    """Configuration for MjData state domain randomizer."""

    noise: NoiseModel
    """Noise model for sampling state values, see :class:`NoiseModel`."""

    attr: str
    """Name of attribute of MjData to randomize. (e.g., `qpos`)."""

    entry_sel: SelectorType = slice(None)
    """Selector for individual entries on state arrays. Default value is `slice(None)`
    (all entries).
    """

    post_proc: DataPostProc | None = None
    """Optional post-processor for data after parameter updates. Default value is None."""

    def __post_init__(self):
        if self.post_proc is None:

            def do_nothing(_: mujoco.MjData): ...

            self.post_proc = do_nothing


class DataStateRandomizer:
    """MjData state domain randomizer.

    Args:
        cfg: Randomizer configuration, see :class:`DataStateRandomizerCfg`.
        data: Optional MuJoCo data. If not given, nominal values are lazily initialized
            on the first call. Otherwise, they're initialized during construction.
            Default value is None.
    """

    def __init__(self, cfg: DataStateRandomizerCfg, data: mujoco.MjData | None = None):
        self.cfg = cfg
        self.nominal = None if data is None else self._get_nominal(data)

    def __call__(self, _: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.nominal is None:
            self.nominal = self._get_nominal(data)
        self.set_view(data, self.cfg.noise.sample(self.nominal, rng))
        self.cfg.post_proc(data)

    def set_view(self, data: mujoco.MjData, values: NDArray) -> None:
        getattr(data, self.cfg.attr)[self.cfg.entry_sel] = values

    def _get_nominal(self, data: mujoco.MjData) -> NDArray:
        assert hasattr(data, self.cfg.attr), f"Data has no attribute named {self.cfg.attr}."
        return getattr(data, self.cfg.attr)[self.cfg.entry_sel].copy()


# region Helpers


def jnt_sel_to_qpos_sel(model: mujoco.MjModel, jnt_sel: SelectorType) -> SelectorType:
    """Map instance/entry selector from model joints to qpos."""
    if isinstance(jnt_sel, int):
        qpos_adr = model.jnt_qposadr[jnt_sel]
        qpos_dim = _get_jnt_qpos_dim(model.jnt_type[jnt_sel])
        return qpos_adr if qpos_dim == 1 else slice(qpos_adr, qpos_adr + qpos_dim)
    if isinstance(jnt_sel, slice) and jnt_sel.step in [None, 1]:
        jnt_types = model.jnt_type[jnt_sel]
        qpos_adr = (
            model.jnt_qposadr[0] if jnt_sel.start is None else model.jnt_qposadr[jnt_sel.start]
        )
        qpos_dim = sum(_get_jnt_qpos_dim(jnt_type) for jnt_type in jnt_types)
        return slice(qpos_adr, qpos_adr + qpos_dim)
    # Non-contiguous slice, sequence, or array of joints -> sequence of qpos
    jnt_types = model.jnt_type[jnt_sel]
    qpos_adrs = model.jnt_qposadr[jnt_sel]
    qpos_dims = [_get_jnt_qpos_dim(jnt_type) for jnt_type in jnt_types]
    return sum(
        [
            tuple(range(qpos_adr, qpos_adr + qpos_dim))
            for qpos_adr, qpos_dim in zip(qpos_adrs, qpos_dims)
        ],
        start=(),
    )


def jnt_sel_to_dof_sel(model: mujoco.MjModel, jnt_sel: SelectorType) -> SelectorType:
    """Map instance/entry selector from model joints to dofs."""
    if isinstance(jnt_sel, int):
        dof_adr = model.jnt_dofadr[jnt_sel]
        dof_dim = _get_jnt_dof_dim(model.jnt_type[jnt_sel])
        return dof_adr if dof_dim == 1 else slice(dof_adr, dof_adr + dof_dim)
    if isinstance(jnt_sel, slice) and jnt_sel.step in [None, 1]:  # contiguous block
        jnt_types = model.jnt_type[jnt_sel]
        dof_adr = model.jnt_dofadr[0] if jnt_sel.start is None else model.jnt_dofadr[jnt_sel.start]
        dof_dim = sum(_get_jnt_dof_dim(jnt_type) for jnt_type in jnt_types)
        return slice(dof_adr, dof_adr + dof_dim)
    # Non-contiguous slice, sequence, or array of joints -> sequence of dofs
    jnt_types = model.jnt_type[jnt_sel]
    dof_adrs = model.jnt_dofadr[jnt_sel]
    dof_dims = [_get_jnt_dof_dim(jnt_type) for jnt_type in jnt_types]
    return sum(
        [tuple(range(dof_adr, dof_adr + dof_dim)) for dof_adr, dof_dim in zip(dof_adrs, dof_dims)],
        start=(),
    )


def _get_jnt_qpos_dim(jnt_type: mujoco.mjtJoint) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _get_jnt_dof_dim(jnt_type: mujoco.mjtJoint) -> int:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1
