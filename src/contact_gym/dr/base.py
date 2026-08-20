from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mujoco
    import numpy as np
    from numpy.typing import NDArray

    SelectorType: TypeAlias = int | slice | Sequence[int] | NDArray[np.integer]


class DomainRandomizer(Protocol):
    """Protocol implemented by MuJoCo domain randomization models."""

    def __call__(
        self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
    ) -> None:
        """Apply randomization to `model` or `data` parameters in place."""
        ...
