from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .base import CurriculumTerm

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gymnasium as gym
    from numpy.typing import NDArray

    FloatArray: TypeAlias = NDArray[np.floating]

# region Fixed


class FixedCurriculumTerm(CurriculumTerm):
    """Base class for curriculum terms with a fixed, environment step-based schedules.

    Args:
        paths: Sequence of dot-separated paths for target attributes
            (e.g., "reward.weights[0]" or "model.dof_damping").
        end: Final values for parameters at `end_step` and beyond. Each entry may be
            a scalar or array matching the shape of the target parameter.
        end_step: Final environment step for the curriculum. Parameter values are
            clamped to `end` for subsequent steps.
        start: Initial values for parameters at `start_step`, matching `end`'s
            per-parameter shapes. If None, initial values are lazily loaded from the
            environment at the first call.
        start_step: Initial environment step for the curriculum. Default value is 0.
    """

    def __init__(
        self,
        paths: Sequence[str],
        end: Sequence[float | FloatArray],
        end_step: int,
        start: Sequence[float | FloatArray] | None = None,
        start_step: int = 0,
    ):
        super().__init__(paths)
        assert len(paths) == len(end), (
            f"Number of parameters must match end values. Got {len(paths)} and {len(end)}."
        )
        assert end_step > start_step, (
            f"End step must be greater than start step. Got {end_step} and {start_step}."
        )
        self.end, self.sizes = self._to_flat(end)
        self.start: FloatArray | None = None
        if start is not None:
            assert len(start) == len(end), (
                f"Number of start and end values must match. Got {len(start)} and {len(end)}."
            )
            self.start, start_sizes = self._to_flat(start)
            assert start_sizes == self.sizes, (
                "Per-parameter start and end shapes must match. "
                f"Got sizes {start_sizes} and {self.sizes}."
            )
        self.start_step = start_step
        self.end_step = end_step
        self.shapes: list[tuple[int, ...]] | None = None

    @abstractmethod
    def __call__(self, env: gym.Env, step: int) -> dict[str, Any]:
        """Update environment parameter values in place according to preset curriculum.

        Args:
            env: Gymnasium environment.
            step: Current environment step number.

        Returns:
            Dictionary of current parameters and their values.
        """
        pass

    def setup(self, env: gym.Env) -> None:
        """Cache parameter getters/setters, their shapes, and load start values if unset."""
        super().setup(env)
        current = [np.asarray(addr.get()) for addr in self.addrs]
        self.shapes = [arr.shape for arr in current]
        if self.start is None:
            self.start, start_sizes = self._to_flat(current)
            assert start_sizes == self.sizes, (
                "Per-parameter start (lazily loaded from env) and end shapes must match. "
                f"Got sizes {start_sizes} and {self.sizes}."
            )
        else:
            env_sizes = [int(np.prod(shape)) if shape else 1 for shape in self.shapes]
            assert env_sizes == self.sizes, (
                "Per-parameter shapes found in the environment do not match the given "
                f"start/end values. Got sizes {env_sizes} and {self.sizes}."
            )

    def _normalized_step(self, step: int) -> float:
        """Clamp and normalize `step` to [0, 1] over [start_step, end_step]."""
        return min(max((step - self.start_step) / (self.end_step - self.start_step), 0.0), 1.0)

    @staticmethod
    def _to_flat(values: Sequence[float | FloatArray]) -> tuple[FloatArray, list[int]]:
        """Flatten a sequence of per-parameter scalar/array values into one 1D array.

        Args:
            values: Sequence of scalar or array values to flatten into a combined 1D array.

        Returns:
            Tuple containg the concatenated array together with each parameter's size, so
            the flat array can later be split back into its original segments.
        """
        arrs = [np.ravel(np.atleast_1d(np.asarray(v))) for v in values]
        sizes = [arr.size for arr in arrs]
        return np.concatenate(arrs), sizes

    def _apply_flat_values(self, values: FloatArray) -> dict[str, Any]:
        """Split 1D flattened array back into per-parameter segments and write each
        segment through its cached address.

        Args:
            values: Concatenated 1D flattened array of combined parameter values.

        Returns:
            Dictionary of current parameters and their values.
        """
        idx = 0
        param_dict: dict[str, Any] = {}
        for path, addr, size, shape in zip(self.paths, self.addrs, self.sizes, self.shapes):
            segment = values[idx : idx + size]
            value = float(segment[0]) if shape == () else segment.reshape(shape)
            addr.set(value)
            param_dict[path] = value
            idx += size
        return param_dict


# region Linear


class LinearCurriculumTerm(FixedCurriculumTerm):
    """Curriculum learning term for linearly varying parameters over step count.

    See :class:`FixedCurriculumTerm` for arguments. Each entry in `paths` may target
    either a scalar or array parameter.
    """

    def __call__(self, env: gym.Env, step: int) -> dict[str, Any]:
        """Update environment parameter values in place according to preset curriculum.

        Args:
            env: Gymnasium environment.
            step: Current environment step number.

        Returns:
            Dictionary of current parameters and their values.
        """
        if self.addrs is None:
            self.setup(env)
        t = self._normalized_step(step)
        values = self.start + t * (self.end - self.start)
        return self._apply_flat_values(values)


# region Cosine


class CosineAnnealingCurriculumTerm(FixedCurriculumTerm):
    """Curriculum learning term for varying parameters over step count following a cosine
    annealing schedule.

    See :class:`FixedCurriculumTerm` for arguments. Each entry in `paths` may target
    either a scalar or array parameter.
    """

    def __call__(self, env: gym.Env, step: int) -> dict[str, Any]:
        """Update environment parameter values in place according to preset curriculum.

        Args:
            env: Gymnasium environment.
            step: Current environment step number.

        Returns:
            Dictionary of current parameters and their values.
        """
        if self.addrs is None:
            self.setup(env)
        t = self._normalized_step(step)
        factor = 0.5 * (1.0 + math.cos(math.pi * t))
        values = self.end + factor * (self.start - self.end)
        return self._apply_flat_values(values)
