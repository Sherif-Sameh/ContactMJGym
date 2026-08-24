from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .base import CurriculumTerm

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gymnasium as gym
    from numpy.typing import NDArray

    FloatArray: TypeAlias = NDArray[np.floating]

# region Linear


class LinearCurriculumTerm(CurriculumTerm):
    """Curriculum learning term for linearly varying parameters over step count.

    Args:
        paths: Sequence of dot-separated paths for target attributes
            (e.g., "env.reward.weights[0]").
        end: Final values for parameters at `end_step` and beyond.
        end_step: Final environment step for linear curriculum. Parameter values are
            clamped to `end` for subsequent steps.
        start: Initial values for parameters at `start_step`. If None, initial values are
            lazily loaded from the environment at the first call.
        start_step: Initial environment step for linear curriculum. Default value is 0.
    """

    def __init__(
        self,
        paths: Sequence[str],
        end: Sequence[float] | FloatArray,
        end_step: int,
        start: Sequence[float] | FloatArray | None = None,
        start_step: int = 0,
    ):
        super().__init__(paths)
        assert len(paths) == len(end), (
            f"Number of parameters must match end values. Got {len(paths)} and {len(end)}."
        )
        assert end_step > start_step, (
            f"End step must be greater than start step. Got {end_step} and {start_step}."
        )
        self.start = start
        self.end = np.asarray(end)
        self.start_step = start_step
        self.end_step = end_step
        if self.start is not None:
            assert len(start) == len(end), (
                f"Number of start and end values must match. Got {len(start)} and {len(end)}."
            )
            self.start = np.asarray(start)

    def __call__(self, env: gym.Env, step: int) -> dict[str, Any] | None:
        """Update environment parameter values in place according to preset curriculum.

        Args:
            env: Gymnasium environment.
            step: Current environment step number.

        Returns:
            Optional dictionary of current parameters and their values.
        """
        if self.addrs is None:
            self.setup(env)
        # Compute updated values
        t = min(max((step - self.start_step) / (self.end_step - self.start_step), 0.0), 1.0)
        values = self.start + t * (self.end - self.start)
        # Set parameters and build parameter dict
        param_dict = {}
        for i, value in enumerate(values):
            self.addrs[i].set(value)
            param_dict[self.paths[i]] = value
        return param_dict

    def setup(self, env: gym.Env) -> None:
        """Cache parameter getters and setters and load start values if unset."""
        super().setup(env)
        if self.start is None:
            self.start = np.array([addr.get() for addr in self.addrs])


# region Cosine


class CosineAnnealingCurriculumTerm(CurriculumTerm):
    """Curriculum learning term for varying parameters over step count following a cosine
    annealing schedule.

    Args:
        paths: Sequence of dot-separated paths for target attributes
            (e.g., "env.reward.weights[0]").
        end: Final values for parameters at `end_step` and beyond.
        end_step: Final environment step for cosine annealing curriculum. Parameter values
            are clamped to `end` for subsequent steps.
        start: Initial values for parameters at `start_step`. If None, initial values are
            lazily loaded from the environment at the first call.
        start_step: Initial environment step for cosine annealing curriculum. Default value
            is 0.
    """

    def __init__(
        self,
        paths: Sequence[str],
        end: Sequence[float] | FloatArray,
        end_step: int,
        start: Sequence[float] | FloatArray | None = None,
        start_step: int = 0,
    ):
        super().__init__(paths)
        assert len(paths) == len(end), (
            f"Number of parameters must match end values. Got {len(paths)} and {len(end)}."
        )
        assert end_step > start_step, (
            f"End step must be greater than start step. Got {end_step} and {start_step}."
        )
        self.start = start
        self.end = np.asarray(end)
        self.start_step = start_step
        self.end_step = end_step
        if self.start is not None:
            assert len(start) == len(end), (
                f"Number of start and end values must match. Got {len(start)} and {len(end)}."
            )
            self.start = np.asarray(start)

    def __call__(self, env: gym.Env, step: int) -> dict[str, Any] | None:
        """Update environment parameter values in place according to preset curriculum.

        Args:
            env: Gymnasium environment.
            step: Current environment step number.

        Returns:
            Optional dictionary of current parameters and their values.
        """
        if self.addrs is None:
            self.setup(env)
        # Compute updated values
        t = min(max((step - self.start_step) / (self.end_step - self.start_step), 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * t))
        values = self.end + factor * (self.start - self.end)
        # Set parameters and build parameter dict
        param_dict = {}
        for i, value in enumerate(values):
            self.addrs[i].set(value)
            param_dict[self.paths[i]] = value
        return param_dict

    def setup(self, env: gym.Env) -> None:
        """Cache parameter getters and setters and load start values if unset."""
        super().setup(env)
        if self.start is None:
            self.start = np.array([addr.get() for addr in self.addrs])
