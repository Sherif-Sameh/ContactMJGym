from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike, NDArray

    FloatArray: TypeAlias = NDArray[np.floating]
    ParamType: TypeAlias = float | Sequence[float] | FloatArray
    Operation: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]


# region Noise


class NoiseModel(Protocol):
    """Protocol implemented by functional noise models."""

    def sample(self, nominal: ParamType, rng: np.random.Generator) -> FloatArray:
        """Sample noise and apply it to the nominal value."""
        ...


class Noise:
    """Functional noise model combining a sampler with a fixed operation.

    The sampler generates the noise and the selected operation combines that noise with
    the nominal value. The operation is resolved during construction.

    Args:
        sampler: Noise sampler, following the :class:`Sampler` protocol.
        operation: Operation to combine noise with nominal value. Must be one of
            ["add", "scale", "abs"]. Default value is "add".
    """

    def __init__(self, sampler: Sampler, operation: Literal["add", "scale", "abs"] = "add") -> None:
        assert operation in _OPERATIONS, (
            f"Unsupported operation {operation}. Expected one of {tuple(_OPERATIONS)}"
        )
        self._sampler = sampler
        self._operation = _OPERATIONS[operation]

    def sample(self, nominal: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Sample noise and apply the configured operation to the nominal value."""
        noise = self._sampler.sample(rng)
        return self._operation(nominal, noise)


# region Samplers


class Sampler(Protocol):
    """Protocol implemented by noise samplers."""

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a noise sample using the supplied random-number generator."""
        ...


class Constant:
    """Sampler that always returns the configured constant value.

    Args:
        value: Constant value returned by the `sample` method.
        dtype: Optional datatype for sampler parameters. If None, the default dtype is
            derived from the parameters if they're float arrays, otherwise it defaults
            to `np.float64`. Default value is None.
    """

    def __init__(self, value: ParamType, *, dtype: DTypeLike | None = None) -> None:
        assert dtype is None or np.issubdtype(dtype, np.floating)
        self._value = _as_float_array(value, dtype)

    def sample(self, _: np.random.Generator) -> FloatArray:
        """Return the configured constant, ignoring the random generator."""
        return self._value


class Uniform:
    """Sampler for a uniform distribution over a closed parameter range.

    Args:
        min: Minimum value for the uniform distribution.
        max: Maximum value for the uniform distribution.
        dtype: Optional datatype for sampler parameters. If None, the default dtype is
            derived from the parameters if they're float arrays, otherwise it defaults
            to `np.float64`. Default value is None.
    """

    def __init__(self, min: ParamType, max: ParamType, *, dtype: DTypeLike | None = None) -> None:
        assert dtype is None or np.issubdtype(dtype, np.floating)
        self._min = _as_float_array(min, dtype)
        self._max = _as_float_array(max, dtype)
        assert np.all(self._min <= self._max), (
            f"min must be less than or equal to max. Got min {self._min} and max {self._max}."
        )

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a uniformly distributed sample."""
        return rng.uniform(self._min, self._max)


class Gaussian:
    """Sampler for a Gaussian distribution.

    Args:
        mean: Mean value for the Gaussian distribution. Default value is 0.
        std: Standard deviation for the Gaussian distribution. Default value is 1.
        dtype: Optional datatype for sampler parameters. If None, the default dtype is
            derived from the parameters if they're float arrays, otherwise it defaults
            to `np.float64`. Default value is None.
    """

    def __init__(
        self, mean: ParamType = 0.0, std: ParamType = 1.0, *, dtype: DTypeLike | None = None
    ) -> None:
        assert dtype is None or np.issubdtype(dtype, np.floating)
        self._mean = _as_float_array(mean, dtype)
        self._std = _as_float_array(std, dtype)
        assert np.all(self._std >= 0), f"std must be non-negative. Got std {self._std}."

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a normally distributed sample."""
        return rng.normal(self._mean, self._std)


class SquashedGaussian:
    """Sampler for a squashed Gaussian distribution through tanh.

    Args:
        mean: Mean value for the Gaussian distribution. Default value is 0.
        std: Standard deviation for the Gaussian distribution. Default value is 1.
        scale: Scale factor for the squashed Gaussian distribution. Default value is 1.
        dtype: Optional datatype for sampler parameters. If None, the default dtype is
            derived from the parameters if they're float arrays, otherwise it defaults
            to `np.float64`. Default value is None.
    """

    def __init__(
        self,
        mean: ParamType = 0.0,
        std: ParamType = 1.0,
        scale: ParamType = 1.0,
        *,
        dtype: DTypeLike | None = None,
    ) -> None:
        assert dtype is None or np.issubdtype(dtype, np.floating)
        self._mean = _as_float_array(mean, dtype)
        self._std = _as_float_array(std, dtype)
        self._scale = _as_float_array(scale, dtype)
        assert np.all(self._std >= 0), f"std must be non-negative. Got std {self._std}."
        assert np.all(self._scale >= 0), f"scale must be non-negative. Got scale {self._scale}."

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw Gaussian noise and squash the result to the range [-`scale`, `scale`]."""
        return self._scale * np.tanh(rng.normal(self._mean, self._std))


# region Helpers


def _add(nominal: FloatArray, noise: FloatArray) -> FloatArray:
    return nominal + noise


def _scale(nominal: FloatArray, noise: FloatArray) -> FloatArray:
    return nominal * noise


def _abs(_: FloatArray, noise: FloatArray) -> FloatArray:
    return noise


_OPERATIONS: dict[str, Operation] = {"add": _add, "scale": _scale, "abs": _abs}


def _as_float_array(value: ParamType, dtype: DTypeLike | None) -> FloatArray:
    """Convert a sampling parameter to a NumPy floating-point array."""
    if dtype is None:
        dtype = np.float64 if not _is_float_array(value) else value.dtype
    return np.asarray(value, dtype=dtype)


def _is_float_array(value: Any) -> bool:
    """Check if the input type is a NumPy floating-point array."""
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating)
