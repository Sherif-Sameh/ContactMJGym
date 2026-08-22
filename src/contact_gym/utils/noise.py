from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation as R

from .transform import add_to_quat

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
    the nominal value. The operation is resolved during construction. Supports vector,
    SO(3) and SE(3) operations. For SO(3) and SE(3) operations, orientation noise is
    assumed to be represented in the tangent space of the nominal orientation and is
    combined through the exponential map of SO(3).

    Args:
        sampler: Noise sampler, following the :class:`Sampler` protocol.
        operation: Operation to combine noise with nominal value. Must be one of
            ["add", "scale", "abs", "add_so3", "abs_so3", "add_se3", "abs_se3"].
            Default value is "add".
        scalar_first: Whether the scalar component goes first or last on quaternions.
            Relevant only for SO(3) and SE(3) operations. Default value is False.
    """

    def __init__(
        self,
        sampler: Sampler,
        operation: Literal[
            "add", "scale", "abs", "add_so3", "abs_so3", "add_se3", "abs_se3"
        ] = "add",
        *,
        scalar_first: bool = False,
    ) -> None:
        assert operation in _OPERATIONS, (
            f"Unsupported operation {operation}. Expected one of {tuple(_OPERATIONS)}"
        )
        self.sampler = sampler
        self.operation = (
            _OPERATIONS[operation]
            if operation in ["add", "scale", "abs"]
            else partial(_OPERATIONS[operation], scalar_first=scalar_first)
        )

    def sample(self, nominal: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Sample noise and apply the configured operation to the nominal value."""
        noise = self.sampler.sample(rng)
        return self.operation(nominal, noise)


# region Samplers


class Sampler(Protocol):
    """Protocol implemented by noise samplers."""

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a noise sample using the supplied random-number generator."""
        ...


class ConstantSampler:
    """Sampler that always returns the configured constant value.

    Args:
        value: Constant value returned by the `sample` method.
        dtype: Optional datatype for sampler parameters. If None, the default dtype is
            derived from the parameters if they're float arrays, otherwise it defaults
            to `np.float64`. Default value is None.
    """

    def __init__(self, value: ParamType, *, dtype: DTypeLike | None = None) -> None:
        assert dtype is None or np.issubdtype(dtype, np.floating)
        self.value = _as_float_array(value, dtype)

    def sample(self, _: np.random.Generator) -> FloatArray:
        """Return the configured constant, ignoring the random generator."""
        return self.value.copy()


class CategoricalSampler:
    """Sampler that draws values from a configured set of categories with replacement.

    Args:
        categories: Values to sample from. The first axis represents the categories and
            any remaining axes represent the shape of each sampled value.
        probabilities: Optional probabilities for each category. If given, its length
            must match the number of categories. Probabilities are normalized internally.
            If None, all categories are sampled with equal probability. Default value is
            None.
        dtype: Optional datatype for sampler parameters. If None, the default dtype is
            derived from the parameters if they're float arrays, otherwise it defaults
            to `np.float64`. Default value is None.
    """

    def __init__(
        self,
        categories: ParamType,
        probabilities: ParamType | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> None:
        assert dtype is None or np.issubdtype(dtype, np.floating)
        self.categories = _as_float_array(categories, dtype)
        assert self.categories.ndim > 0, "categories must have at least one axis"
        if probabilities is not None:
            self.probabilities = _as_float_array(probabilities, dtype)
            self.probabilities /= np.sum(self.probabilities)
            assert self.probabilities.shape == self.categories.shape[:1], (
                "Length of probabilities must match number of categories."
            )
            assert not np.any(self.probabilities < 0), "Probabilities must be non-negative."
        else:
            self.probabilities = None

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a categorical sample using the supplied random-number generator."""
        return rng.choice(self.categories, p=self.probabilities, axis=0)


class UniformSampler:
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
        self.min = _as_float_array(min, dtype)
        self.max = _as_float_array(max, dtype)
        assert np.all(self.min <= self.max), (
            f"min must be less than or equal to max. Got min {self.min} and max {self.max}."
        )

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a uniformly distributed sample."""
        return rng.uniform(self.min, self.max)


class GaussianSampler:
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
        self.mean = _as_float_array(mean, dtype)
        self.std = _as_float_array(std, dtype)
        assert np.all(self.std >= 0), f"std must be non-negative. Got std {self.std}."

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw a normally distributed sample."""
        return rng.normal(self.mean, self.std)


class SquashedGaussianSampler:
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
        self.mean = _as_float_array(mean, dtype)
        self.std = _as_float_array(std, dtype)
        self.scale = _as_float_array(scale, dtype)
        assert np.all(self.std >= 0), f"std must be non-negative. Got std {self.std}."
        assert np.all(self.scale >= 0), f"scale must be non-negative. Got scale {self.scale}."

    def sample(self, rng: np.random.Generator) -> FloatArray:
        """Draw Gaussian noise and squash the result to the range [-`scale`, `scale`]."""
        return self.scale * np.tanh(rng.normal(self.mean, self.std))


# region Helpers


def _add(nominal: FloatArray, noise: FloatArray) -> FloatArray:
    return nominal + noise


def _scale(nominal: FloatArray, noise: FloatArray) -> FloatArray:
    return nominal * noise


def _abs(_: FloatArray, noise: FloatArray) -> FloatArray:
    return noise


def _add_so3(nominal: FloatArray, noise: FloatArray, *, scalar_first: bool = False) -> FloatArray:
    return add_to_quat(nominal, noise, scalar_first=scalar_first)


def _abs_so3(_: FloatArray, noise: FloatArray, *, scalar_first: bool = False) -> FloatArray:
    return R.from_rotvec(noise).as_quat(scalar_first=scalar_first)


def _add_se3(nominal: FloatArray, noise: FloatArray, *, scalar_first: bool = False) -> FloatArray:
    out = np.empty_like(nominal)
    out[..., :3] = nominal[..., :3] + noise[..., :3]
    out[..., 3:] = _add_so3(nominal[..., 3:], noise[..., 3:], scalar_first=scalar_first)
    return out


def _abs_se3(nominal: FloatArray, noise: FloatArray, *, scalar_first: bool = False) -> FloatArray:
    out = np.empty_like(nominal)
    out[..., :3] = noise[..., :3]
    out[..., 3:] = _abs_so3(nominal, noise[..., 3:], scalar_first=scalar_first)
    return out


_OPERATIONS: dict[str, Operation] = {
    "add": _add,
    "scale": _scale,
    "abs": _abs,
    "add_so3": _add_so3,
    "abs_so3": _abs_so3,
    "add_se3": _add_se3,
    "abs_se3": _abs_se3,
}


def _as_float_array(value: ParamType, dtype: DTypeLike | None) -> FloatArray:
    """Convert a sampling parameter to a NumPy floating-point array."""
    if dtype is None:
        dtype = np.float64 if not _is_float_array(value) else value.dtype
    return np.asarray(value, dtype=dtype)


def _is_float_array(value: Any) -> bool:
    """Check if the input type is a NumPy floating-point array."""
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating)
