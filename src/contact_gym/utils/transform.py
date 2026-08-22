from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

# TODO: Is SciPy efficient enough for this?


def add_to_quat(quat: NDArray, rvec: NDArray, *, scalar_first: bool = False) -> NDArray:
    """Add a rotation vector to the current orientation expressed as a unit quaternion.

    Args:
        quat: Current orientation. Shape is (4,) or (N, 4).
        rvec: Rotation vector to apply to current orientation. Shape is (3,) or (N, 3).
        scalar_first: Whether the scalar component goes first or last. Default value is False.

    Returns:
        Output orientation after applying rotation offset to input orientation. Shape is
        the same as the input `quat`.
    """
    rot = R.from_quat(quat, scalar_first=scalar_first)
    rot *= R.from_rotvec(rvec)
    return rot.as_quat(scalar_first=scalar_first)


def add_to_rmat(rmat: NDArray, rvec: NDArray, *, assume_valid: bool = False) -> NDArray:
    """Add a rotation vector to the current orientation expressed as a rotation matrix.

    Args:
        quat: Current orientation. Shape is (3, 3) or (N, 3, 3).
        rvec: Rotation vector to apply to current orientation. Shape is (3,) or (N, 3).
        assume_valid: If True, normalization steps are skipped. Default value is False.

    Returns:
        Output orientation after applying rotation offset to input orientation. Shape is
        the same as the input `rmat`.
    """
    rot = R.from_matrix(rmat, assume_valid=assume_valid)
    rot *= R.from_rotvec(rvec)
    return rot.as_matrix()
