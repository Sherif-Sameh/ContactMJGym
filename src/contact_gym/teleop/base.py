from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias

from ..wrappers.controllers.task_space import TaskSpaceControllerAction

if TYPE_CHECKING:
    import gymnasium as gym
    import numpy as np
    from numpy.typing import NDArray

    FloatArray: TypeAlias = NDArray[np.floating]


class Teleop(ABC):
    """Base manipulator teleoperation abstract class."""

    def __init__(self, env: gym.Env):
        assert isinstance(env, TaskSpaceControllerAction), (
            f"Teleop assumes task-space control. Got env class {env.__class__.__name__}."
        )
        action_dim = env.action_space.shape[0]
        assert action_dim in (6, 7), (
            f"Action dim should be 6 (w/o gripper) or 7 (w/ gripper). Got {action_dim}."
        )
        self.has_gripper = action_dim == 7
        self.gripper_range = (
            (env.action_space.low[6], env.action_space.high[6]) if self.has_gripper else (0, 0)
        )
        self.dtype = env.action_space.dtype

    @abstractmethod
    def get_action(self) -> FloatArray:
        """Get the current action.

        Should be non-blocking, with heavy computation or blocking queries offloaded to
        additonal threads.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset teleop to a default initial state."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any reserved resources or additonal threads before shutdown."""
        pass
