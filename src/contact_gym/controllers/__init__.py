"""Registry for controllers."""

from .mink import MinkCfg, MinkControllerAction
from .mocap import MocapControllerAction
from .task_space import TaskSpaceControllerCfg

# Supported controllers
ALL_CONTROLLERS = ("mocap", "mink")

__all__ = ["MinkCfg", "MinkControllerAction", "MocapControllerAction", "TaskSpaceControllerCfg"]
