"""Registry for controllers."""

from .mink import MinkCfg, MinkControllerAction
from .mocap import MocapControllerAction

# Supported controllers
ALL_CONTROLLERS = ("mocap", "mink")

__all__ = ["MinkCfg", "MinkControllerAction", "MocapControllerAction"]
