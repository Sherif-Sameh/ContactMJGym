"""Registry for controllers."""

from .mink import MinkControllerAction, MinkControllerCfg
from .mocap import MocapControllerAction, MocapControllerCfg

# Supported controllers
ALL_CONTROLLERS = ("mink", "mocap")
CONTROLLER_TO_CLS = {"mink": MinkControllerAction, "mocap": MocapControllerAction}
CONTROLLER_TO_CFG_CLS = {"mink": MinkControllerCfg, "mocap": MocapControllerCfg}

__all__ = [
    "MinkControllerAction",
    "MinkControllerCfg",
    "MocapControllerAction",
    "MocapControllerCfg",
]
