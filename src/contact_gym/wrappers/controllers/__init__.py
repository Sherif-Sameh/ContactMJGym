"""Registry for controllers."""

from .mink import MinkControllerAction, MinkControllerCfg
from .mocap import MocapControllerAction, MocapControllerCfg
from .osc import OscControllerAction, OscControllerCfg

# Supported controllers
ALL_CONTROLLERS = ("mink", "mocap", "osc")
CONTROLLER_TO_CLS = {
    "mink": MinkControllerAction,
    "mocap": MocapControllerAction,
    "osc": OscControllerAction,
}
CONTROLLER_TO_CFG_CLS = {
    "mink": MinkControllerCfg,
    "mocap": MocapControllerCfg,
    "osc": OscControllerCfg,
}

__all__ = [
    "MinkControllerAction",
    "MinkControllerCfg",
    "MocapControllerAction",
    "MocapControllerCfg",
    "OscControllerAction",
    "OscControllerCfg",
]
