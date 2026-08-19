"""Registry for robots and grippers."""

from pathlib import Path

# Supported robots and grippers
ALL_ROBOTS = ("fr3", "panda", "ur10e")
ALL_GRIPPERS = ("panda_hand",)

# MJCF (XML) paths
SCENE_PATHS = {
    "fr3": str(Path(__file__).parent / "franka_fr3/scene.xml"),
    "panda": str(Path(__file__).parent / "franka_emika_panda/scene.xml"),
    "ur10e": str(Path(__file__).parent / "universal_robots_ur10e/scene.xml"),
}
GRIPPER_PATHS = {"panda_hand": str(Path(__file__).parent / "franka_emika_panda/hand.xml")}

# Per-env per-robot and gripper home keyframes
ROBOT_HOME_KEYS = {
    "EdgeGrasp": {
        "fr3": {
            "qpos": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.0),
            "ctrl": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.0),
        },
        "panda": {
            "qpos": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.7853),
            "ctrl": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.7853),
        },
        "ur10e": {
            "qpos": (-1.95, -1.95, 1.92, -1.5708, -1.5708, 1.19),
            "ctrl": (-1.95, -1.95, 1.92, -1.5708, -1.5708, 1.19),
        },
    }
}
GRIPPER_HOME_KEYS = {"EdgeGrasp": {"panda_hand": {"qpos": (0.04, 0.04), "ctrl": (255.0,)}}}


# Helpers for getting qpos and ctrl dimensions
def get_qpos_dim(name: str) -> int:
    """Retrieve the `qpos` dimenension of the given asset."""
    env = next(iter(ROBOT_HOME_KEYS.keys()))
    if name in ALL_ROBOTS:
        return len(ROBOT_HOME_KEYS[env][name]["qpos"])
    assert name in ALL_GRIPPERS, (
        f"Unrecognized name {name}. Must be one of {ALL_ROBOTS + ALL_GRIPPERS}."
    )
    return len(GRIPPER_HOME_KEYS[env][name]["qpos"])


def get_ctrl_dim(name: str) -> int:
    """Retrieve the `ctrl` dimenension of the given asset."""
    env = next(iter(ROBOT_HOME_KEYS.keys()))
    if name in ALL_ROBOTS:
        return len(ROBOT_HOME_KEYS[env][name]["ctrl"])
    assert name in ALL_GRIPPERS, (
        f"Unrecognized name {name}. Must be one of {ALL_ROBOTS + ALL_GRIPPERS}."
    )
    return len(GRIPPER_HOME_KEYS[env][name]["ctrl"])
