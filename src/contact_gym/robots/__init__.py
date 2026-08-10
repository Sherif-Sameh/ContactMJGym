"""Registry for robots."""

from pathlib import Path

# Supported robots
ALL_ROBOTS = ("fr3", "panda", "ur10e")

# MJCF (XML) paths
ROBOT_PATHS = {
    "fr3": str(Path(__file__).parent / "franka_fr3/scene.xml"),
    "panda": str(Path(__file__).parent / "franka_emika_panda/scene.xml"),
    "ur10e": str(Path(__file__).parent / "universal_robots_ur10e/scene.xml"),
}

# Per-env per-robot home keyframes
HOME_KEYS = {
    "EdgeGrasp": {
        "fr3": {
            "qpos": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.0, 0.04, 0.04),
            "ctrl": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.0, 255.0),
        },
        "panda": {
            "qpos": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.7853, 0.04, 0.04),
            "ctrl": (1.57079, -0.178, 0.0, -1.74, 0.0, 1.57079, 0.7853, 255.0),
        },
        "ur10e": {
            "qpos": (-1.95, -1.95, 1.92, -1.5708, -1.5708, 1.19, 0.04, 0.04),
            "ctrl": (-1.95, -1.95, 1.92, -1.5708, -1.5708, 1.19, 255.0),
        },
    }
}
