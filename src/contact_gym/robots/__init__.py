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
