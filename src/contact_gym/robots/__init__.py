"""Registry for robots."""

from pathlib import Path

# Supported robots
ALL_ROBOTS = ("fr3", "ur10e")

# MJCF (XML) paths
ROBOT_PATHS = {
    "fr3": Path(__file__).parent / "franka_fr3/fr3.xml",
    "ur10e": Path(__file__).parent / "universal_robots_ur10e/ur10e.xml",
}
