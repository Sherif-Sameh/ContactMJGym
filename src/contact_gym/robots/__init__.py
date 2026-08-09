"""Registry for robots and grippers."""

from pathlib import Path

# Supported robots
ALL_ROBOTS = ("fr3", "panda", "ur10e")

# Supported grippers
ALL_GRIPPERS = ("panda", "robotiq")

# MJCF (XML) paths
ROBOT_PATHS = {
    "fr3": Path(__file__).parent / "franka_fr3/fr3.xml",
    "panda": Path(__file__).parent / "franka_emika_panda/panda_nohand.xml",
    "ur10e": Path(__file__).parent / "universal_robots_ur10e/ur10e.xml",
}
GRIPPER_PATHS = {
    "panda": Path(__file__).parent / "franka_emika_panda/hand.xml",
    "robotiq": Path(__file__).parent / "robotiq_2f85/2f85.xml",
}
