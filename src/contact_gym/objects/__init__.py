"""Registry for objects."""

from pathlib import Path

# Supported objects
ALL_OBJECTS = ("block", "puck", "table")

# MJCF (XML) paths
OBJECT_PATHS = {
    "block": Path(__file__).parent / "block/block.xml",
    "puck": Path(__file__).parent / "puck/puck.xml",
    "table": Path(__file__).parent / "table/table.xml",
}
