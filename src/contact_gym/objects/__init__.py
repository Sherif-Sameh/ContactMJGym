"""Registry for objects."""

from pathlib import Path

# Supported objects
ALL_OBJECTS = ("block", "puck")

# MJCF (XML) paths
OBJECT_PATHS = {
    "block": str(Path(__file__).parent / "block/block.xml"),
    "puck": str(Path(__file__).parent / "puck/puck.xml"),
    "table": str(Path(__file__).parent / "table/table.xml"),
}
