"""Registry for scene builders."""

from .builder import build_edge_grasp

# Supported scenes
ALL_SCENES = ("EdgeGrasp",)

# Scene builders
SCENE_BUILDERS = {"EdgeGrasp": build_edge_grasp}

__all__ = ["build_edge_grasp"]
