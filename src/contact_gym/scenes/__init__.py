"""Registry for scene builders."""

from .edge_grasp_builder import build_scene as build_edge_grasp_scene

# Supported scenes
ALL_SCENES = ("EdgeGrasp",)

# Scene builders
SCENE_BUILDERS = {"EdgeGrasp": build_edge_grasp_scene}

__all__ = ["build_edge_grasp_scene"]
