"""Registry for environments."""

from .edge_grasp import MujocoEdgeGraspEnv

# Supported environments
ALL_ENVS = (MujocoEdgeGraspEnv,)

__all__ = ["MujocoEdgeGraspEnv"]
