"""Registry for environments."""

from .edge_grasp import EdgeGraspRewardCfg, MujocoEdgeGraspEnv

# Supported environments
ALL_ENVS = (MujocoEdgeGraspEnv,)

__all__ = ["EdgeGraspRewardCfg", "MujocoEdgeGraspEnv"]
