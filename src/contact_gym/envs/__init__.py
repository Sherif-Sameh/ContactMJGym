"""Registry for environments."""

from .edge_grasp import EdgeGraspEnv, EdgeGraspRewardCfg

# Supported environments
ALL_ENVS = (EdgeGraspEnv,)

__all__ = ["EdgeGraspRewardCfg", "EdgeGraspEnv"]
