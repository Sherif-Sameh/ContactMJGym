"""Registry for environments."""

from .edge_grasp import EdgeGraspEnv, EdgeGraspEnvCfg

# Supported environments
ALL_ENVS = (EdgeGraspEnv,)

__all__ = ["EdgeGraspEnvCfg", "EdgeGraspEnv"]
