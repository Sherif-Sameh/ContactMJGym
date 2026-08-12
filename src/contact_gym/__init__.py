from gymnasium.envs.registration import register

register(
    id="contact_gym/MujocoEdgeGrasp-v0",
    entry_point="contact_gym.envs.edge_grasp:MujocoEdgeGraspEnv",
    max_episode_steps=1500,
)
