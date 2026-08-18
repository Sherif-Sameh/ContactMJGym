from gymnasium.envs.registration import register

register(
    id="contact_gym/EdgeGrasp-v0",
    entry_point="contact_gym.envs.edge_grasp:EdgeGraspEnv",
    max_episode_steps=1500,
)
