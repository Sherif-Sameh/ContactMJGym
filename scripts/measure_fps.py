import time

import fire
import gymnasium as gym
import numpy as np

import contact_gym  # noqa: F401

INTERNAL_ENV_IDS = [
    env_id.split("/")[-1]
    for env_id, spec in gym.registry.items()
    if isinstance(spec.entry_point, str) and spec.entry_point.startswith("contact_gym.")
]


def main(env_name: str, n_episodes: int = 5, seed: int = 0, kwargs: dict | None = None):
    """Measure steps/sec throughput of a registered MuJoCo gymnasium environment.

    The environment is run with a fixed action (the initial `data.ctrl` after reset)
    until the episode terminates/truncates, whichever comes first.

    Args:
        env_name: Registered gymnasium environment ID (e.g., "MujocoEdgeGrasp-v0").
        n_episodes: Number of episodes to time. Default value is 5.
        seed: Base seed; episode `i` is seeded with `seed + i`. Default value is 1.
        kwargs: Optional extra kwargs forwarded to gym.make (e.g., '{"frame_skip": 10}').

    Usage:
        python measure_fps.py --env_name "MujocoEdgeGrasp-v0"

        python measure_fps.py \
            --env_name "MujocoEdgeGrasp-v0" \
            --n_episodes 4 \
            --seed 0 \
            --kwargs '{"frame_skip": 20}'
    """
    env_name = f"contact_gym/{env_name}" if env_name in INTERNAL_ENV_IDS else env_name
    kwargs = kwargs if kwargs else {}
    env = gym.make(env_name, **kwargs)
    unwrapped = env.unwrapped
    assert hasattr(unwrapped, "model") and hasattr(unwrapped, "data"), (
        "Environment does not expose MuJoCo model and data structs."
    )
    env.reset(seed=seed)
    frame_skip = getattr(unwrapped, "frame_skip", 1)
    action = unwrapped.data.ctrl.copy()

    print(f"Environment:    {env}")
    print(f"Kwargs:         {kwargs}")
    print(f"Frame skip:     {frame_skip}")
    print("-" * 64)

    episode_env_fps = []
    episode_sim_fps = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        elapsed, steps, done = 0, 0, False
        while not done:
            start = time.perf_counter_ns()
            _, _, terminated, truncated, _ = env.step(action)
            elapsed += time.perf_counter_ns() - start
            steps += 1
            done = terminated or truncated

        elapsed *= 1e-9
        env_fps = steps / elapsed if elapsed > 0 else float("inf")
        sim_fps = env_fps * frame_skip

        episode_env_fps.append(env_fps)
        episode_sim_fps.append(sim_fps)

        print(
            f"Episode {ep + 1}/{n_episodes}: "
            f"{steps:>5} steps in {elapsed:7.3f}s | "
            f"env FPS: {env_fps:9.1f} (steps/sec) | "
            f"sim FPS: {sim_fps:9.1f} (steps/sec)"
        )

    env.close()

    print("-" * 64)
    print(f"Average env FPS over {n_episodes} episodes: {np.mean(episode_env_fps):.1f} (steps/sec)")
    print(f"Average sim FPS over {n_episodes} episodes: {np.mean(episode_sim_fps):.1f} (steps/sec)")


if __name__ == "__main__":
    fire.Fire(main)
