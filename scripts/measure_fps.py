import time
from typing import Callable

import fire
import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray

import contact_gym  # noqa: F401
from contact_gym.controllers import ALL_CONTROLLERS, CONTROLLER_TO_CLS
from contact_gym.envs import EdgeGraspEnvCfg

SceneCfg = EdgeGraspEnvCfg.SceneCfg

INTERNAL_ENV_IDS = [
    env_id.split("/")[-1]
    for env_id, spec in gym.registry.items()
    if isinstance(spec.entry_point, str) and spec.entry_point.startswith("contact_gym.")
]


def _build_action_fn(
    act_scale: float, controller: str | None
) -> Callable[[gym.Env, mujoco.MjData], NDArray]:
    if controller is None:
        return lambda env, data: data.ctrl + env.action_space.sample() * act_scale
    return lambda env, _: env.action_space.sample() * act_scale


def main(
    env_name: str,
    n_episodes: int = 10,
    act_scale: float = 0.0,
    seed: int = 0,
    controller: str | None = None,
    kwargs: dict | None = None,
    scene_kwargs: dict | None = None,
):
    """Measure steps/sec throughput of a registered MuJoCo gymnasium environment.

    Actions are sampled randomly at each step from the action space, with the magnitude
    of the random actions being determined the `act_scale` parameter.
    
    Args:
        env_name: Registered gymnasium environment ID (e.g., "EdgeGrasp-v0").
        n_episodes: Number of episodes to time. Default value is 10.
        act_scale: Scale factor for random actions. Default value is 0.
        seed: Base seed; episode `i` is seeded with `seed + i`. Default value is 0.
        controller: Optional controller to wrap environment with. Default value is None.
        kwargs: Optional extra kwargs forwarded to gym.make (e.g., '{"frame_skip": 20}').
        scene_kwargs: Optional kwargs for scene configuration (e.g.,'{"robot": "fr3"}' ). 

    Usage:
        python measure_fps.py --env_name "EdgeGrasp-v0"

        python measure_fps.py \
            --env_name "EdgeGrasp-v0" \
            --n_episodes 5 \
            --act_scale 0.05
            --seed 0 \
            --controller mocap \
            --kwargs '{"frame_skip": 20}'
            --scene_kwargs '{"robot": "fr3"}'
    """
    env_name = f"contact_gym/{env_name}" if env_name in INTERNAL_ENV_IDS else env_name
    kwargs = kwargs if kwargs else {}
    scene_kwargs = scene_kwargs if scene_kwargs else {}
    cfg = EdgeGraspEnvCfg(scene_cfg=SceneCfg(**scene_kwargs))
    env = gym.make(env_name, cfg=cfg, **kwargs)
    if controller is not None:
        assert controller in ALL_CONTROLLERS
        env = CONTROLLER_TO_CLS[controller](env)
    unwrapped = env.unwrapped
    assert hasattr(unwrapped, "model") and hasattr(unwrapped, "data"), (
        "Environment does not expose MuJoCo model and data structs."
    )
    env.reset(seed=seed)
    frame_skip = getattr(unwrapped, "frame_skip", 1)
    action_fn = _build_action_fn(act_scale, controller)

    print(f"Environment:    {env}")
    print(f"Action scale:   {act_scale:.4f}")
    print(f"Controller:     {controller}")
    print(f"Kwargs:         {kwargs}")
    print(f"Scene kwargs:   {scene_kwargs}")
    print(f"Frame skip:     {frame_skip}")
    print("-" * 64)

    episode_env_fps = []
    episode_sim_fps = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        elapsed, steps, done = 0, 0, False
        while not done:
            action = action_fn(env, unwrapped.data)
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
