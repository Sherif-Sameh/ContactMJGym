import time
from typing import Callable

import fire
import gymnasium as gym
import mujoco
import mujoco.viewer
from numpy.typing import NDArray

import contact_gym  # noqa: F401
import contact_gym.controllers
from contact_gym.controllers import ALL_CONTROLLERS

INTERNAL_ENV_IDS = [
    env_id.split("/")[-1]
    for env_id, spec in gym.registry.items()
    if isinstance(spec.entry_point, str) and spec.entry_point.startswith("contact_gym.")
]
CONTROLLER_REGISTRY = {
    k: getattr(contact_gym.controllers, f"{k.title()}ControllerAction") for k in ALL_CONTROLLERS
}


def _build_action_fn(
    act_scale: float, controller: str | None
) -> Callable[[gym.Env, mujoco.MjData], NDArray]:
    if controller is None:
        return lambda env, data: data.ctrl + env.action_space.sample() * act_scale
    return lambda env, _: env.action_space.sample() * act_scale


def main(
    env_name: str,
    act_scale: float = 3e-4,
    seed: int | None = None,
    controller: str | None = None,
    kwargs: dict | None = None,
):
    """Launch a live MuJoCo viewer window for a registered gymnasium env.

    Actions are sampled randomly at each step from the action space, with the magnitude
    of the random actions being determined the `act_scale` parameter.

    Args:
        env_name: Registered gymnasium environment ID (e.g., "EdgeGrasp-v0").
        act_scale: Scale factor for random actions. Default value is 3e-4.  
        seed: Optional seed for the environment. Default value is None.
        controller: Optional controller to wrap environment with. Default value is None.
        kwargs: Optional extra kwargs forwarded to gym.make (e.g., '{"frame_skip": 20}').

    Usage:
        python simple_agent.py --env_name "EdgeGrasp-v0"

        python simple_agent.py \
            --env_name "EdgeGrasp-v0" \
            --act_scale 0.05 \
            --seed 0 \
            --controller mocap \
            --kwargs '{"frame_skip": 20}'
    """
    env_name = f"contact_gym/{env_name}" if env_name in INTERNAL_ENV_IDS else env_name
    kwargs = kwargs if kwargs else {}
    env = gym.make(env_name, **kwargs)
    if controller is not None:
        assert controller in CONTROLLER_REGISTRY
        env = CONTROLLER_REGISTRY[controller](env)
    unwrapped = env.unwrapped
    assert hasattr(unwrapped, "model") and hasattr(unwrapped, "data"), (
        "Environment does not expose MuJoCo model and data structs."
    )
    env.reset(seed=seed)
    action_fn = _build_action_fn(act_scale, controller)
    frame_dt = 1 / unwrapped.metadata["render_fps"]

    with mujoco.viewer.launch_passive(
        unwrapped.model, unwrapped.data, show_right_ui=False
    ) as viewer:
        next_frame = time.perf_counter()
        while viewer.is_running():
            # Sample and apply action
            action = action_fn(env, unwrapped.data)
            _, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            if terminated or truncated:
                env.reset(seed=seed)
            # Rate limit loop
            next_frame += frame_dt
            sleep_time = next_frame - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame = time.perf_counter()
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
