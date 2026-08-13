import fire
import gymnasium as gym
import mujoco
import mujoco.viewer

import contact_gym  # noqa: F401

ENV_IDS = [
    env_id
    for env_id, spec in gym.registry.items()
    if isinstance(spec.entry_point, str) and spec.entry_point.startswith("contact_gym.")
]


def main(
    env: str,
    rand_act: bool = False,
    pert_scale: float = 3e-4,
    render_interval: int = 1,
    max_steps: int | None = None,
    seed: int | None = None,
    kwargs: dict | None = None,
):
    """Launch a live MuJoCo viewer window for a registered gymnasium env.

    Args:
        env: Registered gymnasium environment ID (e.g., "MujocoEdgeGrasp-v0").
        rand_act: Sample a random perturbation to the env's current ctrl at every step.
            Otherwise, apply the initial ctrl at every step. Default value is False.
        pert_scale: Scale factor for random action perturbations. Default value is 3e-4.  
        render_interval: Number of env steps to skip between viewer sync calls.
            Default value is 1.
        max_steps: Optional limit on the total number of steps to run before exiting.
            If `None`, runs until the viewer window is closed. Default value is `None`.
        seed: Optional seed for the environment. Default value is `None`.
        kwargs: Optional extra kwargs forwarded to gym.make (e.g., '{"frame_skip": 10}').

    Usage:
        python simple_agent.py --env "MujocoEdgeGrasp-v0"

        python simple_agent.py \
            --env "MujocoEdgeGrasp-v0" \
            --rand_act \
            --render_interval 3 \
            --max_steps 500 \
            --seed 0 \
            --kwargs '{"frame_skip": 20}'
    """
    env = f"contact_gym/{env}"
    assert env in ENV_IDS, f"Unsupported environment {env}. Must be one of {ENV_IDS}."
    kwargs = kwargs if kwargs else {}
    env = gym.make(env, **kwargs)
    unwrapped = env.unwrapped
    assert hasattr(unwrapped, "model") and hasattr(unwrapped, "data"), (
        "Environment does not expose MuJoCo model and data structs."
    )
    if seed is not None:
        env.action_space.seed(seed)
    env.reset(seed=seed)
    action = unwrapped.data.ctrl.copy()

    step_count = 0
    with mujoco.viewer.launch_passive(
        unwrapped.model, unwrapped.data, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            if max_steps is not None and step_count >= max_steps:
                break
            # Sample and apply action
            action = action + env.action_space.sample() * pert_scale if rand_act else action
            _, _, terminated, truncated, _ = env.step(action)
            if step_count % render_interval == 0:
                viewer.sync()
            if terminated or truncated:
                env.reset(seed=seed)
                action = unwrapped.data.ctrl.copy()
            step_count += 1
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
