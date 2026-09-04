import time

import fire
import gymnasium as gym

import contact_gym  # noqa: F401
from contact_gym.controllers import ALL_CONTROLLERS, CONTROLLER_TO_CFG_CLS, CONTROLLER_TO_CLS
from contact_gym.controllers.task_space import TaskSpaceControllerCfg
from contact_gym.envs import EdgeGraspEnvCfg
from contact_gym.teleop import Keyboard

SceneCfg = EdgeGraspEnvCfg.SceneCfg
CompensationCfg = TaskSpaceControllerCfg.CompensationCfg

INTERNAL_ENV_IDS = [
    env_id.split("/")[-1]
    for env_id, spec in gym.registry.items()
    if isinstance(spec.entry_point, str) and spec.entry_point.startswith("contact_gym.")
]


def main(
    env_name: str,
    gripper_step: float = 0.2,
    speed_init: float = 0.3,
    speed_step: float = 0.1,
    controller: str = "mocap",
    seed: int | None = None,
    kwargs: dict | None = None,
    scene_kwargs: dict | None = None,
):
    """Launch a live MuJoCo viewer window for a registered gymnasium env with keyboard teleop.

    Args:
        env_name: Registered gymnasium environment ID (e.g., "EdgeGrasp-v0").
        gripper_step: Gripper position step per env `step()` call. Default value is 0.2.
        speed_init: Initial value for speed scale factor. Default value is 0.5.
        speed_step: Speed step size for each "speed_up" or "speed_down" key press.
            Default value is 0.1.
        controller: Controller to wrap environment with. Default value is mocap.
        seed: Optional seed for the environment. Default value is None.
        kwargs: Optional extra kwargs forwarded to gym.make (e.g., '{"frame_skip": 20}').
        scene_kwargs: Optional kwargs for scene configuration (e.g.,'{"robot": "fr3"}' ). 

    Usage:
        python simple_agent.py --env_name "EdgeGrasp-v0"

        python simple_agent.py \
            --env_name "EdgeGrasp-v0" \
            --act_scale 0.05 \
            --seed 0 \
            --controller mocap \
            --kwargs '{"frame_skip": 20}' \
            --scene_kwargs '{"robot": "fr3"}'
    """
    assert env_name in INTERNAL_ENV_IDS
    env_name = f"contact_gym/{env_name}"
    kwargs = kwargs if kwargs else {}
    scene_kwargs = scene_kwargs if scene_kwargs else {}
    cfg = EdgeGraspEnvCfg(scene_cfg=SceneCfg(**scene_kwargs))
    env = gym.make(env_name, cfg=cfg, render_mode="human", **kwargs)
    assert controller in ALL_CONTROLLERS
    controller_cfg = CONTROLLER_TO_CFG_CLS[controller](comp_cfg=CompensationCfg(bias_mult=1))
    env: gym.Env = CONTROLLER_TO_CLS[controller](env, controller_cfg)
    unwrapped = env.unwrapped
    assert hasattr(unwrapped, "viewer_is_running"), (
        "Environment does not have a viewer_is_running property."
    )
    env.reset(seed=seed)
    teleop = Keyboard(env, gripper_step=gripper_step, speed_init=speed_init, speed_step=speed_step)
    teleop.print_controls()

    last_status = ""
    frame_dt = 1 / unwrapped.metadata["render_fps"]
    next_frame = time.perf_counter()
    while unwrapped.viewer_is_running:
        # Get and apply action
        action = teleop.get_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            teleop.reset()
        # Print updated teleop status
        status = teleop.get_status_str()
        if status != last_status:
            print("\r" + status + " " * 10, end="", flush=True)
            last_status = status
        # Rate limit loop
        next_frame += frame_dt
        sleep_time = next_frame - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_frame = time.perf_counter()
    env.close()
    teleop.close()


if __name__ == "__main__":
    fire.Fire(main)
