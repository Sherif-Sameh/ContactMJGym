import logging

import fire
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import contact_gym  # noqa: F401
import contact_gym.dr as dr
from contact_gym.curriculum.fixed import LinearCurriculumTerm
from contact_gym.utils.noise import GaussianSampler, Noise

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s\n%(message)s\n")


def get_randomizers(pos_std: float, yaw_std: float) -> list[dr.DomainRandomizer]:
    jnt_qpos_rand = dr.DataStateRandomizer(
        dr.state.qpos_state_cfg(
            Noise(GaussianSampler(std=0.0), operation="add"), entry_sel=slice(7)
        )
    )
    obj_qpos_rand = dr.DataStateRandomizer(
        dr.state.qpos_state_cfg(
            Noise(
                GaussianSampler(std=[pos_std, pos_std, 0, 0, 0, yaw_std]),
                operation="add_se3",
                scalar_first=True,
            ),
            entry_sel=slice(9, None),
        )
    )
    return [jnt_qpos_rand, obj_qpos_rand]


def get_curriculum(jnt_std: float) -> LinearCurriculumTerm:
    return LinearCurriculumTerm(
        paths=[
            "domain_randomizers[0].cfg.noise.sampler.std",
            "domain_randomizers[1].cfg.noise.sampler.std",
        ],
        end=[jnt_std, np.zeros(6)],
        end_step=5_000,
    )


# region Main


def main(
    jnt_std: float = 0.2,
    pos_std: float = 0.2,
    yaw_std: float = 0.5,
    object: str = "block",
    seed: int | None = None,
):
    """Launch a live MuJoCo viewer window for a registered gymnasium env with domain randomization.

    Args:
        jnt_std: Final standard deviation for initial joint position randomization.
            Default value is 0.2.
        pos_std: Initial standard deviation for initial object position randomization in XY-plane.
            Default value is 0.2.
        yaw_std: Initial standard deviation for initial object yaw angle randomization.
            Default value is 0.5.
        object: Object type for environment. One of ["block", "puck"]. Default value is block.
        seed: Optional seed for the environment. Default value is None.

    Usage:
        python sample_curriculum.py

        python sample_curriculum.py \
            --jnt_std 0.2 \
            --pos_std 0.1 \
            --yaw_std 0.6 \
            --object puck
            --seed 0
    """
    env_name = "contact_gym/EdgeGrasp-v0"
    env = gym.make(
        env_name,
        domain_randomizers=get_randomizers(pos_std, yaw_std),
        curriculum_terms=[get_curriculum(jnt_std)],
        object=object,
        max_episode_steps=500,
    )
    unwrapped = env.unwrapped
    env.reset(seed=seed)
    action = unwrapped.data.ctrl.copy()
    action[:7] = unwrapped.data.qpos[:7]

    with mujoco.viewer.launch_passive(
        unwrapped.model, unwrapped.data, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            # Sample and apply action
            _, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            if terminated or truncated:
                env.reset()
                action[:7] = unwrapped.data.qpos[:7]
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
