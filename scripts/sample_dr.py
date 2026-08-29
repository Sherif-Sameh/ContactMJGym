import fire
import gymnasium as gym
import mujoco
import numpy as np

import contact_gym  # noqa: F401
import contact_gym.dr as dr
from contact_gym.dr import DataStateRandomizer, DomainRandomizer, ModelParamRandomizer
from contact_gym.envs import EdgeGraspEnvCfg
from contact_gym.utils.noise import CategoricalSampler, GaussianSampler, Noise, UniformSampler

SceneCfg = EdgeGraspEnvCfg.SceneCfg


def get_actuator_randomizers() -> list[DomainRandomizer]:
    noise = Noise(UniformSampler(0.9, 1.1), operation="scale")
    kp_rand = ModelParamRandomizer(dr.actuator.actuator_position_kp_cfg(noise, inst_sel=slice(7)))
    neg_kv_rand = ModelParamRandomizer(
        dr.actuator.actuator_position_neg_kv_cfg(noise, inst_sel=slice(7))
    )
    return [kp_rand, neg_kv_rand]


def get_geom_randomizers(model: mujoco.MjModel) -> list[DomainRandomizer]:
    obj_geom_id = model.geom("object-geom").id
    colors = np.array(
        [
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
        ]
    )
    rgb_rand = ModelParamRandomizer(
        dr.geom.geom_rgb_cfg(
            Noise(CategoricalSampler(colors), operation="abs"), inst_sel=obj_geom_id
        )
    )
    return [rgb_rand]


def get_joint_randomizers() -> list[DomainRandomizer]:
    armature_rand = ModelParamRandomizer(
        dr.joint.joint_armature_cfg(
            Noise(UniformSampler(0.9, 1.1), operation="scale"), inst_sel=slice(9)
        )
    )
    frictionloss_rand = ModelParamRandomizer(
        dr.joint.joint_frictionloss_cfg(
            Noise(GaussianSampler(std=0.1), operation="add"), inst_sel=slice(9)
        )
    )
    return [armature_rand, frictionloss_rand]


def get_state_randomizers(jnt_std: float, pos_std: float, yaw_std: float) -> list[DomainRandomizer]:
    jnt_qpos_rand = DataStateRandomizer(
        dr.state.qpos_state_cfg(
            Noise(GaussianSampler(std=jnt_std), operation="add"), entry_sel=slice(7)
        )
    )
    obj_qpos_rand = DataStateRandomizer(
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


# region Main


def main(
    jnt_std: float = 0.1,
    pos_std: float = 0.1,
    yaw_std: float = 0.5,
    object: str = "block",
    seed: int | None = None,
):
    """Launch a live MuJoCo viewer window for a registered gymnasium env with domain randomization.

    Args:
        jnt_std: Standard deviation for initial joint position randomization.
            Default value is 0.1.
        pos_std: Standard deviation for initial object position randomization in XY-plane.
            Default value is 0.1.
        yaw_std: Standard deviation for initial object yaw angle randomization.
            Default value is 0.5.
        object: Object type for environment. One of ["block", "puck"]. Default value is block.
        seed: Optional seed for the environment. Default value is None.

    Usage:
        python sample_dr.py

        python sample_dr.py \
            --jnt_std 0.2 \
            --pos_std 0.1 \
            --yaw_std 0.6 \
            --object puck
            --seed 0
    """
    env_name = "contact_gym/EdgeGrasp-v0"
    cfg = EdgeGraspEnvCfg(scene_cfg=SceneCfg(object=object))
    env = gym.make(
        env_name,
        cfg=cfg,
        render_mode="human",
        domain_randomizers=get_actuator_randomizers()
        + get_joint_randomizers()
        + get_state_randomizers(jnt_std, pos_std, yaw_std),
    )
    unwrapped = env.unwrapped
    assert hasattr(unwrapped, "viewer_is_running"), (
        "Environment does not have a viewer_is_running property."
    )
    unwrapped.domain_randomizers.extend(get_geom_randomizers(unwrapped.model))
    env.reset(seed=seed)
    action = unwrapped.data.ctrl.copy()
    action[:7] = unwrapped.data.qpos[:7]

    while unwrapped.viewer_is_running:
        # Sample and apply action
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            action[:7] = unwrapped.data.qpos[:7]
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
