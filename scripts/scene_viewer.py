import fire
import mujoco
import mujoco.viewer

from contact_gym.scenes import ALL_SCENES, SCENE_BUILDERS


def main(scene: str = "EdgeGrasp", robot: str = "panda", object: str = "block") -> None:
    """Build a scene and open it in the interactive MuJoCo viewer.

    Usage:
        python scene_viewer.py --scene EdgeGrasp --robot panda --object block
    """
    assert scene in ALL_SCENES, f"Unsupported scene {scene}. Must be one of {ALL_SCENES}."
    spec = SCENE_BUILDERS[scene](robot=robot, object=object)
    model = spec.compile()
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object-joint")
    object_qpos_adr = model.jnt_qposadr[object_joint_id]
    if key_id != -1:
        data.qpos[:object_qpos_adr] = model.key_qpos[key_id, :object_qpos_adr]
        data.ctrl[:object_qpos_adr] = model.key_ctrl[key_id, :object_qpos_adr]
    mujoco.mj_forward(model, data)
    print(f"Loaded scene: scene={scene}, robot={robot}, object={object}")
    print(f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    fire.Fire(main)
