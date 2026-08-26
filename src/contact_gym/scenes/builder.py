import mujoco
from mujoco import mjtConDataField, mjtObj

from ..objects import ALL_OBJECTS, OBJECT_PATHS
from ..robots import (
    ALL_GRIPPERS,
    ALL_ROBOTS,
    GRIPPER_HOME_KEYS,
    GRIPPER_PATHS,
    ROBOT_HOME_KEYS,
    SCENE_PATHS,
)
from . import templates


def build_edge_grasp(
    robot: str = "panda", gripper: str = "panda_hand", object: str = "block"
) -> mujoco.MjSpec:
    """Dynamically build the scene for the EdgeGrasp environment.

    Composes a scene from separate MJCF sources: a robot manipulator scene (arm + ground
    + light), a gripper, a table, and a graspable object are loaded independently and
    stitched together at runtime, so any supported gripper can be attached to any
    supported robot.

    Args:
        robot: Choice of robot manipulator, see :func:`ALL_ROBOTS` for options. Default
            value is panda.
        gripper: Choice of gripper, see :func:`ALL_GRIPPERS` for options. Default value
            is panda_hand.
        object: Choice of object to grasp, see :func:`ALL_OBJECTS` for options. Default
            value is block.

    Returns:
        Composed environment MjSpec.
    """
    assert robot in ALL_ROBOTS, f"Unsupported robot {robot}. Must be one of {ALL_ROBOTS}."
    assert gripper in ALL_GRIPPERS, f"Unsupported gripper {gripper}. Must be one of {ALL_GRIPPERS}."
    assert object in ALL_OBJECTS, f"Unsupported object {object}. Must be one of {ALL_OBJECTS}."
    # Load robot scene and attach gripper
    scene_spec = mujoco.MjSpec.from_file(SCENE_PATHS[robot])
    robot_bodyname = templates.get_articulation_parent_bodyname(scene_spec)
    gripper_spec = mujoco.MjSpec.from_file(GRIPPER_PATHS[gripper])
    gripper_bodyname = templates.get_articulation_parent_bodyname(gripper_spec)
    scene_spec = templates.add_body_at_site(
        scene_spec, gripper_spec, "attachment_site", gripper_bodyname, prefix="gripper-"
    )
    # Combine robot and gripper home keys
    home_key = scene_spec.key("home")
    assert home_key is not None, "Scene does not have a home key."
    robot_home = ROBOT_HOME_KEYS["EdgeGrasp"][robot]
    gripper_home = GRIPPER_HOME_KEYS["EdgeGrasp"][gripper]
    scene_home = {k: vr + vg for (k, vr), vg in zip(robot_home.items(), gripper_home.values())}
    home_key.qpos = scene_home["qpos"]
    home_key.ctrl = scene_home["ctrl"]
    # Add mocap body and inactive weld constraint
    scene_spec = templates.add_mocap_body(scene_spec, add_site=True, add_frame=True)
    scene_spec = templates.add_weld_equality(
        scene_spec, mjtObj.mjOBJ_SITE, "mocap", "gripper-tcp", active=False
    )
    # Load table and attach it
    TABLE_POS = (0, 0.5, 0.0)
    table_spec = mujoco.MjSpec.from_file(OBJECT_PATHS["table"])
    table_site = table_spec.site("topcenter")
    table_spec.delete(table_site)
    table_frame = scene_spec.worldbody.add_frame(pos=list(TABLE_POS))
    scene_spec.attach(table_spec, frame=table_frame, prefix="table-")
    # Load object and attach it
    object_pos = [t + s for t, s in zip(TABLE_POS, table_site.pos)]
    object_spec = mujoco.MjSpec.from_file(OBJECT_PATHS[object])
    object_bodyname = templates.get_articulation_parent_bodyname(object_spec)
    object_frame = scene_spec.worldbody.add_frame(pos=object_pos)
    scene_spec.attach(object_spec, frame=object_frame, prefix="object-")
    # Add frame and contact sensors
    scene_spec = templates.add_frame_sensors(
        scene_spec, mjtObj.mjOBJ_SITE, "gripper-tcp", sensors=["linvel", "angvel"]
    )
    scene_spec = templates.add_contact_sensor(
        scene_spec,
        mjtObj.mjOBJ_XBODY,
        f"gripper-{gripper_bodyname}",
        mjtObj.mjOBJ_BODY,
        f"object-{object_bodyname}",
        prefix="gripper_object",
    )
    scene_spec = templates.add_contact_sensor(
        scene_spec,
        mjtObj.mjOBJ_XBODY,
        robot_bodyname,
        prefix="robot",
        data=1 << mjtConDataField.mjCONDATA_FORCE.value,
        reduce=2,
    )
    return scene_spec
