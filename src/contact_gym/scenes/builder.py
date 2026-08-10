import mujoco

from contact_gym.objects import ALL_OBJECTS, OBJECT_PATHS
from contact_gym.robots import ALL_ROBOTS, HOME_KEYS, ROBOT_PATHS


def build_edge_grasp(robot: str = "panda", object: str = "block") -> mujoco.MjSpec:
    """Dynamically build the scene for the EdgeGrasp environment.

    Loads the robot + gripper + common elements (ground, light) spec, sets robot home keyframe, and
    adds object on top of the table.

    Args:
        robot: Choice of robot manipulator. Default is panda.
        object: Choice of object to grasp. Default is block.

    Returns:
        Composed environment MjSpec.
    """
    assert robot in ALL_ROBOTS, (
        f"Unsupported robot {robot}. Must be one of {ALL_ROBOTS}."
    )
    assert object in ALL_OBJECTS, (
        f"Unsupported object {object}. Must be one of {ALL_OBJECTS}."
    )
    # Load robot + gripper scene
    scene_spec = mujoco.MjSpec.from_file(ROBOT_PATHS[robot])
    home_key = scene_spec.key("home")
    assert home_key is not None, "Scene does not have a home key."
    home_key.qpos = HOME_KEYS["EdgeGrasp"][robot]["qpos"]
    home_key.ctrl = HOME_KEYS["EdgeGrasp"][robot]["ctrl"]
    # Load table and attach it
    TABLE_POS = (0, 0.5, 0.0)
    table_spec = mujoco.MjSpec.from_file(OBJECT_PATHS["table"])
    site_pos = table_spec.site("tabletop_center").pos
    table_frame = scene_spec.worldbody.add_frame(pos=list(TABLE_POS))
    scene_spec.attach(table_spec, frame=table_frame, prefix="table-")
    # Load object and attach it
    object_pos = [t + s for t, s in zip(TABLE_POS, site_pos)]
    object_spec = mujoco.MjSpec.from_file(OBJECT_PATHS[object])
    object_frame = scene_spec.worldbody.add_frame(pos=object_pos)
    scene_spec.attach(object_spec, frame=object_frame, prefix="object-")
    return scene_spec
