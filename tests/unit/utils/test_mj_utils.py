import mujoco
import pytest

from contact_gym.utils.mj_utils import (
    disable_actuators,
    filter_actuators,
    filter_joints,
    get_dof_dim_from_joints,
)

_DOF_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="free_body">
      <joint type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="ball_body" pos="1 0 0">
      <joint type="ball"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="hinge_body1" pos="2 0 0">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="hinge_body2" pos="3 0 0">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def dof_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_DOF_MODEL_XML)


# region get_dof_dim


@pytest.mark.unit
def test_get_dof_dim_from_joints_valid_inputs(dof_model):
    model = dof_model

    # Sanity check model
    assert model.nq == 13
    assert model.nv == 11

    # Full model -> dof dim equals nv
    assert get_dof_dim_from_joints(model, qpos_adr=0, qpos_dim=model.nq) == model.nv
    # qpos[0:7) -> free joint only -> dof dim 6
    assert get_dof_dim_from_joints(model, qpos_adr=0, qpos_dim=7) == 6
    # qpos[7:11) -> ball joint only -> dof dim 3
    assert get_dof_dim_from_joints(model, qpos_adr=7, qpos_dim=4) == 3
    # qpos[11:13) -> two hinge joints -> dof dim 2
    assert get_dof_dim_from_joints(model, qpos_adr=11, qpos_dim=2) == 2
    # qpos[11:12) -> first hinge joint only -> dof dim 1
    assert get_dof_dim_from_joints(model, qpos_adr=11, qpos_dim=1) == 1
    # qpos[0:11) -> free + ball joints -> dof dim 6 + 3 = 9
    assert get_dof_dim_from_joints(model, qpos_adr=0, qpos_dim=11) == 9


@pytest.mark.unit
def test_get_dof_dim_from_joints_invalid_inputs(dof_model):
    model = dof_model

    # qpos_adr negative
    with pytest.raises(AssertionError):
        get_dof_dim_from_joints(model, qpos_adr=-1, qpos_dim=1)
    # qpos_adr == nq (out of bounds)
    with pytest.raises(AssertionError):
        get_dof_dim_from_joints(model, qpos_adr=model.nq, qpos_dim=1)
    # qpos_adr well beyond nq
    with pytest.raises(AssertionError):
        get_dof_dim_from_joints(model, qpos_adr=model.nq + 5, qpos_dim=1)
    # qpos_adr in bounds but qpos_adr + qpos_dim - 1 overruns nq
    with pytest.raises(AssertionError):
        get_dof_dim_from_joints(model, qpos_adr=model.nq - 1, qpos_dim=5)
    # qpos_dim zero
    with pytest.raises(AssertionError):
        get_dof_dim_from_joints(model, qpos_adr=0, qpos_dim=0)
    # qpos_dim negative
    with pytest.raises(AssertionError):
        get_dof_dim_from_joints(model, qpos_adr=0, qpos_dim=-2)


# region filter_joints

_JOINT_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="shoulder_body">
      <joint name="arm_shoulder_joint" type="hinge" axis="0 0 1" group="0"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="elbow_body" pos="1 0 0">
      <joint name="arm_elbow_joint" type="hinge" axis="0 0 1" group="1"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="wrist_body" pos="2 0 0">
      <joint name="arm_wrist_joint" type="hinge" axis="0 0 1" group="1"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="gripper_body" pos="3 0 0">
      <joint name="gripper_slide_joint" type="slide" axis="1 0 0" group="2"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def joint_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_JOINT_MODEL_XML)


@pytest.mark.unit
def test_filter_joints_valid_inputs(joint_model):
    model = joint_model

    # Filter by exact names
    assert filter_joints(
        model, names=["arm_elbow_joint", "gripper_slide_joint"], flags=(True,)
    ) == [1, 3]
    # Filter by common substring shared by a subset of joints
    assert filter_joints(model, substring="arm_", flags=(True,)) == [0, 1, 2]
    # Filter by substring shared by all joints
    assert filter_joints(model, substring="_joint", flags=(True,)) == [0, 1, 2, 3]
    # Filter by joint type
    assert filter_joints(model, type=mujoco.mjtJoint.mjJNT_HINGE, flags=(True,)) == [0, 1, 2]
    assert filter_joints(model, type=mujoco.mjtJoint.mjJNT_SLIDE, flags=(True,)) == [3]
    # Filter by group
    assert filter_joints(model, group=1, flags=(True,)) == [1, 2]
    # Combine multiple filters (all flags True -> AND semantics)
    assert filter_joints(
        model, substring="arm_", type=mujoco.mjtJoint.mjJNT_HINGE, flags=(True, True)
    ) == [0, 1, 2]
    # A False flag inverts that filter's condition (exclude rather than include)
    assert filter_joints(model, type=mujoco.mjtJoint.mjJNT_HINGE, flags=(False,)) == [3]
    assert filter_joints(model, substring="arm_", group=1, flags=(True, False)) == [0]


@pytest.mark.unit
def test_filter_joints_invalid_inputs(joint_model):
    model = joint_model

    # No filters defined at all
    with pytest.raises(AssertionError):
        filter_joints(model)
    # No flags defined at all
    with pytest.raises(AssertionError):
        filter_joints(model, group=0)
    # Wrong number of flags (must match the filters)
    with pytest.raises(AssertionError):
        filter_joints(model, names=["arm_elbow_joint"], flags=(True, True))


# region filter_actuators

_ACTUATOR_MODEL_XML = (
    _JOINT_MODEL_XML.replace(
        '<joint name="arm_wrist_joint" type="hinge" axis="0 0 1" group="1"/>',
        '<joint name="arm_wrist_joint" type="hinge" axis="0 0 1" group="1"/>\n'
        '      <site name="wrist_site" pos="0 0 0"/>',
    ).replace("</mujoco>", "")
    + """
  <actuator>
    <motor name="arm_shoulder_motor" joint="arm_shoulder_joint" group="0"/>
    <motor name="arm_elbow_motor" joint="arm_elbow_joint" group="1"/>
    <position name="arm_wrist_position" joint="arm_wrist_joint" group="1"/>
    <motor name="gripper_slide_motor" joint="gripper_slide_joint" group="2"/>
    <motor name="wrist_site_motor" site="wrist_site" group="3"/>
  </actuator>
</mujoco>
"""
)


@pytest.fixture
def actuator_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_ACTUATOR_MODEL_XML)


@pytest.mark.unit
def test_filter_actuators_valid_inputs(actuator_model):
    model = actuator_model
    # Filter by exact names
    assert filter_actuators(
        model, names=["arm_elbow_motor", "gripper_slide_motor"], flags=(True,)
    ) == [1, 3]
    # Filter by common substring shared by a subset of actuators
    assert filter_actuators(model, substring="arm_", flags=(True,)) == [0, 1, 2]
    assert filter_actuators(model, substring="_motor", flags=(True,)) == [0, 1, 3, 4]
    # Filter by transmission type
    assert filter_actuators(model, trntype=mujoco.mjtTrn.mjTRN_JOINT, flags=(True,)) == [0, 1, 2, 3]
    assert filter_actuators(model, trntype=mujoco.mjtTrn.mjTRN_SITE, flags=(True,)) == [4]
    # Filter by group
    assert filter_actuators(model, group=1, flags=(True,)) == [1, 2]
    # Combine multiple filters (all flags True -> AND semantics)
    assert filter_actuators(
        model, substring="arm_", trntype=mujoco.mjtTrn.mjTRN_JOINT, flags=(True, True)
    ) == [0, 1, 2]
    # A False flag inverts that filter's condition (exclude rather than include)
    assert filter_actuators(model, group=1, flags=(False,)) == [0, 3, 4]
    assert filter_actuators(model, substring="_motor", group=1, flags=(True, False)) == [0, 3, 4]


@pytest.mark.unit
def test_filter_actuators_invalid_inputs(actuator_model):
    model = actuator_model

    # No filters defined at all
    with pytest.raises(AssertionError):
        filter_actuators(model)
    # No flags defined at all
    with pytest.raises(AssertionError):
        filter_actuators(model, group=0)
    # Wrong number of flags (must match the filters)
    with pytest.raises(AssertionError):
        filter_actuators(model, names=["arm_elbow_motor"], flags=(True, True))


# region disable_acts


@pytest.mark.unit
def test_disable_actuators_valid_inputs(actuator_model):
    model = actuator_model

    # Starting groups and no actuator groups disabled yet
    assert list(model.actuator_group) == [0, 1, 1, 2, 3]
    assert model.opt.disableactuator == 0

    disable_actuators(model, [1, 3])

    # Groups 0-3 are all taken, so actuators 1 and 3 get moved to the next
    # free group, 4
    assert list(model.actuator_group) == [0, 4, 1, 4, 3]
    # Group 4 is now disabled via the bitmask
    assert model.opt.disableactuator == 1 << 4


@pytest.mark.unit
def test_disable_actuators_invalid_inputs(actuator_model):
    model = actuator_model

    # Empty actuator id
    with pytest.raises(AssertionError):
        disable_actuators(model, [])
    # Negative actuator id
    with pytest.raises(AssertionError):
        disable_actuators(model, [-1])
    # Actuator id == nactuator (out of bounds)
    with pytest.raises(AssertionError):
        disable_actuators(model, [model.nactuator])
    # Actuator id well beyond nactuator
    with pytest.raises(AssertionError):
        disable_actuators(model, [model.nactuator + 3])
