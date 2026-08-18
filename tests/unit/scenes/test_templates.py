import mujoco
import numpy as np
import pytest

from contact_gym.scenes.templates import (
    add_body_at_site,
    add_contact_sensor,
    add_frame_sensors,
    add_mocap_body,
    add_weld_equality,
    get_articulation_parent_bodyname,
)

# region get_articulation

_SINGLE_PARENT_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="arm_base">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="arm_link1" pos="0 0 0.2">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MULTI_PARENT_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="arm_base1">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="arm_base2" pos="1 0 0">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

_NO_PARENT_MODEL_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def single_parent_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_SINGLE_PARENT_MODEL_XML)


@pytest.fixture
def multi_parent_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_MULTI_PARENT_MODEL_XML)


@pytest.fixture
def no_parent_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_NO_PARENT_MODEL_XML)


@pytest.mark.unit
def test_get_articulation_parent_bodyname_valid_inputs(single_parent_spec):
    spec = single_parent_spec
    # Sanity check spec
    assert len(spec.worldbody.bodies) == 1
    # Single parent spec -> arm_base
    assert get_articulation_parent_bodyname(spec) == "arm_base"


@pytest.mark.unit
def test_get_articulation_parent_bodyname_invalid_inputs(multi_parent_spec, no_parent_spec):
    # worldbody has two direct children -> ambiguous parent
    with pytest.raises(AssertionError):
        get_articulation_parent_bodyname(multi_parent_spec)
    # worldbody has zero direct children -> no parent to return
    with pytest.raises(AssertionError):
        get_articulation_parent_bodyname(no_parent_spec)


# region add_body_site

_ATTACH_PARENT_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="attach_parent">
      <geom type="sphere" size="0.05"/>
      <site name="attachment_site" pos="0.5 0 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""

_ATTACHABLE_BODY_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="tool_body">
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.unit
def test_add_body_at_site_valid_inputs():
    spec1 = mujoco.MjSpec.from_string(_ATTACH_PARENT_MODEL_XML)
    spec2 = mujoco.MjSpec.from_string(_ATTACHABLE_BODY_MODEL_XML)

    # Attach tool_body from spec2 at attachment_site in spec1
    site_pos = np.copy(spec1.site("attachment_site").pos)
    site_quat = np.copy(spec1.site("attachment_site").quat)
    add_body_at_site(
        spec1, spec2, sitename="attachment_site", bodyname="tool_body", prefix="attached_"
    )
    # New body should now exist in spec1, prefixed as expected
    attached_body = spec1.body("attached_tool_body")
    assert attached_body is not None
    # Old attachment site must be gone
    assert spec1.site("attachment_site") is None

    # Compile and check the attached body ends up at the site's pose
    model = spec1.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = model.body("attached_tool_body").id
    np.testing.assert_allclose(data.xpos[body_id], site_pos, atol=1e-6)
    np.testing.assert_allclose(data.xquat[body_id], site_quat, atol=1e-6)


@pytest.mark.unit
def test_add_body_at_site_invalid_inputs():
    # Nonexistent site name.
    spec1 = mujoco.MjSpec.from_string(_ATTACH_PARENT_MODEL_XML)
    spec2 = mujoco.MjSpec.from_string(_ATTACHABLE_BODY_MODEL_XML)
    with pytest.raises(AssertionError):
        add_body_at_site(spec1, spec2, sitename="nonexistent_site", bodyname="tool_body")
    # Nonexistent body name.
    spec1 = mujoco.MjSpec.from_string(_ATTACH_PARENT_MODEL_XML)
    spec2 = mujoco.MjSpec.from_string(_ATTACHABLE_BODY_MODEL_XML)
    with pytest.raises(AssertionError):
        add_body_at_site(spec1, spec2, sitename="attachment_site", bodyname="nonexistent_body")


# region add_mocap

_BASE_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="anchor">
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def base_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_BASE_MODEL_XML)


@pytest.mark.unit
def test_add_mocap_body_valid_inputs_default(base_spec):
    spec = base_spec

    # Add mocap with default args
    add_mocap_body(spec, name="mocap", group=4, size=0.005, opacity=0.7)
    body = spec.body("mocap")
    assert body is not None
    assert body.mocap
    # Only the base box geom should be present; no site, no frame geoms
    assert len(body.geoms) == 1
    geom = body.geoms[0]
    assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
    assert geom.group == 4
    assert geom.contype == 0
    assert geom.conaffinity == 0
    np.testing.assert_allclose(geom.size, [0.005] * 3)
    np.testing.assert_allclose(geom.rgba, [0, 0.5, 0, 0.7])
    # No sites should exist
    assert len(body.sites) == 0
    assert spec.site("mocap") is None


@pytest.mark.unit
def test_add_mocap_body_valid_inputs_non_default(base_spec):
    spec = base_spec

    # Add mocap with site and frame
    add_mocap_body(
        spec,
        name="mocap",
        group=2,
        size=0.01,
        opacity=0.5,
        add_site=True,
        add_frame=True,
        frame_opacity=0.2,
    )
    body = spec.body("mocap")
    assert body is not None
    assert body.mocap
    # Base geom + 3 frame axis geoms
    assert len(body.geoms) == 4
    for geom in body.geoms:
        assert geom.group == 2
        assert geom.contype == 0
        assert geom.conaffinity == 0
    np.testing.assert_allclose(body.geoms[0].size, [0.01] * 3)
    np.testing.assert_allclose(body.geoms[0].rgba, [0, 0.5, 0, 0.5])
    # Site should be added on the body with matching name/size/group
    assert len(body.sites) == 1
    site = body.sites[0]
    assert site.name == "mocap"
    assert site.group == 2
    np.testing.assert_allclose(site.size[0], 0.01)
    assert spec.site("mocap") is not None


# region add_weld

_WELD_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="body1">
      <joint type="free"/>
      <geom type="sphere" size="0.05"/>
      <site name="site1" pos="0 0 0"/>
    </body>
    <body name="body2" pos="1 0 0">
      <joint type="free"/>
      <geom type="sphere" size="0.05"/>
      <site name="site2" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def weld_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_WELD_MODEL_XML)


@pytest.mark.unit
def test_add_weld_equality_valid_inputs(weld_spec):
    spec = weld_spec

    # mjOBJ_BODY: obj2name may be omitted (welds body1 to the world)
    add_weld_equality(
        spec,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        obj1name="body1",
        obj2name=None,
        name="body_weld_to_world",
    )
    eq0 = spec.equality("body_weld_to_world")
    assert eq0.type == mujoco.mjtEq.mjEQ_WELD
    assert eq0.objtype == mujoco.mjtObj.mjOBJ_BODY
    assert eq0.name1 == "body1"
    assert eq0.name2 == ""
    # mjOBJ_BODY: both objects specified (welds body1 to body2)
    add_weld_equality(
        spec,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        obj1name="body1",
        obj2name="body2",
        name="body_weld_to_body",
    )
    eq1 = spec.equality("body_weld_to_body")
    assert eq1.objtype == mujoco.mjtObj.mjOBJ_BODY
    assert eq1.name1 == "body1"
    assert eq1.name2 == "body2"
    # mjOBJ_SITE: both objects required
    add_weld_equality(
        spec, objtype=mujoco.mjtObj.mjOBJ_SITE, obj1name="site1", obj2name="site2", name="site_weld"
    )
    eq2 = spec.equality("site_weld")
    assert eq2.objtype == mujoco.mjtObj.mjOBJ_SITE
    assert eq2.name1 == "site1"
    assert eq2.name2 == "site2"
    assert len(spec.equalities) == 3
    # Should compile without error
    spec.compile()


@pytest.mark.unit
def test_add_weld_equality_invalid_inputs(weld_spec):
    spec = weld_spec

    # objtype not body/site
    with pytest.raises(AssertionError):
        add_weld_equality(
            spec, objtype=mujoco.mjtObj.mjOBJ_GEOM, obj1name="body1", obj2name="body2"
        )
    # mjOBJ_BODY requires obj1name
    with pytest.raises(AssertionError):
        add_weld_equality(spec, objtype=mujoco.mjtObj.mjOBJ_BODY, obj1name=None, obj2name="body2")
    # mjOBJ_SITE requires obj1name
    with pytest.raises(AssertionError):
        add_weld_equality(spec, objtype=mujoco.mjtObj.mjOBJ_SITE, obj1name=None, obj2name="site2")
    # mjOBJ_SITE requires obj2name
    with pytest.raises(AssertionError):
        add_weld_equality(spec, objtype=mujoco.mjtObj.mjOBJ_SITE, obj1name="site1", obj2name=None)


# region add_frame

_FRAME_SENSOR_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="body1">
      <joint type="free"/>
      <geom name="geom1" type="sphere" size="0.05"/>
    </body>
    <body name="body2" pos="1 0 0">
      <joint type="free"/>
      <geom name="geom2" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_SENSOR_TYPE_MAP = {
    "pos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
    "quat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    "xaxis": mujoco.mjtSensor.mjSENS_FRAMEXAXIS,
    "yaxis": mujoco.mjtSensor.mjSENS_FRAMEYAXIS,
    "zaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    "linvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
    "angvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
    "linacc": mujoco.mjtSensor.mjSENS_FRAMELINACC,
    "angacc": mujoco.mjtSensor.mjSENS_FRAMEANGACC,
}


@pytest.fixture
def frame_sensor_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_FRAME_SENSOR_MODEL_XML)


@pytest.mark.unit
@pytest.mark.parametrize("sensor_key,sensor_type", list(_SENSOR_TYPE_MAP.items()))
def test_add_frame_sensors_valid_inputs_sensor_types(frame_sensor_spec, sensor_key, sensor_type):
    spec = frame_sensor_spec

    # No ref: sensor measured w.r.t. global frame
    add_frame_sensors(spec, objtype=mujoco.mjtObj.mjOBJ_BODY, objname="body1", sensors=[sensor_key])
    sensor = spec.sensor(f"body1_{sensor_key}")
    assert sensor is not None
    assert sensor.type == sensor_type
    assert sensor.objtype == mujoco.mjtObj.mjOBJ_BODY
    assert sensor.objname == "body1"
    assert sensor.reftype is None or sensor.reftype == mujoco.mjtObj.mjOBJ_UNKNOWN
    assert sensor.refname == ""
    # Ref specified
    add_frame_sensors(
        spec,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="body1",
        reftype=mujoco.mjtObj.mjOBJ_BODY,
        refname="body2",
        sensors=[sensor_key],
    )
    sensor = spec.sensor(f"body2_body1_{sensor_key}")
    assert sensor is not None
    assert sensor.type == sensor_type
    assert sensor.objtype == mujoco.mjtObj.mjOBJ_BODY
    assert sensor.objname == "body1"
    assert sensor.reftype == mujoco.mjtObj.mjOBJ_BODY
    assert sensor.refname == "body2"


@pytest.mark.unit
def test_add_frame_sensors_invalid_inputs(frame_sensor_spec):
    spec = frame_sensor_spec

    # objtype not specified
    with pytest.raises(AssertionError):
        add_frame_sensors(spec, objtype=None, objname="body1", sensors=["pos"])
    # objname not specified
    with pytest.raises(AssertionError):
        add_frame_sensors(spec, objtype=mujoco.mjtObj.mjOBJ_BODY, objname=None, sensors=["pos"])
    # objtype outside the allowed set
    with pytest.raises(AssertionError):
        add_frame_sensors(spec, objtype=mujoco.mjtObj.mjOBJ_JOINT, objname="body1", sensors=["pos"])
    # reftype outside the allowed set
    with pytest.raises(AssertionError):
        add_frame_sensors(
            spec,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname="body1",
            reftype=mujoco.mjtObj.mjOBJ_JOINT,
            refname="body2",
            sensors=["pos"],
        )


# region add_contact


_CONTACT_SENSOR_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="body1">
      <joint type="free"/>
      <geom name="geom1" type="sphere" size="0.05"/>
      <site name="site1"/>
    </body>
    <body name="body2" pos="1 0 0">
      <joint type="free"/>
      <geom name="geom2" type="sphere" size="0.05"/>
      <site name="site2"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def contact_sensor_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_CONTACT_SENSOR_MODEL_XML)


@pytest.mark.unit
def test_add_contact_sensor_valid_inputs(contact_sensor_spec):
    spec = contact_sensor_spec

    # No objects specified
    add_contact_sensor(spec, obj1type=None, obj1name=None)
    sensor = spec.sensor("_contact")
    assert sensor is not None
    assert sensor.type == mujoco.mjtSensor.mjSENS_CONTACT
    assert sensor.objname == ""
    assert sensor.refname == ""
    list(sensor.intprm) == [1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value, 0, 1]
    # Objects specified
    data = (1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value) | (
        1 << mujoco.mjtConDataField.mjCONDATA_FORCE.value
    )
    add_contact_sensor(
        spec,
        obj1type=mujoco.mjtObj.mjOBJ_GEOM,
        obj1name="geom1",
        obj2type=mujoco.mjtObj.mjOBJ_GEOM,
        obj2name="geom2",
        data=data,
        reduce=2,
        num=3,
    )
    sensor = spec.sensor("geom1_geom2_contact")
    assert sensor is not None
    assert sensor.type == mujoco.mjtSensor.mjSENS_CONTACT
    assert sensor.objtype == mujoco.mjtObj.mjOBJ_GEOM
    assert sensor.objname == "geom1"
    assert sensor.reftype == mujoco.mjtObj.mjOBJ_GEOM
    assert sensor.refname == "geom2"
    assert list(sensor.intprm) == [data, 2, 3]


@pytest.mark.unit
def test_add_contact_sensor_invalid_inputs(contact_sensor_spec):
    spec = contact_sensor_spec

    # obj1type outside the allowed set
    with pytest.raises(AssertionError):
        add_contact_sensor(spec, obj1type=mujoco.mjtObj.mjOBJ_JOINT, obj1name="body1")
    # obj2type outside the allowed set
    with pytest.raises(AssertionError):
        add_contact_sensor(
            spec,
            obj1type=mujoco.mjtObj.mjOBJ_BODY,
            obj1name="body1",
            obj2type=mujoco.mjtObj.mjOBJ_JOINT,
            obj2name="body2",
        )
    # reduce negative
    with pytest.raises(AssertionError):
        add_contact_sensor(spec, obj1type=None, obj1name=None, reduce=-1)
    # reduce at upper illegal boundary
    with pytest.raises(AssertionError):
        add_contact_sensor(spec, obj1type=None, obj1name=None, reduce=4)
    # num zero
    with pytest.raises(AssertionError):
        add_contact_sensor(spec, obj1type=None, obj1name=None, num=0)
    # num negative
    with pytest.raises(AssertionError):
        add_contact_sensor(spec, obj1type=None, obj1name=None, num=-1)
