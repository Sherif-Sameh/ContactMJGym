from functools import partial

import mujoco
from mujoco import mjtObj, mjtSensor

# region Body


def get_articulation_parent_bodyname(spec: mujoco.MjSpec) -> str:
    """Get an articulation's parent body's name.

    The parent body must be a child of the world body. The world body should have exactly
    one child body.

    Args:
        spec: MuJoCo spec.

    Returns:
        Name of the articulation's parent body.
    """
    nbody = len(spec.worldbody.bodies)
    assert nbody == 1, f"Articulation must have a single parent body. Got {nbody}."
    return spec.worldbody.bodies[0].name


def add_body_at_site(
    spec1: mujoco.MjSpec, spec2: mujoco.MjSpec, sitename: str, bodyname: str, *, prefix: str = ""
) -> None:
    """Attach a body from `spec2` to an attachment site from `spec1`.

    **Note**: The attachment site is deleted after the body is attached.

    Args:
        spec1: First MuJoCo spec with an attachment site.
        spec2: Second MuJoCo spec with the body to attach.
        sitename: Name of the attachment site in `spec1`.
        bodyname: Name of the body to attach in `spec2`.
        prefix: Prefix when attaching the body from `spec2` to `spec1`. Default value is "".

    Returns:
        Updated MuJoCo `spec1` after body attachment.
    """
    # Get attachment site body and pose
    site = spec1.site(sitename)
    assert site is not None, f"{site} site does not exist within spec1."
    attach_body = site.parent
    attach_pos, attach_quat = site.pos, site.quat
    # Add frame at attachment body and pose then delete site
    attach_frame = attach_body.add_frame(pos=attach_pos, quat=attach_quat)
    spec1.delete(site)
    # Attach new body at attachment frame
    body = spec2.body(bodyname)
    assert body is not None, f"{body} body does not exist within spec2."
    attach_frame.attach_body(body, prefix)
    return spec1


def add_mocap_body(
    spec: mujoco.MjSpec,
    name: str = "mocap",
    group: int = 4,
    size: float = 0.005,
    opacity: float = 0.7,
    *,
    add_site: bool = False,
    add_frame: bool = False,
    frame_opacity: float = 0.1,
) -> mujoco.MjSpec:
    """Add a mocap body to the given spec.

    Args:
        name: Name of the mocap body. Default value is mocap.
        group: Group for child geoms. Default value is 4.
        size: Size of the box geom added to the mocap body. Default value is 0.005.
        opacity: Opacity of the box geom added to the mocap body. Default value is 0.7.
        add_site: If True, add a site to the mocap body. Default value is False.
        add_frame: If True, add geoms visualizing the full frame of the mocap body. The
            axes are visualized in red: x; green: y; blue: z. Default value is False.
        frame_opacity: Opacity of the box geoms used for frame visualization. Default
            value is 0.1.

    Returns:
        Updated MuJoCo spec.
    """
    body = spec.worldbody.add_body(name=name, mocap=True)
    common = dict(type=mujoco.mjtGeom.mjGEOM_BOX, group=group, contype=0, conaffinity=0)
    body.add_geom(**common, size=[size] * 3, rgba=[0, 0.5, 0, opacity])
    if add_site:
        body.add_site(name=name, size=size, group=group)
    if add_frame:
        for rgba, fromto in [
            ([1, 0, 0, frame_opacity], [0, 0, 0, 0.3, 0, 0]),
            ([0, 1, 0, frame_opacity], [0, 0, 0, 0, 0.3, 0]),
            ([0, 0, 1, frame_opacity], [0, 0, 0, 0, 0, 0.3]),
        ]:
            body.add_geom(**common, size=[size] * 3, rgba=rgba, fromto=fromto)
    return spec


# region Equality


def add_weld_equality(
    spec: mujoco.MjSpec,
    objtype: mujoco.mjtObj,
    obj1name: str | None,
    obj2name: str | None,
    name: str = "",
    *,
    active: bool = True,
    solimp: tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2.0),
    solref: tuple[float, ...] = (0.02, 1.0),
) -> mujoco.MjSpec:
    """Add a weld equality constraint to the given spec.

    Args:
        obj1name: Name of the first object in the weld constraint.
        obj2name: Name of the second object in the weld constraint.
        type: Type of the objects being welded. Must be either mjOBJ_BODY or mjOBJ_SITE.
        name: Name of the weld equality constraint. Default value is empty.
        active: If True, the equality constraint is initially active. Default value
            is True.
        solimp: Solver impedance parameters for the equality constraint. Default
            value is (0.9, 0.95, 0.001, 0.5, 2.0).
        solref: Solver reference parameters for the equality constraint. Default
            value is (0.02, 1.0).

    Returns:
        Updated MuJoCo spec.
    """
    assert objtype in [mjtObj.mjOBJ_BODY, mjtObj.mjOBJ_SITE]
    if objtype == mjtObj.mjOBJ_BODY:
        assert obj1name is not None, "Object 1 must be specified with mjOBJ_BODY object type."
    else:
        assert obj1name is not None and obj2name is not None, (
            "Both objects must be specified with mjOBJ_SITE object type."
        )
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name=name,
        objtype=objtype,
        name1=obj1name,
        name2=obj2name,
        active=active,
        solimp=solimp,
        solref=solref,
    )
    return spec


# region Sensor


def add_frame_sensors(
    spec: mujoco.MjSpec,
    objtype: mjtObj,
    objname: str,
    reftype: mjtObj | None = None,
    refname: str | None = None,
    *,
    prefix: str | None = None,
    sensors: tuple[str, ...] = (),
) -> mujoco.MjSpec:
    """Add a collection of frame-based sensors to the given spec.

    Args:
        spec: MuJoCo spec.
        objtype: Type of the object to which the sensors are attached.
        objname: Name of the object to which th sensors are attached.
        reftype: Type of the object to which the frame of reference is attached. If not given,
            sensor values are measured with respect to the global frame. Default value is None.
        refname: Name of the object to which the frame of reference is attached. If not given,
            sensor values are measured with respect to the global frame. Default value is None.
        prefix: Optional prefix for sensor names. If None, a prefix is derived from `objname`
            and `refname`. Default value is None.
        sensors: Names of the frame-based sensors to add. Names are expected without the "frame"
            prefix (e.g., "pos", "quat", "linvel"). Default value is empty.

    Returns:
        Updated MuJoCo spec.
    """
    assert objtype is not None, "Object 1's type must be specified."
    assert objname is not None, "Object 1's name must be specified."
    for oname, otype in zip([objname, refname], [objtype, reftype]):
        assert otype is None or otype in [
            mjtObj.mjOBJ_BODY,
            mjtObj.mjOBJ_XBODY,
            mjtObj.mjOBJ_GEOM,
            mjtObj.mjOBJ_SITE,
            mjtObj.mjOBJ_CAMERA,
        ], f"Invalid object type for object {oname}."
    if prefix is None:
        prefix = (
            objname.split("-")[-1]
            if refname is None
            else f"{refname.split('-')[-1]}_{objname.split('-')[-1]}"
        )
    add_sensor = partial(
        spec.add_sensor, objtype=objtype, objname=objname, reftype=reftype, refname=refname
    )
    if "pos" in sensors:
        add_sensor(name=f"{prefix}_pos", type=mjtSensor.mjSENS_FRAMEPOS)
    if "quat" in sensors:
        add_sensor(name=f"{prefix}_quat", type=mjtSensor.mjSENS_FRAMEQUAT)
    if "xaxis" in sensors:
        add_sensor(name=f"{prefix}_xaxis", type=mjtSensor.mjSENS_FRAMEXAXIS)
    if "yaxis" in sensors:
        add_sensor(name=f"{prefix}_yaxis", type=mjtSensor.mjSENS_FRAMEYAXIS)
    if "zaxis" in sensors:
        add_sensor(name=f"{prefix}_zaxis", type=mjtSensor.mjSENS_FRAMEZAXIS)
    if "linvel" in sensors:
        add_sensor(name=f"{prefix}_linvel", type=mjtSensor.mjSENS_FRAMELINVEL)
    if "angvel" in sensors:
        add_sensor(name=f"{prefix}_angvel", type=mjtSensor.mjSENS_FRAMEANGVEL)
    if "linacc" in sensors:
        add_sensor(name=f"{prefix}_linacc", type=mjtSensor.mjSENS_FRAMELINACC)
    if "angacc" in sensors:
        add_sensor(name=f"{prefix}_angacc", type=mjtSensor.mjSENS_FRAMEANGACC)
    return spec


def add_contact_sensor(
    spec: mujoco.MjSpec,
    obj1type: mjtObj | None,
    obj1name: str | None,
    obj2type: mjtObj | None = None,
    obj2name: str | None = None,
    *,
    prefix: str | None = None,
    data: int = 1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value,
    reduce: int = 0,
    num: int = 1,
) -> mujoco.MjSpec:
    """Add a contact sensor to the given spec.

    Args:
        spec: MuJoCo spec.
        obj1type: Type of the first object for contact matching.
        obj1name: Name of the first object for contact matching.
        obj2type: Type of the second object for contact matching.
        obj2name: Name of the second object for contact matching.
        prefix: Optional prefix for sensor name. If None, a prefix is derived from `obj1name`
            and `obj2name`. Default value is None.
        data: Specifies which data fields are reported for contacts. Defined through bit
            shifting using the :class:`mujoco.mjtConDataField` enum. Default value
            reports "found" only.
        reduce: Contact reduction mode. 0: none; 1: mindist; 2: maxforce; 3:netforce.
            Illegal values will trigger an error. Default value is 0.
        num: Number of contacts to report. Defaut value is 1.

    Returns:
        Updated MuJoCo spec.
    """
    for otype in [obj1type, obj2type]:
        assert otype is None or otype in [
            mjtObj.mjOBJ_BODY,
            mjtObj.mjOBJ_XBODY,
            mjtObj.mjOBJ_GEOM,
            mjtObj.mjOBJ_SITE,
        ]
    assert 0 <= reduce < 4, f"reduce must be in [0, 3]. Got {reduce}."
    assert num > 0, f"num must be > 0. Got {num}."
    if prefix is None:
        prefix = "" if obj1name is None else obj1name.split("-")[-1]
        prefix += "" if obj2name is None else f"_{obj2name.split('-')[-1]}"
    spec.add_sensor(
        name=f"{prefix}_contact",
        type=mjtSensor.mjSENS_CONTACT,
        objtype=obj1type,
        objname=obj1name,
        reftype=obj2type,
        refname=obj2name,
        intprm=[data, reduce, num],
    )
    return spec
