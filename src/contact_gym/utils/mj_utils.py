import mujoco


def mjtjoint_to_qpos_dim(jnt_type: mujoco.mjtJoint) -> int:
    """Get the qpos dimension of the input `jnt_type`."""
    _MJTJOINT_TO_QPOS_DIM = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    return _MJTJOINT_TO_QPOS_DIM[int(jnt_type)]


def mjtjoint_to_dof_dim(jnt_type: mujoco.mjtJoint) -> int:
    """Get the dof dimension of the input `jnt_type`."""
    _MJTJOINT_TO_DOF_DIM = {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    return _MJTJOINT_TO_DOF_DIM[int(jnt_type)]


def get_dof_dim_from_joints(model: mujoco.MjModel, qpos_adr: int, qpos_dim: int = 1) -> int:
    """Get the `dof` dimension of a set of joints according to their types.

    Args:
        model: MuJoCo model.
        qpos_adr: Start address in `qpos`.
        qpos_dim: Dimension of `qpos` to consider. Default value is 1.

    Returns:
        `dof` dimension of the associated set of joints.
    """
    assert 0 <= qpos_adr < model.nq
    assert 0 <= qpos_adr + qpos_dim - 1 < model.nq
    assert qpos_dim > 0
    dof_dim = 0

    def get_dof_dim_jnt(type: mujoco.mjtJoint) -> int:
        if type == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if type == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        return 1

    for jnt in range(model.njnt):
        if qpos_adr <= model.jnt_qposadr[jnt] < qpos_adr + qpos_dim:
            dof_dim += get_dof_dim_jnt(model.jnt_type[jnt])
    return dof_dim


def filter_joints(
    model: mujoco.MjModel,
    names: list[str] | None = None,
    substring: str | None = None,
    type: mujoco.mjtJoint | None = None,
    group: int | None = None,
    flags: tuple[bool, ...] | None = None,
) -> list[int]:
    """Filter model joints by a combination of filters.

    A given model joint is included only if it satisfies all defined filter + flag combinations.
    Each filter is disabled by default. At least a single filter must be passed in. Each flag
    affects the postionally corresponding filter. If a flag is True, then the filter must be
    satisfied. If False, then the inverse condition is required (i.e. exclude rather than include).

    Args:
        model: MuJoCo model to filter through. All joints are considered.
        names: Names of target joints. Default value is `None`.
        substring: Common substring in names of target joints. Default value is `None`.
        type: Joint type (`mujoco.mjtJoint`) of target joints. Default value is `None`.
        group: Joint group of target joints. Default value is `None`.
        flags: Tuple of boolean flags that determine how filters are interpreted. Length
            must match the number of active filters. For details, check above description.
            Default value is None.

    Returns:
        List of joint ids of model joints that satisfy all defined filters.
    """
    f_is_not_none = [f is not None for f in [names, substring, type, group]]
    n_filters = sum(f_is_not_none)
    assert any(f_is_not_none), "Filtering joints requires at least a single filter. None defined."
    assert flags is not None, "Filter flags must be defined. None recieved."
    assert len(flags) == sum(f_is_not_none), (
        f"Number of flags must match the {n_filters} filters. Got {len(flags)}."
    )
    filters = (
        None if names is None else (lambda joint: joint.name in names),
        None if substring is None else (lambda joint: substring in joint.name),
        None if type is None else (lambda joint: joint.type == type),
        None if group is None else (lambda joint: joint.group == group),
    )
    active_filters = tuple([f for i, f in enumerate(filters) if f_is_not_none[i]])
    jnt_ids = [
        i
        for i in range(model.njnt)
        if all(fltr(model.joint(i)) == flag for fltr, flag in zip(active_filters, flags))
    ]
    return jnt_ids


def filter_actuators(
    model: mujoco.MjModel,
    names: list[str] | None = None,
    substring: str | None = None,
    trntype: mujoco.mjtTrn | None = None,
    group: int | None = None,
    flags: tuple[bool, ...] | None = None,
) -> list[int]:
    """Filter model actuators by a combination of filters.

    A given model actuator is included only if it satisfies all defined filter + flag combinations.
    Each filter is disabled by default. At least a single filter must be passed in. Each flag
    affects the postionally corresponding filter. If a flag is True, then the filter must be
    satisfied. If False, then the inverse condition is required (i.e. exclude rather than include).

    Args:
        model: MuJoCo model to filter through. All actuators are considered.
        names: Names of target actuators. Default value is `None`.
        substring: Common substring in names of target actuators. Default value is `None`.
        trntype: Transmission type (`mujoco.mjtTrn`) of target actuators. Default value is `None`.
        group: Actuator group of target actuators. Default value is `None`.
        flags: Tuple of boolean flags that determine how filters are interpreted. Length
            must match the number of active filters. For details, check above description.
            Default value is None.

    Returns:
        List of actuator ids of model actuators that satisfy all defined filters.
    """
    f_is_not_none = [f is not None for f in [names, substring, trntype, group]]
    n_filters = sum(f_is_not_none)
    assert any(f_is_not_none), (
        "Filtering actuators requires at least a single filter. None defined."
    )
    assert flags is not None, "Filter flags must be defined. None recieved."
    assert len(flags) == sum(f_is_not_none), (
        f"Number of flags must match the {n_filters} filters. Got {len(flags)}."
    )
    filters = (
        None if names is None else (lambda actuator: actuator.name in names),
        None if substring is None else (lambda actuator: substring in actuator.name),
        None if trntype is None else (lambda actuator: actuator.trntype == trntype),
        None if group is None else (lambda actuator: actuator.group == group),
    )
    active_filters = tuple([f for i, f in enumerate(filters) if f_is_not_none[i]])
    act_ids = [
        i
        for i in range(model.nactuator)
        if all(fltr(model.actuator(i)) == flag for fltr, flag in zip(active_filters, flags))
    ]
    return act_ids


def disable_actuators(model: mujoco.MjModel, actuator_ids: list[int]) -> None:
    """Moves all selected actuators to a common free group then disables that group.

    Args:
        model: MuJoCo model to update.
        actuator_ids: IDs of all actuators that should be disabled.
    """
    assert actuator_ids, "Got an empty actuator id list."
    assert all([0 <= i < model.nactuator for i in actuator_ids]), (
        f"Invalid actuator ids. Valid values for this model are [0, {model.nactuator - 1}]."
    )
    # Find a free actuator group to disable
    free_group = 0
    active_groups = set(model.actuator_group)
    while free_group in active_groups:
        free_group += 1
    # Move all actuators to the new empty group and disable it
    model.actuator_group[actuator_ids] = free_group
    model.opt.disableactuator |= 1 << free_group
