from pathlib import Path

import mujoco
import pytest

from contact_gym.robots import (
    ALL_GRIPPERS,
    ALL_ROBOTS,
    GRIPPER_HOME_KEYS,
    GRIPPER_PATHS,
    ROBOT_HOME_KEYS,
    SCENE_PATHS,
    get_ctrl_dim,
    get_qpos_dim,
)

# region assets


@pytest.mark.unit
@pytest.mark.parametrize("asset_path", list(SCENE_PATHS.values()) + list(GRIPPER_PATHS.values()))
def test_asset_path(asset_path: str):
    path = Path(asset_path)
    assert path.exists()
    assert path.is_file()
    assert path.suffix == ".xml"


@pytest.mark.unit
@pytest.mark.parametrize("robot", ALL_ROBOTS)
def test_robot(robot: str):
    # Validate completeness
    assert robot in SCENE_PATHS
    for env_home in ROBOT_HOME_KEYS.values():
        assert robot in env_home

    # Validate model and attachment site existence
    model = mujoco.MjModel.from_xml_path(SCENE_PATHS[robot])
    model.site("attachment_site")


@pytest.mark.unit
@pytest.mark.parametrize("gripper", ALL_GRIPPERS)
def test_gripper(gripper: str):
    # Validate completeness
    assert gripper in GRIPPER_PATHS
    for env_home in GRIPPER_HOME_KEYS.values():
        assert gripper in env_home

    # Validate model and tcp existence
    model = mujoco.MjModel.from_xml_path(GRIPPER_PATHS[gripper])
    model.site("tcp")


@pytest.mark.unit
@pytest.mark.parametrize("robot", ALL_ROBOTS)
def test_robot_home_key(robot: str):
    # Validate completeness
    for env_home in ROBOT_HOME_KEYS.values():
        assert "qpos" in env_home[robot]
        assert "ctrl" in env_home[robot]

    # Validate consistency of dimensions
    env_name = next(iter(ROBOT_HOME_KEYS.keys()))
    qpos_dim = len(ROBOT_HOME_KEYS[env_name][robot]["qpos"])
    ctrl_dim = len(ROBOT_HOME_KEYS[env_name][robot]["ctrl"])
    for env_home in ROBOT_HOME_KEYS.values():
        assert len(env_home[robot]["qpos"]) == qpos_dim
        assert len(env_home[robot]["ctrl"]) == ctrl_dim


@pytest.mark.unit
@pytest.mark.parametrize("gripper", ALL_GRIPPERS)
def test_gripper_home_key(gripper: str):
    # Validate completeness
    for env_home in GRIPPER_HOME_KEYS.values():
        assert "qpos" in env_home[gripper]
        assert "ctrl" in env_home[gripper]

    # Validate consistency of dimensions
    env_name = next(iter(ROBOT_HOME_KEYS.keys()))
    qpos_dim = len(GRIPPER_HOME_KEYS[env_name][gripper]["qpos"])
    ctrl_dim = len(GRIPPER_HOME_KEYS[env_name][gripper]["ctrl"])
    for env_home in GRIPPER_HOME_KEYS.values():
        assert len(env_home[gripper]["qpos"]) == qpos_dim
        assert len(env_home[gripper]["ctrl"]) == ctrl_dim


# region helpers


@pytest.mark.unit
@pytest.mark.parametrize("asset_name", ALL_ROBOTS + ALL_GRIPPERS)
def test_get_dim_valid_inputs(asset_name: str):
    HOME_REGISTRY = ROBOT_HOME_KEYS if asset_name in ALL_ROBOTS else GRIPPER_HOME_KEYS
    env_name = next(iter(HOME_REGISTRY.keys()))

    # Check qpos dimension
    qpos_dim = get_qpos_dim(asset_name)
    assert qpos_dim == len(HOME_REGISTRY[env_name][asset_name]["qpos"])
    # Check ctrl dimension
    ctrl_dim = get_ctrl_dim(asset_name)
    assert ctrl_dim == len(HOME_REGISTRY[env_name][asset_name]["ctrl"])


@pytest.mark.unit
def test_get_dim_invalid_inputs():
    asset_name = "a" * 4 + "dd" * 3 + "y" * 2
    # Non-existent asset name
    with pytest.raises(AssertionError):
        get_qpos_dim(asset_name)
    with pytest.raises(AssertionError):
        get_ctrl_dim(asset_name)
