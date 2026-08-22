from __future__ import annotations

import mujoco
import numpy as np
import pytest

from contact_gym.dr import DataStateRandomizer
from contact_gym.dr.state import (
    qpos_state_cfg,
    qpos_state_cfg_from_jnt_sel,
    qvel_state_cfg,
    qvel_state_cfg_from_jnt_sel,
)
from contact_gym.utils.noise import ConstantSampler, Noise

# ---------------------------------------------------------------------------

_STATE_DR_XML = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 0">
      <joint name="hinge1" type="hinge" axis="0 0 1"/>
      <geom name="geom1" type="box" size="0.05 0.05 0.05"/>
      <body name="body2" pos="0.1 0 0">
        <joint name="ball1" type="ball"/>
        <geom name="geom2" type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_STATE_DR_XML)


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _const_noise(value, dtype=np.float64) -> Noise:
    return Noise(ConstantSampler(value, dtype=dtype), operation="add")


# region qpos


@pytest.mark.unit
def test_qpos_state_cfg(model, data, rng):
    hinge_id = model.joint("hinge1").id
    qpos_adr = model.jnt_qposadr[hinge_id]
    nominal_qpos = data.qpos[qpos_adr].copy()

    noise_value = 0.1
    cfg = qpos_state_cfg(_const_noise(noise_value), entry_sel=qpos_adr)
    randomizer = DataStateRandomizer(cfg, data)

    randomizer(model, data, rng)
    expected = nominal_qpos + noise_value
    assert np.isclose(data.qpos[qpos_adr], expected)

    # Test that cached nominal value remains unchaged after sampling
    randomizer(model, data, rng)
    assert np.isclose(data.qpos[qpos_adr], expected)


# region qpos_from_jnt


@pytest.mark.unit
def test_qpos_state_cfg_from_jnt_sel(model, data, rng):
    ball_id = model.joint("ball1").id
    qpos_adr = model.jnt_qposadr[ball_id]
    qpos_slice = slice(qpos_adr, qpos_adr + 4)
    nominal_qpos = data.qpos[qpos_slice].copy()
    assert np.isclose(np.linalg.norm(nominal_qpos), 1.0)

    noise_value = np.array([0.05, -0.02, 0.01, 0.0])
    cfg = qpos_state_cfg_from_jnt_sel(model, _const_noise(noise_value), jnt_sel=ball_id)
    randomizer = DataStateRandomizer(cfg, data)

    randomizer(model, data, rng)
    expected = nominal_qpos + noise_value
    assert np.allclose(data.qpos[qpos_slice], expected)
    assert not np.isclose(np.linalg.norm(data.qpos[qpos_slice]), 1.0)  # no renormalization

    randomizer(model, data, rng)
    assert np.allclose(data.qpos[qpos_slice], expected)


# region qvel


@pytest.mark.unit
def test_qvel_state_cfg(model, data, rng):
    hinge_id = model.joint("hinge1").id
    dof_adr = model.jnt_dofadr[hinge_id]
    nominal_qvel = data.qvel[dof_adr].copy()

    noise_value = 0.3
    cfg = qvel_state_cfg(_const_noise(noise_value), entry_sel=dof_adr)
    randomizer = DataStateRandomizer(cfg, data)

    randomizer(model, data, rng)
    expected = nominal_qvel + noise_value
    assert np.isclose(data.qvel[dof_adr], expected)

    randomizer(model, data, rng)
    assert np.isclose(data.qvel[dof_adr], expected)


# region qvel_from_jnt


@pytest.mark.unit
def test_qvel_state_cfg_from_jnt_sel(model, data, rng):
    ball_id = model.joint("ball1").id
    dof_adr = model.jnt_dofadr[ball_id]
    dof_slice = slice(dof_adr, dof_adr + 3)
    nominal_qvel = data.qvel[dof_slice].copy()

    noise_value = np.array([0.01, -0.02, 0.03])
    cfg = qvel_state_cfg_from_jnt_sel(model, _const_noise(noise_value), jnt_sel=ball_id)
    randomizer = DataStateRandomizer(cfg, data)

    randomizer(model, data, rng)
    expected = nominal_qvel + noise_value
    assert np.allclose(data.qvel[dof_slice], expected)

    randomizer(model, data, rng)
    assert np.allclose(data.qvel[dof_slice], expected)
