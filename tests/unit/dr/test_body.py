from __future__ import annotations

import mujoco
import numpy as np
import pytest

from contact_gym.dr import ModelParamRandomizer
from contact_gym.dr.body import (
    body_gravitycomp_cfg,
    body_inertia_cfg,
    body_mass_cfg,
    body_pos_cfg,
    body_quat_cfg,
)
from contact_gym.utils.noise import ConstantSampler, Noise

_BODY_DR_XML = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0.1 0.2 0.3" quat="0.9689124 0.247404 0 0" gravcomp="0.5">
      <inertial pos="0 0 0" mass="1.5" diaginertia="0.010 0.020 0.030"/>
      <geom name="geom1" type="box" size="0.05 0.05 0.05"/>
      <body name="body2" pos="0.05 0 0" quat="1 0 0 0" gravcomp="0.0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
        <geom name="geom2" type="box" size="0.03 0.03 0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_BODY_DR_XML)


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _const_noise(value, dtype=np.float64) -> Noise:
    return Noise(ConstantSampler(value, dtype=dtype), operation="add")


# region pos


@pytest.mark.unit
def test_body_pos_cfg(model, data, rng):
    body_id = model.body("body1").id
    nominal_pos = model.body_pos[body_id].copy()

    noise_value = np.array([0.01, -0.02, 0.03])
    cfg = body_pos_cfg(_const_noise(noise_value), inst_sel=body_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_pos + noise_value
    assert np.allclose(model.body_pos[body_id], expected)

    # Test that cached nominal value remains unchaged after sampling
    randomizer(model, data, rng)
    assert np.allclose(model.body_pos[body_id], expected)


# region quat


@pytest.mark.unit
def test_body_quat_cfg(model, data, rng):
    body_id = model.body("body1").id
    nominal_quat = model.body_quat[body_id].copy()
    assert np.isclose(np.linalg.norm(nominal_quat), 1.0)

    noise_value = np.array([0.05, 0.02, 0.0, 0.0])
    cfg = body_quat_cfg(_const_noise(noise_value), inst_sel=body_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_quat + noise_value
    expected /= np.linalg.norm(expected)
    assert np.isclose(np.linalg.norm(model.body_quat[body_id]), 1.0)
    assert np.allclose(model.body_quat[body_id], expected)

    randomizer(model, data, rng)
    assert np.isclose(np.linalg.norm(model.body_quat[body_id]), 1.0)
    assert np.allclose(model.body_quat[body_id], expected)


# region mass


@pytest.mark.unit
def test_body_mass_cfg(model, data, rng):
    body_id = model.body("body1").id
    nominal_mass = model.body_mass[body_id].copy()

    noise_value = 0.2
    cfg = body_mass_cfg(_const_noise(noise_value), inst_sel=body_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_mass + noise_value
    assert np.isclose(model.body_mass[body_id], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.body_mass[body_id], expected)


# region inertia


@pytest.mark.unit
def test_body_inertia_cfg(model, data, rng):
    body_id = model.body("body1").id
    nominal_inertia = model.body_inertia[body_id].copy()

    noise_value = np.array([0.001, 0.002, 0.003])
    cfg = body_inertia_cfg(_const_noise(noise_value), inst_sel=body_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_inertia + noise_value
    assert np.allclose(model.body_inertia[body_id], expected)

    randomizer(model, data, rng)
    assert np.allclose(model.body_inertia[body_id], expected)


# region gravitycomp


@pytest.mark.unit
def test_body_gravitycomp_cfg(model, data, rng):
    body_id = model.body("body1").id
    nominal_gravcomp = model.body_gravcomp[body_id].copy()

    noise_value = 0.1
    cfg = body_gravitycomp_cfg(_const_noise(noise_value), inst_sel=body_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_gravcomp + noise_value
    assert np.isclose(model.body_gravcomp[body_id], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.body_gravcomp[body_id], expected)
