from __future__ import annotations

import mujoco
import numpy as np
import pytest

from contact_gym.dr import ModelParamRandomizer
from contact_gym.dr.geom import (
    geom_adhesion_cfg,
    geom_friction_cfg,
    geom_rgb_cfg,
    geom_rgba_cfg,
    geom_sliding_friction_cfg,
    geom_sliding_torsional_friction_cfg,
    geom_solimp_impedance_cfg,
    geom_solref_cfg,
    geom_surface_angvel_cfg,
    geom_surface_linvel_cfg,
)
from contact_gym.utils.noise import ConstantSampler, Noise

_GEOM_DR_XML = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 0">
      <geom
        name="geom1"
        type="box"
        size="0.05 0.05 0.05"
        friction="0.8 0.05 0.001"
        rgba="0.8 0.1 0.1 0.9"
        solref="0.01 0.9"
        solimp="0.85 0.95 0.002 0.4 2"
        surfacevel="0.01 0 0 0 0 0.02"
        adhesion="0.025"
      />
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_GEOM_DR_XML)


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _const_noise(value, dtype=np.float64) -> Noise:
    return Noise(ConstantSampler(value, dtype=dtype), operation="add")


# region solref


@pytest.mark.unit
def test_geom_solref_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_solref = model.geom_solref[geom_id].copy()

    noise_value = np.array([0.001, 0.02])
    cfg = geom_solref_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_solref + noise_value
    assert np.allclose(model.geom_solref[geom_id], expected)

    # Test that cached nominal value remains unchaged after sampling
    randomizer(model, data, rng)
    assert np.allclose(model.geom_solref[geom_id], expected)


# region solimp_imp


@pytest.mark.unit
def test_geom_solimp_impedance_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_solimp = model.geom_solimp[geom_id].copy()

    noise_value = np.array([0.02, -0.01])
    cfg = geom_solimp_impedance_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected_impedance = nominal_solimp[:2] + noise_value
    assert np.allclose(model.geom_solimp[geom_id, :2], expected_impedance)
    assert np.allclose(model.geom_solimp[geom_id, 2:], nominal_solimp[2:])

    randomizer(model, data, rng)
    assert np.allclose(model.geom_solimp[geom_id, :2], expected_impedance)
    assert np.allclose(model.geom_solimp[geom_id, 2:], nominal_solimp[2:])


# region sliding_friction


@pytest.mark.unit
def test_geom_sliding_friction_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_friction = model.geom_friction[geom_id].copy()

    noise_value = 0.05
    cfg = geom_sliding_friction_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected_sliding = nominal_friction[0] + noise_value
    assert np.isclose(model.geom_friction[geom_id, 0], expected_sliding)
    assert np.allclose(model.geom_friction[geom_id, 1:], nominal_friction[1:])

    randomizer(model, data, rng)
    assert np.isclose(model.geom_friction[geom_id, 0], expected_sliding)
    assert np.allclose(model.geom_friction[geom_id, 1:], nominal_friction[1:])


# region torsional_friction


@pytest.mark.unit
def test_geom_sliding_torsional_friction_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_friction = model.geom_friction[geom_id].copy()

    noise_value = np.array([0.05, 0.002])
    cfg = geom_sliding_torsional_friction_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_friction[:2] + noise_value
    assert np.allclose(model.geom_friction[geom_id, :2], expected)
    assert np.isclose(model.geom_friction[geom_id, 2], nominal_friction[2])

    randomizer(model, data, rng)
    assert np.allclose(model.geom_friction[geom_id, :2], expected)
    assert np.isclose(model.geom_friction[geom_id, 2], nominal_friction[2])


# region friction


@pytest.mark.unit
def test_geom_friction_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_friction = model.geom_friction[geom_id].copy()

    noise_value = np.array([0.05, 0.002, 0.0001])
    cfg = geom_friction_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_friction + noise_value
    assert np.allclose(model.geom_friction[geom_id], expected)

    randomizer(model, data, rng)
    assert np.allclose(model.geom_friction[geom_id], expected)


# region surface_linvel


@pytest.mark.unit
def test_geom_surface_linvel_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_surfacevel = model.geom_surfacevel[geom_id].copy()

    noise_value = np.array([0.01, 0.02, -0.01])
    cfg = geom_surface_linvel_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected_linvel = nominal_surfacevel[:3] + noise_value
    assert np.allclose(model.geom_surfacevel[geom_id, :3], expected_linvel)
    assert np.allclose(model.geom_surfacevel[geom_id, 3:], nominal_surfacevel[3:])

    randomizer(model, data, rng)
    assert np.allclose(model.geom_surfacevel[geom_id, :3], expected_linvel)
    assert np.allclose(model.geom_surfacevel[geom_id, 3:], nominal_surfacevel[3:])


# region surface_angvel


@pytest.mark.unit
def test_geom_surface_angvel_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_surfacevel = model.geom_surfacevel[geom_id].copy()

    noise_value = np.array([0.0, 0.0, 0.05])
    cfg = geom_surface_angvel_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected_angvel = nominal_surfacevel[3:] + noise_value
    assert np.allclose(model.geom_surfacevel[geom_id, 3:], expected_angvel)
    assert np.allclose(model.geom_surfacevel[geom_id, :3], nominal_surfacevel[:3])

    randomizer(model, data, rng)
    assert np.allclose(model.geom_surfacevel[geom_id, 3:], expected_angvel)
    assert np.allclose(model.geom_surfacevel[geom_id, :3], nominal_surfacevel[:3])


# region adhesion


@pytest.mark.unit
def test_geom_adhesion_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_adhesion = model.geom_adhesion[geom_id].copy()

    noise_value = 0.1
    cfg = geom_adhesion_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_adhesion + noise_value
    assert np.isclose(model.geom_adhesion[geom_id], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.geom_adhesion[geom_id], expected)


# region rgb


@pytest.mark.unit
def test_geom_rgb_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_rgba = model.geom_rgba[geom_id].copy()

    noise_value = np.array([0.05, -0.02, 0.03])
    cfg = geom_rgb_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected_rgb = nominal_rgba[:3] + noise_value
    assert np.allclose(model.geom_rgba[geom_id, :3], expected_rgb)
    assert np.isclose(model.geom_rgba[geom_id, 3], nominal_rgba[3])

    randomizer(model, data, rng)
    assert np.allclose(model.geom_rgba[geom_id, :3], expected_rgb)
    assert np.isclose(model.geom_rgba[geom_id, 3], nominal_rgba[3])


# region rgba


@pytest.mark.unit
def test_geom_rgba_cfg(model, data, rng):
    geom_id = model.geom("geom1").id
    nominal_rgba = model.geom_rgba[geom_id].copy()

    noise_value = np.array([0.05, -0.02, 0.03, -0.1])
    cfg = geom_rgba_cfg(_const_noise(noise_value), inst_sel=geom_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_rgba + noise_value
    assert np.allclose(model.geom_rgba[geom_id], expected)

    randomizer(model, data, rng)
    assert np.allclose(model.geom_rgba[geom_id], expected)
