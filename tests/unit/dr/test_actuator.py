from __future__ import annotations

import mujoco
import numpy as np
import pytest

from contact_gym.dr import ModelParamRandomizer
from contact_gym.dr.actuator import (
    actuator_armature_cfg,
    actuator_linear_damping_cfg,
    actuator_motor_gain_cfg,
    actuator_position_kp_cfg,
    actuator_position_neg_kv_cfg,
)
from contact_gym.utils.noise import ConstantSampler, Noise

_ACTUATOR_DR_XML = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" armature="0.01" damping="0.10"/>
      <geom name="geom1" type="box" size="0.05 0.05 0.05"/>
      <body name="body2" pos="0.1 0 0">
        <joint name="joint2" type="hinge" axis="0 1 0" armature="0.02" damping="0.20"/>
        <geom name="geom2" type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrlrange="-1 1"/>
    <position name="pos1" joint="joint2" kp="50" kv="5" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_ACTUATOR_DR_XML)


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _const_noise(value: float, dtype=np.float64) -> Noise:
    return Noise(ConstantSampler(value, dtype=dtype), operation="add")


# region motor_gain


@pytest.mark.unit
def test_actuator_motor_gain_cfg(model, data, rng):
    motor_id = model.actuator("motor1").id
    nominal_gain = model.actuator_gainprm[motor_id, 0].copy()

    noise_value = 0.3
    cfg = actuator_motor_gain_cfg(_const_noise(noise_value), inst_sel=motor_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_gain + noise_value
    assert np.isclose(model.actuator_gainprm[motor_id, 0], expected)

    # Test that cached nominal value remains unchaged after sampling
    randomizer(model, data, rng)
    assert np.isclose(model.actuator_gainprm[motor_id, 0], expected)


# region position_kp


@pytest.mark.unit
def test_actuator_position_kp_cfg(model, data, rng):
    pos_id = model.actuator("pos1").id
    nominal_kp = model.actuator_gainprm[pos_id, 0].copy()
    # sanity check before mutating
    assert np.isclose(model.actuator_biasprm[pos_id, 1], -nominal_kp)

    noise_value = 10.0
    cfg = actuator_position_kp_cfg(_const_noise(noise_value), inst_sel=pos_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected_kp = nominal_kp + noise_value
    assert np.isclose(model.actuator_gainprm[pos_id, 0], expected_kp)
    assert np.isclose(model.actuator_biasprm[pos_id, 1], -expected_kp)

    randomizer(model, data, rng)
    assert np.isclose(model.actuator_gainprm[pos_id, 0], expected_kp)
    assert np.isclose(model.actuator_biasprm[pos_id, 1], -expected_kp)


# region position_neg_kv


@pytest.mark.unit
def test_actuator_position_neg_kv_cfg(model, data, rng):
    pos_id = model.actuator("pos1").id
    nominal_neg_kv = model.actuator_biasprm[pos_id, 2].copy()
    assert nominal_neg_kv < 0

    noise_value = -1.5  # noise added to negative kv
    cfg = actuator_position_neg_kv_cfg(_const_noise(noise_value), inst_sel=pos_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_neg_kv + noise_value
    assert np.isclose(model.actuator_biasprm[pos_id, 2], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.actuator_biasprm[pos_id, 2], expected)


# region linear_damping


@pytest.mark.unit
def test_actuator_linear_damping_cfg(model, data, rng):
    motor_id = model.actuator("motor1").id
    nominal_damping = model.actuator_damping[motor_id].copy()

    noise_value = 0.05
    cfg = actuator_linear_damping_cfg(_const_noise(noise_value), inst_sel=motor_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_damping + noise_value
    assert np.isclose(model.actuator_damping[motor_id], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.actuator_damping[motor_id], expected)


# region armature


@pytest.mark.unit
def test_actuator_armature_cfg(model, data, rng):
    pos_id = model.actuator("pos1").id
    nominal_armature = model.actuator_armature[pos_id].copy()

    noise_value = 0.01
    cfg = actuator_armature_cfg(_const_noise(noise_value), inst_sel=pos_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_armature + noise_value
    assert np.isclose(model.actuator_armature[pos_id], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.actuator_armature[pos_id], expected)
