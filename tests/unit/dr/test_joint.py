from __future__ import annotations

import mujoco
import numpy as np
import pytest

from contact_gym.dr import ModelParamRandomizer
from contact_gym.dr.joint import (
    joint_armature_cfg,
    joint_frictionloss_cfg,
    joint_linear_damping_cfg,
    joint_linear_stiffness_cfg,
)
from contact_gym.utils.noise import ConstantSampler, Noise

_JOINT_DR_XML = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 0">
      <joint name="hinge1" type="hinge" axis="0 0 1" stiffness="5.0" frictionloss="0.10" armature="0.010" damping="0.20"/>
      <geom name="geom1" type="box" size="0.05 0.05 0.05"/>
      <body name="body2" pos="0.1 0 0">
        <joint name="ball1" type="ball" frictionloss="0.05" armature="0.020" damping="0.30"/>
        <geom name="geom2" type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_JOINT_DR_XML)


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _const_noise(value, dtype=np.float64) -> Noise:
    return Noise(ConstantSampler(value, dtype=dtype), operation="add")


# region linear_stiffness


@pytest.mark.unit
def test_joint_linear_stiffness_cfg(model, data, rng):
    hinge_id = model.joint("hinge1").id
    nominal_stiffness = model.jnt_stiffness[hinge_id].copy()

    noise_value = 0.5
    cfg = joint_linear_stiffness_cfg(_const_noise(noise_value), inst_sel=hinge_id)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_stiffness + noise_value
    assert np.isclose(model.jnt_stiffness[hinge_id], expected)

    # Test that cached nominal value remains unchaged after sampling
    randomizer(model, data, rng)
    assert np.isclose(model.jnt_stiffness[hinge_id], expected)


# region frictionloss


@pytest.mark.unit
def test_joint_frictionloss_cfg(model, data, rng):
    hinge_id = model.joint("hinge1").id
    dof_adr = model.jnt_dofadr[hinge_id]
    nominal_frictionloss = model.dof_frictionloss[dof_adr].copy()

    noise_value = 0.02
    cfg = joint_frictionloss_cfg(_const_noise(noise_value), inst_sel=hinge_id, model=model)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_frictionloss + noise_value
    assert np.isclose(model.dof_frictionloss[dof_adr], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.dof_frictionloss[dof_adr], expected)


# region armature


@pytest.mark.unit
def test_joint_armature_cfg(model, data, rng):
    hinge_id = model.joint("hinge1").id
    dof_adr = model.jnt_dofadr[hinge_id]
    nominal_armature = model.dof_armature[dof_adr].copy()

    noise_value = 0.005
    cfg = joint_armature_cfg(_const_noise(noise_value), inst_sel=hinge_id, model=model)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_armature + noise_value
    assert np.isclose(model.dof_armature[dof_adr], expected)

    randomizer(model, data, rng)
    assert np.isclose(model.dof_armature[dof_adr], expected)


# region linear_damping


@pytest.mark.unit
def test_joint_linear_damping_cfg(model, data, rng):
    ball_id = model.joint("ball1").id
    dof_adr = model.jnt_dofadr[ball_id]
    dof_slice = slice(dof_adr, dof_adr + 3)
    nominal_damping = model.dof_damping[dof_slice].copy()

    noise_value = np.array([0.01, -0.02, 0.03])
    cfg = joint_linear_damping_cfg(_const_noise(noise_value), inst_sel=ball_id, model=model)
    randomizer = ModelParamRandomizer(cfg, model)

    randomizer(model, data, rng)
    expected = nominal_damping + noise_value
    assert np.allclose(model.dof_damping[dof_slice], expected)

    randomizer(model, data, rng)
    assert np.allclose(model.dof_damping[dof_slice], expected)
