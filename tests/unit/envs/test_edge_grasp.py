import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import contact_gym  # noqa: F401
from contact_gym.objects import ALL_OBJECTS
from contact_gym.robots import ALL_GRIPPERS, ALL_ROBOTS


@pytest.fixture
def env() -> gym.Env:
    return gym.make("contact_gym/EdgeGrasp-v0")


def make_env(**kwargs) -> gym.Env:
    return gym.make("contact_gym/EdgeGrasp-v0", **kwargs)


# region Gym API


@pytest.mark.unit
def test_gymnasium_api_compliance(env):
    check_env(env.unwrapped, skip_render_check=True)


# region Reset


@pytest.mark.unit
def test_reset(env: gym.Env):
    # Return values/types valid
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    # Accepts different seeds
    env.reset(seed=1)
    env.reset(seed=1000)
    env.reset()

    # Successive resets without step()
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    # Options apply initial state and ctrl overrides
    data = env.unwrapped.data
    qpos = np.zeros_like(data.qpos) + np.random.normal(scale=1e-3, size=data.qpos.shape)
    qvel = np.zeros_like(data.qvel) + np.random.normal(scale=1e-3, size=data.qvel.shape)
    ctrl = np.zeros_like(data.ctrl) + np.random.normal(scale=1e-3, size=data.ctrl.shape)
    options = {"qpos": qpos, "qvel": qvel, "ctrl": ctrl}
    env.reset(seed=0, options=options)

    np.testing.assert_allclose(data.qpos, qpos)
    np.testing.assert_allclose(data.qvel, qvel)
    np.testing.assert_allclose(data.ctrl, ctrl)

    # Options don't share memory with data arrays
    qpos += 1.0
    qvel += 1.0
    ctrl += 1.0

    assert not np.allclose(env.unwrapped.data.qpos, qpos)
    assert not np.allclose(env.unwrapped.data.qvel, qvel)
    assert not np.allclose(env.unwrapped.data.ctrl, ctrl)


# region Step


@pytest.mark.unit
def test_step(env: gym.Env):
    # Return values/types valid
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Debug info flag set -> info dict contains reward terms
    env = make_env(debug_info=True)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info, dict)
    assert len(info) > 0
    for term_name, term_value in info.items():
        assert np.isfinite(term_value), f"reward term {term_name} is not finite"


# region Determinism


@pytest.mark.unit
def test_determinism():
    env1 = make_env()
    env2 = make_env()

    # Two environments seeded identically should sample the same actions
    env1.reset(seed=0)
    env2.reset(seed=0)
    for _ in range(5):
        act1 = env1.action_space.sample()
        act2 = env2.action_space.sample()
        np.testing.assert_array_equal(act1, act2)

    # Deterministic observations without explicit domain randomization
    obs1, _ = env1.reset(seed=0)
    obs2, _ = env1.reset(seed=1)
    obs3, _ = env1.reset()

    np.testing.assert_allclose(obs1, obs2)
    np.testing.assert_allclose(obs1, obs3)


# region Random Rollout


@pytest.mark.unit
def test_random_rollout(env: gym.Env):
    # Environment remains stable with random rollout
    env.reset(seed=0)
    for _ in range(25):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.all(np.isfinite(env.unwrapped.data.qpos))
        assert np.all(np.isfinite(env.unwrapped.data.qvel))

        if terminated or truncated:
            env.reset()


# region Initial State


@pytest.mark.unit
def test_initial_state(env: gym.Env):
    # No immediate termination after a reset
    env.reset(seed=0)
    data = env.unwrapped.data
    action = np.copy(data.ctrl)  # don't change initial state
    _, _, terminated, _, _ = env.step(action)
    assert not terminated

    # No huge contact forces (possible peneration on reset)
    max_force = np.max(np.abs(data.cfrc_ext)) if data.cfrc_ext.size else 0.0
    assert max_force < 1e3


# region Asset Combos


@pytest.mark.unit
@pytest.mark.parametrize("robot", ALL_ROBOTS)
@pytest.mark.parametrize("gripper", ALL_GRIPPERS)
@pytest.mark.parametrize("object", ALL_OBJECTS)
def test_asset_combos(robot, gripper, object):
    # Environment works with all possible asset combinations
    env = make_env(robot=robot, gripper=gripper, object=object)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset()


# region Rendering


@pytest.mark.unit
def test_rgb_array_render():
    # RGB array rendering works and returns valid frames
    height, width = 320, 480
    env = make_env(render_mode="rgb_array", renderer_kwargs={"height": height, "width": width})
    env.reset(seed=0)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape == (height, width, 3)
    assert frame.dtype == np.uint8
