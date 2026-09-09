import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import contact_gym  # noqa: F401
from contact_gym.envs import EdgeGraspEnvCfg
from contact_gym.objects import ALL_OBJECTS
from contact_gym.robots import ALL_GRIPPERS, ALL_ROBOTS

SceneCfg = EdgeGraspEnvCfg.SceneCfg


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
    assert isinstance(obs, dict)
    assert all(isinstance(o, np.ndarray) and o.dtype == np.float32 for o in obs.values())
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
    assert isinstance(obs, dict)
    assert all(isinstance(o, np.ndarray) and o.dtype == np.float32 for o in obs.values())
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert len(info) == 1
    assert "is_success" in info
    assert np.isfinite(info["is_success"])


# region Reward


@pytest.mark.unit
def test_compute_reward_sparse():
    rng = np.random.default_rng(0)

    def sample_goal() -> np.ndarray:
        goal = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
        goal[[3, 5]] = rng.integers(0, 2)
        return goal

    sparse_env = make_env(reward_type="sparse")
    sparse_env.reset(seed=0)
    state_rew = -0.01
    terminated = False
    achieved_goal = sample_goal()
    achieved_goal[4] = state_rew
    achieved_goal[5] = float(terminated)

    # Achieved = desired goal -> sparse guidance term is 0
    reward = sparse_env.unwrapped.compute_reward(achieved_goal, achieved_goal.copy(), {})
    assert isinstance(reward, np.ndarray)
    assert reward.shape == ()
    assert np.isfinite(reward)
    np.testing.assert_allclose(reward, state_rew)

    # Achieved goal far from desired -> sparse guidance term is -1
    desired_far = achieved_goal.copy()
    desired_far[:3] += 10.0
    reward = sparse_env.unwrapped.compute_reward(achieved_goal, desired_far, {})
    np.testing.assert_allclose(reward, -1.0 + state_rew)

    # Purely functional -> reward is the same for identical args before/after stepping
    sparse_env.step(sparse_env.action_space.sample())
    reward_after_step = sparse_env.unwrapped.compute_reward(achieved_goal, desired_far, {})
    np.testing.assert_allclose(reward, reward_after_step)

    # Prepare batched inputs
    batch_size = 6
    achieved_goals = np.stack([sample_goal() for _ in range(batch_size)])
    desired_goals = np.stack([sample_goal() for _ in range(batch_size)])

    # Batch -> reward = reward -> batch
    rewards = sparse_env.unwrapped.compute_reward(achieved_goals, desired_goals, {})
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (batch_size,)
    assert np.all(np.isfinite(rewards))
    single_rewards = np.array(
        [
            sparse_env.unwrapped.compute_reward(achieved_goals[i], desired_goals[i], {})
            for i in range(batch_size)
        ]
    )
    np.testing.assert_allclose(rewards, single_rewards, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_compute_reward_dense():
    rng = np.random.default_rng(0)

    def sample_goal() -> np.ndarray:
        goal = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
        goal[[3, 5]] = rng.integers(0, 2)
        return goal

    achieved_goal = sample_goal()
    state_rew = -0.01
    terminated = False
    achieved_goal[4] = state_rew
    achieved_goal[5] = float(terminated)

    dense_env = make_env(reward_type="dense")
    dense_env.reset(seed=0)
    weights = dense_env.unwrapped.cfg.weights
    mdata = dense_env.unwrapped._mdata
    task_cfg = dense_env.unwrapped.cfg.task_cfg
    state_rew = -0.01
    terminated = False

    # tgt_flag = 0 (table target) -> table term disabled regardless of object height
    achieved_table = sample_goal()
    achieved_table[4] = state_rew
    achieved_table[5] = float(terminated)
    desired_same = achieved_table.copy()
    desired_far = achieved_table.copy()
    desired_far[0] += 10.0
    desired_same[3] = desired_far[3] = 0.0

    reward_table_same = dense_env.unwrapped.compute_reward(achieved_table, desired_same, {})
    reward_table_far = dense_env.unwrapped.compute_reward(achieved_table, desired_far, {})
    np.testing.assert_allclose(reward_table_same, state_rew, atol=1e-5)
    np.testing.assert_allclose(
        reward_table_far, weights.tgt_dist * -np.tanh(10.0) + state_rew, atol=1e-4
    )

    # tgt_flag = 1 and object on the table -> target term forced to 1 and table term active
    table_pos = dense_env.unwrapped.data.xpos[mdata.table_body_id]
    achieved_lift = np.array(
        [table_pos[0] + 0.05, table_pos[1], mdata.table_height, 0.0, state_rew, 0.0],
        dtype=np.float32,
    )
    desired_lift = np.array(
        [achieved_lift[0], achieved_lift[1], achieved_lift[2] + 0.2, 1.0, 0.0, 0.0],
        dtype=np.float32,
    )
    reward_lift = dense_env.unwrapped.compute_reward(achieved_lift, desired_lift, {})

    obj_dist = np.sqrt(np.sum((achieved_lift[:2] - table_pos[:2]) ** 2)) / mdata.table_extent
    expected_tbl_rew = np.tanh(task_cfg.dist_mult * obj_dist) - 1
    expected_reward = weights.tgt_dist * -1.0 + weights.tbl_dist * expected_tbl_rew + state_rew
    np.testing.assert_allclose(reward_lift, expected_reward, atol=1e-5)

    # tgt_flag = 1 and object off the table -> target term active and table term forced to 0
    achieved_lift = np.array(
        [table_pos[0] + 0.05, table_pos[1], mdata.table_height + 0.05, 1.0, state_rew, 0.0],
        dtype=np.float32,
    )
    reward_lift = dense_env.unwrapped.compute_reward(achieved_lift, desired_lift, {})

    tgt_dist = np.sqrt(np.sum((achieved_lift[:3] - desired_lift[:3]) ** 2))
    expected_tgt_rew = -np.tanh(tgt_dist)
    expected_reward = weights.tgt_dist * expected_tgt_rew + state_rew
    np.testing.assert_allclose(reward_lift, expected_reward, atol=1e-5)

    # Prepare batched inputs
    batch_size = 6
    achieved_goals = np.stack([sample_goal() for _ in range(batch_size)])
    desired_goals = np.stack([sample_goal() for _ in range(batch_size)])

    # Batch -> reward = reward -> batch
    rewards = dense_env.unwrapped.compute_reward(achieved_goals, desired_goals, {})
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (batch_size,)
    assert np.all(np.isfinite(rewards))
    single_rewards = np.array(
        [
            dense_env.unwrapped.compute_reward(achieved_goals[i], desired_goals[i], {})
            for i in range(batch_size)
        ]
    )
    np.testing.assert_allclose(rewards, single_rewards, rtol=1e-5, atol=1e-6)


# region Terminated


@pytest.mark.unit
def test_compute_terminated(env: gym.Env):
    rng = np.random.default_rng(0)
    env.reset(seed=0)

    def sample_goal() -> np.ndarray:
        goal = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
        goal[[3, 5]] = rng.integers(0, 2)
        return goal

    # Test terminated is goal-independent
    achieved_goal, desired_goal = sample_goal(), sample_goal()
    for terminated in (True, False):
        achieved_goal[5] = float(terminated)
        result = env.unwrapped.compute_terminated(achieved_goal, desired_goal, {})
        assert isinstance(result, np.ndarray)
        assert bool(result) == terminated

    # Purely functional -> terminated is the same for identical args before/after stepping
    env.step(env.action_space.sample())
    other_goal = sample_goal()
    other_goal[5] = float(True)
    result_other_goal = env.unwrapped.compute_terminated(other_goal, other_goal, {})
    assert bool(result_other_goal) is True

    # Prepare batched inputs
    batch_size = 6
    achieved_goals = np.stack([sample_goal() for _ in range(batch_size)])
    desired_goals = np.stack([sample_goal() for _ in range(batch_size)])

    # Batch -> reward = reward -> batch
    result_batch = env.unwrapped.compute_terminated(achieved_goals, desired_goals, {})
    assert isinstance(result_batch, np.ndarray)
    assert result_batch.shape == (batch_size,)
    np.testing.assert_array_equal(result_batch, achieved_goals[:, 5].astype(np.bool_))
    single_results = np.array(
        [
            env.unwrapped.compute_terminated(achieved_goals[i], desired_goals[i], {})
            for i in range(batch_size)
        ]
    )
    np.testing.assert_array_equal(result_batch, single_results)


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

    np.testing.assert_allclose(obs1["observation"], obs2["observation"])
    np.testing.assert_allclose(obs1["observation"], obs3["observation"])


# region Random Rollout


@pytest.mark.unit
def test_random_rollout(env: gym.Env):
    # Environment remains stable with random rollout
    env.reset(seed=0)
    for _ in range(25):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        assert np.all(np.isfinite(np.concatenate(list(obs.values()))))
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
    cfg = EdgeGraspEnvCfg(scene_cfg=SceneCfg(robot=robot, gripper=gripper, object=object))
    env = make_env(cfg=cfg)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.all(np.isfinite(np.concatenate(list(obs.values()))))
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
