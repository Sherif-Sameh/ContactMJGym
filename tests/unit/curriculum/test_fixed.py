import gymnasium as gym
import numpy as np
import pytest

import contact_gym  # noqa: F401
import contact_gym.dr as dr
from contact_gym.curriculum.fixed import (
    CosineAnnealingCurriculumTerm,
    FixedCurriculumTerm,
    LinearCurriculumTerm,
)
from contact_gym.utils.noise import Noise, UniformSampler

TERM_CLASSES = [LinearCurriculumTerm, CosineAnnealingCurriculumTerm]
DR_MIN_PATH = "domain_randomizers[0].cfg.noise.sampler.min"
DR_MAX_PATH = "domain_randomizers[0].cfg.noise.sampler.max"
REWARD_QVEL_PATH = "_rcfg.weights.qvel_l2"
PATHS = [DR_MIN_PATH, DR_MAX_PATH, REWARD_QVEL_PATH]

START = [1.0, np.ones(7), 0.0]
END = [0.8, np.full((7,), 1.2), -1e-2]
END_STEP = 50_000


@pytest.fixture
def env() -> gym.Env:
    noise = Noise(UniformSampler(1.0, np.ones(7)), operation="scale")
    rand = dr.ModelParamRandomizer(dr.actuator.actuator_position_kp_cfg(noise, inst_sel=slice(7)))
    return gym.make("contact_gym/EdgeGrasp-v0", domain_randomizers=[rand]).unwrapped


def _get_values(env: gym.Env) -> list[float | np.ndarray]:
    """Read current values directly off the env, bypassing cached addresses."""
    return [
        env.domain_randomizers[0].cfg.noise.sampler.min,
        env.domain_randomizers[0].cfg.noise.sampler.max,
        env._rcfg.weights.qvel_l2,
    ]


def _to_flat(values: list[float | np.ndarray]) -> np.ndarray:
    return np.concatenate([np.ravel(np.atleast_1d(np.asarray(v))) for v in values])


# region Shared


@pytest.mark.unit
@pytest.mark.parametrize("term_cls", TERM_CLASSES)
def test_term_setup(env: gym.Env, term_cls: type[FixedCurriculumTerm]):
    term = term_cls(paths=PATHS, end=END, end_step=END_STEP, start=START)
    term.setup(env)
    # Test parameter address resolution
    term.setup(env)
    assert term.addrs is not None
    assert len(term.addrs) == len(PATHS)
    # Test explicit start is not overwritten
    np.testing.assert_allclose(term.start, _to_flat(START))
    # Test lazy start loading from env
    term = term_cls(paths=PATHS, end=END, end_step=END_STEP)
    term.setup(env)
    np.testing.assert_allclose(term.start, _to_flat(_get_values(env)))


@pytest.mark.unit
@pytest.mark.parametrize("term_cls", TERM_CLASSES)
def test_term_call(env: gym.Env, term_cls: type[FixedCurriculumTerm]):
    # Test call at step=0 (i.e. step=start_step)
    term = term_cls(paths=PATHS, end=END, end_step=END_STEP, start=START)
    term(env, step=0)
    np.testing.assert_allclose(_to_flat(_get_values(env)), _to_flat(START))
    # Test call at step=end_step
    term(env, step=END_STEP)
    np.testing.assert_allclose(_to_flat(_get_values(env)), _to_flat(END), atol=1e-8)
    # Test call beyond end_step
    term(env, step=END_STEP + 5000)
    np.testing.assert_allclose(_to_flat(_get_values(env)), _to_flat(END), atol=1e-8)
    # Test address validity after mutiple calls
    for step in range(0, 10_000, 2000):
        params = term(env, step=step)
    np.testing.assert_allclose(
        _to_flat(_get_values(env)), _to_flat(list(params.values())), atol=1e-8
    )
    # Test return dict keys match paths
    assert list(params.keys()) == PATHS


# region Linear


@pytest.mark.unit
def test_linear_term(env: gym.Env):
    term = LinearCurriculumTerm(paths=PATHS, end=END, end_step=END_STEP, start=START)
    # Test mid-point
    mid_step = (END_STEP) // 2
    params = term(env, step=mid_step)
    expected = [(s + e) / 2.0 for s, e in zip(START, END)]
    for val, exp in zip(params.values(), expected):
        np.testing.assert_allclose(val, exp, atol=1e-8)
    # Test constant rate of change
    steps = np.linspace(0, END_STEP, 6).astype(int)
    values = np.array([_to_flat(list(term(env, step=s).values())) for s in steps])
    diffs = np.diff(values, axis=0)
    np.testing.assert_allclose(diffs, np.tile(diffs[0], (5, 1)), atol=1e-6)


# region Cosine


@pytest.mark.unit
def test_cosine_annealing_term(env: gym.Env):
    term = CosineAnnealingCurriculumTerm(paths=PATHS, end=END, end_step=END_STEP, start=START)
    # Test mid-point
    mid_step = (END_STEP) // 2
    params = term(env, step=mid_step)
    expected = [(s + e) / 2.0 for s, e in zip(START, END)]
    for val, exp in zip(params.values(), expected):
        np.testing.assert_allclose(val, exp, atol=1e-8)
    # Test arbitrary step
    step = END_STEP // 4
    t = step / END_STEP
    factor = 0.5 * (1.0 + np.cos(np.pi * t))
    expected = [e + factor * (s - e) for s, e in zip(START, END)]
    params = term(env, step=step)
    for val, exp in zip(params.values(), expected):
        np.testing.assert_allclose(val, exp, atol=1e-8)
