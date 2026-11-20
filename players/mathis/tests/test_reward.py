"""Vérifications de la récompense et du contrôleur PD de référence."""

import numpy as np
import pytest

from rocket_mathis.baseline_pd import PDController
from rocket_mathis.config import EnvConfig
from rocket_mathis.env import RocketLandingEnv
from rocket_mathis.evaluate import evaluate, random_policy

from .test_env import OFF, place, run


@pytest.fixture
def env():
    e = RocketLandingEnv()
    yield e
    e.close()


def test_landing_is_rewarded(env):
    place(env, base_height=0.02)
    _, _, _, info = run(env, OFF)
    assert info["outcome"] == "landed"
    assert 100.0 <= info["reward_terms"]["final"] <= 150.0


def test_failure_is_penalized(env):
    place(env, base_height=0.5, vz=-4.0)
    _, _, _, info = run(env, OFF)
    assert info["outcome"] == "hard_landing"
    assert info["reward_terms"]["final"] == -100.0


def test_shaping_telescopes(env):
    """Σ γ^t · guidage = γ^T·Φ(s_T) − Φ(s0) : seul compte l'état de fin, pas le chemin."""
    gamma = env.config.gamma
    pd = PDController()
    obs, _ = env.reset(seed=4)
    phi0 = env._phi  # Φ(s0)
    total, t, done = 0.0, 0, False
    while not done:
        obs, _, terminated, truncated, info = env.step(pd(obs))
        total += gamma**t * info["reward_terms"]["shaping"]
        t += 1
        done = terminated or truncated
    assert terminated
    assert total == pytest.approx(gamma**t * env._phi - phi0, rel=1e-9)


def test_pd_controller_beats_random():
    seeds = range(10)
    pd = evaluate(PDController(), EnvConfig(phase=0), seeds)
    rnd = evaluate(random_policy(), EnvConfig(phase=0), seeds)
    assert pd["success_rate"] >= 0.9
    assert rnd["success_rate"] == 0.0
    assert pd["return_mean"] > rnd["return_mean"] + 100
    assert np.abs(pd["impact_speed_mean"]) < 2.0
