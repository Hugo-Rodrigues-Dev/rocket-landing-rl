"""Vérifications du curriculum : la difficulté ne change que le départ, entre facile et contrat."""

import numpy as np
import pytest

from rocket_mathis.config import EASY_START, EnvConfig
from rocket_mathis.env import RocketLandingEnv


def test_difficulty_bounds():
    full = EnvConfig()
    assert full.at_difficulty(1.0) == full
    easy = full.at_difficulty(0.0)
    for key, value in EASY_START.items():
        assert getattr(easy, key) == pytest.approx(value)
    assert full.at_difficulty(-1) == easy and full.at_difficulty(2) == full


def test_difficulty_only_changes_the_start():
    full, half = EnvConfig(phase=1), EnvConfig(phase=1).at_difficulty(0.5)
    changed = {k for k in full.__dataclass_fields__ if getattr(full, k) != getattr(half, k)}
    assert changed <= set(EASY_START)
    assert half.altitude == pytest.approx(((15 + 80) / 2, (25 + 120) / 2))


def test_env_applies_difficulty_at_next_reset():
    env = RocketLandingEnv()
    env.set_difficulty(0.0)
    altitudes = [env.reset(seed=s)[0][2] * 100 for s in range(30)]
    assert max(altitudes) <= 25.0 + 1e-3
    env.set_difficulty(1.0)
    altitudes = [env.reset(seed=s)[0][2] * 100 for s in range(30)]
    assert min(altitudes) >= 80.0 - 1e-3
    assert np.mean(altitudes) > 90
