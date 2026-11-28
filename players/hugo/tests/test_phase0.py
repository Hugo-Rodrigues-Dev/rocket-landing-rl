"""Vérifie le lien entre les commandes de référence et l'expérience PPO/SAC."""

import numpy as np
import pytest

from rocket_hugo.env import RocketLandingEnv
from rocket_hugo.phase0 import DEVELOPMENT_SEEDS, OFFICIAL_SEEDS, Phase0Experiment
from rocket_hugo.phase0_pilot import pilot_action


# Une correction nulle doit reproduire exactement la transition du pilote seul.
def test_zero_residual_reproduces_pilot_transition():
    raw = RocketLandingEnv()
    experiment = Phase0Experiment("residual")
    obs, _ = raw.reset(seed=7200)
    experiment.reset(seed=7200)
    raw_obs, _, raw_terminated, raw_truncated, raw_info = raw.step(pilot_action(obs))
    exp_obs, _, exp_terminated, exp_truncated, exp_info = experiment.step(np.zeros(3))
    np.testing.assert_array_equal(raw_obs, exp_obs)
    assert (raw_terminated, raw_truncated) == (exp_terminated, exp_truncated)
    assert raw_info["outcome"] == exp_info["outcome"]
    assert exp_info["correction_norm"] == 0


# Les commandes directes restent disponibles et passent au simulateur sans transformation.
@pytest.mark.parametrize("action", [(-1, 0, 0), (0.2, -0.3, 0.4)])
def test_direct_actions_are_forwarded(action):
    raw, experiment = RocketLandingEnv(), Phase0Experiment("direct")
    raw.reset(seed=7200)
    experiment.reset(seed=7200)
    np.testing.assert_array_equal(raw.step(action)[0], experiment.step(action)[0])


# Le développement ne doit pas réutiliser les graines réservées au contrat.
def test_development_seeds_are_separate_from_official_seeds():
    assert set(DEVELOPMENT_SEEDS).isdisjoint(OFFICIAL_SEEDS)
    assert len(OFFICIAL_SEEDS) == 100


# Le corps retourné heurte le pont ; hors du pont, le passage sous sa hauteur est une chute à l'eau.
@pytest.mark.parametrize(
    ("position", "quaternion", "outcome"),
    [((0, 0, 5.9), (0, 1, 0, 0), "crash"), ((20, 0, 6.3), (1, 0, 0, 0), "water")],
)
def test_phase0_physical_failure_is_terminal(position, quaternion, outcome):
    env = RocketLandingEnv()
    env.reset(seed=7200)
    env.data.qpos[:] = (*position, *quaternion)
    env.data.qvel[:] = 0.0
    _, reward, terminated, truncated, info = env.step((-1, 0, 0))
    assert terminated and not truncated
    assert reward == -100 and info["outcome"] == outcome
