"""Vérifie le contrat d'interface Gymnasium et les principales transitions du simulateur 3D."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocket_hugo.demo_vertical import braking_action
from rocket_hugo.env import RocketLandingEnv
from rocket_hugo.vertical import VerticalState


def test_gymnasium_api_and_seeded_start():
    """Vérifie la conformité Gymnasium et le caractère reproductible du départ."""
    env = RocketLandingEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=12)
    same, _ = env.reset(seed=12)
    np.testing.assert_array_equal(obs, same)
    assert 80 <= obs[2] * 100 <= 120
    assert np.linalg.norm(obs[:2]) * 100 <= 30


@pytest.mark.parametrize(("action", "expected_speed"), [(-1, -10.4905), (0, -10), (1, -9.5095)])
def test_vertical_commands_match_analytical_speed(action, expected_speed):
    """Compare trois commandes verticales aux vitesses calculées à la main."""
    env = RocketLandingEnv()
    env.reset(options={"start": "vertical"})
    obs, reward, terminated, truncated, info = env.step((action, 0, 0))
    assert obs[5] * 20 == pytest.approx(expected_speed, abs=1e-5)
    assert info["time_s"] == pytest.approx(0.05)
    assert reward == 0 and not terminated and not truncated


def test_gimbal_causes_translation_and_rotation():
    """Vérifie qu'une tuyère orientée déplace la fusée et produit le couple attendu."""
    env = RocketLandingEnv()
    env.reset(options={"start": "vertical"})
    obs, *_ = env.step((0, 1, 0))
    assert obs[4] != 0
    assert obs[12] < 0


@pytest.mark.parametrize(
    ("brake", "outcome", "reward"), [(False, "hard_landing", -100), (True, "landed", 100)]
)
def test_first_contact_speed_drives_the_outcome(brake, outcome, reward):
    """Compare les issues et récompenses d'une chute libre et d'une arrivée freinée."""
    env = RocketLandingEnv()
    obs, _ = env.reset(options={"start": "vertical"})
    for _ in range(600):
        state = VerticalState(obs[2] * 100, obs[5] * 20)
        action = braking_action(state) if brake and state.height_m > 0 else -1
        obs, actual_reward, terminated, truncated, info = env.step((action, 0, 0))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["outcome"] == outcome
    assert actual_reward == reward
    assert (info["impact_vertical_m_s"] >= -2) == brake
    if brake:
        np.testing.assert_array_equal(obs[-4:], np.ones(4))


def test_hover_times_out_instead_of_terminating():
    """Vérifie qu'un vol stationnaire atteint 30 s et se termine par troncature."""
    env = RocketLandingEnv()
    env.reset(options={"start": "vertical"})
    env.data.qvel[:] = 0
    for _ in range(600):
        _, reward, terminated, truncated, info = env.step((0, 0, 0))
    assert not terminated and truncated
    assert info["outcome"] == "timeout"
    assert reward == -25 and info["time_s"] == pytest.approx(30)
