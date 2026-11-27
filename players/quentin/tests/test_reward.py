import numpy as np
import pytest

from rocket_quentin.common.env import MAX_DECISIONS, RocketLandingEnv
from rocket_quentin.common.reward import SAC_DISCOUNT, ProgressReward, landing_potential


def test_potential_uses_height_above_deck_and_horizontal_distance():
    assert landing_potential(np.array([0, 2, 1])) == pytest.approx(-10)
    assert landing_potential(np.array([0, 0, -0.02])) == 0


@pytest.mark.parametrize("truncate", [False, True])
def test_shaping_telescopes_with_the_correct_last_potential(truncate):
    base = RocketLandingEnv(start_difficulty=0.2)
    wrapped = ProgressReward(RocketLandingEnv(start_difficulty=0.2))
    initial, _ = wrapped.reset(seed=12)
    base.reset(seed=12)
    if truncate:
        base._decisions = MAX_DECISIONS - 2
        wrapped.env._decisions = MAX_DECISIONS - 2

    discounted_shaping = 0.0
    discount = 1.0
    while True:
        obs, reward, terminated, truncated, info = base.step([-1, 0, 0])
        shaped_obs, shaped_reward, shaped_terminated, shaped_truncated, shaped_info = wrapped.step([-1, 0, 0])
        assert np.array_equal(obs, shaped_obs)
        assert (terminated, truncated, info) == (shaped_terminated, shaped_truncated, shaped_info)
        discounted_shaping += discount * (shaped_reward - reward)
        discount *= SAC_DISCOUNT
        if terminated or truncated:
            break

    assert truncated == truncate
    final_potential = landing_potential(obs) if truncate else 0
    assert discounted_shaping == pytest.approx(discount * final_potential - landing_potential(initial))
