import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocket_quentin.env import RocketLandingEnv
from rocket_quentin.pilot_attitude import attitude_action
from rocket_quentin.pilot_lateral import lateral_action
from rocket_quentin.pilot_vertical import vertical_action


def test_gymnasium_api():
    check_env(RocketLandingEnv(), skip_render_check=True)


def test_seeded_start_meets_contract():
    env = RocketLandingEnv()
    first, _ = env.reset(seed=12)
    same, _ = env.reset(seed=12)
    different, _ = env.reset(seed=13)
    assert np.array_equal(first, same)
    assert not np.array_equal(first, different)
    for seed in range(40):
        obs, _ = env.reset(seed=seed)
        assert 80 <= obs[2] <= 120
        assert np.linalg.norm(obs[:2]) <= 30
        assert np.linalg.norm(obs[3:5]) <= 3
        assert -15 <= obs[5] <= -5
        assert np.degrees(np.arccos(np.clip(obs[14], -1, 1))) <= 10
        assert not obs[-4:].any()


def test_curriculum_changes_only_the_next_start():
    env = RocketLandingEnv()
    official, _ = env.reset(seed=12)
    initial_pose = env.data.qpos.copy()
    env.set_start_difficulty(0.0)
    assert np.array_equal(env.data.qpos, initial_pose)
    with pytest.raises(ValueError):
        env.set_start_difficulty(1.1)

    easy, _ = env.reset(seed=12)
    assert 0.04 <= easy[2] <= 0.09
    assert np.linalg.norm(easy[:2]) <= 0.2
    assert -0.45 <= easy[5] <= -0.2
    env.set_start_difficulty(0.1)
    next_level, _ = env.reset(seed=12)
    assert easy[2] < next_level[2] < official[2]
    env.set_start_difficulty(1.0)
    restored, _ = env.reset(seed=12)
    assert np.array_equal(restored, official)


def test_altitude_only_curriculum_preserves_other_start_variables():
    env = RocketLandingEnv(start_difficulty=0.0, altitude_only=True)
    easy, _ = env.reset(seed=12)
    env.set_start_difficulty(0.2)
    lower, _ = env.reset(seed=12)
    env.set_start_difficulty(0.3)
    higher, _ = env.reset(seed=12)
    standard, _ = RocketLandingEnv(start_difficulty=0.3).reset(seed=12)

    assert 0.67 <= lower[2] <= 1.05
    assert 2.19 <= higher[2] <= 3.33
    assert higher[2] == standard[2]
    assert np.array_equal(np.delete(easy, 2), np.delete(lower, 2))
    assert np.array_equal(np.delete(lower, 2), np.delete(higher, 2))


def test_curriculum_can_mix_starts_from_the_previous_level():
    env = RocketLandingEnv(start_difficulty=0.15)
    env.set_start_difficulty(0.2, previous_level_share=0.25)
    altitudes = np.array([env.reset(seed=seed)[0][2] for seed in range(200)])
    previous = (altitudes >= 0.30) & (altitudes <= 0.50)
    current = (altitudes >= 0.67) & (altitudes <= 1.05)
    assert np.all(previous | current)
    assert 0.15 < previous.mean() < 0.35

    env.set_start_difficulty(0.2)
    reset, _ = env.reset(seed=12)
    reference, _ = RocketLandingEnv(start_difficulty=0.2).reset(seed=12)
    assert np.array_equal(reset, reference)


def test_episode_reports_the_level_actually_drawn_at_reset():
    env = RocketLandingEnv(start_difficulty=0.15)
    env.set_start_difficulty(0.2, previous_level_share=0.25)
    observation, _ = env.reset(seed=3)
    assert 0.30 <= observation[2] <= 0.50
    env.set_start_difficulty(0.3, previous_level_share=0.25)

    for decision in range(1, 601):
        _, _, terminated, truncated, info = env.step([-1, 0, 0])
        if terminated or truncated:
            assert info["start_difficulty"] == 0.15
            assert info["episode_steps"] == decision
            return
    pytest.fail("L'épisode ne s'est pas terminé.")


def place_upright(env, position):
    env.reset(seed=123)
    joint = env.data.joint("rocket_free")
    joint.qpos[:] = [*position, 1, 0, 0, 0]
    joint.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize(("action", "velocity"), [(-1, -0.4905), (0, 0.0), (1, 0.4905)])
def test_action_holds_for_50_ms(action, velocity):
    env = RocketLandingEnv()
    place_upright(env, [0, 0, 100])
    obs, _, terminated, truncated, _ = env.step([action, 0, 0])
    assert env.data.time == pytest.approx(0.05)
    assert obs[5] == pytest.approx(velocity)
    assert not (terminated or truncated)


@pytest.mark.parametrize(
    ("position", "outcome", "reward"),
    [
        ([0, 0, 6.77], "landed", 100.0),
        ([0, 0, 7.12], "hard_landing", -100.0),
        ([0, 0, 5.9], "crash", -100.0),
        ([20, 0, 5.9], "water", -100.0),
        ([101, 0, 100], "out_of_bounds", -100.0),
    ],
)
def test_physical_endings(position, outcome, reward):
    env = RocketLandingEnv()
    place_upright(env, position)
    for _ in range(40):
        obs, received, terminated, truncated, info = env.step([-1, 0, 0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["outcome"] == outcome
    assert received == reward
    if outcome in ("landed", "hard_landing"):
        assert info["impact_tilt_deg"] == pytest.approx(0)
        assert (info["impact_speed_m_s"] > 2) == (outcome == "hard_landing")
    if outcome == "landed":
        assert obs[-4:].all()


def test_time_limit_is_a_truncation():
    env = RocketLandingEnv()
    place_upright(env, [0, 0, 100])
    for _ in range(599):
        _, _, terminated, truncated, _ = env.step([0, 0, 0])
        assert not (terminated or truncated)
    _, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert truncated and not terminated
    assert reward == 0
    assert info["outcome"] == "timeout"
    assert env.data.time == pytest.approx(30)


def test_out_of_bounds_distinguishes_excessive_tilt():
    env = RocketLandingEnv()
    place_upright(env, [0, 0, 100])
    rotation = env.data.joint("rocket_free").qpos[3:]
    mujoco.mju_axisAngle2Quat(rotation, [0, 1, 0], np.radians(65))
    mujoco.mj_forward(env.model, env.data)

    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated
    assert info["outcome"] == "out_of_bounds"
    assert info["out_of_bounds_tilt"] is True

    place_upright(env, [101, 0, 100])
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated
    assert info["outcome"] == "out_of_bounds"
    assert info["out_of_bounds_tilt"] is False


@pytest.mark.parametrize(("use_pilot", "outcome"), [(True, "landed"), (False, "hard_landing")])
def test_vertical_pilot_against_freefall(use_pilot, outcome):
    env = RocketLandingEnv()
    obs, _ = env.reset(options={"vertical_demo": True})
    assert obs[:6] == pytest.approx([0, 0, 100, 0, 0, -10])
    for _ in range(600):
        action = vertical_action(obs) if use_pilot else [-1, 0, 0]
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert terminated and not truncated
    assert info["outcome"] == outcome
    assert (info["impact_speed_m_s"] < 2) == use_pilot


@pytest.mark.parametrize(
    ("initial_pitch", "pilot", "expected_outcome"),
    [(5.0, attitude_action, "landed"), (-5.0, attitude_action, "landed"), (5.0, vertical_action, "water")],
)
def test_attitude_correction_from_tilted_start(initial_pitch, pilot, expected_outcome):
    env = RocketLandingEnv()
    obs, _ = env.reset(options={"vertical_demo": True, "initial_pitch_deg": initial_pitch})
    assert obs[12] == pytest.approx(np.sin(np.radians(initial_pitch)))
    for _ in range(40):
        obs, _, terminated, truncated, _ = env.step(pilot(obs))
        assert not (terminated or truncated)

    tilt = np.degrees(np.arctan2(np.linalg.norm(obs[12:14]), obs[14]))
    expected_tilt = 0.0 if pilot is attitude_action else 5.0
    assert tilt == pytest.approx(expected_tilt, abs=0.5)

    for _ in range(560):
        obs, _, terminated, truncated, info = env.step(pilot(obs))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["outcome"] == expected_outcome


@pytest.mark.parametrize(
    ("pilot", "expected_outcome"),
    [(lateral_action, "landed"), (attitude_action, "water")],
)
def test_lateral_guidance_from_offset_and_drift(pilot, expected_outcome):
    env = RocketLandingEnv()
    observation, _ = env.reset(
        options={
            "vertical_demo": True,
            "initial_position_xy": (12.0, -8.0),
            "initial_velocity_xy": (1.0, 0.0),
        }
    )
    assert observation[:5] == pytest.approx([12, -8, 100, 1, 0])
    for _ in range(600):
        observation, _, terminated, truncated, info = env.step(pilot(observation))
        if terminated or truncated:
            break

    assert terminated and not truncated
    assert info["outcome"] == expected_outcome
    if expected_outcome == "landed":
        assert np.linalg.norm(observation[:2]) < 0.5
        assert info["impact_speed_m_s"] < 2
