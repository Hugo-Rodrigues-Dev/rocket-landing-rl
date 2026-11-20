"""Vérifications de l'environnement Gymnasium : API, départ, fins d'épisode, barge mobile."""

import warnings

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rocket_mathis  # noqa: F401  (enregistre l'environnement)
from rocket_mathis.config import EnvConfig
from rocket_mathis.env import BASE_DEPTH, RocketLandingEnv
from rocket_mathis.scene import DECK_TOP_Z

OFF = np.array([-1.0, 0.0, 0.0])  # moteur éteint
HOVER = np.array([0.0, 0.0, 0.0])  # T_MAX / 2 = poids : vol stationnaire


@pytest.fixture
def env():
    e = RocketLandingEnv()
    yield e
    e.close()


def place(env, base_height, vz=0.0, x=0.0):
    """Fusée droite, immobile, bas des pieds à base_height au-dessus du pont."""
    env.reset(seed=0)
    qpos = env.init_qpos.copy()
    qpos[:3] = [x, 0.0, DECK_TOP_Z + base_height + BASE_DEPTH]
    qpos[3:7] = [1, 0, 0, 0]
    qvel = np.zeros(env.model.nv)
    qvel[2] = vz
    env.set_state(qpos, qvel)
    mujoco.mj_forward(env.model, env.data)


def run(env, action, max_steps=1000):
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return obs, terminated, truncated, info
    raise AssertionError("l'épisode ne s'est pas terminé")


def test_gymnasium_api():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Box(-inf, inf) volontaire
        check_env(gym.make("rocket_mathis/RocketLanding-v0").unwrapped, skip_render_check=True)


def test_same_seed_same_episode(env):
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(40, 3))
    trajectories = []
    for _ in range(2):
        obs, _ = env.reset(seed=123)
        trajectories.append([obs] + [env.step(a)[0] for a in actions])
    np.testing.assert_array_equal(np.array(trajectories[0]), np.array(trajectories[1]))


def test_reset_follows_contract(env):
    c = EnvConfig()
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        pos, vel = obs[0:3] * 100, obs[3:6] * 20
        tilt = np.degrees(np.arccos(obs[11]))
        assert c.altitude[0] <= pos[2] <= c.altitude[1] + 1e-3
        assert np.linalg.norm(pos[:2]) <= c.offset_radius + 1e-3
        assert c.vertical_speed[0] - 0.5 <= vel[2] <= c.vertical_speed[1] + 0.5  # + rotation
        assert tilt <= c.max_tilt_deg + 1e-3
        assert not obs[17:].any()


def test_resting_on_deck_is_landed(env):
    place(env, base_height=0.02)
    obs, terminated, _, info = run(env, OFF)
    assert terminated and info["outcome"] == "landed" and info["is_success"]
    assert obs[17:].all()
    assert np.linalg.norm(obs[0:3] * 100) < 0.1


def test_fast_touchdown_is_hard_landing(env):
    place(env, base_height=0.5, vz=-4.0)
    _, terminated, _, info = run(env, OFF)
    assert terminated and info["outcome"] == "hard_landing"
    assert info["impact_speed"] < -2.0


def test_free_fall_from_start_fails(env):
    env.reset(seed=1)
    _, terminated, _, info = run(env, OFF)
    assert terminated and info["outcome"] in {"crash", "hard_landing"}


def test_missing_the_deck_is_water(env):
    place(env, base_height=3.0, x=20.0)
    _, terminated, _, info = run(env, OFF)
    assert terminated and info["outcome"] == "water"


def test_hovering_until_timeout(env):
    place(env, base_height=50.0)
    _, terminated, truncated, info = run(env, HOVER)
    assert truncated and not terminated and info["outcome"] == "timeout"
    assert env.data.time == pytest.approx(30.0, abs=0.06)


def test_barge_moves_in_phase_1():
    env = RocketLandingEnv(EnvConfig(phase=1))
    env.reset(seed=3)
    start = env.data.mocap_pos[env._barge].copy()
    speeds = []
    for _ in range(100):
        obs, *_ = env.step(HOVER)
        speeds.append(np.linalg.norm(obs[15:17] * 3))
    assert np.linalg.norm(env.data.mocap_pos[env._barge] - start) > 0.5
    assert max(speeds) <= 3.0 + 1e-9
    assert env.data.mocap_pos[env._barge][2] == 0.0


def test_barge_still_in_phase_0(env):
    env.reset(seed=3)
    for _ in range(20):
        obs, *_ = env.step(HOVER)
    assert not env.data.mocap_pos[env._barge].any()
    assert not obs[15:17].any()
