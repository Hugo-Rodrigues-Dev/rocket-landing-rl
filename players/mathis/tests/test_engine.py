"""Vérifications du moteur : vol stationnaire, accélération, sens et intensité du couple."""

import mujoco
import numpy as np
import pytest

from rocket_mathis.engine import GIMBAL_MAX, T_MAX, Engine, thrust_direction
from rocket_mathis.scene import load_model

G = 9.81


@pytest.fixture
def sim():
    model = load_model()
    data = mujoco.MjData(model)
    data.qpos[2] = 50.0  # en l'air, loin de tout contact
    mujoco.mj_forward(model, data)
    return model, data, Engine(model)


def fly(model, data, engine, seconds, thrust, gimbal):
    while data.time < seconds:
        engine.apply(data, thrust, gimbal)
        mujoco.mj_step(model, data)


def mass(model):
    return model.body_subtreemass[model.body("fusee").id]


def test_thrust_direction_is_along_axis_at_zero_gimbal():
    assert thrust_direction(np.zeros(2)) == pytest.approx([0, 0, 1])
    assert np.linalg.norm(thrust_direction(np.array([0.1, -0.15]))) == pytest.approx(1)


def test_hover_when_thrust_equals_weight(sim):
    model, data, engine = sim
    fly(model, data, engine, 2.0, mass(model) * G, [0, 0])
    assert data.qpos[2] == pytest.approx(50.0, abs=1e-3)
    assert np.abs(data.qvel).max() < 1e-6


def test_full_thrust_accelerates_at_one_g(sim):
    model, data, engine = sim
    fly(model, data, engine, 1.0, T_MAX, [0, 0])
    assert data.qvel[2] == pytest.approx(G * data.time, rel=1e-3)


@pytest.mark.parametrize("axis", [0, 1])
def test_gimbal_torque_sign_and_magnitude(sim, axis):
    model, data, engine = sim
    gimbal = np.zeros(2)
    gimbal[axis] = GIMBAL_MAX
    thrust = mass(model) * G
    fly(model, data, engine, 0.2, thrust, gimbal)

    body = model.body("fusee").id
    lever = model.site_pos[model.site("moteur").id][2] - model.body_ipos[body][2]  # < 0 : sous le CdM
    expected = thrust * np.sin(GIMBAL_MAX) * lever / model.body_inertia[body][axis]
    omega = data.qvel[3:]  # vitesse angulaire, repère de la fusée
    assert omega[axis] < 0
    assert omega[axis] / data.time == pytest.approx(expected, rel=1e-3)
    assert abs(omega[1 - axis]) < 1e-9 and abs(omega[2]) < 1e-9


def test_commands_are_clipped(sim):
    model, data, engine = sim
    engine.apply(data, 10 * T_MAX, [1.0, -1.0])
    force = np.linalg.norm(data.xfrc_applied[model.body("fusee").id, :3])
    assert force == pytest.approx(T_MAX)
    engine.apply(data, -5.0, [0, 0])
    assert not data.xfrc_applied.any()
