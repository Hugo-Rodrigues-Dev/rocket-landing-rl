"""Vérifications de la scène de l'étape 1 : masses, chute libre, atterrissage stable."""

import mujoco
import numpy as np
import pytest
from rocket_mathis.scene import DECK_TOP_Z, FEET, load_model

REST_Z = DECK_TOP_Z + 6.1  # origine de la fusée posée : pont + distance origine → bas des pieds


@pytest.fixture
def sim():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("petite_chute").id)
    return model, data


def run_until(model, data, t):
    while data.time < t:
        mujoco.mj_step(model, data)


def tilt_deg(model, data):
    z_axis_world_z = data.xmat[model.body("fusee").id][8]
    return np.degrees(np.arccos(np.clip(z_axis_world_z, -1.0, 1.0)))


def contact_pairs(model, data):
    names = model.geom
    return {frozenset((names(c.geom1).name, names(c.geom2).name)) for c in data.contact[: data.ncon]}


def test_rocket_matches_contract(sim):
    model, _ = sim
    assert model.body_subtreemass[model.body("fusee").id] == pytest.approx(1000.0)
    # Centre de masse sous le milieu du réservoir, grâce au moteur lourd
    assert model.body_ipos[model.body("fusee").id][2] < 0


def test_free_fall_follows_gravity(sim):
    model, data = sim
    z0 = data.qpos[2]
    run_until(model, data, 0.3)  # avant le contact
    assert data.ncon == 0
    assert data.qpos[2] == pytest.approx(z0 - 0.5 * 9.81 * data.time**2, abs=0.01)


def test_rocket_settles_on_its_four_feet(sim):
    model, data = sim
    run_until(model, data, 3.0)
    assert contact_pairs(model, data) == {frozenset((foot, "pont")) for foot in FEET}
    assert data.qpos[2] == pytest.approx(REST_Z, abs=0.02)  # ne s'enfonce pas
    assert np.linalg.norm(data.qvel) < 1e-3  # immobile
    assert tilt_deg(model, data) < 0.1


def test_tilted_landing_recovers(sim):
    model, data = sim
    half = np.radians(6) / 2
    data.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
    data.qvel[[0, 2]] = [1.0, -1.5]
    run_until(model, data, 5.0)
    assert contact_pairs(model, data) == {frozenset((foot, "pont")) for foot in FEET}
    assert tilt_deg(model, data) < 0.1


def test_descent_keyframe_is_contract_like():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("descente").id)
    mujoco.mj_forward(model, data)
    altitude = data.qpos[2] - REST_Z
    assert 80 <= altitude <= 120
    assert np.linalg.norm(data.qpos[:2]) <= 30
    assert -15 <= data.qvel[2] <= -5
    assert tilt_deg(model, data) <= 10
