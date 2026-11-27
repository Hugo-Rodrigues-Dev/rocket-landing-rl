from math import cos, pi, sin

import mujoco
import numpy as np
import pytest

from rocket_quentin.common.env import SCENE_PATH
from rocket_quentin.common.thrust import apply_thrust


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    return model, mujoco.MjData(model)


def test_scene_dimensions_and_initial_clearance(scene):
    model, data = scene
    mujoco.mj_forward(model, data)
    assert model.body("rocket").mass[0] == pytest.approx(1000)
    assert model.geom("hull").size[:2] == pytest.approx([0.5, 5])
    assert model.geom("deck").size == pytest.approx([7.5, 7.5, 0.5])
    assert model.geom("deck").pos[2] + model.geom("deck").size[2] == pytest.approx(1)
    assert model.site("nozzle").pos == pytest.approx([0, 0, -5])
    assert data.joint("rocket_free").qpos[2] == pytest.approx(6.77)
    assert data.ncon == 0


@pytest.mark.parametrize(
    ("thrust_n", "expected_acceleration"),
    [(-100, -9.81), (9_810, 0.0), (25_000, 9.81)],
)
def test_vertical_acceleration(scene, thrust_n, expected_acceleration):
    model, data = scene
    apply_thrust(model, data, thrust_n)
    mujoco.mj_step(model, data)

    assert data.qacc[2] == pytest.approx(expected_acceleration)
    assert data.qacc[3:] == pytest.approx([0, 0, 0])


@pytest.mark.parametrize(
    ("tilt_x", "tilt_y", "expected_direction", "torque_axis"),
    [
        (25, 0, [0, -sin(pi / 18), cos(pi / 18)], 0),
        (0, -25, [-sin(pi / 18), 0, cos(pi / 18)], 1),
    ],
)
def test_gimbal_angle_limits_and_rotation(scene, tilt_x, tilt_y, expected_direction, torque_axis):
    model, data = scene
    apply_thrust(model, data, 9_810, tilt_x, tilt_y)
    force, torque = data.xfrc_applied[model.body("rocket").id].reshape(2, 3)

    assert force / 9_810 == pytest.approx(expected_direction)
    assert torque[torque_axis] != 0
    mujoco.mj_step(model, data)
    assert data.qacc[3 + torque_axis] != 0


def test_thrust_follows_the_rocket_and_can_be_turned_off(scene):
    model, data = scene
    # Une rotation de 90° autour de y envoie le +z de la fusée vers le +x du monde.
    data.joint("rocket_free").qpos[3:] = [cos(pi / 4), 0, sin(pi / 4), 0]
    apply_thrust(model, data, 9_810)
    wrench = data.xfrc_applied[model.body("rocket").id]
    assert wrench == pytest.approx([9_810, 0, 0, 0, 0, 0])

    apply_thrust(model, data, 0)
    assert wrench == pytest.approx(np.zeros(6))
