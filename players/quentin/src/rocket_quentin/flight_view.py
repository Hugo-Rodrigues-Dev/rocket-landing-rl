"""Effets visuels et télémétrie pour le lecteur MuJoCo."""

from math import atan2, cos, degrees, sin, tau

import mujoco
import numpy as np

from rocket_quentin.env import RocketLandingEnv
from rocket_quentin.thrust import MAX_GIMBAL_DEG, MAX_THRUST_N

ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
LINE = mujoco.mjtGeom.mjGEOM_LINE
IDENTITY = np.eye(3).ravel()
RIPPLE_GRID = tuple(
    (x + 2.4 * sin(y), y + 1.8 * sin(x), 0.13 * x + 0.31 * y)
    for x in range(-36, 43, 9)
    for y in range(-36, 43, 9)
)
DECK_RING = np.array([[2.8 * cos(t), 2.8 * sin(t), 0.015] for t in np.linspace(0, tau, 25)])
OUTCOMES = {
    None: "En vol",
    "landed": "Posee",
    "hard_landing": "Contact dur",
    "crash": "Crash",
    "water": "A l'eau",
    "out_of_bounds": "Hors zone",
    "timeout": "Temps ecoule",
}


def _geom(scene, kind, size, position, rotation, rgba, *, emission=0.0):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, size, position, rotation, rgba)
    geom.emission = emission
    scene.ngeom += 1
    return geom


def draw_flame(scene: mujoco.MjvScene, nozzle: np.ndarray, force: np.ndarray, time: float) -> None:
    power = np.linalg.norm(force) / MAX_THRUST_N
    if power <= 0:
        return
    direction = -force / (power * MAX_THRUST_N)
    quaternion = np.empty(4)
    rotation = np.empty(9)
    mujoco.mju_quatZ2Vec(quaternion, direction)
    mujoco.mju_quat2Mat(rotation, quaternion)
    length = 6.0 * power * (0.96 + 0.04 * sin(39 * time))
    width = 0.55 * np.sqrt(power)
    layers = (
        (0.48, 0.47, 0.86, (1.0, 0.31, 0.05, 0.66)),
        (0.30, 0.30, 0.70, (1.0, 0.68, 0.20, 0.94)),
        (0.14, 0.14, 0.54, (1.0, 0.93, 0.63, 1.0)),
    )
    for offset, half_length, radius, color in layers:
        _geom(
            scene,
            ELLIPSOID,
            [radius * width, radius * width, half_length * length],
            nozzle + direction * offset * length,
            rotation,
            color,
            emission=0.9,
        )


def draw_ripples(scene: mujoco.MjvScene, time: float) -> None:
    for x, y, phase in RIPPLE_GRID:
        shimmer = 0.5 + 0.5 * sin(1.5 * time + phase)
        _geom(
            scene,
            ELLIPSOID,
            [1.6 + 0.55 * cos(phase), 0.11, 0.005],
            [x + time * 0.45, y + 0.65 * sin(time + phase), 0.012],
            IDENTITY,
            [0.65, 0.87, 0.95, 0.08 + 0.26 * shimmer],
            emission=0.35,
        )


def draw_deck_markings(scene: mujoco.MjvScene, center: np.ndarray, rotation: np.ndarray) -> None:
    points = DECK_RING @ rotation.T + center
    for first, second in zip(points[:-1], points[1:], strict=True):
        line = _geom(scene, LINE, [0, 0, 0], first, IDENTITY, [0.90, 0.97, 1, 1], emission=0.5)
        mujoco.mjv_connector(line, LINE, 2, first, second)


def telemetry(
    env: RocketLandingEnv, observation: np.ndarray, action, info: dict
) -> list[tuple[int, int, str, str]]:
    data = env.data
    deck = data.geom("deck")
    deck_rotation = deck.xmat.reshape(3, 3)
    deck_top = deck.xpos + deck_rotation[:, 2] * env.model.geom("deck").size[2]
    left = [("ENVIRONNEMENT", ""), ("Temps simule", f"{data.time:5.2f} s")]
    if env.phase >= 4:
        wind_x, wind_y = observation[25:27]
        left.extend(
            (
                ("Vent", f"{np.hypot(wind_x, wind_y):.2f} m/s"),
                ("Vent X / Y", f"{wind_x:+.1f} / {wind_y:+.1f} m/s"),
            )
        )
    if env.phase >= 1:
        left.extend(
            (
                ("Barge X / Y", f"{deck_top[0]:+.1f} / {deck_top[1]:+.1f} m"),
                ("Vitesse X / Y", f"{observation[15]:+.2f} / {observation[16]:+.2f} m/s"),
            )
        )
    left.append(("Hauteur du pont", f"{deck_top[2]:.2f} m"))
    if env.phase >= 3:
        roll = degrees(data.joint("barge_roll").qpos[0])
        pitch = degrees(data.joint("barge_pitch").qpos[0])
        left.append(("Roulis / tangage", f"{roll:+.1f} / {pitch:+.1f} deg"))
        left.append(("Vitesse verticale", f"{observation[24]:+.2f} m/s"))

    axis_x, axis_y, axis_z = observation[12:15]
    x_angle, y_angle = degrees(atan2(axis_x, axis_z)), degrees(atan2(axis_y, axis_z))
    thrust = np.linalg.norm(env.thrust_force)
    right = [
        ("FUSEE", ""),
        ("Etat", OUTCOMES[info["outcome"]]),
        ("Altitude du pont", f"{observation[2]:.1f} m"),
        ("Distance au centre", f"{np.linalg.norm(observation[:2]):.2f} m"),
        ("Vitesse verticale rel.", f"{observation[5]:+.2f} m/s"),
        ("Inclinaison X / Y", f"{x_angle:+.1f} / {y_angle:+.1f} deg"),
        ("Rotation", f"{degrees(np.linalg.norm(observation[6:9])):.1f} deg/s"),
        ("Poussee", f"{100 * thrust / MAX_THRUST_N:.0f} % ({thrust / 1000:.1f} kN)"),
        ("Tuyere X / Y", f"{action[1] * MAX_GIMBAL_DEG:+.1f} / {action[2] * MAX_GIMBAL_DEG:+.1f} deg"),
        ("Jambes au contact", f"{int(observation[-4:].sum())}/4"),
    ]
    if env.phase >= 2:
        right.append(("Carburant", f"{info['fuel_remaining_kg']:.1f} kg"))
    if env.phase >= 4:
        right.append(("RCS X / Y / Z", " / ".join(f"{value * 100:+.0f}" for value in action[3:]) + " %"))

    result = []
    for position, rows in (
        (mujoco.mjtGridPos.mjGRID_TOPLEFT, left),
        (mujoco.mjtGridPos.mjGRID_TOPRIGHT, right),
    ):
        labels, values = zip(*rows, strict=True)
        result.append((mujoco.mjtFontScale.mjFONTSCALE_100, position, "\n".join(labels), "\n".join(values)))
    return result
