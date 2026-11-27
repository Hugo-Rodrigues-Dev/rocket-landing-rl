"""Pilote de référence pour la collecte de démonstrations."""

import numpy as np

from rocket_quentin.common.fuel import FUEL_OBSERVATION_INDEX, MIN_THRUST_FRACTION
from rocket_quentin.common.wind import drag_force
from rocket_quentin.dagger.pilot_lateral import lateral_action
from rocket_quentin.dagger.wave_approach import APPROACH_HEIGHT_M, wave_reference


def wind_reference(observation: np.ndarray) -> tuple[list[float], np.ndarray]:
    velocity = observation[3:6].copy()
    velocity[:2] += observation[15:17] - observation[25:27]
    deck_omega = observation[21:24]
    velocity[2] += observation[24] + deck_omega[0] * observation[1] - deck_omega[1] * observation[0]
    mass = 1000 + float(observation[FUEL_OBSERVATION_INDEX])
    action = wave_reference(observation, drag_force(velocity)[:2] / mass)

    body_x, body_z = observation[9:12], observation[12:15]
    body_y = np.cross(body_z, body_x)
    body = np.column_stack((body_x, body_y, body_z))
    relative_rate = observation[6:9].copy()
    correction = -6 * relative_rate
    if observation[2] < APPROACH_HEIGHT_M:
        normal = observation[18:21]
        relative_rate -= body.T @ deck_omega
        correction[:2] = 14 * np.array([-body_y @ normal, body_x @ normal]) - 10 * relative_rate[:2]
    return action, np.clip(correction, -1, 1)


def teaching_targets(observation: np.ndarray, phase: int) -> np.ndarray:
    """Étiquette la demande moteur et les commandes des actionneurs."""
    rcs = ()
    if phase == 4:
        (throttle, tilt_x, tilt_y), rcs = wind_reference(observation)
    else:
        throttle, tilt_x, tilt_y = (wave_reference if phase == 3 else lateral_action)(observation)
    threshold = 0.05 if phase >= 3 else 0.0
    mass_ratio = 1 + float(observation[FUEL_OBSERVATION_INDEX]) / 1000
    powered = float(np.clip((throttle + 1) * mass_ratio - 1, 2 * MIN_THRUST_FRACTION - 1, 1))
    return np.array([throttle - threshold, powered, tilt_x, tilt_y, *rcs], dtype=np.float32)


def reference_action(observation: np.ndarray, phase: int) -> list[float]:
    if phase < 2:
        return lateral_action(observation)
    return teaching_action(teaching_targets(observation, phase))


def teaching_action(targets: np.ndarray) -> list[float]:
    margin, *action = targets.tolist()
    if margin < 0:
        action[0] = -1.0
    return action
