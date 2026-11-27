"""Mouvement prescrit du pont mobile dans MuJoCo."""

from math import radians, sqrt, tau

import mujoco
import numpy as np

BARGE_THETA = 0.5
BARGE_SIGMA = 1.0
MAX_BARGE_SPEED = 3.0
MAX_BARGE_ACCELERATION = 0.5
HEAVE_AMPLITUDE = 0.5
HEAVE_PERIOD = 8.0
TILT_AMPLITUDE = radians(3)


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector * (maximum / norm) if norm > maximum else vector


class BargeMotion:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, waves: bool = False):
        self.model = model
        self.data = data
        first_slide = model.joint("barge_x").id
        position_index = model.jnt_qposadr[first_slide]
        velocity_index = model.jnt_dofadr[first_slide]
        self.position = data.qpos[position_index : position_index + 2]
        self.velocity = data.qvel[velocity_index : velocity_index + 2]
        self.target_velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.waves = WaveMotion(model, data) if waves else None
        if self.waves is not None:
            self._barge_dofs = slice(velocity_index, velocity_index + 5)
            self._prescribed_acceleration = np.zeros(model.nv)
            self._inertial_force = np.zeros(model.nv)

    def reset(self) -> None:
        self.target_velocity.fill(0)
        self.acceleration.fill(0)

    def choose_target(self, rng: np.random.Generator, interval: float) -> None:
        target = self.target_velocity * (1 - BARGE_THETA * interval)
        target += BARGE_SIGMA * sqrt(interval) * rng.standard_normal(2)
        self.target_velocity[:] = _limit_norm(target, MAX_BARGE_SPEED)
        self.acceleration[:] = _limit_norm(
            (self.target_velocity - self.velocity) / interval, MAX_BARGE_ACCELERATION
        )

    def step(self) -> None:
        dt = self.model.opt.timestep
        delta_velocity = self.acceleration * dt
        next_position = self.position + (self.velocity + 0.5 * delta_velocity) * dt
        next_velocity = self.velocity + delta_velocity
        if self.waves is not None:
            # Applique les efforts nécessaires à la trajectoire prescrite pendant le calcul
            # des contacts, y compris la compensation de la gravité sur la glissière verticale.
            self._prescribed_acceleration[self._barge_dofs] = (
                *self.acceleration,
                *(-(self.waves.frequency**2) * self.waves.position),
            )
            mujoco.mj_mulM(self.model, self.data, self._inertial_force, self._prescribed_acceleration)
            dofs = self._barge_dofs
            self.data.qfrc_applied[dofs] = (
                self._inertial_force[dofs] + self.data.qfrc_bias[dofs] - self.data.qfrc_passive[dofs]
            )
        # Le solveur voit la vitesse des glissières et transmet le frottement à la fusée.
        mujoco.mj_step(self.model, self.data)
        # La barge suit son mouvement prescrit même si la fusée lui applique une force.
        self.position[:] = next_position
        self.velocity[:] = next_velocity
        if self.waves is not None:
            self.waves.update(self.data.time)


class WaveMotion:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        first_joint = model.joint("barge_heave").id
        position_index = model.jnt_qposadr[first_joint]
        velocity_index = model.jnt_dofadr[first_joint]
        self.position = data.qpos[position_index : position_index + 3]
        self.velocity = data.qvel[velocity_index : velocity_index + 3]
        self.amplitude = np.array([HEAVE_AMPLITUDE, TILT_AMPLITUDE, TILT_AMPLITUDE])

    def reset(self, rng: np.random.Generator) -> None:
        self.frequency = tau / np.array([HEAVE_PERIOD, *rng.uniform(6.0, 10.0, size=2)])
        self.phase = rng.uniform(0, tau, size=3)
        self.update(0.0)

    def update(self, time: float) -> None:
        angle = self.frequency * time + self.phase
        self.position[:] = self.amplitude * np.sin(angle)
        self.velocity[:] = self.amplitude * self.frequency * np.cos(angle)
