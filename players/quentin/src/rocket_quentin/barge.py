"""Mouvement horizontal prescrit du pont mobile dans MuJoCo."""

from math import sqrt

import mujoco
import numpy as np

BARGE_THETA = 0.5
BARGE_SIGMA = 1.0
MAX_BARGE_SPEED = 3.0
MAX_BARGE_ACCELERATION = 0.5


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector * (maximum / norm) if norm > maximum else vector


class BargeMotion:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        first_slide = model.joint("barge_x").id
        position_index = model.jnt_qposadr[first_slide]
        velocity_index = model.jnt_dofadr[first_slide]
        self.position = data.qpos[position_index : position_index + 2]
        self.velocity = data.qvel[velocity_index : velocity_index + 2]
        self.target_velocity = np.zeros(2)
        self.acceleration = np.zeros(2)

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
        # Le solveur voit la vitesse des glissières et transmet le frottement à la fusée.
        mujoco.mj_step(self.model, self.data)
        # La barge suit son mouvement prescrit même si la fusée lui applique une force.
        self.position[:] = next_position
        self.velocity[:] = next_velocity
