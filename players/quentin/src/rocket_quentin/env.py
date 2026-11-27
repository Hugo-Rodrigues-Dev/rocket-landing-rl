"""Environnement Gymnasium de la phase 0, sur une barge fixe."""

from math import cos, radians, sin, sqrt, tau
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rocket_quentin.thrust import MAX_GIMBAL_DEG, MAX_THRUST_N, apply_thrust

SCENE_PATH = Path(__file__).parent / "assets" / "scene.xml"
DECISION_INTERVAL = 0.05
MAX_DECISIONS = 600
MAX_FLIGHT_TILT_DEG = 60.0
LEG_NAMES = ("leg_ne", "leg_nw", "leg_se", "leg_sw")
CURRICULUM_POWER = 3
EASY_START = {
    "altitude_m": (0.04, 0.09),
    "offset_radius_m": 0.2,
    "vertical_speed_m_s": (-0.45, -0.2),
    "horizontal_speed_m_s": 0.03,
    "max_tilt_deg": 0.15,
}


class RocketLandingEnv(gym.Env):
    """19 observations en unités SI, trois commandes continues dans [-1, 1]."""

    def __init__(self, start_difficulty: float = 1.0, *, altitude_only: bool = False):
        self.set_start_difficulty(start_difficulty)
        self.altitude_only = altitude_only
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
        self._substeps = round(DECISION_INTERVAL / self.model.opt.timestep)
        self._joint = self.data.joint("rocket_free")
        self._rocket = self.data.body("rocket")
        self._hull_id = self.model.geom("hull").id
        self._sea_id = self.model.geom("sea").id
        deck = self.model.geom("deck")
        self._deck_id = deck.id
        self._deck_height = float(deck.pos[2] + deck.size[2])
        self._deck_half_size = deck.size[:2].copy()
        self._legs = [self.model.geom(name).id for name in LEG_NAMES]
        self._decisions = 0
        self._impact_speed = None
        self._impact_tilt = None
        self._contacts = np.zeros(4, dtype=bool)
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._decisions = 0
        self._impact_speed = None
        self._impact_tilt = None
        self._episode_start_difficulty = None
        self._contacts.fill(False)
        self._needs_reset = False

        if options and options.get("vertical_demo"):
            self._joint.qpos[:3] = 0
            self._joint.qpos[:2] = options.get("initial_position_xy", (0.0, 0.0))
            pitch = radians(options.get("initial_pitch_deg", 0.0))
            mujoco.mju_axisAngle2Quat(self._joint.qpos[3:], [0, 1, 0], pitch)
            self._joint.qvel[:2] = options.get("initial_velocity_xy", (0.0, 0.0))
            self._joint.qvel[2] = -10.0
            altitude = 100.0
        else:
            altitude = self._sample_start()
        mujoco.mj_kinematics(self.model, self.data)
        self._joint.qpos[2] += altitude - self._altitude()
        mujoco.mj_forward(self.model, self.data)
        return self._observation(self._contacts), {"outcome": None}

    def set_start_difficulty(self, difficulty: float, *, previous_level_share: float = 0.0) -> None:
        """Règle les prochains départs et la part tirée au palier précédent."""
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("La difficulté doit être comprise entre 0 et 1.")
        if not 0.0 <= previous_level_share < 1.0:
            raise ValueError("La part de départs précédents doit être comprise entre 0 et 1 exclu.")
        self._previous_start_difficulty = self.start_difficulty if previous_level_share else difficulty
        self._previous_level_share = previous_level_share
        self.start_difficulty = difficulty

    def _sample_start(self):
        """Tire un départ du curriculum et renvoie l'altitude visée."""
        # Les premiers niveaux restent proches du pont.
        difficulty = self.start_difficulty
        if self._previous_level_share and self.np_random.random() < self._previous_level_share:
            difficulty = self._previous_start_difficulty
        self._episode_start_difficulty = difficulty
        altitude_blend = difficulty**CURRICULUM_POWER
        other_blend = 0.0 if self.altitude_only else altitude_blend

        def mix(easy, full, blend=other_blend):
            return full if blend == 1.0 else easy + blend * (full - easy)

        direction = self.np_random.uniform(0, tau)
        tilt = radians(self.np_random.uniform(0, mix(EASY_START["max_tilt_deg"], 10)))
        roll = self.np_random.uniform(0, tau)
        inclination = np.empty(4)
        spin = np.empty(4)
        mujoco.mju_axisAngle2Quat(inclination, [cos(direction), sin(direction), 0], tilt)
        mujoco.mju_axisAngle2Quat(spin, [0, 0, 1], roll)
        mujoco.mju_mulQuat(self._joint.qpos[3:], inclination, spin)

        self._joint.qpos[:2] = self._sample_disk(mix(EASY_START["offset_radius_m"], 30))
        easy_low_altitude, easy_high_altitude = EASY_START["altitude_m"]
        altitude = self.np_random.uniform(
            mix(easy_low_altitude, 80, altitude_blend), mix(easy_high_altitude, 120, altitude_blend)
        )
        self._joint.qvel[:2] = self._sample_disk(mix(EASY_START["horizontal_speed_m_s"], 3))
        easy_min_speed, easy_max_speed = EASY_START["vertical_speed_m_s"]
        self._joint.qvel[2] = self.np_random.uniform(mix(easy_min_speed, -15), mix(easy_max_speed, -5))
        return altitude

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Appelle reset() avant de commencer un épisode.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Une action doit contenir trois nombres finis.")
        throttle, tilt_x, tilt_y = np.clip(action, -1, 1)
        thrust = (throttle + 1) * MAX_THRUST_N / 2
        self._decisions += 1

        for _ in range(self._substeps):
            apply_thrust(self.model, self.data, thrust, tilt_x * MAX_GIMBAL_DEG, tilt_y * MAX_GIMBAL_DEG)
            mujoco.mj_step(self.model, self.data)
            # Relit les contacts à la position atteinte, avant le prochain pas de physique.
            mujoco.mj_forward(self.model, self.data)
            contacts, outcome = self._check_outcome()
            if outcome is not None:
                break

        terminated = outcome is not None
        truncated = not terminated and self._decisions >= MAX_DECISIONS
        self._needs_reset = terminated or truncated
        info = {"outcome": "timeout" if truncated else outcome}
        if self._needs_reset:
            info["start_difficulty"] = self._episode_start_difficulty
            info["episode_steps"] = self._decisions
        if outcome == "out_of_bounds":
            info["out_of_bounds_tilt"] = bool(self._rocket.xmat[8] < cos(radians(MAX_FLIGHT_TILT_DEG)))
        if self._impact_speed is not None:
            info["impact_speed_m_s"] = self._impact_speed
            info["impact_tilt_deg"] = self._impact_tilt
        reward = 0.0
        if terminated:
            reward = 100.0 if outcome == "landed" else -100.0
        return self._observation(contacts), reward, terminated, truncated, info

    def _sample_disk(self, radius):
        direction = self.np_random.uniform(0, tau)
        # La racine répartit uniformément les points sur la surface du disque.
        return radius * sqrt(self.np_random.random()) * np.array([cos(direction), sin(direction)])

    def _altitude(self):
        # Le bas d'une capsule est son centre moins sa demi-longueur projetée et son rayon.
        centers = self.data.geom_xpos[self._legs, 2]
        axis_z = self.data.geom_xmat[self._legs, 8]
        radii, half_lengths = self.model.geom_size[self._legs, :2].T
        return float(np.min(centers - np.abs(axis_z) * half_lengths - radii) - self._deck_height)

    def _observation(self, contacts):
        rotation = self._rocket.xmat.reshape(3, 3)
        position = [*self._joint.qpos[:2], self._altitude()]
        return np.concatenate((position, self._joint.qvel, rotation[:, 0], rotation[:, 2], contacts)).astype(
            np.float32
        )

    def _check_outcome(self):
        contacts = np.zeros(4, dtype=bool)
        rotation = self._rocket.xmat.reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
        velocity = self._joint.qvel[:3]
        angular_velocity = self._joint.qvel[3:]
        crash = False
        impact_speeds = []
        for contact in self.data.contact:
            first, second = contact.geom
            pair = (first, second)
            if self._hull_id in pair and (self._deck_id in pair or self._sea_id in pair):
                crash = True
            if self._deck_id not in pair or abs(contact.frame[2]) <= 0.5:
                continue
            leg = second if first == self._deck_id else first
            if leg not in self._legs or contact.pos[2] < self._deck_height - 0.5:
                continue
            index = self._legs.index(leg)
            contacts[index] = True
            if not self._contacts[index]:
                lever = contact.pos - self._rocket.xpos
                point_velocity = velocity + np.cross(rotation @ angular_velocity, lever)
                impact_speeds.append(float(-point_velocity[2]))

        self._contacts = contacts
        if impact_speeds and self._impact_speed is None:
            self._impact_speed = max(impact_speeds)
            self._impact_tilt = tilt
        if crash:
            return contacts, "crash"
        if max(impact_speeds, default=0.0) > 2.0:
            return contacts, "hard_landing"
        if contacts.all() and np.linalg.norm(velocity) < 0.2 and np.linalg.norm(angular_velocity) < 0.1:
            return contacts, "landed"
        base = self.data.site("nozzle").xpos
        if base[2] < self._deck_height and np.any(np.abs(base[:2]) > self._deck_half_size):
            return contacts, "water"
        if np.linalg.norm(self._joint.qpos[:2]) > 100 or self._altitude() > 200 or tilt > MAX_FLIGHT_TILT_DEG:
            return contacts, "out_of_bounds"
        return contacts, None
