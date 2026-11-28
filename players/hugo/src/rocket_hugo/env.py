"""Environnement Gymnasium de phase 0 et simulation des vols libres dans MuJoCo."""

from math import cos, hypot, pi, radians, sin, sqrt
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rocket_hugo.vertical import CONTROL_DT_S, MAX_DECISIONS, thrust_newtons

SCENE = Path(__file__).with_name("assets") / "scene.xml"
GIMBAL_LIMIT_RAD = radians(10)
MIN_UPRIGHT_Z = cos(radians(60))
DECK_HALF_M = 7.5
FEET = ("foot_ne", "foot_nw", "foot_se", "foot_sw")
OUTCOME_REWARDS = {"landed": 100.0, "timeout": -25.0, "flying": 0.0}


def _disk(rng, radius):
    """Tire un point uniforme dans un disque, pour la position ou la vitesse."""
    angle = rng.uniform(0.0, 2 * pi)
    return radius * sqrt(rng.random()) * np.array((cos(angle), sin(angle)))


class RocketLandingEnv(gym.Env):
    """Simule une fusée avec tuyère orientable au-dessus d'une barge immobile.

    Une instance possède son propre état MuJoCo. Elle expose des mesures
    normalisées et applique les critères de fin de vol de la phase 0.
    """

    def __init__(self):
        """Charge la scène, crée l'état physique et mémorise les identifiants des objets suivis."""
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.substeps = round(CONTROL_DT_S / self.model.opt.timestep)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
        self._rocket = self.model.body("rocket").id
        self._deck = self.model.geom("deck").id
        self._sea = self.model.geom("sea").id
        self._base = self.model.site("base").id
        self._nozzle = self.model.site("nozzle").id
        self._deck_center = self.model.site("deck_center").id
        self._feet = {self.model.geom(name).id: i for i, name in enumerate(FEET)}
        self._episode_done = True

    def reset(self, *, seed=None, options=None):
        """Réinitialise la physique, les compteurs et l'historique des contacts.

        Par défaut, le départ est tiré selon le contrat. Les options `vertical` et
        `tilted` fixent un départ de démonstration, avec 0° ou 5° d'inclinaison.
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        options = options or {}
        start = options.get("start", "contract")
        if start in ("vertical", "tilted"):
            tilt = radians(5) if start == "tilted" else 0.0
            self.data.qpos[3:7] = (cos(tilt / 2), sin(tilt / 2), 0, 0)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
            base = self.model.site_pos[self._deck_center] + (0, 0, 100)
            self.data.qpos[:3] = base - rotation.reshape(3, 3) @ self.model.site_pos[self._base]
            self.data.qvel[2] = -10.0
        elif start == "contract":
            self._sample_start()
        else:
            raise ValueError(f"Départ inconnu : {start!r}")
        mujoco.mj_forward(self.model, self.data)
        self._decisions = 0
        self._episode_done = False
        self._impact_speed = None
        self._impact_tilt = None
        self._previous_feet = set()
        return self._observation(self._contacts()[0]), self._info("flying")

    def _sample_start(self):
        """Tire les conditions du contrat et positionne la base dans le repère du pont."""
        rng = self.np_random
        tilt = radians(rng.uniform(0.0, 10.0))
        heading, roll = rng.uniform(0.0, 2 * pi, size=2)
        q_tilt = np.array((cos(tilt / 2), cos(heading) * sin(tilt / 2), sin(heading) * sin(tilt / 2), 0))
        q_roll = np.array((cos(roll / 2), 0, 0, sin(roll / 2)))
        quaternion, rotation = np.empty(4), np.empty(9)
        mujoco.mju_mulQuat(quaternion, q_tilt, q_roll)
        mujoco.mju_quat2Mat(rotation, quaternion)
        base = self.model.site_pos[self._deck_center] + np.array((*_disk(rng, 30), rng.uniform(80, 120)))
        self.data.qpos[:3] = base - rotation.reshape(3, 3) @ self.model.site_pos[self._base]
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:3] = (*_disk(rng, 3), rng.uniform(-15, -5))

    def step(self, action):
        """Applique la poussée et les deux angles de tuyère pendant une décision.

        Inspecte les contacts avant de résoudre chaque sous-pas pour mesurer la
        vitesse d'impact. Renvoie observation, récompense, fin physique, troncature
        et informations. Une fin physique peut écourter la décision.
        """
        if self._episode_done:
            raise gym.error.ResetNeeded("Appelle reset() avant de commencer un nouvel épisode.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError("Trois commandes finies dans [-1, 1] sont attendues.")

        angle_x, angle_y = action[1:] * GIMBAL_LIMIT_RAD
        # Rotation locale Rx puis Ry du vecteur vertical.
        local_force = thrust_newtons(action[0]) * np.array(
            (sin(angle_y) * cos(angle_x), -sin(angle_x), cos(angle_y) * cos(angle_x))
        )
        outcome = "flying"
        for _ in range(self.substeps):
            # mj_step1 met à jour les positions et contacts avant leur résolution par mj_step2.
            mujoco.mj_step1(self.model, self.data)
            feet, outcome = self._inspect_state()
            if outcome != "flying":
                break
            self._apply_force(local_force)
            mujoco.mj_step2(self.model, self.data)
        else:
            # Observe l'état effectivement atteint après le dernier sous-pas.
            mujoco.mj_step1(self.model, self.data)
            feet, outcome = self._inspect_state()

        self._decisions += 1
        terminated = outcome != "flying"
        truncated = not terminated and self._decisions >= MAX_DECISIONS
        if truncated:
            outcome = "timeout"
        self._episode_done = terminated or truncated
        reward = OUTCOME_REWARDS.get(outcome, -100.0)
        return self._observation(feet), reward, terminated, truncated, self._info(outcome)

    def _apply_force(self, local_force):
        """Convertit la poussée locale en force et couple dans le repère du monde."""
        force = self.data.xmat[self._rocket].reshape(3, 3) @ local_force
        lever = self.data.site_xpos[self._nozzle] - self.data.xipos[self._rocket]
        wrench = self.data.xfrc_applied[self._rocket]
        wrench[:3] = force
        mujoco.mju_cross(wrench[3:], lever, force)

    def _contacts(self):
        """Recense les pieds sur le dessus du pont et les contacts avec le corps ou la mer."""
        feet = {}
        crash = water = False
        for contact in self.data.contact:
            first, second = contact.geom
            normal_z = contact.frame[2]
            if second in (self._deck, self._sea):
                first, second = second, first
                normal_z = -normal_z
            if first == self._deck and second in self._feet:
                # La normale, orientée du pont vers le pied, distingue dessus et parois.
                if normal_z > 0.5:
                    feet[self._feet[second]] = contact.pos.copy()
            elif first == self._sea and second in self._feet:
                water = True
            elif first in (self._deck, self._sea):
                crash = True
        return feet, crash, water

    def _inspect_state(self):
        """Détermine l'issue courante et mémorise la vitesse et l'angle au premier contact."""
        feet, crash, water = self._contacts() if self.data.ncon else ({}, False, False)
        x, y, height = self._relative_position()
        new_feet = feet.keys() - self._previous_feet
        self._previous_feet = set(feet)
        hard = False
        if new_feet:
            # La vitesse inclut la rotation : deux pieds peuvent toucher à des vitesses différentes.
            speeds = [self._velocity_at(feet[i])[2] for i in sorted(new_feet)]
            if self._impact_speed is None:
                self._impact_speed = float(min(speeds))
                self._impact_tilt = self._tilt_deg()
            hard = min(speeds) < -2.0
        if crash:
            return feet, "crash"
        if water or (height < 0 and max(abs(x), abs(y)) > DECK_HALF_M):
            return feet, "water"
        if hard:
            return feet, "hard_landing"
        if (
            len(feet) == 4
            and np.linalg.norm(self._velocity_at(self.data.site_xpos[self._base])) < 0.2
            and np.linalg.norm(self.data.qvel[3:6]) < 0.1
        ):
            return feet, "landed"
        if hypot(x, y) > 100 or height > 200 or self.data.xmat[self._rocket, 8] < MIN_UPRIGHT_Z:
            return feet, "out_of_bounds"
        return feet, "flying"

    def _relative_position(self):
        """Donne le déplacement de la base par rapport au centre du pont, en mètres."""
        return self.data.site_xpos[self._base] - self.data.site_xpos[self._deck_center]

    def _velocity_at(self, point):
        """Calcule la vitesse d'un point solidaire de la fusée, en m/s dans le repère du monde.

        La vitesse de translation s'ajoute à la contribution de la rotation.
        """
        angular_world = self.data.xmat[self._rocket].reshape(3, 3) @ self.data.qvel[3:6]
        return self.data.qvel[:3] + np.cross(angular_world, point - self.data.xpos[self._rocket])

    def _tilt_deg(self):
        """Mesure l'angle en degrés entre l'axe de la fusée et la verticale."""
        return float(np.degrees(np.arccos(np.clip(self.data.xmat[self._rocket, 8], -1.0, 1.0))))

    def _observation(self, feet):
        """Construit le vecteur float32 lu par l'agent, dans un ordre fixe.

        Ordre : position (3), vitesse de la base (3), axes x et z (6), vitesse
        angulaire (3) et contacts des pieds dans l'ordre de `FEET` (4).
        """
        rotation = self.data.xmat[self._rocket].reshape(3, 3)
        return np.array(
            (
                *(self._relative_position() / 100),
                *(self._velocity_at(self.data.site_xpos[self._base]) / 20),
                *rotation[:, 0],
                *rotation[:, 2],
                *(self.data.qvel[3:6] / 2),
                *(float(i in feet) for i in range(4)),
            ),
            dtype=np.float32,
        )

    def _info(self, outcome):
        """Expose le temps, l'issue et les mesures d'atterrissage pour la démo et l'évaluation."""
        return {
            "outcome": outcome,
            "is_success": outcome == "landed",
            "time_s": float(self.data.time),
            "distance_to_center_m": float(np.linalg.norm(self._relative_position()[:2])),
            "impact_vertical_m_s": self._impact_speed,
            "impact_tilt_deg": self._impact_tilt,
        }
