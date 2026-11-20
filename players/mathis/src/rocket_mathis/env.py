"""L'environnement Gymnasium : faire poser la fusée sur la barge.

    env = gymnasium.make("rocket_mathis/RocketLanding-v0", config=EnvConfig(phase=0))

Observation (21 valeurs, float32) :
    [0:3]   position du bas des pieds par rapport au centre du pont      ÷ 100 m
    [3:6]   vitesse de ce point par rapport à la barge                    ÷ 20 m/s
    [6:9]   axe x de la fusée dans le monde (colonne de la matrice de rotation)
    [9:12]  axe z de la fusée dans le monde (son axe : (0, 0, 1) = droite)
    [12:15] vitesse angulaire, repère de la fusée                         ÷ 2 rad/s
    [15:17] vitesse horizontale de la barge                               ÷ 3 m/s
    [17:21] contact de chaque pied avec le pont (0 / 1)

Action (3 valeurs dans [-1, 1]) :
    [0]   poussée     -1 → 0, +1 → T_MAX
    [1:3] tuyère      ± GIMBAL_MAX autour des axes x et y de la fusée

Récompense : R = γ·Φ(s') − Φ(s) + R_fin (voir EnvConfig). Φ guide vers le pont à chaque pas ;
R_fin vaut +100 (+50 au centre) si posée, −100 si échec.
"""

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from rocket_mathis.config import EnvConfig
from rocket_mathis.engine import GIMBAL_MAX, T_MAX, Engine
from rocket_mathis.scene import DECK_TOP_Z, FEET, SCENE_XML

DECK_HALF = 7.5  # demi-côté du pont (m)
BASE_DEPTH = 6.1  # distance de l'origine de la fusée au bas des pieds (m)
OBS_SCALES = np.array([100] * 3 + [20] * 3 + [1] * 6 + [2] * 3 + [3] * 2 + [1] * 4, dtype=np.float64)

FAILURES = ("crash", "water", "hard_landing", "out_of_bounds")


class RocketLandingEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        width: int = 960,
        height: int = 540,
        camera_name: str = "suivi",
    ):
        self.config = config or EnvConfig()
        timestep = mujoco.MjModel.from_xml_path(str(SCENE_XML)).opt.timestep
        frame_skip = round(self.config.control_dt / timestep)
        self.metadata = {**self.metadata, "render_fps": round(1 / self.config.control_dt)}
        super().__init__(
            str(SCENE_XML),
            frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=camera_name,
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        m = self.model
        self.engine = Engine(m)
        self._body = m.body("fusee").id
        self._base = m.site("base").id
        self._barge_body = m.body("barge").id
        first = m.jnt_dofadr[m.joint("barge_x").id]  # barge_x puis barge_y
        self._barge_dofs = slice(first, first + 2)
        self._deck = m.geom("pont").id
        self._sea = m.geom("mer").id
        self._feet = np.array([m.geom(name).id for name in FEET])
        rocket_geoms = np.flatnonzero((m.geom_bodyid == self._body) & (m.geom_contype > 0))
        self._hull = np.setdiff1d(rocket_geoms, self._feet)  # tout sauf les pieds
        self._barge_vel = np.zeros(2)
        self._full_config = self.config

    def set_difficulty(self, difficulty: float) -> None:
        """Curriculum : 0 = départ facile, 1 = départ du contrat. Pris en compte au prochain reset."""
        self.config = self._full_config.at_difficulty(difficulty)

    # ------------------------------------------------------------------ reset

    def reset_model(self):
        c, rng = self.config, self.np_random

        tilt = np.radians(rng.uniform(0, c.max_tilt_deg))
        heading, roll = rng.uniform(0, 2 * np.pi, size=2)
        q_roll = np.array([np.cos(roll / 2), 0, 0, np.sin(roll / 2)])
        axis = np.array([np.cos(heading), np.sin(heading), 0.0])
        q_tilt = np.concatenate([[np.cos(tilt / 2)], np.sin(tilt / 2) * axis])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q_tilt, q_roll)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)

        base = np.zeros(3)
        base[:2] = _uniform_disk(rng, c.offset_radius)
        base[2] = DECK_TOP_Z + rng.uniform(*c.altitude)
        qpos = self.init_qpos.copy()
        qpos[:3] = base + rot.reshape(3, 3)[:, 2] * BASE_DEPTH
        qpos[3:7] = quat

        qvel = np.zeros(self.model.nv)
        qvel[:2] = _uniform_disk(rng, c.horizontal_speed)
        qvel[2] = rng.uniform(*c.vertical_speed)
        qvel[3:5] = rng.uniform(-c.max_angular_speed, c.max_angular_speed, size=2)

        self._barge_vel = np.zeros(2)  # barge au centre (qpos des glissières à 0)
        if c.phase >= 1:
            std = c.barge_sigma / np.sqrt(2 * c.barge_theta)  # loi stationnaire du processus
            self._barge_vel = _clip_norm(rng.normal(0, std, size=2), c.barge_max_speed)
        self._barge_target = self._barge_vel.copy()
        qvel[self._barge_dofs] = self._barge_vel
        self.set_state(qpos, qvel)

        self._touched = False
        self._impact = {}
        self.engine.apply(self.data, 0.0, np.zeros(2))  # éteint, flamme cachée
        mujoco.mj_forward(self.model, self.data)
        self._phi = self._potential()
        return self._get_obs()

    # ------------------------------------------------------------------ step

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thrust = T_MAX * (a[0] + 1) / 2
        gimbal = a[1:] * GIMBAL_MAX

        events = {"crash": False, "water": False}
        if self._touched and self.config.engine_cutoff_on_touchdown:
            thrust = 0.0  # moteur coupé depuis le contact, comme le Falcon 9
        self.engine.command(thrust, gimbal)
        self._update_barge_target(self.config.control_dt)
        for _ in range(self.frame_skip):
            self._move_barge(self.model.opt.timestep)
            self.engine.push(self.data)
            vz_before = self.data.qvel[2] if not self._touched else 0.0
            mujoco.mj_step(self.model, self.data)
            if self.data.ncon == 0:
                continue

            feet_on_deck, crash, water = self._contacts()
            events["crash"] |= crash
            events["water"] |= water
            if feet_on_deck.any() and not self._touched:
                self._touched = True
                self._impact = {"impact_speed": vz_before, "impact_tilt_deg": self._tilt_deg()}
                if self.config.engine_cutoff_on_touchdown:
                    self.engine.command(0.0, gimbal)

        outcome = self._outcome(events)
        terminated = outcome is not None
        truncated = not terminated and self.data.time >= self.config.max_time
        if truncated:
            outcome = "timeout"

        obs = self._get_obs()
        shaping, final = self._reward(outcome, terminated)
        distance = float(np.linalg.norm(self._relative_position()[:2]))
        info = {
            "outcome": outcome or "flying",
            "is_success": outcome == "landed",
            "distance_to_center": distance,
            "reward_terms": {"shaping": shaping, "final": final},
            **self._impact,
        }
        if self.render_mode == "human":
            self.render()
        return obs, shaping + final, terminated, truncated, info

    # ------------------------------------------------------------------ récompense

    def _potential(self) -> float:
        c = self.config
        return (
            -(
                c.w_pos * np.linalg.norm(self._relative_position())
                + c.w_vel * np.linalg.norm(self._relative_velocity())
                + c.w_tilt * np.radians(self._tilt_deg())
                + c.w_omega * np.linalg.norm(self.data.qvel[3:6])
            )
            + c.w_feet * self._feet_on_deck().sum()
        )

    def _reward(self, outcome: str | None, terminated: bool) -> tuple[float, float]:
        """(guidage γ·Φ(s') − Φ(s), récompense de fin)."""
        c = self.config
        # Φ calculé aussi en fin d'épisode : finir loin ou de travers coûte, finir posé rapporte.
        # (Avec Φ = 0 imposé à la fin, sortir de la zone loin du pont « encaissait » +|Φ| d'un coup.)
        phi_next = self._potential()
        shaping = c.gamma * phi_next - self._phi
        self._phi = phi_next
        final = 0.0
        if outcome == "landed":
            distance = np.linalg.norm(self._relative_position()[:2])
            final = c.landed_reward + c.precision_bonus * max(0.0, 1.0 - distance / DECK_HALF)
        elif outcome in FAILURES:
            final = c.failure_reward
        return float(shaping), float(final)

    # ------------------------------------------------------------------ état

    def _get_obs(self) -> np.ndarray:
        rot = self.data.xmat[self._body].reshape(3, 3)
        feet_on_deck = self._feet_on_deck()
        obs = np.concatenate(
            [
                self._relative_position(),
                self._relative_velocity(),
                rot[:, 0],
                rot[:, 2],
                self.data.qvel[3:6],
                self._barge_vel,
                feet_on_deck.astype(np.float64),
            ]
        )
        return (obs / OBS_SCALES).astype(np.float32)

    def _target(self) -> np.ndarray:
        """Centre de la surface du pont, dans le monde."""
        return self.data.xpos[self._barge_body] + np.array([0.0, 0.0, DECK_TOP_Z])

    def _relative_position(self) -> np.ndarray:
        return self.data.site_xpos[self._base] - self._target()

    def _relative_velocity(self) -> np.ndarray:
        """Vitesse du bas des pieds par rapport à la barge."""
        rot = self.data.xmat[self._body].reshape(3, 3)
        omega_world = rot @ self.data.qvel[3:6]
        lever = self.data.site_xpos[self._base] - self.data.xpos[self._body]
        v_base = self.data.qvel[:3] + np.cross(omega_world, lever)
        return v_base - np.append(self.data.qvel[self._barge_dofs], 0.0)

    def _tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.data.xmat[self._body][8], -1.0, 1.0))))

    def _feet_on_deck(self) -> np.ndarray:
        """Capteur de pied : le bas du pied est à moins de 1 cm du pont, au-dessus du pont.

        Un test de contact strict ne suffit pas : posée bien droite, une fusée à 4 pieds peut
        reposer sur 3 pieds avec le quatrième à 2 mm, et ne serait jamais déclarée « posée ».
        """
        rot = self.data.xmat[self._barge_body].reshape(3, 3)
        rel = (self.data.geom_xpos[self._feet] - self.data.xpos[self._barge_body]) @ rot  # repère barge
        bottom = rel[:, 2] - self.model.geom_size[self._feet, 0] - DECK_TOP_Z
        over_deck = np.abs(rel[:, :2]).max(axis=1) <= DECK_HALF
        return over_deck & (bottom < 0.01)

    def _contacts(self):
        """(pieds sur le pont [4], crash, à l'eau) d'après les contacts du pas courant."""
        feet_on_deck = np.zeros(4, dtype=bool)
        crash = water = False
        for con in self.data.contact[: self.data.ncon]:
            pair = {con.geom1, con.geom2}
            for i, foot in enumerate(self._feet):
                if pair == {foot, self._deck}:
                    feet_on_deck[i] = True
                elif pair == {foot, self._sea}:
                    water = True
            if pair & set(self._hull) and pair & {self._deck, self._sea}:
                crash = True
        return feet_on_deck, crash, water

    def _outcome(self, events) -> str | None:
        c = self.config
        if events["crash"]:
            return "crash"
        rel = self._relative_position()
        below_deck = rel[2] < -0.5 and np.abs(rel[:2]).max() > DECK_HALF
        if events["water"] or below_deck:
            return "water"
        if self._touched and self._impact["impact_speed"] < -c.hard_landing_speed:
            return "hard_landing"
        if (
            self._feet_on_deck().all()
            and np.linalg.norm(self._relative_velocity()) < c.landed_speed
            and np.linalg.norm(self.data.qvel[3:6]) < c.landed_angular_speed
        ):
            return "landed"
        if (
            np.linalg.norm(rel[:2]) > c.max_distance
            or rel[2] > c.max_altitude
            or self._tilt_deg() > c.max_flight_tilt_deg
        ):
            return "out_of_bounds"
        return None

    # ------------------------------------------------------------------ barge

    def _update_barge_target(self, dt: float):
        """Vitesse visée de la barge : processus d'Ornstein-Uhlenbeck, tiré une fois par décision."""
        c = self.config
        if c.phase >= 1:
            noise = self.np_random.normal(size=2)
            v = (
                self._barge_target
                - c.barge_theta * self._barge_target * dt
                + c.barge_sigma * np.sqrt(dt) * noise
            )
            self._barge_target = _clip_norm(v, c.barge_max_speed)

    def _move_barge(self, dt: float):
        """La barge rejoint la vitesse visée avec une accélération limitée.

        Vitesse réimposée à chaque sous-pas : la fusée qui se pose ne doit pas pousser la barge.
        """
        step = _clip_norm(self._barge_target - self._barge_vel, self.config.barge_max_accel * dt)
        self._barge_vel = self._barge_vel + step
        self.data.qvel[self._barge_dofs] = self._barge_vel


def _uniform_disk(rng: np.random.Generator, radius: float) -> np.ndarray:
    r = radius * np.sqrt(rng.uniform())
    angle = rng.uniform(0, 2 * np.pi)
    return r * np.array([np.cos(angle), np.sin(angle)])


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v * (max_norm / norm) if norm > max_norm else v
