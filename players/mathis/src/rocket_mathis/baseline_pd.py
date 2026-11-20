"""Contrôleur PD de référence, écrit à la main : la base à battre par le RL.

Il ne lit que l'observation de l'agent, comme le ferait une politique apprise :
1. vitesse verticale visée, proportionnelle à l'altitude (descente de plus en plus douce) ;
2. accélération horizontale visée, PD sur la position et la vitesse par rapport au pont ;
3. d'où la direction de poussée voulue, que l'on obtient en inclinant la fusée ;
4. PD d'attitude : l'erreur d'orientation donne un couple, que la tuyère produit.
"""

import numpy as np

from rocket_mathis.engine import GIMBAL_MAX, T_MAX

MASS = 1000.0
INERTIA = 11_632.0  # tangage / lacet (kg·m²), calculé par MuJoCo
LEVER = 3.854  # distance centre de masse → pivot de la tuyère (m)
G = 9.81


class PDController:
    def __init__(
        self,
        descent_gain=0.25,  # 1/s : vitesse visée = gain × altitude
        max_descent=15.0,  # m/s
        min_descent=0.6,  # m/s, vitesse au contact
        vertical_gain=1.5,  # 1/s
        position_gain=0.16,  # 1/s²
        velocity_gain=0.8,  # 1/s
        max_horizontal_accel=2.5,  # m/s², soit environ 14° d'inclinaison
        attitude_freq=2.5,  # rad/s
        attitude_damping=0.9,
    ):
        self.descent_gain, self.max_descent, self.min_descent = descent_gain, max_descent, min_descent
        self.kv = vertical_gain
        self.kp, self.kd = position_gain, velocity_gain
        self.max_horizontal_accel = max_horizontal_accel
        self.att_kp = INERTIA * attitude_freq**2
        self.att_kd = 2 * attitude_damping * attitude_freq * INERTIA

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        dp, dv = obs[0:3] * 100, obs[3:6] * 20
        x_axis, z_axis, omega = obs[6:9], obs[9:12], obs[12:15] * 2
        if obs[17:21].any():  # un pied touche : on coupe, la fusée se pose
            return np.array([-1.0, 0.0, 0.0])

        # 1. vertical
        vz_target = -np.clip(self.descent_gain * dp[2], self.min_descent, self.max_descent)
        az = self.kv * (vz_target - dv[2])

        # 2. horizontal, adouci près du pont pour toucher bien droit
        a_xy = -self.kp * dp[:2] - self.kd * dv[:2]
        max_accel = self.max_horizontal_accel * np.clip(dp[2] / 10.0, 0.2, 1.0)
        norm = np.linalg.norm(a_xy)
        if norm > max_accel:
            a_xy *= max_accel / norm

        # 3. poussée voulue (par unité de masse, dans le monde)
        f = np.array([a_xy[0], a_xy[1], max(G + az, 1.0)])
        z_target = f / np.linalg.norm(f)
        thrust = float(np.clip(MASS * f @ z_axis, 0.0, T_MAX))

        # 4. attitude : rotation qui amène l'axe actuel sur l'axe voulu, exprimée dans la fusée
        rotation = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
        error = rotation.T @ np.cross(z_axis, z_target)
        torque = self.att_kp * error[:2] - self.att_kd * omega[:2]
        authority = max(thrust, 0.3 * T_MAX) * LEVER  # couple disponible par unité de sin(tuyère)
        gimbal = np.arcsin(np.clip(-torque / authority, -1.0, 1.0))

        return np.clip(np.concatenate([[2 * thrust / T_MAX - 1], gimbal / GIMBAL_MAX]), -1.0, 1.0)
