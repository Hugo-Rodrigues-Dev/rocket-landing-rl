"""Paramètres de l'environnement. Valeurs par défaut = contrat (CONTRACT.md)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    phase: int = 0  # 0 : barge immobile, 1 : barge mobile

    # Temps
    control_dt: float = 0.05  # une décision toutes les 0,05 s (20 Hz)
    max_time: float = 30.0

    # Départ (tiré au hasard à chaque reset)
    altitude: tuple[float, float] = (80.0, 120.0)  # bas des pieds au-dessus du pont (m)
    offset_radius: float = 30.0  # décalage horizontal (m)
    vertical_speed: tuple[float, float] = (-15.0, -5.0)  # m/s
    horizontal_speed: float = 3.0  # m/s
    max_tilt_deg: float = 10.0
    max_angular_speed: float = 0.1  # rad/s, en tangage et en lacet (pas de roulis)

    # Barge mobile (phase 1) : vitesse visée d'Ornstein-Uhlenbeck (tirée à chaque décision),
    # que la barge suit avec une accélération limitée, comme un vrai navire.
    barge_theta: float = 0.5  # 1/s, rappel vers 0
    barge_sigma: float = 1.0  # m/s/√s
    barge_max_speed: float = 3.0  # m/s
    barge_max_accel: float = 0.5  # m/s²

    # Fins d'épisode
    landed_speed: float = 0.2  # m/s
    landed_angular_speed: float = 0.1  # rad/s
    hard_landing_speed: float = 2.0  # m/s, vitesse verticale au premier contact
    max_distance: float = 100.0  # m, horizontal
    max_altitude: float = 200.0  # m
    max_flight_tilt_deg: float = 60.0

    # Récompense : R = γ·Φ(s') − Φ(s) + R_fin, avec Φ = 0 en fin d'épisode (Ng et al. 1999).
    # gamma doit être le même que celui de l'algorithme pour ne pas changer la politique optimale.
    gamma: float = 0.995
    # Φ(s) = −(w_pos·‖Δp‖ + w_vel·‖Δv‖ + w_tilt·inclinaison + w_omega·‖ω‖)
    w_pos: float = 0.5  # par m
    w_vel: float = 1.0  # par m/s
    w_tilt: float = 10.0  # par rad
    w_omega: float = 5.0  # par rad/s
    landed_reward: float = 100.0
    precision_bonus: float = 50.0  # en plus, au centre exact (0 au bord du pont)
    failure_reward: float = -100.0
