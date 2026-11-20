"""Paramètres de l'environnement. Valeurs par défaut = contrat (CONTRACT.md)."""

from dataclasses import dataclass, replace

# Départ facile (difficulté 0) : quelques mètres au-dessus de la cible, droit et lent,
# pour que l'agent découvre l'atterrissage (et la coupure du moteur au contact) dès le début.
EASY_START = {
    "altitude": (2.0, 6.0),
    "offset_radius": 1.0,
    "vertical_speed": (-1.5, -0.5),
    "horizontal_speed": 0.2,
    "max_tilt_deg": 1.0,
    "max_angular_speed": 0.01,
}


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
    engine_cutoff_on_touchdown: bool = True  # le moteur s'éteint au premier contact d'un pied
    max_distance: float = 100.0  # m, horizontal
    max_altitude: float = 200.0  # m
    max_flight_tilt_deg: float = 60.0

    # Récompense : R = γ·Φ(s') − Φ(s) + R_fin (guidage par potentiel, Ng et al. 1999).
    # gamma doit être le même que celui de l'algorithme pour ne pas changer la politique optimale.
    gamma: float = 0.995
    # Φ(s) = −(w_pos·‖Δp‖ + w_vel·‖Δv‖ + w_tilt·inclinaison + w_omega·‖ω‖) + w_feet·(pieds posés)
    w_pos: float = 0.5  # par m
    w_vel: float = 1.0  # par m/s
    w_tilt: float = 10.0  # par rad
    w_omega: float = 5.0  # par rad/s
    w_feet: float = 2.5  # par pied posé sur le pont
    landed_reward: float = 100.0
    precision_bonus: float = 50.0  # en plus, au centre exact (0 au bord du pont)
    failure_reward: float = -100.0

    def at_difficulty(self, difficulty: float) -> "EnvConfig":
        """Départ interpolé entre EASY_START (0) et ce config (1). Seul le départ change."""
        d = min(max(difficulty, 0.0), 1.0)

        def mix(easy, full):
            if isinstance(full, tuple):
                return tuple(e + d * (f - e) for e, f in zip(easy, full, strict=True))
            return easy + d * (full - easy)

        return replace(self, **{k: mix(v, getattr(self, k)) for k, v in EASY_START.items()})
