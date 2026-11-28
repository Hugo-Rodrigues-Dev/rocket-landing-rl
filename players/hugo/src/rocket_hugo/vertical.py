"""Référence analytique à un axe, partagée par la démo et l'exercice PPO."""

from dataclasses import dataclass
from math import isfinite, sqrt

MASS_KG = 1_000.0
GRAVITY = 9.81
MAX_THRUST_N = 2 * MASS_KG * GRAVITY
CONTROL_DT_S = 0.05
MAX_DECISIONS = 600


# État immuable d'un vol vertical.
#
# `height_m` se mesure au-dessus du pont. `velocity_m_s` est négative en descente.
@dataclass(frozen=True)
class VerticalState:
    height_m: float
    velocity_m_s: float


# Convertit une commande finie de [-1, 1] en newtons.
#
# -1 coupe le moteur, 0 compense le poids et +1 donne la poussée maximale.
def thrust_newtons(action: float) -> float:
    if not isfinite(action) or not -1.0 <= action <= 1.0:
        raise ValueError("La commande doit appartenir à [−1, 1].")
    return (action + 1.0) * MAX_THRUST_N / 2.0


# Intègre le mouvement avec une poussée constante pour une décision.
#
# Renvoie le nouvel état et le temps réellement écoulé en secondes. Si la
# trajectoire atteint le pont avant la fin du pas, renvoie la hauteur zéro
# et la vitesse juste avant le contact.
def advance(state: VerticalState, action: float) -> tuple[VerticalState, float]:
    height, velocity = state.height_m, state.velocity_m_s
    if height <= 0:
        raise ValueError("Le modèle démarre au-dessus du pont.")

    acceleration = thrust_newtons(action) / MASS_KG - GRAVITY
    # Résout h + v*t + a*t²/2 = 0. Cette forme de la racine évite de diviser par a
    # et fonctionne aussi quand la poussée compense exactement le poids.
    discriminant = velocity**2 - 2 * acceleration * height
    if discriminant >= 0:
        speed_at_contact = sqrt(discriminant)
        denominator = speed_at_contact - velocity
        if denominator > 0:
            contact_time = 2 * height / denominator
            if contact_time <= CONTROL_DT_S:
                return VerticalState(0.0, -speed_at_contact), contact_time

    # Sans contact pendant la décision, applique la cinématique sur le pas complet.
    dt = CONTROL_DT_S
    next_height = height + velocity * dt + acceleration * dt**2 / 2
    next_velocity = velocity + acceleration * dt
    return VerticalState(next_height, next_velocity), dt
