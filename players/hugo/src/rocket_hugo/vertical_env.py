"""Transforme la référence analytique en exercice Gymnasium pour apprendre la poussée."""

from math import exp

import gymnasium as gym
import numpy as np

from rocket_hugo.vertical import VerticalState, advance

EXERCISE_STEPS = 240
SOFT_CONTACT_M_S = 0.2
HEIGHT_SCALE_M = 10.0
SPEED_SCALE_M_S = 5.0


# Exercice avec deux observations, une commande et des départs proches du pont.
#
# Le contact termine le vol. Le seuil de 0,2 m/s évalue le contact doux de cet
# exercice. Les critères de pose de la phase 0 sont implémentés dans `env.py`.
class VerticalLandingEnv(gym.Env):
    # Déclare les espaces Gymnasium et impose un reset avant la première action.
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._done = True

    # Tire une hauteur de 1 à 6 m et une vitesse entre -1,5 et -0,4 m/s.
    #
    # Réinitialise le temps et renvoie la première observation avec ses informations.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = VerticalState(self.np_random.uniform(1, 6), self.np_random.uniform(-1.5, -0.4))
        self._steps = 0
        self._done = False
        self.time_s = 0.0
        return self._observation(), self._info("flying")

    # Avance d'une commande, calcule le score et classe l'issue du vol.
    #
    # Un contact plus lent rapporte davantage. En vol, le score pénalise le temps
    # et l'écart à la vitesse visée. La limite de temps donne une troncature.
    def step(self, action):
        if self._done:
            raise gym.error.ResetNeeded("Appelle reset() avant un nouveau vol.")
        action = np.asarray(action)
        if action.shape != (1,):
            raise ValueError("Une commande de poussée est attendue.")
        self.state, elapsed = advance(self.state, float(action[0]))
        self._steps += 1
        self.time_s += elapsed

        height, speed = self.state.height_m, self.state.velocity_m_s
        outcome = "flying"
        truncated = False
        if height == 0:
            outcome = "landed" if abs(speed) < SOFT_CONTACT_M_S else "hard_landing"
            # Un signal continu distingue deux contacts trop rapides malgré la même issue.
            reward = 15 * exp(-abs(speed) / 0.4) - min(abs(speed), 10)
        elif height > 10:
            outcome, reward = "out_of_bounds", -15.0
        elif self._steps == EXERCISE_STEPS:
            outcome, reward, truncated = "timeout", -15.0, True
        else:
            # La vitesse cible diminue près du pont. Elle sert au score, pas à commander le moteur.
            target_speed = -min(2.0, max(0.12, 0.8 * height))
            reward = -0.02 - 0.05 * min(abs(speed - target_speed), 5)
        terminated = outcome != "flying" and not truncated
        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(outcome)

    # Normalise la hauteur et la vitesse en deux valeurs float32 pour le réseau.
    def _observation(self):
        return np.array(
            (self.state.height_m / HEIGHT_SCALE_M, self.state.velocity_m_s / SPEED_SCALE_M_S),
            dtype=np.float32,
        )

    # Expose l'issue, le temps simulé et la vitesse au contact pour l'évaluation.
    def _info(self, outcome):
        return {
            "outcome": outcome,
            "is_success": outcome == "landed",
            "impact_vertical_m_s": self.state.velocity_m_s if self.state.height_m == 0 else None,
            "time_s": self.time_s,
        }
