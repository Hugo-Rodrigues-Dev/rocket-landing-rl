"""Protocole commun de phase 0 : commandes, récompense et mesures des vols."""

from collections import Counter
from statistics import mean, median

import gymnasium as gym
import numpy as np

from rocket_hugo.env import RocketLandingEnv
from rocket_hugo.phase0_pilot import pilot_action

DEVELOPMENT_SEEDS = range(7200, 7220)
OFFICIAL_SEEDS = range(10_000, 10_100)
RESIDUAL_SCALE = np.array((0.12, 0.22, 0.22))
CONTROL_MODES = ("residual", "direct")


# Additionne la correction au pilote, sauf quand celui-ci coupe le moteur à l'arrivée.
def apply_residual(observation, action):
    base_action = pilot_action(observation)
    if base_action[0] == -1.0:
        return base_action, 0.0
    command = np.clip(base_action + RESIDUAL_SCALE * action, -1, 1)
    return command, float(np.linalg.norm(command - base_action))


# Interprète l'action du réseau et l'évalue dans le simulateur inchangé.
# En mode résiduel, le réseau corrige un pilote manuel aux trois commandes.
class Phase0Experiment(gym.Wrapper):
    # Crée un simulateur indépendant et fixe la façon d'interpréter les actions.
    def __init__(self, control):
        if control not in CONTROL_MODES:
            raise ValueError(f"Mode de pilotage inconnu : {control!r}")
        super().__init__(RocketLandingEnv())
        self.control = control

    # Mémorise l'observation de départ pour composer la première commande.
    def reset(self, **kwargs):
        self.observation, info = super().reset(**kwargs)
        return self.observation, info

    # Joue une commande et ajoute un signal de précision commun aux deux algorithmes.
    def step(self, action):
        command = action
        correction = 0.0
        if self.control == "residual":
            command, correction = apply_residual(self.observation, action)

        self.observation, reward, terminated, truncated, info = self.env.step(command)
        distance = info["distance_to_center_m"]
        # Un petit coût par décision décourage l'attente ; le bonus départage les poses.
        reward -= 0.015 + min(distance, 30.0) * 0.001
        if info["is_success"]:
            reward += 10.0 * np.exp(-distance) - 2.0 * abs(info["impact_vertical_m_s"] or 0.0)
        info["correction_norm"] = correction
        return self.observation, float(reward), terminated, truncated, info


# Rejoue le même lot de graines pour n'importe quel pilote, réseau ou commande fixe.
def evaluate(env, policy, seeds):
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total_reward = corrections = 0.0
        decisions = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            decisions += 1
            total_reward += reward
            corrections += info.get("correction_norm", 0.0)
            if terminated or truncated:
                episodes.append(
                    {
                        "seed": seed,
                        **info,
                        "return": total_reward,
                        "decisions": decisions,
                        "mean_correction_norm": corrections / decisions,
                    }
                )
                break

    outcomes = Counter(item["outcome"] for item in episodes)
    contacts = [item for item in episodes if item["impact_vertical_m_s"] is not None]
    distances = [item["distance_to_center_m"] for item in episodes if item["is_success"]]
    return {
        "episode_count": len(episodes),
        "landed": outcomes["landed"],
        "outcomes": dict(outcomes),
        "median_distance_m": median(distances) if distances else None,
        "mean_impact_speed_m_s": mean(item["impact_vertical_m_s"] for item in contacts) if contacts else None,
        "mean_impact_tilt_deg": mean(item["impact_tilt_deg"] for item in contacts) if contacts else None,
        "mean_correction_norm": mean(item["mean_correction_norm"] for item in episodes),
        "episodes": episodes,
    }
