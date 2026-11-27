"""Mesures et graines d'évaluation communes aux méthodes d'apprentissage."""

from collections import Counter
from collections.abc import Callable

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from rocket_quentin.common.env import RocketLandingEnv

CONTRACT_SEEDS = range(10_000, 10_100)
DEVELOPMENT_SEEDS = range(5_000, 5_020)
COMPARISON_SEEDS = range(5_100, 5_120)
COMPARISON_DIFFICULTIES = (0.0, 0.1, 0.2)


def evaluate_policy(
    policy: Callable,
    seeds: range,
    difficulty: float = 1.0,
    *,
    altitude_only=False,
    phase: int = 0,
    reset_policy: Callable[[], None] | None = None,
) -> dict:
    env = RocketLandingEnv(start_difficulty=difficulty, altitude_only=altitude_only, phase=phase)
    outcomes = Counter()
    out_of_bounds_tilt = 0
    distances = []
    speeds = []
    tilts = []
    fuel = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            if reset_policy is not None:
                reset_policy()
            while True:
                observation, _, terminated, truncated, info = env.step(policy(observation))
                if terminated or truncated:
                    break
            outcome = info["outcome"]
            outcomes[outcome] += 1
            out_of_bounds_tilt += int(info.get("out_of_bounds_tilt", False))
            if outcome == "landed":
                distances.append(np.linalg.norm(observation[:2]))
                if "fuel_remaining_kg" in info:
                    fuel.append(info["fuel_remaining_kg"])
            if "impact_speed_m_s" in info:
                speeds.append(info["impact_speed_m_s"])
                tilts.append(info["impact_tilt_deg"])
    finally:
        env.close()

    result = {
        "episodes": len(seeds),
        "success_rate": outcomes["landed"] / len(seeds),
        "precision_median_m": float(np.median(distances)) if distances else None,
        "impact_speed_mean_m_s": float(np.mean(speeds)) if speeds else None,
        "impact_tilt_mean_deg": float(np.mean(tilts)) if tilts else None,
        "outcomes": dict(outcomes),
        "out_of_bounds_tilt": out_of_bounds_tilt,
    }
    if phase >= 2:
        result["fuel_remaining_mean_kg"] = float(np.mean(fuel)) if fuel else None
    return result


def sac_policy(model: SAC, normalizer: VecNormalize) -> Callable:
    def action(observation: np.ndarray) -> np.ndarray:
        normalized = normalizer.normalize_obs(observation)
        return model.predict(normalized, deterministic=True)[0]

    return action
