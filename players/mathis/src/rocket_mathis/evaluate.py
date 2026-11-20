"""Évaluation selon le contrat : 100 épisodes, graines 10 000 à 10 099, politique déterministe.

uv run python -m rocket_mathis.evaluate pd        # contrôleur PD de référence
uv run python -m rocket_mathis.evaluate random    # actions au hasard
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable

import numpy as np

from rocket_mathis.config import EnvConfig
from rocket_mathis.env import RocketLandingEnv

EVAL_SEEDS = range(10_000, 10_100)  # réservées à l'évaluation : ne jamais entraîner dessus

Policy = Callable[[np.ndarray], np.ndarray]


def evaluate(policy: Policy, config: EnvConfig | None = None, seeds=EVAL_SEEDS) -> dict:
    env = RocketLandingEnv(config)
    outcomes, returns, distances, impact_speeds, impact_tilts = Counter(), [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
        returns.append(total)
        if info["is_success"]:
            distances.append(info["distance_to_center"])
        if "impact_speed" in info:
            impact_speeds.append(info["impact_speed"])
            impact_tilts.append(info["impact_tilt_deg"])
    env.close()
    n = len(returns)
    return {
        "episodes": n,
        "success_rate": outcomes["landed"] / n,
        "precision_median_m": float(np.median(distances)) if distances else None,
        "impact_speed_mean": float(np.mean(impact_speeds)) if impact_speeds else None,
        "impact_tilt_mean_deg": float(np.mean(impact_tilts)) if impact_tilts else None,
        "return_mean": float(np.mean(returns)),
        "outcomes": dict(outcomes),
    }


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1, 1, size=3)


def main():
    parser = argparse.ArgumentParser(description="Évaluation selon le contrat")
    parser.add_argument("policy", choices=["pd", "random"])
    parser.add_argument("--phase", type=int, default=0)
    args = parser.parse_args()
    if args.policy == "pd":
        from rocket_mathis.baseline_pd import PDController

        policy = PDController()
    else:
        policy = random_policy()
    print(json.dumps(evaluate(policy, EnvConfig(phase=args.phase)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
