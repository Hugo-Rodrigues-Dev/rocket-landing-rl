"""Évalue une politique sans apprentissage sur les départs de la phase 0."""

from argparse import ArgumentParser
from collections import Counter
from collections.abc import Callable
from json import dumps
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_quentin.env import RocketLandingEnv
from rocket_quentin.pilot_lateral import lateral_action

CONTRACT_SEEDS = range(10_000, 10_100)
DEVELOPMENT_SEEDS = range(5_000, 5_020)
COMPARISON_SEEDS = range(5_100, 5_120)
COMPARISON_DIFFICULTIES = (0.0, 0.1, 0.2)


def evaluate_policy(policy: Callable, seeds: range, difficulty: float = 1.0, *, altitude_only=False) -> dict:
    env = RocketLandingEnv(start_difficulty=difficulty, altitude_only=altitude_only)
    outcomes = Counter()
    out_of_bounds_tilt = 0
    distances = []
    speeds = []
    tilts = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            while True:
                observation, _, terminated, truncated, info = env.step(policy(observation))
                if terminated or truncated:
                    break
            outcome = info["outcome"]
            outcomes[outcome] += 1
            out_of_bounds_tilt += int(info.get("out_of_bounds_tilt", False))
            if outcome == "landed":
                distances.append(np.linalg.norm(observation[:2]))
            if "impact_speed_m_s" in info:
                speeds.append(info["impact_speed_m_s"])
                tilts.append(info["impact_tilt_deg"])
    finally:
        env.close()

    return {
        "episodes": len(seeds),
        "success_rate": outcomes["landed"] / len(seeds),
        "precision_median_m": float(np.median(distances)) if distances else None,
        "impact_speed_mean_m_s": float(np.mean(speeds)) if speeds else None,
        "impact_tilt_mean_deg": float(np.mean(tilts)) if tilts else None,
        "outcomes": dict(outcomes),
        "out_of_bounds_tilt": out_of_bounds_tilt,
    }


def sac_policy(model: SAC, normalizer: VecNormalize) -> Callable:
    def action(observation: np.ndarray) -> np.ndarray:
        normalized = normalizer.normalize_obs(observation)
        return model.predict(normalized, deterministic=True)[0]

    return action


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="dossier du modèle SAC et de sa normalisation")
    parser.add_argument(
        "--development",
        action="store_true",
        help="compare les niveaux 0, 0,1 et 0,2 sur les graines 5100 à 5119",
    )
    parser.add_argument(
        "--development-level",
        type=float,
        help="évalue un seul niveau de difficulté sur les graines 5100 à 5119",
    )
    parser.add_argument(
        "--altitude-only", action="store_true", help="fait varier seulement la hauteur de départ"
    )
    args = parser.parse_args()
    if args.development_level is not None and not 0 <= args.development_level <= 1:
        parser.error("--development-level doit être compris entre 0 et 1.")
    if args.altitude_only and not (args.development or args.development_level is not None):
        parser.error("--altitude-only est réservé aux mesures de développement.")

    def report(policy):
        if args.development_level is not None:
            return evaluate_policy(
                policy, COMPARISON_SEEDS, args.development_level, altitude_only=args.altitude_only
            )
        if args.development:
            return {
                str(difficulty): evaluate_policy(
                    policy, COMPARISON_SEEDS, difficulty, altitude_only=args.altitude_only
                )
                for difficulty in COMPARISON_DIFFICULTIES
            }
        return evaluate_policy(policy, CONTRACT_SEEDS)

    if args.run_dir is None:
        print(dumps(report(lateral_action), indent=2))
        return

    torch.set_num_threads(1)
    normalizer = VecNormalize.load(args.run_dir / "vec_normalize.pkl", DummyVecEnv([RocketLandingEnv]))
    normalizer.training = False
    try:
        model = SAC.load(args.run_dir / "model.zip", device="cpu")
        print(dumps(report(sac_policy(model, normalizer)), indent=2))
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
