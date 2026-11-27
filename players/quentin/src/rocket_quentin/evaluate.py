"""Évalue une politique sans apprentissage sur une barge fixe ou mobile."""

from argparse import ArgumentParser
from collections import Counter
from collections.abc import Callable
from json import dumps
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_quentin.demonstrations import ENTROPY_COMPARISON_SEEDS
from rocket_quentin.env import OBSERVATION_SIZES, RocketLandingEnv
from rocket_quentin.pilot_lateral import lateral_action
from rocket_quentin.pulse_throttle import PulseThrottle

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
    if phase == 2:
        result["fuel_remaining_mean_kg"] = float(np.mean(fuel)) if fuel else None
    return result


def sac_policy(model: SAC, normalizer: VecNormalize) -> Callable:
    def action(observation: np.ndarray) -> np.ndarray:
        normalized = normalizer.normalize_obs(observation)
        return model.predict(normalized, deterministic=True)[0]

    return action


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="dossier du modèle et de sa normalisation")
    source.add_argument("--model", type=Path, help="modèle .zip accompagné d'un fichier .pkl du même nom")
    parser.add_argument(
        "--phase", type=int, choices=(0, 1, 2), default=0, help="0 : fixe, 1 : mobile, 2 : carburant"
    )
    parser.add_argument(
        "--pulse-throttle", action="store_true", help="adapte la poussée au moteur de phase 2"
    )
    parser.add_argument(
        "--entropy-comparison", action="store_true", help="compare sur les graines 6200 à 6239"
    )
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
    if args.entropy_comparison and (args.phase or args.development or args.development_level is not None):
        parser.error("--entropy-comparison utilise les départs complets de phase 0.")
    if args.pulse_throttle and args.phase != 2:
        parser.error("--pulse-throttle nécessite --phase 2.")

    def report(policy):
        controller = PulseThrottle(policy) if args.pulse_throttle else None
        if controller is not None:
            policy = controller
        options = {
            "altitude_only": args.altitude_only,
            "phase": args.phase,
            "reset_policy": controller.reset if controller is not None else None,
        }
        if args.entropy_comparison:
            return evaluate_policy(policy, ENTROPY_COMPARISON_SEEDS)
        if args.development_level is not None:
            return evaluate_policy(policy, COMPARISON_SEEDS, args.development_level, **options)
        if args.development:
            return {
                str(difficulty): evaluate_policy(policy, COMPARISON_SEEDS, difficulty, **options)
                for difficulty in COMPARISON_DIFFICULTIES
            }
        return evaluate_policy(policy, CONTRACT_SEEDS, **options)

    if args.run_dir is None and args.model is None:
        print(dumps(report(lateral_action), indent=2))
        return

    if args.model:
        model_path = args.model
        normalizer_path = model_path.with_suffix(".pkl")
    else:
        model_path = args.run_dir / "model.zip"
        normalizer_path = args.run_dir / "vec_normalize.pkl"
    torch.set_num_threads(1)
    model = SAC.load(model_path, device="cpu")
    observation_size = model.observation_space.shape[0]
    if observation_size not in OBSERVATION_SIZES[: args.phase + 1]:
        parser.error("Le format d'observation de ce modèle est incompatible avec la phase demandée.")
    normalizer_phase = OBSERVATION_SIZES.index(observation_size)
    normalizer = VecNormalize.load(
        normalizer_path, DummyVecEnv([lambda: RocketLandingEnv(phase=normalizer_phase)])
    )
    normalizer.training = False
    try:
        policy = sac_policy(model, normalizer)
        if normalizer_phase != args.phase:
            # Les nouvelles mesures sont insérées avant les quatre contacts.
            prefix = observation_size - 4
            result = report(
                lambda observation: policy(np.concatenate((observation[:prefix], observation[-4:])))
            )
        else:
            result = report(policy)
        print(dumps(result, indent=2))
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
