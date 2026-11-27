"""Évalue une politique sans apprentissage sur une barge fixe ou mobile."""

from argparse import ArgumentParser
from functools import partial
from json import dumps
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_quentin.common.env import ACTION_SIZES, OBSERVATION_SIZES, RocketLandingEnv
from rocket_quentin.common.evaluation import (
    COMPARISON_DIFFICULTIES,
    COMPARISON_SEEDS,
    CONTRACT_SEEDS,
    evaluate_policy,
    sac_policy,
)
from rocket_quentin.common.evaluation import (
    DEVELOPMENT_SEEDS as DEVELOPMENT_SEEDS,
)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="dossier du modèle et de sa normalisation")
    source.add_argument("--model", type=Path, help="réseau .pt ou modèle .zip avec sa normalisation .pkl")
    parser.add_argument(
        "--phase",
        type=int,
        choices=range(len(OBSERVATION_SIZES)),
        default=0,
        help="0 : fixe, 1 : mobile, 2 : carburant, 3 : vagues, 4 : vent et RCS",
    )
    parser.add_argument(
        "--pulse-throttle", action="store_true", help="adapte la poussée au moteur de phase 2"
    )
    parser.add_argument("--wave-approach", action="store_true", help="aligne la fusée sur le pont au toucher")
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
    if args.pulse_throttle and args.phase not in (2, 3):
        parser.error("--pulse-throttle est réservé aux pilotes historiques des phases 2 et 3.")
    if args.wave_approach and (args.phase < 3 or not args.pulse_throttle):
        parser.error("--wave-approach nécessite la phase 3 et --pulse-throttle.")
    if (args.model or args.run_dir) and (args.pulse_throttle or args.wave_approach):
        parser.error("Les options du pilote de référence sont réservées aux essais sans modèle.")

    def report(policy, reset_policy=None):
        options = {
            "altitude_only": args.altitude_only,
            "phase": args.phase,
            "reset_policy": reset_policy,
        }
        if args.entropy_comparison:
            from rocket_quentin.dagger.demonstrations import ENTROPY_COMPARISON_SEEDS

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
        from rocket_quentin.dagger.pilot_lateral import lateral_action
        from rocket_quentin.dagger.pulse_throttle import PulseThrottle
        from rocket_quentin.dagger.reference import reference_action
        from rocket_quentin.dagger.wave_approach import wave_reference

        if args.wave_approach:
            policy = wave_reference
        elif args.pulse_throttle:
            policy = lateral_action
        else:
            policy = partial(reference_action, phase=args.phase)
        if args.pulse_throttle:
            policy = PulseThrottle(policy)
            result = report(policy, policy.reset)
        else:
            result = report(policy)
        print(dumps(result, indent=2))
        return

    torch.set_num_threads(1)
    if args.model and args.model.suffix == ".pt":
        from rocket_quentin.dagger.learned_landing import LandingPolicy

        policy = LandingPolicy.load(args.model)
        if policy.phase != args.phase:
            parser.error("Le réseau doit être évalué dans sa phase d'entraînement.")
        print(dumps(report(policy.action), indent=2))
        return
    if args.model:
        model_path = args.model
        normalizer_path = model_path.with_suffix(".pkl")
    else:
        model_path = args.run_dir / "model.zip"
        normalizer_path = args.run_dir / "vec_normalize.pkl"
    model = SAC.load(model_path, device="cpu")
    if model.action_space.shape != (ACTION_SIZES[args.phase],):
        parser.error("Le modèle doit produire toutes les commandes de la phase demandée.")
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
