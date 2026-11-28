"""Rejoue la phase 0 et produit un rapport par graine."""

import argparse
import json
from pathlib import Path

import torch

from rocket_hugo.algorithms import ALGORITHMS
from rocket_hugo.env import RocketLandingEnv
from rocket_hugo.phase0 import CONTROL_MODES, DEVELOPMENT_SEEDS, OFFICIAL_SEEDS, Phase0Experiment, evaluate
from rocket_hugo.phase0_pilot import pilot_action


# Choisit le candidat sur les seules données de développement avant toute mesure officielle.
def select_candidate(directories, algo, control):
    candidates = []
    for directory in directories:
        report = json.loads((directory / "development.json").read_text(encoding="utf-8"))
        if (
            report["algo"] != algo
            or report["control"] != control
            or [item["seed"] for item in report["episodes"]] != list(DEVELOPMENT_SEEDS)
        ):
            raise ValueError(f"Rapport de développement incompatible : {directory}")
        candidates.append((directory / "model.zip", report))

    # La précision départage les modèles ayant le même nombre de poses.
    return min(
        candidates,
        key=lambda item: (
            -item[1]["landed"],
            item[1]["median_distance_m"] if item[1]["median_distance_m"] is not None else float("inf"),
            item[1]["seed"],
        ),
    )


# Sélectionne un modèle ou le pilote seul, puis évalue le lot choisi en mode déterministe.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=("pilot", *ALGORITHMS), required=True)
    parser.add_argument("--control", choices=CONTROL_MODES, default="residual")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", type=Path)
    models.add_argument(
        "--candidates", type=Path, nargs="+", help="Dossiers avec model.zip et development.json."
    )
    parser.add_argument("--split", choices=("development", "official"), default="development")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.algo != "pilot" and not (args.model or args.candidates):
        parser.error("PPO ou SAC requiert --model ou --candidates.")
    if args.algo == "pilot" and (args.model or args.candidates):
        parser.error("Le pilote de référence fonctionne sans modèle entraîné.")

    torch.set_num_threads(1)
    selection = None
    if args.algo == "pilot":
        env, policy = RocketLandingEnv(), pilot_action
    else:
        if args.candidates:
            args.model, report = select_candidate(args.candidates, args.algo, args.control)
            selection = {
                "seed": report["seed"],
                "decisions": report["actual_steps"],
                "model": str(args.model),
            }
        model = ALGORITHMS[args.algo].load(args.model, device="cpu")
        env = Phase0Experiment(args.control)

        # Joue la sortie déterministe du modèle sans exploration.
        def policy(obs):
            return model.predict(obs, deterministic=True)[0]

    seeds = OFFICIAL_SEEDS if args.split == "official" else DEVELOPMENT_SEEDS
    report = {
        "algo": args.algo,
        "control": args.control if args.model else "pilot",
        "split": args.split,
        **evaluate(env, policy, seeds),
    }
    if selection:
        report["selection"] = selection
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
