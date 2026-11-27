"""Enrichit l'imitation avec les corrections du pilote sur les vols du réseau."""

from argparse import ArgumentParser
from json import dumps
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_quentin.demonstrations import (
    DAGGER_COMPARISON_SEEDS,
    DEMONSTRATION_SEEDS,
    IMITATION_VALIDATION_SEEDS,
    successful_examples,
)
from rocket_quentin.env import RocketLandingEnv
from rocket_quentin.evaluate import evaluate_policy, sac_policy
from rocket_quentin.pilot_lateral import lateral_action
from rocket_quentin.train_imitation import fit_actor

DAGGER_COLLECTION = (range(3000, 3032), range(3032, 3064))
DAGGER_EPOCHS = 10


def collect_corrections(policy, seeds: range) -> tuple[np.ndarray, np.ndarray]:
    observations, corrections = [], []
    env = RocketLandingEnv()
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            while True:
                observations.append(observation)
                corrections.append(lateral_action(observation))
                observation, _, terminated, truncated, _ = env.step(policy(observation))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(corrections, dtype=np.float32)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--imitation-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("players/quentin/runs/demonstrations-phase0.npz"))
    args = parser.parse_args()
    if not args.data.is_file() or not all(
        (args.imitation_dir / name).is_file() for name in ("model.zip", "vec_normalize.pkl")
    ):
        parser.error("Indique les démonstrations et le dossier du checkpoint d'imitation.")
    with np.load(args.data) as archive:
        if not np.array_equal(archive["seeds"], list(DEMONSTRATION_SEEDS)):
            parser.error("Le fichier de démonstrations doit utiliser les graines 2000 à 2255.")
        examples = successful_examples(archive)

    torch.set_num_threads(1)
    model = SAC.load(args.imitation_dir / "model.zip", device="cpu")
    runs = Path("players/quentin/runs")
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix=f"dagger-s{model.seed}-", dir=runs))
    print(f"Résultats : {run_dir}", flush=True)
    normalizer = VecNormalize.load(args.imitation_dir / "vec_normalize.pkl", DummyVecEnv([RocketLandingEnv]))
    normalizer.training = False
    config = {
        "seed": model.seed,
        "imitation_dir": str(args.imitation_dir),
        "data": str(args.data),
        "collection_seeds": [list(seeds) for seeds in DAGGER_COLLECTION],
        "epochs_per_round": DAGGER_EPOCHS,
        "validation_seeds": list(IMITATION_VALIDATION_SEEDS),
        "test_seeds": list(DAGGER_COMPARISON_SEEDS),
    }
    (run_dir / "config.json").write_text(dumps(config, indent=2), encoding="utf-8")
    try:
        with wandb.init(
            project="rocket-landing-rl",
            group="quentin",
            tags=["phase0", "dagger", "imitation"],
            name=run_dir.name,
            mode="offline",
            dir=str(run_dir),
            config=config,
        ) as run:
            before = evaluate_policy(sac_policy(model, normalizer), DAGGER_COMPARISON_SEEDS)
            print(f"Imitation seule | poses {before['outcomes'].get('landed', 0)}/40", flush=True)
            rounds = []
            for number, seeds in enumerate(DAGGER_COLLECTION, start=1):
                corrections = collect_corrections(sac_policy(model, normalizer), seeds)
                examples = (
                    np.concatenate((examples[0], corrections[0])),
                    np.concatenate((examples[1], corrections[1])),
                )
                directory = run_dir / f"round-{number}"
                print(f"DAgger | tour {number} | {len(corrections[0])} nouvelles corrections", flush=True)
                history = fit_actor(
                    model, normalizer, examples, DAGGER_EPOCHS, model.seed + number, directory, run
                )
                rounds.append({"round": number, "corrections": len(corrections[0]), "validation": history})
                model = SAC.load(directory / "imitation" / "model.zip", device="cpu")

            selected = max(
                rounds, key=lambda result: max(row["success_rate"] for row in result["validation"])
            )
            number = selected["round"]
            model = SAC.load(run_dir / f"round-{number}" / "imitation" / "model.zip", device="cpu")
            after = evaluate_policy(sac_policy(model, normalizer), DAGGER_COMPARISON_SEEDS)
            report = {
                "imitation_test": before,
                "rounds": rounds,
                "selected_round": number,
                "dagger_test": after,
            }
            (run_dir / "report.json").write_text(dumps(report, indent=2), encoding="utf-8")
            print(
                f"DAgger | tour {number} retenu | poses {after['outcomes'].get('landed', 0)}/40", flush=True
            )
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
