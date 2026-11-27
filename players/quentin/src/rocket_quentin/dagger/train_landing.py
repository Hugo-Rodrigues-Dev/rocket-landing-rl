"""Entraîne une politique autonome avec des démonstrations à partir de la phase 2."""

from argparse import ArgumentParser
from json import dumps
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import wandb
from torch.nn import functional as F

from rocket_quentin.common.evaluation import evaluate_policy
from rocket_quentin.dagger.learned_landing import LandingPolicy
from rocket_quentin.dagger.phase_demonstrations import (
    DAGGER_SEEDS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    collect_episode,
)


def fit(policy, observations, targets, epochs, generator, rate):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=rate)
    for _ in range(epochs):
        for indices in torch.randperm(len(observations), generator=generator).split(256):
            samples, labels = observations[indices], targets[indices]
            predictions = policy(samples)
            error = (predictions - labels).square().mean(dim=1)
            gate = F.binary_cross_entropy_with_logits(
                24 * predictions[:, 0], (labels[:, 0] >= 0).float(), reduction="none"
            )
            weights = 1 + (samples[:, 2] < 3).float()
            loss = ((error + 0.075 * gate) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--rounds", type=int, choices=range(4), default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("--epochs doit être positif et --seed positif ou nul.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    runs = Path("players/quentin/runs")
    path = runs / f"demonstrations-phase{args.phase}.npz"
    if not path.is_file():
        parser.error(f"Collecte d'abord les vols dans {path}.")
    with np.load(path) as archive:
        if not np.array_equal(archive["seeds"], list(TRAINING_SEEDS)):
            parser.error("Graines de démonstration inattendues.")
        observations = torch.from_numpy(archive["observations"])
        targets = torch.from_numpy(archive["targets"])
    policy = LandingPolicy(args.phase)
    policy.set_statistics(observations)
    directory = Path(mkdtemp(prefix=f"landing-p{args.phase}-s{args.seed}-", dir=runs))
    config = {**vars(args), "data": str(path), "validation_seeds": list(VALIDATION_SEEDS)}
    (directory / "config.json").write_text(dumps(config, indent=2), encoding="utf-8")
    print(directory, flush=True)
    history = []
    best = (-1.0, -float("inf"))
    with wandb.init(
        project="rocket-landing-rl",
        group="quentin",
        tags=[f"phase{args.phase}", "dagger", "imitation"],
        name=directory.name,
        dir=str(directory),
        mode="offline",
        config=config,
    ) as run:
        for number in range(args.rounds + 1):
            if number:
                corrections = [
                    collect_episode(seed, args.phase, policy.action) for seed in DAGGER_SEEDS[number - 1]
                ]
                observations = torch.cat(
                    (observations, torch.from_numpy(np.concatenate([x[0] for x in corrections])))
                )
                targets = torch.cat((targets, torch.from_numpy(np.concatenate([x[1] for x in corrections]))))
            fit(
                policy,
                observations,
                targets,
                args.epochs if number == 0 else 15,
                generator,
                0.001 if number == 0 else 0.0006,
            )
            report = evaluate_policy(policy.action, VALIDATION_SEEDS, phase=args.phase)
            history.append({"round": number, "transitions": len(targets), **report})
            policy.save(directory / f"round-{number}.pt")
            score = report["success_rate"], -(report["precision_median_m"] or 0)
            if score > best:
                best = score
                policy.save(directory / "best.pt")
            (directory / "report.json").write_text(dumps(history, indent=2), encoding="utf-8")
            run.log({"round": number, "validation/success_rate": score[0], "transitions": len(targets)})
            print(
                f"Tour {number} | {report['outcomes']} | précision {report['precision_median_m']}", flush=True
            )


if __name__ == "__main__":
    main()
