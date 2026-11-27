"""Collecte des vols et des corrections du pilote à partir de la phase 2."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from rocket_quentin.env import RocketLandingEnv
from rocket_quentin.reference import teaching_action, teaching_targets

TRAINING_SEEDS = range(4000, 4160)
DAGGER_SEEDS = (range(4200, 4232), range(4232, 4264), range(4264, 4296))
VALIDATION_SEEDS = range(6700, 6720)


def collect_episode(seed: int, phase: int, policy=None) -> tuple[np.ndarray, np.ndarray, str]:
    env = RocketLandingEnv(phase=phase)
    observations, targets = [], []
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            target = teaching_targets(observation, phase)
            observations.append(observation)
            targets.append(target)
            if policy is None:
                action = teaching_action(target)
            else:
                action = policy(observation)
            observation, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
    finally:
        env.close()
    return np.asarray(observations), np.asarray(targets), info["outcome"]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, type=int, choices=(2, 3, 4))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    path = Path(f"players/quentin/runs/demonstrations-phase{args.phase}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        parser.error(f"Le fichier existe déjà : {path}")
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        episodes = list(workers.map(partial(collect_episode, phase=args.phase), TRAINING_SEEDS))
    print(Counter(episode[2] for episode in episodes), flush=True)
    successful = [episode for episode in episodes if episode[2] == "landed"]
    np.savez_compressed(
        path,
        observations=np.concatenate([episode[0] for episode in successful]),
        targets=np.concatenate([episode[1] for episode in successful]),
        seeds=list(TRAINING_SEEDS),
        outcomes=[episode[2] for episode in episodes],
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
