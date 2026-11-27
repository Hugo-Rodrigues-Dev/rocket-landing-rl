"""Enregistre les trajectoires du pilote sur des départs complets de phase 0."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rocket_quentin.env import RocketLandingEnv
from rocket_quentin.pilot_lateral import lateral_action
from rocket_quentin.reward import ProgressReward

DEMONSTRATION_SEEDS = range(2000, 2256)
IMITATION_VALIDATION_SEEDS = range(6000, 6020)
IMITATION_TEST_SEEDS = range(6100, 6140)
ENTROPY_COMPARISON_SEEDS = range(6200, 6240)
DAGGER_COMPARISON_SEEDS = range(6300, 6340)


def successful_examples(data) -> tuple[np.ndarray, np.ndarray]:
    episode_lengths = np.diff(np.r_[-1, np.flatnonzero(data["dones"])])
    successful_steps = np.repeat(data["outcomes"] == "landed", episode_lengths)
    return data["observations"][successful_steps], data["actions"][successful_steps]


def collect_episode(seed: int) -> dict:
    env = ProgressReward(RocketLandingEnv())
    try:
        observation, _ = env.reset(seed=seed)
        observations, actions, next_observations, rewards, dones, timeouts = [], [], [], [], [], []
        while True:
            action = lateral_action(observation)
            successor, reward, terminated, truncated, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            next_observations.append(successor)
            rewards.append(reward)
            dones.append(terminated or truncated)
            timeouts.append(truncated)
            observation = successor
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "timeouts": np.asarray(timeouts, dtype=bool),
        "outcome": info["outcome"],
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("players/quentin/runs/demonstrations-phase0.npz"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers doit être positif.")
    if args.output.exists():
        parser.error(f"Le fichier existe déjà : {args.output}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = list(pool.map(collect_episode, DEMONSTRATION_SEEDS))
    fields = ("observations", "actions", "next_observations", "rewards", "dones", "timeouts")
    data = {key: np.concatenate([episode[key] for episode in episodes]) for key in fields}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **data,
        seeds=list(DEMONSTRATION_SEEDS),
        outcomes=[episode["outcome"] for episode in episodes],
    )
    successes = sum(episode["outcome"] == "landed" for episode in episodes)
    print(
        f"{successes}/{len(episodes)} poses | {len(data['actions'])} transitions | {args.output}", flush=True
    )


if __name__ == "__main__":
    main()
