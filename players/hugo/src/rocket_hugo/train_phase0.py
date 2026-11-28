"""Entraîne PPO ou SAC sur les départs complets de la phase 0."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3.common.callbacks import BaseCallback

from rocket_hugo.algorithms import ALGORITHMS, make_model
from rocket_hugo.phase0 import CONTROL_MODES, DEVELOPMENT_SEEDS, Phase0Experiment, evaluate


# Sauvegarde la version initiale et des versions intermédiaires après leurs mises à jour.
class TrainingSnapshots(BaseCallback):
    # Reçoit un intervalle en décisions, aligné sur les blocs de collecte de PPO et SAC.
    def __init__(self, directory, interval):
        super().__init__()
        self.directory, self.interval = directory, interval

    # Le prochain bloc commence après les mises à jour du bloc précédent.
    def _on_rollout_start(self):
        if self.num_timesteps % self.interval == 0:
            self.model.save(self.directory / f"step-{self.num_timesteps:07d}.zip")

    # La collecte se poursuit jusqu'au budget demandé par le script.
    def _on_step(self):
        return True


# Entraîne un modèle local, l'évalue sur les départs de développement et sauvegarde tous ses résultats.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGORITHMS, required=True)
    parser.add_argument("--control", choices=CONTROL_MODES, default="residual")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument(
        "--snapshot-every", type=int, help="Intervalle des sauvegardes de rejeu, multiple de 512."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--wandb", action="store_true", help="Journalise en local avec WandB (mode hors ligne)."
    )
    args = parser.parse_args()
    if not 0 <= args.seed < 1000 or args.steps <= 0 or args.steps % 512:
        parser.error("La graine doit être entre 0 et 999 et le budget un multiple positif de 512.")
    if args.snapshot_every is not None and (args.snapshot_every <= 0 or args.snapshot_every % 512):
        parser.error("--snapshot-every doit être un multiple positif de 512.")

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    config = {"algo": args.algo, "control": args.control, "seed": args.seed, "steps": args.steps}
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="rocket-landing-rl",
            group="hugo",
            tags=["phase0", args.algo],
            config=config,
            mode="offline",
            dir=str(args.out.resolve()),
        )
    try:
        env = Phase0Experiment(args.control)
        model = make_model(args.algo, env, args.seed)
        started = perf_counter()
        snapshots = TrainingSnapshots(args.out, args.snapshot_every) if args.snapshot_every else None
        model.learn(total_timesteps=args.steps, callback=snapshots)
        elapsed = perf_counter() - started
        model.save(args.out / "model.zip")
        result = evaluate(env, lambda obs: model.predict(obs, deterministic=True)[0], DEVELOPMENT_SEEDS)
        report = {**config, "actual_steps": model.num_timesteps, "learn_seconds": elapsed, **result}
        (args.out / "development.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if run:
            run.log(
                {key: value for key, value in report.items() if isinstance(value, (float, int))},
                step=model.num_timesteps,
            )
        print(
            f"{args.algo} {args.seed}: {result['landed']}/{result['episode_count']} en développement",
            flush=True,
        )
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
