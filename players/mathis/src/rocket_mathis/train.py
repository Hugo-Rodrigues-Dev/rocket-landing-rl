"""Entraînement avec Stable-Baselines3, suivi dans WandB.

    uv run python -m rocket_mathis.train --algo ppo --phase 0 --steps 3000000
    uv run python -m rocket_mathis.train --algo sac --phase 0 --steps 500000

Curriculum : on part d'un départ facile (difficulté 0) et on monte de 0,1 dès que
80 % des 100 derniers épisodes d'entraînement se posent, jusqu'au départ du contrat (1).
À la fin, le meilleur modèle est évalué selon le contrat (graines 10 000 à 10 099).

Sorties dans runs/<nom>/ (ignoré par git) : best_model.zip, final.zip, evaluation.json.
WandB : en ligne si `wandb login` a été fait, sinon hors ligne (`wandb sync runs/<nom>/wandb/...`).
"""

import argparse
import json
import netrc
import os
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rocket_mathis.config import EnvConfig
from rocket_mathis.env import RocketLandingEnv
from rocket_mathis.evaluate import evaluate

RUNS_DIR = Path.cwd() / "runs"
OUTCOMES = ("landed", "hard_landing", "crash", "water", "out_of_bounds", "timeout")


# ---------------------------------------------------------------------------- WandB


class WandbWriter(KVWriter):
    """Envoie dans WandB tout ce que SB3 journalise (récompenses, pertes, entropie…)."""

    def write(self, key_values, key_excluded, step=0):
        numbers = {k: v for k, v in key_values.items() if isinstance(v, int | float | np.number)}
        wandb.log(numbers, step=step)

    def close(self):
        pass


def wandb_mode(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    try:
        logged_in = netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        logged_in = False
    return "online" if logged_in else "offline"


# ---------------------------------------------------------------------------- curriculum


class CurriculumCallback(BaseCallback):
    """Monte la difficulté du départ quand l'agent réussit, et journalise les issues des épisodes."""

    def __init__(self, enabled: bool, threshold=0.8, increment=0.1, window=100):
        super().__init__()
        self.enabled, self.threshold, self.increment = enabled, threshold, increment
        self.recent = deque(maxlen=window)
        self.difficulty = 0.0 if enabled else 1.0

    def _on_training_start(self):
        self.training_env.env_method("set_difficulty", self.difficulty)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.recent.append(info["outcome"])
        if not self.recent:
            return True
        counts = Counter(self.recent)
        for outcome in OUTCOMES:
            self.logger.record(f"outcomes/{outcome}", counts[outcome] / len(self.recent))
        success = counts["landed"] / len(self.recent)
        self.logger.record("curriculum/difficulty", self.difficulty)
        if (
            self.enabled
            and self.difficulty < 1.0
            and len(self.recent) == self.recent.maxlen
            and success >= self.threshold
        ):
            self.difficulty = round(min(1.0, self.difficulty + self.increment), 2)
            self.training_env.env_method("set_difficulty", self.difficulty)
            self.recent.clear()
            print(f"[curriculum] difficulté → {self.difficulty} (pas {self.num_timesteps:,})", flush=True)
        return True


# ---------------------------------------------------------------------------- modèles


def make_model(algo: str, env, gamma: float, seed: int, net: list[int]):
    policy_kwargs = {"net_arch": net}
    if algo == "ppo":
        policy_kwargs["log_std_init"] = -0.5  # écart-type initial 0,6 au lieu de 1 : moins de chaos au début
        return PPO(
            "MlpPolicy",
            env,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
        )
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=gamma,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Entraînement SB3 de la fusée")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=None, help="par défaut : 8 pour PPO, 1 pour SAC")
    parser.add_argument("--net", type=int, nargs="+", default=[256, 256], help="couches cachées")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--wandb", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)  # petits réseaux : 1 thread est 2× plus rapide que la valeur par défaut
    config = EnvConfig(phase=args.phase)
    n_envs = args.n_envs or (8 if args.algo == "ppo" else 1)
    name = args.name or f"mathis-phase{args.phase}-{args.algo}-s{args.seed}-{time.strftime('%m%d-%H%M')}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project="rocket-landing-rl",
        group="mathis",
        name=name,
        tags=[f"phase{args.phase}", args.algo],
        config={**vars(args), "n_envs": n_envs, "env": asdict(config)},
        mode=wandb_mode(args.wandb),
        dir=run_dir,
    )

    def make_env(rank: int):
        def _init():
            return Monitor(RocketLandingEnv(config))

        return _init

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = vec_cls([make_env(i) for i in range(n_envs)])
    train_env.seed(args.seed)
    eval_env = DummyVecEnv([lambda: Monitor(RocketLandingEnv(config))])  # toujours au départ du contrat
    eval_env.seed(args.seed + 1_000)  # graines d'entraînement, jamais celles de l'évaluation finale

    model = make_model(args.algo, train_env, config.gamma, args.seed, args.net)
    model.set_logger(Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), WandbWriter()]))
    callbacks = CallbackList(
        [
            CurriculumCallback(enabled=not args.no_curriculum),
            EvalCallback(
                eval_env,
                n_eval_episodes=20,
                eval_freq=max(25_000 // n_envs, 1),
                best_model_save_path=str(run_dir),
                deterministic=True,
                verbose=0,
            ),
        ]
    )

    start = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    model.save(run_dir / "final")
    print(f"Entraînement terminé en {(time.time() - start) / 60:.1f} min")

    # Évaluation du contrat sur le meilleur modèle (sinon le dernier)
    best_path = run_dir / "best_model.zip"
    best = model.__class__.load(best_path if best_path.exists() else run_dir / "final.zip")
    results = evaluate(lambda obs: best.predict(obs, deterministic=True)[0], config)
    (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    run.summary.update({f"contract/{k}": v for k, v in results.items() if not isinstance(v, dict)})
    run.summary.update({f"contract/outcome_{k}": v for k, v in results["outcomes"].items()})
    train_env.close()
    eval_env.close()
    run.finish()


if __name__ == "__main__":
    main()
