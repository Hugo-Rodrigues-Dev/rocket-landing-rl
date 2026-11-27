"""Entraîne SAC par paliers de difficulté sur la barge fixe."""

from argparse import ArgumentParser
from json import dumps
from pathlib import Path
from tempfile import mkdtemp

import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_quentin.env import CURRICULUM_POWER, EASY_START, RocketLandingEnv
from rocket_quentin.evaluate import (
    COMPARISON_SEEDS,
    CONTRACT_SEEDS,
    DEVELOPMENT_SEEDS,
    evaluate_policy,
    sac_policy,
)
from rocket_quentin.reward import SAC_DISCOUNT, ProgressReward
from rocket_quentin.training_stats import TrainingStatsCallback

STANDARD_LEVELS = tuple(level / 10 for level in range(11))
CURRICULA = {
    "standard": STANDARD_LEVELS,
    "intermediate": (*STANDARD_LEVELS[:2], 0.15, *STANDARD_LEVELS[2:]),
}
PROMOTION_SUCCESS_RATE = 0.8
LEARNING_STARTS = 500
SAC_SETTINGS = {
    "learning_rate": 3e-4,
    "gamma": SAC_DISCOUNT,
    "buffer_size": 100_000,
    "learning_starts": LEARNING_STARTS,
    "batch_size": 128,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "policy_kwargs": {"net_arch": [64, 64]},
}


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_every: int,
        total_steps: int,
        levels: tuple[float, ...],
        run,
        run_dir: Path,
        *,
        altitude_only=False,
        previous_level_share=0.0,
    ):
        super().__init__()
        self.eval_every = eval_every
        self.total_steps = total_steps
        self.levels = levels
        self.altitude_only = altitude_only
        self.previous_level_share = previous_level_share
        self.run = run
        self.run_dir = run_dir
        self.level = 0
        self.history = []

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if LEARNING_STARTS <= steps < self.total_steps and steps % self.eval_every == 0:
            self._assess(allow_promotion=True)
        return True

    def _on_training_end(self) -> None:
        # SB3 appelle _on_step avant la mise à jour du réseau du pas courant.
        self._assess(allow_promotion=False)

    def _assess(self, *, allow_promotion: bool) -> None:
        difficulty = self.levels[self.level]
        report = evaluate_policy(
            sac_policy(self.model, self.training_env),
            DEVELOPMENT_SEEDS,
            difficulty,
            altitude_only=self.altitude_only,
        )
        entry = {"steps": self.num_timesteps, "difficulty": difficulty, **report}
        success_rate = report["success_rate"]
        if allow_promotion and self.level < len(self.levels) - 1 and success_rate >= PROMOTION_SUCCESS_RATE:
            self.level += 1
            self.training_env.env_method(
                "set_start_difficulty",
                self.levels[self.level],
                previous_level_share=self.previous_level_share,
            )
        entry["next_difficulty"] = self.levels[self.level]
        self.history.append(entry)
        (self.run_dir / "curriculum.json").write_text(dumps(self.history, indent=2), encoding="utf-8")
        self.run.log({f"curriculum/{key}": value for key, value in entry.items()}, step=self.num_timesteps)
        print(
            f"{self.num_timesteps} pas | difficulté {difficulty:.2f} | réussite {success_rate:.0%} "
            f"| prochain départ {entry['next_difficulty']:.2f}",
            flush=True,
        )


class FixedLevelCallback(BaseCallback):
    """Sauvegarde et mesure les modèles sur des départs non utilisés pour les promotions."""

    def __init__(self, interval: int, curriculum: CurriculumCallback, probe_levels: list[float]):
        super().__init__()
        self.interval = interval
        self.curriculum = curriculum
        self.probe_levels = probe_levels
        self.history = []

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if steps < self.curriculum.total_steps and steps % self.interval == 0:
            folder = self.curriculum.run_dir / "checkpoints" / f"step-{steps:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            self.model.save(folder / "model.zip")
            self.training_env.save(folder / "vec_normalize.pkl")
            self._assess()
        return True

    def _on_training_end(self) -> None:
        self._assess()

    def _assess(self) -> None:
        policy = sac_policy(self.model, self.training_env)
        index = self.curriculum.level
        current = self.curriculum.levels[index]
        previous = self.curriculum.levels[index - 1] if index else None
        levels = {*self.probe_levels, current}
        if previous is not None:
            levels.add(previous)
        scores = {
            str(level): evaluate_policy(
                policy, COMPARISON_SEEDS, level, altitude_only=self.curriculum.altitude_only
            )
            for level in sorted(levels)
        }
        self.history.append(
            {
                "steps": self.num_timesteps,
                "current_level": current,
                "previous_level": previous,
                "levels": scores,
            }
        )
        report_file = self.curriculum.run_dir / "fixed-levels.json"
        report_file.write_text(dumps(self.history, indent=2), encoding="utf-8")
        summary = " | ".join(
            f"niveau {level} : {report['success_rate']:.0%}" for level, report in scores.items()
        )
        print(f"{self.num_timesteps} pas | comparaison fixe | {summary}", flush=True)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward", choices=("sparse", "progress"), default="sparse")
    parser.add_argument("--curriculum", choices=tuple(CURRICULA), default="standard")
    parser.add_argument("--max-difficulty", type=float, default=1.0, help="dernier palier d'entraînement")
    parser.add_argument("--altitude-only", action="store_true")
    parser.add_argument("--previous-level-share", type=float, default=0.0)
    parser.add_argument(
        "--probe-every", type=int, default=0, help="intervalle de mesure fixe, 0 pour désactiver"
    )
    parser.add_argument("--probe-levels", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--wandb", choices=("offline", "disabled"), default="offline")
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0:
        parser.error("--steps et --eval-every doivent être positifs.")
    if args.probe_every < 0:
        parser.error("--probe-every doit être positif ou nul.")
    if any(not 0.0 <= level <= 1.0 for level in args.probe_levels):
        parser.error("--probe-levels doit contenir des difficultés entre 0 et 1.")
    levels = CURRICULA[args.curriculum]
    if args.max_difficulty not in levels:
        parser.error("--max-difficulty doit correspondre à un palier du curriculum choisi.")
    levels = levels[: levels.index(args.max_difficulty) + 1]
    if not 0.0 <= args.previous_level_share < 1.0:
        parser.error("--previous-level-share doit être compris entre 0 et 1 exclu.")
    reserved_seeds = (CONTRACT_SEEDS, DEVELOPMENT_SEEDS, COMPARISON_SEEDS)
    if args.seed < 0 or any(args.seed in seeds for seeds in reserved_seeds):
        parser.error("Choisis une graine positive ou nulle en dehors des plages réservées à l'évaluation.")

    runs_dir = Path("players") / "quentin" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{args.curriculum}-altitude" if args.altitude_only else args.curriculum
    if args.previous_level_share:
        variant += "-mixed"
    if args.max_difficulty < 1.0:
        cap_label = str(args.max_difficulty).replace(".", "p")
        variant += f"-cap{cap_label}"
    run_dir = Path(mkdtemp(prefix=f"sac-{args.reward}-{variant}-s{args.seed}-", dir=runs_dir))
    print(f"Résultats : {run_dir}", flush=True)
    config = {
        **vars(args),
        **SAC_SETTINGS,
        "easy_start": EASY_START,
        "curriculum_power": CURRICULUM_POWER,
        "curriculum_levels": levels,
        "promotion_success_rate": PROMOTION_SUCCESS_RATE,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "comparison_seeds": list(COMPARISON_SEEDS) if args.probe_every else [],
    }
    (run_dir / "config.json").write_text(dumps(config, indent=2), encoding="utf-8")

    def make_training_env():
        base_env = RocketLandingEnv(start_difficulty=0.0, altitude_only=args.altitude_only)
        return ProgressReward(base_env) if args.reward == "progress" else base_env

    torch.set_num_threads(1)
    env = VecNormalize(DummyVecEnv([make_training_env]), norm_reward=False)
    try:
        with wandb.init(
            project="rocket-landing-rl",
            group="quentin",
            tags=["phase0", "sac", args.reward, variant],
            name=run_dir.name,
            mode=args.wandb,
            dir=str(run_dir),
            config=config,
        ) as run:
            model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", **SAC_SETTINGS)
            curriculum = CurriculumCallback(
                args.eval_every,
                args.steps,
                levels,
                run,
                run_dir,
                altitude_only=args.altitude_only,
                previous_level_share=args.previous_level_share,
            )
            callbacks = [TrainingStatsCallback(args.eval_every, args.steps, run_dir)]
            if args.probe_every:
                # Mesure le modèle au palier en cours avant une éventuelle promotion au même pas.
                callbacks.append(FixedLevelCallback(args.probe_every, curriculum, args.probe_levels))
            callbacks.append(curriculum)
            model.learn(total_timesteps=args.steps, callback=callbacks)
            model.save(run_dir / "model.zip")
            env.save(run_dir / "vec_normalize.pkl")
    finally:
        env.close()


if __name__ == "__main__":
    main()
