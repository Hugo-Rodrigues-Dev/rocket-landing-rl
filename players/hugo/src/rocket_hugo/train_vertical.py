"""Entraîne PPO sur l'exercice vertical et enregistre les métriques de chaque version."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rocket_hugo.vertical_env import VerticalLandingEnv

ROLLOUT_STEPS = 512
# Les deux jeux de départs sont distincts et hors des graines officielles du contrat.
DEVELOPMENT_SEEDS = range(5100, 5120)
TEST_SEEDS = range(6100, 6120)


def evaluate(model, seeds):
    """Évalue la politique déterministe sur les départs donnés, sans mise à jour.

    Compte les issues, calcule la moyenne des retours et mesure la vitesse
    moyenne sur les seuls épisodes qui touchent le pont.
    """
    env = VerticalLandingEnv()
    outcomes = Counter()
    speeds = []
    returns = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        score = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                outcomes[info["outcome"]] += 1
                if info["impact_vertical_m_s"] is not None:
                    speeds.append(abs(info["impact_vertical_m_s"]))
                returns.append(score)
                break
    return {
        "episodes": len(returns),
        "soft_contacts": outcomes["landed"],
        "outcomes": dict(sorted(outcomes.items())),
        "mean_impact_speed_m_s": mean(speeds) if speeds else None,
        "mean_return": mean(returns),
    }


def main():
    """Entraîne un PPO sur CPU et sauvegarde ses versions dans un dossier neuf.

    Mesure chaque version sur les mêmes départs de développement. Recharge
    le dernier réseau depuis son fichier pour le tester sur d'autres graines.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--every", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed < 5000:
        parser.error("La graine d'entraînement doit être comprise entre 0 et 4999.")
    if min(args.steps, args.every) <= 0 or args.steps % ROLLOUT_STEPS or args.every % ROLLOUT_STEPS:
        parser.error(f"--steps et --every doivent être des multiples positifs de {ROLLOUT_STEPS}.")

    torch.set_num_threads(1)
    check_env(VerticalLandingEnv(), skip_render_check=True)
    args.out.mkdir(parents=True, exist_ok=False)
    model = PPO(
        "MlpPolicy",
        VerticalLandingEnv(),
        seed=args.seed,
        device="cpu",
        n_steps=ROLLOUT_STEPS,
        batch_size=128,
        gamma=0.995,
        policy_kwargs={"net_arch": [32, 32], "log_std_init": -0.6},
    )
    report = {
        "algorithm": "PPO",
        "training_seed": args.seed,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "checkpoints": [],
    }
    learn_seconds = 0.0
    while True:
        # Le premier passage capture le réseau initial, avant la première collecte de vols.
        steps = model.num_timesteps
        filename = f"step-{steps:07d}.zip"
        model.save(args.out / filename)
        metrics = evaluate(model, DEVELOPMENT_SEEDS)
        report["checkpoints"].append({"steps": steps, "file": filename, **metrics})
        report["learn_seconds"] = round(learn_seconds, 2)
        print(
            f"{steps:>7} décisions | {metrics['soft_contacts']:>2}/{metrics['episodes']} contacts doux",
            flush=True,
        )
        if steps == args.steps:
            # Un jeu séparé mesure le dernier modèle, sans choisir un checkpoint a posteriori.
            report["final_test"] = evaluate(PPO.load(args.out / filename, device="cpu"), TEST_SEEDS)
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if steps == args.steps:
            break
        started = perf_counter()
        # Conserve le compteur et l'optimiseur entre deux sauvegardes d'un même entraînement.
        model.learn(total_timesteps=min(args.every, args.steps - steps), reset_num_timesteps=False)
        learn_seconds += perf_counter() - started


if __name__ == "__main__":
    main()
