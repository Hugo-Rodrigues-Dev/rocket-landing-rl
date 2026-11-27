"""Préentraîne l'acteur SAC par imitation, puis affine éventuellement avec SAC."""

from argparse import ArgumentParser
from json import dumps
from math import log
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_quentin.common.env import RocketLandingEnv
from rocket_quentin.common.evaluation import (
    COMPARISON_SEEDS,
    CONTRACT_SEEDS,
    DEVELOPMENT_SEEDS,
    evaluate_policy,
    sac_policy,
)
from rocket_quentin.common.reward import ProgressReward
from rocket_quentin.dagger.demonstrations import (
    DAGGER_COMPARISON_SEEDS,
    DEMONSTRATION_SEEDS,
    ENTROPY_COMPARISON_SEEDS,
    IMITATION_TEST_SEEDS,
    IMITATION_VALIDATION_SEEDS,
    successful_examples,
)
from rocket_quentin.sac.train_sac import SAC_SETTINGS

BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-3
BC_EVAL_INTERVAL = 5
SAC_EVAL_INTERVAL = 2000
SAC_INITIAL_ACTION_STD = 0.1
IMITATION_NET = [128, 128]
FINETUNE_SETTINGS = {
    **SAC_SETTINGS,
    "learning_rate": 1e-4,
    "learning_starts": 0,
    "policy_kwargs": {"net_arch": IMITATION_NET},
}


def save_snapshot(model: SAC, normalizer: VecNormalize, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.zip")
    normalizer.save(directory / "vec_normalize.pkl")


def fit_actor(
    model: SAC, normalizer: VecNormalize, examples, epochs: int, seed: int, run_dir: Path, run
) -> list:
    expert_observations, expert_actions = examples
    observations = torch.from_numpy(normalizer.normalize_obs(expert_observations))
    actions = torch.from_numpy(expert_actions)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=BC_LEARNING_RATE)
    history = []
    best_rate = -1.0

    for epoch in range(1, epochs + 1):
        model.actor.train()
        squared_error = 0.0
        for indices in torch.randperm(len(actions), generator=generator).split(BC_BATCH_SIZE):
            predicted = model.actor(observations[indices], deterministic=True)
            loss = torch.nn.functional.mse_loss(predicted, actions[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_error += loss.detach().item() * len(indices)

        if epoch % BC_EVAL_INTERVAL == 0 or epoch == epochs:
            validation = evaluate_policy(sac_policy(model, normalizer), IMITATION_VALIDATION_SEEDS)
            score = validation["success_rate"]
            report = {"epoch": epoch, "action_mse": squared_error / len(actions), **validation}
            history.append(report)
            run.log(
                {
                    "imitation/epoch": epoch,
                    "imitation/mse": report["action_mse"],
                    "imitation/success_rate": score,
                }
            )
            print(
                f"Imitation | époque {epoch} | poses {validation['outcomes'].get('landed', 0)}/20", flush=True
            )
            if score > best_rate:
                best_rate = score
                save_snapshot(model, normalizer, run_dir / "imitation")
    return history


def seed_replay(model: SAC, data) -> None:
    fields = ("observations", "next_observations", "actions", "rewards", "dones", "timeouts")
    for observation, successor, action, reward, done, timeout in zip(
        *(data[key] for key in fields), strict=True
    ):
        model.replay_buffer.add(
            observation, successor, action, reward, done, [{"TimeLimit.truncated": bool(timeout)}]
        )


def anchor_actor(model: SAC, observations: np.ndarray, weight: float, seed: int):
    inputs = torch.from_numpy(observations)
    with torch.no_grad():
        initial_actions = model.actor(inputs, deterministic=True)
    generator = torch.Generator().manual_seed(seed)

    def add_anchor_gradient(_optimizer, _args, _kwargs) -> None:
        indices = torch.randint(len(inputs), (BC_BATCH_SIZE,), generator=generator)
        with torch.enable_grad():
            actions = model.actor(inputs[indices], deterministic=True)
            loss = weight * torch.nn.functional.mse_loss(actions, initial_actions[indices])
            # SAC a déjà calculé son gradient. On ajoute celui de la pénalité.
            loss.backward()

    return model.actor.optimizer.register_step_pre_hook(add_anchor_gradient)


def finetune(model: SAC, normalizer: VecNormalize, data, steps: int, run_dir: Path, run) -> list:
    # L'acteur vient d'être préentraîné sur son action déterministe. On initialise
    # séparément la dispersion des actions stochastiques pour le passage à SAC.
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(log(SAC_INITIAL_ACTION_STD))
    seed_replay(model, data)

    history = []
    remaining = steps
    while remaining:
        chunk = min(remaining, SAC_EVAL_INTERVAL)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        remaining -= chunk
        validation = evaluate_policy(sac_policy(model, normalizer), IMITATION_VALIDATION_SEEDS)
        decisions = model.num_timesteps
        history.append({"steps": decisions, **validation})
        save_snapshot(model, normalizer, run_dir / "checkpoints" / f"sac-{decisions:05d}")
        run.log({"sac/steps": decisions, "sac/validation_success_rate": validation["success_rate"]})
        print(f"SAC | {decisions} décisions | poses {validation['outcomes'].get('landed', 0)}/20", flush=True)
    return history


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("players/quentin/runs/demonstrations-phase0.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sac-steps", type=int, default=0)
    parser.add_argument("--imitation-dir", type=Path, help="checkpoint d'imitation à réutiliser")
    parser.add_argument(
        "--sac-anchor", type=float, default=0.0, help="poids de la pénalité d'écart à l'acteur initial"
    )
    parser.add_argument(
        "--sac-initial-entropy", type=float, default=1.0, help="coefficient initial, ajusté par SAC"
    )
    parser.add_argument(
        "--entropy-comparison", action="store_true", help="compare sur les graines 6200 à 6239"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.sac_steps < 0:
        parser.error("--epochs doit être positif et --sac-steps positif ou nul.")
    if not np.isfinite(args.sac_anchor) or args.sac_anchor < 0:
        parser.error("--sac-anchor doit être fini et positif ou nul.")
    if not np.isfinite(args.sac_initial_entropy) or args.sac_initial_entropy <= 0:
        parser.error("--sac-initial-entropy doit être fini et strictement positif.")
    if args.imitation_dir and not all(
        (args.imitation_dir / name).is_file() for name in ("model.zip", "vec_normalize.pkl")
    ):
        parser.error("--imitation-dir doit contenir le modèle et les statistiques de normalisation.")
    reserved = (
        DEVELOPMENT_SEEDS,
        COMPARISON_SEEDS,
        IMITATION_VALIDATION_SEEDS,
        IMITATION_TEST_SEEDS,
        ENTROPY_COMPARISON_SEEDS,
        DAGGER_COMPARISON_SEEDS,
        CONTRACT_SEEDS,
    )
    if args.seed < 0 or any(args.seed in seeds for seeds in reserved):
        parser.error("La graine d'entraînement doit être en dehors des plages d'évaluation.")
    if not args.data.is_file():
        parser.error("Collecte d'abord les démonstrations avec rocket_quentin.dagger.demonstrations.")

    test_seeds = ENTROPY_COMPARISON_SEEDS if args.entropy_comparison else IMITATION_TEST_SEEDS
    entropy = "auto" if args.sac_initial_entropy == 1.0 else f"auto_{args.sac_initial_entropy}"
    settings = {**FINETUNE_SETTINGS, "ent_coef": entropy}

    runs = Path("players/quentin/runs")
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix=f"imitation-s{args.seed}-", dir=runs))
    print(f"Résultats : {run_dir}", flush=True)
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: ProgressReward(RocketLandingEnv())])
    if args.imitation_dir:
        normalizer = VecNormalize.load(args.imitation_dir / "vec_normalize.pkl", env)
    else:
        normalizer = VecNormalize(env, norm_reward=False)
    try:
        with np.load(args.data) as archive:
            data = dict(archive.items())
        if not np.array_equal(data["seeds"], list(DEMONSTRATION_SEEDS)):
            parser.error("Le fichier de démonstrations doit utiliser les graines 2000 à 2255.")

        examples = successful_examples(data)
        if not args.imitation_dir:
            normalizer.obs_rms.update(examples[0])
        normalizer.training = False
        config = {
            **vars(args),
            "data": str(args.data),
            "imitation_dir": str(args.imitation_dir) if args.imitation_dir else None,
            "demo_episodes": len(data["seeds"]),
            "demo_transitions": len(data["actions"]),
            "bc_episodes": int((data["outcomes"] == "landed").sum()),
            "bc_transitions": len(examples[1]),
            "demonstration_seeds": list(DEMONSTRATION_SEEDS),
            "validation_seeds": list(IMITATION_VALIDATION_SEEDS),
            "test_seeds": list(test_seeds),
            "bc_batch_size": BC_BATCH_SIZE,
            "bc_learning_rate": BC_LEARNING_RATE,
            "sac_initial_action_std": SAC_INITIAL_ACTION_STD,
            "sac": settings,
        }
        (run_dir / "config.json").write_text(dumps(config, indent=2), encoding="utf-8")
        with wandb.init(
            project="rocket-landing-rl",
            group="quentin",
            tags=["phase0", "sac", "imitation"],
            name=run_dir.name,
            mode="offline",
            dir=str(run_dir),
            config=config,
        ) as run:
            history = []
            imitation_dir = args.imitation_dir or run_dir / "imitation"
            if not args.imitation_dir:
                model = SAC("MlpPolicy", normalizer, seed=args.seed, device="cpu", **settings)
                history = fit_actor(model, normalizer, examples, args.epochs, args.seed, run_dir, run)
            model = SAC.load(
                imitation_dir / "model.zip", env=normalizer, device="cpu", seed=args.seed, ent_coef=entropy
            )
            before = evaluate_policy(sac_policy(model, normalizer), test_seeds)
            report = {"imitation_validation": history, "imitation_test": before}
            report_path = run_dir / "report.json"
            report_path.write_text(dumps(report, indent=2), encoding="utf-8")
            print(f"Imitation seule | poses {before['outcomes'].get('landed', 0)}/40", flush=True)

            if args.sac_steps:
                # Le chargement restaure aussi le coefficient enregistré dans le checkpoint.
                with torch.no_grad():
                    model.log_ent_coef.fill_(log(args.sac_initial_entropy))
                run.log({"sac/initial_entropy_coefficient": model.log_ent_coef.exp().item()})
                anchor = None
                if args.sac_anchor:
                    observations = normalizer.normalize_obs(examples[0])
                    anchor = anchor_actor(model, observations, args.sac_anchor, args.seed)
                try:
                    report["sac_validation"] = finetune(model, normalizer, data, args.sac_steps, run_dir, run)
                finally:
                    if anchor is not None:
                        anchor.remove()
                save_snapshot(model, normalizer, run_dir / "sac")
                after = evaluate_policy(sac_policy(model, normalizer), test_seeds)
                report["sac_test"] = after
                print(f"Après SAC | poses {after['outcomes'].get('landed', 0)}/40", flush=True)
            report_path.write_text(dumps(report, indent=2), encoding="utf-8")
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
