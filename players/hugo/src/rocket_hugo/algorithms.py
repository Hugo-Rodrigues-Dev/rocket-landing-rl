"""Réglages propres à PPO et SAC ; l'environnement et le protocole sont partagés."""

from stable_baselines3 import PPO, SAC

ALGORITHMS = {"ppo": PPO, "sac": SAC}
PPO_SETTINGS = {"n_steps": 512, "batch_size": 128, "n_epochs": 5}
SAC_SETTINGS = {"batch_size": 128, "learning_starts": 1024, "train_freq": 4, "gradient_steps": 1}


# Construit le modèle avec la même taille de couches, puis applique les réglages de son algorithme.
def make_model(name, env, seed):
    settings = PPO_SETTINGS if name == "ppo" else SAC_SETTINGS
    return ALGORITHMS[name](
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        gamma=0.995,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": [32, 32]},
        **settings,
    )
