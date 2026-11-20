"""Code de Mathis : environnement, agents, entraînement."""

import gymnasium as gym

gym.register(id="rocket_mathis/RocketLanding-v0", entry_point="rocket_mathis.env:RocketLandingEnv")
