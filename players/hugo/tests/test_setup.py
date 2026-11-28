"""Vérifie l'installation de la pile et le fonctionnement de base de MuJoCo."""

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
import torch
import wandb


def test_stack_imports():
    """Vérifie que les bibliothèques du projet exposent une version après import."""
    for module in (gym, mujoco, stable_baselines3, torch, wandb):
        assert module.__version__


def test_mujoco_simulates_free_fall():
    """Compare une chute libre MuJoCo à la trajectoire analytique après une seconde."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size="0.5"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    duration = 1.0
    while data.time < duration:
        mujoco.mj_step(model, data)
    # z ≈ -g t² / 2 après 1 s de chute libre
    assert np.isclose(data.qpos[2], -0.5 * 9.81 * data.time**2, atol=0.1)
