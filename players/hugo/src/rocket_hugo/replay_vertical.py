"""Compare des versions PPO sauvegardées sur une trajectoire verticale reproductible."""

import argparse
import json
from pathlib import Path
from queue import Empty, SimpleQueue
from time import perf_counter, sleep

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from rocket_hugo.demo_3d import _draw_hud, _frame_descent
from rocket_hugo.env import SCENE
from rocket_hugo.vertical import CONTROL_DT_S
from rocket_hugo.vertical_env import VerticalLandingEnv


def main():
    """Recharge les réseaux d'un essai et rejoue chacun depuis la graine choisie.

    Le mode console affiche les issues. La fenêtre permet de changer de réseau
    et d'avancer pas à pas. MuJoCo dessine la trajectoire calculée par l'exercice.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=5100, help="Graine identique pour chaque version rejouée."
    )
    parser.add_argument("--headless", action="store_true", help="Rejoue chaque version et affiche son issue.")
    args = parser.parse_args()
    checkpoints = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))["checkpoints"]
    torch.set_num_threads(1)
    policies = [PPO.load(args.run / item["file"], device="cpu") for item in checkpoints]
    exercise = VerticalLandingEnv()

    if args.headless:
        for checkpoint, policy in zip(checkpoints, policies, strict=True):
            obs, _ = exercise.reset(seed=args.seed)
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = exercise.step(action)
                if terminated or truncated:
                    impact = info["impact_vertical_m_s"]
                    contact = f", contact {impact:+.2f} m/s" if impact is not None else ""
                    print(f"{checkpoint['steps']:>7} décisions | {info['outcome']}{contact}")
                    break
        return

    from mujoco import viewer

    # MuJoCo sert ici d'écran. Le mouvement affiché vient du même modèle analytique que l'entraînement.
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    deck_z = model.site_pos[model.site("deck_center").id, 2]
    bottom_z = model.site_pos[model.site("base").id, 2]
    # Le visualiseur transmet les événements clavier depuis son propre thread.
    keys = SimpleQueue()
    index = 0
    obs, info = exercise.reset(seed=args.seed)
    action = None
    paused = True
    done = False
    controls = ("1-9 Version | [ ] Version prec. / suiv.", "R Rejouer | P Pause | N Un pas")
    for number, item in enumerate(checkpoints, 1):
        print(f"Version {number}: {item['steps']} décisions")

    with viewer.launch_passive(
        model, data, key_callback=keys.put, show_left_ui=False, show_right_ui=False
    ) as window:
        window.cam.azimuth, window.cam.elevation = 125, -22
        while window.is_running():
            frame_start = perf_counter()
            single_step = False
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                if key in (ord("["), ord("]"), ord("R")) or 0 <= key - ord("1") < min(9, len(policies)):
                    if key == ord("["):
                        index = max(0, index - 1)
                    elif key == ord("]"):
                        index = min(len(policies) - 1, index + 1)
                    elif key != ord("R"):
                        index = key - ord("1")
                    # Chaque version repart du même état, même après un vol terminé.
                    obs, info = exercise.reset(seed=args.seed)
                    done, action = False, None
                elif key == ord("P"):
                    paused = not paused
                elif key == ord("N"):
                    paused = single_step = True

            if not done and (not paused or single_step):
                action, _ = policies[index].predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = exercise.step(action)
                done = terminated or truncated
            state = exercise.state
            # Convertit la hauteur relative au pont en position du corps dans la scène.
            data.qpos[2] = deck_z - bottom_z + state.height_m
            data.time = info["time_s"]
            mujoco.mj_forward(model, data)
            display_obs = np.zeros(19)
            display_obs[2] = state.height_m / 100
            display_obs[5] = state.velocity_m_s / 20
            display_obs[11] = 1.0
            display_action = None if action is None else (float(action[0]), 0, 0)
            _frame_descent(window.cam, display_obs)
            label = f"PPO {checkpoints[index]['steps']} decisions"
            _draw_hud(window, label, display_obs, display_action, info, paused, controls=controls)
            sleep(max(0, CONTROL_DT_S - (perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
