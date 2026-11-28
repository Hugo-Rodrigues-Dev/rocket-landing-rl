"""Rejoue les scénarios de pilotage manuel dans l'environnement 3D de Hugo."""

import argparse
from contextlib import nullcontext
from math import acos, atan2, degrees
from queue import Empty, SimpleQueue
from time import perf_counter, sleep

import mujoco
import numpy as np

from rocket_hugo.demo_vertical import braking_action
from rocket_hugo.env import GIMBAL_LIMIT_RAD, RocketLandingEnv
from rocket_hugo.vertical import CONTROL_DT_S, MAX_THRUST_N, VerticalState, thrust_newtons

MODES = ("freefall", "weight", "brake", "tilt-open", "tilt-stabilize", "gimbal", "random")
FIXED_ACTIONS = {"freefall": (-1, 0, 0), "weight": (0, 0, 0), "gimbal": (0.3, 0.65, 0)}
OUTCOMES = {
    "flying": "En vol",
    "landed": "Posee",
    "hard_landing": "Impact dur",
    "crash": "Corps touche",
    "water": "A l'eau",
    "out_of_bounds": "Hors zone",
    "timeout": "Temps ecoule",
}


def _frame_descent(camera, observation):
    """Ajuste le cadrage pour garder la fusée et la barge visibles durant la descente."""
    height_m = max(0.0, float(observation[2]) * 100)
    camera.lookat[:] = (float(observation[0]) * 50, float(observation[1]) * 50, 0.5 * height_m + 6)
    camera.distance = 25 + 1.35 * height_m


def _action(mode, observation, rng):
    """Choisit la commande du scénario, fixe, aléatoire ou calculée par une règle manuelle."""
    if mode in FIXED_ACTIONS:
        return FIXED_ACTIONS[mode]
    if mode == "random":
        return rng.uniform(-1, 1, size=3)

    height, speed = float(observation[2]) * 100, float(observation[5]) * 20
    thrust = braking_action(VerticalState(height, speed))
    gimbal_x = 0.0
    if mode == "tilt-stabilize":
        # Diagnostic dans le plan yz. Une tuyère positive produit un couple autour de -x.
        angle_x = atan2(-observation[10], observation[11])
        rate_x = float(observation[12]) * 2
        gimbal_x = float(np.clip((1.2 * angle_x + 0.8 * rate_x) / GIMBAL_LIMIT_RAD, -1, 1))
    return thrust, gimbal_x, 0.0


def _draw_hud(window, mode, obs, action, info, paused, controls=None):
    """Convertit les observations normalisées en unités lisibles et rafraîchit le visualiseur."""
    status = OUTCOMES[info["outcome"]]
    if paused and info["outcome"] == "flying":
        status = "Pause"
    tilt = degrees(acos(float(np.clip(obs[11], -1, 1))))
    thrust = 0.0 if action is None else thrust_newtons(action[0]) / MAX_THRUST_N * 100
    gimbal = "-" if action is None else f"{action[1] * 10:+.1f} / {action[2] * 10:+.1f} deg"
    telemetry = [
        ("Scenario", mode),
        ("Etat", status),
        ("Temps simule", f"{info['time_s']:.2f} s"),
        ("Hauteur", f"{obs[2] * 100:.1f} m"),
        ("Vitesse verticale", f"{obs[5] * 20:+.2f} m/s"),
        ("Inclinaison", f"{tilt:.1f} deg"),
        ("Poussee", f"{thrust:.0f} %"),
        ("Tuyere X / Y", gimbal),
    ]
    if info["impact_vertical_m_s"] is not None:
        telemetry.append(("Premier contact", f"{info['impact_vertical_m_s']:+.2f} m/s"))
    if controls is None:
        controls = (
            "1 Chute | 2 Poids | 3 Freinage | 4 Angle libre | 5 Redressement",
            "R Rejouer | P Pause | N Un pas",
        )
    window.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "\n".join(label for label, _ in telemetry),
                "\n".join(value for _, value in telemetry),
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                *controls,
            ),
        ]
    )
    window.sync()


def _reset(env, mode, seed):
    """Prépare le type de départ et la source aléatoire associés au scénario."""
    start = "vertical"
    if mode.startswith("tilt-"):
        start = "tilted"
    elif mode == "random":
        start = "contract"
    obs, info = env.reset(seed=seed, options={"start": start})
    return obs, info, np.random.default_rng(seed)


def main():
    """Traite les raccourcis, avance la simulation et rapporte l'issue du scénario.

    En mode sans fenêtre, termine après un vol. Avec la fenêtre, conserve
    l'image finale pour permettre de changer de scénario ou de recommencer.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="brake")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--headless", action="store_true", help="Calcule le vol sans ouvrir de fenêtre.")
    args = parser.parse_args()
    env, mode, keys = RocketLandingEnv(), args.mode, SimpleQueue()
    obs, info, rng = _reset(env, mode, args.seed)
    paused = done = False
    action = None
    total_reward = 0.0

    viewer_context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer

        viewer_context = viewer.launch_passive(
            env.model, env.data, key_callback=keys.put, show_left_ui=False, show_right_ui=False
        )

    with viewer_context as window:
        if window:
            window.cam.azimuth, window.cam.elevation = 125, -22
            _frame_descent(window.cam, obs)
            print("1 chute, 2 poids, 3 freinage, 4 angle libre, 5 redressement. R rejoue, P pause, N avance.")
        while window is None or window.is_running():
            frame_start = perf_counter()
            single_step = False
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                if ord("1") <= key <= ord("5") or key == ord("R"):
                    if key != ord("R"):
                        mode = MODES[key - ord("1")]
                    obs, info, rng = _reset(env, mode, args.seed)
                    done, action, total_reward = False, None, 0.0
                    if window:
                        _frame_descent(window.cam, obs)
                elif key == ord("P"):
                    paused = not paused
                elif key == ord("N"):
                    paused = single_step = True

            if not done and (not paused or single_step):
                action = _action(mode, obs, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if window:
                    _frame_descent(window.cam, obs)
                if done:
                    print(f"{mode}: {info['outcome']}, t={info['time_s']:.2f} s, retour={total_reward:+.0f}")
                    if info["impact_vertical_m_s"] is not None:
                        print(f"Premier contact : {info['impact_vertical_m_s']:+.2f} m/s")
                    if window is None:
                        break
            if window:
                _draw_hud(window, mode, obs, action, info, paused)
                sleep(max(0, CONTROL_DT_S - (perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
