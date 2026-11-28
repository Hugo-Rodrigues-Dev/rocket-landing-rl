"""Rejoue et compare les pilotes de Hugo dans le visualiseur 3D."""

import argparse
from contextlib import nullcontext
from math import acos, atan2, degrees
from pathlib import Path
from queue import Empty, SimpleQueue
from time import perf_counter, sleep

import mujoco
import numpy as np

from rocket_hugo.demo_vertical import braking_action
from rocket_hugo.env import GIMBAL_LIMIT_RAD, RocketLandingEnv
from rocket_hugo.phase0 import CONTROL_MODES, apply_residual
from rocket_hugo.phase0_pilot import pilot_action
from rocket_hugo.vertical import CONTROL_DT_S, MAX_THRUST_N, VerticalState, thrust_newtons

MODES = ("freefall", "weight", "brake", "tilt-open", "tilt-stabilize", "gimbal", "random", "pilot", "learned")
COMPARISON_MODES = ("pilot", "ppo", "sac")
COMPARISON_LABELS = {"pilot": "Pilote seul", "ppo": "Pilote + PPO", "sac": "Pilote + SAC"}
COMPARISON_CONTROLS = (
    "1 = Pilote seul | 2 = Pilote + PPO | 3 = Pilote + SAC\n"
    "D Changer le depart | V Version precedente | B Version suivante\n"
    "P Lecture / pause | N Un pas | R Rejouer",
    "",
)
COMPARISON_SEEDS = (10046, 7200)
FINAL_TRAINING_STEPS = 8192
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
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


# Ajuste le cadrage pour garder la fusée et la barge visibles durant la descente.
def _frame_descent(camera, observation):
    # Les trois premières observations sont des distances divisées par 100 m.
    # Viser à mi-chemin du pont et de la fusée laisse les deux dans le champ.
    height_m = max(0.0, float(observation[2]) * 100)
    camera.lookat[:] = (float(observation[0]) * 50, float(observation[1]) * 50, 0.5 * height_m + 6)
    camera.distance = 25 + 1.35 * height_m


# Choisit la commande du scénario, fixe, aléatoire ou calculée par une règle manuelle.
def _action(mode, observation, rng, policy=None, control="residual"):
    if mode in FIXED_ACTIONS:
        return FIXED_ACTIONS[mode]
    if mode == "random":
        return rng.uniform(-1, 1, size=3)
    if mode == "pilot":
        return pilot_action(observation)
    if mode in ("learned", "ppo", "sac"):
        command = policy(observation)
        return apply_residual(observation, command)[0] if control == "residual" else command

    # La règle de freinage attend des mètres et des m/s, comme dans la démo verticale.
    height, speed = float(observation[2]) * 100, float(observation[5]) * 20
    thrust = braking_action(VerticalState(height, speed))
    gimbal_x = 0.0
    if mode == "tilt-stabilize":
        # Les composantes y et z de l'axe de la fusée donnent son angle dans le plan yz.
        angle_x = atan2(-observation[10], observation[11])
        rate_x = float(observation[12]) * 2
        # Une tuyère positive produit un couple autour de -x. Le terme de vitesse
        # amortit le redressement, et la division convertit l'angle en commande normalisée.
        gimbal_x = float(np.clip((1.2 * angle_x + 0.8 * rate_x) / GIMBAL_LIMIT_RAD, -1, 1))
    return thrust, gimbal_x, 0.0


# Convertit les observations normalisées en unités lisibles et rafraîchit le visualiseur.
def _draw_hud(window, mode, obs, action, info, paused, controls=None, comparison=None):
    status = OUTCOMES[info["outcome"]]
    if paused and info["outcome"] == "flying":
        status = "Pause"
    # obs[11] est le cosinus de l'inclinaison. Le bornage protège arccos des arrondis.
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
    if comparison is not None:
        telemetry.extend(
            (
                ("Distance au centre", f"{info['distance_to_center_m']:.2f} m"),
                ("Vitesse horizontale", f"{np.linalg.norm(obs[3:5]) * 20:.2f} m/s"),
            )
        )
    if controls is None:
        controls = (
            "1 Chute | 2 Poids | 3 Freinage | 4 Angle libre | 5 Redressement",
            "R Rejouer | P Pause | N Un pas",
        )
    overlays = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "\n".join(label for label, _ in telemetry),
            "\n".join(value for _, value in telemetry),
        ),
        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, *controls),
    ]
    if comparison is not None:
        seed, results, stage, stages = comparison
        # Le repère se déplace entre les versions effectivement disponibles sur disque.
        indicator = "-".join("[o]" if step == stage else " o " for step in stages)
        # Conserve les résultats terminés lorsque l'utilisateur change de pilote ou rejoue un vol.
        overlays.append(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "\n".join(("Phase 0", "Depart", "Apprentissage PPO/SAC", *COMPARISON_LABELS.values())),
                "\n".join(
                    (
                        "Barge fixe",
                        f"{seed}",
                        f"{stage} decisions {indicator}",
                        *(_result_line(results.get(m)) for m in COMPARISON_MODES),
                    )
                ),
            )
        )
    window.set_texts(overlays)
    window.sync()


# Résume un vol avec des unités communes, y compris lorsque la fusée ne touche jamais le pont.
def _result_line(info):
    if info is None:
        return "A comparer"
    impact = info["impact_vertical_m_s"]
    contact = "--" if impact is None else f"{impact:+.2f} m/s"
    return f"{OUTCOMES[info['outcome']]} | {info['distance_to_center_m']:.2f} m | {contact}"


# Charge un modèle sauvegardé et expose seulement sa commande déterministe.
def _load_policy(algo, path):
    import torch

    from rocket_hugo.algorithms import ALGORITHMS

    torch.set_num_threads(1)
    model = ALGORITHMS[algo].load(path, device="cpu")

    # Utilise la même prédiction que l'évaluation officielle.
    def predict(observation):
        return model.predict(observation, deterministic=True)[0]

    return predict


# Recense les versions locales communes aux deux algorithmes et ajoute les modèles évalués.
def _comparison_models():
    learned_modes = COMPARISON_MODES[1:]
    runs_dir = MODELS_DIR.parent / "runs"
    paths = {
        algo: {
            int(path.stem.removeprefix("step-")): path
            for path in (runs_dir / f"phase0-history-{algo}").glob("step-*.zip")
            if path.stem.removeprefix("step-").isdigit()
        }
        for algo in learned_modes
    }
    for algo in learned_modes:
        paths[algo][FINAL_TRAINING_STEPS] = MODELS_DIR / f"phase0-{algo}-residual.zip"
    return paths, sorted(set.intersection(*(set(paths[algo]) for algo in learned_modes)))


# Prépare le type de départ et la source aléatoire associés au scénario.
def _reset(env, mode, seed):
    start = "vertical"
    if mode.startswith("tilt-"):
        start = "tilted"
    elif mode in ("random", "pilot", "learned", "ppo", "sac"):
        start = "contract"
    obs, info = env.reset(seed=seed, options={"start": start})
    return obs, info, np.random.default_rng(seed)


# Traite les raccourcis, avance la simulation et rapporte l'issue du scénario.
#
# En mode sans fenêtre, termine après un vol. Avec la fenêtre, conserve
# l'image finale pour permettre de changer de scénario ou de recommencer.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "compare"), default="brake")
    parser.add_argument("--seed", type=int, help="Par défaut : 10046 en comparaison, 12 sinon.")
    parser.add_argument(
        "--checkpoint", type=int, help="Nombre de décisions apprises à rejouer en comparaison."
    )
    parser.add_argument("--headless", action="store_true", help="Calcule le vol sans ouvrir de fenêtre.")
    parser.add_argument("--algo", choices=("ppo", "sac"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--control", choices=CONTROL_MODES, default="residual")
    args = parser.parse_args()
    compare = args.mode == "compare"
    seed = args.seed if args.seed is not None else (COMPARISON_SEEDS[0] if compare else 12)
    policies = {}
    results = {}
    if not compare and args.checkpoint is not None:
        parser.error("--checkpoint s'utilise avec --mode compare.")
    if compare:
        if args.algo or args.model or args.control != "residual":
            parser.error(
                "La comparaison utilise les deux modèles résiduels de phase 0 conservés dans models/."
            )
        paths, stages = _comparison_models()
        stage = args.checkpoint if args.checkpoint is not None else FINAL_TRAINING_STEPS
        if stage not in stages:
            parser.error(f"Version introuvable. Décisions disponibles : {', '.join(map(str, stages))}.")
        for versions in paths.values():
            if not versions[stage].is_file():
                parser.error(f"Modèle de comparaison introuvable : {versions[stage]}")
    if args.mode == "learned":
        if args.model is None or args.algo is None:
            parser.error("--mode learned nécessite --algo et --model.")
        policies["learned"] = _load_policy(args.algo, args.model)

    env, mode, keys = RocketLandingEnv(), ("pilot" if compare else args.mode), SimpleQueue()
    obs, info, rng = _reset(env, mode, seed)
    paused, done = compare and not args.headless, False
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
            if compare:
                print(f"Départ {seed}\n{COMPARISON_CONTROLS[0]}")
                if len(stages) == 1:
                    print(
                        "Seule la version finale est disponible ici. Voir le README pour recréer les autres."
                    )
            else:
                print("1 chute, 2 poids, 3 freinage, 4 angle libre, 5 redressement.")
                print("R rejoue, P pause, N avance.")
        shortcuts = COMPARISON_MODES if compare else MODES[:5]
        while window is None or window.is_running():
            frame_start = perf_counter()
            single_step = False
            # Le visualiseur poste les touches dans une file. Seule cette boucle modifie le vol.
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                if key == ord("P"):
                    paused = not paused
                elif key == ord("N"):
                    paused = single_step = True
                elif (
                    key == ord("R")
                    or 0 <= key - ord("1") < len(shortcuts)
                    or (compare and key in (ord("D"), ord("V"), ord("B")))
                ):
                    if compare and key == ord("D"):
                        seed = COMPARISON_SEEDS[1] if seed == COMPARISON_SEEDS[0] else COMPARISON_SEEDS[0]
                        results.clear()
                        paused, single_step = True, False
                    elif compare and key in (ord("V"), ord("B")):
                        offset = -1 if key == ord("V") else 1
                        index = min(len(stages) - 1, max(0, stages.index(stage) + offset))
                        if stages[index] == stage:
                            continue
                        stage = stages[index]
                        # Le pilote seul ne dépend pas du nombre de décisions apprises.
                        results.pop("ppo", None)
                        results.pop("sac", None)
                        paused, single_step = True, False
                    elif key != ord("R"):
                        mode = shortcuts[key - ord("1")]
                    obs, info, rng = _reset(env, mode, seed)
                    done, action, total_reward = False, None, 0.0
                    if window:
                        _frame_descent(window.cam, obs)

            if not done and (not paused or single_step):
                policy = policies.get(mode)
                if compare and mode in paths:
                    key = (mode, stage)
                    if key not in policies:
                        policies[key] = _load_policy(mode, paths[mode][stage])
                    policy = policies[key]
                action = _action(mode, obs, rng, policy, args.control)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if window:
                    _frame_descent(window.cam, obs)
                if done:
                    if compare:
                        results[mode] = info
                        label = COMPARISON_LABELS[mode]
                        if mode != "pilot":
                            label += f" ({stage} décisions)"
                        print(f"{label} : {_result_line(info)}")
                    else:
                        print(
                            f"{mode}: {info['outcome']}, t={info['time_s']:.2f} s, retour={total_reward:+.0f}"
                        )
                        if info["impact_vertical_m_s"] is not None:
                            print(f"Premier contact : {info['impact_vertical_m_s']:+.2f} m/s")
                    if window is None:
                        if not compare or len(results) == len(COMPARISON_MODES):
                            break
                        mode = COMPARISON_MODES[len(results)]
                        obs, info, rng = _reset(env, mode, seed)
                        done = False
            if window:
                _draw_hud(
                    window,
                    COMPARISON_LABELS[mode] if compare else mode,
                    obs,
                    action,
                    info,
                    paused,
                    controls=COMPARISON_CONTROLS if compare else None,
                    comparison=(seed, results, stage, stages) if compare else None,
                )
                # Cadence le rendu en temps réel et retire le temps passé à calculer l'image.
                sleep(max(0, CONTROL_DT_S - (perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
