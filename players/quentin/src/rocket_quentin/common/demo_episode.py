"""Rejoue en 3D un épisode Gymnasium."""

from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from threading import Event
from time import perf_counter, sleep

import mujoco
import numpy as np
from mujoco import viewer

from rocket_quentin.common.env import DECISION_INTERVAL, OBSERVATION_SIZES, RocketLandingEnv
from rocket_quentin.common.flight_view import draw_deck_markings, draw_flame, draw_ripples, telemetry
from rocket_quentin.dagger.pilot_attitude import attitude_action
from rocket_quentin.dagger.pilot_lateral import lateral_action
from rocket_quentin.dagger.pilot_vertical import vertical_action
from rocket_quentin.dagger.pulse_throttle import PulseThrottle
from rocket_quentin.dagger.reference import reference_action
from rocket_quentin.dagger.wave_approach import wave_reference


def frame_episode(camera: mujoco.MjvCamera, rocket_position: np.ndarray, deck_position: np.ndarray) -> None:
    """Cadre à la fois la fusée et la barge pendant la descente."""
    camera.lookat[:] = (rocket_position + deck_position) / 2
    separation = np.linalg.norm(rocket_position - deck_position)
    camera.distance = max(36.0, 1.7 * separation)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=int,
        choices=range(len(OBSERVATION_SIZES)),
        help="0 : fixe, 1 : mobile, 2 : carburant, 3 : vagues, 4 : vent et RCS (déduite du modèle)",
    )
    parser.add_argument(
        "--pulse-throttle", action="store_true", help="adapte la poussée au moteur de phase 2"
    )
    parser.add_argument("--wave-approach", action="store_true", help="aligne la fusée sur le pont au toucher")
    parser.add_argument("--seed", type=int, default=12, help="graine de départ (par défaut : 12)")
    pilot = parser.add_mutually_exclusive_group()
    pilot.add_argument("--model", type=Path, help="checkpoint autonome .pt")
    pilot.add_argument(
        "--mode",
        choices=(
            "freefall",
            "vertical",
            "vertical-freefall",
            "attitude",
            "attitude-uncorrected",
            "lateral",
            "lateral-uncorrected",
            "pilot",
        ),
        default="freefall",
        help="départ du contrat, essai vertical, départ incliné ou départ décalé",
    )
    args = parser.parse_args()
    model_policy = None
    if args.model is not None:
        if args.pulse_throttle or args.wave_approach:
            parser.error("Les options du pilote de référence ne s'appliquent pas au modèle.")
        if args.model.suffix != ".pt" or not args.model.is_file():
            parser.error("--model doit désigner un checkpoint autonome .pt existant.")
        import torch

        from rocket_quentin.dagger.learned_landing import LandingPolicy

        torch.set_num_threads(1)
        model_policy = LandingPolicy.load(args.model)
        if args.phase is not None and args.phase != model_policy.phase:
            parser.error("Le checkpoint ne correspond pas à la phase demandée.")
        args.phase = model_policy.phase
    elif args.phase is None:
        args.phase = 0
    if args.pulse_throttle and (args.phase not in (2, 3) or args.mode == "freefall"):
        parser.error("--pulse-throttle nécessite la phase 2 ou 3 et un mode de pilotage.")
    if args.phase == 4 and args.mode not in ("pilot", "freefall"):
        parser.error("La phase 4 est disponible avec --mode pilot ou --mode freefall.")
    if args.wave_approach and (args.phase < 3 or not args.pulse_throttle or args.mode != "pilot"):
        parser.error("--wave-approach nécessite la phase 3, --mode pilot et --pulse-throttle.")
    env = RocketLandingEnv(phase=args.phase)
    freefall_action = [-1.0] + [0.0] * (env.action_space.shape[0] - 1)
    options = {"vertical_demo": True} if args.mode not in ("freefall", "pilot") else None
    if args.mode in ("attitude", "attitude-uncorrected"):
        options = {"vertical_demo": True, "initial_pitch_deg": 5.0}
    if args.mode in ("lateral", "lateral-uncorrected"):
        options = {
            "vertical_demo": True,
            "initial_position_xy": (12.0, -8.0),
            "initial_velocity_xy": (1.0, 0.0),
        }
    policy = {
        "vertical": vertical_action,
        "attitude": attitude_action,
        "attitude-uncorrected": vertical_action,
        "lateral": lateral_action,
        "lateral-uncorrected": attitude_action,
        "pilot": lateral_action,
    }.get(args.mode)
    if args.phase >= 2 and args.mode == "pilot" and not args.pulse_throttle:
        policy = partial(reference_action, phase=args.phase)
    if args.wave_approach:
        policy = wave_reference
    if args.pulse_throttle:
        policy = PulseThrottle(policy)
    if model_policy is not None:
        policy = model_policy.action
    observation, info = env.reset(seed=args.seed, options=options)
    action = freefall_action
    env.model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    deck = env.data.geom("deck")
    deck_half_height = env.model.geom("deck").size[2]
    deck_position = deck.xpos
    rocket_position = env.data.body("rocket").xpos
    nozzle_position = env.data.site("nozzle").xpos
    force = env.thrust_force
    restart = Event()

    def on_key(key: int) -> None:
        if key == ord("R"):
            restart.set()

    label = args.model.stem if args.model else args.mode
    print(
        f"{label} | phase {args.phase} | altitude initiale {observation[2]:.1f} m | R pour rejouer",
        flush=True,
    )
    finished = False
    try:
        with viewer.launch_passive(
            env.model, env.data, key_callback=on_key, show_left_ui=False, show_right_ui=False
        ) as window:
            with window.lock():
                window.cam.azimuth = 115
                window.cam.elevation = -12
                frame_episode(window.cam, rocket_position, deck_position)
            window.sync()

            while window.is_running():
                frame_start = perf_counter()
                with window.lock():
                    if restart.is_set():
                        restart.clear()
                        observation, info = env.reset(seed=args.seed, options=options)
                        action = freefall_action
                        if isinstance(policy, PulseThrottle):
                            policy.reset()
                        finished = False
                    if not finished:
                        action = policy(observation) if policy is not None else freefall_action
                        observation, reward, terminated, truncated, info = env.step(action)
                        frame_episode(window.cam, rocket_position, deck_position)
                        finished = terminated or truncated
                        if finished:
                            print(f"Fin à {env.data.time:.2f} s | {info['outcome']} | récompense {reward:+g}")
                            impact_speed = info.get("impact_speed_m_s")
                            if impact_speed is not None:
                                print(f"Vitesse verticale au premier contact : {impact_speed:.2f} m/s")
                            if "fuel_remaining_kg" in info:
                                print(f"Carburant restant : {info['fuel_remaining_kg']:.2f} kg")
                            print(f"Distance horizontale au centre : {np.linalg.norm(observation[:2]):.2f} m")
                    window.user_scn.ngeom = 0
                    draw_ripples(window.user_scn, env.data.time)
                    deck_rotation = deck.xmat.reshape(3, 3)
                    center = deck_position + deck_half_height * deck_rotation[:, 2]
                    draw_deck_markings(window.user_scn, center, deck_rotation)
                    draw_flame(window.user_scn, nozzle_position, force, env.data.time)
                    overlay = telemetry(env, observation, action, info)
                window.set_texts(overlay)
                window.sync()
                sleep(max(0.0, DECISION_INTERVAL - (perf_counter() - frame_start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
