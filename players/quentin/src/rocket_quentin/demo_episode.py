"""Rejoue en 3D un épisode Gymnasium."""

from argparse import ArgumentParser
from threading import Event
from time import perf_counter, sleep

import mujoco
import numpy as np
from mujoco import viewer

from rocket_quentin.demo_thrust import draw_thrust_arrow
from rocket_quentin.env import DECISION_INTERVAL, RocketLandingEnv
from rocket_quentin.pilot_attitude import attitude_action
from rocket_quentin.pilot_lateral import lateral_action
from rocket_quentin.pilot_vertical import vertical_action


def frame_episode(camera: mujoco.MjvCamera, rocket_position: np.ndarray, deck_position: np.ndarray) -> None:
    """Cadre à la fois la fusée et la barge pendant la descente."""
    camera.lookat[:] = (rocket_position + deck_position) / 2
    separation = np.linalg.norm(rocket_position - deck_position)
    camera.distance = max(36.0, 1.7 * separation)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", type=int, choices=(0, 1), default=0, help="0 : barge fixe, 1 : barge mobile"
    )
    parser.add_argument("--seed", type=int, default=12, help="graine de départ (par défaut : 12)")
    parser.add_argument(
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
    env = RocketLandingEnv(phase=args.phase)
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
    observation, _ = env.reset(seed=args.seed, options=options)
    env.model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    deck_position = env.data.geom("deck").xpos
    rocket_position = env.data.body("rocket").xpos
    nozzle_position = env.data.site("nozzle").xpos
    force = env.data.xfrc_applied[env.model.body("rocket").id, :3]
    restart = Event()

    def on_key(key: int) -> None:
        if key == ord("R"):
            restart.set()

    print(f"{args.mode} | altitude initiale {observation[2]:.1f} m | R pour rejouer", flush=True)
    finished = False
    try:
        with viewer.launch_passive(env.model, env.data, key_callback=on_key, show_left_ui=False) as window:
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
                        observation, _ = env.reset(seed=args.seed, options=options)
                        finished = False
                    if not finished:
                        action = policy(observation) if policy is not None else [-1, 0, 0]
                        observation, reward, terminated, truncated, info = env.step(action)
                        frame_episode(window.cam, rocket_position, deck_position)
                        window.user_scn.ngeom = 0
                        draw_thrust_arrow(window.user_scn, nozzle_position, force)
                        finished = terminated or truncated
                        if finished:
                            print(f"Fin à {env.data.time:.2f} s | {info['outcome']} | récompense {reward:+g}")
                            impact_speed = info.get("impact_speed_m_s")
                            if impact_speed is not None:
                                print(f"Vitesse verticale au premier contact : {impact_speed:.2f} m/s")
                            print(f"Distance horizontale au centre : {np.linalg.norm(observation[:2]):.2f} m")
                window.sync()
                sleep(max(0.0, DECISION_INTERVAL - (perf_counter() - frame_start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
