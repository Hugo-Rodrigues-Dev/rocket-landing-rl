"""Démo visuelle du moteur, à lancer avec python -m rocket_quentin.demo_thrust."""

from pathlib import Path
from queue import Empty, SimpleQueue
from time import perf_counter, sleep

import mujoco
import numpy as np
from mujoco import viewer

from rocket_quentin.thrust import MAX_GIMBAL_DEG, MAX_THRUST_N, apply_thrust

SCENE_PATH = Path(__file__).parent / "assets" / "scene.xml"
COMMANDS = {
    "1": (0.0, 0.0, "moteur coupé"),
    "2": (MAX_THRUST_N / 2, 0.0, "vol stationnaire"),
    "3": (MAX_THRUST_N, 0.0, "pleine poussée"),
    "4": (MAX_THRUST_N / 2, MAX_GIMBAL_DEG, "tuyère inclinée autour de y"),
}


def draw_thrust_arrow(scene: mujoco.MjvScene, nozzle: np.ndarray, force: np.ndarray) -> None:
    """Trace sous la tuyère une flèche orientée comme la force."""
    if not force.any():
        return

    arrow_type = mujoco.mjtGeom.mjGEOM_ARROW
    arrow = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(arrow, arrow_type, [0, 0, 0], nozzle, np.eye(3).ravel(), [1, 0.8, 0.1, 1])
    arrow.emission = 0.35
    tail = nozzle - 4.0 * force / MAX_THRUST_N
    mujoco.mjv_connector(arrow, arrow_type, 0.18, tail, nozzle)
    scene.ngeom += 1


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    rocket_id = model.body("rocket").id
    # L'essai démarre assez haut pour que la flèche soit visible sous la fusée.
    model.qpos0[2] += 5.0
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    frames_per_trial = 70
    steps_per_frame = 4
    frame_duration = steps_per_frame * model.opt.timestep
    keys = SimpleQueue()
    keys.put(ord("2"))
    remaining_frames = 0
    thrust_n = tilt_y_deg = 0.0

    print("1 : moteur coupé | 2 : stationnaire | 3 : pleine poussée | 4 : tuyère inclinée", flush=True)
    print("Chaque touche recommence l'essai. La flèche jaune représente la poussée.", flush=True)
    with viewer.launch_passive(model, data, key_callback=keys.put, show_left_ui=False) as window:
        with window.lock():
            window.cam.lookat[:] = [0, 0, 11]
            window.cam.distance = 36
            window.cam.azimuth = 115
            window.cam.elevation = -15

        while window.is_running():
            frame_start = perf_counter()
            try:
                command = COMMANDS.get(chr(keys.get_nowait()))
            except Empty:
                command = None

            with window.lock():
                if command is not None:
                    thrust_n, tilt_y_deg, description = command
                    mujoco.mj_resetData(model, data)
                    remaining_frames = frames_per_trial
                    print(f"{description} | {thrust_n:,.0f} N | angle y = {tilt_y_deg:g}°", flush=True)

                if remaining_frames:
                    for _ in range(steps_per_frame):
                        apply_thrust(model, data, thrust_n, tilt_y_deg=tilt_y_deg)
                        mujoco.mj_step(model, data)
                    remaining_frames -= 1

                # Recalcule le repère de la tuyère dans la dernière position affichée.
                apply_thrust(model, data, thrust_n, tilt_y_deg=tilt_y_deg)
                mujoco.mj_kinematics(model, data)
                window.user_scn.ngeom = 0
                draw_thrust_arrow(window.user_scn, data.site("nozzle").xpos, data.xfrc_applied[rocket_id, :3])

            window.sync()
            sleep(max(0, frame_duration - (perf_counter() - frame_start)))


if __name__ == "__main__":
    main()
