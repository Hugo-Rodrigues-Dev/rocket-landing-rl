"""Filmer des épisodes en GIF pour le journal.

    uv run python -m rocket_mathis.video runs/<run>/best_model.zip --variant v4 --difficulty 0 \\
        --seeds 0 1 2 --out docs/captures/etape-5/ppo-v4.gif --title "v4 : PPO"

La caméra suit le milieu entre la fusée et la barge et recule quand elles s'éloignent,
pour que l'on voie toujours les deux. Un bandeau indique la version, le temps et l'altitude.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO, SAC

from rocket_mathis.config import VARIANTS, variant_config
from rocket_mathis.env import RocketLandingEnv

OUTCOME_LABELS = {
    "landed": "POSÉE",
    "hard_landing": "ATTERRISSAGE DUR",
    "crash": "CRASH",
    "water": "À L'EAU",
    "out_of_bounds": "HORS ZONE",
    "timeout": "TEMPS ÉCOULÉ",
}


def _font(size: int):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _camera(env: RocketLandingEnv, cam: mujoco.MjvCamera) -> None:
    rocket = env.data.site_xpos[env._base]
    barge = env._target()
    gap = np.linalg.norm(rocket - barge)
    target_lookat = (rocket + barge) / 2 + np.array([0, 0, 4.0])
    target_distance = max(35.0, 1.4 * gap + 15)
    # lissage pour éviter que la caméra tremble
    cam.lookat[:] = 0.8 * cam.lookat + 0.2 * target_lookat
    cam.distance = 0.85 * cam.distance + 0.15 * target_distance


def record(env, policy, seeds, title: str, fps=10, speed=2.0, size=(400, 225)) -> tuple[list, list, list]:
    """Joue un épisode par graine et renvoie (images, durées en ms, issues). Accéléré ×speed."""
    frames, durations, outcomes = [], [], []
    every = max(1, round(speed / (fps * env.config.control_dt)))
    font, small = _font(14), _font(12)
    with mujoco.Renderer(env.model, size[1], size[0]) as renderer:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation = 135, -12
            cam.lookat[:] = (env.data.site_xpos[env._base] + env._target()) / 2
            cam.distance = 60
            done, step, info = False, 0, {"outcome": ""}
            while True:
                if step % every == 0 or done:
                    _camera(env, cam)
                    renderer.update_scene(env.data, camera=cam)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, size[0], 40), fill=(15, 20, 28))
                    draw.text((8, 3), title, fill=(255, 255, 255), font=font)
                    altitude = obs[2] * 100
                    status = f"t = {env.data.time:4.1f} s  h = {altitude:5.1f} m  ×{speed:g}"
                    draw.text((8, 22), status, fill=(200, 210, 220), font=small)
                    if done:
                        label = OUTCOME_LABELS.get(info["outcome"], info["outcome"])
                        color = (60, 200, 90) if info["outcome"] == "landed" else (230, 70, 60)
                        draw.rectangle((0, size[1] - 34, size[0], size[1]), fill=(15, 20, 28))
                        draw.text((8, size[1] - 28), label, fill=color, font=font)
                        frames.append(image)
                        durations.append(1500)  # arrêt sur l'issue
                        break
                    frames.append(image)
                    durations.append(int(1000 / fps))
                obs, _, terminated, truncated, info = env.step(policy(obs))
                done = terminated or truncated
                step += 1
            outcomes.append(info["outcome"])
    return frames, durations, outcomes


def save_gif(frames: list, durations: list, path: Path) -> Path:
    """Une seule palette de 64 couleurs, sans tramage : fichier bien plus léger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = frames[len(frames) // 2].quantize(colors=64, dither=Image.Dither.NONE)
    images = [f.quantize(palette=reference, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return path


def load_policy(path: str):
    algo = SAC if "sac" in Path(path).as_posix().lower() else PPO
    model = algo.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Filmer un agent en GIF")
    parser.add_argument("model", help="chemin du .zip SB3, ou « pd »")
    parser.add_argument("--variant", choices=list(VARIANTS), default="v4")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--title", default="")
    parser.add_argument("--speed", type=float, default=2.0, help="accélération de la vidéo")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = RocketLandingEnv(variant_config(args.variant, phase=args.phase))
    env.set_difficulty(args.difficulty)
    if args.model == "pd":
        from rocket_mathis.baseline_pd import PDController

        policy = PDController()
    else:
        policy = load_policy(args.model)
    frames, durations, outcomes = record(env, policy, args.seeds, args.title, speed=args.speed)
    path = save_gif(frames, durations, Path(args.out))
    print(f"{path} : {len(frames)} images, {path.stat().st_size / 1e6:.1f} Mo, issues {outcomes}")


if __name__ == "__main__":
    main()
