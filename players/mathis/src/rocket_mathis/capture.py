"""Captures d'écran pour documenter l'avancement du projet.

Les images sont rangées par étape dans players/mathis/docs/captures/<etape>/.

    uv run python -m rocket_mathis.capture etape-1

Pour une étape future, réutiliser render() et save() depuis son propre code
(par exemple pour filmer un épisode joué par l'agent).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from rocket_mathis.engine import GIMBAL_MAX, Engine
from rocket_mathis.scene import load_model

CAPTURES_DIR = Path(__file__).resolve().parents[2] / "docs" / "captures"


def render(model, data, camera: str | mujoco.MjvCamera, width=960, height=540, trails=()) -> np.ndarray:
    """Image RGB vue depuis une caméra du XML (son nom) ou une caméra libre (MjvCamera).

    trails : liste de (points [N, 3], couleur RGBA) dessinés en petites sphères.
    """
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        for points, rgba in trails:
            for p in points:
                if scene.ngeom >= scene.maxgeom:
                    break
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.6, 0, 0]),
                    np.asarray(p, dtype=np.float64),
                    np.eye(3).ravel(),
                    np.asarray(rgba, dtype=np.float32),
                )
                scene.ngeom += 1
        return renderer.render()


def free_camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def save(image: np.ndarray, step: str, name: str) -> Path:
    """Enregistre une image dans docs/captures/<step>/<name>.png."""
    path = CAPTURES_DIR / step / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, optimize=True)
    return path


def simulate_from_keyframe(model, data, key: str, seconds: float):
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    while data.time < seconds:
        mujoco.mj_step(model, data)


def capture_scene(step: str) -> list[Path]:
    """Étape 1 : la scène seule (fusée posée, vue d'ensemble, descente)."""
    model = load_model()
    data = mujoco.MjData(model)
    shots = [
        ("petite_chute", 2.0, "pont", "fusee-posee"),
        ("petite_chute", 2.0, "large", "vue-ensemble"),
        ("descente", 1.0, "suivi", "descente"),
    ]
    paths = []
    for key, seconds, camera, name in shots:
        simulate_from_keyframe(model, data, key, seconds)
        paths.append(save(render(model, data, camera), step, name))
    return paths


def capture_engine(step: str) -> list[Path]:
    """Étape 2 : le moteur (vol stationnaire, tuyère orientée, fusée qui bascule)."""
    model = load_model()
    data = mujoco.MjData(model)
    engine = Engine(model)
    hover = model.body_subtreemass[model.body("fusee").id] * 9.81
    shots = [
        # (durée, poussée, tuyère, caméra, nom)
        (1.0, hover, [0, 0], "pont", "vol-stationnaire"),
        (0.1, hover, [0, GIMBAL_MAX], "tuyere", "tuyere-orientee"),
        (1.5, hover, [0, GIMBAL_MAX], "suivi", "basculement"),
    ]
    paths = []
    for seconds, thrust, gimbal, camera, name in shots:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 22.0  # 15 m au-dessus de la position posée
        while data.time < seconds:
            engine.apply(data, thrust, gimbal)
            mujoco.mj_step(model, data)
        paths.append(save(render(model, data, camera), step, name))
    return paths


OUTCOME_COLORS = {"landed": (0.2, 0.85, 0.3, 1), "timeout": (1, 0.8, 0.2, 1)}
FAILURE_COLOR = (0.95, 0.25, 0.2, 1)


def capture_env(step: str) -> list[Path]:
    """Étape 3 : l'environnement Gymnasium (départs aléatoires du contrat, fin d'épisode)."""
    from rocket_mathis.env import RocketLandingEnv  # import local : évite un cycle au chargement

    env = RocketLandingEnv()
    hover = np.zeros(3)  # poussée = poids : la fusée garde sa vitesse de départ
    trails = []
    for seed in range(8):
        env.reset(seed=seed)
        points, done = [], False
        while not done:
            points.append(env.data.site_xpos[env._base].copy())
            _, _, terminated, truncated, info = env.step(hover)
            done = terminated or truncated
        color = OUTCOME_COLORS.get(info["outcome"], FAILURE_COLOR)
        trails.append((points[::2], color))
    cam = free_camera(lookat=(0, 0, 50), distance=230, azimuth=135, elevation=-12)
    paths = [save(render(env.model, env.data, cam, trails=trails), step, "departs-aleatoires")]

    env.reset(seed=1)
    done = False
    while not done:  # moteur coupé : chute libre jusqu'à l'échec
        _, _, terminated, truncated, info = env.step(np.array([-1.0, 0, 0]))
        done = terminated or truncated
    paths.append(save(render(env.model, env.data, "large"), step, f"chute-libre-{info['outcome']}"))
    env.close()
    return paths


STEPS = {"etape-1": capture_scene, "etape-2": capture_engine, "etape-3": capture_env}


def main():
    parser = argparse.ArgumentParser(description="Captures d'écran de documentation")
    parser.add_argument("step", choices=STEPS, help="étape à capturer")
    args = parser.parse_args()
    for path in STEPS[args.step](args.step):
        print(path.relative_to(CAPTURES_DIR.parents[3]))


if __name__ == "__main__":
    main()
