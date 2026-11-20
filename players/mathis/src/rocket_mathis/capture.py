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


def render(model: mujoco.MjModel, data: mujoco.MjData, camera: str, width=960, height=540) -> np.ndarray:
    """Image RGB de la scène vue depuis une caméra nommée du XML."""
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data, camera=camera)
        return renderer.render()


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


STEPS = {"etape-1": capture_scene, "etape-2": capture_engine}


def main():
    parser = argparse.ArgumentParser(description="Captures d'écran de documentation")
    parser.add_argument("step", choices=STEPS, help="étape à capturer")
    args = parser.parse_args()
    for path in STEPS[args.step](args.step):
        print(path.relative_to(CAPTURES_DIR.parents[3]))


if __name__ == "__main__":
    main()
