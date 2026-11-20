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


def main():
    parser = argparse.ArgumentParser(description="Captures d'écran de documentation")
    parser.add_argument("step", help="nom de l'étape, par exemple etape-1")
    args = parser.parse_args()
    for path in capture_scene(args.step):
        print(path.relative_to(CAPTURES_DIR.parents[3]))


if __name__ == "__main__":
    main()
