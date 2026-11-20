"""Chargement de la scène MuJoCo et noms utiles pour le reste du code."""

from pathlib import Path

import mujoco

SCENE_XML = Path(__file__).parent / "assets" / "scene.xml"

DECK_TOP_Z = 1.0  # hauteur du pont au-dessus de la mer (m)
FEET = ("pied_1", "pied_2", "pied_3", "pied_4")
LEGS = ("jambe_1", "jambe_2", "jambe_3", "jambe_4")


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))
