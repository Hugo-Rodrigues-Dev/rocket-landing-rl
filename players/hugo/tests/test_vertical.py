"""Teste la trajectoire analytique et la détection du premier passage au niveau du pont."""

import unittest
from math import sqrt

from rocket_hugo.vertical import CONTROL_DT_S, GRAVITY, VerticalState, advance


# Regroupe les situations verticales de référence sans dépendance à MuJoCo.
class TestVerticalModel(unittest.TestCase):
    # Vérifie qu'une poussée égale au poids conserve la vitesse de descente.
    def test_weight_matching_preserves_nonzero_velocity(self):
        next_state, dt = advance(VerticalState(100.0, -10.0), action=0.0)
        self.assertEqual(dt, CONTROL_DT_S)
        self.assertEqual(next_state, VerticalState(99.5, -10.0))

    # Vérifie le temps et la vitesse d'un impact survenant entre deux décisions.
    def test_free_fall_reports_impact_before_end_of_step(self):
        state, dt = advance(VerticalState(0.01, 0.0), action=-1.0)
        self.assertAlmostEqual(dt, sqrt(2 * 0.01 / GRAVITY))
        self.assertLess(dt, CONTROL_DT_S)
        self.assertEqual(state.height_m, 0.0)
        self.assertAlmostEqual(state.velocity_m_s, -GRAVITY * dt)

    # Vérifie la détection d'un impact même si la trajectoire repasse ensuite au-dessus du pont.
    def test_upward_acceleration_cannot_undo_an_earlier_contact(self):
        state, dt = advance(VerticalState(0.001, -0.2), action=1.0)
        self.assertLess(dt, CONTROL_DT_S)
        self.assertEqual(state.height_m, 0.0)
        self.assertLess(state.velocity_m_s, 0.0)
