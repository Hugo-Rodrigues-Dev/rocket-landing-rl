"""Réservoir et masse de la fusée en phase 2."""

import mujoco

from rocket_quentin.thrust import MAX_THRUST_N

INITIAL_FUEL_KG = 80.0
FUEL_OBSERVATION_INDEX = 17
SPECIFIC_IMPULSE_S = 300.0
MIN_THRUST_FRACTION = 0.6
MIN_THRUST_N = MIN_THRUST_FRACTION * MAX_THRUST_N


class FuelSystem:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.rocket_id = model.body("rocket").id
        self.dry_mass_kg = float(model.body_mass[self.rocket_id])
        self.dry_inertia = model.body_inertia[self.rocket_id].copy()
        self.exhaust_speed = SPECIFIC_IMPULSE_S * float(-model.opt.gravity[2])
        self.workspace = mujoco.MjData(model)
        self.reset()

    def _set_remaining(self, fuel_kg: float) -> None:
        self.remaining_kg = fuel_kg
        mass = self.dry_mass_kg + fuel_kg
        self.model.body_mass[self.rocket_id] = mass
        # Le carburant conserve le centre de masse et la répartition de masse à vide.
        self.model.body_inertia[self.rocket_id] = self.dry_inertia * (mass / self.dry_mass_kg)
        mujoco.mj_setConst(self.model, self.workspace)

    def reset(self) -> None:
        self._set_remaining(INITIAL_FUEL_KG)

    def burn(self, requested_thrust: float, interval: float) -> float:
        """Consomme le carburant et renvoie la durée réellement propulsée."""
        if requested_thrust < MIN_THRUST_N or self.remaining_kg == 0:
            return 0.0
        flow = requested_thrust / self.exhaust_speed
        available_duration = self.remaining_kg / flow
        if available_duration <= interval:
            self._set_remaining(0.0)
            return available_duration
        self._set_remaining(self.remaining_kg - flow * interval)
        return interval
