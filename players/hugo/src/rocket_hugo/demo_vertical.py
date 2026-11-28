"""Compare en console trois stratégies sur le même départ vertical."""

from rocket_hugo.vertical import GRAVITY, MAX_DECISIONS, VerticalState, advance


def braking_action(state: VerticalState) -> float:
    # La consigne ralentit la descente près du pont. Le retour de vitesse ajuste la poussée.
    """Calcule une poussée à partir de l'écart à la vitesse de descente visée.

    La consigne décroît près du pont. La commande normalisée est saturée
    entre moteur coupé et pleine poussée.
    """
    target_velocity = -min(12.0, max(0.12, 0.5 * state.height_m))
    acceleration = (target_velocity - state.velocity_m_s) / 0.25
    return max(-1.0, min(1.0, acceleration / GRAVITY))


def main() -> None:
    """Rejoue chaque stratégie jusqu'au contact ou à la limite de temps et affiche le bilan."""
    print(f"{'Scénario':<17} {'Durée (s)':>9} {'Hauteur (m)':>12} {'Vitesse (m/s)':>14}  Événement")
    scenarios = {"moteur coupé": -1.0, "poussée = poids": 0.0, "freinage": None}
    for mode, fixed_action in scenarios.items():
        state = VerticalState(height_m=100.0, velocity_m_s=-10.0)
        elapsed = 0.0
        for _ in range(MAX_DECISIONS):
            action = braking_action(state) if fixed_action is None else fixed_action
            state, dt = advance(state, action)
            elapsed += dt
            if state.height_m == 0:
                break
        event = "contact" if state.height_m == 0 else "temps écoulé"
        print(f"{mode:<17} {elapsed:>9.2f} {state.height_m:>12.2f} {state.velocity_m_s:>14.2f}  {event}")


if __name__ == "__main__":
    main()
