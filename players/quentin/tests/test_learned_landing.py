import numpy as np
import pytest
import torch

from rocket_quentin.env import OBSERVATION_SIZES
from rocket_quentin.learned_landing import MIN_POWERED_COMMAND, LandingPolicy


@pytest.mark.parametrize(("phase", "rcs"), [(3, ()), (4, (0.5, 0.1, -0.5))])
def test_saved_policy_owns_the_motor_decision_even_with_a_contact(tmp_path, phase, rcs):
    model = LandingPolicy(phase)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(torch.tensor([1.0, -1.0, 0.25, -0.25, *rcs]))
    path = tmp_path / "landing.pt"
    model.save(path)

    loaded = LandingPolicy.load(path)
    observation = np.zeros(OBSERVATION_SIZES[phase], dtype=np.float32)
    observation[-1] = 1
    assert loaded.phase == phase
    expected = [MIN_POWERED_COMMAND, *np.tanh([0.25, -0.25, *rcs])]
    assert loaded.action(observation) == pytest.approx(expected)
    with torch.no_grad():
        loaded.network[-1].bias[0] = -1
    assert loaded.action(observation)[0] == -1
