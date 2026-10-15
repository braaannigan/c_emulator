import numpy as np
import pytest
import torch

from src.models.stepper import StateStepper, StepperEmulator


class AddOne(torch.nn.Module):
    def forward(self, x):
        return x[:, :6] + 1.0


class Noise(torch.nn.Module):
    def forward(self, x):
        return torch.randn_like(x[:, :6])


def make(target_mode="residual", mass_conserving=False, net=None):
    return StateStepper(net or AddOne(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode=target_mode, mass_conserving=mass_conserving)


def test_stepper_rejects_unknown_target_mode():
    with pytest.raises(ValueError):
        make(target_mode="bogus")


def test_next_state_mode_returns_network_output():
    out = make("next_state")(torch.full((1, 6, 2, 2), 3.0), torch.zeros(1, 2, 2))
    torch.testing.assert_close(out, torch.full((1, 6, 2, 2), 4.0))


def test_residual_mode_adds_increment():
    out = make("residual")(torch.full((1, 6, 2, 2), 3.0), torch.zeros(1, 2, 2))
    torch.testing.assert_close(out, torch.full((1, 6, 2, 2), 7.0))


@pytest.mark.parametrize("mode", ["residual", "next_state"])
def test_mass_conserving_keeps_layer_volume_but_not_velocity(mode):
    torch.manual_seed(0)
    state = torch.randn(2, 6, 4, 4) + 10
    out = make(mode, mass_conserving=True, net=Noise())(state, torch.zeros(2, 4, 4))
    torch.testing.assert_close(out[:, :2].mean(dim=(2, 3)), state[:, :2].mean(dim=(2, 3)))
    assert not torch.allclose(out[:, 2:].mean(dim=(2, 3)), state[:, 2:].mean(dim=(2, 3)))


def test_emulator_rollout_feeds_back_its_own_predictions():
    state = np.ones((5, 6, 2, 2), dtype="float32")
    emulator = StepperEmulator(make("next_state"), torch.device("cpu"))
    pred = emulator.rollout(state[:2], np.zeros((5, 2, 2), dtype="float32"), 3)
    assert pred.shape == (3, 6, 2, 2)
    np.testing.assert_allclose(pred[:, 0, 0, 0], [2.0, 3.0, 4.0])
