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


def test_fixed_cells_are_pinned_and_mass_is_conserved_over_wet_cells():
    torch.manual_seed(0)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0] = True
    values = np.zeros((6, 4, 4), dtype="float32")
    values[:2] = 200.0
    stepper = StateStepper(Noise(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", mass_conserving=True, fixed_mask=mask, fixed_values=values)
    state = torch.from_numpy(np.broadcast_to(values, (2, 6, 4, 4)).copy())
    state[:, :2, 1:] += torch.randn(2, 2, 3, 4)
    out = stepper(state, torch.zeros(2, 4, 4))
    torch.testing.assert_close(out[:, :, 0], torch.from_numpy(values[:, 0]).expand(2, -1, -1))
    torch.testing.assert_close(out[:, :2].sum(dim=(2, 3)), state[:, :2].sum(dim=(2, 3)))
    assert not torch.allclose(out[:, :2, 1:], state[:, :2, 1:])


def test_fixed_cells_without_mass_projection_and_argument_pairing():
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    stepper = StateStepper(AddOne(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", fixed_mask=mask, fixed_values=np.full((6, 2, 2), 5.0))
    out = stepper(torch.zeros(1, 6, 2, 2), torch.zeros(1, 2, 2))
    assert float(out[0, 0, 0, 0]) == 5.0 and float(out[0, 0, 1, 1]) == 1.0
    with pytest.raises(ValueError):
        StateStepper(AddOne(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), fixed_mask=mask)


class FirstExtraChannel(torch.nn.Module):
    """Residual net returning the last input channel (the stochastic one) for all outputs."""

    def forward(self, x):
        return x[:, -1:].expand(-1, 6, -1, -1)


def test_stochastic_channels_make_each_call_a_fresh_member():
    torch.manual_seed(0)
    stepper = StateStepper(FirstExtraChannel(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", stochastic_channels=1)
    state, wind = torch.zeros(1, 6, 3, 3), torch.zeros(1, 3, 3)
    first, second = stepper(state, wind), stepper(state, wind)
    assert not torch.allclose(first, second)
    deterministic = StateStepper(FirstExtraChannel(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual")
    torch.testing.assert_close(deterministic(state, wind), deterministic(state, wind))


class Spike(torch.nn.Module):
    """Residual net returning a single-cell spike in every channel."""

    def forward(self, x):
        out = torch.zeros_like(x[:, :6])
        out[:, :, 4, 4] = 1.0
        return out


def test_increment_smoothing_spreads_spikes_and_spares_land():
    with pytest.raises(ValueError):
        StateStepper(Spike(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), increment_smoothing=-1.0)
    mask = np.zeros((9, 9), dtype=bool)
    mask[0] = True
    values = np.zeros((6, 9, 9), dtype="float32")
    smooth = StateStepper(Spike(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", fixed_mask=mask, fixed_values=values, increment_smoothing=1.0)
    out = smooth(torch.zeros(1, 6, 9, 9), torch.zeros(1, 9, 9))
    assert float(out[0, 0, 4, 4]) < 0.5 and float(out[0, 0, 4, 5]) > 0.0
    torch.testing.assert_close(out[:, :, 0], torch.zeros(1, 6, 9))
    plain = StateStepper(Spike(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", increment_smoothing=1.0, mass_conserving=True)
    out = plain(torch.zeros(1, 6, 9, 9), torch.zeros(1, 9, 9))
    torch.testing.assert_close(out[:, :2].mean(dim=(2, 3)), torch.zeros(1, 2), atol=1e-6, rtol=0)
    assert float(out[0, 2, 4, 4]) < 0.5


def test_coarse_stochastic_noise_is_smooth_and_validated():
    with pytest.raises(ValueError):
        StateStepper(FirstExtraChannel(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), stochastic_channels=1, stochastic_scale=0)
    torch.manual_seed(0)
    fine = StateStepper(FirstExtraChannel(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", stochastic_channels=1)
    coarse = StateStepper(FirstExtraChannel(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", stochastic_channels=1, stochastic_scale=4)
    state, wind = torch.zeros(1, 6, 30, 22), torch.zeros(1, 30, 22)
    rough = lambda out: float((out[0, 0, :, 1:] - out[0, 0, :, :-1]).abs().mean())
    a, b = fine(state, wind), coarse(state, wind)
    assert b.shape == a.shape
    assert rough(b) < 0.5 * rough(a)
