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


class SeesWind(torch.nn.Module):
    """Residual net whose output is the sum of all forcing channels (inputs 6..)."""

    def forward(self, x):
        return x[:, 6:].sum(dim=1, keepdim=True).expand(-1, 6, -1, -1)


def test_forcing_window_feeds_consecutive_winds_to_network():
    stepper = StateStepper(SeesWind(), np.zeros(8), np.ones(8), np.zeros(6), np.ones(6), target_mode="residual", forcing_window=2)
    out = stepper(torch.zeros(1, 6, 2, 2), torch.stack([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)])[None])
    torch.testing.assert_close(out, torch.full((1, 6, 2, 2), 3.0))
    with pytest.raises(ValueError):
        StateStepper(SeesWind(), np.zeros(8), np.ones(8), np.zeros(6), np.ones(6), forcing_window=0)


def test_emulator_rollout_uses_wind_at_step_start_and_end():
    stepper = StateStepper(SeesWind(), np.zeros(8), np.ones(8), np.zeros(6), np.ones(6), target_mode="residual", forcing_window=2)
    forcing = np.arange(5, dtype="float32")[:, None, None] * np.ones((5, 2, 2), dtype="float32")
    # start = 1; steps use winds (1, 2), (2, 3), (3, 4) -> increments 3, 5, 7.
    pred = StepperEmulator(stepper, torch.device("cpu")).rollout(np.zeros((2, 6, 2, 2), dtype="float32"), forcing, 3)
    np.testing.assert_allclose(pred[:, 0, 0, 0], [3.0, 8.0, 15.0])


class SeesTendency(torch.nn.Module):
    """Residual net that repeats the tendency channels (after state + one wind channel)."""

    def forward(self, x):
        return x[:, 7:13]


def two_frame(net=None):
    return StateStepper(net or SeesTendency(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", history_frames=2, tendency_mean=np.zeros(6), tendency_std=np.ones(6))


def test_two_frame_stepper_validates_arguments():
    with pytest.raises(ValueError):
        StateStepper(AddOne(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), history_frames=3)
    with pytest.raises(ValueError):
        StateStepper(AddOne(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), history_frames=2)
    with pytest.raises(ValueError):
        two_frame()(torch.zeros(1, 6, 2, 2), torch.zeros(1, 2, 2))


def test_two_frame_stepper_feeds_tendency_and_rolls_history_forward():
    stepper = two_frame()
    out = stepper(torch.full((1, 6, 2, 2), 3.0), torch.zeros(1, 2, 2), torch.full((1, 6, 2, 2), 1.0))
    torch.testing.assert_close(out, torch.full((1, 6, 2, 2), 5.0))  # 3 + (3 - 1)
    history = np.stack([np.full((6, 2, 2), v, dtype="float32") for v in (0.0, 1.0)])
    # Constant-velocity extrapolation: 1 -> 2 -> 3 -> 4.
    pred = StepperEmulator(stepper, torch.device("cpu")).rollout(history, np.zeros((5, 2, 2), dtype="float32"), 3)
    np.testing.assert_allclose(pred[:, 0, 0, 0], [2.0, 3.0, 4.0])
