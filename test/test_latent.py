import numpy as np
import pytest
import torch

from src.models.latent import LatentEmulator, LatentModel


def make(mask=None, static=True, levels=2, latent_noise_std=0.0):
    values = None
    if mask is not None:
        values = np.zeros((6, 16, 8), dtype="float32")
        values[:2] = 200.0
    return LatentModel(
        np.full(6, 0.0), np.ones(6), 0.0, 1.0, np.ones(6),
        static=np.zeros((2, 16, 8), dtype="float32") if static else None,
        fixed_mask=mask, fixed_values=values, latent_channels=4, hidden_channels=4, levels=levels, stepper_channels=4, stepper_depth=1, latent_noise_std=latent_noise_std,
    )


def test_latent_model_validates_arguments():
    with pytest.raises(ValueError):
        make(levels=0)
    with pytest.raises(ValueError):
        LatentModel(np.zeros(6), np.ones(6), 0.0, 1.0, np.ones(6), fixed_mask=np.zeros((16, 8), dtype=bool))


def test_encode_step_decode_shapes_and_volume_without_mask():
    torch.manual_seed(0)
    model = make(static=False)
    state = torch.randn(2, 6, 16, 8) + 200 * torch.tensor([1, 1, 0, 0, 0, 0.0]).view(1, 6, 1, 1)
    latent = model.encode(state)
    assert latent.shape == (2, 4, 4, 2)
    stepped = model.step(latent, torch.zeros(2, 16, 8))
    decoded = model.decode(stepped, (16, 8), model.wet_volume(state))
    assert decoded.shape == (2, 6, 16, 8)
    torch.testing.assert_close(decoded[:, :2].mean(dim=(2, 3)), state[:, :2].mean(dim=(2, 3)))


def test_decode_pins_land_and_restores_wet_volume():
    torch.manual_seed(0)
    mask = np.zeros((16, 8), dtype=bool)
    mask[0] = True
    model = make(mask=mask)
    state = torch.from_numpy(np.broadcast_to(model.fixed_values[0].numpy(), (2, 6, 16, 8)).copy())
    state[:, :2, 1:] += torch.randn(2, 2, 15, 8)
    decoded = model.decode(model.encode(state), (16, 8), model.wet_volume(state))
    torch.testing.assert_close(decoded[:, :, 0], model.fixed_values[0, :, 0].expand(2, -1, -1))
    torch.testing.assert_close(model.wet_volume(decoded), model.wet_volume(state))


@pytest.mark.parametrize("masked", [False, True])
def test_window_loss_is_finite_and_differentiable(masked):
    torch.manual_seed(0)
    mask = np.zeros((16, 8), dtype=bool)
    mask[0] = True
    model = make(mask=mask if masked else None)
    states = torch.randn(2, 3, 6, 16, 8)
    loss = model.window_loss(states, torch.zeros(2, 2, 16, 8), noise_std=1.0)
    loss.backward()
    assert torch.isfinite(loss) and model.stepper.head.weight.grad is not None


def test_latent_emulator_rollout_shape():
    model = make()
    history = np.zeros((3, 6, 16, 8), dtype="float32")
    pred = LatentEmulator(model, torch.device("cpu")).rollout(history, np.zeros((7, 16, 8), dtype="float32"), 4)
    assert pred.shape == (4, 6, 16, 8)


def test_latent_noise_changes_the_rollout_loss():
    states = torch.randn(2, 3, 6, 16, 8)
    winds = torch.zeros(2, 2, 16, 8)
    torch.manual_seed(0)
    clean = make(latent_noise_std=0.0)
    torch.manual_seed(0)
    noisy = make(latent_noise_std=1.0)
    torch.manual_seed(1)
    a = clean.window_loss(states, winds)
    torch.manual_seed(1)
    b = noisy.window_loss(states, winds)
    assert float(a) != float(b)
