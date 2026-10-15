import pytest
import torch

from src.models.unet import StateUNet


@pytest.mark.parametrize("shape", [(16, 8), (25, 13)])
def test_unet_preserves_shape_on_even_and_odd_grids(shape):
    model = StateUNet(9, 6, base_channels=4, depth=2)
    assert model(torch.zeros(2, 9, *shape)).shape == (2, 6, *shape)


def test_unet_receptive_field_spans_coarse_scales():
    torch.manual_seed(0)
    model = StateUNet(1, 1, base_channels=4, depth=3)
    x = torch.zeros(1, 1, 32, 32, requires_grad=True)
    model(x)[0, 0, 16, 16].backward()
    # a plain 4-layer 3x3 CNN would only see 9x9 cells around the centre
    assert x.grad[0, 0, 0, 0] != 0


def test_unet_rejects_zero_depth():
    with pytest.raises(ValueError):
        StateUNet(1, 1, depth=0)


def test_bottleneck_attention_couples_distant_coarse_cells():
    torch.manual_seed(0)
    model = StateUNet(1, 1, base_channels=8, depth=1, attention_layers=1)
    x = torch.zeros(1, 1, 40, 40, requires_grad=True)
    model(x)[0, 0, 0, 0].backward()
    # Without attention a depth-1 U-Net only sees nearby cells; with it the far corner matters.
    assert x.grad[0, 0, -1, -1] != 0
    assert model(torch.zeros(2, 1, 25, 13)).shape == (2, 1, 25, 13)
