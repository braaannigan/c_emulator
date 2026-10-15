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
