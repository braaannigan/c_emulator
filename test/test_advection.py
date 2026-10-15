import pytest
import torch

from src.models.advection import advect_state


def blob_state(ny=20, nx=40, u=0.0, v=0.0):
    state = torch.zeros(1, 6, ny, nx)
    state[0, 0, 10, 10] = 1.0  # h0 blob at (y=10, x=10)
    state[0, 1, 10, 10] = 2.0  # h1 blob
    state[0, 2] = u  # u0
    state[0, 4] = v  # v0
    return state


def test_zero_velocity_is_identity():
    state = blob_state()
    torch.testing.assert_close(advect_state(state, 7 * 86400.0, 1e4, 4), state)


def test_uniform_eastward_flow_moves_upper_layer_only():
    # 1 m/s for 50 000 s with 10 km cells moves 5 cells east.
    out = advect_state(blob_state(u=1.0), 50_000.0, 1e4, 5)
    assert out[0, 0].argmax().item() == 10 * 40 + 15
    assert out[0, 1].argmax().item() == 10 * 40 + 10  # lower layer has no flow
    torch.testing.assert_close(out[0, 2], torch.ones(20, 40))


def test_uniform_northward_flow_moves_along_y():
    out = advect_state(blob_state(v=1.0), 30_000.0, 1e4, 3)
    assert out[0, 0].argmax().item() == 13 * 40 + 10


def test_substeps_must_be_positive():
    with pytest.raises(ValueError):
        advect_state(blob_state(), 1.0, 1.0, 0)
