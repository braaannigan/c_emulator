import numpy as np
import xarray as xr

from src.data_state import NextStepStateDataset, fit_channel_normalisation, load_two_layer_state_arrays


def two_layer_dataset():
    return xr.Dataset(
        data_vars={
            "layer_thickness": (("time_days", "layers", "y", "x"), np.ones((4, 2, 3, 4), dtype="float32")),
            "zonal_velocity": (("time_days", "layers", "y", "xp1"), np.ones((4, 2, 3, 5), dtype="float32") * 2),
            "meridional_velocity": (("time_days", "layers", "yp1", "x"), np.ones((4, 2, 4, 4), dtype="float32") * 3),
        },
        coords={"time_days": np.arange(4), "layers": [0, 1], "y": np.arange(3), "x": np.arange(4), "xp1": np.arange(5), "yp1": np.arange(4)},
        attrs={"wind_stress_max": 0.05, "wind_shift_amplitude_m": 0.0, "wind_shift_period_days": 10.0},
    )


def test_load_two_layer_state_arrays(tmp_path):
    path = tmp_path / "two.nc"
    two_layer_dataset().to_netcdf(path)

    inputs, state, _ = load_two_layer_state_arrays(path)

    assert inputs.shape == (4, 7, 3, 4)
    assert state.shape == (4, 6, 3, 4)
    np.testing.assert_allclose(state[:, 0:2], 1.0)
    np.testing.assert_allclose(state[:, 2:4], 2.0)
    np.testing.assert_allclose(state[:, 4:6], 3.0)


def test_load_two_layer_state_arrays_rejects_wrong_layer_count(tmp_path):
    path = tmp_path / "one.nc"
    ds = two_layer_dataset().isel(layers=[0])
    ds.to_netcdf(path)

    try:
        load_two_layer_state_arrays(path)
    except ValueError as exc:
        assert "two layers" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_state_dataset_normalises_next_state():
    inputs = np.ones((4, 7, 2, 2), dtype="float32")
    state = np.ones((4, 6, 2, 2), dtype="float32") * np.arange(4, dtype="float32")[:, None, None, None]
    indices = np.array([0, 1, 2])
    norm = fit_channel_normalisation(inputs, state, indices)
    dataset = NextStepStateDataset(inputs, state, indices, norm)

    x, y = dataset[0]

    assert x.shape == (7, 2, 2)
    assert y.shape == (6, 2, 2)
