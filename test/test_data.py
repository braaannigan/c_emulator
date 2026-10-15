import numpy as np
import pytest
import xarray as xr

from src.data import analytic_surface_wind_stress, centre_meridional_velocity, centre_zonal_velocity, chronological_pair_split


def test_centre_zonal_velocity_averages_adjacent_x_faces():
    u = np.array([[[1.0, 3.0, 7.0]]])

    centred = centre_zonal_velocity(u)

    np.testing.assert_allclose(centred, np.array([[[2.0, 5.0]]]))


def test_centre_meridional_velocity_averages_adjacent_y_faces():
    v = np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])

    centred = centre_meridional_velocity(v)

    np.testing.assert_allclose(centred, np.array([[3.0, 5.0], [7.0, 9.0]]))


def test_chronological_pair_split_uses_first_80_percent_for_training_pairs():
    train, eval_ = chronological_pair_split(10, 0.8)

    np.testing.assert_array_equal(train, np.arange(0, 7))
    np.testing.assert_array_equal(eval_, np.arange(7, 9))


def test_chronological_pair_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        chronological_pair_split(10, 1.0)


def test_chronological_pair_split_requires_at_least_three_times():
    with pytest.raises(ValueError):
        chronological_pair_split(2, 0.8)


def test_analytic_surface_wind_stress_uses_metadata_and_thickness_grid():
    ds = xr.Dataset(
        coords={"time_days": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 1.0]), "x": np.array([0.5, 1.5, 2.5, 3.5])},
        attrs={"wind_stress_max": 0.05, "wind_shift_amplitude_m": 0.0, "wind_shift_period_days": 10.0},
    )

    wind = analytic_surface_wind_stress(ds)

    assert wind.shape == (3, 2, 4)
    assert wind.dtype == np.float32
    assert np.isclose(np.abs(wind).max(), 0.05)
