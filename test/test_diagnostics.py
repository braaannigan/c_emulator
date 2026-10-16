import numpy as np
import pytest

from src.diagnostics import (
    SPATIAL_BANDS,
    anomalies,
    autocorrelation,
    band_power,
    block_statistics,
    compare_statistics,
    efolding_lag,
    eke_map,
    extended_forcing,
    frequency_spectrum,
    kinetic_energy_series,
    pattern_stats,
    variance_map,
    wavenumber_spectrum,
    wind_response_r2,
)


def test_anomalies_remove_time_mean():
    field = np.arange(12, dtype=float).reshape(3, 2, 2)
    np.testing.assert_allclose(anomalies(field).mean(axis=0), 0.0)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_wavenumber_spectrum_peaks_at_the_wave_number(axis):
    n, k = 64, 8  # 8 cycles across 64 cells -> 0.125 cycles per cell
    wave = np.sin(2 * np.pi * k * np.arange(n) / n)
    amplitude = np.random.default_rng(0).normal(size=(30, 1))
    field = amplitude[:, :, None] * wave[None, None, :] if axis == "x" else amplitude[:, :, None] * wave[None, :, None]
    field = np.broadcast_to(field, (30, n, n) if axis == "y" else (30, 5, n)).copy()
    freq, power = wavenumber_spectrum(field, axis)
    assert freq[np.argmax(power)] == pytest.approx(k / n)


def test_wavenumber_spectrum_rejects_bad_axis():
    with pytest.raises(ValueError):
        wavenumber_spectrum(np.zeros((2, 2, 2)), "z")


def test_frequency_spectrum_peaks_at_the_signal_frequency_and_sums_to_variance():
    t = np.arange(200)
    field = np.sin(2 * np.pi * 0.1 * t)[:, None, None] * np.ones((1, 3, 3))
    freq, power = frequency_spectrum(field)
    assert freq[np.argmax(power)] == pytest.approx(0.1)
    assert power[1:].sum() == pytest.approx(anomalies(field).var(axis=0).mean() + 0 * 1, rel=1e-6)
    _, flat = frequency_spectrum(np.zeros((10, 2, 2)))
    np.testing.assert_allclose(flat, 0.0)


def test_band_power_sums_bins_and_skips_zero_frequency():
    freq = np.array([0.0, 0.1, 0.3, 0.45])
    power = np.array([100.0, 1.0, 2.0, 3.0])
    assert band_power(freq, power, SPATIAL_BANDS) == {"large": 0.0, "eddy": 1.0, "small": 5.0}


def test_eke_and_variance_maps():
    u = np.array([[[0.0]], [[2.0]]])
    v = np.array([[[0.0]], [[0.0]]])
    np.testing.assert_allclose(eke_map(u, v), [[0.5]])
    np.testing.assert_allclose(variance_map(u), [[1.0]])


def test_pattern_stats():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    stats = pattern_stats(2 * truth, truth)
    assert stats["ratio"] == pytest.approx(2.0) and stats["pattern_correlation"] == pytest.approx(1.0)
    assert np.isnan(pattern_stats(np.ones((2, 2)), truth)["pattern_correlation"])


def test_autocorrelation_of_ar1_matches_theory_and_efolding():
    rng = np.random.default_rng(0)
    phi, n = 0.8, 20000
    x = np.zeros((n, 2, 1))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(size=(2, 1))
    acf = autocorrelation(x, 5)
    np.testing.assert_allclose(acf, phi ** np.arange(6), atol=0.03)
    assert efolding_lag(acf) == pytest.approx(-1 / np.log(phi), abs=0.3)
    with pytest.raises(ValueError):
        autocorrelation(x[:3], 3)


def test_efolding_lag_never_crossing_is_infinite():
    assert efolding_lag(np.array([1.0, 0.9, 0.8])) == float("inf")


def test_wind_response_r2_detects_forced_signal():
    times = np.arange(0, 7 * 300, 7.0)
    phase = 2 * np.pi * times / 40.0
    forced = (np.sin(phase) + 0.5 * np.cos(phase))[:, None, None] * np.ones((1, 2, 2))
    assert wind_response_r2(forced, times, 40.0) == pytest.approx(1.0)
    noise = np.random.default_rng(0).normal(size=(300, 2, 2))
    assert wind_response_r2(noise, times, 40.0) < 0.05


def test_kinetic_energy_series_and_block_statistics():
    states = np.zeros((4, 6, 2, 2))
    states[:, 2] = 2.0  # u0 = 2 -> KE0 = 2
    states[2:, :2] = np.array([1.0, -1.0])[None, None, None, :]
    ke = kinetic_energy_series(states)
    np.testing.assert_allclose(ke[:, 0], 2.0)
    np.testing.assert_allclose(ke[:, 1], 0.0)
    blocks = block_statistics(states, 2)
    assert blocks["kinetic_energy"] == [1.0, 1.0]
    assert blocks["thickness_std"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        block_statistics(states, 0)


def test_compare_statistics_of_identical_runs_is_perfect():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(60, 6, 44, 40))
    times = np.arange(60) * 7.0
    report = compare_statistics(truth, truth, times, times, 40.0, max_lag=5)
    assert report["kinetic_energy_ratio"] == pytest.approx(1.0)
    assert report["thickness_std_ratio"] == pytest.approx(1.0)
    layer = report["layers"]["0"]
    assert layer["eke_map"]["pattern_correlation"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in layer["thickness_spectrum_x_band_ratio"].values())
    assert all(v == pytest.approx(1.0) for v in layer["thickness_frequency_band_ratio"].values())
    assert layer["thickness_efolding_steps"]["model"] == layer["thickness_efolding_steps"]["truth"]
    assert len(layer["thickness_autocorrelation"]["model"]) == 6


def test_compare_statistics_band_ratio_is_nan_when_truth_band_is_empty():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(20, 6, 12, 10))  # 6-cell interior: no large-scale x bin
    times = np.arange(20) * 7.0
    report = compare_statistics(truth, truth, times, times, 40.0, max_lag=3)
    assert np.isnan(report["layers"]["0"]["thickness_spectrum_x_band_ratio"]["large"])


def test_extended_forcing_continues_the_record_exactly():
    import xarray as xr

    from src.data import analytic_surface_wind_stress

    ds = xr.Dataset(
        coords={"time_days": np.arange(5) * 7.0, "y": np.arange(4) * 10.0, "x": np.arange(3) * 10.0},
        attrs={"wind_stress_max": 0.05, "wind_shift_amplitude_m": 10.0, "wind_shift_period_days": 40.0, "output_interval_days": "7.0"},
    )
    times, forcing = extended_forcing(ds, 3)
    np.testing.assert_allclose(times, np.arange(8) * 7.0)
    assert forcing.shape == (8, 4, 3)
    np.testing.assert_allclose(forcing[:5], analytic_surface_wind_stress(ds))


def test_relative_vorticity_of_solid_body_rotation_is_twice_the_rate():
    from src.diagnostics import relative_vorticity

    y, x = np.meshgrid(np.arange(10.0), np.arange(12.0), indexing="ij")
    omega = 0.5
    u, v = -omega * y, omega * x  # counter-clockwise rotation
    np.testing.assert_allclose(relative_vorticity(u, v, 1.0), 2 * omega)
    np.testing.assert_allclose(relative_vorticity(u, v, 2.0), omega)
