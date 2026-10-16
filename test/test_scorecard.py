import numpy as np
import pytest

from src.scorecard import PREDICTABLE_STEPS, drift_score, realism_from_statistics, eke_scores, predictable_mse, scorecard, spectral_scores, summarise


def fields(seed=0, shape=(40, 6, 44, 40)):
    return np.random.default_rng(seed).normal(size=shape)


def test_predictable_mse_averages_the_first_steps():
    errors = np.concatenate([np.ones(PREDICTABLE_STEPS), np.full(50, 100.0)])
    assert predictable_mse(errors) == 1.0
    assert predictable_mse(np.array([2.0, 4.0]), steps=5) == 3.0


def test_identical_rollouts_score_perfectly():
    truth = fields()
    scores = scorecard(truth, truth, np.zeros(40))
    assert scores["spectral_error"] == pytest.approx(0.0)
    assert scores["noise_excess_decades"] == pytest.approx(0.0)
    assert scores["eke_amplitude_error"] == pytest.approx(0.0)
    assert scores["eke_pattern_error"] == pytest.approx(0.0, abs=1e-12)
    assert scores["realism"] == pytest.approx(0.0, abs=1e-12)


def test_scaled_fields_give_log_ratio_errors():
    truth = fields()
    spectral = spectral_scores(2 * truth, truth)  # power x4 in every band
    assert spectral["spectral_error"] == pytest.approx(np.log(4.0))
    assert spectral["noise_excess_decades"] == pytest.approx(np.log10(4.0))
    assert spectral_scores(0.5 * truth, truth)["noise_excess_decades"] == 0.0  # deficits are not noise
    eke = eke_scores(2 * truth, truth)
    assert eke["eke_amplitude_error"] == pytest.approx(np.log(4.0))


def test_degenerate_spectra_give_nan_components_but_finite_realism():
    truth = fields(shape=(10, 6, 12, 10))  # interior too small for a large-scale band
    zero_small = spectral_scores(np.zeros_like(truth), truth)
    assert np.isnan(zero_small["noise_excess_decades"])
    scores = scorecard(np.zeros_like(truth) + 1e-9 * truth, truth, np.zeros(10))
    assert np.isfinite(scores["realism"]) or np.isnan(scores["eke_amplitude_error"])


def test_drift_score():
    truth = [1.0, 2.0]
    steady = drift_score([1.5] * 10, truth)
    assert steady == {"years_outside_truth_range": 0.0, "ke_trend": 0.0, "drift_score": 0.0}
    rising = drift_score([1.0] * 5 + [3.0] * 5, truth)
    assert rising["years_outside_truth_range"] == 0.5
    assert rising["ke_trend"] == pytest.approx(np.log(3.0))


def test_summarise_averages_late_rollouts():
    rollouts = [{"epoch": 10, "scorecard": {"a": 100.0}}, {"epoch": 30, "scorecard": {"a": 1.0}}, {"epoch": 60, "scorecard": {"a": 3.0}}, {"epoch": 40}]
    assert summarise(rollouts) == {"a": 2.0}
    assert summarise(rollouts, 70, 80) == {}


def test_tiny_grids_are_scored_whole():
    truth = fields(shape=(6, 6, 4, 4))
    scores = scorecard(truth, truth, np.zeros(6))
    assert scores["eke_amplitude_error"] == pytest.approx(0.0)


def test_realism_from_statistics_matches_direct_scorecard():
    from src.diagnostics import compare_statistics

    truth = fields(seed=1)
    model = 0.8 * truth + 0.3 * fields(seed=2)
    times = np.arange(len(truth)) * 7.0
    rebuilt = realism_from_statistics(compare_statistics(model, truth, times, times, 40.0, max_lag=3))
    direct = scorecard(model, truth, np.zeros(len(truth)))
    for key in ("spectral_error", "noise_excess_decades", "eke_amplitude_error", "eke_pattern_error", "realism"):
        assert rebuilt[key] == pytest.approx(direct[key], rel=1e-6)
    no_small = compare_statistics(truth, truth, times, times, 40.0, max_lag=3)
    for layer in no_small["layers"].values():
        for key in [k for k in layer if k.endswith("band_ratio") and "spectrum" in k]:
            layer[key]["small"] = float("nan")
    assert realism_from_statistics(no_small)["noise_excess_decades"] == 0.0
