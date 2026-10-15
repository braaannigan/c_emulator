import numpy as np
import pytest

from src.evaluation import evaluate_rollout


class PersistenceEmulator:
    def __init__(self):
        self.calls = []

    def rollout(self, history, forcing, steps):
        self.calls.append((history.shape, forcing.shape, steps))
        return np.repeat(history[-1][None], steps, axis=0)


def make_state(n_times=10):
    state = np.zeros((n_times, 6, 2, 3), dtype="float32")
    state[:, :2] = np.arange(n_times, dtype="float32")[:, None, None, None]
    state[:, 2:] = 2 * np.arange(n_times, dtype="float32")[:, None, None, None]
    return state


def test_evaluator_only_exposes_past_state_and_known_forcing():
    state = make_state()
    forcing = np.zeros((10, 2, 3), dtype="float32")
    emulator = PersistenceEmulator()

    evaluate_rollout(emulator, state, forcing, np.array([7, 8]), None)

    assert emulator.calls == [((8, 6, 2, 3), (10, 2, 3), 2)]


def test_evaluator_scores_thickness_and_velocity_separately():
    state = make_state()
    forcing = np.zeros((10, 2, 3), dtype="float32")

    result = evaluate_rollout(PersistenceEmulator(), state, forcing, np.array([7, 8]), None)

    np.testing.assert_allclose(result.thickness_mse_by_time, [1.0, 4.0])
    np.testing.assert_allclose(result.velocity_mse_by_time, [4.0, 16.0])
    assert result.eval_mse_mean == 2.5
    assert result.eval_mse_last == 4.0
    assert result.velocity_mse_mean == 10.0
    assert result.truth.shape == result.prediction.shape == (2, 6, 2, 3)


def test_evaluator_truncates_to_max_steps():
    state = make_state()
    forcing = np.zeros((10, 2, 3), dtype="float32")
    emulator = PersistenceEmulator()

    result = evaluate_rollout(emulator, state, forcing, np.array([6, 7, 8]), 1)

    assert emulator.calls == [((7, 6, 2, 3), (8, 2, 3), 1)]
    assert result.thickness_mse_by_time.shape == (1,)


def test_evaluator_rejects_bad_shapes():
    state = make_state()
    forcing = np.zeros((10, 2, 3), dtype="float32")

    with pytest.raises(ValueError):
        evaluate_rollout(PersistenceEmulator(), state[:, :2], forcing, np.array([7, 8]))
    with pytest.raises(ValueError):
        evaluate_rollout(PersistenceEmulator(), state, forcing[:5], np.array([7, 8]))
    with pytest.raises(ValueError):
        evaluate_rollout(PersistenceEmulator(), state, forcing, np.array([7, 8]), 0)

    class WrongShape:
        def rollout(self, history, forcing, steps):
            return np.zeros((steps, 2, 2, 3), dtype="float32")

    with pytest.raises(ValueError):
        evaluate_rollout(WrongShape(), state, forcing, np.array([7, 8]))
