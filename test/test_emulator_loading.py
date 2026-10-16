import numpy as np

from src.emulator_loading import default_checkpoint, load_trained_emulator
from src.models.stepper import StepperEmulator
from src.pipelines.dgsw2l_baseline import run_experiment
from test_dgsw2l_pipeline import write_config


def test_load_trained_emulator_round_trips_a_saved_checkpoint(tmp_path):
    config = write_config(tmp_path)
    run_experiment(config)
    assert default_checkpoint(config).exists()
    loaded = load_trained_emulator(config)
    assert isinstance(loaded.emulator, StepperEmulator)
    assert loaded.forcing.shape == (len(loaded.state), *loaded.state.shape[2:])
    assert len(loaded.train_indices) + len(loaded.eval_indices) <= len(loaded.state)
    pred = loaded.emulator.rollout(loaded.state[:3], loaded.forcing[:5], 2)
    assert pred.shape == (2, *loaded.state.shape[1:]) and np.isfinite(pred).all()
