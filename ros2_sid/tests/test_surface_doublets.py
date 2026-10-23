"""Check DDS doublets against the actual legacy generator and CSV loader."""

import json
from pathlib import Path

import numpy as np
import pytest
from ros2_sid.input_designs import multi_step
from ros2_sid.surface_doublets import generate_doublets
from ros2_sid.surface_signals import Experiment


def test_generated_doublets_match_legacy_and_preserve_menu(tmp_path: Path) -> None:
    """Pulse signs, sampled boundaries, axis isolation and menu IDs stay reproducible."""
    plan = {
        "units": "normalized_surface",
        "maneuvers": [{"filename": "old.csv", "order": 6}],
    }
    (tmp_path / "mission_plan.json").write_text(json.dumps(plan))
    generate_doublets(tmp_path)
    generate_doublets(
        tmp_path
    )  # Idempotent regeneration never duplicates menu entries.
    assert (
        len(json.loads((tmp_path / "mission_plan.json").read_text())["maneuvers"]) == 4
    )
    times, legacy = multi_step(0.1, 1, [1, 1], 1, 0.02, 15)
    for index, axis in enumerate(("roll", "pitch", "yaw")):
        config = {
            "units": "normalized_surface",
            "rate": 50,
            "duration": 15,
            "csv": f"doublet_{axis}.csv",
        }
        experiment = Experiment(tmp_path / "spec.json", config=config)
        samples = np.array([experiment.sample(float(t)) for t in times])
        np.testing.assert_allclose(samples[:, index], legacy, atol=1e-12)
        assert np.max(samples[:, index]) == 0.1
        assert np.min(samples[:, index]) == -0.1
        assert np.all(samples[:, [i for i in range(3) if i != index]] == 0)
        assert np.all(samples[times < 1] == 0)
        assert np.all(samples[times > 1.5] == 0)


def test_reject_invalid_amplitude_and_legacy_units(tmp_path: Path) -> None:
    """Do not reinterpret radians or permit invalid normalized amplitudes."""
    for amplitude in (0, -1, 2, float("nan")):
        with pytest.raises(ValueError):
            generate_doublets(tmp_path, amplitude)
    (tmp_path / "mission_plan.json").write_text('{"units":"radians"}')
    with pytest.raises(ValueError):
        generate_doublets(tmp_path)
