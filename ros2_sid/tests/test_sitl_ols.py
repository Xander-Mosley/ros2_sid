"""Numerical and data-boundary checks for the SITL coefficient benchmark."""

import numpy as np
import pytest

from ros2_sid.realtime_ols_utils import RecursiveFourierTransform
from ros2_sid.sitl_ols import fit_models, flight_arrays, maneuver_mask, servo_inputs


def test_scaled_ols_recovers_independent_coefficients_on_holdout():
    rng = np.random.default_rng(123)
    x = np.column_stack((np.ones(1000), rng.normal(size=(1000, 3))))
    x *= [1, .01, 100, .001]
    expected = np.array([.045, -.7, -20, 1.0])
    y = x @ expected
    train = np.arange(1000) < 700
    results, _ = fit_models({'pitch': (x, y, ['bias','alpha','q','elevator'])}, train, ~train)
    np.testing.assert_allclose(list(results['pitch']['coefficients'].values()), expected, atol=1e-8)
    assert results['pitch']['validation']['r2'] == pytest.approx(1)


def test_rank_deficiency_is_not_reported_as_identification():
    x = np.ones((100, 2))
    train = np.arange(100) < 60
    with pytest.raises(ValueError, match='rank deficient'):
        fit_models({'bad': (x, np.ones(100), ['a', 'b'])}, train, ~train)


def test_servo_scaling_and_lag():
    rows = [dict(TimeUS=0, C1=1500, C2=1500, C4=1500, C3=1000),
            dict(TimeUS=1000000, C1=1900, C2=1100, C4=1500, C3=2000)]
    t = np.array([.5, 1.14, 2.0])
    result = servo_inputs(rows, t, .14)
    np.testing.assert_allclose(result[0], 0)
    assert result[1, 0] == pytest.approx(.8 * (1 - np.exp(-1)), abs=.005)
    assert result[-1, 0] == pytest.approx(.8, abs=.001)
    assert result[-1, 1] == pytest.approx(-.8, abs=.001)
    assert result[-1, 3] == pytest.approx(1, abs=.001)


def test_maneuver_selection_uses_firmware_time_and_enabled_sequences(tmp_path):
    records = []
    for i, axis in enumerate(('roll', 'pitch', 'yaw')):
        (tmp_path / f'chirp_{axis}-commands.csv').write_text(
            f'sequence,enable\n{i*10},0\n{i*10+1},1\n{i*10+2},1\n')
        records += [dict(TimeUS=(i*4+1)*1000000, Seq=i*10+1),
                    dict(TimeUS=(i*4+3)*1000000, Seq=i*10+2)]
    times = np.arange(12, dtype=float)
    mask, intervals = maneuver_mask(tmp_path, {'SXC': records}, times, 'chirp')
    assert list(times[mask]) == [2, 6, 10]
    assert len(intervals) == 3


def test_explicit_fourier_frequencies_need_no_generated_config():
    transform = RecursiveFourierTransform(frequencies=np.array([.1, .3]))
    np.testing.assert_equal(transform.frequencies, [.1, .3])


def test_force_normalization_matches_simulator_eas_convention():
    data = {k: [] for k in ('SIM', 'SIM2', 'IMU', 'RCOU', 'SXO')}
    for stamp in range(0, 4000001, 20000):
        data['SIM'].append(dict(TimeUS=stamp, Alt=1000, Q1=1, Q2=0, Q3=0, Q4=0))
        data['SIM2'].append(dict(TimeUS=stamp, VN=20, VE=0, VD=0))
        data['IMU'].append(dict(TimeUS=stamp, GyrX=0, GyrY=0, GyrZ=0, AccX=0, AccY=0, AccZ=-10))
        data['RCOU'].append(dict(TimeUS=stamp, C1=1500, C2=1500, C3=1000, C4=1500))
    _, models, _ = flight_arrays(data, {})
    # Standard atmosphere at 1000 m: rho ~= 1.11166 kg/m^3. The simulator
    # applies both the density and EAS correction, unlike normal TAS qbar.
    expected = 20 / (.5 * 1.11166 * (20**2 * 1.11166 / 1.225) * .45)
    np.testing.assert_allclose(models['lift'][1], expected, rtol=2e-5)
