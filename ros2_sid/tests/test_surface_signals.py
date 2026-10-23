import json
import math
import pytest
from ros2_sid.surface_signals import Experiment, Cadence, ClockSync, RCGate


def mission(tmp_path, axes=None, **kw):
    p = tmp_path / "mission.json"
    p.write_text(
        json.dumps(
            dict(
                units="normalized_surface", rate=100, duration=10, axes=axes or {}, **kw
            )
        )
    )
    return Experiment(p)


def test_constant_simultaneous_and_duration(tmp_path):
    e = mission(
        tmp_path,
        {
            a: dict(signal="constant", amplitude=v)
            for a, v in zip(("roll", "pitch", "yaw"), (0.1, -0.2, 0.3))
        },
    )
    assert e.sample(1) == (0.1, -0.2, 0.3)
    assert e.sample(10) == (0, 0, 0)


def test_chirp_phase(tmp_path):
    e = mission(
        tmp_path,
        {
            "roll": dict(
                signal="chirp", amplitude=0.1, start_frequency=0.2, end_frequency=3
            )
        },
    )
    for t in [0, 0.25, 1.2, 9.9]:
        assert e.sample(t)[0] == pytest.approx(
            0.1 * math.sin(2 * math.pi * (0.2 * t + 2.8 * t * t / 20))
        )


def test_multisine_peak_and_normalization(tmp_path):
    spec = dict(
        signal="multisine",
        amplitude=0.1,
        frequencies=[1, 2],
        amplitudes=[0.1, 0.1],
        phases=[0, 1],
    )
    with pytest.raises(ValueError):
        mission(tmp_path, {"yaw": spec})
    spec["normalize"] = True
    e = mission(tmp_path, {"yaw": spec})
    assert all(abs(e.sample(i / 100)[2]) <= 0.1 for i in range(1000))


def test_units_reject_radians(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text('{"units":"radians","duration":10}')
    with pytest.raises(ValueError):
        Experiment(p)


def test_csv_interpolation_and_bounds(tmp_path):
    (tmp_path / "samples.csv").write_text(
        "time,roll,pitch,yaw\n0,0,0,0\n10,.1,-.2,.3\n"
    )
    e = mission(tmp_path, csv="samples.csv")
    assert e.sample(5) == pytest.approx((0.05, -0.1, 0.15))
    (tmp_path / "samples.csv").write_text("time,roll,pitch,yaw\n0,0,0,0\n10,1.1,0,0\n")
    with pytest.raises(ValueError):
        mission(tmp_path, csv="samples.csv")


def test_cadence_skips_and_aborts():
    c = Cadence(100, 1)
    assert c.due(1) == 0
    assert c.due(1.005) is None
    assert c.due(1.025) == pytest.approx(0.025)
    assert c.deadline == pytest.approx(1.03)
    with pytest.raises(RuntimeError):
        c.due(1.06)


def test_clock_stale_jump_and_recovery():
    c = ClockSync()
    with pytest.raises(RuntimeError):
        c.stamp(1)
    c.update(100, 1)
    assert c.stamp(1.2) == pytest.approx(100.2)
    with pytest.raises(RuntimeError):
        c.stamp(3)
    with pytest.raises(RuntimeError):
        c.update(1000, 2)
    with pytest.raises(RuntimeError):
        c.stamp(2)
    c.update(1001, 3)
    assert c.stamp(3) == 1001


def test_hitl_gate_fails_closed():
    g = RCGate(0)
    g.update([1900], 1, 1)
    assert not g.allowed(1)
    g.update([1000], 1, 2)
    g.update([1900], 1, 2.1)
    assert g.allowed(2.2)
    assert not g.allowed(3)
    g.update([1900], 1, 3.1)
    assert not g.allowed(3.1)
    g.update([], 0, 4)
    assert not g.allowed(4)


def test_multisine_maneuver_examples():
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "config/surface_excitation"
    plan = json.loads((config / "mission_plan.json").read_text())
    for maneuver in plan["maneuvers"]:
        Experiment(config / maneuver["experiment"])
    e = Experiment(config / "multisine_all_axes.json")
    for i in range(int(e.duration * e.rate)):
        t = i / e.rate
        values = e.sample(t)
        for j, axis in enumerate(("roll", "pitch", "yaw")):
            spec = e.axes[axis]
            expected = sum(
                a * math.sin(2 * math.pi * f * t + phase)
                for a, f, phase in zip(
                    spec["amplitudes"], spec["frequencies"], spec["phases"]
                )
            )
            assert values[j] == pytest.approx(expected)
            assert abs(values[j]) <= spec["amplitude"]
    assert len(set(e.sample(0.37))) == 3
    assert e.sample(e.duration) == (0.0, 0.0, 0.0)


def test_legacy_rc_defaults_and_gap_reactivation():
    """Channel eight is high at 1550; a delivery gap requires another low edge."""
    gate = RCGate()
    channels = [1500] * 8
    gate.update(channels, 255, 0)
    channels[7] = 1550
    gate.update(channels, 255, .1)
    assert gate.allowed(.1)
    gate.update(channels, 255, 1)
    assert not gate.allowed(1)
    channels[7] = 1549
    gate.update(channels, 255, 1.1)
    assert not gate.allowed(1.1)
    channels[7] = 1550
    gate.update(channels, 255, 1.2)
    assert gate.allowed(1.2)
