"""Timing checks do not require ROS or replay actuator commands."""

from ros2_sid.air_data_analysis import timing


def row(t, seq, flags=31):
    return dict(time_us=t, sequence_id=seq, flags=flags)


def test_fifty_hz_and_invalid():
    result = timing([row(0, 1), row(20000, 2), row(40000, 3, 0)])
    assert result["mean_received_hz"] == 50
    assert result["valid_fraction"] == 2 / 3


def test_duplicates_gaps_resets_and_wrap():
    result = timing(
        [row(0, 2**32 - 1), row(20000, 0), row(60000, 2), row(60000, 2), row(1, 1)]
    )
    assert result["sequence_gap_samples"] == 1
    assert result["duplicate_samples"] == 1
    assert result["clock_or_sequence_resets"] == 2


def test_absent():
    assert not timing([])["available"]


def test_delay_and_flat_signals():
    import numpy as np
    from ros2_sid.air_data_analysis import signal_delay

    t = np.arange(500) * 0.02
    truth = np.sin(t * 2) + 0.4 * np.sin(t * 7)
    estimated = np.sin((t - 0.06) * 2) + 0.4 * np.sin((t - 0.06) * 7)
    result = signal_delay(t, estimated, truth)
    assert abs(result["lag_ms"] - 60) < 1
    assert not signal_delay(t, truth * 0, truth)["available"]


def test_wall_clock_jump_is_separate_from_acquisition_clock():
    rows = [dict(row(0, 1), clock_us=1000000), dict(row(20000, 2), clock_us=2020000)]
    result = timing(rows)
    assert result["wall_clock_jumps"] == 1
    assert result["mean_received_hz"] == 50
