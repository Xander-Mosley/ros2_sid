"""Check lossless numeric export of the interactive surface message fields."""

from ardupilot_msgs.msg import ControlSurfaceExcitation, ControlSurfaceOutput

from ros2_sid.post_process.data_parser import flatten_surface_message


def test_command_stamp_and_normalized_axes() -> None:
    """Keep source time and signed commands without angle conversion."""
    message = ControlSurfaceExcitation()
    message.sequence_id = 4294967295
    message.stamp.sec = 12
    message.stamp.nanosec = 345
    message.roll = -0.125
    row = flatten_surface_message(message)
    assert row['stamp_sec'] == 12
    assert row['stamp_nanosec'] == 345
    assert row['sequence_id'] == 4294967295
    assert row['roll'] == -0.125


def test_output_preserves_channel_indices_and_clock() -> None:
    """Retain calibration and coherent response in the same output row."""
    message = ControlSurfaceOutput()
    message.time_us = 123456
    message.clock_us = 987654321
    message.count = 4
    message.baseline[0] = 1600
    message.pwm[0] = 1640
    message.delta[0] = 40
    message.rates[1] = -0.5
    row = flatten_surface_message(message)
    assert row['time_us'] == 123456
    assert row['clock_us'] == 987654321
    assert row['pwm_0'] - row['baseline_0'] == row['delta_0'] == 40
    assert row['rates_1'] == -0.5
    assert row['count'] == 4
