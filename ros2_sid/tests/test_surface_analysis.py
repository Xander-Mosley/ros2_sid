import numpy as np
import pytest
from ros2_sid.surface_analysis import expected_pwm


def test_unsynchronized_command_stamp_uses_receipt_estimate():
    from ros2_sid.surface_bag import command_time

    frame = (1000000000, 1700000000000000, 5000000)
    assert command_time(1700000005000000, 1000000000, frame) == (5000000, True)
    assert command_time(0, 1001000000, frame) == (5001000, False)


def test_zero_preserves_disabled_channel_output():
    assert expected_pwm(0, 0, 1100, 1500, 1900, False) == 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("u", [-0.1, 0.1])
def test_unsigned_ros_scalars(u, reverse):
    args = (1600, 1100, 1500, 1900)
    assert expected_pwm(np.float32(u), *map(np.uint16, args), reverse) == expected_pwm(
        float(np.float32(u)), *args, reverse
    )


def test_dataflash_comparison_matches_timestamps_and_detects_differences(monkeypatch):
    from ros2_sid import surface_analysis as analysis

    row = dict(TimeUS=100, Ch=0, Seq=1, Rev=1, Base=1500, Delta=40, PWM=1540, OK=1)
    data = {"bag": [("SXO", dict(row))], "bin": [("SXO", dict(row))]}
    monkeypatch.setattr(analysis, "records", lambda source: iter(data[source]))
    assert analysis.compare_outputs("bag", "bin")["passed"]
    data["bin"][0][1]["PWM"] = 1541
    assert analysis.compare_outputs("bag", "bin")["mismatched"] == 1
    data["bin"] = []
    assert analysis.compare_outputs("bag", "bin")["missing"] == 1


def test_maneuver_requires_full_duration_and_matching_axes(tmp_path, monkeypatch):
    from ros2_sid import surface_analysis as analysis

    path = tmp_path / "commands.csv"
    path.write_text(
        "stamp_s,sequence,enable,roll,pitch,yaw\n"
        + "".join(f"{100 + i / 100},{i},1,.1,0,0\n" for i in range(101))
    )
    frames = [
        ("SXA", dict(ClockUS=100000000 + i * 10000, State=2, Seq=i, R=0.1, P=0, Y=0))
        for i in range(0, 101, 2)
    ]
    monkeypatch.setattr(analysis, "records", lambda _: iter(frames))
    assert analysis.validate_maneuver("bag", path)["passed"]
    for _, frame in frames[:40]:
        frame["State"] = 1
    assert not analysis.validate_maneuver("bag", path)["passed"]
    for _, frame in frames:
        frame["State"] = 2
    frames[10][1]["P"] = 0.1
    assert analysis.validate_maneuver("bag", path)["mismatched"] == 1
