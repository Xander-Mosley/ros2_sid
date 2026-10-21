"""Read rosbag2 without replaying commands to a live flight controller."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def command_time(
    stamp_us: int, recorded_ns: int, frame: tuple[int, int, int]
) -> tuple[int, bool]:
    """Map a command to boot microseconds, returning whether its source stamp was usable."""
    recorded_frame_ns, clock_offset_us, boot_us = frame
    source_time = stamp_us - clock_offset_us
    receipt_estimate = boot_us + (recorded_ns - recorded_frame_ns) // 1000
    valid = abs(source_time - receipt_estimate) <= 2000000
    # Pre-synchronization or deliberately invalid stamps have no usable clock
    # mapping. Show their estimated bag receipt time, explicitly marked invalid.
    return (source_time if valid else receipt_estimate), valid


def records(path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield DataFlash-shaped records from a ROS bag without replaying actuator commands."""
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    path = Path(path)
    metadata = yaml.safe_load((path / "metadata.yaml").read_text())
    storage = metadata["rosbag2_bagfile_information"]["storage_identifier"]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    pending_commands = []
    clock_offsets = []
    while reader.has_next():
        topic, data, recorded_ns = reader.read_next()
        if topic == "/ap/sid/control_surface":
            pending_commands.append(
                (recorded_ns, deserialize_message(data, topics[topic]))
            )
            continue
        if topic != "/ap/sid/output":
            continue
        m = deserialize_message(data, topics[topic])
        clock_offsets.append(
            (recorded_ns, int(m.clock_us) - int(m.time_us), int(m.time_us))
        )
        common = dict(
            TimeUS=int(m.time_us),
            Seq=int(m.sequence_id),
            Rev=int(m.configuration_revision),
        )
        yield (
            "SXC",
            dict(
                common,
                Stamp=int(m.clock_us),
                Recv=int(m.time_us),
                En=int(m.state == 2),
                R=float(m.requested[0]),
                P=float(m.requested[1]),
                Y=float(m.requested[2]),
                Valid=1,
            ),
        )
        yield (
            "SXA",
            dict(
                common,
                ClockUS=int(m.clock_us),
                State=int(m.state),
                R=float(m.applied[0]),
                P=float(m.applied[1]),
                Y=float(m.applied[2]),
            ),
        )
        yield (
            "SXS",
            dict(common, State=int(m.state), Reason=int(m.reason), Mode=int(m.mode)),
        )
        yield (
            "ATT",
            dict(
                common,
                **{
                    k: math.degrees(float(v))
                    for k, v in zip(("Roll", "Pitch", "Yaw"), m.attitude)
                },
            ),
        )
        yield (
            "IMU",
            dict(
                common,
                **{k: float(v) for k, v in zip(("GyrX", "GyrY", "GyrZ"), m.rates)},
            ),
        )
        yield "ARSP", dict(common, Airspeed=float(m.airspeed))
        for i in range(min(m.count, 32)):
            if m.function[i] not in (4, 19, 21, 24, 25, 77, 78, 79, 80):
                continue
            yield (
                "SXCF",
                dict(
                    common,
                    Ch=i,
                    Fn=int(m.function[i]),
                    Min=int(m.minimum[i]),
                    Trim=int(m.trim[i]),
                    Max=int(m.maximum[i]),
                    Revrs=int(m.reversed[i]),
                    Gain=float(m.mixing_gain),
                    Offset=float(m.mixing_offset),
                ),
            )
            yield (
                "SXO",
                dict(
                    common,
                    Ch=i,
                    Fn=int(m.function[i]),
                    U=float(m.excitation[i]),
                    Base=int(m.baseline[i]),
                    PWM=int(m.pwm[i]),
                    Delta=int(m.delta[i]),
                    Clip=int(m.clipped[i]),
                    OK=int(m.accepted),
                    BagTimeNS=int(recorded_ns),
                ),
            )
    if clock_offsets:
        import bisect

        times = [v[0] for v in clock_offsets]
        for recorded_ns, m in pending_commands:
            i = max(0, bisect.bisect_right(times, recorded_ns) - 1)
            stamp = m.stamp.sec * 1000000 + m.stamp.nanosec // 1000
            mapped, valid = command_time(stamp, recorded_ns, clock_offsets[i])
            yield (
                "SXR",
                dict(
                    TimeUS=mapped,
                    TimestampValid=valid,
                    Seq=m.sequence_id,
                    En=int(m.enable),
                    R=m.roll,
                    P=m.pitch,
                    Y=m.yaw,
                ),
            )
