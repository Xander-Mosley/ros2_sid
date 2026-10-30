"""Replot recorded HITL RC authorization alongside DDS commands and applied axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Read a rosbag without replay and save RC/command/output timing plots."""
    import matplotlib
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rc-source", choices=("mavros", "dds"), default="mavros")
    parser.add_argument("--rc-channel", type=int, default=8)
    parser.add_argument("--rc-threshold", type=int, default=1550)
    args = parser.parse_args()
    if not 1 <= args.rc_channel <= 16:
        parser.error("RC channel must be in [1, 16]")
    rc_topic = "/mavros/rc/in" if args.rc_source == "mavros" else "/ap/rcin"
    topics = {rc_topic, "/ap/sid/control_surface", "/ap/sid/output"}
    metadata = yaml.safe_load((args.bag / "metadata.yaml").read_text())
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(args.bag),
            storage_id=metadata["rosbag2_bagfile_information"]["storage_identifier"],
        ),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {
        t.name: get_message(t.type)
        for t in reader.get_all_topics_and_types()
        if t.name in topics
    }
    rc, commands, outputs = [], [], []
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in types:
            continue
        message = deserialize_message(data, types[topic])
        t = timestamp / 1e9
        if topic == rc_topic:
            values = message.channels if args.rc_source == "mavros" else message.values
            index = args.rc_channel - 1
            valid = index < len(values)
            if args.rc_source == "dds":
                valid = valid and bool(message.valid_mask & (1 << index))
            rc.append((t, int(values[index]) if valid else None))
        elif topic == "/ap/sid/control_surface":
            commands.append((t, bool(message.enable)))
        else:
            outputs.append((t, *(float(value) for value in message.applied)))
    if not rc or not commands or not outputs:
        parser.error(
            "Bag must contain RC, excitation commands and applied output telemetry"
        )
    start = min(rc[0][0], commands[0][0], outputs[0][0])
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].step(
        [t - start for t, _ in rc],
        [v if v is not None else float("nan") for _, v in rc],
        where="post",
    )
    axes[0].axhline(
        args.rc_threshold, linestyle="--", color="red", label="Run threshold"
    )
    axes[0].set_ylabel(f"RC {args.rc_channel} (µs)")
    axes[0].legend()
    axes[1].step(
        [t - start for t, _ in commands], [int(v) for _, v in commands], where="post"
    )
    axes[1].set_ylabel("DDS enable")
    for index, axis in enumerate(("roll", "pitch", "yaw"), start=1):
        axes[2].plot(
            [row[0] - start for row in outputs],
            [row[index] for row in outputs],
            label=axis,
        )
    axes[2].set_ylabel("Applied normalized axes")
    axes[2].set_xlabel("Bag receipt elapsed time (s)")
    axes[2].legend()
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"hitl_rc_timing.{suffix}")
    plt.close(fig)
    (args.output / "hitl_rc_samples.json").write_text(
        json.dumps(
            {
                "rc": rc,
                "commands": commands,
                "applied": outputs,
                "note": "Receipt timestamps include transport latency; plots alone are not safety acceptance.",
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
