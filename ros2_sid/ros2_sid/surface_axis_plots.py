"""Render individual-axis SITL command/response figures from recorded ROS bags."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .surface_bag import records

AXES = ("roll", "pitch", "yaw")
FIELDS = ("R", "P", "Y")
FUNCTIONS = {"roll": 4, "pitch": 19, "yaw": 21}


def render(run: Path, output: Path) -> dict[str, Any]:
    """Save six PNG/PDF plots and verify that other logical axes stayed unexcited.

    Command time comes from publisher AP-clock stamps. Applied outputs and aircraft
    response come from coherent firmware snapshots in the ROS bag. Attitude is
    unwrapped and expressed relative to the plotted window's first sample; the
    normal GUIDED response has not been subtracted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation = json.loads((run / "plots/report.json").read_text())
    if not validation["passed"] or len(validation.get("maneuvers", [])) != 6:
        raise ValueError("Six validated single-axis maneuvers are required")
    data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kind, row in records(run / "rosbag"):
        if kind in ("SXA", "SXO", "ATT", "IMU"):
            data[kind].append(row)
    for series in data.values():
        series.sort(key=lambda row: row["TimeUS"])
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"source_bag": str(run / "rosbag"), "experiments": []}
    for axis_index, axis in enumerate(AXES):
        for waveform in ("chirp", "multisine"):
            name = f"{waveform}_{axis}"
            spec = json.loads((run / f"{name}.json").read_text())
            if set(spec["axes"]) != {axis}:
                raise ValueError(f"{name}: experiment is not single-axis")
            with (run / f"{name}-commands.csv").open() as stream:
                commands = [row for row in csv.DictReader(stream) if int(row["enable"])]
            if not commands or any(
                float(row[other]) != 0
                for row in commands
                for other in AXES
                if other != axis
            ):
                raise ValueError(
                    f"{name}: missing commands or unexpected other-axis input"
                )
            start = float(commands[0]["stamp_s"])
            end = float(commands[-1]["stamp_s"])
            frames = [
                row
                for row in data["SXA"]
                if start - 0.5 <= row["ClockUS"] / 1e6 <= end + 1
            ]
            times = {row["TimeUS"]: row["ClockUS"] / 1e6 - start for row in frames}
            active = [row for row in frames if start <= row["ClockUS"] / 1e6 <= end]
            if not active:
                raise ValueError(f"{name}: no firmware output frames")
            other_axis_max = max(
                abs(row[field])
                for row in active
                for index, field in enumerate(FIELDS)
                if index != axis_index
            )
            if other_axis_max > 1e-6:
                raise ValueError(f"{name}: another logical axis was excited")
            fig, panels = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
            command_times = [float(row["stamp_s"]) - start for row in commands]
            panels[0].plot(
                command_times,
                [float(row[axis]) for row in commands],
                label=f"Requested {axis}",
                linewidth=1.3,
            )
            panels[0].plot(
                [times[row["TimeUS"]] for row in frames],
                [row[FIELDS[axis_index]] for row in frames],
                "--",
                label="Firmware applied logical excitation",
                linewidth=1.1,
            )
            panels[0].set_ylabel("Normalized excitation")
            physical = [row for row in data["SXO"] if row["TimeUS"] in times]
            other_pwm_max = max(
                (abs(row["Delta"]) for row in physical if row["Fn"] != FUNCTIONS[axis]),
                default=0,
            )
            if other_pwm_max:
                raise ValueError(f"{name}: another surface has a nonzero perturbation")
            for surface, function in FUNCTIONS.items():
                rows = [row for row in physical if row["Fn"] == function]
                panels[1].plot(
                    [times[row["TimeUS"]] for row in rows],
                    [row["Delta"] for row in rows],
                    label=f"{surface}: final − baseline",
                )
            panels[1].set_ylabel("Applied ΔPWM (µs)")
            rate_rows = [row for row in data["IMU"] if row["TimeUS"] in times]
            for field, label in zip(
                ("GyrX", "GyrY", "GyrZ"), ("p (roll)", "q (pitch)", "r (yaw)")
            ):
                panels[2].plot(
                    [times[row["TimeUS"]] for row in rate_rows],
                    [row[field] for row in rate_rows],
                    label=label,
                )
            panels[2].set_ylabel("Body rate (rad/s)")
            attitude_rows = [row for row in data["ATT"] if row["TimeUS"] in times]
            if not attitude_rows or not rate_rows:
                raise ValueError(f"{name}: aircraft response data is missing")
            for field in (axis.capitalize(),):
                angles = np.rad2deg(
                    np.unwrap(np.deg2rad([row[field] for row in attitude_rows]))
                )
                panels[3].plot(
                    [times[row["TimeUS"]] for row in attitude_rows],
                    angles - angles[0],
                    label=field,
                )
            panels[3].set_ylabel(f"{axis.capitalize()} change (deg)")
            panels[3].set_xlabel("Time since first enabled command (s)")
            for panel in panels:
                panel.axvline(0, color="black", alpha=0.3, linewidth=0.8)
                panel.axvline(end - start, color="black", alpha=0.3, linewidth=0.8)
                panel.grid(alpha=0.25)
                panel.legend(loc="upper right", fontsize=9)
            fig.suptitle(
                f"ArduPlane SITL — {axis.capitalize()} {waveform} — {spec['rate']:g} Hz publication"
            )
            fig.text(
                0.5,
                0.01,
                "Only the named axis is commanded. GUIDED remains active; response includes nominal motion and cross-axis coupling.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.025, 1, 0.97))
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"{name}.{suffix}", dpi=150)
            plt.close(fig)
            summary["experiments"].append(
                dict(
                    name=name,
                    rate=spec["rate"],
                    duration=end - start,
                    output_frames=len(active),
                    other_axis_max=other_axis_max,
                    other_surface_delta_max=other_pwm_max,
                    figure=f"{name}.png",
                )
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    """Render report figures from a completed individual-axis SITL run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
