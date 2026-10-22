"""Generate individual-axis maneuver figures from rosbag2 without replaying commands."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .surface_bag import records


def plot_maneuvers(
    bag: Path, command_directory: Path, output: Path
) -> list[dict[str, Any]]:
    """Plot six conventional-airframe maneuvers using recorded firmware timestamps."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = {}
    for kind, row in records(bag):
        grouped.setdefault(kind, []).append(row)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for axis, field, function in (
        ("roll", "R", 4),
        ("pitch", "P", 19),
        ("yaw", "Y", 21),
    ):
        for waveform in ("chirp", "multisine"):
            name = f"{waveform}_{axis}"
            with (command_directory / f"{name}-commands.csv").open() as stream:
                commands = [r for r in csv.DictReader(stream) if int(r["enable"])]
            if not commands:
                raise ValueError(f"No enabled commands for {name}")
            start, end = (float(commands[i]["stamp_s"]) * 1e6 for i in (0, -1))
            reference = min(grouped["SXA"], key=lambda r: abs(r["ClockUS"] - start))
            origin = reference["TimeUS"] + start - reference["ClockUS"]
            duration = (end - start) / 1e6

            def window(kind: str) -> list[dict[str, Any]]:
                return sorted(
                    [
                        r
                        for r in grouped[kind]
                        if -1 <= (r["TimeUS"] - origin) / 1e6 <= duration + 0.5
                    ],
                    key=lambda r: r["TimeUS"],
                )

            def times(rows: list[dict[str, Any]]) -> list[float]:
                return [(r["TimeUS"] - origin) / 1e6 for r in rows]

            requested, applied = window("SXR"), window("SXA")
            surfaces = [r for r in window("SXO") if r["Fn"] == function]
            if not surfaces:
                raise ValueError(
                    f"No conventional {axis} actuator; mixed-surface plotting requires function selection"
                )
            attitude, rates, airspeed = window("ATT"), window("IMU"), window("ARSP")
            fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
            axes[0].plot(
                times(requested),
                [r[field] for r in requested],
                label="DDS requested",
                color="black",
                linewidth=1,
            )
            axes[0].plot(
                times(applied),
                [r[field] for r in applied],
                label="Applied logical excitation",
                color="tab:orange",
                linewidth=1,
            )
            axes[0].set_ylabel("Normalized excitation")
            for channel in sorted({r["Ch"] for r in surfaces}):
                samples = [r for r in surfaces if r["Ch"] == channel]
                axes[1].plot(
                    times(samples),
                    [r["Base"] for r in samples],
                    "--",
                    label=f"SERVO{channel + 1} baseline",
                )
                axes[1].plot(
                    times(samples),
                    [r["PWM"] for r in samples],
                    label=f"SERVO{channel + 1} final",
                )
            axes[1].set_ylabel("Commanded PWM (µs)")
            for component, label in zip(
                ("GyrX", "GyrY", "GyrZ"), ("p: roll", "q: pitch", "r: yaw")
            ):
                axes[2].plot(times(rates), [r[component] for r in rates], label=label)
            axes[2].set_ylabel("Angular rate (rad/s)")
            for component in ("Roll", "Pitch", "Yaw"):
                values = np.degrees(
                    np.unwrap(np.radians([r[component] for r in attitude]))
                )
                baseline = np.interp(0, times(attitude), values)
                axes[3].plot(times(attitude), values - baseline, label=component)
            axes[3].set_ylabel("Attitude change (deg)\nfrom maneuver start")
            axes[4].plot(
                times(airspeed), [r["Airspeed"] for r in airspeed], label="Airspeed"
            )
            axes[4].set_ylabel("Airspeed (m/s)")
            for ax in axes:
                ax.axvspan(0, duration, color="green", alpha=0.07, linewidth=0)
                ax.axvline(0, color="grey", linestyle=":")
                ax.axvline(duration, color="grey", linestyle=":")
                ax.grid(alpha=0.2)
                ax.legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel(
                "Time from first enabled command (s); shaded = commanded maneuver"
            )
            title = "Frequency sweep (chirp)" if waveform == "chirp" else "Multisine"
            fig.suptitle(
                f"{axis.title()} — {title}\nRecorded straight-flight SITL; normal controllers remain active"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{name}.{extension}", dpi=160)
            plt.close(fig)
            summary.append(
                dict(
                    maneuver=name,
                    commanded_duration_s=duration,
                    output_frames=len(applied),
                    physical_channels=sorted({r["Ch"] + 1 for r in surfaces}),
                    png=f"{name}.png",
                    pdf=f"{name}.pdf",
                )
            )
    (output / "manifest.json").write_text(
        json.dumps(dict(bag=str(bag), maneuvers=summary), indent=2) + "\n"
    )
    return summary


def main() -> None:
    """Run offline plotting against a bag and its six publisher command logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--command-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            plot_maneuvers(args.bag, args.command_directory, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
