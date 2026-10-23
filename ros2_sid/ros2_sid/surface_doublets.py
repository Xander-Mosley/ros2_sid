"""Generate normalized DDS doublets using the unchanged legacy pulse generator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ros2_sid.input_designs import multi_step


def generate_doublets(destination: Path, amplitude: float = 0.1) -> None:
    """Append three 50 Hz doublets, preserving existing mission menu ordering.

    Legacy defaults: 1 Hz natural frequency, one-second delay, 15-second
    duration, and pulses [1, 1]. The original sample-boundary zero gaps remain.
    Amplitude is a surface fraction, not the legacy angle in radians.
    """
    if not 0 < amplitude <= 1:
        raise ValueError("Amplitude must be in (0, 1]")
    destination.mkdir(parents=True, exist_ok=True)
    time, signal = multi_step(amplitude, 1.0, [1, 1], 1.0, 0.02, 15.0)
    plan_path = destination / "mission_plan.json"
    plan = (
        json.loads(plan_path.read_text())
        if plan_path.exists()
        else {"name": "dds_surface", "units": "normalized_surface", "maneuvers": []}
    )
    if plan.get("units") != "normalized_surface":
        raise ValueError("Refusing to change a legacy angle/rate library")
    for index, axis in enumerate(("roll", "pitch", "yaw")):
        filename = f"doublet_{axis}.csv"
        with (destination / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time", "roll", "pitch", "yaw"))
            for t, value in zip(time, signal):
                axes = [0.0, 0.0, 0.0]
                axes[index] = float(value)
                writer.writerow((f"{t:.6f}", *axes))
        if not any(m["filename"] == filename for m in plan["maneuvers"]):
            order = max((m["order"] for m in plan["maneuvers"]), default=0) + 1
            plan["maneuvers"].append(
                {"filename": filename, "name": f"{axis} doublet", "order": order}
            )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")


def main() -> None:
    """Generate a library in an explicitly selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--amplitude", type=float, default=0.1)
    args = parser.parse_args()
    generate_doublets(args.destination, args.amplitude)


if __name__ == "__main__":
    main()
