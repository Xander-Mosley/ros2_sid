"""Analyze acquired air data from a closed rosbag without replaying commands.

FRD velocities and radians are retained in CSV; display plots use degrees.
Simulation truth never substitutes for a missing estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure acquisition cadence per continuous clock/sequence segment."""
    if len(rows) < 2:
        return {"samples": len(rows), "available": False}
    t = np.array([r["time_us"] for r in rows], dtype=np.int64)
    seq = np.array([r["sequence_id"] for r in rows], dtype=np.int64)
    dt = np.diff(t) / 1e6
    ds = (np.diff(seq) + 2**32) % 2**32
    continuous = (dt > 0) & (ds > 0) & (ds < 2**31)
    steps = dt[continuous]
    clock_offsets = np.array(
        [r.get("clock_us", r["time_us"]) - r["time_us"] for r in rows], dtype=np.int64
    )
    clock_jumps = int(np.sum(np.abs(np.diff(clock_offsets)) > 100000))
    return dict(
        samples=len(rows),
        wall_clock_jumps=clock_jumps,
        available=True,
        mean_received_hz=float(1 / steps.mean()) if len(steps) else None,
        sequence_gap_samples=int(np.sum(ds[continuous] - 1)),
        duplicate_samples=int(np.sum(ds == 0)),
        clock_or_sequence_resets=int(np.sum((dt <= 0) | (ds >= 2**31))),
        interval_p50_ms=float(np.median(steps) * 1000) if len(steps) else None,
        interval_p99_ms=float(np.percentile(steps, 99) * 1000) if len(steps) else None,
        valid_fraction=float(np.mean([r["flags"] == 31 for r in rows])),
    )


def signal_delay(
    times: np.ndarray, estimate: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    """Estimate lag over contiguous valid samples; positive means estimate lags truth.

    This is a correlation diagnostic, not calibrated sensor latency. Flat signals
    and gaps cannot establish a delay and are excluded.
    """
    if len(times) < 30:
        return {"available": False}
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return {"available": False}
    candidates = []
    for lag in range(-min(25, int(0.5 / dt)), min(25, int(0.5 / dt)) + 1):
        start = max(0, lag)
        end = min(len(times), len(times) + lag)
        a = estimate[start:end]
        b = truth[start - lag : end - lag]
        continuous = (
            np.abs((times[start:end] - times[start - lag : end - lag]) - lag * dt)
            < dt * 0.5
        )
        a, b = a[continuous], b[continuous]
        if len(a) < 30 or np.std(a) < 1e-4 or np.std(b) < 1e-4:
            continue
        candidates.append((float(np.corrcoef(a, b)[0, 1]), lag))
    if not candidates:
        return {"available": False}
    correlation, lag = max(candidates)
    return {
        "available": True,
        "lag_ms": lag * dt * 1000,
        "correlation": correlation,
        "note": "Bounded ±0.5 s correlation diagnostic; waveform periodicity can make delay ambiguous.",
    }


def analyze_air_data(bag: str | Path, output: str | Path) -> dict[str, Any]:
    """Export air-data samples, timing, paired truth errors and response figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag, output = Path(bag), Path(output) / "air_data"
    if not bag.is_dir() or not (bag / "metadata.yaml").exists():
        return {"available": False, "reason": "Input is not a ROS bag"}
    reader = rosbag2_py.SequentialReader()
    storage = yaml.safe_load((bag / "metadata.yaml").read_text())[
        "rosbag2_bagfile_information"
    ]["storage_identifier"]
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
        rosbag2_py.ConverterOptions("", ""),
    )
    names = {"/ap/air_data", "/ap/air_data/truth", "/ap/sid/output"}
    types = {
        t.name: get_message(t.type)
        for t in reader.get_all_topics_and_types()
        if t.name in names
    }
    if "/ap/air_data" not in types:
        return {
            "available": False,
            "reason": "Bag predates air-data telemetry or topic was unavailable",
        }
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {"estimate": [], "truth": [], "surface": []}
    fields = (
        "time_us",
        "clock_us",
        "sequence_id",
        "flags",
        "rate_hz",
        "source",
        "alpha",
        "beta",
        "u",
        "v",
        "w",
    )
    while reader.has_next():
        topic, raw, receipt = reader.read_next()
        if topic not in types:
            continue
        m = deserialize_message(raw, types[topic])
        if topic == "/ap/sid/output":
            rows["surface"].append(
                (
                    m.time_us,
                    list(m.applied),
                    list(m.rates),
                    int(m.state),
                    list(m.requested),
                )
            )
        else:
            row = {k: getattr(m, k) for k in fields}
            row["bag_receipt_ns"] = receipt
            rows["truth" if topic.endswith("/truth") else "estimate"].append(row)
    report: dict[str, Any] = {
        "available": True,
        "estimate": timing(rows["estimate"]),
        "truth": timing(rows["truth"]),
        "units": "radians; body FRD m/s",
        "limitations": [
            "Acquisition cadence is not independent sensor bandwidth.",
            "Wind estimate may lag or suppress sideslip.",
            "Receipt latency requires a shared wall-clock domain.",
        ],
    }
    for key in ("estimate", "truth"):
        values = rows[key]
        if values:
            with (output / f"{key}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    est = rows["estimate"]
    truth = {
        (r["time_us"], r["sequence_id"]): r for r in rows["truth"] if r["flags"] == 31
    }
    paired = [
        (r, truth[(r["time_us"], r["sequence_id"])])
        for r in est
        if r["flags"] == 31 and (r["time_us"], r["sequence_id"]) in truth
    ]
    surface = rows["surface"]
    surface_times = np.array([r[0] for r in surface])
    active_pairs = []
    for pair in paired:
        i = np.searchsorted(surface_times, pair[0]["time_us"], side="right") - 1
        if (
            i >= 0
            and surface[i][3] == 2
            and pair[0]["time_us"] - surface[i][0] <= 100000
        ):
            active_pairs.append(pair)
    report["active_pairs"] = len(active_pairs)
    for angle in ("alpha", "beta"):
        errors = np.array([a[angle] - b[angle] for a, b in paired])
        report[angle] = {
            "paired_samples": len(errors),
            "bias_deg": float(np.degrees(errors.mean())) if len(errors) else None,
            "rmse_deg": float(np.degrees(np.sqrt(np.mean(errors**2))))
            if len(errors)
            else None,
        }
    report["active_error"] = {}
    for angle in ("alpha", "beta"):
        errors = np.array([a[angle] - b[angle] for a, b in active_pairs])
        report["active_error"][angle] = {
            "rmse_deg": float(np.degrees(np.sqrt(np.mean(errors**2))))
            if len(errors)
            else None,
            "bias_deg": float(np.degrees(errors.mean())) if len(errors) else None,
        }
    report["active_delay"] = {
        angle: signal_delay(
            np.array([a["time_us"] / 1e6 for a, b in active_pairs]),
            np.array([a[angle] for a, b in active_pairs]),
            np.array([b[angle] for a, b in active_pairs]),
        )
        for angle in ("alpha", "beta")
    }
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    origin = est[0]["time_us"] if est else 0
    for key in ("estimate", "truth"):
        valid = rows[key]
        ts = [(r["time_us"] - origin) / 1e6 for r in valid]
        for ax, angle in zip(axes[:2], ("alpha", "beta")):
            ax.plot(
                ts,
                [np.degrees(r[angle]) if r["flags"] == 31 else np.nan for r in valid],
                label=key,
            )
            ax.set_ylabel(angle + " (deg)")
            ax.legend()
    surface = rows["surface"]
    for i, name in enumerate(("roll", "pitch", "yaw")):
        axes[2].plot(
            [(r[0] - origin) / 1e6 for r in surface],
            [r[1][i] for r in surface],
            label=name,
        )
        axes[2].plot(
            [(r[0] - origin) / 1e6 for r in surface],
            [r[4][i] for r in surface],
            linestyle="--",
            linewidth=0.6,
            label=name + " requested",
        )
    axes[2].set_ylabel("Surface excitation")
    axes[2].legend()
    if len(est) > 1:
        axes[3].plot(
            [(r["time_us"] - origin) / 1e6 for r in est[1:]],
            np.diff([r["time_us"] for r in est]) / 1000,
        )
    axes[3].set_ylabel("Sample interval (ms)")
    for angle in ("alpha", "beta"):
        axes[4].plot(
            [(a["time_us"] - origin) / 1e6 for a, b in paired],
            [np.degrees(a[angle] - b[angle]) for a, b in paired],
            label=angle,
        )
    axes[4].set_ylabel("Estimate − truth (deg)")
    axes[4].legend()
    for i, name in enumerate(("p", "q", "r")):
        axes[5].plot(
            [(r[0] - origin) / 1e6 for r in surface],
            [r[2][i] for r in surface],
            label=name,
        )
    axes[5].set_ylabel("Body rate (rad/s)")
    axes[5].legend()
    axes[5].set_xlabel("Firmware elapsed time (s)")
    # Shade actual firmware ACTIVE intervals; other intervals include setup/recovery.
    if surface:
        ts = np.array([(r[0] - origin) / 1e6 for r in surface])
        active = np.array([r[3] == 2 for r in surface])
        for ax in axes:
            ax.fill_between(
                ts,
                0,
                1,
                where=active,
                transform=ax.get_xaxis_transform(),
                color="green",
                alpha=0.08,
            )
    latency = np.array([r["bag_receipt_ns"] / 1e9 - r["clock_us"] / 1e6 for r in est])
    if (
        len(latency)
        and np.all(np.abs(latency) < 60)
        and report["estimate"].get("wall_clock_jumps", 0) == 0
    ):
        report["transport_age_ms"] = {
            "p50": float(np.median(latency) * 1000),
            "p99": float(np.percentile(latency, 99) * 1000),
            "excess_over_median_p99": float(
                (np.percentile(latency, 99) - np.median(latency)) * 1000
            ),
            "note": "Includes host/firmware clock offset; not a one-way latency calibration.",
        }
    else:
        report["transport_age_ms"] = {
            "available": False,
            "reason": "Clock domains not demonstrably aligned",
        }
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Estimated air data and simulation truth"
        if rows["truth"]
        else "Estimated air data — truth unavailable"
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"air_data.{ext}")
    plt.close(fig)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(
        "# Recorded air data\n\n![Air data](air_data.png)\n\n"
        "Angles are estimates, not measured airflow. Conventional beta uses asin(v/V); "
        "SIM_Plane model beta can be reconstructed as atan2(v,u) from the CSV.\n\n"
        "```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )
    return report


def main() -> None:
    """Analyze a saved bag independently of the full excitation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_air_data(args.bag, args.output), indent=2))


if __name__ == "__main__":
    main()
