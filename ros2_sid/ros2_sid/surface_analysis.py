"""Reconstruct and verify calibrated excitation from DataFlash; render standalone figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def expected_pwm(
    u: float, baseline: int, minimum: int, trim: int, maximum: int, reversed_servo: bool
) -> int:
    """Calculate calibrated additive PWM in microseconds, preserving the zero-output path."""
    # ROS fixed-size arrays contain NumPy unsigned scalars. Convert before
    # subtraction/addition so negative perturbations cannot wrap or overflow.
    u = float(u)
    baseline, minimum, trim, maximum = map(int, (baseline, minimum, trim, maximum))
    if u == 0:
        return baseline  # Disabled/zero path preserves normal output, including PWM=0.
    signed = -u if reversed_servo else u
    # ArduPilot truncates the magnitude of the trim-relative integer delta.
    delta = int(abs(signed) * (maximum - trim if signed > 0 else trim - minimum))
    if signed < 0:
        delta = -delta
    return min(maximum, max(minimum, baseline + delta))


def records(source: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield typed record names and heterogeneous fields from a ROS bag or DataFlash log."""
    if Path(source).is_dir():
        from .surface_bag import records as bag_records

        yield from bag_records(source)
    else:
        from pymavlink import mavutil

        connection = mavutil.mavlink_connection(str(source))
        try:
            while True:
                msg = connection.recv_match()
                if msg is None:
                    break
                yield msg.get_type(), msg.to_dict()
        finally:
            connection.close()


def analyze(log_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Validate physical PWM and export timestamp-aligned response CSV, figures, and a report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {}
    actuators = []
    commands = []
    raw_commands = []
    axes = []
    attitude = []
    rates = []
    airspeed = []
    states = []
    for kind, d in records(log_path):
        if kind == "SXCF":
            config[(d["Rev"], d["Ch"])] = d
        elif kind == "SXO":
            actuators.append(d)
        elif kind == "SXC":
            commands.append(d)
        elif kind == "SXR":
            raw_commands.append(d)
        elif kind == "SXA":
            axes.append(d)
        elif kind == "SXS":
            states.append(d)
        elif kind == "ATT":
            attitude.append(d)
        elif kind == "IMU" and d.get("I", 0) == 0:
            rates.append(d)
        elif kind == "ARSP":
            airspeed.append(d)
    if raw_commands:
        commands = raw_commands
    # Best-effort DDS reception order is not firmware sample order. Sorting is
    # essential both for line plots and for interpolation/spectral analysis.
    for series in (actuators, commands, axes, attitude, rates, airspeed, states):
        series.sort(key=lambda d: d["TimeUS"])
    if not actuators or not commands or not axes:
        raise ValueError(
            "Missing required SXC/SXA/SXO records; cannot validate excitation"
        )
    origin = min(x["TimeUS"] for x in actuators)
    rows = []
    missing = 0
    for d in actuators:
        c = config.get((d["Rev"], d["Ch"]))
        if c is None:
            missing += 1
            continue
        expected = (
            expected_pwm(d["U"], d["Base"], c["Min"], c["Trim"], c["Max"], c["Revrs"])
            if d["OK"]
            else d["Base"]
        )
        span = (c["Max"] - c["Trim"]) if d["Delta"] >= 0 else (c["Trim"] - c["Min"])
        applied_normalized = (
            (d["Delta"] / span) * (-1 if c["Revrs"] else 1) if span > 0 else 0.0
        )
        rows.append(
            dict(
                applied_normalized=applied_normalized,
                time_s=(d["TimeUS"] - origin) / 1e6,
                sequence=d["Seq"],
                channel=d["Ch"] + 1,
                function=d["Fn"],
                normalized=d["U"],
                baseline_pwm=d["Base"],
                expected_pwm=expected,
                final_pwm=d["PWM"],
                delta_pwm=d["Delta"],
                error_pwm=d["PWM"] - expected,
                minimum=c["Min"],
                maximum=c["Max"],
                clipped=d["Clip"],
                accepted=d["OK"],
            )
        )
    if not rows:
        raise ValueError("No reconstructable actuator samples")
    with (output / "actuators.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def save(fig: Figure, name: str) -> None:
        """Save a standalone PNG/PDF pair and release the matplotlib figure."""
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=160)
        fig.savefig(output / (name + ".pdf"))
        plt.close(fig)

    channels = sorted(set(r["channel"] for r in rows))
    fig, axs = plt.subplots(
        len(channels), 2, figsize=(13, 3 * len(channels)), squeeze=False
    )
    for i, ch in enumerate(channels):
        rr = [r for r in rows if r["channel"] == ch]
        t = [r["time_s"] for r in rr]
        for key, label in [
            ("baseline_pwm", "Baseline"),
            ("expected_pwm", "Expected"),
            ("final_pwm", "Recorded"),
        ]:
            axs[i, 0].plot(t, [r[key] for r in rr], label=label, linewidth=1)
        axs[i, 0].plot(t, [r["minimum"] for r in rr], "k:", alpha=0.5)
        axs[i, 0].plot(t, [r["maximum"] for r in rr], "k:", alpha=0.5)
        axs[i, 0].set_ylabel(f"SERVO{ch} PWM (µs)")
        axs[i, 0].legend()
        axs[i, 1].plot(t, [r["error_pwm"] for r in rr], label="Recorded − expected")
        axs[i, 1].axhspan(-1, 1, color="green", alpha=0.15)
        axs[i, 1].set_ylabel("Error (µs)")
        axs[i, 1].legend()
        for ax in axs[i]:
            ax.set_xlabel("Firmware time relative to first output record (s)")
            ax.grid(alpha=0.2)
    save(fig, "actuator_validation")
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    labels = ("Roll", "Pitch", "Yaw")
    for i, key in enumerate(("R", "P", "Y")):
        axs[i].plot(
            [(d["TimeUS"] - origin) / 1e6 for d in commands],
            [d[key] for d in commands],
            label="Requested",
        )
        axs[i].plot(
            [(d["TimeUS"] - origin) / 1e6 for d in axes],
            [d[key] for d in axes],
            label="Firmware logical applied",
        )
        axs[i].set_ylabel(labels[i] + " fraction")
        axs[i].legend()
        axs[i].grid(alpha=0.2)
    axs[3].step(
        [(d["TimeUS"] - origin) / 1e6 for d in states],
        [d["State"] for d in states],
        where="post",
    )
    axs[3].set_ylabel("State code")
    axs[3].set_xlabel("Firmware time (s)")
    save(fig, "excitation_tracking")
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ch in channels:
        rr = [r for r in rows if r["channel"] == ch]
        axs[0].plot(
            [r["time_s"] for r in rr], [r["delta_pwm"] for r in rr], label=f"SERVO{ch}"
        )
    axs[0].set_ylabel("Applied ΔPWM (µs)")
    for field in ("Roll", "Pitch", "Yaw"):
        if attitude:
            axs[1].plot(
                [(d["TimeUS"] - origin) / 1e6 for d in attitude],
                [d[field] for d in attitude],
                label=field,
            )
    axs[1].set_ylabel("Attitude (deg)")
    for field in ("GyrX", "GyrY", "GyrZ"):
        if rates:
            axs[2].plot(
                [(d["TimeUS"] - origin) / 1e6 for d in rates],
                [d[field] for d in rates],
                label=field,
            )
    axs[2].set_ylabel("Angular rate (rad/s)")
    if airspeed:
        axs[3].plot(
            [(d["TimeUS"] - origin) / 1e6 for d in airspeed],
            [d["Airspeed"] for d in airspeed],
            label="Airspeed",
        )
    axs[3].set_ylabel("Airspeed (m/s)")
    axs[3].set_xlabel("Firmware time (s)")
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, "aircraft_response")
    # Align response to actuator sample time without extrapolating missing data.
    sample_times = np.array([r["time_s"] for r in rows])
    for response_records, fields in [
        (attitude, ("Roll", "Pitch", "Yaw")),
        (rates, ("GyrX", "GyrY", "GyrZ")),
        (airspeed, ("Airspeed",)),
    ]:
        ts = np.array([(d["TimeUS"] - origin) / 1e6 for d in response_records])
        for field in fields:
            values = (
                np.interp(
                    sample_times,
                    ts,
                    [d[field] for d in response_records],
                    left=np.nan,
                    right=np.nan,
                )
                if len(ts)
                else np.full(len(rows), np.nan)
            )
            for r, value in zip(rows, values):
                r[field] = float(value)
    with (output / "aligned_response.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    spectral = []
    from scipy.signal import coherence, csd, welch

    for ch in channels:
        rr = [r for r in rows if r["channel"] == ch]
        if not any(abs(r["delta_pwm"]) > 0 for r in rr):
            continue
        if len(rr) < 128:
            continue
        t = np.array([r["time_s"] for r in rr])
        dt = np.median(np.diff(t))
        if dt <= 0:
            continue
        grid = np.arange(t[0], t[-1], dt)
        u = np.interp(grid, t, [r["delta_pwm"] for r in rr])
        # Each actuator to each rate: closed-loop descriptive estimate, not an independent MIMO identification.
        fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        for field in ("GyrX", "GyrY", "GyrZ"):
            y = np.interp(grid, t, [r[field] for r in rr])
            if not np.all(np.isfinite(y)):
                continue
            n = min(256, len(grid) // 2)
            f, px = welch(u, 1 / dt, nperseg=n)
            _, xy = csd(u, y, 1 / dt, nperseg=n)
            _, coh = coherence(u, y, 1 / dt, nperseg=n)
            valid = (f > 0) & (px > max(px.max() * 1e-6, 1e-20))
            h = xy[valid] / px[valid]
            axs[3].loglog(
                f[valid], px[valid], label="Input PSD" if field == "GyrX" else None
            )
            fy, py = welch(y, 1 / dt, nperseg=n)
            axs[3].loglog(fy[fy > 0], py[fy > 0], label=field + " PSD")
            axs[0].semilogx(
                f[valid], 20 * np.log10(np.maximum(abs(h), 1e-20)), label=field
            )
            axs[1].semilogx(f[valid], np.angle(h, deg=True), label=field)
            axs[2].semilogx(f[valid], coh[valid], label=field)
        for ax, label in zip(
            axs,
            (
                "Gain (dB, rad/s per µs)",
                "Phase (deg)",
                "Coherence",
                "PSD (input µs²/Hz; rate (rad/s)²/Hz)",
            ),
        ):
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.2)
        axs[-1].set_xlabel("Frequency (Hz)")
        fig.suptitle(
            f"SERVO{ch}: closed-loop descriptive estimate; correlated inputs may bias results"
        )
        save(fig, f"frequency_response_servo{ch}")
        spectral.append(ch)
    report = dict(
        raw_command_samples=len(raw_commands),
        source_kind="rosbag2" if Path(log_path).is_dir() else "DataFlash",
        samples=len(rows),
        missing_configuration=missing,
        commands_using_receipt_time=sum(
            not d.get("TimestampValid", True) for d in raw_commands
        ),
        max_pwm_error=max(abs(r["error_pwm"]) for r in rows),
        nonzero_applied_samples=sum(abs(r["delta_pwm"]) > 0 for r in rows),
        attitude_samples=len(attitude),
        rate_samples=len(rates),
        airspeed_samples=len(airspeed),
        spectral_channels=spectral,
    )
    report["passed"] = bool(
        not missing
        and report["max_pwm_error"] <= 1
        and report["nonzero_applied_samples"] > 0
        and attitude
        and rates
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare_outputs(bag: str | Path, dataflash: str | Path) -> dict[str, int | bool]:
    """Match DDS samples to authoritative DataFlash by boot time and channel."""
    fields = ("Seq", "Rev", "Base", "Delta", "PWM", "OK")
    logged = {
        (d["TimeUS"], d["Ch"]): d for kind, d in records(dataflash) if kind == "SXO"
    }
    matched = missing = mismatched = 0
    for kind, d in records(bag):
        if kind != "SXO":
            continue
        other = logged.get((d["TimeUS"], d["Ch"]))
        if other is None:
            missing += 1
            continue
        matched += 1
        mismatched += any(d[f] != other[f] for f in fields)
    return dict(
        matched=matched,
        missing=missing,
        mismatched=mismatched,
        passed=matched > 0 and missing == 0 and mismatched == 0,
    )


def validate_maneuver(bag: str | Path, command_log: str | Path) -> dict[str, Any]:
    """Require observed activation across the waveform, not just publisher success.

    Intended for the supplied bounded examples with active slew disabled. Logical
    axes must match the accepted command sequence; physical mixing is checked by
    the separate calibrated-PWM validation.
    """
    with open(command_log) as stream:
        commands = [r for r in csv.DictReader(stream) if int(r["enable"])]
    if not commands:
        return dict(passed=False, reason="No enabled commands")
    start, end = float(commands[0]["stamp_s"]), float(commands[-1]["stamp_s"])
    by_sequence = {int(r["sequence"]): r for r in commands}
    frames = [
        d
        for kind, d in records(bag)
        if kind == "SXA" and start <= d["ClockUS"] / 1e6 <= end
    ]
    active = [d for d in frames if d["State"] == 2]
    times = sorted(d["ClockUS"] for d in active)
    max_gap = max((b - a for a, b in zip(times, times[1:])), default=0) / 1e6
    span = (
        (max(d["ClockUS"] for d in active) - min(d["ClockUS"] for d in active)) / 1e6
        if active
        else 0
    )
    mismatched = 0
    for d in active:
        command = by_sequence.get(d["Seq"])
        if command is None or any(
            abs(d[field] - float(command[axis])) > 1e-6
            for field, axis in zip(("R", "P", "Y"), ("roll", "pitch", "yaw"))
        ):
            mismatched += 1
    fraction = len(active) / len(frames) if frames else 0
    return dict(
        command_log=str(command_log),
        duration=end - start,
        active_span=span,
        active_fraction=fraction,
        max_gap=max_gap,
        frames=len(frames),
        mismatched=mismatched,
        passed=bool(
            end > start
            and span >= 0.9 * (end - start)
            and fraction >= 0.95
            and max_gap <= 0.2
            and not mismatched
        ),
    )


def main() -> int:
    """Run offline analysis and return a failing exit status when any requested check fails."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="rosbag2 directory (preferred), or DataFlash .BIN")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--compare-dataflash",
        help="Cross-check bag actuator samples against this firmware .BIN",
    )
    p.add_argument(
        "--command-log",
        action="append",
        default=[],
        help="Validate full maneuver application against publisher CSV (repeatable)",
    )
    p.add_argument(
        "--straight-flight",
        action="store_true",
        help="Also export coordinator events and navigation plots from the bag",
    )
    a = p.parse_args()
    if not a.log.strip():
        p.error("Input path is empty. Set BAG to the recorded bag directory first.")
    report = analyze(a.log, a.output)
    if a.straight_flight:
        from .straight_analysis import analyze_straight

        report["straight_flight"] = analyze_straight(a.log, a.output)
        report["passed"] = report["passed"] and report["straight_flight"]["passed"]
    if a.compare_dataflash:
        report["dataflash_comparison"] = compare_outputs(a.log, a.compare_dataflash)
        report["passed"] = report["passed"] and report["dataflash_comparison"]["passed"]
    if a.command_log:
        report["maneuvers"] = [validate_maneuver(a.log, path) for path in a.command_log]
        report["passed"] = report["passed"] and all(
            r["passed"] for r in report["maneuvers"]
        )
    (Path(a.output) / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1
