"""Plot recorded straight-flight phases and targets directly from rosbag2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check each recorded segment's entry dwell, fixed targets, and zero-before-turn."""
    # Bag storage order can differ from publisher order during clock synchronization.
    # A single coordinator's monotonic event time is authoritative for phase ordering.
    if rows and all("monotonic_s" in row for row in rows):
        rows = sorted(rows, key=lambda row: row["monotonic_s"])
    segments: list[list[dict[str, Any]]] = []
    for row in rows:
        # A new publisher's initial WAITING event may precede DDS discovery.
        # A recorded DONE still closes its attempt; never merge a later maneuver.
        after_done = bool(
            segments and segments[-1][-1]["phase"] == "DONE" and row["phase"] != "DONE"
        )
        if row["phase"] == "WAITING" or not segments or after_done:
            segments.append([])
        segments[-1].append(row)
    reports = []
    for segment in segments:
        final = segment[-1]
        if (
            final["phase"] == "DONE"
            and final.get("reason") == "No navigation ownership"
            and all(r["phase"] in ("RETURNING", "DONE") for r in segment)
        ):
            reports.append(
                dict(
                    passed=False,
                    skipped=True,
                    active_events=0,
                    final_reason=final["reason"],
                    skip_reason="Idle shutdown without a recorded maneuver attempt",
                )
            )
            continue
        active = [r for r in segment if r["phase"] == "ACTIVE"]
        ready = next((r for r in segment if r["phase"] == "READY"), None)
        stable_since = None
        dwell_ok = False
        for row in segment:
            if row["phase"] not in ("SETTLING", "READY"):
                continue
            c = row["configuration"]
            stable = (
                abs(row["bank_deg"]) <= c["bank_tolerance_deg"]
                and abs(
                    (row["course_deg"] - row["target_course_deg"] + 180) % 360 - 180
                )
                <= c["course_tolerance_deg"]
                and abs(row["altitude_relative_m"] - c["altitude_relative_m"])
                <= c["altitude_tolerance_m"]
                and abs(row["airspeed_m_s"] - c["airspeed_m_s"])
                <= c["airspeed_tolerance_m_s"]
                and abs(row["climb_m_s"]) <= c["climb_tolerance_m_s"]
                and row.get("rates_rad_s") is not None
                and max(abs(v) for v in row["rates_rad_s"]) <= c["rate_tolerance_rad_s"]
            )
            if not stable:
                stable_since = None
            elif stable_since is None:
                stable_since = row["monotonic_s"]
            if row["phase"] == "READY":
                # Events are sampled at 10 Hz, unlike the control gate itself.
                dwell_ok = (
                    stable_since is not None
                    and row["monotonic_s"] - stable_since >= c["dwell_s"] - 0.15
                )
                break
        recovery = next((r for r in segment if r["phase"] == "RECOVERY"), None)
        zero_before_turn = bool(
            recovery
            and recovery.get("applied_axes") is not None
            and all(abs(v) < 1e-6 for v in recovery["applied_axes"])
            and all(v == 0 for v in recovery["applied_delta_pwm"])
        )
        final = segment[-1]
        fixed_targets = bool(
            active
            and ready
            and all(
                r["target_course_deg"] == ready["target_course_deg"]
                and r["configuration"] == ready["configuration"]
                for r in active
            )
        )
        loiter = final["phase"] == "DONE" and final.get("vehicle_mode") == "LOITER"
        saved = final.get("recovery_start_location")
        target = final.get("recovery_target_latlon")
        # Historical logs lack this field; do not reinterpret them as start-return evidence.
        start_target_verified = bool(
            saved
            and target
            and all(abs(a - b) < 1e-6 for a, b in zip(saved[:2], target))
            and "target confirmed" in final.get("reason", "")
        )
        recovery_ok = (
            start_target_verified if "recovery_start_location" in final else loiter
        )
        reports.append(
            dict(
                return_to_start_target_verified=start_target_verified,
                passed=bool(active)
                and dwell_ok
                and fixed_targets
                and zero_before_turn
                and loiter
                and recovery_ok,
                active_events=len(active),
                entry_dwell_verified=dwell_ok,
                fixed_targets=fixed_targets,
                zero_before_turn=zero_before_turn,
                loiter_confirmed=loiter,
                final_reason=final["reason"],
            )
        )
    return reports


def segments_passed(reports: list[dict[str, Any]]) -> bool:
    """Require a real evaluated maneuver and success for every evaluated attempt."""
    evaluated = [report for report in reports if not report.get("skipped", False)]
    return bool(evaluated) and all(report["passed"] for report in evaluated)


def analyze_straight(bag: str, output: str) -> dict[str, Any]:
    """Export coordinator events and navigation plots without replaying commands."""
    import matplotlib
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = yaml.safe_load((Path(bag) / "metadata.yaml").read_text())
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=bag,
            storage_id=metadata["rosbag2_bagfile_information"]["storage_identifier"],
        ),
        rosbag2_py.ConverterOptions("", ""),
    )
    rows = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/sid/straight_flight":
            rows.append(json.loads(deserialize_message(data, String).data))
    if not rows:
        raise ValueError("Bag contains no /sid/straight_flight events")
    rows.sort(key=lambda row: row["monotonic_s"])
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "straight_events.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    start = rows[0]["monotonic_s"]
    t = [r["monotonic_s"] - start for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, field, label, target in zip(
        axes,
        ("course_deg", "bank_deg", "altitude_relative_m", "airspeed_m_s"),
        ("Course (deg)", "Bank (deg)", "Altitude above home (m)", "Airspeed (m/s)"),
        ("target_course_deg", None, "altitude_relative_m", "airspeed_m_s"),
    ):
        # WAITING may precede the first navigation sample of a new process.
        # A gap avoids inventing zero altitude or interpolating through recovery.
        ax.plot(
            t,
            [r[field] if r["phase"] != "WAITING" else float("nan") for r in rows],
            label="Observed",
        )
        if target:
            values = [
                r[target]
                if target == "target_course_deg"
                else r["configuration"][target]
                for r in rows
            ]
            values = [
                v if r["phase"] != "WAITING" else float("nan")
                for v, r in zip(values, rows)
            ]
            ax.plot(t, values, "--", label="Fixed target")
        else:
            ax.axhline(0, linestyle="--", color="grey")
        window_start = None
        for row in rows:
            if row["phase"] == "ACTIVE" and window_start is None:
                window_start = row["monotonic_s"] - start
            elif row["phase"] != "ACTIVE" and window_start is not None:
                ax.axvspan(
                    window_start,
                    row["monotonic_s"] - start,
                    color="green",
                    alpha=0.12,
                    linewidth=0,
                )
                window_start = None
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Coordinator monotonic elapsed time (s); green = ACTIVE")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"straight_navigation.{suffix}")
    plt.close(fig)
    points = [r for r in rows if r.get("position_enu_m") is not None]
    fig, ax = plt.subplots(figsize=(8, 8))
    for phase in dict.fromkeys(r["phase"] for r in points):
        selected = [r["position_enu_m"] for r in points if r["phase"] == phase]
        ax.scatter([p[0] for p in selected], [p[1] for p in selected], s=4, label=phase)
    ax.set(
        xlabel="East (m)",
        ylabel="North (m)",
        title="Recorded ground track by phase",
        aspect="equal",
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "straight_track.png")
    plt.close(fig)
    active = [r for r in rows if r["phase"] == "ACTIVE"]
    segments = validate_events(rows)
    report = {
        "passed": segments_passed(segments),
        "segments": segments,
        "event_count": len(rows),
        "active_event_count": len(active),
        "phases": list(dict.fromkeys(r["phase"] for r in rows)),
        "final_reason": rows[-1]["reason"],
        "max_active_bank_deg": max((abs(r["bank_deg"]) for r in active), default=None),
        "max_active_course_error_deg": max(
            (
                abs((r["course_deg"] - r["target_course_deg"] + 180) % 360 - 180)
                for r in active
            ),
            default=None,
        ),
        "note": "Navigation evidence only; actuator validation is reported separately. Does not certify physical flight readiness.",
    }
    (destination / "straight_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
