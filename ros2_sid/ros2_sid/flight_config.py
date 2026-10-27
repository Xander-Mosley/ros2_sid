"""Shared validation for straight-flight settings and RC timeout overrides."""

import argparse
import json
import math
from pathlib import Path


NULLABLE_CHECKS = (
    "course_tolerance_deg",
    "bank_tolerance_deg",
    "rate_tolerance_rad_s",
    "altitude_tolerance_m",
    "airspeed_tolerance_m_s",
    "climb_tolerance_m_s",
    "max_bank_deg",
    "max_altitude_error_m",
    "max_course_error_deg",
    "max_duration_s",
    "min_airspeed_m_s",
    "max_airspeed_m_s",
)


def within_limit(value: float, limit: float | None) -> bool:
    """A null upper bound disables only this comparison."""
    return limit is None or value <= limit


def validate_rc_timeout(value):
    """Accept a disabled freshness check or a finite timeout up to two seconds."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("rc_timeout_s must be null or a number in (0, 2]")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("rc_timeout_s must be null or a number in (0, 2]") from exc
    if not math.isfinite(timeout) or not 0 < timeout <= 2:
        raise ValueError("rc_timeout_s must be null or a number in (0, 2]")
    return timeout


def parse_rc_timeout(value):
    """Parse CLI numbers or the literal null; absence is handled separately."""
    try:
        return validate_rc_timeout(None if value == "null" else value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def resolve_rc_timeout(args, config):
    """Explicit CLI values, including null, take precedence over JSON/defaults."""
    timeout = validate_rc_timeout(
        args.rc_timeout
        if hasattr(args, "rc_timeout")
        else (config.get("rc_timeout_s", 0.5) if config is not None else 0.5)
    )
    args.rc_timeout = timeout
    if config is not None:
        config["rc_timeout_s"] = timeout
    return timeout


def load_straight_flight_config(path):
    """Read and validate settings before constructing ROS endpoints."""
    config = json.loads(Path(path).read_text())
    required = (
        "altitude_relative_m",
        "airspeed_m_s",
        "heading_accel_m_s2",
        "dwell_s",
        "setup_timeout_s",
        "recovery_timeout_s",
        "telemetry_timeout_s",
        "state_timeout_s",
    )
    for key in (*required, *NULLABLE_CHECKS):
        if key not in config:
            raise ValueError(f"Missing required setting: {key}")
        if config[key] is None and key in NULLABLE_CHECKS:
            continue
        try:
            if isinstance(config[key], bool):
                raise ValueError()
            value = float(config[key])
        except (TypeError, ValueError) as exc:
            suffix = " or null" if key in NULLABLE_CHECKS else ""
            raise ValueError(f"{key} must be finite and positive{suffix}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        config[key] = value
    try:
        climb_rate = float(config["climb_rate_m_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("climb_rate_m_s must be finite and nonnegative") from exc
    if not math.isfinite(climb_rate) or climb_rate < 0:
        raise ValueError(
            "climb_rate_m_s must be finite and nonnegative (zero uses firmware maximum)"
        )
    config["climb_rate_m_s"] = climb_rate
    course = config.get("course_deg")
    if course is not None and (not math.isfinite(course) or not 0 <= course < 360):
        raise ValueError("course_deg must be null or in [0,360)")
    if config.get("recovery_mode") != "LOITER":
        raise ValueError("This coordinator requires explicit recovery_mode LOITER")
    if (
        (
            config["min_airspeed_m_s"] is not None
            and config["airspeed_m_s"] <= config["min_airspeed_m_s"]
        )
        or (
            config["max_airspeed_m_s"] is not None
            and config["airspeed_m_s"] >= config["max_airspeed_m_s"]
        )
    ):
        raise ValueError(
            "Target airspeed must lie inside the active airspeed envelope"
        )
    config["rc_timeout_s"] = validate_rc_timeout(config.get("rc_timeout_s", 0.5))
    return config
