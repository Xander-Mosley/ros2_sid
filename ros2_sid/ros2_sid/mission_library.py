"""Locate the shared legacy/DDS mission library in source and installed packages."""

import math
from pathlib import Path


def mission_amplitude_modifier(selection: dict) -> float:
    """Read the mission scale shared by the SITL, HITL, and DDS backends."""
    modifier = selection.get("amplitude_modifier")
    if not isinstance(modifier, (int, float)) or isinstance(modifier, bool):
        modifier = 1.0
    modifier = float(modifier)
    if not math.isfinite(modifier):
        raise ValueError("amplitude_modifier must be finite")
    return modifier


def mission_library_path() -> Path:
    """Return the source library when present, otherwise the installed share copy."""
    source = Path(__file__).resolve().parent / "setup/mission_library"
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("ros2_sid")) / "ros2_sid/setup/mission_library"
    )
