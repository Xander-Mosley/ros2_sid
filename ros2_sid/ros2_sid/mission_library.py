"""Locate the shared legacy/DDS mission library in source and installed packages."""

from pathlib import Path


def mission_library_path() -> Path:
    """Return the source library when present, otherwise the installed share copy."""
    source = Path(__file__).resolve().parent / "setup/mission_library"
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("ros2_sid")) / "ros2_sid/setup/mission_library"
    )
