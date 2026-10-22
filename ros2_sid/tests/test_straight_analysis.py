"""Regression checks for idle shutdowns and incomplete navigation recordings."""

from ros2_sid.straight_analysis import segments_passed, validate_events


def test_idle_shutdown_is_skipped_but_cannot_pass_alone() -> None:
    """An idle publisher shutdown supplies no evidence of a successful maneuver."""
    rows = [
        {"phase": "RETURNING", "reason": "Stopping"},
        {"phase": "DONE", "reason": "No navigation ownership"},
    ]
    reports = validate_events(rows)
    assert reports[0]["skipped"]
    assert not segments_passed(reports)
    assert segments_passed(reports + [{"passed": True}])
    assert not segments_passed(reports + [{"passed": False}])
    assert not segments_passed([])


def test_started_attempt_is_not_skipped() -> None:
    """An attempt that entered WAITING still fails if it never became active."""
    rows = [
        {"phase": "WAITING", "reason": "Waiting"},
        {"phase": "RETURNING", "reason": "Stopping"},
        {"phase": "DONE", "reason": "No navigation ownership"},
    ]
    reports = validate_events(rows)
    assert not reports[0].get("skipped", False)
    assert not segments_passed(reports)


def test_incomplete_shutdown_is_not_skipped() -> None:
    """A truncated recovery cannot be treated as a confirmed idle shutdown."""
    reports = validate_events([{"phase": "RETURNING", "reason": "Stopping"}])
    assert not reports[0].get("skipped", False)
    assert not segments_passed(reports)
