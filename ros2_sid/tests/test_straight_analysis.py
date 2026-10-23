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


def test_done_boundary_separates_maneuvers_when_waiting_is_missing() -> None:
    """Lost discovery-time WAITING cannot merge two independently verified attempts."""
    rows = []
    for offset, course in [(0, 20), (10, 90)]:
        config = dict(
            bank_tolerance_deg=5,
            course_tolerance_deg=5,
            altitude_relative_m=100,
            altitude_tolerance_m=5,
            airspeed_m_s=23,
            airspeed_tolerance_m_s=2,
            climb_tolerance_m_s=1,
            rate_tolerance_rad_s=0.15,
            dwell_s=0.25,
        )
        for i, phase in enumerate(
            ("COMMANDING", "SETTLING", "READY", "ACTIVE", "RECOVERY", "DONE")
        ):
            rows.append(
                dict(
                    phase=phase,
                    monotonic_s=offset + i,
                    configuration=config,
                    bank_deg=0,
                    course_deg=course,
                    target_course_deg=course,
                    altitude_relative_m=100,
                    airspeed_m_s=23,
                    climb_m_s=0,
                    rates_rad_s=[0, 0, 0],
                    applied_axes=[0, 0, 0],
                    applied_delta_pwm=[0, 0, 0],
                    vehicle_mode="LOITER" if phase == "DONE" else "GUIDED",
                    recovery_start_location=[1, 2, 100],
                    recovery_target_latlon=[1, 2],
                    reason="LOITER target confirmed at saved start",
                )
            )
    reports = validate_events(rows)
    assert len(reports) == 2
    assert segments_passed(reports)
    # Missing READY remains a failure; boundary handling does not relax entry checks.
    assert not segments_passed(
        validate_events([r for r in rows if r["phase"] != "READY"])
    )
