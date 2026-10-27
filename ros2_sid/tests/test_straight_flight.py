"""Deterministic navigation coordination and shutdown tests; no vehicle commands."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ros2_sid.straight_flight import FlightPhase, StraightFlight, course_error


@pytest.fixture
def coordinator(tmp_path):
    """Construct real adapter with mocked ROS endpoints and fresh flight telemetry."""
    node = Mock()
    node.create_client.return_value = Mock()
    path = (
        Path(__file__).parents[1]
        / "config/surface_excitation/straight_flight_sitl.json"
    )
    # Keep deterministic test conditions independent of operator-edited settings.
    config = json.loads(path.read_text())
    config.update(
        course_deg=None,
        airspeed_m_s=18,
        min_airspeed_m_s=10,
        max_airspeed_m_s=30,
        altitude_relative_m=100,
        heading_accel_m_s2=2,
        climb_rate_m_s=0,
        dwell_s=3,
        setup_timeout_s=90,
        recovery_timeout_s=8,
        telemetry_timeout_s=1,
        state_timeout_s=3,
        course_tolerance_deg=5,
        bank_tolerance_deg=5,
        rate_tolerance_rad_s=0.15,
        altitude_tolerance_m=5,
        airspeed_tolerance_m_s=2,
        climb_tolerance_m_s=1.5,
        max_bank_deg=35,
        max_altitude_error_m=30,
        max_course_error_deg=40,
        max_duration_s=30,
        recovery_mode="LOITER",
    )
    path = tmp_path / "test-settings.json"
    path.write_text(json.dumps(config))
    flight = StraightFlight(node, str(path), str(tmp_path / "events.jsonl"))
    flight.command_client = Mock()
    flight.loiter_client = Mock()
    flight.command_client.service_is_ready.return_value = True
    flight.loiter_client.service_is_ready.return_value = True
    future = Mock()
    future.done.return_value = True
    future.result.return_value = SimpleNamespace(success=True, result=0, mode_sent=True)
    flight.command_client.call_async.return_value = future
    flight.loiter_client.call_async.return_value = future
    flight.state = SimpleNamespace(
        armed=True, connected=True, mode="GUIDED", system_status=4
    )
    flight.output = SimpleNamespace(
        clock_us=1,
        attitude=[0, 0, 0],
        rates=[0, 0, 0],
        airspeed=18,
        applied=[0, 0, 0],
        delta=[0],
        count=1,
    )
    flight.global_position = (35.0, -106.0)
    flight.start_location = (35.0, -106.0, 100.0)
    flight.altitude = 100
    flight.course = 359
    flight.speed = 18
    yield flight
    flight.event_file.close()


def fresh(flight, now):
    """Refresh all receipts while advancing the test clock."""
    flight.state_time = flight.output_time = flight.velocity_time = (
        flight.altitude_time
    ) = now
    flight.global_position_time = now


def test_settling_commands_once_and_wrap(coordinator):
    """Commands are sequential, target course is captured, and dwell is continuous."""
    f = coordinator
    for t in range(5):
        fresh(f, t)
        assert not f.step(t, True, False, 10)
    assert f.phase == FlightPhase.SETTLING
    f.output.attitude[0] = 0.3
    fresh(f, 5)
    assert not f.step(5, True, False, 10)
    f.output.attitude[0] = 0
    for t in range(6, 10):
        fresh(f, t)
        ready = f.step(t, True, False, 10)
    assert ready
    assert f.command_client.call_async.call_count == 4
    fresh(f, 10)
    assert f.step(10, True, True, 10)
    assert f.phase == FlightPhase.ACTIVE
    assert course_error(1, 359) == 2


@pytest.mark.parametrize("armed,status", [(True, 4), (False, 3), (False, 4)])
def test_return_waits_for_zero_and_confirms_loiter(coordinator, armed, status):
    """Quit must wait for zero output before requesting a turn and for mode telemetry."""
    f = coordinator
    f.owned = True
    f.state.armed = armed
    f.state.system_status = status
    f.output.applied = [0.05, 0, 0]
    f.output.delta = [20]
    f.request_stop(10)
    fresh(f, 11)
    assert not f.recover(11)
    f.loiter_client.call_async.assert_not_called()
    f.output.applied = [0, 0, 0]
    f.output.delta = [0]
    assert not f.recover(11)
    assert f.phase == FlightPhase.RECOVERY
    request = f.loiter_client.call_async.call_args.args[0]
    assert (request.command, request.frame, request.x, request.y, request.z) == (
        17,
        3,
        350000000,
        -1060000000,
        100.0,
    )
    f.recovery_target = f.start_location[:2]
    f.recovery_target_time = 12
    f.state.mode = "LOITER"
    f.recovery_target = f.start_location[:2]
    f.recovery_target_time = 12
    assert f.recover(12)


def test_operator_mode_and_stale_recovery(coordinator):
    """Do not override operator mode changes, or claim recovery without telemetry."""
    f = coordinator
    f.owned = True
    f.request_stop(10)
    fresh(f, 11)
    f.state.mode = "RTL"
    assert f.recover(11)
    f.loiter_client.call_async.assert_not_called()
    f.phase = FlightPhase.RETURNING
    f.state.mode = "GUIDED"
    assert f.recover(20)
    assert f.failed


def test_rejected_ack_and_stale_active(coordinator):
    """Navigation failures or loss of telemetry stop setup/activation."""
    f = coordinator
    fresh(f, 0)
    f.step(0, True, False, 10)
    f.future.result.return_value = SimpleNamespace(success=False, result=3)
    fresh(f, 1)
    with pytest.raises(RuntimeError, match="rejected"):
        f.step(1, True, False, 10)
    f.phase = FlightPhase.ACTIVE
    with pytest.raises(RuntimeError, match="lost"):
        f.step(20, True, True, 10)


def test_stop_during_setup_cancels_remaining_commands(coordinator):
    """An interrupted setup cannot issue late altitude/course commands."""
    f = coordinator
    fresh(f, 0)
    f.step(0, True, False, 10)
    fresh(f, 0.05)
    f.step(0.05, True, False, 10)
    future = f.future
    f.request_stop(0.1)
    future.cancel.assert_called_once()
    assert not f.commands
    fresh(f, 0.2)
    assert not f.step(0.2, True, False, 10)
    assert f.command_client.call_async.call_count == 2
    f.recover(0.2)
    assert f.loiter_client.call_async.call_count == 1


def test_command_fields_and_recovery_service_failure(coordinator):
    """Encode relative altitude and ground course correctly; service faults fail exit."""
    f = coordinator
    for t in range(5):
        fresh(f, t)
        f.step(t, True, False, 10)
    requests = [c.args[0] for c in f.command_client.call_async.call_args_list]
    assert [r.command for r in requests] == [511, 43000, 43001, 43002]
    requests = requests[1:]
    assert requests[0].param1 == 0 and requests[0].param2 == 18
    assert requests[1].param7 == 100 and requests[1].param3 == 0
    assert requests[2].param1 == 0 and requests[2].param2 == 359
    f.request_stop(5)
    fresh(f, 6)
    f.loiter_client.call_async.return_value.result.side_effect = RuntimeError(
        "transport failed"
    )
    assert f.recover(6)
    assert f.failed and "service failed" in f.reason


def test_offline_events_reject_unsafe_or_unsettled_segment(coordinator):
    """Offline acceptance requires a stable entry, fixed targets, and a zero-output handoff."""
    from ros2_sid.straight_analysis import validate_events

    base = dict(
        configuration=coordinator.config,
        bank_deg=0,
        course_deg=359,
        target_course_deg=359,
        altitude_relative_m=100,
        airspeed_m_s=18,
        climb_m_s=0,
        rates_rad_s=[0, 0, 0],
        applied_axes=[0, 0, 0],
        applied_delta_pwm=[0],
        reason="",
        vehicle_mode="GUIDED",
    )
    rows = [
        dict(base, phase=phase, monotonic_s=t)
        for t, phase in [
            (0, "WAITING"),
            (1, "SETTLING"),
            (4, "READY"),
            (5, "ACTIVE"),
            (6, "RETURNING"),
            (7, "RECOVERY"),
            (8, "DONE"),
        ]
    ]
    rows[-1]["vehicle_mode"] = "LOITER"
    assert validate_events(rows)[0]["passed"]
    rows[-1]["recovery_start_location"] = [35.0, -106.0, 100.0]
    rows[-1]["recovery_target_latlon"] = [36.0, -105.0]
    rows[-1]["reason"] = "LOITER target confirmed at saved start"
    assert not validate_events(rows)[0]["passed"]
    rows[-1]["recovery_target_latlon"] = [35.0, -106.0]
    assert validate_events(rows)[0]["return_to_start_target_verified"]
    # rosbag reception-time ordering may put SETTLING before earlier WAITING.
    reordered = rows[1:3] + rows[:1] + rows[3:]
    assert validate_events(reordered)[0]["passed"]
    rows[-2]["applied_delta_pwm"] = [12]
    assert not validate_events(rows)[0]["passed"]
    rows[-2]["applied_delta_pwm"] = [0]
    rows[3]["target_course_deg"] = 120
    assert not validate_events(rows)[0]["passed"]
    rows[3]["target_course_deg"] = 359
    rows[1]["bank_deg"] = 20
    assert not validate_events(rows)[0]["passed"]


def test_ros_array_events_and_recording_failure_recovery(coordinator):
    """ROS fixed arrays contain NumPy scalars; recording failure cannot block exit."""
    import numpy as np

    f = coordinator
    f.output.applied = np.array([0, 0, 0], dtype=np.float32)
    f.output.delta = np.array([0], dtype=np.int16)
    f.output.rates = np.array([0, 0, 0], dtype=np.float32)
    fresh(f, 10)
    f.emit(10, True)
    assert not f.failed
    assert json.loads(f.events.publish.call_args.args[0].data)["applied_axes"] == [
        0,
        0,
        0,
    ]
    f.event_file.close()
    f.owned = True
    f.request_stop(10)
    assert f.failed
    fresh(f, 11)
    assert not f.recover(11)
    f.state.mode = "LOITER"
    f.recovery_target = f.start_location[:2]
    f.recovery_target_time = 12
    assert f.recover(12)


def test_critical_vehicle_state_yields_without_mode_command(coordinator):
    """A failsafe indication must not be overridden even if GUIDED remains reported."""
    f = coordinator
    f.owned = True
    fresh(f, 10)
    f.state.system_status = 5
    assert not f.authorized(10)
    f.request_stop(10)
    assert f.recover(10.1)
    f.loiter_client.call_async.assert_not_called()


def test_start_fix_is_fresh_captured_once_and_reset(coordinator):
    """Capture before setup, never move the recovery point with the aircraft."""
    f = coordinator
    fresh(f, 0)
    f.global_position_time = -10
    assert not f.step(0, True, False, 10)
    assert f.start_location is None
    fresh(f, 1)
    f.step(1, True, False, 10)
    assert f.start_location == (35.0, -106.0, 100.0)
    f.global_position = (36.0, -105.0)
    fresh(f, 2)
    f.step(2, True, False, 10)
    assert f.start_location == (35.0, -106.0, 100.0)
    f.begin(3, 10)
    assert f.start_location is None


def test_old_firmware_wrong_target_is_not_success(coordinator):
    """LOITER mode alone cannot confirm return to start on an older firmware."""
    f = coordinator
    f.owned = True
    f.request_stop(10)
    fresh(f, 11)
    assert not f.recover(11)
    f.state.mode = "LOITER"
    f.recovery_target = (36.0, -105.0)
    f.recovery_target_time = 12
    fresh(f, 12)
    assert not f.recover(12)
    fresh(f, 20)
    assert f.recover(20)
    assert f.failed and "unconfirmed" in f.reason


def test_stale_target_cannot_confirm_recovery(coordinator):
    """An echo predating the return command is not proof of the new target."""
    f = coordinator
    f.owned = True
    f.request_stop(10)
    fresh(f, 11)
    f.recover(11)
    f.state.mode = "LOITER"
    f.recovery_target = f.start_location[:2]
    f.recovery_target_time = 10
    assert not f.recover(11.1)


def test_telemetry_request_alone_does_not_own_navigation(coordinator):
    """Cancel before sending flight targets without changing the vehicle mode."""
    f = coordinator
    fresh(f, 0)
    f.step(0, True, False, 10)
    f.request_stop(0.01)
    assert f.recover(0.02)
    assert not f.owned
    f.loiter_client.call_async.assert_not_called()


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("status", [3, 4])
def test_navigation_authority_independent_of_arming(coordinator, armed, status):
    """Healthy standby/active state permits setup, but stale/mode loss still stops it."""
    f = coordinator
    fresh(f, 10)
    f.state.armed = armed
    f.state.system_status = status
    assert f.authorized(10)
    assert not f.authorized(10 + f.config["state_timeout_s"] + 1)
    f.state.mode = "MANUAL"
    assert not f.authorized(10)


@pytest.mark.parametrize("key,field,value", [
    ("bank_tolerance_deg", "bank", 10),
    ("altitude_tolerance_m", "altitude", 110),
    ("course_tolerance_deg", "course", 10),
    ("rate_tolerance_rad_s", "rate", 1),
    ("airspeed_tolerance_m_s", "airspeed", 25),
    ("climb_tolerance_m_s", "climb", 3),
])
def test_nullable_settling_check(coordinator, key, field, value):
    f = coordinator
    f.phase = FlightPhase.SETTLING
    f.course = f.target_course = 0
    set_measurement(f, field, value)
    fresh(f, 1)
    assert not f.step(1, True, False, 10)
    assert f.dwell_start is None
    f.config[key] = None
    fresh(f, 2)
    assert not f.step(2, True, False, 10)
    assert f.dwell_start == 2
    fresh(f, 5)
    assert f.step(5, True, False, 10)


def set_measurement(f, field, value):
    import math
    if field == "bank":
        f.output.attitude[0] = math.radians(value)
    elif field == "rate":
        f.output.rates[0] = value
    elif field == "airspeed":
        f.output.airspeed = value
    else:
        setattr(f, field, value)


@pytest.mark.parametrize("key,field,value", [
    ("max_bank_deg", "bank", 50),
    ("max_altitude_error_m", "altitude", 150),
    ("max_course_error_deg", "course", 90),
    ("min_airspeed_m_s", "airspeed", 5),
    ("max_airspeed_m_s", "airspeed", 40),
])
@pytest.mark.parametrize("phase", [FlightPhase.READY, FlightPhase.ACTIVE])
def test_nullable_envelope_check(coordinator, key, field, value, phase):
    f = coordinator
    f.phase = phase
    f.target_course = f.course = 0
    fresh(f, 1)
    set_measurement(f, field, value)
    with pytest.raises(RuntimeError, match="envelope"):
        f.step(1, True, phase == FlightPhase.ACTIVE, 10)
    f.config[key] = None
    assert f.step(1, True, phase == FlightPhase.ACTIVE, 10)
    # An unrelated enabled limit still aborts.
    f.config["max_altitude_error_m"] = 30
    f.altitude = 200
    with pytest.raises(RuntimeError, match="envelope"):
        f.step(1, True, phase == FlightPhase.ACTIVE, 10)


def test_nullable_duration(coordinator):
    f = coordinator
    with pytest.raises(RuntimeError, match="maximum duration"):
        f.begin(0, 100)
    f.config["max_duration_s"] = None
    f.begin(0, 100)
    assert f.phase == FlightPhase.WAITING


def test_all_settling_checks_disabled_keep_dwell_and_freshness(coordinator):
    f = coordinator
    for key in f.config:
        if "tolerance" in key:
            f.config[key] = None
    f.phase = FlightPhase.SETTLING
    f.output.attitude[0] = 0.2
    fresh(f, 1)
    assert not f.step(1, True, False, 10)
    fresh(f, 2)
    assert not f.step(2, True, False, 10)
    fresh(f, 4)
    assert f.step(4, True, False, 10)
    with pytest.raises(RuntimeError, match="telemetry lost"):
        f.step(10, True, True, 10)


def test_phase_enum_preserves_event_and_log_names(coordinator):
    f = coordinator
    assert f.phase is FlightPhase.IDLE
    f.transition(FlightPhase.SETTLING, "test transition")
    assert f.phase is FlightPhase.SETTLING
    f.node.get_logger().info.assert_called_with(
        "Straight flight: SETTLING: test transition"
    )
    record = json.loads(Path(f.event_file.name).read_text().splitlines()[-1])
    assert record["phase"] == "SETTLING"


@pytest.fixture
def guided(coordinator):
    coordinator.config["recovery_mode"] = "GUIDED"
    return coordinator


def target_message(latitude=34.9, longitude=-105.9, altitude=1550.0, **kwargs):
    fields = dict(latitude=latitude, longitude=longitude, altitude=altitude,
                  coordinate_frame=0, type_mask=0)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def echo(f, now, monkeypatch, **kwargs):
    monkeypatch.setattr("ros2_sid.straight_flight.time.monotonic", lambda: now)
    f.on_recovery_target(target_message(**kwargs))


def setup_guided(f, monkeypatch):
    fresh(f, 0)
    assert not f.step(0, True, False, 10)  # Request stream, no navigation changes.
    echo(f, 0.1, monkeypatch)
    fresh(f, 0.1)
    assert not f.step(0.1, True, False, 10)  # Capture original and send speed.
    assert f.owned


def start_guided_recovery(f, monkeypatch):
    setup_guided(f, monkeypatch)
    f.request_stop(1)
    fresh(f, 1.1)
    assert not f.recover(1.1)
    assert f.command_client.call_async.call_args.args[0].command == 17


def confirm_loiter(f, monkeypatch):
    fresh(f, 1.2)
    f.state.mode = "LOITER"
    assert not f.recover(1.2)
    return f.loiter_client.call_async.call_args.args[0]


def test_guided_capture_waits_for_target_before_navigation(guided, monkeypatch):
    from ros2_sid.straight_flight import GuidedTarget
    f = guided
    fresh(f, 0)
    echo(f, -1, monkeypatch)  # Old target is insufficient even if otherwise valid.
    assert not f.step(0, True, False, 10)
    fresh(f, 0.1)
    assert not f.step(0.1, True, False, 10)
    assert not f.owned and f.saved_guided_target is None
    assert [c.args[0].command for c in f.command_client.call_async.call_args_list] == [511]
    echo(f, 0.2, monkeypatch)
    fresh(f, 0.2)
    f.step(0.2, True, False, 10)
    saved = f.saved_guided_target
    assert saved == GuidedTarget(34.9, -105.9, 1550.0)
    assert saved.latitude != f.start_location[0]
    echo(f, 0.3, monkeypatch, latitude=36.0, altitude=2000.0)
    fresh(f, 0.3)
    f.step(0.3, True, False, 10)
    assert f.saved_guided_target == saved
    f.begin(5, 10)
    assert f.saved_guided_target is None
    assert not f.target_stream_requested


@pytest.mark.parametrize("fields", [
    dict(coordinate_frame=3), dict(type_mask=4), dict(latitude=float("nan")),
    dict(latitude=91), dict(longitude=181), dict(altitude=float("inf")),
    dict(latitude=0, longitude=0),
])
def test_guided_capture_rejects_invalid_target(guided, monkeypatch, fields):
    fresh(guided, 0)
    guided.step(0, True, False, 10)
    echo(guided, 0.1, monkeypatch, **fields)
    fresh(guided, 0.1)
    guided.step(0.1, True, False, 10)
    assert guided.saved_guided_target is None and not guided.owned
    assert guided.command_client.call_async.call_count == 1
    fresh(guided, 100)
    with pytest.raises(RuntimeError, match="setup timeout"):
        guided.step(100, True, False, 10)


def test_guided_zero_then_loiter_then_guided_original_target(guided, monkeypatch):
    from ros2_sid.straight_flight import RecoveryStage
    f = guided
    setup_guided(f, monkeypatch)
    f.request_stop(1)
    fresh(f, 1.1)
    f.output.applied = [0.1, 0, 0]
    assert not f.recover(1.1)
    assert f.command_client.call_async.call_count == 2
    f.output.applied = [0, 0, 0]
    assert not f.recover(1.1)
    assert f.recovery_stage == RecoveryStage.WAIT_LOITER
    assert f.command_client.call_async.call_args.args[0].command == 17
    f.loiter_client.call_async.assert_not_called()
    assert not f.recover(1.15)  # ACK alone is not a confirmed mode change.
    request = confirm_loiter(f, monkeypatch)
    assert (request.command, request.frame, request.param2) == (192, 0, 1.0)
    assert (request.x, request.y, request.z) == (349000000, -1059000000, 1550.0)
    assert f.recovery_stage == RecoveryStage.WAIT_GUIDED
    fresh(f, 1.3)
    f.state.mode = "GUIDED"
    assert not f.recover(1.3)  # Old echo cannot confirm restoration.
    echo(f, 1.4, monkeypatch)
    fresh(f, 1.4)
    assert f.recover(1.4)
    assert f.recovery_stage == RecoveryStage.COMPLETE and not f.failed
    assert f.state.mode == "GUIDED"
    assert f.recover(2)
    assert f.loiter_client.call_async.call_count == 1


@pytest.mark.parametrize("bad", [dict(latitude=36), dict(altitude=100), dict(coordinate_frame=3)])
def test_guided_wrong_target_cannot_confirm(guided, monkeypatch, bad):
    f = guided
    start_guided_recovery(f, monkeypatch)
    confirm_loiter(f, monkeypatch)
    fresh(f, 1.3)
    f.state.mode = "GUIDED"
    echo(f, 1.3, monkeypatch, **bad)
    assert not f.recover(1.3)
    assert f.recover(10)
    assert f.failed and "unconfirmed" in f.reason


@pytest.mark.parametrize("stage", ["before_loiter", "before_guided", "after_guided"])
def test_guided_operator_takeover_is_latched(guided, monkeypatch, stage):
    f = guided
    setup_guided(f, monkeypatch)
    f.request_stop(1)
    fresh(f, 1.1)
    if stage != "before_loiter":
        f.recover(1.1)
    if stage == "after_guided":
        confirm_loiter(f, monkeypatch)
    n = f.loiter_client.call_async.call_count
    monkeypatch.setattr("ros2_sid.straight_flight.time.monotonic", lambda: 1.3)
    f.on_state(SimpleNamespace(connected=True, mode="RTL", system_status=4))
    f.on_state(SimpleNamespace(connected=True, mode="GUIDED", system_status=4))
    assert f.recover(1.3)
    assert "Yielded" in f.reason
    assert f.loiter_client.call_async.call_count == n


@pytest.mark.parametrize("which", ["loiter", "guided"])
@pytest.mark.parametrize("failure", ["reject", "exception", "pending"])
def test_guided_recovery_failures(guided, monkeypatch, which, failure):
    f = guided
    start_guided_recovery(f, monkeypatch)
    if which == "guided":
        confirm_loiter(f, monkeypatch)
    future = f.future
    if failure == "reject":
        future.result.return_value = SimpleNamespace(success=False, result=4)
    elif failure == "exception":
        future.result.side_effect = RuntimeError("transport failure")
    else:
        future.done.return_value = False
    fresh(f, 1.3)
    if failure == "pending":
        assert not f.recover(1.3)
        assert f.recover(10)
    else:
        assert f.recover(1.3)
    assert f.failed


def test_guided_stale_state_and_output_do_not_restore(guided, monkeypatch):
    f = guided
    start_guided_recovery(f, monkeypatch)
    f.state.mode = "LOITER"
    f.state_time = -10
    assert not f.recover(1.2)
    fresh(f, 1.3)
    f.output_time = -10
    assert not f.recover(1.3)
    f.loiter_client.call_async.assert_not_called()
    assert f.recover(10) and f.failed


def test_guided_cancel_stream_request_has_no_navigation_ownership(guided):
    fresh(guided, 0)
    guided.step(0, True, False, 10)
    guided.request_stop(0.1)
    assert guided.recover(0.2)
    assert not guided.owned
    guided.loiter_client.call_async.assert_not_called()


def test_guided_stale_mode_cannot_confirm_loiter(guided, monkeypatch):
    f = guided
    start_guided_recovery(f, monkeypatch)
    f.state.mode = "LOITER"
    f.state_time = 1.0  # Fresh by age, but older than the LOITER request.
    assert not f.recover(1.2)
    f.loiter_client.call_async.assert_not_called()


def test_guided_missing_saved_target_fails_without_command(guided):
    guided.owned = True
    guided.request_stop(1)
    fresh(guided, 1.1)
    assert guided.recover(1.1) and guided.failed
    guided.command_client.call_async.assert_not_called()
    guided.loiter_client.call_async.assert_not_called()


def test_guided_offline_analysis_requires_full_handoff(guided):
    from copy import deepcopy
    from ros2_sid.straight_analysis import validate_events

    # Disabled tolerances must remain usable in the offline evidence path.
    config = dict(guided.config, bank_tolerance_deg=None)
    base = dict(
        configuration=config, bank_deg=0, course_deg=0, target_course_deg=0,
        altitude_relative_m=100, airspeed_m_s=18, climb_m_s=0,
        rates_rad_s=[0, 0, 0], applied_axes=[0, 0, 0], applied_delta_pwm=[0],
        reason="", vehicle_mode="GUIDED", recovery_stage="IDLE",
    )
    rows = [dict(base, phase=phase, monotonic_s=t) for t, phase in [
        (0, "WAITING"), (0.1, "WAITING"), (1, "SETTLING"), (4, "READY"),
        (5, "ACTIVE"), (6, "RETURNING"), (7, "RECOVERY"), (8, "RECOVERY"), (9, "DONE"),
    ]]
    rows[-3]["recovery_stage"] = "WAIT_LOITER"
    rows[-2].update(recovery_stage="WAIT_GUIDED", vehicle_mode="LOITER")
    target = dict(latitude=34.9, longitude=-105.9, altitude=1550, coordinate_frame=0)
    rows[-1].update(
        saved_guided_target=target, observed_guided_target=dict(target),
        recovery_stage="COMPLETE", recovery_confirmed=True,
        reason="Original GUIDED target confirmed",
    )
    report = validate_events(rows)
    assert len(report) == 1 and report[0]["passed"]
    assert report[0]["original_guided_target_verified"]
    for field, value in [("vehicle_mode", "LOITER"), ("recovery_confirmed", False),
                         ("recovery_stage", "WAIT_GUIDED")]:
        bad = deepcopy(rows)
        bad[-1][field] = value
        assert not validate_events(bad)[0]["passed"]
    bad = deepcopy(rows)
    bad[-1]["observed_guided_target"]["altitude"] = 100
    assert not validate_events(bad)[0]["passed"]
    assert not validate_events(rows[:-2] + rows[-1:])[0]["passed"]
