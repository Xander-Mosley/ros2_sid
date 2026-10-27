"""Deterministic navigation coordination and shutdown tests; no vehicle commands."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ros2_sid.straight_flight import StraightFlight, course_error


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
        airspeed_m_s=18, min_airspeed_m_s=10, altitude_relative_m=100, dwell_s=3
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
    assert f.phase == "SETTLING"
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
    assert f.phase == "ACTIVE"
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
    assert f.phase == "RECOVERY"
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
    f.phase = "RETURNING"
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
    f.phase = "ACTIVE"
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
    f.phase = "SETTLING"
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
@pytest.mark.parametrize("phase", ["READY", "ACTIVE"])
def test_nullable_envelope_check(coordinator, key, field, value, phase):
    f = coordinator
    f.phase = phase
    f.target_course = f.course = 0
    fresh(f, 1)
    set_measurement(f, field, value)
    with pytest.raises(RuntimeError, match="envelope"):
        f.step(1, True, phase == "ACTIVE", 10)
    f.config[key] = None
    assert f.step(1, True, phase == "ACTIVE", 10)
    # An unrelated enabled limit still aborts.
    f.config["max_altitude_error_m"] = 30
    f.altitude = 200
    with pytest.raises(RuntimeError, match="envelope"):
        f.step(1, True, phase == "ACTIVE", 10)


def test_nullable_duration(coordinator):
    f = coordinator
    with pytest.raises(RuntimeError, match="maximum duration"):
        f.begin(0, 100)
    f.config["max_duration_s"] = None
    f.begin(0, 100)
    assert f.phase == "WAITING"


def test_all_settling_checks_disabled_keep_dwell_and_freshness(coordinator):
    f = coordinator
    for key in f.config:
        if "tolerance" in key:
            f.config[key] = None
    f.phase = "SETTLING"
    f.output.attitude[0] = 0.2
    fresh(f, 1)
    assert not f.step(1, True, False, 10)
    fresh(f, 2)
    assert not f.step(2, True, False, 10)
    fresh(f, 4)
    assert f.step(4, True, False, 10)
    with pytest.raises(RuntimeError, match="telemetry lost"):
        f.step(10, True, True, 10)
