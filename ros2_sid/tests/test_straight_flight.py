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
    config.update(airspeed_m_s=18, min_airspeed_m_s=10, altitude_relative_m=100, dwell_s=3)
    path = tmp_path / "test-settings.json"
    path.write_text(json.dumps(config))
    flight = StraightFlight(node, str(path), str(tmp_path / "events.jsonl"))
    flight.command_client = Mock()
    flight.mode_client = Mock()
    flight.command_client.service_is_ready.return_value = True
    flight.mode_client.service_is_ready.return_value = True
    future = Mock()
    future.done.return_value = True
    future.result.return_value = SimpleNamespace(success=True, result=0, mode_sent=True)
    flight.command_client.call_async.return_value = future
    flight.mode_client.call_async.return_value = future
    flight.state = SimpleNamespace(armed=True, connected=True, mode="GUIDED", system_status=4)
    flight.output = SimpleNamespace(
        clock_us=1,
        attitude=[0, 0, 0],
        rates=[0, 0, 0],
        airspeed=18,
        applied=[0, 0, 0],
        delta=[0],
        count=1,
    )
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


def test_settling_commands_once_and_wrap(coordinator):
    """Commands are sequential, target course is captured, and dwell is continuous."""
    f = coordinator
    for t in range(4):
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
    assert f.command_client.call_async.call_count == 3
    fresh(f, 10)
    assert f.step(10, True, True, 10)
    assert f.phase == "ACTIVE"
    assert course_error(1, 359) == 2


def test_return_waits_for_zero_and_confirms_loiter(coordinator):
    """Quit must wait for zero output before requesting a turn and for mode telemetry."""
    f = coordinator
    f.owned = True
    f.output.applied = [0.05, 0, 0]
    f.output.delta = [20]
    f.request_stop(10)
    fresh(f, 11)
    assert not f.recover(11)
    f.mode_client.call_async.assert_not_called()
    f.output.applied = [0, 0, 0]
    f.output.delta = [0]
    assert not f.recover(11)
    assert f.phase == "RECOVERY"
    assert f.mode_client.call_async.call_args.args[0].custom_mode == "LOITER"
    f.state.mode = "LOITER"
    assert f.recover(12)


def test_operator_mode_and_stale_recovery(coordinator):
    """Do not override operator mode changes, or claim recovery without telemetry."""
    f = coordinator
    f.owned = True
    f.request_stop(10)
    fresh(f, 11)
    f.state.mode = "RTL"
    assert f.recover(11)
    f.mode_client.call_async.assert_not_called()
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
    future = f.future
    f.request_stop(0.1)
    future.cancel.assert_called_once()
    assert not f.commands
    fresh(f, 0.2)
    assert not f.step(0.2, True, False, 10)
    assert f.command_client.call_async.call_count == 1
    f.recover(0.2)
    assert f.mode_client.call_async.call_count == 1


def test_command_fields_and_recovery_service_failure(coordinator):
    """Encode relative altitude and ground course correctly; service faults fail exit."""
    f = coordinator
    for t in range(4):
        fresh(f, t)
        f.step(t, True, False, 10)
    requests = [c.args[0] for c in f.command_client.call_async.call_args_list]
    assert [r.command for r in requests] == [43000, 43001, 43002]
    assert requests[0].param1 == 0 and requests[0].param2 == 18
    assert requests[1].param7 == 100 and requests[1].param3 == 0
    assert requests[2].param1 == 0 and requests[2].param2 == 359
    f.request_stop(4)
    fresh(f, 5)
    f.mode_client.call_async.return_value.result.side_effect = RuntimeError("transport failed")
    assert f.recover(5)
    assert f.failed and "service failed" in f.reason


def test_offline_events_reject_unsafe_or_unsettled_segment(coordinator):
    """Offline acceptance requires a stable entry, fixed targets, and a zero-output handoff."""
    from ros2_sid.straight_analysis import validate_events

    base = dict(configuration=coordinator.config, bank_deg=0, course_deg=359,
                target_course_deg=359, altitude_relative_m=100, airspeed_m_s=18,
                climb_m_s=0, rates_rad_s=[0, 0, 0], applied_axes=[0, 0, 0],
                applied_delta_pwm=[0], reason="", vehicle_mode="GUIDED")
    rows = [dict(base, phase=phase, monotonic_s=t) for t, phase in
            [(0, "WAITING"), (1, "SETTLING"), (4, "READY"), (5, "ACTIVE"),
             (6, "RETURNING"), (7, "RECOVERY"), (8, "DONE")]]
    rows[-1]["vehicle_mode"] = "LOITER"
    assert validate_events(rows)[0]["passed"]
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
    assert json.loads(f.events.publish.call_args.args[0].data)["applied_axes"] == [0, 0, 0]
    f.event_file.close()
    f.owned = True
    f.request_stop(10)
    assert f.failed
    fresh(f, 11)
    assert not f.recover(11)
    f.state.mode = "LOITER"
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
    f.mode_client.call_async.assert_not_called()
