"""ROS-node tests with controlled clocks and synthetic RC/status input (no aircraft)."""

import csv
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import pytest

rclpy = pytest.importorskip("rclpy")
from ros2_sid.surface_mission import SurfaceMission
from ros2_sid.straight_flight import FlightPhase


@pytest.mark.parametrize("mode", ["SITL", "HITL"])
def test_entrypoint_uses_current_mission_amplitude(tmp_path, monkeypatch, mode):
    """Both DDS entrypoints resolve the mission scale before constructing a node."""
    import ros2_sid.surface_mission as surface

    (tmp_path / "selected").mkdir()
    (tmp_path / "selected/mission_plan.json").write_text("{}")
    (tmp_path / "current_mission.json").write_text(
        '{"subfolder":"selected", "amplitude_modifier":0.35}'
    )
    monkeypatch.setattr(surface, "mission_library_path", lambda: tmp_path)
    monkeypatch.setattr(surface, "resolve_straight_flight", lambda args: None)
    captured = []

    class NodeConstructed(Exception):
        pass

    def capture_node(args, hitl):
        captured.append((args.amplitude_modifier, hitl))
        raise NodeConstructed

    monkeypatch.setattr(surface, "SurfaceMission", capture_node)
    script = Path(__file__).resolve().parents[1] / "scripts" / f"run_mission_{mode}.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--interface", "dds", "--no-straight-flight"])
    try:
        with pytest.raises(NodeConstructed):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    assert captured == [(0.35, mode == "HITL")]


@pytest.fixture
def node(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            dict(
                units="normalized_surface",
                rate=100,
                duration=1,
                axes={"roll": dict(signal="constant", amplitude=0.1)},
            )
        )
    )
    args = SimpleNamespace(
        experiment=str(path),
        mission_plan=None,
        amplitude_modifier=1.0,
        autostart=True,
        rc_channel=7,
        rc_threshold=1800,
        rc_timeout=0.5,
        wait_timeout=10,
        command_log=str(tmp_path / "commands.csv"),
    )
    rclpy.init()
    n = SurfaceMission(args)
    yield n
    n.logfile.close()
    n.destroy_node()
    rclpy.shutdown()


def test_sitl_start_complete_disable(node, monkeypatch):
    current = [100.0]
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: current[0])
    node.clock_sync.update(1000.0, 100.0)
    node.status = SimpleNamespace(state=1)
    node.status_time = 100.0
    node.disable_acknowledged = True
    sent = []
    node.send = lambda enable, values=(0, 0, 0): sent.append((enable, values))
    node.tick()
    assert sent[-1] == (True, (0.1, 0.0, 0.0))
    node.active_seen = True
    current[0] = 101.0
    node.clock_sync.update(1001.0, 101.0)
    node.status_time = 101.0
    # Deadline violation is a fault, not a burst of stale samples.
    node.tick()
    assert node.failed and node.stopping is not None
    current[0] = 101.1
    node.tick()
    assert sent[-1][0] is False
    current[0] = 101.6
    node.tick()
    assert node.done


def test_hitl_stale_and_high_start(node, monkeypatch):
    current = [100.0]
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: current[0])
    node.hitl = True
    node.requested = False
    node.clock_sync.update(1000.0, 100.0)
    node.status = SimpleNamespace(state=1)
    node.status_time = 100.0
    node.disable_acknowledged = True
    sent = []
    node.send = lambda enable, values=(0, 0, 0): sent.append(enable)
    values = [1500] * 16
    values[6] = 1900
    node.on_rc(SimpleNamespace(values=values, valid_mask=0xFFFF))
    node.tick()
    assert True not in sent
    values[6] = 1000
    node.on_rc(SimpleNamespace(values=values, valid_mask=0xFFFF))
    values[6] = 1900
    node.on_rc(SimpleNamespace(values=values, valid_mask=0xFFFF))
    node.tick()
    assert sent[-1]
    current[0] = 100.6
    node.tick()
    assert node.stopping is not None
    current[0] = 100.7
    node.tick()
    assert not sent[-1]


def test_clean_completion(node, monkeypatch):
    current = [100.0]
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: current[0])
    node.disable_acknowledged = True
    sent = []
    node.send = lambda enable, values=(0, 0, 0): sent.append(enable)
    node.status = SimpleNamespace(state=1)
    node.active_seen = True
    for i in range(101):
        current[0] = 100.0 + i * 0.01
        node.clock_sync.update(1000.0 + i * 0.01, current[0])
        node.status_time = current[0]
        node.tick()
    assert node.stopping is not None and not node.failed
    current[0] = 101.1
    node.tick()
    assert not sent[-1]


def test_new_publisher_requires_own_disable_ack(node, monkeypatch):
    current = [100.0]
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: current[0])
    node.clock_sync.update(1000.0, 100.0)
    node.on_status(SimpleNamespace(state=1, sequence_id=977))
    node.tick()
    assert node.cadence is None and node.last_enabled_sequence is None
    assert node.disable_sequences
    node.on_status(SimpleNamespace(state=1, sequence_id=node.disable_sequences[-1]))
    current[0] += 0.01
    node.tick()
    assert node.cadence is not None and node.last_enabled_sequence is not None
    current[0] += 0.21
    node.clock_sync.update(1000.22, current[0])
    node.tick()
    assert node.failed


@pytest.mark.parametrize("override, expected", [(None, 0.4), (0.2, 0.2), (0.0, 0.0)])
def test_default_mission_source(tmp_path, monkeypatch, override, expected):
    """Load the current DDS plan relative to the installed package config."""
    from ros2_sid.surface_mission import resolve_mission_source

    config = tmp_path / "library"
    config.mkdir(parents=True)
    (config / "current_mission.json").write_text(
        '{"subfolder":"selected", "amplitude_modifier":0.4}'
    )
    (config / "selected").mkdir()
    plan = config / "selected/mission_plan.json"
    plan.write_text("{}")
    monkeypatch.setattr("ros2_sid.surface_mission.mission_library_path", lambda: config)
    args = SimpleNamespace(experiment=None, mission_plan=None, amplitude_modifier=override)
    resolve_mission_source(args)
    assert args.mission_plan == str(plan)
    assert args.amplitude_modifier == expected
    plan.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_mission_source(SimpleNamespace(experiment=None, mission_plan=None))


@pytest.mark.parametrize("override, expected", [(None, 1.0), (0.3, 0.3), (0.0, 0.0)])
def test_explicit_mission_source_bypasses_default(monkeypatch, override, expected):
    """Explicit CLI selections work independently of the packaged pointer."""
    from ros2_sid.surface_mission import resolve_mission_source

    def unexpected_lookup():
        raise AssertionError("Explicit sources must bypass default lookup")

    monkeypatch.setattr(
        "ros2_sid.surface_mission.mission_library_path", unexpected_lookup
    )
    for field in ("experiment", "mission_plan"):
        args = SimpleNamespace(experiment=None, mission_plan=None, amplitude_modifier=override)
        setattr(args, field, "custom.json")
        resolve_mission_source(args)
        assert getattr(args, field) == "custom.json"
        assert args.amplitude_modifier == expected


@pytest.mark.parametrize("modifier", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_amplitude_modifier_rejected(modifier):
    from ros2_sid.surface_mission import resolve_mission_source

    args = SimpleNamespace(experiment="custom.json", mission_plan=None, amplitude_modifier=modifier)
    with pytest.raises(ValueError, match="must be finite"):
        resolve_mission_source(args)


def test_send_scales_all_axes_and_command_log(node, monkeypatch):
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.clock_sync.update(1000.0, 100.0)
    node.args.amplitude_modifier = 0.4
    sent = []
    node.publisher = SimpleNamespace(publish=sent.append)
    node.send(True, (0.2, -0.3, 0.1))
    assert (sent[-1].roll, sent[-1].pitch, sent[-1].yaw) == pytest.approx((0.08, -0.12, 0.04))
    node.send(False)
    assert (sent[-1].roll, sent[-1].pitch, sent[-1].yaw) == (0.0, 0.0, 0.0)
    with open(node.args.command_log) as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(float(rows[0][axis]) for axis in ("roll", "pitch", "yaw")) == pytest.approx((0.08, -0.12, 0.04))


def test_csv_plan_order_units_and_cadence(tmp_path):
    """Match legacy ordering and CSV timing while rejecting attitude plans."""
    from ros2_sid.surface_mission import load_mission_experiments

    (tmp_path / "wave.csv").write_text(
        "time,roll,pitch,yaw\n0,0.1,0,0\n0.02,0.05,0,0\n0.04,0,0,0\n"
    )
    path = tmp_path / "mission_plan.json"
    plan = dict(
        units="normalized_surface",
        maneuvers=[
            dict(filename="wave.csv", name="Second", order=2),
            dict(filename="wave.csv", name="First", order=1),
        ],
    )
    path.write_text(json.dumps(plan))
    loaded = load_mission_experiments(path)
    assert [name for name, _ in loaded] == ["First", "Second"]
    assert loaded[0][1].rate == 50
    assert loaded[0][1].sample(0.02) == (0.05, 0, 0)
    del plan["units"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="normalized_surface"):
        load_mission_experiments(path)


def test_console_quit_stops_reading(node, monkeypatch):
    """Read batched manual commands and quit without another blocking read."""
    calls = []
    monkeypatch.setattr(
        "ros2_sid.surface_mission.sys.stdin", SimpleNamespace(fileno=lambda: 123)
    )
    monkeypatch.setattr(
        "ros2_sid.surface_mission.select.select", lambda *args: ([123], [], [])
    )

    def read_once(fd, count):
        calls.append(1)
        assert len(calls) == 1
        return b"list\nselect 1\nstart\nstop\nquit\n"

    monkeypatch.setattr("ros2_sid.surface_mission.os.read", read_once)
    node.console()
    assert [node.events.get_nowait() for _ in range(5)] == [
        "list",
        "select 1",
        "start",
        "stop",
        "quit",
    ]
    assert len(calls) == 1


@pytest.mark.parametrize("command", ["stop", "quit"])
def test_straight_console_waits_for_recovery(node, monkeypatch, command):
    """Manual exit keeps sending disable until navigation recovery is confirmed."""
    from unittest.mock import Mock

    current = [100.0]
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: current[0])
    node.straight = Mock(failed=False)
    node.straight.recover.return_value = False
    node.events.put(command)
    sent = []
    node.send = lambda enable, values=(0, 0, 0): sent.append(enable)
    node.tick()
    node.straight.request_stop.assert_called_once_with(100.0)
    current[0] = 101
    node.tick()
    assert not node.done and sent == [False, False]
    node.straight.recover.return_value = True
    node.tick()
    assert node.done and True not in sent


def test_clock_loss_during_straight_setup_stops(node, monkeypatch):
    """A lost clock during navigation setup must not strand a held course."""
    from unittest.mock import Mock

    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.straight = Mock(owned=True, phase=FlightPhase.SETTLING)
    node.tick()
    assert node.failed and node.stopping == 100.0
    node.straight.request_stop.assert_called_once_with(100.0)


def test_default_navigation_cli(tmp_path, monkeypatch):
    """Default enables packaged settings; custom settings and explicit opt-out work."""
    import argparse
    from ros2_sid.surface_mission import add_navigation_arguments, resolve_straight_flight

    config = tmp_path / "config/surface_excitation/straight_flight_sitl.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", lambda name: str(tmp_path))
    parser = argparse.ArgumentParser()
    add_navigation_arguments(parser)
    args = parser.parse_args([])
    resolve_straight_flight(args)
    assert args.straight_flight == str(config)
    custom = tmp_path / "custom.json"
    custom.write_text("{}")
    args = parser.parse_args(["--straight-flight", str(custom)])
    resolve_straight_flight(args)
    assert args.straight_flight == str(custom)
    config.unlink()
    args = parser.parse_args(["--no-straight-flight"])
    resolve_straight_flight(args)
    assert args.straight_flight is None
    with pytest.raises(FileNotFoundError):
        resolve_straight_flight(parser.parse_args([]))
    with pytest.raises(SystemExit):
        parser.parse_args(["--straight-flight", str(custom), "--no-straight-flight"])


@pytest.mark.parametrize("phase", ["waiting", "active"])
@pytest.mark.parametrize("channels", [[1500] * 8, []])
def test_hitl_legacy_low_or_missing_stops_between_ticks(node, monkeypatch, phase, channels):
    """RC low/missing cancels preparation or sampling before a subsequent high callback."""
    from ros2_sid.surface_signals import RCGate, Cadence

    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.hitl = True
    node.rc = RCGate()
    node.requested = True
    if phase == "active":
        node.cadence = Cadence(100, 100)
    node.on_mavros_rc(SimpleNamespace(channels=channels))
    assert node.stopping == 100
    assert node.cadence is None and not node.requested
    node.on_mavros_rc(SimpleNamespace(channels=[1500] * 7 + [1900]))
    assert not node.rc.allowed(100)
    node.clock_sync.update(1000, 100)
    sent = []
    node.send = lambda enable, values=(0, 0, 0): sent.append(enable)
    node.tick()
    assert sent == [False]


def test_hitl_console_cannot_start(node, monkeypatch):
    """Console start cannot bypass the physical RC gate."""
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.hitl = True
    node.requested = False
    node.clock_sync.update(1000, 100)
    node.events.put("start")
    sent = []
    node.send = lambda enable, values=(0, 0, 0): sent.append(enable)
    node.tick()
    assert not node.requested and True not in sent


@pytest.mark.parametrize("source,topic", [("mavros", "/mavros/rc/in"), ("dds", "/ap/rcin")])
def test_hitl_constructor_uses_selected_rc_source(node, tmp_path, source, topic):
    """The real HITL constructor subscribes to exactly the requested RC transport."""
    args = SimpleNamespace(**vars(node.args))
    args.rc_source = source
    args.command_log = str(tmp_path / (source + ".csv"))
    child = SurfaceMission(args, hitl=True)
    try:
        topics = {sub.topic_name for sub in child.subscriptions}
        assert topic in topics
        assert ({"/mavros/rc/in", "/ap/rcin"} - {topic}).isdisjoint(topics)
        assert not child.requested
    finally:
        child.logfile.close()
        child.destroy_node()


def test_hitl_waiting_status_is_clear_and_not_repeated(node, monkeypatch):
    """Waiting for RC is visible without flooding the console every timer tick."""
    from unittest.mock import Mock

    logger = Mock()
    monkeypatch.setattr(node, "get_logger", lambda: logger)
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.hitl = True
    node.requested = False
    node.clock_sync.update(1000, 100)
    node.send = lambda *args: None
    node.tick()
    node.tick()
    assert logger.info.call_count == 1
    assert "RC input missing" in logger.info.call_args.args[0]
    node.events.put("start")
    node.tick()
    assert any("console start is ignored" in c.args[0] for c in logger.info.call_args_list)
    assert not node.requested


def test_hitl_running_message_requires_firmware_ack(node, monkeypatch):
    """Only an ACTIVE acknowledgement for this publisher can announce RUNNING."""
    from unittest.mock import Mock
    from ros2_sid.surface_signals import Cadence

    logger = Mock()
    monkeypatch.setattr(node, "get_logger", lambda: logger)
    node.hitl = True
    node.cadence = Cadence(100, 100)
    node.last_enabled_sequence = 1000
    node.on_status(SimpleNamespace(state=2, sequence_id=10))
    assert not node.active_seen and not logger.info.called
    node.on_status(SimpleNamespace(state=2, sequence_id=1000))
    node.on_status(SimpleNamespace(state=2, sequence_id=1000))
    assert node.active_seen and logger.info.call_count == 1
    assert "RUNNING" in logger.info.call_args.args[0]


@pytest.mark.parametrize("state,reason,expected", [
    (4, 1, "state=FAULT (4), reason=UNAUTHORIZED (1)"),
    (99, 99, "state=UNKNOWN (99), reason=UNKNOWN (99)"),
])
def test_authorization_wait_prints_names(node, monkeypatch, state, reason, expected):
    """Real timer diagnostics name firmware codes without crashing on newer codes."""
    from unittest.mock import Mock

    logger = Mock()
    monkeypatch.setattr(node, "get_logger", lambda: logger)
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.hitl = True
    node.started = 100.0
    node.requested = True
    node.rc.allowed = lambda now: True
    node.clock_sync.update(1000, 100)
    node.send = lambda *args: None
    node.status = SimpleNamespace(state=state, reason=reason)
    node.status_time = 100.0
    node.tick()
    assert any(expected in call.args[0] for call in logger.info.call_args_list)


@pytest.mark.parametrize("override,expected", [({}, None), ({"rc_timeout": 0.25}, 0.25)])
def test_constructor_resolves_json_rc_timeout(node, tmp_path, override, expected):
    from pathlib import Path

    args = SimpleNamespace(**vars(node.args))
    del args.rc_timeout
    args.straight_flight = str(
        Path(__file__).parents[1] / "config/surface_excitation/straight_flight_sitl.json"
    )
    args.command_log = str(tmp_path / "configured.csv")
    for key, value in override.items():
        setattr(args, key, value)
    child = SurfaceMission(args, hitl=True)
    try:
        assert child.rc.timeout == expected
        assert child.straight.config["rc_timeout_s"] == expected
    finally:
        child.straight.event_file.close()
        child.logfile.close()
        child.destroy_node()


def test_hitl_null_timeout_waiting_and_stop(node, monkeypatch):
    node.hitl = True
    node.requested = False
    node.rc.timeout = None
    monkeypatch.setattr("ros2_sid.surface_mission.time.monotonic", lambda: 100.0)
    node.send = lambda *args, **kwargs: None
    node.tick()  # Missing RC still waits; status formatting accepts None.
    assert not node.requested
    low = [1000] * 8
    high = [1900] * 8
    node.rc.update(low, 255, 0)
    node.rc.update(high, 255, 1)
    node.clock_sync.update(1000, 100)
    node.status = SimpleNamespace(state=1)
    node.status_time = 100
    node.disable_acknowledged = True
    node.tick()
    assert node.requested and node.stopping is None
    node.on_mavros_rc(SimpleNamespace(channels=low))
    assert node.stopping == 100 and not node.requested
