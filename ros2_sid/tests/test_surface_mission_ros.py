"""ROS-node tests with controlled clocks and synthetic RC/status input (no aircraft)."""

import json
from types import SimpleNamespace
import pytest

rclpy = pytest.importorskip("rclpy")
from ros2_sid.surface_mission import SurfaceMission


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


def test_default_mission_source(tmp_path, monkeypatch):
    """Load the current DDS plan relative to the installed package config."""
    from ros2_sid.surface_mission import resolve_mission_source

    config = tmp_path / "library"
    config.mkdir(parents=True)
    (config / "current_mission.json").write_text('{"subfolder":"selected"}')
    (config / "selected").mkdir()
    plan = config / "selected/mission_plan.json"
    plan.write_text("{}")
    monkeypatch.setattr("ros2_sid.surface_mission.mission_library_path", lambda: config)
    args = SimpleNamespace(experiment=None, mission_plan=None)
    resolve_mission_source(args)
    assert args.mission_plan == str(plan)
    plan.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_mission_source(SimpleNamespace(experiment=None, mission_plan=None))


def test_explicit_mission_source_bypasses_default(monkeypatch):
    """Explicit CLI selections work independently of the packaged pointer."""
    from ros2_sid.surface_mission import resolve_mission_source

    def unexpected_lookup():
        raise AssertionError("Explicit sources must bypass default lookup")

    monkeypatch.setattr(
        "ros2_sid.surface_mission.mission_library_path", unexpected_lookup
    )
    for field in ("experiment", "mission_plan"):
        args = SimpleNamespace(experiment=None, mission_plan=None)
        setattr(args, field, "custom.json")
        resolve_mission_source(args)
        assert getattr(args, field) == "custom.json"


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
    node.straight = Mock(owned=True, phase="SETTLING")
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
