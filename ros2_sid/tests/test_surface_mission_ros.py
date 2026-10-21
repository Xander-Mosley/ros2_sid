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
