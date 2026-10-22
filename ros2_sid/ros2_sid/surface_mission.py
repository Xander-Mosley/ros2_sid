"""DDS mission backend shared by run_mission_SITL.py and run_mission_HITL.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import secrets
import select
import signal
import sys
import threading
import time
from collections import deque
from enum import IntEnum
from pathlib import Path

import rclpy
from ardupilot_msgs.msg import ControlSurfaceExcitation, ControlSurfaceStatus, RcIn
from builtin_interfaces.msg import Time
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .mission_library import mission_library_path
from .surface_signals import AxisSample, Cadence, ClockSync, Experiment, RCGate

QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def add_navigation_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose default straight-flight coordination and an explicit opt-out."""
    navigation = parser.add_mutually_exclusive_group()
    navigation.add_argument(
        "--straight-flight",
        metavar="CONFIG",
        help="Override packaged straight-flight settings (MAVROS required by default)",
    )
    navigation.add_argument(
        "--no-straight-flight",
        action="store_true",
        help="Disable navigation coordination and LOITER recovery; publish DDS excitation only",
    )


def resolve_straight_flight(args: argparse.Namespace) -> None:
    """Resolve default settings from the installed package unless explicitly disabled."""
    if args.no_straight_flight:
        args.straight_flight = None
        return
    if args.straight_flight is None:
        from ament_index_python.packages import get_package_share_directory

        args.straight_flight = str(
            Path(get_package_share_directory("ros2_sid"))
            / "config/surface_excitation/straight_flight_sitl.json"
        )
    if not Path(args.straight_flight).is_file():
        raise FileNotFoundError(
            f"Straight-flight configuration not found: {args.straight_flight}"
        )


def resolve_mission_source(args: argparse.Namespace) -> None:
    """Resolve the packaged current DDS mission unless an explicit source was given."""
    if args.experiment or args.mission_plan:
        return
    library = mission_library_path()
    pointer = library / "current_mission.json"
    selection = json.loads(pointer.read_text())
    folder = selection.get("subfolder")
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError(f"{pointer} requires a nonempty subfolder")
    path = library / folder.strip() / "mission_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"DDS mission plan not found: {path}")
    args.mission_plan = str(path)


def load_mission_experiments(plan_path: Path) -> list[tuple[str, Experiment]]:
    """Load ordered legacy-layout CSVs or explicit DDS waveform definitions.

    CSV plans must declare normalized_surface units; legacy angle data is rejected.
    Sampling rate and duration are inferred from the CSV time column.
    """
    plan = json.loads(plan_path.read_text())
    if plan.get("units") != "normalized_surface":
        raise ValueError(
            "DDS mission plan requires normalized_surface units; select a surface mission"
        )
    entries = plan["maneuvers"]
    if not entries:
        raise ValueError("empty mission plan")
    if any("filename" in entry for entry in entries):
        if not all(
            isinstance(entry.get("order"), int)
            and isinstance(entry.get("filename"), str)
            for entry in entries
        ):
            raise ValueError(
                "CSV mission entries require filename, name and integer order"
            )
        entries = sorted(entries, key=lambda entry: entry["order"])
    result = []
    for entry in entries:
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError("maneuver name must be nonempty")
        if "filename" in entry:
            path = plan_path.parent / entry["filename"]
            with path.open() as stream:
                reader = csv.reader(stream)
                next(reader)
                times = [float(row[0]) for row in reader]
            if len(times) < 2:
                raise ValueError("CSV needs at least two samples")
            dt = times[1] - times[0]
            if dt <= 0 or any(
                abs((b - a) - dt) > max(1e-9, abs(dt) * 1e-6)
                for a, b in zip(times, times[1:])
            ):
                raise ValueError("CSV sample spacing must be positive and uniform")
            experiment = Experiment(
                plan_path,
                config={
                    "units": "normalized_surface",
                    "csv": entry["filename"],
                    "rate": 1 / dt,
                    "duration": times[-1],
                },
            )
        else:
            experiment = Experiment(plan_path.parent / entry["experiment"])
        result.append((entry["name"], experiment))
    return result


class ExcitationState(IntEnum):
    """State codes published by the firmware ControlSurfaceStatus interface."""

    DISABLED = 0
    ARMED = 1
    ACTIVE = 2
    TIMEOUT = 3
    FAULT = 4


class SurfaceMission(Node):
    """ROS 2 DDS excitation publisher with clock, firmware, RC, and cadence safeguards."""

    def __init__(self, args: argparse.Namespace, hitl: bool = False) -> None:
        """Create the DDS endpoints, validated experiment, authorization state, and steady timer."""
        super().__init__("surface_excitation_mission")
        self.args = args
        self.hitl = hitl
        if args.experiment:
            self.experiments = [
                (Path(args.experiment).stem, Experiment(args.experiment))
            ]
        else:
            plan_path = Path(args.mission_plan)
            self.experiments = load_mission_experiments(plan_path)
        self.experiment = self.experiments[0][1]
        self.quit_requested = False
        self.clock_sync = ClockSync()
        self.rc = RCGate(args.rc_channel - 1, args.rc_threshold, args.rc_timeout)
        self.publisher = self.create_publisher(
            ControlSurfaceExcitation, "/ap/sid/control_surface", QOS
        )
        self.create_subscription(Time, "/ap/time", self.on_time, QOS)
        self.create_subscription(
            ControlSurfaceStatus, "/ap/sid/status", self.on_status, QOS
        )
        if hitl:
            self.create_subscription(RcIn, "/ap/rcin", self.on_rc, QOS)
        self.sequence = secrets.randbits(32)
        self.status: ControlSurfaceStatus | None = None
        self.status_time = 0.0
        self.cadence: Cadence | None = None
        self.events: queue.Queue[str] = queue.Queue()
        self.requested = args.autostart and not hitl
        self.done = False
        self.failed = False
        self.active_seen = False
        self.stopping: float | None = None
        self.started = time.monotonic()
        self.last_disable = 0.0
        self.disable_acknowledged = False
        self.disable_sequences: deque[int] = deque(maxlen=16)
        self.last_enabled_sequence: int | None = None
        self.logfile = open(args.command_log, "w", newline="")
        self.log = csv.writer(self.logfile)
        self.log.writerow(
            ["monotonic_s", "stamp_s", "sequence", "enable", "roll", "pitch", "yaw"]
        )
        self.straight = None
        if getattr(args, "straight_flight", None):
            from .straight_flight import StraightFlight

            self.straight = StraightFlight(
                self, args.straight_flight, args.command_log + ".straight.jsonl"
            )
        self.timer = self.create_timer(
            1 / self.experiment.rate,
            self.tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        if not args.autostart:
            threading.Thread(target=self.console, daemon=True).start()

    def console(self) -> None:
        """Queue console lines without holding Python stdin locks during shutdown."""
        pending = b""
        descriptor = sys.stdin.fileno()
        print("DDS mission: list / select N / start / stop / quit:", flush=True)
        while not self.done:
            ready, _, _ = select.select([descriptor], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(descriptor, 4096)
            if not chunk:
                self.events.put("quit")
                return
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                command = line.decode("utf-8", errors="replace").strip().lower()
                self.events.put(command)
                if command == "quit":
                    return

    def on_time(self, msg: Time) -> None:
        """Update AP-clock synchronization and abort on a detected clock discontinuity."""
        try:
            self.clock_sync.update(msg.sec + msg.nanosec * 1e-9, time.monotonic())
        except RuntimeError as exc:
            self.stop(str(exc), failed=True)

    def on_status(self, msg: ControlSurfaceStatus) -> None:
        """Track firmware status and acknowledge only samples from this publisher epoch."""
        self.status = msg
        self.status_time = time.monotonic()
        if (
            msg.state == ExcitationState.ARMED
            and msg.sequence_id in self.disable_sequences
        ):
            self.disable_acknowledged = True
        if (
            self.cadence is not None
            and msg.state == ExcitationState.ACTIVE
            and self.last_enabled_sequence is not None
            and ((self.last_enabled_sequence - msg.sequence_id) & 0xFFFFFFFF) < 100
        ):
            self.active_seen = True

    def on_rc(self, msg: RcIn) -> None:
        """Refresh the HITL RC gate from the message validity mask and channel values."""
        self.rc.update(msg.values, msg.valid_mask, time.monotonic())

    def send(self, enable: bool, values: AxisSample = (0.0, 0.0, 0.0)) -> None:
        """Publish one synchronized normalized sample and retain its sequence in the command CSV."""
        now = time.monotonic()
        stamp = self.clock_sync.stamp(now)
        msg = ControlSurfaceExcitation()
        msg.sequence_id = self.sequence
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        msg.enable = enable
        msg.stamp.sec = int(stamp)
        msg.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        msg.roll, msg.pitch, msg.yaw = map(float, values)
        self.publisher.publish(msg)
        if enable:
            self.last_enabled_sequence = msg.sequence_id
        else:
            self.disable_sequences.append(msg.sequence_id)
        self.log.writerow([now, stamp, msg.sequence_id, int(enable), *values])
        self.logfile.flush()

    def stop(self, reason: str, failed: bool = False) -> None:
        """Begin repeated disable transmission and revoke the current mission authorization."""
        if self.stopping is None:
            self.get_logger().info(reason)
            self.stopping = time.monotonic()
            if self.straight is not None:
                self.straight.request_stop(self.stopping)
        self.failed |= failed
        self.requested = False
        self.cadence = None
        self.rc.reset()
        self.disable_acknowledged = False
        self.disable_sequences.clear()
        self.last_enabled_sequence = None

    def tick(self) -> None:
        """Advance authorization, sampling, and bounded shutdown using monotonic deadlines."""
        now = time.monotonic()
        while not self.events.empty():
            command = self.events.get_nowait()
            if command == "start" and not self.hitl and self.stopping is None:
                self.requested = True
            elif command in ("stop", "quit"):
                self.quit_requested = command == "quit"
                self.stop(command)
            elif command == "list":
                for i, (name, _) in enumerate(self.experiments):
                    self.get_logger().info(f"{i}: {name}")
            elif (
                command.startswith("select ")
                and self.cadence is None
                and not self.requested
            ):
                try:
                    index = int(command.split()[1])
                    if not 0 <= index < len(self.experiments):
                        raise ValueError("index out of range")
                    self.experiment = self.experiments[index][1]
                    self.timer.timer_period_ns = int(1e9 / self.experiment.rate)
                    self.get_logger().info("Selected " + self.experiments[index][0])
                except (ValueError, IndexError) as exc:
                    self.get_logger().error(str(exc))
        if self.stopping is not None:
            if now - self.last_disable >= 0.05:
                try:
                    self.send(False)
                except RuntimeError:
                    pass  # Firmware monotonic timeout remains authoritative.
                self.last_disable = now
            recovered = self.straight is None or self.straight.recover(now)
            if self.straight is not None:
                self.failed |= self.straight.failed
            if now - self.stopping >= 0.5 and recovered:
                if self.args.autostart or self.quit_requested:
                    self.done = True
                else:
                    self.stopping = None
                    self.started = now
                    self.active_seen = False
            return
        if self.hitl:
            allowed = self.rc.allowed(now)
            if (
                self.cadence
                or (
                    self.straight
                    and self.straight.owned
                    and self.straight.phase != "DONE"
                )
            ) and not allowed:
                self.stop("RC switch off, invalid, or stale")
                return
            self.requested = allowed
        try:
            self.clock_sync.stamp(now)
        except RuntimeError as exc:
            if self.cadence or (
                self.straight and self.straight.owned and self.straight.phase != "DONE"
            ):
                self.stop(str(exc), failed=True)
            elif now - self.started > self.args.wait_timeout:
                self.stop("No fresh AP clock", failed=True)
            return
        if self.straight is not None and self.requested:
            try:
                ready = self.straight.step(
                    now, True, self.cadence is not None, self.experiment.duration
                )
                if self.straight.failed:
                    raise RuntimeError("Straight-flight event recording failed")
            except (RuntimeError, ValueError) as exc:
                self.stop(str(exc), failed=True)
                return
            if not ready:
                self.started = now
                if now - self.last_disable >= 0.05:
                    self.send(False)
                    self.last_disable = now
                return
        if self.cadence is None:
            if now - self.last_disable >= 0.05:
                self.send(False)
                self.last_disable = now
            if not self.requested:
                return
            if (
                not self.disable_acknowledged
                or self.status is None
                or now - self.status_time > 0.5
                or self.status.state != ExcitationState.ARMED
            ):
                if now - self.started > self.args.wait_timeout:
                    self.stop("Firmware did not authorize mission", failed=True)
                return
            self.cadence = Cadence(self.experiment.rate, now)
        if (
            self.status is None
            or now - self.status_time > 0.5
            or self.status.state in (ExcitationState.TIMEOUT, ExcitationState.FAULT)
        ):
            self.stop("Firmware timeout/fault or stale status", failed=True)
            return
        try:
            elapsed = self.cadence.due(now)
            if elapsed is None:
                return
            if elapsed > 0.2 and not self.active_seen:
                self.stop("Firmware did not acknowledge enabled samples", failed=True)
                return
            if elapsed >= self.experiment.duration:
                self.stop("Maneuver complete", failed=not self.active_seen)
                return
            self.send(True, self.experiment.sample(elapsed))
        except (RuntimeError, ValueError) as exc:
            self.stop(str(exc), failed=True)


def main(hitl: bool = False) -> int:
    """Run the shared SITL/HITL DDS mission CLI; always attempt disable during shutdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=["dds"], required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--experiment", help="JSON with explicit normalized_surface units"
    )
    source.add_argument(
        "--mission-plan", help="normalized_surface plan with named experiment paths"
    )
    parser.add_argument(
        "--autostart",
        action="store_true",
        help="SITL automated start after firmware authorization",
    )
    add_navigation_arguments(parser)
    parser.add_argument("--command-log", default="surface_commands.csv")
    parser.add_argument("--wait-timeout", type=float, default=30.0)
    parser.add_argument(
        "--rc-channel", type=int, default=7, help="one-based RC switch channel (HITL)"
    )
    parser.add_argument("--rc-threshold", type=int, default=1800)
    parser.add_argument("--rc-timeout", type=float, default=0.5)
    args, ros_args = parser.parse_known_args()
    if not 1 <= args.rc_channel <= 16 or not 0 < args.rc_timeout <= 2:
        parser.error("invalid RC gate configuration")
    if hitl and args.autostart:
        parser.error("HITL requires a fresh low-to-high RC transition")
    try:
        resolve_mission_source(args)
        resolve_straight_flight(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"DDS mission source: {args.experiment or args.mission_plan}")
    print(
        f"Straight-flight coordination: {args.straight_flight or 'disabled (--no-straight-flight)'}"
    )
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = SurfaceMission(args, hitl)
    previous_term = signal.signal(signal.SIGTERM, lambda *_: node.events.put("quit"))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        node.quit_requested = True
        node.stop("Interrupted")
    except Exception as exc:
        node.quit_requested = True
        node.stop(f"Exception: {exc}", failed=True)
    finally:
        # Keep ROS alive to observe the return ramp and complete the recovery handoff.
        if not node.done:
            node.quit_requested = True
            node.stop("Exiting")
            budget = (
                node.straight.config["recovery_timeout_s"] + 1 if node.straight else 1
            )
            deadline = time.monotonic() + budget
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                while rclpy.ok() and not node.done and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.02)
            except Exception as exc:
                node.failed = True
                node.get_logger().error(f"Shutdown executor failed: {exc}")
            finally:
                signal.signal(signal.SIGINT, previous_int)
            if not node.done:
                node.failed = True
                node.get_logger().error(
                    "Shutdown recovery unconfirmed; operator intervention required"
                )
        # Repeated disable on interrupt and exceptions; never depend on delivery alone.
        for _ in range(5):
            try:
                node.send(False)
            except Exception:
                pass
            time.sleep(0.02)
        signal.signal(signal.SIGTERM, previous_term)
        if node.straight:
            node.straight.event_file.close()
        node.logfile.close()
        node.destroy_node()
        rclpy.shutdown()
    return 1 if node.failed else 0
