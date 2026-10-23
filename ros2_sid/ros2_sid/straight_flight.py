"""Optional MAVROS straight-flight coordination around DDS surface excitation.

COMMAND_LONG altitude is relative to home in this ArduPlane fork. Course is true
north clockwise; MAVROS local velocity is ENU. No arming or GUIDED request is sent.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ardupilot_msgs.msg import ControlSurfaceOutput
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from mavros_msgs.msg import GlobalPositionTarget, State
    from rclpy.node import Node
    from sensor_msgs.msg import NavSatFix
    from std_msgs.msg import Float64


def course_error(actual: float, target: float) -> float:
    """Return signed shortest course error in degrees."""
    return (actual - target + 180) % 360 - 180


class StraightFlight:
    """Coordinate acknowledged targets, continuous settling, and bounded recovery."""

    def __init__(self, node: Node, path: str, event_path: str) -> None:
        """Create optional ROS endpoints; importing this module does not start flight."""
        from ardupilot_msgs.msg import ControlSurfaceOutput
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from mavros_msgs.msg import GlobalPositionTarget, State
        from mavros_msgs.srv import CommandInt, CommandLong
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import NavSatFix
        from std_msgs.msg import Float64, String

        self.node = node
        self.config = json.loads(Path(path).read_text())
        required = (
            "altitude_relative_m",
            "airspeed_m_s",
            "heading_accel_m_s2",
            "dwell_s",
            "setup_timeout_s",
            "recovery_timeout_s",
            "telemetry_timeout_s",
            "state_timeout_s",
            "course_tolerance_deg",
            "bank_tolerance_deg",
            "rate_tolerance_rad_s",
            "altitude_tolerance_m",
            "airspeed_tolerance_m_s",
            "climb_tolerance_m_s",
            "max_bank_deg",
            "max_altitude_error_m",
            "max_course_error_deg",
            "max_duration_s",
            "min_airspeed_m_s",
            "max_airspeed_m_s",
        )
        for key in required:
            value = float(self.config[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            self.config[key] = value
        climb_rate = float(self.config["climb_rate_m_s"])
        if not math.isfinite(climb_rate) or climb_rate < 0:
            raise ValueError(
                "climb_rate_m_s must be finite and nonnegative (zero uses firmware maximum)"
            )
        self.config["climb_rate_m_s"] = climb_rate
        course = self.config.get("course_deg")
        if course is not None and (not math.isfinite(course) or not 0 <= course < 360):
            raise ValueError("course_deg must be null or in [0,360)")
        if self.config.get("recovery_mode") != "LOITER":
            raise ValueError("This coordinator requires explicit recovery_mode LOITER")
        if (
            not self.config["min_airspeed_m_s"]
            < self.config["airspeed_m_s"]
            < self.config["max_airspeed_m_s"]
        ):
            raise ValueError(
                "Target airspeed must lie inside the active airspeed envelope"
            )
        self.phase = "IDLE"
        self.reason = ""
        self.failed = False
        self.owned = False
        self.started = 0.0
        self.dwell_start: float | None = None
        self.return_started = 0.0
        self.future: Any = None
        self.future_started = 0.0
        self.commands: list[tuple[int, dict[str, float]]] = []
        self.state: Any = None
        self.output: Any = None
        self.state_time = self.output_time = self.velocity_time = (
            self.altitude_time
        ) = -math.inf
        self.course = self.climb = self.altitude = self.speed = 0.0
        self.target_course = 0.0
        self.position: tuple[float, float] | None = None
        self.last_event = -math.inf
        self.mode_requested = False
        self.global_position: tuple[float, float] | None = None
        self.global_position_time = -math.inf
        self.recovery_target: tuple[float, float] | None = None
        self.recovery_target_time = -math.inf
        self.start_location: tuple[float, float, float] | None = None
        self.loiter_client = node.create_client(CommandInt, "/mavros/cmd/command_int")
        self.command_client = node.create_client(CommandLong, "/mavros/cmd/command")
        self.events = node.create_publisher(String, "/sid/straight_flight", 10)
        self.event_file = open(event_path, "w")
        node.create_subscription(
            NavSatFix,
            "/mavros/global_position/global",
            self.on_global_position,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            GlobalPositionTarget,
            "/mavros/setpoint_raw/target_global",
            self.on_recovery_target,
            qos_profile_sensor_data,
        )
        node.create_subscription(State, "/mavros/state", self.on_state, 10)
        node.create_subscription(
            ControlSurfaceOutput,
            "/ap/sid/output",
            self.on_output,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self.on_velocity,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.on_position,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            Float64,
            "/mavros/global_position/rel_alt",
            self.on_altitude,
            qos_profile_sensor_data,
        )

    def on_state(self, msg: State) -> None:
        """Retain vehicle authority telemetry and local receipt time."""
        self.state, self.state_time = msg, time.monotonic()

    def on_output(self, msg: ControlSurfaceOutput) -> None:
        """Retain coherent applied-output and aircraft-response telemetry."""
        if self.output is not None and msg.time_us <= self.output.time_us:
            return
        self.output, self.output_time = msg, time.monotonic()

    def on_velocity(self, msg: TwistStamped) -> None:
        """Convert ENU velocity to north-clockwise ground course and climb."""
        v = msg.twist.linear
        if not all(math.isfinite(x) for x in (v.x, v.y, v.z)):
            return
        self.course = math.degrees(math.atan2(v.x, v.y)) % 360
        self.speed = math.hypot(v.x, v.y)
        self.climb = v.z
        self.velocity_time = time.monotonic()

    def on_position(self, msg: PoseStamped) -> None:
        """Retain ENU horizontal position for track plots; not used as an altitude source."""
        point = msg.pose.position
        if math.isfinite(point.x) and math.isfinite(point.y):
            self.position = (point.x, point.y)

    def on_recovery_target(self, msg: GlobalPositionTarget) -> None:
        """Observe the firmware navigation target, independently of command ACKs."""
        if math.isfinite(msg.latitude) and math.isfinite(msg.longitude):
            self.recovery_target = (msg.latitude, msg.longitude)
            self.recovery_target_time = time.monotonic()

    def on_global_position(self, msg: NavSatFix) -> None:
        """Retain a valid global fix for the test-start recovery location."""
        if (
            msg.status.status >= 0
            and math.isfinite(msg.latitude)
            and math.isfinite(msg.longitude)
            and -90 <= msg.latitude <= 90
            and -180 <= msg.longitude <= 180
        ):
            self.global_position = (msg.latitude, msg.longitude)
            self.global_position_time = time.monotonic()

    def on_altitude(self, msg: Float64) -> None:
        """Receive altitude above home in meters, matching COMMAND_LONG semantics."""
        if math.isfinite(msg.data):
            self.altitude, self.altitude_time = msg.data, time.monotonic()

    def authorized(self, now: float) -> bool:
        """Require fresh healthy GUIDED state, independently of vehicle arming."""
        return bool(
            self.state
            and now - self.state_time <= self.config["state_timeout_s"]
            and self.state.connected
            and self.state.system_status in (3, 4)  # MAV_STATE_STANDBY / ACTIVE
            and self.state.mode == "GUIDED"
        )

    def fresh(self, now: float) -> bool:
        """Require all navigation and output streams to be recent and finite."""
        return (
            self.output is not None
            and self.speed > 3
            and all(
                now - stamp <= self.config["telemetry_timeout_s"]
                for stamp in (self.output_time, self.velocity_time, self.altitude_time)
            )
            and all(
                math.isfinite(float(v))
                for v in [
                    self.output.airspeed,
                    *self.output.attitude,
                    *self.output.rates,
                ]
            )
        )

    def emit(self, now: float, force: bool = False) -> None:
        """Record phase, targets, and response at 10 Hz for bag-based verification."""
        from std_msgs.msg import String

        if not force and now - self.last_event < 0.1:
            return
        record = dict(
            monotonic_s=now,
            ap_time_us=int(self.output.clock_us) if self.output else 0,
            applied_axes=[float(v) for v in self.output.applied]
            if self.output
            else None,
            applied_delta_pwm=[int(v) for v in self.output.delta[: self.output.count]]
            if self.output
            else None,
            firmware_time_us=int(self.output.time_us)
            if self.output and hasattr(self.output, "time_us")
            else None,
            phase=self.phase,
            receipt_age_s=dict(
                state=now - self.state_time,
                output=now - self.output_time,
                velocity=now - self.velocity_time,
                altitude=now - self.altitude_time,
            ),
            vehicle_mode=self.state.mode if self.state else None,
            reason=self.reason,
            course_deg=self.course,
            target_course_deg=self.target_course,
            altitude_relative_m=self.altitude,
            climb_m_s=self.climb,
            configuration=self.config,
            position_enu_m=self.position,
            recovery_start_location=self.start_location,
            recovery_target_latlon=self.recovery_target,
            bank_deg=math.degrees(self.output.attitude[0]) if self.output else None,
            airspeed_m_s=float(self.output.airspeed) if self.output else None,
            groundspeed_m_s=self.speed,
            rates_rad_s=[float(v) for v in self.output.rates] if self.output else None,
        )
        try:
            text = json.dumps(record)
            self.events.publish(String(data=text))
            self.event_file.write(text + "\n")
            self.event_file.flush()
        except Exception as exc:
            # Evidence failures revoke the maneuver, but must not break recovery.
            self.failed = True
            self.node.get_logger().error(
                f"Straight-flight event recording failed: {exc}"
            )
        self.last_event = now

    def transition(self, phase: str, reason: str = "") -> None:
        """Expose a phase transition in ROS, logs, and the operator console."""
        self.phase, self.reason = phase, reason
        self.node.get_logger().info(f"Straight flight: {phase}: {reason}")
        self.emit(time.monotonic(), True)

    def begin(self, now: float, duration: float) -> None:
        """Start setup only on user request; do not command until telemetry is valid."""
        if duration > self.config["max_duration_s"]:
            raise RuntimeError("Maneuver exceeds straight-flight maximum duration")
        self.start_location = None
        self.started = now
        self.failed = self.owned = self.mode_requested = False
        self.dwell_start = None
        self.future = None
        self.transition("WAITING")

    def request_stop(self, now: float) -> None:
        """Revoke setup and enter return-to-baseline before any planned recovery turn."""
        if self.phase in ("RETURNING", "RECOVERY", "DONE"):
            return
        self.commands.clear()
        # A sent command cannot be recalled; never issue further setup commands.
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self.return_started = now
        self.transition("RETURNING")

    def step(self, now: float, requested: bool, active: bool, duration: float) -> bool:
        """Advance setup without blocking; return true only when excitation may start."""
        from mavros_msgs.srv import CommandLong

        if self.phase in ("IDLE", "DONE"):
            if not requested:
                return False
            self.begin(now, duration)
        if self.phase in ("RETURNING", "RECOVERY"):
            return False
        if (
            now - self.started > self.config["setup_timeout_s"]
            and self.phase != "ACTIVE"
        ):
            raise RuntimeError("Straight-flight settling/setup timeout")
        if self.phase == "WAITING":
            if not self.authorized(now) or not self.fresh(now):
                return False
            if (
                self.global_position is None
                or now - self.global_position_time > self.config["telemetry_timeout_s"]
            ):
                return False
            if self.global_position == (0.0, 0.0):
                raise RuntimeError("Zero latitude/longitude is reserved for loiter-here commands")
            self.start_location = (*self.global_position, self.altitude)
            self.target_course = self.config.get("course_deg")
            if self.target_course is None:
                self.target_course = self.course
            self.commands = [
                (511, dict(param1=87.0, param2=200000.0)),  # target echo at 5 Hz
                (
                    43000,
                    dict(param1=0.0, param2=self.config["airspeed_m_s"], param3=1.0),
                ),
                (
                    43001,
                    dict(
                        param3=self.config["climb_rate_m_s"],
                        param7=self.config["altitude_relative_m"],
                    ),
                ),
                (
                    43002,
                    dict(
                        param1=0.0,
                        param2=float(self.target_course),
                        param3=self.config["heading_accel_m_s2"],
                    ),
                ),
            ]
            self.transition("COMMANDING")
        if not self.authorized(now) or not self.fresh(now):
            raise RuntimeError("Straight-flight authorization/telemetry lost")
        if self.phase == "COMMANDING":
            if self.future is not None:
                if not self.future.done():
                    if now - self.future_started > 5:
                        raise RuntimeError("Navigation command ACK timeout")
                    return False
                response = self.future.result()
                self.future = None
                if not response or not response.success or response.result != 0:
                    raise RuntimeError("Navigation command rejected")
            if self.commands:
                if not self.command_client.service_is_ready():
                    return False
                command, fields = self.commands.pop(0)
                self.future = self.command_client.call_async(
                    CommandLong.Request(command=command, **fields)
                )
                self.future_started = now
                if command != 511:  # A telemetry-rate request does not own navigation.
                    self.owned = True
                return False
            self.transition("SETTLING")
        bank = abs(math.degrees(self.output.attitude[0]))
        altitude_error = abs(self.altitude - self.config["altitude_relative_m"])
        course = abs(course_error(self.course, self.target_course))
        if self.phase in ("READY", "ACTIVE"):
            if (
                not self.config["min_airspeed_m_s"]
                <= self.output.airspeed
                <= self.config["max_airspeed_m_s"]
                or bank > self.config["max_bank_deg"]
                or altitude_error > self.config["max_altitude_error_m"]
                or course > self.config["max_course_error_deg"]
            ):
                raise RuntimeError("Straight-flight envelope exceeded")
            if active and self.phase != "ACTIVE":
                self.transition("ACTIVE")
            self.emit(now)
            return True
        stable = (
            bank <= self.config["bank_tolerance_deg"]
            and altitude_error <= self.config["altitude_tolerance_m"]
            and course <= self.config["course_tolerance_deg"]
            and max(abs(v) for v in self.output.rates)
            <= self.config["rate_tolerance_rad_s"]
            and abs(self.output.airspeed - self.config["airspeed_m_s"])
            <= self.config["airspeed_tolerance_m_s"]
            and abs(self.climb) <= self.config["climb_tolerance_m_s"]
        )
        if not stable:
            self.dwell_start = None
        elif self.dwell_start is None:
            self.dwell_start = now
        elif now - self.dwell_start >= self.config["dwell_s"]:
            self.transition("READY")
        self.emit(now)
        return self.phase == "READY"

    def recover(self, now: float) -> bool:
        """After zero excitation, command LOITER about the saved setup-start fix."""
        from mavros_msgs.srv import CommandInt

        if self.phase == "DONE":
            return True
        if not self.owned:
            self.transition("DONE", "No navigation ownership")
            return True
        # Never replace an operator/failsafe-selected mode. Arming is independent.
        state_fresh = (
            self.state is not None
            and now - self.state_time <= self.config["state_timeout_s"]
        )
        expected_loiter = bool(
            state_fresh
            and self.mode_requested
            and self.state.mode == "LOITER"
            and self.state.system_status in (3, 4)
        )
        if (
            state_fresh
            and not expected_loiter
            and (
                self.state.mode != "GUIDED"
                or self.state.system_status not in (3, 4)
            )
        ):
            self.transition(
                "DONE",
                f"Yielded to vehicle mode {self.state.mode}, system status {self.state.system_status}",
            )
            return True
        target_matches = bool(
            self.start_location
            and self.recovery_target
            and self.recovery_target_time > self.future_started
            and now - self.recovery_target_time <= self.config["telemetry_timeout_s"]
            and all(
                abs(a - b) < 1e-6
                for a, b in zip(self.start_location[:2], self.recovery_target)
            )
        )
        if expected_loiter and target_matches and self.future.done():
            try:
                result = self.future.result()
            except Exception as exc:
                self.failed = True
                self.transition(
                    "DONE",
                    f"LOITER service failed: {exc}; operator intervention required",
                )
                return True
            if result and result.success:
                self.transition(
                    "DONE",
                    "LOITER target confirmed at saved start; aircraft returning (arrival not confirmed)",
                )
                return True
        if now - self.return_started > self.config["recovery_timeout_s"]:
            self.failed = True
            self.transition(
                "DONE", "Recovery unconfirmed: operator intervention required"
            )
            return True
        if not self.authorized(now) and not expected_loiter:
            return False
        zero = (
            self.output is not None
            and self.output_time > self.return_started
            and now - self.output_time <= self.config["telemetry_timeout_s"]
            and all(abs(v) < 1e-6 for v in self.output.applied)
            and all(v == 0 for v in self.output.delta[: self.output.count])
        )
        if zero and not self.mode_requested and self.loiter_client.service_is_ready():
            if self.start_location is None:
                self.failed = True
                self.transition(
                    "DONE",
                    "Missing saved start location; operator intervention required",
                )
                return True
            try:
                latitude, longitude, altitude = self.start_location
                self.future = self.loiter_client.call_async(
                    CommandInt.Request(
                        command=17,
                        frame=3,  # NAV_LOITER_UNLIM, GLOBAL_RELATIVE_ALT
                        x=round(latitude * 1e7),
                        y=round(longitude * 1e7),
                        z=float(altitude),
                    )
                )
            except Exception as exc:
                self.failed = True
                self.transition(
                    "DONE",
                    f"LOITER service failed: {exc}; operator intervention required",
                )
                return True
            self.future_started = now
            self.mode_requested = True
            self.transition(
                "RECOVERY",
                "Zero applied perturbation confirmed; requesting LOITER at saved start",
            )
        if self.mode_requested and self.future.done():
            try:
                result = self.future.result()
            except Exception as exc:
                self.failed = True
                self.transition(
                    "DONE",
                    f"LOITER service failed: {exc}; operator intervention required",
                )
                return True
            if not result or not result.success:
                self.failed = True
                self.transition(
                    "DONE", "LOITER request rejected; operator intervention required"
                )
                return True
        self.emit(now)
        return False
