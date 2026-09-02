#!/usr/bin/env python3

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from drone_interfaces.msg import (
    SysIdDataStream,
    SysIdLeastSquares,
)
from ros2_sid.realtime_ols_utils import ordinary_least_squares


@dataclass
class OLSResult:
    measured_output: float
    regressors: list[float]
    parameters: list[float]


class SIDols(Node):
    CURRENT_AIRCRAFT_FILE = (
        Path(__file__).resolve().parents[1]
        / "ros2_sid"
        / "setup"
        / "aircraft_library"
        / "current_aircraft.json"
    )

    FOURIER_PREFIX = "/sid/fourier/"
    DIFFER_PREFIX = "/sid/differ/"
    OLS_PREFIX = "/sid/ols/"

    FOURIER_TOPICS = {
        f"{FOURIER_PREFIX}imu/gx": None,
        f"{FOURIER_PREFIX}imu/gy": None,
        f"{FOURIER_PREFIX}imu/gz": None,
        f"{FOURIER_PREFIX}rcout/ail": None,
        f"{FOURIER_PREFIX}rcout/elv": None,
        f"{FOURIER_PREFIX}rcout/rud": None,
    }
    DIFFER_TOPICS = {
        f"{DIFFER_PREFIX}imu/gx": None,
        f"{DIFFER_PREFIX}imu/gy": None,
        f"{DIFFER_PREFIX}imu/gz": None,
    }
    BLACK_TOPICS = [
        # f"{FOURIER_PREFIX}rcout/ail",
        # f"{DIFFER_PREFIX}imu/gx",
    ]

    SYS_ID_DATA_STREAM_TYPE = (
        "drone_interfaces/msg/SysIdDataStream"
    )

    DYNAMIC_SUBS = True

    TOPIC_DISCOVERY_PERIOD = 1.0
    OLS_PUBLISH_PERIOD = 0.04


    def __init__(self, ns=''):
        super().__init__("sid_ols")
        self.subs: Dict[str, Subscription] = {}
        self.pubs: Dict[str, Publisher] = {}

        self.fourier_streams: Dict[str, SysIdDataStream] = {}
        self.differ_streams: Dict[str, SysIdDataStream] = {}

        self.aircraft: Dict = {}
        self.load_current_aircraft()

        self.add_static_topics()

        if self.DYNAMIC_SUBS:
            self.discovery_timer = self.create_timer(
                self.TOPIC_DISCOVERY_PERIOD,
                self.discover_topics,
            )

        self.ols_timer = self.create_timer(
            self.OLS_PUBLISH_PERIOD,
            self.calculate_ols,
        )

        self.get_logger().info(
            "SID ols node initialized"
            f" | dynamic_subs={self.DYNAMIC_SUBS}"
        )

    def load_current_aircraft(self) -> None:
        """
        Load the current aircraft configuration from JSON.
        """
        try:
            with self.CURRENT_AIRCRAFT_FILE.open("r", encoding="utf-8") as file:
                self.aircraft = json.load(file)
        except FileNotFoundError:
            self.get_logger().error(
                f"Current aircraft file not found: "
                f"{self.CURRENT_AIRCRAFT_FILE}"
            )
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f"Invalid JSON in current aircraft file: {error}"
            )
        except OSError as error:
            self.get_logger().error(
                f"Unable to read current aircraft file: {error}"
            )


    def add_static_topics(self) -> None:
        """Create all statically configured Fourier and differ streams."""
        for topic, stream_name in self.FOURIER_TOPICS.items():
            if not topic.startswith(self.FOURIER_PREFIX):
                self.get_logger().error(
                    f"Cannot add invalid Fourier topic: {topic}"
                )
                continue
            if stream_name is None:
                stream_name = topic.removeprefix(self.FOURIER_PREFIX)
            self.add_fourier_subscription(
                fourier_topic=topic,
                stream_name=stream_name,
                static=True,
            )

        for topic, stream_name in self.DIFFER_TOPICS.items():
            if not topic.startswith(self.DIFFER_PREFIX):
                self.get_logger().error(
                    f"Cannot add invalid differ topic: {topic}"
                )
                continue
            if stream_name is None:
                stream_name = topic.removeprefix(self.DIFFER_PREFIX)
            self.add_differ_subscription(
                differ_topic=topic,
                stream_name=stream_name,
                static=True,
            )

    def discover_topics(self) -> None:
        """
        Discover new SysIdDataStream topics under
        /sid/fourier/ and /sid/differ/.
        """
        topic_list = self.get_topic_names_and_types()

        for topic_name, topic_type in topic_list:
            if not topic_type:
                continue
            if self.SYS_ID_DATA_STREAM_TYPE not in topic_type:
                continue
            if topic_name in self.subs:
                continue
            if topic_name in self.BLACK_TOPICS:
                continue

            if topic_name.startswith(self.FOURIER_PREFIX):
                configured_name = self.FOURIER_TOPICS.get(topic_name)
                if configured_name is None:
                    stream_name = topic_name.removeprefix(self.FOURIER_PREFIX)
                else:
                    stream_name = configured_name

                self.add_fourier_subscription(
                    fourier_topic=topic_name,
                    stream_name=stream_name,
                    static=False,
                )

            elif topic_name.startswith(self.DIFFER_PREFIX):
                configured_name = self.DIFFER_TOPICS.get(topic_name)
                if configured_name is None:
                    stream_name = topic_name.removeprefix(self.DIFFER_PREFIX)
                else:
                    stream_name = configured_name
                    
                self.add_differ_subscription(
                    differ_topic=topic_name,
                    stream_name=stream_name,
                    static=False,
                )


    def _make_fourier_callback(
        self,
        stream_name: str,
        ) -> Callable[[SysIdDataStream], None]:
        """Create a callback for a Fourier stream."""
        def callback(msg: SysIdDataStream) -> None:
            self.fourier_callback(msg, stream_name)
        return callback

    def _make_differ_callback(
        self,
        stream_name: str,
        ) -> Callable[[SysIdDataStream], None]:
        """Create a callback for a differentiated stream."""
        def callback(msg: SysIdDataStream) -> None:
            self.differ_callback(msg, stream_name)
        return callback

    def add_fourier_subscription(
        self,
        fourier_topic: str,
        stream_name: str,
        static: bool = False,
        ) -> None:
        """Create a subscription to a Fourier stream."""
        if fourier_topic in self.subs:
            return

        if not fourier_topic.startswith(self.FOURIER_PREFIX):
            self.get_logger().error(
                f"Cannot add invalid Fourier topic: "
                f"{fourier_topic}"
            )
            return

        subscription = self.create_subscription(
            SysIdDataStream,
            fourier_topic,
            self._make_fourier_callback(stream_name),
            10,
        )

        self.subs[fourier_topic] = subscription

        source = "static" if static else "dynamic"
        self.get_logger().info(
            f"Added {source} Fourier subscription named {stream_name}"
            f" | {fourier_topic}"
        )

    def add_differ_subscription(
        self,
        differ_topic: str,
        stream_name: str,
        static: bool = False,
        ) -> None:
        """Create a subscription to a differentiated stream."""
        if differ_topic in self.subs:
            return

        if not differ_topic.startswith(self.DIFFER_PREFIX):
            self.get_logger().error(
                f"Cannot add invalid differ topic: "
                f"{differ_topic}"
            )
            return

        subscription = self.create_subscription(
            SysIdDataStream,
            differ_topic,
            self._make_differ_callback(stream_name),
            10,
        )

        self.subs[differ_topic] = subscription

        source = "static" if static else "dynamic"
        self.get_logger().info(
            f"Added {source} Differ subscription named {stream_name}"
            f" | {differ_topic}"
        )


    def fourier_callback(
        self,
        msg: SysIdDataStream,
        stream_name: str,
        ) -> None:
        """
        Store the most recent Fourier message.
        The message is copied so the stored value is independent of the
        ROS callback message object.
        """
        self.fourier_streams[stream_name] = copy(msg)

    def differ_callback(
        self,
        msg: SysIdDataStream,
        stream_name: str,
        ) -> None:
        """
        Store the most recent differentiated message.
        The message is copied so the stored value is independent of the
        ROS callback message object.
        """
        self.differ_streams[stream_name] = copy(msg)


    def get_fourier_stream(
        self,
        stream_name: str,
        ) -> SysIdDataStream | None:
        """Return the most recent Fourier message for a stream."""
        return self.fourier_streams.get(stream_name)
    
    def get_fourier_value(
        self,
        stream_name: str,
        ) -> float | None:
        """Return the most recent Fourier value."""
        stream = self.get_fourier_stream(stream_name)
        if stream is None:
            return None
        return float(stream.value)
    
    def get_fourier_trend(
        self,
        stream_name: str,
        ) -> float | None:
        """Return the most recent Fourier trend."""
        stream = self.get_fourier_stream(stream_name)
        if stream is None:
            return None
        return float(stream.trend)

    def get_differ_stream(
        self,
        stream_name: str,
        ) -> SysIdDataStream | None:
        """Return the most recent differentiated message for a stream."""
        return self.differ_streams.get(stream_name)
    
    def get_differ_value(
        self,
        stream_name: str,
        ) -> float | None:
        """Return the most recent differentiated value."""
        stream = self.get_differ_stream(stream_name)
        if stream is None:
            return None
        return float(stream.value)
    
    def get_differ_trend(
        self,
        stream_name: str,
        ) -> float | None:
        """Return the most recent differentiated trend."""
        stream = self.get_differ_stream(stream_name)
        if stream is None:
            return None
        return float(stream.trend)


    def frequency_ols(
        self,
        measured_output: SysIdDataStream,
        regressors: list[SysIdDataStream],
        ) -> OLSResult:
        """
        Perform frequency-domain ordinary least squares and return the
        corresponding time-domain values for publication.
        """
        output_real = np.asarray(measured_output.spectrum_real, dtype=float)
        output_imag = np.asarray(measured_output.spectrum_imag, dtype=float)
        output_spectrum = output_real + 1j * output_imag

        regressor_spectra = []

        for stream in regressors:
            real = np.asarray(stream.spectrum_real, dtype=float)
            imag = np.asarray(stream.spectrum_imag, dtype=float)
            regressor_spectra.append(real + 1j * imag)

        regressor_matrix = np.column_stack(regressor_spectra)

        parameters = ordinary_least_squares(
            output_spectrum,
            regressor_matrix,
        )

        return OLSResult(
            measured_output=float(measured_output.value),
            regressors=[float(stream.value) for stream in regressors],
            parameters=parameters.tolist(),
        )

    def calculate_ols(self) -> None:
        """
        Programmer-defined frequency-domain least-squares calculations.

        The Fourier and differentiated data required by the equations should
        be obtained using get_fourier_*() and get_differ_*().

        The resulting measured output and regressors published by publish_ols()
        should remain in the time domain.
        """
        m = self.aircraft["geometry"]["mass_kg"] if self.aircraft["geometry"]["mass_kg"] is not None else 1.0
        b = self.aircraft["geometry"]["wing_span_m"] if self.aircraft["geometry"]["wing_span_m"] is not None else 1.0
        S = self.aircraft["geometry"]["wing_area_m2"] if self.aircraft["geometry"]["wing_area_m2"] is not None else 1.0
        c = self.aircraft["geometry"]["mac_m"] if self.aircraft["geometry"]["mac_m"] is not None else 1.0

        Ixx = self.aircraft["inertia"]["Ixx_kgm2"] if self.aircraft["inertia"]["Ixx_kgm2"] is not None else 1.0
        Iyy = self.aircraft["inertia"]["Iyy_kgm2"] if self.aircraft["inertia"]["Iyy_kgm2"] is not None else 1.0
        Izz = self.aircraft["inertia"]["Izz_kgm2"] if self.aircraft["inertia"]["Izz_kgm2"] is not None else 1.0
        Ixy = self.aircraft["inertia"]["Ixy_kgm2"] if self.aircraft["inertia"]["Ixy_kgm2"] is not None else 1.0
        Ixz = self.aircraft["inertia"]["Ixz_kgm2"] if self.aircraft["inertia"]["Ixz_kgm2"] is not None else 1.0
        Iyz = self.aircraft["inertia"]["Iyz_kgm2"] if self.aircraft["inertia"]["Iyz_kgm2"] is not None else 1.0

        p_dot = self.get_differ_stream("imu/gx")
        q_dot = self.get_differ_stream("imu/gy")
        r_dot = self.get_differ_stream("imu/gz")
        p = self.get_fourier_stream("imu/gx")
        q = self.get_fourier_stream("imu/gy")
        r = self.get_fourier_stream("imu/gz")
        ail = self.get_fourier_stream("rcout/ail")
        elv = self.get_fourier_stream("rcout/elv")
        rud = self.get_fourier_stream("rcout/rud")

        if p_dot is not None and p is not None and ail is not None:
            result = self.frequency_ols(
                measured_output=p_dot,
                regressors=[p, ail]
            )
            self.publish_ols(
                ols_name="rol",
                measured_output=result.measured_output,
                regressor=result.regressors,
                parameter=result.parameters,
            )
        if q_dot is not None and q is not None and elv is not None:
            result = self.frequency_ols(
                measured_output=q_dot,
                regressors=[q, elv]
            )
            self.publish_ols(
                ols_name="pit",
                measured_output=result.measured_output,
                regressor=result.regressors,
                parameter=result.parameters,
            )
        if r_dot is not None and r is not None and rud is not None:
            result = self.frequency_ols(
                measured_output=r_dot,
                regressors=[r, rud]
            )
            self.publish_ols(
                ols_name="yaw",
                measured_output=result.measured_output,
                regressor=result.regressors,
                parameter=result.parameters,
            )
        
        nd_p = self.get_fourier_stream("non_dim/p")
        nd_r = self.get_fourier_stream("non_dim/r")
        nd_ail = self.get_fourier_stream("non_dim/ail")
        nd_rud = self.get_fourier_stream("non_dim/rud")
        nd_qr = self.get_fourier_stream("non_dim/qr")
        nd_rpq = self.get_fourier_stream("non_dim/rpq")

        if p_dot is not None and nd_p is not None and nd_r is not None and nd_ail is not None and nd_rud is not None and nd_qr is not None and nd_rpq is not None:
            result = self.frequency_ols(
                measured_output=p_dot,
                regressors=[nd_p, nd_r, nd_ail, nd_rud, nd_qr, nd_rpq]
            )
            result.parameters[0] = (0.5 * S * b ** 2) / Ixx * result.parameters[0]
            result.parameters[1] = (0.5 * S * b ** 2) / Ixx * result.parameters[1]
            result.parameters[2] = (S * b) / Ixx * result.parameters[2]
            result.parameters[3] = (S * b) / Ixx * result.parameters[3]
            self.publish_ols(
                ols_name="nondim/rol",
                measured_output=result.measured_output,
                regressor=result.regressors,
                parameter=result.parameters,
            )


    def _add_ols_publisher(
            self,
            ols_name: str,
        ) -> None:
        """Create a publisher for an OLS result."""
        if ols_name in self.pubs:
            return

        topic = f"{self.OLS_PREFIX}{ols_name}"

        self.pubs[ols_name] = self.create_publisher(
            SysIdLeastSquares,
            topic,
            10,
        )

        self.get_logger().info(
            f"Created ols publisher: {topic}"
        )

    def publish_ols(
        self,
        ols_name: str,
        measured_output: float,
        regressor: list[float],
        parameter: list[float],
        ) -> None:
        """
        Publish a least-squares result.

        Parameters
        ----------
        ols_name : str
            Name of the OLS result. Used as the final component of the
            /sid/ols/ topic.

        measured_output : float
            Time-domain measured output corresponding to this OLS equation.

        regressor : list[float]
            Time-domain regressor values corresponding to the OLS equation.

        parameter : list[float]
            Estimated least-squares parameters.
        """
        if ols_name not in self.pubs:
            self._add_ols_publisher(ols_name)

        msg = SysIdLeastSquares()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.measured_output = float(measured_output)
        msg.regressor = [float(value) for value in regressor]
        msg.parameter = [float(value) for value in parameter]
        self.pubs[ols_name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    ols_node = SIDols()

    while rclpy.ok():
        try:
            rclpy.spin_once(ols_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    ols_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()