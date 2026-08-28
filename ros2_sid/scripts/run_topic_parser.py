#!/usr/bin/env python3

import json
from pathlib import Path

import mavros
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription

from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure, Temperature
from std_msgs.msg import Float64, Float64MultiArray, String

from ardupilot_msgs.msg import Pitot, Propulsion, RcIn, RcOut
from drone_interfaces.msg import SysIdDataStream

from ros2_sid.signal_processing_utils import ButterworthLowPass_2Ovdt


class SIDParser(Node):
    FREQUENCY_CONFIG_FILE = (
        Path(__file__).resolve().parents[1]
        / "ros2_sid"
        / "setup"
        / "frequency_config.json"
    )

    PARSER_PREFIX = "/sid/parser/"

    LOWPASS_FILTER = True

    TIMESTAMP_ROLLOVER_PERIOD = 1.0


    def __init__(self, ns=''):
        super().__init__('sid_parser')
        self.load_frequency_config()
        alias_frequency = self.frequency_config["alias_frequency_hz"]
        self.default_cutoff_frequency = alias_frequency

        self.previous_timestamps = {}
        
        self.setup_streams()
        self.setup_subs()

        self.get_logger().info(
            "SID parser node initialized"
        )

    def load_frequency_config(self) -> None:
        """Load frequency_config.json."""
        try:
            with self.FREQUENCY_CONFIG_FILE.open("r", encoding="utf-8") as file:
                self.frequency_config = json.load(file)
        except FileNotFoundError:
            self.get_logger().error(
                f"Frequency config file not found: "
                f"{self.FREQUENCY_CONFIG_FILE}"
            )
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f"Invalid JSON in frequency config file: {error}"
            )
        except OSError as error:
            self.get_logger().error(
                f"Unable to read frequency config file: {error}"
            )

        if "alias_frequency_hz" not in self.frequency_config:
            raise KeyError(
                "frequency_config.json is missing "
                "'alias_frequency_hz'."
            )
        if self.frequency_config["alias_frequency_hz"] <= 0:
            raise ValueError(
                "'alias_frequency_hz' must be greater than zero."
            )


    def _create_stream(
            self,
            topic: str,
            cutoff_frequency: float | None = None
            ) -> dict:
            """
            Create a parser output stream.

            Parameters
            ----------
            topic : str
                ROS topic on which the SysIdDataStream will be published.

            cutoff_frequency : float | None
                Cutoff frequency for the low-pass filter.
                None means no filter is created.
            """
            publisher = self.create_publisher(
                SysIdDataStream,
                topic,
                10
            )

            signal_filter = None
            if cutoff_frequency is not None:
                signal_filter = ButterworthLowPass_2Ovdt(cutoff_frequency)

            return {
                "publisher": publisher,
                "filter": signal_filter
            }

    def setup_streams(self):
        upper_cutoff = self.default_cutoff_frequency

        self.imu_streams = {
            "gx": self._create_stream(
                topic=f"{self.PARSER_PREFIX}imu/gx",
                cutoff_frequency=upper_cutoff
            ),
            "gy": self._create_stream(
                topic=f"{self.PARSER_PREFIX}imu/gy",
                cutoff_frequency=upper_cutoff
            ),
            "gz": self._create_stream(
                topic=f"{self.PARSER_PREFIX}imu/gz",
                cutoff_frequency=upper_cutoff
            ),
        }

        self.rcout_streams = {
            "ail": self._create_stream(
                topic=f"{self.PARSER_PREFIX}rcout/ail",
                cutoff_frequency=upper_cutoff
            ),
            "elv": self._create_stream(
                topic=f"{self.PARSER_PREFIX}rcout/elv",
                cutoff_frequency=upper_cutoff
            ),
            "rud": self._create_stream(
                topic=f"{self.PARSER_PREFIX}rcout/rud",
                cutoff_frequency=upper_cutoff
            ),
        }


    def _get_dt(self, stream_name: str, header) -> float:
        """
        Calculate the time difference between consecutive messages.
        Each input topic maintains its own previous timestamp.

        Parameters
        ----------
        stream_name : str
            Identifier for the input stream.

        header :
            ROS message header containing the timestamp.

        Returns
        -------
        float
            Time difference in seconds.
        """
        current_time = float(header.stamp.nanosec) * 1e-9
        previous_time = self.previous_timestamps.get(stream_name, 0.0)

        dt = current_time - previous_time
        if dt <= 0.0:
            dt += self.TIMESTAMP_ROLLOVER_PERIOD

        self.previous_timestamps[stream_name] = current_time

        return dt

    def _publish_stream(
        self,
        stream: dict,
        header,
        value: float,
        dt: float
        ) -> None:
        """
        Create and publish a SysIdDataStream message.

        Filtering is performed here so that individual callbacks do
        not need to duplicate filtering and message construction code.
        """
        pub_msg: SysIdDataStream = SysIdDataStream()

        pub_msg.header = header
        pub_msg.dt = dt

        signal_filter = stream["filter"]

        if self.LOWPASS_FILTER and signal_filter is not None:
            pub_msg.value = signal_filter.update(
                value,
                dt
            )
        else:
            pub_msg.value = value

        stream["publisher"].publish(pub_msg)

    def setup_subs(self):
        self.imu_sub: Subscription = self.create_subscription(
            Imu,
            '/ap/imu/experimental/data',
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )

        self.rcout_sub: Subscription = self.create_subscription(
            RcOut,
            '/ap/rcout',
            self.rcout_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, sub_msg: Imu) -> None:
        """
        Input:
            /ap/imu/experimental/data

        Outputs:
            /sid/parser/imu/gx
            /sid/parser/imu/gy
            /sid/parser/imu/gz
        """
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        dt = self._get_dt(
            stream_name="imu",
            header=sub_msg.header
        )

        self._publish_stream(
            stream=self.imu_streams["gx"],
            header=sub_msg.header,
            value=sub_msg.angular_velocity.x,
            dt=dt
        )
        self._publish_stream(
            stream=self.imu_streams["gy"],
            header=sub_msg.header,
            value=sub_msg.angular_velocity.y,
            dt=dt
        )
        self._publish_stream(
            stream=self.imu_streams["gz"],
            header=sub_msg.header,
            value=sub_msg.angular_velocity.z,
            dt=dt
        )

    def rcout_callback(self, sub_msg: RcOut) -> None:
        """
        Input:
            /ap/rcout

        Outputs:
            sid/parser/rcout/ail
            sid/parser/rcout/elv
            sid/parser/rcout/rud
        """
        dt = self._get_dt(
            stream_name="rcout",
            header=sub_msg.header
        )

        self._publish_stream(
            stream=self.rcout_streams["ail"],
            header=sub_msg.header,
            value=float(sub_msg.values[0]) - 1500.0,
            dt=dt
        )
        self._publish_stream(
            stream=self.rcout_streams["elv"],
            header=sub_msg.header,
            value=float(sub_msg.values[1]) - 1500.0,
            dt=dt
        )
        self._publish_stream(
            stream=self.rcout_streams["rud"],
            header=sub_msg.header,
            value=float(sub_msg.values[3]) - 1500.0,
            dt=dt
        )


def main(args=None):
    rclpy.init(args=args)
    parser_node = SIDParser()

    while rclpy.ok():
        try:
            rclpy.spin_once(parser_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    parser_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()