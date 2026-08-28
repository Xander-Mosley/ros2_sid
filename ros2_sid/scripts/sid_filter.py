#!/usr/bin/env python3

from pathlib import Path
from typing import Callable, Dict

import json
import numpy as np

import mavros
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure, Temperature
from std_msgs.msg import Float64, Float64MultiArray, String

from ardupilot_msgs.msg import Pitot, Propulsion, RcIn, RcOut
from drone_interfaces.msg import SysIdDataStream

from ros2_sid.signal_processing_utils import (
    ButterworthLowPass_2Ovdt,
    ButterworthHighPass_4OvdtCascaded,
    ButterworthLowPass_4OvdtCascaded,
)


class SIDFilter(Node):
    FREQUENCY_CONFIG_FILE = (
        Path(__file__).resolve().parents[1]
        / "ros2_sid"
        / "setup"
        / "frequency_config.json"
    )

    FILTER_PREFIX = "/sid/filter/"

    # If "detrending_cutoff" and "pre_filter_cutoff" are
    # omitted, the default cutoffs are used.
    FILTER_TOPICS = {
        "imu/gx": {},
        "imu/gy": {},
        "imu/gz": {},

        "rcout/ail": {},
        "rcout/elv": {},
        "rcout/rud": {},

        "pitot/dyn_pres": {},
        "pitot/airspeed": {},

        "propulsion/prop_speed": {},

        # "imu/gx": {
        #     "detrending_cutoff": 0.05,
        #     "pre_filter_cutoff": 5.0,
        # },
    }

    USE_PRE_FILTER = True
    USE_DETRENDING = True

    TIMESTAMP_ROLLOVER_PERIOD = 1.0


    def __init__(self, ns=''):
        super().__init__('sid_filter')
        self.load_frequency_config()
        alias_frequency = self.frequency_config["alias_frequency_hz"]
        minimum_frequency = self.frequency_config["minimum_frequency_hz"]
        self.default_pre_filter_cutoff = alias_frequency
        self.default_detrending_cutoff = minimum_frequency

        self.pubs: dict[str, Publisher] = {}

        self.previous_timestamps: dict[str, float] = {}
        self.filters: Dict[str, dict] = {}
        
        self.setup_streams()
        self.setup_subs()

        pre_filter_cutoff = (
            f", so default_pre_filter_cutoff={self.default_pre_filter_cutoff:.3f} Hz"
            if self.USE_PRE_FILTER
            else ""
        )
        detrending_cutoff = (
            f", so default_detrending_cutoff={self.default_detrending_cutoff:.3f} Hz"
            if self.USE_DETRENDING
            else ""
        )
        self.get_logger().info(
            f"SID filter node initialized"
            f" | use_pre_filter={self.USE_PRE_FILTER}"
            f"{pre_filter_cutoff}"
            f" | use_detrending={self.USE_DETRENDING}"
            f"{detrending_cutoff}"
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

        if "minimum_frequency_hz" not in self.frequency_config:
            raise KeyError(
                "frequency_config.json is missing "
                "'minimum_frequency_hz'."
            )
        if self.frequency_config["minimum_frequency_hz"] <= 0:
            raise ValueError(
                "'minimum_frequency_hz' must be greater than zero."
            )


    def _create_stream(
            self,
            stream_name: str,
            pre_filter_cutoff: float,
            detrending_cutoff: float,
            ) -> None:
            """ Create the filtering chain and publisher for one stream. """
            filter_topic = self.FILTER_PREFIX + stream_name
            filters = {}

            if self.USE_PRE_FILTER:
                filters["pre_filter"] = ButterworthLowPass_2Ovdt(pre_filter_cutoff)
            else:
                filters["pre_filter"] = None

            if self.USE_DETRENDING:
                filters["high_pass"] = ButterworthHighPass_4OvdtCascaded(detrending_cutoff)
                filters["low_pass"] = ButterworthLowPass_4OvdtCascaded(detrending_cutoff)
            else:
                filters["high_pass"] = None
                filters["low_pass"] = None

            publisher = self.create_publisher(
                SysIdDataStream,
                filter_topic,
                10
            )

            self.filters[stream_name] = filters
            self.pubs[stream_name] = publisher
            
            pre_filter_status = (
                f"{pre_filter_cutoff:.3f} Hz pre-filter"
                if self.USE_PRE_FILTER
                else "pre-filter disabled"
            )
            detrending_status = (
                f"{detrending_cutoff:.3f} Hz detrending"
                if self.USE_DETRENDING
                else "detrending disabled"
            )
            self.get_logger().info(
                f"Added filter stream: "
                f"{filter_topic}"
                f" | {pre_filter_status}"
                f" | {detrending_status}"
            )

    def setup_streams(self):
        """ Create the filters and publisher for every output stream. """
        for stream_name, config in self.FILTER_TOPICS.items():
            pre_filter_cutoff = config.get("pre_filter_cutoff", self.default_pre_filter_cutoff)
            detrending_cutoff = config.get("detrending_cutoff", self.default_detrending_cutoff)
            self._create_stream(
                stream_name=stream_name,
                pre_filter_cutoff=pre_filter_cutoff,
                detrending_cutoff=detrending_cutoff,
                )


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
        stream_name: str,
        header,
        value: float,
        dt: float
        ) -> None:
        """Create and publish a SysIdDataStream message."""
        if stream_name not in self.filters:
            self.get_logger().error(
                f"Cannot publish invalid stream name: "
                f"{stream_name}"
            )
            return

        if dt <= 0.0:
            self.get_logger().warning(
                f"Invalid dt={dt:.6f} for "
                f"{stream_name}; skipping sample."
            )
            return

        filters = self.filters[stream_name]
        filtered_value = value

        if self.USE_PRE_FILTER:
            pre_filter = filters["pre_filter"]
            if pre_filter is None:
                return
            try:
                filtered_value = pre_filter.update(value, dt)
            except Exception as error:
                self.get_logger().error(
                    f"Pre-filter error on "
                    f"{stream_name}: {error}"
                )
                return

        new_value = filtered_value
        new_trend = filtered_value

        if self.USE_DETRENDING:
            high_pass = filters["high_pass"]
            low_pass = filters["low_pass"]
            if high_pass is None or low_pass is None:
                return
            try:
                new_value = high_pass.update(filtered_value, dt)
                new_trend = low_pass.update(filtered_value, dt)
            except Exception as error:
                self.get_logger().error(
                    f"Detrending filter error on "
                    f"{stream_name}: {error}"
                )
                return

        pub_msg: SysIdDataStream = SysIdDataStream()
        pub_msg.header = header
        pub_msg.dt = dt
        pub_msg.value = new_value
        pub_msg.trend = new_trend
        self.pubs[stream_name].publish(pub_msg)

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

        self.pitot_sub: Subscription = self.create_subscription(
            Pitot,
            '/ap/pitot',
            self.pitot_callback,
            qos_profile=SENSOR_QOS
        )

        self.propulsion_sub: Subscription = self.create_subscription(
            Propulsion,
            '/ap/propulsion',
            self.propulsion_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, sub_msg: Imu) -> None:
        """
        Input:
            /ap/imu/experimental/data

        Outputs:
            /sid/filter/imu/gx
            /sid/filter/imu/gy
            /sid/filter/imu/gz
        """
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        dt = self._get_dt(
            stream_name="imu",
            header=sub_msg.header
        )

        self._publish_stream(
            stream_name="imu/gx",
            header=sub_msg.header,
            value=sub_msg.angular_velocity.x,
            dt=dt
        )
        self._publish_stream(
            stream_name="imu/gy",
            header=sub_msg.header,
            value=sub_msg.angular_velocity.y,
            dt=dt
        )
        self._publish_stream(
            stream_name="imu/gz",
            header=sub_msg.header,
            value=sub_msg.angular_velocity.z,
            dt=dt
        )

    def rcout_callback(self, sub_msg: RcOut) -> None:
        """
        Input:
            /ap/rcout

        Outputs:
            sid/filter/rcout/ail
            sid/filter/rcout/elv
            sid/filter/rcout/rud
        """
        dt = self._get_dt(
            stream_name="rcout",
            header=sub_msg.header
        )

        self._publish_stream(
            stream_name="rcout/ail",
            header=sub_msg.header,
            value=float(sub_msg.values[0]) - 1500.0,
            dt=dt
        )
        self._publish_stream(
            stream_name="rcout/elv",
            header=sub_msg.header,
            value=float(sub_msg.values[1]) - 1500.0,
            dt=dt
        )
        self._publish_stream(
            stream_name="rcout/rud",
            header=sub_msg.header,
            value=float(sub_msg.values[3]) - 1500.0,
            dt=dt
        )

    def pitot_callback(self, sub_msg: Pitot) -> None:
        """
        Input:
            /ap/pitot

        Outputs:
            sid/filter/pitot/dyn_pres
            sid/filter/pitot/airspeed
        """
        dt = self._get_dt(
            stream_name="pitot",
            header=sub_msg.header
        )

        self._publish_stream(
            stream_name="pitot/dyn_pres",
            header=sub_msg.header,
            value=sub_msg.dynamic_pressure,
            dt=dt
        )
        self._publish_stream(
            stream_name="pitot/airspeed",
            header=sub_msg.header,
            value=sub_msg.true_airspeed,    # Might want calibarated_airspeed
            dt=dt
        )

    def propulsion_callback(self, sub_msg: Propulsion) -> None:
        """
        Input:
            /ap/propulsion

        Outputs:
            sid/filter/propulsion/prop_speed
        """
        dt = self._get_dt(
            stream_name="propulsion",
            header=sub_msg.header
        )

        self._publish_stream(
            stream_name="propulsion/prop_speed",
            header=sub_msg.header,
            value=sub_msg.rpm / 30 * np.pi,
            dt=dt
        )


def main(args=None):
    rclpy.init(args=args)
    filter_node = SIDFilter()

    while rclpy.ok():
        try:
            rclpy.spin_once(filter_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()