#!/usr/bin/env python3

from copy import copy
from typing import Callable

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from drone_interfaces.msg import SysIdDataStream
from ros2_sid.realtime_ols_utils import RecursiveFourierTransform


class SIDFourier(Node):
    FILTER_PREFIX = "/sid/filter/"
    TERMS_PREFIX = "/sid/terms/"
    FOURIER_PREFIX = "/sid/fourier/"

    # If "eff" or "frequencies" are omitted, the default
    # values are used from the frequency config file.
    FILTER_TOPICS = {
        f"{FILTER_PREFIX}imu/gx": {},
        f"{FILTER_PREFIX}imu/gy": {},
        f"{FILTER_PREFIX}imu/gz": {},
        # f"{FILTER_PREFIX}imu/gx": {
        #     "eff": 0.98,
        #     "frequencies": np.array([0.1, 0.2, 0.3]),
        # },
    }
    TERMS_TOPICS = {
        # f"{TERMS_PREFIX}roll/term0": {
        #     "eff": 0.98,
        #     "frequencies": np.array([0.1, 0.2, 0.3]),
        # },
    }
    BLACK_TOPICS = [
        f"{FILTER_PREFIX}rcout/ail",
        f"{FILTER_PREFIX}rcout/elv",
        f"{FILTER_PREFIX}rcout/rud",
        f"{FILTER_PREFIX}pitot/dyn_pres",
        f"{FILTER_PREFIX}pitot/airspeed",
        f"{FILTER_PREFIX}propulsion/prop_speed",
        # f"{TERMS_PREFIX}roll/term0",
    ]

    SYS_ID_DATA_STREAM_TYPE = (
        "drone_interfaces/msg/SysIdDataStream"
    )

    DYNAMIC_SUBS = True

    TOPIC_DISCOVERY_PERIOD = 1.0


    def __init__(self, ns=''):
        super().__init__("sid_fourier")
        self.subs: dict[str, Subscription] = {}
        self.pubs: dict[str, Publisher] = {}

        self.fourier_transforms: dict[str, RecursiveFourierTransform] = {}
        self.fourier_initialized: dict[str, bool] = {}
        self.callbacks: dict[str, Callable[[SysIdDataStream], None],] = {}

        self.add_static_topics()

        if self.DYNAMIC_SUBS:
            self.discovery_timer = self.create_timer(
                self.TOPIC_DISCOVERY_PERIOD,
                self.discover_topics,
            )

        self.get_logger().info(
            "SID Fourier node initialized"
            f" | dynamic_subs={self.DYNAMIC_SUBS}"
        )


    def add_static_topics(self) -> None:
        """Create all statically configured filter and terms streams."""
        for topic, config in self.FILTER_TOPICS.items():
            if not topic.startswith(self.FILTER_PREFIX):
                self.get_logger().error(
                    f"Cannot add invalid filter topic: {topic}"
                )
                continue
            self.add_stream(
                input_topic=topic,
                eff=config.get("eff"),
                frequencies=config.get("frequencies"),
                static=True,
            )

        for topic, config in self.TERMS_TOPICS.items():
            if not topic.startswith(self.TERMS_PREFIX):
                self.get_logger().error(
                    f"Cannot add invalid terms topic: {topic}"
                )
                continue
            self.add_stream(
                input_topic=topic,
                eff=config.get("eff"),
                frequencies=config.get("frequencies"),
                static=True,
            )

    def discover_topics(self) -> None:
        """
        Discover new SysIdDataStream topics under
        /sid/filter/ and /sid/terms/.
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

            if topic_name.startswith(self.FILTER_PREFIX):
                self.add_stream(
                    input_topic=topic_name,
                    static=False,
                )

            elif topic_name.startswith(self.TERMS_PREFIX):
                self.add_stream(
                    input_topic=topic_name,
                    static=False,
                )


    def _is_valid_input_topic(
        self,
        input_topic: str,
        ) -> bool:
        """
        Return True when the topic belongs to
        one of the supported input namespaces.
        """
        return (
            input_topic.startswith(self.FILTER_PREFIX)
            or
            input_topic.startswith(self.TERMS_PREFIX)
        )

    def _get_fourier_topic(
        self,
        input_topic: str,
        ) -> str:
        """
        Convert an input topic into its Fourier output topic.

        Example:
            /sid/filter/imu/gx
                -> /sid/fourier/imu/gx

            /sid/terms/roll/0
                -> /sid/fourier/roll/0
        """
        if input_topic.startswith(self.FILTER_PREFIX):
            stream_name = input_topic[len(self.FILTER_PREFIX):]

        elif input_topic.startswith(self.TERMS_PREFIX):
            stream_name = input_topic[len(self.TERMS_PREFIX):]

        else:
            raise ValueError(
                f"Unsupported Fourier input topic: "
                f"{input_topic}"
            )

        return self.FOURIER_PREFIX + stream_name

    def _make_callback(
        self,
        input_topic: str,
        ) -> Callable[[SysIdDataStream], None]:
        """
        Create a callback associated with one input stream.
        Filter and terms streams use exactly the same Fourier processing.
        """
        def callback(msg: SysIdDataStream,) -> None:
            self.fourier_callback(msg, input_topic)
        return callback

    def add_stream(
        self,
        input_topic: str,
        eff: float | None = None,
        frequencies: np.ndarray | None = None,
        static: bool = False,
        ) -> None:
        """
        Create a Fourier transform and ROS interfaces for an input stream.

        Parameters
        ----------
        input_topic:
            Full /sid/filter/... or /sid/terms/... input topic.

        eff:
            Optional recursive Fourier forgetting factor.

        frequencies:
            Optional Fourier frequencies.

        static:
            True when created from FILTER_TOPICS/TERMS_TOPICS.
            False when dynamically discovered.
        """
        if input_topic in self.subs:
            return
        
        if not self._is_valid_input_topic(input_topic):
            self.get_logger().error(
                f"Cannot add invalid Fourier input topic: "
                f"{input_topic}"
            )
            return

        fourier_topic = self._get_fourier_topic(input_topic)

        for existing_input in self.pubs:
            existing_output = self._get_fourier_topic(existing_input)
            if existing_output == fourier_topic:
                self.get_logger().error(
                    "Cannot create Fourier stream because the output topic "
                    f"'{fourier_topic}' is already used by "
                    f"'{existing_input}'."
                )
                return

        if eff is None and frequencies is None:
            fourier = RecursiveFourierTransform()
        elif frequencies is None:
            fourier = RecursiveFourierTransform(eff=eff)
        elif eff is None:
            fourier = RecursiveFourierTransform(frequencies=frequencies)
        else:
            fourier = RecursiveFourierTransform(
                eff=eff,
                frequencies=frequencies,
            )
        self.fourier_transforms[input_topic] = fourier
        self.fourier_initialized[input_topic] = False

        publisher = self.create_publisher(
            SysIdDataStream,
            fourier_topic,
            10,
        )
        
        subscription = self.create_subscription(
            SysIdDataStream,
            input_topic,
            self._make_callback(input_topic),
            10,
        )
        
        self.pubs[input_topic] = publisher
        self.subs[input_topic] = subscription

        source = "static" if static else "dynamic"
        eff_info = f" | eff={eff}" if eff is not None else ""
        frequency_info = (
            f" | frequencies={len(frequencies)}"
            if frequencies is not None
            else ""
        )
        self.get_logger().info(
            f"Added {source} Fourier stream: "
            f"{input_topic} --> {fourier_topic}"
            f"{eff_info}"
            f"{frequency_info}"
        )


    def fourier_callback(
        self,
        sub_msg: SysIdDataStream,
        input_topic: str,
        ) -> None:
        """Perform and publish one recursive Fourier update."""
        fourier = self.fourier_transforms.get(input_topic)
        if fourier is None:
            return

        dt = float(sub_msg.dt)

        if dt <= 0.0:
            self.get_logger().warning(
                f"Invalid dt={dt:.6f} for "
                f"{input_topic}; skipping sample."
            )
            return

        if not self.fourier_initialized[input_topic]:
            current_time = (
                sub_msg.header.stamp.sec +
                sub_msg.header.stamp.nanosec * 1e-9
            )
            fourier.update_cp_time(current_time)
            self.fourier_initialized[input_topic] = True
        else:
            fourier.update_cp_timestep(dt)

        spectrum = fourier.update_spectrum(sub_msg.value)

        pub_msg = copy(sub_msg)
        pub_msg.frequencies_hz = fourier.frequencies.tolist()
        pub_msg.spectrum_real = (spectrum.real.tolist())
        pub_msg.spectrum_imag = (spectrum.imag.tolist())
        self.pubs[input_topic].publish(pub_msg)


def main(args=None):
    rclpy.init(args=args)
    fourier_node = SIDFourier()

    while rclpy.ok():
        try:
            rclpy.spin_once(fourier_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    fourier_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()