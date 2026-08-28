#!/usr/bin/env python3

from copy import copy
from pathlib import Path
from typing import Callable, Dict

import json

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from drone_interfaces.msg import SysIdDataStream
from ros2_sid.signal_processing_utils import (
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

    PARSER_PREFIX = "/sid/parser/"
    FILTER_PREFIX = "/sid/filter/"

    # If "cutoff_frequency_hz" is omitted, the default cutoff is used.
    PARSER_TOPICS = {
        f"{PARSER_PREFIX}imu/gx": {},
        f"{PARSER_PREFIX}imu/gy": {},
        f"{PARSER_PREFIX}imu/gz": {},

        f"{PARSER_PREFIX}rcout/ail": {},
        f"{PARSER_PREFIX}rcout/elv": {},
        f"{PARSER_PREFIX}rcout/rud": {},

        # f"{PARSER_PREFIX}imu/gx": {
        #     "cutoff_frequency_hz": 0.0,
        # },
    }

    SYS_ID_DATA_STREAM_TYPE = (
        "drone_interfaces/msg/SysIdDataStream"
    )

    DYNAMIC_SUBS = False
    TOPIC_DISCOVERY_PERIOD = 1.0


    def __init__(self, ns=''):
        super().__init__("sid_filter")
        self.load_frequency_config()
        minimum_frequency = self.frequency_config["minimum_frequency_hz"]
        self.default_cutoff_frequency = 0.5 * minimum_frequency
        
        self.filters: Dict[str, dict] = {}

        self.subs: dict[str, Subscription] = {}
        self.pubs: dict[str, Publisher] = {}

        self.add_static_topics()

        if self.DYNAMIC_SUBS:
            self.discovery_timer = self.create_timer(
                self.TOPIC_DISCOVERY_PERIOD,
                self.discover_topics,
            )

        self.get_logger().info(
            "SID filter node initialized"
            f" | dynamic_subs={self.DYNAMIC_SUBS}"
        )

        self.get_logger().info(
            f"Default cutoff frequency: "
            f"{self.default_cutoff_frequency:.3f} Hz"
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

        if "minimum_frequency_hz" not in self.frequency_config:
            raise KeyError(
                "frequency_config.json is missing "
                "'minimum_frequency_hz'."
            )
        if self.frequency_config["minimum_frequency_hz"] <= 0:
            raise ValueError(
                "'minimum_frequency_hz' must be greater than zero."
            )


    def add_static_topics(self) -> None:
        """
        Create topics that always exist.
        """
        for topic, config in self.PARSER_TOPICS.items():
            cutoff_frequency = config.get("cutoff_frequency_hz")
            if cutoff_frequency is None:
                cutoff_frequency = self.default_cutoff_frequency

            self.add_stream(
                parser_topic=topic,
                cutoff_frequency=cutoff_frequency,
                static=True,
            )

    def discover_topics(self):
        """
        Discover new SysIdDataStream topics under /sid/parser/.
        """
        topic_list = self.get_topic_names_and_types()

        for topic_name, topic_types in topic_list:
            if not topic_name.startswith(self.PARSER_PREFIX):
                continue
            if topic_name in self.subs:
                continue
            if not topic_types:
                continue

            if self.SYS_ID_DATA_STREAM_TYPE not in topic_types:
                self.get_logger().warning(
                    f"Ignoring parser topic '{topic_name}'. "
                    f"Expected {self.SYS_ID_DATA_STREAM_TYPE}, "
                    f"found {topic_types}"
                )
                continue

            config = self.PARSER_TOPICS.get(
                topic_name,
                {}
            )
            cutoff_frequency = config.get(
                "cutoff_frequency_hz"
            )

            if cutoff_frequency is None:
                cutoff_frequency = self.default_cutoff_frequency

            self.add_stream(
                parser_topic=topic_name,
                cutoff_frequency=cutoff_frequency,
                static=False,
            )


    def _make_callback(
        self,
        parser_topic: str,
        ) -> Callable[[SysIdDataStream], None]:
        """Create a typed callback for a parser topic."""
        def callback(msg: SysIdDataStream) -> None:
            self.filter_callback(msg, parser_topic)
        return callback

    def add_stream(
        self,
        parser_topic: str,
        cutoff_frequency: float,
        static: bool = False,
        ):
        """
        Create the subscriber, filter, and publisher for one stream.
        """
        if parser_topic in self.subs:
            return
        if not parser_topic.startswith(self.PARSER_PREFIX):
            self.get_logger().error(
                f"Cannot add invalid parser topic: {parser_topic}"
            )
            return

        stream_name = parser_topic[len(self.PARSER_PREFIX):]
        filter_topic = (self.FILTER_PREFIX + stream_name)

        try:
            self.filters[parser_topic] = {
                "high_pass": ButterworthHighPass_4OvdtCascaded(cutoff_frequency),
                "low_pass": ButterworthLowPass_4OvdtCascaded(cutoff_frequency),
            }
        except Exception as error:
            self.get_logger().error(
                f"Failed to create filter for "
                f"{parser_topic}: {error}"
            )
            return

        publisher = self.create_publisher(
            SysIdDataStream,
            filter_topic,
            10,
        )

        subscription = self.create_subscription(
            SysIdDataStream,
            parser_topic,
            self._make_callback(parser_topic),
            10,
        )

        self.pubs[parser_topic] = publisher
        self.subs[parser_topic] = subscription

        source = "static" if static else "dynamic"
        self.get_logger().info(
            f"Added {source} filter stream: "
            f"{parser_topic} --> {filter_topic}"
            f" | {cutoff_frequency:.3f} Hz detrend"
        )


    def filter_callback(
        self,
        sub_msg: SysIdDataStream,
        parser_topic: str,
        ):
        """
        Filter one SysIdDataStream message.
        """
        dt = float(sub_msg.dt)

        if dt <= 0.0:
            self.get_logger().warning(
                f"Invalid dt={dt:.6f} for "
                f"{parser_topic}; skipping sample."
            )
            return

        original_value = sub_msg.value

        try:
            high_pass = self.filters[parser_topic]["high_pass"].update(
                    original_value,
                    dt,
                )
            low_pass = self.filters[parser_topic]["low_pass"].update(
                    original_value,
                    dt,
                )
        except Exception as error:
            self.get_logger().error(
                f"Filter error on {parser_topic}: {error}"
            )
            return

        pub_msg = copy(sub_msg)
        pub_msg.value = high_pass
        pub_msg.trend = low_pass
        self.pubs[parser_topic].publish(pub_msg)


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