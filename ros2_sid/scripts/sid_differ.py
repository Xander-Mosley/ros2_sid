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

from drone_interfaces.msg import SysIdDataStream
from ros2_sid.signal_processing_utils import (
    poly_diff,
    ButterworthLowPass_2Ovdt,
)
from ros2_sid.realtime_ols_utils import CircularBuffer


DIFF_BUFFER_SIZE = 5
DIFF_POLYORDER = 3


@dataclass
class DifferStreamState:
    """Differentiation state for one signal stream."""
    time_buffer: CircularBuffer
    value_buffer: CircularBuffer
    trend_buffer: CircularBuffer
    sample_count: int = 0

    @classmethod
    def create(cls) -> "DifferStreamState":
        return cls(
            time_buffer=CircularBuffer(DIFF_BUFFER_SIZE),
            value_buffer=CircularBuffer(DIFF_BUFFER_SIZE),
            trend_buffer=CircularBuffer(DIFF_BUFFER_SIZE),
        )

    def add_sample(
        self,
        time: float,
        value: float,
        trend: float,
        ) -> None:

        self.time_buffer.add(time)
        self.value_buffer.add(value)
        self.trend_buffer.add(trend)

        self.sample_count = min(
            self.sample_count + 1,
            DIFF_BUFFER_SIZE,
        )

    @property
    def ready(self) -> bool:
        return self.sample_count >= DIFF_POLYORDER + 1

    def differentiate(self) -> tuple[float, float]:
        if not self.ready:
            raise RuntimeError(
                "Not enough samples for differentiation."
            )

        time = self.time_buffer.get_all()
        value = self.value_buffer.get_all()
        trend = self.trend_buffer.get_all()

        # TODO: Check that eval_point should be "end"
        derivative_value = poly_diff(
            time=time,
            data=value,
            polyorder=DIFF_POLYORDER,
            eval_point="end",
        )
        derivative_trend = poly_diff(
            time=time,
            data=trend,
            polyorder=DIFF_POLYORDER,
            eval_point="end",
        )

        return derivative_value, derivative_trend


class SIDDiffer(Node):
    FREQUENCY_CONFIG_FILE = (
        Path(__file__).resolve().parents[1]
        / "ros2_sid"
        / "setup"
        / "frequency_config.json"
    )

    FOURIER_PREFIX = "/sid/fourier/"
    DIFFER_PREFIX = "/sid/differ/"

    # If "cutoff_frequency_hz" is omitted, the default cutoff is used.
    FOURIER_TOPICS = {
        f"{FOURIER_PREFIX}imu/gx": {},
        f"{FOURIER_PREFIX}imu/gy": {},
        f"{FOURIER_PREFIX}imu/gz": {},
        # f"{FOURIER_PREFIX}imu/gx": {
        #     "cutoff_frequency_hz": 0.0,
        # },
    }
    BLACK_TOPICS = [
        # f"{FILTER_PREFIX}rcout/ail",
    ]

    SYS_ID_DATA_STREAM_TYPE = (
        "drone_interfaces/msg/SysIdDataStream"
    )

    DYNAMIC_SUBS = False

    TOPIC_DISCOVERY_PERIOD = 1.0
    
    USE_POSTFILTER = True


    def __init__(self, ns=''):
        super().__init__("sid_differ")
        self.load_frequency_config()
        alias_frequency = self.frequency_config["alias_frequency_hz"]
        self.default_cutoff_frequency = alias_frequency

        self.subs: dict[str, Subscription] = {}
        self.pubs: dict[str, Publisher] = {}

        self.stream_states: dict[str, DifferStreamState] = {}
        self.filters: Dict[str, dict] = {}

        self.add_static_topics()

        if self.DYNAMIC_SUBS:
            self.discovery_timer = self.create_timer(
                self.TOPIC_DISCOVERY_PERIOD,
                self.discover_topics,
            )

        default_cutoff = (
            f", so default_cutoff_frequency={self.default_cutoff_frequency:.3f} Hz"
            if self.USE_POSTFILTER
            else ""
        )
        self.get_logger().info(
            "SID differ node initialized"
            f" | dynamic_subs={self.DYNAMIC_SUBS}"
            f" | use_postfilter={self.USE_POSTFILTER}"
            f"{default_cutoff}"
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


    def add_static_topics(self) -> None:
        """Create topics that always exist."""
        for topic, config in self.FOURIER_TOPICS.items():
            if not topic.startswith(self.FOURIER_PREFIX):
                self.get_logger().error(
                    f"Cannot add invalid Fourier topic: {topic}"
                )
                continue
            cutoff_frequency = config.get("cutoff_frequency_hz")
            if cutoff_frequency is None:
                cutoff_frequency = self.default_cutoff_frequency

            self.add_stream(
                fourier_topic=topic,
                cutoff_frequency=cutoff_frequency,
                static=True,
            )

    def discover_topics(self):
        """Discover new SysIdDataStream topics under /sid/fourier/."""
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

            if not topic_name.startswith(self.FOURIER_PREFIX):
                continue
            config = self.FOURIER_TOPICS.get(topic_name, {})

            cutoff_frequency = config.get("cutoff_frequency_hz")
            if cutoff_frequency is None:
                cutoff_frequency = self.default_cutoff_frequency

            self.add_stream(
                fourier_topic=topic_name,
                cutoff_frequency=cutoff_frequency,
                static=False,
            )


    def _make_callback(
        self,
        fourier_topic: str,
        ) -> Callable[[SysIdDataStream], None]:
        """Create a typed callback for a fourier topic."""
        def callback(msg: SysIdDataStream) -> None:
            self.differ_callback(msg, fourier_topic)
        return callback

    def add_stream(
        self,
        fourier_topic: str,
        cutoff_frequency: float,
        static: bool = False,
        ):
        """
        Create the state, filters, subscriber, and
        publisher for one Fourier stream.
        """
        if fourier_topic in self.subs:
            return
        
        if not fourier_topic.startswith(self.FOURIER_PREFIX):
            self.get_logger().error(
                f"Cannot add invalid fourier topic: "
                f"{fourier_topic}"
            )
            return

        stream_name = fourier_topic[len(self.FOURIER_PREFIX):]
        differ_topic = (self.DIFFER_PREFIX + stream_name)

        self.stream_states[fourier_topic] = DifferStreamState.create()

        if self.USE_POSTFILTER:
            try:
                self.filters[fourier_topic] = {
                    "value": ButterworthLowPass_2Ovdt(cutoff_frequency),
                    "trend": ButterworthLowPass_2Ovdt(cutoff_frequency),
                }
            except Exception as error:
                self.stream_states.pop(fourier_topic, None)
                self.filters.pop(fourier_topic, None)
                self.get_logger().error(
                    f"Failed to create filter for "
                    f"{fourier_topic}: {error}"
                )
                return

        publisher = self.create_publisher(
            SysIdDataStream,
            differ_topic,
            10,
        )

        subscription = self.create_subscription(
            SysIdDataStream,
            fourier_topic,
            self._make_callback(fourier_topic),
            10,
        )

        self.pubs[fourier_topic] = publisher
        self.subs[fourier_topic] = subscription

        source = "static" if static else "dynamic"
        filter_status = (
            f"{cutoff_frequency:.3f} Hz"
            if self.USE_POSTFILTER
            else "disabled"
        )
        self.get_logger().info(
            f"Added {source} differ stream: "
            f"{fourier_topic} --> {differ_topic}"
            f" |  {filter_status} filter"
        )


    def differ_callback(
        self,
        sub_msg: SysIdDataStream,
        fourier_topic: str,
        ) -> None:
        """Process and publish one Fourier data-stream sample."""
        dt = float(sub_msg.dt)

        if dt <= 0.0:
            self.get_logger().warning(
                f"Invalid dt={dt:.6f} for "
                f"{fourier_topic}; skipping sample."
            )
            return

        state = self.stream_states[fourier_topic]

        timestamp = (
            float(sub_msg.header.stamp.sec)
            + float(sub_msg.header.stamp.nanosec) * 1e-9
        )

        state.add_sample(
            time=timestamp,
            value=float(sub_msg.value),
            trend=float(sub_msg.trend),
        )

        if not state.ready:
            return

        try:
            value_derivative, trend_derivative = state.differentiate()
        except Exception as error:
            self.get_logger().error(
                f"Differentiation error on "
                f"{fourier_topic}: {error}"
            )
            return

        if self.USE_POSTFILTER:
            try:
                value_derivative = self.filters[fourier_topic]["value"].update(value_derivative, dt)
                trend_derivative = self.filters[fourier_topic]["trend"].update(trend_derivative, dt)
            except Exception as error:
                self.get_logger().error(
                    f"Filter error on "
                    f"{fourier_topic}: {error}"
                )
                return

        frequencies = np.asarray(
            sub_msg.frequencies_hz,
            dtype=float,
        )
        spectrum_real = np.asarray(
            sub_msg.spectrum_real,
            dtype=float,
        )
        spectrum_imag = np.asarray(
            sub_msg.spectrum_imag,
            dtype=float,
        )

        if not (frequencies.size == spectrum_real.size == spectrum_imag.size):
            self.get_logger().error(
                f"Invalid Fourier spectrum on "
                f"{fourier_topic}: "
                f"frequencies={frequencies.size}, "
                f"real={spectrum_real.size}, "
                f"imag={spectrum_imag.size}"
            )
            return

        spectrum = (spectrum_real + 1j * spectrum_imag) # X = real + j*imag
        differentiated_spectrum = (1j * 2.0 * np.pi * frequencies * spectrum)   # dX/dt = j * (2*pi*f) * X

        pub_msg = copy(sub_msg)
        pub_msg.value = value_derivative
        pub_msg.trend = trend_derivative
        pub_msg.spectrum_real = differentiated_spectrum.real.tolist()
        pub_msg.spectrum_imag = differentiated_spectrum.imag.tolist()
        self.pubs[fourier_topic].publish(pub_msg)


def main(args=None):
    rclpy.init(args=args)
    differ_node = SIDDiffer()

    while rclpy.ok():
        try:
            rclpy.spin_once(differ_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    differ_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()