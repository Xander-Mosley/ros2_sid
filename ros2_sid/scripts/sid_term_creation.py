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


class SIDTerms(Node):
    CURRENT_AIRCRAFT_FILE = (
        Path(__file__).resolve().parents[1]
        / "ros2_sid"
        / "setup"
        / "aircraft_library"
        / "current_aircraft.json"
    )

    FILTER_PREFIX = "/sid/filter/"
    DIFFER_PREFIX = "/sid/differ/"
    TERMS_PREFIX = "/sid/terms/"

    FILTER_TOPICS = {
        f"{FILTER_PREFIX}imu/gx": None,
        f"{FILTER_PREFIX}imu/gy": None,
        f"{FILTER_PREFIX}imu/gz": None,

        f"{FILTER_PREFIX}rcout/ail": None,
        f"{FILTER_PREFIX}rcout/elv": None,
        f"{FILTER_PREFIX}rcout/rud": None,

        f"{FILTER_PREFIX}pitot/dyn_pres": None,
        f"{FILTER_PREFIX}pitot/airspeed": None,

        f"{FILTER_PREFIX}propulsion/prop_speed": None,
    }
    DIFFER_TOPICS = {
        f"{DIFFER_PREFIX}imu/gx": None,
        f"{DIFFER_PREFIX}imu/gy": None,
        f"{DIFFER_PREFIX}imu/gz": None,
    }
    BLACK_TOPICS = [
        # f"{FILTER_PREFIX}rcout/ail",
        # f"{DIFFER_PREFIX}imu/gx",
    ]

    SYS_ID_DATA_STREAM_TYPE = (
        "drone_interfaces/msg/SysIdDataStream"
    )

    DYNAMIC_SUBS = False

    TOPIC_DISCOVERY_PERIOD = 1.0
    TERMS_PUBLISH_PERIOD = 0.04

    TIMESTAMP_ROLLOVER_PERIOD = 1.0


    def __init__(self, ns=''):
        super().__init__("sid_terms")
        self.subs: Dict[str, Subscription] = {}
        self.pubs: Dict[str, Publisher] = {}
        
        self.filter_streams: Dict[str, SysIdDataStream] = {}
        self.differ_streams: Dict[str, SysIdDataStream] = {}

        self.previous_publish_time: Dict[str, float] = {}

        self.aircraft: Dict = {}
        self.load_current_aircraft()

        self.add_static_topics()

        if self.DYNAMIC_SUBS:
            self.discovery_timer = self.create_timer(
                self.TOPIC_DISCOVERY_PERIOD,
                self.discover_topics,
            )

        self.terms_timer = self.create_timer(
            self.TERMS_PUBLISH_PERIOD,
            self.calculate_terms,
        )

        self.get_logger().info(
            "SID terms node initialized"
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
        """Create all statically configured filter and differ streams."""
        for topic, stream_name in self.FILTER_TOPICS.items():
            if not topic.startswith(self.FILTER_PREFIX):
                self.get_logger().error(
                    f"Cannot add invalid filter topic: {topic}"
                )
                continue
            if stream_name is None:
                stream_name = topic.removeprefix(self.FILTER_PREFIX)
            self.add_filter_subscription(
                filter_topic=topic,
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

    def discover_topics(self):
        """
        Discover new SysIdDataStream topics under
        /sid/filter/ and /sid/differ/.
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
                configured_name = self.FILTER_TOPICS.get(topic_name)
                if configured_name is None:
                    stream_name = topic_name.removeprefix(self.FILTER_PREFIX)
                else:
                    stream_name = configured_name

                self.add_filter_subscription(
                    filter_topic=topic_name,
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


    def _make_filter_callback(
        self,
        stream_name: str,
        ) -> Callable[[SysIdDataStream], None]:
        """Create a callback for a filter stream."""
        def callback(msg: SysIdDataStream) -> None:
            self.filter_callback(msg, stream_name)
        return callback

    def _make_differ_callback(
        self,
        stream_name: str,
        ) -> Callable[[SysIdDataStream], None]:
        """Create a callback for a differentiated stream."""
        def callback(msg: SysIdDataStream) -> None:
            self.differ_callback(msg, stream_name)
        return callback

    def add_filter_subscription(
        self,
        filter_topic: str,
        stream_name: str,
        static: bool = False,
        ) -> None:
        """Create a subscription to a filter stream."""
        if filter_topic in self.subs:
            return

        if not filter_topic.startswith(self.FILTER_PREFIX):
            self.get_logger().error(
                f"Cannot add invalid filter topic: "
                f"{filter_topic}"
            )
            return

        subscription = self.create_subscription(
            SysIdDataStream,
            filter_topic,
            self._make_filter_callback(stream_name),
            10,
        )

        self.subs[filter_topic] = subscription

        source = "static" if static else "dynamic"
        self.get_logger().info(
            f"Added {source} filter subscription named {stream_name}"
            f" | {filter_topic}"
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


    def filter_callback(
        self,
        msg: SysIdDataStream,
        stream_name: str,
        ) -> None:
        """
        Store the most recent filtered message.
        The message is copied so the stored value is independent of the
        ROS callback message object.
        """
        self.filter_streams[stream_name] = copy(msg)

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


    def get_filter_stream(
        self,
        stream_name: str,
        ) -> SysIdDataStream | None:
        """Return the most recent filter message for a stream."""
        return self.filter_streams.get(stream_name)
    
    def get_filter_value(
        self,
        stream_name: str,
        ) -> float | None:
        """Return the most recent filter value."""
        stream = self.get_filter_stream(stream_name)
        if stream is None:
            return None
        return float(stream.value)
    
    def get_filter_trend(
        self,
        stream_name: str,
        ) -> float | None:
        """Return the most recent filter trend."""
        stream = self.get_filter_stream(stream_name)
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
    

    def calculate_terms(self) -> None:
        """
        Calculate programmer-defined aircraft equation terms.

        This function is where the aircraft-specific system identification
        equations should be implemented.
        """
        # TODO: Could simplify this process by not using the aircraft constants yet.
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

        p = self.get_filter_stream("imu/gx")
        q = self.get_filter_stream("imu/gy")
        r = self.get_filter_stream("imu/gz")
        
        ail = self.get_filter_stream("rcout/ail")
        elv = self.get_filter_stream("rcout/elv")
        rud = self.get_filter_stream("rcout/rud")

        dyn_pres = self.get_filter_stream("pitot/dyn_pres")
        airspeed = self.get_filter_stream("pitot/airspeed")
        
        prop_speed = self.get_filter_stream("propulsion/prop_speed")

        if dyn_pres is None or airspeed is None:
            return

        if p is None or q is None or r is None:
            return

        lp_value = (0.5 * (b ** 2) * S / Ixx) * p.value * dyn_pres.trend / airspeed.trend
        lp_trend = (0.5 * (b ** 2) * S / Ixx) * p.trend * dyn_pres.trend / airspeed.trend
        self.publish_term(
            term_name="lp",
            value=lp_value,
            trend=lp_trend,
        )

        lr_value = (0.5 * (b ** 2) * S / Ixx) * r.value * dyn_pres.trend / airspeed.trend
        lr_trend = (0.5 * (b ** 2) * S / Ixx) * r.trend * dyn_pres.trend / airspeed.trend
        self.publish_term(
            term_name="lr",
            value=lr_value,
            trend=lr_trend,
        )

        mq_value = (0.5 * (c ** 2) * S / Iyy) * q.value * dyn_pres.trend / airspeed.trend
        mq_trend = (0.5 * (c ** 2) * S / Iyy) * q.trend * dyn_pres.trend / airspeed.trend
        self.publish_term(
            term_name="mq",
            value=mq_value,
            trend=mq_trend,
        )

        np_value = (0.5 * (b ** 2) * S / Izz) * p.value * dyn_pres.trend / airspeed.trend
        np_trend = (0.5 * (b ** 2) * S / Izz) * p.trend * dyn_pres.trend / airspeed.trend
        self.publish_term(
            term_name="np",
            value=np_value,
            trend=np_trend,
        )

        nr_value = (0.5 * (b ** 2) * S / Izz) * r.value * dyn_pres.trend / airspeed.trend
        nr_trend = (0.5 * (b ** 2) * S / Izz) * r.trend * dyn_pres.trend / airspeed.trend
        self.publish_term(
            term_name="nr",
            value=nr_value,
            trend=nr_trend,
        )

        if ail is None or elv is None or rud is None:
            return

        lail_value = (0.5 * b * S / Ixx) * ail.value * dyn_pres.trend
        lail_trend = (0.5 * b * S / Ixx) * ail.trend * dyn_pres.trend
        self.publish_term(
            term_name="lail",
            value=lail_value,
            trend=lail_trend,
        )

        lrud_value = (0.5 * b * S / Ixx) * rud.value * dyn_pres.trend
        lrud_trend = (0.5 * b * S / Ixx) * rud.trend * dyn_pres.trend
        self.publish_term(
            term_name="lrud",
            value=lrud_value,
            trend=lrud_trend,
        )

        melv_value = (0.5 * c * S / Iyy) * elv.value * dyn_pres.trend
        melv_trend = (0.5 * c * S / Iyy) * elv.trend * dyn_pres.trend
        self.publish_term(
            term_name="melv",
            value=melv_value,
            trend=melv_trend,
        )

        nail_value = (0.5 * b * S / Iyy) * ail.value * dyn_pres.trend
        nail_trend = (0.5 * b * S / Iyy) * ail.trend * dyn_pres.trend
        self.publish_term(
            term_name="nail",
            value=nail_value,
            trend=nail_trend,
        )

        nrud_value = (0.5 * b * S / Iyy) * rud.value * dyn_pres.trend
        nrud_trend = (0.5 * b * S / Iyy) * rud.trend * dyn_pres.trend
        self.publish_term(
            term_name="nrud",
            value=nrud_value,
            trend=nrud_trend,
        )

        qr_value = q.value * r.value
        qr_trend = q.trend * r.trend
        self.publish_term(
            term_name="qr",
            value=qr_value,
            trend=qr_trend,
        )

        pr_value = p.value * r.value
        pr_trend = p.trend * r.trend
        self.publish_term(
            term_name="pr",
            value=pr_value,
            trend=pr_trend,
        )

        pq_value = p.value * q.value
        pq_trend = p.trend * q.trend
        self.publish_term(
            term_name="pq",
            value=pq_value,
            trend=pq_trend,
        )

        r2p2_value = (r.value ** 2) - (p.value ** 2)
        r2p2_trend = (r.trend ** 2) - (p.trend ** 2)
        self.publish_term(
            term_name="r2p2",
            value=r2p2_value,
            trend=r2p2_trend,
        )

        if p_dot is None or q_dot is None or r_dot is None:
            return

        rpq_value = r_dot.value + p.value * q.value
        rpq_trend = r_dot.trend + p.trend * q.trend
        self.publish_term(
            term_name="rpq",
            value=rpq_value,
            trend=rpq_trend,
        )

        pqr_value = p_dot.value - q.value * r.value
        pqr_trend = p_dot.trend - q.trend * r.trend
        self.publish_term(
            term_name="pqr",
            value=pqr_value,
            trend=pqr_trend,
        )

        if prop_speed is None:
            return
        
        omega_r_value = prop_speed.value * r.value
        omega_r_trend = prop_speed.trend * r.trend
        self.publish_term(
            term_name="omega_r",
            value=omega_r_value,
            trend=omega_r_trend,
        )
        
        omega_p_value = -prop_speed.value * q.value
        omega_p_trend = -prop_speed.trend * q.trend
        self.publish_term(
            term_name="omega_p",
            value=omega_p_value,
            trend=omega_p_trend,
        )

    def _add_term_publisher(
            self,
            term_name: str,
        ) -> None:
        """Create a publisher for a generated equation term."""
        if term_name in self.pubs:
            return
        
        topic = f"{self.TERMS_PREFIX}{term_name}"

        self.pubs[term_name] = self.create_publisher(
            SysIdDataStream,
            topic,
            10,
        )

        self.get_logger().info(
            f"Created term publisher: {topic}"
        )

    def publish_term(
        self,
        term_name: str,
        value: float,
        trend: float,
        ) -> None:
        """
        Publish a generated equation term.

        Parameters
        ----------
        term_name : str
            Name of the generated term.

        value : float
            Programmer-defined value equation.

        trend : float
            Programmer-defined trend equation.
        """
        if term_name not in self.pubs:
            self._add_term_publisher(term_name)

        timestamp = self.get_clock().now()
        current_time = float(timestamp.nanoseconds) * 1e-9
        previous_time = self.previous_publish_time.get(term_name, 0.0)

        dt = current_time - previous_time
        if dt <= 0.0:
            dt += self.TIMESTAMP_ROLLOVER_PERIOD

        self.previous_publish_time[term_name] = current_time

        msg = SysIdDataStream()
        msg.header.stamp = timestamp.to_msg()
        msg.dt = dt
        msg.value = value
        msg.trend = trend
        self.pubs[term_name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    terms_node = SIDTerms()

    while rclpy.ok():
        try:
            rclpy.spin_once(terms_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    terms_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()