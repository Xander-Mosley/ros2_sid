#!/usr/bin/env python3

import json
from pathlib import Path
from re import S

import mavros
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
import time

from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure, Temperature
from std_msgs.msg import Float64, Float64MultiArray, String

from ardupilot_msgs.msg import Pitot, Propulsion, RcIn, RcOut
from drone_interfaces.msg import CtlTraj, Telem

from ros2_sid.rt_ols import CircularBuffer
from ros2_sid.signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT,
    ButterworthLowPass_VDT_2O, ButterworthHighPass_VDT_2O
    )


class Differentiating(Node):
    def __init__(self, ns=''):
        super().__init__('differentiating_node')
        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()
        
    def setup_vars(self):
        package_subroot = Path(__file__).resolve().parents[1]
        frequency_config_file = (package_subroot / "ros2_sid" / "setup" / "frequency_config.json")
        if not frequency_config_file.exists():
            raise FileNotFoundError(
                f"Frequency configuration file not found:\n"
                f"{frequency_config_file}"
            )
        with frequency_config_file.open("r", encoding="utf-8") as file:
            self.frequency_config = json.load(file)
        upper_cutoff = self.frequency_config["alias_frequency_hz"]
        
        self.acc_times = CircularBuffer(5)
        self.rol_velo = CircularBuffer(5)
        self.pit_velo = CircularBuffer(5)
        self.yaw_velo = CircularBuffer(5)
        self.acc_times.add(0)

        self.rol_accel_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)
        self.pit_accel_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)
        self.yaw_accel_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)


    def setup_subs(self):
        self.imu_filt_sub: Subscription = self.create_subscription(
            Imu,
            '/sid/filtered/imu',
            self.imu_filt_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_filt_callback(self, sub_msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        new_nanosec: float = sub_msg.header.stamp.nanosec * 1E-9
        if new_nanosec < self.acc_times.latest:
            new_nanosec += 1.0
        dt = new_nanosec - self.acc_times.latest
        self.acc_times.add(new_nanosec)
        if np.all(self.acc_times.get_all() >= 1.0):
            self.acc_times.apply_to_all(lambda x: x - 1.0)

        self.rol_velo.add(sub_msg.angular_velocity.x)
        self.pit_velo.add(sub_msg.angular_velocity.y)
        self.yaw_velo.add(sub_msg.angular_velocity.z)
        if self.rol_velo.size < self.rol_velo._capacity:
            return
        
        pub_msg: Imu = Imu()
        pub_msg.header = sub_msg.header
        pub_msg.angular_velocity.x = self.rol_accel_lpf.update(poly_diff(self.acc_times.get_all(), self.rol_velo.get_all()), dt)
        pub_msg.angular_velocity.y = self.pit_accel_lpf.update(poly_diff(self.acc_times.get_all(), self.pit_velo.get_all()), dt)
        pub_msg.angular_velocity.z = self.yaw_accel_lpf.update(poly_diff(self.acc_times.get_all(), self.yaw_velo.get_all()), dt)
        self.imu_diff.publish(pub_msg)


    def setup_pubs(self):
        self.imu_diff: Publisher = self.create_publisher(
            Imu, 'sid/differentiated/imu', 10)


def main(args=None):
    rclpy.init(args=args)
    differentiating_node = Differentiating()

    while rclpy.ok():
        try:
            rclpy.spin_once(differentiating_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    differentiating_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()