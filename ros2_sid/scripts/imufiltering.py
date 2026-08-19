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

from ros2_sid.signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT,
    ButterworthLowPass_VDT_2O, ButterworthHighPass_VDT_2O
    )


class Filtering(Node):
    def __init__(self, ns=''):
        super().__init__('filtering_node')
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
        
        self.imu_prev_nanosec = 0.0
        self.rol_velo_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)
        self.pit_velo_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)
        self.yaw_velo_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)

        self.rcout_prev_nanosec = 0.0
        self.ail_pwm_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)
        self.elv_pwm_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)
        self.rud_pwm_lpf = ButterworthLowPass_VDT_2O(upper_cutoff)


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
        # self.replay_rcout_sub: Subscription = self.create_subscription(
        #     Float64MultiArray,
        #     '/replay/RCOU/data',
        #     self.replay_rcout_callback,
        #     qos_profile=SENSOR_QOS
        # )

    def imu_callback(self, sub_msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        new_nanosec: float = sub_msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.imu_prev_nanosec) % 1.0
        self.imu_prev_nanosec = new_nanosec
        
        pub_msg: Imu = Imu()
        pub_msg.header = sub_msg.header
        pub_msg.angular_velocity.x = self.rol_velo_lpf.update(sub_msg.angular_velocity.x, dt)
        pub_msg.angular_velocity.y = self.pit_velo_lpf.update(sub_msg.angular_velocity.y, dt)
        pub_msg.angular_velocity.z = self.yaw_velo_lpf.update(sub_msg.angular_velocity.z, dt)
        self.imu_filt.publish(pub_msg)
    
    def rcout_callback(self, sub_msg: RcOut) -> None:
        new_nanosec: float = sub_msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.rcout_prev_nanosec) % 1.0
        self.rcout_prev_nanosec = new_nanosec
        
        pub_msg: RcOut = RcOut()
        pub_msg.header = sub_msg.header
        pub_msg.values[0] = self.ail_pwm_lpf.update(sub_msg.values[0], dt)
        pub_msg.values[1] = self.elv_pwm_lpf.update(sub_msg.values[1], dt)
        pub_msg.values[3] = self.rud_pwm_lpf.update(sub_msg.values[3], dt)
        self.rcout_filt.publish(pub_msg)
    
    def replay_rcout_callback(self, sub_msg: Float64MultiArray) -> None:
        new_sec, new_nanosec = divmod(sub_msg.data[0], 1.0)
        dt = (new_nanosec - self.rcout_prev_nanosec) % 1.0
        self.rcout_prev_nanosec = new_nanosec
        
        pub_msg: RcOut = RcOut()
        pub_msg.header = sub_msg.header
        pub_msg.values[0] = self.ail_pwm_lpf.update(sub_msg.values[2], dt)
        pub_msg.values[1] = self.elv_pwm_lpf.update(sub_msg.values[3], dt)
        pub_msg.values[3] = self.rud_pwm_lpf.update(sub_msg.values[5], dt)
        self.rcout_filt.publish(pub_msg)


    def setup_pubs(self):
        self.imu_filt: Publisher = self.create_publisher(
            Imu, 'sid/filtered/imu', 10)
        # self.imu_detrended: Publisher = self.create_publisher(
        #     Imu, 'sid/imu_detrended', 10)
        # self.imu_trend: Publisher = self.create_publisher(
        #     Imu, 'sid/imu_trend', 10)
        self.rcout_filt: Publisher = self.create_publisher(
            RcOut, 'sid/filtered/rcout', 10)


def main(args=None):
    rclpy.init(args=args)
    filtering_node = Filtering()

    while rclpy.ok():
        try:
            rclpy.spin_once(filtering_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    filtering_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()