#!/usr/bin/env python3
from re import S
from collections import deque

import numpy as np
import mavros
import time
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from mavros.base import SENSOR_QOS
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray, String


from ros2_sid.rt_ols import CircularBuffer
from ros2_sid.signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT, ButterworthLowPass_VDT_2O
    )


class IMUDifferentiating(Node):
    def __init__(self, ns=''):
        super().__init__('imu_differentiating_node')
        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()
        
    def setup_vars(self):
        self.acc_time = CircularBuffer(5)
        self.acc_time.fill_all(0)

        self.rol_velo = CircularBuffer(5)
        self.pit_velo = CircularBuffer(5)
        self.yaw_velo = CircularBuffer(5)
        self.rol_velo.fill_all(0)
        self.pit_velo.fill_all(0)
        self.yaw_velo.fill_all(0)

        self.rol_accel_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.pit_accel_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.yaw_accel_lpf = ButterworthLowPass_VDT_2O(1.54)
        
        self.elapsed = 0
        self.max_elapsed = 0
        self.min_elapsed = 1
        self.ema_elapsed = 0


    def setup_subs(self):
        self.imu_filt_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_filt',
            self.imu_filt_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_filt_callback(self, sub_msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        start = time.perf_counter()

        new_nanosec_data: float = sub_msg.header.stamp.nanosec * 1E-9
        if self.acc_time.size > 0 and new_nanosec_data < self.acc_time.latest:
            new_nanosec_data += 1.0
        if self.acc_time.size == 0:
            self.acc_time.add(new_nanosec_data)
            return
        dt = new_nanosec_data - self.acc_time.latest
        if dt > (1.0 / 150.0):
            self.acc_time.add(new_nanosec_data)
            if self.acc_time.size > 0 and np.all(self.acc_time.get_all() >= 1.0):
                self.acc_time.apply_to_all(lambda x: x - 1.0)

            self.rol_velo.add(sub_msg.angular_velocity.x)
            self.pit_velo.add(sub_msg.angular_velocity.y)
            self.yaw_velo.add(sub_msg.angular_velocity.z)

            pub_msg: Imu = Imu()
            pub_msg.header = sub_msg.header
            pub_msg.angular_velocity.x = self.rol_accel_lpf.update(poly_diff(self.acc_time.get_all(), self.rol_velo.get_all()), dt)
            pub_msg.angular_velocity.y = self.pit_accel_lpf.update(poly_diff(self.acc_time.get_all(), self.pit_velo.get_all()), dt)
            pub_msg.angular_velocity.z = self.yaw_accel_lpf.update(poly_diff(self.acc_time.get_all(), self.yaw_velo.get_all()), dt)
            self.imu_diff.publish(pub_msg)

            end = time.perf_counter()

            elapsed = end - start
            self.max_elapsed = np.max([self.max_elapsed, elapsed])
            self.min_elapsed = np.min([self.min_elapsed, elapsed])
            num_pts = 99
            alpha = 2 / (num_pts + 1)
            self.ema_elapsed = (alpha * elapsed) + ((1-alpha) * self.ema_elapsed)            
            pub_msg_2: Float64MultiArray = Float64MultiArray()
            pub_msg_2.data = [
                elapsed,
                self.ema_elapsed,
                self.max_elapsed,
                self.min_elapsed,
            ]
            self.imu_diff_duration.publish(pub_msg_2)


    def setup_pubs(self):
        self.imu_diff: Publisher = self.create_publisher(
            Imu, 'imu_diff', 10)
        
        self.imu_diff_duration: Publisher = self.create_publisher(
            Float64MultiArray, 'imu_diff_duration', 10)


def main(args=None):
    rclpy.init(args=args)
    imu_differentiating_node = IMUDifferentiating()

    while rclpy.ok():
        try:
            rclpy.spin_once(imu_differentiating_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    imu_differentiating_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()