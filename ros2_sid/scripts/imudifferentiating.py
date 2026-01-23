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
        self.acc_times = CircularBuffer(5)
        self.acc_times.add(0)
        self.minimum_dt = 1.0 / 100.0

        self.rol_velo = CircularBuffer(5)
        self.pit_velo = CircularBuffer(5)
        self.yaw_velo = CircularBuffer(5)

        self.rol_accel_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.pit_accel_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.yaw_accel_lpf = ButterworthLowPass_VDT_2O(1.54)
        
        num_pts = 99
        self.alpha = 2 / (num_pts + 1)
        self.acc_max_elapsed = 0
        self.acc_min_elapsed = 1_000_000_000
        self.acc_ema_elapsed = 0


    def setup_subs(self):
        # self.imu_sub: Subscription = self.create_subscription(
        #     Imu,
        #     '/mavros/imu/data',
        #     self.imu_callback,
        #     qos_profile=SENSOR_QOS
        # )
        self.imu_filt_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_filt',
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, sub_msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        start = time.perf_counter()

        new_sec: float = sub_msg.header.stamp.sec
        new_nanosec: float = sub_msg.header.stamp.nanosec * 1E-9
        if new_nanosec < self.acc_times.latest:
            new_nanosec += 1.0
        dt = new_nanosec - self.acc_times.latest
        if dt > self.minimum_dt:
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

            end = time.perf_counter()

            elapsed = end - start
            self.acc_max_elapsed = np.max([self.acc_max_elapsed, elapsed])
            self.acc_min_elapsed = np.min([self.acc_min_elapsed, elapsed])
            self.acc_ema_elapsed = (self.alpha * elapsed) + ((1-self.alpha) * self.acc_ema_elapsed)
            pub_msg_2: Float64MultiArray = Float64MultiArray()
            pub_msg_2.data = [
                elapsed,
                self.acc_ema_elapsed,
                self.acc_max_elapsed,
                self.acc_min_elapsed,
            ]
            self.imu_diff_duration.publish(pub_msg_2)

        else:
            print(f"IMU differentiation skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")


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