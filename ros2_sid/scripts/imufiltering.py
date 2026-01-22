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


class IMUFiltering(Node):
    def __init__(self, ns=''):
        super().__init__('imu_filtering_node')
        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()
        
    def setup_vars(self):
        self.imu_time = CircularBuffer(2)

        self.rol_velo_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.pit_velo_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.yaw_velo_lpf = ButterworthLowPass_VDT_2O(1.54)
        
        self.elapsed = 0
        self.max_elapsed = 0
        self.min_elapsed = 1
        self.ema_elapsed = 0


    def setup_subs(self):
        # self.imu_sub: Subscription = self.create_subscription(
        #     Imu,
        #     '/mavros/imu/data',
        #     self.imu_callback,
        #     qos_profile=SENSOR_QOS
        # )
        
        self.replay_imu_sub: Subscription = self.create_subscription(
            Float64MultiArray,
            '/replay/IMU/data',
            self.replay_imu_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, sub_msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        start = time.perf_counter()

        new_nanosec_data: float = sub_msg.header.stamp.nanosec * 1E-9
        if self.imu_time.size > 0 and new_nanosec_data < self.imu_time.latest:
            new_nanosec_data += 1.0
        if self.imu_time.size == 0:
            self.imu_time.add(new_nanosec_data)
            return
        dt = new_nanosec_data - self.imu_time.latest
        if dt >= (1.0 / 100.0):
            self.imu_time.add(new_nanosec_data)
            if self.imu_time.size > 0 and np.all(self.imu_time.get_all() >= 1.0):
                self.imu_time.apply_to_all(lambda x: x - 1.0)
            
            pub_msg: Imu = Imu()
            pub_msg.header = sub_msg.header
            pub_msg.angular_velocity.x = self.rol_velo_lpf.update(sub_msg.angular_velocity.x, dt)
            pub_msg.angular_velocity.y = self.pit_velo_lpf.update(sub_msg.angular_velocity.y, dt)
            pub_msg.angular_velocity.z = self.yaw_velo_lpf.update(sub_msg.angular_velocity.z, dt)
            self.imu_filt.publish(pub_msg)

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
            self.imu_filt_duration.publish(pub_msg_2)

    def replay_imu_callback(self, sub_msg: Float64MultiArray) -> None:
        start = time.perf_counter()

        seconds = int(sub_msg.data[0])
        nanoseconds = int(round((sub_msg.data[0] - seconds) * 1_000_000_000))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds = 0

        new_nanosec_data: float = nanoseconds * 1E-9
        if self.imu_time.size > 0 and new_nanosec_data < self.imu_time.latest:
            new_nanosec_data += 1.0
        if self.imu_time.size == 0:
            self.imu_time.add(new_nanosec_data)
            return
        dt = new_nanosec_data - self.imu_time.latest
        if dt > (1.0 / 150.0):
            self.imu_time.add(new_nanosec_data)
            if self.imu_time.size > 0 and np.all(self.imu_time.get_all() >= 1.0):
                self.imu_time.apply_to_all(lambda x: x - 1.0)

            pub_msg: Imu = Imu()
            pub_msg.header.stamp.sec = seconds
            pub_msg.header.stamp.nanosec = nanoseconds
            pub_msg.angular_velocity.x = self.rol_velo_lpf.update(sub_msg.data[5], dt)
            pub_msg.angular_velocity.y = self.pit_velo_lpf.update(sub_msg.data[6], dt)
            pub_msg.angular_velocity.z = self.yaw_velo_lpf.update(sub_msg.data[7], dt)
            self.imu_filt.publish(pub_msg)

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
            self.imu_filt_duration.publish(pub_msg_2)


    def setup_pubs(self):
        self.imu_filt: Publisher = self.create_publisher(
            Imu, 'imu_filt', 10)
        
        self.imu_filt_duration: Publisher = self.create_publisher(
            Float64MultiArray, 'imu_filt_duration', 10)


def main(args=None):
    rclpy.init(args=args)
    imu_filtering_node = IMUFiltering()

    while rclpy.ok():
        try:
            rclpy.spin_once(imu_filtering_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    imu_filtering_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()