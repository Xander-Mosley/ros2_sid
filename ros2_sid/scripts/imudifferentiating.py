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


from ros2_sid.rt_ols import ModelStructure, StoredData
from ros2_sid.signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT, ButterworthLowPass_VDT_2O
    )

FIRST_PASS = True

class IMUDifferentiating(Node):
    def __init__(self, ns=''):
        super().__init__('imu_differentiating_node')
        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()
        
    def setup_vars(self):
        self.livetime_sec = StoredData(5, 1)
        self.livetime_nano = deque([0.0, 0.0, 0.0, 0.0, 0.0],maxlen=5)
        self.rol_velo = StoredData(5, 1)
        self.pit_velo = StoredData(5, 1)
        self.yaw_velo = StoredData(5, 1)
        self.lpf_rol_accel = ButterworthLowPass_VDT_2O(1.54)
        self.lpf_pit_accel = ButterworthLowPass_VDT_2O(1.54)
        self.lpf_yaw_accel = ButterworthLowPass_VDT_2O(1.54)
        self.rol_accel = 0.0
        self.pit_accel = 0.0
        self.yaw_accel = 0.0
        
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
        start = time.perf_counter()
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        global FIRST_PASS

        self.livetime_sec.update_data(sub_msg.header.stamp.sec)
        new_nanosec_data: float = sub_msg.header.stamp.nanosec * 1E-9
        if new_nanosec_data < self.livetime_nano[-1]:
            new_nanosec_data += 1.0

        self.dt = new_nanosec_data - self.livetime_nano[-1]
        if self.dt > (1 / 150):
            self.livetime_nano.append(new_nanosec_data)
            if len(self.livetime_nano) > 0 and all(x >= 1.0 for x in self.livetime_nano):
                self.livetime_nano = deque([x - 1.0 for x in self.livetime_nano], maxlen=self.livetime_nano.maxlen)

            self.rol_velo.update_data(sub_msg.angular_velocity.x)
            self.pit_velo.update_data(sub_msg.angular_velocity.y)
            self.yaw_velo.update_data(sub_msg.angular_velocity.z)

            self.rol_accel = self.lpf_rol_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.rol_velo.data), self.dt)
            self.pit_accel = self.lpf_pit_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.pit_velo.data), self.dt)
            self.yaw_accel = self.lpf_yaw_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.yaw_velo.data), self.dt)
            
            pub_msg: Imu = Imu()
            pub_msg.header = sub_msg.header
            pub_msg.angular_velocity.x = np.float64(self.rol_accel)
            pub_msg.angular_velocity.y = np.float64(self.pit_accel)
            pub_msg.angular_velocity.z = np.float64(self.yaw_accel)
            self.imu_diff.publish(pub_msg)

            end = time.perf_counter()
            self.elapsed = end - start
            self.max_elapsed = np.max([self.max_elapsed, self.elapsed])
            self.min_elapsed = np.min([self.min_elapsed, self.elapsed])
            num_pts = 99
            alpha = 2 / (num_pts + 1)
            self.ema_elapsed = (alpha * self.elapsed) + ((1-alpha) * self.ema_elapsed)
            # print(f"imu_callback runtime - {self.elapsed*1e3:.3f} ms\tavg: {self.ema_elapsed*1e3:.3f} ms\tmax: {self.max_elapsed*1e3:.3f} ms\tmin: {self.min_elapsed*1e3:.3f} ms")
            
            pub_msg_2: Float64MultiArray = Float64MultiArray()
            pub_msg_2.data = [
                np.float64(self.elapsed),
                np.float64(self.ema_elapsed),
                np.float64(self.max_elapsed),
                np.float64(self.min_elapsed),
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