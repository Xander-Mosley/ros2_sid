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

class IMUFiltering(Node):
    def __init__(self, ns=''):
        super().__init__('imu_filtering_node')
        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()
        
    def setup_vars(self):
        self.livetime_sec = 0.0
        self.livetime_nano = deque([0.0, 0.0, 0.0, 0.0, 0.0],maxlen=5)
        self.lpf_rol_velo = ButterworthLowPass_VDT_2O(1.54)
        self.lpf_pit_velo = ButterworthLowPass_VDT_2O(1.54)
        self.lpf_yaw_velo = ButterworthLowPass_VDT_2O(1.54)
        self.rol_velo = 0.0
        self.pit_velo = 0.0
        self.yaw_velo = 0.0
        
        self.elapsed = 0
        self.max_elapsed = 0
        self.min_elapsed = 1
        self.ema_elapsed = 0


    def setup_subs(self):
        self.imu_sub: Subscription = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, sub_msg: Imu) -> None:
        start = time.perf_counter()
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        global FIRST_PASS

        self.livetime_sec = sub_msg.header.stamp.sec
        new_nanosec_data: float = sub_msg.header.stamp.nanosec * 1E-9
        if new_nanosec_data < self.livetime_nano[-1]:
            new_nanosec_data += 1.0

        self.dt = new_nanosec_data - self.livetime_nano[-1]
        if self.dt > (1 / 150):
            self.livetime_nano.append(new_nanosec_data)
            if len(self.livetime_nano) > 0 and all(x >= 1.0 for x in self.livetime_nano):
                self.livetime_nano = deque([x - 1.0 for x in self.livetime_nano], maxlen=self.livetime_nano.maxlen)

            self.pit_velo = self.lpf_pit_velo.update(sub_msg.angular_velocity.y, self.dt)
            self.rol_velo = self.lpf_rol_velo.update(sub_msg.angular_velocity.x, self.dt)
            self.yaw_velo = self.lpf_yaw_velo.update(sub_msg.angular_velocity.z, self.dt)
            
            pub_msg: Imu = Imu()
            pub_msg.header = sub_msg.header
            pub_msg.angular_velocity.x = np.float64(self.rol_velo)
            pub_msg.angular_velocity.y = np.float64(self.pit_velo)
            pub_msg.angular_velocity.z = np.float64(self.yaw_velo)
            self.imu_filt.publish(pub_msg)

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