#!/usr/bin/env python3

from re import S

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
        self.minimum_dt = 1.0 / 100.0
        self.imu_prev_nanosec = 0.0

        self.rol_velo_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.pit_velo_lpf = ButterworthLowPass_VDT_2O(1.54)
        self.yaw_velo_lpf = ButterworthLowPass_VDT_2O(1.54)
        
        num_pts = 99
        self.alpha = 2 / (num_pts + 1)
        self.imu_max_elapsed = 0
        self.imu_min_elapsed = 1_000_000_000
        self.imu_ema_elapsed = 0


    def setup_subs(self):
        self.imu_sub: Subscription = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )
        # self.replay_imu_sub: Subscription = self.create_subscription(
        #     Float64MultiArray,
        #     '/replay/IMU/data',
        #     self.replay_imu_callback,
        #     qos_profile=SENSOR_QOS
        # )

    def imu_callback(self, sub_msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        start = time.perf_counter()

        new_sec: float = sub_msg.header.stamp.sec
        new_nanosec: float = sub_msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.imu_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.imu_prev_nanosec = new_nanosec
            
            pub_msg: Imu = Imu()
            pub_msg.header = sub_msg.header
            pub_msg.angular_velocity.x = self.rol_velo_lpf.update(sub_msg.angular_velocity.x, dt)
            pub_msg.angular_velocity.y = self.pit_velo_lpf.update(sub_msg.angular_velocity.y, dt)
            pub_msg.angular_velocity.z = self.yaw_velo_lpf.update(sub_msg.angular_velocity.z, dt)
            self.imu_filt.publish(pub_msg)

            end = time.perf_counter()

            elapsed = end - start
            self.imu_max_elapsed = np.max([self.imu_max_elapsed, elapsed])
            self.imu_min_elapsed = np.min([self.imu_min_elapsed, elapsed])
            self.imu_ema_elapsed = (self.alpha * elapsed) + ((1-self.alpha) * self.imu_ema_elapsed)            
            pub_msg_2: Float64MultiArray = Float64MultiArray()
            pub_msg_2.data = [
                elapsed,
                self.imu_ema_elapsed,
                self.imu_max_elapsed,
                self.imu_min_elapsed,
            ]
            self.imu_filt_duration.publish(pub_msg_2)

        else:
            print(f"IMU filter skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def replay_imu_callback(self, sub_msg: Float64MultiArray) -> None:
        start = time.perf_counter()
        
        new_sec, new_nanosec = divmod(sub_msg.data[0], 1.0)
        dt = (new_nanosec - self.imu_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.imu_prev_nanosec = new_nanosec

            pub_msg: Imu = Imu()
            pub_msg.header.stamp.sec = int(new_sec)
            pub_msg.header.stamp.nanosec = int(new_nanosec * 1e9)
            pub_msg.angular_velocity.x = self.rol_velo_lpf.update(sub_msg.data[5], dt)
            pub_msg.angular_velocity.y = self.pit_velo_lpf.update(sub_msg.data[6], dt)
            pub_msg.angular_velocity.z = self.yaw_velo_lpf.update(sub_msg.data[7], dt)
            self.imu_filt.publish(pub_msg)

            end = time.perf_counter()

            elapsed = end - start
            self.imu_max_elapsed = np.max([self.imu_max_elapsed, elapsed])
            self.imu_min_elapsed = np.min([self.imu_min_elapsed, elapsed])
            self.imu_ema_elapsed = (self.alpha * elapsed) + ((1-self.alpha) * self.imu_ema_elapsed)            
            pub_msg_2: Float64MultiArray = Float64MultiArray()
            pub_msg_2.data = [
                elapsed,
                self.imu_ema_elapsed,
                self.imu_max_elapsed,
                self.imu_min_elapsed,
            ]
            self.imu_filt_duration.publish(pub_msg_2)

        else:
            print(f"IMU filter skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")


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