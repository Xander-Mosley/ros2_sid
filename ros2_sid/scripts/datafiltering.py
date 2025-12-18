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

class DataFiltering(Node):
    def __init__(self, ns=''):
        super().__init__('data_filtering_node')

        self.node_streamrate = 50

        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()
        
    def setup_vars(self):
        self.livetime_sec = StoredData(5, 1)
        self.livetime_nano = deque([0.0, 0.0, 0.0, 0.0, 0.0],maxlen=5)
        self.rol_velo = StoredData(5, 1)
        self.lpf_rol_velo = ButterworthLowPass_VDT_2O(1.54)
        self.pit_velo = StoredData(5, 1)
        self.lpf_pit_velo = ButterworthLowPass_VDT_2O(1.54)
        self.yaw_velo = StoredData(5, 1)
        self.lpf_yaw_velo = ButterworthLowPass_VDT_2O(1.54)
        self.rol_accel = StoredData(5, 1)
        self.lpf_rol_accel = ButterworthLowPass_VDT_2O(1.54)
        self.pit_accel = StoredData(5, 1)
        self.lpf_pit_accel = ButterworthLowPass_VDT_2O(1.54)
        self.yaw_accel = StoredData(5, 1)
        self.lpf_yaw_accel = ButterworthLowPass_VDT_2O(1.54)
        
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

    def imu_callback(self, msg: Imu) -> None:
        start = time.perf_counter()
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        global FIRST_PASS

        self.livetime_sec.update_data(msg.header.stamp.sec)
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if new_nanosec_data < self.livetime_nano[-1]:
            new_nanosec_data += 1.0

        self.dt = new_nanosec_data - self.livetime_nano[-1]
        if self.dt > (1 / 150):
            self.livetime_nano.append(new_nanosec_data)
            if len(self.livetime_nano) > 0 and all(x >= 1.0 for x in self.livetime_nano):
                self.livetime_nano = deque([x - 1.0 for x in self.livetime_nano], maxlen=self.livetime_nano.maxlen)

            if FIRST_PASS:
                FIRST_PASS = False
                ModelStructure.update_shared_cp_time(self.livetime_nano[0])
            else:
                ModelStructure.update_shared_cp_timestep(self.dt)

            self.rol_velo.update_data(self.lpf_rol_velo.update(msg.angular_velocity.x, self.dt))
            self.pit_velo.update_data(self.lpf_pit_velo.update(msg.angular_velocity.y, self.dt))
            self.yaw_velo.update_data(self.lpf_yaw_velo.update(msg.angular_velocity.z, self.dt))

            self.rol_accel.update_data(self.lpf_rol_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.rol_velo.data), self.dt))
            self.pit_accel.update_data(self.lpf_pit_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.pit_velo.data), self.dt))
            self.yaw_accel.update_data(self.lpf_yaw_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.yaw_velo.data), self.dt))

            end = time.perf_counter()
            self.elapsed = end - start
            self.max_elapsed = np.max([self.max_elapsed, self.elapsed])
            self.min_elapsed = np.min([self.min_elapsed, self.elapsed])
            num_pts = 99
            alpha = 2 / (num_pts + 1)
            self.ema_elapsed = (alpha * self.elapsed) + ((1-alpha) * self.ema_elapsed)
            # print(f"imu_callback runtime - {self.elapsed*1e3:.3f} ms\tavg: {self.ema_elapsed*1e3:.3f} ms\tmax: {self.max_elapsed*1e3:.3f} ms\tmin: {self.min_elapsed*1e3:.3f} ms")


    def setup_pubs(self):
        self.timer_period: float = (1 / self.node_streamrate)

        self.imu_filtered: Publisher = self.create_publisher(
            Float64MultiArray, 'imu_filtered', 10)
        self.timer = self.create_timer(
            self.timer_period, self.imu_filtered_pub)
        
        self.imu_filter_duration: Publisher = self.create_publisher(
            Float64MultiArray, 'imu_filter_duration', 10)
        self.timer = self.create_timer(
            self.timer_period, self.imu_filter_duration_pub)
        
    def imu_filtered_pub(self) -> None:
        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.rol_velo.data.item(0)),
            np.float64(self.pit_velo.data.item(0)),
            np.float64(self.yaw_velo.data.item(0)),
            np.float64(self.rol_accel.data.item(0)),
            np.float64(self.pit_accel.data.item(0)),
            np.float64(self.yaw_accel.data.item(0))
        ]
        self.imu_filtered.publish(msg)
        
    def imu_filter_duration_pub(self) -> None:
        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.elapsed),
            np.float64(self.ema_elapsed),
            np.float64(self.max_elapsed),
            np.float64(self.min_elapsed),
        ]
        self.imu_filter_duration.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    data_filtering_node = DataFiltering()

    while rclpy.ok():
        try:
            rclpy.spin_once(data_filtering_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    data_filtering_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()