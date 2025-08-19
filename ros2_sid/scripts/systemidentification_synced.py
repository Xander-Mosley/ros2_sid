#!/usr/bin/env python3
import math
import threading
from re import S

import mavros
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher

from drone_interfaces.msg import CtlTraj, Telem
from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray, String

from ros2_sid.rt_ols import ModelStructure, StoredData, diff
from ros2_sid.rotation_utils import euler_from_quaternion


class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('ols_node')
        self.setup_storeddatas()
        self.setup_modelstructures()

        self.setup_synced_subscriptions(ns)

        self.setup_all_publishers()


    def setup_storeddatas(self) -> None:
        # initialize stored data objects
        self.livetime = StoredData(6, 1)
        self.rol_velo = StoredData(6, 1)
        self.pit_velo = StoredData(6, 1)
        self.yaw_velo = StoredData(6, 1)
        self.rol_accel = StoredData(1, 1)
        self.pit_accel = StoredData(1, 1)
        self.yaw_accel = StoredData(1, 1)
        self.ail_pwm = StoredData(1, 1)
        self.elv_pwm = StoredData(1, 1)
        self.rud_pwm = StoredData(1, 1)
        # self.aoa = StoredData(1, 1)
        # self.ssa = StoredData(1, 1)
        # self.airspeed = StoredData(6, 1)
        # self.dyn_pres = StoredData(1, 1)

    def setup_modelstructures(self) -> None:
        # define class variables
        ModelStructure.class_eff = 0.999

        # initialize model structure objects
        self.rol = ModelStructure(2)
        self.pit = ModelStructure(2)
        self.yaw = ModelStructure(2)
    

    def setup_synced_subscriptions(self, ns: str) -> None:
        self.imu_sub = Subscriber(
            self,
            Imu,
            '/mavros/imu/data',
            qos_profile=SENSOR_QOS
        )
        self.rcout_sub = Subscriber(
            self,
            RCOut,
            '/mavros/rc/out',
            qos_profile=SENSOR_QOS
        )
        self.odom_sub = Subscriber(
            self,
            mavros.local_position.Odometry,
            'mavros/local_position/odom',
            qos_profile=SENSOR_QOS
        )
        
        self.sync = ApproximateTimeSynchronizer(
            [self.imu_sub, self.rcout_sub, self.odom_sub],
            queue_size = 20,
            slop = 0.5
        )
        self.sync.registerCallback(self.synced_callback)

        # self.get_logger().info("Synchronized IMU, RCOut, Odometry, and Telem subscriptions initialized.")

    def synced_callback(
            self,
            imu_msg: Imu,
            rcout_msg: RCOut,
            odom_msg: Odometry
            ) -> None:
        
        self.livetime.update_data((imu_msg.header.stamp.nanosec * 1e-9))    # TODO: Confirm that it is acceptable to ignore stamp.sec, or find a way to incorporate it.
        self.rol_velo.update_data(imu_msg.angular_velocity.x)
        self.pit_velo.update_data(imu_msg.angular_velocity.y)
        self.yaw_velo.update_data(imu_msg.angular_velocity.z)
        self.rol_accel.update_data(diff(self.livetime.data, self.rol_velo.data))
        self.pit_accel.update_data(diff(self.livetime.data, self.pit_velo.data))
        self.yaw_accel.update_data(diff(self.livetime.data, self.yaw_velo.data))
        ModelStructure.update_shared_cp_time(self.livetime.data[0])

        self.ail_pwm.update_data(rcout_msg.channels[0])
        self.elv_pwm.update_data(rcout_msg.channels[1])
        self.rud_pwm.update_data(rcout_msg.channels[2])


    def setup_all_publishers(self) -> None:
        self.ols_rol_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol_s', 10)
        timer_period: float = 0.02
        self.ols_rol_timer = self.create_timer(
            timer_period, self.publish_ols_rol_data)

        self.ols_pit_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_pit_s', 10)
        timer_period: float = 0.02
        self.ols_pit_timer = self.create_timer(
            timer_period, self.publish_ols_pit_data)

        self.ols_yaw_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_yaw_s', 10)
        timer_period: float = 0.02
        self.ols_yaw_timer = self.create_timer(
            timer_period, self.publish_ols_yaw_data)
        
    def publish_ols_rol_data(self) -> None:
        self.rol.update_model(self.rol_accel.data[0], [self.rol_velo.data[0], self.ail_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.rol_accel.data.item(0)),
            np.float64(self.rol_velo.data.item(0)),
            np.float64(self.ail_pwm.data.item(0)),

            self.rol.parameters[0],
            self.rol.parameters[1]
            ]
        self.ols_rol_publisher.publish(msg)
        
    def publish_ols_pit_data(self) -> None:
        self.pit.update_model(self.pit_accel.data[0], [self.pit_velo.data[1], self.elv_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.pit_accel.data.item(0)),

            np.float64(self.pit_velo.data.item(0)),
            np.float64(self.elv_pwm.data.item(0)),

            self.pit.parameters[0],
            self.pit.parameters[1]
            ]
        self.ols_pit_publisher.publish(msg)
        
    def publish_ols_yaw_data(self) -> None:
        self.yaw.update_model(self.yaw_accel.data[0], [self.yaw_velo.data[0], self.rud_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.yaw_accel.data.item(0)),

            np.float64(self.yaw_velo.data.item(0)),
            np.float64(self.rud_pwm.data.item(0)),

            self.yaw.parameters[0],
            self.yaw.parameters[1]
            ]
        self.ols_yaw_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    perform_sid = OLSNode()

    while rclpy.ok():
        try:
            rclpy.spin_once(perform_sid, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    perform_sid.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()