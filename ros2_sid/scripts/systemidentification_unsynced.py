#!/usr/bin/env python3
import math
import threading
from re import S

import mavros
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from drone_interfaces.msg import CtlTraj, Telem
from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray, String

from ros2_sid.rt_ols import ModelStructure, StoredData, diff


def euler_from_quaternion(x:float, y:float, z:float, w:float) -> tuple:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('ols_node')
        self.setup_storeddatas()
        self.setup_modelstructures()

        self.setup_all_subscriptions()

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
    

    def setup_all_subscriptions(self) -> None:
        self.imu_sub: Subscription = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )

        self.rcout_sub: Subscription = self.create_subscription(
            RCOut,
            '/mavros/rc/out',
            self.rcout_callback,
            qos_profile=SENSOR_QOS
        )

        self.odom_sub: Subscription = self.create_subscription(
            mavros.local_position.Odometry,
            'mavros/local_position/odom',
            self.odom_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        self.livetime.update_data((msg.header.stamp.nanosec * 1e-9))
        self.rol_velo.update_data(msg.angular_velocity.x)
        self.pit_velo.update_data(msg.angular_velocity.y)
        self.yaw_velo.update_data(msg.angular_velocity.z)

        self.rol_accel.update_data(diff(self.livetime.data, self.rol_velo.data))
        self.pit_accel.update_data(diff(self.livetime.data, self.pit_velo.data))
        self.yaw_accel.update_data(diff(self.livetime.data, self.yaw_velo.data))

        ModelStructure.update_shared_cp_time(self.livetime.data[0])
        
    def rcout_callback(self, msg: RCOut) -> None:
        self.ail_pwm.update_data(msg.channels[0])
        self.elv_pwm.update_data(msg.channels[1])
        self.rud_pwm.update_data(msg.channels[2])

    def odom_callback(self, msg: mavros.local_position.Odometry) -> None:
        """
        Converts NED to ENU and publishes the trajectory
        https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
        Twist Will show velocity in linear and rotational 
        """
        state_x = msg.pose.pose.position.x
        state_y = msg.pose.pose.position.y
        state_z = msg.pose.pose.position.z

        # quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        state_phi, state_theta, state_psi = euler_from_quaternion(
            qx, qy, qz, qw)
        
        state_phi = state_phi
        state_theta = state_theta
        state_psi = state_psi   # (yaw+ (2*np.pi) ) % (2*np.pi);

        # get magnitude of velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        state_airspeed = np.sqrt(vx**2 + vy**2 + vz**2)

        self.odom_data: list[float] = [
            state_x,
            state_y,
            state_z,
            state_phi,
            state_theta,
            state_psi,
            state_airspeed
        ]


    def setup_all_publishers(self) -> None:
        self.ols_rol_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol', 10)
        timer_period: float = 0.02
        self.ols_rol_timer = self.create_timer(
            timer_period, self.publish_ols_rol_data)

        self.ols_pit_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_pit', 10)
        timer_period: float = 0.02
        self.ols_pit_timer = self.create_timer(
            timer_period, self.publish_ols_pit_data)

        self.ols_yaw_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_yaw', 10)
        timer_period: float = 0.02
        self.ols_yaw_timer = self.create_timer(
            timer_period, self.publish_ols_yaw_data)
        
    def publish_ols_rol_data(self) -> None:
        self.rol.update_model(self.rol_accel.data[0], [self.rol_velo.data[0], self.ail_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            self.rol.parameters[0],
            self.rol.parameters[1]
            ]
        self.ols_rol_publisher.publish(msg)
        
    def publish_ols_pit_data(self) -> None:
        self.pit.update_model(self.pit_accel.data[0], [self.pit_velo.data[1], self.elv_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            self.pit.parameters[0],
            self.pit.parameters[1]
            ]
        self.ols_pit_publisher.publish(msg)
        
    def publish_ols_yaw_data(self) -> None:
        self.yaw.update_model(self.yaw_accel.data[0], [self.yaw_velo.data[0], self.rud_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
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