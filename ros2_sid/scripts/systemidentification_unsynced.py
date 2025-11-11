#!/usr/bin/env python3
import math
import threading
from re import S

import mavros
import numpy as np
from collections import deque

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
from sensor_msgs.msg import FluidPressure
from sensor_msgs.msg import Temperature
from std_msgs.msg import Float64, Float64MultiArray, String

from ros2_sid.rt_ols import ModelStructure, StoredData, diff, sg_diff
from ros2_sid.rotation_utils import euler_from_quaternion
from ros2_sid.discrete_diff import ButterworthLowPassVariableDT, ButterworthLowPass


FIRST_PASS = True


class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('ols_node')
        self.setup_storeddatas()
        self.setup_modelstructures()

        self.setup_all_subscriptions()

        self.setup_all_publishers()


    def setup_storeddatas(self) -> None:
        # initialize stored data objects
        self.livetime_sec = StoredData(5, 1)
        self.livetime_nano = deque([0.0, 0.0, 0.0, 0.0, 0.0],maxlen=5)
        self.rol_velo = StoredData(5, 1)
        self.pit_velo = StoredData(5, 1)
        self.yaw_velo = StoredData(5, 1)
        self.rol_accel = StoredData(5, 1)
        self.pit_accel = StoredData(5, 1)
        self.yaw_accel = StoredData(5, 1)

        self.ail_pwm = StoredData(1, 1)
        self.elv_pwm = StoredData(1, 1)
        self.rud_pwm = StoredData(1, 1)

        # self.aoa = StoredData(1, 1)
        # self.ssa = StoredData(1, 1)
        
        self.dyn_pres = StoredData(1, 1)
        self.dyn_pres.update_data(1)
        self.stat_pres = StoredData(1, 1)
        self.stat_pres.update_data(1)
        self.temp = StoredData(1, 1)
        self.temp.update_data(1)
        self.airspeed = StoredData(1, 1)

        self.xdir_accel = StoredData(1, 1)
        self.ydir_accel = StoredData(1, 1)
        self.zdir_accel = StoredData(1, 1)

        self.mass = 1
        self.wing_span = 3.868  # [m]
        self.wing_area = 1.065634   # [m²]


        self.lpf1 = ButterworthLowPass(1.54)
        self.lpf2 = ButterworthLowPass(1.54)

    def setup_modelstructures(self) -> None:
        # define class variables
        ModelStructure.class_eff = 0.999

        # initialize model structure objects
        self.rol = ModelStructure(2)
        self.pit = ModelStructure(2)
        self.yaw = ModelStructure(2)
        self.rol_large = ModelStructure(4)
        self.rol_yaw = ModelStructure(5)
        self.rol_moment = ModelStructure(2)
        self.Y_dim = ModelStructure(4)
        self.rol_nondim = ModelStructure(4)
    

    def setup_all_subscriptions(self) -> None:
        # TODO: Subscribe to more published data.
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
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            qos_profile=SENSOR_QOS
        )

        self.diff_pressure_sub: Subscription = self.create_subscription(
            FluidPressure,
            '/mavros/imu/diff_pressure',
            self.diff_pressure_callback,
            qos_profile=SENSOR_QOS
        )

        self.static_pressure_sub: Subscription = self.create_subscription(
            FluidPressure,
            '/mavros/imu/static_pressure',
            self.static_pressure_callback,
            qos_profile=SENSOR_QOS
        )

        self.temperature_baro_sub: Subscription = self.create_subscription(
            Temperature,
            '/mavros/imu/temperature_baro',
            self.temperature_baro_callback,
            qos_profile=SENSOR_QOS
        )

    def imu_callback(self, msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        self.livetime_sec.update_data(msg.header.stamp.sec)

        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        while len(self.livetime_nano) > 0 and new_nanosec_data < self.livetime_nano[-1]:
            new_nanosec_data += 1.0
        self.livetime_nano.append(new_nanosec_data)
        if len(self.livetime_nano) > 0 and all(x >= 1.0 for x in self.livetime_nano):
            self.livetime_nano = deque([x - 1.0 for x in self.livetime_nano], maxlen=self.livetime_nano.maxlen)

        dt = self.livetime_nano[-1] - self.livetime_nano[-2]
        # ModelStructure.update_shared_cp_time(self.livetime_nano[0])     # TODO: Test if this works better with seconds or nanoseconds
        if FIRST_PASS:
            ModelStructure.update_shared_cp_time(self.livetime_nano[0])
        else:
            ModelStructure.update_shared_cp_timestep(dt)

        # self.rol_velo.update_data(msg.angular_velocity.x)
        self.pit_velo.update_data(msg.angular_velocity.y)
        self.yaw_velo.update_data(msg.angular_velocity.z)
        self.xdir_accel.update_data(msg.linear_acceleration.x)
        self.ydir_accel.update_data(msg.linear_acceleration.y)
        self.zdir_accel.update_data(msg.linear_acceleration.z)
        # self.rol_accel.update_data(sg_diff(self.livetime_nano.data, self.rol_velo.data))
        self.pit_accel.update_data(sg_diff(np.array(self.livetime_nano)[::-1], self.pit_velo.data))
        self.yaw_accel.update_data(sg_diff(np.array(self.livetime_nano)[::-1], self.yaw_velo.data))
        

        # cutoff_frequency = 18   # [Hz] 1.2*f_system_dynamics (15 Hz)
        # dt = 0.02   # Assumed average dt    TODO: Confirm that this is the proper dt with the mixed IMUs
        # alpha = 1 - np.exp(-2 * np.pi * cutoff_frequency * dt)
        # self.rol_velo.update_data((alpha * msg.angular_velocity.x) + ((1- alpha) * np.mean(self.rol_velo.data[1:])))
        # self.pit_velo.update_data((alpha * msg.angular_velocity.y) + ((1- alpha) * np.mean(self.pit_velo.data[1:])))
        # self.yaw_velo.update_data((alpha * msg.angular_velocity.z) + ((1- alpha) * np.mean(self.yaw_velo.data[1:])))
        
        self.rol_velo.update_data(self.lpf1.update(msg.angular_velocity.x, dt))

        # cutoff_frequency = 4   # [Hz] ~(0.5-0.7) the value of the gyro cutoff frequency
        # dt = 0.02   # Assumed average dt    TODO: Confirm that this is the proper dt with the mixed IMUs
        # alpha = 1 - np.exp(-2 * np.pi * cutoff_frequency * dt)
        # # dt = self.livetime_nano.data[-1] - self.livetime_nano.data[-2]
        # # alpha = np.clip(dt / 0.05, 0.1, 0.9)
        # # alpha = np.exp(-dt / 0.03)  # estimate_derivative()
        # self.rol_accel.update_data((alpha * sg_diff(self.livetime_nano.data, self.rol_velo.data)) + ((1 - alpha) * np.mean(self.rol_accel.data[1:])))
        # self.pit_accel.update_data((alpha * sg_diff(self.livetime_nano.data, self.pit_velo.data)) + ((1 - alpha) * np.mean(self.pit_accel.data[1:])))
        # self.yaw_accel.update_data((alpha * sg_diff(self.livetime_nano.data, self.yaw_velo.data)) + ((1 - alpha) * np.mean(self.yaw_accel.data[1:])))

        self.rol_accel.update_data(self.lpf2.update(sg_diff(np.array(self.livetime_nano)[::-1], self.rol_velo.data), dt))
                
    def rcout_callback(self, msg: RCOut) -> None:
        self.ail_pwm.update_data(msg.channels[0])
        self.elv_pwm.update_data(msg.channels[1])
        self.rud_pwm.update_data(msg.channels[2])

    def odom_callback(self, msg: Odometry) -> None:
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

    def diff_pressure_callback(self, msg: FluidPressure) -> None:
        self.dyn_pres.update_data(msg.fluid_pressure)

    def static_pressure_callback(self, msg: FluidPressure) -> None:
        self.stat_pres.update_data(msg.fluid_pressure)

    def temperature_baro_callback(self, msg: Temperature) -> None:
        self.temp.update_data(msg.temperature)


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

        self.ols_rol_large_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol_large', 10)
        timer_period: float = 0.02
        self.ols_rol_large_timer = self.create_timer(
            timer_period, self.publish_ols_rol_large_data)

        self.ols_rol_yaw_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol_yaw', 10)
        timer_period: float = 0.02
        self.ols_rol_yaw_timer = self.create_timer(
            timer_period, self.publish_ols_rol_yaw_data)

        self.ols_rol_moment_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol_moment', 10)
        timer_period: float = 0.02
        self.ols_rol_moment_timer = self.create_timer(
            timer_period, self.publish_ols_rol_moment_data)

        self.ols_Y_dim_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_Y_dim', 10)
        timer_period: float = 0.02
        self.ols_Y_dim_timer = self.create_timer(
            timer_period, self.publish_ols_Y_dim_data)

        self.ols_rol_nondim_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol_nondim', 10)
        timer_period: float = 0.02
        self.ols_rol_nondim_timer = self.create_timer(
            timer_period, self.publish_ols_rol_nondim_data)
        
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
        
    def publish_ols_rol_large_data(self) -> None:
        self.rol_large.update_model(self.rol_accel.data[0], [self.rol_velo.data[0], self.ail_pwm.data[0], self.yaw_velo.data[0], self.rud_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.rol_accel.data.item(0)),

            np.float64(self.rol_velo.data.item(0)),
            np.float64(self.ail_pwm.data.item(0)),
            np.float64(self.yaw_velo.data.item(0)),
            np.float64(self.rud_pwm.data.item(0)),

            self.rol_large.parameters[0],
            self.rol_large.parameters[1],
            self.rol_large.parameters[2],
            self.rol_large.parameters[3]
            ]
        self.ols_rol_large_publisher.publish(msg)

    def publish_ols_rol_yaw_data(self) -> None:
        self.rol_yaw.update_model(self.rol_accel.data[0], [self.rol_velo.data[0], self.ail_pwm.data[0], self.yaw_velo.data[0], self.rud_pwm.data[0], self.yaw_accel.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.rol_accel.data.item(0)),

            np.float64(self.rol_velo.data.item(0)),
            np.float64(self.ail_pwm.data.item(0)),
            np.float64(self.yaw_velo.data.item(0)),
            np.float64(self.rud_pwm.data.item(0)),
            np.float64(self.yaw_accel.data.item(0)),

            self.rol_yaw.parameters[0],
            self.rol_yaw.parameters[1],
            self.rol_yaw.parameters[2],
            self.rol_yaw.parameters[3],
            self.rol_yaw.parameters[4]
            ]
        self.ols_rol_yaw_publisher.publish(msg)

    def publish_ols_rol_moment_data(self) -> None:
        moment = (self.rol_accel.data[0] - self.yaw_accel.data[0]) - self.pit_velo.data[0] * (self.rol_velo.data[0] - self.yaw_velo.data[0])

        self.rol_moment.update_model(moment, [self.rol_velo.data[0], self.ail_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(moment.item(0)),

            np.float64(self.rol_velo.data.item(0)),
            np.float64(self.ail_pwm.data.item(0)),

            self.rol_moment.parameters[0],
            self.rol_moment.parameters[1]
            ]
        self.ols_rol_moment_publisher.publish(msg)

    def publish_ols_Y_dim_data(self) -> None:
        Y_force_dim = self.mass * self.ydir_accel.data[0]

        self.Y_dim.update_model(Y_force_dim, [self.rol_velo.data[0], self.ail_pwm.data[0], self.yaw_velo.data[0], self.rud_pwm.data[0]])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(Y_force_dim.item(0)),

            np.float64(self.rol_velo.data.item(0)),
            np.float64(self.ail_pwm.data.item(0)),
            np.float64(self.yaw_velo.data.item(0)),
            np.float64(self.rud_pwm.data.item(0)),

            self.Y_dim.parameters[0],
            self.Y_dim.parameters[1],
            self.Y_dim.parameters[2],
            self.Y_dim.parameters[3]
            ]
        self.ols_Y_dim_publisher.publish(msg)

    def publish_ols_rol_nondim_data(self) -> None:
        airdensity = self.stat_pres.data[0] / (1 * self.temp.data[0])
        airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
        Z = self.rol_accel.data[0]
        X1 = self.rol_velo.data[0] * (self.dyn_pres.data[0] / airspeed)
        X2 = (self.ail_pwm.data[0] * self.dyn_pres.data[0])
        X3 = (self.pit_velo.data[0] * self.yaw_velo.data[0])
        X4 = self.yaw_accel.data[0] + (self.rol_velo.data[0] * self.pit_velo.data[0])

        self.rol_nondim.update_model(Z, [X1, X2, X3, X4])

        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.rol_accel.data.item(0)),

            np.float64(self.rol_velo.data.item(0) * (self.dyn_pres.data.item(0) / airspeed.item(0))),
            np.float64((self.ail_pwm.data.item(0) * self.dyn_pres.data.item(0))),
            np.float64((self.pit_velo.data.item(0) * self.yaw_velo.data.item(0))),
            np.float64(self.yaw_accel.data.item(0) + (self.rol_velo.data.item(0) * self.pit_velo.data.item(0))),

            self.rol_nondim.parameters[0],
            self.rol_nondim.parameters[1],
            self.rol_nondim.parameters[2],
            self.rol_nondim.parameters[3]
            ]
        self.ols_rol_nondim_publisher.publish(msg)


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