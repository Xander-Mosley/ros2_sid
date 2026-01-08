#!/usr/bin/env python3

import math
import threading
from re import S
from collections import deque

import numpy as np
import mavros
import pickle as pkl
import time
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure, Temperature
from std_msgs.msg import Float64, Float64MultiArray, String


from drone_interfaces.msg import CtlTraj, Telem
from ros2_sid.rt_ols import ModelStructure, StoredData, RecursiveFourierTransform, ordinary_least_squares
from ros2_sid.rotation_utils import euler_from_quaternion
from ros2_sid.signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT, ButterworthLowPass_VDT_2O
    )


FIRST_PASS = True
IMU_PASS = True
ACC_PASS = True
RCO_PASS = True

class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('ols_node')
        self.setup_variables()
        self.setup_all_subscriptions()
        self.setup_all_publishers()

    def setup_variables(self) -> None:
        # initialize stored data objects
        # self.livetime_sec = StoredData(5, 1)
        # self.livetime_nano = deque([0.0, 0.0, 0.0, 0.0, 0.0],maxlen=5)
        # self.rol_velo = StoredData(5, 1)
        # self.pit_velo = StoredData(5, 1)
        # self.yaw_velo = StoredData(5, 1)
        # self.rol_accel = StoredData(5, 1)
        # self.pit_accel = StoredData(5, 1)
        # self.yaw_accel = StoredData(5, 1)

        # self.ail_pwm = StoredData(5, 1)
        # self.elv_pwm = StoredData(5, 1)
        # self.rud_pwm = StoredData(5, 1)

        self.rol_velo_td = StoredData(1, 1)
        self.rol_accel_td = StoredData(1, 1)
        self.ail_pwm_td = StoredData(1, 1)

        RecursiveFourierTransform.set_defaults(eff=0.999)
        self.rol_velo = RecursiveFourierTransform()
        self.rol_accel = RecursiveFourierTransform()
        self.ail_pwm = RecursiveFourierTransform()

        self.imu_time = deque([0.0, 0.0],maxlen=2)
        self.acc_time = deque([0.0, 0.0],maxlen=2)
        self.rco_time = deque([0.0, 0.0],maxlen=2)


        # self.aoa = StoredData(1, 1)
        # self.ssa = StoredData(1, 1)
        
        # self.dyn_pres = StoredData(1, 1)
        # self.dyn_pres.update_data(1)
        # self.stat_pres = StoredData(1, 1)
        # self.stat_pres.update_data(1)
        # self.temp = StoredData(1, 1)
        # self.temp.update_data(1)
        # self.airspeed = StoredData(1, 1)
        # self.airspeed.update_data(1)

        # self.mass = 25              # [kg]
        # self.wing_span = 3.868      # [m]
        # self.wing_area = 1.065634   # [m²]
        # self.wing_chord = 0.2755    # [m]

        # self.dt = 0.0


        # define class variables
        # ModelStructure.class_eff = 0.999

        # initialize model structure objects
        # self.rol = ModelStructure(2)
        # self.rol_slowed = ModelStructure(2)
        # self.rol_nondim = ModelStructure(2)
        # self.rol_nondim_inertias = ModelStructure(4)
        # self.rol_ssa = ModelStructure(3)
        # self.rol_ssa_nondim = ModelStructure(3)
        # self.rol_ssa_nondim_inertias = ModelStructure(5)

        # self.rol_large = ModelStructure(4)
        # self.rol_large_nondim = ModelStructure(4)
        # self.rol_large_nondim_inertias = ModelStructure(6)
        # self.rol_large_ssa = ModelStructure(5)
        # self.rol_large_ssa_nondim = ModelStructure(5)
        # self.rol_large_ssa_nondim_inertias = ModelStructure(7)

        # self.pit = ModelStructure(2)
        # self.pit_nondim = ModelStructure(2)
        # self.pit_nondim_inertias = ModelStructure(4)
        # self.pit_aoa = ModelStructure(3)
        # self.pit_aoa_nondim = ModelStructure(3)
        # self.pit_aoa_nondim_inertias = ModelStructure(5)

        # self.yaw = ModelStructure(2)
        # self.yaw_nondim = ModelStructure(2)
        # self.yaw_nondim_inertias = ModelStructure(4)
        # self.yaw_ssa = ModelStructure(3)
        # self.yaw_ssa_nondim = ModelStructure(3)
        # self.yaw_ssa_nondim_inertias = ModelStructure(5)

        # self.yaw_large = ModelStructure(4)
        # self.yaw_large_nondim = ModelStructure(4)
        # self.yaw_large_nondim_inertias = ModelStructure(6)
        # self.yaw_large_ssa = ModelStructure(5)
        # self.yaw_large_ssa_nondim = ModelStructure(5)
        # self.yaw_large_ssa_nondim_inertias = ModelStructure(7)



    def setup_all_subscriptions(self) -> None:
        # # TODO: Subscribe to more published data.
        # self.imu_sub: Subscription = self.create_subscription(
        #     Imu,
        #     '/mavros/imu/data',
        #     self.imu_callback,
        #     qos_profile=SENSOR_QOS
        # )

        # self.telem_sub: Subscription = self.create_subscription(
        #     Telem,
        #     '/telem',
        #     self.telem_callback,
        #     qos_profile=SENSOR_QOS
        # )

        self.imu_filtered_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_filt',
            self.imu_filtered_callback,
            qos_profile=SENSOR_QOS
        )

        self.imu_differentiated_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_diff',
            self.imu_differentiated_callback,
            qos_profile=SENSOR_QOS
        )

        self.rcout_sub: Subscription = self.create_subscription(
            RCOut,
            '/mavros/rc/out',
            self.rcout_callback,
            qos_profile=SENSOR_QOS
        )

        # self.odom_sub: Subscription = self.create_subscription(
        #     Odometry,
        #     '/mavros/local_position/odom',
        #     self.odom_callback,
        #     qos_profile=SENSOR_QOS
        # )

        # self.diff_pressure_sub: Subscription = self.create_subscription(
        #     FluidPressure,
        #     '/mavros/imu/diff_pressure',
        #     self.diff_pressure_callback,
        #     qos_profile=SENSOR_QOS
        # )

        # self.static_pressure_sub: Subscription = self.create_subscription(
        #     FluidPressure,
        #     '/mavros/imu/static_pressure',
        #     self.static_pressure_callback,
        #     qos_profile=SENSOR_QOS
        # )

        # self.temperature_baro_sub: Subscription = self.create_subscription(
        #     Temperature,
        #     '/mavros/imu/temperature_baro',
        #     self.temperature_baro_callback,
        #     qos_profile=SENSOR_QOS
        # )

    # def imu_callback(self, msg: Imu) -> None:
    #     # start = time.perf_counter()
    #     # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
    #     global FIRST_PASS

    #     self.livetime_sec.update_data(msg.header.stamp.sec)
    #     new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
    #     if new_nanosec_data < self.livetime_nano[-1]:
    #         new_nanosec_data += 1.0

    #     self.dt = new_nanosec_data - self.livetime_nano[-1]
    #     if self.dt > (1 / 150):
    #         self.livetime_nano.append(new_nanosec_data)
    #         if len(self.livetime_nano) > 0 and all(x >= 1.0 for x in self.livetime_nano):
    #             self.livetime_nano = deque([x - 1.0 for x in self.livetime_nano], maxlen=self.livetime_nano.maxlen)

    #         # ModelStructure.update_shared_cp_time(self.livetime_nano[0])     # TODO: Test if this works better with seconds or nanoseconds
    #         if FIRST_PASS:
    #             FIRST_PASS = False
    #             ModelStructure.update_shared_cp_time(self.livetime_nano[0])
    #             # self.max_elapsed = 0
    #             # self.min_elapsed = 1
    #             # self.ema_elapsed = 0
    #         else:
    #             ModelStructure.update_shared_cp_timestep(self.dt)

    #         self.rol_velo.update_data(self.lpf_rol_velo.update(msg.angular_velocity.x, self.dt))
    #         self.pit_velo.update_data(self.lpf_pit_velo.update(msg.angular_velocity.y, self.dt))
    #         self.yaw_velo.update_data(self.lpf_yaw_velo.update(msg.angular_velocity.z, self.dt))

    #         self.rol_accel.update_data(self.lpf_rol_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.rol_velo.data), self.dt))
    #         self.pit_accel.update_data(self.lpf_pit_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.pit_velo.data), self.dt))
    #         self.yaw_accel.update_data(self.lpf_yaw_accel.update(poly_diff(np.array(self.livetime_nano)[::-1], self.yaw_velo.data), self.dt))

    #         # first_filter_record = {
    #         #     "dt": self.dt,
    #         #     "new_data": msg.angular_velocity.x,
    #         #     "filtered_data": self.lpf_rol_velo.current()
    #         # }
    #         # deriv_record = {
    #         #     "time": np.array(self.livetime_nano)[::-1],
    #         #     "input_data": self.rol_velo.data,
    #         #     "input_time": poly_diff(np.array(self.livetime_nano)[::-1], self.rol_velo.data)
    #         # }
    #         # second_filter_record = {
    #         #     "dt": self.dt,
    #         #     "new_data": poly_diff(np.array(self.livetime_nano)[::-1], self.rol_velo.data),
    #         #     "filtered_data": self.lpf_rol_accel.current()
    #         # }
    #         # with open("first_filter_record.pkl", "ab") as f1:
    #         #     pkl.dump(first_filter_record, f1)
    #         # with open("deriv_record.pkl", "ab") as f2:
    #         #     pkl.dump(deriv_record, f2)
    #         # with open("second_filter_record.pkl", "ab") as f3:
    #         #     pkl.dump(second_filter_record, f3)

    #         # end = time.perf_counter()
    #         # elapsed = end - start
    #         # self.max_elapsed = np.max([self.max_elapsed, elapsed])
    #         # self.min_elapsed = np.min([self.min_elapsed, elapsed])
    #         # num_pts = 99
    #         # alpha = 2 / (num_pts + 1)
    #         # self.ema_elapsed = (alpha * elapsed) + ((1-alpha) * self.ema_elapsed)
    #         # print(f"imu_callback runtime - avg: {self.ema_elapsed*1e3:.3f} ms\tmax: {self.max_elapsed*1e3:.3f} ms\tmin: {self.min_elapsed*1e3:.3f} ms")

    # def telem_callback(self, msg: Telem) -> None:
    #     self.aoa.update_data(msg.alpha)
    #     self.ssa.update_data(msg.beta)

    def imu_filtered_callback(self, msg: Imu) -> None:
        # global FIRST_PASS
        # if FIRST_PASS:
        #     FIRST_PASS = False
        #     self.last_time = None
        # now = time.monotonic()
        # if self.last_time is not None:
        #     ModelStructure.update_shared_cp_timestep(now - self.last_time)
        # else:
        #     ModelStructure.update_shared_cp_time(0)
        # self.last_time = now
        
        # self.rol_velo.update_data(msg.angular_velocity.x)
        # self.pit_velo.update_data(msg.angular_velocity.y)
        # self.yaw_velo.update_data(msg.angular_velocity.z)
        
        global IMU_PASS
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if new_nanosec_data < self.imu_time[-1]:
            new_nanosec_data += 1.0
        dt = new_nanosec_data - self.imu_time[-1]
        if dt > (1 / 150):
            self.imu_time.append(new_nanosec_data)
            if len(self.imu_time) > 0 and all(x >= 1.0 for x in self.imu_time):
                self.imu_time = deque([x - 1.0 for x in self.imu_time], maxlen=self.imu_time.maxlen)
            if IMU_PASS:
                IMU_PASS = False
                self.rol_velo.update_cp_time(self.imu_time[-1])
            else:
                self.rol_velo.update_cp_timestep(dt)
            
            self.rol_velo.update_spectrum(msg.angular_velocity.x)
            self.rol_velo_td.update_data(msg.angular_velocity.x)

    def imu_differentiated_callback(self, msg: Imu) -> None:
        # self.rol_accel.update_data(msg.data[0])
        # self.pit_accel.update_data(msg.data[1])
        # self.yaw_accel.update_data(msg.data[2])
        
        global ACC_PASS
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if new_nanosec_data < self.acc_time[-1]:
            new_nanosec_data += 1.0
        dt = new_nanosec_data - self.acc_time[-1]
        if dt > (1 / 150):
            self.acc_time.append(new_nanosec_data)
            if len(self.acc_time) > 0 and all(x >= 1.0 for x in self.acc_time):
                self.acc_time = deque([x - 1.0 for x in self.acc_time], maxlen=self.acc_time.maxlen)
            if ACC_PASS:
                ACC_PASS = False
                self.rol_accel.update_cp_time(self.acc_time[-1])
            else:
                self.rol_accel.update_cp_timestep(dt)
        
            self.rol_accel.update_spectrum(msg.angular_velocity.x)
            self.rol_accel_td.update_data(msg.angular_velocity.x)

    def rcout_callback(self, msg: RCOut) -> None:
        # self.ail_pwm.update_data(msg.channels[0])
        # self.elv_pwm.update_data(msg.channels[1])
        # self.rud_pwm.update_data(msg.channels[2])
        
        global RCO_PASS
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if new_nanosec_data < self.rco_time[-1]:
            new_nanosec_data += 1.0
        dt = new_nanosec_data - self.rco_time[-1]
        if dt > (1 / 150):
            self.rco_time.append(new_nanosec_data)
            if len(self.rco_time) > 0 and all(x >= 1.0 for x in self.rco_time):
                self.rco_time = deque([x - 1.0 for x in self.rco_time], maxlen=self.rco_time.maxlen)
            if RCO_PASS:
                RCO_PASS = False
                self.ail_pwm.update_cp_time(self.rco_time[-1])
            else:
                self.ail_pwm.update_cp_timestep(dt)
        
            self.ail_pwm.update_spectrum(msg.channels[0])
            self.ail_pwm_td.update_data(msg.channels[0])

    # def odom_callback(self, msg: Odometry) -> None:
    #     """
    #     Converts NED to ENU and publishes the trajectory
    #     https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
    #     Twist Will show velocity in linear and rotational 
    #     """
    #     # state_x = msg.pose.pose.position.x
    #     # state_y = msg.pose.pose.position.y
    #     # state_z = msg.pose.pose.position.z

    #     # # quaternion attitudes
    #     # qx = msg.pose.pose.orientation.x
    #     # qy = msg.pose.pose.orientation.y
    #     # qz = msg.pose.pose.orientation.z
    #     # qw = msg.pose.pose.orientation.w
    #     # state_phi, state_theta, state_psi = euler_from_quaternion(
    #     #     qx, qy, qz, qw)
        
    #     # state_phi = state_phi
    #     # state_theta = state_theta
    #     # state_psi = state_psi   # (yaw+ (2*np.pi) ) % (2*np.pi);

    #     # get magnitude of velocity
    #     vx = msg.twist.twist.linear.x
    #     vy = msg.twist.twist.linear.y
    #     vz = msg.twist.twist.linear.z
    #     self.airspeed.update_data(np.sqrt(vx**2 + vy**2 + vz**2))

    #     # self.odom_data: list[float] = [
    #     #     state_x,
    #     #     state_y,
    #     #     state_z,
    #     #     state_phi,
    #     #     state_theta,
    #     #     state_psi,
    #     #     self.airspeed
    #     # ]

    # def diff_pressure_callback(self, msg: FluidPressure) -> None:
    #     self.dyn_pres.update_data(msg.fluid_pressure)   # [Pa]

    # def static_pressure_callback(self, msg: FluidPressure) -> None:
    #     self.stat_pres.update_data(msg.fluid_pressure)  # [Pa]

    # def temperature_baro_callback(self, msg: Temperature) -> None:
    #     # self.temp.update_data(msg.temperature)              # [°C]
    #     self.temp.update_data(msg.temperature + 273.15)     # [°K]



    def setup_all_publishers(self) -> None:
        self.ols_rol_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_rol', 10)
        timer_period: float = 0.25
        self.ols_rol_timer = self.create_timer(
            timer_period, self.publish_ols_rol_data)

        # self.ols_rol_slowed_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_slowed', 10)
        # timer_period: float = 0.25
        # self.ols_rol_slowed_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_slowed_data)
        
        # self.ols_rol_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_rol_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_nondim_data)

        # self.ols_rol_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_rol_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_nondim_inertias_data)

        # self.ols_rol_ssa_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_ssa', 10)
        # timer_period: float = 0.02
        # self.ols_rol_ssa_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_ssa_data)

        # self.ols_rol_ssa_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_ssa_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_rol_ssa_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_ssa_nondim_data)

        # self.ols_rol_ssa_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_ssa_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_rol_ssa_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_ssa_nondim_inertias_data)


        # self.ols_rol_large_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_large', 10)
        # timer_period: float = 0.02
        # self.ols_rol_large_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_large_data)

        # self.ols_rol_large_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_large_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_rol_large_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_large_nondim_data)

        # self.ols_rol_large_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_large_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_rol_large_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_large_nondim_inertias_data)

        # self.ols_rol_large_ssa_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_large_ssa', 10)
        # timer_period: float = 0.02
        # self.ols_rol_large_ssa_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_large_ssa_data)

        # self.ols_rol_large_ssa_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_large_ssa_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_rol_large_ssa_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_large_ssa_nondim_data)

        # self.ols_rol_large_ssa_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_rol_large_ssa_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_rol_large_ssa_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_rol_large_ssa_nondim_inertias_data)


        # self.ols_pit_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_pit', 10)
        # timer_period: float = 0.02
        # self.ols_pit_timer = self.create_timer(
        #     timer_period, self.publish_ols_pit_data)
        
        # self.ols_pit_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_pit_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_pit_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_pit_nondim_data)
        
        # self.ols_pit_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_pit_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_pit_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_pit_nondim_inertias_data)
        
        # self.ols_pit_aoa_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_pit_aoa', 10)
        # timer_period: float = 0.02
        # self.ols_pit_aoa_timer = self.create_timer(
        #     timer_period, self.publish_ols_pit_aoa_data)
        
        # self.ols_pit_aoa_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_pit_aoa_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_pit_aoa_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_pit_aoa_nondim_data)
        
        # self.ols_pit_aoa_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_pit_aoa_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_pit_aoa_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_pit_aoa_nondim_inertias_data)


        # self.ols_yaw_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_data)
        
        # self.ols_yaw_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_nondim_data)
        
        # self.ols_yaw_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_nondim_inertias_data)
        
        # self.ols_yaw_ssa_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_ssa', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_ssa_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_ssa_data)
        
        # self.ols_yaw_ssa_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_ssa_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_ssa_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_ssa_nondim_data)
        
        # self.ols_yaw_ssa_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_ssa_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_ssa_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_ssa_nondim_inertias_data)


        # self.ols_yaw_large_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_large', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_large_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_large_data)
        
        # self.ols_yaw_large_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_large_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_large_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_large_nondim_data)
        
        # self.ols_yaw_large_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_large_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_large_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_large_nondim_inertias_data)
        
        # self.ols_yaw_large_ssa_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_large_ssa', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_large_ssa_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_large_ssa_data)
        
        # self.ols_yaw_large_ssa_nondim_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_large_ssa_nondim', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_large_ssa_nondim_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_large_ssa_nondim_data)
        
        # self.ols_yaw_large_ssa_nondim_inertias_publisher: Publisher = self.create_publisher(
        #         Float64MultiArray, 'ols_yaw_large_ssa_nondim_inertias', 10)
        # timer_period: float = 0.02
        # self.ols_yaw_large_ssa_nondim_inertias_timer = self.create_timer(
        #     timer_period, self.publish_ols_yaw_large_ssa_nondim_inertias_data)

    def publish_ols_rol_data(self) -> None:
        # Z = self.rol_accel.data[0]
        # X1 = self.rol_velo.data[0]
        # X2 = self.ail_pwm.data[0]

        # self.rol.update_model(Z, [X1, X2])

        # msg: Float64MultiArray = Float64MultiArray()
        # msg.data = [
        #     np.float64(Z.item(0)),

        #     np.float64(X1.item(0)),
        #     np.float64(X2.item(0)),

        #     self.rol.parameters[0],
        #     self.rol.parameters[1]
        #     ]
        # self.ols_rol_publisher.publish(msg)

        Z = self.rol_accel.current_spectrum
        X1 = self.rol_velo.current_spectrum
        X2 = self.ail_pwm.current_spectrum
        parameters = ordinary_least_squares(Z, np.column_stack((X1, X2)))
        
        msg: Float64MultiArray = Float64MultiArray()
        msg.data = [
            np.float64(self.rol_accel_td.data.item(0)),

            np.float64(self.rol_velo_td.data.item(0)),
            np.float64(self.ail_pwm_td.data.item(0)),

            parameters[0],
            parameters[1]
            ]
        self.ols_rol_publisher.publish(msg)



    # def publish_ols_rol_slowed_data(self) -> None:
    #     Z = self.rol_accel.data[0]
    #     X1 = self.rol_velo.data[0]
    #     X2 = self.ail_pwm.data[0]

    #     self.rol_slowed.update_model(Z, [X1, X2])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),

    #         self.rol_slowed.parameters[0],
    #         self.rol_slowed.parameters[1]
    #         ]
    #     self.ols_rol_slowed_publisher.publish(msg)   

    # def publish_ols_rol_nondim_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X2 = self.dyn_pres.data[0] * self.ail_pwm.data[0]

    #     self.rol_nondim.update_model(Z, [X1, X2])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),

    #         self.rol_nondim.parameters[0],
    #         self.rol_nondim.parameters[1]
    #         ]
    #     self.ols_rol_nondim_publisher.publish(msg)

    # def publish_ols_rol_nondim_inertias_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X2 = self.dyn_pres.data[0] * self.ail_pwm.data[0]
    #     X3 = self.pit_velo.data[0] * self.yaw_velo.data[0]
    #     X4 = self.yaw_accel.data[0] + (self.rol_velo.data[0] * self.pit_velo.data[0])

    #     self.rol_nondim_inertias.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.rol_nondim_inertias.parameters[0],
    #         self.rol_nondim_inertias.parameters[1],
    #         self.rol_nondim_inertias.parameters[2],
    #         self.rol_nondim_inertias.parameters[3]
    #         ]
    #     self.ols_rol_nondim_inertias_publisher.publish(msg)

    # def publish_ols_rol_ssa_data(self) -> None:
    #     self.rol_ssa.update_model(self.rol_accel.data[0], [self.ssa.data[0], self.rol_velo.data[0], self.ail_pwm.data[0]])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(self.rol_accel.data.item(0)),

    #         np.float64(self.ssa.data.item(0)),
    #         np.float64(self.rol_velo.data.item(0)),
    #         np.float64(self.ail_pwm.data.item(0)),

    #         self.rol_ssa.parameters[0],
    #         self.rol_ssa.parameters[1],
    #         self.rol_ssa.parameters[2]
    #         ]
    #     self.ols_rol_ssa_publisher.publish(msg)

    # def publish_ols_rol_ssa_nondim_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.ssa.data[0]
    #     X2 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X3 = self.dyn_pres.data[0] * self.ail_pwm.data[0]

    #     self.rol_ssa_nondim.update_model(Z, [X1, X2, X3])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),

    #         self.rol_ssa_nondim.parameters[0],
    #         self.rol_ssa_nondim.parameters[1],
    #         self.rol_ssa_nondim.parameters[2]
    #         ]
    #     self.ols_rol_ssa_nondim_publisher.publish(msg)

    # def publish_ols_rol_ssa_nondim_inertias_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.ssa.data[0]
    #     X2 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X3 = self.dyn_pres.data[0] * self.ail_pwm.data[0]
    #     X4 = self.pit_velo.data[0] * self.yaw_velo.data[0]
    #     X5 = self.yaw_accel.data[0] + (self.rol_velo.data[0] * self.pit_velo.data[0])

    #     self.rol_ssa_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),

    #         self.rol_ssa_nondim_inertias.parameters[0],
    #         self.rol_ssa_nondim_inertias.parameters[1],
    #         self.rol_ssa_nondim_inertias.parameters[2],
    #         self.rol_ssa_nondim_inertias.parameters[3],
    #         self.rol_ssa_nondim_inertias.parameters[4]
    #         ]
    #     self.ols_rol_ssa_nondim_inertias_publisher.publish(msg)


    # def publish_ols_rol_large_data(self) -> None:
    #     Z = self.rol_accel.data[0]
    #     X1 = self.rol_velo.data[0]
    #     X2 = self.ail_pwm.data[0]
    #     X3 = self.yaw_velo.data[0]
    #     X4 = self.rud_pwm.data[0]

    #     self.rol_large.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.rol_large.parameters[0],
    #         self.rol_large.parameters[1],
    #         self.rol_large.parameters[2],
    #         self.rol_large.parameters[3]
    #         ]
    #     self.ols_rol_large_publisher.publish(msg)

    # def publish_ols_rol_large_nondim_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X2 = self.dyn_pres.data[0] * self.yaw_velo.data[0] / airspeed
    #     X3 = self.dyn_pres.data[0] * self.ail_pwm.data[0]
    #     X4 = self.dyn_pres.data[0] * self.rud_pwm.data[0]

    #     self.rol_large_nondim.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.rol_large_nondim.parameters[0],
    #         self.rol_large_nondim.parameters[1],
    #         self.rol_large_nondim.parameters[2],
    #         self.rol_large_nondim.parameters[3]
    #         ]
    #     self.ols_rol_large_nondim_publisher.publish(msg)

    # def publish_ols_rol_large_nondim_inertias_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X2 = self.dyn_pres.data[0] * self.yaw_velo.data[0] / airspeed
    #     X3 = self.dyn_pres.data[0] * self.ail_pwm.data[0]
    #     X4 = self.dyn_pres.data[0] * self.rud_pwm.data[0]
    #     X5 = self.pit_velo.data[0] * self.yaw_velo.data[0]
    #     X6 = self.yaw_accel.data[0] + (self.rol_velo.data[0] * self.pit_velo.data[0])

    #     self.rol_large_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5, X6])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),
    #         np.float64(X6.item(0)),

    #         self.rol_large_nondim_inertias.parameters[0],
    #         self.rol_large_nondim_inertias.parameters[1],
    #         self.rol_large_nondim_inertias.parameters[2],
    #         self.rol_large_nondim_inertias.parameters[3],
    #         self.rol_large_nondim_inertias.parameters[4],
    #         self.rol_large_nondim_inertias.parameters[5]
    #         ]
    #     self.ols_rol_large_nondim_inertias_publisher.publish(msg)
        
    # def publish_ols_rol_large_ssa_data(self) -> None:
    #     self.rol_large_ssa.update_model(self.rol_accel.data[0], [self.ssa.data[0], self.rol_velo.data[0], self.ail_pwm.data[0], self.yaw_velo.data[0], self.rud_pwm.data[0]])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(self.rol_accel.data.item(0)),

    #         np.float64(self.ssa.data.item(0)),
    #         np.float64(self.rol_velo.data.item(0)),
    #         np.float64(self.ail_pwm.data.item(0)),
    #         np.float64(self.yaw_velo.data.item(0)),
    #         np.float64(self.rud_pwm.data.item(0)),

    #         self.rol_large_ssa.parameters[0],
    #         self.rol_large_ssa.parameters[1],
    #         self.rol_large_ssa.parameters[2],
    #         self.rol_large_ssa.parameters[3],
    #         self.rol_large_ssa.parameters[4]
    #         ]
    #     self.ols_rol_large_ssa_publisher.publish(msg)

    # def publish_ols_rol_large_ssa_nondim_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.ssa.data[0]
    #     X2 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X3 = self.dyn_pres.data[0] * self.yaw_velo.data[0] / airspeed
    #     X4 = self.dyn_pres.data[0] * self.ail_pwm.data[0]
    #     X5 = self.dyn_pres.data[0] * self.rud_pwm.data[0]

    #     self.rol_large_ssa_nondim.update_model(Z, [X1, X2, X3, X4, X5])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),

    #         self.rol_large_ssa_nondim.parameters[0],
    #         self.rol_large_ssa_nondim.parameters[1],
    #         self.rol_large_ssa_nondim.parameters[2],
    #         self.rol_large_ssa_nondim.parameters[3],
    #         self.rol_large_ssa_nondim.parameters[4]
    #         ]
    #     self.ols_rol_large_ssa_nondim_publisher.publish(msg)

    # def publish_ols_rol_large_ssa_nondim_inertias_data(self) -> None:
    #     R_dryair = 287.05    # [J/kg-K], specific gas constant of dry air. TODO: Consider humidity of the air?
    #     airdensity = self.stat_pres.data[0] / (R_dryair * self.temp.data[0])
    #     airspeed = np.sqrt((2 * self.dyn_pres.data[0]) / airdensity)
        
    #     Z = self.rol_accel.data[0]
    #     X1 = self.dyn_pres.data[0] * self.ssa.data[0]
    #     X2 = self.dyn_pres.data[0] * self.rol_velo.data[0] / airspeed
    #     X3 = self.dyn_pres.data[0] * self.yaw_velo.data[0] / airspeed
    #     X4 = self.dyn_pres.data[0] * self.ail_pwm.data[0]
    #     X5 = self.dyn_pres.data[0] * self.rud_pwm.data[0]
    #     X6 = self.pit_velo.data[0] * self.yaw_velo.data[0]
    #     X7 = self.yaw_accel.data[0] + (self.rol_velo.data[0] * self.pit_velo.data[0])

    #     self.rol_large_ssa_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5, X6, X7])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),
    #         np.float64(X6.item(0)),
    #         np.float64(X7.item(0)),

    #         self.rol_large_ssa_nondim_inertias.parameters[0],
    #         self.rol_large_ssa_nondim_inertias.parameters[1],
    #         self.rol_large_ssa_nondim_inertias.parameters[2],
    #         self.rol_large_ssa_nondim_inertias.parameters[3],
    #         self.rol_large_ssa_nondim_inertias.parameters[4],
    #         self.rol_large_ssa_nondim_inertias.parameters[5],
    #         self.rol_large_ssa_nondim_inertias.parameters[6]
    #         ]
    #     self.ols_rol_large_ssa_nondim_inertias_publisher.publish(msg)


    # def publish_ols_pit_data(self) -> None:
    #     Z = self.pit_accel.data[0]
    #     X1 = self.pit_velo.data[0]
    #     X2 = self.elv_pwm.data[0]
        
    #     self.pit.update_model(Z, [X1, X2])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),

    #         self.pit.parameters[0],
    #         self.pit.parameters[1]
    #         ]
    #     self.ols_pit_publisher.publish(msg)

    # def publish_ols_pit_nondim_data(self) -> None:
    #     Z = self.pit_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_chord / 1  # assuming Iy=1 for now
    #     X1 = moment_coefficients * self.pit_velo.data[0] * (self.wing_chord / (2 * self.airspeed.data[0]))
    #     X2 = moment_coefficients * self.elv_pwm.data[0]
        
    #     self.pit_nondim.update_model(Z, [X1, X2])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),

    #         self.pit_nondim.parameters[0],
    #         self.pit_nondim.parameters[1]
    #         ]
    #     self.ols_pit_nondim_publisher.publish(msg)

    # def publish_ols_pit_nondim_inertias_data(self) -> None:
    #     Z = self.pit_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_chord / 1  # assuming Iy=1 for now
    #     X1 = moment_coefficients * self.pit_velo.data[0] * (self.wing_chord / (2 * self.airspeed.data[0]))
    #     X2 = moment_coefficients * self.elv_pwm.data[0]
    #     X3 = self.rol_velo.data[0] * self.yaw_velo.data[0]
    #     X4 = (self.yaw_velo.data[0] * self.yaw_velo.data[0]) - (self.rol_velo.data[0] * self.rol_velo.data[0])
        
    #     self.pit_nondim_inertias.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.pit_nondim_inertias.parameters[0],
    #         self.pit_nondim_inertias.parameters[1],
    #         self.pit_nondim_inertias.parameters[2],
    #         self.pit_nondim_inertias.parameters[3]
    #         ]
    #     self.ols_pit_nondim_inertias_publisher.publish(msg)

    # def publish_ols_pit_aoa_data(self) -> None:
    #     Z = self.pit_accel.data[0]
    #     X1 = self.aoa.data[0]
    #     X2 = self.pit_velo.data[0]
    #     X3 = self.elv_pwm.data[0]
        
    #     self.pit_aoa.update_model(Z, [X1, X2, X3])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),

    #         self.pit_aoa.parameters[0],
    #         self.pit_aoa.parameters[1],
    #         self.pit_aoa.parameters[2]
    #         ]
    #     self.ols_pit_aoa_publisher.publish(msg)

    # def publish_ols_pit_aoa_nondim_data(self) -> None:
    #     Z = self.pit_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_chord / 1  # assuming Iy=1 for now
    #     X1 = moment_coefficients * self.aoa.data[0]
    #     X2 = moment_coefficients * self.pit_velo.data[0] * (self.wing_chord / (2 * self.airspeed.data[0]))
    #     X3 = moment_coefficients * self.elv_pwm.data[0]
        
    #     self.pit_aoa_nondim.update_model(Z, [X1, X2, X3])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),

    #         self.pit_aoa_nondim.parameters[0],
    #         self.pit_aoa_nondim.parameters[1],
    #         self.pit_aoa_nondim.parameters[2]
    #         ]
    #     self.ols_pit_aoa_nondim_publisher.publish(msg)

    # def publish_ols_pit_aoa_nondim_inertias_data(self) -> None:
    #     Z = self.pit_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_chord / 1  # assuming Iy=1 for now
    #     X1 = moment_coefficients * self.aoa.data[0]
    #     X2 = moment_coefficients * self.pit_velo.data[0] * (self.wing_chord / (2 * self.airspeed.data[0]))
    #     X3 = moment_coefficients * self.elv_pwm.data[0]
    #     X4 = self.rol_velo.data[0] * self.yaw_velo.data[0]
    #     X5 = (self.yaw_velo.data[0] * self.yaw_velo.data[0]) - (self.rol_velo.data[0] * self.rol_velo.data[0])
        
    #     self.pit_aoa_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),

    #         self.pit_aoa_nondim_inertias.parameters[0],
    #         self.pit_aoa_nondim_inertias.parameters[1],
    #         self.pit_aoa_nondim_inertias.parameters[2],
    #         self.pit_aoa_nondim_inertias.parameters[3],
    #         self.pit_aoa_nondim_inertias.parameters[4]
    #         ]
    #     self.ols_pit_aoa_nondim_inertias_publisher.publish(msg)


    # def publish_ols_yaw_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     X1 = self.yaw_velo.data[0]
    #     X2 = self.rud_pwm.data[0]

    #     self.yaw.update_model(Z, [X1, X2])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),

    #         self.yaw.parameters[0],
    #         self.yaw.parameters[1]
    #         ]
    #     self.ols_yaw_publisher.publish(msg)

    # def publish_ols_yaw_nondim_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X2 = moment_coefficients * self.rud_pwm.data[0]

    #     self.yaw_nondim.update_model(Z, [X1, X2])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),

    #         self.yaw_nondim.parameters[0],
    #         self.yaw_nondim.parameters[1]
    #         ]
    #     self.ols_yaw_nondim_publisher.publish(msg)

    # def publish_ols_yaw_nondim_inertias_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X2 = moment_coefficients * self.rud_pwm.data[0]
    #     X3 = self.rol_velo.data[0] * self.pit_velo.data[0]
    #     X4 = self.rol_accel.data[0] - (self.pit_velo.data[0] * self.yaw_velo.data[0])

    #     self.yaw_nondim_inertias.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.yaw_nondim_inertias.parameters[0],
    #         self.yaw_nondim_inertias.parameters[1],
    #         self.yaw_nondim_inertias.parameters[2],
    #         self.yaw_nondim_inertias.parameters[3]
    #         ]
    #     self.ols_yaw_nondim_inertias_publisher.publish(msg)

    # def publish_ols_yaw_ssa_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     X1 = self.ssa.data[0]
    #     X2 = self.yaw_velo.data[0]
    #     X3 = self.rud_pwm.data[0]

    #     self.yaw_ssa.update_model(Z, [X1, X2, X3])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),

    #         self.yaw_ssa.parameters[0],
    #         self.yaw_ssa.parameters[1],
    #         self.yaw_ssa.parameters[2]
    #         ]
    #     self.ols_yaw_ssa_publisher.publish(msg)

    # def publish_ols_yaw_ssa_nondim_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.ssa.data[0]
    #     X2 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X3 = moment_coefficients * self.rud_pwm.data[0]

    #     self.yaw_ssa_nondim.update_model(Z, [X1, X2, X3])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),

    #         self.yaw_ssa_nondim.parameters[0],
    #         self.yaw_ssa_nondim.parameters[1],
    #         self.yaw_ssa_nondim.parameters[2]
    #         ]
    #     self.ols_yaw_ssa_nondim_publisher.publish(msg)

    # def publish_ols_yaw_ssa_nondim_inertias_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.ssa.data[0]
    #     X2 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X3 = moment_coefficients * self.rud_pwm.data[0]
    #     X4 = self.rol_velo.data[0] * self.pit_velo.data[0]
    #     X5 = self.rol_accel.data[0] - (self.pit_velo.data[0] * self.yaw_velo.data[0])

    #     self.yaw_ssa_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),

    #         self.yaw_ssa_nondim_inertias.parameters[0],
    #         self.yaw_ssa_nondim_inertias.parameters[1],
    #         self.yaw_ssa_nondim_inertias.parameters[2],
    #         self.yaw_ssa_nondim_inertias.parameters[3],
    #         self.yaw_ssa_nondim_inertias.parameters[4]
    #         ]
    #     self.ols_yaw_ssa_nondim_inertias_publisher.publish(msg)


    # def publish_ols_yaw_large_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     X1 = self.yaw_velo.data[0]
    #     X2 = self.rud_pwm.data[0]
    #     X3 = self.rol_velo.data[0]
    #     X4 = self.ail_pwm.data[0]

    #     self.yaw_large.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.yaw_large.parameters[0],
    #         self.yaw_large.parameters[1],
    #         self.yaw_large.parameters[2],
    #         self.yaw_large.parameters[3]
    #         ]
    #     self.ols_yaw_large_publisher.publish(msg)

    # def publish_ols_yaw_large_nondim_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X2 = moment_coefficients * self.rud_pwm.data[0]
    #     X3 = moment_coefficients * self.rol_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X4 = moment_coefficients * self.ail_pwm.data[0]

    #     self.yaw_large_nondim.update_model(Z, [X1, X2, X3, X4])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),

    #         self.yaw_large_nondim.parameters[0],
    #         self.yaw_large_nondim.parameters[1],
    #         self.yaw_large_nondim.parameters[2],
    #         self.yaw_large_nondim.parameters[3]
    #         ]
    #     self.ols_yaw_large_nondim_publisher.publish(msg)

    # def publish_ols_yaw_large_nondim_inertias_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X2 = moment_coefficients * self.rud_pwm.data[0]
    #     X3 = moment_coefficients * self.rol_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X4 = moment_coefficients * self.ail_pwm.data[0]
    #     X5 = self.rol_velo.data[0] * self.pit_velo.data[0]
    #     X6 = self.rol_accel.data[0] - (self.pit_velo.data[0] * self.yaw_velo.data[0])

    #     self.yaw_large_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5, X6])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),
    #         np.float64(X6.item(0)),

    #         self.yaw_large_nondim_inertias.parameters[0],
    #         self.yaw_large_nondim_inertias.parameters[1],
    #         self.yaw_large_nondim_inertias.parameters[2],
    #         self.yaw_large_nondim_inertias.parameters[3],
    #         self.yaw_large_nondim_inertias.parameters[4],
    #         self.yaw_large_nondim_inertias.parameters[5]
    #         ]
    #     self.ols_yaw_large_nondim_inertias_publisher.publish(msg)

    # def publish_ols_yaw_large_ssa_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     X1 = self.ssa.data[0]
    #     X2 = self.yaw_velo.data[0]
    #     X3 = self.rud_pwm.data[0]
    #     X4 = self.rol_velo.data[0]
    #     X5 = self.ail_pwm.data[0]

    #     self.yaw_large_ssa.update_model(Z, [X1, X2, X3, X4, X5])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),

    #         self.yaw_large_ssa.parameters[0],
    #         self.yaw_large_ssa.parameters[1],
    #         self.yaw_large_ssa.parameters[2],
    #         self.yaw_large_ssa.parameters[3],
    #         self.yaw_large_ssa.parameters[4]
    #         ]
    #     self.ols_yaw_large_ssa_publisher.publish(msg)

    # def publish_ols_yaw_large_ssa_nondim_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.ssa.data[0]
    #     X2 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X3 = moment_coefficients * self.rud_pwm.data[0]
    #     X4 = moment_coefficients * self.rol_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X5 = moment_coefficients * self.ail_pwm.data[0]

    #     self.yaw_large_ssa_nondim.update_model(Z, [X1, X2, X3, X4, X5])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),

    #         self.yaw_large_ssa_nondim.parameters[0],
    #         self.yaw_large_ssa_nondim.parameters[1],
    #         self.yaw_large_ssa_nondim.parameters[2],
    #         self.yaw_large_ssa_nondim.parameters[3],
    #         self.yaw_large_ssa_nondim.parameters[4]
    #         ]
    #     self.ols_yaw_large_ssa_nondim_publisher.publish(msg)

    # def publish_ols_yaw_large_ssa_nondim_inertias_data(self) -> None:
    #     Z = self.yaw_accel.data[0]
    #     moment_coefficients = self.dyn_pres.data[0] * self.wing_area * self.wing_span / 1  # assuming Iz=1 for now
    #     X1 = moment_coefficients * self.ssa.data[0]
    #     X2 = moment_coefficients * self.yaw_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X3 = moment_coefficients * self.rud_pwm.data[0]
    #     X4 = moment_coefficients * self.rol_velo.data[0] * (self.wing_span / (2 * self.airspeed.data[0]))
    #     X5 = moment_coefficients * self.ail_pwm.data[0]
    #     X6 = self.rol_velo.data[0] * self.pit_velo.data[0]
    #     X7 = self.rol_accel.data[0] - (self.pit_velo.data[0] * self.yaw_velo.data[0])

    #     self.yaw_large_ssa_nondim_inertias.update_model(Z, [X1, X2, X3, X4, X5, X6, X7])

    #     msg: Float64MultiArray = Float64MultiArray()
    #     msg.data = [
    #         np.float64(Z.item(0)),

    #         np.float64(X1.item(0)),
    #         np.float64(X2.item(0)),
    #         np.float64(X3.item(0)),
    #         np.float64(X4.item(0)),
    #         np.float64(X5.item(0)),
    #         np.float64(X6.item(0)),
    #         np.float64(X7.item(0)),

    #         self.yaw_large_ssa_nondim_inertias.parameters[0],
    #         self.yaw_large_ssa_nondim_inertias.parameters[1],
    #         self.yaw_large_ssa_nondim_inertias.parameters[2],
    #         self.yaw_large_ssa_nondim_inertias.parameters[3],
    #         self.yaw_large_ssa_nondim_inertias.parameters[4],
    #         self.yaw_large_ssa_nondim_inertias.parameters[5],
    #         self.yaw_large_ssa_nondim_inertias.parameters[6]
    #         ]
    #     self.ols_yaw_large_ssa_nondim_inertias_publisher.publish(msg)


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