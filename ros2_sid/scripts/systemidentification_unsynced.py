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
from rclpy.timer import Timer
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from mavros.base import SENSOR_QOS
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure, Temperature
from std_msgs.msg import Float64, Float64MultiArray, String


from drone_interfaces.msg import CtlTraj, Telem
from ros2_sid.rt_ols import CircularBuffer, RecursiveFourierTransform, RegressorData, ordinary_least_squares
from ros2_sid.rotation_utils import euler_from_quaternion
from ros2_sid.signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT, ButterworthLowPass_VDT_2O
    )


class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('ols_node')
        self.setup_variables()
        self.setup_all_subscriptions()
        self.setup_all_publishers()

    def setup_variables(self) -> None:
        self.imu_time = CircularBuffer(2)
        self.acc_time = CircularBuffer(2)
        self.rco_time = CircularBuffer(2)

        self._imu_pass = True
        self._acc_pass = True
        self._rco_pass = True

        self.rol_velo = RegressorData(delay=0, eff=0.999)
        self.pit_velo = RegressorData(delay=0, eff=0.999)
        self.yaw_velo = RegressorData(delay=0, eff=0.999)
        
        self.rol_accel = RegressorData(delay=0, eff=0.999)
        self.pit_accel = RegressorData(delay=0, eff=0.999)
        self.yaw_accel = RegressorData(delay=0, eff=0.999)

        self.ail_pwm = RegressorData(delay=11, eff=0.999)
        self.elv_pwm = RegressorData(delay=11, eff=0.999)
        self.rud_pwm = RegressorData(delay=11, eff=0.999)

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


    def setup_all_subscriptions(self) -> None:
        # self.imu_sub: Subscription = self.create_subscription(
        #     Imu,
        #     '/mavros/imu/data',
        #     self.imu_callback,
        #     qos_profile=SENSOR_QOS
        # )
        self.imu_filtered_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_filt',
            self.imu_filtered_callback,
            qos_profile=SENSOR_QOS
        )
        # self.telem_sub: Subscription = self.create_subscription(
        #     Telem,
        #     '/telem',
        #     self.telem_callback,
        #     qos_profile=SENSOR_QOS
        # )

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
        # self.replay_rcout_sub: Subscription = self.create_subscription(
        #     Float64MultiArray,
        #     '/replay/RCOU/data',
        #     self.replay_rcout_callback,
        #     qos_profile=SENSOR_QOS
        # )

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

    def imu_callback(self, msg: Imu) -> None:
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if self.imu_time.size > 0 and new_nanosec_data < self.imu_time.latest:
            new_nanosec_data += 1.0
        if self.imu_time.size > 0:
            dt = new_nanosec_data - self.imu_time.latest
        else:
            dt = 0.0
        if dt > (1.0 / 150.0) or self.imu_time.size == 0:
            self.imu_time.add(new_nanosec_data)
            if self.imu_time.size > 0 and np.all(self.imu_time.get_all() >= 1.0):
                self.imu_time.apply_to_all(lambda x: x - 1.0)

            if self._imu_pass:
                self._imu_pass = False
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_time(self.imu_time.oldest)
            else:
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_timestep(dt)

            self.rol_velo.update(msg.angular_velocity.x)
            self.pit_velo.update(msg.angular_velocity.y)
            self.yaw_velo.update(msg.angular_velocity.z)
    def imu_filtered_callback(self, msg: Imu) -> None:
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if self.imu_time.size > 0 and new_nanosec_data < self.imu_time.latest:
            new_nanosec_data += 1.0
        if self.imu_time.size > 0:
            dt = new_nanosec_data - self.imu_time.latest
        else:
            dt = 0.0
        if dt > (1.0 / 150.0) or self.imu_time.size == 0:
            self.imu_time.add(new_nanosec_data)
            if self.imu_time.size > 0 and np.all(self.imu_time.get_all() >= 1.0):
                self.imu_time.apply_to_all(lambda x: x - 1.0)

            if self._imu_pass:
                self._imu_pass = False
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_time(self.imu_time.oldest)
            else:
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_timestep(dt)

            self.rol_velo.update(msg.angular_velocity.x)
            self.pit_velo.update(msg.angular_velocity.y)
            self.yaw_velo.update(msg.angular_velocity.z)
    # def telem_callback(self, msg: Telem) -> None:
        # self.aoa.update_data(msg.alpha)
        # self.ssa.update_data(msg.beta)


    def imu_differentiated_callback(self, msg: Imu) -> None:
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if self.acc_time.size > 0 and new_nanosec_data < self.acc_time.latest:
            new_nanosec_data += 1.0
        if self.acc_time.size > 0:
            dt = new_nanosec_data - self.acc_time.latest
        else:
            dt = 0.0
        if dt > (1.0 / 150.0) or self.acc_time.size == 0:
            self.acc_time.add(new_nanosec_data)
            if self.acc_time.size > 0 and np.all(self.acc_time.get_all() >= 1.0):
                self.acc_time.apply_to_all(lambda x: x - 1.0)

            if self._acc_pass:
                self._acc_pass = False
                for accel in [self.rol_accel, self.pit_accel, self.yaw_accel]:
                    accel.spectrum.update_cp_time(self.acc_time.oldest)
            else:
                for accel in [self.rol_accel, self.pit_accel, self.yaw_accel]:
                    accel.spectrum.update_cp_timestep(dt)

            self.rol_accel.update(msg.angular_velocity.x)
            self.pit_accel.update(msg.angular_velocity.y)
            self.yaw_accel.update(msg.angular_velocity.z)

    def rcout_callback(self, msg: RCOut) -> None:
        new_nanosec_data: float = msg.header.stamp.nanosec * 1E-9
        if self.rco_time.size > 0 and new_nanosec_data < self.rco_time.latest:
            new_nanosec_data += 1.0
        if self.rco_time.size > 0:
            dt = new_nanosec_data - self.rco_time.latest
        else:
            dt = 0.0
        if dt > (1.0 / 150.0) or self.rco_time.size == 0:
            self.rco_time.add(new_nanosec_data)
            if self.rco_time.size > 0 and np.all(self.rco_time.get_all() >= 1.0):
                self.rco_time.apply_to_all(lambda x: x - 1.0)

            if self._rco_pass:
                self._rco_pass = False
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_time(self.rco_time.oldest)
            else:
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_timestep(dt)

            self.ail_pwm.update(msg.channels[0] - 1500)
            self.elv_pwm.update(msg.channels[1] - 1500)
            self.rud_pwm.update(msg.channels[2] - 1500)
    def replay_rcout_callback(self, msg: Float64MultiArray) -> None:
        seconds = int(msg.data[0])
        nanoseconds = int(round((msg.data[0] - seconds) * 1_000_000_000))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds = 0

        new_nanosec_data: float = nanoseconds * 1E-9
        if self.rco_time.size > 0 and new_nanosec_data < self.rco_time.latest:
            new_nanosec_data += 1.0
        if self.rco_time.size > 0:
            dt = new_nanosec_data - self.rco_time.latest
        else:
            dt = 0.0
        if dt > (1.0 / 150.0) or self.rco_time.size == 0:
            self.rco_time.add(new_nanosec_data)
            if self.rco_time.size > 0 and np.all(self.rco_time.get_all() >= 1.0):
                self.rco_time.apply_to_all(lambda x: x - 1.0)

            if self._rco_pass:
                self._rco_pass = False
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_time(self.rco_time.oldest)
            else:
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_timestep(dt)

            self.ail_pwm.update(msg.data[2] - 1500)
            self.elv_pwm.update(msg.data[3] - 1500)
            self.rud_pwm.update(msg.data[5] - 1500)

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
        publisher_periods = {
            "ols_rol": 1 / 25,
            # "ols_rol_nondim": 0.02,
            # "ols_rol_nondim_inertias": 0.02,
            # "ols_rol_ssa": 0.02,
            # "ols_rol_ssa_nondim": 0.02,
            # "ols_rol_ssa_nondim_inertias": 0.02,

            "ols_rol_large": 1 / 25,
            # "ols_rol_large_nondim": 0.02,
            # "ols_rol_large_nondim_inertias": 0.02,
            # "ols_rol_large_ssa": 0.02,
            # "ols_rol_large_ssa_nondim": 0.02,
            # "ols_rol_large_ssa_nondim_inertias": 0.02,

            "ols_pit": 1 / 25,
            # "ols_pit_nondim": 0.02,
            # "ols_pit_nondim_inertias": 0.02,
            # "ols_pit_aoa": 0.02,
            # "ols_pit_aoa_nondim": 0.02,
            # "ols_pit_aoa_nondim_inertias": 0.02,

            "ols_yaw": 1 / 25,
            # "ols_yaw_nondim": 0.02,
            # "ols_yaw_nondim_inertias": 0.02,
            # "ols_yaw_ssa": 0.02,
            # "ols_yaw_ssa_nondim": 0.02,
            # "ols_yaw_ssa_nondim_inertias": 0.02,

            "ols_yaw_large": 1 / 25,
            # "ols_yaw_large_nondim": 0.02,
            # "ols_yaw_large_nondim_inertias": 0.02,
            # "ols_yaw_large_ssa": 0.02,
            # "ols_yaw_large_ssa_nondim": 0.02,
            # "ols_yaw_large_ssa_nondim_inertias": 0.02,
        }

        self.model_publishers: dict[str, Publisher] = {}
        self.model_timers: dict[str, Timer] = {}

        for name, period in publisher_periods.items():
            callback_name = f"publish_{name}_data"
            if not hasattr(self, callback_name):
                self.get_logger().warn(f"Missing callback: {callback_name}")
                continue
            callback = getattr(self, callback_name)

            self.model_publishers[name] = self.create_publisher(
                Float64MultiArray,
                name,
                10,
            )
            self.model_timers[name] = self.create_timer(
                period,
                callback,
            )
    def _publish_ols(
            self,
            publisher_name: str,
            Z,
            Xs: list,
            ) -> None:
        parameters = ordinary_least_squares(Z.spectrum.current_spectrum,
                            np.column_stack([X.spectrum.current_spectrum for X in Xs]))
        msg = Float64MultiArray()
        msg.data = ([Z.timedata.oldest]
                    + [X.timedata.oldest for X in Xs]
                    + parameters[:len(Xs)].tolist())
        self.model_publishers[publisher_name].publish(msg)

    def publish_ols_rol_data(self) -> None:
        self._publish_ols("ols_rol",
                          self.rol_accel,
                          [self.rol_velo, self.ail_pwm])
        
    def publish_ols_rol_large_data(self) -> None:
        self._publish_ols("ols_rol_large",
                          self.rol_accel,
                          [self.rol_velo, self.ail_pwm, self.yaw_velo, self.rud_pwm])
        
    def publish_ols_pit_data(self) -> None:
        self._publish_ols("ols_pit",
                          self.pit_accel,
                          [self.pit_velo, self.elv_pwm])
        
    def publish_ols_yaw_data(self) -> None:
        self._publish_ols("ols_yaw",
                          self.yaw_accel,
                          [self.yaw_velo, self.rud_pwm])
        
    def publish_ols_yaw_large_data(self) -> None:
        self._publish_ols("ols_yaw_large",
                          self.yaw_accel,
                          [self.yaw_velo, self.rud_pwm, self.rol_velo, self.ail_pwm])



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