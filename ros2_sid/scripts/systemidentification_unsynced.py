#!/usr/bin/env python3

import math
import threading
from re import S

import numpy as np
import mavros
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


class OLSNode(Node):
    def __init__(self, ns=''):
        super().__init__('ols_node')
        self.setup_vars()
        self.setup_subs()
        self.setup_pubs()

    def setup_vars(self) -> None:
        self.minimum_dt = 1.0 / 100.0

        self.imu_prev_nanosec = 0.0
        self.accel_prev_nanosec = 0.0
        self.rcout_prev_nanosec = 0.0
        self.telem_prev_nanosec = 0.0
        self.odom_prev_nanosec = 0.0
        self.diff_pres_prev_nanosec = 0.0
        self.stat_pres_prev_nanosec = 0.0
        self.temp_baro_prev_nanosec = 0.0

        self._imu_first_pass = True
        self._accel_first_pass = True
        self._rcout_first_pass = True
        self._telem_first_pass = True
        self._odom_first_pass = True
        self._diff_pres_first_pass = True
        self._stat_pres_first_pass = True
        self._temp_baro_first_pass = True

        self.odom_avg = 0.0
        self.diff_pres_avg = 0.0

        self.rol_velo = RegressorData(eff=0.999)
        self.pit_velo = RegressorData(eff=0.999)
        self.yaw_velo = RegressorData(eff=0.999)
        
        self.rol_accel = RegressorData(eff=0.999)
        self.pit_accel = RegressorData(eff=0.999)
        self.yaw_accel = RegressorData(eff=0.999)

        self.ail_pwm = RegressorData(eff=0.999)
        self.elv_pwm = RegressorData(eff=0.999)
        self.rud_pwm = RegressorData(eff=0.999)

        self.aoa = RegressorData(eff=0.999)
        self.ssa = RegressorData(eff=0.999)

        self.airspeed = RegressorData(eff=0.999)

        self.dyn_pres = RegressorData(eff=0.999)

        self.stat_pres = RegressorData(eff=0.999)

        self.temp_baro = RegressorData(eff=0.999)

        self.mass = 25              # [kg]
        self.wing_span = 3.868      # [m]
        self.wing_area = 1.065634   # [m²]
        self.wing_chord = 0.2755    # [m]
        
        groups = {
            "rol": ["rol_accel", "rol_velo", "ail_pwm"],
            "rol_nondim": ["rol_accel", "rol_velo", "ail_pwm"],
            # "rol_ssa": ["rol_accel", "rol_velo", "ail_pwm", "ssa"],
            "rol_large": ["rol_accel", "rol_velo", "ail_pwm", "yaw_velo", "rud_pwm"],
            # "rol_large_ssa": ["rol_accel", "rol_velo", "ail_pwm", "ssa", "yaw_velo", "rud_pwm"],

            "pit": ["pit_accel", "pit_velo", "elv_pwm"],
            # "pit_aoa": ["pit_accel", "pit_velo", "elv_pwm", "aoa"],

            "yaw": ["yaw_accel", "yaw_velo", "rud_pwm"],
            # "yaw_ssa": ["yaw_accel", "yaw_velo", "rud_pwm", "ssa"],
            "yaw_large": ["yaw_accel", "yaw_velo", "rud_pwm", "rol_velo", "ail_pwm"],
            # "yaw_large_ssa": ["yaw_accel", "yaw_velo", "rud_pwm", "ssa", "rol_velo", "ail_pwm"],
        }

        self.ols = {}
        for group, signals in groups.items():
            self.ols[group] = {
                signal: RecursiveFourierTransform(eff=0.999)
                for signal in signals
            }


    def setup_subs(self) -> None:
        self.imu_filt_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_filt',
            self.imu_callback,
            qos_profile=SENSOR_QOS
        )

        self.imu_diff_sub: Subscription = self.create_subscription(
            Imu,
            '/imu_diff',
            self.imu_diff_callback,
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

        self.telem_sub: Subscription = self.create_subscription(
            Telem,
            '/telem',
            self.telem_callback,
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

        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.imu_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.imu_prev_nanosec = new_nanosec

            if self._imu_first_pass:
                self._imu_first_pass = False
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_time(self.imu_prev_nanosec)
                for group, filters in self.ols.items():
                    for signal, rft in filters.items():
                        rft.update_cp_time(self.imu_prev_nanosec)
            else:
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_timestep(dt)
                for group, filters in self.ols.items():
                    for signal, rft in filters.items():
                        rft.update_cp_timestep(self.imu_prev_nanosec)

            self.rol_velo.update(msg.angular_velocity.x)
            self.pit_velo.update(msg.angular_velocity.y)
            self.yaw_velo.update(msg.angular_velocity.z)

        else:
            print(f"IMU update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def imu_diff_callback(self, msg: Imu) -> None:
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.accel_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.accel_prev_nanosec = new_nanosec

            if self._accel_first_pass:
                self._accel_first_pass = False
                for accel in [self.rol_accel, self.pit_accel, self.yaw_accel]:
                    accel.spectrum.update_cp_time(self.accel_prev_nanosec)
            else:
                for accel in [self.rol_accel, self.pit_accel, self.yaw_accel]:
                    accel.spectrum.update_cp_timestep(dt)

            self.rol_accel.update(msg.angular_velocity.x)
            self.pit_accel.update(msg.angular_velocity.y)
            self.yaw_accel.update(msg.angular_velocity.z)

        else:
            print(f"ACCEL update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def rcout_callback(self, msg: RCOut) -> None:
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.rcout_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.rcout_prev_nanosec = new_nanosec

            if self._rcout_first_pass:
                self._rcout_first_pass = False
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_time(self.rcout_prev_nanosec)
            else:
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_timestep(dt)

            self.ail_pwm.update(msg.channels[0] - 1500)
            self.elv_pwm.update(msg.channels[1] - 1500)
            self.rud_pwm.update(msg.channels[3] - 1500)

        else:
            print(f"RCOut update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def replay_rcout_callback(self, msg: Float64MultiArray) -> None:
        new_sec, new_nanosec = divmod(msg.data[0], 1.0)
        dt = (new_nanosec - self.rcout_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.rcout_prev_nanosec = new_nanosec

            if self._rcout_first_pass:
                self._rcout_first_pass = False
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_time(self.rcout_prev_nanosec)
            else:
                for pwm in [self.ail_pwm, self.elv_pwm, self.rud_pwm]:
                    pwm.spectrum.update_cp_timestep(dt)

            self.ail_pwm.update(msg.data[2] - 1500)
            self.elv_pwm.update(msg.data[3] - 1500)
            self.rud_pwm.update(msg.data[5] - 1500)

        else:
            print(f"RCOut update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def telem_callback(self, msg: Telem) -> None:
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.telem_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.telem_prev_nanosec = new_nanosec

            if self._telem_first_pass:
                self._telem_first_pass = False
                for xxx in [self.aoa, self.ssa]:
                    xxx.spectrum.update_cp_time(self.telem_prev_nanosec)
            else:
                for xxx in [self.aoa, self.ssa]:
                    xxx.spectrum.update_cp_timestep(dt)

            self.aoa.update(msg.alpha)
            self.ssa.update(msg.beta)

        else:
            print(f"Telem update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def odom_callback(self, msg: Odometry) -> None:
        """
        Converts NED to ENU and publishes the trajectory
        https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
        Twist Will show velocity in linear and rotational 
        """
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.odom_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.odom_prev_nanosec = new_nanosec

            if self._odom_first_pass:
                self._odom_first_pass = False
                self.airspeed.spectrum.update_cp_time(self.odom_prev_nanosec)
            else:
                self.airspeed.spectrum.update_cp_timestep(dt)

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            airspeed = np.sqrt(vx**2 + vy**2 + vz**2)
            self.airspeed.update(airspeed - self.odom_avg)

            sampling_period = 1 / 2
            time_constant = 8
            alpha = 1 - np.exp(sampling_period/time_constant)
            self.odom_avg = self.odom_avg + alpha * (airspeed - self.odom_avg)

        else:
            print(f"Odom update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def diff_pressure_callback(self, msg: FluidPressure) -> None:
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.diff_pres_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.diff_pres_prev_nanosec = new_nanosec

            if self._diff_pres_first_pass:
                self._diff_pres_first_pass = False
                self.dyn_pres.spectrum.update_cp_time(self.diff_pres_prev_nanosec)
            else:
                self.dyn_pres.spectrum.update_cp_timestep(dt)

            self.dyn_pres.update(msg.fluid_pressure - self.diff_pres_avg)   # [Pa]

            sampling_period = 1 / 2
            time_constant = 8
            alpha = 1 - np.exp(sampling_period/time_constant)
            self.diff_pres_avg = self.diff_pres_avg + alpha * (msg.fluid_pressure - self.diff_pres_avg)

        else:
            print(f"Diff Pressure update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def static_pressure_callback(self, msg: FluidPressure) -> None:
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.stat_pres_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.stat_pres_prev_nanosec = new_nanosec

            if self._stat_pres_first_pass:
                self._stat_pres_first_pass = False
                self.stat_pres.spectrum.update_cp_time(self.stat_pres_prev_nanosec)
            else:
                self.stat_pres.spectrum.update_cp_timestep(dt)

            self.stat_pres.update(msg.fluid_pressure)  # [Pa]

        else:
            print(f"Static Pressure update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")

    def temperature_baro_callback(self, msg: Temperature) -> None:
        new_sec: float = msg.header.stamp.sec
        new_nanosec: float = msg.header.stamp.nanosec * 1E-9
        dt = (new_nanosec - self.temp_baro_prev_nanosec) % 1.0
        if dt >= self.minimum_dt:
            self.temp_baro_prev_nanosec = new_nanosec

            if self._temp_baro_first_pass:
                self._temp_baro_first_pass = False
                self.temp_baro.spectrum.update_cp_time(self.temp_baro_prev_nanosec)
            else:
                self.temp_baro.spectrum.update_cp_timestep(dt)

            # self.temp_baro.update(msg.temperature)              # [°C]
            # self.temp_baro.update(msg.temperature + 273.15)     # [°K]
            self.temp_baro.update(msg.temperature - 25.0)       # [°C - ambient tempterature]

        else:
            print(f"Temperature Baro update skipped (dt={dt:.6f} < {self.minimum_dt:.6f}s) at {new_sec + new_nanosec}s.")


    def setup_pubs(self) -> None:
        default_pub_rate = 1 / 25
        publisher_periods = {
            "ols_rol": default_pub_rate,
            "ols_rol_nondim": default_pub_rate,
            # "ols_rol_nondim_inertias": default_pub_rate,
            # "ols_rol_ssa": default_pub_rate,
            # "ols_rol_ssa_nondim": default_pub_rate,
            # "ols_rol_ssa_nondim_inertias": default_pub_rate,

            "ols_rol_large": default_pub_rate,
            # "ols_rol_large_nondim": default_pub_rate,
            # "ols_rol_large_nondim_inertias": default_pub_rate,
            # "ols_rol_large_ssa": default_pub_rate,
            # "ols_rol_large_ssa_nondim": default_pub_rate,
            # "ols_rol_large_ssa_nondim_inertias": default_pub_rate,

            "ols_pit": default_pub_rate,
            # "ols_pit_nondim": default_pub_rate,
            # "ols_pit_nondim_inertias": default_pub_rate,
            # "ols_pit_aoa": default_pub_rate,
            # "ols_pit_aoa_nondim": default_pub_rate,
            # "ols_pit_aoa_nondim_inertias": default_pub_rate,

            "ols_yaw": default_pub_rate,
            # "ols_yaw_nondim": default_pub_rate,
            # "ols_yaw_nondim_inertias": default_pub_rate,
            # "ols_yaw_ssa": default_pub_rate,
            # "ols_yaw_ssa_nondim": default_pub_rate,
            # "ols_yaw_ssa_nondim_inertias": default_pub_rate,

            "ols_yaw_large": default_pub_rate,
            # "ols_yaw_large_nondim": default_pub_rate,
            # "ols_yaw_large_nondim_inertias": default_pub_rate,
            # "ols_yaw_large_ssa": default_pub_rate,
            # "ols_yaw_large_ssa_nondim": default_pub_rate,
            # "ols_yaw_large_ssa_nondim_inertias": default_pub_rate,


            "ols_rol_old": default_pub_rate,
            "ols_rol_nondim_old": default_pub_rate,
            "ols_rol_large_old": default_pub_rate,
            "ols_pit_old": default_pub_rate,
            "ols_yaw_old": default_pub_rate,
            "ols_yaw_large_old": default_pub_rate,
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

    def publish_ols_rol_nondim_data(self) -> None:
        airspeed_inv = np.conj(self.airspeed.spectrum.current_spectrum) / (np.abs(self.airspeed.spectrum.current_spectrum)**2 + 1e-6)
        Z = self.rol_accel.spectrum.current_spectrum
        X1 = np.convolve(self.rol_velo.spectrum.current_spectrum, self.dyn_pres.spectrum.current_spectrum, mode='same') * airspeed_inv
        X2 = np.convolve(self.ail_pwm.spectrum.current_spectrum, self.dyn_pres.spectrum.current_spectrum, mode='same')

        # print(self.dyn_pres.spectrum.current_spectrum)

        parameters = ordinary_least_squares(Z,
                           np.column_stack([X1, X2]))
        msg = Float64MultiArray()
        msg.data = [
            self.rol_accel.timedata.oldest,
            self.dyn_pres.timedata.oldest * self.rol_velo.timedata.oldest / self.airspeed.timedata.oldest,
            self.dyn_pres.timedata.oldest * self.ail_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
            ]
        self.model_publishers["ols_rol_nondim"].publish(msg)

    def publish_ols_rol_ssa_data(self) -> None:
        self._publish_ols("ols_rol_ssa",
                          self.rol_accel,
                          [self.rol_velo, self.ail_pwm, self.ssa])

    def publish_ols_rol_large_data(self) -> None:
        self._publish_ols("ols_rol_large",
                          self.rol_accel,
                          [self.rol_velo, self.ail_pwm, self.yaw_velo, self.rud_pwm])

    def publish_ols_rol_large_ssa_data(self) -> None:
        self._publish_ols("ols_rol_large_ssa",
                          self.rol_accel,
                          [self.rol_velo, self.ail_pwm, self.ssa, self.yaw_velo, self.rud_pwm])

    def publish_ols_pit_data(self) -> None:
        self._publish_ols("ols_pit",
                          self.pit_accel,
                          [self.pit_velo, self.elv_pwm])

    def publish_ols_pit_aoa_data(self) -> None:
        self._publish_ols("ols_pit_aoa",
                          self.pit_accel,
                          [self.pit_velo, self.elv_pwm, self.aoa])

    def publish_ols_yaw_data(self) -> None:
        self._publish_ols("ols_yaw",
                          self.yaw_accel,
                          [self.yaw_velo, self.rud_pwm])

    def publish_ols_yaw_ssa_data(self) -> None:
        self._publish_ols("ols_yaw_ssa",
                          self.yaw_accel,
                          [self.yaw_velo, self.rud_pwm, self.ssa])

    def publish_ols_yaw_large_data(self) -> None:
        self._publish_ols("ols_yaw_large",
                          self.yaw_accel,
                          [self.yaw_velo, self.rud_pwm, self.rol_velo, self.ail_pwm])

    def publish_ols_yaw_large_ssa_data(self) -> None:
        self._publish_ols("ols_yaw_large_ssa",
                          self.yaw_accel,
                          [self.yaw_velo, self.rud_pwm, self.ssa, self.rol_velo, self.ail_pwm])


    def publish_ols_rol_old_data(self) -> None:
        self.ols["rol"]["rol_accel"].update_spectrum(self.rol_accel.timedata.oldest)
        self.ols["rol"]["rol_velo"].update_spectrum(self.rol_velo.timedata.oldest)
        self.ols["rol"]["ail_pwm"].update_spectrum(self.ail_pwm.timedata.oldest)

        parameters = ordinary_least_squares(self.ols["rol"]["rol_accel"].current_spectrum,
                           np.column_stack([self.ols["rol"]["rol_velo"].current_spectrum,
                                            self.ols["rol"]["ail_pwm"].current_spectrum]))
        msg = Float64MultiArray()
        msg.data = (
            self.rol_accel.timedata.oldest,
            self.rol_velo.timedata.oldest,
            self.ail_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
        )
        self.model_publishers["ols_rol_old"].publish(msg)
        
    def publish_ols_rol_nondim_old_data(self) -> None:
        if self.dyn_pres.timedata.oldest == 0.0 or self.airspeed.timedata.oldest == 0.0:
            return
        self.ols["rol_nondim"]["rol_accel"].update_spectrum(self.rol_accel.timedata.oldest)
        self.ols["rol_nondim"]["rol_velo"].update_spectrum(self.dyn_pres.timedata.oldest * self.rol_velo.timedata.oldest / self.airspeed.timedata.oldest)
        self.ols["rol_nondim"]["ail_pwm"].update_spectrum(self.dyn_pres.timedata.oldest * self.ail_pwm.timedata.oldest)

        parameters = ordinary_least_squares(self.ols["rol_nondim"]["rol_accel"].current_spectrum,
                           np.column_stack([self.ols["rol_nondim"]["rol_velo"].current_spectrum,
                                            self.ols["rol_nondim"]["ail_pwm"].current_spectrum]))
        msg = Float64MultiArray()
        msg.data = (
            self.rol_accel.timedata.oldest,
            self.dyn_pres.timedata.oldest * self.rol_velo.timedata.oldest / self.airspeed.timedata.oldest,
            self.dyn_pres.timedata.oldest * self.ail_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
        )
        self.model_publishers["ols_rol_nondim_old"].publish(msg)

    def publish_ols_rol_large_old_data(self) -> None:
        self.ols["rol_large"]["rol_accel"].update_spectrum(self.rol_accel.timedata.oldest)
        self.ols["rol_large"]["rol_velo"].update_spectrum(self.rol_velo.timedata.oldest)
        self.ols["rol_large"]["ail_pwm"].update_spectrum(self.ail_pwm.timedata.oldest)
        self.ols["rol_large"]["yaw_velo"].update_spectrum(self.yaw_velo.timedata.oldest)
        self.ols["rol_large"]["rud_pwm"].update_spectrum(self.rud_pwm.timedata.oldest)

        parameters = ordinary_least_squares(self.ols["rol_large"]["rol_accel"].current_spectrum,
                           np.column_stack([self.ols["rol_large"]["rol_velo"].current_spectrum,
                                            self.ols["rol_large"]["ail_pwm"].current_spectrum,
                                            self.ols["rol_large"]["yaw_velo"].current_spectrum,
                                            self.ols["rol_large"]["rud_pwm"].current_spectrum]))
        msg = Float64MultiArray()
        msg.data = (
            self.rol_accel.timedata.oldest,
            self.rol_velo.timedata.oldest,
            self.ail_pwm.timedata.oldest,
            self.yaw_velo.timedata.oldest,
            self.rud_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
        )
        self.model_publishers["ols_rol_large_old"].publish(msg)

    def publish_ols_pit_old_data(self) -> None:
        self.ols["pit"]["pit_accel"].update_spectrum(self.pit_accel.timedata.oldest)
        self.ols["pit"]["pit_velo"].update_spectrum(self.pit_velo.timedata.oldest)
        self.ols["pit"]["elv_pwm"].update_spectrum(self.elv_pwm.timedata.oldest)

        parameters = ordinary_least_squares(self.ols["pit"]["pit_accel"].current_spectrum,
                           np.column_stack([self.ols["pit"]["pit_velo"].current_spectrum,
                                            self.ols["pit"]["elv_pwm"].current_spectrum]))
        msg = Float64MultiArray()
        msg.data = (
            self.pit_accel.timedata.oldest,
            self.pit_velo.timedata.oldest,
            self.elv_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
        )
        self.model_publishers["ols_pit_old"].publish(msg)

    def publish_ols_yaw_old_data(self) -> None:
        self.ols["yaw"]["yaw_accel"].update_spectrum(self.yaw_accel.timedata.oldest)
        self.ols["yaw"]["yaw_velo"].update_spectrum(self.yaw_velo.timedata.oldest)
        self.ols["yaw"]["rud_pwm"].update_spectrum(self.rud_pwm.timedata.oldest)

        parameters = ordinary_least_squares(self.ols["yaw"]["yaw_accel"].current_spectrum,
                           np.column_stack([self.ols["yaw"]["yaw_velo"].current_spectrum,
                                            self.ols["yaw"]["rud_pwm"].current_spectrum]))
        msg = Float64MultiArray()
        msg.data = (
            self.yaw_accel.timedata.oldest,
            self.yaw_velo.timedata.oldest,
            self.rud_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
        )
        self.model_publishers["ols_yaw_old"].publish(msg)
        
    def publish_ols_yaw_large_old_data(self) -> None:
        self.ols["yaw_large"]["yaw_accel"].update_spectrum(self.yaw_accel.timedata.oldest)
        self.ols["yaw_large"]["yaw_velo"].update_spectrum(self.yaw_velo.timedata.oldest)
        self.ols["yaw_large"]["ail_pwm"].update_spectrum(self.ail_pwm.timedata.oldest)
        self.ols["yaw_large"]["rol_velo"].update_spectrum(self.rol_velo.timedata.oldest)
        self.ols["yaw_large"]["ail_pwm"].update_spectrum(self.ail_pwm.timedata.oldest)

        parameters = ordinary_least_squares(self.ols["yaw_large"]["yaw_accel"].current_spectrum,
                           np.column_stack([self.ols["yaw_large"]["yaw_velo"].current_spectrum,
                                            self.ols["yaw_large"]["ail_pwm"].current_spectrum,
                                            self.ols["yaw_large"]["rol_velo"].current_spectrum,
                                            self.ols["yaw_large"]["ail_pwm"].current_spectrum]))
        msg = Float64MultiArray()
        msg.data = (
            self.yaw_accel.timedata.oldest,
            self.yaw_velo.timedata.oldest,
            self.ail_pwm.timedata.oldest,
            self.rol_velo.timedata.oldest,
            self.ail_pwm.timedata.oldest,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
        )
        self.model_publishers["ols_yaw_large_old"].publish(msg)


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