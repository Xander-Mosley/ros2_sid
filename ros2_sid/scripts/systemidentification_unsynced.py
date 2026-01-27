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

        self._imu_first_pass = True
        self._accel_first_pass = True
        self._rcout_first_pass = True
        self._telem_first_pass = True

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
        
        # self.dyn_pres = StoredData(1, 1)
        # self.dyn_pres.update_data(1)
        # self.stat_pres = StoredData(1, 1)
        # self.stat_pres.update_data(1)
        # self.temp = StoredData(1, 1)
        # self.temp.update_data(1)
        # self.airspeed = StoredData(1, 1)
        # self.airspeed.update_data(1)

        self.mass = 25              # [kg]
        self.wing_span = 3.868      # [m]
        self.wing_area = 1.065634   # [m²]
        self.wing_chord = 0.2755    # [m]


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
            else:
                for velo in [self.rol_velo, self.pit_velo, self.yaw_velo]:
                    velo.spectrum.update_cp_timestep(dt)

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


    def setup_pubs(self) -> None:
        default_pub_rate = 1 / 25
        publisher_periods = {
            "ols_rol": default_pub_rate,
            # "ols_rol_nondim": default_pub_rate,
            # "ols_rol_nondim_inertias": default_pub_rate,
            "ols_rol_ssa": default_pub_rate,
            # "ols_rol_ssa_nondim": default_pub_rate,
            # "ols_rol_ssa_nondim_inertias": default_pub_rate,

            "ols_rol_large": default_pub_rate,
            # "ols_rol_large_nondim": default_pub_rate,
            # "ols_rol_large_nondim_inertias": default_pub_rate,
            "ols_rol_large_ssa": default_pub_rate,
            # "ols_rol_large_ssa_nondim": default_pub_rate,
            # "ols_rol_large_ssa_nondim_inertias": default_pub_rate,

            "ols_pit": default_pub_rate,
            # "ols_pit_nondim": default_pub_rate,
            # "ols_pit_nondim_inertias": default_pub_rate,
            "ols_pit_aoa": default_pub_rate,
            # "ols_pit_aoa_nondim": default_pub_rate,
            # "ols_pit_aoa_nondim_inertias": default_pub_rate,

            "ols_yaw": default_pub_rate,
            # "ols_yaw_nondim": default_pub_rate,
            # "ols_yaw_nondim_inertias": default_pub_rate,
            "ols_yaw_ssa": default_pub_rate,
            # "ols_yaw_ssa_nondim": default_pub_rate,
            # "ols_yaw_ssa_nondim_inertias": default_pub_rate,

            "ols_yaw_large": default_pub_rate,
            # "ols_yaw_large_nondim": default_pub_rate,
            # "ols_yaw_large_nondim_inertias": default_pub_rate,
            "ols_yaw_large_ssa": default_pub_rate,
            # "ols_yaw_large_ssa_nondim": default_pub_rate,
            # "ols_yaw_large_ssa_nondim_inertias": default_pub_rate,
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