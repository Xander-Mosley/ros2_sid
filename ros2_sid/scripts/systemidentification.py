#!/usr/bin/env python3
from re import S
import rclpy
import math
import numpy as np

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from drone_interfaces.msg import Telem, CtlTraj
import threading

from rclpy.subscription import Subscription
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Imu
from mavros_msgs.msg import RCOut
from nav_msgs.msg import Odometry
from mavros.base import SENSOR_QOS
import mavros

from ros2_sid.rt_ols import StoredData, ModelStructure, diff


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
        self.setup_models()
        self.setup_all_subscriptions()
        # self.setup_synced_subscriptions(ns)
        self.setup_all_publishers()


    def setup_models(self) -> None:
        # define class variables
        ModelStructure.class_eff = 0.999

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

        # initialize model structure objects
        self.rol = ModelStructure(2)
        self.pit = ModelStructure(2)
        self.yaw = ModelStructure(2)
    

    def setup_all_subscriptions(self) -> None:
        self.imu_sub: Subscription = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            SENSOR_QOS
        )

        self.rc_sub: Subscription = self.create_subscription(
            RCOut,
            '/mavros/rc/out',
            self.rc_callback,
            qos_profile=SENSOR_QOS
        )

        # self.state_sub: Subscription = self.create_subscription(
        #     mavros.local_position.Odometry,
        #     'mavros/local_position/odom',
        #     self.mavros_state_callback,
        #     qos_profile=SENSOR_QOS
        # )

    def imu_callback(self, msg: Imu) -> None:
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html, body frame
        self.livetime.update_data((msg.header.stamp.nanosec))
        self.rol_velo.update_data(msg.angular_velocity.x)
        self.pit_velo.update_data(msg.angular_velocity.y)
        self.yaw_velo.update_data(msg.angular_velocity.z)

        self.rol_accel.update_data(diff(self.livetime.data, self.rol_velo.data))
        self.pit_accel.update_data(diff(self.livetime.data, self.pit_velo.data))
        self.yaw_accel.update_data(diff(self.livetime.data, self.yaw_velo.data))

        ModelStructure.update_shared_cp_time(self.livetime.data[0])
        
    def rc_callback(self, msg: RCOut) -> None:
        self.ail_pwm.update_data(msg.channels[0])
        self.elv_pwm.update_data(msg.channels[1])
        self.rud_pwm.update_data(msg.channels[2])

    # def mavros_state_callback(self, msg: mavros.local_position.Odometry) -> None:
    #     """
    #     Converts NED to ENU and publishes the trajectory
    #     https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html
    #     Twist Will show velocity in linear and rotational 
    #     """

    #     self.current_state: list[float] = [
    #             0.0,  # x
    #             0.0,  # y
    #             0.0,  # z
    #             0.0,  # phi
    #             0.0,  # theta
    #             0.0,  # psi
    #             0.0,  # airspeed
    #         ]
        
    #     self.current_state[0] = msg.pose.pose.position.x
    #     self.current_state[1] = msg.pose.pose.position.y
    #     self.current_state[2] = msg.pose.pose.position.z

    #     # quaternion attitudes
    #     qx = msg.pose.pose.orientation.x
    #     qy = msg.pose.pose.orientation.y
    #     qz = msg.pose.pose.orientation.z
    #     qw = msg.pose.pose.orientation.w
    #     roll, pitch, yaw = euler_from_quaternion(
    #         qx, qy, qz, qw)

    #     self.current_state[3] = roll
    #     self.current_state[4] = pitch
    #     self.current_state[5] = yaw  # (yaw+ (2*np.pi) ) % (2*np.pi);

    #     vx = msg.twist.twist.linear.x
    #     vy = msg.twist.twist.linear.y
    #     vz = msg.twist.twist.linear.z
    #     # get magnitude of velocity
    #     self.current_state[6] = np.sqrt(vx**2 + vy**2 + vz**2)



    # def setup_synced_subscriptions(self, ns: str) -> None:
    #     self.imu_sub = Subscriber(
    #         self,
    #         Imu,
    #         '/mavros/imu/data',
    #         qos_profile=SENSOR_QOS
    #     )
    #     self.rc_sub = Subscriber(
    #         self,
    #         RCOut,
    #         '/mavros/rc/out',
    #         qos_profile=SENSOR_QOS
    #     )
    #     self.odom_sub = Subscriber(
    #         self,
    #         Odometry,
    #         'mavros/local_position/odom',
    #         qos_profile=SENSOR_QOS
    #     )
    #     self.telem_sub = Subscriber(
    #         self,
    #         Telem,
    #         '/telem_data',
    #         qos_profile=SENSOR_QOS
    #     )
        
    #     self.sync = ApproximateTimeSynchronizer(
    #         [self.imu_sub],
    #         queue_size = 20,
    #         slop = 0.05
    #     )
    #     self.sync.registerCallback(self.synced_callback)

    #     self.get_logger().info("Synchronized IMU, RCOut, Odometry, and Telem subscriptions initialized.")

    #     print("Rol Velo:")
    #     print(self.rol_velo)

    # def synced_callback(
    #         self,
    #         imu_msg: Imu,
    #         rc_msg: RCOut,
    #         odom_msg: Odometry,
    #         telem_msg: Telem
    #         ) -> None:
        
    #     self.rol_velo = imu_msg.angular_velocity.x
    #     self.pit_velo = imu_msg.angular_velocity.y
    #     self.rud_velo = imu_msg.angular_velocity.z

    #     self.ail_pwm = rc_msg.channels[0]
    #     self.elv_pwm = rc_msg.channels[1]
    #     self.rud_pwm = rc_msg.channels[2]

    #     self.position = odom_msg.pose.pose.position
    #     self.orientation = odom_msg.pose.pose.orientation
    #     self.velocity = odom_msg.twist.twist.linear

    #     self.get_logger().info(
    #         f"Gryo: [{self.rol_velo:.2f}, {self.pit_velo:.2f}, {self.rud_velo:.2f}],"
    #         f"RC PWM: [Ail: {self.ail_pwm}, Elv: {self.elv_pwm}, Rud: {self.rud_pwm}],"
    #         f"Velo: [{self.velocity.x:.2f}, {self.velocity.y:.2f}, {self.velocity.z:.2f}]"
    #     )



    def setup_all_publishers(self):
        self.ols_publisher: Publisher = self.create_publisher(
                Float64MultiArray, 'ols_data', 10)
            
        self.timer_period: float = 0.02 # should later automate to match the shared time becuase this determines the rate at which the models update (which should be the same rate as the stored data)
        self.timer = self.create_timer(
            self.timer_period, self.publish_ols_data)
        
    def publish_ols_data(self) -> None:
        self.rol.update_model(self.rol_accel.data[0], [self.rol_velo.data[0], self.ail_pwm.data[0]])
        self.pit.update_model(self.pit_accel.data[0], [self.pit_velo.data[1], self.elv_pwm.data[0]])
        self.yaw.update_model(self.yaw_accel.data[0], [self.yaw_velo.data[0], self.rud_pwm.data[0]])

        ols_data = [
            self.rol.parameters[0],
            self.rol.parameters[1],
            self.pit.parameters[0],
            self.pit.parameters[1],
            self.yaw.parameters[0],
            self.yaw.parameters[1]
            ]
        
        msg: Float64MultiArray = Float64MultiArray()
        msg.data = ols_data
        self.ols_publisher.publish(msg)

    
 

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