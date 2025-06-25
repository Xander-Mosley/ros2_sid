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

from ros2_sid.inputdesign import FrequencySweep, MultiStep

class PubExample(Node):
    def __init__(self, ns=''):
        super().__init__('DroneSignal')

        self.some_publisher: Publisher = self.create_publisher(
            String, 'adele', 10)
        self.timer_period: float = 0.5
        self.timer = self.create_timer(
            self.timer_period, self.publish_message)
        
        time, signal = FrequencySweep(1, 1, 5, 0.1, 10)
        print(signal)

    def publish_message(self) -> None:
        msg = String()
        msg.data = "Hello, it's me!"
        self.some_publisher.publish(msg)

class PubInputSignals(Node):
    def __init__(self, ns=''):
        super().__init__('excitation_node')
        # self.swtich: int = 1
        # self.mode: int = 0

        self.trajectory: Publisher = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        
        amplitude: float = 1
        time_step: float = 0.1
        _, self.sweep = FrequencySweep(amplitude, 1, 5, time_step, 15)
        self.step_count = 0
        
        self.timer_period: float = 0.1
        self.timer = self.create_timer(
            self.timer_period, self.PublishSignal)

    def PublishSignal(self) -> None:
        trajectory: CtlTraj = CtlTraj()

        trajectory.roll = [self.sweep[self.step_count]]
        self.step_count += 1
        trajectory.pitch = [0.]
        trajectory.yaw = [0.]
        trajectory.idx = 0

        self.trajectory.publish(trajectory)
        # Publish control trajectory
    
    # def updateTimer(self, new_timer: float) -> None:
    #     """
    #     Updates the time based on a switch.
    #     Args:
    #         timer_period: float the new timer

    #     Returns:
    #         None
    #     """
    #     self.timer_period: float = new_timer
    #     self.timer = self.create_timer(
    #         self.timer_period, self.execute)
        
    # def execute(self) -> None:

    #     """
    #     Check the on swtich.
    #     If off
    #         continue
    #     elif on
    #         Check input method type.
    #         If input type is roll
    #             Check update time period
    #             send freq sweep
    #         elif input type is pit
    #             ""
    #             ""
    #         else input type is yaw
    #             ""
    #             ""
            
    #     """

    # def MessageValue(self, rol, pit, yaw) -> float:
    #     if (self.switch == 1):
    #         if (self.mode == 0):
    #             message_value: float = rol.signal(rol.step_count)
    #             rol.step_count += 1
    #         elif (self.mode == 1):
    #             message_value: float = pit.signal(pit.step_count)
    #             pit.step_count += 1
    #         elif (self.mode ==2):
    #             message_value: float = yaw.signal(yaw.step_count)
    #             yaw.step_count += 1
    #         else:
    #             message_value = -1
    #     else:
    #         rol.step_count: int = 0
    #         pit.step_count: int = 0
    #         yaw.step_count: int = 0
    #         message_value: float = 0
        
    #     return message_value


def main(args=None):
    rclpy.init(args=args)
    pub_example = PubInputSignals()

    while rclpy.ok():
        try:
            rclpy.spin_once(pub_example, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    pub_example.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
