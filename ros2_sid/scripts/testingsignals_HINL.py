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
from rclpy.subscription import Subscription
from mavros_msgs.msg import RCIn
from mavros.base import SENSOR_QOS

from ros2_sid.inputdesign import frequency_sweep, multi_step

class PubInputSignals(Node):
    def __init__(self, ns=''):
        super().__init__('excitation_node')
        self.setup_subscriptions()
        self.kill_switch: float = 0.
        self.kill_switch_channel: int = 5
        self.kill_switch_threshold: float = 1550

        self.run_switch: int = 0
        self.maneuver_mode: int = 1
        self.maneuvers()
        self.initial_counter: int = 0
        self.initial_timer_period: float = 0.02

        self.counter: int = self.initial_counter
        self.input_signal: Publisher = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        self.current_timer_period: float = self.initial_timer_period
        self.timer = self.create_timer(
            self.current_timer_period, self.logic_loop)
        
    def setup_subscriptions(self) -> None:
        self.rcin_sub: Subscription = self.create_subscription(
            RCIn,
            '/mavros/rc/in',
            self.rcin_callback,
            qos_profile=SENSOR_QOS
        )

    def rcin_callback(self, msg: RCIn) -> None:
        self.kill_switch = msg.channels[self.kill_switch_channel]


    def maneuvers(self) -> None:
        # maneuvers must have the shape (N, 4)
        # where the columns (in order) are:
        # time, roll signal, pitch signal, yaw signal;
        # and the first time value must be zero

        amplitude: float = np.deg2rad(45) 
        minimum_frequency: float = 0.1
        maximum_frequency: float = 1.5
        time_step: float = 0.02
        final_time: float = 15.
        time, sweep = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        empty = np.zeros_like(time)
        self.rolsweep = np.array([time, sweep, empty, empty]).T

        amplitude: float = np.deg2rad(45)  
        natural_frequency: float = 1.0
        pulses: list = [1, 1]
        time_delay: float = 1.0
        time_step: float = 0.02
        final_time: float = 15.
        time, doublet = multi_step(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        empty = np.zeros_like(time)
        self.roldoublet = np.array([time, doublet, empty, empty]).T
        
        amplitude: float = np.deg2rad(10)
        minimum_frequency: float = 0.1
        maximum_frequency: float = 1.5
        time_step: float = 0.02
        final_time: float = 15.
        time, sweep = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        empty = np.zeros_like(time)
        self.pitsweep = np.array([time, empty, sweep, empty]).T

        natural_frequency: float = 1.0
        amplitude: float = np.deg2rad(10)
        pulses: list = [1, 1]
        time_delay: float = 1.0
        time_step: float = 0.02
        final_time: float = 15.
        time, doublet = multi_step(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        empty = np.zeros_like(time)
        self.pitdoublet = np.array([time, empty, doublet, empty]).T
        

        amplitude: float = np.deg2rad(30)
        minimum_frequency: float = 0.1
        maximum_frequency: float = 1.5
        time_step: float = 0.02
        final_time: float = 15.
        time, sweep = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        empty = np.zeros_like(time)
        self.yawsweep = np.array([time, empty, empty, sweep]).T
        
        amplitude: float = 1.
        natural_frequency: float = 1.0
        pulses: list = [1, 1]
        time_delay: float = 1.0
        time_step: float = 0.02
        final_time: float = 15.
        time, doublet = multi_step(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        empty = np.zeros_like(time)
        self.yawdoublet = np.array([time, empty, empty, doublet]).T

    def logic_loop(self) -> None:
        if (self.kill_switch >= self.kill_switch_threshold):
            # self.switch could be a variable defined by the control's...
            # function, so the function isn't running at each if statement
            if (self.run_switch == 1):
                # self.maneuver_mode should be a variable defined by the control's...
                # function, so the function isn't running at each if statement
                if (self.counter == 0):
                    if (self.maneuver_mode == 1):
                        self.current_maneuver = self.rolsweep
                    elif (self.maneuver_mode == 2):
                        self.current_maneuver = self.roldoublet
                    elif (self.maneuver_mode == 3):
                        self.current_maneuver = self.pitsweep
                    elif (self.maneuver_mode == 4):
                        self.current_maneuver = self.pitdoublet
                    elif (self.maneuver_mode == 5):
                        self.current_maneuver = self.yawsweep
                    elif (self.maneuver_mode == 6):
                        self.current_maneuver = self.yawdoublet
                    else:
                        self.current_maneuver = None
                    
                    if (self.current_maneuver is not None):
                        maneuver_timer_period: float = self.current_maneuver[1, 0]
                    else:
                        maneuver_timer_period: float = self.initial_timer_period
                    
                    if (maneuver_timer_period != self.current_timer_period):
                        self.update_timer_period(maneuver_timer_period)


                if (self.current_maneuver is not None):
                    if (self.counter < len(self.current_maneuver)):
                        self.publish_trajectory()

                    else:
                        self.run_switch = 0
                        print("MANEUVER COMPLETE")

                else:
                    # could this be done earlier as well???
                    self.run_switch = 0


            elif (self.run_switch == 0):
                self.counter = self.initial_counter


        else:
            self.run_switch = 1

    def update_timer_period(self, new_timer_period) -> None:
        if (self.timer is not None):
            self.timer.cancel()

        self.current_timer_period = new_timer_period
        self.timer = self.create_timer(
            self.current_timer_period, self.logic_loop)
        
    def publish_trajectory(self) -> None:
        trajectory: CtlTraj = CtlTraj()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.roll  = [self.current_maneuver[self.counter, 1], self.current_maneuver[self.counter, 1]]
        trajectory.pitch = [self.current_maneuver[self.counter, 2], self.current_maneuver[self.counter, 2]]
        trajectory.yaw   = [self.current_maneuver[self.counter, 3], self.current_maneuver[self.counter, 3]]
        trajectory.thrust = [0.5, 0.5]
        trajectory.idx = 0
        self.input_signal.publish(trajectory)
        #print(f"Publishing trajectory: {trajectory.roll}, {trajectory.pitch}, {trajectory.yaw}")
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    pub_signals = PubInputSignals()

    while rclpy.ok():
        try:
            rclpy.spin_once(pub_signals, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    pub_signals.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
