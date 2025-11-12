#!/usr/bin/env python3

"""
testingsignals_HINL.py - ROS2 node for publishing drone excitation signals using a kill switch.

This script defines a ROS2 node, 'PubInputSignals', which generates and publishes
predefined or live-generated input maneuvers to a drone's trajectory topic for testing
and system identification. Unlike 'testingsignals_SINL.py', this node maps the run/stop
switch to a physical RC kill switch channel on the flight controller, enabling live
hardware control over signal execution.

Key Features
------------
- Load maneuvers from CSV files to ensure reproducibility.
- Generate and publish different excitation signals on the 'trajectory' topic.
- Run/stop execution is controlled by a physical kill switch input.
- User-selectable maneuver modes when the kill switch is inactive.
- Adjustable publishing timer to match maneuver time steps.
- ROS2 threading allows concurrent handling of user input and signal publication.
- Subscribes to '/mavros/rc/in' to read RC input channels.

Maneuver Format
---------------
- All maneuvers are expected as arrays of shape (N, 4):
  '[time, roll_signal, pitch_signal, yaw_signal]'
- Time values must start at zero.

Usage
-----
1. Launch the node:
    '''bash
    ros2 run ros2_sid testingsignals_HINL.py
    '''
2. Use the RC kill switch to start/stop maneuver execution.
3. When the kill switch is inactive (low), select maneuvers via console input.

Custom Dependencies
-------------------
- Custom messages: 'drone_interfaces/CtlTraj', 'drone_interfaces/Telem'
- Input signal utilities from 'ros2_sid.input_design'

Author
------
Xander D. Mosley
Email: XanderDMosley.Engineer@gmail.com
Date: 11 Jul 2025
"""


import math
import threading
from re import S

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray, String
from drone_interfaces.msg import CtlTraj, Telem
from mavros_msgs.msg import RCIn
from mavros.base import SENSOR_QOS

from ros2_sid.input_design import frequency_sweep, multi_sine, multi_step


__all__ = ['PubInputSignals']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


class PubInputSignals(Node):
    """
    ROS2 node for publishing predefined excitation signals to a drone's trajectory topic.

    This node interfaces with a physical RC kill switch on the flight controller to
    control the start/stop of maneuver execution. It publishes maneuver data as 
    'CtlTraj' messages for use in hardware-in-the-loop (HIL) or flight tests.

    Maneuvers can be loaded from pre-saved CSV files (e.g., multisines, doublets, sweeps)
    or generated dynamically. The user can select which maneuver to execute through
    a console interface when the kill switch is inactive.

    Attributes
    ----------
    kill_switch : float
        Current RC channel value corresponding to the kill switch.
    rc_bias : int
        Index offset for RC channel numbering.
    kill_switch_channel : int
        Index of the kill switch RC channel.
    kill_switch_threshold : float
        PWM threshold distinguishing ON/OFF states.
    run_switch : int
        Internal execution flag (1 = running, 0 = stopped).
    maneuver_mode : int
        Index of the currently selected maneuver (0-9).
    initial_counter : int
        Initial value for the trajectory index counter.
    initial_timer_period : float
        Default timer update period (s).
    counter: int
        Current index in the active maneuver trajectory array.
    input_signal : Publisher
        ROS2 publisher for 'CtlTraj' messages.
    current_timer_period : float
        Current ROS2 timer period (s).
    timer : Timer
        ROS2 timer controlling publishing rate.
    userthread : Thread
        Thread for user console input.

    Author
    ------
    Xander D. Mosley

    History
    -------
    11 Jul 2025 - Created, XDM.
    """
    def __init__(self, ns=''):
        """
        Initialize the excitation node and set up publishers, subscribers, and timers.

        Parameters
        ----------
        ns : str, optional
            Namespace for the ROS2 node. Defaults to an empty string.

        Notes
        -----
        - A separate thread is launched for user input ('self.userthread').
        - Maneuver data is preloaded from CSV files during initialization.
        """
        super().__init__('excitation_node')
        self.setup_subscriptions()
        self.kill_switch: float = 0.0
        self.rc_bias: int = 1   # Channel 1 starts at index 0
        self.kill_switch_channel: int = 9 - self.rc_bias
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
        
        self.userthread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.userthread.start()
        
    def setup_subscriptions(self) -> None:
        """
        Create subscriptions for receiving RC input data.

        Subscribes to '/mavros/rc/in' to read RC channel data and determine
        the state of the physical kill switch.
        """
        self.rcin_sub: Subscription = self.create_subscription(
            RCIn,
            '/mavros/rc/in',
            self.rcin_callback,
            qos_profile=SENSOR_QOS
            )

    def rcin_callback(self, msg: RCIn) -> None:
        """
        Callback for RC input messages.

        Parameters
        ----------
        msg : RCIn
            MAVROS RC input message containing PWM values for all channels.

        Notes
        -----
        - Updates 'self.kill_switch' with the PWM value from the configured channel.
        """
        self.kill_switch = msg.channels[self.kill_switch_channel]

    def maneuvers(self) -> None:
        """
        Load predefined maneuver trajectories from CSV files or generate live
        when the node starts.

        Loads multiple sets of maneuvers such as multisine, doublet, and sweep
        excitations for roll, pitch, and yaw axes. Each maneuver file defines
        a (N, 4) array with columns: [time, roll, pitch, yaw].

        To create and save a maneuver, use the 'save_maneuver()'
        function in 'maneuver_utils.py'.

        Raises
        ------
        OSError
            If one or more maneuver CSV files cannot be found or read.

        Notes
        -----
        - The first time value in each file must be zero.
        - Maneuver arrays are stored as:
            - 'self.allsines', 'self.rolsines', 'self.pitsines', 'self.yawsines'
            - 'self.roldoublet', 'self.pitdoublet', 'self.yawdoublet'
            - 'self.rolsweep', 'self.pitsweep', 'self.yawsweep'
        """
        file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/sines_7deg_15s_0.1-1.5.csv"
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        time = data[:, 0]
        empty = np.zeros_like(time)
        self.allsines = np.array([time, data[:, 1], data[:, 2], data[:, 3]]).T
        self.rolsines = np.array([time, data[:, 1], empty, empty]).T
        self.pitsines = np.array([time, empty, data[:, 2], empty]).T
        self.yawsines = np.array([time, empty, empty, data[:, 3]]).T

        file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/doublet_7deg_15s_0.1-1.5.csv"
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        time = data[:, 0]
        empty = np.zeros_like(time)
        self.roldoublet = np.array([time, data[:, 1], empty, empty]).T
        self.pitdoublet = np.array([time, empty, data[:, 2], empty]).T
        self.yawdoublet = np.array([time, empty, empty, data[:, 3]]).T

        file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/sweep_7deg_15s_0.1-1.5.csv"
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        time = data[:, 0]
        empty = np.zeros_like(time)
        self.rolsweep = np.array([time, data[:, 1], empty, empty]).T
        self.pitsweep = np.array([time, empty, data[:, 2], empty]).T
        self.yawsweep = np.array([time, empty, empty, data[:, 3]]).T

    def user_input_loop(self) -> None:
        """
        Handle user input for selecting maneuvers when kill switch is inactive.

        Runs in a separate thread and waits for console input to change
        the 'maneuver_mode'. The menu is displayed only when the kill switch
        is below the configured threshold (inactive).

        Notes
        -----
        - User can select between 10 maneuver options (0-9).
        - Input validation ensures only valid integers are accepted.
        """
        while rclpy.ok():
            # TODO: Make this code match the number and type of maneuvers.
            if (self.kill_switch <= self.kill_switch_threshold):
                print("\nManeuvers")
                print("=========")
                print("0: All Multisines")
                print("1: Multisine - Roll")
                print("2: Multisine - Pitch")
                print("3: Multisine - Yaw")
                print("4: Doublet - Roll")
                print("5: Doublet - Pitch")
                print("6: Doublet - Yaw")
                print("7: Sweep - Roll")
                print("8: Sweep - Pitch")
                print("9: Sweep - Yaw")
                while True:
                    try:
                        maneuver_input = int(input("\nEnter a Maneuver (0-9):\n"))
                        if maneuver_input not in range(0, 10):
                            print("Invalid maneuver. Enter a number between 0 and 9.")
                            continue
                        self.maneuver_mode = maneuver_input
                        break
                    except ValueError:
                        print("Invalid input. Please enter an integer between 0 and 9.")

    def logic_loop(self) -> None:
        """
        Main control loop triggered by the ROS2 timer.

        Publishes maneuver data based on the current kill switch state and 
        selected maneuver mode. Handles maneuver initialization, timing updates,
        and automatic stopping when the end of a trajectory is reached.

        Behavior
        --------
        - When kill switch is **high** (>= threshold):
            Executes the selected maneuver.
        - When kill switch is **low** (< threshold):
            Stops maneuver execution and allows new selection.

        Notes
        -----
        - The timer period is dynamically updated to match the maneuver timestep.
        - Resets the counter when execution stops.
        """
        # print(self.kill_switch)
        if (self.kill_switch >= self.kill_switch_threshold):
            # print("kill switch high")
            if (self.run_switch == 1):
                if (self.counter == 0):
                    # TODO: Make this code match the number and type of maneuvers.
                    if (self.maneuver_mode == 0):
                        self.current_maneuver = self.allsines
                    elif (self.maneuver_mode == 1):
                        self.current_maneuver = self.rolsines
                    elif (self.maneuver_mode == 2):
                        self.current_maneuver = self.pitsines
                    elif (self.maneuver_mode == 3):
                        self.current_maneuver = self.yawsines
                    elif (self.maneuver_mode == 4):
                        self.current_maneuver = self.roldoublet
                    elif (self.maneuver_mode == 5):
                        self.current_maneuver = self.pitdoublet
                    elif (self.maneuver_mode == 6):
                        self.current_maneuver = self.yawdoublet
                    elif (self.maneuver_mode == 7):
                        self.current_maneuver = self.rolsweep
                    elif (self.maneuver_mode == 8):
                        self.current_maneuver = self.pitsweep
                    elif (self.maneuver_mode == 9):
                        self.current_maneuver = self.yawsweep
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
                        self.counter += 1
                    else:
                        self.run_switch = 0
                        print("MANEUVER COMPLETE")

                else:
                    self.run_switch = 0
                    print("NO CURRENT MANEUVER")

            elif (self.run_switch == 0):
                self.counter = self.initial_counter

        else:
            # print("kill switch low")
            self.run_switch = 1
            self.counter = self.initial_counter

    def update_timer_period(self, new_timer_period) -> None:
        """
        Update the ROS2 timer period for maneuver publishing.

        Cancels the existing timer and creates a new one with the specified period.

        Parameters
        ----------
        new_timer_period : float
            Desired timer period in seconds.

        Raises
        ------
        ValueError
            If 'new_timer_period' is non-positive.
        """
        if (self.timer is not None):
            self.timer.cancel()

        self.current_timer_period = new_timer_period
        self.timer = self.create_timer(
            self.current_timer_period, self.logic_loop)
        
    def publish_trajectory(self) -> None:
        """
        Publish a single 'CtlTraj' message corresponding to the current maneuver step.

        Constructs and publishes a 'CtlTraj' message with roll, pitch, yaw, and thrust
        values based on the current time index in the maneuver array.

        Notes
        -----
        - The trajectory message includes duplicated control values (2 elements per list)
            to ensure compatibility with downstream systems expecting paired inputs.
        - A timestamp is attached to each message via 'self.get_clock().now().to_msg()'.
        """
        if (self.current_maneuver is not None):
            trajectory: CtlTraj = CtlTraj()
            trajectory.header.stamp = self.get_clock().now().to_msg() # type: ignore
            trajectory.roll  = [self.current_maneuver[self.counter, 1], self.current_maneuver[self.counter, 1]]
            trajectory.pitch = [self.current_maneuver[self.counter, 2], self.current_maneuver[self.counter, 2]]
            trajectory.yaw   = [self.current_maneuver[self.counter, 3], self.current_maneuver[self.counter, 3]]
            trajectory.thrust = [0.5, 0.5]
            trajectory.idx = 0
            self.input_signal.publish(trajectory)
            # print(f"Publishing trajectory: {trajectory.roll}, {trajectory.pitch}, {trajectory.yaw}")


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