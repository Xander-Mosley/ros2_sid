#!/usr/bin/env python3

"""
testingsignals_SINL.py - ROS2 node for publishing simulated drone excitation signals.

This script defines a ROS2 node, 'PubInputSignals', which generates and publishes
predefined or live-generated input maneuvers to a drone's trajectory topic for testing
and system identification purposes. The node supports multiple types of maneuvers
including multisines, doublets, and frequency sweeps for roll, pitch, and yaw axes.

Key Features
------------
- Load maneuvers from CSV files to ensure reproducibility.
- Generate and publish different excitation signals on the 'trajectory' topic.
- User-selectable maneuver modes.
- Adjustable publishing timer to match maneuver time steps.
- Runs in a ROS2 environment and leverages threading to handle user input concurrently.

Maneuver Format
---------------
- All maneuvers are expected as arrays of shape (N, 4):
  [time, roll_signal, pitch_signal, yaw_signal]
- Time values must start at zero.

Usage
-----
1. Launch the node:
    '''bash
    ros2 run ros2_sid testingsignals_SINL.py
    '''
2. Toggle execution and select maneuvers via console input.

Custom Dependencies:
- Custom message: drone_interfaces/CtlTraj, drone_interfaces/Telem
- Input signal utilities from 'ros2_sid.input_design'

Author
------
Xander D. Mosley
Email: XanderDMosley.Engineer@gmail.com
Date: 11 Jul 2025
"""


import math
import os
import threading
from re import S

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray, String
from drone_interfaces.msg import CtlTraj, Telem

from ros2_sid.input_design import frequency_sweep, multi_sine, multi_step


__all__ = ['PubInputSignals']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


class PubInputSignals(Node):
    """
    ROS2 node that publishes predefined or generated input signals (maneuvers) 
    as control trajectories for aircraft system identification or control testing.

    This node allows users to select and execute predefined excitation maneuvers 
    interactively. Maneuvers can be loaded from CSV files or generated dynamically.

    Attributes
    ----------
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
        Continuously listen for user input to control the excitation process.

        Provides a simple CLI interface for:
            - Starting or stopping the signal publishing ('run_switch').
            - Selecting a maneuver mode (0-9) when publishing is stopped.

        Notes
        -----
        - This method runs in a separate daemon thread to avoid blocking
           the ROS2 executor thread.
        """
        while rclpy.ok():
            try:
                userswitch = int(input("\nTesting Switch (0-1):\n"))
                if userswitch not in [0, 1]:
                    print("Invalid switch. Enter 0 or 1.")
                    continue
            except ValueError:
                print("Invalid input. Please enter an integer (0 or 1).")
                continue
            if userswitch != self.run_switch:
                self.run_switch = userswitch

            # TODO: Make this code match the number and type of maneuvers.
            if self.run_switch == 0:
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
        Main logic loop for executing and publishing maneuvers.

        This callback is triggered periodically by the ROS2 timer. It manages:
            - Selecting the appropriate maneuver based on 'maneuver_mode'.
            - Publishing trajectory points to the 'CtlTraj' topic.
            - Resetting the state when a maneuver completes.

        Notes
        -----
        - The timer period can change dynamically to match the maneuver's sampling time.
        """
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

    def update_timer_period(self, new_timer_period) -> None:
        """
        Update the ROS2 timer period used for publishing trajectories.

        Cancels the current timer and creates a new one with the specified period.

        Parameters
        ----------
        new_timer_period : float
            New timer period in seconds.

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
        Publish a single trajectory message based on the current maneuver.

        Creates and publishes a 'CtlTraj' message containing the current roll, 
        pitch, and yaw excitation signals and a constant thrust value.

        Notes
        -----
        - The published message corresponds to the current index ('self.counter')
            within the maneuver array.
        - The timestamp is currently not set but can be added using 
            'self.get_clock().now().to_msg()'.
        """
        if (self.current_maneuver is not None):
            trajectory: CtlTraj = CtlTraj()
            # trajectory.header.stamp = self.get_clock().now().to_msg()
            # TODO: Check if two indexes are required.
            trajectory.roll  = [self.current_maneuver[self.counter, 1]]
            trajectory.pitch = [self.current_maneuver[self.counter, 2]]
            trajectory.yaw   = [self.current_maneuver[self.counter, 3]]
            trajectory.thrust = [0.5]
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