#!/usr/bin/env python3

"""
run_mission_HITL.py - ROS2 node for publishing drone excitation signals using a kill switch.

This script defines a ROS2 node, 'PubInputSignals', which generates and publishes
predefined or live-generated input maneuvers to a drone's trajectory topic for testing
and system identification. Unlike 'run_mission_SITL.py', this node maps the run/stop
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
- DDS messages: 'ardupilot_msgs/RcIn

Author
------
Xander D. Mosley
Email: XanderDMosley.Engineer@gmail.com
Date: 11 Jul 2025
"""


from pathlib import Path
import json
import math
import os
from re import S
import textwrap
import threading

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray, String
from drone_interfaces.msg import CtlTraj, Telem
from mavros_msgs.msg import RCIn
from mavros.base import SENSOR_QOS

from ardupilot_msgs.msg import RcIn


__all__ = ['PubInputSignals']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"

SCREEN_WIDTH = 60
LABEL_WIDTH = 18

class PubInputSignals(Node):
    """
    ROS2 node for publishing predefined excitation signals to a drone's trajectory topic.

    This node interfaces with a physical RC kill switch on the flight controller to
    control the start/stop of maneuver execution. It publishes maneuver data as 
    'CtlTraj' messages for use in hardware-in-the-loop (HITL) or flight tests.

    Maneuvers can be loaded from pre-saved CSV files (e.g., multisines, doublets, sweeps).
    The user can select which maneuver to execute through a console interface
    when the kill switch is inactive.

    Author
    ------
    Xander D. Mosley

    History
    -------
    11 Jul 2025 - Created, XDM.
    19 Aug 2026 - Pull maneuvers from a dedicated mission plan within a mission library, XDM.
    19 Aug 2026 - Improved the user display, XDM.
    19 Aug 2026 - Removed the if-else loop complexity from the logic loop, XDM.
    """
    def __init__(self, ns=''):
        """
        Initialize the excitation node and set up publishers, subscribers, and timers.

        Parameters
        ----------
        ns : str, optional
            Namespace for the ROS2 node. Defaults to an empty string.

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

        Notes
        -----
        - A separate thread is launched for user input ('self.userthread').
        - Maneuver data is preloaded from CSV files during initialization.
        """
        super().__init__('excitation_node')
        self.clear_screen()
        self.print_header()
        self.print_developer_info()

        # Paths
        package_subroot = Path(__file__).resolve().parents[1]
        self.mission_library = (package_subroot / "ros2_sid" / "setup" / "mission_library")
        self.current_mission_file = (self.mission_library / "current_mission.json")

        # Controller
        self.setup_subs()
        self.kill_switch: float = 0.0
        self.rc_bias: int = 1   # Channel 1 starts at index 0
        self.kill_switch_channel: int = 9 - self.rc_bias
        self.kill_switch_threshold: float = 1550
        
        # Execution States
        self.run_switch: int = 0
        self.maneuver_mode: int = 0
        self.initial_counter: int = 0
        self.counter: int = self.initial_counter
        self.initial_timer_period: float = 0.02
        self.current_timer_period: float = self.initial_timer_period
        self.current_maneuver = None
        self.current_maneuver_name = None

        # Load Mission Maneuvers
        self.maneuvers()

        # ROS Publisher and Timer
        self.input_signal: Publisher = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        self.timer = self.create_timer(
            self.current_timer_period, self.logic_loop)
            
        # User Input Thread
        self.userthread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.userthread.start()
    

    def setup_subs(self) -> None:
        """
        Create subscriptions for receiving RC input data.

        Subscribes to '/mavros/rc/in' or '/ap/rcin'
        to read RC channel data and determine
        the state of the physical kill switch.
        """
        self.rcin_sub: Subscription = self.create_subscription(
            RCIn,
            '/mavros/rc/in',
            self.rcin_callback,
            qos_profile=SENSOR_QOS
        )
        # self.rcin_sub: Subscription = self.create_subscription(
        #     RcIn,
        #     '/ap/rcin',
        #     self.dds_rcin_callback,
        #     qos_profile=SENSOR_QOS
        # )
    
    def rcin_callback(self, sub_msg: RCIn) -> None:
        """
        Callback for RC input messages.

        Parameters
        ----------
        sub_msg : RCIn
            MAVROS RC input message containing PWM values for all channels.

        Notes
        -----
        - Updates 'self.kill_switch' with the PWM value from the configured channel.
        """
        self.kill_switch = float(sub_msg.channels[self.kill_switch_channel])
    
    def dds_rcin_callback(self, sub_msg: RcIn) -> None:
        """
        Callback for RC input messages.

        Parameters
        ----------
        sub_msg : RcIn
            DDS RC input message containing PWM values for all channels.

        Notes
        -----
        - Updates 'self.kill_switch' with the PWM value from the configured channel.
        """
        self.kill_switch = float(sub_msg.values[self.kill_switch_channel])


    def load_current_mission(self) -> Path:
        """
        Load the currently selected mission from current_mission.json.

        Returns
        -------
        Path
            Path to the current mission directory.
        """
        if not self.current_mission_file.exists():
            raise FileNotFoundError(
                f"Current mission file not found:\n"
                f"{self.current_mission_file}"
            )
        with self.current_mission_file.open("r", encoding="utf-8") as file:
            mission_data = json.load(file)
        
        mission_name = mission_data.get("subfolder")
        if not isinstance(mission_name, str) or not mission_name.strip():
            raise ValueError(
                "current_mission.json does not contain a valid "
                "'subfolder' field."
            )
        mission_name = mission_name.strip()
        mission_path = self.mission_library / mission_name
        if not mission_path.exists():
            raise FileNotFoundError(
                f"Current mission directory does not exist:\n"
                f"{mission_path}"
            )
        if not mission_path.is_dir():
            raise NotADirectoryError(
                f"Current mission path is not a directory:\n"
                f"{mission_path}"
            )
        
        self.current_mission_name = mission_name
        self.current_mission_path = mission_path

        return mission_path
    
    def load_mission_plan(self, mission_path: Path) -> dict:
        """
        Load mission_plan.json from the current mission directory.

        Parameters
        ----------
        mission_path : Path
            Path to the mission directory.

        Returns
        -------
        dict
            Mission plan data.
        """
        mission_plan_file = mission_path / "mission_plan.json"

        if not mission_plan_file.exists():
            raise FileNotFoundError(
                f"Mission plan not found:\n"
                f"{mission_path_file}"
            )

        with mission_plan_file.open("r", encoding="utf-8") as file:
            mission_plan = json.load(file)

        if not isinstance(mission_plan, dict):
            raise ValueError(
                f"mission_plan.json must contain a JSON object."
            )
        if "maneuvers" not in mission_plan:
            raise ValueError(
                f"Mission plan does not contain a 'maneuvers' list:\n"
                f"{mission_plan_file}"
            )
        if not isinstance(mission_plan["maneuvers"], list):
            raise ValueError(
                f"'maneuvers' in mission_plan.json must be a list."
            )
        
        return mission_plan
    
    def load_maneuver(self, file_path: Path) -> np.ndarray:
        """
        Load a maneuver CSV file.

        Parameters
        ----------
        file_path : Path
            Path to maneuver CSV.

        Returns
        -------
        np.ndarray
            Maneuver array with columns:
                [time, roll, pitch, yaw]

        Raises
        ------
        ValueError
            If the maneuver does not have the required structure.
        """
        try:
            data = np.loadtxt(
                file_path,
                delimiter=",",
                skiprows=1
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to load maneuver:\n"
                f"{file_path}\n"
                f"Error: {exc}"
            ) from exc
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 4:
            raise ValueError(
                f"Maneuver file must contain at least four columns:\n"
                f"{file_path}\n"
                f"Expected: time, roll, pitch, yaw"
            )
        data = data[:, :4]

        time = data[:, 0]
        if len(time) < 2:
            raise ValueError(
                f"Maneuver must contain at least two time samples:\n"
                f"{file_path}"
            )
        if not np.isclose(time[0], 0.0):
            raise ValueError(
                f"Maneuver time must start at zero:\n"
                f"{file_path}\n"
                f"First time value: {time[0]}"
            )
        time_steps = np.diff(time)
        if np.any(time_steps <= 0.0):
            raise ValueError(
                f"Maneuver time values must be strictly increasing:\n"
                f"{file_path}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"Maneuver contains non-finite values:\n"
                f"{file_path}"
            )
        
        return data
    
    def maneuvers(self) -> None:
        """
        Load maneuvers defined by the current mission's mission_plan.json.

        Maneuvers are loaded according to their 'order' field rather than
        filesystem order.

        Each maneuver is stored as:
            {
                "filename": ...,
                "order": ...,
                "name": ...,
                "path": ...,
                "data": ...
            }
        """
        mission_path = self.load_current_mission()
        mission_plan = self.load_mission_plan(mission_path)
        self.mission_plan = mission_plan

        for maneuver in mission_plan["maneuvers"]:
            if not isinstance(maneuver, dict):
                raise ValueError(
                    "Each maneuver entry must be a JSON object."
                )
            if not isinstance(maneuver.get("filename"), str):
                raise ValueError(
                    "Each maneuver must contain a valid 'filename'."
                )
            if not isinstance(maneuver.get("name"), str):
                raise ValueError(
                    "Each maneuver must contain a valid 'name'."
                )
            if not isinstance(maneuver.get("order"), int):
                raise ValueError(
                    f"Maneuver '{maneuver.get('name')}' "
                    "must contain an integer 'order'."
                )

        maneuver_definitions = sorted(
            mission_plan["maneuvers"],
            key=lambda maneuver: maneuver.get("order", 0)
        )
        self.maneuver_list = []

        for maneuver_definition in maneuver_definitions:
            filename = maneuver_definition.get("filename")
            order = maneuver_definition.get("order")
            name = maneuver_definition.get("name")

            if not isinstance(filename, str) or not filename.strip():
                print(
                    "\nWARNING: Maneuver entry has no valid filename"
                )
                continue
            if not isinstance(name, str) or not name.strip():
                print(
                    f"\nWARNING: Maneuver '{filename}' has no valid name."
                )
                continue

            file_path = mission_path / filename

            if not file_path.exists():
                print(
                    f"\nWARNING: Maneuver file does not exist:\n"
                    f"{file_path}"
                )
                continue
            if not file_path.is_file():
                print(
                    f"\nWARNING: Maneuver path is not a file:\n"
                    f"{file_path}"
                )
                continue
            
            try:
                data = self.load_maneuver(file_path)
            except ValueError as exc:
                print(
                    f"\nWARNING: Skipping invalid maneuver:"
                )
                print(exc)
                continue
            
            self.maneuver_list.append(
                {
                    "filename": filename,
                    "order": order,
                    "name": name,
                    "path": file_path,
                    "data": data,
                }
            )
        
        if not self.maneuver_list:
            raise ValueError(
                f"No valid maneuvers were loaded for mission"
                f"'{self.current_mission_name}'."
            )
        
        self.print_section("Current Mission")
        print(f"{'Mission Name':<{LABEL_WIDTH}} : {mission_plan.get('name', self.current_mission_name)}")
        print(f"{'No. of Maneuvers':<{LABEL_WIDTH}} : {len(self.maneuver_list)}")
        wrapped_lines = textwrap.wrap(mission_plan.get('description', ''), width=(SCREEN_WIDTH-LABEL_WIDTH))
        if wrapped_lines:
            print(f"{'Description':<{LABEL_WIDTH}} : {wrapped_lines[0]}")
            for line in wrapped_lines[1:]:
                print(f"{'':<{LABEL_WIDTH}}   {line}")
        else:
            print(f"{'Description':<{LABEL_WIDTH}} : ")
        print()

        self.print_section("Mission Plan")
        for index, maneuver in enumerate(self.maneuver_list):
            duration = maneuver["data"][-1, 0]
            print(
                f" {index}. "
                f"{maneuver['name']}, "
                f"order {maneuver['order']}, "
                f"{duration:.3f} s"
            )
        
        print()
        self.print_section("Default Selection")
        default_maneuver = self.maneuver_list[self.maneuver_mode]
        print(
            f"{'Maneuver':<{LABEL_WIDTH}} : "
            f"{self.maneuver_mode}. "
            f"{default_maneuver['name']}"
        )
        print()


    def user_input_loop(self) -> None:
        """
        Handle user input for selecting maneuvers when kill switch is inactive.

        Runs in a separate thread and waits for console input to change
        the 'maneuver_mode'. The menu is displayed only when the kill switch
        is below the configured threshold (inactive).

        Notes
        -----
        - User can select between the current mission's maneuvers.
        - Input validation ensures only valid integers are accepted.
        """
        while rclpy.ok():
            if (self.kill_switch <= self.kill_switch_threshold):
                self.clear_screen()
                self.print_page(f"Mission: {self.current_mission_name}")
                self.print_maneuver_menu()
                
                while True:
                    try:
                        maneuver_input = int(
                            input(
                                f"{f'Enter a Maneuver (0-{len(self.maneuver_list) - 1})':<{LABEL_WIDTH}} : "
                            )
                        )
                    except ValueError:
                        print(
                            "Invalid Input. "
                            "Please enter an integer."
                        )
                        continue
                    if maneuver_input not in range(len(self.maneuver_list)):
                        print(
                            "Invalid Input. "
                            "Please enter an integer between "
                            f"0 and {len(self.maneuver_list) - 1}"
                        )
                        continue
                    
                    self.maneuver_mode = maneuver_input
                    selected = self.maneuver_list[self.maneuver_mode]
                    print()
                    print(
                        f"Selected maneuver: "
                        f"{selected['name']}"
                    )
                    print()

                    break

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
        if (self.kill_switch < self.kill_switch_threshold):
            # print("kill switch low")
            if self.current_maneuver is not None:
                self.maneuver_stop(completed=False)
            else:
                self.counter = self.initial_counter
                self.run_switch = 1
            return
        
        # print("kill switch high")
        if (self.run_switch == 0):
            # If run_switch == 0, we are NOT allowed to start another
            # maneuver. This occurs after a maneuver has completed.
            return
        
        if self.current_maneuver is None:
            # Select maneuver at beginning of execution
            if not self.maneuver_list:
                self.run_switch = 0
                print("NO MANEUVERS LOADED")
                return
            if self.maneuver_mode not in range(len(self.maneuver_list)):
                self.run_switch = 0
                print("INVALID MANEUVER SELECTION")
                return
            selected_maneuver = self.maneuver_list[self.maneuver_mode]
            self.current_maneuver = selected_maneuver["data"]
            self.current_maneuver_name = selected_maneuver["name"]

            time_steps = np.diff(self.current_maneuver[:, 0])
            maneuver_timer_period = float(time_steps[0])
            if not np.allclose(
                time_steps,
                maneuver_timer_period,
                rtol=1e-6,
                atol=1e-9
                ):
                print(
                    f"\nWARNING: Maneuver "
                    f"'{self.current_maneuver_name}' "
                    f"has nonuniform time steps."
                )
            if maneuver_timer_period <= 0.0:
                self.run_switch = 0
                self.current_maneuver = None
                self.current_maneuver_name = None
                print("INVALID MANEUVER TIMESTEP")
                return
            
            if not np.isclose(
                maneuver_timer_period,
                self.current_timer_period
                ):
                self.update_timer_period(maneuver_timer_period)
            
            print(
                f"\nSTARTING MANEUVER: "
                f"{self.current_maneuver_name}"
            )

        if self.counter < len(self.current_maneuver):
            self.publish_trajectory()
            self.counter += 1
        else:
            # Maneuver is complete.
            self.maneuver_stop(completed=True)

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
        Publish a single trajectory message corresponding to the current maneuver count.

        Constructs and publishes a 'CtlTraj' message with roll, pitch, yaw, and thrust
        values based on the current time index in the maneuver array and a constant thrust value.

        Notes
        -----
        - The published message corresponds to the current index ('self.counter')
            within the maneuver array.
        - A timestamp is attached to each message via 'self.get_clock().now().to_msg()'.
        """
        if (self.current_maneuver is not None):
            """
            Check that 'self.control_type' in drone_ros/drone_ros/scripts/Drone.py
            is set to control type 1; and that control type 1 looks like the following:
            '''python3
                    elif self.control_method == self.control_type[1]:
                        roll_cmd = np.rad2deg(roll_traj[idx_command])
                        pitch_cmd = np.rad2deg(pitch_traj[idx_command])
                        yaw_cmd = np.rad2deg(yaw_traj[idx_command])
                        thrust_cmd = float(thrust_traj[idx_command])
                        print("roll_cmd: ", roll_cmd)
                        print("pitch_cmd: ", pitch_cmd)
                        print("yaw_cmd: ", yaw_cmd)
                        print("thrust cmd", thrust_cmd)
                        self.sendAttitudeTarget(roll_angle=roll_cmd,
                                                pitch_angle=pitch_cmd,
                                                yaw_angle=yaw_cmd,
                                                thrust=thrust_cmd)
            '''
            """
            trajectory: CtlTraj = CtlTraj()
            trajectory.header.stamp = self.get_clock().now().to_msg()
            trajectory.roll  = [self.current_maneuver[self.counter, 1], self.current_maneuver[self.counter, 1]]
            trajectory.pitch = [self.current_maneuver[self.counter, 2], self.current_maneuver[self.counter, 2]]
            trajectory.yaw   = [self.current_maneuver[self.counter, 3], self.current_maneuver[self.counter, 3]]
            trajectory.thrust = [0.5, 0.5]
            trajectory.idx = 0
            self.input_signal.publish(trajectory)
            # print(f"Publishing trajectory: {trajectory.roll}, {trajectory.pitch}, {trajectory.yaw}")


    def clear_screen(self):
        """Clear the terminal screen."""
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            # Fall back to printing blank lines
            print("\n" * 100)
    
    def print_header(self):
        print("=" * SCREEN_WIDTH)
        print("MISSION PLAN PUBLISHER")
        print("=" * SCREEN_WIDTH)

    def print_developer_info(self):
        print("Developed by Xander D. Mosley")
        print("-" * SCREEN_WIDTH)
        print()

    def print_page(self, title: str):
        """Print a page header."""
        self.clear_screen()
        self.print_header()
        print(title)
        print("-" * SCREEN_WIDTH)
        print()

    def print_section(self, title: str):
        """Print a section header."""
        print(title)
        print(("- " * ((len(title) + 1) // 2)).rstrip())

    def print_maneuver_menu(self) -> None:
        self.print_section("Maneuvers")
        for index, maneuver in enumerate(self.maneuver_list):
            duration = maneuver["data"][-1, 0]
            print(
                f" {index}. {maneuver['name']}"
                f" ({duration:.3f} s)"
            )
        print()
    
    def maneuver_stop(self, completed: bool = False) -> None:
        if self.current_maneuver is None:
            return
        
        if completed:
            print(
                f"MANEUVER COMPLETE: "
                f"{self.current_maneuver_name}"
            )
        else:
            print(
                f"MANEUVER STOPPED: "
                f"{self.current_maneuver_name}"
            )

        self.run_switch = 0
        self.counter = self.initial_counter
        self.current_maneuver = None
        self.current_maneuver_name = None
        

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