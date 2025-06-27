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

from ros2_sid.inputdesign import frequency_sweep, multi_step


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


class NewSolution(Node):
    def __init__(self, ns=''):
        super().__init__('excitation_node')
        self.switch: int = 1
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
        # maneuvers must have the shape (N, 4)
        # where the columns (in order) are:
        # time, roll signal, pitch signal, yaw signal;
        # and the first time value must be zero

        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, sweep = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        empty = np.zeros_like(time)
        self.rolsweep = np.array([time, sweep, empty, empty]).T

        amplitude: float = 1.
        natural_frequency: float = 0.5
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.01
        final_time: float = 15.
        time, doublet = multi_step(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        empty = np.zeros_like(time)
        self.roldoublet = np.array([time, doublet, empty, empty]).T
        

        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.05
        final_time: float = 15.
        time, sweep = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        empty = np.zeros_like(time)
        self.pitsweep = np.array([time, empty, sweep, empty]).T

        amplitude: float = 1.
        natural_frequency: float = 0.2
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, doublet = multi_step(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        empty = np.zeros_like(time)
        self.pitdoublet = np.array([time, empty, doublet, empty]).T
        

        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, sweep = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        empty = np.zeros_like(time)
        self.yawsweep = np.array([time, empty, empty, sweep]).T
        
        amplitude: float = 1.
        natural_frequency: float = 2.
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, doublet = multi_step(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        empty = np.zeros_like(time)
        self.yawdoublet = np.array([time, empty, empty, doublet]).T

    def user_input_loop(self) -> None:
        while rclpy.ok():
            userswitch = int(input("\nTesting Switch (0-1):\n"))
            if (userswitch != self.switch):
                self.switch = userswitch
            if (self.switch == 0):
                print("\nManeuvers")
                print("=========")
                print("1: Roll  - Sweep")
                print("2: Roll  - Doublet")
                print("3: Pitch - Sweep")
                print("4: Pitch - Doublet")
                print("5: Yaw   - Sweep")
                print("6: Yaw   - Doublet")
                self.maneuver_mode: int = int(input("\nEnter a Maneuver (1-6):\n"))

    def logic_loop(self) -> None:
        # self.switch could be a variable defined by the control's...
        # function, so the function isn't running at each if statement
        if (self.switch == 1):
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
                    self.switch = 0
                    print("MANEUVER COMPLETE")


            else:
                # could this be done earlier as well???
                self.switch = 0
                print("NOOO!")
            


        elif (self.switch == 0):
            self.counter = self.initial_counter

    def update_timer_period(self, new_timer_period) -> None:
        if (self.timer is not None):
            self.timer.cancel()

        self.current_timer_period = new_timer_period
        self.timer = self.create_timer(
            self.current_timer_period, self.logic_loop)
        
    def publish_trajectory(self) -> None:
        trajectory: CtlTraj = CtlTraj()
        # trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.roll  = [self.current_maneuver[self.counter, 1]]
        trajectory.pitch = [self.current_maneuver[self.counter, 2]]
        trajectory.yaw   = [self.current_maneuver[self.counter, 3]]
        trajectory.idx = 0
        self.input_signal.publish(trajectory)
        self.counter += 1




class PotentialSolution(Node):
    def __init__(self, ns=''):
        super().__init__('excitation_node')
        self.switch: int = 1
        self.maneuver_mode: int = 0
        self.maneuvers()
        self.counter = 0
        
        self.input_signal: Publisher = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        
        self.maneuver_timer = None
        self.end_maneuver_timer = False
        self.main_timerperiod: float = 0.02
        self.main_timer = self.create_timer(
            self.main_timerperiod, self.execute)
        
        self.userthread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.userthread.start()

    def user_input_loop(self):
        while rclpy.ok():
            userswitch = int(input("Testing Switch (0-1): "))
            if (userswitch != self.switch):
                self.switch = userswitch
                if (self.switch == 0):
                    self.maneuver_mode: int = int(input("Enter a Maneuver (0-6): "))
        
    def maneuvers(self) -> None:
        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, sweep = FrequencySweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        self.rolsweep = np.array([time, sweep]).T

        amplitude: float = 1.
        natural_frequency: float = 2.
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, doublet = MultiStep(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        self.roldoublet = np.array([time, doublet]).T
        

        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, sweep = FrequencySweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        self.pitsweep = np.array([time, sweep]).T

        amplitude: float = 1.
        natural_frequency: float = 2.
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, doublet = MultiStep(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        self.pitdoublet = np.array([time, doublet]).T
        

        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, sweep = FrequencySweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        self.yawsweep = np.array([time, sweep]).T
        
        amplitude: float = 1.
        natural_frequency: float = 2.
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, doublet = MultiStep(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        self.yawdoublet = np.array([time, doublet]).T
        
    def execute(self):
        if (self.switch == 1):
            # Currently no logic for if switching from one maneuver mode to another.
            if (self.maneuver_timer is not None) and (self.end_maneuver_timer is True):
                self.end_maneuver_timer = False
                self.maneuver_timer.cancel()
                self.maneuver_timer = None
                self.counter = 0
            
            if (self.maneuver_timer is None):
                if (self.maneuver_mode == 0):
                    maneuver_timerperiod: float = self.rolsweep[1, 0]
                elif (self.maneuver_mode == 1):
                    maneuver_timerperiod: float = self.roldoublet[1, 0]
                elif (self.maneuver_mode == 2):
                    maneuver_timerperiod: float = self.pitsweep[1, 0]
                elif (self.maneuver_mode == 3):
                    maneuver_timerperiod: float = self.pitdoublet[1, 0]
                elif (self.maneuver_mode == 4):
                    maneuver_timerperiod: float = self.yawsweep[1, 0]
                elif (self.maneuver_mode == 5):
                    maneuver_timerperiod: float = self.yawdoublet[1, 0]
                else:
                    maneuver_timerperiod: float = self.main_timerperiod
                
                self.maneuver_timer = self.create_timer(
                    maneuver_timerperiod, self.publish_trajectory)
            
        elif (self.switch == 0):
            if (self.maneuver_timer is not None):
                self.end_maneuver_timer = False
                self.maneuver_timer.cancel()
                self.maneuver_timer = None
                self.counter = 0

    def publish_trajectory(self):
        self.trajectory: CtlTraj = CtlTraj()

        self.determine_trajectory()

        self.trajectory.idx = 0
        self.input_signal.publish(self.trajectory)

    def determine_trajectory(self):
        if (self.maneuver_mode == 0):
            if (self.counter < len(self.rolsweep)):
                self.trajectory.roll  = [self.rolsweep[self.counter, 1]]
                self.trajectory.pitch = [0.]
                self.trajectory.yaw   = [0.]
                self.counter += 1
            else:
                self.end_maneuver_timer = True

        elif (self.maneuver_mode == 1):
            if (self.counter < len(self.roldoublet)):
                self.trajectory.roll  = [self.roldoublet[self.counter, 1]]
                self.trajectory.pitch = [0.]
                self.trajectory.yaw   = [0.]
                self.counter += 1
            else:
                self.end_maneuver_timer = True

        elif (self.maneuver_mode == 2):
            if (self.counter < len(self.pitsweep)):
                self.trajectory.roll  = [0.]
                self.trajectory.pitch = [self.pitsweep[self.counter, 1]]
                self.trajectory.yaw   = [0.]
                self.counter += 1
            else:
                self.end_maneuver_timer = True

        elif (self.maneuver_mode == 3):
            if (self.counter < len(self.pitdoublet)):
                self.trajectory.roll  = [0.]
                self.trajectory.pitch = [self.pitdoublet[self.counter, 1]]
                self.trajectory.yaw   = [0.]
                self.counter += 1
            else:
                self.end_maneuver_timer = True

        elif (self.maneuver_mode == 4):
            if (self.counter < len(self.yawsweep)):
                self.trajectory.roll  = [0.]
                self.trajectory.pitch = [0.]
                self.trajectory.yaw   = [self.yawsweep[self.counter, 1]]
                self.counter += 1
            else:
                self.end_maneuver_timer = True

        elif (self.maneuver_mode == 5):
            if (self.counter < len(self.yawdoublet)):
                self.trajectory.roll  = [0.]
                self.trajectory.pitch = [0.]
                self.trajectory.yaw   = [self.yawdoublet[self.counter, 1]]
                self.counter += 1
            else:
                self.end_maneuver_timer = True

        else:
            self.trajectory.roll  = [0.]
            self.trajectory.pitch = [0.]
            self.trajectory.yaw   = [0.]


class PubInputSignals(Node):
    def __init__(self, ns=''):
        super().__init__('excitation_node')
        self.swtich: int = 1
        self.mode: int = 0

        self.input_signal: Publisher = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        
        self.Maneuvers()
        self.step_count: int = 0

        self.timer_period: float = 0.1
        self.timer = self.create_timer(
            self.timer_period, self.PublishSignal)
        
    def Maneuvers(self) -> None:
        amplitude: float = 1.
        minimum_frequency: float = 1.
        maximum_frequency: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, sweep = FrequencySweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
        self.rolsweep = np.array([time, sweep]).T

        amplitude: float = 1.
        natural_frequency: float = 2.
        pulses: list = [1, 1]
        time_delay: float = 5.
        time_step: float = 0.1
        final_time: float = 15.
        time, doublet = MultiStep(amplitude, natural_frequency, pulses, time_delay, time_step, final_time)
        self.roldoublet = np.array([time, doublet]).T

    def PublishSignal(self) -> None:
        trajectory: CtlTraj = CtlTraj()

        if (self.step_count < len(self.rolsweep)):
            trajectory.roll = [self.rolsweep[self.step_count, 1]]
            trajectory.pitch = [0.]
            trajectory.yaw = [0.]
            self.step_count += 1
        else:
            self.step_count = 0

        trajectory.idx = 0
        self.input_signal.publish(trajectory)
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
        
    # def Execute(self) -> None:
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
    #     if (self.swtich == 1):
    #         if (self.mode == 0):
    #             self.timer_period = 0.1
    #         else:
    #             continue
    #     else:
    #         continue

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
    pub_example = NewSolution()

    while rclpy.ok():
        try:
            rclpy.spin_once(pub_example, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    pub_example.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
