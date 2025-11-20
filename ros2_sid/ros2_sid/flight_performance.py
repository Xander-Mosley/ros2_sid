#!/usr/bin/env python3#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system

import numpy as np
import re
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from typing import Union, Optional
from plotter_class import PlotFigure


def _ensure_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
        
def _recenter_angles(data: np.ndarray) -> np.ndarray:
    x = _ensure_numpy(data)
    recentered = np.where(x >= 0, 180 - x, -180 - x)
    return recentered


def plot_overall(
        dataframe: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        plot_labels: Optional[dict] = None
        ) -> PlotFigure:

    if dataframe is None or dataframe.empty or 'timestamp' not in dataframe.columns:
        raise ValueError("Invalid DataFrame provided.")
    if plot_labels is None:
        plot_labels = {}
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    ail_def = dataframe["rcout_ch1"] - 1500
    elv_def = dataframe["rcout_ch2"] - 1500
    rud_def = dataframe["rcout_ch4"] - 1500
    rol_rate = dataframe["gx"]
    pit_rate = dataframe["gy"]
    yaw_rate = dataframe["gz"]
    rol_deg = dataframe["roll_deg"]
    pit_deg = dataframe["pitch_deg"]
    yaw_deg = dataframe["yaw_deg"]
    airspeed = dataframe["airspeed"]
    altitude = dataframe["altitude"]

    fig = PlotFigure(nrows=5, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Overall Figure"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Controls Over Time", ylabel="PWM Signal\n[±1500]")
    fig.add_data(0, time, ail_def, label='Aileron', color='blue')
    fig.add_data(0, time, elv_def, label='Elevator', color='red')
    fig.add_data(0, time, rud_def, label='Rudder', color='green')

    fig.define_subplot(1, title="Rates Over Time", ylabel="Angular Velocity\n[rad/s]")
    fig.add_data(1, time, rol_rate, label='Roll', color='blue')
    fig.add_data(1, time, pit_rate, label='Pitch', color='red')
    fig.add_data(1, time, yaw_rate, label='Yaw', color='green')

    fig.define_subplot(2, title="Attitude Over Time", ylabel="Orientation\n[deg]")
    fig.add_data(2, time, rol_deg, label='Roll', color='blue')
    fig.add_data(2, time, pit_deg, label='Pitch', color='red')
    fig.add_data_secondary_y(2, time, yaw_deg, label='Yaw', color='green')

    fig.define_subplot(3, title="Airspeed Over Time", ylabel="Airspeed\n[m/s]")
    fig.add_data(3, time, airspeed, color='black')

    fig.define_subplot(4, title="Altitude Over Time", ylabel="Altitude\n[m]", xlabel="Time [s]")
    fig.add_data(4, time, altitude, color='black')

    fig.set_all_legends(loc='upper right', fontsize='medium')

    return fig

def plot_signals(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 1"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    ail_def = dataframe["rcout_ch1"] - 1500
    elv_def = dataframe["rcout_ch2"] - 1500
    rud_def = dataframe["rcout_ch4"] - 1500
    thrust = dataframe["rcout_ch3"]
    
    axs[0].plot(time, ail_def, label='Aileron', color="blue", linestyle="-")
    axs[1].plot(time, elv_def, label='Elevator', color="red", linestyle="-")
    axs[2].plot(time, rud_def, label='Rudder', color="green", linestyle="-")
    axs[3].plot(time, thrust, label='Thrust', color="black", linestyle="-")
    
    # --- Final Formatting ---
    axs[0].set_title("Aileron Signal Over Time")
    axs[0].set_ylabel("PWM Signal\n[±1500]")
    axs[1].set_title("Elevator Signal Over Time")
    axs[1].set_ylabel("PWM Signal\n[±1500]")
    axs[2].set_title("Rudder Signal Over Time")
    axs[2].set_ylabel("PWM Signal\n[±1500]")
    axs[3].set_title("Thrust Signal Over Time")
    axs[3].set_ylabel("PWM Signal\n[0-3000]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")

def plot_rates(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 2"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    rol_rate = dataframe["gx"]
    pit_rate = dataframe["gy"]
    yaw_rate = dataframe["gz"]
    
    axs[0].scatter(time, rol_rate, label='Roll', color="blue")
    axs[1].scatter(time, pit_rate, label='Pitch', color="red")
    axs[2].scatter(time, yaw_rate, label='Yaw', color="green")
    
    # --- Final Formatting ---
    axs[0].set_title("Roll Rate Over Time")
    axs[0].set_ylabel("Rate\n[rad/s]")
    axs[1].set_title("Pitch Rate Over Time")
    axs[1].set_ylabel("Rate\n[rad/s]")
    axs[2].set_title("Yaw Rate Over Time")
    axs[2].set_ylabel("Rate\n[rad/s]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")

def plot_raw_rates(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 2"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    rol_rate = dataframe["gx_raw"]
    pit_rate = dataframe["gy_raw"]
    yaw_rate = dataframe["gz_raw"]
    
    axs[0].scatter(time, rol_rate, label='Roll', color="blue")
    axs[1].scatter(time, pit_rate, label='Pitch', color="red")
    axs[2].scatter(time, yaw_rate, label='Yaw', color="green")
    
    # --- Final Formatting ---
    axs[0].set_title("Roll Rate Over Time")
    axs[0].set_ylabel("Rate\n[rad/s]")
    axs[1].set_title("Pitch Rate Over Time")
    axs[1].set_ylabel("Rate\n[rad/s]")
    axs[2].set_title("Yaw Rate Over Time")
    axs[2].set_ylabel("Rate\n[rad/s]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")

def plot_angles(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 3"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    rol_deg = dataframe["roll_deg"]
    pit_deg = dataframe["pitch_deg"]
    yaw_deg = _recenter_angles(dataframe["yaw_deg"])
    
    axs[0].scatter(time, rol_deg, label='Roll', color="blue")
    axs[1].scatter(time, pit_deg, label='Pitch', color="red")
    axs[2].scatter(time, yaw_deg, label='Yaw', color="green")
    
    # --- Final Formatting ---
    axs[0].set_title("Roll Angle Over Time")
    axs[0].set_ylabel("Deflection\n[deg]")
    axs[1].set_title("Pitch Angle Over Time")
    axs[1].set_ylabel("Deflection\n[deg]")
    axs[2].set_title("Yaw Angle Over Time")
    axs[2].set_ylabel("Deflection\n[deg]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")

def plot_energy(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 4"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    airspeed = dataframe["airspeed"]
    altitude = dataframe["altitude"]
    
    axs[0].plot(time, airspeed, label='Airspeed', color="black")
    axs[1].plot(time, altitude, label='Altitude', color="black")
    
    # --- Final Formatting ---
    axs[0].set_title("Airspeed Over Time")
    axs[0].set_ylabel("Airspeed\n[m/s]")
    axs[1].set_title("Altitude Over Time")
    axs[1].set_ylabel("Altitude\n[m]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")

def plot_trajectory(
        dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if dataframe is None:
        raise ValueError("No DataFrame provided.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 5"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    ail_def = dataframe["roll_cmd"]
    elv_def = dataframe["yaw_cmd"]
    rud_def = dataframe["pitch_cmd"]
    
    axs[0].scatter(time, ail_def, label='Aileron', color="blue", linestyle="-")
    axs[1].scatter(time, elv_def, label='Elevator', color="red", linestyle="-")
    axs[2].scatter(time, rud_def, label='Rudder', color="green", linestyle="-")
    
    # --- Final Formatting ---
    axs[0].set_title("Aileron Trajectory Over Time")
    axs[0].set_ylabel("Deflection\n[deg]")
    axs[1].set_title("Elevator Trajectory Over Time")
    axs[1].set_ylabel("Deflection\n[deg]")
    axs[2].set_title("Rudder Trajectory Over Time")
    axs[2].set_ylabel("Deflection\n[deg]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")
    
def plot_input_performance(
        trajectory_dataframe: pd.DataFrame,
        odometry_dataframe: pd.DataFrame,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if trajectory_dataframe is None:
        raise ValueError("No DataFrame provided.")
    if trajectory_dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in trajectory_dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if odometry_dataframe is None:
        raise ValueError("No DataFrame provided.")
    if odometry_dataframe.empty:
        raise ValueError("DataFrame is empty.")
    if 'timestamp' not in odometry_dataframe.columns:
        raise ValueError("DataFrame has no 'timestamp' column.")
    if plot_labels is None:
        plot_labels = {}
        
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Figure 6"
    subtitle = plot_labels.get("subtitle", "Last Flight Test")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
        
    if start_time is not None:
        trajectory_dataframe = trajectory_dataframe[trajectory_dataframe["timestamp"] >= start_time]
    if end_time is not None:
        trajectory_dataframe = trajectory_dataframe[trajectory_dataframe["timestamp"] <= end_time]
    if start_time is not None:
        odometry_dataframe = odometry_dataframe[odometry_dataframe["timestamp"] >= start_time]
    if end_time is not None:
        odometry_dataframe = odometry_dataframe[odometry_dataframe["timestamp"] <= end_time]

    trajectory_time = trajectory_dataframe["timestamp"]
    ail_def = trajectory_dataframe["pitch_cmd"]
    elv_def = trajectory_dataframe["yaw_cmd"]
    rud_def = trajectory_dataframe["roll_cmd"]
    
    odometry_time = odometry_dataframe["timestamp"]
    rol_deg = odometry_dataframe["roll_deg"]
    pit_deg = odometry_dataframe["pitch_deg"]
    yaw_deg = _recenter_angles(odometry_dataframe["yaw_deg"])
    
    axs[0].plot(trajectory_time, ail_def, label='Desired', color="black", linestyle="--")
    axs[1].plot(trajectory_time, elv_def, label='Desired', color="black", linestyle="--")
    axs[2].plot(trajectory_time, rud_def, label='Desired', color="black", linestyle="--")
    axs[0].plot(odometry_time, rol_deg, label='Actual', color="blue", linestyle="-")
    axs[1].plot(odometry_time, pit_deg, label='Actual', color="red", linestyle="-")
    axs[2].plot(odometry_time, yaw_deg, label='Actual', color="green", linestyle="-")
    
    # --- Final Formatting ---
    axs[0].set_title("Roll Over Time")
    axs[0].set_ylabel("Deflection\n[deg]")
    axs[1].set_title("Pitch Over Time")
    axs[1].set_ylabel("Deflection\n[deg]")
    axs[2].set_title("Yaw Over Time")
    axs[2].set_ylabel("Deflection\n[deg]")
    
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='medium')
    axs[-1].set_xlabel("Time [s]")


if __name__ == "__main__":
    start_time = 0
    end_time = 999999
    
    folder_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/"
    
    plot_overall(pd.read_csv(f"{folder_path}synced_all_data.csv"), start_time, end_time)
    # plot_signals(pd.read_csv(f"{folder_path}rcout_data.csv"), start_time, end_time)
    # # plot_rates(pd.read_csv(f"{folder_path}telem_data.csv"), start_time, end_time)
    # plot_rates(pd.read_csv(f"{folder_path}imu_data.csv"), start_time, end_time)
    # plot_raw_rates(pd.read_csv(f"{folder_path}imu_raw_data.csv"), start_time, end_time)
    # plot_angles(pd.read_csv(f"{folder_path}odometry_data.csv"), start_time, end_time)
    # plot_energy(pd.read_csv(f"{folder_path}synced_all_data.csv"), start_time, end_time)
    # plot_trajectory(pd.read_csv(f"{folder_path}trajectory_data.csv"), start_time, end_time)
    # plot_input_performance(pd.read_csv(f"{folder_path}trajectory_data.csv"), pd.read_csv(f"{folder_path}odometry_data.csv"), start_time, end_time)

    plt.show()
