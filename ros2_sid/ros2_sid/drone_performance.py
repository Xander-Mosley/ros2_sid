#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from plotter_class import PlotFigure

ArrayLike = Union[float, Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]


__all__ = ['flight_data', 'flight_envelope', 'aerodynamic_performance', 'climb_performance', 'landing_performance', 'control_performance']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def _ensure_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
        
def _recenter_angles(data: ArrayLike) -> np.ndarray:
    x = _ensure_numpy(data)
    recentered = np.where(x >= 0, 180 - x, -180 - x)
    return recentered


def plot_overall(
        dataframes: list[pd.DataFrame],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        plot_labels: Optional[dict] = None
        ) -> PlotFigure:
    
    if not dataframes or len(dataframes) != 5:
        raise ValueError("Expected a list of 5 DataFrames (one per subplot).")
    if plot_labels is None:
        plot_labels = {}

    fig = PlotFigure(nrows=5, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Overall"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or 'timestamp' not in df.columns:
            raise ValueError("Invalid DataFrame provided.")
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]
        return df

    # ---------- Subplot 0: Control Commands ----------
    df = preprocess(dataframes[0])
    t = df["timestamp"]
    fig.define_subplot(0, title="Control Commands Over Time", ylabel="PWM Signal")
    fig.add_data(0, t, df["rcout_ch1"], label="Aileron", color="tab:blue")
    fig.add_data(0, t, df["rcout_ch2"], label="Elevator", color="tab:red")
    fig.add_data(0, t, df["rcout_ch4"], label="Rudder", color="tab:green")
    fig.add_data(0, t, df["rcout_ch3"], label="Thrust", color="black")

    # ---------- Subplot 1: Rates ----------
    df = preprocess(dataframes[1])
    t = df["timestamp"]
    fig.define_subplot(1, title="Rates Over Time", ylabel="Angular Velocity\n[rad/s]")
    fig.add_data(1, t, df["gx"], label="Roll", color="tab:blue")
    fig.add_data(1, t, df["gy"], label="Pitch", color="tab:red")
    fig.add_data(1, t, df["gz"], label="Yaw", color="tab:green")

    # ---------- Subplot 2: Attitude ----------
    df = preprocess(dataframes[2])
    t = df["timestamp"]
    fig.define_subplot(2, title="Attitude Over Time", ylabel="Roll & Pitch\n[deg]", y2label="Yaw\n[deg]")
    fig.add_data(2, t, df["roll_deg"], label="Roll", color="tab:blue")
    fig.add_data(2, t, df["pitch_deg"], label="Pitch", color="tab:red")
    fig.add_data_secondary_y(2, t, df["yaw_deg"], label="Yaw", color="tab:green", axis_color="green")

    # ---------- Subplot 3: Airspeed ----------
    df = preprocess(dataframes[3])
    t = df["timestamp"]
    fig.define_subplot(3, title="Airspeed Over Time", ylabel="Airspeed\n[m/s]")
    fig.add_data(3, t, df["airspeed"], color="black")

    # ---------- Subplot 4: Altitude ----------
    df = preprocess(dataframes[4])
    t = df["timestamp"]
    fig.define_subplot(4, title="Altitude Over Time", ylabel="Altitude\n[m]", xlabel="Time [s]")
    fig.add_data(4, t, df["altitude"], color="black")

    fig.set_all_legends(loc="upper right", fontsize="medium")
    fig.set_all_grids(True, alpha=0.5)

    return fig

def plot_controls(
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
    ail_def = dataframe["rcout_ch1"]
    elv_def = dataframe["rcout_ch2"]
    rud_def = dataframe["rcout_ch4"]
    thrust = dataframe["rcout_ch3"]

    fig = PlotFigure(nrows=4, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Control Commands"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Aileron Signal Over Time", ylabel="PWM Signal")
    fig.add_data(0, time, ail_def, color='tab:blue')

    fig.define_subplot(1, title="Elevator Signal Over Time", ylabel="PWM Signal")
    fig.add_data(1, time, elv_def, color='tab:red')

    fig.define_subplot(2, title="Rudder Signal Over Time", ylabel="PWM Signal")
    fig.add_data(2, time, rud_def, color='tab:green')

    fig.define_subplot(3, title="Thrust Signal Over Time", ylabel="PWM Signal", xlabel="Time [s]")
    fig.add_data(3, time, thrust, color='black')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_rates(
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
    rol_rate = dataframe["gx"]
    pit_rate = dataframe["gy"]
    yaw_rate = dataframe["gz"]

    fig = PlotFigure(nrows=3, ncols=1, figsize=(12, 6))
    base_title = "Flight Performance - Rates"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Roll Rate Over Time", ylabel="Angular Velocity\n[rad/s]")
    fig.add_data(0, time, rol_rate, color='tab:blue')

    fig.define_subplot(1, title="Pitch Rate Over Time", ylabel="Angular Velocity\n[rad/s]")
    fig.add_data(1, time, pit_rate, color='tab:red')

    fig.define_subplot(2, title="Yaw Rate Over Time", ylabel="Angular Velocity\n[rad/s]", xlabel="Time [s]")
    fig.add_data(2, time, yaw_rate, color='tab:green')
    
    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig
def plot_raw_rates(
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
    rol_rate = dataframe["gx_raw"]
    pit_rate = dataframe["gy_raw"]
    yaw_rate = dataframe["gz_raw"]

    fig = PlotFigure(nrows=3, ncols=1, figsize=(12, 6))
    base_title = "Flight Performance - Raw Rates"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Raw Roll Rate Over Time", ylabel="Angular Velocity\n[rad/s]")
    fig.add_data(0, time, rol_rate, color='tab:blue')

    fig.define_subplot(1, title="Raw Pitch Rate Over Time", ylabel="Angular Velocity\n[rad/s]")
    fig.add_data(1, time, pit_rate, color='tab:red')

    fig.define_subplot(2, title="Raw Yaw Rate Over Time", ylabel="Angular Velocity\n[rad/s]", xlabel="Time [s]")
    fig.add_data(2, time, yaw_rate, color='tab:green')
    
    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_attitude(
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
    rol_deg = dataframe["roll_deg"]
    pit_deg = dataframe["pitch_deg"]
    yaw_deg = dataframe["yaw_deg"]
    # yaw_deg = _recenter_angles(dataframe["yaw_deg"])

    fig = PlotFigure(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Attitude"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Roll Over Time", ylabel="Angle\n[deg]")
    fig.add_data(0, time, rol_deg, color='tab:blue')

    fig.define_subplot(1, title="Pitch Over Time", ylabel="Angle\n[deg]")
    fig.add_data(1, time, pit_deg, color='tab:red')

    fig.define_subplot(2, title="Yaw Over Time", ylabel="Angle\n[deg]", xlabel="Time [s]")
    fig.add_data(2, time, yaw_deg, color='tab:green')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_energy(
        dataframes: list[pd.DataFrame],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        plot_labels: Optional[dict] = None
        ) -> PlotFigure:
    
    if not dataframes or len(dataframes) != 2:
        raise ValueError("Expected a list of 2 DataFrames (one per subplot).")
    if plot_labels is None:
        plot_labels = {}

    fig = PlotFigure(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Energy"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or 'timestamp' not in df.columns:
            raise ValueError("Invalid DataFrame provided.")
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]
        return df
    
    # ---------- Subplot 0: Airspeed ----------
    df = preprocess(dataframes[0])
    t = df["timestamp"]
    fig.define_subplot(0, title="Airspeed Over Time", ylabel="Airspeed\n[m/s]")
    fig.add_data(0, t, df["airspeed"], color="black")

    # ---------- Subplot 1: Altitude ----------
    df = preprocess(dataframes[1])
    t = df["timestamp"]
    fig.define_subplot(1, title="Altitude Over Time", ylabel="Altitude\n[m]", xlabel="Time [s]")
    fig.add_data(1, t, df["altitude"], color="black")

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig


def plot_trajectory(
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
    roll_cmd = dataframe["roll_cmd"]
    pitch_cmd = dataframe["pitch_cmd"]
    yaw_cmd = dataframe["yaw_cmd"]

    fig = PlotFigure(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Trajectory Commands"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Roll Trajectory Over Time", ylabel="Angle\n[deg]")
    fig.add_data(0, time, roll_cmd, color='tab:blue')

    fig.define_subplot(1, title="Pitch Trajectory Over Time", ylabel="Angle\n[deg]")
    fig.add_data(1, time, pitch_cmd, color='tab:red')

    fig.define_subplot(2, title="Yaw Trajectory Over Time", ylabel="Angle\n[deg]", xlabel="Time [s]")
    fig.add_data(2, time, yaw_cmd, color='tab:green')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig
    
def plot_trajectory_delay(
        dataframes: list[pd.DataFrame],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        plot_labels: Optional[dict] = None,
        time_offset: float = 0.0
        ) -> PlotFigure:
    
    if not dataframes or len(dataframes) != 2:
        raise ValueError("Expected a list of 2 DataFrames (one per subplot).")
    if plot_labels is None:
        plot_labels = {}

    fig = PlotFigure(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Flight Performance - Trajectory Delay"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or 'timestamp' not in df.columns:
            raise ValueError("Invalid DataFrame provided.")
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]
        return df

    # ---------- Data Sources ----------
    trajectory_df = preprocess(dataframes[0])
    odometry_df = preprocess(dataframes[1])

    trajectory_time = trajectory_df["timestamp"]
    rol_cmd = trajectory_df["roll_cmd"]
    pit_cmd = trajectory_df["pitch_cmd"]
    yaw_cmd = trajectory_df["yaw_cmd"]
    
    odometry_time = odometry_df["timestamp"] - time_offset
    rol_deg = odometry_df["roll_deg"]
    pit_deg = odometry_df["pitch_deg"]
    yaw_deg = odometry_df["yaw_deg"]
    # yaw_deg = _recenter_angles(odometry_df["yaw_deg"])

    fig.define_subplot(0, title="Roll Over Time", ylabel="Angle\n[deg]")
    fig.add_data(0, trajectory_time, rol_cmd, label="Command", color='black', linestyle="--")
    fig.add_scatter(0, odometry_time, rol_deg, label="Response", color='tab:blue')

    fig.define_subplot(1, title="Pitch Over Time", ylabel="Angle\n[deg]")
    fig.add_data(1, trajectory_time, pit_cmd, label="Command", color='black', linestyle="--")
    fig.add_scatter(1, odometry_time, pit_deg, label="Response", color='tab:red')

    fig.define_subplot(2, title="Yaw Over Time", ylabel="Command Angle\n[deg]", y2label="Response Angle\n[deg]", xlabel="Time [s]")
    fig.add_data(2, trajectory_time, yaw_cmd, color='black', linestyle="--")
    fig.add_scatter_secondary_y(2, odometry_time, yaw_deg, color='tab:green', axis_color='green')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig


def flight_data(folder_path, start_time, end_time):
    plot_overall(
        [pd.read_csv(f"{folder_path}rcout_data.csv"),
         pd.read_csv(f"{folder_path}imu_data.csv"),
         pd.read_csv(f"{folder_path}odometry_data.csv"),
         pd.read_csv(f"{folder_path}odometry_data.csv"),
         pd.read_csv(f"{folder_path}altitude_data.csv")
         ], start_time, end_time)
    # plot_controls(pd.read_csv(f"{folder_path}rcout_data.csv"), start_time, end_time)
    # # plot_rates(pd.read_csv(f"{folder_path}telem_data.csv"), start_time, end_time)
    # plot_rates(pd.read_csv(f"{folder_path}imu_data.csv"), start_time, end_time)
    # plot_raw_rates(pd.read_csv(f"{folder_path}imu_raw_data.csv"), start_time, end_time)
    # plot_attitude(pd.read_csv(f"{folder_path}odometry_data.csv"), start_time, end_time)
    # plot_energy([
    #      pd.read_csv(f"{folder_path}odometry_data.csv"),
    #      pd.read_csv(f"{folder_path}altitude_data.csv")
    #      ], start_time, end_time)

def flight_envelope(folder_path, start_time, end_time):
    pass

def aerodynamic_performance(folder_path, start_time, end_time):
    pass

def climb_performance(folder_path, start_time, end_time):
    # Probably won't do these plots.
    pass

def landing_performance(folder_path, start_time, end_time):
    # Probably won't do these plots.
    pass

def control_performance(folder_path, start_time, end_time):
    # plot_trajectory(pd.read_csv(f"{folder_path}trajectory_data.csv"), start_time, end_time)
    plot_trajectory_delay([
        pd.read_csv(f"{folder_path}trajectory_data.csv"),
        pd.read_csv(f"{folder_path}odometry_data.csv")
        ], start_time, end_time, time_offset=0.0)
    # TODO: Make a lag plot for the command and response angle


def main():
    start_time = 0
    end_time = 999999

    folder_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/"
    # TODO: quickly program the drone performance plots
    flight_data(folder_path, start_time, end_time)
    flight_envelope(folder_path, start_time, end_time)
    aerodynamic_performance(folder_path, start_time, end_time)
    climb_performance(folder_path, start_time, end_time)
    landing_performance(folder_path, start_time, end_time)
    control_performance(folder_path, start_time, end_time)
    plt.show()


if __name__ == "__main__":
    main()