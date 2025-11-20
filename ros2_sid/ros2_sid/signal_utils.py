#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
signal_utils.py - Signal processing and analysis utilities.

Description
-----------
This script provides functions and tools for analyzing time-series data, 
specifically focused on derivative computation, low-pass filtering, and 
signal characterization in both the time and frequency domains. 

Features
--------
- Compute rolling derivatives using local linear or polynomial fitting.
- Apply low-pass filtering using multiple methods, including:
  * Fixed timestep filters (LPF, Butterworth)
  * Variable timestep filters (LPF_VDT, Butterworth_VDT)
  * Higher order filters
- Perform FFT-based frequency analysis, including magnitude and phase spectra.
- Generate time-domain, frequency-domain, and Bode plots using the PlotFigure class.
- Compute basic time statistics (min, max, mean, std) and sampling rate.

Modules and Classes
-------------------
- 'rolling_diff' : Compute rolling derivatives of a signal.
- 'apply_filter' : Apply a variety of low-pass filters to a signal.
- 'time_statistics' : Print basic timing and sampling rate statistics.
- '_compute_fft' : Compute FFT of one or more signals (supports custom frequency grids).
- 'signal_analysis' : Generate time-domain and frequency-domain plots.
- '_analyze_signal' : High-level function to filter, differentiate, and analyze a signal.

Custom Dependencies
------------
- plotter_class.py
- signal_processing.py

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 30 Oct 2025
"""


from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotter_class import PlotFigure
from signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT, ButterworthLowPass_VDT_2O
    )


__all__ = ['rolling_diff', 'apply_filter', 'time_statistics', 'plot_analysis']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def rolling_diff(
    time: np.ndarray,
    data: np.ndarray,
    method: str = "linear",
    window_size: int = 5
    ) -> np.ndarray:
    """
    Compute local derivatives over a rolling window
    using a specified differentiation method.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values.
    data : np.ndarray
        1D array of data values.
    window_size : int, optional
        Number of samples in each local window (default 5).
    method : {'linear', 'poly'}, optional
        Differentiation method to use:
        - 'linear' → local linear least-squares fit
        - 'poly' → Savitzky-Golay (polynomial fit)
        
    Returns
    -------
    np.ndarray
        Array of derivative estimates, length = len(data).

    Author
    ------
    Xander D. Mosley

    History
    -------
    30 Oct 2025 - Created, XDM.
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    
    if len(time) != len(data):
        raise ValueError("Arguments 'time' and 'data' must have the same length.")
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise ValueError("Argument 'window_size' must be an integer.")
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if len(time) < window_size:
        raise ValueError("Data length must exceed window size.")
    
    derivatives = np.full(len(time), 0.0, dtype=float)

    if method == "linear":
        for i in range(window_size, len(data) + 1):
            t_window = time[i - window_size : i]
            x_window = data[i - window_size : i]
            derivatives[i-1] = linear_diff(t_window, x_window)

    elif method == "poly":
        for i in range(window_size, len(data) + 1):
            t_window = time[i - window_size : i]
            x_window = data[i - window_size : i]
            derivatives[i-1] = poly_diff(t_window, x_window)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear' or 'poly'.")

    return derivatives

def apply_filter(
    time: np.ndarray,
    data: np.ndarray,
    filter_type: Literal[
        "LPF", "LPF_VDT",
        "Butter1", "Butter1_VDT", "Butter2_VDT"
        ],
    cutoff_frequency: float,
    num_dts: int = 1
    ) -> np.ndarray:
    """
    Apply a selected low-pass filter to an entire dataset sequentially (real-time approximation).

    Parameters
    ----------
    time : np.ndarray
        1D array of time values.
    data : np.ndarray
        1D array of input signal values.
    filter_type : str
        Type of filter to apply. Options:
        - "LPF" → LowPassFilter (fixed timestep)
        - "LPF_VDT" → LowPassFilter_VDT (variable timestep)
        - "Butter1" → ButterworthLowPass (fixed timestep)
        - "Butter1_VDT" → ButterworthLowPass_VDT (variable timestep)
        - "Butter2_VDT" → ButterworthLowPass_VDT_2O (second-order, variable timestep)
    cutoff_frequency : float
        Desired cutoff frequency in Hz.
    num_dts : int, optional
        Smoothing parameter for variable timestep LPF_VDT, default 1.

    Returns
    -------
    np.ndarray
        Array of filtered signal values, same length as 'data'.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.

    Author
    ------
    Xander D. Mosley

    History
    -------
    12 Nov 2025 - Created, XDM.
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    
    if len(time) != len(data):
        raise ValueError("'time' and 'data' must have the same length.")
    
    filtered = np.zeros_like(data)

    if filter_type == "LPF":
        dt = time[1] - time[0]
        filt = LowPassFilter(cutoff_frequency=cutoff_frequency, dt=dt, initial_value=data[0])
        for i, val in enumerate(data):
            filtered[i] = filt.update(val)

    elif filter_type == "LPF_VDT":
        filt = LowPassFilter_VDT(cutoff_frequency=cutoff_frequency, num_dts=num_dts, initial_value=data[0])
        for i, val in enumerate(data):
            if i > 0:
                dt_i = time[i] - time[i-1]
                filtered[i] = filt.update(val, dt_i)

    elif filter_type == "Butter1":
        dt = time[1] - time[0]
        filt = ButterworthLowPass(cutoff_frequency=cutoff_frequency, dt=dt)
        for i, val in enumerate(data):
            filtered[i] = filt.update(val)

    elif filter_type == "Butter1_VDT":
        filt = ButterworthLowPass_VDT(cutoff_frequency=cutoff_frequency)
        for i, val in enumerate(data):
            if i > 0:
                dt_i = time[i] - time[i-1]
                filtered[i] = filt.update(val, dt_i)

    elif filter_type == "Butter2_VDT":
        filt = ButterworthLowPass_VDT_2O(cutoff_frequency=cutoff_frequency)
        for i, val in enumerate(data):
            if i > 0:
                dt_i = time[i] - time[i-1]
                filtered[i] = filt.update(val, dt_i)

    else:
        raise ValueError(f"Unknown filter type '{filter_type}'.")
    
    return filtered


def time_statistics(t):
    dt = np.diff(t)
    print("")
    print("Time Step\t\tSampling Rate")
    print("=========\t\t=============")
    print(f"Min: {round(np.min(dt), 4)} s\t\tMax: {round(np.max(1/dt), 2)} Hz")
    print(f"Max: {round(np.max(dt), 4)} s\t\tMin: {round(np.min(1/dt), 2)} Hz")
    print(f"Avg: {round(np.mean(dt), 4)} s\t\tAvg: {round(np.mean(1/dt), 2)} Hz")
    print(f"Std: {round(np.std(dt), 4)} s\t\tStd: {round(np.std(1/dt), 2)} Hz")
    print("")

def _compute_fft(t, *signals, f=None):
    t = np.asarray(t)
    signals = [np.asarray(sig) for sig in signals]
    if f is None:
        dt_mean = np.mean(np.diff(t))
        f_max = 0.5 / dt_mean
        n_freqs = len(t)
        f = np.linspace(0, f_max, n_freqs)
    ffts = []
    for sig in signals:
        Xf = np.array([np.sum(sig * np.exp(-2j * np.pi * freq * t)) for freq in f])
        ffts.append(Xf)
    return f, ffts

def plot_analysis(t, x, y):
    time_figure = PlotFigure("Signal Analysis - Time Domain")
    time_figure.define_subplot(0, ylabel="Amplitude", xlabel="Time [s]", grid=True)
    time_figure.add_scatter(0, t, x, label="Input", color="tab:blue")
    time_figure.add_data(0, t, y, label="Output", color="tab:orange")
    time_figure.set_all_legends()

    f, (X, Y) = _compute_fft(t, x, y)
    def to_dB(x):
        return 20 * np.log10(np.abs(x) + 1e-12)
    
    freq_figure = PlotFigure("Signal Analysis - Frequency Spectrum",  nrows=2, sharex=True)
    freq_figure.define_subplot(0, ylabel="Magnitude", grid=True)
    freq_figure.add_data(0, f, np.abs(X), label="Input", color="tab:blue")
    freq_figure.add_data(0, f, np.abs(Y), label="Output", color="tab:orange")
    freq_figure.define_subplot(1, ylabel="Magnitude [dB]", xlabel="Frequency [Hz]", grid=True)
    freq_figure.add_data(1, f, to_dB(np.abs(X)), label="Input", color="tab:blue")
    freq_figure.add_data(1, f, to_dB(np.abs(Y)), label="Output", color="tab:orange")
    freq_figure.set_all_legends()
    
    H = Y / X
    mag = to_dB(H)
    phase = np.angle(H, deg=True)
    bode_figure = PlotFigure("Signal Analysis - Bode Plot", nrows=2, sharex=True)
    bode_figure.define_subplot(0, ylabel="Magnitude [dB]", grid=True)
    bode_figure.set_log_scale(0, axis='x')
    bode_figure.add_data(0, f, mag, label="Magnitude", color="tab:blue")
    bode_figure.add_line(0, -3, orientation='h', color='tab:red', label='-3 dB')
    bode_figure.define_subplot(1, ylabel="Phase [deg]", xlabel="Frequency [Hz]", grid=True)
    bode_figure.set_log_scale(1, axis='x')
    bode_figure.add_data(1, f, phase, label="Phase")
    bode_figure.set_all_legends()

    # dt = np.mean(np.diff(t))
    # fs = 1 / dt
    # T = t[-1] - t[0]
    # f_min = 1 / T
    # f_max = fs / 2
    # valid = (f >= f_min) & (f <= f_max)
    # phase_rad = np.unwrap(np.angle(H))
    # df = np.gradient(f[valid])
    # group_delay_sec = np.full_like(f, np.nan)
    # group_delay_sec[valid] = -np.gradient(phase_rad[valid], df) / (2 * np.pi)
    # group_delay_samples = np.full_like(f, np.nan)
    # group_delay_samples[valid] = group_delay_sec[valid] * fs

    # delay_figure = PlotFigure("Signal Analysis - Phase Delay", nrows=2, sharex=True)
    # delay_figure.define_subplot(0, ylabel="Phase Delay [s]", grid=True)
    # delay_figure.add_scatter(0, f, group_delay_sec, color="tab:blue", label="Phase Delay (s)")
    # delay_figure.define_subplot(1, ylabel="Phase Delay [samples]", xlabel="Frequency [Hz]", grid=True)
    # delay_figure.add_scatter(1, f, group_delay_samples, color="tab:orange", label="Phase Delay (samples)")
    # delay_figure.set_all_legends()


def _analyze_signal(t, x):
    # TODO: Determine the ideal filter for input data.
    fx = apply_filter(t, x, 'Butter2_VDT', 1.54)
    xp = rolling_diff(t, fx, "poly")
    fxp = apply_filter(t, xp, 'Butter2_VDT', 1.54)

    time_statistics(t)
    # plot_analysis(t, x, fx)
    # plot_analysis(t, fx, xp)
    # plot_analysis(t, xp, fxp)
    plot_analysis(t, x, fxp)

    plt.show()


def main():
    # file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/saved_maneuver.csv"
    file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/imu_data.csv"
    # file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/imu_raw_data.csv"
    # file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/telem_data.csv"

    df = pd.read_csv(file_path)
    print("Columns in CSV:", df.columns.tolist())
    time = df['timestamp'].to_numpy()
    data = df['gx'].to_numpy()

    start, end = 0, 999999999
    time = time[start:end]
    data = data[start:end]
    # time = time[::4]
    # data = data[::4]

    _analyze_signal(time, data)


if __name__ == "__main__":
    main()