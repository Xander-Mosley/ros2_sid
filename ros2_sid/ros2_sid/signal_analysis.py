#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
signal_analysis.py - Signal processing and analysis utilities.

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
from matplotlib import cm

from plotter_class import PlotFigure
from signal_processing import (
    linear_diff, poly_diff,
    LowPassFilter, LowPassFilter_VDT,
    ButterworthLowPass, ButterworthLowPass_VDT, ButterworthLowPass_VDT_2O
    )
from rt_ols import RecursiveFourierTransform


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

def compute_rft_spectrum(t, *signals, frequencies=None, eff=1.0):
    """
    Compute the Recursive Fourier Transform spectrum for one or more signals.

    Parameters
    ----------
    t : array_like
        1D array of timestamps (length N).
    *signals : array_like
        One or more signal arrays (length N each).
    frequencies : array_like, optional
        Frequencies (Hz) to compute the spectrum for. 
        If None, will default to 0.1 to 1.5 Hz in 0.04 Hz steps.
    eff : float, optional
        Exponential smoothing factor in [0,1].

    Returns
    -------
    f : np.ndarray
        Frequency array (Hz) used.
    spectra : list of np.ndarray
        List of complex spectra (one per signal).
    """
    t = np.asarray(t)
    signals = [np.asarray(sig) for sig in signals]

    if frequencies is None:
        frequencies = np.arange(0.1, 1.54, 0.04)
    frequencies = np.asarray(frequencies)

    spectra = []

    for sig in signals:
        if len(sig) != len(t):
            raise ValueError("Each signal must have the same length as timestamps")
        
        rft = RecursiveFourierTransform(eff=eff, frequencies=frequencies)
        rft.update_cp_time(t[0])
        rft.update_spectrum(sig[0])

        for n in range(1, len(sig)):
            dt = t[n] - t[n-1]
            if dt < 0:
                raise ValueError("Timestamps must be non-decreasing")
            rft.update_cp_timestep(dt)
            rft.update_spectrum(sig[n])

        spectra.append(rft.current_spectrum)

    return frequencies, spectra


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

def plot_model_spectrums(filepaths, rft_args=None, plot_labels=None):
    data = {}
    spectrums = {}

    df_z = pd.read_csv(filepaths["z"]["filepath"])
    data["z"] = {
        "time": df_z["timestamp"].to_numpy(),
        "data": df_z[filepaths["z"]["tag"]].to_numpy()
    }

    freqs_z, spec_z = compute_rft_spectrum(data["z"]["time"], data["z"]["data"])
    spectrums["z"] = {
        "frequencies": freqs_z,
        "spectrum": spec_z[0]
    }

    data["x"] = {}
    spectrums["x"] = {}

    for key, cfg in filepaths["x"].items():
        df = pd.read_csv(cfg["filepath"])
        x_data = df[cfg["tag"]].to_numpy()

        # Optional per-regressor preprocessing
        if key == "2":
            x_data = x_data - 1500

        data["x"][key] = {
            "time": df["timestamp"].to_numpy(),
            "data": x_data
        }

        freqs, spec = compute_rft_spectrum(data["x"][key]["time"], x_data)
        spectrums["x"][key] = {
            "frequencies": freqs,
            "spectrum": spec[0]
        }

    n_regressors = len(spectrums["x"])
    freq_figure = PlotFigure(
        "Spectrum Analysis - Model Spectrums",
        nrows=1 + n_regressors,
        sharex=True
    )

    z_label = plot_labels["z"]["name"] if plot_labels else "Measured Output"
    freq_figure.define_subplot(0, ylabel="Magnitude", grid=True)
    freq_figure.add_data(
        0,
        spectrums["z"]["frequencies"],
        np.abs(spectrums["z"]["spectrum"]),
        label=z_label
    )
    for i, key in enumerate(spectrums["x"], start=1):
        label = (
            plot_labels["x"][key]["name"]
            if plot_labels else f"Regressor {key}"
        )
        freq_figure.define_subplot(i, ylabel="Magnitude", xlabel="Freqyency [Hz]" if i == n_regressors else None, grid=True)
        freq_figure.add_data(
            i,
            spectrums["x"][key]["frequencies"],
            np.abs(spectrums["x"][key]["spectrum"]),
            label=label
        )
    freq_figure.set_all_legends()

    def to_dB(x):
        return 20 * np.log10(np.abs(x) + 1e-12)

    freqs_match = all(
        np.allclose(freqs_z, spectrums["x"][k]["frequencies"])
        for k in spectrums["x"]
    )

    if freqs_match:
        bode_figure = PlotFigure(
            "Spectrum Analysis - Bode Plot of Regressors",
            nrows=2,
            sharex=True
        )

        bode_figure.define_subplot(0, ylabel="Magnitude [dB]", grid=True)
        bode_figure.set_log_scale(0, axis='x')
        bode_figure.add_line(0, -3, orientation='h', color='tab:red', label='-3 dB')

        bode_figure.define_subplot(
            1,
            ylabel="Phase [deg]",
            xlabel="Frequency [Hz]",
            grid=True
        )
        bode_figure.set_log_scale(1, axis='x')

        for key in spectrums["x"]:
            label = (
                plot_labels["x"][key]["name"]
                if plot_labels else f"Regressor {key}"
            )

            ratio = spectrums["z"]["spectrum"] / spectrums["x"][key]["spectrum"]

            bode_figure.add_data(
                0, freqs_z, to_dB(ratio), label=label
            )
            bode_figure.add_data(
                1, freqs_z, np.angle(ratio, deg=True), label=label
            )

        bode_figure.set_all_legends()

    def get_rft_params(rft_args, group, key=None):
        if rft_args is None:
            return {"frequencies": None, "eff": None}

        if key is None:
            return rft_args.get(group, {})

        return rft_args.get(group, {}).get(key, {})
    
    z_rft = get_rft_params(rft_args, "z")

    plot_normalized_rft_over_time(
        data["z"]["time"],
        data["z"]["data"],
        frequencies=z_rft.get("frequencies"),
        eff=z_rft.get("eff"), # type: ignore
        subtitle=plot_labels["z"]["name"] # type: ignore
    )

    for key in data["x"]:
        x_rft = get_rft_params(rft_args, "x", key)

        plot_normalized_rft_over_time(
            data["x"][key]["time"],
            data["x"][key]["data"],
            frequencies=x_rft.get("frequencies"),
            eff=x_rft.get("eff"), # type: ignore
            subtitle=plot_labels["x"][key]["name"] # type: ignore
        )

def plot_normalized_rft_over_time(time, signal, frequencies=None, eff=0.999, subtitle=""):
    """
    Compute RFT at each time step, normalize magnitude spectra, and plot 3D surface highlighting peaks.

    Parameters
    ----------
    time : np.ndarray
        Array of timestamps.
    signal : np.ndarray
        Signal values at each timestamp.
    frequencies : np.ndarray, optional
        Frequencies to compute. Defaults to 0.1-1.5 Hz with 0.04 Hz step.
    eff : float
        RFT smoothing factor.
    """
    if frequencies is None:
        frequencies = np.arange(0.1, 1.54, 0.04)

    n_time = len(time)
    n_freq = len(frequencies)

    # Initialize array to store magnitude spectra over time
    mag_spectra = np.zeros((n_time, n_freq))

    # Initialize RFT
    rft = RecursiveFourierTransform(eff=eff, frequencies=frequencies)

    # Loop over time steps
    for i in range(n_time):
        if i == 0:
            rft.update_cp_time(time[i])
        else:
            dt = time[i] - time[i-1]
            if dt < 0:
                raise ValueError("Timestamps must be non-decreasing")
            rft.update_cp_timestep(dt)
        rft.update_spectrum(signal[i])
        mag_spectra[i, :] = np.abs(rft.current_spectrum)

    # --- Normalize magnitude spectra across each time step ---
    mag_spectra_norm = mag_spectra / np.max(mag_spectra)

    # max_freq_indicies = np.argmax(mag_spectra_norm, axis=1)
    # max_frequencies = frequencies[max_freq_indicies]

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(time, frequencies, mag_spectra_norm.T, shading='auto', cmap='viridis')
    # plt.plot(time, max_frequencies, color='white', lw=2, label='Max Frequency')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Normalized Magnitude')
    plt.title(f'Time-Resolved Normalized RFT Spectrum\n{subtitle}')
    # plt.legend()

def plot_timestep_distribution(file_directory, file_name):
    df = pd.read_csv(file_directory + file_name)

    df = df.sort_values("timestamp")
    df["dt"] = df["timestamp"].diff()
    dt = df["dt"].dropna()

    mean_dt = dt.mean()
    num_sigma = 2
    std_dt = dt.std()
    low_end = mean_dt - num_sigma * std_dt
    high_end = mean_dt + num_sigma * std_dt

    figtitle = f"Distribution of Time Step Sizes\nLog: {file_name}"
    dt_dist = PlotFigure(fig_title=figtitle)
    dt_dist.define_subplot(0, ylabel="Quantity", xlabel="Time Step, dt [s]")
    dt_dist.add_hist(0, dt, 74, edgecolor="black")
    dt_dist.add_line(0, low_end, 'v', label=f"Mean - {num_sigma}σ = {low_end:.3} s", color="grey", linestyle="--")
    dt_dist.add_line(0, mean_dt, 'v', label=f"Mean = {mean_dt:.3} s", color="red", linewidth=2)
    dt_dist.add_line(0, high_end, 'v', label=f"Mean + {num_sigma}σ = {high_end:.3} s", color="grey", linestyle="--")
    dt_dist.set_all_legends()

def plot_timestep_overtime(file_directory, file_name):
    df = pd.read_csv(file_directory + file_name)

    df = df.sort_values("timestamp")
    df["dt"] = df["timestamp"].diff()
    df["dt"][0] = 0

    figtitle = f"Time Step Sizes Over Time\nLog: {file_name}"
    dt_vt = PlotFigure(fig_title=figtitle)
    dt_vt.define_subplot(0, ylabel="Time Step, dt [s]", xlabel="Time [s]")
    dt_vt.add_data(0, df["timestamp"], df["dt"])
    dt_vt.set_all_legends()


def _analyze_input_signals():
    # file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/saved_maneuver.csv"
    file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/imu_data.csv"
    # file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/imu_raw_data.csv"
    # file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/telem_data.csv"

    df = pd.read_csv(file_path)
    print("Columns in CSV:", df.columns.tolist())
    t = df['timestamp'].to_numpy()
    x = df['gx'].to_numpy()

    start, end = 0, 999999999
    t = t[start:end]
    x = x[start:end]
    # t = t[::2]
    # x = x[::2]
    
    fx = apply_filter(t, x, 'Butter2_VDT', 1.54)
    xp = rolling_diff(t, fx, "poly")
    fxp = apply_filter(t, xp, 'Butter2_VDT', 1.54)

    time_statistics(t)
    plt.plot(t, fxp)
    # plot_analysis(t, x, fx)
    # plot_analysis(t, fx, xp)
    # plot_analysis(t, xp, fxp)
    # plot_analysis(t, x, fxp)

    plt.show()

def _analyze_regressor_spectrums():
    filepaths = {
        "z": {"tag": "gax",
              "filepath": "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/imu_diff_data.csv"},
        "x": {
            "1": {"tag": "gx", "filepath": "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/imu_data.csv"},
            # "2": {"tag": "rcout_ch1", "filepath": "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/rcout_data.csv"},
        }
    }

    rft_args = {
        "z": {
            "frequencies": None, "eff": 0.999
        },
        "x": {
            "1": {"frequencies": None, "eff": 0.999},
            "2": {"frequencies": None, "eff": 0.969},
        }
    }
    
    plot_labels = {
        "z": {"name": "Roll Angular Acceleration", "unit": "m/s²"},
        "x": {
            "1": {"name": "Roll Angular Rate", "unit": "rad/s"},
            "2": {"name": "Aileron Command", "unit": "µs"},
        }
    }

    plot_model_spectrums(filepaths=filepaths, rft_args=rft_args, plot_labels=plot_labels)

    plt.show()

def _analyze_time_steps():
    file_directory = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/"

    plot_timestep_distribution(file_directory=file_directory, file_name="imu_diff_data.csv")
    plot_timestep_distribution(file_directory=file_directory, file_name="imu_data.csv")
    # plot_timestep_distribution(file_directory=file_directory, file_name="rcout_data.csv")
    # plot_timestep_overtime(file_directory=file_directory, file_name="imu_diff_data.csv")
    # plot_timestep_overtime(file_directory=file_directory, file_name="imu_data.csv")
    # plot_timestep_overtime(file_directory=file_directory, file_name="rcout_data.csv")

    plt.show()




def main():
    _analyze_input_signals()
    # _analyze_regressor_spectrums()
    # _analyze_time_steps()

if __name__ == "__main__":
    main()