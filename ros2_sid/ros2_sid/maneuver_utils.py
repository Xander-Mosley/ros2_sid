#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system

import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional, Union
import warnings
from scipy.optimize import minimize
import os
import csv

from inputdesign import frequency_sweep, multi_step, multi_sine


__all__ = ['save_maneuver', 'plot_maneuver', 'plot_maneuver_spectrum']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def save_maneuver(
        maneuver: np.ndarray,
        filename: Optional[str] = "saved_maneuver.csv"
        ) -> None:
    """
    Save a signal and its corresponding time vector to a CSV file in the same directory.

    Parameters
    ----------
    maneuver : np.ndarray
        2D array with shape (n_channels, n_samples), where the first row is the time vector.
    filename : str, optional
        Name of the output CSV file (default is 'saved_maneuver.csv').

    Notes
    -----
    - The maneuver should be stacked row-wise: [time; signal1; signal2; ...].
    - The CSV will be saved in the same directory as this script.

    Author
    ------
    Xander D. Mosley

    History
    -------
    19 Aug 2025 - Created, XDM.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(str(current_dir), "maneuvers", str(filename))
    
    num_channels = maneuver.shape[1] - 1
    header = ['time'] + [f'channel_{i+1}' for i in range(num_channels)]

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(maneuver)

    print(f"\nSignal saved to: {filepath}\n")


def plot_maneuver(
        maneuver: np.ndarray
        ) -> None:
    """
    Plot maneuver signals in grouped panels:
      - Top: rotational channels (Roll, Pitch, Yaw)
      - Bottom: translational channels (X, Y, Z), if present.

    Parameters
    ----------
    maneuver : np.ndarray
        Array of shape (N, M), where:
          - Column 0 is time [s]
          - Columns 1-3 are rotational channels
          - Columns 4-6 (optional) are translational channels

    Author
    ------
    Xander D. Mosley

    History
    -------
    30 Oct 2025 - Created, XDM.
    """
    if maneuver.ndim != 2 or maneuver.shape[1] < 2:
        raise ValueError("Input 'maneuver' must be a 2D array with at least two columns (time + one channel).")

    time = maneuver[:, 0]
    signal = maneuver[:, 1:]

    num_channels = signal.shape[1]
    rot_labels = ['Roll', 'Pitch', 'Yaw']
    trans_labels = ['X', 'Y', 'Z']

    # Split signals into rotational and translational components
    if num_channels <= 3:
        rot_signals = signal
        trans_signals = None
    else:
        rot_signals = signal[:, :3]
        trans_signals = signal[:, 3:num_channels]

    # Create figure
    nrows = 2 if trans_signals is not None else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 6), sharex=True)

    if nrows == 1:
        axs = [axs]

    # --- Rotational Maneuver Plot ---
    for i in range(rot_signals.shape[1]):
        axs[0].plot(time, rot_signals[:, i], label=rot_labels[i])
    axs[0].set_title("Rotational Maneuver Inputs")
    axs[0].set_ylabel(f"Rotation Signals")
    axs[0].set_xlabel("Time [s]")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    # --- Translational Maneuver Plot (if present) ---
    if trans_signals is not None:
        for i in range(trans_signals.shape[1]):
            axs[1].plot(time, trans_signals[:, i], label=trans_labels[i])
        axs[1].set_title("Translational Maneuver Inputs")
        axs[1].set_ylabel("Translation Signals")
        axs[1].set_xlabel("Time [s]")
        axs[1].legend(loc="upper right")
        axs[1].grid(True)

    fig.suptitle("Generated Maneuver Inputs", fontsize=12)


def plot_maneuver_spectrum(maneuver: np.ndarray) -> None:
    """
    Plot the single-sided amplitude spectrum of maneuver signals.

    Parameters
    ----------
    maneuver : np.ndarray
        Array of shape (N, M), where:
          - Column 0 is time [s]
          - Columns 1-3 are rotational channels [rad or deg]
          - Columns 4-6 (optional) are translational channels [m]

    Author
    ------
    Xander D. Mosley

    History
    -------
    30 Oct 2025 - Created, XDM.
    """
    if maneuver.ndim != 2 or maneuver.shape[1] < 2:
        raise ValueError("Input 'maneuver' must be 2D with at least two columns (time + one channel).")

    time = maneuver[:, 0]
    signal = maneuver[:, 1:]
    num_channels = signal.shape[1]

    rot_labels = ['Roll', 'Pitch', 'Yaw']
    trans_labels = ['X', 'Y', 'Z']

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    N = len(time)
    freqs = np.fft.rfftfreq(N, d=dt)

    # Split signals
    if num_channels <= 3:
        rot_signals = signal
        trans_signals = None
    else:
        rot_signals = signal[:, :3]
        trans_signals = signal[:, 3:num_channels]

    # Detect radians vs degrees
    if np.mean(np.abs(rot_signals)) < 1.5:
        rot_signals = np.rad2deg(rot_signals)
        rot_unit = "[deg]"
    else:
        rot_unit = "[rad]"

    # Helper to compute single-sided amplitude spectrum
    def amplitude_spectrum(sig):
        fft_vals = np.fft.rfft(sig, axis=0)
        amp = (2.0 / N) * np.abs(fft_vals)
        return amp

    # Compute amplitude spectra
    rot_amp = amplitude_spectrum(rot_signals)
    trans_amp = amplitude_spectrum(trans_signals) if trans_signals is not None else None

    # Determine subplot layout
    nrows = 2 if trans_signals is not None else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 6), sharex=True)

    if nrows == 1:
        axs = [axs]

    # --- Rotational Spectrum ---
    for i in range(rot_signals.shape[1]):
        axs[0].semilogx(freqs, rot_amp[:, i], label=rot_labels[i])
    axs[0].set_ylabel(f"Amplitude {rot_unit}")
    axs[0].set_title("Rotational Maneuver Frequency Spectrum")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].legend()

    # --- Translational Spectrum ---
    if trans_signals is not None:
        for i in range(trans_signals.shape[1]):
            axs[1].semilogx(freqs, trans_amp[:, i], label=trans_labels[i])
        axs[1].set_ylabel("Amplitude [m]")
        axs[1].set_title("Translational Maneuver Frequency Spectrum")
        axs[1].grid(True, which="both", ls="--", alpha=0.5)
        axs[1].legend()

    axs[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Maneuver Input Spectra", fontsize=12)


def _test_maneuver():
    # TODO: Change this description if the maneuvers expand to require more than RPY signals.
    # maneuvers must have the shape (N, 4) where the columns (in order) are:
    # time, roll signal, pitch signal, yaw signal; and the first time value must be zero
    # TODO: Justify why these inputs. Why this amplitude, these frequencies, and these time values?
    amplitude: float = np.deg2rad(7)
    minimum_frequency: float = 0.1
    maximum_frequency: float = 1.5
    time_step: float = 0.02
    final_time: float = 15.
    num_channels: int = 3
    # time, signal = frequency_sweep(amplitude, minimum_frequency, maximum_frequency, time_step, final_time)
    # time, signal = multi_step(amplitude, ((minimum_frequency + maximum_frequency) / 2), [3, 2, 1, 1], (5 * time_step), time_step, final_time)
    time, signal, *_ = multi_sine(amplitude, minimum_frequency, maximum_frequency, time_step, final_time, num_channels)
    empty = np.zeros_like(time)

    maneuver = np.column_stack((time, signal))

    save_maneuver(maneuver)
    plot_maneuver(maneuver)
    plot_maneuver_spectrum(maneuver)
    plt.show()


if (__name__ == '__main__'):
    _test_maneuver()