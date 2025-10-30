#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
maneuver_utils.py — Helpful functions for maneuvers.

This module provides functions for saving, visualizing, and analyzing
maneuver signals commonly used in simulation and control experiments. It
includes functions for:

- save_maneuver(): Save maneuver time-series data to CSV.
- plot_maneuver(): Plot rotational and translational maneuver signals.
- plot_maneuver_spectrum(): Plot single-sided magnitude spectra of maneuvers.

All functions operate on NumPy arrays with time in the first column and
signal channels in subsequent columns, supporting up to three rotational
and three translational channels.

Author: Xander D. Mosley
Date: 30 Oct 2025
"""


import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt

from inputdesign import frequency_sweep, multi_step, multi_sine


__all__ = ['save_maneuver', 'plot_maneuver', 'plot_maneuver_spectrum']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def save_maneuver(
        maneuver: np.ndarray,
        filename: Optional[str] = "saved_maneuver.csv"
        ) -> None:
    """
    Save a maneuver and its corresponding time vector to a CSV file in the same directory.

    Parameters
    ----------
    maneuver : np.ndarray
        2D array with shape (n_samples, n_channels), where the first column is the time vector.
    filename : str, optional
        Name of the output CSV file (default is 'saved_maneuver.csv').

    Notes
    -----
    - The maneuver should be stacked column-wise: [time, signal1, signal2, ...].
    - The CSV will be saved in a folder named 'maneuvers' in the same directory as this script.

    Author
    ------
    Xander D. Mosley

    History
    -------
    19 Aug 2025 - Created, XDM.
    """
    if maneuver.ndim != 2:
        raise ValueError("Maneuver must be a 2D numpy array with shape (n_samples, n_channels).")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(current_dir, "maneuvers")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, str(filename))
    
    num_channels = maneuver.shape[1] - 1
    header = ','.join(['time'] + [f'channel_{i+1}' for i in range(num_channels)])

    np.savetxt(filepath, maneuver, delimiter=',', header=header, comments='', fmt='%.6f')
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
        Array of shape (n_samples, n_channels), where:
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
    if maneuver.shape[1] > 7:
        raise ValueError("Input 'maneuver' can have at most six channels (plus one time column).")

    time = maneuver[:, 0]
    signal = maneuver[:, 1:]

    num_channels = signal.shape[1]
    rot_labels = ['Roll', 'Pitch', 'Yaw']
    trans_labels = ['X', 'Y', 'Z']  # TODO: Determine if adding translational maneuvers is possible.

    if num_channels <= 3:
        rot_signals = signal
        trans_signals = None
    else:
        rot_signals = signal[:, :3]
        trans_signals = signal[:, 3:num_channels]

    nrows = 2 if trans_signals is not None else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 6), sharex=True)

    if nrows == 1:
        axs = [axs]

    # --- Rotational Maneuver Plot ---
    for i in range(rot_signals.shape[1]):
        axs[0].plot(time, rot_signals[:, i], label=rot_labels[i])
    axs[0].set_title("Rotational Trajectory")
    axs[0].set_ylabel(f"Rotational\nAmplitude")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    # --- Translational Maneuver Plot (if present) ---
    if trans_signals is not None:
        for i in range(trans_signals.shape[1]):
            axs[1].plot(time, trans_signals[:, i], label=trans_labels[i])
        axs[1].set_title("Translational Trajectory")
        axs[1].set_ylabel("Translational\nAmplitude")
        axs[1].legend(loc="upper right")
        axs[1].grid(True)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Generated Maneuver", fontsize=12)


def plot_maneuver_spectrum(
    maneuver: np.ndarray
    ) -> None:
    """
    Plot the single-sided magnitude spectrum of maneuver signals.

    Parameters
    ----------
    maneuver : np.ndarray
        Array of shape (n_samples, n_channels), where:
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
        raise ValueError("Input 'maneuver' must be 2D with at least two columns (time + one channel).")
    if maneuver.shape[1] > 7:
        raise ValueError("Input 'maneuver' can have at most six channels (plus one time column).")

    time = maneuver[:, 0]
    signal = maneuver[:, 1:]
    num_channels = signal.shape[1]

    rot_labels = ['Roll', 'Pitch', 'Yaw']
    trans_labels = ['X', 'Y', 'Z']

    if num_channels <= 3:
        rot_signals = signal
        trans_signals = None
    else:
        rot_signals = signal[:, :3]
        trans_signals = signal[:, 3:num_channels]

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    N = len(time)
    freqs = np.fft.rfftfreq(N, d=dt)

    def amplitude_spectrum(sig):
        fft_vals = np.fft.rfft(sig, axis=0)
        amp = (2.0 / N) * np.abs(fft_vals)
        return amp

    nrows = 2 if trans_signals is not None else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 6), sharex=True)

    if nrows == 1:
        axs = [axs]

    # --- Rotational Spectrum ---
    rot_amp = amplitude_spectrum(rot_signals)
    for i in range(rot_signals.shape[1]):
        axs[0].semilogx(freqs, rot_amp[:, i], label=rot_labels[i])
    axs[0].set_title("Rotational Trajectory - Frequency Spectrum")
    axs[0].set_ylabel(f"Magnitude")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].legend()

    # --- Translational Spectrum (if present) ---
    if trans_signals is not None:
        trans_amp = amplitude_spectrum(trans_signals)
        for i in range(trans_signals.shape[1]):
            axs[1].semilogx(freqs, trans_amp[:, i], label=trans_labels[i])
        axs[1].set_title("Translational Trajectory - Frequency Spectrum")
        axs[1].set_ylabel("Magnitude")
        axs[1].grid(True, which="both", ls="--", alpha=0.5)
        axs[1].legend()

    axs[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Generated Maneuver - Frequency Spectrum", fontsize=12)


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