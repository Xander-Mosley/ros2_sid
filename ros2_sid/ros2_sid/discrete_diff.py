#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script description
"""

import numpy as np
from typing import Optional, Sequence, Union
import warnings
import weakref

__all__ = ['linear_diff', 'savitzky_golay_diff', 'rolling_diff', 'LowPassFilter', 'smooth_data_array', 'LowPassFilterVariableDT', 'smooth_data_with_timestamps']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def linear_diff(
        time: np.ndarray,
        data: np.ndarray
        ) -> float:
    """
    Estimates the derivative of 'data' with respect to 'time' using
    a least-squares linear regression - originally designed for use
    over six samples.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length >= 2).
    data : np.ndarray
        1D array of data values, same length as 'time'.
        
    Returns
    -------
    float
        Estimated derivative (slope) of data with respect to time.
    """
    
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()
    n_samples = time.size

    if time.size != data.size:
        raise ValueError("Arguments 'time' and 'data' must have the same length.")
    if n_samples < 2:
        raise ValueError("At least two points are required for differentiation")
        
    sum_xt = np.dot(data, time)
    sum_t = time.sum()
    sum_x = data.sum()
    sum_t2 = np.dot(time, time)
    
    denominator = (n_samples * sum_t2) - (sum_t ** 2)
    if (denominator == 0):
        raise ZeroDivisionError("Denominator in derivative computation is zero.")
        
    numerator = (n_samples * sum_xt) - (sum_x * sum_t)
        
    return numerator / denominator


def savitzky_golay_diff(
        time: np.ndarray,
        data: np.ndarray,
        polyorder: Optional[int] = 3,
        eval_point: Optional[str] = "center"
        ) -> float:
    """
    Estimates the derivative of 'data' with respect to 'time' using a
    least-squares cubic regression over several samples. Follows
    the Savitzky-Golay style for differentiation.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length >= 2).
    data : np.ndarray
        1D array of data values, same length as 'time'.
        
    Returns
    -------
    float
        Estimated derivative (slope) of data with respect to time.
    """
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()

    if time.size != data.size:
        raise ValueError("Arguments 'time' and 'data' must have the same length.")
    if not isinstance(polyorder, int) or isinstance(polyorder, bool):
        raise ValueError("Argument 'polyorder' must be an integer.")
    if time.size <= polyorder:
        raise ValueError("Number of samples must exceed polynomial order.")

    if eval_point == "center":
        eval_idx = len(time) // 2   # TODO: Ensure this is the center of five data points. Center of six data points?
    elif eval_point == "start":
        eval_idx = 0
    elif eval_point == "end":
        eval_idx = -1
    else:
        raise ValueError("eval_point must be 'start', 'center', or 'end'.")
    
    shifted_time = time - time[eval_idx]
    A = np.vander(shifted_time, N=polyorder + 1, increasing=True)

    coeffs, *_ = np.linalg.lstsq(A, data, rcond=None)
    deriv_coeffs = np.array([i * coeffs[i] for i in range(1, len(coeffs))]) #TODO: Check if this produces the same results.
    derivative = np.polyval(deriv_coeffs[::-1], 0.0)

    # Using a 4th order polynomial starts to fit to noise.
    # TODO: Check if results improve with a better input data smoother.
    
    return np.float64(derivative)


def rolling_diff(
    time: np.ndarray,
    data: np.ndarray,
    method: Optional[str] = "linear",
    window_size: Optional[int] = 6
) -> np.ndarray:
    """
    Compute local derivatives over a rolling window using a specified method.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values.
    data : np.ndarray
        1D array of data values.
    window_size : int, optional
        Number of samples in each local window (default 6).
    method : {'linear', 'sg'}, optional
        Differentiation method to use:
        - 'linear' → local linear least-squares fit
        - 'sg' → Savitzky-Golay (polynomial fit)
        
    Returns
    -------
    np.ndarray
        Array of derivative estimates, length = len(data).
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
    
    derivatives = np.full(len(time), np.nan)

    if method == "linear":
        for i in range(window_size, len(data) + 1):
            t_window = time[i - window_size : i]
            x_window = data[i - window_size : i]
            derivatives[i-1] = linear_diff(t_window, x_window)

    elif method == "sg":
        for i in range(window_size, len(data) + 1):
            t_window = time[i - window_size : i]
            x_window = data[i - window_size : i]
            derivatives[i-1] = savitzky_golay_diff(t_window, x_window)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear' or 'sg'.")

    return derivatives


class LowPassFilter:
    def __init__(self, cutoff_frequency, dt, initial_value=0.0):
        """
        Standard EMA low-pass filter for a single data stream.
        
        :param cutoff_frequency: Cutoff frequency in Hz
        :param dt: Time step in seconds
        :param initial_value: Starting value for the filter
        """
        self.alpha = 1 - np.exp(-2 * np.pi * cutoff_frequency * dt)
        self.filtered_value = initial_value

    def update(self, new_value):
        """
        Update the filter with a new value.
        
        :param new_value: New input value
        :return: Filtered output
        """
        self.filtered_value = (self.alpha * new_value) + ((1 - self.alpha) * self.filtered_value)
        return self.filtered_value

def smooth_data_array(data, cutoff_frequency=18, dt=0.02):
    """
    Smooth a 1D data array using an EMA low-pass filter.
    
    :param data: 1D array-like sequence of data points
    :param cutoff_frequency: Filter cutoff frequency in Hz
    :param dt: Time step between samples in seconds
    :return: Numpy array of filtered values
    """
    if len(data) == 0:
        return np.array([])

    # Initialize filter with the first value of the array
    lpf = LowPassFilter(cutoff_frequency, dt, initial_value=data[0])
    
    filtered = []
    for value in data:
        filtered.append(lpf.update(value))
    
    return np.array(filtered)

class LowPassFilterVariableDT:
    def __init__(self, cutoff_frequency, initial_value=0.0):
        """
        EMA low-pass filter that handles variable time steps.
        :param cutoff_frequency: Cutoff frequency in Hz
        :param initial_value: Starting value
        """
        num_data = 5
        self.smoofactor = 2 / (1 + num_data)
        self.average_dt = 0.1
        
        self.fc = cutoff_frequency
        self.filtered_value = initial_value

    def update(self, new_value, dt):
        """
        Update the filter with a new value and the timestep since last update.
        :param new_value: New input value
        :param dt: Time difference since previous sample
        :return: Filtered output
        """
        self.average_dt = (dt * self.smoofactor) + ((1 - self.smoofactor) * self.average_dt)
        alpha = 1 - np.exp(-2 * np.pi * self.fc * self.average_dt)

        # alpha = 1 - np.exp(-2 * np.pi * self.fc * dt)
        self.filtered_value = (alpha * new_value) + ((1 - alpha) * self.filtered_value)
        return self.filtered_value, alpha

def smooth_data_with_timestamps(data, timestamps, cutoff_frequency=18):
    """
    Smooth a 1D data array with variable time steps.
    
    :param data: 1D array of data points
    :param timestamps: 1D array of timestamps corresponding to each data point
    :param cutoff_frequency: Filter cutoff frequency in Hz
    :return: Numpy array of filtered values
    """
    if len(data) == 0:
        return np.array([])

    lpf = LowPassFilterVariableDT(cutoff_frequency, initial_value=data[0])
    filtered = [data[0]]
    alphas = [0.0]  # first alpha can be 0 (no filtering yet)

    for i in range(1, len(data)):
        dt = timestamps[i] - timestamps[i-1]
        filtered_value, alpha = lpf.update(data[i], dt)
        filtered.append(filtered_value)
        alphas.append(alpha)

    return np.array(filtered), np.array(alphas)

if (__name__ == '__main__'):
    file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/saved_maneuver.csv"
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    t = data[:, 0]
    x = data[:, 1]
    rolling_diff(t, x)

