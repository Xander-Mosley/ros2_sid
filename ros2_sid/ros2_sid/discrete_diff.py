#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script description
"""

import numpy as np
from typing import Optional, Sequence, Union
import warnings
import weakref

__all__ = ['linear_diff', 'savitzky_golay_diff']
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
            if i < 10:
                print(i)
            t_window = time[i - window_size : i]
            x_window = data[i - window_size : i]
            derivatives[i-1] = linear_diff(t_window, x_window)

    elif method == "sg":
        for i in range(window_size, len(data) + 1):
            t_window = time[i - window_size : i]
            x_window = data[i - window_size : i]
            derivatives[i] = savitzky_golay_diff(t_window, x_window)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear' or 'sg'.")

    return derivatives


if (__name__ == '__main__'):
    file_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/saved_maneuver.csv"
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    t = data[:, 0]
    x = data[:, 1]
    rolling_diff(t, x)

