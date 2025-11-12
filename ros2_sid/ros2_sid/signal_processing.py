#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
signal_processing.py - 


Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 4 Nov 2025
"""


import numpy as np
import warnings


__all__ = ['linear_diff', 'poly_diff',
           'LowPassFilter', 'LowPassFilter_VDT',
           'ButterworthLowPass', 'ButterworthLowPass_VDT', 'ButterworthLowPass_VDT_2O']
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

    Author
    ------
    Xander D. Mosley

    History
    -------
    4 Nov 2025 - Created, XDM.
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

def poly_diff(
        time: np.ndarray,
        data: np.ndarray,
        polyorder: int = 3,
        eval_point: str = "center"
        ) -> float:
    """
    Estimates the first derivative of 'data' with respect to 'time' using a
    local least-squares polynomial fit, following the Savitzky-Golay
    differentiation method.

    A polynomial of degree 'polyorder' is fit to the provided samples, and the
    derivative is evaluated at a specified point within the window.

    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length > polyorder).
    data : np.ndarray
        1D array of data values, same length as 'time'.
    polyorder : int, optional
        Order of the polynomial to fit. Must be less than the number of samples.
        Default is 3.
    eval_point : {'start', 'center', 'end'}, optional
        Location within the window where the derivative is evaluated:
        - 'start': derivative at the first sample,
        - 'center': derivative at the midpoint sample,
        - 'end': derivative at the last sample.
        Default is 'center'.

    Returns
    -------
    float
        Estimated first derivative of 'data' with respect to 'time' at the
        specified evaluation point.

    Raises
    ------
    ValueError
        If 'time' and 'data' lengths differ.
    ValueError
        If 'polyorder' is not a valid integer.
    ValueError
        If 'eval_point' is not one of {'start', 'center', 'end'}.

    Author
    ------
    Xander D. Mosley

    History
    -------
    4 Nov 2025 - Created, XDM.
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
    deriv_coeffs = np.array([i * coeffs[i] for i in range(1, len(coeffs))])
    derivative = np.polyval(deriv_coeffs[::-1], 0.0)
    # NOTE: Using a 4th order polynomial starts to fit to noise.
    return np.float64(derivative)


class LowPassFilter:
    # First order low pass exponential moving average filter
    def __init__(
            self,
            cutoff_frequency: float,
            dt: float,
            initial_value: float = 0.0
            ) -> None:
        self.alpha = 1 - np.exp(-2 * np.pi * cutoff_frequency * dt)
        self.filtered_value = initial_value

    def update(self, new_value: float) -> float:
        self.filtered_value = (self.alpha * new_value) + ((1 - self.alpha) * self.filtered_value)
        return self.filtered_value

class LowPassFilter_VDT:
    # First order low pass exponential moving average filter with variable time steps
    def __init__(
            self,
            cutoff_frequency: float,
            num_dts: int = 1,
            initial_value: float = 0.0
            ) -> None:
        num_dts = num_dts
        self.smoothing_factor = 2 / (1 + num_dts)
        self.average_dt = 0.0
        self.fc = cutoff_frequency
        self.filtered_value = initial_value

    def update(self, new_value: float, dt: float) -> float:
        self.average_dt = (dt * self.smoothing_factor) + ((1 - self.smoothing_factor) * self.average_dt)
        alpha = 1 - np.exp(-2 * np.pi * self.fc * self.average_dt)
        self.filtered_value = (alpha * new_value) + ((1 - alpha) * self.filtered_value)
        return self.filtered_value


class ButterworthLowPass:
    # First order low pass butterworth filter
    def __init__(self, cutoff_frequency: float, dt: float):
        fc = cutoff_frequency
        self.y_filtered = 0.0
        self.x_previous = 0.0

        fc_safe = min(fc, 0.45 / dt)  # keep cutoff frequency below Nyquist (< 0.5/dt)
        if fc > (0.45 / dt):
            print("Warning: Cutoff frequency too high; clamped to 0.45 * fs.")
        gamma = np.tan(np.pi * fc_safe * dt)

        b0_prime = gamma
        b1_prime = b0_prime
        a1_prime = gamma - 1
        D = (gamma ** 2) + (np.sqrt(2) * gamma) + 1
        self.b0 = b0_prime / D
        self.b1 = b1_prime / D
        self.a1 = a1_prime / D

    def update(self, x_new: float):
        y_new = (self.b0 * x_new) + (self.b1 * self.x_previous) - (self.a1 * self.y_filtered)
        self.x_previous = x_new
        self.y_filtered = y_new

        return y_new

class ButterworthLowPass_VDT:
    # First order low pass butterworth filter with variable time steps
    def __init__(self, cutoff_frequency: float):
        self.fc = cutoff_frequency
        self.y_filtered = 0.0
        self.x_previous = 0.0

    def update(self, x_new: float, dt: float):
        fc_safe = min(self.fc, 0.45 / dt)  # keep cutoff frequency below Nyquist (< 0.5/dt)
        if self.fc > (0.45 / dt):
            print("Warning: Cutoff frequency too high; clamped to 0.45 * fs.")
        gamma = np.tan(np.pi * fc_safe * dt)

        b0_prime = gamma
        b1_prime = b0_prime
        a1_prime = gamma - 1
        D = (gamma ** 2) + (np.sqrt(2) * gamma) + 1
        b0 = b0_prime / D
        b1 = b1_prime / D
        a1 = a1_prime / D

        y_new = (b0 * x_new) + (b1 * self.x_previous) - (a1 * self.y_filtered)
        self.x_previous = x_new
        self.y_filtered = y_new

        return y_new

class ButterworthLowPass_VDT_2O:
    # Second order low pass butterworth filter with variable time steps
    def __init__(self, cutoff_frequency: float):
        self.fc = cutoff_frequency
        self.y_filtered = [0.0, 0.0]
        self.x_previous = [0.0, 0.0]

    def update(self, x_new: float, dt: float):
        fc_safe = min(self.fc, 0.45 / dt)  # keep cutoff frequency below Nyquist (< 0.5/dt)
        if self.fc > (0.45 / dt):
            print("Warning: Cutoff frequency too high; clamped to 0.45 * fs.")
        gamma = np.tan(np.pi * fc_safe * dt)

        b0_prime = gamma ** 2
        b1_prime = 2 * b0_prime
        b2_prime = b0_prime
        a1_prime = 2 * ((gamma ** 2) - 1)
        a2_prime = (gamma ** 2) - (np.sqrt(2) * gamma) + 1
        D = (gamma ** 2) + (np.sqrt(2) * gamma) + 1
        b0 = b0_prime / D
        b1 = b1_prime / D
        b2 = b2_prime / D
        a1 = a1_prime / D
        a2 = a2_prime / D

        y_new = (b0 * x_new) + (b1 * self.x_previous[0]) + (b2 * self.x_previous[1]) - (a1 * self.y_filtered[0]) - (a2 * self.y_filtered[1])
        self.x_previous[1] = self.x_previous[0]
        self.x_previous[0] = x_new
        self.y_filtered[1] = self.y_filtered[0]
        self.y_filtered[0] = y_new

        return y_new


if (__name__ == '__main__'):
    warnings.warn(
        "This script defines several functions and classes for"
        " signal processing, such as filtering and differentiating."
        "It is intented to be imported, not executed directly."
        "\n\tImport functions and class structures from this script using:\t"
        "from signal_processing import linear_diff, LowPassFilter, ButterworthLowPass_VDT"
        "\nMore functions and class structures are available within this script"
        " than the ones shown for example.",
        UserWarning)

