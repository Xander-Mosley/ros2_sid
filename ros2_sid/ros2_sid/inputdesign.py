#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inputdesign.py — Importable signal generation functions.

This module provides functions for generating standard excitation signals used
in system identification and control experiments, including:
- frequency_sweep()
- multi_step()
- multi_sine()

All functions return NumPy arrays suitable for simulation or experiment playback.

Author: Xander D. Mosley (Python adaptation of Morelli's original MATLAB code)
Date: 2025-07-23
"""


import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional, Union
import warnings
from scipy.optimize import minimize


__all__ = ['frequency_sweep', 'multi_step', 'multi_sine']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def _rms(
        input_array: np.ndarray
        ) -> float | np.ndarray:
    """
    Compute the root-mean-square (RMS) value of a vector or matrix.
    
    The RMS value is a measure of the magnitude of a varying quantity and
    is defined as the square root of the arithmetic mean of the squares of the values.
    
    Parameters
    ----------
    input_array : np.ndarray
        Input vector or matrix of column vectors.
        - If 1D, computes the RMS of the vector.
        - If 2D, computes the RMS of each column independently.
    
    Returns
    -------
    rms_values : float or np.ndarray
        RMS value(s).
        - Scalar if input is 1D.
        - 1D array of RMS values for each column if input is 2D.
    
    Raises
    ------
    ValueError
        If input_array is not 1D or 2D.
    
    Notes
    -----
    - The RMS is commonly used to quantify the magnitude of a signal.
    - For a matrix input, the RMS is computed column-wise.
    
    Author
    ------
    Eugene A. Morelli (MATLAB original); Python adaptation by Xander D. Mosley.
    
    History
    -------
    25 Feb 1998 - Created and debugged, EAM.
    23 Jul 2025 - Adapted to Python, XDM.
    """
    input_array = np.asarray(input_array)

    if input_array.ndim not in (1, 2):
        raise ValueError("Input must be a 1D or 2D array.")

    return np.sqrt(np.mean(input_array ** 2, axis=0))


def _peakfactor(
        input_array: np.ndarray
        ) -> float | np.ndarray:
    """
    Compute the relative peak factor of a time series or multiple time series.
    
    The relative peak factor is a measure of signal "peakiness," defined as:
        peak_factor = (max(y) - min(y)) / (2 * sqrt(2) * rms(y))
    where 'y' is the input signal. For a pure sinusoid, the relative peak factor equals 1.
    
    Parameters
    ----------
    input_array : np.ndarray
        Vector or matrix of column vector time histories.
        - 1D array represents a single time history.
        - 2D array represents multiple time histories in its columns.
    
    Returns
    -------
    peak_factor : float or np.ndarray
        Relative peak factor(s).
        - Scalar if input is 1D.
        - 1D array of relative peak factors for each column if input is 2D.
    
    Raises
    ------
    ValueError
        If input_array is not 1D or 2D.
    
    See Also
    --------
    _rms : Compute the root mean square of the input signal.
        
    Notes
    -----
    - The relative peak factor quantifies the ratio between the peak-to-peak amplitude
      and the root-mean-square amplitude of the signal.
    - It is commonly used in signal processing to assess waveform characteristics.
    - A lower peak factor indicates a less "peaky" and more uniform signal amplitude.
    
    Author
    ------
    Eugene A. Morelli (MATLAB original); Python adaptation by Xander D. Mosley.
    
    History
    -------
    18 Mar 2003 - Created and debugged, EAM.
    23 Jul 2025 - Adapted to Python, XDM.
    """
    input_array = np.asarray(input_array)

    if input_array.ndim not in (1, 2):
        raise ValueError("Input must be a 1D or 2D array.")

    rms_vals = _rms(input_array)
    peak_to_peak = np.ptp(input_array, axis=0)
    return peak_to_peak / (2 * np.sqrt(2) * rms_vals)


def _peakfactorcost(
        phases: np.ndarray,
        frequencies: np.ndarray,
        powers: np.ndarray,
        time: np.ndarray
        ) -> tuple[float | np.ndarray, NDArray[np.float64] | Any]:
    """
    Compute the relative peak factor cost for optimizing multisine inputs.
    
    This function evaluates the relative peak factor of a multi-frequency cosine
    signal with specified phases, frequencies, and power spectrum, serving as the
    objective function for phase optimization to minimize signal peakiness.
    
    Parameters
    ----------
    phases : np.ndarray
        Vector of phase angles for each frequency component (radians).
    frequencies : np.ndarray
        Frequency components (Hz).
    powers : np.ndarray
        Relative power for each frequency component; elements should sum to 1.
    time : np.ndarray
        Time vector over which the signal is evaluated.
    
    Returns
    -------
    cost : float
        Relative peak factor of the composite signal.
    signals : np.ndarray
        Time-domain signal composed of weighted cosine components.
    
    See Also
    --------
    _peakfactor : Function to compute the relative peak factor metric.
    
    Notes
    -----
    - The peak factor is a measure of signal peakiness relative to its RMS level.
    - Used primarily in phase optimization to produce multisine inputs with minimized peak factors.
    - This function assumes powers sum to unity to properly weight components.
    
    Author
    ------
    Eugene A. Morelli (MATLAB original); Python adaptation by Xander D. Mosley.
    
    History
    -------
    24 Nov 2002 - Created and debugged, EAM.
    25 May 2004 - Included arbitrary power spectrum, EAM.
    23 Jul 2025 - Adapted to Python, XDM.
    """
    angular_frequencies = 2 * np.pi * frequencies

    cos_matrix = np.cos(np.outer(time, angular_frequencies) + phases)
    signals = cos_matrix @ np.sqrt(powers)

    cost = _peakfactor(signals)
    return cost, signals


def frequency_sweep(
        amplitude: float,
        minimum_frequency: float,
        maximum_frequency: float,
        time_step: float,
        final_time: float,
        function_type: Optional[str] = 'linear',
        noise_amplitude: Optional[float] = 0
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a frequency sweep signal using linear or logarithmic frequency variation.
    
    This function generates a frequency sweep sinusoidal signal of specified amplitude 
    that covers frequencies from 'minimum_frequency' to 'maximum_frequency' over the 
    duration 'final_time' with sampling interval 'time_step'. The sweep can be linear 
    or logarithmic in frequency progression. Optional Gaussian noise can be added.
    
    Parameters
    ----------
    amplitude : float
        Amplitude of the output sinusoidal sweep signal.
    minimum_frequency : float
        Starting frequency in Hertz (Hz).
    maximum_frequency : float
        Ending frequency in Hertz (Hz).
    time_step : float
        Sampling interval in seconds.
    final_time : float
        Total duration of the sweep signal in seconds.
    function_type : str, optional
        Sweep type, either 'linear' or 'logarithmic' (default is 'linear').
    noise_amplitude : float, optional
        Amplitude of added Gaussian noise (default is 0, no noise).
    
    Returns
    -------
    time : np.ndarray
        Time vector corresponding to the generated signal samples.
    signal : np.ndarray
        Generated frequency sweep signal with specified amplitude and noise.
    
    Raises
    ------
    ValueError
        If 'function_type' is not provided.
    ValueError
        If 'function_type' is not 'linear' or 'logarithmic'.
    
    Notes
    -----
    - Frequencies vary continuously from minimum_frequency to maximum_frequency.
    - Logarithmic sweep uses exponential frequency increase controlled by constants.
    - The output signal is normalized to the specified amplitude after noise addition.
    
    Author
    ------
    Eugene A. Morelli (MATLAB original); Python adaptation by Xander D. Mosley.
    
    History
    -------
    12 Sep 1997 - Created and debugged, EAM.
    19 Nov 2005 - Changed arguments of the sine functions, EAM.
    23 Jun 2025 - Adapted to Python, XDM.
    """
    time = np.arange(0, final_time + time_step, time_step)
    min_omega = 2 * np.pi * minimum_frequency
    max_omega = 2 * np.pi * maximum_frequency

    if function_type is None:
        raise ValueError("function_type must be provided")

    ftype = function_type.lower()

    if ftype == 'linear':
        K = time / final_time
    elif ftype == 'logarithmic':
        C1 = 4
        C2 = 0.0187
        K = C2 * (np.exp(C1 * (time / final_time)) - 1)
    else:
        raise ValueError("Invalid function_type: choose 'linear' or 'logarithmic'")
    
    omega = min_omega + K * (max_omega - min_omega)
    angle = np.zeros_like(omega)
    angle[1:] = np.cumsum(omega[1:] * time_step)
    
    signal = np.sin(angle)
    if noise_amplitude is not None and (noise_amplitude != 0.0):
        signal += noise_amplitude * np.random.randn(len(signal))
    signal = amplitude * (signal / max(abs(signal)))
    
    return time, signal


def multi_step(
        amplitude: float | np.ndarray | list,
        natural_frequency: float,
        pulses: np.ndarray | list,
        time_delay: float,
        time_step: float,
        final_time: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a multi-step alternating square wave signal composed of pulses with varying widths.
    
    This function creates an alternating square wave signal of total duration
    'final_time', sampled at intervals 'time_step'. Each pulse width is defined
    by an integer multiplier in 'pulses', scaled by the quarter period
    corresponding to the 'natural_frequency'. The signal starts after
    'time_delay' seconds, alternating sign (+/-) with each pulse, and each
    pulse has a specified amplitude.
    
    Parameters
    ----------
    amplitude : float or array_like
        Amplitude(s) of each pulse.
        If a scalar is provided, all pulses have the same amplitude.
        If array-like, each element specifies the amplitude of the corresponding pulse.
    natural_frequency : float
        Target natural frequency (Hz) defining the base pulse time as one quarter period (1 / (4 * natural_frequency)).
    pulses : array_like of int
        Vector of integer pulse widths.
        Each element defines the number of base pulse intervals in that pulse.
    time_delay : float
        Time delay in seconds before the square wave starts.
    time_step : float
        Sampling interval in seconds.
    final_time : float
        Total duration of the output signal in seconds.
    
    Returns
    -------
    time : np.ndarray
        Time vector corresponding to the generated signal samples.
    signal : np.ndarray
        Generated alternating square wave signal.
    
    Notes
    -----
    - The square wave alternates sign with each pulse.
    - Pulse widths are integer multiples of the base pulse time, defined by the natural frequency.
    - The output signal is zero before the time delay.
    
    Author
    ------
    Eugene A. Morelli (MATLAB original); Python adaptation by Xander D. Mosley.
    
    History
    -------
    01 May 1997 - Created and debugged, EAM.
    23 Jun 2025 - Adapted to Python, XDM.
    """    
    pulse_time = 1 / (4 * natural_frequency)
    time = np.arange(0, final_time + time_step, time_step)
    num_samples = len(time)
    signal = np.zeros(num_samples)
    
    pulses = np.abs(np.array(pulses))
    pulses = pulses[pulses != 0]
    pulses = np.round(pulses).astype(int)
    num_pulses = len(pulses)
    
    if np.isscalar(amplitude):
        amplitude_array = np.full(num_pulses, amplitude)
    else:
        amplitude_array = np.array(amplitude)
        if (len(amplitude_array) != num_pulses):
            amplitude_array = np.full(num_pulses, amplitude_array[0])
    
    n0 = int(round(time_delay / time_step))
    sign = 1
    
    for j in range(num_pulses):
        n1 = n0 + int(round(pulses[j] * pulse_time / time_step))
        if ((n0 + 1) < num_samples):
            signal[(n0 + 1):min(n1, num_samples)] = sign * amplitude_array[j]
        n0 = n1
        sign = -sign
    
    return time, signal


def multi_sine(
        amplitude: Union[float, list, np.ndarray],
        minimum_frequency: float,
        maximum_frequency: float,
        time_step: float,
        total_time: float,
        num_channels: Optional[int] = 1,
        user_frequencies: Optional[np.ndarray] = None,
        power_spectrum: Optional[np.ndarray] = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate orthogonal multisine excitation signals with minimized relative peak factor.
    
    This function generates 'num_channels' orthogonal multisine signals of
    duration 'total_time' and sampling interval 'time_step'. Each signal is
    composed of harmonic components with frequencies between 'minimum_frequency'
    and 'maximum_frequency', optionally shaped by a user-defined power spectrum
    and/or specific frequency assignments. The harmonic components are assigned
    to channels in a way that maximizes orthogonality in time and frequency,
    and the signal phases are optimized to minimize the relative peak factor of
    each channel.
    
    Parameters
    ----------
    amplitude : float or array_like
        Amplitude(s) of the output signals.
        If a scalar is provided, all signals will use that value.
        If array-like, must be of shape (number of channels,) and each value applies to a corresponding signal.
    minimum_frequency : float
        Minimum frequency (Hz) of the harmonic components.
    maximum_frequency : float
        Maximum frequency (Hz) of the harmonic components.
    time_step : float
        Sampling interval in seconds.
    total_time : float
        Total duration of the signal in seconds.
    num_channels : int, optional
        Number of output signals to generate (default is 1).
    user_frequencies : np.ndarray, optional
        Optional frequency matrix of shape (number of frequencies, number of channels).
        Each column defines the harmonic components (Hz) to use for that channel.
        If not provided, harmonic components are auto-assigned based on 'minimum_frequency', 'maximum_frequency', 'time_step', and 'total_time'.
    power_spectrum : np.ndarray, optional
        Optional power spectrum matrix of shape (number of frequencies, number of channels).
        Each column defines the relative power for each frequency of the corresponding signal.
        Columns must each sum to 1. If not provided, a flat (uniform) spectrum is used.
    
    Returns
    -------
    time_vector : np.ndarray
        Array of shape (number of samples,).
        The time values corresponding to the generated signal samples.
    signal : np.ndarray
        Array of shape (number of samples, number of channels).
        The generated orthogonal multisine signals.
    peak_factors : np.ndarray
        Array of shape (number of channels,).
        The relative peak factor for each signal.
    frequency_matrix : np.ndarray
        Array of shape (maximum number of frequencies, number of channels).
        Matrix of harmonic component frequencies used in each signal (Hz).
    num_frequencies_per_channel : np.ndarray
        Array of shape (number of channels,).
        Number of harmonic components used in each channel.
    phase_matrix : np.ndarray
        Array of shape (maximum number of frequencies, number of channels).
        Optimized phase angles (radians) for each frequency component in each signal.
    
    Raises
    ------
    ValueError
        If the minimum and maximum frequency bounds are invalid or reversed.
    ValueError
        If provided user frequency and power spectrum matrices have mismatched dimensions.
    
    See Also
    --------
    _peakfactor : Function to compute the relative peak factor.
    _peakfactorcost : Objective function used in peak factor minimization.
    
    Notes
    -----
    - Frequencies are selected based on integer multiples of 1 / total_time.
    - Frequency and power allocation ensure orthogonality between channels.
    - The signal's phases are optimized to reduce peak factor while respecting power spectrum shape.
    - Signals begin and end at (or near) zero to improve usability in physical system tests.
    
    Reference
    ---------
    Morelli, E.A., "Multiple Input Design for Real-Time Parameter Estimation in the Frequency Domain,"
    Paper REG-360, 13th IFAC Symposium on System Identification, Rotterdam, The Netherlands, August 2003.
    
    Author
    ------
    Eugene A. Morelli (MATLAB original); Python adaptation by Xander D. Mosley
    
    History
    -------
    24 Nov 2002 - Created and debugged, EAM.
    18 Mar 2003 - Incorporated peakfactor.m, made resulting inputs orthogonal
                  in time and frequency domains, added peak factor optimization
                  loop, interleaved frequencies for multiple inputs, EAM.
    27 Mar 2003 - Added fu input for arbitrary frequency selection, EAM.
    25 May 2004 - Added arbitrary power spectrum capability, EAM.
    23 Jul 2025 - Adapted to Python, XDM.
    """
    time = np.arange(0, (total_time + time_step), time_step)
    num_time_points = len(time)
    
    if num_channels is None:
        num_channels = 1
    else:
        num_channels = round(num_channels)
        
    amplitude_array = np.atleast_1d(amplitude)
    if (amplitude_array.size != num_channels):
        amplitude_array = np.full(num_channels, amplitude_array[0])
        
    signal = np.zeros((num_time_points, num_channels))
    
    if maximum_frequency <= minimum_frequency:
        raise ValueError('Illegal frequency bounds')
        
    minimum_frequency = max(minimum_frequency, (2 / total_time))
    maximum_frequency = min(maximum_frequency, (1 / (2 * time_step)))
    
    minimum_frequency = np.floor(minimum_frequency / (1 / total_time)) * (1 / total_time)
    maximum_frequency = np.ceil(maximum_frequency / (1 / total_time)) * (1 / total_time)
    
    if user_frequencies is None:
        base_frequencies = np.arange(minimum_frequency, (maximum_frequency + (1 / total_time)), (1 / total_time))
        num_base_frequencies = len(base_frequencies)
        frequency_matrix = np.zeros((num_base_frequencies, num_channels))
        num_frequencies_per_channel = np.zeros(num_channels, dtype=int)
        
        for channel_index in range(num_channels):
            channel_frequencies = base_frequencies[channel_index::num_channels]
            channel_frequencies = np.sort(channel_frequencies)
            num_frequencies_per_channel[channel_index] = len(channel_frequencies)
            frequency_matrix[:num_frequencies_per_channel[channel_index], channel_index] = channel_frequencies
            
    else:
        user_frequencies = np.atleast_2d(user_frequencies).T if user_frequencies.ndim == 1 else user_frequencies
        frequency_matrix = user_frequencies
        num_frequencies_per_channel = np.array([(frequency_matrix[:, channel] != 0).sum() for channel in range(frequency_matrix.shape[1])])
        
    peak_factors = np.zeros(num_channels)
    phase_matrix = np.zeros((np.max(num_frequencies_per_channel), num_channels))
    
    if power_spectrum is None:
        power_spectrum_array = np.full((np.max(num_frequencies_per_channel), num_channels), (1 / np.max(num_frequencies_per_channel)))
    else:
        power_spectrum_array = np.atleast_2d(power_spectrum)
        if power_spectrum_array.shape[0] != np.max(num_frequencies_per_channel):
            raise ValueError("Input size mismatch for user_frequencies and power_spectrum.")
        if power_spectrum_array.shape[1] != num_channels:
            power_spectrum_array = np.tile(power_spectrum_array[:, [0]], (1, num_channels))
            
    for channel_index in range(num_channels):
        channel_frequencies = frequency_matrix[:num_frequencies_per_channel[channel_index], channel_index]
        angular_frequencies = (2 * np.pi) * channel_frequencies
        phase_vector = np.zeros(num_frequencies_per_channel[channel_index])
        channel_power = power_spectrum_array[:num_frequencies_per_channel[channel_index], channel_index]
        
        for freq_index in range(1, num_frequencies_per_channel[channel_index]):
            time_until_prev = total_time * np.sum(channel_power[:freq_index])
            phase_vector[freq_index] = phase_vector[freq_index - 1] - ((2 * np.pi) * (channel_frequencies[freq_index] - channel_frequencies[freq_index - 1])) * time_until_prev
            
        max_iterations = 50
        peak_factor_goal = 1.01
        peak_factors[channel_index] = _peakfactorcost(phase_vector, channel_frequencies, channel_power, time)[0]
        
        # print(f"\n\n Starting phase optimization for input number {channel_index + 1} ...\n")
        
        for iteration in range(max_iterations):
            if peak_factors[channel_index] > peak_factor_goal:
                # print(f"\t Currently on iteration {iteration + 1} of {max_iterations} (max) ...\n")
                
                result = minimize(lambda phases: _peakfactorcost(phases, channel_frequencies, channel_power, time)[0], phase_vector, method='Nelder-Mead', options={'disp': False})
                optimized_phases = result.x
                phase_vector = optimized_phases
                
                phase_offset = np.zeros(num_frequencies_per_channel[channel_index])
                phase_increment = 0.0001 * np.ones(num_frequencies_per_channel[channel_index])
                
                initial_sign = np.sign(np.sum(np.sqrt(channel_power) * np.cos(phase_vector)))
                
                while np.sign(np.sum(np.sqrt(channel_power) * np.cos(phase_vector + phase_offset))) == initial_sign:
                    phase_offset += (phase_increment * angular_frequencies)
                    
                phase_vector += phase_offset
                peak_factors[channel_index] = _peakfactorcost(phase_vector, channel_frequencies, channel_power, time)[0]
                
        phase_vector = phase_vector - ((2 * np.pi) * np.floor(phase_vector / (2 * np.pi)))
        
        for phase_index in range(num_frequencies_per_channel[channel_index]):
            if abs(phase_vector[phase_index]) > np.pi:
                phase_vector[phase_index] -= (np.sign(phase_vector[phase_index]) * (2 * np.pi))
                
        phase_matrix[:num_frequencies_per_channel[channel_index], channel_index] = phase_vector
        
        for phase_index in range(num_frequencies_per_channel[channel_index]):
            signal[:, channel_index] += (np.sqrt(channel_power[phase_index]) * np.cos((angular_frequencies[phase_index] * time) + phase_vector[phase_index]))
            
        signal[:, channel_index] *= amplitude_array[channel_index]
        peak_factors[channel_index] = _peakfactor(signal[:, channel_index])
        
    max_num_frequencies = np.max(num_frequencies_per_channel)
    frequency_matrix = frequency_matrix[:max_num_frequencies, :]
    phase_matrix = phase_matrix[:max_num_frequencies, :]
    
    return time, signal, peak_factors, frequency_matrix, num_frequencies_per_channel, phase_matrix


# TODO: Add a ramp input.


if (__name__ == '__main__'):
    warnings.warn(
        "This script defines signal generation functions and is intended "
        "to be imported, not executed directly. "
        "\n\tImport this script using:\t"
        "from inputdesign import frequency_sweep, multi_step, multi_sine",
        UserWarning)