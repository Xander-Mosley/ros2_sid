# Input Designs
# Functions to create various input signals.
# V1.0: Developed FrequencySweep and Multistep. Created an empty MultiSine function.
# Xander Mosley - 20250623105231
# V1.1: Fixed naming conventions for PEP8.
# Xander Mosley - 20250701143041

import numpy as np
from typing import Optional
import warnings


__all__ = ['frequency_sweep', 'multi_step']


def _rms(
        input_array: np.ndarray
        ) -> np.float64 | np.ndarray:
    """
    Computes the root-mean-square (RMS) value of a vector or matrix.
    
    Parameters
    ----------
    input_array : np.ndarray
        Input vector or 2D array of column vectors.

    Returns
    -------
    rms_values : np.float64 | np.ndarray
        If input is 1D, returns a scalar.
        If input is 2D, returns 1D array with RMS for each column.
    """
    input_array = np.asarray(input_array)
    
    if input_array.ndim == 1:
        return np.sqrt(np.mean(input_array ** 2))
    elif input_array.ndim == 2:
        n, m = input_array.shape
        rms_values = np.zeros(m)
        for j in range(m):
            rms_values[j] = np.sqrt(np.dot(input_array[:, j], input_array[:, j]) / n)
        return rms_values
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def _peakfactor(
        input_array: np.ndarray
        ) -> np.float64 | np.ndarray:
    """
    Computes the relative peak factor of a time series.

    Parameters
    ----------
    input_array : np.ndarray
        Input vector or 2D array of column vectors.

    Returns
    -------
    peak_factor : np.float64 | np.ndarray
        Relative peak factor.
        If input is 1D, returns a scalar.
        If input is 2D, returns a 1D array with peak factor for each column.
    """
    input_array = np.asarray(input_array)

    if input_array.ndim == 1:
        peak_to_peak = np.max(input_array) - np.min(input_array)
        return peak_to_peak / (2 * np.sqrt(2) * _rms(input_array))
    elif input_array.ndim == 2:
        peak_to_peak = np.max(input_array, axis=0) - np.min(input_array, axis=0)
        return peak_to_peak / (2 * np.sqrt(2) * _rms(input_array))
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def _peakfactorcost(
        phases: np.ndarray,
        frequencies: np.ndarray,
        powers: np.ndarray,
        time: np.ndarray
        ) -> tuple[np.float64, np.ndarray]:
    """
    Computes the relative peak factor cost for optimizing multi-sine inputs.

    Parameters
    ----------
    phases : np.ndarray
        Vector of phase angles (radians).
    frequencies : np.ndarray
        Frequency vector (Hz).
    powers : np.ndarray
        Power for each component. Sum should equal 1.
    time : np.ndarray
        Time vector.

    Returns
    -------
    cost : np.float64
        Relative peak factor cost.
    signals : np.ndarray
        Multi-frequency sum of cosine signals.
    """
    angular_frequencies = 2 * np.pi * frequencies

    signals = np.zeros(len(time))
    for j in range(len(frequencies)):
        signals += np.sqrt(powers[j]) * np.cos(angular_frequencies[j] * time + phases[j])

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
    Creates a frequency sweep for a given range using either a linear or logarithmic function.
    
    Parameters
    ----------
    amplitude : float
        Amplitude of the sinusoidal signal.
    minimum_frequency : float
        Minimum frequency in Hz.
    maximum_frequency : float
        Maximum frequency in Hz.
    time_step : float
        Time step in seconds.
    final_time : float
        Final time in seconds.
    function_type : str, optional
        The default is 'linear'.
        Function type utilized: 'linear' or 'logarithmic'.
    noise_amplitude : float, optional
        The default is 0.
        Amplitude of added noise.

    Raises
    ------
    ValueError
        Invalid function_type: choose 'linear' or 'logarithmic'

    Returns
    -------
    time : np.ndarray
        Time array of the signal.
    signal : np.ndarray
        Sweep signal.

    """
    
    time = np.arange(0, final_time + time_step, time_step)
    min_omega = 2 * np.pi * minimum_frequency
    max_omega = 2 * np.pi * maximum_frequency
    
    if (function_type.lower() == 'linear'):
        K = time / final_time
    elif (function_type.lower() == 'logarithmic'):
        C1 = 4
        C2 = 0.0187
        K = C2 * (np.exp(C1 * (time / final_time)) - 1)
    else:
        raise ValueError("Invalid function_type: choose 'linear' or 'logarithmic'")
    
    omega = min_omega + K * (max_omega - min_omega)
    angle = np.zeros_like(omega)
    angle[1:] = np.cumsum(omega[1:] * time_step)
    
    signal = np.sin(angle)
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
    Creates a multi-step alternating square wave signal.

    Parameters
    ----------
    amplitude : float | np.ndarray | list
        Amplitudes for every pulse or each pulse.
    natural_frequency : float
        The desired target frequency.
    pulses : np.ndarray | list
        List of integer pulse widths.
    time_delay : float
        Time delay before square wave starts in seconds.
    time_step : float
        Sampling interval in seconds.
    final_time : float
        Total time duration in seconds.

    Returns
    -------
    time : np.ndarray
        Time array of the signal.
    signal : np.ndarray
        Alternating square wave signal.

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
        amplitude = np.full(num_pulses, amplitude)
    else:
        amplitude = np.array(amplitude)
        if (len(amplitude) != num_pulses):
            amplitude = np.full(num_pulses, amplitude[0])
    
    n0 = int(round(time_delay / time_step))
    sign = 1
    
    for j in range(num_pulses):
        n1 = n0 + int(round(pulses[j] * pulse_time / time_step))
        if ((n0 + 1) < num_samples):
            signal[(n0 + 1):min(n1, num_samples)] = sign * amplitude[j]
        n0 = n1
        sign = -sign
    
    signal = signal[:num_samples]
    
    return time, signal


def multi_sine():
    None


if (__name__ == '__main__'):
    warnings.warn(
        "This script is not intended to be run as a standalone program."
        " It contains structures and functions to be imported and used in other scripts.",
        UserWarning)