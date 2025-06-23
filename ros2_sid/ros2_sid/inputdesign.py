# Input Designs
# Functions to create various input signals.
# V1.0: Developed FrequencySweep and Multistep. Created an empty MultiSine function.
# Xander Mosley - 20250623105231

import numpy as np
from typing import Optional
import warnings


__all__ = ['FrequencySweep', 'MultiStep']


def FrequencySweep(
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


def MultiStep(
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


def MultiSine():
    None

    


if (__name__ == '__main__'):
    warnings.warn(
        "This script is not intended to be run as a standalone program."
        " It contains structures and functions to be imported and used in other scripts.",
        UserWarning)