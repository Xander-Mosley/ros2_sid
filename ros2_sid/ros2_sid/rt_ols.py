#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rt_ols.py — structures for real-time ordinary-least-squares

Description
-----------
Real-time data storage and frequency-domain model structures for recursive
ordinary-least-squares (OLS) estimation in the frequency domain.

This module provides two primary classes:
- 'StoredData': A FIFO-style rolling data container for time-indexed signals.
- 'ModelStructure': A frequency-domain model class supporting exponential
  forgetting, complex exponential basis functions, and recursive parameter
  estimation.

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 3 Jul 2025
"""


import warnings
import weakref
from typing import Optional, Sequence, Union

import numpy as np


__all__ = ['CircularBuffer', 'RecursiveFourierTransform', 'RegressorData', 'ordinary_least_squares']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


class CircularBuffer:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._data = np.zeros(capacity, dtype=float)
        self._index = 0
        self._size = 0

    def add(self, value: float) -> None:
        self._data[self._index] = value
        self._index = (self._index + 1) % self._capacity
        self._size = min(self._size, self._capacity)
        self._size += 1

    @property
    def latest(self) -> float:
        if self._size == 0:
            raise IndexError("Buffer is empty")
        return self._data[self._index - 1]

    @property
    def oldest(self) -> float:
        if self._size == 0:
            raise IndexError("Buffer is empty")
        if self._size < self._capacity:
            # Not yet full: oldest is at index 0
            return self._data[0]
        # Buffer full: oldest is at the current index
        return self._data[self._index]

    @property
    def size(self) -> int:
        return self._size

    def get_all(self) -> np.ndarray:
        if self._size < self._capacity:
            return self._data[:self._size].copy()
        # Return in chronological order
        return np.concatenate((self._data[self._index:], self._data[:self._index]))

    def apply_to_all(self, func) -> None:
        self._data[:self._size] = func(self.get_all())
        self._index = 0
        
    def fill_all(self, value: float) -> None:
        self._data.fill(value)
        self._index = 0
        self._size = self._capacity


class RecursiveFourierTransform:
    default_eff: float = 0.999
    default_frequencies: np.ndarray = np.arange(0.1, 1.54, 0.04)

    def __init__(
            self,
            eff: Optional[float] = None,
            frequencies: Optional[np.ndarray] = None
            ) -> None:
        
        self._eff = self.default_eff if eff is None else eff
        self._frequencies = (
            self.default_frequencies.copy()
            if frequencies is None else frequencies.copy()
        )

        if not (0.0 <= self._eff <= 1.0):
            raise ValueError("eff should be in [0, 1]")
        if self._frequencies.ndim != 1:
            raise ValueError("frequencies must be a 1D array")
        if np.any(self._frequencies < 0):
            raise ValueError("frequencies must be non-negative")

        self._complex_products = np.ones(self._frequencies.size, dtype=complex)
        self._frequencydata = np.zeros(self._frequencies.size, dtype=complex)
        
        self._omega = -2j * self._frequencies * np.pi   # went from (-1j * f) to (-2j * pi * f) for the correct rotation
        self._phase_initialized = False

    def update_cp_time(self, current_time: float) -> None:
        self._complex_products = np.exp(self._omega * current_time)
        self._phase_initialized = True
    
    def update_cp_timestep(self, time_step: float) -> None:
        if not self._phase_initialized:
            raise RuntimeError("Call update_cp_time() before update_cp_timestep()")
        self._complex_products *= np.exp(self._omega * time_step)

    def update_spectrum(
            self,
            timedata: float,
            ) -> None:
        
        self._frequencydata = (
            self._eff * self._frequencydata +
            timedata * self._complex_products
            )
    
    @property
    def current_spectrum(self) -> np.ndarray:
        return self._frequencydata.copy()
    
    @property
    def eff(self) -> float:
        return self._eff

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies.copy()

    @classmethod
    def set_defaults(
            cls,
            *,
            eff: Optional[float] = None,
            frequencies: Optional[np.ndarray] = None
            ) -> None:
        
        if eff is not None:
            if not isinstance(eff, (int, float)):
                raise TypeError("eff must be a float")
            eff = float(eff)
            if not (0.0 <= eff <= 1.0):
                raise ValueError("default eff must be between 0.0 and 1.0")
            cls.default_eff = eff

        if frequencies is not None:
            if not isinstance(frequencies, np.ndarray):
                raise TypeError("frequencies must be a numpy array")
            if frequencies.ndim != 1:
                raise ValueError("frequencies must be a 1D array")
            if np.any(frequencies < 0):
                raise ValueError("frequencies must be non-negative")
            cls.default_frequencies = frequencies.copy()


class RegressorData:
    def __init__(
            self,
            delay: int = 0,
            eff: Optional[float] = None,
            frequencies: Optional[np.ndarray] = None
            ) -> None:
        
        self.timedata = CircularBuffer(capacity=delay+1)
        self.timedata.fill_all(0)
        self.spectrum = RecursiveFourierTransform(eff=eff, frequencies=frequencies)

    def update(self, new_value) -> None:
        self.timedata.add(new_value)
        self.spectrum.update_spectrum(self.timedata.oldest)


def ordinary_least_squares(
        measured_output: np.ndarray,
        regressors: np.ndarray
        ) -> np.ndarray:
    if measured_output.shape[0] != regressors.shape[0]:
        raise ValueError("Number of samples in measured_output and regressors must match")
    parameters = np.real(
        np.linalg.pinv(regressors.T @ regressors)
            @ (regressors.T @ measured_output)
        ).ravel()
    return parameters


if (__name__ == '__main__'):
    warnings.warn(
        "This script defines the structures necessary for real-time "
        "ordinary-least-squares; and is intended to be imported, not "
        "executed directly."
        "\n\tImport this script using:\t"
        "from rt_ols import StoredData, ModelStructure",
        UserWarning)