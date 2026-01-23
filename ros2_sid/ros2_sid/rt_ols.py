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


__all__ = ['StoredData', 'ModelStructure', 'CircularBuffer', 'RecursiveFourierTransform', 'RegressorData', 'ordinary_least_squares']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


class StoredData:
    """
    This class stores a 2D array of time-updating data. It supports
    FIFO-style updates where new rows push out the oldest ones.
    """
    def __init__(
            self,
            num_rows: int,
            num_cols: int
            ) -> None:
        """
        Initializes a zero-filled 2D array of shape (num_rows, num_cols).
        
        Parameters
        ----------
        num_rows : int
            Number of historical data entries to store.
        num_cols : int
            Number of channels (e.g., time, sensor1, sensor2, etc.).
        """
        
        self._data: np.ndarray = np.zeros((num_rows, num_cols), dtype=float)
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_rows={self.num_rows}, num_cols={self.num_cols})"
    
    def __str__(self) -> str:
        return f"\nNumber of Rows: {self.num_rows}\nNumber of Columns: {self.num_cols}\n"
    
    def update_data(
            self,
            newestdata
            ) -> None:
        """
        Shifts data down by one row and inserts newestdata at the first row.
        
        Parameters
        ----------
        newestdata : array-like, float, or int
            The latest data row. Must match the number of columns.
            
        Raises
        ------
        ValueError
            If the number of elements in newestdata does not match the number of columns.
        """
        
        newestdata = np.atleast_1d(newestdata).astype(float)
        if newestdata.size != self.num_cols:
            raise ValueError(
                f"Incorrect number of elements in new data: expected {self.num_cols}, got {newestdata.size}."
            )
        
        self._data = np.roll(self._data, shift=1, axis=0)
        self._data[0] = newestdata
        
    @property
    def num_rows(self) -> int:
        return self._data.shape[0]
    
    @property
    def num_cols(self) -> int:
        return self._data.shape[1]
    
    @property
    def latest(self) -> np.ndarray:
        return self._data[0]
    
    @property
    def data(self) -> np.ndarray:
        return self._data


class _ModelStructureMeta(type):
    """
    Metaclass for managing class-level properties in ModelStructure.
    Specifically handles the class-level exponential forgetting factor (eff),
    enforcing validation and encapsulation through a property interface.
    """
    @property
    def class_eff(cls) -> float:
        """
        Gets the class-wide exponential forgetting factor (eff).

        Returns
        -------
        float
            The current class-level exponential forgetting factor (eff).
        """
        return cls._class_eff
    
    @class_eff.setter
    def class_eff(cls, value: float) -> None:
        """
        Sets the class-wide exponential forgetting factor (eff), ensuring it is within (0.0 - 1.0).

        Parameters
        ----------
        value : float
            The new exponential forgetting factor value to apply at the class level.

        Raises
        ------
        ValueError
            If the given value is not in the range (0.0 - 1.0).
        """
        if not (0 < value <= 1.0):
            raise ValueError("Exponential forgetting factor (eff) must be in the range (0.0 - 1.0).")
        cls._class_eff = value


class ModelStructure(metaclass=_ModelStructureMeta):
    """
    Represents a frequency-domain model structure for real-time parameter estimation
    using an exponential forgetting factor and complex exponential basis functions.
    This class allows both shared (class-wide) and unique (instance-specific)
    frequency representations, enabling efficient modeling across multiple
    model structures with customizable configurations.
    
    Attributes
    ----------
    _class_eff : float
        Default class-wide exponential forgetting factor (eff) used if no instance-specific value is set.
    _instances : weakref.WeakSet
        Tracks all active instances of ModelStructure for efficient class-wide updates.
    frequencies : np.ndarray
        Class-level default frequency array used when no custom frequency vector is provided to an instance.
    """
    _class_eff: float = 1.0
    _instances = weakref.WeakSet()
    frequencies: np.ndarray = np.arange(0.1, 1.54, 0.04)
    
    
    
    def __init__(
            self,
            number_of_regressors: int,
            frequencies: Optional[Union[Sequence[float], np.ndarray]] = None,
            exponential_forgetting_factor: Optional[float] = None
            ) -> None:
        """
        Initialize a ModelStructure instance for time-to-frequency domain
        transformation and adaptive parameter estimation.
        
        Parameters
        ----------
        number_of_regressors : int
            Number of regressors (inputs or basis functions) used in the model.
        frequencies : Union[Sequence[float], np.ndarray], optional
            Custom frequency vector for this instance (if None, class-wide default is used).
        exponential_forgetting_factor : float, optional
            Instance-specific exponential forgetting factor; if None, class-wide EFF is used.
        
        Attributes
        ----------
        _instance_eff : Optional[float]
            Instance-level exponential forgetting factor.
        _num_regressors : int
            Number of regressors (inputs or basis functions) used in the model.
        measuredoutput_timedata : float
            Most recent scalar measured output from time-domain data.
        regressors_timedata : np.ndarray
            Time-domain input regressors (shape = (num_regressors,)).
        frequencies : np.ndarray
            Instance-level frequency array, either custom or copied from class.
        _unique_frequencies : bool
            Flag indicating whether the instance uses a custom frequency vector.
        complex_products : np.ndarray
            Complex exponentials computed from frequencies for time mapping.
        measuredoutput_frequencydata : np.ndarray
            Frequency-domain transformed output data.
        regressors_frequencydata : np.ndarray
            Frequency-domain transformed regressors.
        parameters : np.ndarray
            Real-valued parameter estimates calculated using least squares.
        modeledoutput : float
            Most recent model prediction based on current parameters and inputs.
        """
        self.__class__._instances.add(self)
        self._num_regressors = number_of_regressors
        self._instance_eff: Optional[float] = exponential_forgetting_factor
        
        self.measuredoutput_timedata: float = 0.0
        self.regressors_timedata: np.ndarray = np.zeros(self.num_regressors, dtype=float)
        
        if (frequencies is not None):
            self.frequencies = np.asarray(frequencies, dtype=float).flatten()
            self._unique_frequencies = True
        else:
            self.frequencies = np.copy(self.__class__.frequencies)
            self._unique_frequencies = False
            
        self.complex_products = np.zeros(self.len_frequencies, dtype=complex)
        self.measuredoutput_frequencydata: np.ndarray = np.zeros(self.len_frequencies, dtype=complex)
        self.regressors_frequencydata: np.ndarray = np.zeros((self.len_frequencies, self.num_regressors), dtype=complex)
        
        self.parameters: np.ndarray = np.zeros(self.num_regressors, dtype=float)
        self.modeledoutput: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"ModelStructure(num_regressors={self.num_regressors!r}, "
            f"frequencies={np.round(self.frequencies, decimals=6).tolist()!r})"
            )
    
    def __str__(self) -> str:
        return (
            f"ModelStructure\n"
            f"--------------------------\n"
            f"Number of Regressors: {self.num_regressors}\n"
            f"Frequencies: {np.round(self.frequencies, decimals=6).tolist()}\n"
            f"EFF: {self.eff:.4f}\n"
            )
    
    
    
    def update_model(
            self,
            measuredoutput_timedata: float,
            regressors_timedata: Union[np.ndarray, list[float], float]
            ) -> None:
        """
        Update the model's internal state based on the latest time-domain data.
        This method:
        - Stores the new measured output value.
        - Updates the frequency-domain representations of both the output and regressors
          using the exponential forgetting factor (EFF).
        - Computes the model parameters using least squares regression in the frequency domain.
        - Calculates the new modeled output using the updated parameters.
        
        Parameters
        ----------
        measuredoutput_timedata : float
            The current measured output value in the time domain.
        regressors_timedata : Union[np.ndarray, list[float], float]
            The current regressor values in the time domain. Must match the number of regressors.
        
        Raises
        ------
        ValueError
            If the number of provided regressors does not match the expected count.
        """
    
        self.measuredoutput_timedata = measuredoutput_timedata
        self.measuredoutput_frequencydata = (
            self.eff * self.measuredoutput_frequencydata +
            self.measuredoutput_timedata * self.complex_products
            )
        
        self.regressors_timedata = np.atleast_1d(regressors_timedata).astype(float).flatten()
        if (self.regressors_timedata.size != self.num_regressors):
            raise ValueError("Mismatch in number of regressors.")
        self.regressors_frequencydata = (
            self.eff * self.regressors_frequencydata +
            np.outer(self.complex_products, self.regressors_timedata)
            )
        
        self.parameters = np.real(
            np.linalg.pinv(self.regressors_frequencydata.T @ self.regressors_frequencydata)
                @ (self.regressors_frequencydata.T @ self.measuredoutput_frequencydata)
            ).ravel()
        
        self.modeledoutput = float(np.dot(self.regressors_timedata, self.parameters))
    
    
    def predict_model(self, regressors: Union[np.ndarray, list[float], float]) -> float:
        """
        Predict the modeled output given a new set of regressors.
        
        Parameters
        ----------
        regressors : Union[np.ndarray, list[float], float]
            The regressor values to use for prediction. Must match the trained model dimensions.
        
        Returns
        -------
        float
            The predicted output from the model.
        """
        regressors = np.asarray(regressors, dtype=float).flatten()
        return float(np.dot(regressors, self.parameters))
    
    
    def update_cp_time(self, current_time: float) -> None:
        """
        Recompute the complex exponential terms used in the frequency-domain transformation
        based on an absolute time value.
        
        Parameters
        ----------
        current_time : float
            The current time at which to compute the complex exponentials.
        """
        self.complex_products = np.exp(-1j * self.frequencies * current_time)
    
    
    def update_cp_timestep(self, time_step: float) -> None:
        """
        Recompute the complex exponential terms used in the frequency-domain transformation
        based on an absolute time value.
        
        Parameters
        ----------
        current_time : float
            The current time at which to compute the complex exponentials.
        """
        self.complex_products *= np.exp(-1j * self.frequencies * time_step)
    
    
    
    @property
    def eff(self) -> float:
        """
        Get the exponential forgetting factor (eff) for the instance.
        Returns the instance-specific eff if set; otherwise, returns the class-wide default.
        
        Returns
        -------
        float
            The exponential forgetting factor (eff) used for recursive updates.
        """
        return self._instance_eff if self._instance_eff is not None else ModelStructure._class_eff
    @eff.setter
    def eff(self, value: float) -> None:
        """
        Set the exponential forgetting factor (eff) for the instance.
        
        Parameters
        ----------
        value : float
            The new exponential forgetting factor (eff).
            Must be in the range (0.0 - 1.0).
        
        Raises
        ------
        ValueError
            If the provided value is outside the valid range.
        """
        if not (0 < value <= 1.0):
            raise ValueError("Exponential forgetting factor (eff) must be in the range (0.0 - 1.0).")
        self._instance_eff = value
    def clear_eff(self) -> None:
        """
        Clear the instance-specific exponential forgetting factor (eff),
        reverting to the class-wide default.
        """
        self._instance_eff = None
    
    @property
    def len_frequencies(self) -> int:
        return self.frequencies.size
    
    @property
    def num_regressors(self) -> int:
        return self._num_regressors
    
    
    
    @classmethod
    def class_frequencies(cls, frequencies: Union[np.ndarray, Sequence[float]]) -> None:
        """
        Set the default frequency array for all future instances of the class.
        
        Parameters
        ----------
        frequencies : Union[np.ndarray, Sequence[float]]
            A sequence or NumPy array of frequency values to be used as the
            default for new instances that do not specify a custom frequency array.
        """
        cls.frequencies = np.asarray(frequencies, dtype=float).flatten()
    
    @classmethod
    def update_shared_cp_time(cls, current_time: float) -> None:
        """
        Update the complex frequency products at a specific time for all
        instances sharing the class-level frequency array.
        This method affects only those instances that did not define a unique
        frequency array during initialization.
        
        Parameters
        ----------
        current_time : float
            The time at which to compute the complex frequency products.
        """
        common_cp = np.exp(-1j * cls.frequencies * current_time)
        for inst in cls._instances:
            if not inst._unique_frequencies:
                inst.complex_products = common_cp.copy()
    
    @classmethod
    def update_shared_cp_timestep(cls, time_step: float) -> None:
        """
        Incrementally update the complex frequency products for all instances
        sharing the class-level frequency array by a given time step.
        
        Parameters
        ----------
        time_step : float
            The time step used to update the existing complex products by
            multiplying with delta copmlex products.
        """
        delta_cp = np.exp(-1j * cls.frequencies * time_step)
        for inst in cls._instances:
            if not inst._unique_frequencies:
                inst.complex_products *= delta_cp


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