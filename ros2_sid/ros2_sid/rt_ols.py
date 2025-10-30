# System-Identification, Real-Time Ordinary-Least-Squares
# Class structures and functions for performing real-time OLS on several model instances.
# V1.0: Created a reformatted StoredData and diff().
# Xander Mosley - 20250703102331
# V1.1: Created a reformatted ModelStructure (with the additiion of _ModelStructureMeta).
# Xander Mosley - 20250707162841


import numpy as np
from typing import Optional, Sequence, Union
import warnings
import weakref


__all__ = ['StoredData', 'ModelStructure', 'diff', 'sg_diff']


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


def diff(
        time: np.ndarray,
        data: np.ndarray
        ) -> float:
    """
    Estimates the derivative of 'data' with respect to 'time' using a 
    least-squares linear regression over six samples.
    
    Assumes both 'time' and 'data' contain exactly six elements.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length 6).
    data : np.ndarray
        1D array of data values (length 6), corresponding to the time points.
        
    Returns
    -------
    float
        The estimated derivative (slope) of data with respect to time.
    """
    
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()

    if time.size != 6 or data.size != 6:
        raise ValueError("Both 'time' and 'data' must be 1D arrays of length 6.")
        
    sum_xt = np.dot(data, time)
    sum_t = time.sum()
    sum_x = data.sum()
    sum_t2 = np.dot(time, time)
    
    denominator = (6 * sum_t2) - (sum_t ** 2)
    if (denominator == 0):
        raise ZeroDivisionError("Denominator in derivative computation is zero.")
        
    numerator = (6 * sum_xt) - (sum_x * sum_t)
        
    return numerator / denominator


def sg_diff(    # Savitzky-Golay sytle differentiation
        time: np.ndarray,
        data: np.ndarray
        ) -> float:
    """
    Estimates the derivative of 'data' with respect to 'time' using a
    least-squares quadratic regression over several samples.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length 6).
    data : np.ndarray
        1D array of data values (length 6), corresponding to the time points.
        
    Returns
    -------
    float
        The estimated derivative (slope) of data with respect to time.
    """
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()

    if time.size != data.size:
        raise ValueError("Both 'time' and 'data' must be 1D arrays of the same length.")

    # Shift time to improve numerical stability (set last point to t=0)
    center_idx = -2
    shifted_time = time - time[center_idx]  # t[-1] becomes 0

    # Design matrix for quadratic fit: [t^2, t, 1]
    A = np.vstack([shifted_time**3, shifted_time**2, shifted_time, np.ones_like(shifted_time)]).T

    # Solve least squares: find coefficients [a, b, c] for ax^2 + bx + c
    coeffs, *_ = np.linalg.lstsq(A, data, rcond=None)
    a, b, c, _ = coeffs

    # Using a 4th order polynomial starts to fit to noise. TODO: Check if results improve with a better input data smoother.
    # Derivative of ax^3 + bx^2 + cx is 3a*t^2 2b*t + c
    # Evaluate at t=0 (last point)
    derivative_at_last_point = c
    
    return derivative_at_last_point



if (__name__ == '__main__'):
    warnings.warn(
        "This script is not intended to be run as a standalone program."
        " It contains structures and functions to be imported and used in other scripts.",
        UserWarning)