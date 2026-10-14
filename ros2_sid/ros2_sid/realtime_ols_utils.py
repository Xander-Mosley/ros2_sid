#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realtime_ols_utils.py — structures for real-time ordinary-least-squares

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


import json
from pathlib import Path
import warnings

import numpy as np


__all__ = [
    'CircularBuffer', 'RecursiveFourierTransform', 'RegressorData', 'ordinary_least_squares',
    'modeled_outputs', 'residuals', 'mean_squared_errors', 'coefficient_of_determination', 'confidence_intervals',
    'variance_inflation_factors', 'singular_value_decomposition', 'matrix_correlation',
    ]
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


PACKAGE_SUBROOT = Path(__file__).resolve().parents[1]
SETUP_DIR = PACKAGE_SUBROOT / "ros2_sid" / "setup"
FREQUENCY_CONFIG_FILE = SETUP_DIR / "frequency_config.json"


def _load_frequency_config() -> dict:
    """Load frequency configuration from frequency_config.json."""
    try:
        with FREQUENCY_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError:
        print("Error: frequency_config.json contains invalid JSON.")
        raise
    except OSError as error:
        print(f"Error loading frequency configuration: {error}")
        raise
    return config

def _euclidean_norm(
        matrix: np.ndarray,
        ) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Matrix must be a 2D array")
    is_complex = np.iscomplexobj(matrix)

    if is_complex:
        matrix_mean = 0
    else:
        matrix_mean = np.mean(matrix, axis=0, keepdims=True)

    sjj = np.sum(np.abs(matrix - matrix_mean) ** 2, axis=0, keepdims=True)
    sjj_safe = np.where(sjj == 0, 1, sjj)
    matrix_norm = (matrix - matrix_mean) / np.sqrt(sjj_safe)

    return matrix_norm


class CircularBuffer:
    def __init__(self, capacity: int, start_value: float = 0.0):
        if self._capacity < 1:
            raise ValueError("Capacity must be a positive integer")
        self._capacity = capacity
        self._data = np.zeros(capacity, dtype=float)
        self._index = 0
        self._size = 0
        self.fill_all(start_value)

    def add(self, value: float) -> None:
        self._data[self._index] = value
        self._index = (self._index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

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
    config = _load_frequency_config()
    default_frequencies: np.ndarray = np.asarray(
        config['frequencies_hz'],
        dtype=float,
    )
    default_eff: float = 0.999

    def __init__(
            self,
            eff: float | None = None,
            frequencies: np.ndarray | None = None
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

    def update_spectrum(self,timedata: float,) -> np.ndarray:
        self._frequencydata = (
            self._eff * self._frequencydata +
            timedata * self._complex_products
            )
        return self._frequencydata.copy()
    
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
            eff: float | None = None,
            frequencies: np.ndarray | None = None
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
            eff: float | None = None,
            frequencies: np.ndarray | None = None
            ) -> None:
        if delay < 1:
            raise ValueError("Delay must be a postive integer")
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
    measured_output = np.asarray(measured_output).reshape(-1)
    regressors = np.asarray(regressors)
    if measured_output.shape[0] != regressors.shape[0]:
        raise ValueError("Number of samples in measured_output and regressors must match")
    
    XhX = np.conj(regressors).T @ regressors
    XhZ = np.conj(regressors).T @ measured_output
    parameters = np.linalg.pinv(XhX) @ XhZ
    return parameters.ravel()

def modeled_outputs(
        regressors: np.ndarray,
        parameters: np.ndarray,
        ) -> float | np.ndarray:
    regressors = np.asarray(regressors)
    parameters = np.asarray(parameters)
    if regressors.shape[-1] != parameters.shape[-1]:
        raise ValueError("Number of regressors and parameters must match")
    return regressors @ parameters

def residuals(
        measured_output: float | np.ndarray,
        regressors: np.ndarray,
        parameters: np.ndarray,
        ) -> float | np.ndarray:
    measured_output = np.asarray(measured_output).reshape(-1)
    regressors = np.asarray(regressors)
    parameters = np.asarray(parameters)
    if regressors.shape[-1] != parameters.shape[-1]:
        raise ValueError("Number of regressors and parameters must match")

    res = measured_output - modeled_outputs(regressors, parameters)
    return res.item() if res.size == 1 else res

def mean_squared_errors(
        measured_output: float | np.ndarray,
        regressors: np.ndarray,
        parameters: np.ndarray,
        ) -> float:
    measured_output = np.asarray(measured_output).reshape(-1)
    regressors = np.asarray(regressors)
    parameters = np.asarray(parameters)
    if regressors.shape[-1] != parameters.shape[-1]:
        raise ValueError("Number of regressors and parameters must match")
    
    residual = residuals(measured_output, regressors, parameters)
    mse = np.mean(np.abs(residual) ** 2)
    return float(mse)

def coefficient_of_determination(
        measured_output: np.ndarray,
        regressors: np.ndarray,
        parameters: np.ndarray,
        *,
        recentered: bool = True,
        adjusted: bool = False,
        ) -> float:
    measured_output = np.asarray(measured_output).reshape(-1)
    regressors = np.asarray(regressors)
    parameters = np.asarray(parameters)
    
    if regressors.ndim != 2:
        raise ValueError("Regressors must be a 2D array")
    num_observations, num_regressors = regressors.shape
    if num_observations != measured_output.shape[0]:
        raise ValueError("Number of observations in regressors and measured_output must match")
    if num_regressors != parameters.shape[-1]:
        raise ValueError("Number of regressors and parameters must match")

    if recentered:
        measured_output_mean = np.mean(measured_output)
    else:
        measured_output_mean = 0
    ss_tot = np.sum(np.abs(measured_output - measured_output_mean) ** 2)
    if ss_tot == 0:
        return np.nan

    residual = residuals(measured_output, regressors, parameters)
    ss_res = np.sum(np.abs(residual) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    if adjusted:
        if num_observations - num_regressors - 1 <= 0:
            raise ValueError("Degrees of freedom must be positive for R² calculation")
        r2 = 1 - (1 - r2) * ((num_observations - 1) / (num_observations - num_regressors - 1))
    return r2

def confidence_intervals(
        measured_output: np.ndarray,
        regressors: np.ndarray,
        parameters: np.ndarray,
        *,
        confidence_multiplier: float = 1.96,
        ) -> tuple[float | np.ndarray, np.ndarray]:
    measured_output = np.asarray(measured_output).reshape(-1)
    regressors = np.asarray(regressors)
    parameters = np.asarray(parameters)
    
    if regressors.ndim != 2:
        raise ValueError("Regressors must be a 2D array")
    num_observations, num_regressors = regressors.shape
    dof = num_observations - num_regressors
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive for confidence interval calculation")
    if num_observations != measured_output.shape[0]:
        raise ValueError("Number of observations in regressors and measured_output must match")
    if num_regressors != parameters.shape[-1]:
        raise ValueError("Number of regressors and parameters must match")
    is_complex = (
        np.iscomplexobj(measured_output) or
        np.iscomplexobj(regressors)
    )
    
    residual = residuals(measured_output, regressors, parameters)
    sigma_squared = np.sum(np.abs(residual) ** 2) / dof
    XtX_inv = np.linalg.pinv(np.conj(regressors).T @ regressors)
    param_covariance = sigma_squared * XtX_inv

    if not is_complex:
        param_variance = np.real(np.diag(param_covariance))
        x_i = regressors[-1, :].reshape(1, -1)
        model_variance = sigma_squared * np.real(x_i @ XtX_inv @ x_i.T).item()
    else:
        param_variance = np.real(np.diag(param_covariance)) / 2     # circular complex Gaussian assumption
        model_variance = np.einsum(
            'ij,jk,ik->i',
            regressors,
            param_covariance,
            np.conj(regressors)
            ).real / 2

    param_ci = confidence_multiplier * np.sqrt(np.maximum(param_variance, 0))
    model_ci = confidence_multiplier * np.sqrt(np.maximum(model_variance, 0))

    return model_ci, param_ci

def variance_inflation_factors(
        regressors: np.ndarray,
        *,
        return_cod: bool = False,
        ) -> np.ndarray:
    regressors = np.asarray(regressors)
    if regressors.ndim != 2:
        raise ValueError("Regressors must be a 2D array")

    regressors_norm = _euclidean_norm(regressors)

    column_norms = np.linalg.norm(regressors_norm, axis=0)
    valid = column_norms > 0
    vif = np.full(regressors.shape[1], np.nan, dtype=float)
    if np.count_nonzero(valid) < 2:
        return vif
    regressors_norm_safe = regressors_norm[:, valid]

    XtX = np.conj(regressors_norm_safe).T @ regressors_norm_safe
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return vif

    vif_valid = np.real(np.diag(XtX_inv))
    vif[valid] = vif_valid
    if not return_cod:
        return vif

    cod = np.full_like(vif, np.nan)
    positive = vif > 0
    cod[positive] = 1 - (1 / vif[positive])
    return cod

def singular_value_decomposition(
    regressors: np.ndarray,
    *,
    return_cond_nums: bool = False,
    ) -> np.ndarray:
    regressors = np.asarray(regressors)
    if regressors.ndim != 2:
        raise ValueError("Regressors must be a 2D array")

    regressors_norm = _euclidean_norm(regressors)
    singular_values = np.linalg.svd(regressors_norm, compute_uv=False)

    if not return_cond_nums:
        return singular_values
    
    with np.errstate(divide="ignore", invalid="ignore"):
        condition_numbers = singular_values[0] / singular_values
    return condition_numbers
    
def matrix_correlation(
        matrix: np.ndarray,
        ) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Matrix must be a 2D array")

    matrix_norm = _euclidean_norm(matrix)
    return np.conj(matrix_norm).T @ matrix_norm


if (__name__ == '__main__'):
    warnings.warn(
        "This module provides utilities for real-time "
        "frequency-domain OLS estimation. It is intended "
        "to be imported, not executed directly.",
        UserWarning)