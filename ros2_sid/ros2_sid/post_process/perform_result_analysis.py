#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import re

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ros2_sid.realtime_ols_utils import (
    ordinary_least_squares,
    modeled_outputs,
    residuals,
    mean_squared_errors,
    coefficient_of_determination,
    confidence_intervals,
    variance_inflation_factors,
    singular_value_decomposition,
    matrix_correlation,
)


__all__ = ['SysIdData', 'OLSResult', 'load_data_topic', 'load_model_topic']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


DATA_DIRECTORY = Path(__file__).parent / "topic_data_files"


@dataclass
class SysIdData:
    timestamp: np.ndarray = field(default_factory=lambda: np.array([]))
    timestep: np.ndarray = field(default_factory=lambda: np.array([]))
    trend: np.ndarray = field(default_factory=lambda: np.array([]))
    value: np.ndarray = field(default_factory=lambda: np.array([]))
    frequency: np.ndarray = field(default_factory=lambda: np.array([]))
    spectrum: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=complex)
    )

@dataclass
class OLSResult:
    timestamp: np.ndarray = field(default_factory=lambda: np.array([]))
    measured_output: np.ndarray = field(default_factory=lambda: np.array([]))
    regressors: np.ndarray = field(default_factory=lambda: np.array([]))
    parameters: np.ndarray = field(default_factory=lambda: np.array([]))

    modeled_output: np.ndarray = field(default_factory=lambda: np.array([]))
    residual: np.ndarray = field(default_factory=lambda: np.array([]))
    mse: np.ndarray = field(default_factory=lambda: np.array([]))
    cod: np.ndarray = field(default_factory=lambda: np.array([]))

    modeled_output_ci: np.ndarray = field(default_factory=lambda: np.array([]))
    modeled_output_cip: np.ndarray = field(default_factory=lambda: np.array([]))
    parameter_cis: np.ndarray = field(default_factory=lambda: np.array([]))
    parameter_cips: np.ndarray = field(default_factory=lambda: np.array([]))

    regressor_cods: np.ndarray = field(default_factory=lambda: np.array([]))
    regressor_conds: np.ndarray = field(default_factory=lambda: np.array([]))
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))


def load_data_topic(topicname: str) -> SysIdData:
    """
    Load a topic CSV and return its contents in a SysIdData instance.

    Expected file path: DATA_DIRECTORY / f"{topicname}_data.csv"

    Expected columns:
        timestamp
        {topicname}_timestep
        {topicname}_trend
        {topicname}_value
        {topicname}_frequency_{i}
        {topicname}_spec_real_{i}
        {topicname}_spec_imag_{i}

    Spectral data is returned as a complex matrix with:
        Rows    = instances of time
        Columns = frequency bins
    """
    filepath = DATA_DIRECTORY / f"{topicname}_data.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data topic file not found: {filepath}"
        )
    df = pd.read_csv(filepath)

    data = SysIdData()

    if "timestamp" in df.columns:
        data.timestamp = df["timestamp"].to_numpy()

    for attribute in ("timestep", "trend", "value"):
        column = f"{topicname}_{attribute}"
        if column in df.columns:
            setattr(data, attribute, df[column].to_numpy())

    frequency_pattern = re.compile(rf"^{re.escape(topicname)}_frequency_(\d+)$")
    frequency_indices = sorted(
        int(match.group(1))
        for column in df.columns
        if (match := frequency_pattern.match(column))
    )
    if not frequency_indices:
        return data
    
    frequency_columns = [f"{topicname}_frequency_{i}" for i in frequency_indices]
    frequency_matrix = df[frequency_columns].to_numpy()
    if not np.allclose(frequency_matrix, frequency_matrix[0]):
        raise ValueError(f"Frequency vector changes over time in {topicname}")
    data.frequency = frequency_matrix[0]

    real_columns = [f"{topicname}_spec_real_{i}" for i in frequency_indices]
    imag_columns = [f"{topicname}_spec_imag_{i}" for i in frequency_indices]
    if all(column in df.columns for column in real_columns + imag_columns):
        real = df[real_columns].to_numpy()
        imag = df[imag_columns].to_numpy()
        data.spectrum = real + 1j * imag

    return data

def load_model_topic(model_name: str) -> OLSResult:
    """
    Load model results from a CSV file into an OLSResult instance.

    Parameters
    ----------
    model_name : str
        Model name. The expected filename is {model_name}_data.csv.

    Returns
    -------
    OLSResult
        Loaded model results. Missing data fields remain empty.
    """
    filepath = DATA_DIRECTORY / f"{model_name}_data.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model topic file not found: {filepath}"
        )
    df = pd.read_csv(filepath)

    result = OLSResult()

    if "timestamp" in df.columns:
        result.timestamp = df["timestamp"].to_numpy()
    if f"{model_name}_measured_output" in df.columns:
        result.measured_output = df[f"{model_name}_measured_output"].to_numpy()
    if f"{model_name}_modeled_output" in df.columns:
        result.modeled_output = df[f"{model_name}_modeled_output"].to_numpy()
    if f"{model_name}_residual" in df.columns:
        result.residual = df[f"{model_name}_residual"].to_numpy()
    if f"{model_name}_mse" in df.columns:
        result.mse = df[f"{model_name}_mse"].to_numpy()
    if f"{model_name}_cod" in df.columns:
        result.cod = df[f"{model_name}_cod"].to_numpy()
    if f"{model_name}_modeled_output_ci" in df.columns:
        result.modeled_output_ci = df[f"{model_name}_modeled_output_ci"].to_numpy()
    if f"{model_name}_modeled_output_cip" in df.columns:
        result.modeled_output_cip = df[f"{model_name}_modeled_output_cip"].to_numpy()

    def get_columns(prefix: str) -> list[str]:
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        columns = [
            (int(match.group(1)), column)
            for column in df.columns
            if (match := pattern.match(column))
        ]
        return [column for _, column in sorted(columns)]

    regressor_columns = get_columns(f"{model_name}_regressor")
    if regressor_columns:
        result.regressors = df[regressor_columns].to_numpy()

    parameter_columns = get_columns(f"{model_name}_parameter")
    if parameter_columns:
        result.parameters = df[parameter_columns].to_numpy()

    parameter_ci_columns = get_columns(f"{model_name}_parameter_ci")
    if parameter_ci_columns:
        result.parameter_cis = df[parameter_ci_columns].to_numpy()

    parameter_cip_columns = get_columns(f"{model_name}_parameter_cip")
    if parameter_cip_columns:
        result.parameter_cips = df[parameter_cip_columns].to_numpy()

    regressor_cod_columns = get_columns(f"{model_name}_regressor_cod")
    if regressor_cod_columns:
        result.regressor_cods = df[regressor_cod_columns].to_numpy()

    regressor_cond_columns = get_columns(f"{model_name}_regressor_cond")
    if regressor_cond_columns:
        result.regressor_conds = df[regressor_cond_columns].to_numpy()

    def get_correlation_matrix(model_name: str) -> np.ndarray:
        pattern = re.compile(rf"^{re.escape(model_name)}_correlation_(\d+)_to_(\d+)$")
        columns = []
        for column in df.columns:
            match = pattern.match(column)
            if match:
                j = int(match.group(1)) - 1
                k = int(match.group(2)) - 1
                columns.append((j, k, column))
        if not columns:
            return np.empty((len(df), 0, 0))
        num_regressors = max(
            max(j, k) for j, k, _ in columns
        ) + 1
        matrix = np.full(
            (len(df), num_regressors, num_regressors),
            np.nan,
            dtype=float,
        )
        for j, k, column in columns:
            matrix[:, j, k] = df[column].to_numpy()
        indices = np.arange(num_regressors)
        matrix[:, indices, indices] = 1.0
        return matrix

    result.correlation_matrix = get_correlation_matrix(model_name)

    return result

def save_model_result(
        result: OLSResult,
        model_name: str,
        ) -> None:
    """
    Save an OLSResult instance to a CSV file in DATA_DIRECTORY.

    Parameters
    ----------
    result : OLSResult
        System identification results to save.

    model_name : str
        Name of the model for which to save results.

    Raises
    ------
    FileExistsError
        If a file with the specified name already exists.
    """
    filepath = DATA_DIRECTORY / f"{model_name}_data.csv"
    if filepath.exists():
        response = input(
            f"File already exists: {filepath}\n"
            "Would you like to overwrite it? [y/N]: "
        ).strip().lower()

        if response not in ("y", "yes"):
            print("Save cancelled.")
            return
    data = {}

    data["timestamp"] = result.timestamp
    data[f"{model_name}_measured_output"] = result.measured_output
    for i in range(result.regressors.shape[1]):
        data[f"{model_name}_regressor_{i + 1}"] = result.regressors[:, i]
    for i in range(result.parameters.shape[1]):
        data[f"{model_name}_parameter_{i + 1}"] = result.parameters[:, i]

    data[f"{model_name}_modeled_output"] = result.modeled_output
    data[f"{model_name}_residual"] = result.residual
    data[f"{model_name}_mse"] = result.mse
    data[f"{model_name}_cod"] = result.cod

    data[f"{model_name}_modeled_output_ci"] = result.modeled_output_ci
    data[f"{model_name}_modeled_output_cip"] = result.modeled_output_cip
    for i in range(result.parameter_cis.shape[1]):
        data[f"{model_name}_parameter_ci_{i + 1}"] = result.parameter_cis[:, i]
    for i in range(result.parameter_cips.shape[1]):
        data[f"{model_name}_parameter_cip_{i + 1}"] = result.parameter_cips[:, i]

    for i in range(result.regressor_cods.shape[1]):
        data[f"{model_name}_regressor_cod_{i + 1}"] = result.regressor_cods[:, i]
    for i in range(result.regressor_conds.shape[1]):
        data[f"{model_name}_regressor_cond_{i + 1}"] = result.regressor_conds[:, i]

    for j in range(result.regressors.shape[1]):
        for k in range(result.regressors.shape[1]):
            if j == k:
                continue
            correlation_element = result.correlation_matrix[:, j, k]
            correlation_element_label = f"{model_name}_correlation_{j + 1}_to_{k + 1}"
            data[correlation_element_label] = correlation_element

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")


def rerun_live_sid(
        measured_output: SysIdData,
        regressors: list[SysIdData],
        *,
        calculation_rate_hz: float | None = None,
        measured_output_lag: int = 0,
        regressors_lags: list[int] | None = None,
        ) -> OLSResult:
    """
    Calculate complex-valued least-squares parameters over time using
    recursively updated frequency spectra.

    Parameters
    ----------
    measured_output : SysIdData
        Measured output signal.

    regressors : list[SysIdData]
        Regressor signals used to estimate the parameters.

    calculation_rate_hz : float | None
        Desired parameter estimation rate in Hz. If None, the slowest
        sampling rate among the measured output and regressors is used.

    Returns
    -------
    np.ndarray
        Complex parameter estimates with shape (N, P), where N is the
        number of parameter estimation instances and P is the number
        of regressors.
    """
    if not regressors:
        raise ValueError("At least one regressor is required.")
    signals = [measured_output, *regressors]
    for signal in signals:
        if signal.timestamp.size == 0:
            raise ValueError("All signals must contain timestamps.")
        if signal.spectrum.size == 0:
            raise ValueError("All signals must contain spectral data.")
        if signal.timestep.size != signal.timestamp.size:
            raise ValueError("Signal timesteps must match timestamp length.")
        if signal.trend.size != signal.timestamp.size:
            raise ValueError("Signal trends must match timestamp length.")
        if signal.value.size != signal.timestamp.size:
            raise ValueError("Signal values must match timestamp length.")
        if signal.spectrum.shape[0] != signal.timestamp.size:
            raise ValueError("Spectrum rows must match timestamp length.")
        
    if regressors_lags is None:
        regressors_lags = [0] * len(regressors)
    if len(regressors_lags) != len(regressors):
        raise ValueError("Number of regressor lags must match number of regressors.")
    all_lags = [measured_output_lag, *regressors_lags]
    if any(not isinstance(lag, (int, np.integer)) for lag in all_lags):
        raise ValueError("All lags must be integers.")
    if any(lag < 0 for lag in all_lags):
        raise ValueError("All lags must be non-negative.")

    frequency = measured_output.frequency
    for signal in signals[1:]:
        if not np.allclose(signal.frequency, frequency):
            raise ValueError("All signals must have matching frequency vectors.")
    
    sampling_rates = []
    for signal in signals:
        dt = signal.timestep
        if np.any(dt <= 0):
            raise ValueError("Timestamps must be strictly increasing.")
        sampling_rates.append(1.0 / np.median(dt))
    if calculation_rate_hz is None:
        calculation_rate_hz = min(sampling_rates)
    if calculation_rate_hz <= 0: # type: ignore
        raise ValueError("Calculation rate must be positive.")

    start_time = max(signal.timestamp[0] for signal in signals)
    end_time = min(signal.timestamp[-1] for signal in signals)
    if start_time > end_time:
        raise ValueError("Signals have no overlapping time interval.")
    calculation_timestamps = np.arange(
        start_time,
        end_time,
        1.0 / calculation_rate_hz # type: ignore
    )

    num_instances = calculation_timestamps.size
    num_parameters = len(regressors)
    result = OLSResult(
        timestamp=calculation_timestamps,
        measured_output=np.full((num_instances,), np.nan, dtype=float),
        regressors=np.full((num_instances, num_parameters), np.nan, dtype=float),
        parameters=np.full((num_instances, num_parameters), np.nan, dtype=float),
        modeled_output=np.full((num_instances,), np.nan, dtype=float),
        residual=np.full((num_instances,), np.nan, dtype=float),
        mse=np.full((num_instances,), np.nan, dtype=float),
        cod=np.full((num_instances,), np.nan, dtype=float),
        modeled_output_ci=np.full((num_instances,), np.nan, dtype=float),
        modeled_output_cip=np.full((num_instances,), np.nan, dtype=float),
        parameter_cis=np.full((num_instances, num_parameters), np.nan, dtype=float),
        parameter_cips=np.full((num_instances, num_parameters), np.nan, dtype=float),
        regressor_cods=np.full((num_instances, num_parameters), np.nan, dtype=float),
        regressor_conds=np.full((num_instances, num_parameters), np.nan, dtype=float),
        correlation_matrix=np.full((num_instances, num_parameters, num_parameters), np.nan, dtype=float),
    )

    for i, timestamp in enumerate(calculation_timestamps):
        indices = [
            np.searchsorted(
                signal.timestamp,
                timestamp,
                side="right"
            ) - 1
            for signal in signals
        ]   # Find the latest available sample for each signal.
        indices = [index - lag for index, lag in zip(indices, all_lags)]
        if any(index < 0 for index in indices):
            continue  # Not enough data yet for this timestamp.

        Z = measured_output.spectrum[indices[0], :]
        X = np.column_stack([
            signal.spectrum[index, :]
            for signal, index in zip(regressors, indices[1:])
        ])
        theta = ordinary_least_squares(Z, X)
        
        Y = modeled_outputs(X, theta)

        Y_ci, theta_ci = confidence_intervals(Z, X, theta)
        Y_cip = Y_ci / np.abs(Y)
        theta_cip = theta_ci / np.abs(theta)

        result.measured_output[i] = measured_output.value[indices[0]]
        result.regressors[i, :] = [
            signal.value[index]
            for signal, index in zip(regressors, indices[1:])
        ]
        result.parameters[i, :] = np.real(theta)
        result.modeled_output[i] = modeled_outputs(result.regressors[i, :], result.parameters[i, :])
        result.residual[i] = residuals(result.measured_output[i], result.regressors[i, :], result.parameters[i, :])
        result.mse[i] = mean_squared_errors(Z, X, theta)
        result.cod[i] = coefficient_of_determination(Z, X, theta, recentered=False, adjusted=True)
        result.modeled_output_cip[i] = np.sqrt(np.mean(Y_cip ** 2))
        result.modeled_output_ci[i] = np.abs(result.modeled_output[i]) * result.modeled_output_cip[i]
        result.parameter_cis[i, :] = np.real(theta_ci)
        result.parameter_cips[i, :] = np.real(theta_cip)
        result.regressor_cods[i, :] = np.real(variance_inflation_factors(X, return_cod=True))
        result.regressor_conds[i, :] = singular_value_decomposition(X, return_cond_nums=True)
        result.correlation_matrix[i, :, :] = np.real(matrix_correlation(X))
    return result

def reconstruct_missing_results(
        existing_result: OLSResult,
        *,
        window_size: int = 100,
        ) -> OLSResult:
    if window_size < 1:
        raise ValueError("Window size must be a positive integer")
    result = OLSResult(
        timestamp=existing_result.timestamp.copy(),
        measured_output=existing_result.measured_output.copy(),
        regressors=existing_result.regressors.copy(),
        parameters=existing_result.parameters.copy(),
        modeled_output=existing_result.modeled_output.copy(),
        residual=existing_result.residual.copy(),
        mse=existing_result.mse.copy(),
        cod=existing_result.cod.copy(),
        modeled_output_ci=existing_result.modeled_output_ci.copy(),
        modeled_output_cip=existing_result.modeled_output_cip.copy(),
        parameter_cis=existing_result.parameter_cis.copy(),
        parameter_cips=existing_result.parameter_cips.copy(),
        regressor_cods=existing_result.regressor_cods.copy(),
        regressor_conds=existing_result.regressor_conds.copy(),
        correlation_matrix=existing_result.correlation_matrix.copy(),
    )

    measured_output = np.asarray(result.measured_output).reshape(-1)
    regressors = np.asarray(result.regressors)

    if regressors.ndim != 2:
        raise ValueError("Regressors must be a 2D array.")
    if measured_output.size != regressors.shape[0]:
        raise ValueError("Measured output and regressors must have the same number of samples.")

    num_samples, num_parameters = regressors.shape

    if measured_output.size != num_samples:
        raise ValueError("Measured output and regressors must have the same number of samples.")
    if num_samples == 0 or num_parameters == 0:
        raise ValueError("Measured output and regressors must not be empty.")
    if result.timestamp.size != num_samples:
        raise ValueError("Timestamp and signal arrays must have the same number of samples.")

    # Initialize missing result arrays.
    def initialize_array(attribute: str, shape: tuple[int, ...]) -> np.ndarray:
        array = getattr(result, attribute)
        if array.size == 0:
            array = np.full(shape, np.nan, dtype=float)
            setattr(result, attribute, array)
        elif array.shape != shape:
            raise ValueError(f"{attribute} must have shape {shape}, got {array.shape}.")
        return array

    result.parameters = initialize_array("parameters", (num_samples, num_parameters))
    result.modeled_output = initialize_array("modeled_output", (num_samples,))
    result.residual = initialize_array("residual", (num_samples,))
    result.mse = initialize_array("mse", (num_samples,))
    result.cod = initialize_array("cod", (num_samples,))
    result.modeled_output_ci = initialize_array("modeled_output_ci", (num_samples,))
    result.modeled_output_cip = initialize_array("modeled_output_cip", (num_samples,))
    result.parameter_cis = initialize_array("parameter_cis", (num_samples, num_parameters))
    result.parameter_cips = initialize_array("parameter_cips", (num_samples, num_parameters))
    result.regressor_cods = initialize_array("regressor_cods", (num_samples, num_parameters))
    result.regressor_conds = initialize_array("regressor_conds", (num_samples, num_parameters))
    result.correlation_matrix = initialize_array("correlation_matrix", (num_samples, num_parameters, num_parameters))

    for i in range(num_samples):
        start = max(0, i - window_size + 1)
        z = result.measured_output[start:i + 1]
        x = result.regressors[start:i + 1, :]

        if np.any(np.isnan(z)) or np.any(np.isnan(x)):
            continue  # Skip if there are NaNs in the current window.

        if np.isnan(result.parameters[i, :]).any():
            theta = ordinary_least_squares(z, x)
            result.parameters[i, :] = theta

        if np.isnan(result.modeled_output[i]):
            result.modeled_output[i] = modeled_outputs(result.regressors[i, :], result.parameters[i, :])

        if np.isnan(result.residual[i]):
            result.residual[i] = residuals(result.measured_output[i], result.regressors[i, :], result.parameters[i, :])

        if np.isnan(result.mse[i]):
            result.mse[i] = mean_squared_errors(z, x, result.parameters[i, :])

        if (np.isnan(result.cod[i]) and
            i > window_size - result.regressors.shape[1]):
            result.cod[i] = coefficient_of_determination(z, x, result.parameters[i, :], adjusted=True)

        if ((np.isnan(result.modeled_output_ci[i]) or
            np.isnan(result.modeled_output_cip[i]) or
            np.isnan(result.parameter_cis[i, :]).any() or
            np.isnan(result.parameter_cips[i, :]).any()) and 
            i > window_size - result.regressors.shape[1]):
            y_ci, theta_ci = confidence_intervals(z, x, result.parameters[i, :])
            result.modeled_output_ci[i] = y_ci
            result.modeled_output_cip[i] = y_ci / np.abs(result.modeled_output[i])
            result.parameter_cis[i, :] = theta_ci
            result.parameter_cips[i, :] = theta_ci / np.abs(result.parameters[i, :])

        if np.isnan(result.regressor_cods[i, :]).any():
            result.regressor_cods[i, :] = variance_inflation_factors(x, return_cod=True)

        if np.isnan(result.regressor_conds[i, :]).any():
            result.regressor_conds[i, :] = singular_value_decomposition(x, return_cond_nums=True)

        if np.isnan(result.correlation_matrix[i, :, :]).any():
            result.correlation_matrix[i, :, :] = matrix_correlation(x)

    return result

def array_shift(
        structure: SysIdData | OLSResult,
        array_name: str,
        shift: int,
        ) -> SysIdData | OLSResult:
    if not hasattr(structure, array_name):
        raise AttributeError(f"Structure has no attribute '{array_name}'.")
    array = getattr(structure, array_name)
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Attribute '{array_name}' must be a NumPy array.")
    setattr(
        structure,
        array_name,
        np.roll(array, shift=shift, axis=0),
    )
    return structure


def main() -> None:
    rol_accel = load_data_topic("differ_imu_gx")
    rol_velo = load_data_topic("fourier_imu_gx")
    ail = load_data_topic("fourier_rcout_ail")
    # rol_accel = array_shift(rol_accel, "value", -12)
    awesome_result = rerun_live_sid(
        measured_output=rol_accel,
        regressors=[rol_velo, ail],
        calculation_rate_hz=25.0,
        # measured_output_lag=30,
        # regressors_lags=[0, 15],
    )
    save_model_result(awesome_result, "rol_2")

    rol_model = load_model_topic("rol")
    rol_model = reconstruct_missing_results(rol_model)
    save_model_result(rol_model, "rol_fixed")

    nd_p = load_data_topic("nondim_p")
    nd_r = load_data_topic("nondim_r")
    nd_ail = load_data_topic("nondim_ail")
    nd_rud = load_data_topic("nondim_rud")
    nd_qr = load_data_topic("nondim_qr")
    nd_rpq = load_data_topic("nondim_rpq")

    nondim_result = rerun_live_sid(
        measured_output=rol_accel,
        regressors=[nd_p, nd_r, nd_ail, nd_rud, nd_qr, nd_rpq],
        calculation_rate_hz=25.0,
        measured_output_lag=(3*25+5)*6,
    )
    save_model_result(nondim_result, "nondim_2")

if (__name__ == "__main__"):
    main()