#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

from plotter_class import PlotFigure

ArrayLike = Union[float, Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]


# __all__ = ['']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def _ensure_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

def _matrix_norm(matrix: np.ndarray) -> np.ndarray:
    matrix_mean = np.mean(matrix, axis=0)
    sjj = np.sum((matrix - matrix_mean) ** 2, axis=0)
    sjj_safe = np.where(sjj == 0, 1, sjj)
    matrix_norm = (matrix - matrix_mean) / np.sqrt(sjj_safe)
    return matrix_norm

def _extract_param_number(col: str) -> Union[int, float]:
    match = re.search(r'parameter_(\d+)', col)
    return int(match.group(1)) if match else float('inf')


def _sliding_mse(
        true_values: np.ndarray | pd.Series | list,
        pred_values: np.ndarray | pd.Series | list,
        window_size: int = 6
        ) -> np.ndarray:
    
    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
    
    sliding_mse = list(np.full(window_size - 1, np.nan))

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, len(true_values) - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        
        if np.isnan(true_window).any() or np.isnan(pred_window).any():
            mse = np.nan
        else:
            mse = np.mean((true_window - pred_window) ** 2)
            
        sliding_mse.append(mse)
        
    return np.array(sliding_mse)

def _sliding_cod(
        true_values: np.ndarray | pd.Series | list,
        pred_values: np.ndarray | pd.Series | list,
        window_size: int = 30
        ) -> np.ndarray:
    
    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
        
    sliding_cod = list(np.full(window_size - 1, np.nan))

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, len(true_values) - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        
        if np.isnan(true_window).any() or np.isnan(pred_window).any():
            cod = np.nan
        else:
            ss_total = np.sum((true_window - np.mean(true_window)) ** 2)
            ss_res = np.sum((true_window - pred_window) ** 2)
            
            if ss_total == 0:
                cod = np.nan
            else:
                cod = 1 - (ss_res / ss_total)
                
        sliding_cod.append(cod)
        
    return np.array(sliding_cod)

def _sliding_adjusted_cod(
        true_values: np.ndarray | pd.Series | list,
        pred_values: np.ndarray | pd.Series | list,
        num_predictors: int,
        window_size: int = 30
        ) -> np.ndarray:
    
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for adjusted R².")

    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
    
    sliding_cod = list(np.full(window_size - 1, np.nan))

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, len(true_values) - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        
        if np.isnan(true_window).any() or np.isnan(pred_window).any():
            cod = np.nan
        else:
            ss_total = np.sum((true_window - np.mean(true_window)) ** 2)
            ss_res = np.sum((true_window - pred_window) ** 2)
            
            if ss_total == 0 or (window_size - num_predictors - 1) <= 0:
                cod = np.nan
            else:
                r2 = 1 - (ss_res / ss_total)
                cod = 1 - (((1 - r2) * (window_size - 1)) / (window_size - num_predictors - 1))
                
        sliding_cod.append(cod)
    
    return np.array(sliding_cod)

def _sliding_confidence_intervals(
    true_values: np.ndarray | pd.Series | list,
    pred_values: np.ndarray | pd.Series | list,
    predictors: np.ndarray | pd.Series | pd.DataFrame | list,
    window_size: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
    
    true_values = _ensure_numpy(true_values)
    pred_values = _ensure_numpy(pred_values)
    if true_values.shape != pred_values.shape:
        raise ValueError("The true values array and predicted values array must have the same shape.")
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_predictors = predictors.shape
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for confidence intervals.")

    output_ci_array = np.full(num_samples, np.nan)
    param_ci_array = np.full((num_samples, num_predictors), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        true_window = true_values[i - offset_left : i + offset_right + 1]
        pred_window = pred_values[i - offset_left : i + offset_right + 1]
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(true_window).any() or np.isnan(pred_window).any() or np.isnan(X).any():
            continue

        residuals = true_window - pred_window
        sigma_squared = np.sum(residuals ** 2) / (window_size - num_predictors)
        XtX_inv = np.linalg.pinv(X.T @ X)
        djj = np.diag(XtX_inv)
        param_ci_array[i, :] = 2 * np.sqrt(sigma_squared * djj)

        x_i = X[-1, :].reshape(1, -1)
        output_var = (x_i @ XtX_inv @ x_i.T).item()
        output_ci_array[i] = 2 * np.sqrt(sigma_squared * output_var)

    return output_ci_array, param_ci_array

def _sliding_vif_cod(
    predictors: np.ndarray | pd.Series | pd.DataFrame | list,
    window_size: int = 30
    ) -> np.ndarray:
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_predictors = predictors.shape
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for variance inflation factors.")

    cod_array = np.full((num_samples, num_predictors), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(X).any():
            continue
        
        X_norm = _matrix_norm(X)
        XtX = X_norm.T @ X_norm
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        
        vif = np.diag(XtX_inv)
        cod = 1 - 1 / vif
        cod_array[i, :] = cod

    return cod_array

def _sliding_svd_cond(
    predictors: np.ndarray | pd.Series | pd.DataFrame | list,
    window_size: int = 30
    ) -> np.ndarray:
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_predictors = predictors.shape
    if num_predictors >= window_size:
        raise ValueError("Number of predictors must be less than window size for singular value decomposition.")

    cond_array = np.full((num_samples, num_predictors), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(X).any():
            continue
        
        X_norm = _matrix_norm(X)
        
        try:
            U, singular_values, Vt = np.linalg.svd(X_norm, full_matrices=False)
            if np.any(singular_values == 0):
                continue
            
            max_sv = np.max(singular_values)
            cond = max_sv / singular_values
            cond_array[i, :len(cond)] = cond
            
        except np.linalg.LinAlgError:
            continue

    return cond_array

def _sliding_correlation_matrix(
    predictors: np.ndarray | pd.DataFrame | list,
    window_size: int = 30
    ) -> np.ndarray:
    
    predictors = _ensure_numpy(predictors)
    
    num_samples, num_features = predictors.shape
    if num_features >= window_size:
        raise ValueError("Number of predictors must be less than window size.")
        
    corr_matrices = np.full((num_samples, num_features, num_features), np.nan)

    if window_size % 2 == 0:
        half_window = window_size // 2
        offset_left = half_window - 1
        offset_right = half_window
    else:
        half_window = window_size // 2
        offset_left = offset_right = half_window

    for i in range(offset_left, num_samples - offset_right):
        X = predictors[i - offset_left : i + offset_right + 1, :]

        if np.isnan(X).any():
            continue
        
        X_norm = _matrix_norm(X)
        
        corr_matrix = (X_norm.T @ X_norm)   # / (window_size - 1)
        corr_matrices[i] = corr_matrix

    return corr_matrices


def extract_model(dataframe, prefix):
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty.")
    if 'timestamp' not in dataframe.columns:
        raise ValueError("Missing required 'timestamp' column in DataFrame.")
    if not any(col.startswith(prefix) for col in dataframe.columns):
        raise ValueError(f"No columns found for prefix '{prefix}' in provided data frame.")
    
    regressor_cols = sorted(
        [col for col in dataframe.columns if col.startswith(prefix + 'regressor_')],
        key=_extract_param_number
    )
    parameter_cols = sorted(
        [col for col in dataframe.columns if col.startswith(prefix + 'parameter_')],
        key=_extract_param_number
    )
    if len(regressor_cols) == 0:
        raise ValueError(f"Missing a '{prefix}regressor_#' column.")
    if len(parameter_cols) == 0:
        raise ValueError(f"Missing a '{prefix}parameter_#' column.")
    if len(regressor_cols) != len(parameter_cols):
        raise ValueError(
            f"Mismatched number of regressors ({len(regressor_cols)}) and parameters ({len(parameter_cols)})."
        )
    
    extracted_cols = ['timestamp', f"{prefix}measured_output"] + regressor_cols + parameter_cols
    
    return dataframe[extracted_cols]

def process_models(dataframes: list[pd.DataFrame]):
    for i, df in enumerate(dataframes):
        if df.empty:
            raise ValueError(f"DataFrame at index {i} is empty.")
        if 'timestamp' not in df.columns:
            raise ValueError(f"DataFrame at index {i} missing 'timestamp' column.")
        
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        regressor_cols = sorted(
            [col for col in df.columns if 'regressor_' in col],
            key=_extract_param_number
            )
        parameter_cols = sorted(
            [col for col in df.columns if 'parameter_' in col],
            key=_extract_param_number
            )
        if len(regressor_cols) == 0:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_regressor_#' column.")
        if len(parameter_cols) == 0:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_parameter_#' column.")
        if (len(regressor_cols) != len(parameter_cols)):
            raise ValueError(f"DataFrame at index {i} has a mismatched number of regressors ({len(regressor_cols)}) and parameters ({len(parameter_cols)}).")
        
        # modeled output
        regressors = df[regressor_cols].to_numpy()
        parameters = df[parameter_cols].to_numpy()
        modeled_output = np.sum(regressors * parameters, axis=1)
        modeled_output_label = f"{prefix}modeled_output"
        df.insert(loc=2, column=modeled_output_label, value=modeled_output)
        
        # modeled output confidence intervals (cis)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        modeled_output_cis, parameter_cis = _sliding_confidence_intervals(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], regressors, window_size=100)
        modeled_output_cis_label = f"{prefix}modeled_output_cis"
        df.insert(loc=3, column=modeled_output_cis_label, value=modeled_output_cis)
        
        # modeled output confidence interval percentages (cips)
        with np.errstate(divide='ignore', invalid='ignore'):
            modeled_output_cips = (df[f"{prefix}modeled_output_cis"] / df[f"{prefix}modeled_output"]).abs() * 100
            modeled_output_cips = modeled_output_cips.mask(~np.isfinite(modeled_output_cips), np.nan)
        modeled_output_cips_label = f"{prefix}modeled_output_cips"
        df.insert(loc=4, column=modeled_output_cips_label, value=modeled_output_cips)
        
        # residuals
        residuals = df[f"{prefix}measured_output"] - df[f"{prefix}modeled_output"]
        residuals_label = f"{prefix}residuals"
        df.insert(loc=5, column=residuals_label, value=residuals)
        
        # mean squared error (mse)
        mse = _sliding_mse(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=6)
        mse_label = f"{prefix}mse"
        df.insert(loc=6, column=mse_label, value=mse)
        total_mse = _sliding_mse(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=6)
        print(total_mse)
        
        # coefficient of determination (cod)
        num_regressors = len([col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)])
        # cod = _sliding_cod(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=30)
        cod = _sliding_adjusted_cod(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], num_regressors, window_size=100)
        cod_label = f"{prefix}cod"
        df.insert(loc=7, column=cod_label, value=cod)
        
        # regressor coefficient of determinations (cod) from variance inflation factors (vif)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        regressors_cod = _sliding_vif_cod(regressors, window_size=100)
        for j in range(regressors_cod.shape[1]):
            regressors_cod_label = f"{prefix}regressor_{j+1}_cod"
            df.insert(loc=len(df.columns), column=regressors_cod_label, value=regressors_cod[:, j])
        
        # regressor condition numbers (cond) from singular value decomposition (svd)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        regressors_cond = _sliding_svd_cond(regressors, window_size=100)
        for j in range(regressors_cond.shape[1]):
            regressors_cond_label = f"{prefix}regressor_{j+1}_cond"
            df.insert(loc=len(df.columns), column=regressors_cond_label, value=regressors_cond[:, j])
                             
        # paramter confidence intervals (cis)
        for j in range(parameter_cis.shape[1]):
            parameter_cis_label = f"{prefix}parameter_{j+1}_cis"
            df.insert(loc=len(df.columns), column=parameter_cis_label, value=parameter_cis[:, j])
        
        # parameter confidence interval percentages (cips)
        for j in range(parameter_cis.shape[1]):
            with np.errstate(divide='ignore', invalid='ignore'):
                parameter_cips = (df[f"{prefix}parameter_{j+1}_cis"] / df[f"{prefix}parameter_{j+1}"]).abs() * 100
                parameter_cips = parameter_cips.mask(~np.isfinite(parameter_cips), np.nan)
            parameter_cips_label = f"{prefix}parameter_{j+1}_cips"
            df.insert(loc=len(df.columns), column=parameter_cips_label, value=parameter_cips)
            
        # correlation matrix
        regressor_cols = [col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]
        regressors = df[regressor_cols]
        num_regressors = len(regressor_cols)
        correlation_matrix = _sliding_correlation_matrix(regressors, window_size=100)
        for j in range(num_regressors):
            for k in range(num_regressors):
                if j != k:
                    correlation_element = correlation_matrix[:, j, k]
                    correlation_element_label = f"{prefix}correlation_{j}_to_{k}"
                    df.insert(loc=len(df.columns), column=correlation_element_label, value=correlation_element)
        
    return dataframes


def plot_models(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_params = 0
    for i, df in enumerate(dataframes):
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        max_num_params = max(max_num_params, num_params)
    total_subplots = 1 + max_num_params
    fig, axs = plt.subplots(total_subplots, 1, figsize=(12, (2 + 2*max_num_params)), sharex=True)
    
    base_title = "Parameter Estimator Performance - Figure 1"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        measured_output = df[f"{prefix}measured_output"]
        modeled_output = df[f"{prefix}modeled_output"]
        parameters_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        parameters = df[parameters_cols].to_numpy()
        num_params = len(parameters_cols)
    
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1  ---
        if i == 0:
            axs[0].plot(time, measured_output, label='Measured', linestyle='--', color='black')
        axs[0].plot(time, modeled_output, label=f'Estimate: {prefix}', linestyle='-', color=model_color)

        # --- Subplot 2+ ---
        for j in range(num_params):
            axs[1 + j].plot(time, parameters[:, j], label=f'{prefix}', linestyle='-', color=model_color)

    # --- Final Formatting ---
    axs[0].set_title("Measured vs Estimated Outputs")
    axs[0].set_ylabel(plot_labels.get("output_amp", "Output Amplitude"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    axs[0].legend(loc='upper right', fontsize='medium')
    
    for i in range(max_num_params):
        axs[1 + i].set_title(f"Parameter {chr(65+i)} Over Time")
        param_key = f"param_{i+1}_amp"
        default_label = f"Parameter {chr(65+i)}'s\nAmplitude"
        axs[1 + i].set_ylabel(plot_labels.get(param_key, default_label))
    axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    for ax in axs:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True)

def plot_regressor_data(
    dataframes: list[pd.DataFrame],
    start_time: float | None = None,
    end_time: float | None = None,
    plot_labels: dict | None = None
    ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_params = 0
    for i, df in enumerate(dataframes):
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        max_num_params = max(max_num_params, num_params)
    total_subplots = 1 + max_num_params
    fig, axs = plt.subplots(total_subplots, 1, figsize=(12, (2 + 2*max_num_params)), sharex=True)
    
    base_title = "Parameter Estimator Performance - Figure 1"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        measured_output = df[f"{prefix}measured_output"]
        modeled_output = df[f"{prefix}modeled_output"]
        regressors_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)],
            key=_extract_param_number
            )
        regressors = df[regressors_cols].to_numpy()
        num_params = len(regressors_cols)
    
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1  ---
        if i == 0:
            axs[0].plot(time, measured_output, label='Measured', linestyle='--', color='black')
        axs[0].plot(time, modeled_output, label=f'Estimate: {prefix}', linestyle='-', color=model_color)

        # --- Subplot 2+ ---
        for j in range(num_params):
            axs[1 + j].scatter(time, regressors[:, j], label=f'{prefix}', linestyle='-', color=model_color)

    # --- Final Formatting ---
    axs[0].set_title("Measured vs Estimated Outputs")
    axs[0].set_ylabel(plot_labels.get("output_amp", "Output Amplitude"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    axs[0].legend(loc='upper right', fontsize='medium')
    
    for i in range(max_num_params):
        axs[1 + i].set_title(f"Parameter {chr(65+i)} Over Time")
        param_key = f"param_{i+1}_amp"
        default_label = f"Parameter {chr(65+i)}'s\nAmplitude"
        axs[1 + i].set_ylabel(plot_labels.get(param_key, default_label))
    axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    for ax in axs:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True)

def plot_confidence(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_params = 0
    for i, df in enumerate(dataframes):
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        max_num_params = max(max_num_params, num_params)
    total_subplots = 1 + max_num_params
    fig, axs = plt.subplots(total_subplots, 1, figsize=(12, (2 + 2*max_num_params)), sharex=True)
    
    base_title = "Parameter Estimator Performance - Figure 2"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        measured_output = df[f"{prefix}measured_output"]
        modeled_output = df[f"{prefix}modeled_output"]
        modeled_output_cis = df[f"{prefix}modeled_output_cis"]
        parameter_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        parameter_cis_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+_cis", col)],
            key=_extract_param_number
            )
        parameters = df[parameter_cols].to_numpy()
        parameters_cis = df[parameter_cis_cols].to_numpy()
        num_params = len(parameter_cols)
    
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1  ---
        if i == 0:
            axs[0].plot(time, measured_output, label='Measured', linestyle='--', color='black')
        axs[0].plot(time, modeled_output, label=f'Estimate: {prefix}', linestyle='-', color=model_color)
        axs[0].fill_between(time, modeled_output - modeled_output_cis, modeled_output + modeled_output_cis, color=model_color, alpha=0.3, label=f"Estimate: {prefix} CI's")

        # --- Subplot 2+ ---
        for j in range(num_params):
            axs[1 + j].plot(time, parameters[:, j], label=f'{prefix}', linestyle='-', color=model_color)
            axs[1 + j].fill_between(time, parameters[:, j] - parameters_cis[:, j], parameters[:, j] + parameters_cis[:, j], color=model_color, alpha=0.3, label=f"{prefix} CI's")

    # --- Final Formatting ---
    axs[0].set_title("Measured vs Estimated Outputs")
    axs[0].set_ylabel(plot_labels.get("output_amp", "Output Amplitude"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    axs[0].legend(loc='upper right', fontsize='medium')
    
    for i in range(max_num_params):
        axs[1 + i].set_title(f"Parameter {chr(65+i)} Over Time")
        param_key = f"param_{i+1}_amp"
        default_label = f"Parameter {chr(65+i)}'s\nAmplitude"
        axs[1 + i].set_ylabel(plot_labels.get(param_key, default_label))
    axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    for ax in axs:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True)

def plot_percent_confidence(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_params = 0
    for i, df in enumerate(dataframes):
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        max_num_params = max(max_num_params, num_params)
    total_subplots = 1 + max_num_params
    fig, axs = plt.subplots(total_subplots, 1, figsize=(12, (2 + 2*max_num_params)), sharex=True)
    
    base_title = "Parameter Estimator Performance - Figure 3"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        modeled_output_cips = df[f"{prefix}modeled_output_cips"]
        parameter_cips_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+_cips", col)],
            key=_extract_param_number
            )
        parameters_cips = df[parameter_cips_cols].to_numpy()
        num_params = len(parameter_cips_cols)
    
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1  ---
        axs[0].plot(time, modeled_output_cips, label=f'Estimate: {prefix}', linestyle='-', color=model_color)

        # --- Subplot 2+ ---
        for j in range(num_params):
            axs[1 + j].plot(time, parameters_cips[:, j], label=f'{prefix}', linestyle='-', color=model_color)

    # --- Final Formatting ---
    axs[0].set_title("Estimated Output's Percent Confidence Over Time")
    axs[0].set_ylabel(plot_labels.get("output_percent_confidence", "Output Percent\nConfidence [%]"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    axs[0].legend(loc='upper right', fontsize='medium')
    
    for i in range(max_num_params):
        axs[1 + i].set_title(f"Parameter {chr(65+i)}'s Percent Confidence Over Time")
        param_key = f"param_{i+1}_percent_confidence"
        default_label = f"Parameter {chr(65+i)}'s\nPercent\nConfidence [%]"
        axs[1 + i].set_ylabel(plot_labels.get(param_key, default_label))
    axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    for ax in axs:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True)

def plot_error(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax_upper_left = fig.add_subplot(gs[0, 0])
    ax_lower_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[:, 1])
    axs = [ax_upper_left, ax_lower_left, ax_right]
    
    base_title = "Parameter Estimator Performance - Figure 4"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        measured_output = df[f"{prefix}measured_output"]
        residuals = df[f"{prefix}residuals"]
        mse = df[f"{prefix}mse"]
    
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1 (upper left) ---
        axs[0].scatter(time, residuals, label=prefix, s=10, alpha=0.7, color=model_color)
        axs[0].plot(time, residuals, linewidth=1, alpha=0.3, color=model_color)
        
        # --- Subplot 2 (lower left) ---
        axs[1].scatter(time, mse, label=prefix, s=10, alpha=0.7, color=model_color)
        axs[1].plot(time, mse, linewidth=1, alpha=0.3, color=model_color)
        
        # --- Subplot 3 (right) ---
        axs[2].scatter(measured_output, residuals, label=prefix, s=10, alpha=0.7, color=model_color)
        
    # --- Final Formatting ---
    axs[0].set_title("Residuals Over Time")
    axs[0].set_ylabel(plot_labels.get("residuals", "Residuals"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    axs[1].set_title("Mean Squared Error (MSE) Over Time")
    axs[1].set_ylabel(plot_labels.get("mse", "MSE"))
    axs[1].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    axs[2].set_title("Residuals vs Measured Output")
    axs[2].set_ylabel(plot_labels.get("residuals", "Residuals"))
    axs[2].set_xlabel(plot_labels.get("measured_output", "Measured Output"))
    lims = [
        min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]),
        max(axs[2].get_xlim()[1], axs[2].get_ylim()[1])
        ]
    axs[2].set_xlim(lims[0], lims[1])
    axs[2].set_ylim(lims[0], lims[1])
    axs[2].set_aspect('equal')
    
    for ax in axs:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True)
        axs[0].legend(loc='upper right', fontsize='medium')

def plot_fit(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_params = 0
    for i, df in enumerate(dataframes):
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        max_num_params = max(max_num_params, num_params)
    total_subplots = 1 + max_num_params
    fig, axs = plt.subplots(total_subplots, 1, figsize=(12, (2 + 2*max_num_params)), sharex=True)
    
    base_title = "Parameter Estimator Performance - Figure 5"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        cod = df[f"{prefix}cod"]
        regressor_cod_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}regressor_\d+_cod", col)],
            key=_extract_param_number
            )
        regressors_cod = df[regressor_cod_cols].to_numpy()
        num_params = len(regressor_cod_cols)
        
        measured_output = df[f"{prefix}measured_output"]
        modeled_output = df[f"{prefix}modeled_output"]
        regressor_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)],
            key=_extract_param_number
            )
        regressors = df[regressor_cols]
        cod_batch = _sliding_adjusted_cod(measured_output, modeled_output, num_params, len(measured_output))[-1]
        regressors_cod_batch = _sliding_vif_cod(regressors, len(regressors))[-1]
        
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1  ---
        axs[0].plot(time, cod, label=f'Estimate: {prefix}', linestyle='-', color=model_color)
        axs[0].axhline(cod_batch, label=f'Batched: {prefix}', linestyle=':', color=model_color, alpha=0.7)

        # --- Subplot 2+ ---
        for j in range(num_params):
            axs[1 + j].plot(time, regressors_cod[:, j], label=f'{prefix}', linestyle='-', color=model_color)
            axs[1 + j].axhline(regressors_cod_batch[j], label=f'{prefix}', linestyle=':', color=model_color, alpha=0.7)

    # --- Final Formatting ---
    axs[0].set_title("Coefficient of Determination (R²) Over Time")
    axs[0].set_ylabel(plot_labels.get("cod_amp", "R² [%]"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    axs[0].legend(loc='upper right', fontsize='medium')
    axs[0].set_ylim(0, 1)
    axs[0].axhspan(0.75, 1.00, color='#A8D5BA', alpha=0.3)  # green
    axs[0].axhspan(0.50, 0.75, color='#FFF3B0', alpha=0.3)  # yellow
    axs[0].axhspan(0.00, 0.50, color='#F4CCCC', alpha=0.3)  # red
    
    for i in range(max_num_params):
        axs[1 + i].set_title(f"Parameter {chr(65+i)}'s Fit (r²) Over Time")
        param_key = f"param_{i+1}_cod_amp"
        default_label = f"Parameter {chr(65+i)}'s\nr² [%]"
        axs[1 + i].set_ylabel(plot_labels.get(param_key, default_label))
        axs[1 + i].set_ylim(0, 1)
        axs[1 + i].axhspan(0.90, 1.00, color='#F4CCCC', alpha=0.3)  # red
        axs[1 + i].axhspan(0.75, 0.90, color='#FFF3B0', alpha=0.3)  # yellow
        axs[1 + i].axhspan(0.00, 0.75, color='#A8D5BA', alpha=0.3)  # green
    axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    for ax in axs:
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True)

def plot_conditioning(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_params = 0
    for i, df in enumerate(dataframes):
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        max_num_params = max(max_num_params, num_params)
    total_subplots = 1 + max_num_params
    fig, axs = plt.subplots(total_subplots, 1, figsize=(12, (2 + 2*max_num_params)), sharex=True)
    
    base_title = "Parameter Estimator Performance - Figure 6"
    subtitle = plot_labels.get("subtitle", "Multiple Models")
    full_title = f"{base_title}\n{subtitle}" if subtitle else base_title
    fig.suptitle(full_title, fontsize=14, weight='bold')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
            
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        regressor_cond_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}regressor_\d+_cond", col)],
            key=_extract_param_number
            )
        regressors_cond = df[regressor_cond_cols].to_numpy()
        num_params = len(regressor_cond_cols)
        max_cond = df[regressor_cond_cols].max(axis=1).to_numpy()
        
        regressor_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)],
            key=_extract_param_number
            )
        regressors = df[regressor_cols]
        regressors_cond_batch = _sliding_svd_cond(regressors, len(regressors))[-1]
        max_cond_batch = max(regressors_cond_batch)
    
        model_color = color_cycle[i % len(color_cycle)]
        
        # --- Subplot 1  ---
        axs[0].plot(time, max_cond, label=f'Estimate: {prefix}', linestyle='-', color=model_color)
        axs[0].axhline(max_cond_batch, label=f'Batched: {prefix}', linestyle=':', color=model_color, alpha=0.7)

        # --- Subplot 2+ ---
        for j in range(num_params):
            axs[1 + j].plot(time, regressors_cond[:, j], label=f'{prefix}', linestyle='-', color=model_color)
            axs[1 + j].axhline(regressors_cond_batch[j], label=f'{prefix}', linestyle=':', color=model_color, alpha=0.7)

    # --- Final Formatting ---
    axs[0].set_title("Max Condition Number Over Time")
    axs[0].set_ylabel(plot_labels.get("cond_amp", "Amplitude"))
    axs[0].set_xlabel(plot_labels.get("time", "Time [s]"))
    axs[0].legend(loc='upper right', fontsize='medium')
    
    for i in range(max_num_params):
        axs[1 + i].set_title(f"Parameter {chr(65+i)} Over Time")
        param_key = f"param_{i+1}_cond_amp"
        default_label = "Amplitude"
        axs[1 + i].set_ylabel(plot_labels.get(param_key, default_label))
    axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))
    
    for ax in axs:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_yscale("log")
        ax.grid(True)

def plot_correlation(
        dataframes: list[pd.DataFrame],
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
    
    for i, df in enumerate(dataframes):
        if df.empty:
            print(f"Skipping empty DataFrame at index {i}")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame at index {i} due to missing 'timestamp'")
            continue
        
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame at index {i} missing a 'prefix_measured_output' column.")
        param_cols = sorted(
            [col for col in df.columns if re.match(rf"{prefix}parameter_\d+$", col)],
            key=_extract_param_number
            )
        num_params = len(param_cols)
        fig, axs = plt.subplots(num_params, 1, figsize=(12, (2*num_params)), sharex=True)
        if isinstance(axs, np.ndarray):
            axs = axs.tolist()
        else:
            axs = [axs]
        
        base_title = "Parameter Estimator Performance - Figure 7"
        subtitle = plot_labels.get("subtitle")
        full_title = f"{base_title}\n{subtitle} - {prefix}" if subtitle else f"{base_title}\n{prefix}"
        fig.suptitle(full_title, fontsize=14, weight='bold')
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        time = df["timestamp"]
        
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        correlation_matrix_batch = _sliding_correlation_matrix(regressors, len(regressors))[-1]
        
        for j in range(num_params):
            ax = axs[j]
            for k in range(num_params):
                if j == k:
                    continue
                
                param_color = color_cycle[k % len(color_cycle)]
                
                col_name = f"{prefix}correlation_{j}_to_{k}"
                if col_name in df.columns:
                    ax.plot(time, df[col_name], label=f"corr({chr(65+j)}, {chr(65+k)})", color=param_color)
                    
                ax.axhline(correlation_matrix_batch[j, k], linestyle=':', color=param_color, alpha=0.7)
                    
            ax.set_title(f"Parameter {chr(65 + j)}'s Correlation Over Time")
            ax.set_ylabel(f"Param {chr(65 + j)}\nCorrelation")
            ax.legend(loc='upper right', fontsize='medium')
            ax.grid(True)
            ax.set_ylim(-1, 1)
            ax.axhspan( 0.90,  1.00, color='#F4CCCC', alpha=0.3)  # red
            ax.axhspan( 0.75,  0.90, color='#FFF3B0', alpha=0.3)  # yellow
            ax.axhspan(-0.75,  0.75, color='#A8D5BA', alpha=0.3)  # green
            ax.axhspan(-0.90, -0.75, color='#FFF3B0', alpha=0.3)  # yellow
            ax.axhspan(-1.00, -0.90, color='#F4CCCC', alpha=0.3)  # red
            
        axs[-1].set_xlabel(plot_labels.get("time", "Time [s]"))

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        for ax in axs:
            ax.yaxis.set_major_formatter(formatter)

def plot_filter_duration(
        dataframe: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        plot_labels: Optional[dict] = None
        ) -> PlotFigure:

    if dataframe is None or dataframe.empty or 'timestamp' not in dataframe.columns:
        raise ValueError("Invalid DataFrame provided.")
    if plot_labels is None:
        plot_labels = {}
        
    if start_time is not None:
        dataframe = dataframe[dataframe["timestamp"] >= start_time]
    if end_time is not None:
        dataframe = dataframe[dataframe["timestamp"] <= end_time]

    time = dataframe["timestamp"]
    elapsed = dataframe["elapsed"] * 1000
    ema_elapsed = dataframe["ema_elapsed"] * 1000
    max_elapsed = dataframe["max_elapsed"] * 1000
    min_elapsed = dataframe["min_elapsed"] * 1000

    fig = PlotFigure(nrows=1, ncols=1, figsize=(12, 6), sharex=True)
    base_title = "Filter Performance - Duration"
    subtitle = plot_labels.get("subtitle", "Last Test")
    fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

    fig.define_subplot(0, title="Filter Duration Over Time", ylabel="Time\n[ms]", xlabel="Time [s]")
    fig.add_scatter(0, time, elapsed, color='tab:blue')
    fig.add_data(0, time, ema_elapsed, color='black')
    fig.add_fill_between(0, time, max_elapsed, min_elapsed, "Bounds", color="tab:blue")

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

if __name__ == "__main__":
    csv_path = "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/ols_rol_nondim_data.csv"
    
    models = ['ols_rol_nondim_']
    
    start_time = 0
    end_time = 999999
    # TODO: Add model labels as well
    plot_labels = {
    "subtitle": "Roll Models",
    "time": "Time [s]",
    "measured_output": "Measured Roll\nAcceleration [deg/s²]",
    "output_amp": "Roll Acceleration\n[deg/s²]",
    "output_percent_confidence": "Confidence [%]",
    "cod_amp": "R²",
    "residuals": "Roll Acceleration\nResiduals [deg/s²]",
    "mse": "Roll Acceleration\nSquared Error\n[(deg/s²)²]",
    "param_1_amp": "Roll Velocity\nParameter [1/s]",
    "param_2_amp": "Aileron Parameter\n[1/s²]",
    "param_3_amp": "Yaw Velocity\nParameter [1/s]",
    "param_4_amp": "Rudder Parameter\n[1/s²]",
    "param_1_cod_amp": "Roll Velocity\nParameter's r²\n[%]",
    "param_2_cod_amp": "Aileron\nParameter's r²\n[%]",
    "param_3_cod_amp": "Yaw Velocity\nParameter's r²\n[%]",
    "param_4_cod_amp": "Rudder\nParameter's r²\n[%]",
    "param_1_cond_amp": "Roll Velocity\nParameter's\nConditioning",
    "param_2_cond_amp": "Aileron\nParameter's\nConditioning",
    "param_3_cond_amp": "Yaw Velocity\nParameter's\nConditioning",
    "param_4_cond_amp": "Rudder\nParameter's\nConditioning",
    }

    csv = pd.read_csv(csv_path)
    model_dfs = {prefix: extract_model(csv, prefix) for prefix in models}
    processed_models = process_models(list(model_dfs.values()))
    # TODO: ADD BATCH RESULTS!
    plot_models(processed_models, start_time, end_time, plot_labels)
    plot_regressor_data(processed_models, start_time, end_time, plot_labels)  # TODO: Fix labels of this plotter function
    # plot_confidence(processed_models,  start_time, end_time, plot_labels)
    # TODO: Recenter around data, not confidence intervals.
    # plot_percent_confidence(processed_models,  start_time, end_time, plot_labels)
    plot_error(processed_models, start_time, end_time, plot_labels)
    # plot_fit(processed_models,  start_time, end_time, plot_labels)
    # TODO: Review the R² calculations.
    # plot_conditioning(processed_models, start_time, end_time, plot_labels)
    # plot_correlation(processed_models, start_time, end_time, plot_labels)

    # TODO: Add FFT plotter
    # TODO: Add Bode plotter

    plot_filter_duration(pd.read_csv("/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/filt_duration_data.csv"))
    plot_filter_duration(pd.read_csv("/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/diff_duration_data.csv"))

    plt.show()