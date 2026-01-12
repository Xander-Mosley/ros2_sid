#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional, Sequence, Union

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

# def _extract_param_number(col: str) -> Union[int, float]:
#     match = re.search(r'parameter_(\d+)', col)
#     return int(match.group(1)) if match else float('inf')


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

def process_models(dataframes: Dict[str, pd.DataFrame]):
    for name, df in dataframes.items():
        if df.empty:
            raise ValueError(f"DataFrame named {name} is empty.")
        if 'timestamp' not in df.columns:
            raise ValueError(f"DataFrame named {name} missing 'timestamp' column.")
        
        prefix = next((match.group(1) for col in df.columns if (match := re.fullmatch(r"(.*)measured_output", col))), None)
        if prefix is None:
            raise ValueError(f"DataFrame named {name} missing a 'prefix_measured_output' column.")
        
        regressor_cols = sorted(
            [col for col in df.columns if 'regressor_' in col],
            key=_extract_param_number
            )
        parameter_cols = sorted(
            [col for col in df.columns if 'parameter_' in col],
            key=_extract_param_number
            )
        if len(regressor_cols) == 0:
            raise ValueError(f"DataFrame named {name} missing a 'prefix_regressor_#' column.")
        if len(parameter_cols) == 0:
            raise ValueError(f"DataFrame named {name} missing a 'prefix_parameter_#' column.")
        if (len(regressor_cols) != len(parameter_cols)):
            raise ValueError(f"DataFrame named {name} has a mismatched number of regressors ({len(regressor_cols)}) and parameters ({len(parameter_cols)}).")
        
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





def _extract_model_prefix(df: pd.DataFrame) -> str:
    prefix = next(
        (match.group(1)
         for col in df.columns
         if (match := re.fullmatch(r"(.*)measured_output", col))),
        None
    )
    if prefix is None:
        raise ValueError("Missing '<prefix>measured_output' column.")
    return prefix

def _extract_param_number(col: str) -> Union[int, float]:
    match = re.search(r'parameter_(\d+)', col)
    return int(match.group(1)) if match else float('inf')

def _get_indexed_columns(
        df: pd.DataFrame,
        prefix: str,
        pattern: str
        ) -> list[str]:
    regex = rf"{prefix}{pattern}"
    return sorted(
        (col for col in df.columns if re.match(regex, col)),
        key=_extract_param_number
    )

def _get_max_num_terms(dataframes: dict[str, pd.DataFrame]) -> int:
    max_num_terms = 0
    for name, df in dataframes.items():
        param_cols = _get_indexed_columns(df, _extract_model_prefix(df), r"parameter_\d+$")
        max_num_terms = max(max_num_terms, len(param_cols))
    return max_num_terms


def plot_parameter_data(
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(dataframes)
    terms = plot_labels.get("terms", {})

    fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
    base_title = "Model Performance - Parameters"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    units = term_info.get("units", "")
    fig.define_subplot(0, title=f"Measured vs Modeled {term_name}", ylabel=f"{term_name}'s\nAmplitude\n{units}")
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term_name = term_info.get("term")
        param_units = term_info.get("param_units", "")
        if term_name:
            title = f"{term_name}'s Parameter Over Time"
            ylabel = f"{term_name}'s\nParameter Amplitude\n{param_units}"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"{term_name} Over Time"
            ylabel = f"{term_name}'s\nAmplitude\n{param_units}"
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(dataframes.keys())
    }

    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        prefix = _extract_model_prefix(df)
        parameter_cols = _get_indexed_columns(df, prefix, r"parameter_\d+$")

        time = df["timestamp"]
        modeled_output = df[f"{prefix}modeled_output"]
        parameters = df[parameter_cols].to_numpy()
        num_terms = len(parameter_cols)

        if i == 0:
            measured_output = df[f"{prefix}measured_output"]
            fig.add_data(0, time, measured_output, label='Measured', color='black', linestyle='--')
        fig.add_data(0, time, modeled_output, label=name, color=model_colors[name])
        
        for j in range(num_terms):
            fig.add_data(1+j, time, parameters[:, j], label=name, color=model_colors[name], linestyle='-', marker='.')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_regressor_data(
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(dataframes)
    terms = plot_labels.get("terms", {})

    fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
    base_title = "Model Performance - Regressors"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    term_info = terms.get(0, {})
    term = term_info.get("term", "Output")
    units = term_info.get("units", "")
    fig.define_subplot(0, title=f"Measured vs Modeled {term}", ylabel=f"{term}'s\nAmplitude\n{units}")
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term = term_info.get("term", f"Regressor {chr(65+i)}")
        units = term_info.get("units", "")
        fig.define_subplot(1 + i,
                        title=f"{term} Over Time",
                        ylabel=f"{term}'s\n Amplitude\n{units}",
                        xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None
                        )
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(dataframes.keys())
    }

    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        prefix = _extract_model_prefix(df)
        regressor_cols = _get_indexed_columns(df, prefix, r"regressor_\d+$")

        time = df["timestamp"]
        modeled_output = df[f"{prefix}modeled_output"]
        regressors = df[regressor_cols].to_numpy()
        num_terms = len(regressor_cols)

        if i == 0:
            measured_output = df[f"{prefix}measured_output"]
            fig.add_data(0, time, measured_output, label='Measured', color='black', linestyle='--')
        fig.add_data(0, time, modeled_output, label=name, color=model_colors[name])
        
        for j in range(num_terms):
            fig.add_data(1+j, time, regressors[:, j], label=name, color=model_colors[name], linestyle='-', marker='.')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_confidence(
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(dataframes)
    terms = plot_labels.get("terms", {})

    fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
    base_title = "Model Performance - Confidence Intervals (CIs)"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    units = term_info.get("units", "")
    fig.define_subplot(0, title=f"Measured vs Modeled {term_name}", ylabel=f"{term_name}'s\nAmplitude\n{units}")
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term_name = term_info.get("term")
        param_units = term_info.get("param_units", "")
        if term_name:
            title = f"{term_name}'s Parameter Over Time"
            ylabel = f"{term_name}'s\nParameter Amplitude\n{param_units}"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"{term_name} Over Time"
            ylabel = f"{term_name}'s\nAmplitude\n{param_units}"
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(dataframes.keys())
    }
    
    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        prefix = _extract_model_prefix(df)
        parameter_cols = _get_indexed_columns(df, prefix, r"parameter_\d+$")
        parameter_cis_cols = _get_indexed_columns(df, prefix, r"parameter_\d+_cis")

        time = df["timestamp"]
        modeled_output = df[f"{prefix}modeled_output"]
        modeled_output_cis = df[f"{prefix}modeled_output_cis"]
        parameters = df[parameter_cols].to_numpy()
        parameters_cis = df[parameter_cis_cols].to_numpy()
        num_params = len(parameter_cols)
        
        if i == 0:
            measured_output = df[f"{prefix}measured_output"]
            fig.add_data(0, time, measured_output, label='Measured', color='black', linestyle='--')
        fig.add_data(0, time, modeled_output, color=model_colors[name])
        fig.add_fill_between(0, time, modeled_output - modeled_output_cis, modeled_output + modeled_output_cis, label=f"{name}'s CIs", color=model_colors[name], alpha=0.3)
        fig.autoscale_from_lines(0)

        for j in range(num_params):
            fig.add_data(1+j, time, parameters[:, j], color=model_colors[name], linestyle='-', marker='.')
            fig.add_fill_between(1+j, time, parameters[:, j] - parameters_cis[:, j], parameters[:, j] + parameters_cis[:, j], label=f"{name}'s CIs", color=model_colors[name], alpha=0.3)
            fig.autoscale_from_lines(1+j)


    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_percent_confidence(
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ):
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(dataframes)
    terms = plot_labels.get("terms", {})

    fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
    base_title = "Model Performance - Percent Confidence"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    fig.define_subplot(0, title=f"Percent Confidence of Modeled {term_name} Over Time", ylabel=f"{term_name}'s\nPercent Confidence\n[%]")
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"Percent Confidence of {term_name}'s Parameter Over Time"
            ylabel = f"{term_name}'s\nPercent Confidence\n[%]"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"Percent Confidence of {term_name} Over Time"
            ylabel = f"{term_name}'s\nPercent Confidence\n[%]"
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(dataframes.keys())
    }
    
    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        prefix = _extract_model_prefix(df)
        parameter_cips_cols = _get_indexed_columns(df, prefix, r"parameter_\d+_cips")

        time = df["timestamp"]
        modeled_output_cips = df[f"{prefix}modeled_output_cips"]
        parameters_cips = df[parameter_cips_cols].to_numpy()
        num_params = len(parameter_cips_cols)
        
        fig.add_data(0, time, modeled_output_cips, label=name, color=model_colors[name])

        for j in range(num_params):
            fig.add_data(1+j, time, parameters_cips[:, j], label=name, color=model_colors[name])

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig


def plot_error(
        dataframes: dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> None:
    
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
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(dataframes)
    terms = plot_labels.get("terms", {})

    fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
    base_title = "Model Performance - Model Fit"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    term_info = terms.get(0, {})
    fig.define_subplot(0, title="Coefficient of Determination (R²) Over Time", ylabel="R² [%]")
    fig.shade_subplot(0, y_range=(0.75, 1.00), color='#A8D5BA', alpha=0.3)  # green
    fig.shade_subplot(0, y_range=(0.50, 0.75), color='#FFF3B0', alpha=0.3)  # yellow
    fig.shade_subplot(0, y_range=(0.00, 0.50), color='#F4CCCC', alpha=0.3)  # red
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"{term_name}'s Parameter Fit (r²) Over Time"
            ylabel = f"{term_name}'s\nParameter r² [%]"
        else:
            term_name = f"Parameter {chr(65+i)}'s Fit (r²) Over Time"
            title = f"{term_name} Over Time"
            ylabel = f"{term_name}'s\nr² [%]"
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
        fig.shade_subplot(1 + i, y_range=(0.75, 1.00), color='#F4CCCC', alpha=0.3)  # red
        fig.shade_subplot(1 + i, y_range=(0.50, 0.75), color='#FFF3B0', alpha=0.3)  # yellow
        fig.shade_subplot(1 + i, y_range=(0.00, 0.50), color='#A8D5BA', alpha=0.3)  # green
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(dataframes.keys())
    }

    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        prefix = _extract_model_prefix(df)
        regressor_cols = _get_indexed_columns(df, prefix, r"regressor_\d+$")
        regressor_cod_cols = _get_indexed_columns(df, prefix, r"regressor_\d+_cod")

        time = df["timestamp"]
        cod = df[f"{prefix}cod"]
        regressors_cod = df[regressor_cod_cols].to_numpy()
        num_params = len(regressor_cod_cols)
        
        measured_output = df[f"{prefix}measured_output"]
        modeled_output = df[f"{prefix}modeled_output"]
        regressors = df[regressor_cols].to_numpy()
        cod_batch = _sliding_adjusted_cod(measured_output, modeled_output, num_params, len(measured_output))[-1]
        regressors_cod_batch = _sliding_vif_cod(regressors, len(regressors))[-1]
        
        fig.add_data(0, time, cod, label=name, color=model_colors[name], marker='.')
        fig.add_line(0, cod_batch, 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

        for j in range(num_params):
            fig.add_data(1+j, time, regressors_cod[:, j], label=name, color=model_colors[name], marker='.')
            fig.add_line(1+j, regressors_cod_batch[j], 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    fig.apply_to_all_axes('set_ylim', [0, 1])
    return fig

def plot_conditioning(
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(dataframes)
    terms = plot_labels.get("terms", {})

    fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
    base_title = "Model Performance - Conditioning"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    fig.define_subplot(0, title="Max Condition Number Over Time", ylabel="Amplitude")
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"{term_name}'s Parameter Conditioning Over Time"
            ylabel = "Amplitude"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"{term_name}'s Conditioning Over Time"
            ylabel = "Amplitude"
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(dataframes.keys())
    }

    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]

        prefix = _extract_model_prefix(df)
        regressor_cols = _get_indexed_columns(df, prefix, r"regressor_\d+$")
        regressor_cond_cols = _get_indexed_columns(df, prefix, r"regressor_\d+_cond")

        time = df["timestamp"]
        regressors_cond = df[regressor_cond_cols].to_numpy()
        num_params = len(regressor_cond_cols)
        max_cond = df[regressor_cond_cols].max(axis=1).to_numpy()
 
        regressors = df[regressor_cols].to_numpy()
        regressors_cond_batch = _sliding_svd_cond(regressors, len(regressors))[-1]
        max_cond_batch = max(regressors_cond_batch)
        
        fig.add_data(0, time, max_cond, label=name, color=model_colors[name])
        fig.add_line(0, max_cond_batch, 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

        for j in range(num_params):
            fig.add_data(1+j, time, regressors_cond[:, j], label=name, color=model_colors[name])
            fig.add_line(1+j, regressors_cond_batch[j], 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_correlation(
        dataframes: Dict[str, pd.DataFrame],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> list[PlotFigure]:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}

    figures: list[PlotFigure] = []
    
    for i, (name, df) in enumerate(dataframes.items()):
        if df.empty:
            print(f"Skipping empty DataFrame named {name}.")
            continue
        if 'timestamp' not in df.columns:
            print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
            continue
        
        if start_time is not None:
            df = df[df["timestamp"] >= start_time]
        if end_time is not None:
            df = df[df["timestamp"] <= end_time]
        
        prefix = _extract_model_prefix(df)
        parameter_cols =_get_indexed_columns(df, prefix, r"parameter_\d+$")
        max_num_terms = len(parameter_cols)

        fig = PlotFigure(nrows=max_num_terms, ncols=1, figsize=(12, (2*max_num_terms)), sharex=True)
        base_title = "Model Performance - Regressor Correlations"
        subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
        fig.set_figure_title(f"{base_title}\n{subtitle}")

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        time = df["timestamp"]
        
        regressors = df[_get_indexed_columns(df, prefix, r"regressor_\d+$")].to_numpy()
        correlation_matrix_batch = _sliding_correlation_matrix(regressors, len(regressors))[-1]
        
        for j in range(max_num_terms):
            title = f"Parameter {chr(65 + j)}'s Correlation Over Time"
            ylabel = f"Param {chr(65 + j)}\nCorrelation"
            fig.define_subplot(j, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if j == max_num_terms - 1 else None)
            fig.shade_subplot(j, y_range=( 0.90,  1.00), color='#F4CCCC', alpha=0.3)  # red
            fig.shade_subplot(j, y_range=( 0.75,  0.90), color='#FFF3B0', alpha=0.3)  # yellow
            fig.shade_subplot(j, y_range=(-0.75,  0.75), color='#A8D5BA', alpha=0.3)  # green
            fig.shade_subplot(j, y_range=(-0.90, -0.75), color='#FFF3B0', alpha=0.3)  # yellow
            fig.shade_subplot(j, y_range=(-1.00, -0.90), color='#F4CCCC', alpha=0.3)  # red

            for k in range(max_num_terms):
                if j == k:
                    continue
                
                param_color = color_cycle[k % len(color_cycle)]
                
                col_name = f"{prefix}correlation_{j}_to_{k}"
                if col_name in df.columns:
                    fig.add_data(j, time, df[col_name], label=f"corr({chr(65+j)}, {chr(65+k)})", color=param_color)
                    
                fig.add_line(j, correlation_matrix_batch[j, k], 'h', color=param_color, linestyle=':', alpha=0.7)

        fig.set_all_legends(loc='upper right', fontsize='medium')
        fig.set_all_grids(True, alpha=0.5)
        fig.apply_to_all_axes('set_ylim', [-1, 1])

        figures.append(fig)

    return figures


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




def plot_models(csv_files, start_time, end_time, plot_labels, separate = False):
    model_dataframes: Dict[str, pd.DataFrame] = {}
    for name, info in csv_files.items():
        try:
            df = pd.read_csv(info["path"])
            model_dataframes[name] = extract_model(df, info["prefix"])
        except Exception as error:
            raise RuntimeError(f"Failed processing '{name}'") from error
    processed_models = process_models(model_dataframes)

    # TODO: Allow for separate figures to be plotted. Simple if-else statement once all plots are complete.
    # plot_parameter_data(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)  # TODO: Add batch results.
    # plot_regressor_data(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_confidence(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_percent_confidence(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)  # TODO: Add batch results.
    plot_error(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)   # TODO: Need to add the gridspec to the PlotFigure class (if possible).
    # plot_fit(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)  # TODO: Review the R² method.
    # plot_conditioning(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_correlation(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    plt.show()
    
    # TODO: Add FFT plotter, Bode plots, and 3D RFT progressions

def main():
    # TODO: Isolated this into three scripts: model_processing, model_performance, and model_spectrums.
    csv_files = {
        "Small Roll": {"prefix": "ols_rol_", "path": "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/ols_rol_data.csv"},
    }

    start_time = 0
    end_time = 999999

    plot_labels = {
        "subtitle": "",
        "time": "Time [s]",

        "terms":{
            0: {"term": "Roll Acceleration", "units": "[rad/s²]"},
            1: {"term": "Roll Rate", "units": "[rad/s]", "param_units": "[1/s]"},
            2: {"term": "Aileron Command", "units": "[PWM]", "param_units": "[rad/s²-PWM]"},
        },
    }

    plot_models(csv_files, start_time, end_time, plot_labels)

if __name__ == "__main__":
    main()