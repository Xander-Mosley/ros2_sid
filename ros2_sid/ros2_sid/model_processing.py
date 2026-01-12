#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system

ArrayLike = Union[float, Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]


__all__ = ['extract_model', 'process_models',
    'sliding_mse', 'sliding_cod', 'sliding_adjusted_cod', 'sliding_confidence_intervals', 'sliding_vif_cod', 'sliding_svd_cond', 'sliding_correlation_matrix']
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


def sliding_mse(
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

def sliding_cod(
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

def sliding_adjusted_cod(
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

def sliding_confidence_intervals(
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

def sliding_vif_cod(
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

def sliding_svd_cond(
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

def sliding_correlation_matrix(
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
        modeled_output_cis, parameter_cis = sliding_confidence_intervals(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], regressors, window_size=100)
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
        mse = sliding_mse(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=6)
        mse_label = f"{prefix}mse"
        df.insert(loc=6, column=mse_label, value=mse)
        total_mse = sliding_mse(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=6)
        
        # coefficient of determination (cod)
        num_regressors = len([col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)])
        # cod = sliding_cod(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], window_size=30)
        cod = sliding_adjusted_cod(df[f"{prefix}measured_output"], df[f"{prefix}modeled_output"], num_regressors, window_size=100)
        cod_label = f"{prefix}cod"
        df.insert(loc=7, column=cod_label, value=cod)
        
        # regressor coefficient of determinations (cod) from variance inflation factors (vif)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        regressors_cod = sliding_vif_cod(regressors, window_size=100)
        for j in range(regressors_cod.shape[1]):
            regressors_cod_label = f"{prefix}regressor_{j+1}_cod"
            df.insert(loc=len(df.columns), column=regressors_cod_label, value=regressors_cod[:, j])
        
        # regressor condition numbers (cond) from singular value decomposition (svd)
        regressors = df[[col for col in df.columns if re.match(rf"{prefix}regressor_\d+$", col)]]
        regressors_cond = sliding_svd_cond(regressors, window_size=100)
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
        correlation_matrix = sliding_correlation_matrix(regressors, window_size=100)
        for j in range(num_regressors):
            for k in range(num_regressors):
                if j != k:
                    correlation_element = correlation_matrix[:, j, k]
                    correlation_element_label = f"{prefix}correlation_{j}_to_{k}"
                    df.insert(loc=len(df.columns), column=correlation_element_label, value=correlation_element)
        
    return dataframes


if __name__ == "__main__":
    pass