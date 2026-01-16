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

from model_processing import (extract_model, process_models,
                              sliding_adjusted_cod, sliding_vif_cod, sliding_svd_cond, sliding_correlation_matrix)
from plotter_class import PlotFigure

ArrayLike = Union[float, Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]


# __all__ = ['']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


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
        ) -> PlotFigure:
    
    if not dataframes:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}

    layout = {
        0: (slice(0, 1), slice(0, 1)),   # upper-left
        1: (slice(1, 2), slice(0, 1)),   # lower-left
        2: (slice(0, 2), slice(1, 2)),   # right (spans rows)
    }
    fig = PlotFigure(nrows=2, ncols=2, figsize=(12, 6), gridspec=True, layout=layout)
    base_title = "Model Performance - Error"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    units = term_info.get("units", "")
    
    fig.define_subplot(
        0,
        title="Residuals Over Time",
        ylabel=f"Residuals {units}",
        xlabel=plot_labels.get("time", "Time [s]"),
        grid=True,
    )
    fig.define_subplot(
        1,
        title="Mean Squared Error (MSE) Over Time",
        ylabel="MSE",
        xlabel=plot_labels.get("time", "Time [s]"),
        grid=True,
    )
    fig.define_subplot(
        2,
        title="Residuals vs Measured Output",
        ylabel=f"Residuals {units}",
        xlabel=f"Measured Output {units}",
        grid=True,
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

        time = df["timestamp"]
        measured_output = df[f"{prefix}measured_output"]
        residuals = df[f"{prefix}residuals"]
        mse = df[f"{prefix}mse"]
        
        # --- Subplot 1 (upper left) ---
        fig.add_scatter(0, time, residuals, label=name, color=model_colors[name], s=10, alpha=0.7)
        fig.add_data(0, time, residuals, color=model_colors[name], alpha=0.3, linewidth=1)
        
        # --- Subplot 2 (lower left) ---
        fig.add_scatter(1, time, mse, label=name, color=model_colors[name], s=10, alpha=0.7)
        fig.add_data(1, time, mse, color=model_colors[name], alpha=0.3, linewidth=1)
        
        # --- Subplot 3 (right) ---
        fig.add_scatter(2, measured_output, residuals, label=name, color=model_colors[name], s=10, alpha=0.7)
        
    ax = fig._get_ax(2)
    lims = (
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

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
        cod_batch = sliding_adjusted_cod(measured_output, modeled_output, num_params, len(measured_output))[-1]
        regressors_cod_batch = sliding_vif_cod(regressors, len(regressors))[-1]
        
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
        regressors_cond_batch = sliding_svd_cond(regressors, len(regressors))[-1]
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
        correlation_matrix_batch = sliding_correlation_matrix(regressors, len(regressors))[-1]
        
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


# def plot_filter_duration(
#         dataframe: pd.DataFrame,
#         start_time: Optional[float] = None,
#         end_time: Optional[float] = None,
#         plot_labels: Optional[dict] = None
#         ) -> PlotFigure:

#     if dataframe is None or dataframe.empty or 'timestamp' not in dataframe.columns:
#         raise ValueError("Invalid DataFrame provided.")
#     if plot_labels is None:
#         plot_labels = {}
        
#     if start_time is not None:
#         dataframe = dataframe[dataframe["timestamp"] >= start_time]
#     if end_time is not None:
#         dataframe = dataframe[dataframe["timestamp"] <= end_time]

#     time = dataframe["timestamp"]
#     elapsed = dataframe["elapsed"] * 1000
#     ema_elapsed = dataframe["ema_elapsed"] * 1000
#     max_elapsed = dataframe["max_elapsed"] * 1000
#     min_elapsed = dataframe["min_elapsed"] * 1000

#     fig = PlotFigure(nrows=1, ncols=1, figsize=(12, 6), sharex=True)
#     base_title = "Filter Performance - Duration"
#     subtitle = plot_labels.get("subtitle", "Last Test")
#     fig.set_figure_title(f"{base_title}\n{subtitle}" if subtitle else base_title)

#     fig.define_subplot(0, title="Filter Duration Over Time", ylabel="Time\n[ms]", xlabel="Time [s]")
#     fig.add_scatter(0, time, elapsed, color='tab:blue')
#     fig.add_data(0, time, ema_elapsed, color='black')
#     fig.add_fill_between(0, time, max_elapsed, min_elapsed, "Bounds", color="tab:blue")

#     fig.set_all_legends(loc='upper right', fontsize='medium')
#     fig.set_all_grids(True, alpha=0.5)
#     return fig



def plot_models(csv_files, start_time, end_time, plot_labels, separate = False):
    model_dataframes: Dict[str, pd.DataFrame] = {}
    for name, info in csv_files.items():
        try:
            df = pd.read_csv(info["path"])
            model_dataframes[name] = extract_model(df, info["prefix"])
        except Exception as error:
            raise RuntimeError(f"Failed processing '{name}'") from error
    processed_models = process_models(model_dataframes)

    if not separate:
        # plot_parameter_data(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)  # TODO: Add batch results.
        plot_regressor_data(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
        plot_confidence(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
        # plot_percent_confidence(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)  # TODO: Add batch results.
        plot_error(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
        plot_fit(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)  # TODO: Review the R² method.
        plot_conditioning(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
        plot_correlation(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
        # TODO: Add FFT plotter, Bode plots, and 3D RFT progressions
    else:
        for i in enumerate(processed_models.items()):
            # plot_parameter_data(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            plot_regressor_data(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            plot_confidence(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            # plot_percent_confidence(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            plot_error(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            # plot_fit(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            # plot_conditioning(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
            # plot_correlation(processed_models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    
    plt.show()

def main():
    # TODO: Isolated this into three scripts: model_processing, model_performance, and model_spectrums.
    csv_files = {
        "Small Roll": {"prefix": "ols_rol_", "path": "/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/topic_data_files/ols_rol_data.csv"},
    }

    start_time = 0
    end_time = 55

    plot_labels = {
        # "subtitle": "",
        # "time": "Time [s]",

        "terms":{
            0: {"term": "Roll Acceleration", "units": "[rad/s²]"},
            1: {"term": "Roll Rate", "units": "[rad/s]", "param_units": "[1/s]"},
            2: {"term": "Aileron Command", "units": "[PWM]", "param_units": "[rad/s²-PWM]"},
        },
    }

    plot_models(csv_files, start_time, end_time, plot_labels)

if __name__ == "__main__":
    main()