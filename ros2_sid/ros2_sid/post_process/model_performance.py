#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from ros2_sid.plotter_class import PlotFigure
from ros2_sid.post_process.perf_result_analysis import (
    load_model_topic, OLSResult
)
from ros2_sid.realtime_ols_utils import (
    ordinary_least_squares,
    coefficient_of_determination,
    variance_inflation_factors
)

ArrayLike = Union[float, Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]


POST_PROCESS_DIR = Path(__file__).resolve().parent
DATA_FILE_DIR = POST_PROCESS_DIR / "topic_data_files"


# __all__ = ['']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def _get_max_num_terms(models: dict[str, OLSResult]) -> int:
    max_num_terms = 0
    for name, model in models.items():
        max_num_terms = max(max_num_terms, model.regressors.shape[1])
    return max_num_terms

def _validate_model_data(
        model: OLSResult,
        model_name: str = "Model"
        ) -> bool:
    if model.timestamp.size == 0:
        print(f"Skipping {model_name}: missing timestamps.")
        return False
    if model.measured_output.size == 0:
        print(f"Skipping {model_name}: missing measured output.")
        return False
    if model.regressors.size == 0:
        print(f"Skipping {model_name}: missing parameters.")
        return False
    if model.parameters.size == 0:
        print(f"Skipping {model_name}: missing regressors.")
        return False
    if model.modeled_output.size == 0:
        print(f"Skipping {model_name}: missing modeled output.")
        return False
    if model.modeled_output_ci.size == 0:
        print(f"Skipping {model_name}: missing modeled output confidence intervals.")
        return False
    if model.modeled_output_cip.size == 0:
        print(f"Skipping {model_name}: missing modeled output confidence interval percentages.")
        return False
    if model.parameter_cis.size == 0:
        print(f"Skipping {model_name}: missing parameter confidence intervals.")
        return False
    if model.parameter_cips.size == 0:
        print(f"Skipping {model_name}: missing parameter confidence interval percentages.")
        return False
    
    num_samples = model.timestamp.size

    if (
        model.measured_output.shape[0] != num_samples
        or model.parameters.shape[0] != num_samples
        or model.regressors.shape[0] != num_samples
        or model.modeled_output.shape[0] != num_samples
        or model.modeled_output_ci.shape[0] != num_samples
        or model.modeled_output_cip.shape[0] != num_samples
        or model.parameter_cis.shape[0] != num_samples
        or model.parameter_cips.shape[0] != num_samples
    ):
        print(f"Skipping {model_name}: mismatched array lengths.")
        return False

    return True

def _get_time_mask(
        time: np.ndarray,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> np.ndarray:
    mask = np.ones(time.shape, dtype=bool)
    if start_time is not None:
        mask &= time >= start_time
    if end_time is not None:
        mask &= time <= end_time
    return mask


def plot_regressor_data(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(models)

    fig = PlotFigure(nrows=1 + max_num_terms, ncols=1, figsize=(12, 2 + 2 * max_num_terms), sharex=True)
    base_title = "Model Performance - Regressors"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    term_units = term_info.get("units", "")
    if term_units != "":
        term_units = f"\n[{term_units}]"
    fig.define_subplot(0, title=f"Measured {term_name}", ylabel=f"Amplitude{term_units}")
    for i in range(max_num_terms):
        term_info = terms.get(1 + i, {})
        term_name = term_info.get("term")
        if not term_name:
            term_name = f"Regressor {chr(65+i)}"
        term_units = term_info.get("units", "")
        if term_units != "":
            term_units = f"\n[{term_units}]"
        title = f"{term_name} Over Time"
        ylabel = f"Amplitude{term_units}"
        xlabel = plot_labels.get("time", "Time [s]") if i == max_num_terms - 1 else None
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=xlabel)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}

    measured_output_plotted = False

    for name, model in models.items():
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        regressors = model.regressors[mask]

        if not measured_output_plotted:
            measured_output = model.measured_output[mask]
            fig.add_data(0, time, measured_output, label='Measured', color='black', linestyle='-', marker='.', alpha=0.7)
            measured_output_plotted = True
        
        for j in range(regressors.shape[1]):
            fig.add_data(1 + j, time, regressors[:, j], label=name, color=model_colors[name], linestyle='-', marker='.')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_parameter_data(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(models)

    fig = PlotFigure(nrows=1 + max_num_terms, ncols=1, figsize=(12, 2 + 2 * max_num_terms), sharex=True)
    base_title = "Model Performance - Parameters"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    term_units = term_info.get("units", "")
    if term_units != "":
        term_units = f"\n[{term_units}]"
    fig.define_subplot(0, title=f"Measured vs Modeled {term_name}", ylabel=f"Amplitude{term_units}")
    for i in range(max_num_terms):
        term_info = terms.get(1 + i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"{term_name}'s Parameter Over Time"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"{term_name} Over Time"
        term_units = term_info.get("units", "")
        if term_units != "":
            term_units = f"\n[{term_units}]"
        ylabel = f"Amplitude{term_units}"
        xlabel = plot_labels.get("time", "Time [s]") if i == max_num_terms - 1 else None
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=xlabel)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}

    references_plotted = False

    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue
        
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        parameters = model.parameters[mask]
        modeled_output = model.modeled_output[mask]

        if not references_plotted:
            measured_output = model.measured_output[mask]
            regressors = model.regressors[mask]
            post_parameters = ordinary_least_squares(measured_output, regressors)
            fig.add_data(0, time, measured_output, label='Measured', color='black', linestyle='--', alpha=0.7)
            for j in range(parameters.shape[1]):
                fig.add_line(1 + j, post_parameters[j], 'h', label=f"Post Process: {post_parameters[j]:.2e}", color='black', linestyle='--', alpha=0.7)
            references_plotted = True

        fig.add_data(0, time, modeled_output, label=name, color=model_colors[name])
        
        for j in range(parameters.shape[1]):
            fig.add_data(1 + j, time, parameters[:, j], label=name, color=model_colors[name], linestyle='-', marker='.')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_confidence(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(models)

    fig = PlotFigure(nrows=1 + max_num_terms, ncols=1, figsize=(12, 2 + 2 * max_num_terms), sharex=True)
    base_title = "Model Performance - Confidence Intervals (CIs)"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    term_units = term_info.get("units", "")
    if term_units != "":
        term_units = f"\n[{term_units}]"
    fig.define_subplot(0, title=f"Measured vs Modeled {term_name}", ylabel=f"Amplitude{term_units}")
    for i in range(max_num_terms):
        term_info = terms.get(1 + i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"{term_name}'s Parameter Over Time"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"{term_name} Over Time"
        term_units = term_info.get("units", "")
        if term_units != "":
            term_units = f"\n[{term_units}]"
        ylabel = f"Amplitude{term_units}"
        xlabel = plot_labels.get("time", "Time [s]") if i == max_num_terms - 1 else None
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=xlabel)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}

    legend_handles = [[] for _ in range(1 + max_num_terms)]
    legend_labels = [[] for _ in range(1 + max_num_terms)]

    measured_output_plotted = False

    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue
        
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        parameters = model.parameters[mask]
        parameters_cis = model.parameter_cis[mask]
        modeled_output = model.modeled_output[mask]
        modeled_output_ci = model.modeled_output_ci[mask]

        if not measured_output_plotted:
            measured_output = model.measured_output[mask]
            regressors = model.regressors[mask]
            post_parameters = ordinary_least_squares(measured_output, regressors)
            fig.add_data(0, time, measured_output, label='Measured', color='black', linestyle='--')
            for j in range(parameters.shape[1]):
                post_handle = fig.add_line(1 + j, post_parameters[j], 'h', color='black', linestyle='--', alpha=0.7)
                legend_handles[1 + j].append(post_handle)
                legend_labels[1 + j].append(f"Post: {post_parameters[j]:.2e}")
            measured_output_plotted = True

        line_handle = fig.add_data(0, time, modeled_output, label=name, color=model_colors[name])
        fill_handle = fig.add_fill_between(0, time, modeled_output - modeled_output_ci, modeled_output + modeled_output_ci, label=f"{name} ± 2σ", color=model_colors[name], alpha=0.3)
        legend_handles[0].append((line_handle, fill_handle))
        legend_labels[0].append(f"{name} ± 2σ")
        fig.autoscale_from_lines(0)
                
        for j in range(parameters.shape[1]):
            line_handle = fig.add_data(1 + j, time, parameters[:, j], color=model_colors[name], linestyle='-', marker='.')
            fill_handle = fig.add_fill_between(1 + j, time, parameters[:, j] - parameters_cis[:, j], parameters[:, j] + parameters_cis[:, j], color=model_colors[name], alpha=0.3)
            legend_handles[1 + j].append((line_handle, fill_handle))
            legend_labels[1 + j].append(f"{name} ± 2σ")
            fig.autoscale_from_lines(1 + j)
        
    ax1 = fig._get_ax(0)
    ax1.title.set_fontsize('xx-large')
    ax1.yaxis.label.set_fontsize('xx-large')
    ax1.tick_params(axis='y', labelsize='x-large')

    ax2 = fig._get_ax(1)
    ax2.title.set_fontsize('xx-large')
    ax2.yaxis.label.set_fontsize('xx-large')
    ax2.tick_params(axis='y', labelsize='x-large')

    ax3 = fig._get_ax(2)
    ax3.title.set_fontsize('xx-large')
    ax3.yaxis.label.set_fontsize('xx-large')
    ax3.tick_params(axis='y', labelsize='x-large')
    ax3.xaxis.label.set_fontsize('xx-large')
    ax3.tick_params(axis='x', labelsize='x-large')

    for i, (handles, labels) in enumerate(zip(legend_handles, legend_labels)):
        ax = fig._get_ax(i)
        if i == 0 and measured_output_plotted:
            handles = [ax.lines[0], *handles]
            labels = ["Measured", *labels]
        ax.legend(
            handles,
            labels,
            loc="upper right",
            fontsize="medium",
            handler_map={tuple: HandlerTuple(ndivide=1)}
        )
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_percent_confidence(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(models)

    fig = PlotFigure(nrows=1 + max_num_terms, ncols=1, figsize=(12, 2 + 2 * max_num_terms), sharex=True)
    base_title = "Model Performance - Percent Confidence"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    fig.define_subplot(0, title=f"Percent Confidence of Modeled {term_name} Over Time", ylabel=f"Percent\nConfidence [%]")
    for i in range(max_num_terms):
        term_info = terms.get(1 + i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"Percent Confidence of {term_name}'s Parameter Over Time"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"Percent Confidence of {term_name} Over Time"
        ylabel = f"Percent\nConfidence [%]"
        xlabel = plot_labels.get("time", "Time [s]") if i == max_num_terms - 1 else None
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=xlabel)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}
    
    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue
        
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        modeled_output_cip = model.modeled_output_cip[mask]
        parameters_cips = model.parameter_cips[mask]
        
        fig.add_data(0, time, modeled_output_cip, label=name, color=model_colors[name])

        for j in range(parameters_cips.shape[1]):
            fig.add_data(1+j, time, parameters_cips[:, j], label=name, color=model_colors[name])

    for i in fig.axes:
        fig.set_log_scale(i, 'y')
    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_error(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(models)

    layout = {
        0: (slice(0, 1), slice(0, 1)),   # upper-left
        1: (slice(1, 2), slice(0, 1)),   # lower-left
        2: (slice(0, 2), slice(1, 2)),   # right (spans rows)
    }
    fig = PlotFigure(nrows=2, ncols=2, figsize=(12, 6), gridspec=True, layout=layout)
    base_title = "Model Performance - Error"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    term_units = term_info.get("units", "")
    if term_units != "":
        term_units = f"\n[{term_units}]"
    fig.define_subplot(
        0,
        title="Residuals Over Time",
        ylabel=f"Residual {term_name}{term_units}",
        # xlabel=plot_labels.get("time", "Time [s]"),
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
        title=f"Residuals vs Modeled {term_name}",
        ylabel=f"Residual {term_name}{term_units}",
        xlabel=f"Modeled {term_name}{term_units}",
        grid=True,
    )
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}

    legend_handles = [[] for _ in range(1 + max_num_terms)]
    legend_labels = [[] for _ in range(1 + max_num_terms)]

    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue
        
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        modeled_output = model.modeled_output[mask]
        residual = model.residual[mask]
        mse = model.mse[mask]

        total_mse = float(np.nanmean(residual ** 2))
        
        # --- Subplot 1 (upper left) ---
        dots_handle = fig.add_scatter(0, time, residual, color=model_colors[name], s=10, alpha=0.7)
        line_handle = fig.add_data(0, time, residual, color=model_colors[name], alpha=0.3, linewidth=1)
        legend_handles[0].append((dots_handle, line_handle))
        legend_labels[0].append(name)
        
        # --- Subplot 2 (lower left) ---
        dots_handle = fig.add_scatter(1, time, mse, color=model_colors[name], s=10, alpha=0.7)
        line_handle = fig.add_data(1, time, mse, color=model_colors[name], alpha=0.3, linewidth=1)
        legend_handles[1].append((dots_handle, line_handle))
        legend_labels[1].append(name)
        fig.add_line(1, total_mse, 'h', color=model_colors[name], linestyle=':', alpha=0.7)
        fig.annotate(1, f"{total_mse:.2e}", xy=(start_time, total_mse),
                     xycoords="data", textcoords="offset points",
                     xytext=(-10, 5), ha="left", va="top",
                     fontsize="large", fontweight="bold", color=model_colors[name],
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.3"))
        
        # --- Subplot 3 (right) ---
        dots_handle = fig.add_scatter(2, modeled_output, residual, color=model_colors[name], s=10, alpha=0.7)
        legend_handles[2].append(dots_handle)
        legend_labels[2].append(name)
        
    # ax1 = fig._get_ax(0)
    # ax1.title.set_fontsize('xx-large')
    # ax1.yaxis.label.set_fontsize('xx-large')
    # # ax1.xaxis.label.set_fontsize('xx-large')
    # ax1.tick_params(axis='y', labelsize='x-large')
    # ax1.tick_params(axis='x', labelbottom=False)
    # # ax1.set_xticklabels([])   # Does not work with sharex()

    # ax2 = fig._get_ax(1)
    # ax2.title.set_fontsize('xx-large')
    # ax2.yaxis.label.set_fontsize('xx-large')
    # ax2.xaxis.label.set_fontsize('xx-large')
    # ax2.tick_params(axis='y', labelsize='x-large')
    # ax2.tick_params(axis='x', labelsize='x-large')
    # ax2.sharex(ax1)

    ax3 = fig._get_ax(2)
    # ax3.title.set_fontsize('xx-large')
    # ax3.yaxis.label.set_fontsize('xx-large')
    # ax3.xaxis.label.set_fontsize('xx-large')
    # ax3.tick_params(axis='y', labelsize='x-large')
    # ax3.tick_params(axis='x', labelsize='x-large')
    lims = (
        min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
        max(ax3.get_xlim()[1], ax3.get_ylim()[1]),
    )
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect("equal")

    ax3.set_aspect("equal")
    for i, (handles, labels) in enumerate(zip(legend_handles, legend_labels)):
        ax = fig._get_ax(i)
        ax.legend(
            handles,
            labels,
            loc="upper right",
            fontsize="medium",
            handler_map={tuple: HandlerTuple(ndivide=1)}
        )
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_error_kde(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    fig = PlotFigure(figsize=(12, 6))
    base_title = "Model Performance - Error KDE"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    term_info = terms.get(0, {})
    term_name = term_info.get("term", "Output")
    term_units = term_info.get("units", "")
    if term_units != "":
        term_units = f"\n[{term_units}]"
    
    fig.define_subplot(
        0,
        title="Kernel Density Plot (KDE) of Residuals",
        ylabel=f"Density",
        xlabel=f"Residuals{term_units}",
        grid=True,
    )
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}

    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue
        
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        residual = model.residual[mask]

        x = np.asarray(residual)
        kde = gaussian_kde(x, bw_method='scott')
        x_grid = np.linspace(x.min(), x.max(), int(np.ceil(0.4 * x.size)))
        kde_vals = kde(x_grid)

        mean = x.mean()
        sigma = x.std(ddof=1)
        mask = (x_grid >= mean - 2*sigma) & (x_grid <= mean + 2*sigma)
        kde_at_max = kde(mean)[0]

        fig.add_data(0, x_grid, kde_vals, label=name, color=model_colors[name])
        fig.add_fill_between(0, x_grid[mask], kde_vals[mask], alpha=0.25, color=model_colors[name])
        fig.add_line(0, mean, 'v', min_val=0, max_val=kde_at_max, color=model_colors[name], linestyle="--")
        
    subplot = fig._get_ax(0)
    legend1 = subplot.legend(loc="upper right", fontsize='medium')
    subplot.add_artist(legend1)
    mean_proxy = Line2D([0], [0], color="grey", linestyle="--", linewidth=2)
    sigma_proxy = Patch(facecolor="grey", alpha=0.25)
    subplot.legend(handles=[mean_proxy, sigma_proxy], labels=["Mean", "±2σ Region"], loc="upper left", frameon=True)
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_fit(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> PlotFigure:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}
        
    max_num_terms = _get_max_num_terms(models)

    fig = PlotFigure(nrows=1 + max_num_terms, ncols=1, figsize=(12, 2 + 2 * max_num_terms), sharex=True)
    base_title = "Model Performance - Model Fit"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
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
        else:
            title = f"Parameter {chr(65+i)}'s Fit (r²) Over Time"
        ylabel = f"Parameter\nr² [%]"
        xlabel = plot_labels.get("time", "Time [s]") if i == max_num_terms - 1 else None
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=xlabel)
        fig.shade_subplot(1 + i, y_range=(0.75, 1.00), color='#F4CCCC', alpha=0.3)  # red
        fig.shade_subplot(1 + i, y_range=(0.50, 0.75), color='#FFF3B0', alpha=0.3)  # yellow
        fig.shade_subplot(1 + i, y_range=(0.00, 0.50), color='#A8D5BA', alpha=0.3)  # green
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(models)}

    references_plotted = False

    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue
        
        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        cod = model.cod[mask]
        regressor_cods = model.regressor_cods[mask]

        if not references_plotted:
            measured_output = model.measured_output[mask]
            regressors = model.regressors[mask]
            post_parameters = ordinary_least_squares(measured_output, regressors)
            post_cod = coefficient_of_determination(measured_output, regressors, post_parameters, adjusted=True)
            post_regressor_cods = variance_inflation_factors(regressors, return_cod=True)
            fig.add_line(0, post_cod, 'h', label=f'Post Process: {post_cod:.2e}', color='black', linestyle=':', alpha=0.7)
            for j in range(regressors.shape[1]):
                fig.add_line(1 + j, post_regressor_cods[j], 'h', label=f"Post Process: {post_regressor_cods[j]:.2e}", color='black', linestyle=':', alpha=0.7)
            references_plotted = True

        fig.add_data(0, time, cod, label=name, color=model_colors[name], linestyle='-', marker='.')
        
        for j in range(regressor_cods.shape[1]):
            fig.add_data(1 + j, time, regressor_cods[:, j], label=name, color=model_colors[name], linestyle='-', marker='.')

    fig.set_all_legends(loc='upper right', fontsize='medium')
    fig.set_all_grids(True, alpha=0.5)
    return fig

# def plot_fit(
#         dataframes: Dict[str, pd.DataFrame],
#         *,
#         start_time: float | None = None,
#         end_time: float | None = None,
#         plot_labels: dict | None = None
#         ) -> PlotFigure:
    
#     if not dataframes:
#         raise ValueError("No models provided.")
#     if plot_labels is None:
#         plot_labels = {}
        
#     max_num_terms = _get_max_num_terms(dataframes)
#     terms = plot_labels.get("terms", {})

#     fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
#     base_title = "Model Performance - Model Fit"
#     subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
#     fig.set_figure_title(f"{base_title}\n{subtitle}")

#     term_info = terms.get(0, {})
#     fig.define_subplot(0, title="Coefficient of Determination (R²) Over Time", ylabel="R² [%]")
#     fig.shade_subplot(0, y_range=(0.75, 1.00), color='#A8D5BA', alpha=0.3)  # green
#     fig.shade_subplot(0, y_range=(0.50, 0.75), color='#FFF3B0', alpha=0.3)  # yellow
#     fig.shade_subplot(0, y_range=(0.00, 0.50), color='#F4CCCC', alpha=0.3)  # red
#     for i in range(max_num_terms):
#         term_info = terms.get(1+i, {})
#         term_name = term_info.get("term")
#         if term_name:
#             title = f"{term_name}'s Parameter Fit (r²) Over Time"
#             ylabel = f"{term_name}'s\nParameter r² [%]"
#         else:
#             term_name = f"Parameter {chr(65+i)}'s Fit (r²) Over Time"
#             title = f"{term_name} Over Time"
#             ylabel = f"{term_name}'s\nr² [%]"
#         fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
#         fig.shade_subplot(1 + i, y_range=(0.75, 1.00), color='#F4CCCC', alpha=0.3)  # red
#         fig.shade_subplot(1 + i, y_range=(0.50, 0.75), color='#FFF3B0', alpha=0.3)  # yellow
#         fig.shade_subplot(1 + i, y_range=(0.00, 0.50), color='#A8D5BA', alpha=0.3)  # green
            
#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     model_colors = {
#         name: color_cycle[i % len(color_cycle)]
#         for i, name in enumerate(dataframes.keys())
#     }

#     for i, (name, df) in enumerate(dataframes.items()):
#         if df.empty:
#             print(f"Skipping empty DataFrame named {name}.")
#             continue
#         if 'timestamp' not in df.columns:
#             print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
#             continue
        
#         if start_time is not None:
#             df = df[df["timestamp"] >= start_time]
#         if end_time is not None:
#             df = df[df["timestamp"] <= end_time]

#         prefix = _extract_model_prefix(df)
#         regressor_cols = _get_indexed_columns(df, prefix, r"regressor_\d+$")
#         regressor_cod_cols = _get_indexed_columns(df, prefix, r"regressor_\d+_cod")

#         time = df["timestamp"]
#         cod = df[f"{prefix}cod"]
#         regressors_cod = df[regressor_cod_cols].to_numpy()
#         num_params = len(regressor_cod_cols)
        
#         measured_output = df[f"{prefix}measured_output"]
#         modeled_output = df[f"{prefix}modeled_output"]
#         regressors = df[regressor_cols].to_numpy()
#         cod_batch = sliding_adjusted_cod(measured_output, modeled_output, num_params, len(measured_output))[-1]
#         regressors_cod_batch = sliding_vif_cod(regressors, len(regressors))[-1]
        
#         fig.add_data(0, time, cod, label=name, color=model_colors[name], marker='.')
#         fig.add_line(0, cod_batch, 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

#         for j in range(num_params):
#             fig.add_data(1+j, time, regressors_cod[:, j], label=name, color=model_colors[name], marker='.')
#             fig.add_line(1+j, regressors_cod_batch[j], 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

#     fig.set_all_legends(loc='upper right', fontsize='medium')
#     fig.set_all_grids(True, alpha=0.5)
#     fig.apply_to_all_axes('set_ylim', [0, 1])
#     return fig

# def plot_conditioning(
#         dataframes: Dict[str, pd.DataFrame],
#         *,
#         start_time: float | None = None,
#         end_time: float | None = None,
#         plot_labels: dict | None = None
#         ) -> PlotFigure:
    
#     if not dataframes:
#         raise ValueError("No models provided.")
#     if plot_labels is None:
#         plot_labels = {}
        
#     max_num_terms = _get_max_num_terms(dataframes)
#     terms = plot_labels.get("terms", {})

#     fig = PlotFigure(nrows=1+max_num_terms, ncols=1, figsize=(12, (2 + 2*max_num_terms)), sharex=True)
#     base_title = "Model Performance - Conditioning"
#     subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
#     fig.set_figure_title(f"{base_title}\n{subtitle}")

#     fig.define_subplot(0, title="Max Condition Number Over Time", ylabel="Amplitude")
#     for i in range(max_num_terms):
#         term_info = terms.get(1+i, {})
#         term_name = term_info.get("term")
#         if term_name:
#             title = f"{term_name}'s Parameter Conditioning Over Time"
#             ylabel = "Amplitude"
#         else:
#             term_name = f"Parameter {chr(65+i)}"
#             title = f"{term_name}'s Conditioning Over Time"
#             ylabel = "Amplitude"
#         fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None)
            
#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     model_colors = {
#         name: color_cycle[i % len(color_cycle)]
#         for i, name in enumerate(dataframes.keys())
#     }

#     for i, (name, df) in enumerate(dataframes.items()):
#         if df.empty:
#             print(f"Skipping empty DataFrame named {name}.")
#             continue
#         if 'timestamp' not in df.columns:
#             print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
#             continue
        
#         if start_time is not None:
#             df = df[df["timestamp"] >= start_time]
#         if end_time is not None:
#             df = df[df["timestamp"] <= end_time]

#         prefix = _extract_model_prefix(df)
#         regressor_cols = _get_indexed_columns(df, prefix, r"regressor_\d+$")
#         regressor_cond_cols = _get_indexed_columns(df, prefix, r"regressor_\d+_cond")

#         time = df["timestamp"]
#         regressors_cond = df[regressor_cond_cols].to_numpy()
#         num_params = len(regressor_cond_cols)
#         max_cond = df[regressor_cond_cols].max(axis=1).to_numpy()
 
#         regressors = df[regressor_cols].to_numpy()
#         regressors_cond_batch = sliding_svd_cond(regressors, len(regressors))[-1]
#         max_cond_batch = max(regressors_cond_batch)
        
#         fig.add_data(0, time, max_cond, label=name, color=model_colors[name])
#         fig.add_line(0, max_cond_batch, 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

#         for j in range(num_params):
#             fig.add_data(1+j, time, regressors_cond[:, j], label=name, color=model_colors[name])
#             fig.add_line(1+j, regressors_cond_batch[j], 'h', label=f"Batched {name}", color=model_colors[name], linestyle=':', alpha=0.7)

#     fig.set_all_legends(loc='upper right', fontsize='medium')
#     fig.set_all_grids(True, alpha=0.5)
#     return fig

# def plot_correlation(
#         dataframes: Dict[str, pd.DataFrame],
#         *,
#         start_time: float | None = None,
#         end_time: float | None = None,
#         plot_labels: dict | None = None
#         ) -> list[PlotFigure]:
    
#     if not dataframes:
#         raise ValueError("No models provided.")
#     if plot_labels is None:
#         plot_labels = {}

#     figures: list[PlotFigure] = []
    
#     for i, (name, df) in enumerate(dataframes.items()):
#         if df.empty:
#             print(f"Skipping empty DataFrame named {name}.")
#             continue
#         if 'timestamp' not in df.columns:
#             print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
#             continue
        
#         if start_time is not None:
#             df = df[df["timestamp"] >= start_time]
#         if end_time is not None:
#             df = df[df["timestamp"] <= end_time]
        
#         prefix = _extract_model_prefix(df)
#         parameter_cols =_get_indexed_columns(df, prefix, r"parameter_\d+$")
#         max_num_terms = len(parameter_cols)

#         fig = PlotFigure(nrows=max_num_terms, ncols=1, figsize=(12, (2*max_num_terms)), sharex=True)
#         base_title = "Model Performance - Regressor Correlations"
#         subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
#         fig.set_figure_title(f"{base_title}\n{subtitle}")

#         color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#         time = df["timestamp"]
        
#         regressors = df[_get_indexed_columns(df, prefix, r"regressor_\d+$")].to_numpy()
#         correlation_matrix_batch = sliding_correlation_matrix(regressors, len(regressors))[-1]
        
#         for j in range(max_num_terms):
#             title = f"Parameter {chr(65 + j)}'s Correlation Over Time"
#             ylabel = f"Param {chr(65 + j)}\nCorrelation"
#             fig.define_subplot(j, title=title, ylabel=ylabel, xlabel=plot_labels.get(f"time", "Time [s]") if j == max_num_terms - 1 else None)
#             fig.shade_subplot(j, y_range=( 0.90,  1.00), color='#F4CCCC', alpha=0.3)  # red
#             fig.shade_subplot(j, y_range=( 0.75,  0.90), color='#FFF3B0', alpha=0.3)  # yellow
#             fig.shade_subplot(j, y_range=(-0.75,  0.75), color='#A8D5BA', alpha=0.3)  # green
#             fig.shade_subplot(j, y_range=(-0.90, -0.75), color='#FFF3B0', alpha=0.3)  # yellow
#             fig.shade_subplot(j, y_range=(-1.00, -0.90), color='#F4CCCC', alpha=0.3)  # red

#             for k in range(max_num_terms):
#                 if j == k:
#                     continue
                
#                 param_color = color_cycle[k % len(color_cycle)]
                
#                 col_name = f"{prefix}correlation_{j}_to_{k}"
#                 if col_name in df.columns:
#                     fig.add_data(j, time, df[col_name], label=f"corr({chr(65+j)}, {chr(65+k)})", color=param_color)
                    
#                 fig.add_line(j, correlation_matrix_batch[j, k], 'h', color=param_color, linestyle=':', alpha=0.7)

#         fig.set_all_legends(loc='upper right', fontsize='medium')
#         fig.set_all_grids(True, alpha=0.5)
#         fig.apply_to_all_axes('set_ylim', [-1, 1])

#         figures.append(fig)

#     return figures


# def polor_plot(
#         dataframes: dict[str, pd.DataFrame],
#         *,
#         start_time: float | None = None,
#         end_time: float | None = None,
#         plot_labels: dict | None = None
#         ) -> PlotFigure:
    
#     if not dataframes:
#         raise ValueError("No models provided.")
#     if plot_labels is None:
#         plot_labels = {}

#     fig = PlotFigure(figsize=(12, 6))
#     base_title = "Polar Plot"
#     subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(dataframes)
#     fig.set_figure_title(f"{base_title}\n{subtitle}")

#     terms = plot_labels.get("terms", {})
#     term_info = terms.get(0, {})
#     units = term_info.get("units", "")
    
#     fig.define_subplot(
#         0,
#         title="Cl-Alpha Polar",
#         ylabel=f"Cl",
#         xlabel=f"Alpha",
#         grid=True,
#     )

#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     model_colors = {
#         name: color_cycle[i % len(color_cycle)]
#         for i, name in enumerate(dataframes.keys())
#     }
    
#     for i, (name, df) in enumerate(dataframes.items()):
#         if df.empty:
#             print(f"Skipping empty DataFrame named {name}.")
#             continue
#         if 'timestamp' not in df.columns:
#             print(f"Skipping DataFrame named {name} due to missing 'timestamp'.")
#             continue
        
#         if start_time is not None:
#             df = df[df["timestamp"] >= start_time]
#         if end_time is not None:
#             df = df[df["timestamp"] <= end_time]

#         prefix = _extract_model_prefix(df)

#         cl_param = df[f"{prefix}parameter_3"]
#         alpha = df[f"{prefix}regressor_3"]

#         fig.add_scatter(0, alpha, cl_param, label=name, color=model_colors[name])
        
#     fig.set_all_grids(True, alpha=0.5)
#     return fig


def generate_plots(
    models: dict[str, OLSResult],
    *,
    combine_models: bool = True,
    start_time: float | None = None,
    end_time: float | None = None,
    plot_labels: dict | None = None,
    ) -> None:
    if not combine_models:
        for name, model in models.items():
            generate_plots(
                models={name: model},
                combine_models=True,
                start_time=start_time,
                end_time=end_time,
                plot_labels=plot_labels,
            )
        return
    # plot_regressor_data(models=models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_parameter_data(models=models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    plot_confidence(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_percent_confidence(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_error(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_error_kde(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_fit(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_conditioning(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)
    # plot_correlation(models, start_time=start_time, end_time=end_time, plot_labels=plot_labels)

def main() -> None:
    rol = load_model_topic("rol")
    rol2 = load_model_topic("rol_2")
    rol_fixed = load_model_topic("rol_fixed")
    rol_time = load_model_topic("rol_time")
    
    generate_plots(
        models={
            # "Roll": rol,
            # "Small Roll": rol2,
            # "Roll Fixed": rol_fixed,
            "Small Roll": rol_time,
            },
        # combine_models=False,
         start_time=16,
        end_time=16+16,
        plot_labels={
            # "title": "",
            # "subtitle": "",
            # "time": "Time [s]",
            "terms":{
                0: {"term": "Roll Acceleration", "units": "rad/s²"},
                1: {"term": "Roll Rate", "units": "rad/s", "param_units": "1/s"},
                2: {"term": "Aileron Command", "units": "PWM", "param_units": "rad/s²-PWM"},
            }
        },
    )

    plt.show()

if __name__ == "__main__":
    main()