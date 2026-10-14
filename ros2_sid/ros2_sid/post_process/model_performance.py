#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from ros2_sid.plotter_class import PlotFigure
from ros2_sid.post_process.perform_result_analysis import (
    load_model_topic, OLSResult
)
from ros2_sid.realtime_ols_utils import (
    ordinary_least_squares,
    coefficient_of_determination,
    variance_inflation_factors,
    singular_value_decomposition,
    matrix_correlation,
)


POST_PROCESS_DIR = Path(__file__).resolve().parent
DATA_FILE_DIR = POST_PROCESS_DIR / "topic_data_files"
SAVE_PLOT_DIR = POST_PROCESS_DIR / "model_plots"


__all__ = ['generate_plots']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def _convert_to_filename(name: str) -> str:
    """Convert a name into a filesystem-friendly filename."""
    return "_".join(
        "".join(
            char.lower() if char.isalnum() else "_"
            for char in name
        ).strip("_").split("_")
    )

def _get_next_combined_folder() -> str:
    """Return the next available Combined_N folder name."""
    pattern = re.compile(r"^Combined_(\d+)$", re.IGNORECASE)
    largest_number = 0

    if SAVE_PLOT_DIR.exists():
        for folder in SAVE_PLOT_DIR.iterdir():
            if not folder.is_dir():
                continue

            match = pattern.match(folder.name)
            if match:
                largest_number = max(
                    largest_number,
                    int(match.group(1)),
                )

    return f"combined_{largest_number + 1}"


def _get_max_num_terms(models: dict[str, OLSResult]) -> int:
    max_num_terms = 0
    for _, model in models.items():
        max_num_terms = max(max_num_terms, model.regressors.shape[1])
    return max_num_terms

def _validate_model_data(
        model: OLSResult,
        model_name: str = "Model"
        ) -> bool:
    required_fields = {
        "timestamp": "timestamps",
        "measured_output": "measured output",
        "regressors": "regressors",
        "parameters": "parameters",
        "modeled_output": "modeled output",
        "residual": "residual",
        "mse": "mse (mean squared error)",
        "cod": "cod (coefficient of determination)",
        "modeled_output_ci": "modeled output confidence intervals",
        "modeled_output_cip": "modeled output confidence interval percentages",
        "parameter_cis": "parameter confidence intervals",
        "parameter_cips": "parameter confidence interval percentages",
        "regressor_cods": (
            "regressor cods (coefficient of determinations) "
            "from vifs (variance inflation factors)"
        ),
        "regressor_conds": (
            "regressor conds (condition numbers) "
            "from svd (singular value decomposition)"
        ),
        "correlation_matrix": "correlation matrix",
    }

    for field, description in required_fields.items():
        if getattr(model, field).size == 0:
            print(f"Skipping {model_name}: missing {description}.")
            return False

    num_samples = model.timestamp.shape[0]
    sample_fields = tuple(required_fields)
    if any(
        getattr(model, field).shape[0] != num_samples
        for field in sample_fields
        if field != "timestamp"
        ):
        print(f"Skipping {model_name}: mismatched array lengths.")
        return False

    num_predictors = model.regressors.shape[1]
    predictor_fields = (
        "parameters",
        "parameter_cis",
        "parameter_cips",
        "regressor_cods",
        "regressor_conds",
    )
    if any(
        getattr(model, field).shape[1] != num_predictors
        for field in predictor_fields
        ):
        print(f"Skipping {model_name}: mismatched predictor dimensions.")
        return False

    if model.correlation_matrix.shape[1:] != (
        num_predictors,
        num_predictors,
        ):
        print(f"Skipping {model_name}: mismatched correlation matrix dimensions.")
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
            fig.add_data(0, time, measured_output, label='Measured', color='black', marker='.', alpha=0.7)
            measured_output_plotted = True
        
        for j in range(regressors.shape[1]):
            fig.add_data(1 + j, time, regressors[:, j], label=name, color=model_colors[name], marker='.')

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

    real_labels = {i: [] for i in range(1 + max_num_terms)}
    real_handles = {i: [] for i in range(1 + max_num_terms)}
    batch_labels = {i: [] for i in range(1 + max_num_terms)}
    batch_handles = {i: [] for i in range(1 + max_num_terms)}
            
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

        measured_output = model.measured_output[mask]
        regressors = model.regressors[mask]
        batch_parameters = ordinary_least_squares(measured_output, regressors)

        if not references_plotted:
            real_labels[0].append('Measured')
            real_handles[0].append(
                fig.add_data(0, time, measured_output, color='black', linestyle='--', alpha=0.7)
            )
            references_plotted = True

        real_labels[0].append(name)
        real_handles[0].append(
            fig.add_data(0, time, modeled_output, color=model_colors[name])
        )
        
        for j in range(parameters.shape[1]):
            index = 1 + j
            real_labels[index].append(name)
            real_handles[index].append(
                fig.add_data(index, time, parameters[:, j], color=model_colors[name], marker='.')
            )
            batch_labels[index].append(f"{batch_parameters[j]:.2e}")
            batch_handles[index].append(
                fig.add_line(index, batch_parameters[j], 'h', color=model_colors[name], linestyle='--', alpha=0.7)
            )

    for i, ax in enumerate(fig.all_axes):
        if batch_handles[i]:
            batch_legend = ax.legend(
                batch_handles[i],
                batch_labels[i],
                title="Batch",
                loc="upper left",
                fontsize="medium",
                bbox_to_anchor=(1.0, 1.0),
            )
            ax.add_artist(batch_legend)
        if real_handles[i]:
            if i == 0:
                ax.legend(
                    real_handles[i],
                    real_labels[i],
                    loc="upper right",
                    fontsize="medium",
                )
                continue
            ax.legend(
                real_handles[i],
                real_labels[i],
                title="Live",
                loc="upper right",
                fontsize="medium",
            )
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

    real_labels = {i: [] for i in range(1 + max_num_terms)}
    real_handles = {i: [] for i in range(1 + max_num_terms)}
    batch_labels = {i: [] for i in range(1 + max_num_terms)}
    batch_handles = {i: [] for i in range(1 + max_num_terms)}
            
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
        parameters_cis = model.parameter_cis[mask]
        modeled_output = model.modeled_output[mask]
        modeled_output_ci = model.modeled_output_ci[mask]

        measured_output = model.measured_output[mask]
        regressors = model.regressors[mask]
        batch_parameters = ordinary_least_squares(measured_output, regressors)

        if not references_plotted:
            real_labels[0].append('Measured')
            real_handles[0].append(
                fig.add_data(0, time, measured_output, color='black', linestyle='--', alpha=0.7)
            )
            references_plotted = True

        real_labels[0].append(f"{name} ± 2σ")
        real_handles[0].append((
            fig.add_data(0, time, modeled_output, color=model_colors[name]),
            fig.add_fill_between(0,
                                 time,
                                 modeled_output - modeled_output_ci,
                                 modeled_output + modeled_output_ci,
                                 color=model_colors[name],
                                 alpha=0.3)
        ))
        fig.autoscale_from_lines(0)
                
        for j in range(parameters.shape[1]):
            index = 1 + j
            real_labels[index].append(f"{name} ± 2σ")
            real_handles[index].append((
                fig.add_data(index, time, parameters[:, j], color=model_colors[name], marker='.'),
                fig.add_fill_between(index, 
                                     time,
                                     parameters[:, j] - parameters_cis[:, j],
                                     parameters[:, j] + parameters_cis[:, j],
                                     color=model_colors[name],
                                     alpha=0.3)
            ))

            batch_labels[index].append(f"{batch_parameters[j]:.2e}")
            batch_handles[index].append(
                fig.add_line(index, batch_parameters[j], 'h', color=model_colors[name], linestyle='--', alpha=0.7)
            )
            fig.autoscale_from_lines(1 + j)

    for i, ax in enumerate(fig.all_axes):
        if batch_handles[i]:
            batch_legend = ax.legend(
                batch_handles[i],
                batch_labels[i],
                title="Batch",
                loc="upper left",
                fontsize="medium",
                bbox_to_anchor=(1.0, 1.0),
            )
            ax.add_artist(batch_legend)
        if real_handles[i]:
            if i == 0:
                ax.legend(
                    real_handles[i],
                    real_labels[i],
                    loc="upper right",
                    fontsize="medium",
                )
                continue
            ax.legend(
                real_handles[i],
                real_labels[i],
                title="Live",
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

    real_labels = [[] for _ in range(3)]
    real_handles = [[] for _ in range(3)]
    batch_labels = [[] for _ in range(3)]
    batch_handles = [[] for _ in range(3)]

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
        real_labels[0].append(name)
        real_handles[0].append((
            fig.add_scatter(0, time, residual, color=model_colors[name], s=10, alpha=0.7),
            fig.add_data(0, time, residual, color=model_colors[name], alpha=0.3, linewidth=1)
        ))
        
        # --- Subplot 2 (lower left) ---
        real_labels[1].append(name)
        real_handles[1].append((
            fig.add_scatter(1, time, mse, color=model_colors[name], s=10, alpha=0.7),
            fig.add_data(1, time, mse, color=model_colors[name], alpha=0.3, linewidth=1)
        ))
        batch_labels[1].append(f"{total_mse:.2e}")
        batch_handles[1].append(
            fig.add_line(1, total_mse, 'h', color=model_colors[name], linestyle='--', alpha=0.7)
        )
        
        # --- Subplot 3 (right) ---
        real_labels[2].append(name)
        real_handles[2].append(
            fig.add_scatter(2, modeled_output, residual, color=model_colors[name], s=10, alpha=0.7)
        )
        
    ax3 = fig._get_ax(2)
    lims = (
        min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
        max(ax3.get_xlim()[1], ax3.get_ylim()[1]),
    )
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect("equal")

    for i, ax in enumerate(fig.all_axes):
        if batch_handles[i]:
            batch_legend = ax.legend(
                batch_handles[i],
                batch_labels[i],
                title="Batch",
                loc="upper left",
                fontsize="medium",
            )
            ax.add_artist(batch_legend)
        if real_handles[i]:
            ax.legend(
                real_handles[i],
                real_labels[i],
                loc="upper right",
                fontsize="medium",
                handler_map={tuple: HandlerTuple(ndivide=1)},
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
    fig.shade_subplot(0, y_range=(75, 100), color='#A8D5BA', alpha=0.3)  # green
    fig.shade_subplot(0, y_range=(50, 75), color='#FFF3B0', alpha=0.3)  # yellow
    fig.shade_subplot(0, y_range=(00, 50), color='#F4CCCC', alpha=0.3)  # red
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
        fig.shade_subplot(1 + i, y_range=(75, 100), color='#F4CCCC', alpha=0.3)  # red
        fig.shade_subplot(1 + i, y_range=(50, 75), color='#FFF3B0', alpha=0.3)  # yellow
        fig.shade_subplot(1 + i, y_range=(00, 50), color='#A8D5BA', alpha=0.3)  # green

    real_labels = {i: [] for i in range(1 + max_num_terms)}
    real_handles = {i: [] for i in range(1 + max_num_terms)}
    batch_labels = {i: [] for i in range(1 + max_num_terms)}
    batch_handles = {i: [] for i in range(1 + max_num_terms)}
            
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
        cod = model.cod[mask] * 100
        regressor_cods = model.regressor_cods[mask] * 100

        measured_output = model.measured_output[mask]
        regressors = model.regressors[mask]
        batch_parameters = ordinary_least_squares(measured_output, regressors)
        batch_cod = coefficient_of_determination(measured_output, regressors, batch_parameters, adjusted=True)
        batch_cod = np.clip(batch_cod * 100, 0, 100)
        batch_regressor_cods = variance_inflation_factors(regressors, return_cod=True)
        batch_regressor_cods = np.clip(batch_regressor_cods * 100, 0, 100)

        real_labels[0].append(name)
        real_handles[0].append(
            fig.add_data(0, time, cod, color=model_colors[name], marker='.')
        )
        batch_labels[0].append(f"{batch_cod:.1f}%")
        batch_handles[0].append(
            fig.add_line(0, batch_cod, 'h', color=model_colors[name], linestyle='--', alpha=0.7)
        )
        
        for j in range(regressor_cods.shape[1]):
            index = 1 + j
            real_labels[index].append(name)
            real_handles[index].append(
                fig.add_data(index, time, regressor_cods[:, j], color=model_colors[name], marker='.')
            )
            batch_labels[index].append(f"{batch_regressor_cods[j]:.1f}%")
            batch_handles[index].append(
                fig.add_line(index, batch_regressor_cods[j], 'h', color=model_colors[name], linestyle='--', alpha=0.7)
            )

    for i, ax in enumerate(fig.all_axes):
        if batch_handles[i]:
            batch_legend = ax.legend(
                batch_handles[i],
                batch_labels[i],
                title="Batch",
                loc="upper left",
                fontsize="medium",
                bbox_to_anchor=(1.0, 1.0),
            )
            ax.add_artist(batch_legend)
        if real_handles[i]:
            ax.legend(
                real_handles[i],
                real_labels[i],
                title="Live",
                loc="upper right",
                fontsize="medium",
            )
    fig.set_all_grids(True, alpha=0.5)
    fig.apply_to_all_axes('set_ylim', [0, 100])
    return fig

def plot_conditioning(
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
    base_title = "Model Performance - Conditioning"
    subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
    fig.set_figure_title(f"{base_title}\n{subtitle}")

    terms = plot_labels.get("terms", {})
    fig.define_subplot(0, title="Max Condition Number Over Time", ylabel="Amplitude")
    for i in range(max_num_terms):
        term_info = terms.get(1+i, {})
        term_name = term_info.get("term")
        if term_name:
            title = f"{term_name}'s Parameter Conditioning Over Time"
        else:
            term_name = f"Parameter {chr(65+i)}"
            title = f"{term_name}'s Conditioning Over Time"
        ylabel = "Amplitude"
        xlabel = plot_labels.get(f"time", "Time [s]") if i == max_num_terms - 1 else None
        fig.define_subplot(1 + i, title=title, ylabel=ylabel, xlabel=xlabel)

    real_labels = {i: [] for i in range(1 + max_num_terms)}
    real_handles = {i: [] for i in range(1 + max_num_terms)}
    batch_labels = {i: [] for i in range(1 + max_num_terms)}
    batch_handles = {i: [] for i in range(1 + max_num_terms)}
            
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
        regressor_conds = model.regressor_conds[mask]
        max_cond = regressor_conds.max(axis=1)

        regressors = model.regressors[mask]
        batch_regressor_conds = singular_value_decomposition(regressors, return_cond_nums=True)
        batch_max_cond = max(batch_regressor_conds)

        real_labels[0].append(name)
        real_handles[0].append(
            fig.add_data(0, time, max_cond, color=model_colors[name], marker='.')
        )
        batch_labels[0].append(f"{batch_max_cond:.1f}")
        batch_handles[0].append(
            fig.add_line(0, batch_max_cond, 'h', color=model_colors[name], linestyle='--')
        )

        for j in range(regressor_conds.shape[1]):
            index = 1 + j
            real_labels[index].append(name)
            real_handles[index].append(
                fig.add_data(index, time, regressor_conds[:, j], color=model_colors[name], marker='.')
            )
            batch_labels[index].append(f"{batch_regressor_conds[j]:.1f}")
            batch_handles[index].append(
                fig.add_line(index, batch_regressor_conds[j], 'h', color=model_colors[name], linestyle='--', alpha=0.7)
            )

    for i, ax in enumerate(fig.all_axes):
        if batch_handles[i]:
            batch_legend = ax.legend(
                batch_handles[i],
                batch_labels[i],
                title="Batch",
                loc="upper left",
                fontsize="medium",
                bbox_to_anchor=(1.0, 1.0),
            )
            ax.add_artist(batch_legend)
        if real_handles[i]:
            ax.legend(
                real_handles[i],
                real_labels[i],
                title="Live",
                loc="upper right",
                fontsize="medium",
            )
    fig.set_all_grids(True, alpha=0.5)
    return fig

def plot_correlation(
        models: Dict[str, OLSResult],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        plot_labels: dict | None = None
        ) -> list[PlotFigure]:
    if not models:
        raise ValueError("No models provided.")
    if plot_labels is None:
        plot_labels = {}

    figures: list[PlotFigure] = []
    for name, model in models.items():
        if not _validate_model_data(model, name):
            continue

        mask = _get_time_mask(model.timestamp, start_time, end_time)
        if not np.any(mask):
            print(f"Skipping {name}: no data in the specified time range.")
            continue

        time = model.timestamp[mask]
        correlation_matrix = model.correlation_matrix[mask]

        regressors = model.regressors[mask]
        batch_correlation_matrix = matrix_correlation(regressors)
        
        max_num_terms = regressors.shape[1]

        fig = PlotFigure(nrows=max_num_terms, ncols=1, figsize=(12, 2 * max_num_terms), sharex=True)
        base_title = "Model Performance - Regressor Correlations"
        subtitle = plot_labels["subtitle"] if plot_labels.get("subtitle") else "Model(s): " + ", ".join(models)
        fig.set_figure_title(f"{base_title}\n{subtitle}")

        real_labels = {i: [] for i in range(1 + max_num_terms)}
        real_handles = {i: [] for i in range(1 + max_num_terms)}
        batch_labels = {i: [] for i in range(1 + max_num_terms)}
        batch_handles = {i: [] for i in range(1 + max_num_terms)}
                
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for j in range(max_num_terms):
            title = f"Parameter {chr(65 + j)}'s Correlation Over Time"
            ylabel = f"Param {chr(65 + j)}\nCorrelation"
            xlabel = plot_labels.get(f"time", "Time [s]") if j == max_num_terms - 1 else None
            fig.define_subplot(j, title=title, ylabel=ylabel, xlabel=xlabel)
            fig.shade_subplot(j, y_range=( 0.90,  1.00), color='#F4CCCC', alpha=0.3)  # red
            fig.shade_subplot(j, y_range=( 0.75,  0.90), color='#FFF3B0', alpha=0.3)  # yellow
            fig.shade_subplot(j, y_range=(-0.75,  0.75), color='#A8D5BA', alpha=0.3)  # green
            fig.shade_subplot(j, y_range=(-0.90, -0.75), color='#FFF3B0', alpha=0.3)  # yellow
            fig.shade_subplot(j, y_range=(-1.00, -0.90), color='#F4CCCC', alpha=0.3)  # red

            for k in range(max_num_terms):
                if j == k:
                    continue
                param_color = color_cycle[k % len(color_cycle)]

                real_labels[j].append(f"corr({chr(65+j)}, {chr(65+k)})")
                real_handles[j].append(
                    fig.add_data(j, time, correlation_matrix[:, j, k], color=param_color)
                )
                batch_labels[j].append(f"{batch_correlation_matrix[j, k]:.2f}")
                batch_handles[j].append(
                    fig.add_line(j, batch_correlation_matrix[j, k], 'h', color=param_color, linestyle='--', alpha=0.7)
                )

        for i, ax in enumerate(fig.all_axes):
            if batch_handles[i]:
                batch_legend = ax.legend(
                    batch_handles[i],
                    batch_labels[i],
                    title="Batch",
                    loc="upper left",
                    fontsize="medium",
                    bbox_to_anchor=(1.0, 1.0),
                )
                ax.add_artist(batch_legend)
            if real_handles[i]:
                ax.legend(
                    real_handles[i],
                    real_labels[i],
                    title="Live",
                    loc="upper right",
                    fontsize="medium",
                )
        fig.set_all_grids(True, alpha=0.5)
        fig.apply_to_all_axes('set_ylim', [-1, 1])
        figures.append(fig)
    return figures


def generate_plots(
    models: dict[str, OLSResult],
    *,
    combine_models: bool = True,
    start_time: float | None = None,
    end_time: float | None = None,
    plot_labels: dict | None = None,
    plot_figures: bool = True,
    save_figures: bool = False,
    group_name: str | None = None,
    ) -> None:
    if not combine_models:
        for name, model in models.items():
            generate_plots(
                models={name: model},
                combine_models=True,
                start_time=start_time,
                end_time=end_time,
                plot_labels=plot_labels,
                save_figures=save_figures,
                group_name=name
            )
        return

    if group_name is not None:
        folder_name = group_name
    elif len(models) == 1:
        folder_name = next(iter(models))
    else:
        folder_name = _get_next_combined_folder()

    if len(models) == 1:
        filename_suffix = _convert_to_filename(next(iter(models)))
    else:
        filename_suffix = _get_next_combined_folder()

    save_dir: Path | None = None
    if save_figures:
        save_dir = SAVE_PLOT_DIR / _convert_to_filename(folder_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        
    plot_functions = [
        ("regs", plot_regressor_data),
        ("params", plot_parameter_data),
        ("conf", plot_confidence),
        ("per_conf", plot_percent_confidence),
        ("err", plot_error),
        ("kde", plot_error_kde),
        ("fit", plot_fit),
        ("cond", plot_conditioning),
        ("corr", plot_correlation),
    ]

    for plot_name, plot_function in plot_functions:
        result = plot_function(
            models,
            start_time=start_time,
            end_time=end_time,
            plot_labels=plot_labels,
        )

        if not save_figures or save_dir is None:
            continue

        figures = result if isinstance(result, list) else [result]
        for i, fig in enumerate(figures, start=1):
            if fig is None:
                continue
            index_suffix = f"_{i}" if len(figures) > 1 else ""
            filename = (
                f"{plot_name}_{filename_suffix}"
                f"{index_suffix}.svg"
            )

            fig.fig.savefig(
                save_dir / filename,
                format="svg",
                bbox_inches="tight",
            )

    if plot_figures:
        plt.show()


def main() -> None:
    rol = load_model_topic("rol")
    rol2 = load_model_topic("rol_2")
    rol_fixed = load_model_topic("rol_fixed")
    rol_nd = load_model_topic("nondim_2")
    
    generate_plots(
        models={
            # "Roll": rol,
            "Small Roll": rol2,
            # "Roll Fixed": rol_fixed,
            # "Small Roll": rol_time,
            # "Nondim Roll": rol_nd,
            },
        # combine_models=False,
         start_time=368,
        end_time=16+368,
        plot_labels={
            # "title": "",
            # "subtitle": "",
            # "time": "Time [s]",
            # "terms":{
            #     0: {"term": "Roll Acceleration", "units": "rad/s²"},
            #     1: {"term": "Roll Rate", "units": "rad/s", "param_units": "1/s"},
            #     2: {"term": "Aileron Command", "units": "PWM", "param_units": "rad/s²-PWM"},
            # }
        },
        plot_figures=False,
        save_figures=True,
    )

if __name__ == "__main__":
    main()