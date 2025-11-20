#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plotter_class.py — A structured wrapper for creating multi-subplot matplotlib figures.

Description
-----------
This module defines the `PlotFigure` class, a high-level wrapper for
matplotlib's figure and axes objects. It simplifies figure creation and
management when working with multiple subplots by providing intuitive
helper methods for common plotting tasks — such as adding lines, scatter
plots, bar charts, histograms, and shaded regions — without requiring
manual subplot indexing or repetitive boilerplate code.

The class also includes convenience methods for applying consistent
titles, grids, legends, and log scaling across all subplots.

Intended Use
------------
Import and use this class as a utility for cleaner, more organized
visualization scripts or notebooks:

    >>> from plotter_class import PlotFigure
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> fig = PlotFigure("Demo", nrows=1, ncols=2)
    >>> fig.add_data(0, x, y, label="sin(x)")
    >>> fig.add_scatter(1, x, np.cos(x), label="cos(x)")
    >>> fig.set_all_legends()

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 11 Nov 2025
"""

import warnings
from typing import Any, Callable, Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ArrayLike = Union[float, Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]


__all__ = ['PlotFigure']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


class PlotFigure:
    """
    A convenience wrapper for managing multi-subplot matplotlib figures.

    This class simplifies creating, updating, and formatting figures
    with multiple subplots, while providing helper methods for adding
    common plot types (lines, scatter, bars, histograms, etc.).
    """

    def __init__(
        self,
        fig_title: Optional[str] = None,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[float, float] = (8, 6),
        sharex: bool = False,
        sharey: bool = False
        ) -> None:
        """
        Initialize a PlotFigure with a grid of subplots.

        Parameters
        ----------
        fig_title : Optional[str], default = None
            Title for the entire figure.
        nrows : int, default = 1
            Number of subplot rows.
        ncols : int, default = 1
            Number of subplot columns.
        figsize : Tuple[float, float], default = (8, 6)
            Size of the figure in inches.
        sharex : bool, default = False
            Share x-axis among subplots.
        sharey : bool, default = False
            Share y-axis among subplots.
        """
        self.fig, self.axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey
            )
        self.nrows, self.ncols = nrows, ncols
        self.plot_data: dict[Tuple[int, Optional[str]], object] = {}
        self.secondary_axes: dict[int, Axes] = {}

        if fig_title:
            self.fig.suptitle(fig_title, fontsize=14)

    @property
    def all_axes(self) -> list[Axes]:
        """
        Get all axes in the figure as a flat iterable.

        Returns
        -------
        list of matplotlib.axes.Axes
            Flattened list of subplot axes.
        """
        if isinstance(self.axes, np.ndarray):
            return list(self.axes.flat)
        return [self.axes]


    def _add_plot_data(
        self,
        ax_pos: int,
        plot_func: Callable[[Axes], object],
        label: Optional[str] = None
        ) -> object:
        """
        Helper method to add a plot element to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        plot_func : Callable[[matplotlib.axes.Axes], object]
            Function that takes an axis and adds a plot.
        label : Optional[str], default = None
            Legend label.

        Returns
        -------
        object
            The created plot object (line, bar, scatter, etc.).
        """
        ax = self._get_ax(ax_pos)
        obj = plot_func(ax)
        if label:
            self._maybe_legend(ax, label)
        self.plot_data[(ax_pos, label)] = obj
        return obj
    
    def _axis_index(self, ax: Axes) -> int:
        """Return the subplot index for a given primary axis."""
        if isinstance(self.axes, np.ndarray):
            for i, a in enumerate(self.axes.flat):
                if a is ax:
                    return i
        else:
            if self.axes is ax:
                return 0
        raise ValueError("Axis not found in figure.")

    def _ensure_numpy(self, data: Optional[ArrayLike]) -> np.ndarray:
        """
        Convert any array-like input to a proper NumPy array.

        Parameters
        ----------
        data : ArrayLike
            Input data: float, list, tuple, np.ndarray, pd.Series, or pd.DataFrame.

        Returns
        -------
        np.ndarray
            A NumPy array with the same data.
        """
        if data is None:
            return None # type: ignore
        if isinstance(data, np.ndarray):
            return data
        try:
            if isinstance(data, pd.Series):
                return np.asarray(data)
            if isinstance(data, pd.DataFrame):
                # Single-column DataFrame becomes 1D array
                if data.shape[1] == 1:
                    return np.asarray(data.iloc[:, 0])
                return np.asarray(data)
        except ImportError:
            pass
        # Default fallback for lists, tuples, etc.
        return np.array(data)

    def _get_ax(self, ax_pos: int) -> Axes:
        """
        Retrieve a specific subplot axis by index.

        Parameters
        ----------
        ax_pos : int
            Subplot index (0-based).

        Returns
        -------
        matplotlib.axes.Axes
            The requested subplot axis.

        Raises
        ------
        ValueError
            If 'ax_pos' is out of range.
        """
        try:
            return (
                self.axes.flat[ax_pos]
                if isinstance(self.axes, np.ndarray)
                else self.axes
                )
        except IndexError:
            raise ValueError(
                f"Invalid subplot index {ax_pos}. "
                f"Figure has {self.nrows * self.ncols} subplots."
                )
        
    def _get_secondary_axis(self, ax_pos: int) -> Axes:
        """
        Get the secondary y-axis for a subplot.
        Creates it if it does not already exist.

        Parameters
        ----------
        ax_pos : int
            Subplot index.

        Returns
        -------
        matplotlib.axes.Axes
            The secondary y-axis (twinx axis).
        """
        if ax_pos in self.secondary_axes:
            return self.secondary_axes[ax_pos]
        ax = self._get_ax(ax_pos)
        sec_ax = ax.twinx()
        self.secondary_axes[ax_pos] = sec_ax
        return sec_ax

    def _maybe_legend(self, ax: Axes, label: Optional[str]) -> None:
        """
        Add a legend to the axis if a label is provided.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to modify.
        label : Optional[str]
            Label for the legend.
        """
        if label:
            ax.legend()


    def set_figure_title(self, title: str, fontsize: int = 14) -> None:
        """
        Set or update the figure title.

        Parameters
        ----------
        title : str
            New figure title.
        fontsize : int, default = 14
            Font size of the title.
        """
        self.fig.suptitle(title, fontsize=fontsize)

    def define_subplot(
        self,
        ax_pos: int,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        y2label: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        y2lim: Optional[Tuple[float, float]] = None,
        grid: bool = False,
        grid_kwargs: Optional[dict] = None
        ) -> None:
        """
        Configure subplot labels, limits, grid, and optional secondary y-axis.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        title : Optional[str]
            Subplot title.
        xlabel : Optional[str]
            X-axis label.
        ylabel : Optional[str]
            Primary y-axis label.
        y2label : Optional[str]
            Secondary y-axis label (creates a secondary y-axis if provided).
        xlim : Optional[Tuple[float, float]]
            X-axis limits.
        ylim : Optional[Tuple[float, float]]
            Primary y-axis limits.
        y2lim : Optional[Tuple[float, float]]
            Secondary y-axis limits.
        grid : bool
            Enable gridlines.
        grid_kwargs : Optional[dict]
            Extra grid style arguments.
        """
        ax = self._get_ax(ax_pos)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if grid:
            grid_kwargs = grid_kwargs or {"alpha": 0.3, "linestyle": "--"}
            ax.grid(True, **grid_kwargs)
        # ---- Secondary y-axis ----
        if y2label or y2lim:
            ax2 = self._get_secondary_axis(ax_pos)
            if y2label:
                ax2.set_ylabel(y2label)
            if y2lim:
                ax2.set_ylim(*y2lim)

    def apply_to_all_axes(self, func: str, *args, **kwargs) -> None:
        """
        Apply a method or attribute call to all subplot axes.

        Parameters
        ----------
        func : str
            Name of an axis method (e.g., 'set_xlim', 'grid').
        *args
            Positional arguments for the method.
        **kwargs
            Keyword arguments for the method.

        Notes
        -----
        Useful for batch-applying the same property across subplots.
        """
        for ax in self.all_axes:
            getattr(ax, func)(*args, **kwargs)

    def set_all_legends(self, **kwargs) -> None:
        """
        Display legends on all subplots, merging primary + secondary axes
        only for the axes that belong together.
        """
        for ax in self.all_axes:
            ax_pos = self._axis_index(ax)
            handles1, labels1 = ax.get_legend_handles_labels()
            sec_ax = self.secondary_axes.get(ax_pos, None)

            if sec_ax is not None:
                handles2, labels2 = sec_ax.get_legend_handles_labels()
            else:
                handles2, labels2 = [], []

            handles = handles1 + handles2
            labels = labels1 + labels2
            if handles:
                ax.legend(handles, labels, **kwargs)

    def set_all_grids(self, enabled: bool = True, **kwargs) -> None:
        """
        Enable or disable grid lines on all subplots.

        Parameters
        ----------
        enabled : bool, default = True
            Whether to enable grid lines.
        **kwargs
            Passed to 'matplotlib.axes.Axes.grid'.
        """
        for ax in self.all_axes:
            ax.grid(enabled, **kwargs)

    def set_log_scale(
        self,
        ax_pos: int,
        axis: str = 'y',
        base: float = 10
        ) -> None:
        """
        Set logarithmic scaling on a subplot axis.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        axis : str, {'x', 'y'}, default = 'y'
            Axis to apply log scale.
        base : float, default = 10
            Logarithmic base.

        Raises
        ------
        ValueError
            If 'axis' is not 'x' or 'y'.
        """
        ax = self._get_ax(ax_pos)
        if axis.lower() == 'y':
            ax.set_yscale('log', base=base)
        elif axis.lower() == 'x':
            ax.set_xscale('log', base=base)
        else:
            raise ValueError("Axis must be 'x' or 'y'")

    def shade_subplot(
        self,
        ax_pos: int,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        color: str = 'gray',
        alpha: float = 0.3,
        **kwargs
        ) -> None:
        """
        Shade a rectangular region on a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x_range : Optional[Tuple[float, float]], default = None
            Range on the x-axis to shade.
        y_range : Optional[Tuple[float, float]], default = None
            Range on the y-axis to shade.
        color : str, default = 'gray'
            Shade color.
        alpha : float, default = 0.3
            Transparency of shading.
        **kwargs
            Passed to 'ax.axvspan' or 'ax.axhspan'.
        """
        ax = self._get_ax(ax_pos)
        if x_range:
            ax.axvspan(x_range[0], x_range[1], color=color, alpha=alpha, **kwargs)
        if y_range:
            ax.axhspan(y_range[0], y_range[1], color=color, alpha=alpha, **kwargs)

    def color_axis(self, ax_pos: int, color: str) -> None:
        """
        Color the primary y-axis spines, ticks, and label.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        color : str
            Color to apply.
        """
        ax = self._get_ax(ax_pos)
        ax.spines["left"].set_color(color)
        ax.tick_params(axis="y", colors=color)
        ylabel = ax.get_ylabel()
        if ylabel:
            ax.yaxis.label.set_color(color)

    def color_axis_secondary_y(self, ax_pos: int, color: str) -> None:
        """
        Color the secondary y-axis spines, ticks, and label.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        color : str
            Color to apply.
        """
        ax2 = self.secondary_axes.get(ax_pos)
        if ax2 is None:
            raise ValueError(
                f"No secondary axis exists for subplot {ax_pos}. "
                f"Create one with add_data_secondary_y() or y2label first."
            )
        ax2.spines["right"].set_color(color)
        ax2.tick_params(axis="y", colors=color)
        ylabel = ax2.get_ylabel()
        if ylabel:
            ax2.yaxis.label.set_color(color)


    def add_data(
        self,
        ax_pos: int,
        x: ArrayLike,
        y: ArrayLike,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a simple line plot to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            X-axis data.
        y : np.ndarray
            Y-axis data.
        label : Optional[str], default = None
            Line label for legend.
        **kwargs
            Additional arguments passed to 'matplotlib.axes.Axes.plot'.

        Returns
        -------
        matplotlib.lines.Line2D
            The created line object.
        """
        line = self._add_plot_data(
            ax_pos,
            lambda ax: ax.plot(
                self._ensure_numpy(x),
                self._ensure_numpy(y),
                label=label,
                **kwargs
                )[0],
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return line

    def add_data_secondary_y(
        self,
        ax_pos: int,
        x: ArrayLike,
        y: ArrayLike,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a line plot to a secondary y-axis on the given subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x, y : array-like
            Data for plotting.
        label : str, optional
            Legend label.
        kwargs :
            Passed to matplotlib.axes.Axes.plot
        """
        line = self._add_plot_data(
            ax_pos,
            lambda ax: self._get_secondary_axis(ax_pos).plot(
                self._ensure_numpy(x),
                self._ensure_numpy(y),
                label=label,
                **kwargs
                )[0],
            label
            )
        if axis_color:
            self.color_axis_secondary_y(ax_pos, axis_color)
        return line

    def add_scatter(
        self,
        ax_pos: int,
        x: ArrayLike,
        y: ArrayLike,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a scatter plot to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            X data.
        y : np.ndarray
            Y data.
        label : Optional[str], default = None
            Label for legend.
        **kwargs
            Passed to 'matplotlib.axes.Axes.scatter'.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        """
        scatter_obj  = self._add_plot_data(
            ax_pos,
            lambda ax: ax.scatter(
                self._ensure_numpy(x),
                self._ensure_numpy(y),
                label=label,
                **kwargs
                ),
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return scatter_obj 

    def add_scatter_secondary_y(
        self,
        ax_pos: int,
        x: ArrayLike,
        y: ArrayLike,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a scatter plot to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            X data.
        y : np.ndarray
            Y data.
        label : Optional[str], default = None
            Label for legend.
        **kwargs
            Passed to 'matplotlib.axes.Axes.scatter'.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object.
        """
        scatter_obj  = self._add_plot_data(
            ax_pos,
            lambda ax: self._get_secondary_axis(ax_pos).scatter(
                self._ensure_numpy(x),
                self._ensure_numpy(y),
                label=label,
                **kwargs
                ),
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return scatter_obj 

    def add_bar(
        self,
        ax_pos: int,
        x: ArrayLike,
        height: ArrayLike,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a bar plot to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            Bar positions.
        height : np.ndarray
            Heights of the bars.
        label : Optional[str], default = None
            Label for the bars.
        **kwargs
            Passed to 'matplotlib.axes.Axes.bar'.

        Returns
        -------
        list of matplotlib.patches.Rectangle
            The bar container object.
        """
        bar_container = self._add_plot_data(
            ax_pos,
            lambda ax: ax.bar(
                self._ensure_numpy(x),
                self._ensure_numpy(height),
                label=label,
                **kwargs
                ),
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return bar_container

    def add_bar_secondary_y(
        self,
        ax_pos: int,
        x: ArrayLike,
        height: ArrayLike,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a bar plot to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            Bar positions.
        height : np.ndarray
            Heights of the bars.
        label : Optional[str], default = None
            Label for the bars.
        **kwargs
            Passed to 'matplotlib.axes.Axes.bar'.

        Returns
        -------
        list of matplotlib.patches.Rectangle
            The bar container object.
        """
        bar_container = self._add_plot_data(
            ax_pos,
            lambda ax: self._get_secondary_axis(ax_pos).bar(
                self._ensure_numpy(x),
                height,
                label=label,
                **kwargs
                ),
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return bar_container

    def add_fill_between(
        self,
        ax_pos: int,
        x: ArrayLike,
        y1: ArrayLike,
        y2: Union[ArrayLike, float] = 0.0,
        label: Optional[str] = None,
        color: str = 'gray',
        alpha: float = 0.3,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a shaded area between two curves or a baseline.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : ArrayLike
            X-axis data.
        y1 : ArrayLike
            Upper curve.
        y2 : Union[ArrayLike, float], default = 0.0
            Lower curve or constant baseline.
        label : Optional[str], default = None
            Legend label.
        color : str, default = 'gray'
            Fill color.
        alpha : float, default = 0.3
            Transparency of fill.
        axis_color : Optional[str], default = None
            Color the primary axis.
        **kwargs
            Passed to `matplotlib.axes.Axes.fill_between`.

        Returns
        -------
        matplotlib.collections.PolyCollection
            The filled area object.
        """
        poly_obj = self._add_plot_data(
            ax_pos,
            lambda ax: ax.fill_between(
                self._ensure_numpy(x),
                self._ensure_numpy(y1),
                self._ensure_numpy(y2),
                label=label,
                color=color,
                alpha=alpha,
                **kwargs
                ),
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return poly_obj
    
    def add_fill_between_secondary_y(
        self,
        ax_pos: int,
        x: ArrayLike,
        y1: ArrayLike,
        y2: Union[ArrayLike, float] = 0.0,
        label: Optional[str] = None,
        color: str = 'gray',
        alpha: float = 0.3,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a shaded area between two curves or a baseline.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : ArrayLike
            X-axis data.
        y1 : ArrayLike
            Upper curve.
        y2 : Union[ArrayLike, float], default = 0.0
            Lower curve or constant baseline.
        label : Optional[str], default = None
            Legend label.
        color : str, default = 'gray'
            Fill color.
        alpha : float, default = 0.3
            Transparency of fill.
        axis_color : Optional[str], default = None
            Color the primary axis.
        **kwargs
            Passed to `matplotlib.axes.Axes.fill_between`.

        Returns
        -------
        matplotlib.collections.PolyCollection
            The filled area object.
        """
        poly_obj = self._add_plot_data(
            ax_pos,
            lambda ax: self._get_secondary_axis(ax_pos).fill_between(
                self._ensure_numpy(x),
                self._ensure_numpy(y1),
                self._ensure_numpy(y2),
                label=label,
                color=color,
                alpha=alpha,
                **kwargs
                ),
            label
            )
        if axis_color:
            self.color_axis(ax_pos, axis_color)
        return poly_obj


    def add_errorbar(
        self,
        ax_pos: int,
        x: ArrayLike,
        y: ArrayLike,
        yerr: Optional[ArrayLike] = None,
        xerr: Optional[ArrayLike] = None,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add an error bar plot to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            X data.
        y : np.ndarray
            Y data.
        yerr : Optional[np.ndarray], default = None
            Y-axis error values.
        xerr : Optional[np.ndarray], default = None
            X-axis error values.
        label : Optional[str], default = None
            Legend label.
        **kwargs
            Passed to 'matplotlib.axes.Axes.errorbar'.

        Returns
        -------
        matplotlib.container.ErrorbarContainer
            The created errorbar object.
        """
        return self._add_plot_data(
            ax_pos,
            lambda ax: ax.errorbar(
                self._ensure_numpy(x),
                self._ensure_numpy(y),
                yerr=self._ensure_numpy(yerr),
                xerr=self._ensure_numpy(xerr),
                label=label,
                **kwargs
                ),
            label
            )

    def add_hist(
        self,
        ax_pos: int,
        data: ArrayLike,
        bins: int = 10,
        label: Optional[str] = None,
        axis_color: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a histogram to a subplot.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        data : np.ndarray
            Input data.
        bins : int, default = 10
            Number of histogram bins.
        label : Optional[str], default = None
            Legend label.
        **kwargs
            Passed to 'matplotlib.axes.Axes.hist'.

        Returns
        -------
        Tuple
            The histogram output (n, bins, patches).
        """
        return self._add_plot_data(
            ax_pos,
            lambda ax: ax.hist(
                self._ensure_numpy(data),
                bins=bins,
                label=label,
                **kwargs
                ),
            label
            )

    def add_line(
        self,
        ax_pos: int,
        value: float,
        orientation: str = 'h',
        label: Optional[str] = None,
        **kwargs
        ) -> object:
        """
        Add a horizontal or vertical reference line.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        value : float
            Location of the line.
        orientation : {'h', 'v'}, default = 'h'
            'h' for horizontal, 'v' for vertical.
        label : Optional[str], default = None
            Label for legend.
        **kwargs
            Passed to 'matplotlib.axes.Axes.axhline' or 'axvline'.

        Returns
        -------
        matplotlib.lines.Line2D
            The created line object.

        Raises
        ------
        ValueError
            If 'orientation' is not 'h' or 'v'.
        """
        if orientation not in ('h', 'v'):
            raise ValueError(
                "orientation must be either 'h' (horizontal) or 'v' (vertical)"
                )
        plot_func = (
            (lambda ax: ax.axhline(y=value, label=label, **kwargs))
            if orientation == 'h'
            else (lambda ax: ax.axvline(x=value, label=label, **kwargs))
            )
        return self._add_plot_data(ax_pos, plot_func, label)

    def add_shade(
        self,
        ax_pos: int,
        x: ArrayLike,
        y1: ArrayLike,
        y2: Union[ArrayLike, float] = 0.0,
        label: Optional[str] = None,
        color: str = 'gray',
        alpha: float = 0.3,
        **kwargs
        ) -> object:
        """
        Add a shaded region between two curves or a baseline.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        x : np.ndarray
            X-axis data.
        y1 : np.ndarray
            Upper curve.
        y2 : Union[np.ndarray, float], default = 0.0
            Lower curve or constant baseline.
        label : Optional[str], default = None
            Label for legend.
        color : str, default = 'gray'
            Fill color.
        alpha : float, default = 0.3
            Transparency of fill.
        **kwargs
            Passed to 'matplotlib.axes.Axes.fill_between'.

        Returns
        -------
        matplotlib.collections.PolyCollection
            The shaded area object.
        """
        return self._add_plot_data(
            ax_pos,
            lambda ax: ax.fill_between(
                self._ensure_numpy(x),
                self._ensure_numpy(y1),
                self._ensure_numpy(y2),
                label=label,
                color=color,
                alpha=alpha,
                **kwargs
                ),
            label
            )


if (__name__ == '__main__'):
    warnings.warn(
        "This script defines a class that visually simplifies lines of code for plotting."
        " It is intented to be imported, not executed directly."
        "\n\tImport the class from this script using:\t"
        "from plotter_class import PlotFigure",
        UserWarning)