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
from typing import Callable, Optional, Union, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", "GTK3Agg", depending on your system
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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
        self.lines: dict[Tuple[int, Optional[str]], object] = {}

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

    def _add_plot(
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
        self.lines[(ax_pos, label)] = obj
        return obj

    def add_data(
        self,
        ax_pos: int,
        x: np.ndarray,
        y: np.ndarray,
        label: Optional[str] = None,
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
        return self._add_plot(
            ax_pos,
            lambda ax: ax.plot(x, y, label=label, **kwargs)[0],
            label
            )

    def add_scatter(
        self,
        ax_pos: int,
        x: np.ndarray,
        y: np.ndarray,
        label: Optional[str] = None,
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
        return self._add_plot(
            ax_pos,
            lambda ax: ax.scatter(x, y, label=label, **kwargs),
            label
            )

    def add_bar(
        self,
        ax_pos: int,
        x: np.ndarray,
        height: np.ndarray,
        label: Optional[str] = None,
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
        return self._add_plot(
            ax_pos,
            lambda ax: ax.bar(x, height, label=label, **kwargs),
            label
            )

    def add_hist(
        self,
        ax_pos: int,
        data: np.ndarray,
        bins: int = 10,
        label: Optional[str] = None,
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
        return self._add_plot(
            ax_pos,
            lambda ax: ax.hist(data, bins=bins, label=label, **kwargs),
            label
            )

    def add_errorbar(
        self,
        ax_pos: int,
        x: np.ndarray,
        y: np.ndarray,
        yerr: Optional[np.ndarray] = None,
        xerr: Optional[np.ndarray] = None,
        label: Optional[str] = None,
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
        return self._add_plot(
            ax_pos,
            lambda ax: ax.errorbar(
                x, y, yerr=yerr, xerr=xerr, label=label, **kwargs
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
        return self._add_plot(ax_pos, plot_func, label)

    def add_shade(
        self,
        ax_pos: int,
        x: np.ndarray,
        y1: np.ndarray,
        y2: Union[np.ndarray, float] = 0.0,
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
        return self._add_plot(
            ax_pos,
            lambda ax: ax.fill_between(
                x, y1, y2, label=label, color=color, alpha=alpha, **kwargs
                ),
            label
            )

    def define_subplot(
        self,
        ax_pos: int,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        grid: bool = False,
        grid_kwargs: Optional[dict] = None
        ) -> None:
        """
        Configure subplot labels, limits, and grid.

        Parameters
        ----------
        ax_pos : int
            Subplot index.
        title : Optional[str], default = None
            Subplot title.
        xlabel : Optional[str], default = None
            X-axis label.
        ylabel : Optional[str], default = None
            Y-axis label.
        xlim : Optional[Tuple[float, float]], default = None
            X-axis limits.
        ylim : Optional[Tuple[float, float]], default = None
            Y-axis limits.
        grid : bool, default = False
            Whether to enable gridlines.
        grid_kwargs : Optional[dict], default = None
            Additional arguments passed to 'Axes.grid'.
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

    def set_all_legends(self, **kwargs) -> None:
        """
        Display legends on all subplots.

        Parameters
        ----------
        **kwargs
            Passed to 'matplotlib.axes.Axes.legend'.
        """
        for ax in self.all_axes:
            ax.legend(**kwargs)

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


if (__name__ == '__main__'):
    warnings.warn(
        "This script defines a class that visually simplifies lines of code for plotting."
        " It is intented to be imported, not executed directly."
        "\n\tImport the class from this script using:\t"
        "from plotter_class import PlotFigure",
        UserWarning)