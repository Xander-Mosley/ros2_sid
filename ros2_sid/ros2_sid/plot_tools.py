import math
import numpy as np
import matplotlib.pyplot as plt

from discrete_diff import linear_diff, savitzky_golay_diff, rolling_diff, LowPassFilter, smooth_data_array, LowPassFilterVariableDT, smooth_data_with_timestamps_LP, ButterworthLowPassVariableDT, smooth_data_with_timestamps_Butter, butterworthlowpass_loop



class PlotFigure:
    def __init__(self, fig_title=None, nrows=1, ncols=1, figsize=(8,6), sharex=False, sharey=False):
        self.fig, self.axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey
        )
        self.nrows, self.ncols = nrows, ncols
        self.lines = {}  # {(ax_pos, label): line/collection}
        
        if fig_title:
            self.fig.suptitle(fig_title, fontsize=14)

    @property
    def all_axes(self):
        return self.axes.flat if isinstance(self.axes, np.ndarray) else [self.axes]

    def _get_ax(self, ax_pos):
        try:
            return self.axes.flat[ax_pos] if isinstance(self.axes, np.ndarray) else self.axes
        except IndexError:
            raise ValueError(f"Invalid subplot index {ax_pos}. Figure has {self.nrows * self.ncols} subplots.")

    def _maybe_legend(self, ax, label):
        if label:
            ax.legend()

    def _add_plot(self, ax_pos, plot_func, label=None):
        ax = self._get_ax(ax_pos)
        obj = plot_func(ax)
        if label:
            self._maybe_legend(ax, label)
        self.lines[(ax_pos, label)] = obj
        return obj


    def add_data(self, ax_pos, x, y, label=None, **kwargs):
        return self._add_plot(ax_pos, lambda ax: ax.plot(x, y, label=label, **kwargs)[0], label)

    def add_scatter(self, ax_pos, x, y, label=None, **kwargs):
        return self._add_plot(ax_pos, lambda ax: ax.scatter(x, y, label=label, **kwargs), label)

    def add_bar(self, ax_pos, x, height, label=None, **kwargs):
        return self._add_plot(ax_pos, lambda ax: ax.bar(x, height, label=label, **kwargs), label)

    def add_hist(self, ax_pos, data, bins=10, label=None, **kwargs):
        return self._add_plot(ax_pos, lambda ax: ax.hist(data, bins=bins, label=label, **kwargs), label)

    def add_errorbar(self, ax_pos, x, y, yerr=None, xerr=None, label=None, **kwargs):
        return self._add_plot(ax_pos, lambda ax: ax.errorbar(x, y, yerr=yerr, xerr=xerr, label=label, **kwargs), label)
    
    def add_line(self, ax_pos, value, orientation='h', label=None, **kwargs):
        if orientation not in ('h', 'v'):
            raise ValueError("orientation must be either 'h' for horizontal or 'v' for vertical")
        plot_func = (lambda ax: ax.axhline(y=value, label=label, **kwargs) if orientation == 'h' else lambda ax: ax.axvline(x=value, label=label, **kwargs))
        return self._add_plot(ax_pos, plot_func, label)

    def add_shade(self, ax_pos, x, y1, y2=0, label=None, color='gray', alpha=0.3, **kwargs):
        return self._add_plot(ax_pos, lambda ax: ax.fill_between(x, y1, y2, label=label, color=color, alpha=alpha, **kwargs), label)


    def define_subplot(self, ax_pos, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, grid=False, grid_kwargs=None):
        ax = self._get_ax(ax_pos)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        if grid:
            grid_kwargs = grid_kwargs or {"alpha":0.3, "linestyle":"--"}
            ax.grid(True, **grid_kwargs)

    def shade_subplot(self, ax_pos, x_range=None, y_range=None, color='gray', alpha=0.3, **kwargs):
        ax = self._get_ax(ax_pos)
        if x_range: ax.axvspan(x_range[0], x_range[1], color=color, alpha=alpha, **kwargs)
        if y_range: ax.axhspan(y_range[0], y_range[1], color=color, alpha=alpha, **kwargs)

    def set_log_scale(self, ax_pos, axis='y', base=10):
        ax = self._get_ax(ax_pos)
        if axis.lower() == 'y': ax.set_yscale('log', base=base)
        elif axis.lower() == 'x': ax.set_xscale('log', base=base)
        else: raise ValueError("Axis must be 'x' or 'y'")


    def set_figure_title(self, title, fontsize=14):
        self.fig.suptitle(title, fontsize=fontsize)

    def set_all_grids(self, enabled=True, **kwargs):
        for ax in self.all_axes:
            ax.grid(enabled, **kwargs)

    def set_all_legends(self, **kwargs):
        for ax in self.all_axes:
            ax.legend(**kwargs)

    def apply_to_all_axes(self, func, *args, **kwargs):
        for ax in self.all_axes:
            getattr(ax, func)(*args, **kwargs)



def load_signal_data(file_path, t_slice=slice(0, 9999999)):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    t, x = data[t_slice, 0], data[t_slice, 4]
    # t = t[::2]
    # x = x[::2]
    return t, x


def preprocess_signal(t, x, cutoff_pre=10, cutoff_post=5):
    fx = butterworthlowpass_loop(x, t, cutoff_frequency=cutoff_pre)
    xp = rolling_diff(t, fx, "sg")
    xp_full = np.concatenate(([0] * 5, xp[5:]))
    fxp = butterworthlowpass_loop(xp_full, t, cutoff_frequency=cutoff_post)
    return fx, xp_full, fxp


def compute_fft(t, *signals):
    dt = t[1] - t[0]
    f = np.fft.fftfreq(len(t), dt)
    mask = f >= 0
    f = f[mask]
    ffts = [np.fft.fft(sig)[mask] for sig in signals]
    return f, ffts



def plot_time_domain(t, x, fx, xp, fxp):
    fig = PlotFigure("Discrete Differentiation", nrows=2, sharex=True)
    fig.define_subplot(0, ylabel="Signal Amplitude", xlabel="Time [s]")
    fig.add_scatter(0, t, x, label="Raw")
    fig.add_data(0, t, fx, label="Pre-filtered", color="tab:orange")

    fig.define_subplot(1, ylabel="Derivative", xlabel="Time [s]")
    fig.add_scatter(1, t, xp, label="Differentiated")
    fig.add_data(1, t, fxp, label="Post-filtered", color="tab:orange")
    fig.set_all_legends()
    return fig


def plot_frequency_spectrum(f, X, FX, XP, FXP):
    fig = PlotFigure("Frequency Spectrum Analysis", nrows=2, sharex=False)
    for i, (sig, label) in enumerate([(X, "Raw vs Pre-filtered"), (XP, "Derivative vs Post-filtered")]):
        fig.define_subplot(i, xlabel="Frequency [Hz]", ylabel="Magnitude", grid=True)
        # fig.set_log_scale(i, axis='x')
    fig.add_data(0, f, np.abs(X), label="Raw")
    fig.add_data(0, f, np.abs(FX), label="Pre-filtered")
    fig.add_data(1, f, np.abs(XP), label="Differentiated")
    fig.add_data(1, f, np.abs(FXP), label="Post-filtered")
    fig.set_all_legends()
    return fig


def plot_frequency_spectrum_dB(f, X, FX, XP, FXP):
    def to_dB(x):
        return 20 * np.log10(np.abs(x) + 1e-12)
    fig = PlotFigure("Frequency Spectrum Analysis", nrows=2, sharex=False)
    for i, (sig, label) in enumerate([(X, "Raw vs Pre-filtered"), (XP, "Derivative vs Post-filtered")]):
        fig.define_subplot(i, xlabel="Frequency [Hz]", ylabel="Magnitude", grid=True)
        # fig.set_log_scale(i, axis='x')
    fig.add_data(0, f, to_dB(np.abs(X)), label="Raw")
    fig.add_data(0, f, to_dB(np.abs(FX)), label="Pre-filtered")
    fig.add_data(1, f, to_dB(np.abs(XP)), label="Differentiated")
    fig.add_data(1, f, to_dB(np.abs(FXP)), label="Post-filtered")
    fig.set_all_legends()
    return fig


def plot_bode(f, pairs):
    def bode_components(num, den):
        H = num / den
        return 20 * np.log10(np.abs(H)), np.angle(H, deg=True)

    for title, (num, den) in pairs.items():
        mag, phase = bode_components(num, den)
        fig = PlotFigure(f"Bode Plot: {title}", nrows=2, sharex=True)
        fig.define_subplot(0, ylabel="Magnitude [dB]", grid=True)
        fig.set_log_scale(0, axis='x')
        fig.add_data(0, f, mag, label="Magnitude")
        fig.add_line(0, -3, orientation='h', color='red', label='-3 dB')
        fig.define_subplot(1, xlabel="Frequency [Hz]", ylabel="Phase [deg]", grid=True)
        fig.set_log_scale(1, axis='x')
        fig.add_data(1, f, phase, label="Phase")
        fig.set_all_legends()


def main(file_path):
    t, x = load_signal_data(file_path, t_slice=slice(0, 999999))
    # t, x = load_signal_data(file_path, t_slice=slice(2700, 3450))
    fx, xp, fxp = preprocess_signal(t, x, 1.25, 1.25)
    f, (X, FX, XP, FXP) = compute_fft(t, x, fx, xp, fxp)


    dt = np.diff(t[1:])
    print("")
    print("Min dt: " + str(np.min(dt)))
    print("Max dt: " + str(np.max(dt)))
    print("Avg dt: " + str(np.mean(dt)))
    print("Std dt: " + str(np.std(dt)))
    print("")
    print(f"Max Sample Rate: {round(np.max(1/dt),2)} Hz")
    print(f"Min Sample Rate: {round(np.min(1/dt),2)} Hz")
    print(f"Avg Sample rate: {round(np.mean(1/dt),2)} Hz")
    print(f"Std Sample rate: {round(np.std(1/dt),2)} Hz")
    print("")
    

    plot_time_domain(t, x, fx, xp, fxp)
    plot_frequency_spectrum(f, X, FX, XP, FXP)
    plot_frequency_spectrum_dB(f, X, FX, XP, FXP)
    plot_bode(f, {
        "FX vs X": (FX, X),
        # "XP vs FX": (XP, FX),
        "FXP vs XP": (FXP, XP),
        "FXP vs X": (FXP, X),
    })

    plt.show()


if __name__ == "__main__":
    # main("/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/saved_maneuver.csv")
    # main("/develop_ws/bag_files/topic_data_files/imu_data.csv")
    main("/develop_ws/bag_files/topic_data_files/imu_raw_data.csv")
