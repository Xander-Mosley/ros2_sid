import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lombscargle

from plotter_class import PlotFigure
from signal_processing import linear_diff, savitzky_golay_diff, rolling_diff, LowPassFilter, smooth_data_array, LowPassFilterVariableDT, smooth_data_with_timestamps_LP, ButterworthLowPassVariableDT, smooth_data_with_timestamps_Butter, butterworthlowpass_loop


def load_signal_data(file_path, t_slice=slice(0, 9999999)):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    t, x = data[t_slice, 0], data[t_slice, 4]
    t = t[::2]
    x = x[::2]
    return t, x


def preprocess_signal(t, x, cutoff_pre=10, cutoff_post=5):
    fx = butterworthlowpass_loop(x, t, cutoff_frequency=cutoff_pre)
    xp = rolling_diff(t, fx, "sg")
    xp_full = np.concatenate(([0] * 5, xp[5:]))
    fxp = butterworthlowpass_loop(xp_full, t, cutoff_frequency=cutoff_post)
    return fx, xp_full, fxp


def time_statistics(t):
    dt = np.diff(t)
    print("")
    print(f"Time Step\t\tSampling Rate")
    print(f"=========\t\t=============")
    print(f"Min: {round(np.min(dt), 3)} s\t\tMax: {round(np.max(1/dt), 2)} Hz")
    print(f"Max: {round(np.max(dt), 3)} s\t\tMin: {round(np.min(1/dt), 2)} Hz")
    print(f"Avg: {round(np.mean(dt), 3)} s\t\tAvg: {round(np.mean(1/dt), 2)} Hz")
    print(f"Std: {round(np.std(dt), 3)} s\t\tStd: {round(np.std(1/dt), 2)} Hz")
    print("")
    

def _compute_fft(t, *signals, f=None):
    t = np.asarray(t)
    signals = [np.asarray(sig) for sig in signals]
    if f is None:
        dt_mean = np.mean(np.diff(t))
        f_max = 0.5 / dt_mean
        n_freqs = len(t)
        f = np.linspace(0, f_max, n_freqs)
    ffts = []
    for sig in signals:
        Xf = np.array([np.sum(sig * np.exp(-2j * np.pi * freq * t)) for freq in f])
        ffts.append(Xf)
    return f, ffts

def signal_analysis(t, x, y):
    time_figure = PlotFigure("Signal Analysis - Time Domain")
    time_figure.define_subplot(0, ylabel="Amplitude", xlabel="Time [s]", grid=True)
    time_figure.add_scatter(0, t, x, label="Input", color="tab:blue")
    time_figure.add_data(0, t, y, label="Output", color="tab:orange")
    time_figure.set_all_legends()

    f, (X, Y) = _compute_fft(t, x, y)
    def to_dB(x):
        return 20 * np.log10(np.abs(x) + 1e-12)
    
    freq_figure = PlotFigure("Signal Analysis - Frequency Spectrum",  nrows=2, sharex=True)
    freq_figure.define_subplot(0, ylabel="Magnitude", grid=True)
    freq_figure.add_data(0, f, np.abs(X), label="Input", color="tab:blue")
    freq_figure.add_data(0, f, np.abs(Y), label="Output", color="tab:orange")
    freq_figure.define_subplot(1, ylabel="Magnitude [dB]", xlabel="Frequency [Hz]", grid=True)
    freq_figure.add_data(1, f, to_dB(np.abs(X)), label="Input", color="tab:blue")
    freq_figure.add_data(1, f, to_dB(np.abs(Y)), label="Output", color="tab:orange")
    freq_figure.set_all_legends()
    
    H = Y / X
    mag = to_dB(H)
    phase = np.angle(H, deg=True)
    bode_figure = PlotFigure(f"Signal Analysis - Bode Plot", nrows=2, sharex=True)
    bode_figure.define_subplot(0, ylabel="Magnitude [dB]", grid=True)
    bode_figure.set_log_scale(0, axis='x')
    bode_figure.add_data(0, f, mag, label="Magnitude", color="tab:blue")
    bode_figure.add_line(0, -3, orientation='h', color='tab:red', label='-3 dB')
    bode_figure.define_subplot(1, ylabel="Phase [deg]", xlabel="Frequency [Hz]", grid=True)
    bode_figure.set_log_scale(1, axis='x')
    bode_figure.add_data(1, f, phase, label="Phase")
    bode_figure.set_all_legends()

    # dt = np.mean(np.diff(t))
    # fs = 1 / dt
    # T = t[-1] - t[0]
    # f_min = 1 / T
    # f_max = fs / 2
    # valid = (f >= f_min) & (f <= f_max)
    # phase_rad = np.unwrap(np.angle(H))
    # df = np.gradient(f[valid])
    # group_delay_sec = np.full_like(f, np.nan)
    # group_delay_sec[valid] = -np.gradient(phase_rad[valid], df) / (2 * np.pi)
    # group_delay_samples = np.full_like(f, np.nan)
    # group_delay_samples[valid] = group_delay_sec[valid] * fs

    # delay_figure = PlotFigure("Signal Analysis - Phase Delay", nrows=2, sharex=True)
    # delay_figure.define_subplot(0, ylabel="Phase Delay [s]", grid=True)
    # delay_figure.add_scatter(0, f, group_delay_sec, color="tab:blue", label="Phase Delay (s)")
    # delay_figure.define_subplot(1, ylabel="Phase Delay [samples]", xlabel="Frequency [Hz]", grid=True)
    # delay_figure.add_scatter(1, f, group_delay_samples, color="tab:orange", label="Phase Delay (samples)")
    # delay_figure.set_all_legends()


def main(file_path):
    t, x = load_signal_data(file_path, t_slice=slice(0, 999999))
    fx, xp, fxp = preprocess_signal(t, x, 1.54, 1.54)

    time_statistics(t)
    # signal_analysis(t, x, fx)
    # signal_analysis(t, fx, xp)
    # signal_analysis(t, xp, fxp)
    signal_analysis(t, x, fxp)

    plt.show()


if __name__ == "__main__":
    # main("/develop_ws/src/ros2_sid/ros2_sid/ros2_sid/maneuvers/saved_maneuver.csv")
    main("/develop_ws/bag_files/topic_data_files/imu_data.csv")
    # main("/develop_ws/bag_files/topic_data_files/imu_raw_data.csv")
