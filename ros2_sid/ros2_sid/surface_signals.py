"""Pure normalized-surface waveforms and monotonic scheduling helpers."""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

AxisSample = tuple[float, float, float]
AXES = ("roll", "pitch", "yaw")


class WaveformSpec(TypedDict, total=False):
    """JSON waveform fields; Experiment validates fields needed by each signal kind."""

    signal: str
    amplitude: float
    start_frequency: float
    end_frequency: float
    frequencies: list[float]
    amplitudes: list[float]
    phases: list[float]
    normalize: bool


class Experiment:
    """Validated constant, linear-chirp, multisine, or interpolated CSV experiment."""

    def __init__(self, path: str | Path) -> None:
        """Load and validate a normalized-surface JSON experiment or CSV source."""
        self.path = Path(path)
        self.config: dict[str, Any] = json.loads(self.path.read_text())
        c = self.config
        if c.get("units") != "normalized_surface":
            raise ValueError(
                "DDS missions require units: normalized_surface; radians are not surface fractions"
            )
        self.rate = float(c.get("rate", 100))
        self.duration = float(c["duration"])
        if not (
            math.isfinite(self.rate)
            and 10 <= self.rate <= 400
            and math.isfinite(self.duration)
            and self.duration > 0
        ):
            raise ValueError("invalid rate or duration")
        self.axes: dict[str, WaveformSpec] = c.get("axes", {})
        if set(self.axes) - set(AXES):
            raise ValueError("unknown axis")
        self.rows: list[tuple[float, ...]] | None = None
        if "csv" in c:
            if self.axes:
                raise ValueError("select CSV or generated axes")
            with (self.path.parent / c["csv"]).open() as f:
                reader = csv.reader(f)
                next(reader)
                self.rows = [tuple(map(float, row)) for row in reader]
            if len(self.rows) < 2 or any(
                len(r) != 4 or not all(map(math.isfinite, r)) for r in self.rows
            ):
                raise ValueError("CSV requires finite time,roll,pitch,yaw rows")
            self.times = [r[0] for r in self.rows]
            if self.times[0] != 0 or any(
                b <= a for a, b in zip(self.times, self.times[1:])
            ):
                raise ValueError("CSV time must start at zero and increase")
            if self.duration > self.times[-1] or any(
                abs(v) > 1 for row in self.rows for v in row[1:]
            ):
                raise ValueError("CSV duration or amplitude out of bounds")
        for spec in self.axes.values():
            kind = spec["signal"]
            amplitude = float(spec.get("amplitude", 0.1))
            if not math.isfinite(amplitude) or abs(amplitude) > 1:
                raise ValueError("amplitude must be finite in [-1,1]")
            if kind == "chirp":
                frequencies = [spec["start_frequency"], spec["end_frequency"]]
            elif kind == "multisine":
                frequencies = spec["frequencies"]
                amps = spec["amplitudes"]
                phases = spec.get("phases", [0] * len(amps))
                if not frequencies or not len(frequencies) == len(amps) == len(phases):
                    raise ValueError("multisine component lengths must match")
                if not all(math.isfinite(x) for x in amps + phases):
                    raise ValueError("nonfinite multisine")
                peak = sum(abs(x) for x in amps)
                limit = abs(amplitude)
                if peak > limit:
                    if not spec.get("normalize", False):
                        raise ValueError(
                            "multisine amplitude sum exceeds amplitude; request normalize explicitly"
                        )
                    spec["amplitudes"] = [x * limit / peak for x in amps]
            elif kind == "constant":
                frequencies = []
            else:
                raise ValueError("unknown waveform")
            if any(
                not math.isfinite(f) or f <= 0 or f >= self.rate / 2
                for f in frequencies
            ):
                raise ValueError(
                    "frequency must be positive and below publication Nyquist"
                )

    def sample(self, t: float) -> AxisSample:
        """Return roll, pitch and yaw fractions at elapsed seconds; zero outside the experiment."""
        if not 0 <= t < self.duration:
            return (0.0, 0.0, 0.0)
        if self.rows is not None:
            i = min(len(self.rows) - 2, bisect.bisect_right(self.times, t) - 1)
            a, b = self.rows[i : i + 2]
            fraction = (t - a[0]) / (b[0] - a[0])
            return (
                a[1] + fraction * (b[1] - a[1]),
                a[2] + fraction * (b[2] - a[2]),
                a[3] + fraction * (b[3] - a[3]),
            )
        result = []
        for axis in AXES:
            s = self.axes.get(axis)
            if s is None:
                result.append(0.0)
                continue
            amplitude = s.get("amplitude", 0.1)
            if s["signal"] == "constant":
                v = amplitude
            elif s["signal"] == "chirp":
                f0, f1 = s["start_frequency"], s["end_frequency"]
                v = amplitude * math.sin(
                    2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * self.duration))
                )
            else:
                v = sum(
                    a * math.sin(2 * math.pi * f * t + p)
                    for a, f, p in zip(
                        s["amplitudes"],
                        s["frequencies"],
                        s.get("phases", [0] * len(s["frequencies"])),
                    )
                )
            result.append(v)
        return result[0], result[1], result[2]


class Cadence:
    """Deadline-based scheduling that never publishes a backlog of stale samples."""

    def __init__(self, rate: float, now: float) -> None:
        """Initialize publication frequency in Hz and monotonic start time in seconds."""
        self.period = 1 / rate
        self.start = now
        self.deadline = now

    def due(self, now: float) -> float | None:
        """Return elapsed seconds if due, skip missed deadlines, and reject excessive jitter."""
        if now < self.deadline:
            return None
        if now - self.deadline > 2 * self.period:
            raise RuntimeError("publication jitter exceeds two periods")
        elapsed = now - self.start
        self.deadline = (
            self.start + (math.floor(elapsed / self.period) + 1) * self.period
        )
        return elapsed


class ClockSync:
    """Map local monotonic time to AP time while detecting jumps and stale updates."""

    def __init__(self) -> None:
        """Start without an authorized mapping to the flight-controller clock."""
        self.remote: float | None = None
        self.local: float | None = None

    def update(self, remote: float, now: float) -> None:
        """Accept AP-clock and local monotonic seconds; invalidate discontinuous mappings."""
        if (
            self.remote is not None
            and self.local is not None
            and abs((remote - self.remote) - (now - self.local)) > 0.05
        ):
            self.remote = None
            self.local = None
            raise RuntimeError("flight-controller clock discontinuity")
        self.remote = remote
        self.local = now

    def stamp(self, now: float) -> float:
        """Extrapolate AP-clock seconds; raise RuntimeError for missing or stale synchronization."""
        if self.remote is None or self.local is None or not 0 <= now - self.local <= 1:
            raise RuntimeError("flight-controller clock unavailable or stale")
        return self.remote + now - self.local


class RCGate:
    """Fail-closed RC authorization requiring a fresh low-to-high switch transition."""

    def __init__(
        self, channel: int = 6, threshold: int = 1800, timeout: float = 0.5
    ) -> None:
        """Configure a zero-based RC channel, PWM threshold, and freshness timeout in seconds."""
        self.channel = channel
        self.threshold = threshold
        self.timeout = timeout
        self.last: float | None = None
        self.high = False
        self.saw_low = False

    def update(self, values: Sequence[int], mask: int, now: float) -> None:
        """Observe a valid RC sample and require a low sample before a high switch position."""
        if self.channel >= len(values) or not mask & (1 << self.channel):
            self.last = None
            self.high = False
            self.saw_low = False
            return
        high = values[self.channel] >= self.threshold
        if not high:
            self.saw_low = True
        self.high = high
        self.last = now

    def allowed(self, now: float) -> bool:
        """Return whether the switch is freshly high after observing low; stale input resets it."""
        if self.last is None or now - self.last > self.timeout:
            self.saw_low = False
            return False
        return self.saw_low and self.high

    def reset(self) -> None:
        """Require another low-to-high RC transition before activation."""
        self.saw_low = False
