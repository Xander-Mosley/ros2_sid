# Create your own DDS frequency sweeps and multisines

The default DDS mission uses precomputed CSVs in the same mission-library layout
as legacy. `surface_signals.Experiment` can generate custom signals from JSON for
preview and CSV export; explicit JSON overrides can also generate them at runtime.
This guide configures normalized surface excitation, not attitude commands.

## Automatic mission selection

From a sourced ROS container shell, you can launch without a file argument:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds
```

Both interfaces now use the same library structure:

```text
ros2_sid/setup/mission_library/
  current_mission.json       # {"subfolder": "dds_surface"}
  dds_surface/
    mission_plan.json        # units + ordered filename/name/order entries
    chirp_roll.csv           # time,roll,pitch,yaw
    ...
```

The shared pointer selects the mission folder. DDS preloads its ordered CSVs and
infers cadence/duration from their uniform timestamps. Plans must declare
`"units": "normalized_surface"`; legacy attitude plans are rejected. Select
`mission_1` in the same pointer when using legacy trajectory mode.
The old pointer in `config/surface_excitation` is no longer read.

Explicit `--experiment FILE` still supports runtime waveform generation from JSON,
and `--mission-plan FILE` accepts both existing DDS JSON plans and the CSV layout.
The examples below generate signals for preview or explicit JSON operation; the
last section converts them into the default legacy-style CSV library.

## Create a custom mission folder

Inside the container (the checkout is mounted at `/work`):

```bash
export CUSTOM=/work/ros2_sid/ros2_sid/config/surface_excitation/my_mission
mkdir -p "$CUSTOM"
```

Choose a new folder name for a new experiment series. The commands below write the
named files; edit existing files intentionally rather than overwriting past runs.

## Frequency sweep / chirp

Create a pitch-only linear frequency sweep:

```bash
cat > "$CUSTOM/pitch_chirp.json" <<'JSON'
{
  "units": "normalized_surface",
  "rate": 50,
  "duration": 20,
  "axes": {
    "pitch": {
      "signal": "chirp",
      "amplitude": 0.05,
      "start_frequency": 0.2,
      "end_frequency": 2.0
    }
  }
}
JSON
```

This generates `u(t) = A sin(2π[f0 t + (f1−f0)t²/(2T)])` during the experiment:

| Field | Meaning in this example |
|---|---|
| `rate` | Publish 50 samples/s |
| `duration` | Run for 20 seconds |
| `pitch` | Excite pitch only; omitted axes remain zero |
| `amplitude` | Peak requested perturbation 0.05 = 5% normalized deflection |
| `start_frequency` | Begin at 0.2 Hz |
| `end_frequency` | Sweep linearly toward 2 Hz over the duration |

Change `pitch` to `roll` or `yaw` for another axis. The generated chirp is linear;
there is no logarithmic-sweep selector in this DDS generator. At the duration
boundary, the publisher requests disable and firmware handles the return to baseline.

## Multisine

Create a roll-only multisine:

```bash
cat > "$CUSTOM/roll_multisine.json" <<'JSON'
{
  "units": "normalized_surface",
  "rate": 50,
  "duration": 20,
  "axes": {
    "roll": {
      "signal": "multisine",
      "amplitude": 0.10,
      "frequencies": [0.5, 1.0, 2.0],
      "amplitudes": [0.02, 0.03, 0.04],
      "phases": [0.0, 1.0, 2.0]
    }
  }
}
JSON
```

The signal is the sum `Σ Ak sin(2π fk t + φk)`. Entries at the same index form a
component: 0.5 Hz with amplitude 0.02 and phase 0 radians, then 1 Hz/0.03/1 radian,
then 2 Hz/0.04/2 radians. All three arrays must have equal lengths. If `phases` is
omitted, every component starts with phase zero.

For a multisine, **`amplitude` is a bound, not an extra multiplier**. The component
amplitudes above have absolute sum 0.09, so the conservative peak bound is 9%, below
the declared 10% limit. The actual peak can be lower depending on phases.

If the absolute sum exceeds `amplitude`, the configuration is rejected unless you
add `"normalize": true` to that axis specification. With normalization, every
component amplitude is multiplied by `abs(amplitude) / sum(abs(amplitudes))`.
For example, `[0.08, 0.06, 0.06]` with bound `0.10` becomes `[0.04, 0.03, 0.03]`.
The generator does not optimize multisine phases automatically.

To request simultaneous axes, add independent `roll`, `pitch`, and/or `yaw` objects
under `axes`. Each may use its own waveform and frequencies; the publication rate
and duration are shared. Mixed surfaces may combine axes before the firmware's
physical-surface limit is applied.

## Limits and timing

Values are signed fractions of calibrated surface deflection, not degrees or PWM.
Firmware limits (`SX_R_MAX`, `SX_P_MAX`, `SX_Y_MAX`, and `SX_S_MAX`) and available
servo travel still limit the applied output. Increasing a JSON amplitude does not
override them. Use the applied-output record to determine what reached the outputs.

Frequencies must be positive and strictly below half the configured publication
rate. This parser check is only a sampling constraint; it does not establish a
suitable bandwidth for an aircraft or its actuator/control loop. The examples use
50 Hz, matching the individual-axis SITL experiments. Normal stabilization remains
active, so measured aircraft response is closed-loop response.

Files are loaded at publisher startup. Stop and restart the publisher after editing
amplitudes or frequencies; the console does not hot-reload or edit them.

## Validate and preview without publishing

After sourcing ROS and the installed workspace, run:

```bash
python3 - "$CUSTOM" <<'PY'
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ros2_sid.surface_signals import Experiment

folder = Path(sys.argv[1])
for filename in ("pitch_chirp.json", "roll_multisine.json"):
    experiment = Experiment(folder / filename)
    times = [i / experiment.rate for i in range(int(experiment.duration * experiment.rate))]
    samples = [experiment.sample(t) for t in times]
    stem = Path(filename).stem
    with (folder / f"{stem}_preview.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "roll", "pitch", "yaw"])
        writer.writerows((t, *sample) for t, sample in zip(times, samples))
    fig, ax = plt.subplots()
    for index, axis in enumerate(("roll", "pitch", "yaw")):
        ax.plot(times, [sample[index] for sample in samples], label=axis)
    ax.set(xlabel="Elapsed time (s)", ylabel="Normalized requested excitation", title=stem)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / f"{stem}_preview.png")
    plt.close(fig)
    print(f"Validated {filename}; saved preview CSV and PNG")
PY
```

This calls the same waveform generator used by the DDS publisher, but creates no
ROS node and sends no commands. Preview plots show the requested waveform only;
they are not simulated or measured aircraft responses.

## Add a menu and make it the default

Create a plan next to your two experiments:

```bash
cat > "$CUSTOM/mission_plan.json" <<'JSON'
{
  "units": "normalized_surface",
  "maneuvers": [
    {"name": "My pitch chirp", "experiment": "pitch_chirp.json"},
    {"name": "My roll multisine", "experiment": "roll_multisine.json"}
  ]
}
JSON
```

Experiment paths are relative to the plan. To try it immediately, specify the source
plan explicitly (after the simulator is ready in airborne GUIDED):

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --mission-plan "$CUSTOM/mission_plan.json" --command-log "$CUSTOM/commands.csv"
```

To make these waveforms the automatic default using the legacy file structure,
generate CSVs into a new mission folder:

```bash
python3 - "$CUSTOM" <<'PYCODE'
import csv
import json
import sys
from pathlib import Path
from ros2_sid.surface_signals import Experiment

source = Path(sys.argv[1])
library = Path("/work/ros2_sid/ros2_sid/ros2_sid/setup/mission_library")
destination = library / "my_surface_mission"
destination.mkdir(parents=True, exist_ok=True)
entries = []
for order, filename in enumerate(("pitch_chirp.json", "roll_multisine.json"), 1):
    experiment = Experiment(source / filename)
    csv_name = Path(filename).stem + ".csv"
    with (destination / csv_name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "roll", "pitch", "yaw"])
        for i in range(round(experiment.duration * experiment.rate) + 1):
            t = i / experiment.rate
            writer.writerow([t, *experiment.sample(t)])
    entries.append({"filename": csv_name, "name": Path(filename).stem, "order": order})
(destination / "mission_plan.json").write_text(json.dumps(
    {"units": "normalized_surface", "maneuvers": entries}, indent=2))
(library / "current_mission.json").write_text(json.dumps({"subfolder": destination.name}))
PYCODE
```

Rebuild before the next flight/recording, from the host checkout root:

```bash
docker compose -f docker-compose.sid.yaml restart ros
docker compose -f docker-compose.sid.yaml up -d --wait --wait-timeout 900
```

Reenter the container and source ROS/the installed workspace. Run
`ros2 run ros2_sid run_mission_SITL.py --interface dds`, then `list`, `select 0`,
and `start`. Files are preloaded: restart the publisher after changes. Preserve
CSV/JSON files alongside the bag to reproduce the exact requested input.
