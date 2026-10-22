# DDS surface-excitation development

> **DDS default:** `run_mission_SITL.py --interface dds` now loads the packaged
> `straight_flight_sitl.json` automatically. MAVROS is required: defaults target
> 75 m above home, 23 m/s, and the current ground course, then wait for settling.
> `--straight-flight PATH` overrides these settings; `--no-straight-flight`
> explicitly disables navigation coordination and automatic LOITER recovery.
> The shared HITL DDS entrypoint has the same default and retains its RC gate;
> the packaged settings remain SITL examples, not validated physical-flight settings.
> The legacy trajectory interface is unchanged. Rebuild/restart the ROS service
> before using this new default; do not restart it during a running experiment.


For automatic mission loading and creating your own chirps/multisines, see [Custom DDS waveforms](docs/custom_surface_waveforms.md).

The shared DDS backend is selected from `run_mission_SITL.py` or
`run_mission_HITL.py` with `--interface dds`. The legacy trajectory backend retains
its existing behavior. Architecture, C4 diagrams, usage, build instructions and
experiment evidence are in `ardupilot_sid/docs/control_surface_excitation/` in the
outer workspace.

For a hands-on walkthrough, see [Run the DDS surface-excitation SITL tests yourself](docs/manual_surface_excitation_sitl.md). It covers setup, the six single-axis maneuver tests, interactive maneuver control, ROS bags, and plotting.

To enter the container and send waveforms directly, follow [Manual container waveform commands](docs/manual_container_waveforms.md).

## Automatic ROS setup

After building the DDS-enabled Plane binary, run from the outer workspace:

```bash
docker compose -f docker-compose.sid.yaml up -d --wait --wait-timeout 900
```

The ROS service builds the interfaces and mission packages and runs the surface
Python tests before becoming healthy. See the manual guide for logs and rebuilding
an already running service after source changes.

## Module responsibilities

- `surface_signals.py`: validated waveform definitions, monotonic cadence, clock
  synchronization and RC authorization. Pure Python; no ROS dependency.
- `surface_mission.py`: ROS endpoints, publisher-epoch acknowledgment, experiment
  selection and shutdown. Console input queues events; the timer owns mission state.
- `surface_bag.py`: offline rosbag2 deserialization and clock-domain conversion.
  It never replays commands to an aircraft.
- `surface_analysis.py`: calibrated PWM checks, complete-maneuver validation,
  DataFlash comparison, aligned response export and PNG/PDF figures.
- `surface_axis_plots.py`: six single-axis chirp/multisine report figures from a
  validated ROS bag, with checks that other axes and surfaces remain unexcited.

Public classes/functions have docstrings and type annotations. `WaveformSpec`
describes JSON waveform fields and `AxisSample` fixes the roll/pitch/yaw tuple
shape. Heterogeneous decoded DataFlash/ROS record dictionaries use `Any` at the
serialization boundary; numerical helpers and stateful publisher interfaces are
typed. Dynamic ROS-generated classes have no complete local typing stubs.

## Quality checks

Run from this ROS package directory using a development environment with Ruff and
mypy installed:

```bash
python3 -m ruff check --select E,F,I,ANN,D --ignore E501,D203,D213 \
  ros2_sid/surface*.py scripts/plot_surface_excitation.py
python3 -m ruff format --check ros2_sid/surface*.py \
  scripts/plot_surface_excitation.py tests/test_surface*.py
python3 -m mypy --strict --allow-subclassing-any --ignore-missing-imports \
  --follow-imports=skip ros2_sid/surface*.py
```

The mypy subclass allowance accommodates the external, untyped `rclpy.node.Node`
base class. It does not disable annotation checks on the new modules. Imports are
skipped because firmware-generated ROS interfaces and ROS itself are supplied by
the sourced ROS environment; runtime ROS tests cover those boundaries.

After sourcing ROS Humble and the built workspace:

```bash
python3 -m pytest tests/test_surface_signals.py tests/test_surface_analysis.py \
  tests/test_surface_mission_ros.py
```

The same three suites are registered with `ament_cmake_pytest` for `colcon test`.
A run that skips ROS tests because `rclpy` is unavailable is not a complete test
result. SITL acceptance also validates actual actuator outputs, full waveform
coverage, clock/sequence safeguards and recorded bag/DataFlash agreement.

## Individual-axis flight plots

The SITL runner can execute chirp and multisine separately on roll, pitch and yaw.
From the outer workspace, with the isolated ROS container and agent running:

```bash
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --individual-axes --rate 50 --output sid_results/individual_axes
```

After building/sourcing the ROS package, regenerate the six detailed figures from
the recorded bag (paths below assume the container's `/work` mount):

```bash
python3 -m ros2_sid.surface_axis_plots \
  --run /work/sid_results/individual_axes \
  --output /work/ardupilot_sid/docs/control_surface_excitation/figures/individual_axes
```

The renderer requires the run's successful output-validation report and all six
command CSV/experiment JSON files. It writes PNG/PDF pairs and `summary.json`.
Each plot shows the selected command and applied logical excitation, physical PWM
perturbations, all three body rates, and the selected axis's unwrapped attitude change. Normal GUIDED
motion remains in the response; nonzero cross-axis response is not evidence that
another excitation axis was commanded. The figures are embedded in the firmware
`VERIFICATION.md` report.

## Shared legacy/DDS mission library

Both backends resolve `ros2_sid/setup/mission_library/current_mission.json` and its
`subfolder`. The selected folder contains `mission_plan.json` with ordered
`filename`/`name`/`order` entries and maneuver CSVs. DDS requires the additional
plan marker `"units": "normalized_surface"`. The default selection is now
`dds_surface`; select `mission_1` for legacy trajectory operation. Backend checks
prevent interpreting surface fractions as legacy attitude commands or vice versa.
Run `run_mission_SITL.py --interface dds` with no source argument for the default.
See the custom-waveform guide for generating CSVs and switching missions.

## Select a recording directory

Inside the container, select an existing interactive SITL run without copying its
timestamp:

```bash
source /work/tools/sid/use_run.sh --latest interactive_
```

For a new waveform session, use `--new container_waves` once, then
`--latest container_waves_` in other shells. Check the printed path; `--path DIRECTORY`
selects an exact existing run. Always use `source` so the helper exports `RUN`
into the shell where you will record, publish, or analyze.


## Default straight-flight coordination

See [straight-flight setup and safe exit](docs/straight_flight_plan.md) for the
`--straight-flight` option shared by the SITL/HITL DDS entrypoints. It uses MAVROS
for fixed navigation targets and settling, then the existing DDS waveform path.
Normal completion and manual exit observe zero perturbation before requesting
LOITER, while preserving operator/failsafe mode changes. Bag-based plots include
course, altitude, airspeed, phases, and ground track. Physical flight is unverified.

## Cube Orange Plus Simulation on Hardware

See [installation and flashing](docs/cubeorangeplus_simulation_on_hardware.md) for
building this fork with onboard Plane simulation and custom DDS, connecting the
two USB interfaces, and returning to standard ArduPlane firmware.

The [manual HITL guide](docs/manual_surface_excitation_hitl.md) covers the legacy
RC channel-8 switch, shared DDS mission library, recording and RC timing plots.
