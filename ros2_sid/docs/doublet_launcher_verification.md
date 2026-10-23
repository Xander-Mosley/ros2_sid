# DDS doublets and interactive launcher verification

Implemented on 2026-10-22. Physical Cube/receiver testing was not performed.

## What changed

- `surface_doublets.py` generates normalized CSVs with the unchanged legacy
  `input_designs.multi_step` function: `[1, 1]` pulses, 1 Hz natural frequency,
  one-second delay, 0.02-second samples and 15-second duration. The default
  amplitude is 0.10. Legacy rounding and boundary zero samples are preserved.
- Roll, pitch and yaw doublets are appended at menu indices 6–8 in `dds_surface`.
  Existing chirp/multisine indices remain unchanged. Both mission entrypoints
  use this library; HITL's RC channel-8 gate and stop behavior are unchanged.
- `tools/sid/run_surface_test.sh` starts four tmux panes, initializes/reuses the
  Docker services, rebuilds ROS interfaces/package, snapshots settings and the
  selected library, and chooses unique paths. `session.json` records the paths.
- `tools/sid/plot_surface_test.sh` reads the manifest, refuses an active recording,
  and analyzes the closed bag. Multiple interactive maneuvers are split into
  derived validation CSVs; the raw command log is preserved.
- The recorder reports its actual suffixed path through a sidecar. Successful
  mission shutdown is followed by a configurable recording tail (90 s default).
  Failed recovery leaves recording available. Connections, especially the HITL
  GCS router, remain available until explicitly stopped by the operator.
- `surface_axis_plots.py --waveform doublet` produces per-axis figures from bags.
  No excitation firmware change was needed for doublets.

## Executed checks

| Check | Result |
|---|---|
| ROS package tests, including legacy-generator comparison | 59 passed |
| Offline launcher tests: names, commands, readiness failure, failed recovery, actual bag path, repeated maneuvers | 7 passed |
| Existing live-launcher/recorder regression test | Passed |
| Shell syntax, Python compilation, selected Ruff correctness/import checks | Passed |
| HITL Compose configuration resolution | Passed; no device connection attempted |
| Roll/pitch/yaw doublets through SITL interactive menu | All passed |
| Roll/pitch/yaw doublets through HITL entrypoint with simulated RC | All passed |
| RC-low interruption during a nonzero doublet pulse | Passed |
| Real four-pane launcher: menu selection, doublet, recovery, automatic bag closure | Passed |
| Plot helper on closed SITL and simulated-HITL bags, including RC analysis | Passed |

The first launcher attempt exposed a healthy container with an older installed
recorder. Its flight/recovery passed, but automatic path finalization failed.
The launcher now rebuilds before opening panes and can resolve older finalized
bags from metadata. The first bag was preserved and successfully replotted after
finalization. A second fresh launcher session passed automatic closure without
manual intervention.

## Retained simulation evidence

All paths below are relative to the checkout. Bags include all ROS topics;
command CSVs, DataFlash logs, coordinator events and validation reports are retained.

| Run | Artifacts | PWM comparison |
|---|---|---|
| Three SITL doublets | `sid_results/dds_doublets_sitl_20261022/` | 14,902 frames; max error 1 µs |
| Three simulated-HITL doublets | `sid_results/dds_doublets_hitl_20261022/` | 15,795 frames; max error 1 µs |
| Simulated RC-low stop | `sid_results/dds_doublet_rc_stop_20261022/` | 5,494 frames; max error 1 µs |
| Final launcher integration | `sid_results/sitl_launcher_verified_20261022_211950/` | See `plots/report.json` |
| Initial launcher attempt | `sid_results/sitl_launcher_validation_20261022_210156/` | Preserved for traceability |

At the RC-stop transition, applied roll excitation was approximately **−0.10**.
Zero excitation was observed before the navigation handoff; recovery confirmed
LOITER at the saved start point. These are simulated RC values, not receiver data.

### Commands versus response

Figures show requested/applied excitation, baseline/final servo PWM, angular rates,
attitude changes and airspeed. The green region is the commanded maneuver. Large
motion beyond its end may be the return-to-LOITER turn, not a doublet response.
PWM is commanded output, not measured physical surface position.

| Axis | SITL | HITL entrypoint in PC simulation |
|---|---|---|
| Roll | [PNG](../../../sid_results/dds_doublets_sitl_20261022/plots/doublets/doublet_roll.png) | [PNG](../../../sid_results/dds_doublets_hitl_20261022/plots/doublets/doublet_roll.png) |
| Pitch | [PNG](../../../sid_results/dds_doublets_sitl_20261022/plots/doublets/doublet_pitch.png) | [PNG](../../../sid_results/dds_doublets_hitl_20261022/plots/doublets/doublet_pitch.png) |
| Yaw | [PNG](../../../sid_results/dds_doublets_sitl_20261022/plots/doublets/doublet_yaw.png) | [PNG](../../../sid_results/dds_doublets_hitl_20261022/plots/doublets/doublet_yaw.png) |

PDF versions are saved beside each PNG.

![Recorded roll doublet and response](../../../sid_results/dds_doublets_sitl_20261022/plots/doublets/doublet_roll.png)

## Reproduce

Normal interactive workflow, from the checkout root:

```bash
./tools/sid/run_surface_test.sh sitl
# Or, with the Cube and the documented serial prerequisites:
./tools/sid/run_surface_test.sh hitl

# After quitting the mission and allowing recording to close:
./tools/sid/plot_surface_test.sh --latest sitl
./tools/sid/plot_surface_test.sh --latest hitl
```

The automated harness uses an isolated UDP agent on port 2022 and ROS domain 42,
with a running ROS container. These commands arm/take off **PC SITL**, not hardware:

```bash
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 --output sid_results/NEW_doublets_sitl \
  --interactive-library --straight-flight --doublets-only \
  --dds-domain 42 --dds-port 2022

python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 --output sid_results/NEW_doublets_hitl \
  --hitl-rc --doublets-only --dds-domain 42 --dds-port 2022

python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 --output sid_results/NEW_doublet_stop \
  --hitl-rc --only-experiment doublet_roll --exit-active rc-low --exit-after 1.1 \
  --dds-domain 42 --dds-port 2022
```

The menu-driving launcher test is reproducible with
`python3 tools/sid/test_surface_session_sitl.py SESSION_DIRECTORY` after starting
an isolated `run_surface_test.sh sitl --detach --return-record-seconds 5` session.
It performs normal simulated arming/takeoff via the launcher's MAVProxy UDP 14553.

Inside a sourced ROS container, replot the per-axis harness figures directly:

```bash
python3 -m ros2_sid.surface_axis_plots \
  /work/sid_results/dds_doublets_sitl_20261022/rosbag \
  --command-directory /work/sid_results/dds_doublets_sitl_20261022 \
  --output /work/sid_results/dds_doublets_sitl_20261022/plots/doublets \
  --waveform doublet
```

## Limits

The physical HITL USB/MAVLink-router/GCS path was configured and checked offline,
not exercised on the Cube. Nothing was flashed. The existing return-to-start
firmware revision remains required. Abruptly killing tmux or the container is not
an orderly flight-test stop. Use the mission's quit/RC stop and observe recovery.
