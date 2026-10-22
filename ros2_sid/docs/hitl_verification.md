# HITL entrypoint verification — 2026-10-22

The actual `run_mission_HITL.py --interface dds` entrypoint was exercised against
an isolated **PC ArduPlane SITL**, using simulated RC channel 8 via MAVROS. These
results do not validate the physical Cube's receiver wiring or onboard timing.
The Cube and its domain-0 serial DDS agent were not commanded during these tests.

## Implementation and checks

- Legacy mapping preserved: one-based channel 8 -> zero-based index 7, threshold
  1550 µs; high runs and low stops. Legacy trajectory code is unchanged.
- Shared DDS waveform/library/navigation path; MAVROS RC is the default, DDS RC
  is an explicit alternative. Startup-high, stale input, invalid/missing input,
  inter-tick low/high and console-start rejection have regression coverage.
- MAVROS `rc_io` enabled. The simulator uses `FLTMODE_CH=5`, `RC8_OPTION=0` so its
  experiment switch cannot also change flight mode.
- Final relevant unit/ROS regression suite: **51 tests passed**. ROS package
  build passed. Results are retained in `final_tests.log`.
- Six full experiments passed: chirp and multisine on roll, pitch and yaw.
  Every experiment confirmed LOITER recovery. Safety probe and mode-change checks
  passed. Rosbag/DataFlash analysis passed with 25,036 actuator samples and a
  maximum calibrated PWM prediction error of 1 µs.

[Execution results](../../../sid_results/hitl_rc_verified_20261022/execution.json),
[numeric analysis](../../../sid_results/hitl_rc_verified_20261022/plots/report.json),
[recorded bag](../../../sid_results/hitl_rc_verified_20261022/rosbag), and
[plot/test log](../../../sid_results/hitl_rc_verified_20261022/plot_tests.log)
are retained with all six command CSVs and navigation event logs.

An earlier run (`sid_results/hitl_rc_six_20261022`) exposed the default Plane
channel-8 mode-selection conflict: raising the switch selected MANUAL and the
publisher withheld excitation. That run was stopped and retained; it is not a
successful experiment. No physical Cube assignments were changed to resolve it.

## Recorded command and response plots

These figures use recorded commands, applied output and aircraft response from
the six successful runs; they are not synthetic illustrations. Each also has a
PDF alongside its PNG. PWM is commanded output, not measured surface position.

### Roll frequency sweep

![Roll chirp](../../../sid_results/hitl_rc_verified_20261022/plots/individual_axes/chirp_roll.png)

### Roll multisine

![Roll multisine](../../../sid_results/hitl_rc_verified_20261022/plots/individual_axes/multisine_roll.png)

### Pitch frequency sweep

![Pitch chirp](../../../sid_results/hitl_rc_verified_20261022/plots/individual_axes/chirp_pitch.png)

### Pitch multisine

![Pitch multisine](../../../sid_results/hitl_rc_verified_20261022/plots/individual_axes/multisine_pitch.png)

### Yaw frequency sweep

![Yaw chirp](../../../sid_results/hitl_rc_verified_20261022/plots/individual_axes/chirp_yaw.png)

### Yaw multisine

![Yaw multisine](../../../sid_results/hitl_rc_verified_20261022/plots/individual_axes/multisine_yaw.png)

### RC gate and applied excitation

![RC timing](../../../sid_results/hitl_rc_verified_20261022/plots/hitl_rc_timing.png)

The RC timing plot uses bag receipt timestamps, which include transport latency.
Refer to the [manual walkthrough](manual_surface_excitation_hitl.md) to record
and regenerate these plots from your own experiments.

## RC-low interruption

A separate isolated run lowered simulated RC channel 8 during ACTIVE. The HITL
publisher logged the RC stop, confirmed zero applied perturbation about **0.24 s**
later, and confirmed LOITER about **0.28 s** after stop detection. These intervals
come from publisher logs, not physical switch-to-actuator measurements.

[RC-stop run](../../../sid_results/hitl_rc_stop_20261022/) retains the command CSV,
rosbag, firmware logs and navigation transitions. The maneuver was deliberately
interrupted, so its command log covers only the transmitted part of the chirp.

The RC-stop run also passed its automated RC-stop detection, LOITER recovery,
safety probe and bag validation.

![RC-low abort timing](../../../sid_results/hitl_rc_stop_20261022/plots/hitl_rc_timing.png)
