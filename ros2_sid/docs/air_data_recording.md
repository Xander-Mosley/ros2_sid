# Recording angle of attack and sideslip

Use the same SITL/HITL launcher, maneuver menu and plotting commands as before.
The all-topic recorder includes `/ap/air_data` and, on simulator builds,
`/ap/air_data/truth`. No additional mission command-line flags are required.

```bash
./tools/sid/run_surface_test.sh sitl
# After mission quit, recovery handoff and recorder closure:
./tools/sid/plot_surface_test.sh --latest sitl
```

Use `hitl` in place of `sitl` for the hardware launcher. This feature does not
change RC channel-8 authorization, excitation limits or return-to-start handling.
Compile/install the updated firmware and ROS interfaces together. Physical Cube
operation has not been validated by the PC simulation experiments. In particular,
UDP results do not establish the sustainable rate on a shared USB/serial link.

## Rate parameter

`AHRS_AOA_RATE` accepts **20, 50 or 100 Hz**. Firmware defaults to 20 Hz;
identification parameter examples select 50 Hz. Unsupported values fall back to
20 Hz. Change it through the existing GCS parameter interface, for example in
MAVProxy:

```text
param set AHRS_AOA_RATE 50
param show AHRS_AOA_RATE
```

No reboot is needed for this parameter. It is a maximum calculation cadence,
not a guarantee: the AHRS loop and scheduling determine the achieved rate. The
launcher does not change flight-loop frequency. Selecting 100 Hz on a 50 Hz
loop cannot produce 100 fresh results per second. Period quantization also
matters at 20 Hz on a 50 Hz loop.

## What the values mean

The existing AHRS calculation uses estimated ground velocity minus estimated
wind, rotated to body forward/right/down coordinates. Its legacy formulas and
degree-valued MAVLink getters remain intact. The new DDS message uses **radians**
and body airflow in **m/s**. It includes boot acquisition time, a corresponding
`/ap/time` clock stamp, sequence ID, configured rate, source and validity bits.

Validity bits are velocity available (1), finite airflow (2), speed at least
2 m/s (4), forward airflow (8), and AHRS healthy (16). Valid angles require all
five (31). Invalid angles are NaN. These checks do not establish that wind is
accurate or that every underlying sensor has a fresh independent measurement.

`AIRD` DataFlash records capture the same acquisition snapshot; `AIRQ` captures
simulator airflow separately. The legacy `AOA` record retains its old logging
cadence. The new truth topic is simulation-only and includes wind. Neither
DDS estimates nor commanded servo PWM are physical angle measurements.

Conventional sideslip is `asin(v / V)`. The SIM_Plane aerodynamic model uses
`atan2(v, u)`; coefficient analysis reconstructs the model convention from
recorded airflow rather than silently using the conventional angle.

## Optional live inspection

In a sourced container pane (using the vehicle's ROS domain):

```bash
ros2 interface show ardupilot_msgs/msg/AirData
ros2 topic hz /ap/air_data
ros2 topic echo /ap/air_data --once --qos-reliability best_effort
```

`ros2 topic hz` measures reception; the saved report separately measures acquisition
from firmware timestamps and checks DataFlash agreement. The truth topic is not
expected to contain data on normal non-simulation firmware.

## Plots and offline analysis

The normal plotting helper adds `plots/air_data/` with CSVs, PNG/PDF figures,
`results.json` and a Markdown report. Figures include estimated/truth angles,
errors, applied excitation, body rates and sample intervals. Green regions mark
firmware ACTIVE output; other portions include settling and recovery.
Older bags without the topic remain readable and report air data unavailable.

To analyze a saved bag independently inside the sourced container:

```bash
python3 -m ros2_sid.air_data_analysis "$BAG" --output "$RUN/plots"
```

Timing statistics distinguish unique sequence gaps, duplicates and clock resets.
Transport age is shown only when clock domains appear aligned; it includes clock
offset and is not a calibrated one-way latency measurement.

For the existing simulation coefficient benchmark, `--air-data-source estimate`
selects recorded estimated airflow; `--air-data-source truth` preserves truth
operation. Keep `--state-source` unchanged between comparisons and pass
`--common-air-data-mask` to both fits for identical accepted sample windows.
Crosswind truth fits require `AIRQ`, while legacy zero-wind bags remain supported.

The EKF synthetic zero-sideslip assumption and wind-estimate lag can bias these
angles. Faster computation does not remove those errors. Independent airflow
sensors may still be needed for defensible aerodynamic coefficients.

[Implementation verification and retained plots](air_data_verification.md).
