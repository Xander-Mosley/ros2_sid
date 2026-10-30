# Configurable air-data recording: implementation and verification

The implementation reuses the existing AHRS aerodynamic-angle calculation.
`AHRS_AOA_RATE` selects 20/50/100 Hz; firmware defaults to 20 Hz and identification
parameter examples request 50 Hz. It does not change the flight-loop rate or EKF
fusion. New snapshots carry acquisition time, sequence, configured rate and
validity; invalid estimates are NaN rather than valid zeros. Legacy degree-valued
getters and the legacy `AOA` log retain their conventions.

`/ap/air_data` publishes estimated angles/FRD airflow; `/ap/air_data/truth` is
simulation-only. `AIRD`/`AIRQ` DataFlash records independently retain acquisition
samples. All-topic rosbag recording and the existing plotting helper capture and
analyze these without new mission arguments.

## Evidence

The retained [experiment report](../../../sid_results/air_data_20261023/REPORT.md)
contains measured acquisition/reception rates, angle errors, scheduler records
and held-out identification comparisons. The [reproduction commands](../../../sid_results/air_data_20261023/REPRODUCE.md)
describe the isolated domain-42, speed-1 experiment matrix. Each run retains its
bag, DataFlash log, parameter file, commands, plots and execution results.

- DDS-enabled SITL build: passed.
- DDS-disabled SITL build: passed; DDS-enabled build restored afterwards.
- Cube Orange Plus simulation-on-hardware compile: passed; **not flashed**.
- Native C++ rate/formula tests: passed.
- ROS-package regression suite: 71 passed. Launcher regression suite: 8 passed.
- Installed-package CTest targets (air data, straight-flight analysis, OLS, doublets): 4 passed.
- Six speed-1 rate/wind flights: all final validations passed (54 completed maneuvers).
- Simulated-HITL entrypoint: three doublets passed using simulated RC.
- RC-low interruption: passed; applied excitation stopped, recovery targeted the
  saved start, and the aircraft remained within 130 m of that point in LOITER
  for the required 15-second observation.
- All eight bags' air-data samples matched their DataFlash acquisition records;
  no invalid snapshot contained finite angles. Older bags without air data remain supported.

The initial 20 Hz bag analysis exposed a pre-existing segmentation defect: a lost
initial `WAITING` event merged two maneuvers despite an intervening recorded
`DONE`. The analyzer now treats `DONE` as a segment boundary. A regression test
checks both successful segmentation and continued rejection of missing entry
readiness. The initial failing report is retained beside the reanalysis output.
This change affects offline validation only.

## Measured results and recommendation

| Requested rate | Achieved acquisition, zero / 3 m/s crosswind | DDS reception, zero / crosswind |
|---|---:|---:|
| 20 Hz | 16.67 / 16.67 Hz | 16.67 / 16.67 Hz |
| 50 Hz | 49.52 / 49.52 Hz | 49.31 / 49.52 Hz |
| 100 Hz | 50.00 / 50.00 Hz | 50.00 / 50.00 Hz |

**Use 50 Hz as the candidate recording setting; retain 20 Hz as the firmware
startup default.** Requesting 100 Hz did not provide additional independent
samples beyond the 50 Hz loop. No long loops were reported in the retained PM
records, but these simulator metrics do not measure physical Cube CPU load.
The 50 Hz zero-wind bag lost 64 sequence samples; the report lists reception
losses separately from acquisition cadence. No repeated cached samples were
reported as new sequence IDs.

During active excitation at 50 Hz, alpha RMSE was 0.155° / 0.154° and beta RMSE
was 0.865° / 0.787° for zero wind / crosswind. Faster recording did not remove
estimated-airflow bias. For example, zero-wind held-out yaw R² fell from 0.649
with truth airflow to 0.134 with estimated airflow; pitch remained 0.986 versus
0.983. These results do **not** validate the estimated airflow for recovering all
aerodynamic coefficients. The report includes all coefficients, prediction
figures and common-mask comparisons, including negative prediction scores.

Apparent transport ages include a host/firmware clock offset and are **not**
calibrated one-way latency. Source intervals, sequence gaps, clock-discontinuity
counts and angle correlation diagnostics are retained separately.

![50 Hz estimated versus truth air data](../../../sid_results/air_data_20261023/rate50_wind0/plots/air_data/air_data.png)

![RC-stop return to the saved start](../../../sid_results/air_data_20261023/rate50_wind0_hitl_stop/return_to_start.png)

## Interpretation and limitations

SITL runs at a 50 Hz flight loop. The 20 Hz interval can quantize to about 16.67 Hz;
requesting 100 Hz cannot create a faster AHRS loop. Compare measured source rate
with DDS reception separately; missing sequences are recorded explicitly.

The OLS comparison uses identical output sources, a common validity mask and the
same maneuver split. It changes the airflow source (including speed), not just
an angle field. It is a conditional simulation benchmark with known geometry,
mass, CG, thrust and servo assumptions, not a validated real-aircraft estimator.
There is one flight per rate/wind cell, not a statistically replicated campaign;
small between-flight score differences are not proof of a rate effect.
Wind-estimate lag and synthetic sideslip constraints can bias coefficients even
when the telemetry rate is adequate.

The ordinary maneuver recovery check confirms a saved-start LOITER target,
not arrival. The dedicated RC-stop case additionally records a return track.
SITL scheduler metrics do not establish Cube CPU capacity. Physical USB throughput,
receiver operation, sensor bandwidth and real-flight identification remain unverified.

See [manual recording and parameter setup](air_data_recording.md).
