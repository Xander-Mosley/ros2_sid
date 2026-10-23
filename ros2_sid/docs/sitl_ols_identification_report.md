# SITL aircraft identification with ordinary least squares

Experiment date: 22 October 2026.

## Outcome

Two fresh conventional-plane SITL flights were flown using the existing
`ros2_sid` DDS excitation workflow. An offline OLS estimator recovered several
main lift, roll and pitch derivatives within approximately 1–6% of the values
in `ardupilot_sid/libraries/SITL/SIM_Plane.h`. This is a partial identification
success: yaw and several nominally zero derivatives remain inaccurate.

The primary result uses simulator truth to reconstruct the aircraft state.
The IMU-output comparison is more biased. Neither variant is a sensor-only
estimator, and the validation is not a free-running trajectory simulation.

## Held-out prediction plot

![Recorded and OLS-predicted aerodynamic coefficients on the three held-out multisine maneuvers](figures/sitl_ols/held_out_predictions.png)

Blue points are coefficients reconstructed from the recorded flight; orange
points are OLS predictions using the recorded regressors. The three separated
time windows are the roll, pitch and yaw multisine maneuvers. All three were
excluded from training. Each panel predicts a dimensionless force or moment
coefficient, rather than attitude, position or future aircraft motion.

The figure is copied from the original-frequency flight's
`ols_truth_transients/validation.png`. The plot, [PDF](figures/sitl_ols/held_out_predictions.pdf),
[complete coefficients](figures/sitl_ols/coefficients.csv), and
[numerical results](figures/sitl_ols/results.json) are stored inside this package
so the report does not depend on a local `sid_results` directory.

## Experiments

Both flights used the conventional `plane` SITL model, zero wind, and separate
roll, pitch and yaw excitations at 50 Hz. Normal GUIDED navigation remained
active during excitation. These runs did not enable the runner's optional
straight-flight coordination and did not involve physical hardware.

| Setting | Original-frequency flight | Lower-frequency flight |
|---|---|---|
| Run directory | `ols_identification_20261022` | `ols_low_frequency_20261022` |
| Chirp, each axis | 10 s, 0.2–3 Hz | 30 s, 0.1–0.8 Hz |
| Multisine, each axis | 12 s, 0.5/1/2 Hz | 30 s, 0.1/0.3/0.6 Hz |
| Maximum requested excitation | 0.1 normalized surface | 0.1 normalized surface |
| Training samples, including transients | 7,118 | 10,217 |
| Held-out samples | 1,708 | 4,404 |

The runner applied the selected axis's waveform individually. These sample
counts are points on a 50 Hz analysis grid; they are temporally correlated,
not independent experiment repetitions.

Each flight completed six maneuvers, the DDS safety probe, the mode-change
check, and the recorded ROS-bag versus DataFlash output validation. Raw logs,
command CSVs, launch arguments, parameter defaults and execution summaries
remain under the corresponding outer-workspace `sid_results` directories.

## Estimation method

The implementation is [sitl_ols.py](../ros2_sid/sitl_ols.py). It calls the existing
`ordinary_least_squares` function in `realtime_ols_utils.py`, with column scaling
to reduce numerical conditioning problems. It checks regressor rank and reports
scaled condition numbers. Aerodynamic reference values are read **after**
fitting, solely to calculate comparison errors.

The analysis reads DataFlash `SIM`, `SIM2`, `IMU`, `RCOU`, `SXO` and `SXC`
records. Command sequence IDs are matched to firmware receipt times to identify
whole maneuvers. Quaternion interpolation and velocity interpolation produce
body-frame flow velocity and angle of attack/sideslip on the common time grid.

Both output modes use truth attitude and velocity for alpha and beta:

- `--state-source truth`: derive body rates from the rotation spline and
  accelerations from simulator velocity, with gravity removed in the appropriate
  frame.
- `--state-source imu`: use primary IMU body rates and accelerations for the
  regression outputs and rate regressors.

Surface inputs are reconstructed from applied PWM, not just the requested
perturbations. The estimator prefers the 50 Hz `SXO` surface records and uses
`RCOU` for throttle. It models the default one-pole servo response, including
the recorded `SIM_SERVO_SPEED` value of approximately 0.14 s.

Known calibration is held fixed: mass 2 kg, wing area 0.45 m², span 1.88 m,
chord 0.24 m, CG offset `[-0.15, 0, -0.05]` m, and the simulator's throttle-to-
thrust scale. The fit is therefore conditional on these quantities.

### Simulator conventions that matter

The estimator follows this fork's code rather than substituting conventional
aircraft equations:

1. Axes are forward/right/down; angles and body rates use radians.
2. Surface input is `(PWM - 1500) / 500`, then servo filtering. The header's
   `delta*_max` constants are not applied in the simulated force equations.
3. `beta = atan2(v, u)`, not `asin(v / V)`.
4. `SIM_Aircraft::update_eas_airspeed` stores equivalent airspeed in `airspeed`.
   `SIM_Plane` then uses that speed together with local air density. The estimator
   reproduces this behavior, including the EAS correction.
5. The simulator directly integrates calculated moments as angular acceleration;
   introducing a physical inertia tensor would change the identification target.
6. Thrust is removed before reconstructing aerodynamic force. The torque due to
   `CGOffset × aerodynamic_force` is subtracted before fitting moment coefficients.

With `p_hat = b*p/(2*V)`, `q_hat = c*q/(2*V)`, and `r_hat = b*r/(2*V)`, the
regression columns are:

| Model | Columns, in coefficient order |
|---|---|
| Roll, yaw, side force | `1, beta, p_hat, r_hat, aileron, rudder` |
| Pitch, lift | `1, alpha, q_hat, elevator` |
| Drag | `1, alpha, alpha², q_hat, abs(elevator)` |

Drag is fitted as a polynomial. Its first three reference values are derived by
expanding the source equation
`CD = c_drag_p + (c_lift_0 + c_lift_a*alpha)^2 / (pi*oswald*AR)`.
They are not three independent constants in the source header.

### Filtering and training split

The analysis uses short polynomial filters and numerical differentiation. The
same final nine-point, third-order smoothing is applied to every regressor and
its output. Samples require EAS above 12 m/s, altitude more than 20 m above the
initial altitude, `abs(alpha) < 0.20 rad`, and `abs(beta) < 0.25 rad`.

The initial fit used chirps only. The final `--include-transients` fit also uses
airborne climb and settling samples, broadening the alpha/speed range. Entire
multisines remain excluded from training, with a filter guard around them.
Validation excludes 0.3 s at each maneuver edge. The saved NPZ file records the
actual training and validation masks.

## Results and interpretation

The primary fit is the original-frequency flight with truth outputs and
airborne transients included. Representative nonzero derivatives are:

| Coefficient | Source | OLS | Absolute relative error |
|---|---:|---:|---:|
| `c_lift_0` | 0.56 | 0.56631 | 1.13% |
| `c_lift_a` | 6.9 | 7.01741 | 1.70% |
| `c_l_p` | -1 | -0.98223 | 1.78% |
| `c_l_deltaa` | 0.25 | 0.24443 | 2.23% |
| `c_m_0` | 0.045 | 0.04666 | 3.68% |
| `c_m_a` | -0.7 | -0.66157 | 5.49% |
| `c_m_q` | -20 | -19.31486 | 3.43% |
| `c_m_deltae` | 1 | 0.98806 | 1.19% |
| `c_y_b` | -0.98 | -0.98496 | 0.51% |
| `c_y_deltar` | -0.2 | -0.20143 | 0.71% |

The complete CSV includes unsuccessful estimates as well. In this fit,
`c_l_r` has about 12% error, nonzero yaw derivatives have approximately 13–60%
error, and `c_lift_q` estimates 1.97 despite a source value of zero. Relative
percentage error is undefined for zero reference coefficients; their reports
use absolute error.

| Held-out R² | Original / truth | Original / IMU | Slow / truth | Slow / IMU |
|---|---:|---:|---:|---:|
| Roll | 0.9706 | 0.9646 | 0.8555 | 0.8539 |
| Pitch | 0.9603 | 0.9595 | 0.9350 | 0.9365 |
| Yaw | 0.6584 | 0.6630 | 0.0590 | 0.0565 |
| Side force | 0.9959 | 0.9907 | 0.9943 | 0.9916 |
| Lift | 0.99985 | 0.99875 | 0.99980 | 0.99899 |
| Drag | 0.6569 | 0.6282 | 0.5048 | 0.5293 |

All four columns above include transients. The slower experiment did not
improve every model, particularly yaw.

Pitch illustrates why prediction R² alone is insufficient. The original
chirp-only truth fit had pitch R² about 0.966, but estimated `c_m_q = -0.956`
instead of -20. Adding transients reduced the pitch regressor condition number
from about 231 to 76.5 and recovered `c_m_q = -19.315`, despite a slightly lower
validation R² of 0.960. The wider training coverage was more useful for
coefficient identification than the original response fit score suggested.

The IMU-output fit is still materially biased: with transients, the original
flight estimates `c_m_a = -0.0917` versus -0.7 and `c_m_q = -16.781` versus -20.
Timing, differentiation noise and correlated regressors remain limitations.
No confidence intervals are claimed because residuals are temporally correlated.
The multisines were excluded from coefficient fitting, but were inspected during
development; these are development validation results, not an untouched final
test campaign.

## Changes made and workflow compatibility

| File/change | Purpose |
|---|---|
| `ros2_sid/sitl_ols.py` | New offline coefficient estimator and plot/report exporter |
| `tests/test_sitl_ols.py` | Numerical, servo, timing-selection and EAS normalization tests |
| `config/surface_excitation/ols/` | Optional lower-frequency waveform definitions |
| `CMakeLists.txt` | Register the estimator tests with the ROS package |
| `realtime_ols_utils.py` | Defer Fourier configuration loading until a transform needs it |
| Firmware repo's `surface_sitl.py` | Add optional `--waveform-config-dir` and serialize its path in run metadata |
| Package README and this report | Reproduction instructions, evidence and limitations |

The OLS task did not edit `run_mission_SITL.py`, `run_mission_HITL.py`, excitation
control logic, RC gating, straight-flight logic, or the SITL/HITL manual guides.
The existing runner defaults remain unchanged when the new waveform option is
omitted. Existing modifications to the runner and firmware that predated this
task were preserved. No simulator aerodynamic reference coefficients were changed.

## Reproduction

Prerequisites are the existing DDS-enabled Plane binary, host `pymavlink`, and
the isolated ROS/DDS containers described by the existing SITL guide. From the
outer workspace, with those services healthy, use a fresh run directory:

```bash
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 --individual-axes --rate 50 \
  --output sid_results/ols_new_run

docker exec -e PYTHONPATH=/work/ros2_sid/ros2_sid \
  surface-excitation-sitl-ros-1 python3 -m ros2_sid.sitl_ols \
  --run /work/sid_results/ols_new_run \
  --reference /work/ardupilot_sid/libraries/SITL/SIM_Plane.h \
  --state-source truth --include-transients \
  --output /work/sid_results/ols_new_run/ols_truth_transients
```

For the slower flight add
`--waveform-config-dir ros2_sid/ros2_sid/config/surface_excitation/ols` to the
runner command. For analysis comparisons, select `--state-source imu`, or omit
`--include-transients` to train on chirps only. Use separate output directories
to retain each comparison.

The estimator requires a complete, validated conventional-plane run. It writes
`results.json`, `coefficients.csv`, `regression_data.npz`, `validation.png` and
`validation.pdf`. JSON records the reference-header and DataFlash SHA-256 hashes,
calibration assumptions, per-maneuver validation and condition numbers.

To regenerate the figure embedded in this report from the retained original
flight, substitute `ols_identification_20261022` for `ols_new_run` above, then:

```bash
cp sid_results/ols_identification_20261022/ols_truth_transients/validation.png \
  ros2_sid/ros2_sid/docs/figures/sitl_ols/held_out_predictions.png
cp sid_results/ols_identification_20261022/ols_truth_transients/validation.pdf \
  ros2_sid/ros2_sid/docs/figures/sitl_ols/held_out_predictions.pdf
```

## Verification completed

- Both SITL flights passed their six maneuver checks, safety probe, mode-change
  check and ROS-bag/DataFlash validation.
- Six estimator tests passed, including known-coefficient recovery, rank-deficient
  rejection, servo scaling/lag, firmware-time maneuver selection, explicit Fourier
  initialization without a configuration file, and the simulator EAS convention.
- The estimator tests plus waveform tests passed together: 16 tests.
- The ROS package built successfully; its installed estimator imported, and the
  registered `test_sitl_ols` CTest target passed.
- Python syntax and Git whitespace checks passed for the implementation changes.

No physical HITL identification campaign or free-running model validation was
performed. The next useful work would be stronger independent yaw excitation,
better sensor/actuator time alignment, and validation without simulator-truth
alpha/beta before treating this as a measured-aircraft model.

## Follow-up: estimated air data

A separate [configurable air-data experiment](air_data_verification.md) compares
20/50/100 Hz acquisition settings and estimated versus truth airflow, including
crosswind. The estimator now accepts `--air-data-source estimate` and
`--common-air-data-mask` for controlled comparisons. The historical results in
this report remain truth-based; they are not reclassified as sensor-only results.
