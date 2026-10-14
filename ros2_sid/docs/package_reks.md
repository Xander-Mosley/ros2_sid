# Package Requirements

## Firmware

This data requires flight data stream through Data Distribution Service (DDS).
Therefore, a custom DDS module is required for the ArduPilot firmware.
The following is a link to the custom ardupilot_sid repo:
<https://github.com/Xander-Mosley/ardupilot_sid/tree/master>

A arduplane.apj file is required to (re)flash an ArduPilot cube.
The ardupilot_sid repo can be used to generate this file using the following commands.

```Bash
```

Branches

- 'master': follows the most recent ArduPilot project commit.
- 'Plane-4.6.3': configured for ArduPilot's Plane-4.6.3 tag.

## Messages

### ArduPilot

/ardupilot_msgs/msg/...

Pitot.msg :

```text
# Raw pitot tube / airspeed sensor data.
#
# Units:
#   differential_pressure : Pa     (Measured pitot-static pressure difference, ΔP)
#   dynamic_pressure      : Pa     (q = 0.5 * ρ * V²)
#   calibrated_airspeed   : m/s    (CAS = sqrt(2q / ρ₀))
#   true_airspeed         : m/s    (TAS = CAS * sqrt(ρ₀ / ρ))
#   temperature           : °C     (Measured sensor or ambient air temperature)
#   air_density           : kg/m³  (Local atmospheric air density)
#
# Notes:
# - Values originate from AP_Airspeed and AP_Baro before EKF fusion.
# - ρ₀ = 1.225 kg/m³ (ISA sea-level air density)
# - ρ = local atmospheric air density
# - q = dynamic pressure

std_msgs/Header header

float32 differential_pressure
float32 dynamic_pressure

float32 calibrated_airspeed
float32 true_airspeed

float32 temperature
float32 air_density
```

Propulsion.msg :

```text
# Propulsion system telemetry.
#
# Units:
#   rpm         : revolutions/minute
#   voltage     : V
#   current     : A
#   temperature : °C
#
# Notes:
# - Values originate from ESC or propulsion telemetry, AP_ESC_Telem (when available).
# - Electrical power may be computed as:
#     power = voltage × current (W)

std_msgs/Header header

float32 rpm

float32 voltage
float32 current

float32 temperature
```

RcIn.msg :

```text
# PWM values received by the flight controller from the RC receiver.
#
# Units:
#   values : µs (PWM pulse width)
#
# Notes:
# - RC input backend (AP_RCProtocol / RC_Channel).
# - Bit i of valid_mask corresponds to values[i].
# - A set bit indicates the corresponding PWM value is valid.
# - Clear bits indicate the corresponding array element should be ignored.
# - Example:
#     valid_mask = 0x000F
#     -> values[0]..values[3] are valid
#     -> values[4]..values[15] should be ignored

std_msgs/Header header

uint16 valid_mask
uint16[16] values
```

RcOut.msg :

```text
# PWM values sent by the flight controller to the aircraft servos.
#
# Units:
#   values : µs (PWM pulse width)
#
# Notes:
# - SRV_Channel output PWM values.
# - Bit i of valid_mask corresponds to values[i].
# - A set bit indicates the corresponding PWM value is valid.
# - Clear bits indicate the corresponding array element should be ignored.
# - Example:
#     valid_mask = 0x000F
#     -> values[0]..values[3] are valid
#     -> values[4]..values[15] should be ignored

std_msgs/Header header

uint16 valid_mask
uint16[16] values
```

### Drone Interfaces

/drone_interfaces/drone_interfaces/msg/...

SysIdDataStream.msg :

```text
# Data structure for signal streams during system identification.
# Used ONLY within the ros2_sid package.

std_msgs/Header header          # contains builtin_interfaces/Time stamp + string frame_id
float64 dt                      # time step

float64 trend                   # low-pass portion of the signal before detrending
float64 value                   # time domain data contained in this signal stream

float64[] frequencies_hz        # frequencies each bin represent
float64[] spectrum_real         # real part of the Fourier coefficient
float64[] spectrum_imag         # imaginary part of the Fourier coefficient
```

SysIdLeastSquares.msg :

```text
# Data structure for publishing least squares results from the ros2_sid package.

std_msgs/Header header          # contains builtin_interfaces/Time stamp + string frame_id

float64 measured_output         # time domain measured output for the current least squares calculation 
float64[] regressor             # time domain regressors for the current least squares calculation
float64[] parameter             # estimated parameters from the current least squares calculation

```
