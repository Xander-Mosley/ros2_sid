# Run the DDS control-surface SITL tests yourself

On a new machine, first complete [Docker image setup](../README_surface_excitation.md#first-time-docker-image-setup).
It builds the shared image used by both ROS and the DDS agent from this
repository’s Dockerfile. No separate DDS-agent image is needed.

## Recommended: one launcher, no RUN/BAG variables

From **any checkout location**, open a host terminal in the repository root:

```bash
./tools/sid/run_surface_test.sh sitl
```

This starts/reuses the Docker services, rebuilds `ros2_sid`, creates a unique
`sid_results/sitl_YYYYMMDD_HHMMSS/` directory, and opens four tmux panes:

1. Connections: SITL and interactive MAVProxy.
2. MAVROS: telemetry and straight-flight navigation services.
3. Recorder: **all ROS topics**, including topics discovered later.
4. Mission: SITL console maneuver selection, after DDS/MAVROS/recorder readiness.

Use the connection pane’s MAVProxy/GCS to take off and enter GUIDED as described below. The launcher never arms or takes off automatically.

The default DDS library now includes doublets:

```text
list
select 6    # roll doublet
select 7    # pitch doublet
select 8    # yaw doublet
```

Enter just one `select` command (without the comment). Then enter `start`.
The existing chirps/multisines keep menu numbers 0–5. Each doublet uses the legacy
`multi_step` generator: positive then negative pulses, one-second initial delay,
1 Hz natural frequency, 50 Hz samples, and 15 seconds total. The nominal pulse
width is 0.25 s; the legacy sample rounding and zero boundary samples are retained.
Amplitude is **0.10 normalized surface fraction**, not radians. Existing firmware
limits, RC stop, timeout protection and return-to-start recovery still apply.


### Amplitude modifier and firmware limits

Both `run_mission_SITL.py --interface dds` and `run_mission_HITL.py --interface dds`
read `amplitude_modifier` from `ros2_sid/setup/mission_library/current_mission.json`
inside the package. It defaults to **1.0** and multiplies every roll, pitch, and yaw
sample before publishing and command CSV logging. Restart the mission script after
editing the JSON. `--amplitude-modifier` overrides the JSON value; explicit
`--experiment` or `--mission-plan` sources default to 1.0 unless this override is supplied.

For example, a waveform amplitude of 0.10 with modifier 5.0 sends a peak of 0.50.
The modifier changes amplitude only, not frequency or duration. A raw axis command
of 5.0 is invalid: firmware rejects commands outside [-1, 1], rather than clipping
them into that range.

Firmware defaults and the conventional/elevon/vtail SITL parameter presets now set
`SX_R_MAX`, `SX_P_MAX`, `SX_Y_MAX`, and `SX_S_MAX` to **1.0**. The first three clip
individual axis commands; `SX_S_MAX` bounds each mixed physical surface. Lower
configured limits still apply: with `SX_R_MAX=0.1`, a requested roll of 0.5 is clipped
to 0.1, which can hide the effect of increasing the modifier. Servo calibration,
remaining output travel, mixing, and any configured slew limit also affect motion.
The waveform definitions retain their existing amplitudes (typically 0.10).

These defaults require rebuilding the firmware. Existing saved aircraft/SITL
parameters can retain the old limits after a rebuild; inspect the active values
and explicitly set the following if full normalized excitation range is intended:

```text
param set SX_R_MAX 1.0
param set SX_P_MAX 1.0
param set SX_Y_MAX 1.0
param set SX_S_MAX 1.0
```

A limit of 1.0 permits excitation up to the full calibrated deflection; it does not
command that deflection by itself. Compare the scaled command CSV with firmware
applied-axis/clipping telemetry when diagnosing a smaller-than-requested response.

For an optional session label:

```bash
./tools/sid/run_surface_test.sh sitl --name doublet_roll
```

The label names the session; it does not select a maneuver. No name or log argument
is required. `session.json` records the bag, command CSV and plot paths. A fresh
run gets a fresh directory; collisions receive an index. The selected legacy-style
`current_mission.json` library and straight-flight settings are copied into the run
so the recording retains its configuration. The selected library must declare
`normalized_surface` units.

### Finish and plot

Enter `quit` in the mission pane. After successful recovery handoff, recording
continues for **90 seconds** to capture the return, then closes automatically.
Use `--return-record-seconds N` to change that tail. It is a recording duration,
not proof of arrival. Connections remain running so the GCS stays available.

```bash
./tools/sid/plot_surface_test.sh --latest sitl
# Or replot a particular completed session:
./tools/sid/plot_surface_test.sh sid_results/sitl_YOUR_SESSION
```

The plotting helper reads actual paths from the manifest, runs bag info and surface/
straight-flight analysis, and writes `plots/`. It never replays commands.
It can start the UDP-only ROS container for offline plotting if the original one
is stopped. Do not move the session outside the checkout's `sid_results` directory.

The MAVROS, recorder and mission panes open **inside the ROS container**, with ROS
and the workspace already sourced and `RUN` set. HITL's router pane also runs in
the container; SITL's simulator pane remains on the host. Ctrl-C returns to the
container prompt. Use `run_mavros`, `run_recorder`, `run_mission` or
`run_connections` in its corresponding pane to recall the supervised command
(or press Up, Enter). You can also run `ros2` commands directly there. The short
commands retain readiness checks and recording/recovery tracking internally.
For a new recorded experiment, start a fresh launcher session so logs and shutdown
markers are not reused. Use the mission menu’s `quit` for orderly recovery.

Detach with **Ctrl-B, D**; reattach with `tmux attach -t surface-test`.
Detaching does not stop the test. When ready to close connections, Ctrl-C their
panes, then remove the finished session with `tmux kill-session -t surface-test`.
The launcher refuses a second session or existing MAVROS/router/recorder processes.

If recovery or startup fails, recording/connections are retained for operator
inspection. After taking over and quitting the mission, Ctrl-C the recorder and
wait for its finalization; the plotting helper accepts that finalized bag and
retains any failed test evidence. Never kill the container as a normal test stop.

### Prerequisites and connection overrides

Host: Docker Compose, tmux, and host MAVProxy plus the DDS-enabled ArduPlane binary.
The configured ROS image must contain MAVROS and the existing ROS dependencies.
SITL uses the existing host DDS launcher and UDP 14552 for MAVROS.

[Verification results and doublet plots](doublet_launcher_verification.md).

## Individual commands and troubleshooting

The detailed commands below remain available when diagnosing individual services.


### Return to the test-start location

With the default straight-flight coordinator, each maneuver saves the aircraft's
latitude, longitude, and altitude above home **before straight-flight setup begins**.
After completion, `stop`, `quit`, Ctrl-C, or an HITL RC stop, it:

1. Disables excitation and waits for observed zero applied perturbation.
2. Commands LOITER centered on that saved location, at the saved altitude.
3. Confirms LOITER mode and a fresh navigation-target echo matching the saved coordinates.

The aircraft flies back and circles the saved point using its configured loiter
radius. `DONE` confirms the return target was accepted; it does **not** mean the
aircraft has already arrived. Keep recording to capture the return flight.
Operator mode changes, disarming, and failsafes still take precedence. A new
maneuver captures a new start point. `--no-straight-flight` disables this recovery.

**Setup requirement:** rebuild ArduPlane from this updated repository (including
Cube Simulation-on-Hardware firmware for HITL), rebuild `ros2_sid`, and restart
MAVROS with `config/surface_excitation/mavros_straight.yaml`. It now enables
`setpoint_raw` to receive the navigation-target echo. The coordinator requests that
MAVLink message at 5 Hz automatically. It also requires a fresh global GPS fix
from `/mavros/global_position/global` before setup can start.

Older firmware ignores the location in this LOITER command and may circle the
end point; the coordinator reports recovery unconfirmed if its target does not
match. No hardware is flashed automatically by this change. The eight-second
`recovery_timeout_s` bounds command/target confirmation, not the flight back.
Saved coordinates and target echoes are recorded in `/sid/straight_flight` and
the command log's `.straight.jsonl` file. See the
[return-to-start verification report](return_to_start_verification.md) for tests
and recorded tracks.


> **DDS default:** `run_mission_SITL.py --interface dds` now loads the packaged
> `straight_flight_sitl.json` automatically. MAVROS is required: defaults target
> 75 m above home, 23 m/s, and the current ground course, then wait for settling.
> `--straight-flight PATH` overrides these settings; `--no-straight-flight`
> explicitly disables navigation coordination and automatic LOITER recovery.
> The shared HITL DDS entrypoint has the same default and retains its RC gate;
> the packaged settings remain SITL examples, not validated physical-flight settings.
> The legacy trajectory interface is unchanged. Rebuild/restart the ROS service
> before using this new default; do not restart it during a running experiment.


For automatic mission loading and creating your own chirps/multisines, see [Custom DDS waveforms](custom_surface_waveforms.md).

This guide reproduces the individual roll, pitch, and yaw chirp/multisine tests,
records ROS 2 bags, and generates command-versus-response plots. It also gives an
interactive procedure for starting a maneuver yourself. These commands use a
simulated conventional ArduPlane, with stabilization running in airborne GUIDED.
They do not run `run_mission_HITL.py` or connect to physical hardware.

For the new straight-flight workflow, go to [section 7](#7-run-an-interactive-straight-flight-maneuver)
after completing setup and takeoff. It includes MAVROS, recording, manual exits,
and replotting.

For a publisher-focused walkthrough after flight setup, see [Send waveforms from inside the ROS container](manual_container_waveforms.md).

## 1. Prepare the workspace and tools

Set `SID_WORKSPACE` to your checkout root: the directory containing
`docker-compose.sid.yaml`, `ardupilot_sid/`, and `ros2_sid/`. Replace the example
path below with your actual location:

```bash
export SID_WORKSPACE="/path/to/ros2_trajectory_docker"
cd "$SID_WORKSPACE"
```

You need Docker Compose with working Docker access, a host Python environment with
`pymavlink`, the built DDS generator, and the shared Docker image:

```bash
docker info
docker image inspect ros2_trajectory:compact
python3 -c 'from pymavlink import mavutil; print("pymavlink available")'
test -x Micro-XRCE-DDS-Gen/scripts/microxrceddsgen
```

To build or refresh the shared image, follow [Docker image setup](../README_surface_excitation.md#first-time-docker-image-setup)
before proceeding. Firmware
build dependencies include `empy==3.3.4`, `future`, `lxml`, `pyyaml`, `dronecan`, and
`pexpect`. The ROS image must contain ROS Humble, colcon, rosbag2, NumPy, SciPy,
matplotlib, and pymavlink. The interactive flight option additionally needs host
MAVProxy (`mavproxy.py`).

Use only one instance of this test at a time. The automated test uses TCP ports 5780/5782/5783,
DDS agent UDP port 2021, and ROS domain 0. Stop an older test's agent/SITL instance
before starting this one; do not stop unrelated services.

## 2. Build firmware and start the ROS services

On the host:

```bash
export PATH="$SID_WORKSPACE/Micro-XRCE-DDS-Gen/scripts:$PATH"
cd "$SID_WORKSPACE/ardupilot_sid"
git submodule update --init --recursive
./waf configure --board sitl --enable-dds
./waf plane -j4
cd "$SID_WORKSPACE"
docker compose -f docker-compose.sid.yaml up -d --wait --wait-timeout 900
export SID_ROS_CONTAINER="$(docker compose -f docker-compose.sid.yaml ps -q ros)"
docker compose -f docker-compose.sid.yaml ps
```

The services use host networking and mount the checkout at `/work`. They have no
physical-device mappings. Keep `SID_ROS_CONTAINER` in this host terminal; it avoids
assuming a particular generated container name.

Compose automatically builds `ardupilot_msgs`, `ardupilot_sitl`, and `ros2_sid`
into `/tmp/install`, checks the generated interface imports, and runs the surface and straight-flight
test suites. The ROS service becomes healthy only after these succeed.
`--wait` waits for that readiness; plain `up -d` starts setup in the background.
The firmware build above is still required; Compose installs that existing binary.

Watch setup or diagnose a failed startup with:

```bash
docker compose -f docker-compose.sid.yaml logs -f ros
```

The expected final message is `SID ROS workspace ready: interfaces built and surface
tests passed.` Ctrl-C exits the log viewer without stopping the service. Build/test
failure stops the ROS service and leaves it unready; inspect its logs and correct
the failure before proceeding.

Setup runs on every container start, using incremental build files when available.
After changing message definitions or ROS code, rebuild and rerun startup checks:

```bash
docker compose -f docker-compose.sid.yaml restart ros
docker compose -f docker-compose.sid.yaml up -d --wait --wait-timeout 900
```

An already running healthy service is not rebuilt by `up -d` alone. Recreating the
container rebuilds its `/tmp/install` automatically. Interactive ROS shells still
need to source ROS and `/tmp/install/setup.bash`, as shown below.

## 3. Reproduce the six validated experiments

Run this on the host from the workspace root. Always choose a **new output directory**;
the acceptance runner expects a fresh run directory. The standalone recorder
automatically indexes existing bag names, as described below.

```bash
cd "$SID_WORKSPACE"
export SID_RUN="sid_results/manual_axes_$(date +%Y%m%d_%H%M%S)"
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container "$SID_ROS_CONTAINER" --airframe conventional \
  --individual-axes --rate 50 --output "$SID_RUN"
```

This is a user-invoked automated flight test: the runner launches a fresh simulator,
merges the Plane defaults and excitation parameters, performs normal TAKEOFF,
waits for altitude, enters GUIDED, and executes:

| Axis | Chirp | Multisine |
|---|---|---|
| Roll | 10 s, 0.2–3 Hz, amplitude 0.10 | 12 s, 0.5/1/2 Hz |
| Pitch | Same chirp, pitch only | Same components, pitch only |
| Yaw | Same chirp, yaw only | Same components, yaw only |

Multisine component amplitudes are 0.03/0.03/0.04 and phases are 0/1/2 radians.
The conservative combined amplitude bound is 0.10. Values are normalized surface
perturbations, not attitude angles. Each experiment commands only its selected axis.
The runner uses the roll waveform from the examples and moves it to the selected
axis; the ordinary `multisine.json` example itself also contains pitch/yaw offsets.

Allow several minutes. This firmware fork has two existing 60-second DDS startup
delays. The runner maintains simulated RC/GCS input while the missions run and
records a bag automatically. Do not start a second command publisher or recorder
using the same output path.

At completion, all entries in `execution.json` should be `true`, including the six
maneuvers, `safety_probe`, `mode_change`, and `bag_validation`:

```bash
cat "$SID_RUN/execution.json"
cat "$SID_RUN/plots/report.json"
```

Validation checks actual applied outputs, maneuver coverage, calibration, and
matching DataFlash records. An exit failure, missing result, or failed validation
is not a passing flight, even if DDS messages were received. Retain its artifacts.

## 4. Generate and view the six detailed plots

In the same host terminal:

```bash
docker exec -e SID_RUN="$SID_RUN" "$SID_ROS_CONTAINER" bash -c '
  set -e
  source /opt/ros/humble/setup.bash
  source /tmp/install/setup.bash
  python3 -m ros2_sid.surface_axis_plots \
    --run "/work/$SID_RUN" --output "/work/$SID_RUN/axis_figures"
  ros2 bag info "/work/$SID_RUN/rosbag"
'
```

Open `$SID_RUN/axis_figures` in your file browser. It contains a PNG and PDF for
each of `chirp_roll`, `multisine_roll`, `chirp_pitch`, `multisine_pitch`, `chirp_yaw`,
and `multisine_yaw`, plus `summary.json`. Each figure shows:

1. Requested and applied logical excitation.
2. Actual servo perturbations: final PWM minus normal baseline PWM.
3. Body angular rates p, q, r.
4. The selected axis's attitude change.

Normal GUIDED motion and cross-axis aircraft response remain visible. Servo output
is a command, not measured surface position. The renderer also verifies that other
axes/surfaces were not excited in each maneuver. Inspect `plots/report.json` for
numeric validation; the prior run achieved at most 1 microsecond reconstruction
error. Your run must pass its own checks.

Retain the entire run directory: `rosbag/`, `logs/*.BIN`, command CSVs, experiment
JSONs, `defaults.parm`, `launch.json`, `execution.json`, and analysis logs. You can
rerun the plot command offline with the matching ROS interfaces installed. No bag
playback or running flight is required. Do not replay a command bag into a vehicle.

The existing report with comparison figures is
[VERIFICATION.md](../../../ardupilot_sid/docs/control_surface_excitation/VERIFICATION.md).

## 5. Optional: start maneuvers interactively

Use this instead of step 3 when you want to control flight setup and type `start`
yourself. Complete steps 1–2 first. Do not run the automated runner concurrently.

### Host terminal A: launch using the host script

The launcher uses `ardupilot_sid`, the existing DDS-enabled Plane binary, domain 0,
agent UDP port 2021, and speedup 1. It merges the conventional excitation defaults,
creates all six single-axis waveform JSONs at 50 Hz, and starts MAVProxy itself.
The home location is Richter's Airfield, retained from the original script.

```bash
cd "$SID_WORKSPACE"
bash useful_shell_scripts/run_arduplane_host_dds.sh --headless
```

The launcher creates and prints a unique `sid_results/interactive_...` directory
automatically. The container shells below select it with `use_run.sh`, so you do
not need to copy its timestamp. If you supply `--run-dir` yourself, that directory
must not already exist. `--headless` keeps the MAVProxy terminal without
opening graphical windows; omit it for the map and console if your host has the
required GUI dependencies. This script does not start Docker or rebuild firmware.

Keep this terminal running. **Use its MAVProxy prompt for the commands below; do
not launch another MAVProxy instance.** This interactive launcher uses sim_vehicle's
default MAVLink TCP port 5760, and forwards MAVLink to UDP 14551/14552/14553. The
automated acceptance runner separately uses ports 5780/5782. Do not run both at once.

### In the launcher's MAVProxy prompt: take off and enter GUIDED

At the MAVProxy prompt, initialize simulated RC inputs and take off:

```text
rc 1 1500
rc 2 1500
rc 3 1000
rc 4 1500
param set TKOFF_ALT 100
mode TAKEOFF
arm throttle
```

Wait for successful arming. If prearm checks fail, read and resolve the reported
condition; do not force-arm or disable checks. Then:

```text
rc 3 1800
```

Wait until relative altitude is at least 80 m (`status GLOBAL_POSITION_INT` reports
`relative_alt` in millimeters), then:

```text
mode GUIDED
```

Allow roughly eight seconds to settle. Leave MAVProxy running to maintain the GCS
connection. This interactive procedure follows the runner's flight stages but is
not the automated acceptance harness.

### Container terminals C and D: record and publish

Open two additional host terminals. Set `SID_WORKSPACE` to your checkout root
in each new host shell, then enter the ROS container and source its workspace:

```bash
cd "$SID_WORKSPACE"
docker compose -f docker-compose.sid.yaml exec ros bash
source /opt/ros/humble/setup.bash
source /tmp/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/work/ros2_sid/ros2_sid/config/surface_excitation/fastdds_udp.xml
```

In **both container shells**, select the latest interactive run automatically:

```bash
source /work/tools/sid/use_run.sh --latest interactive_
```

Check the printed path matches your active run. If several runs exist, select the
exact one with `source /work/tools/sid/use_run.sh --path /work/sid_results/RUN_NAME`.
Use `source`, not `bash`: the helper exports `RUN` into your current shell.

In terminal C, check live delivery, then start recording:

```bash
ros2 topic echo /ap/time --once --qos-reliability best_effort
ros2 topic echo /ap/sid/status --once --qos-reliability best_effort
ros2 run ros2_sid record_surface_excitation.sh "$RUN/bag_chirp_roll"
```

In terminal D, you can run the default mission without specifying an experiment:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds --wait-timeout 180
```

With no `--experiment` or `--mission-plan`, the publisher reads the shared
`mission_library/current_mission.json`, follows its `subfolder`, and loads that
folder's `mission_plan.json` and ordered CSV maneuvers. The current default is
`dds_surface`. Type `list`, `select N`, and `start`; it does not automatically
execute every maneuver. The publisher still requires airborne GUIDED authorization.

With no `--command-log`, commands are written to `surface_commands.csv` in the
shell's current working directory. **Starting another publisher invocation at that
location overwrites the file.** This CSV is separate from ROS bag recording; the
publisher does not start a bag recorder.

To keep default mission selection but save the command log alongside this run:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --wait-timeout 180 --command-log "$RUN/surface_commands.csv"
```

Choose a different CSV filename for subsequent publisher invocations if you want
to retain each log. Multiple maneuvers selected within one invocation share its
command CSV. `--wait-timeout 180` allows up to 180 seconds while waiting for startup
readiness/authorization; it does not set the waveform duration or replace the
firmware command-freshness timeout. If omitted, its default is 30 seconds.

Alternatively, to load only the generated roll-chirp experiment, start the publisher
with an explicit source and log path:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --experiment "$RUN/chirp_roll.json" --wait-timeout 180 \
  --command-log "$RUN/chirp_roll-commands.csv"
```

Type `start` at its console. Use `stop` to end early or `quit` to exit. Let the full
maneuver finish for duration validation. After completion, wait at least one second
for the return to baseline, then stop the recorder in terminal C with Ctrl-C.

Repeat with `multisine_roll`, `chirp_pitch`, `multisine_pitch`, `chirp_yaw`, and
`multisine_yaw`, using a new bag path and matching CSV each time. Run only one
publisher at a time. The feature requires valid airborne GUIDED authorization;
status state 2 means ACTIVE, while state 1 means excitation-ready, not vehicle-armed.

### Replot an interactive bag

In a sourced container shell:

```bash
ros2 run ros2_sid plot_surface_excitation.py "$RUN/bag_chirp_roll" \
  --output "$RUN/plots_chirp_roll" \
  --command-log "$RUN/chirp_roll-commands.csv"
```

This produces general tracking, actuator-validation, and response plots plus a
numeric report. The six-panel-set renderer in step 4 expects the automated runner's
combined bag/report layout; use the general plotter for these separate bags.

When finished, stop the simulated firmware cleanly with Ctrl-C in terminal A. Once
its `.BIN` has closed, you can repeat the plot command with
`--compare-dataflash "$RUN/logs/00000001.BIN"` (check the actual log filename first).
Standard DataFlash `RCOU` contains baseline plus excitation; custom records preserve
the baseline and applied delta separately.

## 6. Troubleshooting and cleanup

| Symptom | Check |
|---|---|
| Docker permission denied | Resolve Docker access for your host account before running the harness. |
| Topics listed but no data | Agent port 2021, domain 0, and the UDP-only Fast DDS profile must agree. Allow DDS startup delays. |
| ROS interface/package missing | Build/source `/tmp/install/setup.bash` inside the same ROS container. |
| `ros_env.sh: unexpected end of file` on shell startup | Check `cat -v /usr/local/bin/ros_env.sh` for `^M` Windows line endings. In the container, repair with `sed -i 's/\r$//' /usr/local/bin/ros_env.sh`, then `bash -n /usr/local/bin/ros_env.sh`. Source `/opt/ros/humble/setup.bash` and `/tmp/install/setup.bash` again. The Dockerfile normalizes this generated script for future builds. |
| Plotter cannot find `metadata.yaml` | Set `BAG` to the actual recorded bag directory. Selecting `RUN` does not set `BAG`; check `ros2 bag info "$BAG"` first. |
| Excitation stays inactive | Check `/ap/sid/status`, airborne GUIDED, normal arming, `SX_ENABLE`, failsafes, and calibration. |
| Timeout/jitter abort | Keep speedup 1, use the tested 50 Hz rate, and reduce host load. Preserve the failure logs; do not relax safety deadlines. |
| Repeating a bag name | The recorder adds `_1`, `_2`, etc.; use the printed path for analysis. |
| Plots show other body rates moving | Aircraft coupling and normal stabilization can produce other-axis response even with single-axis excitation. |

After the publisher and recorder have exited and SITL is stopped, stop only this
Compose project's services from the host workspace:

```bash
docker compose -f docker-compose.sid.yaml stop
```

`stop` preserves the ROS container's build workspace for reuse. Keep your run folder
for comparison and offline plotting. Further architecture and parameter details are
in the [firmware guide](../../../ardupilot_sid/docs/control_surface_excitation/README.md).

## Export interactive recordings with the existing data_parser

After stopping the recorder, use a sourced ROS container shell and select the
existing interactive run. Do not create a new directory for analysis:

```bash
source /work/tools/sid/use_run.sh --latest interactive_
python3 -m ros2_sid.post_process.data_parser "$RUN/bag_chirp_roll" \
  --surface --output "$RUN/parsed_surface"
```

Replace `bag_chirp_roll` with your actual bag name when needed. Check the printed
run path before analysis. Choose a new output folder for each export. After
updating package sources, restart the Compose ROS service to rebuild before using
this command (do this before starting a flight or recorder).

The existing parser now exports these topics as numeric CSVs:

| File | Contents |
|---|---|
| `surface_commands_data.csv` | Source stamp, sequence, enable, normalized roll/pitch/yaw requests |
| `surface_status_data.csv` | State, reason, sequence, limits and configuration revision |
| `surface_output_data.csv` | Coherent applied-output snapshots: calibration, baseline/final PWM, applied perturbations, attitude and rates |
| `ap_time_data.csv` | AP clock samples |

Files are produced for topics containing samples; counts are printed on completion.
Every row retains `bag_time_ns` (recorder receipt time). Applied snapshots retain
`time_us` (firmware boot time) and `clock_us` (AP clock); command stamps remain
`stamp_sec` and `stamp_nanosec`. Do not equate these different clock domains.

Array suffixes are **zero-based**: `baseline_0`, `pwm_0`, and `delta_0` correspond
to SERVO1; only indices below `count` describe configured entries. Logical
`requested_0`/`applied_0` are roll, index 1 pitch, index 2 yaw. Attitude is radians,
angular rates rad/s, airspeed m/s, and PWM microseconds. The snapshot preserves the
firmware's accepted sequence; CSV export does not assume the nearest published
command was applied. It does not generate the legacy nearest-time
`synced_all_data.csv` for these surface recordings.

Use the plotting/validation commands above for calibrated output checks and
command-versus-response figures. Successful CSV extraction alone is not validation
of actuator behavior. The parser reads rosbag2 directly, including split bags,
and never publishes or replays commands. Legacy MAVROS/SID `.db3` export remains
available by omitting `--surface` and providing the input and output arguments.

## Optional MAVROS alongside DDS with run_live.sh

The host SITL launcher already forwards MAVLink to UDP 14552. In a sourced ROS
container shell, start MAVROS monitoring without a serial router:

```bash
cd /work
bash useful_shell_scripts/run_live.sh --fcu-url 'udp://:14552@127.0.0.1'
```

The environment needs `tmux` and the `mavros` ROS package, including MAVROS's
GeographicLib datasets. DDS still sends the excitation; MAVROS supplies a parallel
telemetry/service connection. Run the DDS mission publisher separately as above.
The launcher does not start `Drone.py`, set modes, arm, or request the previous
200 Hz message intervals by default. Verify `/mavros/state` reports `connected: true`
and that the desired telemetry topics deliver samples.

The launcher uses tmux **windows**: MAVROS, optional router, optional Drone.py, and
optional recorder. Use Ctrl-B then `w` to choose a window, or Ctrl-B then `d` to
detach. An existing session is preserved; attach with `tmux attach -t MavrosSession`.
Use `--session another_name` for a separate session, with nonconflicting endpoints.

To record both DDS and MAVROS in one bag, choose a new path:

```bash
bash useful_shell_scripts/run_live.sh --detach --bag "$RUN/combined_rosbag"
```

Do not also start the standalone recorder at that path. Alternatively, if MAVROS
is already running, use just the recorder:

```bash
ros2 run ros2_sid record_surface_excitation.sh "$RUN/combined_rosbag"
```

The recorder now records **all discoverable ROS topics**, including hidden topics,
DDS excitation, MAVROS, and publishers that appear after recording starts. The
legacy `--with-mavros` option remains accepted but is no longer necessary. Missing topics are not synthesized;
recording a topic name does not guarantee the vehicle streams it. Existing surface
plotting and `data_parser --surface` work on this bag and select the DDS topics.
For MAVROS CSVs, the parser's legacy `.db3` mode supports its existing topic mapping;
it does not automatically align MAVROS timestamps to firmware boot time.

`--with-drone` explicitly enables `Drone.py`; coordinate any guidance commands it
sends with your experiment. `--serial-device PATH --baud RATE` explicitly enables
a serial MAVLink router forwarding to localhost 14551/14552/14553, for separately
configured environments. The SITL Compose file provides no hardware device access;
this guide uses UDP simulation only. No automatic MAVROS excitation fallback exists.

Before closing the tmux session, quit the DDS publisher and Ctrl-C in the recorder
window so the bag is finalized. Stop MAVROS in its window. The launcher refuses to
kill an existing session automatically. Inspect `--help` for all endpoint options.

### All-topic recording scope

Both `record_surface_excitation.sh BAG_DIRECTORY` and `run_live.sh --bag BAG_DIRECTORY`
record all discoverable topics in the current ROS domain, including hidden topics.
Discovery stays enabled throughout recording. The existing DDS/MAVROS QoS overrides
remain in use; other topics use rosbag2's automatic QoS selection. Only received
messages are saved: this is not a guarantee of lossless delivery or a capture of
raw MAVLink/service calls. Inspect `ros2 bag info BAG_DIRECTORY` after recording to
confirm topics and message counts. All-topic bags can grow substantially faster
than excitation-only bags. Surface plotting and export still select the relevant
DDS topics from the larger recording.

## Automatic mission loading: shared legacy file layout

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds
```

Both interfaces use `ros2_sid/setup/mission_library/current_mission.json`, which
selects a folder using `{"subfolder": "dds_surface"}`. That folder contains
`mission_plan.json` and precomputed CSVs with `time,roll,pitch,yaw` columns.
The plan uses the legacy `filename`, `name`, and integer `order` fields; maneuvers
are sorted by `order`, preloaded, and sampled at the CSV cadence. DDS plans must
also declare `"units": "normalized_surface"`. The six supplied DDS maneuvers are
chirp/multisine on roll, pitch, and yaw separately, at 50 Hz.

The shared pointer currently selects `dds_surface`. To use legacy trajectory
mode, select a legacy folder such as `mission_1` instead. Each backend rejects
the other command units. The old `config/surface_excitation/current_mission.json`
is no longer consulted. Explicit `--experiment` and `--mission-plan` DDS overrides
remain supported. The DDS console still uses `list`, `select N`, `start`, `stop`,
and `quit`; HITL retains its RC activation gate.

The source library is under `ros2_sid/ros2_sid/ros2_sid/setup/mission_library`.
Installed packages use the library under the package share directory (or the
source library if symlink-installed). Rebuild after source edits and restart the
publisher. See [Custom waveforms](custom_surface_waveforms.md) for CSV generation.


## End-to-end test of the manual console and CSV library

With the default `dds_surface` library selected and no other simulator using this
test's ports, the harness can drive the actual interactive console (no
`--autostart`, `--experiment`, or `--mission-plan` argument):

```bash
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container "$SID_ROS_CONTAINER" --interactive-library --rate 50 \
  --output "sid_results/console_library_$(date +%Y%m%d_%H%M%S)"
```

It sends `list`, `select N`, `start`, then `stop`/`quit` after completion, for each
of the six supplied CSV maneuvers. It records the library snapshot, all-topic bag,
command CSVs, and DataFlash log, and validates actual applied actuator outputs.
Inspect `execution.json` and `plots/report.json`. This exercises the console via
stdin rather than replacing it with direct calls to the publisher.

`--dds-domain` and `--dds-port` permit a dedicated test agent/domain when another
simulator is running; that agent must be started separately with matching settings.
Defaults remain domain 0 and UDP 2021. The harness uses TCP 5780/5782/5783.

## Reusing a bag name

You can repeat the same recording command:

```bash
ros2 run ros2_sid record_surface_excitation.sh "$RUN/bag_chirp_roll"
```

If `bag_chirp_roll` exists, the recorder uses the first available indexed name:
`bag_chirp_roll_1`, `bag_chirp_roll_2`, and so on. Existing recordings are preserved.
The recorder prints the selected output path before starting. This also applies to
`run_live.sh --bag`. Use that printed path for `ros2 bag info`, plotting, and parsing;
`RUN` itself does not change. For example, to inspect a second recording:

```bash
ros2 bag info "$RUN/bag_chirp_roll_1"
```

Do not start concurrent recorders with the same requested base name; indexing
selects an unused path at startup but does not reserve it across processes.

## ROS-to-surface command path and straight flight

```text
ROS 2 mission selects and samples the maneuver CSV
                    ↓
/ap/sid/control_surface
(normalized roll, pitch, yaw perturbations)
                    ↓
Micro XRCE-DDS Agent → ArduPilot AP_DDS subscriber
                    ↓
Freshness, enable, mode, failsafe, and amplitude checks
                    ↓
Mix logical perturbations into configured physical surfaces
                    ↓
Convert using each servo’s MIN / TRIM / MAX / REVERSED
                    ↓
Add perturbation to normal controller/mixer output
                    ↓
Clamp to servo limits → final actuator outputs
```

The DDS publisher adds surface perturbations while normal GUIDED navigation and
stabilization continue. It does not itself establish a straight flight path. See
[Straight-flight coordination and safe exit](straight_flight_plan.md) for the optional
implemented coordinator, configuration details, and phase-by-phase tests. The
following section gives the commands to use it in this manual workflow.


## 7. Run an interactive straight-flight maneuver

The DDS mission defaults to using MAVROS to establish a steady course, altitude, and airspeed,
then sends the selected maneuver through DDS. Normal ArduPlane guidance and
stabilization continue throughout. Follow the MAVROS setup below before using the publisher
command in section 5; run only one excitation publisher at a time.

First complete the firmware/Compose setup and the section 5 host SITL launch,
takeoff, and GUIDED steps. Keep the host launcher and DDS agent running. This
section uses the host launcher's MAVROS endpoint, UDP 14552. The automated runner's
TCP endpoints are a separate setup; do not launch both simulators together.

### Terminal B: enter the ROS container and start MAVROS

On the host, from your checkout root:

```bash
cd "$SID_WORKSPACE"
docker compose -f docker-compose.sid.yaml exec ros bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source /tmp/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/work/ros2_sid/ros2_sid/config/surface_excitation/fastdds_udp.xml
source /work/tools/sid/use_run.sh --latest interactive_
export SHARE="$(ros2 pkg prefix --share ros2_sid)"

bash /work/useful_shell_scripts/run_live.sh --detach \
  --fcu-url 'udp://:14552@127.0.0.1' \
  --mavros-config "$SHARE/config/surface_excitation/mavros_straight.yaml"
```

`use_run.sh` selects the existing interactive run directory and exports `RUN`.
Check its printed path. To select a specific run instead, use
`source /work/tools/sid/use_run.sh --path /work/sid_results/YOUR_RUN_DIRECTORY`.
The `/work` mount is independent of your host checkout location.

If MAVROS is already connected on this endpoint, reuse it instead of launching
another instance. The example YAML loads the telemetry and command plugins needed
here without background parameter/mission downloads. Do not use `--with-drone`
or another guidance publisher while the straight-flight coordinator owns targets.

### Which component routes MAVLink?

For this manual's SITL setup, the host launcher already uses MAVProxy to forward
MAVLink to localhost UDP 14552. `run_live.sh` starts MAVROS on that endpoint:

```text
ArduPlane SITL ↔ host launcher / MAVProxy ↔ UDP 14552 ↔ MAVROS
                                                           ↕
                                                straight-flight coordinator

ROS mission ↔ Micro XRCE-DDS Agent ↔ ArduPlane DDS excitation interface
```

MAVROS supplies navigation commands and telemetry. The DDS agent remains necessary
for surface excitation; MAVLink routing does not replace it. `run_live.sh` starts
neither the DDS agent nor the mission publisher.

You can start MAVROS and the all-topic recorder together, using this command
**instead of** the MAVROS-only launch above and the standalone recorder in terminal C:

```bash
bash /work/useful_shell_scripts/run_live.sh --detach \
  --fcu-url 'udp://:14552@127.0.0.1' \
  --mavros-config "$SHARE/config/surface_excitation/mavros_straight.yaml" \
  --bag "$RUN/bag_straight"
```

Attach with `tmux attach -t MavrosSession`. Use Ctrl-B then `w` to select the
`mavros` or `recorder` window; Ctrl-B then `d` detaches without stopping them.
If the session already exists, attach and inspect it rather than starting another
MAVROS instance. Check the recorder window for its actual output path, including
any automatically added index.

The script also supports **mavlink-routerd for a serial connection** through
`--serial-device PATH --baud RATE`. In that mode it starts a `router` window and
forwards to localhost UDP 14551, 14552 (MAVROS), and 14553. Those options require
`mavlink-routerd` and access to the specified serial device. The SITL Compose
configuration intentionally exposes no hardware devices, and this manual does not
use that serial path. For the existing host SITL launcher, retain MAVProxy's
forwarding and omit `--serial-device`; adding another router on the same endpoints
would duplicate the routing setup. `--fcu-url` selects MAVROS's connection but does
not itself launch a UDP/TCP mavlink-routerd instance.

Keep `--with-drone` off during straight-flight tests, because the coordinator owns
the navigation targets. After launch, perform the stream-rate request and connection
checks below. To shut down, quit the mission and wait for recovery, Ctrl-C the
recorder to finalize its bag, then stop MAVROS (and the router if one was used).

Request position telemetry on the MAVROS connection:

```bash
ros2 service call /mavros/set_stream_rate mavros_msgs/srv/StreamRate \
  '{stream_id: 6, message_rate: 10, on_off: true}'
ros2 topic echo /mavros/state --once
ros2 topic hz /mavros/local_position/velocity_local
# Ctrl-C ends this rate display only.
ros2 topic hz /mavros/global_position/rel_alt
# Ctrl-C ends this rate display only.
```

Before starting, verify `connected: true`, `armed: true`, and `mode: GUIDED`.
Position/velocity should arrive at approximately 10 Hz; a successful service call
alone does not prove the stream rate. The default 1 Hz stream is insufficient for
this configuration's one-second freshness check. Reissue the rate request after
reconnecting MAVROS if needed.

### Terminal C: record all ROS topics

Enter another ROS container shell and repeat the sourcing/environment/run-directory
commands above (do not launch MAVROS again). Then:

```bash
ros2 run ros2_sid record_surface_excitation.sh "$RUN/bag_straight"
```

Use an existing all-topic recorder if one is already running. Do not start duplicate
recorders for the same experiment. The script adds an index if the bag path already
exists; retain the actual printed path for plotting. Keep this terminal recording
through settling, excitation, disable, and recovery. It captures DDS commands and
outputs, MAVROS telemetry, and `/sid/straight_flight` events.

### Terminal B: select and start a maneuver

For this simulated airframe, the supplied configuration targets 75 m above home,
23 m/s airspeed, and the ground course measured when setup begins. It requires a
continuous 0.25-second settling dwell before waveform publication.

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --wait-timeout 180 --command-log "$RUN/straight-commands.csv"
```

Neither `--experiment` nor `--mission-plan` is required. The default legacy-layout
`current_mission.json` → selected folder → `mission_plan.json` → ordered CSV files
still supplies the maneuvers. At the mission console, enter:

```text
list
select 0
start
```

Use the indices printed by `list`. In the supplied DDS library, 0/1 select roll
chirp/multisine, 2/3 pitch, and 4/5 yaw. Setup reports `WAITING`, `COMMANDING`, and
`SETTLING`; the waveform begins only after `READY`, followed by `ACTIVE`. Navigation
targets stay fixed during the maneuver. The coordinator does not arm, take off,
or force GUIDED. The example settings are SITL settings, not physical-aircraft
flight-test limits.

To customize navigation targets/tolerances, copy the settings into your run folder
and pass the copied file to `--straight-flight`:

```bash
cp "$SHARE/config/surface_excitation/straight_flight_sitl.json" "$RUN/straight-flight.json"
```

Edit that copy before launching. Maneuver amplitudes and waveforms still come from
the selected CSV or an explicit experiment override; see
[custom waveforms](custom_surface_waveforms.md). Use `--no-straight-flight` to retain excitation-only behavior without this
navigation/LOITER coordinator. Omitting both flags enables the packaged settings.

### Settling tolerances and active limits

These values reflect the current source `straight_flight_sitl.json`: target altitude
**75 m above home**, airspeed **23 m/s**, and `dwell_s: 0.25`. The configuration
file passed to the running publisher is authoritative; installed defaults can
still contain older values until the package is rebuilt.

Before excitation starts, **all** settling conditions must hold continuously:

| Measurement | JSON parameter | Required condition with these targets |
|---|---|---|
| Ground course | `course_tolerance_deg: 5` | Within ±5° of the target course |
| Bank angle | `bank_tolerance_deg: 5` | Within ±5° of wings level |
| Roll, pitch, and yaw rates | `rate_tolerance_rad_s: 0.15` | Each within ±0.15 rad/s (about ±8.6°/s) |
| Altitude above home | `altitude_tolerance_m: 5` | 70–80 m (75 m ±5 m) |
| Airspeed | `airspeed_tolerance_m_s: 2` | 21–25 m/s (23 m/s ±2 m/s) |
| Climb/descent rate | `climb_tolerance_m_s: 1.5` | Within ±1.5 m/s |

With `course_deg: null`, the coordinator captures the measured ground course when
fresh, authorized setup begins. It does not continually change the target to
follow the current course. Any settling-condition violation resets the dwell
timer. `dwell_s: 0.25` means 0.25 continuous seconds inside all tolerances, not
0.25 seconds from typing `start`. Reaching the target condition can take longer.

The existing freshness and authorization checks also apply: required navigation
and output telemetry must be no older than `telemetry_timeout_s: 1`, vehicle state
must be no older than `state_timeout_s: 3`, and the vehicle must be connected,
armed, in GUIDED, and report normal active system status.

Once READY/ACTIVE, wider limits allow the intended excitation response:

| Measurement | JSON parameter | Allowed range with these targets |
|---|---|---|
| Bank magnitude | `max_bank_deg: 35` | At most 35° |
| Altitude error | `max_altitude_error_m: 30` | Within ±30 m of target (45–105 m above home) |
| Ground-course error | `max_course_error_deg: 40` | Within ±40° of the fixed target |
| Airspeed | `min_airspeed_m_s: 20`, `max_airspeed_m_s: 30` | 20–30 m/s |

Exceeding an active limit stops the maneuver and invokes the recovery procedure.
The narrow settling rate/climb tolerances do not remain active abort limits.
Freshness and authorization checks continue throughout.

To shorten settling, change `dwell_s` first and assess the recorded entry conditions
in SITL. Reducing it does not make the aircraft reach its targets faster, and a
shorter dwell provides less evidence that the initikval condition is steady.
`setup_timeout_s: 90` is the maximum setup wait; reducing it causes an earlier
abort. `recovery_timeout_s: 8` bounds recovery confirmation. Neither telemetry
freshness timeout is a settling delay. `max_duration_s: 30` bounds the selected
maneuver duration rather than setting dwell time.

Pass your edited source file with `--straight-flight PATH`, or rebuild the ROS
package to update the installed default. Do not restart services during an active
experiment. These are configured test thresholds, not physical-flight safety
limits. The earlier recorded verification used a three-second dwell and different
target settings; it does not validate this new 0.25-second dwell configuration.

### Stop, quit, and run another segment

- `stop` cancels the maneuver and returns to the console after recovery.
- `quit`, EOF, Ctrl-C, and SIGTERM cancel the maneuver and exit after recovery.
- Normal completion follows the same recovery procedure.

With the default straight-flight coordination, recovery means repeated DDS disable, observation
of a fresh **zero excitation contribution** (not neutral total servo output), then
requesting and confirming LOITER if the aircraft remains armed and authorized in
GUIDED. Look for `Straight flight: DONE: LOITER target confirmed at saved start` before shutting down
supporting services. Operator/failsafe mode changes and disarm take precedence;
the coordinator does not force GUIDED back on.

If recovery is unconfirmed, the process reports failure and the need for operator
intervention. Do not assume that exiting the process proves LOITER was reached.
Killing the process/container or losing communications cannot guarantee navigation
recovery; firmware timeout still removes stale excitation.

To run another maneuver after `stop` or normal completion, use the host MAVProxy
prompt to select `mode GUIDED` again, then `select N` and `start` in the mission
console. Settling is checked again. Use a new command-log filename when launching a
new publisher process so earlier command records are preserved. For the per-maneuver
command-log validation below, run one maneuver per publisher process; quit after
recovery and relaunch with a fresh command-log filename for the next maneuver.

After the last maneuver, `quit` and wait for recovery. Then Ctrl-C the recorder in
terminal C to finalize the bag. Keep MAVROS, DDS, and SITL alive until that handoff
finishes; only then follow the existing shutdown procedure.

### Replot the recorded commands and response

In a sourced container shell with the same `RUN` selected, set `BAG` to the actual
path printed by the recorder (including an index suffix, if added):

```bash
export BAG="$RUN/bag_straight"
ros2 bag info "$BAG"
ros2 run ros2_sid plot_surface_excitation.py "$BAG" \
  --output "$RUN/straight-plots" --straight-flight \
  --command-log "$RUN/straight-commands.csv"
```

This reads the bag offline without replaying flight commands. Outputs include
requested/applied excitation, baseline/final PWM, aircraft response,
`straight_navigation.png`/`.pdf`, `straight_track.png`, exported events, and numeric
validation. Green navigation windows mark ACTIVE excitation; exclude settling and
recovery turns from later identification. Deliberately interrupted maneuvers cover
only the transmitted portion of the waveform, not the full requested sweep.
For a bag spanning multiple publisher processes, repeat `--command-log` for each
process's CSV. The earlier DataFlash comparison option remains available.

Plots can be written even when validation exits with status 1. Inspect
`report.json` and `straight_report.json` for the failing check. A confirmed idle
shutdown (`RETURNING`/`DONE`, reason `No navigation ownership`, with no recorded
maneuver attempt) is reported as skipped, not as a passed maneuver. At least one
evaluated maneuver must pass, and every evaluated attempt must pass. Started but
aborted attempts and incomplete recordings are not excused by this rule. Only the
CSVs supplied with `--command-log` receive command-to-applied-signal validation;
navigation checks alone do not establish that every waveform was delivered.

### Optional automated straight-flight and exit tests

To reproduce the validation automatically instead of operating the manual host
simulator, first stop that simulator cleanly and free the test ports. Keep Compose
running. From the host checkout, use a new output directory for each invocation:

```bash
export SID_ROS_CONTAINER="$(docker compose -f docker-compose.sid.yaml ps -q ros)"
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container "$SID_ROS_CONTAINER" --interactive-library --straight-flight \
  --rate 50 --output "sid_results/straight_$(date +%Y%m%d_%H%M%S)"

# Separate run: intentionally interrupt active maneuvers with
# quit, stop, SIGINT, SIGTERM, quit, and stop.
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container "$SID_ROS_CONTAINER" --interactive-library --straight-flight \
  --exit-active cycle --rate 50 \
  --output "sid_results/straight_exit_$(date +%Y%m%d_%H%M%S)"
```

These use the normal domain-0/UDP-2021 Compose agent, launch their own conventional
Plane, perform takeoff/GUIDED, drive the actual manual console, and retain bags,
DataFlash logs, plots, and validation results. They also launch their own MAVROS
connection and give the safety probe a separate serial endpoint. For measured
results and retained failed attempts, see the
[verification report](../../../ardupilot_sid/docs/control_surface_excitation/VERIFICATION.md).


### Separate plots for every axis and waveform

For a complete six-maneuver recording with one command CSV per maneuver, run:

```bash
python3 -m ros2_sid.surface_axis_plots "$BAG" \
  --command-directory "$RUN" --output "$RUN/individual_axis_plots"
```

The command directory must contain `chirp_roll-commands.csv`,
`multisine_roll-commands.csv`, and the corresponding pitch/yaw files from that bag
(the automated six-maneuver runner writes these names). This produces six PNG/PDF
pairs with command/applied excitation, baseline/final servo PWM, angular rates,
attitude changes, and airspeed. This helper is for conventional surfaces.
See the [report's individual-axis figures](../../../ardupilot_sid/docs/control_surface_excitation/VERIFICATION.md#individual-axis-straight-flight-chirp-and-multisine-plots)
for the recorded examples.


### Default mission command and explicit opt-out

After the MAVROS/telemetry setup above, no navigation JSON flag is needed:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --wait-timeout 180 --command-log "$RUN/straight-commands.csv"
```

To intentionally keep the aircraft's existing GUIDED navigation (which may be
loitering) and send only the excitation:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds --no-straight-flight \
  --wait-timeout 180 --command-log "$RUN/excitation-only-commands.csv"
```

The opt-out still disables excitation on exit, but does not request LOITER. Missing
MAVROS telemetry in the default workflow withholds excitation; it does not silently
fall back to excitation-only mode. The automated `surface_sitl.py` runner retains
its own `--straight-flight` switch: without it, it explicitly opts its publisher
out, preserving the original excitation-only test. The plotting command's
`--straight-flight` flag still requests navigation plots and has not changed.

## Angle-of-attack and sideslip recording

The same recorder and plotting commands now include estimated α/β when the
updated firmware publishes them. The identification configuration requests 50 Hz;
the report measures the achieved rate. No additional maneuver arguments are needed.
See [air-data recording and rate selection](air_data_recording.md).
