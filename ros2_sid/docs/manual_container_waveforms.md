# Send control-surface waveforms from inside the ROS container

For automatic mission loading and creating your own chirps/multisines, see [Custom DDS waveforms](custom_surface_waveforms.md).

This walkthrough uses `ros2_sid` directly: enter the container, create a waveform,
and type `start` to publish it to `/ap/sid/control_surface`. It uses the DDS backend
of `run_mission_SITL.py` and an already running ArduPlane simulation.

The commands are normalized additive surface perturbations. Normal GUIDED guidance
and stabilization continue running; the firmware maps roll/pitch/yaw to configured
servo functions and applies each channel's calibration and reversal.

## 1. Start the environment on the host

Build the DDS-enabled Plane binary first, as described in the
[manual SITL guide](manual_surface_excitation_sitl.md). Then, in a host terminal, set `SID_WORKSPACE` to your checkout root (the
directory containing `docker-compose.sid.yaml`). Replace this example path:

```bash
export SID_WORKSPACE="/path/to/ros2_trajectory_docker"
cd "$SID_WORKSPACE"
docker compose -f docker-compose.sid.yaml up -d --wait --wait-timeout 900
```

Compose builds the ROS interfaces and mission packages and runs their surface tests.
It starts the DDS agent, but **does not launch or fly ArduPlane**.

Start the host launcher in another host terminal:

```bash
cd "$SID_WORKSPACE"
bash useful_shell_scripts/run_arduplane_host_dds.sh --headless
```

It uses the `ardupilot_sid` binary and excitation defaults, creates a fresh log
folder, and opens MAVProxy. Omit `--headless` if you want its graphical map/console.
Follow the manual SITL guide's **step 5, takeoff commands** in that same MAVProxy
prompt to take off and enter GUIDED. Keep this terminal running.
Do not run `surface_sitl.py` concurrently: that harness starts its own simulator
and command publishers.

Before publishing, the simulated aircraft must be armed, airborne in normal
fixed-wing GUIDED, with safety off, no relevant failsafe, valid servo calibration,
and `SX_ENABLE=1`. The supplied conventional SITL parameters configure the feature.
The publisher does not arm the aircraft or change its flight mode.

Set `SID_WORKSPACE` to the same checkout path in each new host shell; exported
variables are not automatically shared between independently opened terminals.

## 2. Enter the ROS container

In another host terminal:

```bash
cd "$SID_WORKSPACE"
docker compose -f docker-compose.sid.yaml exec ros bash
```

Everything below runs **inside the container**, unless explicitly marked otherwise:

```bash
source /opt/ros/humble/setup.bash
source /tmp/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/work/ros2_sid/ros2_sid/config/surface_excitation/fastdds_udp.xml
cd /work
```

For an existing SITL instance configured with domain 41, set `DDS_DOMAIN_ID` to
`0` and restart that simulator, or launch a fresh run with the updated defaults.
Recreate existing Compose services with the `up` command above to apply their new
environment. In existing host ROS shells, also run `export ROS_DOMAIN_ID=0`.

Compose mounts your checkout at `/work` regardless of its host location; files saved there remain available on the
host. Repeat the sourcing commands whenever you open a new container shell.

Verify actual DDS delivery:

```bash
ros2 topic echo /ap/time --once --qos-reliability best_effort
ros2 topic echo /ap/sid/status --once --qos-reliability best_effort
```

Allow more than two minutes for this fork's DDS startup delays. Topic discovery
alone does not establish delivery. If these commands never receive a sample, check
the simulator, agent logs, domain 0, and UDP port 2021 before proceeding.

## 3. Create single-axis chirp and multisine experiments

Make a new output directory and print its path; use that same path in any additional
container shells used for monitoring or recording:

```bash
source /work/tools/sid/use_run.sh --new container_waves
```

Create a roll-only chirp (10 seconds, 0.2–3 Hz, 10% normalized amplitude):

```bash
cat > "$RUN/chirp_roll.json" <<'JSON'
{
  "units": "normalized_surface",
  "rate": 50,
  "duration": 10,
  "axes": {
    "roll": {
      "signal": "chirp",
      "amplitude": 0.10,
      "start_frequency": 0.2,
      "end_frequency": 3.0
    }
  }
}
JSON
```

Create a roll-only multisine (12 seconds, three frequency components):

```bash
cat > "$RUN/multisine_roll.json" <<'JSON'
{
  "units": "normalized_surface",
  "rate": 50,
  "duration": 12,
  "axes": {
    "roll": {
      "signal": "multisine",
      "amplitude": 0.10,
      "frequencies": [0.5, 1.0, 2.0],
      "amplitudes": [0.03, 0.03, 0.04],
      "phases": [0.0, 1.0, 2.0]
    }
  }
}
JSON
```

Generate pitch-only and yaw-only versions and a selectable six-maneuver plan:

```bash
python3 - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
maneuvers = []
for axis in ("roll", "pitch", "yaw"):
    for signal in ("chirp", "multisine"):
        spec = json.loads((run / f"{signal}_roll.json").read_text())
        spec["axes"] = {axis: spec["axes"]["roll"]}
        filename = f"{signal}_{axis}.json"
        (run / filename).write_text(json.dumps(spec, indent=2) + "\n")
        maneuvers.append({"name": f"{axis} {signal}", "experiment": filename})
(run / "mission_plan.json").write_text(json.dumps(
    {"units": "normalized_surface", "maneuvers": maneuvers}, indent=2
) + "\n")
PY
```

Only the named axis is excited; omitted axes are zero. Frequencies are Hz and phases
are radians. The multisine sum-of-absolute-amplitudes bound is 0.10. Firmware limits
and available servo travel can reduce what is actually applied. These files use
50 Hz publication, matching the cadence of the recorded individual-axis tests.

## 4. Optionally start a ROS bag before the maneuver

In a second host terminal, enter the container as in step 2 and source ROS/the
workspace. Select the latest waveform run and check its printed path matches the first shell,
then start recording:

```bash
source /work/tools/sid/use_run.sh --latest container_waves_
ros2 run ros2_sid record_surface_excitation.sh "$RUN/rosbag"
```

Leave the recorder running. It records command, status, applied-output, and AP-time
topics. The applied-output message includes baseline/final PWM and aircraft response.

## 5. Send a waveform through ros2_sid

Back in the first container shell, choose either a single experiment or the menu.
Run **only one publisher at a time**.

For a single roll chirp:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --experiment "$RUN/chirp_roll.json" --wait-timeout 180 \
  --command-log "$RUN/chirp_roll-commands.csv"
```

At the publisher console, type:

```text
start
```

The publisher synchronizes to `/ap/time`, completes a disable/enable handshake, and
waits for firmware authorization. After the maneuver it sends disable samples and
returns to the interactive console. Type `quit` to exit. Use `stop` to end a maneuver
early; a shortened maneuver is not a full-duration validation pass.

To send a different waveform, replace the experiment and CSV filenames, for example:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --experiment "$RUN/multisine_pitch.json" --wait-timeout 180 \
  --command-log "$RUN/multisine_pitch-commands.csv"
```

Alternatively, launch the menu instead of either single-experiment command:

```bash
ros2 run ros2_sid run_mission_SITL.py --interface dds \
  --mission-plan "$RUN/mission_plan.json" --wait-timeout 180 \
  --command-log "$RUN/menu-commands.csv"
```

Type `list`, then `select N`, then `start`. Indices start at zero:

| Index | Maneuver |
|---|---|
| 0 | Roll chirp |
| 1 | Roll multisine |
| 2 | Pitch chirp |
| 3 | Pitch multisine |
| 4 | Yaw chirp |
| 5 | Yaw multisine |

Wait for the maneuver and disable sequence to finish before selecting another one.
The menu CSV contains all selected runs; for separate per-maneuver command CSVs,
use the single-experiment invocation each time.

## 6. Observe and plot what was actually applied

From another sourced container shell, inspect current state or outputs:

```bash
ros2 topic echo /ap/sid/status --qos-reliability best_effort
ros2 topic echo /ap/sid/output --once --qos-reliability best_effort
```

The first command runs until Ctrl-C. State 2 is ACTIVE; state 1 is excitation-ready,
which is distinct from vehicle arming. A received command alone is not proof that
servo excitation was applied. Inspect baseline/final PWM and applied deltas.

When finished, type `quit` in the publisher. Wait at least one second, then Ctrl-C
in the recorder terminal to finalize the bag. In a sourced shell with `RUN` set:

```bash
ros2 bag info "$RUN/rosbag"
ros2 run ros2_sid plot_surface_excitation.py "$RUN/rosbag" --output "$RUN/plots"
```

For a bag containing a complete single roll chirp and its matching command CSV,
include that CSV for command-coverage validation:

```bash
ros2 run ros2_sid plot_surface_excitation.py "$RUN/rosbag" \
  --output "$RUN/plots" --command-log "$RUN/chirp_roll-commands.csv"
```

Open the PNG/PDF figures and `report.json` in the host's corresponding
`sid_results/container_waves_.../plots` folder. Plots separate the applied excitation
from the baseline controller output and show attitude/angular-rate response. The
standard `.BIN` `RCOU` record contains their combined servo output. PWM is commanded
output, not measured physical surface position.

Offline plotting needs no live aircraft and does not replay commands. For the full
six-run acceptance test, exact DataFlash comparison, and six individual-axis report
figures, use the [manual SITL test guide](manual_surface_excitation_sitl.md).

## 7. Exit

After the publisher and recorder have stopped, type `exit` to leave each container
shell. Stop the host launcher and its MAVProxy session in their host terminal. To stop
this Compose project's services, run on the host:

```bash
cd "$SID_WORKSPACE"
docker compose -f docker-compose.sid.yaml stop
```

For the HITL entrypoint, the shared DDS waveform format is the same, but activation
requires its fresh RC low-to-high transition rather than this SITL console `start`
workflow. This guide covers simulation only; see the
[firmware usage documentation](../../../ardupilot_sid/docs/control_surface_excitation/README.md)
for that interface distinction.

## Export interactive recordings with the existing data_parser

After stopping the recorder, use a sourced ROS container shell and select the
existing waveform run. Do not create a new directory for analysis:

```bash
source /work/tools/sid/use_run.sh --latest container_waves_
python3 -m ros2_sid.post_process.data_parser "$RUN/rosbag" \
  --surface --output "$RUN/parsed_surface"
```

For the interactive SITL guide's separate roll-chirp bag, replace `"$RUN/rosbag"`
with `"$RUN/bag_chirp_roll"`. Choose a new output folder for each export. After
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


## Selecting a run without copying timestamps

Source the helper in every container shell that needs `RUN`:

```bash
source /work/tools/sid/use_run.sh --latest container_waves_
```

Use `--latest interactive_` for runs created by the host SITL launcher, or
`--path /work/sid_results/RUN_NAME` to select an exact directory. `--new container_waves`
creates a unique directory for new waveform files. Only use `--new` once per run;
other shells should select that existing directory. The helper prints the selected
path. `source` is necessary to update your current shell's environment. Host use
also works as `source tools/sid/use_run.sh ...` from the checkout root; paths are
resolved relative to the script's workspace, not a hardcoded username.

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
minimal-change approach and phase-by-phase tests. This plan is not yet implemented.


For a settled straight segment, add `--straight-flight` using the SITL configuration
and MAVROS setup in [the straight-flight guide](straight_flight_plan.md). The default
CSV selection still applies. With this option, `stop`, `quit`, EOF, and Ctrl-C
disable excitation, wait for observed zero contribution, and confirm LOITER while
the vehicle remains armed GUIDED. Keep the agent, MAVROS, and recorder running
until the handoff finishes; operator/failsafe mode changes take precedence.
