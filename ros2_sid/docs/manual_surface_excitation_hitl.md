# Manually run HITL DDS surface excitation

On a new machine, first complete [Docker image setup](../README_surface_excitation.md#first-time-docker-image-setup).
It builds the shared image used by both ROS and the DDS agent from this
repository’s Dockerfile. No separate DDS-agent image is needed.

## Ground tests with physical control surfaces

Ground tests require firmware containing `SX_GND_TEST`; rebuild and flash the
appropriate custom Plane firmware before using this parameter. Its default is
`0`, which preserves airborne, normal-flight-stage authorization. Set it to `1`
to require the firmware's `is_flying()` detector to report grounded instead.
The earlier ground-test firmware still required arming; rebuild and flash the
updated image for disarmed operation. Previously generated `.apj` files do not
change when source code is edited.
Other values do not authorize excitation. This parameter does not disable
failsafes: GUIDED, hardware safety off, `SX_ENABLE=1`, healthy RC/failsafe state,
and the existing output limits and command watchdog are still required.

Both ground and flight excitation are independent of the vehicle arming state; keep the vehicle disarmed for
a bench test and physically isolate propulsion. Hardware safety must still be
released to allow surface motion. Excitation only
adds surface perturbations to the controller outputs; it neither commands
absolute surface positions nor disables the motor. Physical servo output also
depends on the board's output configuration. Onboard simulation with physical
output masks disabled will not move physical surfaces; its flight detector
describes the simulated aircraft. Use the appropriate physical-aircraft firmware
and output configuration for a real ground test.

Use the connection/environment setup below and an all-topic recorder, but run
the mission directly rather than the default straight-flight launcher:

1. While disarmed, keep the experiment RC switch LOW and set `SX_GND_TEST=1`
   and `SX_ENABLE=1` through the GCS.
2. With propulsion isolated, stay disarmed, release hardware safety and select
   GUIDED. Retain normal failsafe checks. Excitation also requires `SERVO_AUTO_TRIM=0`,
   `Q_ENABLE=0`, no active system-identification experiment, no active navigation
   script, and no servo emergency stop. This interface supports fixed-wing-only
   configurations; do not change a QuadPlane configuration just to bypass this gate.
3. In the sourced ROS shell, run (choose a fresh command-log path):

   ```bash
   ros2 run ros2_sid run_mission_HITL.py --interface dds \
     --no-straight-flight --wait-timeout 180 \
     --command-log "$RUN/ground-commands.csv"
   ```

   `RUN` is the recording directory selected in the manual setup below.
   `--no-straight-flight` skips the altitude, airspeed, settling and navigation
   checks from `straight_flight_sitl.json`, and disables automatic navigation recovery.
   The RC gate, clock synchronization and firmware acknowledgement still apply.
4. Use `list` and `select N`, then switch LOW to HIGH to start. Confirm the
   console reports firmware-confirmed ACTIVE excitation and verify surface motion.
   Switch LOW to stop; `stop`, `quit` and Ctrl+C also stop excitation. A mode
   change, failsafe, or airborne detection removes ground-test authorization.
   Lost/stale commands retain the existing watchdog and bounded return behavior.
5. Wait for zero excitation, disarm, then restore `SX_ENABLE=0` and
   `SX_GND_TEST=0` before reconnecting propulsion or preparing for flight.
   Do not change the ground-test parameter during an active maneuver.

Disarming alone does not stop excitation. Use the RC switch LOW,
mission stop, hardware safety, or a mode change to stop it. Disarming also does not stop flight-mode excitation. Flight mode still requires
airborne detection and normal flight stage. Normal flight-controller arming and
throttle handling are unchanged.

For bench verification, first confirm `SX_GND_TEST=0` rejects grounded excitation,
then verify activation with `1`, RC-LOW stop, publisher-loss timeout, and
mode-change stop. Record commanded/applied excitation and servo PWM; PWM telemetry
alone does not prove physical surface motion. Ground-test runs have no
straight-flight recovery, so omit `--straight-flight` when plotting their bags.

### If ground excitation waits for firmware authorization

`WAITING: firmware authorization/disable acknowledgement; state=..., reason=...`
now prints names as well as codes, for example
`state=FAULT (4), reason=UNAUTHORIZED (1)`. This means the publisher has not yet received a fresh, experiment-ready status
acknowledging one of its own disable samples. It does not identify a Python
exception, and increasing `--wait-timeout` does not remove the blocking condition.

In another sourced ROS container shell, capture:

```bash
ros2 topic echo /ap/sid/status --once
ros2 topic echo /mavros/state --once
```

If a topic produces no output, stop the echo with Ctrl+C and check the agent,
serial connection, ROS domain and firmware DDS configuration. Use the actual
numeric values from the console/status message to choose the next check:

| Status | Interpretation and next check |
|---|---|
| `state=unavailable` | No status received by this publisher; check DDS connectivity. |
| `state=0` | Firmware excitation is disabled; confirm `SX_ENABLE=1` on the connected Cube. |
| `state=4, reason=1` | Unauthorized. Check the firmware gates listed below. |
| `reason=3` | A command was stale or incorrectly timestamped; check `/ap/time`, transport delay and command freshness. |
| `reason=5` | Invalid or changed configuration; inspect `SX_*` bounds, servo functions/calibration and mixer settings. |
| `reason=6` | Firmware detected a clock discontinuity. |
| `reason=7` | Firmware detected a DDS disconnect. |
| `state=1` but still waiting | Experiment-ready is not enough: status must be fresh and its `sequence_id` must match this publisher's disable samples. Check command delivery and other publishers. |

For `state=4, reason=1`, confirm the connected Cube actually runs the updated
hardware firmware and exposes `SX_GND_TEST=1`; confirm `SX_ENABLE=1`, GUIDED,
hardware safety off (arming is not required in ground mode), no aircraft/RC
failsafe, and all the step-2
conditions above. The firmware must also detect grounded operation. Simulation
firmware evaluates simulated flight state, not whether the physical board is
sitting on a bench. The status currently combines these authorization gates into
one reason, so `reason=1` alone cannot identify which gate failed.

Keep just one excitation publisher running. After correcting the cause, return
the RC experiment switch LOW, stop/relaunch the mission if it timed out, select a
maneuver, then move LOW to HIGH. The publisher sends the disable handshake
automatically; do not bypass it or publish a second command stream manually.

## Recommended: one launcher, no RUN/BAG variables

From **any checkout location**, open a host terminal in the repository root:

```bash
./tools/sid/run_surface_test.sh hitl
```

This starts/reuses the Docker services, rebuilds `ros2_sid`, creates a unique
`sid_results/hitl_YYYYMMDD_HHMMSS/` directory, and opens four tmux panes:

1. Connections: MAVLink router on ACM0, forwarding to MAVROS and GCS.
2. MAVROS: telemetry and straight-flight navigation services.
3. Recorder: **all ROS topics**, including topics discovered later.
4. Mission: HITL RC-gated maneuver selection, after DDS/MAVROS/recorder readiness.

Use the GCS to establish airborne GUIDED operation in Cube onboard simulation. Keep RC channel 8 LOW while selecting; HIGH starts the selected maneuver and LOW stops it.

The default DDS library now includes doublets:

```text
list
select 6    # roll doublet
select 7    # pitch doublet
select 8    # yaw doublet
```

Enter just one `select` command (without the comment). Then move the RC switch LOW → HIGH. Console `start` cannot bypass the RC gate.
The existing chirps/multisines keep menu numbers 0–5. Each doublet uses the legacy
`multi_step` generator: positive then negative pulses, one-second initial delay,
1 Hz natural frequency, 50 Hz samples, and 15 seconds total. The nominal pulse
width is 0.25 s; the legacy sample rounding and zero boundary samples are retained.
Amplitude is **0.10 normalized surface fraction**, not radians. Existing firmware
limits, RC stop, timeout protection and configured recovery still apply.


### Amplitude modifier and firmware limits

Both `run_mission_SITL.py --interface dds` and `run_mission_HITL.py --interface dds`
read `amplitude_modifier` from `ros2_sid/setup/mission_library/current_mission.json`
inside the package. It defaults to **1.0** and multiplies every roll, pitch, and yaw
sample before publishing and command CSV logging. Restart the mission script after
editing the JSON. `--amplitude-modifier` overrides the JSON value; explicit
`--experiment` or `--mission-plan` sources default to 1.0 unless this override is supplied.

The `tools/sid/run_surface_test.sh` launcher records the selected JSON modifier in
`session.json` and passes it as `--amplitude-modifier` alongside its explicit
mission-plan snapshot. Start a new session after changing the source JSON;
restarting a pane in an existing session reuses that session's saved modifier.

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
./tools/sid/run_surface_test.sh hitl --name doublet_roll
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
./tools/sid/plot_surface_test.sh --latest hitl
# Or replot a particular completed session:
./tools/sid/plot_surface_test.sh sid_results/hitl_YOUR_SESSION
```

The plotting helper reads actual paths from the manifest, runs bag info and surface/
straight-flight analysis plus RC analysis, and writes `plots/`. It never replays commands.
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

Host: Docker Compose, tmux, and WSL USB device access and fuser.
The configured ROS image must contain MAVROS and mavlink-router.
HITL defaults: ACM0 for MAVLink, ACM1 for DDS, both 115200 baud; GCS UDP destination 127.0.0.1:14550.

```bash
./tools/sid/run_surface_test.sh hitl --mavlink-device /dev/ttyACM0 \
  --dds-device /dev/ttyACM1 --baud 115200 --dds-baud 115200 \
  --gcs 192.168.1.20:14550
```

Set the GCS address to the machine running Mission Planner/MAVProxy. The serial
router runs **inside the ROS container**; do not also start host `run_soh.sh` on ACM0.
The launcher refuses an already-owned serial device. No Cube flashing is performed.

[Verification results and doublet plots](doublet_launcher_verification.md).

## Individual commands and troubleshooting

The detailed commands below remain available when diagnosing individual services.


### Recovery to the original GUIDED loiter target

The default `"recovery_mode": "GUIDED"` saves the existing GUIDED navigation
target before sending straight-flight commands. This is the loiter waypoint's
latitude, longitude, and absolute altitude, not home or the aircraft's position
at maneuver start. Setup waits for a valid fresh target echo; it never guesses
where the original loiter center was.

After completion, `stop`, `quit`, Ctrl-C, or an HITL RC stop:

1. Disable excitation and wait for fresh zero applied perturbation.
2. Briefly select LOITER to clear the persistent straight-flight overrides.
3. After LOITER is confirmed, restore the saved waypoint with `DO_REPOSITION`
   and return to GUIDED.
4. Confirm fresh GUIDED state and a matching latitude/longitude/altitude echo.

The final mode is **GUIDED**. `DONE` confirms the navigation handoff, not arrival
at the loiter circle. The saved waypoint altitude is restored, even if it differs
from the straight-flight altitude. Firmware-configured loiter radius/direction
are used; a previous custom radius/direction cannot be recovered from target echoes.
Keep RC LOW until recovery finishes, then select and use LOW-to-HIGH for another test.

MAVROS must enable `setpoint_raw` (included in `mavros_straight.yaml`). The mission
requests target telemetry at 5 Hz before capture. Missing/invalid target telemetry
blocks setup until its timeout. The existing `recovery_timeout_s` covers the entire
handoff; failure leaves the current mode alone and reports recovery unconfirmed.
Observed operator/failsafe takeover has priority. State telemetry cannot distinguish
an operator selecting the same expected LOITER/GUIDED mode from our own transition.

No firmware changes are needed for this sequence on the inspected fork. Rebuild
or install the ROS changes, and restart the mission or launcher to load the updated
configuration. `--no-straight-flight` disables navigation and recovery completely.

For the earlier behavior, explicitly set `"recovery_mode": "LOITER"`: return to
the aircraft's setup-start location and finish in LOITER. The
[return-to-start verification report](return_to_start_verification.md) describes
that earlier behavior, not flight validation of the new GUIDED sequence. See the
[recovery reference](straight_flight_plan.md#default-recovery-restore-the-original-guided-loiter-target)
for confirmation criteria, event fields, and limitations.


The simulated HITL procedure below uses the Cube's **Simulation on Hardware**
firmware; the physical ground-test section above uses standard hardware firmware.
Simulation firmware installation
is covered in [Cube setup](cubeorangeplus_simulation_on_hardware.md). The same
mission library, waveforms, plots and straight-flight coordinator used by SITL
are shared by `run_mission_HITL.py --interface dds`.

## RC switch behavior

The defaults match the legacy HITL execution gate:

| Setting | Default |
|---|---|
| Input | `/mavros/rc/in` |
| Channel | 8 (one-based; array index 7) |
| Run | channel 8 >= 1550 microseconds |
| Stop | channel 8 < 1550 microseconds |
| RC receipt timeout | Disabled by `rc_timeout_s: null` in the packaged straight-flight JSON; otherwise 0.5 seconds when the key/config is absent |

**To start a test:** set the switch LOW, select a maneuver, then switch HIGH.
The aircraft settles into straight flight before the test signal begins.
Repeat this sequence after every test or stop; starting with the switch already
HIGH does nothing. Switching LOW—even briefly—or receiving invalid RC input stops
the test. Missing initial RC input blocks activation. If an RC timeout is enabled,
stale input also stops the test; with `rc_timeout_s: null`, message silence alone
does not revoke the last valid HIGH. The console `start` command and `--autostart` cannot start HITL tests.

**Switch LOW:** stops the test signal, then briefly enters LOITER and restores
the original waypoint in GUIDED with the default recovery setting.
It **does not stop the motor or disarm the aircraft**. If you change flight mode
or a failsafe takes over, that takes priority. Keep the switch low until recovery
finishes; raise it again to start another test.

## Configuring optional checks with `null`

For every setting, including command targets, units, timers, and recovery, see the
[complete JSON parameter reference](straight_flight_plan.md#json-parameter-reference).

Edit [straight_flight_sitl.json](../config/surface_excitation/straight_flight_sitl.json)
and restart the mission to load changes. The normal command loads the installed
copy; rebuild/install after editing source, or pass `--straight-flight /work/path/to/settings.json`
to use an explicit file. Launcher snapshots retain the settings copied at launch.

| JSON setting | What `null` disables |
|---|---|
| `rc_timeout_s` | RC sample-age checks; valid input, LOW-to-HIGH activation, and LOW/invalid-input stop still apply |
| `course_tolerance_deg`, `bank_tolerance_deg`, `rate_tolerance_rad_s` | The corresponding course, bank, or angular-rate settling requirement |
| `altitude_tolerance_m`, `airspeed_tolerance_m_s`, `climb_tolerance_m_s` | The corresponding altitude, airspeed, or climb settling requirement |
| `max_bank_deg`, `max_altitude_error_m`, `max_course_error_deg` | The corresponding envelope limit during READY/ACTIVE |
| `min_airspeed_m_s`, `max_airspeed_m_s` | The lower or upper airspeed envelope limit, independently |
| `max_duration_s` | The coordinator's maximum planned maneuver-duration check; the experiment still ends at its own duration |

Only `rc_timeout_s` is disabled in the supplied configuration. To disable another
check, change its existing numeric value to JSON `null`. Restore a positive number
to enable it again. Missing limit/tolerance keys are errors, not opt-outs.

The target `airspeed_m_s` must remain numeric, even when airspeed checks are
disabled. Flight targets, `dwell_s`, telemetry/state freshness, setup/recovery
deadlines, valid measurements, and navigation authorization remain required.
Disabling every settling tolerance still requires dwell. There is no `armed`
setting in this JSON; arming behavior is described in the ground-test section.
`course_deg: null` has a separate meaning: capture the current course.

RC precedence is explicit CLI override, then JSON, then the 0.5-second mission
default. Use `--rc-timeout 0.5` to enable a timeout or `--rc-timeout null` to disable
it; numeric RC timeouts must be in `(0, 2]` seconds. With `--no-straight-flight`,
the JSON is not loaded, so use `--rc-timeout null` explicitly if needed for the
ground-test command. `--wait-timeout` does not change RC freshness.

With RC freshness disabled, the last valid HIGH remains authorized when messages
stop arriving. Firmware failsafes and command watchdogs remain independent and
are not disabled by these JSON settings. Startup logs list disabled checks;
straight-flight JSONL events record the effective configuration and CLI overrides.
See [optional-check details](straight_flight_plan.md#optional-checks-in-the-dds-mission-configuration).

### Examples: change only the checks you want to skip

For an optional check, a **number means enforce that limit** and **`null` means
skip that one comparison**. Use bare `null`, not the string `"null"` or zero.
The snippets below show selected entries: edit those entries in your existing
JSON file and keep the rest of the file. These examples do not change firmware
checks or make missing/invalid telemetry acceptable.

**Example 1: stop RC message gaps from triggering the Python RC timeout**

```json
{
  "rc_timeout_s": null
}
```

With `0.5`, an RC sample older than half a second loses authorization. With
`null`, the Python gate does not expire that sample: after a valid LOW-to-HIGH
transition, the last HIGH remains authorized even if RC messages stop. Receiving
LOW or invalid RC input still stops the maneuver. Starting with HIGH alone still
does not start it. Other checks can still stop the maneuver.

For the ground-test command that uses `--no-straight-flight`, the JSON is ignored.
Add the override to that command instead:

```bash
ros2 run ros2_sid run_mission_HITL.py --interface dds \
  --no-straight-flight --rc-timeout null --wait-timeout 180 \
  --command-log "$RUN/ground-commands.csv"
```

**Example 2: skip airspeed settling, but keep the airspeed envelope**

```json
{
  "airspeed_m_s": 23,
  "airspeed_tolerance_m_s": null,
  "min_airspeed_m_s": 20,
  "max_airspeed_m_s": 30
}
```

The commanded target is still 23 m/s. With tolerance `2`, airspeed must be
between 21 and 25 m/s while settling. With tolerance `null`, an airspeed of
26 m/s no longer blocks settling on that criterion. All other settling checks
and dwell still apply. Once READY/ACTIVE, airspeed below 20 or above 30 m/s
still triggers an envelope stop.

**Example 3: skip all three configurable airspeed checks**

```json
{
  "airspeed_m_s": 23,
  "airspeed_tolerance_m_s": null,
  "min_airspeed_m_s": null,
  "max_airspeed_m_s": null
}
```

The target remains 23 m/s, but neither airspeed settling error nor the lower/upper
airspeed limits block the maneuver. Airspeed telemetry must still be present,
fresh, and finite. The separate groundspeed requirement also remains. Do not set
`airspeed_m_s` itself to `null`: it is a command target, not an optional check.

**Example 4: skip bank settling, but keep the bank envelope**

```json
{
  "bank_tolerance_deg": null,
  "max_bank_deg": 35
}
```

With tolerance `5`, a 10-degree bank prevents settling. With tolerance `null`,
it no longer blocks settling on bank angle. A bank above 35 degrees still stops
the maneuver in READY/ACTIVE. Setting `max_bank_deg` to `null` as well would skip
that envelope check too. The two settings apply at different stages.

**Example 5: remove the coordinator's duration cap**

```json
{
  "max_duration_s": null
}
```

With `30`, the coordinator rejects a planned 45-second maneuver before setup.
With `null`, that particular duration check accepts it. The experiment still ends
at its own configured duration; this does not create an unlimited maneuver.

To restore any check, replace `null` with its numeric limit and restart the
mission. Keep required settings such as `telemetry_timeout_s`, `state_timeout_s`,
`setup_timeout_s`, `recovery_timeout_s`, and `dwell_s` numeric. `course_deg: null`
is the exception to the skip-check meaning: it captures the current course as
the command target. There is no `armed: null` option.

## Start the connections

The recommended setup runs both **mavlink-router and MAVROS inside the ROS
container**. A separate agent container handles USB DDS:

```text
Cube ACM0 (MAVLink) → ROS container: mavlink-router → UDP 14552 → MAVROS
                                                 → UDP 14550 → MAVProxy / GCS
Cube ACM1 (DDS)     → serial DDS agent container → /ap/* ROS topics
```

Use the dedicated `docker-compose.hitl.yaml` below. It maps ACM0 into the ROS
container and ACM1 into the agent; the PC SITL Compose file remains UDP-only.
The HITL startup builds the ROS interfaces/publisher without requiring a PC SITL
firmware binary. The Cube build currently waits 120 seconds before starting DDS.

### 1. Host terminal: start the HITL containers

First finish any running experiment and recovery. Stop any existing `run_soh.sh`,
serial router, or standalone serial DDS agent that owns these USB ports. Do not
run the PC SITL stack alongside this setup in domain 0. If migrating from the
previous walkthrough, stop its services **after the experiment is finished**:

```bash
cd /path/to/ros2_trajectory_docker
docker compose -f docker-compose.sid.yaml stop
# Only if the standalone agent from the installation guide exists:
docker stop cube-soh-agent
```

Check the Cube is attached and identify its ports:

```bash
ls -l /dev/serial/by-id/ /dev/ttyACM*
export SID_MAVLINK_DEVICE=/dev/ttyACM0
export SID_DDS_DEVICE=/dev/ttyACM1
docker compose -f docker-compose.hitl.yaml up -d --wait --wait-timeout 900
docker compose -f docker-compose.hitl.yaml logs --tail 30 agent
docker compose -f docker-compose.hitl.yaml exec ros bash
```

If the ACM numbers change, set the variables to the correct device paths before
starting Compose. You can use the Cube's `/dev/serial/by-id/...-if00` path for
MAVLink and `...-if02` for DDS. Inside the containers they are consistently named
ACM0 and ACM1. After a USB disconnect/reboot that changes the mapped device,
reattach to WSL and recreate the containers when no experiment is running.

### Terminal A: launch router and MAVROS

```bash
source /opt/ros/humble/setup.bash
source /tmp/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/work/ros2_sid/ros2_sid/config/surface_excitation/fastdds_udp.xml
source /work/tools/sid/use_run.sh --new interactive_hitl
export SHARE="$(ros2 pkg prefix --share ros2_sid)"

bash /work/useful_shell_scripts/run_live.sh --detach \
  --session HitlMavros \
  --serial-device /dev/ttyACM0 --baud 115200 \
  --fcu-url 'udp://:14552@127.0.0.1' \
  --mavros-config "$SHARE/config/surface_excitation/mavros_straight.yaml"
```

`use_run.sh` creates the run directory once and prints `RUN`. Reuse that same
run in the recording and mission terminals. If a run already exists, use
`--latest interactive_hitl_` instead of `--new interactive_hitl`.

This starts a `router` tmux window owning ACM0 and a `mavros` window listening on
UDP 14552. The YAML enables navigation and `rc_io` for the kill switch. It does
not launch a mission publisher or arm the aircraft. Keep `--with-drone` off so
the straight-flight coordinator is the only navigation-target publisher. No host MAVProxy process is
needed; **do not also run `run_soh.sh` on ACM0**.

The pattern is similar to `run_live_drone2.sh`, but this single-Cube setup uses
`/mavros` and the default vehicle system ID 1, as expected by the HITL mission.
Do not copy the drone-2 `/uav2`, system-ID-2 or port-14555 settings into this
workflow. Its router script does not forward to 14555 by default, and its
message-interval calls use a different namespace from its MAVROS node.

To inspect either process:

```bash
tmux attach -t HitlMavros
# Or print recent output without attaching:
tmux capture-pane -pt HitlMavros:router
tmux capture-pane -pt HitlMavros:mavros
```

Press **Ctrl+B, then D** to detach and leave the processes running. If the session
already exists, attach rather than starting a duplicate. To restart after a
configuration update, first finish the experiment/recovery, then:

```bash
tmux kill-session -t HitlMavros
# Rerun the run_live.sh command above to restart both router and MAVROS.
```

### GCS terminal: connect MAVProxy to the router

The serial-router launch above now also forwards MAVLink to **127.0.0.1:14550**
by default. MAVROS keeps its separate port 14552. In a **WSL host terminal** with
MAVProxy installed, run:

```bash
mavproxy.py --master=udpin:127.0.0.1:14550 --console --map
```

Leave it running alongside the containers. This connects MAVProxy to the router,
not directly to ACM0. For a text-only console, omit `--console --map`. Look for
a vehicle heartbeat and the expected aircraft before using GCS commands.
The container uses host networking, so this localhost endpoint is on the WSL
host. Do not run `run_soh.sh` as well: it would compete for ACM0.

For a GCS on **Windows or another computer**, set its reachable IP when launching
the container router. Add this option to the `run_live.sh` command above:

```bash
--gcs-endpoint 192.0.2.10:14550
```

That line is an option to append, not a separate command. Replace the example IP
with the GCS computer's actual reachable address; allow inbound UDP 14550 there.
This replaces the default localhost GCS destination. On that computer, configure
the GCS to listen on UDP 14550; MAVProxy can use
`mavproxy.py --master=udpin:0.0.0.0:14550 --console --map`.
Do not point the GCS at MAVROS's port 14552.

If the router is already running, finish any experiment/recovery before restarting
`HitlMavros` with the new option. Keep GCS guidance commands idle while the mission
coordinator owns the straight-flight targets; normal operator mode changes still
take precedence. Routing details follow the
[mavlink-router endpoint documentation](https://github.com/mavlink-router/mavlink-router).

### Check the connections

In the sourced ROS container shell:

```bash
ros2 service call /mavros/set_stream_rate mavros_msgs/srv/StreamRate \
  '{stream_id: 6, message_rate: 10, on_off: true}'
ros2 service call /mavros/set_stream_rate mavros_msgs/srv/StreamRate \
  '{stream_id: 3, message_rate: 10, on_off: true}'
ros2 topic echo /mavros/state --once
ros2 topic echo /mavros/rc/in --once
ros2 topic echo /ap/sid/status --once
ros2 topic hz /mavros/local_position/velocity_local
# Ctrl+C stops this rate display only.
ros2 topic hz /mavros/global_position/rel_alt
# Ctrl+C stops this rate display only.
ros2 topic hz /mavros/rc/in
# Ctrl+C stops this rate display only.
```

Confirm MAVROS reports `connected: true`, with position/velocity and RC updates
near 10 Hz. Reissue the stream requests after reconnecting if necessary. Before
starting a simulated airborne maneuver, confirm `mode: GUIDED` and establish
normal simulated flight. Excitation does not itself require `armed: true`. Physical ground tests may remain disarmed with `SX_GND_TEST=1`. The `hz` command
runs continuously; press Ctrl+C after checking its rate. A topic that does not
produce data needs troubleshooting before starting a test.

Before enabling the test, check `FLTMODE_CH` and `RC8_OPTION` in the flight
controller. **Channel 8 must not also be the flight-mode channel or another
assigned auxiliary function.** Plane commonly uses channel 8 for flight modes.
Assign mode selection to a different receiver channel, or explicitly select a
different `--rc-channel` for the experiment; do not remove your ability to change
flight mode. The automated simulator test uses `FLTMODE_CH=5`, `RC8_OPTION=0`;
these are test assignments, not a prescription for your transmitter.

Check that array element 7 follows your physical switch on both sides of 1550.
Do not infer channel mapping from the transmitter label.

Keep the Cube disarmed and `SX_ENABLE=0` for the initial switch/telemetry checks.
For simulated airborne experiments, keep `SIM_OH_MASK=0` and
`SIM_OH_RELAY_MSK=0`; configure and fly the onboard simulation to stable airborne
GUIDED with normal arming/failsafe checks. Enable `SX_ENABLE` only for the intended
simulation experiment. This guide does not authorize physical flight testing.

## Record and select maneuvers

Use the same workflow as SITL. Keep the router, MAVROS and DDS agent running.
The differences are the **HITL entrypoint** and the **RC switch starts/stops the
test**. You do not need `run_hitl.sh` or an experiment JSON file.

### Prepare the container shells

From your checkout, open a ROS container shell in each terminal you need:

```bash
docker compose -f docker-compose.hitl.yaml exec ros bash
```

Inside each shell, load the environment:

```bash
source /opt/ros/humble/setup.bash
source /tmp/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/work/ros2_sid/ros2_sid/config/surface_excitation/fastdds_udp.xml
```

In both **Terminal B** and **Terminal C**, select the run created in Terminal A:

```bash
source /work/tools/sid/use_run.sh --latest interactive_hitl_
```

Check that all terminals print the same `RUN`. To select a specific run, use
`source /work/tools/sid/use_run.sh --path /work/sid_results/YOUR_RUN_DIRECTORY`.
Do not launch MAVROS again or create another run. `RUN` is a shell variable;
it is not shared automatically.

### Terminal C: record all ROS topics

```bash
ros2 run ros2_sid record_surface_excitation.sh "$RUN/bag_straight"
```

Use an existing all-topic recorder if one is already running. Do not start
duplicate recorders for the same experiment. The script adds an index if the bag
path already exists; retain the actual printed path for plotting. Keep this
terminal recording through settling, excitation, disable and recovery. It captures
DDS commands/outputs, MAVROS telemetry, RC input and `/sid/straight_flight` events.

### Terminal B: select and start a maneuver

Keep the RC switch **LOW**, then run:

```bash
ros2 run ros2_sid run_mission_HITL.py --interface dds \
  --wait-timeout 180 --command-log "$RUN/straight-commands.csv"
```

This uses the default mission library and installed `straight_flight_sitl.json`,
just like SITL. Review that configuration's altitude, airspeed and settling
settings before testing; `--wait-timeout` does not set the settling dwell.
No RC arguments are needed: the defaults are MAVROS channel **8** (index **7**),
with **1550 µs** as the run/stop threshold.

The HITL console reports what it is doing:

- `WAITING`: RC is missing/stale, LOW, needs a LOW-to-HIGH transition, or the
  aircraft clock/firmware authorization is not ready.
- `START REQUESTED` / `PREPARING`: the switch requested a test; flight setup is
  still in progress.
- `STARTING`: commands are being sent, awaiting firmware acknowledgement.
- `RUNNING: firmware confirmed ACTIVE excitation`: the firmware confirmed it is
  applying excitation. Sending a start request alone does not produce this message.

Messages appear when the status changes, rather than on every sample. Typing
`start` in HITL prints a reminder to use the RC switch. Completion and stop/recovery
messages remain visible.

At the mission prompt:

```text
list
select 0
```

The shipped library contains:

| Selection | Maneuver |
|---|---|
| `select 0` | Roll chirp / frequency sweep |
| `select 1` | Roll multisine |
| `select 2` | Pitch chirp / frequency sweep |
| `select 3` | Pitch multisine |
| `select 4` | Yaw chirp / frequency sweep |
| `select 5` | Yaw multisine |

Use the actual `list` output if you have changed the library. Then:

1. Have the onboard simulation airborne in GUIDED with excitation authorized,
   as described in the setup above. Keep physical output masks disabled.
2. Raise the RC switch **HIGH**. The aircraft settles into straight flight, then
   starts the maneuver. Do not type `start`; HITL uses the RC switch.
3. Let the maneuver finish, or switch **LOW** to stop it.
4. Keep LOW and wait for zero excitation, the temporary LOITER transition, and
   final GUIDED target confirmation. If an operator
   mode change or failsafe takes over, that takes priority.
5. Type `quit` to close the publisher. Keep recording until recovery is complete.

To run another maneuver, confirm recovery finished in GUIDED (select GUIDED
manually if using legacy LOITER recovery), keep RC LOW,
launch the publisher again and select the next maneuver. **Use a different
command-log filename for each launch** (for example `roll-multisine-commands.csv`):
the direct ROS command overwrites an existing command CSV. The bag recorder can
continue across all maneuvers. One maneuver per command CSV makes later checks
straightforward.

`stop`, `quit` and Ctrl+C also stop the experiment. This does not disarm the
aircraft or stop its motor. Keep MAVROS and DDS running until recovery finishes.

### Stop recording and plot

After the publisher finishes recovery, press **Ctrl+C in Terminal C** and wait
for the recorder to exit. In either sourced shell with the same `RUN`:

```bash
export BAG="$RUN/bag_straight"  # use the actual printed path, including any suffix
ros2 bag info "$BAG"
ros2 run ros2_sid plot_surface_excitation.py "$BAG" \
  --output "$RUN/hitl-plots" --straight-flight \
  --command-log "$RUN/straight-commands.csv"
python3 -m ros2_sid.surface_rc_analysis "$BAG" --output "$RUN/hitl-plots"
```

For additional maneuvers in the same bag, repeat `--command-log` for each saved
CSV. Only supplied command logs receive command-to-output validation.

Plots and reports are saved under `$RUN/hitl-plots` and are also visible in the
host checkout's `sid_results` directory. They include commanded/applied excitation,
servo PWM, aircraft response, straight-flight recovery and RC-switch timing.
Check `report.json` and `straight_report.json`: plots may still be written when
validation fails. RC timing includes transport latency; PWM is commanded output,
not measured surface position. Retain the bag, CSVs and corresponding Cube `.BIN`
logs so you can replot later without connecting the aircraft.

### Optional overrides

The simple command above is the normal workflow. Only add these when needed:

| Argument | Purpose |
|---|---|
| `--experiment /work/path/to/experiment.json` | Use one custom experiment instead of the library |
| `--straight-flight /work/path/to/settings.json` | Override the installed navigation settings |
| `--rc-source dds` | Read `/ap/rcin` instead of `/mavros/rc/in`; navigation still needs MAVROS |
| `--rc-channel N` | Use a different one-based RC channel |
| `--rc-timeout 0.5` / `--rc-timeout null` | Override JSON RC freshness with a timeout in seconds or disable sample-age checks |
| `--no-straight-flight` | Disable navigation coordination and automatic navigation recovery |

Do not combine `--straight-flight` with `--no-straight-flight`. If you choose DDS
RC, also add `--rc-source dds` to the RC plotting command. See
[custom waveforms](custom_surface_waveforms.md) for generating your own sweeps
and multisines. Existing convenience helpers remain available but are optional.

## Automated tests and evidence

`surface_sitl.py --hitl-rc` runs the real HITL entrypoint against **PC SITL**, using
RC overrides only on the harness-owned simulator. It does not connect to the
Cube. Use an isolated agent/domain/port (for example 42/2022), then:

```bash
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 --hitl-rc --rate 50 \
  --dds-domain 42 --dds-port 2022 --output sid_results/hitl_rc_six
# A separate run tests RC-low during ACTIVE:
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 --hitl-rc --rate 50 \
  --dds-domain 42 --dds-port 2022 --only-experiment chirp_roll \
  --exit-active rc-low --output sid_results/hitl_rc_stop
```

Use fresh output directories. Simulator-backed tests cannot verify your receiver,
physical switch wiring or the Cube's timing under load; perform disarmed physical
switch checks separately before any onboard simulation experiment.

See [executed verification and plots](hitl_verification.md) for the recorded
simulator-backed HITL-entrypoint results and their limitations.

## Angle-of-attack and sideslip recording

The same recorder and plotting commands now include estimated α/β when the
updated firmware publishes them. The identification configuration requests 50 Hz;
the report measures the achieved rate. No additional maneuver arguments are needed.
See [air-data recording and rate selection](air_data_recording.md).
