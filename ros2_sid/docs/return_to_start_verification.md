# Return-to-start recovery verification

The shared DDS coordinator saves a fresh global fix and relative altitude before
straight-flight setup. It removes excitation, requests LOITER at that saved point,
and checks the firmware target echo. RC stop, console stop/quit, completion and
normal signal shutdown share this recovery path. Operator/failsafe authority
still takes precedence. Confirmation is a navigation handoff, not an arrival claim.

## Reproduce in isolated PC simulation

Build from the workspace's `ardupilot_sid` directory:

```bash
./waf configure --board sitl --enable-dds
./waf plane -j4
```

Build the ROS packages and launch a UDP DDS agent as in the
[SITL manual](manual_surface_excitation_sitl.md). The verification below used
`surface-excitation-sitl-ros-1`, ROS domain 42 and agent UDP port 2022. These are
isolated test settings, not the manuals' normal domain-0 defaults.

From the workspace root:

```bash
python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 \
  --output sid_results/return_start_sitl_verified_20261022 \
  --interactive-library --straight-flight --verify-start-return \
  --only-experiment chirp_roll --dds-domain 42 --dds-port 2022

python3 ardupilot_sid/Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py \
  --container surface-excitation-sitl-ros-1 \
  --output sid_results/return_start_hitl_verified_20261022 \
  --hitl-rc --exit-active rc-low --verify-start-return \
  --only-experiment chirp_roll --dds-domain 42 --dds-port 2022
```

Use fresh output directories when repeating these commands. Both use PC SITL;
the second exercises the HITL entrypoint and simulated RC channel 8. Neither
connects to a physical Cube. The return check sets an 80 m loiter radius in its
isolated parameters and requires 15 continuous seconds within 130 m of the saved
center while in LOITER. Firmware authorization and normal arming remain enabled.

## Evidence and limits

The first attempt (`sid_results/return_start_sitl_20261022`) caught a location
parser rejection. Excitation returned to zero and recovery reported unconfirmed.
The Plane handler was corrected to reuse the existing reposition location decoder
without changing legacy zero-coordinate LOITER commands. That failed attempt is
retained separately.

The ROS package regression suite passed 57 tests. The DDS-enabled SITL firmware
built successfully with Python 3.12, empy 3.3.4 and setuptools 80.10.2 in a temporary
build environment. Host Python 3.14 initially failed DroneCAN dependency generation.

Cube/Simulation-on-Hardware firmware has **not** been flashed or tested for this
revision. Rebuild/reflash it before using the new recovery behavior on a Cube.
Older firmware ignores the requested LOITER location; a mismatched target cannot
pass the coordinator's confirmation check. Communication loss or process kill
cannot guarantee a navigation handoff.

## Recorded results

- **Normal SITL completion:** passed. The roll chirp completed, applied excitation
  reached zero, and the saved LOITER target was confirmed. The aircraft returned
  from approximately 514 m away and remained within the 130 m acceptance boundary
  for 15 seconds. Bag validation, zero-before-turn, target matching, mode-change
  and existing safety probes passed.
- Evidence: `sid_results/return_start_sitl_verified_20261022/` contains the all-topic
  rosbag, DataFlash log, command CSV, coordinator events, return-track samples,
  plots, build output, unit-test output and `execution.json`.

![SITL normal completion: return to saved start](../../../sid_results/return_start_sitl_verified_20261022/chirp_roll-return-track.png)

The return-track figure uses the harness's recorded MAVLink GPS samples. The
all-topic rosbag also retains MAVROS global position and target echoes. Existing
`plot_surface_excitation.py --straight-flight` revalidates the coordinator's saved
target and zero-output handoff from the bag. `plot_return.py` beside the samples
recreates the return figure from `chirp_roll-return-track.json` without commanding
a vehicle.


### HITL entrypoint with RC-low stop (PC simulation)

Passed: the switch interrupted active excitation, zero perturbation was observed,
and the saved LOITER center was confirmed. The recorded aircraft returned from
approximately 303 m away and remained within 130 m of the center for 15 seconds.
The RC gate still uses channel 8 (array index 7). Mode-change and safety probes
passed. This verifies the HITL software path using simulated RC, not a physical
receiver or Cube.

Evidence is retained in `sid_results/return_start_hitl_verified_20261022/`.
Initial bag validation exposed out-of-order storage/readback of coordinator events
around clock synchronization. Sorting by the publisher's embedded `monotonic_s`
fixed the offline interpretation; a regression test covers this case. Reanalysis
of the unchanged bag passed, including DataFlash comparison and zero-before-turn.
`execution.before_reanalysis.json` preserves the initial failure;
`execution.json`, `reanalysis.txt`, and `plots/report.json` record the corrected
validation. The final installed ROS package rebuilt successfully and all 57
package tests passed again.

![HITL entrypoint RC stop: simulated return to saved start](../../../sid_results/return_start_hitl_verified_20261022/chirp_roll-return-track.png)

## Changed implementation

- `ArduPlane/GCS_Mavlink.cpp`: explicit-coordinate LOITER handling, destination
  validation, and target installation after mode entry.
- `ros2_sid/straight_flight.py`: fresh start capture, location command, and target
  echo confirmation shared by SITL and HITL.
- `config/surface_excitation/mavros_straight.yaml`: navigation-target echo plugin.
- `ros2_sid/straight_analysis.py`: saved-target validation and monotonic event order.
- `tests/test_straight_flight.py`: recovery, freshness, ownership, and ordering tests.
- `Tools/ros2/ardupilot_dds_tests/scripts/surface_sitl.py`: observed return-track check.
- Both manual guides and `straight_flight_plan.md`: setup and recovery instructions.

No mission CLI flags were added: this is the default DDS straight-flight recovery
for both entrypoints. No physical-aircraft deployment or flight was performed.
