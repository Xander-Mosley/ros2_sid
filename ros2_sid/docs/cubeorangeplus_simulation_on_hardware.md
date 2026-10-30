# Cube Orange Plus: Simulation on Hardware with custom DDS

This builds **ArduPlane** from this workspace's `ardupilot_sid` fork, including
`/ap/sid/control_surface`, `/ap/sid/status`, and `/ap/sid/output`. Simulation on
Hardware (SoH) runs the Plane physics model and simulated sensors on the Cube;
it does not require a separate `sim_vehicle.py` process. It is distinct from
standard firmware using real sensors, and from external-physics HITL.

Reference: [ArduPilot Simulation on Hardware](https://ardupilot.org/dev/docs/sim-on-hardware.html).

## USB and backups

Run the build/upload on the WSL host. Set the checkout directory to your own path:

```bash
cd /path/to/ros2_trajectory_docker
export WORKSPACE="$PWD"
ls -l /dev/serial/by-id/ /dev/ttyACM*
```

Use **CubeOrangePlus**, board ID **1063**; CubeOrange is a different target.
This board has two USB interfaces. Its hardware definition maps interface `if00`
(usually `/dev/ttyACM0`) to MAVLink, and `if02` (usually `/dev/ttyACM1`) to
`SERIAL6`. Prefer `/dev/serial/by-id/` paths; ACM numbers can change after reboot.
Close Mission Planner, MAVProxy and other serial clients before upload.

Save all parameters, missions, fences and rally points first. MAVProxy provides
`param save`, `wp save`, `fence save` and `rally save` commands (load the respective
modules if needed). Keep real-aircraft and simulation backups separate. Firmware
flashing does not reliably erase existing parameter settings.

Physical outputs remain disabled in this guide: `SIM_OH_MASK=0`,
`SIM_OH_RELAY_MSK=0`, and `SX_ENABLE=0`. Do not enable physical actuators while
propellers or hazardous mechanisms are attached. No simulated flight is required
for installation verification.

## Build environment

Use the installed ArduPilot cross compiler and Python 3.12 environment. The
Python 3.14 environment encountered a DroneCAN multiprocessing failure here.
For example, with `uv` installed:

```bash
uv venv --python /usr/bin/python3.12 /tmp/cube-build-venv
uv pip install --python /tmp/cube-build-venv/bin/python \
  'empy==3.3.4' pexpect pyyaml future lxml numpy pyserial pymavlink dronecan setuptools
source /tmp/cube-build-venv/bin/activate
export PATH="/opt/gcc-arm-none-eabi-10-2020-q4-major/bin:$WORKSPACE/Micro-XRCE-DDS-Gen/scripts:$PATH"
arm-none-eabi-g++ --version
microxrceddsgen -version
```

The compiler location is an example: use your installed ArduPilot toolchain.
The checked-out generator needs its built JAR and Java; the existing DDS build
setup supplies these. Initialize required ArduPilot submodules on a fresh clone.

```bash
./tools/sid/build_cubeorangeplus.sh soh
```

The wrapper invokes:

```bash
cd "$WORKSPACE/ardupilot_sid"
python3 Tools/scripts/sitl-on-hardware/sitl-on-hw.py \
  --board CubeOrangePlus --vehicle plane --simclass Plane --frame plane \
  --defaults ../tools/sid/soh/cubeorangeplus.parm --enable-dds
```

The upstream helper adds simulated sensor defaults and `SIM_ENABLED=1`.
Our additional defaults select DDS on `SERIAL6`, domain 0, and disable excitation
and physical outputs. The wrapper preserves the APJ, build log, hardware defines,
source revision/diff and checksum in a timestamped `sid_results` directory.
It does not flash automatically.

## Upload and initialize simulation settings

Use the actual artifact and first USB interface:

```bash
cd "$WORKSPACE/ardupilot_sid"
python3 Tools/scripts/uploader.py --port /dev/ttyACM0 \
  "$WORKSPACE/sid_results/<build-directory>/arduplane-soh.apj"
```

The uploader checks the board ID and verifies programmed flash. Do not use
`--force` to bypass board mismatch. WSL USB passthrough may detach when the Cube
enters its bootloader: if upload waits for the device, reattach the board using
`usbipd list` / `usbipd attach --wsl --busid <BUSID>` from Windows, then retry.
Keep the Cube powered during flashing.

When switching between standard and simulation firmware, reset parameters only
**after saving the backups**. In MAVProxy on the first interface:

```text
param set FORMAT_VERSION 0
reboot
```

Reconnect and fetch parameters. Confirm the embedded defaults were loaded:

```text
param show AHRS_EKF_TYPE
param show GPS1_TYPE
param show SIM_OH_MASK
param show SIM_OH_RELAY_MSK
param show SERIAL6_PROTOCOL
param show DDS_ENABLE
param show DDS_DOMAIN_ID
param show SX_ENABLE
```

Expected: `AHRS_EKF_TYPE=10`, `GPS1_TYPE=100`, both output masks 0,
`SERIAL6_PROTOCOL=45`, `DDS_ENABLE=1`, `DDS_DOMAIN_ID=0`, `SX_ENABLE=0`.
The simulation helper also configures simulation timing and sensor calibration.
Do not load an entire physical-aircraft backup into the simulation.

## Start DDS and MAVROS

The UDP agent from `docker-compose.sid.yaml` does not open the Cube's serial DDS
interface. Start a **serial** Micro XRCE-DDS Agent on the second interface. For a
host-installed agent:

```bash
MicroXRCEAgent serial --dev /dev/ttyACM1 -b 115200
```

After [building the shared image](../README_surface_excitation.md#first-time-docker-image-setup),
use a separate terminal (only if the HITL launcher is not already running an agent):

```bash
docker run --rm --name cube-soh-agent --network host \
  --device=/dev/ttyACM1:/dev/ttyACM1 \
  ros2_trajectory:compact \
  bash -lc 'ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM1 -b 115200 -v4'
```

This checkout currently delays DDS initialization by **120 seconds after boot**
(two 60-second waits in `AP_DDS_Client::main_loop`). Allow more than two minutes
before concluding that the transport failed. This is firmware startup latency,
not the straight-flight settling dwell.

Do not run two agents on that serial port. If a PC SITL is also running in domain
0, stop it before checking these topics so vehicle messages cannot be confused.
Enter the ROS container, source ROS and the built workspace, then inspect data:

```bash
docker compose -f docker-compose.sid.yaml exec ros bash
source /opt/ros/humble/setup.bash
source /tmp/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 topic list | grep '/ap/'
ros2 topic echo /ap/time --once
ros2 topic echo /ap/sid/status --once
```

MAVLink stays on ACM0. To route it to MAVROS, use the existing launcher in an
environment that has MAVROS, tmux and mavlink-routerd installed and can access ACM0:

```bash
./useful_shell_scripts/run_live.sh --serial-device /dev/ttyACM0 --baud 115200
```

The UDP-only Compose service deliberately has no device mappings; host USB ports
are not automatically available inside it. Use a host MAVLink router to forward
to the container's MAVROS UDP port, or explicitly map ACM0 if running the router
inside a container. Do not run `run_soh.sh` and a serial router on ACM0 together.
The existing `run_soh.sh` uses MAVProxy to forward to ports 14550–14553, including
MAVROS on 14552, and writes a simulated starting location.

After installation checks, follow the [manual excitation guide](manual_surface_excitation_sitl.md)
for recording, maneuver selection and plotting. Skip its PC SITL launch. Enabling
excitation and arming the simulated aircraft are separate test operations; they
are not part of flashing. Straight-flight coordination still requires MAVROS.

## Standard ArduPlane firmware, or parameters only

To build **standard ArduPlane with this fork's custom DDS**, without simulated
sensors or physics:

```bash
cd "$WORKSPACE"
./tools/sid/build_cubeorangeplus.sh plane
cd ardupilot_sid
python3 Tools/scripts/uploader.py --port /dev/ttyACM0 \
  "$WORKSPACE/sid_results/<build-directory>/arduplane-plane.apj"
```

This uses `./waf configure --board CubeOrangePlus --enable-dds` and `./waf plane`;
it does not apply the SoH hardware definition or simulation defaults. When
returning from SoH, reset parameters, reboot, and restore the appropriate
real-aircraft configuration/calibrations. Do not restore the simulation sensor
settings. Set `DDS_ENABLE=1`, `DDS_DOMAIN_ID=0`, `SERIAL6_PROTOCOL=45` and reboot
if DDS on the second USB interface is wanted. Leave `SX_ENABLE=0` until the
intended testing setup has been independently checked.

A `.parm` file is configuration, **not firmware**. To change only parameters on
an already compatible firmware, use MAVProxy `param load /path/to/file.parm`,
then fetch/check values and reboot for settings that require it. Loading
parameters cannot add DDS or turn a normal build into Simulation on Hardware.
An official standard firmware download will not contain this fork's custom
control-surface DDS messages.

## Installation evidence

See `sid_results/cubeorangeplus_soh/` for the current installation's original
configuration backup, build log, firmware and verification records. A successful
flash and disarmed DDS check do not demonstrate airborne excitation behavior on
this hardware. The existing SITL verification report applies to its documented
simulator runs, not automatically to this Cube installation.

For the installation performed on 2026-10-22, flash verification and disarmed
custom DDS telemetry checks succeeded. The serial agent `cube-soh-agent` was
left running; inspect it with `docker logs cube-soh-agent`. The unused PC-SITL
UDP agent was stopped to remove a stale duplicate `/ap` participant. To return
to PC SITL later, stop/remove the serial agent with `docker rm -f cube-soh-agent`
and start the UDP service with `docker compose -f docker-compose.sid.yaml up -d agent`.
The existing board configuration was already SoH, so it was preserved rather than
reset. DDS and output-disable parameters were explicitly set and checked.
