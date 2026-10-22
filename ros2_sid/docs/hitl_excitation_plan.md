# HITL DDS excitation implementation and verification plan

1. Compare legacy and DDS entrypoints. Preserve legacy RC channel 8, threshold
   1550 microseconds, high-to-run and low-to-stop behavior. Keep the legacy
   trajectory implementation intact. Verify exact threshold and missing channels.
2. Reuse the shared DDS waveform, CSV library, straight-flight coordination,
   repeated disable and zero-output-before-LOITER recovery. Default to legacy
   `/mavros/rc/in`; allow explicit `/ap/rcin`. Test startup high, low/high rearm,
   stale input, invalid input, a low/high pulse between timer ticks, and console
   start rejection. Keep the existing fresh low/high requirement on DDS startup.
3. Enable MAVROS `rc_io` alongside navigation plugins. Build ROS and test in a
   separate ROS domain. Add a simulator-only harness mode that runs the actual
   HITL entrypoint with simulated RC input; never inject test RC into hardware.
4. Record six individual-axis chirp/multisine experiments and an RC-low abort.
   Compare recorded applied PWM with calibrated expectations; verify navigation
   recovery. Generate command/applied/aircraft-response plots from rosbag2.
5. Publish the manual HITL/SoH walkthrough and retain logs, bags, reports and
   figures. Label simulator-backed HITL-entrypoint tests separately from tests
   on the physical Cube. Never infer physical RC wiring verification from them.

The RC switch stops the **excitation experiment**, not aircraft power or motors.
Normal stabilization continues. RC-low initiates bounded disable/return and
navigation recovery; operator/failsafe mode changes take precedence. Publisher
failure is covered by firmware command timeout, but automatic navigation recovery
requires the publisher and MAVROS link to remain operational.

## Findings incorporated during execution

- Legacy defaults differed from the earlier DDS gate (channel 7 / 1800). DDS now
  defaults to MAVROS channel 8 / 1550, with explicit overrides available.
- The existing MAVROS plugin allowlist omitted `rc_io`; it is now enabled.
- Low/high callbacks between timer ticks must latch a stop, and an RC delivery
  gap must invalidate the previous low edge even if a new packet arrives before
  the timer checks freshness. Both cases now have regression coverage.
- Plane's default `FLTMODE_CH=8` conflicts with the legacy experiment channel.
  The first isolated run withheld excitation after the switch selected MANUAL.
  The harness now uses `FLTMODE_CH=5`, `RC8_OPTION=0`. The manual requires checking
  assignments on the actual receiver; no Cube parameters were changed here.
