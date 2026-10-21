#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_BAG_DIRECTORY" >&2
  exit 2
fi
config="$(ros2 pkg prefix --share ros2_sid)/config/surface_excitation/bag_qos.yaml"
exec ros2 bag record -s sqlite3 -o "$1" --qos-profile-overrides-path "$config" \
  /ap/time /ap/sid/control_surface /ap/sid/status /ap/sid/output
