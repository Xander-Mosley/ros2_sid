#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--with-mavros" ) ]]; then
  echo "Usage: $0 NEW_BAG_DIRECTORY [--with-mavros]" >&2
  exit 2
fi
base="$1"
while [[ "$base" == */ && "$base" != / ]]; do base="${base%/}"; done
[[ -n "$base" && "$base" != / ]] || { echo 'Provide a bag directory name.' >&2; exit 2; }
output="$base"
index=1
while [[ -e "$output" || -L "$output" ]]; do
  output="${base}_${index}"
  index=$((index + 1))
done
printf 'Recording all ROS topics to: %s\n' "$output"
if [[ -n "${SID_BAG_PATH_FILE:-}" ]]; then
  printf '%s\n' "$output" > "$SID_BAG_PATH_FILE"
fi
config="$(ros2 pkg prefix --share ros2_sid)/config/surface_excitation/bag_qos.yaml"
# --with-mavros remains accepted for older launch commands; all topics are now
# recorded by default, including hidden topics and publishers discovered later.
exec ros2 bag record -s sqlite3 -o "$output" --qos-profile-overrides-path "$config" \
  --all --include-hidden-topics
