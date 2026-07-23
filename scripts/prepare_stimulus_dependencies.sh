#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
mkdir -p "$output_dir"
url="https://raw.githubusercontent.com/opencv/opencv/4.12.0/data/haarcascades/haarcascade_frontalface_default.xml"
output="$output_dir/haarcascade_frontalface_default.xml"

curl --fail --location --retry 3 --output "$output" "$url"
sha256sum "$output"
printf 'source_url=%s\n' "$url" > "$output_dir/source.txt"

