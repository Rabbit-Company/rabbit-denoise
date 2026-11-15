#!/usr/bin/env bash
# Cuts short clips from your own videos for the `real` benchmark suite.
#
#   scripts/cut-real.sh [SECONDS]
#
# Reads media/real-clips.tsv, one clip per line: NAME, TAB, path, TAB, start in seconds.
# Each clip becomes media/real/NAME.mkv, SECONDS seconds (default 2) of 10 bit FFV1.
# CLIPS and CLIP_DIR choose another list and folder. Pictures larger than 1080p keep their
# centre 1920x1080 instead of being scaled.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SECONDS_LONG="${1:-2}"
LIST="${CLIPS:-$ROOT/media/real-clips.tsv}"
DIR="${CLIP_DIR:-$ROOT/media/real}"
mkdir -p "$DIR"
[[ -f "$LIST" ]] || { echo "no $LIST" >&2; exit 1; }

while IFS=$'\t' read -r name path start; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  out="$DIR/$name.mkv"
  if [[ -f "$out" ]]; then
    echo "have $out"
    continue
  fi
  echo "cutting $name"
  ffmpeg -v error -nostdin -y -ss "$start" -i "$path" -map 0:v:0 -t "$SECONDS_LONG" \
    -vf "crop=w='min(iw,1920)':h='min(ih,1080)'" -pix_fmt yuv420p10le \
    -c:v ffv1 -level 3 -slices 16 -g 1 "$out.part.mkv"
  mv "$out.part.mkv" "$out"
done <"$LIST"
