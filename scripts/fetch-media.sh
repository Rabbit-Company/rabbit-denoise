#!/usr/bin/env bash
# Downloads the benchmark clips from the Xiph test media collection and keeps the first
# FRAMES frames of each as lossless FFV1, so the full files never touch the disk.
#
#   scripts/fetch-media.sh [FRAMES]
#
# film/  65 mm film scans with real grain. No clean version exists.
# clean/ CGI and 2D animation without grain. Known grain is added to these.
set -euo pipefail

FRAMES="${1:-240}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEDIA="$ROOT/media"
BASE="https://media.xiph.org/video/derf/y4m"

FILM=(ducks_take_off_1080p50 crowd_run_1080p50 old_town_cross_1080p50 park_joy_1080p50)
# Each clean clip starts at a frame past its titles, where there is picture to denoise.
CLEAN=(sintel_trailer_2k_1080p24:240 elephants_dream_1080p24:1500 sita_sings_the_blues_1080p24:2400)

fetch() {
  local url="$1" out="$2" skip="$3" unpack="$4"
  if [[ -f "$out" ]]; then
    echo "have $out"
    return
  fi
  echo "fetching $url"
  local tmp="$out.part.mkv"
  # curl and the unpacker stop with a broken pipe once ffmpeg has read enough frames, so
  # only ffmpeg's status counts.
  set +o pipefail
  curl -sSfL "$url" 2>/dev/null | $unpack 2>/dev/null | ffmpeg -v error -nostdin -y -f yuv4mpegpipe -i - \
    -vf "trim=start_frame=$skip,setpts=PTS-STARTPTS" -frames:v "$FRAMES" \
    -c:v ffv1 -level 3 -slices 16 -g 1 "$tmp"
  local status=${PIPESTATUS[2]}
  set -o pipefail
  if [[ "$status" != 0 ]]; then
    echo "ffmpeg failed on $url" >&2
    rm -f "$tmp"
    return 1
  fi
  local got
  got=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$tmp")
  if [[ "$got" != "$FRAMES" ]]; then
    echo "$url gave $got frames, expected $FRAMES" >&2
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$out"
}

mkdir -p "$MEDIA/film" "$MEDIA/clean"
for name in "${FILM[@]}"; do
  fetch "$BASE/$name.y4m" "$MEDIA/film/$name.mkv" 0 cat
done
for entry in "${CLEAN[@]}"; do
  name="${entry%%:*}"
  skip="${entry##*:}"
  fetch "$BASE/$name.y4m.xz" "$MEDIA/clean/$name.mkv" "$skip" "xz -dc"
done
