#!/usr/bin/env bash
# Measures denoising speed the way a whole episode sees it: the average from start to
# end over a long clip, startup included.
#
#   scripts/speed.sh [DENOISER...]
#
# Runs every denoiser (default "rabbit") over every clip in media/speed, made by
#
#   CLIPS=media/speed-clips.tsv CLIP_DIR=media/speed scripts/cut-real.sh 30
#
# Each run should take 30 seconds or more. Prints one line per run and appends it to
# results/speed.jsonl, labelled with SPEED_LABEL when set.
#
# A warm laptop runs the next test slower, so before every run the script waits until the
# GPU is below COOL_C degrees (default 55).
set -euo pipefail
# Decimal points for printf and bc whatever the locale.
export LC_ALL=C

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/work/speed.log"
mkdir -p "$ROOT/work" "$ROOT/results"
# shellcheck source=denoisers.sh
source "$ROOT/scripts/denoisers.sh"
(cd "$ROOT" && cargo build --release -q -p rabbit-denoise)

NAMES=("$@")
[[ ${#NAMES[@]} -eq 0 ]] && NAMES=(rabbit)
LABEL="${SPEED_LABEL:-}"

# Waits, up to 10 minutes, until the first GPU that reports a temperature is cool enough.
cool_down() {
  local sensor
  sensor=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
  [[ -z "$sensor" ]] && return
  for _ in $(seq 1 120); do
    (($(cat "$sensor") / 1000 < ${COOL_C:-55})) && return
    sleep 5
  done
}

# Denoisers that tune themselves on first use do so before anything is timed.
first=$(ls "$ROOT"/media/speed/*.mkv | head -1)
for denoiser in "${NAMES[@]}"; do
  warm_up_denoiser "$denoiser" "$first"
done

for clip in "$ROOT"/media/speed/*.mkv; do
  name=$(basename "$clip" .mkv)
  frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$clip" | cut -d, -f1)
  for denoiser in "${NAMES[@]}"; do
    cool_down
    start=$(date +%s.%N)
    ffmpeg -v error -nostdin -i "$clip" -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe - | denoise "$denoiser" >/dev/null
    end=$(date +%s.%N)
    seconds=$(echo "$end - $start" | bc -l)
    fps=$(echo "$frames / $seconds" | bc -l)
    printf '%-10s %-8s %5d frames %7.1f s %6.2f fps\n' "$denoiser" "$name" "$frames" "$seconds" "$fps"
    jq -nc --arg d "$denoiser" --arg c "$name" --arg l "$LABEL" --argjson f "$frames" --argjson s "$seconds" \
      '{denoiser: $d, clip: $c, label: $l, frames: $f, seconds: $s, fps: ($f / $s), date: (now | todate)}' >>"$ROOT/results/speed.jsonl"
  done
done
