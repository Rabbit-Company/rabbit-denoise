#!/usr/bin/env bash
# Runs the denoiser benchmark.
#
#   scripts/bench.sh known [DENOISER...]   denoise the clips with known grain, score against the clean picture
#   scripts/bench.sh film  [DENOISER...]   denoise real film, encode, put the grain back, measure bitrate
#   scripts/bench.sh real  [DENOISER...]   the same on your own clips from scripts/cut-real.sh
#
# Without DENOISER, every one in $DENOISERS runs. Results go to results/*.jsonl and a
# rerun only does the missing work. scripts/report.py turns them into tables.
#
# Needs scripts/fetch-media.sh and, for `known`, scripts/make-grained.sh to have run.
#
# Settings, as environment variables:
#   DENOISERS  default "rabbit"
#   CRFS       film: SVT-AV1 CRFs, default "20 28 36"
#   PRESET     film: SVT-AV1 preset, default 6
#   SSIM_EVERY score SSIMULACRA2 on every Nth frame, default 2
#   KNOWN_FRAMES, KNOWN_START  known: score a cut of this many frames from this frame
#              instead of whole clips, written to results/known-START-FRAMES.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REGRAIN="${REGRAIN:-$ROOT/../regrain/target/release/regrain}"
RDBENCH="$ROOT/target/release/rdbench"
DENOISERS="${DENOISERS:-rabbit}"
CRFS="${CRFS:-20 28 36}"
PRESET="${PRESET:-6}"
SSIM_EVERY="${SSIM_EVERY:-2}"
OUT="$ROOT/work/out"
RESULTS="$ROOT/results"
LOG="$ROOT/work/bench.log"
# shellcheck source=denoisers.sh
source "$ROOT/scripts/denoisers.sh"

mkdir -p "$RESULTS" "$OUT"
(cd "$ROOT" && cargo build --release -q -p rdbench -p rabbit-denoise)

y4m() {
  ffmpeg -v error -nostdin -i "$1" -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe -
}

to_ffv1() {
  ffmpeg -v error -nostdin -y -f yuv4mpegpipe -i - -c:v ffv1 -level 3 -slices 16 -g 1 "$1"
}

frames() {
  ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$1" | cut -d, -f1
}

# Denoises IN into OUT and prints the seconds it took, decoding and FFV1 included.
run_denoiser() {
  local name="$1" in="$2" out="$3"
  mkdir -p "$(dirname "$out")"
  local start end
  start=$(date +%s.%N)
  y4m "$in" | denoise "$name" | to_ffv1 "$out.part.mkv"
  end=$(date +%s.%N)
  if [[ "$(frames "$out.part.mkv")" != "$(frames "$in")" ]]; then
    echo "$name dropped frames on $in" >&2
    rm -f "$out.part.mkv"
    return 1
  fi
  mv "$out.part.mkv" "$out"
  echo "$end - $start" | bc
}

# Lets denoisers that tune themselves do so before anything is timed.
warm_up() {
  local clip="$1"
  for name in $DENOISERS; do
    warm_up_denoiser "$name" "$clip"
  done
}

# Has this denoiser, clip and setting already been recorded?
recorded() {
  local file="$1" filter="$2"
  [[ -f "$file" ]] && jq -e -s "any(.[]; $filter)" "$file" >/dev/null
}

# Cuts KNOWN_FRAMES frames from KNOWN_START of a known grain clip, or gives the clip
# whole when KNOWN_FRAMES is unset.
known_input() {
  local in="$1" clip="$2"
  if [[ -z "${KNOWN_FRAMES:-}" ]]; then
    echo "$in"
    return
  fi
  local out
  out="$ROOT/work/cuts/$clip/${KNOWN_START:-0}-$KNOWN_FRAMES/$(basename "$in")"
  if [[ ! -f "$out" ]]; then
    mkdir -p "$(dirname "$out")"
    ffmpeg -v error -nostdin -y -i "$in" -vf "trim=start_frame=${KNOWN_START:-0},setpts=PTS-STARTPTS" -frames:v "$KNOWN_FRAMES" \
      -pix_fmt yuv420p10le -c:v ffv1 -level 3 -slices 16 "$out.part.mkv"
    mv "$out.part.mkv" "$out"
  fi
  echo "$out"
}

known() {
  local suffix=""
  [[ -n "${KNOWN_FRAMES:-}" ]] && suffix="-${KNOWN_START:-0}-$KNOWN_FRAMES"
  local file="$RESULTS/known$suffix.jsonl"
  local first
  first=$(ls "$ROOT"/work/grained/*/medium4.mkv | head -1)
  warm_up "$first"
  for dir in "$ROOT"/work/grained/*/; do
    local clip reference
    clip=$(basename "$dir")
    reference=$(known_input "$dir/reference.mkv" "$clip")
    for whole in "$dir"*.mkv; do
      local profile noisy
      profile=$(basename "$whole" .mkv)
      [[ "$profile" == reference ]] && continue
      noisy=$(known_input "$whole" "$clip")
      for name in $DENOISERS; do
        if recorded "$file" ".denoiser==\"$name\" and .clip==\"$clip\" and .profile==\"$profile\""; then
          continue
        fi
        echo "known: $name $clip $profile"
        local out="$OUT/known$suffix/$name/$clip/$profile.mkv" seconds
        seconds=$(run_denoiser "$name" "$noisy" "$out")
        "$RDBENCH" compare --reference "$reference" --distorted "$out" --noisy "$noisy" --ssim-every "$SSIM_EVERY" \
          | jq -c --arg d "$name" --arg c "$clip" --arg p "$profile" --argjson s "$seconds" \
            '{denoiser: $d, clip: $c, profile: $p, seconds: $s, fps: (.frames / $s)} + .' >>"$file"
      done
    done
  done
}

# Denoises every clip in media/SUITE, encodes it, puts the grain back and records the
# bitrate. film is the Xiph film scans, real the clips from scripts/cut-real.sh.
pipeline() {
  local suite="$1"
  local file="$RESULTS/$suite.jsonl"
  warm_up "$(ls "$ROOT"/media/"$suite"/*.mkv | head -1)"
  for source in "$ROOT"/media/"$suite"/*.mkv; do
    local clip rate count
    clip=$(basename "$source" .mkv)
    rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$source" | cut -d, -f1)
    count=$(frames "$source")
    # `none` encodes the source as it is, grain and all, as a baseline.
    for name in none $DENOISERS; do
      local dir="$OUT/$suite/$name/$clip" denoised table="" seconds=0
      mkdir -p "$dir"
      if [[ "$name" == none ]]; then
        denoised="$source"
      else
        denoised="$dir/denoised.mkv"
        if [[ ! -f "$denoised" ]]; then
          echo "$suite: denoising $clip with $name"
          seconds=$(run_denoiser "$name" "$source" "$denoised")
          echo "$seconds" >"$dir/seconds"
        fi
        seconds=$(cat "$dir/seconds")
        table="$dir/grain.tbl"
        if [[ ! -f "$table" ]]; then
          "$REGRAIN" diff "$source" "$denoised" -o "$table" -y -q
        fi
        if [[ ! -f "$dir/residual.json" ]]; then
          "$RDBENCH" residual --reference "$source" --distorted "$denoised" >"$dir/residual.json"
        fi
      fi
      local residual='null'
      [[ -f "$dir/residual.json" ]] && residual=$(cat "$dir/residual.json")
      for crf in $CRFS; do
        if recorded "$file" ".denoiser==\"$name\" and .clip==\"$clip\" and .crf==$crf"; then
          continue
        fi
        echo "$suite: $name $clip crf $crf"
        local encode="$dir/crf$crf.ivf" shown="$dir/crf$crf.ivf"
        if [[ ! -f "$encode" ]]; then
          ffmpeg -v error -nostdin -y -i "$denoised" -pix_fmt yuv420p10le -c:v libsvtav1 -preset "$PRESET" -crf "$crf" \
            -svtav1-params "film-grain=0" "$encode.part.ivf" 2>/dev/null
          mv "$encode.part.ivf" "$encode"
        fi
        if [[ -n "$table" ]]; then
          shown="$dir/crf$crf.grain.ivf"
          "$REGRAIN" apply "$encode" -t "$table" -o "$shown" -y -q
        fi
        local kbps
        kbps=$(echo "$(stat -c %s "$encode") * 8 / ($count / ($rate)) / 1000" | bc -l)
        "$RDBENCH" compare --reference "$source" --distorted "$shown" --ssim-every "$SSIM_EVERY" \
          | jq -c --arg d "$name" --arg c "$clip" --argjson crf "$crf" --argjson k "$kbps" --argjson s "$seconds" \
            --argjson residual "$residual" \
            '{denoiser: $d, clip: $c, crf: $crf, kbps: $k, denoise_fps: (if $s > 0 then .frames / $s else null end), residual: $residual} + .' >>"$file"
      done
    done
  done
}

suite="${1:-}"
shift || true
[[ $# -gt 0 ]] && DENOISERS="$*"
case "$suite" in
  known) known ;;
  film | real) pipeline "$suite" ;;
  *)
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
