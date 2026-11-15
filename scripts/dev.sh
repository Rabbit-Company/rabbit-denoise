#!/usr/bin/env bash
# Quick check of rabbit-denoise on a short cut.
#
#   scripts/dev.sh CLIP START PROFILE [rabbit-denoise options...]
#
#   scripts/dev.sh CLIP 24 medium4 --step 6
#
# Cuts FRAMES (default 24) frames from work/grained/CLIP/PROFILE.mkv from frame START,
# with the reference cut the same way, and prints PSNR Y noisy and out, grain removed,
# collateral PSNR, chroma PSNR, SSIMULACRA2 and fps. With COMPARE=NAME, it also scores
# NAME's output from `bench.sh known` on the same frames.
set -euo pipefail
# Decimal points for printf and bc whatever the locale.
export LC_ALL=C

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLIP="$1" START="$2" PROFILE="$3"
shift 3
FRAMES="${FRAMES:-24}"
CUT="$ROOT/work/dev/$CLIP-$START-$FRAMES"
mkdir -p "$CUT"
(cd "$ROOT" && cargo build --release -q -p rabbit-denoise -p rdbench)

cut() {
  local in="$1" out="$2"
  [[ -f "$out" ]] && return
  ffmpeg -v error -nostdin -y -i "$in" -vf "trim=start_frame=$START,setpts=PTS-STARTPTS" -frames:v "$FRAMES" \
    -pix_fmt yuv420p10le -c:v ffv1 -level 3 -slices 16 "$out"
}

cut "$ROOT/work/grained/$CLIP/reference.mkv" "$CUT/reference.mkv"
cut "$ROOT/work/grained/$CLIP/$PROFILE.mkv" "$CUT/$PROFILE.mkv"

score() {
  local label="$1" out="$2" fps="$3"
  "$ROOT/target/release/rdbench" compare --reference "$CUT/reference.mkv" --distorted "$out" --noisy "$CUT/$PROFILE.mkv" --ssim-every 4 \
    | jq -r --arg l "$label" --arg fps "$fps" \
      '"\($l)\tY \(.psnr_noisy.y*100|round/100) -> \(.psnr.y*100|round/100)\tremoved \(.grain_removed.y*1000|round/1000)\tcollateral \(.collateral_psnr.y*100|round/100)\tUV \(.psnr.u*100|round/100)/\(.psnr.v*100|round/100)\tSSIMU2 \(.ssimulacra2.mean*100|round/100)\t\($fps) fps"'
}

if [[ -n "${COMPARE:-}" ]]; then
  cut "$ROOT/work/out/known/$COMPARE/$CLIP/$PROFILE.mkv" "$CUT/$PROFILE.$COMPARE.mkv"
  score "$COMPARE" "$CUT/$PROFILE.$COMPARE.mkv" "-"
fi
start=$(date +%s.%N)
ffmpeg -v error -nostdin -i "$CUT/$PROFILE.mkv" -strict -1 -f yuv4mpegpipe - \
  | "$ROOT/target/release/rabbit-denoise" denoise -q "$@" > "$CUT/out.y4m"
end=$(date +%s.%N)
score rabbit "$CUT/out.y4m" "$(echo "$FRAMES / ($end - $start)" | bc -l | xargs printf '%.2f')"
