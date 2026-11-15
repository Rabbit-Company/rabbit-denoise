#!/usr/bin/env bash
# Puts the same crop of the source and of every denoiser's output side by side, for
# judging grain and detail by eye.
#
#   scripts/frames.sh SUITE CLIP FRAME [W:H:X:Y] [DENOISER...]
#
#   scripts/frames.sh real CLIP 24 480:270:700:400 rabbit rabbit-dev
#
# Reads the outputs `bench.sh film` or `bench.sh real` leave in work/out. The crop
# defaults to 480x270 from the centre, shown at twice its size. Writes
# work/frames/CLIP-FRAME.png.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE="$1" CLIP="$2" FRAME="$3"
CROP="${4:-480:270:720:405}"
shift 3
[[ $# -gt 0 ]] && shift
NAMES=("$@")
[[ ${#NAMES[@]} -eq 0 ]] && NAMES=(rabbit)

source="$ROOT/media/$SUITE/$CLIP.mkv"
transfer=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_transfer -of csv=p=0 "$source")
tonemap=""
if [[ "$transfer" == smpte2084 ]]; then
  tonemap="zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
fi
mkdir -p "$ROOT/work/frames"
out="$ROOT/work/frames/$CLIP-$FRAME.png"

inputs=(-i "$source")
labels=(source)
for name in "${NAMES[@]}"; do
  inputs+=(-i "$ROOT/work/out/$SUITE/$name/$CLIP/denoised.mkv")
  labels+=("$name")
done

filters=""
for i in "${!labels[@]}"; do
  filters+="[$i:v]select=eq(n\\,$FRAME),${tonemap}crop=$CROP,scale=iw*2:ih*2:flags=neighbor,format=rgb24,"
  filters+="drawtext=text='${labels[$i]}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6[v$i];"
done
stack=""
for i in "${!labels[@]}"; do stack+="[v$i]"; done
filters+="${stack}vstack=inputs=${#labels[@]}"

ffmpeg -v error -nostdin -y "${inputs[@]}" -filter_complex "$filters" -frames:v 1 "$out"
echo "$out"
