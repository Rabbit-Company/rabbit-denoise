#!/usr/bin/env bash
# Adds known AV1 film grain to the clean clips, so a denoiser can be scored against the
# picture without grain.
#
#   scripts/make-grained.sh
#
# Each clean clip is encoded once with SVT-AV1 at a very high quality. Its decode without
# grain is the reference. regrain then writes a grain table into a copy of that encode and
# dav1d draws the grain, so reference and noisy video differ by exactly the grain.
#
# Writes work/grained/<clip>/reference.mkv and work/grained/<clip>/<profile>.mkv, both
# lossless 10 bit FFV1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REGRAIN="${REGRAIN:-$ROOT/../regrain/target/release/regrain}"
OUT="$ROOT/work/grained"

# name:regrain generate options, or name:@table to use a table from grain/.
# The film-* tables are what `regrain diff` measured on the Xiph film clips.
PROFILES=(
  "fine2:--strength 2 --size fine"
  "medium4:--strength 4 --size medium --chroma 1.5"
  "coarse6:--strength 6 --size coarse --chroma 2"
  "film-ducks:@film-ducks.tbl"
  "film-park:@film-park.tbl"
)

# Decodes an AV1 file with dav1d into lossless 10 bit FFV1: ffv1 INPUT OUTPUT
ffv1() {
  ffmpeg -v error -nostdin -y -c:v libdav1d -i "$1" -pix_fmt yuv420p10le -c:v ffv1 -level 3 -slices 16 -g 1 "$2"
}

for clip in "$ROOT"/media/clean/*.mkv; do
  name="$(basename "$clip" .mkv)"
  dir="$OUT/$name"
  mkdir -p "$dir"
  if [[ ! -f "$dir/reference.mkv" ]]; then
    echo "$name: encoding reference"
    ffmpeg -v error -nostdin -y -i "$clip" -pix_fmt yuv420p10le -c:v libsvtav1 -preset 6 -crf 4 \
      -svtav1-params "film-grain=0" "$dir/base.ivf" 2>/dev/null
    ffv1 "$dir/base.ivf" "$dir/reference.mkv"
  fi
  for profile in "${PROFILES[@]}"; do
    pname="${profile%%:*}"
    options="${profile#*:}"
    [[ -f "$dir/$pname.mkv" ]] && continue
    echo "$name: adding $pname grain"
    if [[ "$options" == @* ]]; then
      cp "$ROOT/grain/${options#@}" "$dir/$pname.tbl"
    else
      # shellcheck disable=SC2086
      "$REGRAIN" generate $options -o "$dir/$pname.tbl" -y -q
    fi
    "$REGRAIN" apply "$dir/base.ivf" -t "$dir/$pname.tbl" -o "$dir/$pname.ivf" -y -q
    ffv1 "$dir/$pname.ivf" "$dir/$pname.mkv"
    rm -f "$dir/$pname.ivf"
  done
done
