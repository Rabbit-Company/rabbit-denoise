# Denoisers the benchmark can run, sourced by bench.sh, speed.sh and dev.sh.
#
#   denoise NAME < input.y4m > output.y4m
#
# Each reads a 10 bit y4m stream on standard input and writes one on standard output.
# Other denoisers to compare with go in scripts/denoisers.local.sh, which stays out of git.
# It may define `local_denoise NAME`, called for names not listed here, and
# `warm_up_denoiser NAME CLIP`, run before anything is timed.

denoise() {
  case "$1" in
    rabbit) "$ROOT/target/release/rabbit-denoise" denoise -q 2>>"$LOG" ;;
    # The working build under another name. RABBIT_DEV_ARGS adds options.
    rabbit-dev) "$ROOT/target/release/rabbit-denoise" denoise -q ${RABBIT_DEV_ARGS:-} 2>>"$LOG" ;;
    *)
      if declare -F local_denoise >/dev/null; then
        local_denoise "$1"
      else
        echo "unknown denoiser $1" >&2
        return 1
      fi
      ;;
  esac
}

# Lets a denoiser that tunes itself on first use do so before it is timed.
warm_up_denoiser() {
  :
}

if [[ -f "$ROOT/scripts/denoisers.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/scripts/denoisers.local.sh"
fi
