# rabbit-denoise

A film grain denoiser for video: it removes the grain and keeps the picture. Use it before any encoder, such as AV1, HEVC or H.264, to spend fewer bits on grain. With AV1 grain synthesis, [regrain](https://github.com/Rabbit-Company/regrain) can then put the grain back.

It runs on the GPU at about 9.2 fps at 1080p on a Radeon 780M.

## Installing

Download an archive for your system from the [releases page](https://github.com/Rabbit-Company/rabbit-denoise/releases). Each one holds the `rabbit-denoise` binary, this README, the changelog and the licence.

| Archive           | System                                                                              |
| ----------------- | ----------------------------------------------------------------------------------- |
| `linux-x86_64-v3` | Linux on Intel Haswell (2013), AMD Excavator (2015) or newer. Pick this when unsure |
| `linux-x86_64-v4` | Linux on CPUs with AVX-512, such as AMD Zen 4 or newer and Intel Xeon               |
| `linux-x86_64-v2` | Linux on older x86-64 CPUs that the v3 build does not run on                        |
| `linux-aarch64`   | Linux on 64 bit ARM                                                                 |
| `macos-aarch64`   | macOS on Apple silicon (M1 and newer)                                               |
| `macos-x86_64`    | macOS on Intel                                                                      |
| `windows-x86_64`  | Windows on x86-64                                                                   |

The v2, v3 and v4 Linux builds differ only in the CPU instructions they may use. To see which levels your CPU supports:

```sh
/lib64/ld-linux-x86-64.so.2 --help | grep supported
```

A build that is too new for the CPU crashes with an illegal instruction error and does no other harm, so trying v3 first is safe.

On Linux and macOS:

```sh
tar -xzf rabbit-denoise-v0.1.5-linux-x86_64-v3.tar.gz
sudo install rabbit-denoise-v0.1.5-linux-x86_64-v3/rabbit-denoise /usr/local/bin/
```

The macOS binaries are not signed. If macOS refuses to open it, clear the quarantine flag once:

```sh
xattr -d com.apple.quarantine /usr/local/bin/rabbit-denoise
```

On Windows, unpack the zip and put `rabbit-denoise.exe` in a folder on the `PATH`.

`SHA256SUMS` on the releases page lists the checksum of every archive:

```sh
sha256sum --check --ignore-missing SHA256SUMS
```

On macOS, use `shasum -a 256 --check --ignore-missing SHA256SUMS`.

To install the manual and completions:

```sh
rabbit-denoise manual > /usr/local/share/man/man1/rabbit-denoise.1
rabbit-denoise completions bash > /etc/bash_completion.d/rabbit-denoise
```

## Building

```sh
cargo build --release
```

The binary is `target/release/rabbit-denoise`. Building for the local CPU with `-C target-cpu=native` made no measurable difference, because the GPU does most of the work.

## Requirements

- A GPU with Vulkan on Linux and Windows, Metal on macOS or DX12 on Windows, and its driver, such as Mesa on Linux. Without one, `--device auto` runs on the CPU, about 20 times slower, and says so on standard error and in the `start` event.
- `ffmpeg` and `ffprobe` for video files other than y4m and for `.mkv` output. Pass `--ffmpeg <path>` when `ffmpeg` is not on the `PATH`. `ffprobe` is expected next to it. y4m in and out needs neither.

## Usage

```sh
rabbit-denoise denoise source.mkv -o denoised.mkv
rabbit-denoise denoise source.mkv -o denoised.y4m
ffmpeg -i source.mkv -strict -1 -f yuv4mpegpipe - | rabbit-denoise denoise -o denoised.y4m
```

`denoise` reads any video file ffmpeg can decode, such as MKV or MP4 with AV1, HEVC or H.264. y4m files and y4m on standard input are read directly. Other files are decoded by running `ffmpeg`, so ffmpeg and ffprobe must be installed for them, as for regrain and retone. Only the first video track is denoised.

An output named `.mkv` holds the denoised video as lossless FFV1, with the source's colour tags and HDR10 mastering display and light level metadata. The source's audio, subtitles, attachments such as fonts, chapters and tags are copied unchanged, unless `--video-only` is given. MP4 subtitles become SubRip, which Matroska holds. Writing MKV also needs ffmpeg. Any other output is y4m, which carries no colour tags.

Two things cannot be kept. Dolby Vision and HDR10+ metadata do not fit in FFV1, so the output is plain HDR10 and a warning says so. A source with a variable frame rate is refused for an MKV with its other tracks, as the denoised video comes back at one frame rate and would drift from the audio. `--video-only` writes it anyway. FFV1 files are large, about 20 to 40 GB for 24 minutes of 1080p at 10 bit.

Interlaced video is refused with `unsupported_input`, as the filter would treat its two fields as one picture. Deinterlace it first, for example with `ffmpeg -i INPUT -vf bwdif=mode=send_frame -strict -1 -f yuv4mpegpipe - | rabbit-denoise denoise -o OUTPUT`. Video flagged interlaced whose frames are progressive can go through ffmpeg with `-vf setfield=prog` instead. The output may not be the input file.

The output keeps the input's bit depth and chroma layout: 8, 10 and 12 bit 4:2:0, 4:2:2 and 4:4:4, and 8 and 12 bit greyscale, at any size. Other pixel formats are converted by ffmpeg to the nearest of these, rounding the bit depth up. y4m input in other colorspaces, such as 10 bit greyscale, stops with `unsupported_input`.

Like regrain and retone, every command takes `--progress <text|json|none>` and `-q`. `completions` and `manual` print a shell completion script and the manual page.

The defaults are the best quality measured on the benchmark below. The dB figures below come from the known grain clips. Every number has a range, shown in `--help`, and values outside it are refused. The whole number ranges are what the GPU runs.

### Input and output

| Option            | Default  | Meaning                                                            |
| ----------------- | -------- | ------------------------------------------------------------------ |
| `[INPUT]`         | `-`      | Video to denoise, or `-` for y4m on standard input                 |
| `-o, --output`    | `-`      | Output `.mkv` or `.y4m`, or `-` for y4m on standard output         |
| `--video-only`    | off      | With an `.mkv` output, leave out the source's other tracks         |
| `-y, --overwrite` | off      | Replace an existing output file                                    |
| `--ffmpeg <PATH>` | `ffmpeg` | ffmpeg for inputs other than y4m, with ffprobe expected next to it |

A file is written under a temporary name and renamed once the run succeeds. Devices and pipes, such as `/dev/null`, are written directly.

### Strength

| Option               | Default | Range     | Meaning                                                              |
| -------------------- | ------- | --------- | -------------------------------------------------------------------- |
| `--sigma-scale <X>`  | 1.0     | 0.25 to 4 | Multiplies the measured grain deviation, every plane and both stages |
| `--chroma-scale <X>` | 1.0     | 0.25 to 4 | Multiplies the chroma planes' grain deviation on top of that         |
| `--lambda <X>`       | 2.7     | 1 to 6    | Grain deviations a coefficient must clear to survive the first stage |
| `--basic`            | off     |           | Return the first stage's result and skip the second                  |

The denoiser measures the grain itself, so these options are corrections that most videos do not need. `--sigma-scale` above 1 removes more grain and more fine detail. Below 1 it leaves some grain in, and regrain then puts back less. `--chroma-scale` does the same for colour alone. `--lambda` is the first stage's threshold. A coefficient survives only where it is this many times the grain at its frequency and brightness. The second stage has no threshold and weighs every coefficient by how much of it is picture.

`--basic` is about twice as fast and about 2 dB worse.

### Matching

| Option                     | Default | Range    | Meaning                                                                  |
| -------------------------- | ------- | -------- | ------------------------------------------------------------------------ |
| `--step <N>`               | 3       | 1 to 6   | Distance between reference patches, in samples                           |
| `--temporal-radius <N>`    | 3       | 0 to 16  | Frames on each side of the current one that matches come from            |
| `--search-radius <N>`      | 5       | 0 to 7   | How far matches are searched for in the patch's own frame                |
| `--matches-per-frame <N>`  | 2       | 1 to 8   | Matches taken into a group from each frame, besides the reference        |
| `--min-shift <N>`          | 4       | 1 to 8   | How far apart two matches from one frame must be                         |
| `--group-size <N>`         | 16      | 2 to 16  | Most patches in a group, the reference included                          |
| `--match-limit <X>`        | 2.0     | 0.5 to 8 | How much a first stage match may differ from its reference               |
| `--wiener-match-limit <X>` | 0.5     | 0.1 to 4 | The same for the second stage, which matches on the first stage's result |

Every patch on a grid `--step` samples apart is filtered together with the patches most like it, from its own frame and `--temporal-radius` frames on each side. These options trade speed against quality:

- `--step 4` is about 30 per cent faster for about 0.3 dB less.
- `--temporal-radius 2` is about 20 per cent faster for about 0.4 dB less on film grain. 5 gains about 0.1 dB over 3. 0 matches within the frame only and removes much less grain.
- `--search-radius 5` scores within 0.1 dB of 7 and is faster.

The others shape the groups. Matches from other frames carry different grain, so averaging them removes it. Patches of one frame a sample or two apart share most of their grain, so they must be `--min-shift` apart. `--group-size` 12 and 10 are no faster than 16 and lose a little colour.

`--match-limit` is in multiples of the difference grain alone makes between two patches, so it follows the grain. Higher blurs, lower removes less grain. Anything from 1.5 to 3 scores within 0.1 dB of 2. The second stage matches on the first stage's result and needs the tighter `--wiener-match-limit`.

### Motion

| Option                       | Default  | Range   | Meaning                                                                   |
| ---------------------------- | -------- | ------- | ------------------------------------------------------------------------- |
| `--follow-radius <N>`        | 1        | 0 to 7  | Search radius in neighbouring frames, around a patch's track and its move |
| `--wiener-follow-radius <N>` | as above | 0 to 7  | The same for the second stage alone                                       |
| `--motion-range <N>`         | 16       | 0 to 32 | How far the motion fields look for a block's move between two frames      |
| `--no-motion`                | off      |         | Follow patches without motion fields, searching around their track alone  |

A patch is followed from frame to frame as far as `--temporal-radius` goes. A motion field gives every 16x16 block its move between two frames, measured on the frames averaged 2x2. The next frame is searched `--follow-radius` around the patch's track and around where the field moves it. Raise `--motion-range` for very fast motion. It is rounded down to an even number.

`--no-motion` searches around the track alone, with `--follow-radius` 3 unless given. It scores 0.15 to 0.25 dB lower, is 16 to 20 per cent slower and is kept for comparison.

### Grain measurement

| Option                    | Default | Meaning                                                                    |
| ------------------------- | ------- | -------------------------------------------------------------------------- |
| `--no-single-frame-cap`   | off     | Use the grain measured between frames without capping it by single frames  |
| `--no-luma-guided-chroma` | off     | Match chroma blocks on chroma rather than on luma when measuring its grain |

Both switch off a safeguard, for comparison. Grain measured between frames reads high where the picture moves, and grain measured in single frames reads high on texture, so the model takes the smaller of the two. Matching chroma on chroma reads its grain 10 to 20 per cent low and costs 0.1 to 1.9 dB of colour.

### Device

| Option         | Default | Meaning                                                                                                 |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `--device <D>` | auto    | `auto` uses the GPU when one opens and the CPU otherwise. `gpu` fails without a GPU. `cpu` uses the CPU |

The CPU is the reference the GPU is tested against, and about 20 times slower.

### Diagnostics

| Option                | Meaning                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--show-noise`        | Print the grain measured for every frame to standard error                                                    |
| `--show-spectrum <N>` | With `--show-noise`, also print the 8x8 luma grain spectrum of brightness band N, 0 to 15 from black to white |
| `--timing`            | Print where the time went at the end, with the time of each GPU stage where the GPU reports it                |

### Progress and errors

With `--progress json`, rabbit-denoise writes one JSON object per line to standard output, or to standard error when the video goes to standard output. Every object has an `event` field:

| Event      | When                                 | Fields                                                                                        |
| ---------- | ------------------------------------ | --------------------------------------------------------------------------------------------- |
| `start`    | After the y4m header was read        | `width`, `height`, `bit_depth`, `colorspace`, `device`, `total`                               |
| `progress` | About twice a second                 | `frames` written, `total`, `percent`, `fps` since the start, `elapsed_seconds`, `eta_seconds` |
| `summary`  | When the command succeeded           | `frames`, `seconds`, `fps`                                                                    |
| `warning`  | When something of the source is lost | `message`                                                                                     |
| `error`    | When the command failed              | `code`, `message`                                                                             |

`device` names the GPU and its backend, or `CPU`. `total` is the frame count the container stores or its duration gives, or the size of a y4m file, and `null` on standard input, where `percent` and `eta_seconds` are `null` too. Errors exit with status 1, or 130 when cancelled, and carry one of these codes:

| Code                 | Meaning                                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `unsupported_input`  | A y4m colorspace rabbit-denoise does not read, a file without a video track, or a variable frame rate for an MKV with other tracks |
| `unsupported_output` | An output name other than `.mkv`, `.y4m` or none                                                                                   |
| `bad_input`          | The input is not a video, or a frame could not be read                                                                             |
| `ffmpeg_missing`     | ffmpeg or ffprobe could not be run                                                                                                 |
| `ffmpeg_failed`      | ffmpeg failed while decoding the input or writing the MKV                                                                          |
| `no_frames`          | The input held no frames                                                                                                           |
| `gpu_unavailable`    | `--device gpu` and no GPU opened, or the settings need more than the GPU filter supports                                           |
| `gpu_failed`         | The GPU stopped responding or ran out of memory while denoising                                                                    |
| `io_error`           | A file could not be read, written or moved, the output exists without `-y`, or it is the input                                     |
| `cancelled`          | Ctrl+C or SIGTERM. A second one ends the process at once                                                                           |
| `error`              | Anything else                                                                                                                      |

A cancelled or failed run removes its unfinished output. When Ctrl+C stops both ffmpeg and rabbit-denoise, the end of ffmpeg's stream is not taken for the end of the video. The text of `message` may change between versions. The codes and field names will not.

## How it works

Two stages in the style of V-BM3D, on 8x8 patches. Every reference patch is grouped with matching patches from its own frame and the frames around it. The group is transformed with an 8x8 DCT and a DCT along the group, shrunk and averaged back into the picture. The first stage keeps a coefficient only where it clears `lambda` grain deviations. The second matches on the first stage's result and applies a Wiener gain.

What sets it apart is how it measures the grain:

- **Per frequency and brightness.** The model holds the grain variance of each of the 64 DCT frequencies for 16 brightness bands, the same things AV1 describes with its AR filter and scaling function.
- **Over time.** A block found again in the next frame differs from it by grain alone. On known grain this lands within a few per cent of the truth, where single frames read 20 to 130 per cent high on texture. Pairs across a scene cut are dropped, and the first frames of a scene are redone once a pair is measured.
- **Only fine grain.** Film grain is strongest at middle and high frequencies, so the model caps every frequency at 6 times the high frequency grain, which is also the most AV1 grain synthesis can draw again. Coarse, blotchy noise beyond that stays, as removing it would mostly take fine dark texture with it.
- **Two measurements that check each other.** Motion reads high between frames and texture reads high in single frames, so the model takes the smaller of the two.
- **Chroma matched on luma.** Chroma's grain is measured with blocks matched on the luma first stage result.
- **Motion fields.** A patch is followed by searching around its track and around where the motion field moves it. That is 0.15 to 0.25 dB better and 16 to 20 per cent faster than tracking alone. Chroma follows the luma field.
- **Independent grain in every group.** Each frame gives its two best matches, at least 4 pixels apart, since shifted patches share their grain.
- **Match limits from the grain.** A match may differ from its reference by twice what grain alone makes.

## GPU

The filter runs on the GPU through wgpu, on Vulkan, Metal or DX12. It finds the same groups as the CPU version, and the outputs are about 70 dB apart. Two GPU runs of the same video also differ, by about 100 dB, as the GPU adds up overlapping patches in a different order each time. The difference is far below anything visible, but the output is not the same byte for byte. `cargo test` checks that whenever a GPU opens. On the Radeon 780M it denoises 1080p at about 9.2 fps over a 30 second clip, against about 0.4 fps on 16 CPU threads.

- `find_groups` gives each 4x4 block of reference patches one workgroup, with the searched region in workgroup memory and four threads per patch.
- `filter_groups` gives each 4x4 block of reference patches one workgroup, which filters their groups in turn with one thread per coefficient. The patches are added up in workgroup memory, and each pixel goes to the accumulators once. Adding every patch with float atomics took 75 per cent of the time on a Radeon RX 9060 XT.
- `half_plane` and `motion` measure the motion fields on luma, and chroma follows them.
- Each entry point runs for all three planes in one pass, and only the motion fields' entry points may write the fields, so the planes run at the same time.
- Frames, the planes averaged 2x2 and the motion fields stay on the GPU, so each is made once.
- `pair_blocks` and `residual_blocks` find the blocks the grain is measured on between frames and in what the first stage removed, in the same submission as the first stage. Its result goes into the ring on the GPU, and the CPU only takes the medians of the blocks. They match the CPU's blocks to within 0.1 per cent.
- A first stage is queued behind the second stage in flight and the next second stage behind it, so the GPU always has work. The CPU takes the medians meanwhile and measures single frames, and threads read and write frames.
- Release builds leave out the shader's bounds checks, which cost about a third of the time. `gpu::supports` keeps every index in range.
- Lists kept in registers are written out in full, since an indexed array spills to slow scratch memory.
- Each shader is built for its step and search radius, so it uses as little workgroup memory as it can.
- `--timing` prints where the time went, with GPU timestamps per entry point and stage.
- `MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaderstats` prints every shader's registers, workgroup memory and waves on RADV.

## Benchmark

```sh
scripts/fetch-media.sh          # Xiph clips, first 240 frames each, as FFV1 in media/
scripts/make-grained.sh         # known grain added to the clean clips, in work/grained/
scripts/bench.sh known          # denoise them, score against the clean picture
scripts/bench.sh film           # denoise real film, encode, regrain, measure bitrate
scripts/cut-real.sh             # short clips of your own videos, listed in media/real-clips.tsv
scripts/bench.sh real           # the same pipeline on them
scripts/frames.sh real CLIP 24  # the same crop of the source and every output, side by side
scripts/report.py               # tables from results/*.jsonl
scripts/dev.sh CLIP START GRAIN # quick check of rabbit-denoise on a 24 frame cut
scripts/speed.sh                # average speed over 30 second clips, start to end
```

Needs ffmpeg with libsvtav1 and libdav1d, a built [regrain](https://github.com/Rabbit-Company/regrain) (`REGRAIN=<path>`, by default next to this repository), jq, bc, Rust and Python 3.

### Suites

**known** has exact answers. Three clean clips from the Xiph test media collection, two CGI and one flat 2D animation, are encoded at CRF 4, and the decode without grain is the reference. dav1d then draws known grain with three profiles: `fine2` (deviation 2), `medium4` (4, film-like) and `coarse6` (6, clumped). `rdbench` reports:

- `psnr` of the denoised picture against the clean one, next to `psnr_noisy` for the input.
- `grain_removed`: the share of the true grain the denoiser took out. 1 is ideal.
- `collateral_psnr`: what the denoiser took out that was not grain. Higher is better.
- SSIMULACRA2 mean and 5% low.

`film-ducks` and `film-park` use grain tables `regrain diff` measured on Xiph film clips, kept in `grain/`. The grain is still drawn by the AV1 model, which favours a denoiser built on that model.

**film** runs the real workflow on four 65 mm film clips from the Xiph SVT set. Each is denoised, encoded with SVT-AV1 at CRF 20, 28 and 36, measured with `regrain diff` and regrained. The `none` row encodes the source directly. Real film has no clean picture, so this suite cannot score fidelity. It reports:

- Bitrate against the direct encode at the same CRF.
- `texture_ratio` from `rdbench residual`, which flags gross smearing but does not rank.
- SSIMULACRA2 of the regrained decode against the source, for reference only. AV1 draws new random grain, so the score does not rank denoisers.

**real** runs the same pipeline on short clips of your own videos, listed in `media/real-clips.tsv` with a name, path and start in seconds, separated by tabs. The file stays out of git. `scripts/cut-real.sh` keeps 2 seconds of each as lossless FFV1.

`KNOWN_START=24 KNOWN_FRAMES=48 scripts/bench.sh known` scores 48 frame cuts instead of whole clips.

Fidelity comes from the known suite. On real film, compare frames by eye with `scripts/frames.sh`.

### Speed

`scripts/speed.sh` measures the average speed over clips long enough that every run takes 30 seconds or more. List them in `media/speed-clips.tsv` and cut them with `CLIPS=media/speed-clips.tsv CLIP_DIR=media/speed scripts/cut-real.sh 30`. A warm laptop runs slower, so the script waits until the GPU is below 55 °C before every run. Runs still vary by about 10 per cent.

### Adding a denoiser

Other denoisers to compare with go in `scripts/denoisers.local.sh`, which stays out of git. Define `local_denoise NAME`, which reads 10 bit y4m on standard input and writes it on standard output, and optionally `warm_up_denoiser NAME CLIP` for a denoiser that tunes itself on first use. Then run `scripts/bench.sh known <name>`, or set `COMPARE=<name>` for `scripts/dev.sh`.
