# Changelog

Error codes and JSON field names stay compatible within a major version. Messages meant for people may change in any release. Before 1.0, a minor version may change the defaults and the options.

## 0.1.3 - 2026-11-16

### Changes

- The grain between frames and in what the first stage removed is measured on the GPU, in the same submission as the first stage. Only the medians are left to the CPU, so a slow CPU no longer holds the GPU back. At 1080p on a Radeon RX 9060 XT with an i7-8700K, the x86-64-v2 build goes from 10.6 to 19.1 fps on video without grain, and grainy video from about 18 to 20 fps. The GPU finds the same blocks as the CPU, to within 0.1 per cent.
- The first stage's result goes into the GPU's ring of frames on the GPU, so the second stage no longer uploads it.
- Without a GPU, what the first stage removed is measured on all threads.

## 0.1.2 - 2026-11-16

### Changes

- About twice as fast on video with little or no grain, from 7.0 to 14.6 fps at 1080p on a Radeon RX 9060 XT with an i7-8700K. Such video measures no grain between frames, so its frames waited to be redone and the GPU and CPU took turns. The second stage is now queued behind the first stage then too, and the output is the same.
- The block search that measures grain between frames is about 40 per cent faster on the CPU.

## 0.1.1 - 2026-11-15

### Changes

- About twice as fast on GPUs with much more compute than float atomic throughput, from 8.8 to 17.5 fps at 1080p on a Radeon RX 9060 XT. `filter_groups` adds up the patches of each 4x4 block of references in workgroup memory, so each pixel goes to the accumulators once. Adding every patch with float atomics took 75 per cent of the time there.
- The grain between frames and in the residuals is measured on all planes at once, and chroma scales the luma basic estimates once for each size.

## 0.1.0 - 2026-11-15

First release: a denoiser for the regrain workflow.

### Features

- `denoise` removes film grain and keeps the picture, before any encoder. With AV1 grain synthesis, regrain can put the grain back. It writes y4m at 8, 10 and 12 bit 4:2:0, 4:2:2 and 4:4:4, and 8 and 12 bit greyscale, at any size.
- Reads any video file ffmpeg can decode, such as MKV or MP4 with AV1, HEVC or H.264, by running an installed ffmpeg as regrain and retone do. `--ffmpeg` picks the executable. y4m files and y4m on standard input are read directly and need no ffmpeg.
- `-o denoised.mkv` writes the denoised video as lossless FFV1 with the source's colour tags and HDR10 metadata, and copies the source's audio, subtitles, attachments, chapters and tags unchanged. `--video-only` leaves the other tracks out. Dolby Vision and HDR10+ metadata cannot be kept, which a `warning` reports. Sources with a variable frame rate are refused with their other tracks, as they would drift out of sync.
- Two stages in the style of V-BM3D. Patches are grouped with matches from their own frame and 3 frames on each side, followed with block motion fields.
- The grain is measured from the video itself, for every DCT frequency and 16 brightness bands. It is measured between frames and capped by the grain of single frames, so neither motion nor texture reads as grain. Scene cuts are detected, and each frame uses only the grain of its own scene.
- Only fine grain is removed. Coarse, blotchy noise beyond 6 times the high frequency grain, which is also the most AV1 grain synthesis can draw again, stays in the picture.
- Runs on the GPU through wgpu, on Vulkan, Metal or DX12, at about 9.2 fps at 1080p on a Radeon 780M. `--device cpu` runs the CPU reference, about 20 times slower.
- Every setting is an option, grouped in `--help` under strength, matching, motion, grain measurement, device and diagnostics. Every number shows its default and range there, and values outside the range are refused. `--show-noise`, `--show-spectrum` and `--timing` print the measured grain and where the time went.
- Progress shows the frames done out of the total, the percent and the time left, and `--progress json` reports them as `total`, `percent`, `elapsed_seconds` and `eta_seconds`. `--progress json` reports `start`, `progress`, `summary` and `error` events. `warning` events report what of the source is lost. Errors carry a stable code: `unsupported_input`, `unsupported_output`, `bad_input`, `no_frames`, `ffmpeg_missing`, `ffmpeg_failed`, `gpu_unavailable`, `gpu_failed`, `io_error`, `cancelled` or `error`.
- Interlaced video is refused with a hint to deinterlace it first, as the filter would treat its two fields as one picture. An output that is the input file is refused.
- Output files are written under a temporary name and renamed when the run succeeds. Ctrl+C or SIGTERM stops cleanly with code `cancelled` and exit status 130, and removes the unfinished output.
- `completions` and `manual` print a shell completion script and the manual page.

### Known limitations

- Two GPU runs of the same video differ by about 100 dB, far below anything visible, as the GPU adds up overlapping patches in a different order each time.
- 10 bit greyscale y4m is not read. Video files in 10 bit greyscale are converted to 12 bit by ffmpeg.
- y4m output carries no colour primaries or transfer. Write `.mkv` to keep them.
