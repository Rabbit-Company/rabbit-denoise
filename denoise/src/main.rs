use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rabbit_denoise::dct::Transforms;
use rabbit_denoise::noise::{GrainTally, NoiseModel, NoiseStats};
use rabbit_denoise::scenes::{CutDetector, blur_into};
use rabbit_denoise::{Denoiser, Device, Frame, FrameModels, Measured, Plane, Settings};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::fmt::{self, Display};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use y4m::Colorspace;

mod ffmpeg;

static CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
// Whether a progress line is on the terminal, unfinished.
static PROGRESS_SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
// Whether standard output was closed, as by `| head`, so nothing more can be printed.
static STDOUT_CLOSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Prints a line to standard output, or notes that it was closed instead of panicking.
fn print_line(line: impl Display) {
  if writeln!(std::io::stdout().lock(), "{line}").is_err() {
    STDOUT_CLOSED.store(true, std::sync::atomic::Ordering::Relaxed);
  }
}

// The first Ctrl+C or SIGTERM stops the denoiser cleanly. A second one ends the process.
fn install_cancel() {
  let _ = ctrlc::set_handler(|| {
    if CANCELLED.swap(true, std::sync::atomic::Ordering::SeqCst) {
      std::process::exit(130);
    }
  });
}

fn check_cancel() -> Result<()> {
  if CANCELLED.load(std::sync::atomic::Ordering::Relaxed) {
    Err(fail("cancelled", "cancelled"))
  } else {
    Ok(())
  }
}

/// The output's temporary file, removed unless the run finished.
struct PartFile(Option<PathBuf>);

impl Drop for PartFile {
  fn drop(&mut self) {
    if let Some(path) = self.0.take() {
      let _ = std::fs::remove_file(path);
    }
  }
}

const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("RABBIT_DENOISE_TARGET"), ")");

#[derive(Parser)]
#[command(
    name = "rabbit-denoise",
    version = VERSION,
    about = "Remove film grain from video and keep the picture"
)]
struct Cli {
  #[command(subcommand)]
  command: Command,
  /// How to report progress and results
  #[arg(long, value_enum, default_value = "text", global = true)]
  progress: Progress,
  /// Print nothing except errors, the same as --progress none
  #[arg(short, long, global = true)]
  quiet: bool,
}

// Parsed once at start, so the size of the largest variant does not matter.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
  /// Denoise a video
  Denoise {
    /// Video to denoise, or - for y4m on standard input
    ///
    /// y4m is read directly. Any other file, such as MKV or MP4 with AV1, HEVC or H.264, is
    /// decoded by ffmpeg, which must then be installed.
    #[arg(default_value = "-")]
    input: PathBuf,
    /// Output video, or - for y4m on standard output
    ///
    /// A .mkv file gets the denoised video as lossless FFV1 with the source's colour tags
    /// and HDR10 metadata, and the source's audio, subtitles, attachments and chapters
    /// unchanged. Any other name gets y4m.
    #[arg(short, long, default_value = "-")]
    output: PathBuf,
    /// Overwrite the output file if it exists
    #[arg(short = 'y', long)]
    overwrite: bool,
    /// With an .mkv output, write the denoised video alone, without the source's other tracks
    #[arg(long)]
    video_only: bool,
    /// ffmpeg executable for files other than y4m, ffprobe is expected next to it
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,

    /// Multiplies the measured grain deviation
    ///
    /// 1.0 removes the grain as measured. Higher removes more grain and more fine detail,
    /// lower leaves some grain in.
    #[arg(long, default_value_t = 1.0, help_heading = "Strength")]
    sigma_scale: f32,
    /// Multiplies the chroma planes' grain deviation, on top of --sigma-scale
    ///
    /// Above 1 removes more colour grain.
    #[arg(long, default_value_t = 1.0, help_heading = "Strength")]
    chroma_scale: f32,
    /// Grain deviations a coefficient must clear to survive the first stage
    ///
    /// The first stage keeps a coefficient only where it is this many times the grain at
    /// its frequency and brightness. Higher removes more.
    #[arg(long, default_value_t = 2.7, help_heading = "Strength")]
    lambda: f32,
    /// Return the first stage's result and skip the second
    ///
    /// About twice as fast and about 2 dB worse.
    #[arg(long, help_heading = "Strength")]
    basic: bool,
    /// Denoise only the scenes whose grain reaches this, and copy the others unchanged
    ///
    /// The grain as `analyze` measures it: the deviation of the luma grain on a 0 to 255
    /// scale, whatever the bit depth. The video is read twice, first to find its scenes and
    /// measure them, so the input must be a file. Clean anime mostly measures below 1 and
    /// grainy openings 3 and more, so 2 denoises only their grainy scenes. Anime with film
    /// grain throughout can measure 1.5 to 2.5, which 2 splits scene by scene: denoise all of
    /// it instead, or use 1.
    #[arg(long, help_heading = "Scenes")]
    min_grain: Option<f32>,
    /// Denoise only these frames, and copy the others unchanged
    ///
    /// First and last frames, counted from 0, such as 0-299,515-1265, or single frames such
    /// as 42. Each run of touching frames is denoised as if the video began and ended there.
    /// Made for a program that chose the scenes itself, such as from what `analyze` printed,
    /// so the video is read once and standard input works too.
    #[arg(long, value_name = "RANGES", value_parser = parse_frame_ranges, conflicts_with = "min_grain", help_heading = "Scenes")]
    denoise_frames: Option<FrameRanges>,

    /// Distance between reference patches, in samples
    ///
    /// 3 is the best quality. 4 is about 30 per cent faster for about 0.3 dB less.
    #[arg(long, default_value_t = 3, help_heading = "Matching")]
    step: u32,
    /// Frames on each side of the current one that matches come from
    ///
    /// More frames remove more grain without blurring. 2 is about 20 per cent faster for
    /// about 0.4 dB less, and 5 gains about 0.1 dB over 3. 0 matches within the frame only.
    #[arg(long, default_value_t = 3, help_heading = "Matching")]
    temporal_radius: u32,
    /// How far from a patch, in samples, matches are searched for in its own frame
    #[arg(long, default_value_t = 5, help_heading = "Matching")]
    search_radius: u32,
    /// Matches taken into a group from each frame, besides the reference
    ///
    /// They must be --min-shift apart.
    #[arg(long, default_value_t = 2, help_heading = "Matching")]
    matches_per_frame: u32,
    /// How far apart, in samples, two matches from one frame must be
    ///
    /// Closer patches share most of their grain, so averaging them removes little.
    #[arg(long, default_value_t = 4, help_heading = "Matching")]
    min_shift: u32,
    /// Most patches in a group, the reference included
    #[arg(long, default_value_t = 16, help_heading = "Matching")]
    group_size: u32,
    /// How much a first stage match may differ from its reference
    ///
    /// In multiples of the difference grain alone makes. Higher blurs more, lower removes
    /// less grain.
    #[arg(long, default_value_t = 2.0, help_heading = "Matching")]
    match_limit: f32,
    /// How much a second stage match may differ from its reference
    ///
    /// As --match-limit, for the second stage, which matches on the first stage's result.
    #[arg(long, default_value_t = 0.5, help_heading = "Matching")]
    wiener_match_limit: f32,

    /// Search radius in neighbouring frames, around a patch's track and its predicted move
    ///
    /// In samples. Defaults to 1 with motion fields and 3 with --no-motion.
    #[arg(long, help_heading = "Motion")]
    follow_radius: Option<u32>,
    /// --follow-radius for the second stage only
    #[arg(long, help_heading = "Motion")]
    wiener_follow_radius: Option<u32>,
    /// How far, in samples, the motion fields look for a block's move between two frames
    ///
    /// Fast motion needs more. Rounded down to an even number.
    #[arg(long, default_value_t = 16, help_heading = "Motion")]
    motion_range: u32,
    /// Follow patches without motion fields, searching around their track alone
    ///
    /// Slower and slightly worse. Kept for comparison.
    #[arg(long, help_heading = "Motion")]
    no_motion: bool,

    /// Use the grain measured between frames without capping it by single frames
    ///
    /// Motion reads high between frames and texture reads high in single frames, so by
    /// default the smaller of the two is used.
    #[arg(long, help_heading = "Grain measurement")]
    no_single_frame_cap: bool,
    /// Match chroma blocks on chroma rather than on luma when measuring its grain
    ///
    /// Matching on chroma reads its grain 10 to 20 per cent low.
    #[arg(long, help_heading = "Grain measurement")]
    no_luma_guided_chroma: bool,

    /// Where to run the filter
    #[arg(long, value_enum, default_value = "auto", help_heading = "Device")]
    device: DeviceArg,

    /// Print the grain measured for every frame to standard error
    #[arg(long, help_heading = "Diagnostics")]
    show_noise: bool,
    /// With --show-noise, also print the 8x8 luma grain spectrum of this brightness band
    ///
    /// Bands 0 to 15 split the brightness range evenly, from black to white.
    #[arg(long, help_heading = "Diagnostics")]
    show_spectrum: Option<u32>,
    /// Print where the time went to standard error at the end
    ///
    /// On the GPU, also the time of each shader stage.
    #[arg(long, help_heading = "Diagnostics")]
    timing: bool,
  },
  /// Find a video's scenes and measure the grain of each, without denoising
  ///
  /// Prints every scene with its first and last frame, its times and its grain: the
  /// deviation of the luma grain on a 0 to 255 scale, whatever the bit depth. Clean anime
  /// mostly measures below 1, grainy openings 3 and more, and anime with film grain
  /// throughout 1.5 to 2.5. The grain is
  /// measured on single frames and between frames, and the smaller is taken, as texture
  /// reads high in the first and motion in the second.
  Analyze {
    /// Video to analyze, or - for y4m on standard input
    #[arg(default_value = "-")]
    input: PathBuf,
    /// ffmpeg executable for files other than y4m, ffprobe is expected next to it
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,
  },
  /// Print a shell completion script
  Completions {
    /// Shell to print the script for
    shell: clap_complete::Shell,
  },
  /// Print the manual page in roff format
  Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DeviceArg {
  /// The GPU when one opens, the CPU otherwise
  Auto,
  /// The GPU, or fail
  Gpu,
  /// The CPU, about 20 times slower
  Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Progress {
  /// Progress and results as text on standard error
  Text,
  /// One JSON object per line on standard output, or on standard error when the video goes to standard output
  Json,
  /// Nothing except errors
  None,
}

/// The smallest and largest value of a numeric option.
#[derive(Clone, Copy)]
enum Limit {
  Whole(u32, u32),
  Fraction(f32, f32),
}

/// Every numeric option of `denoise` with its limits, which parsing enforces and --help
/// shows. The whole number limits are what the GPU filter runs.
const LIMITS: [(&str, Limit); 16] = [
  ("sigma_scale", Limit::Fraction(0.25, 4.0)),
  ("min_grain", Limit::Fraction(0.1, 50.0)),
  ("chroma_scale", Limit::Fraction(0.25, 4.0)),
  ("lambda", Limit::Fraction(1.0, 6.0)),
  ("step", Limit::Whole(1, 6)),
  ("temporal_radius", Limit::Whole(0, 16)),
  ("search_radius", Limit::Whole(0, 7)),
  ("matches_per_frame", Limit::Whole(1, 8)),
  ("min_shift", Limit::Whole(1, 8)),
  ("group_size", Limit::Whole(2, 16)),
  ("match_limit", Limit::Fraction(0.5, 8.0)),
  ("wiener_match_limit", Limit::Fraction(0.1, 4.0)),
  ("follow_radius", Limit::Whole(0, 7)),
  ("wiener_follow_radius", Limit::Whole(0, 7)),
  ("motion_range", Limit::Whole(0, 32)),
  ("show_spectrum", Limit::Whole(0, 15)),
];

/// Gives an option its limits: the parser that enforces them, `<N>` or `<X>` for its value,
/// and its default and range at the end of its help.
fn limited(arg: clap::Arg, limit: Limit) -> clap::Arg {
  let (name, range) = match limit {
    Limit::Whole(min, max) => ("N", format!("{min} to {max}")),
    Limit::Fraction(min, max) => ("X", format!("{min} to {max}")),
  };
  let default = arg.get_default_values().first().map(|v| v.to_string_lossy().into_owned());
  let notes = match default {
    Some(default) => format!("[default: {default}] [range: {range}]"),
    None => format!("[range: {range}]"),
  };
  let help = arg.get_help().map(ToString::to_string).unwrap_or_default();
  let long = arg.get_long_help().map(ToString::to_string).unwrap_or_else(|| help.clone());
  let arg = arg
    .value_name(name)
    .hide_default_value(true)
    .help(format!("{help} {notes}"))
    .long_help(format!("{long}\n\n{notes}"));
  match limit {
    Limit::Whole(min, max) => arg.value_parser(move |text: &str| match text.parse::<u32>() {
      Ok(value) if (min..=max).contains(&value) => Ok(value),
      _ => Err(format!("{text} is not a whole number from {min} to {max}")),
    }),
    Limit::Fraction(min, max) => arg.value_parser(move |text: &str| match text.parse::<f32>() {
      Ok(value) if (min..=max).contains(&value) => Ok(value),
      _ => Err(format!("{text} is not a number from {min} to {max}")),
    }),
  }
}

/// The command line, with every numeric option's limits.
fn cli_command() -> clap::Command {
  <Cli as clap::CommandFactory>::command()
    // The global options come after each command's own.
    .mut_arg("progress", |arg| arg.display_order(1000))
    .mut_arg("quiet", |arg| arg.display_order(1001))
    .mut_subcommand("denoise", |mut denoise| {
      // Changing an option moves it to the end, so every option keeps its place explicitly.
      let order: Vec<String> = denoise.get_arguments().map(|a| a.get_id().to_string()).collect();
      for (place, id) in order.iter().enumerate() {
        let limit = LIMITS.iter().find(|(name, _)| name == id).map(|(_, limit)| *limit);
        denoise = denoise.mut_arg(id, |arg| {
          let arg = arg.display_order(place);
          match limit {
            Some(limit) => limited(arg, limit),
            None => arg,
          }
        });
      }
      denoise
    })
}

/// An error with a stable code for --progress json.
#[derive(Debug)]
struct Failure {
  code: &'static str,
  message: String,
}

impl Display for Failure {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(&self.message)
  }
}

impl std::error::Error for Failure {}

fn fail(code: &'static str, message: impl Display) -> anyhow::Error {
  anyhow::Error::new(Failure {
    code,
    message: message.to_string(),
  })
}

fn error_code(error: &anyhow::Error) -> &'static str {
  error
    .chain()
    .find_map(|cause| cause.downcast_ref::<Failure>().map(|f| f.code))
    .unwrap_or("error")
}

struct Reporter {
  mode: Progress,
  // JSON goes to standard error when the video is on standard output.
  json_to_stderr: bool,
}

impl Reporter {
  fn event(&self, name: &str, fields: Value) {
    if self.mode != Progress::Json {
      return;
    }
    let mut object = json!({ "event": name });
    if let (Some(target), Value::Object(fields)) = (object.as_object_mut(), fields) {
      target.extend(fields);
    }
    if self.json_to_stderr {
      eprintln!("{object}");
    } else {
      print_line(object);
    }
  }

  fn text(&self) -> bool {
    self.mode == Progress::Text
  }
}

/// Width and height of the chroma planes, or None for monochrome.
fn chroma_size(colorspace: Colorspace, width: usize, height: usize) -> Result<Option<(usize, usize)>> {
  use Colorspace::*;
  Ok(match colorspace {
    Cmono | Cmono12 => None,
    C420 | C420p10 | C420p12 | C420jpeg | C420paldv | C420mpeg2 => Some((width.div_ceil(2), height.div_ceil(2))),
    C422 | C422p10 | C422p12 => Some((width.div_ceil(2), height)),
    C444 | C444p10 | C444p12 => Some((width, height)),
    other => return Err(fail("unsupported_input", format!("unsupported colorspace {other:?}"))),
  })
}

/// Converts a plane's samples on all threads, a row at a time.
fn to_plane(bytes: &[u8], width: usize, height: usize, wide: bool) -> Plane {
  to_plane_into(bytes, width, height, wide, vec![0.0; width * height])
}

/// `to_plane`, written into `data`, a buffer to reuse.
fn to_plane_into(bytes: &[u8], width: usize, height: usize, wide: bool, mut data: Vec<f32>) -> Plane {
  data.resize(width * height, 0.0);
  let mut plane = Plane { width, height, data };
  let rows = plane.data.par_chunks_mut(width.max(1));
  if wide {
    rows.zip(bytes.par_chunks(2 * width.max(1))).for_each(|(row, src)| {
      for (v, b) in row.iter_mut().zip(src.as_chunks::<2>().0) {
        *v = u16::from_le_bytes(*b) as f32;
      }
    });
  } else {
    rows.zip(bytes.par_chunks(width.max(1))).for_each(|(row, src)| {
      for (v, &b) in row.iter_mut().zip(src) {
        *v = b as f32;
      }
    });
  }
  plane
}

fn to_bytes(plane: &Plane, maxval: f32, wide: bool) -> Vec<u8> {
  let round = |v: f32| v.round().clamp(0.0, maxval) as u16;
  if wide {
    plane.data.iter().flat_map(|&v| round(v).to_le_bytes()).collect()
  } else {
    plane.data.iter().map(|&v| round(v) as u8).collect()
  }
}

fn is_stdio(path: &std::path::Path) -> bool {
  path.as_os_str() == "-"
}

struct Job {
  input: PathBuf,
  ffmpeg: String,
  video_only: bool,
  output: PathBuf,
  settings: Settings,
  show_spectrum: Option<usize>,
  overwrite: bool,
  min_grain: Option<f32>,
  denoise_frames: Option<FrameRanges>,
}

/// Frames to denoise, as first and last frames: sorted, with touching and overlapping
/// ranges joined.
#[derive(Clone, Debug, PartialEq)]
struct FrameRanges(Vec<(usize, usize)>);

/// Reads ranges such as `0-299,515-1265,2000`.
fn parse_frame_ranges(text: &str) -> std::result::Result<FrameRanges, String> {
  let mut ranges = Vec::new();
  for part in text.split(',').map(str::trim) {
    let number = |s: &str| {
      s.trim()
        .parse::<usize>()
        .map_err(|_| format!("{part} is not a frame or a range of frames such as 0-299"))
    };
    let (first, last) = match part.split_once('-') {
      Some((a, b)) => (number(a)?, number(b)?),
      None => {
        let frame = number(part)?;
        (frame, frame)
      }
    };
    if first > last {
      return Err(format!("{part} ends before it starts"));
    }
    ranges.push((first, last));
  }
  ranges.sort_unstable();
  let mut joined: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
  for (first, last) in ranges {
    match joined.last_mut() {
      Some(previous) if first <= previous.1.saturating_add(1) => previous.1 = previous.1.max(last),
      _ => joined.push((first, last)),
    }
  }
  Ok(FrameRanges(joined))
}

/// The frames `--min-grain` or `--denoise-frames` denoises, as first and last frames of
/// runs, in order.
#[derive(Clone)]
struct Plan {
  runs: Vec<(usize, usize)>,
  /// The frames the scan read. Frames past them follow the last one.
  frames: usize,
}

impl Plan {
  fn denoises(&self, index: usize) -> bool {
    let index = index.min(self.frames.saturating_sub(1));
    let i = self.runs.partition_point(|r| r.1 < index);
    i < self.runs.len() && self.runs[i].0 <= index
  }

  fn denoised_frames(&self) -> usize {
    self.runs.iter().map(|r| r.1 + 1 - r.0).sum()
  }
}

/// What the reader hands the denoiser: a frame to denoise with its grain measured, or one
/// to copy.
enum Item {
  Denoise(Measured),
  Keep(Frame),
}

/// Scans the video for `--min-grain` and returns the frames to denoise.
fn plan(job: &Job, min_grain: f32, reporter: &Reporter) -> Result<Plan> {
  let start = Instant::now();
  if reporter.text() {
    eprintln!("Finding the scenes with grain {min_grain} or more");
  }
  let (mut runs, mut scenes, mut chosen) = (Vec::<(usize, usize)>::new(), 0usize, 0usize);
  let frames = scan(
    &job.input,
    &job.ffmpeg,
    reporter,
    Some("analyze"),
    |_| Ok(()),
    |scene, format, _| {
      scenes += 1;
      // A scene without enough measurable blocks, such as a black screen, has no grain.
      if !scene.luma().deviation(format.maxval).is_some_and(|g| g >= min_grain) {
        return;
      }
      chosen += 1;
      match runs.last_mut() {
        Some(run) if run.1 + 1 == scene.first => run.1 = scene.last,
        _ => runs.push((scene.first, scene.last)),
      }
    },
  )?
  .expect("a scan with a stage does not stop");
  let plan = Plan { runs, frames };
  let denoised = plan.denoised_frames();
  let percent = denoised as f64 * 100.0 / frames as f64;
  if reporter.text() {
    eprintln!(
      "Found {scenes} scenes in {:.1} s. Denoising {chosen} of them, {denoised} of {frames} frames ({percent:.0}%), and copying the rest",
      start.elapsed().as_secs_f64()
    );
  }
  reporter.event(
    "scenes",
    json!({ "scenes": scenes, "denoised_scenes": chosen, "frames": frames, "denoised_frames": denoised, "min_grain": min_grain }),
  );
  Ok(plan)
}

/// What the output is written as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputKind {
  Y4m,
  Mkv,
}

/// MKV for a .mkv name, y4m for standard output, devices, pipes and other names.
fn output_kind(path: &std::path::Path) -> Result<OutputKind> {
  if is_stdio(path) || std::fs::metadata(path).is_ok_and(|m| !m.is_file()) {
    return Ok(OutputKind::Y4m);
  }
  let extension = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase());
  match extension.as_deref() {
    Some("mkv") => Ok(OutputKind::Mkv),
    None | Some("y4m") => Ok(OutputKind::Y4m),
    Some(other) => Err(fail(
      "unsupported_output",
      format!("cannot write .{other} files, name the output .mkv for lossless FFV1 or .y4m"),
    )),
  }
}

/// The error for an interlaced source, which the filter would treat as whole frames.
fn interlaced(path: &std::path::Path, order: &str) -> anyhow::Error {
  fail(
    "unsupported_input",
    format!(
      "{} is interlaced (field order {order}), which rabbit-denoise does not denoise. Deinterlace it first, for example with ffmpeg -i INPUT -vf bwdif=mode=send_frame -strict -1 -f yuv4mpegpipe - | rabbit-denoise denoise -o OUTPUT, or add -vf setfield=prog instead when its frames are progressive despite the flag",
      if is_stdio(path) {
        "the input".to_string()
      } else {
        path.display().to_string()
      }
    ),
  )
}

/// The error to report for a failed write: ffmpeg's when the muxing ffmpeg has stopped.
fn write_error(error: anyhow::Error, muxing: &mut Option<ffmpeg::Process>) -> anyhow::Error {
  muxing.as_mut().and_then(|m| m.failed(Duration::from_secs(5))).unwrap_or(error)
}

/// Prints a warning, or reports it as a `warning` event.
fn warn(reporter: &Reporter, message: &str) {
  if reporter.text() {
    eprintln!("Warning: {message}");
  }
  reporter.event("warning", json!({ "message": message }));
}

/// A video on its way in as y4m: from a y4m file, standard input or a decoding ffmpeg.
struct Input {
  decoding: Option<ffmpeg::Process>,
  /// What ffprobe found, for files ffmpeg decodes.
  source: Option<ffmpeg::Source>,
  reader: Box<dyn Read + Send>,
}

/// Opens the input, starting ffmpeg for files other than y4m. Interlaced files are refused.
fn open_input(path: &std::path::Path, ffmpeg: &str) -> Result<Input> {
  if is_stdio(path) {
    return Ok(Input {
      decoding: None,
      source: None,
      reader: Box::new(std::io::stdin()),
    });
  }
  if ffmpeg::is_y4m(path)? {
    let file = std::fs::File::open(path).map_err(|e| fail("io_error", format!("could not open {}: {e}", path.display())))?;
    return Ok(Input {
      decoding: None,
      source: None,
      reader: Box::new(file),
    });
  }
  let probed = ffmpeg::probe(&ffmpeg::ffprobe_path(ffmpeg), path)?;
  if let Some(order) = &probed.interlaced {
    return Err(interlaced(path, order));
  }
  let (process, stdout) = ffmpeg::decode(ffmpeg, &probed)?;
  Ok(Input {
    decoding: Some(process),
    source: Some(probed),
    reader: Box::new(stdout),
  })
}

/// The picture format of a y4m stream.
#[derive(Clone, Copy)]
struct Format {
  width: usize,
  height: usize,
  colorspace: Colorspace,
  bit_depth: usize,
  /// Two bytes per sample.
  wide: bool,
  maxval: f32,
  /// The chroma planes' size, or None for monochrome.
  chroma: Option<(usize, usize)>,
}

impl Format {
  /// The planes of a y4m frame.
  fn frame(&self, frame: &y4m::Frame) -> Frame {
    let mut planes = vec![to_plane(frame.get_y_plane(), self.width, self.height, self.wide)];
    if let Some((cw, ch)) = self.chroma {
      planes.push(to_plane(frame.get_u_plane(), cw, ch, self.wide));
      planes.push(to_plane(frame.get_v_plane(), cw, ch, self.wide));
    }
    Frame { planes }
  }

  /// `frame`, in buffers taken from `buffers`.
  fn frame_into(&self, frame: &y4m::Frame, buffers: &Buffers) -> Frame {
    let plane = |bytes: &[u8], w: usize, h: usize| to_plane_into(bytes, w, h, self.wide, buffers.take(w * h));
    let mut planes = vec![plane(frame.get_y_plane(), self.width, self.height)];
    if let Some((cw, ch)) = self.chroma {
      planes.push(plane(frame.get_u_plane(), cw, ch));
      planes.push(plane(frame.get_v_plane(), cw, ch));
    }
    Frame { planes }
  }

  /// The frame count, for the percent done and the time left: from ffprobe, or a y4m file's size.
  fn total(&self, source: Option<&ffmpeg::Source>, path: &std::path::Path) -> Option<u64> {
    match source {
      Some(source) => source.frames,
      None if !is_stdio(path) => {
        let samples = self.width * self.height + self.chroma.map_or(0, |(w, h)| 2 * w * h);
        let frame_bytes = (6 + samples * self.colorspace.get_bytes_per_sample()) as u64;
        std::fs::metadata(path).ok().filter(|m| m.is_file()).map(|m| m.len() / frame_bytes)
      }
      None => None,
    }
  }
}

type Decoder = y4m::Decoder<BufReader<Box<dyn Read + Send>>>;

/// Reads the y4m header and returns the decoder, the header's text and the format.
/// Interlaced streams are refused.
fn read_header(
  reader: Box<dyn Read + Send>,
  decoding: &mut Option<ffmpeg::Process>,
  path: &std::path::Path,
) -> Result<(Decoder, String, Format)> {
  let decoded = y4m::decode(BufReader::with_capacity(1 << 20, reader));
  // An empty stream from ffmpeg is ffmpeg's failure.
  if let (Err(_), Some(decoding)) = (&decoded, decoding.as_mut()) {
    check_cancel()?;
    decoding.finish()?;
  }
  let decoder = decoded.map_err(|e| match e {
    y4m::Error::UnknownColorspace => fail(
      "unsupported_input",
      "unsupported y4m colorspace: rabbit-denoise reads 8, 10 and 12 bit 4:2:0, 4:2:2 and 4:4:4, and 8 and 12 bit greyscale. Convert other formats with ffmpeg -pix_fmt, such as gray12le for 10 bit greyscale",
    ),
    e => fail("bad_input", format!("not a y4m stream: {e:?}")),
  })?;
  let header = String::from_utf8_lossy(decoder.get_raw_params()).into_owned();
  // y4m names the field order t, b or m, ffprobe tt, bb or mixed.
  let order = header.split_whitespace().find_map(|t| t.strip_prefix('I')).and_then(|o| match o {
    "t" => Some("tt"),
    "b" => Some("bb"),
    "m" => Some("mixed"),
    _ => None,
  });
  if let Some(order) = order {
    return Err(interlaced(path, order));
  }
  let (width, height) = (decoder.get_width(), decoder.get_height());
  let colorspace = decoder.get_colorspace();
  let bit_depth = colorspace.get_bit_depth();
  let format = Format {
    width,
    height,
    colorspace,
    bit_depth,
    wide: colorspace.get_bytes_per_sample() == 2,
    maxval: ((1u32 << bit_depth) - 1) as f32,
    chroma: chroma_size(colorspace, width, height)?,
  };
  Ok((decoder, header, format))
}

fn denoise(job: Job, reporter: &Reporter) -> Result<()> {
  let kind = output_kind(&job.output)?;
  if !is_stdio(&job.output) && std::fs::metadata(&job.output).is_ok_and(|m| m.is_file()) && !job.overwrite {
    return Err(fail(
      "io_error",
      format!("{} exists, pass -y to overwrite it", job.output.display()),
    ));
  }
  // Writing over the input would lose it, and the output is only renamed into place at the end.
  let same_file = !is_stdio(&job.input)
    && !is_stdio(&job.output)
    && matches!((std::fs::canonicalize(&job.input), std::fs::canonicalize(&job.output)), (Ok(input), Ok(output)) if input == output);
  if same_file {
    return Err(fail(
      "io_error",
      format!("{} is the input, choose another output", job.output.display()),
    ));
  }
  // Standard input, pipes and devices cannot be read a second time.
  if job.min_grain.is_some() && !(!is_stdio(&job.input) && std::fs::metadata(&job.input).is_ok_and(|m| m.is_file())) {
    return Err(fail(
      "unsupported_input",
      "--min-grain reads the video twice, so the input must be a file, not standard input or a pipe",
    ));
  }
  let ffprobe = ffmpeg::ffprobe_path(&job.ffmpeg);

  // Frames are read and written on their own threads. Files other than y4m come through ffmpeg.
  let Input {
    mut decoding,
    source,
    reader,
  } = open_input(&job.input, &job.ffmpeg)?;

  // An MKV takes the source's other tracks unless asked not to, which a variable frame
  // rate would put out of sync, and its HDR metadata.
  let tracks = kind == OutputKind::Mkv && !job.video_only && source.is_some();
  let mut hdr = None;
  if let (OutputKind::Mkv, Some(source)) = (kind, &source) {
    if tracks && source.is_variable_frame_rate(&ffprobe)? {
      return Err(fail(
        "unsupported_input",
        format!(
          "{} has a variable frame rate, so its other tracks would drift out of sync with the denoised video. Pass --video-only to write the video alone",
          source.path.display()
        ),
      ));
    }
    if source.colour.is_hdr() || source.dolby_vision {
      let found = source.hdr(&ffprobe)?;
      if source.dolby_vision || found.dolby_vision {
        warn(reporter, "the source's Dolby Vision metadata is not kept, the output is plain HDR");
      }
      if found.hdr10_plus {
        warn(reporter, "the source's HDR10+ metadata is not kept, the output is plain HDR10");
      }
      hdr = Some(found);
    }
  }

  let (mut decoder, header, format) = read_header(reader, &mut decoding, &job.input)?;
  let total = format.total(source.as_ref(), &job.input);
  let Format {
    width,
    height,
    colorspace,
    bit_depth,
    wide,
    maxval,
    ..
  } = format;
  // The scan runs once the checks above have passed, before any output exists. The second
  // ffmpeg waits meanwhile.
  let plan = match (job.min_grain, &job.denoise_frames) {
    (Some(min_grain), _) => Some(plan(&job, min_grain, reporter)?),
    (None, Some(FrameRanges(runs))) => {
      let plan = Plan {
        runs: runs.clone(),
        // Frames past the last range are copied.
        frames: usize::MAX,
      };
      if reporter.text() {
        eprintln!(
          "Denoising {} frames in {} runs, and copying the rest",
          plan.denoised_frames(),
          plan.runs.len()
        );
      }
      Some(plan)
    }
    (None, None) => None,
  };
  check_cancel()?;

  // Files are written under a temporary name so a failed run leaves nothing that looks
  // finished. Devices and pipes are written directly.
  let special = std::fs::metadata(&job.output).is_ok_and(|m| !m.is_file());
  let temporary = (!is_stdio(&job.output) && !special).then(|| {
    let mut name = job.output.clone().into_os_string();
    name.push(".part");
    PathBuf::from(name)
  });
  let mut part = PartFile(None);
  let mut muxing = None;
  let output: Box<dyn Write + Send> = if is_stdio(&job.output) {
    Box::new(std::io::stdout())
  } else if kind == OutputKind::Mkv {
    let path = temporary.as_ref().unwrap_or(&job.output);
    part.0 = temporary.clone();
    let extras = ffmpeg::Extras {
      source: source.as_ref(),
      tracks,
      hdr: hdr.as_ref(),
    };
    let (process, stdin) = ffmpeg::mux(&job.ffmpeg, path, &job.output, &extras)?;
    muxing = Some(process);
    Box::new(stdin)
  } else {
    let path = temporary.as_ref().unwrap_or(&job.output);
    let file: Box<dyn Write + Send> = Box::new(
      std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(!special)
        .open(path)
        .map_err(|e| fail("io_error", format!("could not create {}: {e}", path.display())))?,
    );
    part.0 = temporary.clone();
    file
  };

  // ffmpeg's colour range and similar options travel in X parameters.
  let mut builder = y4m::encode(width, height, decoder.get_framerate())
    .with_colorspace(colorspace)
    .with_pixel_aspect(decoder.get_pixel_aspect());
  for token in header.split_whitespace() {
    if let Some(extension) = token.strip_prefix('X') {
      builder = builder.append_vendor_extension(y4m::VendorExtensionString::new(extension.as_bytes().to_vec())?);
    }
  }
  let mut sink = BufWriter::with_capacity(1 << 20, output);
  let encoder = match builder.write_header(&mut sink) {
    Ok(encoder) => encoder,
    Err(e) => return Err(write_error(fail("io_error", format!("could not write: {e:?}")), &mut muxing)),
  };

  let mut denoiser = Denoiser::new(job.settings.clone(), bit_depth).map_err(|e| fail("gpu_unavailable", format!("{e:#}")))?;
  let device = denoiser.gpu_name().unwrap_or("CPU").to_string();
  if reporter.text() {
    match denoiser.gpu_note() {
      Some(note) => eprintln!("Denoising on the CPU: {note}"),
      None => eprintln!("Denoising on {device}"),
    }
  }
  reporter.event(
    "start",
    json!({ "width": width, "height": height, "bit_depth": bit_depth, "colorspace": format!("{colorspace:?}"), "device": device, "total": total }),
  );
  let start = Instant::now();
  let mut last_report = Instant::now();
  let spectrum_band = job.show_spectrum;
  let show_noise = |denoiser: &mut Denoiser| {
    for FrameModels { index, hard, wiener } in denoiser.take_models() {
      if let Some(band) = spectrum_band.filter(|&b| b < rabbit_denoise::noise::BANDS) {
        let s = wiener[0].spectrum(band);
        eprintln!("frame {index} luma grain spectrum of band {band}, deviation per coefficient, vertical frequency by row:");
        for v in 0..8 {
          eprintln!(
            "  {}",
            (0..8).map(|u| format!("{:6.2}", s[v * 8 + u].sqrt())).collect::<Vec<_>>().join("")
          );
        }
      }
      let line = |model: &NoiseModel| {
        model
          .band_variance()
          .iter()
          .map(|v| format!("{:5.2}", v.sqrt()))
          .collect::<Vec<_>>()
          .join(" ")
      };
      eprintln!("frame {index} luma grain by band, hard:   {}", line(&hard[0]));
      eprintln!("frame {index} luma grain by band, wiener: {}", line(&wiener[0]));
      if wiener.len() == 3 {
        let middle = |model: &NoiseModel| model.band_variance()[rabbit_denoise::noise::BANDS / 2].sqrt();
        eprintln!(
          "frame {index} chroma grain at mid grey, hard: {:.2} {:.2}, wiener: {:.2} {:.2}",
          middle(&hard[1]),
          middle(&hard[2]),
          middle(&wiener[1]),
          middle(&wiener[2])
        );
      }
    }
  };
  let mut progress = |written: usize, last: bool| {
    if !last && last_report.elapsed().as_secs_f64() < 0.5 {
      return;
    }
    last_report = Instant::now();
    let elapsed = start.elapsed().as_secs_f64();
    let fps = written as f64 / elapsed.max(1e-9);
    // The count is an estimate for some containers, so it never reads below what is done.
    let total = total.map(|t| t.max(written as u64));
    let percent = total.map(|t| (written as f64 * 100.0 / t.max(1) as f64).min(100.0));
    let eta = total
      .filter(|_| fps > 0.0 && written > 0)
      .map(|t| ((t - written as u64) as f64 / fps).round() as u64);
    if reporter.text() {
      let frames = match total {
        Some(total) => format!("{written}/{total} frames ({:.0}%)", percent.unwrap_or(0.0)),
        None => format!("{written} frames"),
      };
      let left = eta.map_or_else(String::new, |s| format!(", eta {}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60));
      eprint!("\rDenoised {frames}, {fps:.2} fps{left}   ");
      PROGRESS_SHOWN.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    reporter.event(
      "progress",
      json!({ "frames": written, "total": total, "percent": percent, "fps": fps, "elapsed_seconds": elapsed, "eta_seconds": eta }),
    );
  };

  // One thread reads frames ahead of the denoiser and another writes them behind it.
  let (mut read, mut written) = (0usize, 0usize);
  std::thread::scope(|scope| -> Result<()> {
    // The reader also measures the grain of each frame to denoise on its own.
    let (frames_in, frames) = std::sync::mpsc::sync_channel::<Result<Item>>(2);
    let measurer = denoiser.measurer();
    let reader_plan = plan.clone();
    scope.spawn(move || {
      for index in 0.. {
        let frame = match decoder.read_frame() {
          Ok(frame) => frame,
          Err(y4m::Error::EOF) => return,
          Err(e) => {
            let _ = frames_in.send(Err(fail("bad_input", format!("could not read frame {index}: {e:?}"))));
            return;
          }
        };
        let frame = format.frame(&frame);
        let item = if reader_plan.as_ref().is_none_or(|p| p.denoises(index)) {
          Item::Denoise(measurer.measure(frame))
        } else {
          Item::Keep(frame)
        };
        if frames_in.send(Ok(item)).is_err() {
          return;
        }
      }
    });
    let (done_out, done) = std::sync::mpsc::sync_channel::<Vec<Frame>>(2);
    let writer = scope.spawn(move || -> Result<()> {
      let mut encoder = encoder;
      for frames in done {
        for frame in frames {
          let bytes: Vec<Vec<u8>> = frame.planes.iter().map(|p| to_bytes(p, maxval, wide)).collect();
          let empty: &[u8] = &[];
          let planes = match bytes.len() {
            1 => [bytes[0].as_slice(), empty, empty],
            _ => [bytes[0].as_slice(), bytes[1].as_slice(), bytes[2].as_slice()],
          };
          encoder
            .write_frame(&y4m::Frame::new(planes, None))
            .map_err(|e| fail("io_error", format!("could not write: {e:?}")))?;
        }
      }
      Ok(())
    });
    // A failed write ends the writer, and its error is reported.
    let write = |frames: Vec<Frame>, written: &mut usize| -> Result<bool> {
      *written += frames.len();
      Ok(frames.is_empty() || done_out.send(frames).is_ok())
    };
    let mut running = true;
    // Whether the denoiser takes frames, and how many were copied since its last run ended.
    // Each run of frames to denoise starts the denoiser afresh, so its frame numbers stay
    // those of the video.
    let (mut open, mut skipped) = (true, 0usize);
    for item in frames {
      check_cancel()?;
      let item = match item {
        Ok(item) => item,
        // A frame cut short is ffmpeg's failure when it has stopped.
        Err(e) => return Err(decoding.as_mut().and_then(|d| d.failed(Duration::ZERO)).unwrap_or(e)),
      };
      read += 1;
      let done = match item {
        Item::Denoise(frame) => {
          if !open {
            denoiser.resume(skipped);
            (open, skipped) = (true, 0);
          }
          let done = denoiser.push_measured(frame).map_err(|e| fail("gpu_failed", format!("{e:#}")))?;
          show_noise(&mut denoiser);
          done
        }
        Item::Keep(frame) => {
          // The run before goes out first, so the frames stay in order.
          let mut done = Vec::new();
          if open {
            done = denoiser.finish().map_err(|e| fail("gpu_failed", format!("{e:#}")))?;
            show_noise(&mut denoiser);
            open = false;
          }
          skipped += 1;
          done.push(frame);
          done
        }
      };
      if !write(done, &mut written)? {
        running = false;
        break;
      }
      progress(written, false);
    }
    if running && open {
      check_cancel()?;
      let done = denoiser.finish().map_err(|e| fail("gpu_failed", format!("{e:#}")))?;
      show_noise(&mut denoiser);
      write(done, &mut written)?;
    }
    drop(done_out);
    let wrote = writer.join().map_err(|_| anyhow::anyhow!("the writing thread failed"))?;
    wrote.map_err(|e| write_error(e, &mut muxing))?;
    progress(written, true);
    Ok(())
  })?;
  // Dropping a BufWriter would hide a failed final write.
  if let Err(e) = sink.flush() {
    return Err(write_error(fail("io_error", format!("could not write: {e}")), &mut muxing));
  }
  // Closing its input lets the muxing ffmpeg finish the file.
  drop(sink);
  // Ctrl+C also stops an ffmpeg feeding the input, which then looks like its end.
  check_cancel()?;
  if let Some(decoding) = decoding.as_mut() {
    decoding.finish()?;
  }
  if written != read {
    bail!("read {read} frames but wrote {written}");
  }
  if read == 0 {
    return Err(fail("no_frames", "no frames could be read"));
  }
  if let Some(muxing) = muxing.as_mut() {
    muxing.finish()?;
  }
  if let Some(path) = &temporary {
    std::fs::rename(path, &job.output).map_err(|e| fail("io_error", format!("could not move {} into place: {e}", path.display())))?;
    part.0 = None;
  }
  let seconds = start.elapsed().as_secs_f64();
  // With --min-grain, the frames planned past the end of the video are not counted.
  let denoised = plan.as_ref().map(|p| (0..written).filter(|&i| p.denoises(i)).count());
  if reporter.text() {
    let copied = denoised.map_or(String::new(), |d| format!(", {} of them copied unchanged", written - d));
    eprintln!(
      "\rDenoised {written} frames in {seconds:.1} s, {:.2} fps{copied}   ",
      written as f64 / seconds
    );
    PROGRESS_SHOWN.store(false, std::sync::atomic::Ordering::Relaxed);
  }
  if job.settings.timing {
    eprintln!("{}", denoiser.timing_report());
  }
  let mut fields = json!({ "frames": written, "seconds": seconds, "fps": written as f64 / seconds });
  if let Some(denoised) = denoised {
    fields["denoised_frames"] = json!(denoised);
  }
  reporter.event("summary", fields);
  Ok(())
}

/// A scene found by `scan`, its frames and their grain.
struct Scene {
  first: usize,
  last: usize,
  /// One tally per plane, measured on single frames.
  tallies: Vec<GrainTally>,
  /// The luma grain measured between the scene's frames.
  pairs: GrainTally,
}

impl Scene {
  /// The luma grain: measured on single frames, capped by the grain measured between them.
  fn luma(&self) -> GrainTally {
    self.tallies[0].capped(&self.pairs)
  }
}

/// A frame's measurements for `scan`: whether it starts a scene, its grain on single frames
/// per plane, and the luma grain between it and the frame before, for the second of a pair.
type Measurement = (bool, Vec<NoiseStats>, Option<NoiseStats>);

/// Plane buffers that `scan` reuses. Fresh memory for every frame cost a fifth of the time,
/// in the system clearing its pages.
#[derive(Default)]
struct Buffers(std::sync::Mutex<Vec<Vec<f32>>>);

impl Buffers {
  /// The most buffers kept. No more are kept than were in use at once.
  const KEPT: usize = 32;

  /// A buffer of `len` samples if one is kept, or an empty one.
  fn take(&self, len: usize) -> Vec<f32> {
    let mut kept = self.0.lock().expect("buffers");
    kept
      .iter()
      .position(|b| b.len() == len)
      .map_or_else(Vec::new, |i| kept.swap_remove(i))
  }

  fn give(&self, data: Vec<f32>) {
    let mut kept = self.0.lock().expect("buffers");
    if kept.len() < Self::KEPT {
      kept.push(data);
    }
  }

  fn give_frame(&self, frame: Frame) {
    frame.planes.into_iter().for_each(|p| self.give(p.data));
  }
}

/// How many pairs `scan` goes on measuring while it waits for the oldest to finish.
const PAIRS_AT_ONCE: usize = 2;

/// Measures a frame and the next, or a frame without a pair, with its cut.
fn measure_frames(
  first: (Frame, bool),
  second: Option<(Frame, bool)>,
  maxval: f32,
  transforms: &Transforms,
  buffers: &Buffers,
) -> Vec<Measurement> {
  let single = |frame: &Frame| -> Vec<NoiseStats> { frame.planes.iter().map(|p| NoiseStats::measure(p, maxval, transforms)).collect() };
  let Some(second) = second else {
    let stats = single(&first.0);
    buffers.give_frame(first.0);
    return vec![(first.1, stats, None)];
  };
  let (a, b) = (&first.0.planes[0], &second.0.planes[0]);
  let ((stats_a, stats_b), pair) = rayon::join(
    || rayon::join(|| single(&first.0), || single(&second.0)),
    || {
      let take = |p: &Plane| buffers.take(p.data.len());
      let (blurred_a, blurred_b) = rayon::join(|| blur_into(a, take(a)), || blur_into(b, take(b)));
      let blocks = NoiseStats::pair_blocks(a, &blurred_a, b, &blurred_b, None, transforms);
      buffers.give(blurred_a.data);
      buffers.give(blurred_b.data);
      NoiseStats::from_pairs(&blocks, false, maxval)
    },
  );
  buffers.give_frame(first.0);
  buffers.give_frame(second.0);
  vec![(first.1, stats_a, None), (second.1, stats_b, Some(pair))]
}

/// Rounds for printing in JSON, where f32 values would show their binary noise.
fn rounded(value: f64, places: i32) -> f64 {
  let scale = 10f64.powi(places);
  (value * scale).round() / scale
}

fn clock(seconds: f64) -> String {
  let millis = (seconds * 1000.0).round() as u64;
  format!(
    "{}:{:02}:{:02}.{:03}",
    millis / 3_600_000,
    millis / 60_000 % 60,
    millis / 1000 % 60,
    millis % 1000
  )
}

/// Clears an unfinished progress line on the terminal.
fn clear_progress() {
  if PROGRESS_SHOWN.swap(false, std::sync::atomic::Ordering::Relaxed) {
    eprint!("\r{:79}\r", "");
  }
}

/// What `scan` knows once the y4m header is read.
struct Scanning<'a> {
  format: Format,
  fps: f64,
  total: Option<u64>,
  source: Option<&'a ffmpeg::Source>,
}

/// Reads a whole video, splits it into scenes and measures the grain of each on single
/// frames. `started` runs once the header is read and `scene` as each scene ends.
/// `progress` events carry `stage` when it is given. Without a stage, as for `analyze`, the
/// scan stops when standard output is closed and returns None. Otherwise it returns the
/// frames read.
fn scan(
  path: &std::path::Path,
  ffmpeg: &str,
  reporter: &Reporter,
  stage: Option<&str>,
  started: impl FnOnce(&Scanning) -> Result<()>,
  mut scene: impl FnMut(&Scene, &Format, f64),
) -> Result<Option<usize>> {
  let Input {
    mut decoding,
    source,
    reader,
  } = open_input(path, ffmpeg)?;
  let (mut decoder, _, format) = read_header(reader, &mut decoding, path)?;
  let total = format.total(source.as_ref(), path);
  let rate = decoder.get_framerate();
  let fps = if rate.den > 0 && rate.num > 0 {
    rate.num as f64 / rate.den as f64
  } else {
    24.0
  };
  started(&Scanning {
    format,
    fps,
    total,
    source: source.as_ref(),
  })?;

  let stderr_terminal = std::io::IsTerminal::is_terminal(&std::io::stderr());
  let start = Instant::now();
  let mut last_report = Instant::now();
  let mut progress = |frames: usize, last: bool| {
    if !last && last_report.elapsed().as_secs_f64() < 0.5 {
      return;
    }
    last_report = Instant::now();
    let elapsed = start.elapsed().as_secs_f64();
    let rate = frames as f64 / elapsed.max(1e-9);
    let total = total.map(|t| t.max(frames as u64));
    let percent = total.map(|t| (frames as f64 * 100.0 / t.max(1) as f64).min(100.0));
    let eta = total
      .filter(|_| rate > 0.0 && frames > 0)
      .map(|t| ((t - frames as u64) as f64 / rate).round() as u64);
    if reporter.text() && stderr_terminal && !last {
      let done = match total {
        Some(total) => format!("{frames}/{total} frames ({:.0}%)", percent.unwrap_or(0.0)),
        None => format!("{frames} frames"),
      };
      let left = eta.map_or_else(String::new, |s| format!(", eta {}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60));
      eprint!("\rAnalyzed {done}, {rate:.2} fps{left}   ");
      PROGRESS_SHOWN.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let mut fields =
      json!({ "frames": frames, "total": total, "percent": percent, "fps": rate, "elapsed_seconds": elapsed, "eta_seconds": eta });
    if let Some(stage) = stage {
      fields["stage"] = json!(stage);
    }
    reporter.event("progress", fields);
  };

  let mut read = 0usize;
  let stopped = std::thread::scope(|scope| -> Result<bool> {
    // One thread reads frames and finds the cuts while another measures the frame before, so
    // that reading, which is mostly on one thread, does not leave the others waiting.
    let (decoded_in, decoded) = std::sync::mpsc::sync_channel::<Result<(Frame, bool)>>(2);
    let buffers = std::sync::Arc::new(Buffers::default());
    let reading = buffers.clone();
    scope.spawn(move || {
      let mut cuts = CutDetector::new(format.maxval);
      for index in 0.. {
        let frame = match decoder.read_frame() {
          Ok(frame) => frame,
          Err(y4m::Error::EOF) => return,
          Err(e) => {
            let _ = decoded_in.send(Err(fail("bad_input", format!("could not read frame {index}: {e:?}"))));
            return;
          }
        };
        let frame = format.frame_into(&frame, &reading);
        let cut = cuts.is_cut(&frame.planes[0]);
        if decoded_in.send(Ok((frame, cut))).is_err() {
          return;
        }
      }
    });
    // Measuring a frame does not need the one before, except for the pairs.
    let (frames_in, frames) = std::sync::mpsc::sync_channel::<Result<Measurement>>(4);
    scope.spawn(move || {
      let transforms = std::sync::Arc::new(Transforms::new(1));
      let maxval = format.maxval;
      // A pair is measured on all threads, and while one waits on a few of its last blocks
      // the next keeps the others busy. They are passed on in order.
      let mut running = std::collections::VecDeque::new();
      let mut start = |first: (Frame, bool), second: Option<(Frame, bool)>| -> bool {
        let (done, measured) = std::sync::mpsc::sync_channel(1);
        let (transforms, buffers) = (transforms.clone(), buffers.clone());
        rayon::spawn(move || {
          let _ = done.send(measure_frames(first, second, maxval, &transforms, &buffers));
        });
        running.push_back(measured);
        while running.len() > PAIRS_AT_ONCE {
          let measured = running.pop_front().expect("running").recv().expect("measured");
          for frame in measured {
            if frames_in.send(Ok(frame)).is_err() {
              return false;
            }
          }
        }
        true
      };
      // Grain is measured between frames 0 and 1, 2 and 3, and so on, which is enough and
      // takes half the time of every pair. A pair across a cut measures nothing useful, so
      // a cut starts the next pair.
      let mut held: Option<(Frame, bool)> = None;
      let mut failed = None;
      for decoded in decoded {
        let (frame, cut) = match decoded {
          Ok(decoded) => decoded,
          Err(e) => {
            failed = Some(e);
            break;
          }
        };
        let going = match held.take() {
          Some(first) if !cut => start(first, Some((frame, cut))),
          Some(first) => {
            held = Some((frame, cut));
            start(first, None)
          }
          None => {
            held = Some((frame, cut));
            true
          }
        };
        if !going {
          return;
        }
      }
      if let Some(first) = held
        && !start(first, None)
      {
        return;
      }
      for measured in running {
        for frame in measured.recv().expect("measured") {
          if frames_in.send(Ok(frame)).is_err() {
            return;
          }
        }
      }
      if let Some(e) = failed {
        let _ = frames_in.send(Err(e));
      }
    });
    let mut current: Option<Scene> = None;
    for measured in frames {
      check_cancel()?;
      // Nobody reads the scenes any more, as after `| head`.
      if stage.is_none() && STDOUT_CLOSED.load(std::sync::atomic::Ordering::Relaxed) {
        // Stopping ffmpeg before its pipe closes keeps it from reporting a broken pipe.
        drop(decoding.take());
        return Ok(true);
      }
      let (cut, stats, pair) = match measured {
        Ok(measured) => measured,
        // A frame cut short is ffmpeg's failure when it has stopped.
        Err(e) => return Err(decoding.as_mut().and_then(|d| d.failed(Duration::ZERO)).unwrap_or(e)),
      };
      if cut {
        if let Some(done) = current.take() {
          scene(&done, &format, fps);
        }
        current = Some(Scene {
          first: read,
          last: read,
          tallies: vec![GrainTally::default(); stats.len()],
          pairs: GrainTally::default(),
        });
      }
      let open = current.as_mut().expect("the first frame starts a scene");
      open.last = read;
      if let Some(pair) = &pair {
        open.pairs.add(pair);
      }
      for (tally, stats) in open.tallies.iter_mut().zip(&stats) {
        tally.add(stats);
      }
      read += 1;
      progress(read, false);
    }
    // Ctrl+C also stops an ffmpeg feeding the input, which then looks like its end.
    check_cancel()?;
    if let Some(done) = current.take() {
      scene(&done, &format, fps);
    }
    Ok(false)
  })?;
  if stopped {
    return Ok(None);
  }
  if let Some(decoding) = decoding.as_mut() {
    decoding.finish()?;
  }
  if read == 0 {
    return Err(fail("no_frames", "no frames could be read"));
  }
  progress(read, true);
  if reporter.text() {
    clear_progress();
  }
  Ok(Some(read))
}

fn analyze(path: PathBuf, ffmpeg: String, reporter: &Reporter) -> Result<()> {
  let start = Instant::now();
  let mut shown_header = false;
  let mut scenes = 0usize;
  let started = |scanning: &Scanning| -> Result<()> {
    if let Some(source) = scanning.source
      && source.is_variable_frame_rate(&ffmpeg::ffprobe_path(&ffmpeg))?
    {
      warn(
        reporter,
        "the source has a variable frame rate, so scene times assume one frame rate and drift. Frame numbers are exact",
      );
    }
    let Scanning { format, fps, total, .. } = scanning;
    if reporter.text() {
      eprintln!(
        "Analyzing {}x{}, {} bit, {fps:.3} fps",
        format.width, format.height, format.bit_depth
      );
    }
    reporter.event(
      "start",
      json!({ "width": format.width, "height": format.height, "bit_depth": format.bit_depth, "colorspace": format!("{:?}", format.colorspace), "fps": fps, "total": total }),
    );
    Ok(())
  };
  let report_scene = |scene: &Scene, format: &Format, fps: f64| {
    let maxval = format.maxval;
    let luma_tally = scene.luma();
    let luma = luma_tally.deviation(maxval);
    let chroma: Option<Vec<Option<f32>>> =
      (scene.tallies.len() == 3).then(|| scene.tallies[1..].iter().map(|t| t.deviation(maxval)).collect());
    let (start, end) = (scene.first as f64 / fps, (scene.last + 1) as f64 / fps);
    let value = |v: Option<f32>| v.map(|v| rounded(v as f64, 3));
    reporter.event(
      "scene",
      json!({
        "index": scenes,
        "first_frame": scene.first,
        "last_frame": scene.last,
        "frames": scene.last + 1 - scene.first,
        "start_seconds": rounded(start, 6),
        "end_seconds": rounded(end, 6),
        "grain": value(luma),
        "chroma_grain": chroma.as_ref().map(|c| c.iter().map(|&v| value(v)).collect::<Vec<_>>()),
        "grain_by_band": luma_tally.bands(maxval).map(value),
      }),
    );
    if reporter.text() {
      clear_progress();
      if !shown_header {
        print_line(format!(
          "{:>6}  {:>15}  {:>12}  {:>9}  {:>6}  {:>11}",
          "Scene", "Frames", "Start", "Length", "Grain", "Chroma"
        ));
        shown_header = true;
      }
      let number = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.2}"));
      let chroma = chroma.map_or(String::new(), |c| format!("{:>5} {:>5}", number(c[0]), number(c[1])));
      print_line(format!(
        "{:>6}  {:>15}  {:>12}  {:>8.3}s  {:>6}  {chroma:>11}",
        scenes + 1,
        format!("{}-{}", scene.first, scene.last),
        clock(start),
        end - start,
        number(luma),
      ));
    }
    scenes += 1;
  };
  let Some(read) = scan(&path, &ffmpeg, reporter, None, started, report_scene)? else {
    return Ok(());
  };
  let seconds = start.elapsed().as_secs_f64();
  if reporter.text() {
    eprintln!(
      "Analyzed {read} frames in {seconds:.1} s, {:.2} fps, {scenes} scenes",
      read as f64 / seconds
    );
  }
  reporter.event(
    "summary",
    json!({ "frames": read, "scenes": scenes, "seconds": seconds, "fps": read as f64 / seconds }),
  );
  Ok(())
}

fn run(command: Command, reporter: &Reporter) -> Result<()> {
  match command {
    Command::Denoise {
      input,
      output,
      overwrite,
      video_only,
      ffmpeg,
      sigma_scale,
      chroma_scale,
      lambda,
      basic,
      min_grain,
      denoise_frames,
      step,
      temporal_radius,
      search_radius,
      matches_per_frame,
      min_shift,
      group_size,
      match_limit,
      wiener_match_limit,
      follow_radius,
      wiener_follow_radius,
      motion_range,
      no_motion,
      no_single_frame_cap,
      no_luma_guided_chroma,
      device,
      show_noise,
      show_spectrum,
      timing,
    } => {
      let mut settings = Settings::new(temporal_radius as usize);
      settings.sigma_scale = sigma_scale;
      settings.chroma_scale = chroma_scale;
      settings.basic_only = basic;
      settings.single_frame_cap = !no_single_frame_cap;
      settings.chroma_luma_matching = !no_luma_guided_chroma;
      settings.device = match device {
        DeviceArg::Auto => Device::Auto,
        DeviceArg::Gpu => Device::Gpu,
        DeviceArg::Cpu => Device::Cpu,
      };
      settings.keep_models = show_noise;
      settings.timing = timing;
      settings.hard.lambda = lambda;
      settings.hard.distance_factor = match_limit;
      settings.wiener.distance_factor = wiener_match_limit;
      let follow = follow_radius.map_or(if no_motion { 3 } else { 1 }, |r| r as usize);
      for params in [&mut settings.hard, &mut settings.wiener] {
        params.step = step as usize;
        params.search_radius = search_radius as usize;
        params.per_frame = matches_per_frame as usize;
        params.min_shift = min_shift as usize;
        params.max_group = group_size as usize;
        params.motion = !no_motion;
        params.motion_range = motion_range as usize;
        params.predict_radius = follow;
      }
      if let Some(r) = wiener_follow_radius {
        settings.wiener.predict_radius = r as usize;
      }
      denoise(
        Job {
          input,
          ffmpeg,
          video_only,
          output,
          settings,
          show_spectrum: show_spectrum.map(|b| b as usize),
          overwrite,
          min_grain,
          denoise_frames,
        },
        reporter,
      )
    }
    Command::Analyze { input, ffmpeg } => analyze(input, ffmpeg, reporter),
    Command::Completions { shell } => {
      let mut command = cli_command();
      let mut text = Vec::new();
      clap_complete::generate(shell, &mut command, "rabbit-denoise", &mut text);
      print_bytes(&text)
    }
    Command::Manual => {
      let command = cli_command();
      let mut text = Vec::new();
      clap_mangen::Man::new(command)
        .render(&mut text)
        .context("could not render the manual")?;
      print_bytes(&text)
    }
  }
}

fn print_bytes(bytes: &[u8]) -> Result<()> {
  match std::io::stdout().lock().write_all(bytes) {
    Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => Err(fail("io_error", error)),
    _ => Ok(()),
  }
}

fn main() -> ExitCode {
  let cli = <Cli as clap::FromArgMatches>::from_arg_matches(&cli_command().get_matches()).unwrap_or_else(|e| e.exit());
  install_cancel();
  let mode = if cli.quiet { Progress::None } else { cli.progress };
  let json_to_stderr = matches!(&cli.command, Command::Denoise { output, .. } if is_stdio(output));
  let reporter = Reporter { mode, json_to_stderr };
  match run(cli.command, &reporter) {
    Ok(()) => ExitCode::SUCCESS,
    Err(error) => {
      if mode == Progress::Json {
        reporter.event("error", json!({ "code": error_code(&error), "message": format!("{error:#}") }));
      } else {
        let newline = if PROGRESS_SHOWN.load(std::sync::atomic::Ordering::Relaxed) {
          "\n"
        } else {
          ""
        };
        eprintln!("{newline}Error: {error:#}");
      }
      if error_code(&error) == "cancelled" {
        ExitCode::from(130)
      } else {
        ExitCode::FAILURE
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn every_numeric_option_has_limits() {
    let command = cli_command();
    command.clone().debug_assert();
    let denoise = command.find_subcommand("denoise").unwrap();
    let text = ["input", "output", "ffmpeg", "device", "progress", "denoise_frames"];
    for arg in denoise.get_arguments() {
      let id = arg.get_id().as_str();
      let takes_value = arg.get_num_args().is_some_and(|n| n.takes_values());
      if takes_value && !text.contains(&id) {
        assert_eq!(LIMITS.iter().filter(|(name, _)| *name == id).count(), 1, "{id}");
      }
    }
  }

  #[test]
  fn plan_denoises_its_runs_and_frames_past_the_scan_follow_the_last() {
    let plan = Plan {
      runs: vec![(3, 5), (9, 9)],
      frames: 10,
    };
    let denoised: Vec<usize> = (0..12).filter(|&i| plan.denoises(i)).collect();
    assert_eq!(denoised, [3, 4, 5, 9, 10, 11]);
    assert_eq!(plan.denoised_frames(), 4);
    let none = Plan { runs: vec![], frames: 4 };
    assert!(!(0..6).any(|i| none.denoises(i)));
  }

  #[test]
  fn frame_ranges_are_sorted_and_joined() {
    assert_eq!(
      parse_frame_ranges("515-1265, 0-299,300-310,42,1266"),
      Ok(FrameRanges(vec![(0, 310), (515, 1266)]))
    );
    assert_eq!(parse_frame_ranges("7"), Ok(FrameRanges(vec![(7, 7)])));
    assert!(parse_frame_ranges("10-5").is_err());
    assert!(parse_frame_ranges("a-5").is_err());
    assert!(parse_frame_ranges("").is_err());
    let parse = |args: &[&str]| cli_command().try_get_matches_from([&["rabbit-denoise", "denoise"], args].concat());
    assert!(parse(&["--denoise-frames", "0-9"]).is_ok());
    assert!(parse(&["--denoise-frames", "0-9", "--min-grain", "2"]).is_err());
  }

  #[test]
  fn limits_are_enforced() {
    let parse = |args: &[&str]| cli_command().try_get_matches_from([&["rabbit-denoise", "denoise"], args].concat());
    assert!(parse(&["--sigma-scale", "4"]).is_ok());
    assert!(parse(&["--sigma-scale", "4.5"]).is_err());
    assert!(parse(&["--sigma-scale", "0.2"]).is_err());
    assert!(parse(&["--sigma-scale", "nan"]).is_err());
    assert!(parse(&["--step", "6"]).is_ok());
    assert!(parse(&["--step", "7"]).is_err());
    assert!(parse(&["--follow-radius", "8"]).is_err());
  }
}
