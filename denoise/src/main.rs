use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rabbit_denoise::noise::NoiseModel;
use rabbit_denoise::{Denoiser, Device, Frame, FrameModels, Measured, Plane, Settings};
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
const LIMITS: [(&str, Limit); 15] = [
  ("sigma_scale", Limit::Fraction(0.25, 4.0)),
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
      println!("{object}");
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

fn to_plane(bytes: &[u8], width: usize, height: usize, wide: bool) -> Plane {
  let data = if wide {
    bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b) as f32).collect()
  } else {
    bytes.iter().map(|&b| b as f32).collect()
  };
  Plane { width, height, data }
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
  let ffprobe = ffmpeg::ffprobe_path(&job.ffmpeg);

  // Frames are read and written on their own threads. Files other than y4m come through ffmpeg.
  let mut decoding = None;
  let mut source = None;
  let input: Box<dyn Read + Send> = if is_stdio(&job.input) {
    Box::new(std::io::stdin())
  } else if ffmpeg::is_y4m(&job.input)? {
    Box::new(std::fs::File::open(&job.input).map_err(|e| fail("io_error", format!("could not open {}: {e}", job.input.display())))?)
  } else {
    let probed = ffmpeg::probe(&ffprobe, &job.input)?;
    if let Some(order) = &probed.interlaced {
      return Err(interlaced(&job.input, order));
    }
    let (process, stdout) = ffmpeg::decode(&job.ffmpeg, &probed)?;
    decoding = Some(process);
    source = Some(probed);
    Box::new(stdout)
  };

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

  let decoded = y4m::decode(BufReader::with_capacity(1 << 20, input));
  // An empty stream from ffmpeg is ffmpeg's failure.
  if let (Err(_), Some(decoding)) = (&decoded, decoding.as_mut()) {
    check_cancel()?;
    decoding.finish()?;
  }
  let mut decoder = decoded.map_err(|e| match e {
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
    return Err(interlaced(&job.input, order));
  }
  let (width, height) = (decoder.get_width(), decoder.get_height());
  let colorspace = decoder.get_colorspace();
  let bit_depth = colorspace.get_bit_depth();
  let wide = colorspace.get_bytes_per_sample() == 2;
  let maxval = ((1u32 << bit_depth) - 1) as f32;
  let chroma = chroma_size(colorspace, width, height)?;
  // The frame count, for the percent done and the time left: from ffprobe, or a y4m file's size.
  let total = match &source {
    Some(source) => source.frames,
    None if !is_stdio(&job.input) => {
      let samples = width * height + chroma.map_or(0, |(w, h)| 2 * w * h);
      let frame_bytes = (6 + samples * colorspace.get_bytes_per_sample()) as u64;
      std::fs::metadata(&job.input)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len() / frame_bytes)
    }
    None => None,
  };

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
    // The reader also measures each frame's grain on its own.
    let (frames_in, frames) = std::sync::mpsc::sync_channel::<Result<Measured>>(2);
    let measurer = denoiser.measurer();
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
        let mut planes = vec![to_plane(frame.get_y_plane(), width, height, wide)];
        if let Some((cw, ch)) = chroma {
          planes.push(to_plane(frame.get_u_plane(), cw, ch, wide));
          planes.push(to_plane(frame.get_v_plane(), cw, ch, wide));
        }
        if frames_in.send(Ok(measurer.measure(Frame { planes }))).is_err() {
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
    for frame in frames {
      check_cancel()?;
      let frame = match frame {
        Ok(frame) => frame,
        // A frame cut short is ffmpeg's failure when it has stopped.
        Err(e) => return Err(decoding.as_mut().and_then(|d| d.failed(Duration::ZERO)).unwrap_or(e)),
      };
      let done = denoiser.push_measured(frame).map_err(|e| fail("gpu_failed", format!("{e:#}")))?;
      read += 1;
      show_noise(&mut denoiser);
      if !write(done, &mut written)? {
        running = false;
        break;
      }
      progress(written, false);
    }
    if running {
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
  if reporter.text() {
    eprintln!(
      "\rDenoised {written} frames in {seconds:.1} s, {:.2} fps   ",
      written as f64 / seconds
    );
    PROGRESS_SHOWN.store(false, std::sync::atomic::Ordering::Relaxed);
  }
  if job.settings.timing {
    eprintln!("{}", denoiser.timing_report());
  }
  reporter.event(
    "summary",
    json!({ "frames": written, "seconds": seconds, "fps": written as f64 / seconds }),
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
        },
        reporter,
      )
    }
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
    let text = ["input", "output", "ffmpeg", "device", "progress"];
    for arg in denoise.get_arguments() {
      let id = arg.get_id().as_str();
      let takes_value = arg.get_num_args().is_some_and(|n| n.takes_values());
      if takes_value && !text.contains(&id) {
        assert_eq!(LIMITS.iter().filter(|(name, _)| *name == id).count(), 1, "{id}");
      }
    }
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
