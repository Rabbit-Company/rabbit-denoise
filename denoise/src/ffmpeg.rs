//! Video files read and written through an external ffmpeg and ffprobe.

use crate::fail;
use anyhow::Result;
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

/// The pixel formats the y4m reader takes as they are.
const NATIVE: [&str; 14] = [
  "yuv420p",
  "yuvj420p",
  "yuv422p",
  "yuvj422p",
  "yuv444p",
  "yuvj444p",
  "yuv420p10le",
  "yuv422p10le",
  "yuv444p10le",
  "yuv420p12le",
  "yuv422p12le",
  "yuv444p12le",
  "gray",
  "gray12le",
];

/// Lossless FFV1 as RabbitEncoder writes it: every frame a keyframe, checksummed slices.
const FFV1: [&str; 16] = [
  "-c:v",
  "ffv1",
  "-level",
  "3",
  "-coder",
  "1",
  "-context",
  "1",
  "-g",
  "1",
  "-slices",
  "24",
  "-slicecrc",
  "1",
  "-threads",
  "0",
];

/// Packets sampled at the start and the middle to find a variable frame rate.
const SAMPLE_PACKETS: usize = 300;

/// Whether the file starts with a y4m header. Pipes are taken for y4m, as reading their
/// start would use it up.
pub fn is_y4m(path: &Path) -> Result<bool> {
  let metadata = std::fs::metadata(path).map_err(|e| fail("io_error", format!("could not open {}: {e}", path.display())))?;
  if !metadata.is_file() {
    return Ok(true);
  }
  let mut file = std::fs::File::open(path).map_err(|e| fail("io_error", format!("could not open {}: {e}", path.display())))?;
  let mut start = [0u8; 10];
  let mut filled = 0;
  while filled < start.len() {
    match file.read(&mut start[filled..]) {
      Ok(0) => break,
      Ok(n) => filled += n,
      Err(e) => return Err(fail("io_error", format!("could not read {}: {e}", path.display()))),
    }
  }
  Ok(&start[..filled] == b"YUV4MPEG2 ")
}

/// ffprobe next to the given ffmpeg, or the one on the PATH.
pub fn ffprobe_path(ffmpeg: &str) -> String {
  let path = Path::new(ffmpeg);
  match path.file_name().and_then(|n| n.to_str()) {
    Some(name) if name.contains("ffmpeg") && path.parent().is_some_and(|p| !p.as_os_str().is_empty()) => path
      .with_file_name(name.replace("ffmpeg", "ffprobe"))
      .to_string_lossy()
      .into_owned(),
    _ => "ffprobe".to_string(),
  }
}

/// Runs ffprobe and returns its standard output.
fn ffprobe(ffprobe: &str, path: &Path, args: &[&str]) -> Result<String> {
  let output = Command::new(ffprobe)
    .args(["-v", "error"])
    .args(args)
    .arg(path)
    .stdin(Stdio::null())
    .output()
    .map_err(|e| fail("ffmpeg_missing", format!("could not run {ffprobe}: {e}")))?;
  if !output.status.success() {
    // ffprobe starts its message with the path.
    let message = String::from_utf8_lossy(&output.stderr);
    let message = message.trim().lines().last().unwrap_or("");
    let message = message.strip_prefix(&format!("{}: ", path.display())).unwrap_or(message);
    return Err(fail("bad_input", format!("ffprobe could not read {}: {message}", path.display())));
  }
  Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A number or a fraction such as 24000/1001.
fn ratio(value: &Value) -> Option<f64> {
  if let Some(number) = value.as_f64() {
    return Some(number);
  }
  let text = value.as_str()?;
  let (top, bottom) = text.split_once('/').unwrap_or((text, "1"));
  let (top, bottom): (f64, f64) = (top.trim().parse().ok()?, bottom.trim().parse().ok()?);
  (bottom != 0.0).then(|| top / bottom)
}

/// A tag ffprobe reports, or None when it is unknown.
fn tag(stream: &Value, key: &str) -> Option<String> {
  stream[key]
    .as_str()
    .filter(|v| !v.is_empty() && *v != "unknown" && *v != "unspecified")
    .map(str::to_string)
}

/// The colour tags of a video track.
#[derive(Clone, Debug, Default)]
pub struct Colour {
  pub range: Option<String>,
  pub primaries: Option<String>,
  pub transfer: Option<String>,
  pub matrix: Option<String>,
  pub chroma_location: Option<String>,
}

impl Colour {
  /// Whether the transfer is PQ or HLG.
  pub fn is_hdr(&self) -> bool {
    matches!(self.transfer.as_deref(), Some("smpte2084" | "arib-std-b67"))
  }
}

/// What ffprobe says about a video file.
pub struct Source {
  pub path: PathBuf,
  pub pixel_format: String,
  pub colour: Colour,
  /// When the first video frame and the file start, in seconds.
  video_start: f64,
  file_start: f64,
  fps: f64,
  average_fps: Option<f64>,
  duration: Option<f64>,
  pub dolby_vision: bool,
  /// The codec of every subtitle track, in order.
  subtitles: Vec<String>,
  /// The field order when the video is interlaced, such as "tt".
  pub interlaced: Option<String>,
  /// How many frames the container says the video has, or its duration gives.
  pub frames: Option<u64>,
}

/// Probes a file's tracks and its first video track.
pub fn probe(ffprobe_path: &str, path: &Path) -> Result<Source> {
  let text = ffprobe(
    ffprobe_path,
    path,
    &[
      "-show_entries",
      "stream=codec_type,codec_name,pix_fmt,color_range,color_space,color_transfer,color_primaries,chroma_location,field_order,start_time,duration,nb_frames,avg_frame_rate,r_frame_rate:stream_tags=NUMBER_OF_FRAMES,NUMBER_OF_FRAMES-eng,DURATION:stream_disposition=attached_pic:stream_side_data=side_data_type:format=start_time,duration",
      "-of",
      "json",
    ],
  )?;
  let json: Value = serde_json::from_str(&text).map_err(|e| fail("bad_input", format!("could not read ffprobe's output: {e}")))?;
  let streams = json["streams"].as_array().cloned().unwrap_or_default();
  let video = streams
    .iter()
    .find(|s| s["codec_type"] == "video" && s["disposition"]["attached_pic"].as_i64() != Some(1))
    .ok_or_else(|| fail("unsupported_input", format!("{} has no video track", path.display())))?;
  let side_data = video["side_data_list"].as_array().cloned().unwrap_or_default();
  let fps = ratio(&video["r_frame_rate"]).filter(|f| *f > 0.0).unwrap_or(24.0);
  let video_start = ratio(&video["start_time"]).unwrap_or(0.0);
  let file_start = ratio(&json["format"]["start_time"]).unwrap_or(0.0);
  // The count MP4 stores, or the track's duration at the frame rate. mkvmerge's frame count
  // is used only where it agrees, as a cut or remux copies it unchanged.
  let count = |v: &Value| v.as_str().and_then(|s| s.parse::<u64>().ok()).filter(|n| *n > 0);
  let duration = ratio(&video["duration"])
    .or_else(|| video["tags"]["DURATION"].as_str().and_then(parse_clock))
    .or_else(|| ratio(&json["format"]["duration"]).map(|d| d - (video_start - file_start)))
    .filter(|d| *d > 0.0);
  let from_duration = duration.map(|d| (d * fps).round() as u64);
  let tagged = count(&video["tags"]["NUMBER_OF_FRAMES"]).or_else(|| count(&video["tags"]["NUMBER_OF_FRAMES-eng"]));
  let agreeing = tagged.filter(|&n| from_duration.is_some_and(|d| n.abs_diff(d) <= 1 + d / 100));
  let frames = count(&video["nb_frames"]).or(agreeing).or(from_duration);
  let interlaced = tag(video, "field_order").filter(|order| ["tt", "bb", "tb", "bt"].contains(&order.as_str()));
  Ok(Source {
    path: path.to_path_buf(),
    pixel_format: tag(video, "pix_fmt").unwrap_or_default(),
    colour: Colour {
      range: tag(video, "color_range"),
      primaries: tag(video, "color_primaries"),
      transfer: tag(video, "color_transfer"),
      matrix: tag(video, "color_space"),
      chroma_location: tag(video, "chroma_location"),
    },
    video_start,
    file_start,
    fps,
    average_fps: ratio(&video["avg_frame_rate"]).filter(|f| *f > 0.0),
    duration: ratio(&json["format"]["duration"]),
    dolby_vision: side_data.iter().any(|d| d["side_data_type"] == "DOVI configuration record"),
    subtitles: streams
      .iter()
      .filter(|s| s["codec_type"] == "subtitle")
      .map(|s| s["codec_name"].as_str().unwrap_or("").to_string())
      .collect(),
    interlaced,
    frames,
  })
}

impl Source {
  /// Whether frames last different times, so the denoised video, which comes back at one
  /// frame rate, would drift from the audio. Containers do not always report it, so the
  /// gaps between a few hundred timestamps are checked too.
  pub fn is_variable_frame_rate(&self, ffprobe_path: &str) -> Result<bool> {
    if self.average_fps.is_some_and(|average| (average - self.fps).abs() > self.fps * 0.01) {
      return Ok(true);
    }
    let middle = self.duration.unwrap_or(0.0) / 2.0;
    let intervals = format!("%+#{SAMPLE_PACKETS},{middle:.3}%+#{SAMPLE_PACKETS}");
    let text = ffprobe(
      ffprobe_path,
      &self.path,
      &[
        "-select_streams",
        "V:0",
        "-read_intervals",
        &intervals,
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
      ],
    )?;
    Ok(irregular_timestamps(&text, self.fps))
  }

  /// The HDR metadata of the first video frame.
  pub fn hdr(&self, ffprobe_path: &str) -> Result<Hdr> {
    let text = ffprobe(
      ffprobe_path,
      &self.path,
      &[
        "-select_streams",
        "V:0",
        "-read_intervals",
        "%+#1",
        "-show_frames",
        "-show_entries",
        "frame=side_data_list",
        "-of",
        "json",
      ],
    )?;
    let json: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let side_data = json["frames"][0]["side_data_list"].as_array().cloned().unwrap_or_default();
    let mut hdr = Hdr::default();
    for data in &side_data {
      match data["side_data_type"].as_str().unwrap_or("") {
        "Mastering display metadata" => hdr.mastering_display = mastering_display(data),
        "Content light level metadata" => {
          if let (Some(max), Some(average)) = (data["max_content"].as_u64(), data["max_average"].as_u64()) {
            hdr.content_light = Some(format!("{max},{average}"));
          }
        }
        kind if kind.contains("HDR10+") => hdr.hdr10_plus = true,
        kind if kind.starts_with("Dolby Vision") => hdr.dolby_vision = true,
        _ => {}
      }
    }
    Ok(hdr)
  }
}

/// Static and dynamic HDR metadata of a video track.
#[derive(Debug, Default)]
pub struct Hdr {
  /// For ffmpeg's -mastering_display.
  pub mastering_display: Option<String>,
  /// For ffmpeg's -content_light: MaxCLL,MaxFALL.
  pub content_light: Option<String>,
  pub hdr10_plus: bool,
  pub dolby_vision: bool,
}

/// Mastering display metadata as ffmpeg's -mastering_display takes it: chromaticities in
/// 1/50000 and luminance in 1/10000 cd/m².
fn mastering_display(data: &Value) -> Option<String> {
  let xy = |key: &str| ratio(&data[key]).map(|v| (v * 50000.0).round() as i64);
  let luminance = |key: &str| ratio(&data[key]).map(|v| (v * 10000.0).round() as i64);
  Some(format!(
    "G({},{})B({},{})R({},{})WP({},{})L({},{})",
    xy("green_x")?,
    xy("green_y")?,
    xy("blue_x")?,
    xy("blue_y")?,
    xy("red_x")?,
    xy("red_y")?,
    xy("white_point_x")?,
    xy("white_point_y")?,
    luminance("max_luminance")?,
    luminance("min_luminance")?,
  ))
}

/// Seconds from a clock such as 00:23:59.940000000.
fn parse_clock(value: &str) -> Option<f64> {
  let mut seconds = 0.0;
  for part in value.split(':') {
    seconds = seconds * 60.0 + part.parse::<f64>().ok()?;
  }
  Some(seconds)
}

/// Whether more than one in fifty gaps between sampled timestamps is off by a quarter
/// frame. Gaps of more than 4 frames join the two samples and are skipped.
fn irregular_timestamps(text: &str, fps: f64) -> bool {
  let mut times: Vec<f64> = text
    .lines()
    .filter_map(|line| line.trim().trim_end_matches(',').parse().ok())
    .collect();
  times.sort_by(f64::total_cmp);
  times.dedup();
  let frame = 1.0 / fps;
  let (mut deltas, mut irregular) = (0usize, 0usize);
  for pair in times.windows(2) {
    let gap = pair[1] - pair[0];
    if gap > frame * 4.0 {
      continue;
    }
    deltas += 1;
    irregular += usize::from((gap - frame).abs() > frame * 0.25);
  }
  deltas >= 20 && irregular * 50 > deltas
}

/// The y4m pixel format to decode a source format to. Formats the reader takes stay as
/// they are. Others keep their chroma layout where y4m has it, and their bit depth rounded
/// up to 8, 10 or 12.
pub fn y4m_format(source: &str) -> String {
  if NATIVE.contains(&source) {
    return source.to_string();
  }
  let bits = source_bits(source);
  if source.starts_with("gray") || source.starts_with("ya") {
    return if bits > 8 { "gray12le" } else { "gray" }.to_string();
  }
  let layout = if ["420", "nv12", "nv21", "p010", "p012", "p016"].iter().any(|s| source.contains(s)) {
    "420"
  } else if [
    "422", "nv16", "nv20", "p210", "p212", "p216", "y210", "y212", "yuyv", "uyvy", "yvyu",
  ]
  .iter()
  .any(|s| source.contains(s))
  {
    "422"
  } else {
    "444"
  };
  match bits {
    0..=8 => format!("yuv{layout}p"),
    9 | 10 => format!("yuv{layout}p10le"),
    _ => format!("yuv{layout}p12le"),
  }
}

/// The bit depth a pixel format name gives, or 8.
fn source_bits(format: &str) -> u32 {
  // Semi-planar and packed formats such as p010, y210 and nv20 name their depth apart.
  for prefix in ["p0", "p2", "p4", "y2", "y4"] {
    if let Some(bits) = format
      .strip_prefix(prefix)
      .and_then(|rest| rest.get(..2))
      .and_then(|d| d.parse().ok())
    {
      return bits;
    }
  }
  if format.starts_with("nv") {
    return if format.starts_with("nv20") { 10 } else { 8 };
  }
  if ["48", "64"].iter().any(|d| format.contains(d)) {
    return 16;
  }
  // Planar formats end with the depth before the byte order, as in yuv420p10le and gray12be.
  let name = format.trim_end_matches("le").trim_end_matches("be");
  let digits = name.len() - name.trim_end_matches(|c: char| c.is_ascii_digit()).len();
  match name[name.len() - digits..].parse::<u32>() {
    Ok(bits) if (9..=16).contains(&bits) => bits,
    _ => 8,
  }
}

/// An ffmpeg run, killed if it is still running when dropped.
pub struct Process {
  child: Option<Child>,
  /// What it does, for errors: "decoding x.mkv".
  task: String,
}

impl Process {
  fn failure(&self, status: std::process::ExitStatus) -> anyhow::Error {
    fail("ffmpeg_failed", format!("ffmpeg failed while {} ({status})", self.task))
  }

  /// The error to report when ffmpeg stops within `wait`, having failed.
  pub fn failed(&mut self, wait: Duration) -> Option<anyhow::Error> {
    let deadline = Instant::now() + wait;
    loop {
      let status = self.child.as_mut()?.try_wait().ok()?;
      if let Some(status) = status {
        self.child = None;
        return (!status.success()).then(|| self.failure(status));
      }
      if Instant::now() >= deadline {
        return None;
      }
      std::thread::sleep(Duration::from_millis(20));
    }
  }

  /// Waits for ffmpeg once its input or output has ended, and fails if ffmpeg did.
  pub fn finish(&mut self) -> Result<()> {
    let Some(mut child) = self.child.take() else { return Ok(()) };
    let status = child
      .wait()
      .map_err(|e| fail("ffmpeg_failed", format!("could not wait for ffmpeg: {e}")))?;
    if !status.success() {
      return Err(self.failure(status));
    }
    Ok(())
  }
}

impl Drop for Process {
  fn drop(&mut self) {
    if let Some(mut child) = self.child.take() {
      let _ = child.kill();
      let _ = child.wait();
    }
  }
}

/// Starts ffmpeg decoding the first video track to y4m on its standard output.
pub fn decode(ffmpeg: &str, source: &Source) -> Result<(Process, ChildStdout)> {
  let format = y4m_format(&source.pixel_format);
  let mut child = Command::new(ffmpeg)
    .args(["-v", "error", "-nostdin", "-i"])
    .arg(&source.path)
    .args(["-map", "0:V:0", "-an", "-sn", "-dn", "-pix_fmt", &format])
    .args(["-fps_mode", "passthrough", "-strict", "-1", "-f", "yuv4mpegpipe", "-"])
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::inherit())
    .spawn()
    .map_err(|e| fail("ffmpeg_missing", format!("could not run {ffmpeg}: {e}")))?;
  let stdout = child.stdout.take().expect("piped standard output");
  let process = Process {
    child: Some(child),
    task: format!("decoding {}", source.path.display()),
  };
  Ok((process, stdout))
}

/// What goes into an MKV besides the denoised video.
pub struct Extras<'a> {
  pub source: Option<&'a Source>,
  /// Copy the source's audio, subtitles, attachments, chapters and tags.
  pub tracks: bool,
  pub hdr: Option<&'a Hdr>,
}

/// Starts ffmpeg writing the y4m on its standard input to `path` as lossless FFV1 in
/// Matroska, with the source's colour tags and HDR10 metadata, and its other tracks.
pub fn mux(ffmpeg: &str, path: &Path, shown: &Path, extras: &Extras) -> Result<(Process, ChildStdin)> {
  let mut command = Command::new(ffmpeg);
  command.args(["-v", "error", "-nostdin", "-nostats", "-y"]);
  if let Some(hdr) = extras.hdr {
    if let Some(display) = &hdr.mastering_display {
      command.args(["-mastering_display", display]);
    }
    if let Some(light) = &hdr.content_light {
      command.args(["-content_light", light]);
    }
  }
  // The video keeps its place against the other tracks.
  if let Some(source) = extras.source.filter(|_| extras.tracks) {
    let offset = source.video_start - source.file_start;
    if offset.abs() > 1e-6 {
      command.args(["-itsoffset", &format!("{offset:.6}")]);
    }
  }
  command.args(["-f", "yuv4mpegpipe", "-i", "-"]);
  let tracks = extras.source.filter(|_| extras.tracks);
  if let Some(source) = tracks {
    command.arg("-i").arg(&source.path);
  }
  command.args(["-map", "0:v:0"]);
  match tracks {
    Some(source) => {
      command.args([
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-map",
        "1:t?",
        "-map_chapters",
        "1",
        "-map_metadata",
        "1",
      ]);
      // A stream metadata map of its own would stop ffmpeg copying every stream's tags,
      // the attachments' MIME types among them.
      command.args(["-c", "copy"]);
      // Matroska has no mov_text, MP4's subtitles. They become SubRip.
      for (i, codec) in source.subtitles.iter().enumerate() {
        if codec == "mov_text" {
          command.args([&format!("-c:s:{i}"), "srt"]);
        }
      }
    }
    None => {
      command.args(["-map_chapters", "-1", "-map_metadata", "-1"]);
    }
  }
  command.args(FFV1);
  if let Some(source) = extras.source {
    // setparams relabels the frames: y4m carries no colour tags, and -colorspace would
    // make ffmpeg convert instead.
    let colour = &source.colour;
    let tags: Vec<String> = [
      ("range", &colour.range),
      ("color_primaries", &colour.primaries),
      ("color_trc", &colour.transfer),
      ("colorspace", &colour.matrix),
    ]
    .iter()
    .filter_map(|(key, value)| value.as_ref().map(|v| format!("{key}={v}")))
    .collect();
    if !tags.is_empty() {
      command.args(["-vf", &format!("setparams={}", tags.join(":"))]);
    }
    if let Some(location) = &colour.chroma_location {
      command.args(["-chroma_sample_location", location]);
    }
  }
  command.args(["-f", "matroska"]).arg(path);
  let mut child = command
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::inherit())
    .spawn()
    .map_err(|e| fail("ffmpeg_missing", format!("could not run {ffmpeg}: {e}")))?;
  let stdin = child.stdin.take().expect("piped standard input");
  let process = Process {
    child: Some(child),
    task: format!("writing {}", shown.display()),
  };
  Ok((process, stdin))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn y4m_formats() {
    for native in NATIVE {
      assert_eq!(y4m_format(native), native);
    }
    let cases = [
      ("yuv420p10be", "yuv420p10le"),
      ("p010le", "yuv420p10le"),
      ("nv12", "yuv420p"),
      ("nv20le", "yuv422p10le"),
      ("yuv420p16le", "yuv420p12le"),
      ("yuv422p10be", "yuv422p10le"),
      ("yuyv422", "yuv422p"),
      ("y210le", "yuv422p10le"),
      ("yuv444p9le", "yuv444p10le"),
      ("gbrp", "yuv444p"),
      ("gbrp10le", "yuv444p10le"),
      ("rgb24", "yuv444p"),
      ("rgb48le", "yuv444p12le"),
      ("yuv411p", "yuv444p"),
      ("gray10le", "gray12le"),
      ("gray16le", "gray12le"),
      ("ya8", "gray"),
    ];
    for (source, expected) in cases {
      assert_eq!(y4m_format(source), expected, "{source}");
    }
  }

  #[test]
  fn ffprobe_next_to_ffmpeg() {
    assert_eq!(ffprobe_path("ffmpeg"), "ffprobe");
    // Joined with the system's separator, a backslash on Windows.
    let expected = Path::new("/opt/ffmpeg/bin").join("ffprobe");
    assert_eq!(ffprobe_path("/opt/ffmpeg/bin/ffmpeg"), expected.to_string_lossy());
    let expected = Path::new("tools").join("ffprobe.exe");
    assert_eq!(
      ffprobe_path(&Path::new("tools").join("ffmpeg.exe").to_string_lossy()),
      expected.to_string_lossy()
    );
    assert_eq!(ffprobe_path("ffmpeg.exe"), "ffprobe");
    assert_eq!(ffprobe_path("custom"), "ffprobe");
  }

  #[test]
  fn mastering_display_units() {
    let data: Value = serde_json::json!({
      "red_x": "35400/50000", "red_y": "14600/50000",
      "green_x": "8500/50000", "green_y": "39850/50000",
      "blue_x": "6550/50000", "blue_y": "2300/50000",
      "white_point_x": "15635/50000", "white_point_y": "16450/50000",
      "min_luminance": "1/10000", "max_luminance": "10000000/10000",
    });
    assert_eq!(
      mastering_display(&data).unwrap(),
      "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)"
    );
  }

  #[test]
  fn clocks() {
    assert_eq!(parse_clock("00:00:09.008000000"), Some(9.008));
    assert_eq!(parse_clock("01:02:03.5"), Some(3723.5));
    assert_eq!(parse_clock("x"), None);
  }

  #[test]
  fn frame_rates() {
    let steady: String = (0..100).map(|i| format!("{:.3}\n", i as f64 / 24.0)).collect();
    assert!(!irregular_timestamps(&steady, 24.0));
    // 24 and 60 fps stretches.
    let mixed: String = (0..50)
      .map(|i| i as f64 / 24.0)
      .chain((0..50).map(|i| 50.0 / 24.0 + i as f64 / 60.0))
      .map(|t| format!("{t:.3}\n"))
      .collect();
    assert!(irregular_timestamps(&mixed, 24.0));
  }
}
