fn main() {
  println!(
    "cargo:rustc-env=RABBIT_DENOISE_TARGET={}",
    std::env::var("TARGET").unwrap_or_default()
  );
  println!("cargo:rerun-if-changed=build.rs");
}
