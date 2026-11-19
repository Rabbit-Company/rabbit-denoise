//! Removes film grain from video and keeps the picture.

#![allow(clippy::needless_range_loop)]

pub mod bm3d;
pub mod dct;
pub mod gpu;
pub mod motion;
pub mod noise;
pub mod pipeline;
pub mod plane;
pub mod scenes;

pub use pipeline::{Denoiser, Device, FrameModels, Measured, Measurer, Settings};
pub use plane::{Frame, Plane};
