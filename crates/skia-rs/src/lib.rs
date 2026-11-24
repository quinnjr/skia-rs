//! # skia-rs
//!
//! A pure Rust implementation of Google's Skia 2D graphics library.
//!
//! This crate provides a comprehensive 2D graphics API for rendering shapes,
//! images, text, and more. It is designed to be a drop-in replacement for
//! the original Skia library while providing a safe, idiomatic Rust API.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use skia_rs::prelude::*;
//!
//! // Create a surface to draw on
//! let mut surface = Surface::new_raster_n32_premul(800, 600).unwrap();
//! let mut canvas = surface.canvas();
//!
//! // Create a paint for styling
//! let mut paint = Paint::default();
//! paint.set_color(Color::RED.into());
//! paint.set_anti_alias(true);
//!
//! // Draw a circle
//! canvas.draw_circle(Point::new(400.0, 300.0), 100.0, &paint);
//! ```
//!
//! ## Feature Flags
//!
//! This crate uses feature flags to control which components are included:
//!
//! - **default** = `["std", "codec", "svg"]` - Standard features for most use cases
//! - **std** - Standard library support (disable for `no_std` environments)
//! - **serde** - Serialization support via serde
//!
//! ### Image Codecs
//! - **codec** - Base image codec support
//! - **codec-png** - PNG encoding/decoding
//! - **codec-jpeg** - JPEG encoding/decoding
//! - **codec-webp** - WebP encoding/decoding
//! - **codec-gif** - GIF decoding
//! - **codec-avif** - AVIF encoding/decoding
//! - **codec-raw** - RAW image processing
//! - **codec-all** - All image codecs
//!
//! ### Specialized Modules
//! - **svg** - SVG parsing and rendering
//! - **pdf** - PDF document generation
//! - **text** - Advanced text rendering and shaping
//! - **skottie** - Lottie animation support
//!
//! ### GPU Backends
//! - **gpu** - Base GPU support
//! - **vulkan** - Vulkan backend
//! - **opengl** - OpenGL backend
//! - **metal** - Metal backend (macOS/iOS only)
//! - **wgpu-backend** / **webgpu** - wgpu backend (cross-platform)
//!
//! ### FFI
//! - **ffi** - C FFI bindings for interoperability
//!
//! ### Full Features
//! - **full** - Enables all features except platform-specific GPU backends
//!
//! ## Module Organization
//!
//! - [`core`] - Fundamental types: `Scalar`, `Point`, `Rect`, `Color`, `Matrix`
//! - [`path`] - Path geometry: `Path`, `PathBuilder`, `PathOps`, `PathEffect`
//! - [`paint`] - Styling: `Paint`, `Shader`, `BlendMode`, `Filter`
//! - [`canvas`] - Drawing: `Canvas`, `Surface`, `Picture`
//! - [`safe`] - High-level ergonomic API (recommended for most users)
//!
//! Optional modules (enabled via features):
//! - [`text`](https://docs.rs/skia-rs-text) - Typography: font loading, text shaping, layout
//! - [`codec`](https://docs.rs/skia-rs-codec) - Image I/O: PNG, JPEG, GIF, WebP, AVIF
//! - [`svg`](https://docs.rs/skia-rs-svg) - SVG parsing and rendering
//! - [`pdf`](https://docs.rs/skia-rs-pdf) - PDF document generation
//! - [`gpu`](https://docs.rs/skia-rs-gpu) - GPU rendering backends
//! - [`skottie`](https://docs.rs/skia-rs-skottie) - Lottie/Skottie animation
//! - [`ffi`](https://docs.rs/skia-rs-ffi) - C FFI bindings

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

// Re-export core crates
pub use skia_rs_canvas as canvas;
pub use skia_rs_core as core;
pub use skia_rs_paint as paint;
pub use skia_rs_path as path;
pub use skia_rs_safe as safe;

// Optional crate re-exports
#[cfg(feature = "text")]
#[cfg_attr(docsrs, doc(cfg(feature = "text")))]
pub use skia_rs_text as text;

#[cfg(feature = "codec")]
#[cfg_attr(docsrs, doc(cfg(feature = "codec")))]
pub use skia_rs_codec as codec;

#[cfg(feature = "svg")]
#[cfg_attr(docsrs, doc(cfg(feature = "svg")))]
pub use skia_rs_svg as svg;

#[cfg(feature = "pdf")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdf")))]
pub use skia_rs_pdf as pdf;

#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
pub use skia_rs_gpu as gpu;

#[cfg(feature = "skottie")]
#[cfg_attr(docsrs, doc(cfg(feature = "skottie")))]
pub use skia_rs_skottie as skottie;

#[cfg(feature = "ffi")]
#[cfg_attr(docsrs, doc(cfg(feature = "ffi")))]
pub use skia_rs_ffi as ffi;

/// Prelude module for convenient imports.
///
/// Import all commonly used types with:
/// ```rust
/// use skia_rs::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use skia_rs_core::{
        AlphaType, Color, Color4f, ColorSpace, ColorType, IPoint, IRect, ISize, ImageInfo, Matrix,
        Point, Rect, Scalar, Size,
    };

    // Path types
    pub use skia_rs_path::{FillType, Path, PathBuilder, PathDirection};

    // Paint types
    pub use skia_rs_paint::{BlendMode, Paint, Style};

    // Canvas types
    pub use skia_rs_canvas::{Canvas, ClipOp, SaveLayerRec, Surface};

    // (skia_rs_safe::prelude re-exports the same core/path/paint/canvas types
    // already re-exported above, so it is intentionally not globbed here.)

    // Optional module prelude re-exports
    #[cfg(feature = "text")]
    #[cfg_attr(docsrs, doc(cfg(feature = "text")))]
    pub use skia_rs_text::{Font, FontStyle, TextBlob, Typeface};

    #[cfg(feature = "codec")]
    #[cfg_attr(docsrs, doc(cfg(feature = "codec")))]
    pub use skia_rs_codec::{ImageDecoder, ImageEncoder, ImageFormat};

    #[cfg(feature = "svg")]
    #[cfg_attr(docsrs, doc(cfg(feature = "svg")))]
    pub use skia_rs_svg::SvgDom;

    #[cfg(feature = "pdf")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pdf")))]
    pub use skia_rs_pdf::PdfDocument;

    #[cfg(feature = "gpu")]
    #[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
    pub use skia_rs_gpu::GpuContext;

    #[cfg(feature = "skottie")]
    #[cfg_attr(docsrs, doc(cfg(feature = "skottie")))]
    pub use skia_rs_skottie::Animation;
}

/// Version information for the skia-rs library.
pub mod version {
    /// The major version number.
    pub const MAJOR: u32 = parse_component(env!("CARGO_PKG_VERSION_MAJOR"));
    /// The minor version number.
    pub const MINOR: u32 = parse_component(env!("CARGO_PKG_VERSION_MINOR"));
    /// The patch version number.
    pub const PATCH: u32 = parse_component(env!("CARGO_PKG_VERSION_PATCH"));

    /// The full version string.
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Returns the version as a tuple.
    #[inline]
    #[must_use]
    pub const fn as_tuple() -> (u32, u32, u32) {
        (MAJOR, MINOR, PATCH)
    }

    /// Parse a decimal version component at compile time.
    ///
    /// Previously each component was a hand-maintained `const` literal; if
    /// one fell behind the workspace version the constants desynced from
    /// `CARGO_PKG_VERSION` and the unit test comparing them against the
    /// string version failed. Deriving each component from its own cargo
    /// env var makes the drift impossible.
    const fn parse_component(s: &str) -> u32 {
        let bytes = s.as_bytes();
        let mut i = 0;
        let mut n: u32 = 0;
        while i < bytes.len() {
            let b = bytes[i];
            assert!(b >= b'0' && b <= b'9', "version component must be decimal");
            n = n * 10 + (b - b'0') as u32;
            i += 1;
        }
        n
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "exact re-export sanity checks on integral float literals"
)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version::VERSION, env!("CARGO_PKG_VERSION"));
        let parts: Vec<u32> = env!("CARGO_PKG_VERSION")
            .split('.')
            .map(|p| p.parse().unwrap())
            .collect();
        assert_eq!(version::as_tuple(), (parts[0], parts[1], parts[2]));
    }

    #[test]
    fn test_core_re_export() {
        let point = core::Point::new(10.0, 20.0);
        assert_eq!(point.x, 10.0);
        assert_eq!(point.y, 20.0);
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let color = Color::RED;
        assert_eq!(color.red(), 255);

        let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0);
        assert_eq!(rect.width(), 100.0);
    }
}
