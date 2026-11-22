//! Raster conformance: every case in `conformance/README.md` must match the
//! committed reference image in `conformance/reference/` within tolerance.
//!
//! Regenerate the references after an intentional rendering change with:
//!
//! ```text
//! cargo run -p skia-rs-safe --example conformance_gen -- --backend raster --output-dir conformance/reference
//! ```

#![cfg(feature = "codec")]

#[path = "conformance/cases.rs"]
mod cases;

use skia_rs_codec::ImageInfo;
use skia_rs_core::{AlphaType, ColorType};
use std::path::PathBuf;

/// Maximum normalized RMSE (0.0 = identical, 1.0 = maximally different)
/// tolerated between a render and its reference. Raster output is pure
/// Rust f32 math, so the only expected drift is last-bit differences in
/// platform `libm` (trig used by arcs/conics); this is the README's
/// cross-backend, non-AA threshold.
const MAX_RMSE: f64 = 0.01;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/reference")
}

fn load_reference(name: &str) -> Vec<u8> {
    let path = reference_dir().join(format!("{name}.png"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {}: {e}. Regenerate with `cargo run -p skia-rs-safe \
             --example conformance_gen -- --output-dir conformance/reference`",
            path.display()
        )
    });
    assert!(
        !bytes.starts_with(b"version https://git-lfs"),
        "{} is a Git LFS pointer; run `git lfs pull`",
        path.display()
    );
    let image = skia_rs_codec::decode_image(&bytes)
        .unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()));
    assert_eq!(
        (image.width(), image.height()),
        (cases::SIZE, cases::SIZE),
        "{name}: reference has the wrong dimensions"
    );

    let info = ImageInfo::new(
        cases::SIZE,
        cases::SIZE,
        ColorType::Rgba8888,
        AlphaType::Premul,
    );
    let row_bytes = usize::try_from(cases::SIZE).unwrap() * 4;
    let mut pixels = vec![0u8; row_bytes * usize::try_from(cases::SIZE).unwrap()];
    assert!(
        image.read_pixels(&info, &mut pixels, row_bytes, 0, 0),
        "{name}: cannot read reference pixels"
    );
    pixels
}

/// Normalized root-mean-square error over all channels.
fn rmse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    #[allow(
        clippy::cast_precision_loss,
        reason = "pixel buffer length is far below 2^52"
    )]
    let n = a.len() as f64;
    (sum / n).sqrt() / 255.0
}

#[test]
fn conformance_raster_matches_references() {
    let mut failures = Vec::new();
    for case in cases::CASES {
        let rendered = cases::render_raster(case);
        let reference = load_reference(case.name);
        let error = rmse(&rendered, &reference);
        if error > MAX_RMSE {
            failures.push(format!("{}: RMSE {error:.5} > {MAX_RMSE}", case.name));
        }
    }
    assert!(
        failures.is_empty(),
        "raster output diverged from conformance/reference:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn conformance_case_names_are_unique() {
    let mut names: Vec<_> = cases::CASES.iter().map(|c| c.name).collect();
    names.sort_unstable();
    let len = names.len();
    names.dedup();
    assert_eq!(names.len(), len, "duplicate conformance case names");
}
