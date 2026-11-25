//! Generate backend conformance images.
//!
//! Renders every case listed in `conformance/README.md` and writes one PNG
//! per case (`<case>.png`) into the output directory. The committed raster
//! references in `conformance/reference/` are produced with:
//!
//! ```text
//! cargo run -p skia-rs-safe --example conformance_gen -- --backend raster --output-dir conformance/reference
//! ```
//!
//! Only the `raster` backend can render canvas scenes today. The GPU
//! backends (`wgpu`, `vulkan`, `opengl`, `metal`) expose device, pipeline and
//! command-buffer primitives but no canvas renderer yet, so requesting one
//! exits with an error instead of writing images that are not real GPU
//! output.

#[cfg(feature = "codec")]
#[path = "../tests/conformance/cases.rs"]
mod cases;

const USAGE: &str =
    "usage: conformance_gen [--backend raster] [--output-dir DIR] [--filter SUBSTRING]";

fn main() {
    #[cfg(feature = "codec")]
    std::process::exit(run());

    #[cfg(not(feature = "codec"))]
    {
        eprintln!("conformance_gen requires the 'codec' feature (enabled by default).");
        std::process::exit(2);
    }
}

#[cfg(feature = "codec")]
fn run() -> i32 {
    use skia_rs_codec::{ImageEncoder, ImageInfo, PngEncoder};
    use skia_rs_core::{AlphaType, ColorType};
    use std::path::PathBuf;

    let mut backend = String::from("raster");
    let mut output_dir = PathBuf::from("conformance_output");
    let mut filter: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next().unwrap_or_else(|| {
                eprintln!("missing value for {flag}\n{USAGE}");
                std::process::exit(2);
            })
        };
        match arg.as_str() {
            "--backend" => backend = value("--backend"),
            "--output-dir" => output_dir = PathBuf::from(value("--output-dir")),
            "--filter" => filter = Some(value("--filter")),
            "-h" | "--help" => {
                println!("{USAGE}");
                return 0;
            }
            other => {
                eprintln!("unknown argument '{other}'\n{USAGE}");
                return 2;
            }
        }
    }

    match backend.as_str() {
        "raster" => {}
        "wgpu" | "vulkan" | "opengl" | "metal" => {
            eprintln!(
                "backend '{backend}' cannot render conformance scenes yet: skia-rs-gpu \
                 provides device/pipeline primitives but no canvas renderer. Only \
                 '--backend raster' is supported."
            );
            return 1;
        }
        other => {
            eprintln!("unknown backend '{other}' (expected: raster)\n{USAGE}");
            return 2;
        }
    }

    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("cannot create {}: {e}", output_dir.display());
        return 1;
    }

    let size = usize::try_from(cases::SIZE).expect("SIZE is positive");
    let info = ImageInfo::new(
        cases::SIZE,
        cases::SIZE,
        ColorType::Rgba8888,
        AlphaType::Premul,
    );
    let encoder = PngEncoder::new();
    let mut written = 0;

    for case in cases::CASES {
        if filter.as_deref().is_some_and(|f| !case.name.contains(f)) {
            continue;
        }
        let pixels = cases::render_raster(case);
        let Some(image) = skia_rs_codec::Image::from_raster_data(&info, &pixels, size * 4) else {
            eprintln!("{}: could not wrap rendered pixels", case.name);
            return 1;
        };
        let png = match encoder.encode_bytes(&image) {
            Ok(png) => png,
            Err(e) => {
                eprintln!("{}: PNG encoding failed: {e}", case.name);
                return 1;
            }
        };
        let path = output_dir.join(format!("{}.png", case.name));
        if let Err(e) = std::fs::write(&path, png) {
            eprintln!("cannot write {}: {e}", path.display());
            return 1;
        }
        println!("wrote {}", path.display());
        written += 1;
    }

    println!("{written} image(s) written to {}", output_dir.display());
    0
}
