# skia-rs

<div align="center">

[![Crates.io](https://img.shields.io/crates/v/skia-rs-safe.svg)](https://crates.io/crates/skia-rs-safe)
[![Documentation](https://docs.rs/skia-rs-safe/badge.svg)](https://docs.rs/skia-rs-safe)
[![License](https://img.shields.io/crates/l/skia-rs-safe.svg)](https://github.com/quinnjr/skia-rs#license)
[![Build Status](https://github.com/quinnjr/skia-rs/workflows/CI/badge.svg)](https://github.com/quinnjr/skia-rs/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](https://blog.rust-lang.org/2024/02/08/Rust-1.85.0.html)

**A 100% Rust implementation of the Skia 2D graphics library**

[Getting Started](#getting-started) •
[Features](#features) •
[Examples](#examples) •
[Documentation](https://docs.rs/skia-rs-safe) •
[Contributing](#contributing)

</div>

---

## Overview

**skia-rs** is a pure Rust reimplementation of [Google's Skia](https://skia.org/) 2D graphics library. It aims to provide complete API compatibility with the original Skia while leveraging Rust's safety, performance, and ergonomics.

### Why skia-rs?

- **Pure Rust** — No C/C++ dependencies, easy to build and deploy
- **Memory Safe** — Leverage Rust's ownership system, no use-after-free or buffer overflows
- **Fast** — Optimized software rasterizer, competitive with native Skia
- **Portable** — Works anywhere Rust compiles: desktop, mobile, WASM
- **API Compatible** — Familiar API for Skia users, smooth migration path

## Getting Started

Add `skia-rs-safe` to your `Cargo.toml`:

```toml
[dependencies]
skia-rs-safe = "0.1"
```

Or use individual crates for more control:

```toml
[dependencies]
skia-rs-core = "0.1"    # Core types: Point, Rect, Matrix, Color
skia-rs-path = "0.1"    # Path geometry and operations
skia-rs-paint = "0.1"   # Paint, shaders, filters
skia-rs-canvas = "0.1"  # Canvas and surface
```

### Quick Example

```rust
use skia_rs_canvas::Surface;
use skia_rs_core::{Color, Rect};
use skia_rs_paint::{Paint, Style};

fn main() {
    // Create a 800x600 RGBA surface
    let mut surface = Surface::new_raster_n32_premul(800, 600)
        .expect("Failed to create surface");

    // Get a canvas to draw on
    let canvas = surface.canvas();

    // Clear with a dark background
    canvas.clear(Color::from_rgb(18, 18, 26));

    // Create a paint with an orange color
    let mut paint = Paint::new();
    paint.set_anti_alias(true);
    paint.set_color(Color::from_rgb(255, 107, 53));
    paint.set_style(Style::Fill);

    // Draw a rounded rectangle
    let rect = Rect::from_xywh(100.0, 100.0, 600.0, 400.0);
    canvas.draw_round_rect(&rect, 20.0, 20.0, &paint);

    // Draw some text
    paint.set_color(Color::WHITE);
    canvas.draw_string("Hello, skia-rs!", 150.0, 320.0, &paint);
}
```

## Features

### Core Graphics
- ✅ **Geometry primitives** — Point, Rect, RRect, Matrix, Matrix44
- ✅ **Color management** — Color, Color4f, ColorSpace, ICC profiles
- ✅ **Path system** — Path construction, boolean operations, effects
- ✅ **Paint & styling** — Stroke, fill, blend modes, shaders

### Drawing Operations
- ✅ **Canvas API** — Full drawing operations with save/restore
- ✅ **Anti-aliased rendering** — High-quality edges
- ✅ **Clipping** — Rect and path-based clipping
- ✅ **Transformations** — Translate, rotate, scale, skew

### Text
- ✅ **Text shaping** — via rustybuzz
- ✅ **Font management** — TrueType/OpenType support
- ✅ **Rich text layout** — Paragraph styling, decorations

### Image Codecs
- ✅ **PNG** — Read/write support
- ✅ **JPEG** — Read/write support
- ✅ **GIF** — Read/write support
- ✅ **WebP** — Read/write support

### Effects
- ✅ **Shaders** — Linear, radial, sweep gradients, image shaders
- ✅ **Color filters** — Matrix, lighting, blend mode filters
- ✅ **Mask filters** — Blur, shader-based masks
- ✅ **Image filters** — Blur, drop shadow, morphology, displacement

### GPU Backends
- 🔄 **wgpu** — Cross-platform GPU rendering (in progress)
- 📋 **Vulkan** — Planned
- 📋 **OpenGL** — Planned
- 📋 **Metal** — Planned

## Crate Structure

```
skia-rs/
├── skia-rs-core     # Foundation: types, geometry, color
├── skia-rs-path     # Path geometry and operations
├── skia-rs-paint    # Paint, shaders, effects
├── skia-rs-canvas   # Canvas, surface, recording
├── skia-rs-text     # Text layout and rendering
├── skia-rs-codec    # Image encoding/decoding
├── skia-rs-gpu      # GPU backends
├── skia-rs-svg      # SVG parsing and rendering
├── skia-rs-pdf      # PDF generation
├── skia-rs-ffi      # C FFI bindings
└── skia-rs-safe     # High-level unified API
```

## Feature Flags

The `skia-rs-safe` crate supports feature flags:

| Feature | Default | Description |
|---------|---------|-------------|
| `codec` | ✅ | Image encoding/decoding (PNG, JPEG, GIF, WebP) |
| `svg`   | ✅ | SVG parsing and rendering |
| `pdf`   | ❌ | PDF document generation |
| `gpu`   | ❌ | GPU backends (wgpu) |

```toml
# All features
skia-rs-safe = { version = "0.1", features = ["codec", "svg", "pdf", "gpu"] }

# Minimal (no codecs or extras)
skia-rs-safe = { version = "0.1", default-features = false }
```

## Performance

skia-rs includes a comprehensive benchmark suite. Key performance highlights:

| Operation | Performance |
|-----------|-------------|
| Rectangle fill (1000×1000) | 224 ns |
| Clear (1000×1000) | 189 ns |
| Anti-aliased line | 502 ns |
| Path boolean union | 12.4 µs |
| SVG path parsing | 890 ns |

Run benchmarks locally:

```bash
cargo bench -p skia-rs-bench
```

## Examples

### Drawing Shapes

```rust
use skia_rs_canvas::Surface;
use skia_rs_core::{Color, Rect, Point};
use skia_rs_paint::{Paint, Style};
use skia_rs_path::PathBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut surface = Surface::new_raster_n32_premul(400, 400)?;
    let canvas = surface.canvas();
    canvas.clear(Color::WHITE);

    // Draw a circle
    let mut paint = Paint::new();
    paint.set_color(Color::from_rgb(66, 133, 244));
    paint.set_anti_alias(true);
    canvas.draw_circle(Point::new(200.0, 200.0), 80.0, &paint);

    // Draw a custom path
    let path = PathBuilder::new()
        .move_to(100.0, 300.0)
        .quad_to(200.0, 250.0, 300.0, 300.0)
        .build();

    paint.set_color(Color::from_rgb(234, 67, 53));
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(4.0);
    canvas.draw_path(&path, &paint);

    Ok(())
}
```

### Gradients

```rust
use skia_rs_paint::{Paint, LinearGradient, TileMode};
use skia_rs_core::{Point, Color4f};

let gradient = LinearGradient::new(
    Point::new(0.0, 0.0),
    Point::new(400.0, 0.0),
    &[Color4f::RED, Color4f::BLUE],
    None,
    TileMode::Clamp,
);

let mut paint = Paint::new();
paint.set_shader(gradient);
canvas.draw_rect(&rect, &paint);
```

### Image I/O

```rust
use skia_rs_codec::{Image, ImageFormat};

// Load an image
let image = Image::from_file("input.png")?;

// Draw it
canvas.draw_image(&image, 0.0, 0.0, None);

// Save to file
surface.save_png("output.png")?;
```

## C FFI

skia-rs provides C bindings for use from other languages:

```c
#include "skia-rs.h"

sk_surface_t* surface = sk_surface_new_raster_n32_premul(800, 600);
sk_canvas_t* canvas = sk_surface_get_canvas(surface);

sk_paint_t* paint = sk_paint_new();
sk_paint_set_color(paint, 0xFFFF6B35);
sk_paint_set_antialias(paint, true);

sk_rect_t rect = { 100, 100, 700, 500 };
sk_canvas_draw_rect(canvas, &rect, paint);

sk_paint_delete(paint);
sk_surface_delete(surface);
```

Build the FFI library:

```bash
cargo build -p skia-rs-ffi --release
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development

```bash
# Clone the repository
git clone https://github.com/quinnjr/skia-rs.git
cd skia-rs

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench -p skia-rs-bench

# Run fuzzing (requires nightly)
cd fuzz && cargo +nightly fuzz run fuzz_path
```

## Roadmap

See [TODO.md](TODO.md) for the complete development roadmap.

**v0.1.0** (Current):
- Core types and geometry ✅
- Path system with boolean operations ✅
- Full paint/shader/filter stack ✅
- Software rasterizer with AA ✅
- Text shaping ✅
- Image codecs (PNG, JPEG, GIF, WebP) ✅
- C FFI bindings ✅

**v0.2.0** (Planned):
- GPU rendering via wgpu
- SVG export
- Performance optimizations
- Extended font support

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Acknowledgments

- [Google Skia](https://skia.org/) — The original inspiration
- [rustybuzz](https://github.com/RazrFalcon/rustybuzz) — Text shaping
- [wgpu](https://github.com/gfx-rs/wgpu) — Cross-platform GPU abstraction

---

<div align="center">
Made with 🦀 by <a href="https://github.com/quinnjr">Joseph R. Quinn</a>
</div>
