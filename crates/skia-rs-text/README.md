# skia-rs-text

Text layout and rendering for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **Typeface**: Font face abstraction
- **Font**: Font with size and style properties
- **Text shaping**: via rustybuzz integration
- **TextBlob**: Positioned glyph runs
- **Paragraph**: Rich text layout with styling
- **Font manager**: Font enumeration and matching

## Usage

```rust
use skia_rs_core::Point;
use skia_rs_text::{Font, TextBlobBuilder, Typeface};
use std::sync::Arc;

// Load the platform's default sans-serif font (Segoe UI, Helvetica,
// DejaVu Sans, ...). `Typeface::default_typeface()` is a data-less
// placeholder that can measure text but draws no glyphs, so it is only a
// fallback for systems without fonts. `Typeface::from_data` loads any
// TTF/OTF file you ship yourself.
let typeface = Typeface::system_default().unwrap_or_else(Typeface::default_typeface);
let font = Font::new(Arc::new(typeface), 24.0);

// Map text to glyphs
let glyphs = font.text_to_glyphs("Hello, world!");

// Build a text blob
let mut builder = TextBlobBuilder::new();
builder.add_text("Hello, world!", &font, Point::zero());
let blob = builder.build().expect("non-empty blob");

// Draw with a canvas (skia-rs-canvas, `text` feature)
canvas.draw_text_blob(&blob, 100.0, 100.0, &paint);
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
