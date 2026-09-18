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
use skia_rs_text::{Font, Typeface, TextBlobBuilder};

// Create a font
let typeface = Typeface::default_typeface();
let font = Font::new(typeface, 24.0);

// Shape text into glyphs
let glyphs = font.text_to_glyphs("Hello, world!");

// Build a text blob
let mut builder = TextBlobBuilder::new();
builder.alloc_run(&font, glyphs.len(), 0.0, 0.0);
let blob = builder.build();

// Draw with canvas
canvas.draw_text_blob(&blob, 100.0, 100.0, &paint);
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
