# skia-rs-svg

SVG parsing and rendering for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **SVG DOM**: Parse SVG documents
- **Elements**: rect, circle, ellipse, path, line, polyline, polygon
- **Attributes**: fill, stroke, transform, opacity
- **Rendering**: Render SVG to canvas

## Usage

```rust
use skia_rs_svg::SvgDom;

// Parse an SVG file
let svg = SvgDom::from_file("icon.svg")?;

// Get the size
let (width, height) = svg.size();

// Render to canvas
svg.render(canvas);
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
