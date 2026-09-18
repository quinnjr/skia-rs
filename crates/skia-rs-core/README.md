# skia-rs-core

Core types for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **Geometry primitives**: `Point`, `IPoint`, `Rect`, `IRect`, `RRect`
- **Transformations**: `Matrix` (3x3), `Matrix44` (4x4)
- **Color**: `Color`, `Color4f`, `ColorSpace`, ICC profiles
- **Pixel storage**: `ImageInfo`, `Pixmap`, `Bitmap`
- **Clipping**: `Region` with boolean operations

## Usage

```rust
use skia_rs_core::{Point, Rect, Matrix, Color};

// Create points and rectangles
let p = Point::new(10.0, 20.0);
let rect = Rect::from_xywh(0.0, 0.0, 100.0, 50.0);

// Transform with matrices
let m = Matrix::rotate(45.0);
let rotated_point = m.map_point(p);

// Work with colors
let red = Color::from_rgb(255, 0, 0);
let transparent_red = red.with_alpha(128);
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
