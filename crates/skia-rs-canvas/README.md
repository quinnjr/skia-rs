# skia-rs-canvas

Canvas, surface, and recording for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **Surface**: CPU-backed drawing targets
- **Canvas**: Full drawing API with save/restore
- **Rasterizer**: Anti-aliased software rendering
- **Transformations**: Translate, rotate, scale, skew
- **Clipping**: Rect and path-based clipping
- **Picture**: Recording and playback

## Usage

```rust
use skia_rs_canvas::Surface;
use skia_rs_core::{Color, Rect, Point};
use skia_rs_paint::Paint;

// Create a surface
let mut surface = Surface::new_raster_n32_premul(800, 600).unwrap();
let canvas = surface.canvas();

// Clear background
canvas.clear(Color::WHITE);

// Draw shapes
let paint = Paint::new();
canvas.draw_rect(&Rect::from_xywh(10.0, 10.0, 100.0, 50.0), &paint);
canvas.draw_circle(Point::new(200.0, 100.0), 50.0, &paint);

// Save and restore state
canvas.save();
canvas.translate(100.0, 100.0);
canvas.rotate(45.0);
canvas.draw_rect(&rect, &paint);
canvas.restore();
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
