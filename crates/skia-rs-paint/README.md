# skia-rs-paint

Paint, shaders, and effects for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **Paint**: Color, style, stroke width, anti-aliasing
- **Blend modes**: All Porter-Duff and advanced blend modes
- **Shaders**: Linear, radial, sweep gradients, image shaders
- **Color filters**: Matrix, lighting, blend mode filters
- **Mask filters**: Blur, shader-based masks
- **Image filters**: Blur, drop shadow, morphology, displacement

## Usage

```rust
use skia_rs_paint::{Paint, Style, BlendMode, LinearGradient};
use skia_rs_core::{Color, Point};

// Create a paint with stroke style
let mut paint = Paint::new();
paint.set_color(Color::from_rgb(255, 107, 53));
paint.set_style(Style::Stroke);
paint.set_stroke_width(4.0);
paint.set_anti_alias(true);

// Add a gradient shader
let gradient = LinearGradient::new(
    Point::new(0.0, 0.0),
    Point::new(100.0, 0.0),
    &[Color4f::RED, Color4f::BLUE],
    None,
    TileMode::Clamp,
);
paint.set_shader(gradient);
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
