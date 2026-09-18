# skia-rs-gpu

GPU backends for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **wgpu backend**: Cross-platform GPU rendering (default)
- **Vulkan backend**: Native Vulkan support (planned)
- **OpenGL backend**: OpenGL ES 2.0+ / OpenGL 3.0+ (planned)

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wgpu-backend` | ✅ | wgpu cross-platform backend |
| `vulkan` | ❌ | Native Vulkan backend |
| `opengl` | ❌ | OpenGL/OpenGL ES backend |

## Usage

```rust
use skia_rs_gpu::{WgpuContext, WgpuSurface};

// Create a GPU context
let context = WgpuContext::new()?;

// Create a GPU-backed surface
let surface = WgpuSurface::new(&context, 800, 600)?;

// Draw on the canvas (same API as CPU)
let canvas = surface.canvas();
canvas.clear(Color::WHITE);
canvas.draw_rect(&rect, &paint);

// Present to screen
surface.present();
```

## Status

⚠️ GPU backends are currently in development. The wgpu backend is functional for basic operations.

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
