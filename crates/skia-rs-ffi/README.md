# skia-rs-ffi

C FFI bindings for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **C-compatible API**: Use skia-rs from C, C++, or any language with C FFI
- **Opaque pointers**: Safe memory management patterns
- **Static/dynamic linking**: Build as `.a` or `.so`/`.dll`

## Building

```bash
# Build static library
cargo build -p skia-rs-ffi --release

# Output: target/release/libskia_rs_ffi.a
```

## Usage (C)

```c
#include "skia-rs.h"

int main() {
    // Create a surface
    sk_surface_t* surface = sk_surface_new_raster_n32_premul(800, 600);
    sk_canvas_t* canvas = sk_surface_get_canvas(surface);

    // Create a paint
    sk_paint_t* paint = sk_paint_new();
    sk_paint_set_color(paint, 0xFFFF6B35);
    sk_paint_set_antialias(paint, true);

    // Draw a rectangle
    sk_rect_t rect = { 100, 100, 700, 500 };
    sk_canvas_draw_rect(canvas, &rect, paint);

    // Cleanup
    sk_paint_delete(paint);
    sk_surface_delete(surface);

    return 0;
}
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
