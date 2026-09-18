# skia-rs-path

Path geometry and operations for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **Path construction**: `Path`, `PathBuilder` with fluent API
- **Path operations**: Boolean union, intersect, difference, xor
- **Path effects**: Dash, corner, discrete, trim effects
- **SVG parsing**: Parse SVG path data strings
- **Path measurement**: Length, position along path

## Usage

```rust
use skia_rs_path::{PathBuilder, PathOps};

// Build a path with fluent API
let path = PathBuilder::new()
    .move_to(0.0, 0.0)
    .line_to(100.0, 0.0)
    .quad_to(150.0, 50.0, 100.0, 100.0)
    .close()
    .build();

// Parse SVG path data
let heart = PathBuilder::from_svg("M 10,30 A 20,20 0,0,1 50,30 ...").build();

// Boolean operations
let union = PathOps::union(&path1, &path2);
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
