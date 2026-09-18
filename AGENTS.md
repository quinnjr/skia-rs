<!-- converted from Cursor rules -->

## Cursor rule: `.cursor/rules/api-design.mdc`

_API design principles and patterns for skia-rs_

Applies to: `["**/*.rs"]`

# API Design Principles

## Skia API Compatibility

The primary goal is API compatibility with Skia. When implementing a feature:

1. **Reference the original**: Check `skia/` submodule for the C++ implementation
2. **Match signatures**: Function names and parameter order should match Skia
3. **Match semantics**: Behavior should be identical to Skia
4. **Document differences**: If Rust requires a different approach, document it

## Builder Pattern

Use builders for complex object construction:

```rust
pub struct PathBuilder {
    path: Path,
    last_move: Option<Point>,
}

impl PathBuilder {
    pub fn new() -> Self { ... }
    pub fn move_to(&mut self, x: Scalar, y: Scalar) -> &mut Self { ... }
    pub fn line_to(&mut self, x: Scalar, y: Scalar) -> &mut Self { ... }
    pub fn build(self) -> Path { ... }
}
```

## Method Chaining

Support fluent interfaces where appropriate:

```rust
impl Paint {
    pub fn set_color(&mut self, color: Color4f) -> &mut Self {
        self.color = color;
        self
    }

    pub fn set_style(&mut self, style: Style) -> &mut Self {
        self.style = style;
        self
    }
}

// Usage:
paint.set_color(Color4f::RED)
     .set_style(Style::Stroke)
     .set_stroke_width(2.0);
```

## Const Correctness

- Use `const fn` where possible
- Define common constants as associated constants

```rust
impl Rect {
    pub const EMPTY: Self = Self { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };

    #[inline]
    pub const fn new(left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Self {
        Self { left, top, right, bottom }
    }
}
```

## Common Patterns

### Cloning vs Referencing

- Prefer references for read-only access
- Clone only when ownership transfer is needed
- Use `Cow<T>` when clone-on-write is beneficial

### Option Patterns

```rust
// Skia-style: return Option for fallible operations
pub fn invert(&self) -> Option<Matrix> {
    let det = self.determinant();
    if det == 0.0 {
        return None;
    }
    Some(self.compute_inverse(det))
}
```

### Iteration

```rust
// Provide iterators for collections
impl Path {
    pub fn iter(&self) -> PathIter<'_> {
        PathIter { path: self, index: 0 }
    }
}

// Make types iterable
impl<'a> IntoIterator for &'a Path {
    type Item = PathElement;
    type IntoIter = PathIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
```


## Cursor rule: `.cursor/rules/architecture.mdc`

_Project architecture and crate structure for skia-rs_

Applies to: `["**/*.rs", "**/Cargo.toml"]`

# Skia-RS Architecture

Skia-RS is a 100% Rust implementation of Google's Skia 2D graphics library, designed for API compatibility with the original C++ Skia library. The project provides both a native Rust API and C FFI bindings for cross-language interoperability.

## Workspace Structure

```
skia-rs/
├── crates/
│   ├── skia-rs-core/     # Foundation: Scalar, Point, Rect, Color, Matrix, ImageInfo
│   ├── skia-rs-path/     # Path geometry: Path, PathBuilder, PathOps, PathEffects
│   ├── skia-rs-paint/    # Styling: Paint, Shaders, BlendModes, Filters
│   ├── skia-rs-canvas/   # Drawing: Canvas, Surface, Picture recording
│   ├── skia-rs-text/     # Typography: Font loading, text shaping, layout
│   ├── skia-rs-gpu/      # GPU backends: Vulkan, OpenGL, WebGPU
│   ├── skia-rs-codec/    # Image I/O: PNG, JPEG, GIF, WebP
│   ├── skia-rs-svg/      # SVG support: parsing and rendering
│   ├── skia-rs-pdf/      # PDF generation
│   ├── skia-rs-ffi/      # C API bindings for FFI
│   ├── skia-rs-safe/     # High-level ergonomic Rust API
│   └── skia-rs-bench/    # Performance benchmarks
├── fuzz/                 # Fuzz testing with cargo-fuzz/libFuzzer
├── skia/                 # Official Skia submodule (reference)
└── TODO.md              # Development roadmap
```

## Crate Dependencies

```
skia-rs-core (no internal deps)
    ↓
skia-rs-path (depends on: core)
    ↓
skia-rs-paint (depends on: core, path)
    ↓
skia-rs-canvas (depends on: core, path, paint)
    ↓
skia-rs-text (depends on: core, path, paint)
    ↓
skia-rs-gpu (depends on: core, path, paint, canvas)
skia-rs-codec (depends on: core)
skia-rs-svg (depends on: core, path, paint, canvas)
skia-rs-pdf (depends on: core, path, paint, canvas, text)
    ↓
skia-rs-ffi (depends on: all above)
skia-rs-safe (depends on: all above, re-exports)
```

## File Organization

Each crate should follow this structure:

```
crates/skia-rs-{name}/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Module declarations and re-exports
│   ├── {feature}.rs    # Feature implementations
│   └── ...
```

## Re-exports

- `lib.rs` should re-export all public types
- Use `pub use module::*;` for complete re-exports
- Group related items in modules

```rust
// lib.rs
pub mod color;
pub mod geometry;
pub mod matrix;

pub use color::*;
pub use geometry::*;
pub use matrix::*;
```


## Cursor rule: `.cursor/rules/benchmarking.mdc`

_Benchmarking guidelines for skia-rs performance testing_

Applies to: `["**/bench/**", "**/benches/**", "crates/skia-rs-bench/**"]`

# Benchmarking

## Using the Benchmark System

```bash
# Run all benchmarks
cargo bench -p skia-rs-bench

# Run specific suite
cargo bench -p skia-rs-bench --bench core_benchmarks

# Quick validation
cargo bench -p skia-rs-bench -- --test
```

## Writing Benchmarks

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_example(c: &mut Criterion) {
    let mut group = c.benchmark_group("Example");

    group.bench_function("operation", |b| {
        b.iter(|| {
            // Use black_box to prevent optimization
            black_box(some_operation())
        })
    });

    group.finish();
}

criterion_group!(benches, bench_example);
criterion_main!(benches);
```

## Performance Targets

- Core operations (Point, Rect, Color): < 10ns
- Matrix operations: < 100ns
- Path operations: < 1µs per segment
- Canvas drawing: comparable to native Skia

## Best Practices

- Use `std::hint::black_box` to prevent compiler optimizations
- Use deterministic RNG for reproducible benchmarks
- Group related benchmarks together
- Include baseline comparisons where possible


## Cursor rule: `.cursor/rules/code-style.mdc`

_Rust code style and conventions for skia-rs_

Applies to: `["**/*.rs"]`

# Code Style & Conventions

## Rust Edition & Version

- **Edition**: 2024
- **MSRV**: 1.85
- All code must compile on stable Rust

## Naming Conventions

- Follow Skia's naming where possible for API compatibility
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and traits
- Prefix internal/private items with underscore only when necessary
- Crate names: `skia-rs-{module}` (hyphenated)
- Module imports: `skia_rs_{module}` (underscored)

## Type Aliases

```rust
// Core type alias - matches Skia's SkScalar
pub type Scalar = f32;
```

## Documentation

- All public items MUST have doc comments
- Use `//!` for module-level documentation
- Include examples in doc comments for complex APIs
- Reference corresponding Skia types/functions in docs

```rust
/// A 2D point with floating-point coordinates.
///
/// Corresponds to Skia's `SkPoint`.
///
/// # Examples
/// ```
/// use skia_rs_core::Point;
/// let p = Point::new(10.0, 20.0);
/// assert_eq!(p.length(), (500.0_f32).sqrt());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Point {
    pub x: Scalar,
    pub y: Scalar,
}
```

## Error Handling

- Use `thiserror` for error types
- Prefer `Option<T>` over `Result<T, E>` for simple failure cases (matching Skia patterns)
- Never panic in library code except for invariant violations
- Use `debug_assert!` for development-time checks

## Memory & Performance

- Use `#[repr(C)]` for FFI-compatible structs
- Derive `bytemuck::{Pod, Zeroable}` for types that need zero-copy operations
- Use `SmallVec` for small, stack-allocated collections
- Prefer `&self` over `&mut self` where possible
- Avoid allocations in hot paths

```rust
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Color(pub u32);
```

## Inline Hints

- Use `#[inline]` for small, frequently-called methods
- Use `#[inline(always)]` sparingly, only for critical hot paths
- Let the compiler decide for complex functions

```rust
impl Point {
    #[inline]
    pub const fn new(x: Scalar, y: Scalar) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn length(&self) -> Scalar {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
```


## Cursor rule: `.cursor/rules/commands.mdc`

_Common workspace commands for skia-rs development_

Applies to: `["**/*"]`

# Workspace Commands

## Development

```bash
# Check all crates
cargo check --workspace

# Build debug
cargo build --workspace

# Build release
cargo build --release --workspace

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench -p skia-rs-bench

# Generate docs
cargo doc --workspace --no-deps --open

# Format code
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings
```

## Git Workflow

- Keep the `skia/` submodule updated for reference
- Commit messages should reference Skia types/functions being implemented
- Use feature branches for new implementations
- Update `TODO.md` as features are completed

## Conventional Commits

Commit messages must follow Conventional Commits:

```
feat(core): add Matrix inversion
fix(path): correct cubic curve bounds calculation
docs: update API documentation
refactor(canvas): simplify save/restore stack
test(paint): add property tests for color conversion
```

## Pre-commit Hooks

The project uses `cargo-husky` for git hooks:

- **pre-commit**: Runs `cargo fmt --check` and `cargo clippy`
- **pre-push**: Runs full test suite
- **commit-msg**: Validates Conventional Commits format


## Cursor rule: `.cursor/rules/ffi.mdc`

_FFI (Foreign Function Interface) guidelines for skia-rs-ffi_

Applies to: `["crates/skia-rs-ffi/**"]`

# FFI Guidelines

## C API Design

- All FFI functions in `skia-rs-ffi`
- Use opaque pointers for complex types
- Follow Skia's C API naming: `sk_{type}_{method}`
- Always check for null pointers

```rust
// skia-rs-ffi/src/lib.rs

/// Create a new paint object.
///
/// # Safety
/// Returns a valid pointer that must be freed with `sk_paint_delete`.
#[no_mangle]
pub unsafe extern "C" fn sk_paint_new() -> *mut Paint {
    Box::into_raw(Box::new(Paint::new()))
}

/// Delete a paint object.
///
/// # Safety
/// `paint` must be a valid pointer returned by `sk_paint_new`.
#[no_mangle]
pub unsafe extern "C" fn sk_paint_delete(paint: *mut Paint) {
    if !paint.is_null() {
        drop(Box::from_raw(paint));
    }
}

/// Set the paint color.
///
/// # Safety
/// `paint` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn sk_paint_set_color(paint: *mut Paint, color: u32) {
    if let Some(p) = paint.as_mut() {
        p.set_color32(Color(color));
    }
}
```

## Type Mappings

| Skia C++ | Rust | C FFI |
|----------|------|-------|
| `SkScalar` | `Scalar` (f32) | `float` |
| `SkPoint` | `Point` | `sk_point_t` |
| `SkRect` | `Rect` | `sk_rect_t` |
| `SkColor` | `Color` | `uint32_t` |
| `SkMatrix` | `Matrix` | `sk_matrix_t` |
| `SkPath*` | `*mut Path` | `sk_path_t*` |
| `SkPaint*` | `*mut Paint` | `sk_paint_t*` |
| `SkCanvas*` | `*mut Canvas` | `sk_canvas_t*` |

## Safety Requirements

- All FFI functions must be marked `unsafe`
- Document safety requirements in doc comments
- Check for null pointers before dereferencing
- Use `#[repr(C)]` for all FFI-visible structs


## Cursor rule: `.cursor/rules/fuzzing.mdc`

_Fuzzing setup and guidelines for skia-rs_

Applies to: `["fuzz/**"]`

# Fuzzing

## Setup

```bash
# Install nightly toolchain and cargo-fuzz
rustup install nightly
cargo install cargo-fuzz
```

## Running Fuzz Tests

```bash
# Run a fuzz target
cd fuzz
cargo +nightly fuzz run fuzz_point

# Run with time limit
cargo +nightly fuzz run fuzz_matrix -- -max_total_time=60

# List all targets
cargo +nightly fuzz list
```

## Available Fuzz Targets

- `fuzz_point` - Point operations
- `fuzz_rect` - Rectangle operations
- `fuzz_matrix` - Matrix transformations
- `fuzz_color` - Color conversions
- `fuzz_path` - Path construction
- `fuzz_path_builder` - PathBuilder shapes
- `fuzz_paint` - Paint configuration
- `fuzz_canvas` - Canvas operations

## Writing Fuzz Targets

```rust
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct MyInput {
    value: f32,
    flag: bool,
}

fuzz_target!(|input: MyInput| {
    // Skip invalid inputs early
    if !input.value.is_finite() {
        return;
    }

    // Test code that should not panic
    let result = my_function(input.value);

    // Assert invariants
    assert!(result.is_valid());
});
```

## Best Practices

- Use `Arbitrary` derive for structured input
- Validate and skip invalid inputs early
- Limit resource usage (iteration counts, sizes)
- Assert invariants, not expected failures
- Keep fuzz targets focused on specific functionality
- The `fuzz` crate uses `edition = "2021"` due to `libfuzzer-sys` requirements


## Cursor rule: `.cursor/rules/gpu.mdc`

_GPU backend guidelines for skia-rs-gpu_

Applies to: `["crates/skia-rs-gpu/**"]`

# GPU Backend Guidelines

## Feature Flags

```toml
[features]
default = ["wgpu-backend"]
vulkan = ["dep:ash"]
opengl = ["dep:glow"]
wgpu-backend = ["dep:wgpu"]
```

## Backend Abstraction

```rust
pub trait GpuBackend: Send + Sync {
    fn create_surface(&self, info: &ImageInfo) -> Result<GpuSurface, GpuError>;
    fn flush(&self);
}
```

## Supported Backends

- **wgpu**: Cross-platform WebGPU abstraction (default)
- **Vulkan**: Low-level via `ash` crate
- **OpenGL**: Via `glow` crate
- **Metal**: macOS/iOS (planned)

## Resource Management

- Use GPU resource pools for frequently allocated objects
- Implement proper cleanup in `Drop` traits
- Handle device lost scenarios gracefully


## Cursor rule: `.cursor/rules/testing.mdc`

_Testing guidelines and patterns for skia-rs_

Applies to: `["**/*.rs", "**/tests/**"]`

# Testing Guidelines

## Unit Tests

- Place tests in the same file as the code
- Test edge cases and error conditions
- Use property-based testing with `proptest` for numeric code

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_addition() {
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1 + p2, Point::new(4.0, 6.0));
    }
}
```

## Property Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn matrix_identity_preserves_point(x in -1000.0f32..1000.0, y in -1000.0f32..1000.0) {
        let p = Point::new(x, y);
        let result = Matrix::IDENTITY.map_point(p);
        prop_assert!((result.x - p.x).abs() < 1e-6);
        prop_assert!((result.y - p.y).abs() < 1e-6);
    }
}
```

## Conformance Testing

- Compare output against Skia reference implementation
- Use the `skia/` submodule for reference
- Document any intentional behavioral differences

## Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p skia-rs-core

# Run with output
cargo test --workspace -- --nocapture
```

