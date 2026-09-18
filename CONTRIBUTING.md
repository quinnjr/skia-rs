# Contributing to skia-rs

Thank you for your interest in contributing to skia-rs! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Rust 1.85 or later (stable)
- Git
- For fuzzing: Rust nightly and cargo-fuzz

### Setup

```bash
# Clone the repository
git clone https://github.com/quinnjr/skia-rs.git
cd skia-rs

# Initialize the Skia submodule (for reference)
git submodule update --init

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace
```

## Development Workflow

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements
- `perf/description` - Performance improvements

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting, no code change
- `refactor` - Code restructuring
- `test` - Adding tests
- `perf` - Performance improvement
- `chore` - Maintenance tasks

Examples:
```
feat(path): add cubic bezier intersection

fix(canvas): correct clip stack restoration

docs(readme): add gradient example

perf(rasterizer): optimize rectangle fill with SIMD
```

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with tests
4. **Ensure** all checks pass:
   ```bash
   cargo fmt --all --check
   cargo clippy --workspace -- -D warnings
   cargo test --workspace
   ```
5. **Submit** a pull request

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- Follow Rust naming conventions
- Add doc comments to all public items
- Include examples in complex API documentation

## Architecture Guidelines

### Skia API Compatibility

The primary goal is API compatibility with Google's Skia. When implementing:

1. **Reference the original** - Check the `skia/` submodule
2. **Match signatures** - Function names and parameters should align
3. **Match semantics** - Behavior should be identical
4. **Document differences** - Note any deviations

### Crate Organization

```
skia-rs-core   → Foundation (no internal deps)
     ↓
skia-rs-path   → Geometry (depends: core)
     ↓
skia-rs-paint  → Styling (depends: core, path)
     ↓
skia-rs-canvas → Drawing (depends: core, path, paint)
```

Lower-level crates should not depend on higher-level ones.

### Memory & Performance

- Use `#[repr(C)]` for FFI-compatible structs
- Derive `bytemuck::{Pod, Zeroable}` for zero-copy types
- Use `SmallVec` for small collections
- Avoid allocations in hot paths
- Prefer `&self` over `&mut self`

### Error Handling

- Use `thiserror` for error types
- Prefer `Option<T>` for simple failures (matching Skia patterns)
- Never panic in library code
- Use `debug_assert!` for development checks

## Testing

### Unit Tests

Place tests in the same file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // ...
    }
}
```

### Property Testing

Use proptest for numeric/geometric code:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn matrix_inverse_roundtrip(
        a in -1000.0f32..1000.0,
        b in -1000.0f32..1000.0,
    ) {
        // ...
    }
}
```

### Benchmarks

Add benchmarks for performance-critical code:

```bash
cargo bench -p skia-rs-bench
```

### Fuzzing

Run fuzz tests for robustness:

```bash
cd fuzz
cargo +nightly fuzz run fuzz_path -- -max_total_time=60
```

## Documentation

### Rustdoc

All public items must have doc comments:

```rust
/// A 2D point with floating-point coordinates.
///
/// Corresponds to Skia's `SkPoint`.
///
/// # Examples
///
/// ```
/// use skia_rs_core::Point;
/// let p = Point::new(10.0, 20.0);
/// assert_eq!(p.length(), (500.0_f32).sqrt());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: Scalar,
    pub y: Scalar,
}
```

### README Updates

Update crate READMEs when adding features.

## Release Process

Releases are managed by maintainers. The process:

1. Update version in `Cargo.toml` (workspace)
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag after merge: `git tag v0.x.y`
5. Publish to crates.io

## Getting Help

- **Issues** - Bug reports and feature requests
- **Discussions** - Questions and ideas
- **Discord** - Real-time chat (coming soon)

## Recognition

Contributors are recognized in:
- Git history
- Release notes
- README acknowledgments

Thank you for contributing to skia-rs! 🦀
