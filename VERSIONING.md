# Versioning Strategy

This document outlines the versioning strategy for skia-rs and its relationship to the original Google Skia library.

## Version Format

skia-rs uses [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes
- **PRERELEASE**: Optional (e.g., `-alpha.1`, `-beta.2`, `-rc.1`)
- **BUILD**: Optional build metadata (e.g., `+skia.m128`)

## Relationship to Skia Releases

### Skia Version Tracking

Google Skia uses milestone-based versioning (e.g., `m128`, `m129`). skia-rs tracks Skia API compatibility through build metadata:

| skia-rs Version | Skia Milestone | API Compatibility |
|-----------------|----------------|-------------------|
| 0.1.0+skia.m128 | m128 | Partial |
| 0.2.0+skia.m129 | m129 | Partial |
| 1.0.0+skia.m135 | m135 | Full |

### Compatibility Tiers

1. **Full Compatibility** - All public APIs from the tracked Skia milestone are implemented
2. **Partial Compatibility** - Core APIs implemented, some advanced features pending
3. **Experimental** - API may change, not recommended for production

## Version Numbering Rules

### Pre-1.0 (Current Phase)

During the 0.x.y development phase:

- `0.1.x` - Initial release, core functionality
- `0.2.x` - GPU backend stabilization
- `0.3.x` - Advanced features (SVG, PDF, Skottie)
- `0.9.x` - Release candidates for 1.0

Breaking changes may occur in minor releases (0.x → 0.y).

### Post-1.0 (Stable)

After 1.0 release:

- **Major bumps (x.0.0)**: Breaking API changes
- **Minor bumps (0.x.0)**: New features, deprecations
- **Patch bumps (0.0.x)**: Bug fixes, performance improvements

## Crate Versioning

All crates in the workspace share the same version:

```toml
[workspace.package]
version = "0.1.0"
```

| Crate | Description |
|-------|-------------|
| `skia-rs-core` | Core types |
| `skia-rs-path` | Path geometry |
| `skia-rs-paint` | Paint & shaders |
| `skia-rs-canvas` | Canvas & surface |
| `skia-rs-text` | Text rendering |
| `skia-rs-gpu` | GPU backends |
| `skia-rs-codec` | Image codecs |
| `skia-rs-svg` | SVG support |
| `skia-rs-pdf` | PDF generation |
| `skia-rs-skottie` | Lottie animation |
| `skia-rs-ffi` | C FFI bindings |
| `skia-rs-safe` | High-level API |

## Deprecation Policy

1. **Deprecation Warning**: Feature marked `#[deprecated]` for at least one minor release
2. **Documentation**: Migration path documented in CHANGELOG
3. **Removal**: Deprecated features removed in next major release

```rust
#[deprecated(since = "0.2.0", note = "Use `new_method()` instead")]
pub fn old_method() { ... }
```

## Release Process

### Version Bump Checklist

1. Update `[workspace.package] version` in root `Cargo.toml`
2. Update `CHANGELOG.md` with release notes
3. Run `cargo test --workspace`
4. Run `cargo clippy --workspace`
5. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
6. Push tag: `git push origin v0.1.0`
7. Run publish script: `./scripts/publish.sh --execute`

### Git Tags

```
v0.1.0          # Release
v0.1.0-alpha.1  # Pre-release
v0.1.0-rc.1     # Release candidate
```

## Feature Flags and Versioning

Feature flags don't affect version numbers but are documented:

```toml
# Cargo.toml
[features]
default = ["std", "codec"]
std = []                    # Standard library (default)
codec = ["png", "jpeg"]     # Image codecs
gpu = ["wgpu-backend"]      # GPU rendering
full = ["std", "codec", "gpu", "svg", "pdf", "text", "skottie"]
```

## Skia Milestone Sync Schedule

| skia-rs Release | Target Skia | Timeline |
|-----------------|-------------|----------|
| 0.1.0 | m128 | Q1 2026 |
| 0.2.0 | m130 | Q2 2026 |
| 0.5.0 | m133 | Q3 2026 |
| 1.0.0 | m135+ | Q4 2026 |

## API Stability Markers

```rust
/// Stable API - will not change without major version bump.
#[stable]
pub fn draw_rect(...) { ... }

/// Unstable API - may change in minor releases.
#[unstable(feature = "gpu_compute")]
pub fn compute_shader(...) { ... }

/// Internal API - not for public use.
#[doc(hidden)]
pub fn internal_helper(...) { ... }
```

## MSRV (Minimum Supported Rust Version)

- Current MSRV: **1.94**
- MSRV bumps are considered **minor** version changes
- MSRV is specified in `Cargo.toml`:

```toml
[workspace.package]
rust-version = "1.94"
```

## Version Compatibility Matrix

| skia-rs | Rust MSRV | Skia API | wgpu |
|---------|-----------|----------|------|
| 0.1.x | 1.85 | m128 | 23.x |
| 0.2.x | 1.85+ | m130 | 24.x |
| 0.4.x | 1.85 | — | 23.x |
| 0.5.x (unreleased) | 1.94 | — | 23.x |
| 1.0.x | TBD | m135+ | TBD |

"—": the targeted Skia milestone is not tracked for these releases.

---

*This versioning strategy ensures predictable releases while tracking upstream Skia development.*
