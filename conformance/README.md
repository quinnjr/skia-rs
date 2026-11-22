# Backend Conformance Testing

This directory holds the reference images used to check that rendering stays
consistent across platforms and, eventually, across backends.

## Structure

```
conformance/
├── README.md     # This file
└── reference/    # Reference images (<case>.png), rendered by the raster backend
```

The test cases themselves live in
[`crates/skia-rs-safe/tests/conformance/cases.rs`](../crates/skia-rs-safe/tests/conformance/cases.rs)
and are shared by:

- `crates/skia-rs-safe/examples/conformance_gen.rs`, which renders every case to `<case>.png`
- `crates/skia-rs-safe/tests/conformance.rs`, which renders every case and compares it with
  `reference/<case>.png`

## Backend Status

| Backend | Platforms | Builds in CI | Renders conformance images |
|---------|-----------|--------------|----------------------------|
| **Raster** | All | ✅ | ✅ |
| **WebGPU/WGPU** | All | ✅ | ❌ no canvas renderer yet |
| **Vulkan** | Linux, Windows | ✅ | ❌ no canvas renderer yet |
| **OpenGL** | Linux, Windows | ✅ | ❌ no canvas renderer yet |
| **Metal** | macOS | ✅ | ❌ no canvas renderer yet |

`skia-rs-gpu` currently provides device, pipeline, and command-buffer
primitives but no renderer that turns canvas draw calls into GPU work, so
`conformance_gen` only accepts `--backend raster`. Requesting a GPU backend
exits with an error rather than producing images that are not real GPU output.

## Running Conformance Tests

```bash
# Compare raster output against the committed references
cargo test -p skia-rs-safe --test conformance

# Render the images into a scratch directory for inspection
cargo run -p skia-rs-safe --example conformance_gen -- --output-dir conformance_output

# Render a subset
cargo run -p skia-rs-safe --example conformance_gen -- --output-dir conformance_output --filter circle
```

### Regenerating the References

After an intentional rendering change, regenerate the references, review the
images, and commit them (they are stored with Git LFS):

```bash
cargo run -p skia-rs-safe --example conformance_gen -- --backend raster --output-dir conformance/reference
```

### CI Integration

The `Conformance` jobs in `.github/workflows/ci.yml`:

1. Build the workspace with each backend's feature on every supported platform
2. Run the `conformance` tests, which compare raster output with `reference/` on that platform
3. For the raster backend, generate the images with `conformance_gen` and, on Linux,
   report the ImageMagick RMSE against `reference/`
4. Upload the generated images as artifacts for manual inspection

## Test Cases

All images are 200×200 on a white background.

### Basic Shapes
- `solid_rect` - Filled rectangle
- `stroked_rect` - Stroked rectangle
- `circle_fill` - Filled circle
- `circle_stroke` - Stroked circle
- `oval` - Filled oval/ellipse

### Lines
- `lines` - Multiple colored lines of increasing width

### Paths
- `path_triangle` - Simple triangle path
- `path_star` - Star shape (self-intersecting, winding fill)
- `path_bezier` - Stroked cubic Bézier curve
- `nested_paths` - Rectangle with hole (even-odd fill)
- `rounded_rect` - Rounded rectangle

### Compositing
- `overlapping_shapes` - Overlapping opaque shapes
- `alpha_blending` - Semi-transparent shapes
- `antialiased_shapes` - Anti-aliased circle, polygon, and line

### Stroke Properties
- `stroke_widths` - Stroke widths 1, 2, 4, 8, and 12

## Tolerance

`tests/conformance.rs` fails when the normalized RMSE between a render and its
reference exceeds **0.01**. Raster output is pure Rust `f32` arithmetic, so the
only expected cross-platform drift is last-bit differences in the platform
`libm` (trigonometry used for arcs and conics).

When GPU renderers exist, these thresholds are the intended targets:

| Comparison | RMSE Threshold |
|------------|----------------|
| Same backend, any platform | < 0.01 |
| Cross-backend (no AA) | < 0.01 |
| Cross-backend (with AA) | < 0.05 |
| GPU vs CPU | < 0.10 |

## Adding New Test Cases

1. Add a drawing function to `crates/skia-rs-safe/tests/conformance/cases.rs`
2. Register it in the `CASES` table
3. Regenerate the references (see above) and review the new image
4. Commit the new reference image

## Troubleshooting

If the `conformance` test fails:

1. Run `conformance_gen` and compare its output with `reference/` visually
2. Decide whether the difference is a bug or an intentional change
3. Fix the bug, or regenerate and commit the references if the change is intended

If the test reports a Git LFS pointer, run `git lfs pull`.
