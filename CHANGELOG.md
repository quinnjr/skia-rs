# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-08-19

### Changed

- **Breaking**: `skia-rs-codec` no longer enables the `webp` feature by
  default, so `libwebp-sys` (a native C build) is only compiled when WebP
  support is explicitly requested. Enable it with the `webp` feature on
  `skia-rs-codec`, or `codec-webp` / `codec-all` on `skia-rs` /
  `skia-rs-safe`. Without the feature, `WebpDecoder` / `WebpEncoder` remain
  available but return `CodecError::Unsupported`.
- Project metadata now points at the new repository home
  (`github.com/quinnjr/skia-rs`) and author (Joseph R. Quinn) across crate
  manifests, READMEs, licenses, and the generated C headers.

## [0.3.0] - 2026-07-12

This release lands a large conformance audit against upstream C++ Skia. It
contains **breaking** behavior and C-ABI changes across most crates (marked
**Breaking** below); consumers should treat it as a minor-version bump
despite the 0.x line, and C consumers must rebuild against the regenerated
headers.

It also folds the node/python bindings into the workspace, hardens CI
(formatting baseline, Windows N32/BGRA coverage, toolchain setup), and
enforces `clippy::pedantic` + `clippy::nursery` with `-D warnings`
workspace-wide (including the `--all-features` examples).

**Known limitations shipped in this release:**
- **Windows raster surfaces:** `ColorType::n32()` now selects `Bgra8888` on
  Windows (matching Skia's build-config default), but the raster backend
  currently supports only RGBA-order 8888 buffers, so `Surface::new_raster`
  with an N32 info (e.g. `ImageInfo::new_n32_premul`) returns `None` on
  Windows targets. Construct surfaces with an explicit `Rgba8888` color type
  on Windows until BGRA raster buffers are supported.
- **PDF Type0/CID fonts:** the `/DescendantFonts` structure is emitted, but
  drawing text through a Type0 font fails closed rather than producing output
  (see the skia-rs-pdf section).

### Changed (skia-rs-canvas — conformance audit)
- `Surface::new_raster` now accepts `Bgra8888` surfaces (previously it
  returned `None` for anything but RGBA-order 8888). This makes N32 raster
  surfaces work on Windows, where `ColorType::n32()` selects BGRA. The
  backing buffer is still stored in RGBA order (the hot path is unchanged);
  the declared BGRA byte order is produced only on `make_image_snapshot`
  readback (R/B swapped). **Caveat:** `Surface::pixels()`/`pixels_mut()`
  expose the raw physical buffer, so for a BGRA surface they return
  RGBA-ordered bytes — use `make_image_snapshot` to get declared-order
  pixels.
- The raster pixel pipeline now stores **premultiplied** pixels end to end
  (`SkSurface_Raster` with `AlphaType::Premul`). Paint colors are
  premultiplied once at the paint→device boundary; blends operate on
  premultiplied values; `clear` premultiplies its color. Translucent draws
  now store premultiplied bytes (e.g. 50% red stores `r == a`, not
  `r == 255`), a visible byte-level change from the previous straight-alpha
  storage.
- `blend_colors` delegates every non-trivial blend mode to
  `skia-rs-paint`'s `BlendMode::apply` on premultiplied values — the full
  separable/non-separable set (Multiply, Overlay, Darken, …, Luminosity) is
  now correct instead of silently falling back to SrcOver.
- Byte-domain color math uses Skia's rounded `SkMulDiv255Round` everywhere,
  including the AVX2 chunk **and** its scalar remainder, and the
  premultiply/unpremultiply span helpers.
- The SIMD SrcOver span blitters (scalar, AVX2, NEON) now blend
  premultiplied source and agree bit-for-bit with `blend_colors`.
- Fixed an out-of-bounds read/write and double-blending bug in the AArch64
  NEON span blitter (`fill_span_blend_neon`): it now processes 8 px per
  iteration with matched `vld4_u8`/`vst4_u8` (32 B) plus a scalar tail.
- `Surface::new_raster` rejects color types the RGBA buffer cannot represent
  (only RGBA-order 8888 is supported) instead of silently mislabeling them.
  Image snapshots carry the surface's true color/alpha type and deliver
  unpremultiplied bytes when the surface's alpha type is `Unpremul`.
- `Canvas::save`/`save_layer` (and `RecordingCanvas::save`) now return the
  save count **before** the save (`SkCanvas::save` returns
  `getSaveCount() - 1`; initial count is 1), not the post-save count.
- `Canvas::restore_to_count` clamps its argument to 1 like
  `SkCanvas::restoreToCount` (passing 0 previously looped forever).
- `DrawCommand::ClipRect`/`ClipPath` record the `ClipOp` and replay it;
  Difference clips recorded into pictures no longer replay as Intersect.
- Picture playback of `SetMatrix` composes with the CTM at playback start
  (`setMatrix(initialCTM * recorded)`) instead of replacing it.
- `Canvas::draw_picture` (and `DrawCommand::DrawPicture` playback) honor the
  optional paint via an implicit `save_layer`.
- Non-AA `clip_path` scan-converts the actual path into a region
  (`SkRegion::setPath` semantics) instead of clipping to the path bounds.
- AA `ClipOp::Difference` on a rect subtracts only the rect: the existing
  clip is retained and coverage inside the rect is zeroed (previously the
  clip was first intersected with the rect, destroying everything outside).
- `clip_path_with_op` Difference rasterizes the actual path coverage and
  subtracts it from the existing clip in every clip-state combination
  (Rect/Region/Mask/RegionAndMask, AA and non-AA); no silent no-ops.
- Non-AA clip rects round to nearest (Skia `rect.round()`), not round-out,
  and clip containment tests sample pixel centers.
- `fill_rect` under a non-axis-aligned CTM scan-converts the mapped quad as
  a path; only the axis-aligned case uses `map_rect` (previously the quad's
  bounding box was filled).
- Stroke geometry honors `stroke_width`: positive widths build the stroke
  outline via `skia-rs-path`'s `stroke_to_fill` (cap/join/miter from the
  paint) and fill it; width 0 remains a hairline. `StrokeAndFill` fills the
  union of fill and stroke geometry exactly once (no double blend of the
  overlap with translucent paint).
- Circles map through the full CTM: rotation/skew/non-uniform scale convert
  to a conic path and go through the path pipeline (ellipse under
  non-uniform scale); `fill_circle` emits each row exactly once (no double
  blending with translucent paint).
- `fill_path` honors `paint.shader()`; shaders are sampled in **local**
  space (device point through the inverse CTM) and go through the clipped
  blitter. The shader branch of `fill_rect` respects the clip.
- `paint.is_anti_alias()` routes fills through the AA scanline filler,
  which no longer keeps edges past their `y_max` and now applies the clip.
- `InverseWinding`/`InverseEvenOdd` fill the complement of the path within
  the clip (both AA and non-AA).
- `Canvas::clear` only clears within the device clip (`drawColor(color,
  kSrc)` semantics); `draw_color` fills the device clip regardless of the
  CTM; new `Canvas::draw_paint` fills the clip with the full paint (shader
  sampled through the CTM).
- `Canvas::clip_rect` under a rotated/skewed CTM clips to the mapped quad
  (as a path), not the quad's bounding box.
- `save_layer` bounds are a content hint in **local** space: mapped through
  the CTM, intersected with the clip, and rounded to a device-space layer
  with one shared origin convention for drawing and compositing.
- `composite_layer` composites through the current clip (per-pixel
  coverage) and applies the layer paint's color filter in addition to its
  alpha and blend mode.
- `draw_image_rect` applies paint alpha exactly once (was applied twice)
  and goes through the clipped, matrix-aware pipeline (per-pixel inverse
  mapping), so rotation and path/AA/difference clips work.
- `draw_vertices` applies the clip per pixel, uses one coverage rule (pixel
  centers) for both flat and interpolated triangles, applies paint alpha to
  vertex colors, and premultiplies before blending.
- `RSXform::to_matrix` places tx/ty in the translation slots applied last
  (`SkMatrix::setRSXform` layout).
- `draw_round_rect` clamps oversized radii uniformly like
  `SkRRect::setRectXY` and uses conic (circular-arc) corner geometry.
- `draw_arc` with |sweep| >= 360 draws the full oval; arcs use curve
  segments from the path crate instead of 10-degree polylines.
- `draw_points` in `Points` mode draws stroke-width-sized squares
  (butt/square cap) or circles (round cap) per `SkDraw::drawPoints`.
- COLR v1 sweep-gradient angles from `skia-rs-text` are now degrees (no
  extra negation); the color-glyph renderer feeds them straight into
  `SkShaders::SweepGradient` instead of converting from radians.

### Changed (skia-rs-text — conformance audit)
- COLR **v1** color glyphs now render: the paint-graph walker emits a layer
  per painted fill using the active clip glyph, instead of dropping every v1
  layer (`paint` no longer requires a preceding `outline_glyph`). ClipList
  boxes are tracked distinctly (exposed as `ColorGlyphLayer::clip_box`) and
  no longer pushed as a sentinel gid 0.
- COLR v1 sweep angles are converted as degrees (`raw * 180`) with no extra
  sign flip. `GlyphPaint::SweepGradient::start_angle`/`end_angle` are now
  documented and delivered in **degrees**, not radians.
- `TextBlob::from_text` / `TextBlobBuilder::add_text` position glyphs by real
  per-glyph `hmtx` advances; blob width now agrees with `Font::measure_text`
  (was a uniform `size * 0.5` per glyph).
- `TextBlob::unique_id()` returns a monotonic `u32` from a process-wide
  counter (was a pointer-address `usize`).
- `GlyphRun::bounds` uses the font's conservative bbox-based `top`/`bottom`
  extents instead of ascent/descent.
- Font metrics now match the FreeType port: `x_height`/`cap_height` are
  **positive**; `top`/`bottom` come from the font bounding box; `avg_char_width`
  from OS/2 `xAvgCharWidth` (fallback bbox width) and `max_char_width` from the
  bbox width; `underline_position` folds in the half-thickness (distance to the
  top of the stroke); negative line gap is clamped to 0.
- Glyph 0 (`.notdef`) is treated like any glyph: it uses its real `hmtx`
  advance, bounding box, and tofu-box outline (previously advance 0, empty
  bounds, no outline).
- `Font::glyph_path` applies `scale_x` and `skew_x` (Skia's
  `MakeTextMatrix`), consistent with `glyph_advance`.
- Shaper output negates HarfBuzz `y_advance`/`y_offset` (y-up → Skia y-down)
  and multiplies shaped x positions by `Font::scale_x`.
- `GlyphImage::top` is converted to y-down (negated bitmap-top bearing).
- `ParagraphBuilder::push_style`/`pop` maintain a real style stack; `pop`
  restores the previously pushed style rather than the default. Empty lines
  take their height from the paragraph's own style, not `Font::default()`.

### Changed (skia-rs-codec — conformance audit)
- PNG decode enables `EXPAND | STRIP_16`: paletted PNG-8, sub-8-bit
  grayscale, and `tRNS` chunks now decode correctly (to RGB(A)/8-bit gray),
  and 16-bit-per-channel PNGs strip to 8-bit instead of failing.
- Image encoders (PNG, JPEG, WebP, BMP) unpremultiply premultiplied input
  before writing, since every format stores straight alpha. Canvas-backed
  images (now `AlphaType::Premul`) round-trip correctly; a 50%-red pixel
  stored as `r == a == 128` encodes as unpremultiplied `(255,0,0,128)`.
- Standalone 32-bit `BI_RGB` BMPs are treated as **opaque** (`kBGRX`): the
  4th byte is padding and ignored. The alpha byte is only honored for
  BMP-in-ICO (matching `SkBmpCodec`).
- BMP `BI_BITFIELDS` (compression 3) now reads and applies the R/G/B/A
  channel masks (16- and 32-bpp); 16-bpp BMPs decode (default X1R5G5B5).
- GIF single-frame decode returns the **logical-screen** canvas size and
  composites the first frame at its left/top offset over a transparent
  background (was the bare, possibly-smaller first frame). `AnimatedImage`
  gains `canvas_width`/`canvas_height` (and `canvas_dimensions()`).
- Malformed ICO/BMP inputs now error instead of panicking or over-allocating:
  directory-entry offsets and embedded-image bounds are checked, header size
  and dimensions are validated (non-positive/absurd sizes rejected before
  allocation, `i32::MIN` height handled), and truncated pixel arrays are
  reported as incomplete.
- `Image::unique_id` and the default `ImageGenerator::unique_id` (plus
  `SolidColorGenerator`) draw from a monotonic atomic counter instead of a
  pointer address, so freed-then-reused memory cannot alias two images.
- `Image::make_scaled_with` filters in premultiplied space (premultiply →
  filter → unpremultiply) so color no longer bleeds out of transparent
  texels; nearest-neighbor sampling uses pixel centers
  (`floor((dst + 0.5)·scale)`).
- `LazyImage` concurrent generation blocks waiters on a condvar until the
  generating thread publishes its result (no "generation in progress"
  error); `peek_pixels` returns the cached pixmap when pixels are already
  generated (no decode side effect) and `None` otherwise.
- `LazyImage::read_pixels` and `GpuImage::read_pixels` return `false` without
  a partial copy when the destination cannot hold the whole image.
- WBMP format sniffing accepts multibyte width/height integers (dimensions
  larger than 127 pixels).
- JPEG dimension scanning handles all SOF markers (C0–C3, C5–C7, C9–CB,
  CD–CF) and skips standalone RST/SOI/EOI markers that carry no length
  payload.
- `DisposalMethod::Background` is documented as clear-to-**transparent**
  (never the GIF background color), matching `SkGifCodec`.

### Changed (skia-rs-gpu — conformance audit)
- Built-in solid/gradient/cover fragment shaders now output **premultiplied**
  color, and `blend_mode_to_state` derives per-mode **alpha** blend components
  from Ganesh's `gBlendTable` (color→alpha factor substitution) instead of a
  hardwired SrcOver alpha. Together these change the bytes written for
  translucent gradient/solid GPU fills to premultiplied form.
- The paint→GPU bridge reads **real gradient geometry and stops** (endpoints,
  radius, straight-alpha colors) via a new `Shader::as_any` downcast, instead
  of probing premultiplied `sample()` values.
- `PipelineKey` now distinguishes pipelines by every state that affects
  compilation (stencil/depth ops, blend operations + alpha, write mask,
  topology, cull mode, entry points, vertex formats); previously-colliding
  distinct pipelines no longer share a cache slot.
- Stencil-cover emits the closing fan triangle and supports inverse fill types
  (`Equal 0` cover over clip bounds); fill tessellation routes multi-contour
  and non-convex paths through stencil-cover so holes stay holes.
- Curve flattening is device-space (`set_view_matrix` scales tolerance to
  source space); subdivision is tolerance-driven up to `MAX_POINTS_PER_CURVE`
  rather than a small fixed cap, so magnified curves are smoother.
- Stroking honors join (miter with limit + bevel fallback) and cap
  (butt/square/round) styles via `StrokeStyle`.
- Gradient LUTs premultiply **after** sRGB encoding and sample at half-texel
  centers (no Repeat seam); sweep `t=0` is the +x axis (Skia
  `xy_to_unit_angle`).
- SDF textures pack **inside as high** (>128, edge at 128) per
  `SkDistanceFieldGen`, with a half-texel edge-distance correction.
- Image tiling handles tile modes per axis (a Clamp axis no longer tiles).
- Atlas: TooLarge accounts for padding; `uv_rect` insets a half texel;
  `compact` drops unplaceable entries instead of re-placing at stale
  coordinates; glyph eviction frees atlas regions and `GlyphBatch::validate`
  flags stale (bumped-generation) batches.
- wgpu: pipelines build from the shader's real bind-group layouts (auto
  layout); clear colors are linearized for `*UnormSrgb` targets; compute
  dispatches record a real `DispatchCompute` (never replayed as a draw);
  `ScissorRect::from_rect` clamps the box instead of sliding its edges.
- Removed undefined behavior in the Vulkan backend (`CString::from_raw` over a
  borrowed device-name array); Metal gates `Depth24Unorm_Stencil8` on device
  support and reports Tier1 argument buffers as available.

### Changed (skia-rs-svg — conformance audit)
- Presentation attributes (`fill`, `stroke`, `stroke-width`, `color`,
  `fill-opacity`, `stroke-opacity`, `fill-rule`, `stroke-linecap`,
  `stroke-linejoin`, `stroke-dasharray`, `stroke-dashoffset`) now **inherit**
  down the element tree via a presentation-context stack. The `fill: black`
  initial value lives at the render root, not on every node, so an element
  with no `fill` correctly inherits its ancestor's paint instead of drawing
  black. `SvgNode`'s inherited fields are now `Option` (unset = inherit) and
  `SvgPaint` gained `CurrentColor` and a `url(#id) <fallback>` form.
- `currentColor` resolves against the inherited `color` property (default
  black) instead of silently falling back to an unset paint.
- Percentage lengths resolve against the viewport per SVG 1.1 §7.10
  (`x`/width → viewport width, `y`/height → height, `r`/font-size →
  `sqrt(w²+h²)/√2`) rather than collapsing to a raw `0..1` fraction.
- Group/element `opacity` composites through a `saveLayer` (alpha), with the
  Skia leaf-node optimization applying it as paint alpha only for a single
  atomic draw (one of fill/stroke, no descendants). Overlapping content in a
  translucent group is no longer double-composited.
- `<polyline>` and `<polygon>` now **fill** by default (previously polyline
  rendered stroke-only); the fill closes the contour implicitly.
- `preserveAspectRatio` is honored (default `xMidYMid meet` uniform-scale +
  centering; `none` non-uniform; all align/meet/slice combinations), instead
  of an unconditional min-scale fit.
- Parsed `fill-rule` (even-odd), `stroke-dasharray`/`-dashoffset`,
  `stroke-linecap`, and `stroke-linejoin` are applied to the path/paint.
- objectBoundingBox gradients map through the bbox matrix composed as
  `bbox × gradientTransform`; radial OBB gradients are now correctly
  elliptical for non-square bounds.
- `<clipPath>` child transforms are honored for all shape kinds.
- `<use>` applies its `x`/`y` translation, and a depth guard prevents
  malformed reference cycles from overflowing the stack.
- CSS declarations apply in document/cascade order (ordered list, not
  `HashMap` iteration); `fill-opacity`/`stroke-opacity` multiply into the
  paint alpha instead of mutating the fill/stroke color.
- A missing `url(#id)` paint reference uses the grammar's fallback color when
  provided.
- SVG-in-OpenType glyph documents render in font units scaled by `ppem/upem`
  (via `render_svg_in_container`) instead of being stretched to the canvas.
- Path export subdivides conics into a quad spline
  (`SkConic::chopIntoQuadsPOW2`-equivalent) instead of emitting a single
  naive, weight-dropping quad.

### skia-rs-ffi / skia-rs-safe

- **Breaking:** `sk_clip_op_t` values now match upstream `SkClipOp`
  (`kDifference = 0`, `kIntersect = 1`) instead of the reversed mapping
  previously used; `sk_canvas_clip_rect`/`sk_canvas_clip_path` now take a
  raw `uint32_t` (decoded via `decode_clip_op`, rejecting out-of-range
  values) instead of a C enum passed by value, which was undefined
  behavior for any value outside the two valid discriminants.
  `sk_region_op_rect`'s `op` and `sk_patheffect_new_trim`'s `mode` are
  likewise raw `uint32_t` decoded with range checks instead of by-value
  Rust enums.
- **Breaking:** `decode_color_type` (used by `sk_surface_new_raster_with_info`)
  now matches upstream `SkColorType` numbering (`kRGB_888x = 5`,
  `kBGRA_8888 = 6`, …) instead of miscounting from index 5, and returns an
  error for any color type it doesn't recognize instead of silently
  falling back to RGBA8888.
- **Breaking:** `sk_paint_set_blend_mode` now decodes all 29 upstream
  `SkBlendMode` values (0-28) instead of only the first 15, returns `bool`
  (`false`, paint left unchanged) for an out-of-range mode instead of
  silently coercing to `SrcOver`, and gained a `sk_paint_get_blend_mode`
  counterpart.
- Fixed `WasmSurface::get_image_data` (wasm32): stopped swapping the R/B
  channels (the surface is RGBA, not BGRA — the swap silently corrupted
  every pixel's red/blue channels) and now unpremultiplies pixels before
  handing them to `ImageData`, which expects straight (unpremultiplied)
  alpha. `WasmSurface::get_pixels` was given the same unpremultiply
  treatment so it stays consistent with its "for ImageData" contract now
  that surfaces store premultiplied bytes.
- `sk_surface_read_pixels` unpremultiplies its output when the surface's
  declared alpha type is `Unpremul` (surfaces store premultiplied bytes
  internally, so the raw buffer no longer contradicts the declared info);
  `sk_surface_peek_pixels`, which hands back a borrowed premultiplied
  buffer, documents that and refuses `Unpremul`-typed surfaces.
- `sk_surface_lock_canvas` now actually enforces its documented one-lock-
  per-surface contract: a second lock while one is outstanding returns
  null, and `sk_surface_clear`/`sk_surface_draw_*` are no-ops while the
  surface is locked (previously both could run concurrently with a locked
  canvas, giving two independent mutable views of the same pixel buffer).
- Fixed undefined behavior in `sk_matrix_concat`, `sk_matrix_map_point`,
  `sk_matrix_invert`, `sk_matrix44_concat`, `sk_matrix44_map_point`, and
  `sk_matrix44_invert` when `result` aliases an input pointer (a supported,
  documented in-place usage): inputs are now copied out via `ptr::read`
  before anything is written through `result`, instead of forming
  simultaneous `&`/`&mut` references into the same memory.
- Recording-canvas handles (`sk_recording_canvas_t`) now carry a shared
  liveness flag from their owning `sk_picture_recorder_t`; calling any
  `sk_recording_canvas_*` function after the recorder has been deleted now
  returns an error/no-op instead of dereferencing freed memory.
- `sk_canvas_clear` now respects the canvas's current clip stack (routes
  through the same clip-aware canvas construction as the other draw
  calls) instead of clearing the whole buffer regardless of clip.
- `sk_version()` now reports the crate's actual `CARGO_PKG_VERSION`
  instead of a hardcoded `"0.1.0"`.
- `SkImageInfoABI`'s size assertion is now pointer-width aware (24 bytes on
  64-bit targets, 20 bytes on 32-bit) instead of unconditionally asserting
  24, and its docs no longer overclaim byte-for-byte binary compatibility
  with upstream `SkImageInfo` (which embeds a ref-counted `sk_sp` smart
  pointer, not a raw pointer).
- Corrected the `sk_refcnt_get_count`/`sk_refcnt_is_unique` documentation:
  the magic-tag check is a best-effort heuristic against non-refcounted
  pointers, not a memory-safety guarantee — the pointer must still be null
  or point to a valid allocation.
- Regenerated the stale, drifted root `include/skia-rs.h` (previously
  exposed types like `Paint`/`Path` as public opaque structs that
  `cbindgen.toml` excludes, and was missing most `sk_*` functions added
  since it was last hand-refreshed) so it matches the crate's current
  exports and compiles as C; added `crates/skia-rs-ffi/tests/header_up_to_date.rs`
  so `cargo test` catches future drift between the committed headers and
  the crate's actual exports.
- `skia-rs-safe` Android: `HardwareBufferFormat` no longer defines a
  nonexistent `R4G4B4A4_UNORM` variant (there is no
  `AHARDWAREBUFFER_FORMAT_R4G4B4A4_UNORM` in the NDK). `BitmapConfig`
  values now match the real `ANDROID_BITMAP_FORMAT_*` native constants
  instead of arbitrary sequential values. `HardwareBuffer::new` on Android
  now returns `None` (fails closed) instead of returning a fake `Some`
  that claimed a hardware buffer was allocated when no real
  `AHardwareBuffer_allocate` call was ever made.

### Changed (skia-rs-node)

- **Breaking:** `Surface.getPixels()` now returns straight
  (unpremultiplied) RGBA. Raster surfaces switched to premultiplied
  internal storage this release; `getPixels` unpremultiplies before
  returning so its documented straight-alpha contract (feeding
  `node-canvas` `ImageData`, PNG writers, etc.) is preserved. Translucent
  pixels are unaffected in value from a caller's perspective, but any code
  that relied on the transient premultiplied bytes should re-check.

### Changed (skia-rs-python)

- **Breaking:** `Surface.pixels()` now returns straight (unpremultiplied)
  RGBA, matching its documented contract and the parallel `skia-rs-node`
  change. Raster surfaces store premultiplied bytes internally this
  release; `pixels()` unpremultiplies before returning, so consumers
  feeding the buffer into NumPy/PIL/PNG writers keep correct colors for
  translucent pixels.

### Changed (skia-rs-core — conformance audit, Task 1)
- `Rect::is_empty` now reports empty for any NaN coordinate (matching
  `SkRect::isEmpty`), so NaN rects no longer survive `union`/`join`.
- `Rect::contains_rect` returns false when either rectangle is empty.
- `Rect::round`/`round_out`/`round_in` and `Size::to_isize_round` use Skia's
  rounding (half toward +∞) with saturating float→int casts; NaN and
  out-of-range values now saturate instead of becoming 0.
- `IRect::from_xywh` uses saturating addition for the right/bottom edges.
- `Matrix::invert` and `Matrix44::invert` use Skia's determinant thresholds
  (double precision; reject only near-zero/zero determinants and non-finite
  inverses); small-scale matrices such as `scale(0.005)` now invert.
- `Matrix44::pre_translate`/`post_translate`/`pre_scale`/`post_scale` had their
  pre/post semantics corrected to match `SkM44` (`preX` = `self * X`).
- `RRect::from_rect_xy`/`from_oval` sort inverted rects instead of panicking,
  scale oversized radii by a single aspect-preserving factor, and square all
  corners when either radius is ≤ 0.
- RGB565→RGBA8888 conversion replicates low bits, so full-scale channels map to
  255 (e.g. 565 white is now `#FFFFFF`, previously `#F8FCF8`).
- RGBA8888→Gray8 uses BT.709 luma coefficients (previously BT.601).
- Premultiplication (`premultiply_color`, `premultiply_in_place`) rounds via
  `SkMulDiv255Round` instead of truncating.
- Per-format alpha (un)premultiplication for `Argb4444`, `Rgba1010102`,
  `Bgra1010102`, and `R16G16B16A16Unorm` (previously corrupted by a generic
  4-byte-RGBA path); alpha-only formats are left unchanged.
- `ColorType::n32` returns `Rgba8888` on all platforms except Windows (`Bgra8888`),
  matching Skia's build-config selection rather than target endianness.
- `ColorType::has_alpha` returns false for the alpha-less `R16G16Unorm` and
  `R16G16Float` formats.
- `Region` canonicalizes to scanline order after `intersect` and `difference`
  (not just `union`), so equality, `is_rect`, `rect_count`, and iteration order
  match `SkRegion`.
- `ImageInfo::new` accepts zero dimensions as a legal empty info (still rejects
  negatives).

### Changed (skia-rs-path — conformance audit, Task 2)
- `Path::contains` accumulates signed winding (`SkPathPriv::Contains`): the
  Winding rule tests `w != 0`, EvenOdd tests parity, inverse fill types are
  XORed and short-circuit outside the (now inclusive) bounds, and every contour
  is implicitly closed for hit-testing.
- `Path::reverse` produces valid verb streams (`SkPathPriv::ReverseAddPath`) —
  no trailing `Move`, no spurious/doubled `Close`; one `Close` per closed contour.
- `Path::direction` uses the dominant-contour extreme-point cross test
  (`ComputeFirstDirection`); `add_rect`'s CW geometry now reports CW in y-down
  device space (previously reported CCW).
- `Path::is_rect` recognizes the crate's own `add_rect` output (Move + 3 Lines +
  Close) and rejects non-rectangular H/V staircases (`IsRectContour`).
- `Path::convexity` is per-contour and verb-aware (`ComputeConvexity`): any
  second contour is Concave; sign-change tests replace the fixed 0.001 threshold.
- `Path::bounds` returns empty bounds for any non-finite coordinate.
- `stroke_to_fill` strokes the closing segment of closed contours with a join at
  every vertex and emits the inner offset ring reversed, so a stroked rect filled
  with the Winding rule renders as a frame with an empty middle (was a filled
  slab); curves are flattened with error-driven subdivision and the correct conic
  form; zero-length contours emit a Round/Square cap dot.
- `PathBuilder`: after `close()`, a line/curve injects `moveTo(last_move)`;
  `current_point()` returns the subpath start after `close()`; repeated `close()`
  is a no-op; consecutive `move_to` overwrite the pending Move; `add_oval` uses
  four √2/2 conics; `add_arc` derives direction from the sweep sign and only
  takes the oval shortcut when the start angle is a multiple of 90°; SVG `arc_to`
  applies the x-axis-rotation to emitted geometry; zero-sweep arcs no longer
  produce NaN control points.
- SVG parser: smooth `S`/`T` reflection is gated on the previous command kind
  (`S` after C/S, `T` after Q/T); numeric data directly after `Z` is a parse
  error; the current point returns to the subpath start after `Z`.
- `PathMeasure::get_point_at`/`get_tangent_at` pin the distance into `[0, length]`
  (returning `None` only for NaN/empty); `get_segment` pins start/stop.
- `DashEffect` dashes the closing segment of a closed contour continuously.
- `CornerEffect` rounds the joint between the last and first segments of a closed
  contour (`SkCornerPathEffect`).
- `simplify` always runs the boolean-ops machinery so self-intersections and
  overlaps resolve (no Union-with-empty short-circuit).

### Changed (skia-rs-paint — conformance audit, Task 3)
- Blend modes Overlay/HardLight/SoftLight add the Porter-Duff edge terms
  `s·(1−da) + d·(1−sa)` outside the branch; SoftLight's dark-dst polynomial is
  the upstream `16m³ − 12m² + 3m` form; ColorDodge/ColorBurn general branches
  use the upstream `da` normalization (`BLEND_MODE(...)` in
  `SkRasterPipeline_opts.h`).
- `Paint::default()` now matches `SkPaint`: `stroke_width` is `0.0` (hairline,
  was `1.0`) and `anti_alias` is `false` (was `true`).
- `Paint::color32()` and `Paint::serialize()` round float color components to
  the nearest byte (0.5 → 128) instead of truncating.
- **`Shader::sample` now returns premultiplied colors** (Skia raster-pipeline
  convention), uniformly across ColorShader, gradients, image shaders, blend/
  compose shaders and runtime effects; `BlendMode::apply` receives premul
  inputs from BlendShader/ComposeShader as it requires.
- Gradient stop positions are pinned monotonic into [0, 1] (`SkTPin`) and a
  first stop above 0 acts as an implicit stop at t=0 carrying the first color
  (`SkGradientBaseShader::fFirstStopIsImplicit`); t below the first explicit
  stop no longer returns the last color.
- Two-point conical gradients pick the larger quadratic root when both
  interpolated radii are valid (upstream well-behaved case), and now honor
  `GradientFlags::INTERPOLATE_PREMUL`.
- `Alpha8` image sampling decodes as black-with-alpha `(0,0,0,a)`, not white.
- `ImageShader::sample` honors `SamplingOptions`: `FilterMode::Linear` does
  bilinear filtering (was always nearest-neighbor).
- Blur filters convert sigma to box radius via
  `SkBlurMask::ConvertSigmaToRadius` (`(sigma − 0.5)/0.57735`, three-box
  approximation) instead of truncating sigma — sigma 0.9 now blurs.
- `BlurMaskFilter` implements `BlurStyle` Solid (src ∪ blur), Outer
  (blur − src) and Inner (blur ∩ src) per `SkBlurMask`.
- `ColorFilterImageFilter` unpremultiplies, applies the color matrix in
  unpremul space, clamps and re-premultiplies (upstream
  `SkColorFilters::Matrix` default).
- `MatrixConvolutionImageFilter` honors its `tile_mode` (was always clamp)
  and, when `convolve_alpha` is false, convolves unpremultiplied RGB and
  re-premultiplies with the original alpha.
- `TileImageFilter` fills only `dst_rect`, leaving the rest of the buffer
  untouched.
- Runtime effect uniforms pack tightly (offset += size, 4-byte granularity,
  no 16-byte alignment) per `SkRuntimeEffect.cpp`; child declarations
  (`uniform shader s;`) are children only — excluded from `uniforms()` and
  `uniform_size()`.
- SkSL: method-style calls parse on any postfix expression and
  `child.eval(coord)` samples the bound child shader in the interpreter;
  multi-component swizzle assignment (`c.rgb *= 0.5`) works; matrix±matrix,
  matrix·scalar, scalar·matrix, vector·matrix and matrix/scalar arithmetic
  are implemented (previously collapsed to 0); float division/modulo by zero
  follow IEEE (±inf/NaN); `&`, `|`, `^`, `<<`, `>>` are wired into the GLSL
  precedence chain and `half(x)` parses as a constructor.

### skia-rs-pdf
- **Breaking (API):** `PdfCanvas::draw_text` and `draw_text_with_font` now
  return `Result<(), PdfError>` instead of `()`. Conditions that previously
  panicked are now recoverable errors: drawing through a Type0/CID font
  (whose live per-glyph CID emission is not yet implemented) and drawing
  with no font selected both return `Err(PdfError::Unsupported)`. A font
  index out of range remains a `debug_assert!` (programmer error — a valid
  index only ever comes from the canvas's own font manager). Callers must
  now handle or propagate the `Result`.
- **Breaking (visual):** fixed text rendering upside-down — the PDF page
  CTM's `1 0 0 -1 0 height` y-flip was never compensated for glyph runs,
  so every drawn string rendered mirrored. Text now sets the text matrix
  to `1 0 0 -1 x y Tm` per upstream `SkPDFDevice::GlyphPositioner`.
- **Breaking (visual):** fixed images rendering vertically flipped —
  `draw_image` now emits the `setScale(1,-1); postTranslate(0,1)`
  counter-flip (`w 0 0 -h x (y+h) cm`) before placement, matching
  `SkPDFDevice::internalDrawImageRect`.
- **Breaking:** non-ASCII text drawn against a simple (Type1/TrueType)
  font is now encoded as single-byte WinAnsiEncoding bytes instead of an
  (invalid, per PDF 32000-1) UTF-16BE hex string; characters outside
  WinAnsi fall back to `?` until per-run Type0 font switching exists.
- **Breaking:** `Path` fill type is now honored when filling/stroking —
  `EvenOdd`/`InverseEvenOdd` paths emit `f*`/`B*` instead of always
  `f`/`B` (nonzero winding).
- Fixed ToUnicode CMap codespace for simple fonts: declares a 1-byte
  codespace (`<00> <FF>`) with 2-hex-digit `bfchar` source codes instead
  of a 2-byte range with 4-digit codes that didn't match the actual
  content-stream byte.
- Type0 (CID) fonts now emit a proper `/DescendantFonts` array
  (`CIDFontType2` with `/CIDSystemInfo`, `/CIDToGIDMap /Identity`, `/W`)
  per PDF 32000-1 §9.7.6, instead of a Type0 dict with no glyph source.
  New `PdfFont::truetype_cid` / `PdfFontManager::register_truetype_cid`.
  Note: only the `/DescendantFonts` *structure* is emitted; drawing text
  through a Type0 font fails closed (the simple-font 1-byte draw path is
  incompatible with Identity-H CID encoding) until per-glyph CID emission
  lands.
- Symbol/ZapfDingbats standard fonts now omit `/Encoding` so readers use
  the font's built-in symbolic encoding, instead of incorrectly
  declaring `/Encoding /StandardEncoding` (which has no entries for
  their glyph names).
- `ExtGraphicsState::cache_key` now covers `soft_mask`, `alpha_is_shape`,
  `text_knockout`, and both overprint flags, fixing a bug where two
  distinct ExtGStates differing only in those fields were wrongly
  deduplicated into one cached `/ExtGState` object.
- PDF/A OutputIntents now embed a real, valid sRGB ICC v2.1 profile
  (`assets/srgb-v2.icc`) instead of a 9-byte placeholder string that
  wasn't parseable ICC data at all.

### skia-rs-skottie

Conformance-audit fixes for the Lottie player — most real Lottie files
previously rendered blank or static:

- `as_vec2()` now accepts `Vec3` values (Bodymovin exports 3-component
  position/anchor arrays)
- Bezier path keyframes (`{"i","o","v","c"}`) parse into real
  `KeyframeValue::Path` geometry and interpolate point-wise; `"sh"`
  shapes and masks now produce real geometry instead of empty paths
- Group `"tr"` transform items parse into a real `Transform`
  (anchor/position/scale/rotation/skew/opacity, animated) and apply to
  the group's children; group opacity multiplies
- Stroke dash arrays (`"d"`) no longer collide with the shape
  `direction` field (now a `DirectionOrDash` union); dashed strokes
  parse and render via `DashEffect`
- Layer transform/opacity/masks now evaluate at the unadjusted
  composition frame; only precomp **content** gets the `st`/`sr`/`tm`
  remap, matching upstream `Layer.cpp`/`PrecompLayer.cpp` — behavior
  change: non-precomp layers with `st`/`sr` set (previously
  misapplied) no longer shift
- `"sr"` (stretch) and `"tm"` (time remap) parse and apply to precomp
  content: `(t - st) / sr`, with `tm` (seconds) overriding when present
- A trailing `{"t":N}`-only keyframe now inherits the previous
  keyframe's value instead of resetting to 0
- Fill rule `"r"` (2 = even-odd) parses and sets the path's fill type
- Mask modes implemented via real polygon boolean ops: Add unions,
  Subtract subtracts, Intersect intersects; `inv` and the first-mask
  source-mode flip are honored (previously all masks behaved as a
  plain intersect clip regardless of mode)
- Layer parenting composes the parent transform chain (cycle-guarded)
  — previously parent references were ignored entirely
- Precomp layers guard against self-referential/cyclic `refId` nesting
  (bounded recursion depth) instead of overflowing the stack
- Trim paths (`s`/`e`/`o`/`m`) implemented via path measure, matching
  upstream start/stop/offset/inverted resolution (previously a no-op);
  `m:1` ("Simultaneously", upstream `kParallel`) trims each shape
  independently and `m:2` ("Individually", `kSerial`) merges all
  geometry and applies one trim across the combined length, per
  `AttachTrimGeometryEffect`
- Track mattes (`td`/`tt`): `tt` selects alpha (1), alpha-inverted (2),
  luma (3), or luma-inverted (4) masking from the matte source (explicit
  `tp` index, or the immediately preceding layer); `td` hides the source
  from the main render list while still rendering it as the matte input.
  Compositing matches upstream `sksg::MaskEffect::onRender` (outer
  `save_layer` collects coverage — luma via a BT.709 luma-to-alpha color
  filter — then an inner `SrcIn`/`SrcOut` `save_layer` composites the
  consumer's content); an invisible matte source contributes zero
  coverage (consumer fully masked out, or fully shown for inverted modes)
- `ti`/`to` spatial bezier position interpolation: position keyframes
  with spatial tangents interpolate along the cubic motion path
  `(v + to) -> (v_next + ti)` by arc length (via `PathMeasure`), using
  the eased factor as the arc-length fraction, matching upstream
  `Vec2KeyframeAnimator`; also applied to 3-component `Vec3` positions
  (bezier on x/y, z linear)
- Gradient fills/strokes parse `g` (stop count + interleaved
  pos/rgb[+alpha]), `s`/`e`, and `t` (linear/radial with highlight
  focal point) and build a real gradient shader (previously a flat
  gray placeholder)
- Skew is now `Skew(tan(-radians(pin(sk, -85, 85))))` (negated,
  pinned), matching upstream `Transform.cpp`
- Rounded-rect corners use circular-arc cubic beziers instead of a
  quadratic approximation, and respect `"d"` (CW/CCW) winding

Not implemented in this pass (tracked as follow-ups): mask
opacity/feather soft-alpha compositing (masks with opacity < 100% or a
feather currently apply as a hard clip — combining Lottie's multi-mask
per-mode boolean stack in coverage/alpha space is a materially larger
change than the single-source track-matte compositing above).

## [0.2.6] - 2026-04-26

Phase 7 complete: resolved every remaining deferral across the
workspace's 10 crates. The substantial items from earlier phases
(boolean ops, ICC profile parsing, TTF subsetting, CSS Color Level
4, COLR v1, RAW demosaic, multi-backend GPU executors, expanded
FFI surface) all landed with real implementations and tests.

### Added (skia-rs-path — 14/14 now resolved)
- Correct polygon boolean ops via the geo crate's sweep-line
  (Difference/Intersect/Xor/Union) — handles concave shapes, holes,
  partial overlaps correctly (GAP-C4, GAP-C5)
- Path::length uses adaptive curve flattening instead of control-
  polygon approximation (GAP-N2)
- Path::contains honors conic weights and uses adaptive flattening
  (GAP-N3)

### Added (skia-rs-core — 12/12 now resolved)
- IccProfile::from_bytes parses rXYZ/gXYZ/bXYZ + rTRC tag table to
  build a real ColorSpace; falls back to sRGB for unsupported tags.
  Display P3 and Adobe RGB profiles now round-trip correctly (GAP-C1)
- CSS Color Level 4: lab(), lch(), oklab(), oklch(), hwb(), and
  color(space) with srgb/srgb-linear/display-p3 support. Out-of-
  gamut values clamped. Slash-separator alpha per spec.

### Added (skia-rs-pdf)
- Byte-level TrueType subsetting (pure-Rust glyf/loca pruning).
  Measured 14-32% size reduction on Roboto and Noto Serif drawing
  "Hello, World!". Composite glyph dependency resolution, checksum
  preservation, /Length1 invariant. (GAP-C3 follow-up)

### Added (skia-rs-text — 14/14 now resolved, no follow-ups)
- COLR v1 typed paint data on ColorGlyphLayer: GlyphPaint::Solid /
  LinearGradient / RadialGradient / SweepGradient with per-layer
  transform, clip, and composite mode. Canvas integration deferred
  to consumers.
- skia_rs_svg::glyph_svg_to_dom: decompress+parse SVG-in-OpenType
  bytes (gzip magic auto-detection)

### Added (skia-rs-codec — 13/13 now resolved)
- demosaic_bayer_rggb: bilinear Bayer demosaic for 16-bit raw data
- AnimatedImage / AnimationFrame / LoopCount / DisposalMethod /
  BlendMethod types; GifCodec::decode_animated walks multi-frame
  GIF streams with per-frame delays
- PNG and JPEG decoders populate ImageInfo.color_space from iCCP /
  APP2-ICC chunks via IccProfile::from_bytes

### Added (skia-rs-gpu — 14/14 now resolved)
- OpenGlContext / VulkanContext / MetalContext gain new_wgpu_executor
  that delegates to wgpu configured with the specific Backends
  mask. CommandBuffer + RenderPipelineDescriptor flow through the
  same abstraction across all four backends.

### Added (skia-rs-ffi)
- Expanded API surface 126 → 181 functions. New primitives:
  sk_canvas_clip_rect/path, sk_canvas_save_layer,
  sk_canvas_draw_text_blob, sk_matrix44_t, sk_colorspace_t,
  sk_textblob_t, sk_picture_recorder_t, sk_picture_t,
  sk_region_t, sk_patheffect_t (Dash + Trim)
- Regenerated include/skia_rs.h via cbindgen

### Changed (skia-rs-paint)
- Paint gains path_effect field and getters/setters
- Canvas::draw_path applies Paint::path_effect before rasterization
  (required for dashed strokes to work end-to-end in FFI)

### Fixed
- ShaderError gained impl std::error::Error so it integrates with
  the workspace error convention
- sk_path_iter_next guards bounds before indexing — malformed paths
  return SK_PATH_VERB_DONE instead of panicking (still caught by
  catch_panic but now explicitly handled)

### Cumulative workspace status (after v0.2.6)
- skia-rs-core: 12/12 ✅
- skia-rs-path: 14/14 ✅
- skia-rs-paint: 31/31 ✅
- skia-rs-canvas: 22/22 ✅
- skia-rs-text: 14/14 ✅
- skia-rs-codec: 13/13 ✅
- skia-rs-ffi: 12/12 (Priority 1+2 of scope-follow-up resolved;
  RuntimeEffect + GPU + SVG/PDF/Skottie FFI + CI C-link harness
  remain as separate efforts)
- skia-rs-svg: 13/13 ✅
- skia-rs-pdf: 14/14 ✅
- skia-rs-gpu: 14/14 ✅

### Test count
- skia-rs-core: 79 → 90
- skia-rs-path: 50 → 57
- skia-rs-text: 23 + 34 → 33 + 34
- skia-rs-codec: 47 → 56
- skia-rs-svg: 39 → 46
- skia-rs-pdf: 62 → 71
- skia-rs-gpu: 104 → 128 (with all features)
- skia-rs-ffi: 28 → 44
- Workspace total: ~735 → 802+, 0 failures

## [0.2.5] - 2026-04-25

Phase 6 complete: audited and completed the remaining six crates
(text, codec, ffi, svg, pdf, gpu). 80 gaps across six GAPS.md files;
all addressed with explicit resolution or documented deferrals.

### Added (skia-rs-text)
- Real Font::metrics from hhea/OS/2/post tables (replaced hardcoded approximations)
- Paragraph::layout routes through functional rustybuzz Shaper (was ad-hoc glyph mapping)
- Per-span style preservation via LineFragment
- Real glyph intercepts with de Casteljau flattening
- Color font table parsing (COLR v0/v1 layers, CBDT/CBLC/sbix/bdat, SVG-in-OpenType bytes, palette access)
- FontMgr::make_from_data, make_from_file, character-coverage matching
- GlyphRun bounds from real advances

### Added (skia-rs-codec)
- Image::read_pixels format conversion via skia_rs_core::convert_pixels
- Image::make_with_filter applies real ImageFilter from skia-rs-paint
- GpuImageBackend trait defining upload/read_back/release contract
- EncodedImageGenerator caches decoded image and reports real info
- SamplingOptions::{Nearest, Linear} and Image::make_scaled_with bilinear
- Fixed WebP RGB→RGBA decode bug (surfaced by new round-trip tests)

### Added (skia-rs-ffi)
- Panic catching wraps every exported sk_* function (soundness fix)
- Tag-validated RefCounted<T> rejects untagged pointers
- sk_init(major, minor) + sk_is_initialized() for runtime ABI check
- Expanded API: sk_canvas_t, sk_image_t, sk_typeface_t/sk_font_t,
  sk_shader_t, sk_colorfilter_t, sk_maskfilter_t, sk_imagefilter_t,
  gradient constructors, blur/saturation/matrix filters, image I/O,
  PNG encode, surface read_pixels, path iteration, matrix helpers
- cbindgen-driven include/skia_rs.h header and examples/draw_rect.c

### Added (skia-rs-svg)
- <text> element renders real glyph outlines via skia-rs-text
- Gradient url(#id) references resolved through Defs symbol table
- Gradient <stop> parsing from attrs or style; spread + transform
- CSS Stylesheet applied during render tree-walk
- Replaced hand-rolled XML parser with roxmltree (entities, CDATA,
  namespaces, text content all fixed)
- <image> data: base64 decode via skia-rs-codec
- <clipPath> dispatch via canvas.clip_path
- Extended color parsing (3/4/6/8-digit hex, rgb/a, hsl/a, ~140 CSS3 names)
- Extended length units (px/pt/pc/em/rem/ex/ch/vw/vh/vmin/vmax/cm/mm/in/%)

### Added (skia-rs-pdf)
- PdfDocument wires PdfFontManager, PdfImageManager, TransparencyManager
  so the Resources dict actually contains /Font, /XObject, /ExtGState
- Per-page used_fonts/used_images/used_ext_gstates tracking
- Real TrueType metrics from ttf_parser::Face (ascender/descender/
  italic_angle/cap_height/x_height/bbox/per-code widths)
- FNV-1a subset-prefix tag on BaseFont; FontFile2 stream emission
- PDF/A validation wired into write_to with XMP metadata + OutputIntent
- ExtGState registration for alpha < 1 or non-Normal blend mode
- set_alpha/set_blend_mode/draw_image/draw_text_with_font methods
- Conic flattening via skia_rs_path::flatten::flatten_conic_adaptive
- ToUnicode CMap from recorded used_chars (surrogate-pair aware)

### Added (skia-rs-gpu)
- WgpuExecutor + WgpuPipelineCache consume CommandBuffer + RenderPipelineDescriptor
- Paint→pipeline bridge (paint_bridge.rs): PipelineSelection, BlendMode→BlendState, uniform packing
- Ear-clipping triangulator replaces fan (handles concave correctly)
- Adaptive conic flattening in tessellation
- WgpuStencilSurface with full depth/stencil translation (wires stencil-cover algorithm)
- Naga WGSL validation in shader compilation
- Felzenszwalb-Huttenlocher O(n) SDF distance transform
- Atlas compact() + free()
- MSAA resolve path in read_pixels

### Fixed
- All six crates: GAPS.md annotated with per-gap Status lines

### Test counts
- skia-rs-text: 19 → 57
- skia-rs-codec: 26 → 47
- skia-rs-ffi: 13 → 28
- skia-rs-svg: 21 → 39
- skia-rs-pdf: 27 → 62
- skia-rs-gpu: 77 → 104 (default), 78 → 115 (wgpu)
- **Workspace total: 695 → 718+ tests, 0 failures**

### Deferred (documented in per-crate GAPS.md)
- Text: COLR v1 gradients/transforms, SVG-in-OpenType rasterization
- Codec: RAW demosaic (needs color-pipeline library), ICC profile chain
- FFI: full API surface expansion (text blobs, codec streaming, runtime effects, matrix44, picture, GPU FFI, SVG/PDF FFI)
- SVG: CIE-lab/oklch color spaces (upstream color-management)
- PDF: byte-level TTF subsetting, external veraPDF validation in CI
- GPU: OpenGL/Vulkan/Metal executors (wgpu path is fully functional), real-GPU CI setup

## [0.2.4] - 2026-04-25

### Added (skia-rs-canvas)
- **Canvas unified via Backing enum**: `Canvas<'a>` now carries
  `Backing::Raster(&mut PixelBuffer) | Recording(&mut Vec<DrawCommand>) | Null`.
  All 11 critical draw methods now dispatch via match and render correctly.
  Picture playback works end-to-end with pixel-identical round-trip (C-1..C-7)
- **ClipStack-based clip**: Canvas clip is now full `ClipStack` with
  Difference ops, AA, and path clipping (C-8, C-9, N-1)
- **draw_points with PointMode**: Points/Lines/Polygon modes (C-4)
- **save_layer offscreen composition**: allocates per-layer buffer,
  composites back on restore with paint alpha + blend mode, nests (C-10)
- **Barycentric color interpolation in draw_vertices**: per-vertex
  Gouraud shading for Triangles/Strip/Fan (N-5)
- **draw_image_lattice + draw_atlas + draw_patch**: real implementations
  dispatching to draw_image_rect / draw_vertices
- **Coons patch draw_patch**: full bicubic surface with boundary-curve
  evaluation and optional corner color interpolation
- **Real glyph outlines in text rendering** via ttf-parser (N-6):
  - New `Font::glyph_path()` returns a Path with screen-space scaling
  - Real `cmap` and `hmtx` parsing in skia-rs-text Typeface
  - `Canvas::draw_string`, `draw_text_blob`, `draw_glyphs` now render
    actual character shapes, not rectangles

### Changed (skia-rs-canvas)
- `Surface::canvas()` now returns a functional raster Canvas (was a stub)
- `RasterCanvas` is a deprecated type alias for `Canvas<'a>` (migration path)
- SSE4.1 `fill_span_blend` dispatch disabled — fell back to scalar due
  to incorrect per-channel alpha handling. AVX2 and NEON paths unchanged.

### Fixed (skia-rs-canvas)
- `ClipStack::clip_rect_aa` now intersects both region and mask in the
  `RegionAndMask` branch (N-4)
- `draw_vertices` flat-shade fallback preserved when no colors provided

### Tests (skia-rs-canvas)
- Test count grew from 39 to 103 in skia-rs-canvas
- New pixel-level round-trip tests for Picture playback (Circle, Path,
  Oval, Arc, RoundRect, Line, ClipRect, Scale)
- Matrix stack, SaveLayerRec, RSXform, FilterMode, PointMode coverage

## [0.2.3] - 2026-04-25

### Added (skia-rs-paint)

- **Full filter pipeline**: Paint now carries `mask_filter`, `color_filter`,
  and `image_filter` fields with getter/setter pairs (GAP-C1)
- **BlendMode::apply** for all 29 blend modes — Porter-Duff (Clear/Src/Dst/
  SrcOver/DstOver/SrcIn/DstIn/SrcOut/DstOut/SrcATop/DstATop/Xor/Plus/Modulate),
  separable (Screen/Overlay/Darken/Lighten/ColorDodge/ColorBurn/HardLight/
  SoftLight/Difference/Exclusion/Multiply), and non-separable HSL
  (Hue/Saturation/Color/Luminosity) (GAP-C6)
- **All shader sample() implementations**: LocalMatrixShader, BlendShader,
  ComposeShader, TwoPointConicalGradient, PerlinNoiseShader, ImageShader
  (with pixel data) (GAP-C3/C4/C5/C7/C8)
- **MaskFilter::apply_mask** for separable box-blur approximation of
  Gaussian blur (GAP-N3)
- **ImageFilter::apply** for all 14 concrete filter types: Blur, DropShadow,
  Morphology, ColorFilter, DisplacementMap, Lighting, Compose, Merge,
  Offset, MatrixConvolution, Tile (and three multi-input types with
  documented limitations) (GAP-N4/N5/N6)
- **SkSL tree-walking interpreter**: RuntimeShader::sample and
  RuntimeColorFilter::filter_color now evaluate SkSL at runtime with
  full Stmt/Expr coverage and approximately 35 built-in functions (GAP-C9/C10)
- **Full WGSL code generation**: every Stmt and Expr variant now emits
  valid WGSL (GAP-C11)
- **Dedicated MSL code generation**: separate emitters using MSL type
  syntax (float4, discard_fragment, etc.) (GAP-C12)
- **SPIR-V compilation**: compile_to(SpirV) and compile_to_spirv() via
  naga WGSL to SPIR-V path (GAP-N9)
- **SkSL semantic validation**: type checking, scope resolution, arity
  checks, return-type validation, builtin signatures (GAP-N10)
- **SkSL preprocessor**: #define, #undef, #ifdef, #ifndef, #else, #endif
  with whole-identifier text substitution (GAP-N11)
- **SkSL layout qualifier parsing**: layout(color) uniform and similar
  annotations now parse cleanly (GAP-N11)
- **ColorMatrixFilter convenience constructors**: brightness, contrast,
  hue_rotate, invert, sepia, grayscale (GAP-N1)
- **EffectKind entry-point validation**: main(vec2)->vec4 for Shader,
  main(vec4)->vec4 for ColorFilter, main(vec4,vec4)->vec4 for Blender
  (GAP-N8)
- **RuntimeEffect compilation caches**: GLSL/WGSL/MSL/SPIR-V results
  cached via OnceLock (GAP-N7)
- **Paint serialization** now round-trips shader, mask_filter, color_filter,
  and image_filter (GAP-C2)

### Changed (skia-rs-paint)

- GradientFlags now uses the bitflags crate (GAP-N2)
- GradientFlags::INTERPOLATE_PREMUL flag is honored in gradient
  interpolation (GAP-N2)
- RuntimeEffectError now uses thiserror derive (GAP-N12)

### Tests (skia-rs-paint)

- Test count grew from 17 to 145+ across skia-rs-paint
- Added property-based tests via proptest for blend mode finiteness
  and serialization round-trips
- All 31 audit gaps have test coverage

## [0.2.2] - 2026-04-25

### Added (skia-rs-path)

- **PathMeasure fully implemented**: `compute_lengths`, `get_point_at`,
  `get_tangent_at`, `get_matrix_at`, `get_segment`, `contour_count`,
  `contour_length` now functional (GAP-C1)
- Adaptive curve flattening utilities (`flatten_quad_adaptive`,
  `flatten_cubic_adaptive`, `flatten_conic_adaptive`) in internal
  `flatten` module
- `StrokeJoin::Round` now generates actual arc segments instead of a
  straight-line midpoint approximation (GAP-N7)
- `Path::tight_bounds()` now computes from quadratic and cubic curve
  extrema, not control points (GAP-N1)
- `Path::is_oval()` verifies cardinal-point endpoint geometry rather
  than only counting verb types (GAP-N4)
- `Path::convexity()` result is now cached via `AtomicU8` for repeat
  calls — Send+Sync compatible (GAP-N5)

### Fixed (skia-rs-path)

- `TrimEffect::apply` now extracts the actual sub-segment using
  `PathMeasure::get_segment` (GAP-C2). `Path1DEffect` (GAP-C3)
  auto-fixed by the same PathMeasure implementation.
- `DashEffect` now flattens curves before applying dash intervals,
  allowing dash transitions mid-curve (GAP-C6)
- `stroke_to_fill` tracks `is_closed` per contour, fixing multi-contour
  paths with mixed open/closed states (GAP-N8)
- Conic weight is now honored in `path_to_polygons` via the
  rational-quadratic evaluator (GAP-N6)
- Removed unused dependencies (`thiserror`, `arrayvec`, `proptest`)
- Cleaned up compiler warnings (unused `Verb` import, unnecessary `mut`)

### Known Limitations (skia-rs-path)

- Boolean operations (`PathOp::Difference`, `Intersect`, `Xor`) still
  have known correctness issues for non-convex inputs and partial
  overlaps. Tracked as GAP-C4 and GAP-C5. Limitations documented in
  the public rustdoc. Planned for a future release with a proper
  polygon-clipping algorithm (Weiler-Atherton or sweep-line).
- `Path::length()` and `Path::contains()` use control-polygon
  approximations for curves. Acceptable for typical use; tracked for
  incremental improvement.

## [0.2.1] - 2026-04-25

### Fixed - skia-rs-core

- `Matrix::map_point` now guards against divide-by-zero in perspective transforms (GAP-C4)
- `Region::contains_rect` correctly handles rects spanning multiple components (GAP-C2)
- `Region::union` merges overlapping rectangles to prevent unbounded growth (GAP-C3)
- `Matrix::skew` uses raw skew factors matching Skia's API (GAP-N1)
- `Matrix::invert` uses f32-appropriate singularity threshold (GAP-N5)
- `RRect::from_rect_xy` clamps radii to half-dimensions (GAP-N2)

### Added - skia-rs-core

- `Matrix44::ortho_checked` and `perspective_checked` variants returning `Option` (GAP-N4)
- `Matrix44::get`/`set` now have explicit bounds-check messages (GAP-N3)
- `convert_pixels` handles alpha type conversion (Premul ↔ Unpremul) (GAP-N6)
- Documented limitation in `IccProfile::from_bytes` regarding non-sRGB profiles (GAP-C1)

### Removed - skia-rs-core

- Orphan source files `matrix.rs` and `scalar.rs` (dead code) (GAP-C5, GAP-N7)

### Known Limitations - skia-rs-core

- `IccProfile::from_bytes` does not yet parse tag tables; non-sRGB profiles will
  report incorrect color space metadata. Tracked as GAP-C1, deferred to a
  future release for full ICC support.

## [0.1.0] - 2026-01-02

### 🎉 Initial Release

The first public release of skia-rs, a pure Rust implementation of the Skia 2D graphics library.

### Added

#### Core Types (`skia-rs-core`)
- `Point`, `IPoint` - 2D point types with arithmetic operations
- `Rect`, `IRect` - Rectangle types with intersection, union, and containment
- `RRect` - Rounded rectangle with per-corner radii
- `Matrix` - 3x3 transformation matrix with all standard operations
- `Matrix44` - 4x4 transformation matrix for 3D graphics
- `Color`, `Color4f` - 32-bit and floating-point color types
- `ColorSpace` - sRGB and linear color space support
- `ImageInfo`, `Pixmap`, `Bitmap` - Pixel storage and metadata
- `Region` - Complex clipping region with boolean operations
- ICC profile support for color management

#### Path System (`skia-rs-path`)
- `Path` - Complete path representation with all verb types
- `PathBuilder` - Fluent API for path construction
- `PathMeasure` - Path length and position calculations
- `PathOps` - Boolean operations (union, intersect, difference, xor)
- `PathEffect` - Dash, corner, discrete, trim effects
- SVG path parsing with full command support
- Arc approximation using cubic Bézier curves

#### Paint & Effects (`skia-rs-paint`)
- `Paint` - Full paint properties (color, stroke, anti-alias, etc.)
- `BlendMode` - All Porter-Duff and advanced blend modes
- `Style` - Fill, stroke, and stroke-and-fill styles
- Shaders:
  - `ColorShader` - Solid color
  - `LinearGradient`, `RadialGradient`, `SweepGradient` - Gradient fills
  - `TwoPointConicalGradient` - Two-point conical gradients
  - `ImageShader` - Image-based shaders
  - `BlendShader`, `ComposeShader` - Shader composition
  - `PerlinNoiseShader` - Procedural noise
- Color filters: matrix, lighting, blend mode
- Mask filters: blur, shader-based, table/gamma
- Image filters: blur, drop shadow, morphology, displacement, lighting, convolution

#### Canvas & Drawing (`skia-rs-canvas`)
- `Surface` - Drawing target with pixel storage
- `Canvas` - Full drawing API with save/restore stack
- Software rasterizer with anti-aliased rendering
- Drawing operations:
  - Shapes: rect, round rect, oval, circle, arc, path
  - Lines and points
  - Images with various sampling options
  - Text (via text crate integration)
- Clipping: rect and path-based
- Transformations: translate, rotate, scale, skew, concat
- `Picture` and `PictureRecorder` for recording/playback

#### Text (`skia-rs-text`)
- `Typeface` - Font face abstraction
- `Font` - Font with size and style properties
- `FontMetrics` - Font measurement data
- `TextBlob`, `TextBlobBuilder` - Positioned glyph runs
- `FontMgr` - Font enumeration and matching
- `Paragraph`, `ParagraphBuilder` - Rich text layout
- Text shaping via rustybuzz integration

#### Image Codecs (`skia-rs-codec`)
- `Image` - Immutable image with pixel access
- PNG encoding and decoding
- JPEG encoding and decoding
- GIF encoding and decoding
- WebP encoding and decoding
- Automatic format detection

#### GPU (`skia-rs-gpu`)
- wgpu backend foundation (in progress)
- `WgpuContext` - GPU context management
- `WgpuSurface` - GPU-backed surfaces
- `WgpuTexture` - Texture management

#### SVG (`skia-rs-svg`)
- SVG DOM parsing
- Basic element support (rect, circle, ellipse, path, etc.)
- Style attribute parsing

#### PDF (`skia-rs-pdf`)
- `PdfDocument` - PDF document creation
- `PdfPage` - Page management
- `PdfCanvas` - Drawing to PDF

#### FFI (`skia-rs-ffi`)
- C-compatible bindings for core types
- Opaque pointer-based API
- Static and dynamic library outputs

#### Safe API (`skia-rs-safe`)
- Unified re-export of all crates
- Feature flags for optional components
- High-level ergonomic API

### Performance

- Optimized software rasterizer achieving 68x speedup for rectangle fills
- SIMD-friendly memory layouts
- Minimal allocations in hot paths
- Comprehensive benchmark suite

### Testing

- Unit tests for all public APIs
- Property-based testing with proptest
- 17 fuzz targets covering major subsystems
- CI/CD with GitHub Actions

### Documentation

- Rustdoc for all public items
- Example code in documentation
- Benchmark documentation (`BENCHMARK.md`)


[0.1.0]: https://github.com/quinnjr/skia-rs/releases/tag/v0.1.0
[0.3.0]: https://github.com/quinnjr/skia-rs/compare/v0.2.7...v0.3.0
[0.4.0]: https://github.com/quinnjr/skia-rs/compare/v0.3.0...v0.4.0
[Unreleased]: https://github.com/quinnjr/skia-rs/compare/v0.4.0...HEAD

