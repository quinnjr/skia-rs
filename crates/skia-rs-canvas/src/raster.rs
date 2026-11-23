//! Rasterizer for drawing primitives to pixel buffers.
//!
//! This module provides software rasterization for basic shapes.
//!
//! ## Active Edge Table Algorithm
//!
//! For path filling, this module implements an optimized Active Edge Table (AET)
//! scanline algorithm with the following characteristics:
//!
//! - **Global Edge Table (GET)**: Edges sorted by `y_min` for efficient activation
//! - **Active Edge Table (AET)**: Only edges intersecting the current scanline
//! - **Winding Number Calculation**: Supports both non-zero and even-odd fill rules
//! - **X-Intersection Sorting**: Uses insertion sort optimized for nearly-sorted data
//!
//! The algorithm has O(n log n) setup time and O(n) per-scanline time, where n is
//! the number of edges, making it efficient for complex paths.
//!
//! ## Clipping
//!
//! The rasterizer supports multiple clipping modes:
//!
//! - **Rectangular clip**: Fast path for simple rectangular clips
//! - **Region-based clip**: Complex clips composed of multiple rectangles
//! - **Anti-aliased clip**: Smooth clip edges using coverage masks

#![allow(
    clippy::suboptimal_flops,
    reason = "faithful port of Skia's fma-free reference arithmetic; mul_add's fused rounding diverges from Skia"
)]

use skia_rs_core::premultiply_color;
use skia_rs_core::{Color, Color4f, IRect, Matrix, Point, Rect, Region, Scalar};
use skia_rs_paint::{BlendMode, Paint, Style};
use skia_rs_path::PathBuilder;

use crate::simd::mul_div_255_round;
use skia_rs_path::{FillType, Path, PathElement, Verb};

use crate::clip::ClipStack;

/// A pixel buffer for rasterization.
#[derive(Debug, Clone)]
pub struct PixelBuffer {
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
    /// RGBA pixel data (4 bytes per pixel).
    pub pixels: Vec<u8>,
    /// Row stride in bytes.
    pub stride: usize,
}

impl PixelBuffer {
    /// Create a new pixel buffer.
    #[must_use]
    pub fn new(width: i32, height: i32) -> Self {
        let stride = usize::try_from(width).unwrap_or(0) * 4;
        let pixels = vec![0u8; usize::try_from(height).unwrap_or(0) * stride];
        Self {
            width,
            height,
            pixels,
            stride,
        }
    }

    /// Clear the buffer with a color.
    ///
    /// `SkCanvas::clear` is `drawColor(color, SkBlendMode::kSrc)`; the buffer
    /// stores premultiplied pixels, so the color is premultiplied first.
    #[inline]
    pub fn clear(&mut self, color: Color) {
        let pm = premultiply_color(color);
        let r = pm.red();
        let g = pm.green();
        let b = pm.blue();
        let a = pm.alpha();

        // Optimize for common case of fully transparent or opaque clear
        if a == 0 && r == 0 && g == 0 && b == 0 {
            self.pixels.fill(0);
            return;
        }

        // Create a 4-byte pattern and fill using chunks
        let pattern = [r, g, b, a];
        for chunk in self.pixels.as_chunks_mut::<4>().0 {
            chunk.copy_from_slice(&pattern);
        }
    }

    /// Get a pixel at (x, y).
    #[inline]
    #[must_use]
    pub fn get_pixel(&self, x: i32, y: i32) -> Option<Color> {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return None;
        }
        let offset =
            usize::try_from(y).unwrap_or(0) * self.stride + usize::try_from(x).unwrap_or(0) * 4;
        Some(Color::from_argb(
            self.pixels[offset + 3],
            self.pixels[offset],
            self.pixels[offset + 1],
            self.pixels[offset + 2],
        ))
    }

    /// Set a pixel at (x, y).
    #[inline]
    pub fn set_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }
        let offset =
            usize::try_from(y).unwrap_or(0) * self.stride + usize::try_from(x).unwrap_or(0) * 4;
        self.pixels[offset] = color.red();
        self.pixels[offset + 1] = color.green();
        self.pixels[offset + 2] = color.blue();
        self.pixels[offset + 3] = color.alpha();
    }

    /// Blend a **premultiplied** pixel at (x, y) using the given blend mode.
    ///
    /// `src` must already be premultiplied; the buffer stores premultiplied
    /// pixels, and `blend_colors` operates on premultiplied inputs.
    #[inline]
    pub fn blend_pixel(&mut self, x: i32, y: i32, src: Color, blend_mode: BlendMode) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }

        // Fast path for fully opaque source with SrcOver (most common case)
        if blend_mode == BlendMode::SrcOver && src.alpha() == 255 {
            self.set_pixel(x, y, src);
            return;
        }

        // Fast path for fully transparent source
        if src.alpha() == 0 && matches!(blend_mode, BlendMode::SrcOver | BlendMode::Src) {
            if blend_mode == BlendMode::Src {
                self.set_pixel(x, y, Color::from_argb(0, 0, 0, 0));
            }
            return;
        }

        let dst = self.get_pixel(x, y).unwrap_or(Color::from_argb(0, 0, 0, 0));
        let blended = blend_colors(src, dst, blend_mode);
        self.set_pixel(x, y, blended);
    }

    /// Blend a pixel with coverage (alpha) for anti-aliasing.
    /// Coverage is 0.0 to 1.0 representing how much of the pixel is covered.
    #[inline]
    pub fn blend_pixel_aa(
        &mut self,
        x: i32,
        y: i32,
        src: Color,
        coverage: f32,
        blend_mode: BlendMode,
    ) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }

        if coverage <= 0.0 {
            return;
        }

        let cov = skia_rs_core::cast::f32_to_u8_sat(coverage.min(1.0) * 255.0);
        self.blend_pixel_coverage(x, y, src, cov, blend_mode);
    }

    /// Blend `src` (premultiplied) into the pixel with partial `coverage`
    /// (0-255), the way Skia's blitters do: `lerp(dst, blend(src, dst),
    /// coverage)`. For `SrcOver` that equals blending the coverage-scaled
    /// source, which is the fast path kept here; other modes (`Src`,
    /// `Clear`, ...) must lerp, or partially covered pixels would be
    /// replaced instead of blended toward.
    pub fn blend_pixel_coverage(
        &mut self,
        x: i32,
        y: i32,
        src: Color,
        coverage: u8,
        blend_mode: BlendMode,
    ) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height || coverage == 0 {
            return;
        }
        if coverage == 255 {
            self.blend_pixel(x, y, src, blend_mode);
            return;
        }
        let dst = self.get_pixel(x, y).unwrap_or(Color::from_argb(0, 0, 0, 0));
        let result = if blend_mode == BlendMode::SrcOver {
            blend_colors(apply_coverage(src, coverage), dst, blend_mode)
        } else {
            let full = blend_colors(src, dst, blend_mode);
            let lerp = |from: u8, to: u8| {
                let (from, to, cov) = (u32::from(from), u32::from(to), u32::from(coverage));
                u8::try_from((from * (255 - cov) + to * cov + 127) / 255).unwrap_or(u8::MAX)
            };
            Color::from_argb(
                lerp(dst.alpha(), full.alpha()),
                lerp(dst.red(), full.red()),
                lerp(dst.green(), full.green()),
                lerp(dst.blue(), full.blue()),
            )
        };
        self.set_pixel(x, y, result);
    }
}

/// Blend two **premultiplied** colors using a blend mode, returning a
/// premultiplied result.
///
/// This is the single blend implementation for the raster pipeline. The
/// hot Porter-Duff cases (`SrcOver`, `Src`, `Dst`, `Clear`) use the exact
/// integer `SkMulDiv255Round` form so they agree bit-for-bit with the SIMD
/// span blitters. Every other mode delegates to `skia-rs-paint`'s
/// [`BlendMode::apply`], which implements the full Porter-Duff / separable /
/// non-separable set on premultiplied [`Color4f`] values (Task 3). There is
/// no second blend implementation here.
fn blend_colors(src: Color, dst: Color, mode: BlendMode) -> Color {
    match mode {
        BlendMode::SrcOver => crate::simd::src_over_premul(src, dst),
        BlendMode::Src => src,
        BlendMode::Dst => dst,
        BlendMode::Clear => Color::TRANSPARENT,
        _ => {
            // Bytes are premultiplied; Color4f::from_color yields premul
            // components in [0, 1], exactly what BlendMode::apply expects.
            let s = Color4f::from_color(src);
            let d = Color4f::from_color(dst);
            mode.apply(s, d).to_color()
        }
    }
}

/// Test-only re-export of [`blend_colors`] so the SIMD differential tests can
/// assert bit-exact agreement with the per-pixel blend path.
#[cfg(test)]
pub(crate) fn blend_colors_for_test(src: Color, dst: Color, mode: BlendMode) -> Color {
    blend_colors(src, dst, mode)
}

/// Scale a **premultiplied** color by an 8-bit coverage value.
///
/// For premultiplied storage, coverage attenuates all four channels
/// (RGB and A), not just alpha. Uses rounded `SkMulDiv255Round`.
#[inline]
fn apply_coverage(color: Color, coverage: u8) -> Color {
    if coverage == 255 {
        return color;
    }
    let c = u32::from(coverage);
    Color::from_argb(
        mul_div_255_round(u32::from(color.alpha()), c),
        mul_div_255_round(u32::from(color.red()), c),
        mul_div_255_round(u32::from(color.green()), c),
        mul_div_255_round(u32::from(color.blue()), c),
    )
}

/// Per-fill pixel source: a premultiplied solid color, or a shader sampled
/// in **local** (pre-CTM) space.
enum PixelSource<'p> {
    /// Premultiplied solid paint color.
    Solid(Color),
    /// Shader with the device→local inverse CTM and the paint's alpha
    /// (0-255) to modulate the shader output.
    Shader {
        shader: &'p dyn skia_rs_paint::Shader,
        inv: Matrix,
        alpha: u8,
    },
}

/// Sample the source at device pixel (x, y). Returns a PREMULTIPLIED color.
///
/// Shaders are sampled at the pixel center mapped through the inverse CTM
/// (local space, per `SkShaderBase::appendRootStages` applying the inverse
/// total matrix); `Shader::sample` returns premul (Task 3 contract) which is
/// then modulated by the paint alpha.
fn source_color_at(source: &PixelSource<'_>, x: i32, y: i32) -> Color {
    match source {
        PixelSource::Solid(c) => *c,
        PixelSource::Shader { shader, inv, alpha } => {
            let local = inv.map_point(Point::new(
                skia_rs_core::cast::scalar_from_i32(x) + 0.5,
                skia_rs_core::cast::scalar_from_i32(y) + 0.5,
            ));
            let c = shader.sample(local.x, local.y).to_color();
            apply_coverage(c, *alpha)
        }
    }
}

/// Compute the fill spans of `edges` at scanline `y` (typically a pixel
/// center). Returns sorted, disjoint `(x0, x1)` pairs.
fn spans_at_scanline(edges: &[Edge], y: f32, fill_type: FillType) -> Vec<(f32, f32)> {
    let mut xs: Vec<(f32, i32)> = edges
        .iter()
        .filter(|e| y >= e.y_min && y < e.y_max)
        .map(|e| (e.x_at(y), e.winding))
        .collect();
    if xs.is_empty() {
        return Vec::new();
    }
    xs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut spans = Vec::new();
    match fill_type {
        FillType::Winding | FillType::InverseWinding => {
            let mut winding = 0i32;
            let mut span_start = 0.0f32;
            for &(x, w) in &xs {
                let was_inside = winding != 0;
                winding += w;
                let is_inside = winding != 0;
                if !was_inside && is_inside {
                    span_start = x;
                } else if was_inside && !is_inside {
                    spans.push((span_start, x));
                }
            }
        }
        FillType::EvenOdd | FillType::InverseEvenOdd => {
            let mut inside = false;
            let mut span_start = 0.0f32;
            for &(x, _) in &xs {
                inside = !inside;
                if inside {
                    span_start = x;
                } else {
                    spans.push((span_start, x));
                }
            }
        }
    }
    spans
}

/// Rasterizer for drawing to a pixel buffer.
pub struct Rasterizer<'a> {
    buffer: &'a mut PixelBuffer,
    /// Simple rectangular clip (for backward compatibility and fast path).
    clip: Rect,
    /// Advanced clip stack with region and AA support.
    clip_stack: ClipStack,
    /// Whether to use the advanced clip stack.
    use_advanced_clip: bool,
    matrix: Matrix,
    /// Device origin of the buffer: buffer pixel (x, y) sits at device pixel
    /// (x + origin.0, y + origin.1). Non-zero for `save_layer` buffers whose
    /// clip stack remains in device coordinates.
    origin: (i32, i32),
}

impl<'a> Rasterizer<'a> {
    /// Create a new rasterizer.
    #[must_use]
    pub const fn new(buffer: &'a mut PixelBuffer) -> Self {
        use skia_rs_core::cast::scalar_from_i32;
        let clip = Rect::from_xywh(
            0.0,
            0.0,
            scalar_from_i32(buffer.width),
            scalar_from_i32(buffer.height),
        );
        let clip_stack = ClipStack::new(&clip);
        Self {
            buffer,
            clip,
            clip_stack,
            use_advanced_clip: false,
            matrix: Matrix::IDENTITY,
            origin: (0, 0),
        }
    }

    /// Set the buffer's device origin (see the `origin` field). The clip
    /// stack stays in device coordinates; all clip queries add this offset.
    pub const fn set_origin(&mut self, x: i32, y: i32) {
        self.origin = (x, y);
    }

    /// The buffer's device origin.
    #[must_use]
    pub const fn origin(&self) -> (i32, i32) {
        self.origin
    }

    /// Set the current transformation matrix.
    pub const fn set_matrix(&mut self, matrix: &Matrix) {
        self.matrix = *matrix;
    }

    /// Set the clip rectangle (simple mode).
    pub const fn set_clip(&mut self, clip: Rect) {
        self.clip = clip;
        self.use_advanced_clip = false;
    }

    /// Takes the clip stack back out (leaving an empty one), for a caller
    /// that lent it with [`set_clip_stack`](Self::set_clip_stack).
    pub const fn take_clip_stack(&mut self) -> ClipStack {
        std::mem::replace(&mut self.clip_stack, ClipStack::new(&Rect::EMPTY))
    }

    /// Set the clip stack (advanced mode).
    pub fn set_clip_stack(&mut self, clip_stack: ClipStack) {
        self.clip_stack = clip_stack;
        self.clip = self.clip_stack.bounds();
        self.use_advanced_clip = true;
    }

    /// Get the device bounds as an `IRect`.
    const fn device_bounds(&self) -> IRect {
        IRect::new(0, 0, self.buffer.width, self.buffer.height)
    }

    /// Save the current clip state.
    pub fn save_clip(&mut self) {
        self.use_advanced_clip = true;
        self.clip_stack.save();
    }

    /// Restore the previous clip state.
    pub fn restore_clip(&mut self) {
        self.clip_stack.restore();
    }

    /// Clip to a region.
    ///
    /// This enables advanced clipping mode with support for complex
    /// multi-rectangle clip areas.
    pub fn clip_region(&mut self, region: &Region) {
        self.use_advanced_clip = true;
        self.clip_stack.clip_region(region);
    }

    /// Clip to a path.
    ///
    /// If `anti_alias` is true, the clip edges will be anti-aliased
    /// using a coverage mask.
    pub fn clip_path(&mut self, path: &Path, anti_alias: bool) {
        self.use_advanced_clip = true;
        let device_bounds = self.device_bounds();
        self.clip_stack.clip_path(path, &device_bounds, anti_alias);
    }

    /// Clip to a rectangle with optional anti-aliasing.
    pub fn clip_rect_aa(&mut self, rect: &Rect, anti_alias: bool) {
        self.use_advanced_clip = true;
        if anti_alias {
            let device_bounds = self.device_bounds();
            self.clip_stack.clip_rect_aa(rect, &device_bounds);
        } else {
            self.clip_stack.clip_rect(rect);
        }
    }

    /// Get the clip coverage at a point in buffer coordinates (0-255).
    /// Returns 255 for simple clips if the point is inside.
    #[inline]
    fn get_clip_coverage(&self, x: i32, y: i32) -> u8 {
        use skia_rs_core::cast::scalar_from_i32;
        // Saturating: callers may pass saturated indices for far-off
        // geometry, and a layer's origin must not overflow them.
        let (dx, dy) = (
            x.saturating_add(self.origin.0),
            y.saturating_add(self.origin.1),
        );
        if self.use_advanced_clip {
            self.clip_stack.get_coverage(dx, dy)
        } else if self
            .clip
            .contains(Point::new(scalar_from_i32(dx), scalar_from_i32(dy)))
        {
            255
        } else {
            0
        }
    }

    /// Get the current clip bounds in buffer coordinates.
    #[must_use]
    pub fn clip_bounds(&self) -> Rect {
        use skia_rs_core::cast::scalar_from_i32;
        let b = if self.use_advanced_clip {
            self.clip_stack.bounds()
        } else {
            self.clip
        };
        if self.origin == (0, 0) {
            b
        } else {
            Rect::new(
                b.left - scalar_from_i32(self.origin.0),
                b.top - scalar_from_i32(self.origin.1),
                b.right - scalar_from_i32(self.origin.0),
                b.bottom - scalar_from_i32(self.origin.1),
            )
        }
    }

    /// Check if the current clip is anti-aliased.
    #[must_use]
    pub const fn is_clip_anti_aliased(&self) -> bool {
        self.use_advanced_clip && self.clip_stack.is_anti_aliased()
    }

    /// Build the pixel source for `paint`: a premultiplied solid color, or
    /// the paint's shader with the inverse of (CTM x shader local matrix)
    /// for local-space sampling (`Shader::sample` expects coordinates in the
    /// shader's own space; the caller applies the local matrix).
    fn make_source<'p>(&self, paint: &'p Paint) -> PixelSource<'p> {
        paint.shader().map_or_else(
            || PixelSource::Solid(premultiply_color(paint.color32())),
            |shader| {
                let total = shader
                    .local_matrix()
                    .map_or(self.matrix, |local| self.matrix.concat(local));
                let inv = total.invert().unwrap_or(Matrix::IDENTITY);
                let alpha =
                    skia_rs_core::cast::f32_to_u8_sat(paint.alpha().clamp(0.0, 1.0) * 255.0);
                PixelSource::Shader {
                    shader: shader.as_ref(),
                    inv,
                    alpha,
                }
            },
        )
    }

    /// Integer pixel bounds of the current clip, clamped to the buffer:
    /// `(x0, y0, x1, y1)` half-open.
    fn clip_pixel_bounds(&self) -> (i32, i32, i32, i32) {
        use skia_rs_core::cast::{ceil_to_i32, floor_to_i32};
        let clip = self.clip_bounds();
        (
            floor_to_i32(clip.left).max(0),
            floor_to_i32(clip.top).max(0),
            ceil_to_i32(clip.right).min(self.buffer.width),
            ceil_to_i32(clip.bottom).min(self.buffer.height),
        )
    }

    /// Blend one pixel with geometry `coverage` (0-255), combining it with
    /// the clip coverage. The source is sampled per pixel (shader) or solid.
    fn blit_pixel_cov(
        &mut self,
        x: i32,
        y: i32,
        coverage: u8,
        source: &PixelSource<'_>,
        blend_mode: BlendMode,
    ) {
        if coverage == 0 {
            return;
        }
        let clip_cov = self.get_clip_coverage(x, y);
        if clip_cov == 0 {
            return;
        }
        let combined = if clip_cov == 255 {
            coverage
        } else {
            mul_div_255_round(u32::from(coverage), u32::from(clip_cov))
        };
        let color = source_color_at(source, x, y);
        self.buffer
            .blend_pixel_coverage(x, y, color, combined, blend_mode);
    }

    /// Blend a full-coverage horizontal span `[x0, x1)` at row `y` through
    /// the clip. Solid sources take the SIMD-accelerated hline path; shader
    /// sources sample per pixel in local space.
    fn blit_span(
        &mut self,
        x0: i32,
        x1: i32,
        y: i32,
        source: &PixelSource<'_>,
        blend_mode: BlendMode,
    ) {
        if x0 >= x1 {
            return;
        }
        match source {
            PixelSource::Solid(c) => self.draw_hline(x0, x1 - 1, y, *c, blend_mode),
            PixelSource::Shader { .. } => {
                let (cx0, cy0, cx1, cy1) = self.clip_pixel_bounds();
                if y < cy0 || y >= cy1 {
                    return;
                }
                for x in x0.max(cx0)..x1.min(cx1) {
                    self.blit_pixel_cov(x, y, 255, source, blend_mode);
                }
            }
        }
    }

    /// Reset the clip to device bounds.
    pub fn reset_clip(&mut self) {
        use skia_rs_core::cast::scalar_from_i32;
        let bounds = Rect::from_xywh(
            0.0,
            0.0,
            scalar_from_i32(self.buffer.width),
            scalar_from_i32(self.buffer.height),
        );
        self.clip = bounds;
        self.clip_stack.reset(&bounds);
        self.use_advanced_clip = false;
    }

    /// Draw a point.
    #[allow(
        clippy::similar_names,
        reason = "point/paint mirrors SkCanvas::drawPoint's (SkPoint, SkPaint) parameter names"
    )]
    pub fn draw_point(&mut self, point: Point, paint: &Paint) {
        use skia_rs_core::cast::floor_to_i32;
        // Under perspective, a point at or behind the w plane maps to a
        // flipped position: draw nothing, as for lines (clip_to_w_plane).
        if clip_to_w_plane(&self.matrix, point, point).is_none() {
            return;
        }
        let device = self.matrix.map_point(point);
        // Points off the device (beyond the 1-px AA square) draw nothing,
        // and must not reach the pixel indexing with saturated values.
        let (w, h) = (
            skia_rs_core::cast::scalar_from_i32(self.buffer.width),
            skia_rs_core::cast::scalar_from_i32(self.buffer.height),
        );
        if !(device.x > -2.0 && device.x < w + 2.0 && device.y > -2.0 && device.y < h + 2.0) {
            return;
        }
        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();
        if !paint.is_anti_alias() {
            // The pixel containing the point (bw_pt_hair_proc floors).
            let (x, y) = (floor_to_i32(device.x), floor_to_i32(device.y));
            self.blit_pixel_cov(x, y, 255, &source, blend_mode);
            return;
        }
        // aa_square_proc: an anti-aliased 1x1 square centered on the point,
        // covering up to four pixels by area.
        let (left, top) = (device.x - 0.5, device.y - 0.5);
        let (x0, y0) = (floor_to_i32(left), floor_to_i32(top));
        let (fx, fy) = (left - left.floor(), top - top.floor());
        for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
            for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
                let coverage = skia_rs_core::cast::f32_to_u8_sat((wx * wy * 255.0).round());
                self.blit_pixel_cov(
                    x0.saturating_add(dx),
                    y0.saturating_add(dy),
                    coverage,
                    &source,
                    blend_mode,
                );
            }
        }
    }

    /// Draw a line segment.
    ///
    /// Matches `SkCanvas::drawLine`, which is `drawPoints(kLines_PointMode)`:
    /// the segment is always stroked (the paint's style is ignored) using
    /// the paint's stroke width, cap and join. A stroke width of zero is a
    /// hairline, drawn with Bresenham's algorithm (aliased) or Wu's
    /// algorithm (anti-aliased).
    ///
    /// A hairline (width 0) ignores the stroke cap, like Skia's
    /// `drawLine`/`drawPoints`, which blit hairlines through `PtProcRec`'s
    /// `bw_line_hair_proc` / `aa_line_hair_proc` (`SkScan::HairLine`).
    /// Only hairline *paths* extend their ends for caps (`hair_path`).
    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        // SkCanvas::drawPoints quick-rejects non-finite points.
        if ![p0.x, p0.y, p1.x, p1.y].iter().all(|v| v.is_finite()) {
            return;
        }
        if paint.stroke_width() > 0.0 {
            let mut b = PathBuilder::new();
            b.move_to(p0.x, p0.y).line_to(p1.x, p1.y);
            let mut stroke = paint.clone();
            stroke.set_style(Style::Stroke);
            self.draw_path(&b.build(), &stroke);
            return;
        }

        if paint.is_anti_alias() {
            self.draw_line_aa(p0, p1, paint);
        } else {
            self.draw_line_aliased(p0, p1, paint);
        }
    }

    /// Draw an aliased hairline, as Skia's `SkScan::HairLine` does.
    ///
    /// Along the major axis the span is half-open, `[round(a0), round(a1))`
    /// (`SkFDot6Round`), so joined segments do not overlap at shared
    /// vertices and a hairline shorter than a pixel draws nothing. Along the
    /// minor axis a DDA is sampled at each pixel's center and floored
    /// (`horiline` / `vertline`).
    fn draw_line_aliased(&mut self, p0: Point, p1: Point, paint: &Paint) {
        use skia_rs_core::cast::{floor_to_i32, scalar_from_i32};

        let Some((mut start, mut end)) = self.device_segment(p0, p1) else {
            return;
        };
        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();
        let horizontal = (end.x - start.x).abs() > (end.y - start.y).abs();
        if !horizontal {
            // Treat the mostly-vertical case as horizontal with x and y
            // swapped, swapping back when plotting.
            std::mem::swap(&mut start.x, &mut start.y);
            std::mem::swap(&mut end.x, &mut end.y);
        }
        if start.x > end.x {
            std::mem::swap(&mut start, &mut end);
        }
        let round = |v: Scalar| floor_to_i32(v + 0.5);
        let (first, past_last) = (round(start.x), round(end.x));
        if first >= past_last {
            return; // too short to draw
        }
        // Only walk the part of the major axis inside the device (Skia clips
        // the line before HairLineRgn), so far-off lines cost nothing.
        let extent = if horizontal {
            self.buffer.width
        } else {
            self.buffer.height
        };
        let (first, past_last) = (first.max(0), past_last.min(extent));
        if first >= past_last {
            return;
        }
        let slope = (end.y - start.y) / (end.x - start.x);
        // Minor-axis value at the center of the first pixel.
        let mut minor = slope * (scalar_from_i32(first) + 0.5 - start.x) + start.y;
        for major in first..past_last {
            let row = floor_to_i32(minor);
            let (x, y) = if horizontal {
                (major, row)
            } else {
                (row, major)
            };
            self.blit_pixel_cov(x, y, 255, &source, blend_mode);
            minor += slope;
        }
    }

    /// Draw line with anti-aliasing using Wu's algorithm.
    fn draw_line_aa(&mut self, p0: Point, p1: Point, paint: &Paint) {
        use skia_rs_core::cast::{floor_to_i32, scalar_from_i32};
        // Fractional part in [0, 1) for negative values too
        // (`f32::fract` is negative below zero, which the half-pixel
        // shift below produces near the top/left edges).
        let fpart = |v: Scalar| v - v.floor();
        // Wu's algorithm assumes pixel i is centered at i; Skia's pixel
        // centers are at i + 0.5. Shift by half a pixel so a line at
        // y = 50.5 lands entirely on row 50 (SkScan_Antihair).
        let Some((t0, t1)) = self.device_segment(p0, p1) else {
            return;
        };

        let mut x0 = t0.x - 0.5;
        let mut y0 = t0.y - 0.5;
        let mut x1 = t1.x - 0.5;
        let mut y1 = t1.y - 0.5;

        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();

        let steep = (y1 - y0).abs() > (x1 - x0).abs();

        if steep {
            std::mem::swap(&mut x0, &mut y0);
            std::mem::swap(&mut x1, &mut y1);
        }

        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }

        let dx = x1 - x0;
        let dy = y1 - y0;
        // dx >= 0 after the swap; a zero extent is handled by the
        // single-column branch below.
        let gradient = if dx > 0.0 { dy / dx } else { 0.0 };

        // A segment inside one column along the major axis covers it once,
        // by its length (as Skia's anti-hairline does for sub-pixel
        // segments); plotting both endpoints there would blend it twice.
        let (first_column, last_column) = (floor_to_i32(x0 + 0.5), floor_to_i32(x1 + 0.5));
        if first_column == last_column {
            let column = first_column;
            let y_mid = f32::midpoint(y0, y1);
            self.plot_wu_endpoint(steep, column, y_mid, dx.min(1.0), &source, blend_mode);
            return;
        }

        // Handle first endpoint
        // Round half up, so endpoints are translation-consistent across
        // zero (`f32::round` rounds half away from zero).
        let xend = (x0 + 0.5).floor();
        let yend = y0 + gradient * (xend - x0);
        let xgap = 1.0 - fpart(x0 + 0.5);
        let xpxl1 = floor_to_i32(xend);

        self.plot_wu_endpoint(steep, xpxl1, yend, xgap, &source, blend_mode);

        let mut intery = yend + gradient;

        // Handle second endpoint
        let xend = (x1 + 0.5).floor();
        let yend = y1 + gradient * (xend - x1);
        let xgap = fpart(x1 + 0.5);
        let xpxl2 = floor_to_i32(xend);

        self.plot_wu_endpoint(steep, xpxl2, yend, xgap, &source, blend_mode);

        // Main loop, limited to the device along the major axis.
        let extent = if steep {
            self.buffer.height
        } else {
            self.buffer.width
        };
        let (first, past_last) = (xpxl1.saturating_add(1).max(0), xpxl2.min(extent));
        // Re-base `intery` to `first` (in floating point: the integer gap
        // can exceed i32 for far-off lines).
        intery += gradient * (scalar_from_i32(first) - scalar_from_i32(xpxl1) - 1.0);
        if steep {
            for x in first..past_last {
                let y = floor_to_i32(intery);
                self.plot_aa(y, x, 1.0 - fpart(intery), &source, blend_mode);
                self.plot_aa(y.saturating_add(1), x, fpart(intery), &source, blend_mode);
                intery += gradient;
            }
        } else {
            for x in first..past_last {
                let y = floor_to_i32(intery);
                self.plot_aa(x, y, 1.0 - fpart(intery), &source, blend_mode);
                self.plot_aa(x, y.saturating_add(1), fpart(intery), &source, blend_mode);
                intery += gradient;
            }
        }
    }

    /// The buffer as a device-space rect.
    const fn device_rect(&self) -> Rect {
        use skia_rs_core::cast::scalar_from_i32;
        Rect::new(
            0.0,
            0.0,
            scalar_from_i32(self.buffer.width),
            scalar_from_i32(self.buffer.height),
        )
    }

    /// The hairline `p0`-`p1` mapped to device space and clipped to the
    /// device (outset by two pixels, so clipped ends stay off-device), as
    /// Skia clips lines before rasterizing them (`SkLineClipper`). This
    /// keeps far-off endpoints from saturating pixel indices or losing f32
    /// precision. `None` when the segment misses the device or is not
    /// finite.
    fn device_segment(&self, p0: Point, p1: Point) -> Option<(Point, Point)> {
        // Under perspective, first keep only the part in front of the w
        // plane (as PerspectiveClip does for paths): points behind it map to
        // flipped coordinates.
        let (p0, p1) = clip_to_w_plane(&self.matrix, p0, p1)?;
        let (a, b) = (self.matrix.map_point(p0), self.matrix.map_point(p1));
        let bounds = (
            -2.0,
            -2.0,
            f64::from(self.buffer.width) + 2.0,
            f64::from(self.buffer.height) + 2.0,
        );
        clip_segment(a, b, bounds)
    }

    /// Plot one Wu endpoint: the pixel pair straddling `yend` in column
    /// `major` (rows and columns swapped for steep lines), weighted by the
    /// endpoint's horizontal gap coverage `xgap`.
    fn plot_wu_endpoint(
        &mut self,
        steep: bool,
        major: i32,
        yend: Scalar,
        xgap: Scalar,
        source: &PixelSource<'_>,
        blend_mode: BlendMode,
    ) {
        use skia_rs_core::cast::floor_to_i32;
        let minor = floor_to_i32(yend);
        let frac = yend - yend.floor();
        let (near, far) = ((1.0 - frac) * xgap, frac * xgap);
        if steep {
            self.plot_aa(minor, major, near, source, blend_mode);
            self.plot_aa(minor.saturating_add(1), major, far, source, blend_mode);
        } else {
            self.plot_aa(major, minor, near, source, blend_mode);
            self.plot_aa(major, minor.saturating_add(1), far, source, blend_mode);
        }
    }

    /// Plot a pixel with coverage for anti-aliasing.
    #[inline]
    fn plot_aa(
        &mut self,
        x: i32,
        y: i32,
        coverage: f32,
        source: &PixelSource<'_>,
        blend_mode: BlendMode,
    ) {
        let clip_coverage = self.get_clip_coverage(x, y);
        if clip_coverage > 0 {
            let color = source_color_at(source, x, y);
            // Combine line AA coverage with clip coverage
            let combined_coverage = coverage * (f32::from(clip_coverage) / 255.0);
            self.buffer
                .blend_pixel_aa(x, y, color, combined_coverage, blend_mode);
        }
    }

    /// Draw a horizontal line (fast path with SIMD optimization).
    ///
    /// Uses SIMD-accelerated blitting when available for:
    /// - SSE4.2 on `x86/x86_64` (4 pixels at a time)
    /// - AVX2 on `x86/x86_64` (8 pixels at a time)
    /// - NEON on ARM/AArch64 (4 pixels at a time)
    fn draw_hline(&mut self, x0: i32, x1: i32, y: i32, color: Color, blend_mode: BlendMode) {
        use skia_rs_core::cast::saturate_to_i32;
        let clip_bounds = self.clip_bounds();
        let (start, end) = if x0 < x1 { (x0, x1) } else { (x1, x0) };
        let start = start.max(saturate_to_i32(clip_bounds.left.trunc()));
        let end = end.min(saturate_to_i32(clip_bounds.right.trunc()) - 1);

        if y < saturate_to_i32(clip_bounds.top.trunc())
            || y >= saturate_to_i32(clip_bounds.bottom.trunc())
        {
            return;
        }

        if start > end {
            return;
        }

        // For advanced clips (region-based or AA), use per-pixel with coverage
        if self.use_advanced_clip {
            // Partial clip coverage blends like every other blit path
            // (lerp toward the blend result for non-SrcOver modes).
            for x in start..=end {
                let coverage = self.get_clip_coverage(x, y);
                self.buffer
                    .blend_pixel_coverage(x, y, color, coverage, blend_mode);
            }
            return;
        }

        // Validate bounds
        if start < 0 || end >= self.buffer.width || y < 0 || y >= self.buffer.height {
            // Fall back to per-pixel with bounds checking
            for x in start..=end {
                self.buffer.blend_pixel(x, y, color, blend_mode);
            }
            return;
        }

        let row_offset = usize::try_from(y).unwrap_or(0) * self.buffer.stride;
        let start_offset = row_offset + usize::try_from(start).unwrap_or(0) * 4;
        let end_offset = row_offset + usize::try_from(end + 1).unwrap_or(0) * 4;

        // SIMD-optimized path for SrcOver blend mode (most common case)
        if blend_mode == BlendMode::SrcOver {
            crate::simd::fill_span_solid(&mut self.buffer.pixels[start_offset..end_offset], color);
            return;
        }

        // For other blend modes, use per-pixel blending
        for x in start..=end {
            self.buffer.blend_pixel(x, y, color, blend_mode);
        }
    }

    /// Returns true when the CTM maps axis-aligned rects to axis-aligned
    /// rects without rotation/skew (`map_rect` is then exact).
    fn matrix_is_axis_aligned(&self) -> bool {
        self.matrix.skew_x() == 0.0 && self.matrix.skew_y() == 0.0
    }

    /// Draw a filled rectangle.
    ///
    /// Under a non-axis-aligned CTM the rect is converted to a path and
    /// scan-converted as the mapped quad; only the axis-aligned case may use
    /// `map_rect` (which would otherwise fill the quad's bounding box).
    pub fn fill_rect(&mut self, rect: &Rect, paint: &Paint) {
        use skia_rs_core::cast::round_to_i32;
        // Perspective maps a rect to a quadrilateral, not its bounding box.
        if !self.matrix_is_axis_aligned() || self.matrix.has_perspective() || paint.is_anti_alias()
        {
            let mut b = PathBuilder::new();
            b.add_rect(rect);
            self.fill_path(&b.build(), paint);
            return;
        }

        let transformed = self.matrix.map_rect(rect);

        let x0 = round_to_i32(transformed.left);
        let x1 = round_to_i32(transformed.right);
        // Rows outside the buffer draw nothing; limiting them keeps huge or
        // far-off rects from looping over billions of empty rows.
        let y0 = round_to_i32(transformed.top).max(0);
        let y1 = round_to_i32(transformed.bottom).min(self.buffer.height);

        let blend_mode = paint.blend_mode();
        let source = self.make_source(paint);
        for y in y0..y1 {
            self.blit_span(x0, x1, y, &source, blend_mode);
        }
    }

    /// Fill a **device-space** rectangle with `paint`, bypassing the CTM for
    /// geometry while still sampling any shader through the CTM (local
    /// space). Used by `clear`/`drawColor`/`drawPaint`, which fill the device
    /// clip regardless of the matrix (`SkDraw::drawPaint`).
    pub fn fill_device_rect(&mut self, rect: &Rect, paint: &Paint) {
        use skia_rs_core::cast::round_to_i32;
        let x0 = round_to_i32(rect.left).saturating_sub(self.origin.0);
        let x1 = round_to_i32(rect.right).saturating_sub(self.origin.0);
        let y0 = round_to_i32(rect.top).saturating_sub(self.origin.1).max(0);
        let y1 = round_to_i32(rect.bottom)
            .saturating_sub(self.origin.1)
            .min(self.buffer.height);

        let blend_mode = paint.blend_mode();
        let source = self.make_source(paint);
        for y in y0..y1 {
            self.blit_span(x0, x1, y, &source, blend_mode);
        }
    }

    /// Draw a stroked rectangle, honoring the paint's stroke width.
    ///
    /// Width 0 is a hairline (single-pixel outline); positive widths build
    /// the stroke outline via `skia-rs-path`'s `stroke_to_fill` and fill it.
    pub fn stroke_rect(&mut self, rect: &Rect, paint: &Paint) {
        if paint.stroke_width() > 0.0 {
            let mut b = PathBuilder::new();
            b.add_rect(rect);
            self.stroke_path(&b.build(), paint);
            return;
        }

        // SkDraw::drawRect: an aliased hairline rect under a rect-preserving
        // CTM (rectStaysRect: scale/translate or a 90-degree rotation) is
        // blitted by SkScan::HairRect as a closed pixel frame. Other
        // hairline rects are four hairline segments, as in Skia:
        // SkScan::AntiHairRect is a 5-point AntiHairLine polyline (drawn
        // segment by segment, so AA corners blend twice there too), and a
        // non-rect-preserving CTM goes through hair_path's per-segment
        // HairLine.
        if !paint.is_anti_alias() && self.matrix.rect_stays_rect() {
            self.hair_rect(rect, paint);
            return;
        }

        let tl = Point::new(rect.left, rect.top);
        let tr = Point::new(rect.right, rect.top);
        let bl = Point::new(rect.left, rect.bottom);
        let br = Point::new(rect.right, rect.bottom);

        self.draw_line(tl, tr, paint);
        self.draw_line(tr, br, paint);
        self.draw_line(br, bl, paint);
        self.draw_line(bl, tl, paint);
    }

    /// `SkScan::HairRect`: blit the one-pixel frame of the device rect,
    /// spanning pixels `floor(left)..=floor(right)` and
    /// `floor(top)..=floor(bottom)` (a solid block when either side is at
    /// most two pixels).
    fn hair_rect(&mut self, rect: &Rect, paint: &Paint) {
        use skia_rs_core::cast::floor_to_i32;
        let dev = self.matrix.map_rect(rect);
        // Trim to the device outset by one (Skia intersects with the clip
        // bounds outset by 1) so the integer extents cannot overflow.
        let (w, h) = (self.buffer.width, self.buffer.height);
        let trim = |v: Scalar, hi: i32| floor_to_i32(v).clamp(-1, hi.saturating_add(1));
        let (left, top) = (trim(dev.left, w), trim(dev.top, h));
        let (right, bottom) = (trim(dev.right + 1.0, w), trim(dev.bottom + 1.0, h));
        if left >= right || top >= bottom {
            return;
        }
        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();
        if right - left <= 2 || bottom - top <= 2 {
            for y in top..bottom {
                self.blit_span(left, right, y, &source, blend_mode);
            }
            return;
        }
        self.blit_span(left, right, top, &source, blend_mode);
        for y in top + 1..bottom - 1 {
            self.blit_span(left, left + 1, y, &source, blend_mode);
            self.blit_span(right - 1, right, y, &source, blend_mode);
        }
        self.blit_span(left, right, bottom - 1, &source, blend_mode);
    }

    /// Draw a rectangle (filled or stroked based on paint style).
    pub fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
        // SkCanvas::drawRect quick-rejects rects with non-finite edges.
        if !rect.is_finite() {
            return;
        }
        match paint.style() {
            Style::Fill => self.fill_rect(rect, paint),
            Style::Stroke => self.stroke_rect(rect, paint),
            Style::StrokeAndFill => {
                let mut b = PathBuilder::new();
                b.add_rect(rect);
                self.stroke_and_fill_path(&b.build(), paint);
            }
        }
    }

    /// Draw a filled circle with disjoint per-row spans.
    ///
    /// Each row is emitted exactly once (analytic half-width per scanline),
    /// so translucent paints are blended exactly once per pixel — the old
    /// midpoint-octant loop emitted overlapping rows.
    ///
    /// Assumes a translate + uniform-scale CTM; [`draw_circle`](Self::draw_circle) routes any
    /// other matrix through the path pipeline.
    pub fn fill_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, round_to_i32, scalar_from_i32};
        let tc = self.matrix.map_point(center);
        let r = radius * self.matrix.scale_x().abs();
        if r <= 0.0 {
            return;
        }

        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();

        // Only rows inside the buffer (huge or far-off circles would
        // otherwise loop over billions of empty rows).
        let y0 = floor_to_i32(tc.y - r).max(0);
        let y1 = ceil_to_i32(tc.y + r).min(self.buffer.height);
        for y in y0..y1 {
            let dy = scalar_from_i32(y) + 0.5 - tc.y;
            if dy.abs() > r {
                continue;
            }
            let half = (r * r - dy * dy).sqrt();
            let x0 = round_to_i32(tc.x - half);
            let x1 = round_to_i32(tc.x + half);
            self.blit_span(x0, x1, y, &source, blend_mode);
        }
    }

    /// Draw a stroked circle.
    pub fn stroke_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        // Skia strokes circles as conic paths (a hairline circle goes
        // through hair_path), which also honors the clip and shader.
        let mut b = PathBuilder::new();
        b.add_circle(center.x, center.y, radius);
        self.stroke_path(&b.build(), paint);
    }

    /// Draw a circle (filled or stroked based on paint style).
    ///
    /// The analytic circle fast paths are only valid for a translate +
    /// uniform-scale CTM. Any other matrix (rotation, skew, non-uniform
    /// scale) converts the circle to a conic path and maps it through the
    /// full CTM via the path pipeline (an ellipse under non-uniform scale).
    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        let m = &self.matrix;
        let uniform = m.skew_x() == 0.0
            && m.skew_y() == 0.0
            && (m.scale_x().abs() - m.scale_y().abs()).abs() <= 1e-6;
        let needs_path = !uniform
            || m.has_perspective()
            || (paint.style() != Style::Fill && paint.stroke_width() > 0.0)
            || paint.style() == Style::StrokeAndFill;
        if needs_path {
            let mut b = PathBuilder::new();
            b.add_circle(center.x, center.y, radius);
            self.draw_path(&b.build(), paint);
            return;
        }

        if paint.is_anti_alias() {
            self.draw_circle_aa(center, radius, paint);
        } else {
            match paint.style() {
                Style::Fill => self.fill_circle(center, radius, paint),
                // Hairline stroke (width 0).
                Style::Stroke => self.stroke_circle(center, radius, paint),
                // StrokeAndFill is routed through the path pipeline above.
                Style::StrokeAndFill => unreachable!(),
            }
        }
    }

    /// Draw an anti-aliased circle.
    fn draw_circle_aa(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, scalar_from_i32};
        let tc = self.matrix.map_point(center);
        let cx = tc.x;
        let cy = tc.y;
        let r = radius * self.matrix.scale_x().abs();
        // An empty circle is Skia's empty oval: nothing for fills and butt
        // hairlines, but hair_path extends a degenerate contour's ends for
        // round/square caps, drawing a dot (as the aliased path does).
        if r.is_nan() || r <= 0.0 {
            if !r.is_nan()
                && paint.style() == Style::Stroke
                && paint.stroke_cap() != skia_rs_paint::StrokeCap::Butt
            {
                let mut b = PathBuilder::new();
                b.add_circle(center.x, center.y, 0.0);
                self.stroke_path(&b.build(), paint);
            }
            return;
        }

        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();

        // Bounding box, limited to the buffer: pixels outside it draw
        // nothing, and a huge or far-off circle must not loop over them.
        let min_x = floor_to_i32(cx - r - 1.0).max(0);
        let max_x = ceil_to_i32(cx + r + 1.0).min(self.buffer.width - 1);
        let min_y = floor_to_i32(cy - r - 1.0).max(0);
        let max_y = ceil_to_i32(cy + r + 1.0).min(self.buffer.height - 1);

        match paint.style() {
            Style::Fill => {
                // For each pixel in bounding box
                for py in min_y..=max_y {
                    for px in min_x..=max_x {
                        // Calculate distance from pixel center to circle center
                        let dx = scalar_from_i32(px) + 0.5 - cx;
                        let dy = scalar_from_i32(py) + 0.5 - cy;
                        let dist_sq = dx * dx + dy * dy;

                        // Calculate coverage using smoothstep
                        let dist = dist_sq.sqrt();
                        let coverage = if dist <= r - 0.5 {
                            1.0
                        } else if dist >= r + 0.5 {
                            0.0
                        } else {
                            // Smooth edge
                            1.0 - (dist - (r - 0.5))
                        };

                        if coverage > 0.0 {
                            self.plot_aa(px, py, coverage, &source, blend_mode);
                        }
                    }
                }
            }
            Style::Stroke => {
                let stroke_width = paint.stroke_width().max(1.0);
                let outer_r = r + stroke_width / 2.0;
                let inner_r = (r - stroke_width / 2.0).max(0.0);

                for py in min_y..=max_y {
                    for px in min_x..=max_x {
                        let dx = scalar_from_i32(px) + 0.5 - cx;
                        let dy = scalar_from_i32(py) + 0.5 - cy;
                        #[allow(
                            clippy::imprecise_flops,
                            reason = "per-pixel hot loop: hypot is markedly slower and its overflow guarding is unnecessary for pixel-space distances"
                        )]
                        let dist = (dx * dx + dy * dy).sqrt();

                        let outer_coverage = if dist <= outer_r - 0.5 {
                            1.0
                        } else if dist >= outer_r + 0.5 {
                            0.0
                        } else {
                            1.0 - (dist - (outer_r - 0.5))
                        };

                        let inner_coverage = if dist <= inner_r - 0.5 {
                            1.0
                        } else if dist >= inner_r + 0.5 {
                            0.0
                        } else {
                            1.0 - (dist - (inner_r - 0.5))
                        };

                        let coverage = outer_coverage - inner_coverage;

                        if coverage > 0.0 {
                            self.plot_aa(px, py, coverage, &source, blend_mode);
                        }
                    }
                }
            }
            Style::StrokeAndFill => {
                self.draw_circle_aa(center, radius, &{
                    let mut p = paint.clone();
                    p.set_style(Style::Fill);
                    p
                });
                self.draw_circle_aa(center, radius, &{
                    let mut p = paint.clone();
                    p.set_style(Style::Stroke);
                    p
                });
            }
        }
    }

    /// Draw an oval (conic-based geometry via the path crate).
    pub fn draw_oval(&mut self, rect: &Rect, paint: &Paint) {
        let center = Point::new(
            f32::midpoint(rect.left, rect.right),
            f32::midpoint(rect.top, rect.bottom),
        );
        let rx = rect.width() / 2.0;
        let ry = rect.height() / 2.0;

        if (rx - ry).abs() < 0.01 {
            // Close to circle, use circle drawing
            self.draw_circle(center, rx, paint);
        } else {
            // Conic-based oval geometry (Task 2's conic-aware add_oval).
            let mut b = PathBuilder::new();
            b.add_oval(rect);
            self.draw_path(&b.build(), paint);
        }
    }

    /// Draw a path.
    pub fn draw_path(&mut self, path: &Path, paint: &Paint) {
        match paint.style() {
            Style::Fill => self.fill_path(path, paint),
            Style::Stroke => self.stroke_path(path, paint),
            Style::StrokeAndFill => self.stroke_and_fill_path(path, paint),
        }
    }

    /// Stroke a path, honoring the paint's stroke width.
    ///
    /// Positive widths build the stroke outline in local space via
    /// `skia-rs-path`'s `stroke_to_fill` (Task 2) and fill it through the
    /// normal (matrix-aware, clipped, AA-aware) fill pipeline. Width 0
    /// remains a hairline.
    fn stroke_path(&mut self, path: &Path, paint: &Paint) {
        // Positive widths always fill the stroke outline, even sub-pixel
        // ones. Skia's SkDrawTreatAsHairline instead draws thin AA strokes
        // with its hairline blitters (modulating alpha by the width), which
        // blend each polyline pixel once; this rasterizer's hairlines draw
        // segment by segment. Filling the anti-aliased outline gives the
        // same coverage while keeping the caps and one blend per pixel.
        if paint.stroke_width() > 0.0 {
            if let Some(outline) = stroke_outline(path, paint) {
                self.fill_path(&outline, paint);
            }
            return;
        }
        self.stroke_path_hairline(path, paint);
    }

    /// `StrokeAndFill`: fill the union of the fill geometry and the stroke
    /// outline exactly once, so translucent paints don't double-blend the
    /// overlap (`SkStrokeRec::kStrokeAndFill_Style`).
    fn stroke_and_fill_path(&mut self, path: &Path, paint: &Paint) {
        if paint.stroke_width() <= 0.0 {
            // Hairline StrokeAndFill degenerates to a plain fill.
            self.fill_path(path, paint);
            return;
        }
        let combined = stroke_outline(path, paint)
            .and_then(|outline| skia_rs_path::op(path, &outline, skia_rs_path::PathOp::Union));
        if let Some(c) = combined {
            self.fill_path(&c, paint);
        } else {
            // Union unavailable (open/degenerate contours): fill then stroke.
            self.fill_path(path, paint);
            self.stroke_path(path, paint);
        }
    }

    /// Draw one hairline curve segment from `start`: `eval` gives the local
    /// point at `t`, `end` the exact end point. `points` is a reusable
    /// buffer.
    fn hairline_curve(
        &mut self,
        points: &mut Vec<Point>,
        hull: &[Point],
        eval: impl Fn(Scalar) -> Point,
        end: Point,
        paint: &Paint,
    ) {
        let start = hull[0];
        let device = self.device_rect();
        flatten_curve_culled(points, eval, end, &self.matrix, hull, Some(&device));
        let mut prev = start;
        for &p in points.iter() {
            self.draw_line(prev, p, paint);
            prev = p;
        }
    }

    /// Stroke a path as a hairline (width 0) by walking its flattened
    /// segments with the line rasterizer.
    #[allow(
        clippy::similar_names,
        reason = "t/mt (parameter and its complement) and per-segment prev_t/prev_mt are the standard Bezier-flattening parameter names"
    )]
    fn stroke_path_hairline(&mut self, path: &Path, paint: &Paint) {
        // Skia's hair_path extends segments for non-butt caps (extend_pts):
        // 0.5 device px for square caps, pi/8 for round. Extension happens
        // in device space, per segment: a segment that starts a contour
        // extends its start, one that ends a contour extends its end, and
        // closed contours only extend degenerate segments.
        let cap_outset = match paint.stroke_cap() {
            skia_rs_paint::StrokeCap::Butt => None,
            skia_rs_paint::StrokeCap::Square => Some(0.5),
            skia_rs_paint::StrokeCap::Round => Some(std::f32::consts::PI / 8.0),
        };
        let caps = cap_outset.and_then(|outset| {
            self.matrix.invert().map(|inverse| HairlineCaps {
                matrix: self.matrix,
                inverse,
                outset,
            })
        });
        // Lookahead (contour ends, closed contours) reads the verb array,
        // which lines up one-to-one with the element iterator.
        let verbs = path.verbs();

        let mut current = Point::zero();
        let mut contour_start = Point::zero();
        let mut closed = false;
        let mut prev_is_move = false;
        // Reused by every curve in the path (no per-segment allocation).
        let mut curve_points: Vec<Point> = Vec::new();

        for (index, element) in path.iter().enumerate() {
            let ends_contour =
                matches!(verbs.get(index + 1), None | Some(Verb::Move | Verb::Close));
            // Extend `pts` when this segment is capped (hair_path's
            // `!isClosed || degenerate` condition).
            let extend = |pts: &mut [Point]| {
                if let Some(caps) = &caps {
                    let degenerate = pts.iter().all(|q| *q == pts[0]);
                    if !closed || degenerate {
                        caps.extend(pts, prev_is_move, ends_contour);
                    }
                }
            };
            match element {
                PathElement::Move(p) => {
                    current = p;
                    contour_start = p;
                    closed = contour_is_closed(verbs.get(index + 1..).unwrap_or_default());
                }
                PathElement::Line(p) => {
                    let mut pts = [current, p];
                    extend(&mut pts);
                    self.draw_line(pts[0], pts[1], paint);
                    current = p;
                }
                // Every curve is flattened against a device-pixel error bound
                // (as in edge collection), so hairlines stay smooth under any
                // CTM rather than showing fixed-step facets when magnified.
                PathElement::Quad(ctrl, end) => {
                    let mut pts = [current, ctrl, end];
                    extend(&mut pts);
                    let [a, b, c] = pts;
                    self.hairline_curve(
                        &mut curve_points,
                        &pts,
                        |t| quad_point(t, 1.0 - t, a, b, c),
                        c,
                        paint,
                    );
                    current = end;
                }
                PathElement::Conic(ctrl, end, weight) => {
                    let mut pts = [current, ctrl, end];
                    extend(&mut pts);
                    let [a, b, c] = pts;
                    self.hairline_curve(
                        &mut curve_points,
                        &pts,
                        |t| conic_point(a, b, c, weight, t),
                        c,
                        paint,
                    );
                    current = end;
                }
                PathElement::Cubic(c1, c2, end) => {
                    let mut pts = [current, c1, c2, end];
                    extend(&mut pts);
                    let [a, b, c, d] = pts;
                    self.hairline_curve(
                        &mut curve_points,
                        &pts,
                        |t| cubic_point(t, 1.0 - t, a, b, c, d),
                        d,
                        paint,
                    );
                    current = end;
                }
                PathElement::Close => {
                    let mut pts = [current, contour_start];
                    // hair_path caps a moveTo/close pair (a degenerate
                    // contour) to match SVG's dot for zero-length subpaths.
                    if prev_is_move && let Some(caps) = &caps {
                        caps.extend(&mut pts, true, ends_contour);
                    }
                    if pts[0] != pts[1] {
                        self.draw_line(pts[0], pts[1], paint);
                    }
                    current = contour_start;
                    // Segments after a Close (without a Move) form a new
                    // contour from the same start: Skia injects a moveTo, so
                    // the cap state is that of a fresh contour.
                    closed = contour_is_closed(verbs.get(index + 1..).unwrap_or_default());
                }
            }
            prev_is_move = matches!(element, PathElement::Move(_) | PathElement::Close);
        }
    }

    /// Fill a path, routing anti-aliased paints through [`fill_path_aa`]
    /// (`paint.isAntiAlias()` selects the AA blitter in `SkDraw::drawPath`).
    fn fill_path(&mut self, path: &Path, paint: &Paint) {
        // SkPathPriv::PerspectiveClip: under perspective, geometry at or
        // behind the w = 0 plane maps to flipped or non-finite points, so the
        // path is first clipped to w > kW0PlaneDistance in local space.
        let clipped = perspective_clip(path, &self.matrix);
        let path = clipped.as_ref().unwrap_or(path);
        if paint.is_anti_alias() {
            self.fill_path_aa(path, paint);
        } else {
            self.fill_path_bw(path, paint);
        }
    }

    /// Fill a path without anti-aliasing.
    ///
    /// Scanline algorithm sampling pixel centers; supports all four fill
    /// types (the inverse types fill the complement of the path within the
    /// clip), shaders (sampled in local space), and the full clip stack.
    #[allow(
        clippy::similar_names,
        reason = "clip_x0/clip_y0/clip_x1/clip_y1 are the natural names for a 2D clip-rect bound tuple"
    )]
    fn fill_path_bw(&mut self, path: &Path, paint: &Paint) {
        use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, round_to_i32, scalar_from_i32};
        let fill_type = path.fill_type();
        let inverse = matches!(
            fill_type,
            FillType::InverseWinding | FillType::InverseEvenOdd
        );
        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();

        let edges = collect_edges(path, &self.matrix, Some(&self.device_rect()));
        if edges.is_empty() && !inverse {
            return;
        }

        let (clip_x0, clip_y0, clip_x1, clip_y1) = self.clip_pixel_bounds();

        // Row range: inverse fills cover the whole clip; regular fills only
        // the path's vertical extent (clamped to the clip).
        let (y_min, y_max) = if inverse {
            (clip_y0, clip_y1)
        } else {
            let ymin = edges.iter().map(|e| e.y_min).fold(f32::INFINITY, f32::min);
            let ymax = edges
                .iter()
                .map(|e| e.y_max)
                .fold(f32::NEG_INFINITY, f32::max);
            (
                floor_to_i32(ymin).max(clip_y0),
                ceil_to_i32(ymax).min(clip_y1),
            )
        };

        for y in y_min..y_max {
            let scanline = scalar_from_i32(y) + 0.5;
            let spans = spans_at_scanline(&edges, scanline, fill_type);
            let mut int_spans: Vec<(i32, i32)> = spans
                .iter()
                .map(|&(a, b)| (round_to_i32(a), round_to_i32(b)))
                .filter(|&(a, b)| a < b)
                .collect();

            if inverse {
                // Complement of the spans within the clip's x range.
                let mut comp = Vec::new();
                let mut cursor = clip_x0;
                for &(a, b) in &int_spans {
                    let a = a.clamp(clip_x0, clip_x1);
                    let b = b.clamp(clip_x0, clip_x1);
                    if a > cursor {
                        comp.push((cursor, a));
                    }
                    cursor = cursor.max(b);
                }
                if cursor < clip_x1 {
                    comp.push((cursor, clip_x1));
                }
                int_spans = comp;
            }

            for (x0, x1) in int_spans {
                self.blit_span(x0, x1, y, &source, blend_mode);
            }
        }
    }

    /// Fill a path with anti-aliasing (4x vertical supersampling with
    /// analytic horizontal coverage).
    ///
    /// Edges past their `y_max` contribute nothing (spans are recomputed per
    /// sample scanline), the clip is applied per pixel, shaders sample in
    /// local space, and inverse fill types fill the complement within the
    /// clip.
    #[allow(
        clippy::similar_names,
        reason = "clip_x0/clip_y0/clip_x1/clip_y1 are the natural names for a 2D clip-rect bound tuple"
    )]
    pub fn fill_path_aa(&mut self, path: &Path, paint: &Paint) {
        use skia_rs_core::cast::{ceil_to_i32, f32_to_u8_sat, floor_to_i32, scalar_from_i32};
        const SAMPLE_OFFSETS: [f32; 4] = [0.125, 0.375, 0.625, 0.875];

        let fill_type = path.fill_type();
        let inverse = matches!(
            fill_type,
            FillType::InverseWinding | FillType::InverseEvenOdd
        );
        let source = self.make_source(paint);
        let blend_mode = paint.blend_mode();

        let edges = collect_edges(path, &self.matrix, Some(&self.device_rect()));
        if edges.is_empty() && !inverse {
            return;
        }

        let (clip_x0, clip_y0, clip_x1, clip_y1) = self.clip_pixel_bounds();
        if clip_x0 >= clip_x1 || clip_y0 >= clip_y1 {
            return;
        }

        let (y_min, y_max) = if inverse {
            (clip_y0, clip_y1)
        } else {
            let ymin = edges.iter().map(|e| e.y_min).fold(f32::INFINITY, f32::min);
            let ymax = edges
                .iter()
                .map(|e| e.y_max)
                .fold(f32::NEG_INFINITY, f32::max);
            (
                floor_to_i32(ymin).max(clip_y0),
                ceil_to_i32(ymax).min(clip_y1),
            )
        };

        let row_width = usize::try_from(clip_x1 - clip_x0).unwrap_or(0);
        let mut coverage = vec![0.0f32; row_width];

        for y in y_min..y_max {
            coverage.fill(0.0);

            for &offset in &SAMPLE_OFFSETS {
                let scanline = scalar_from_i32(y) + offset;
                for (x0, x1) in spans_at_scanline(&edges, scanline, fill_type) {
                    let x0 = x0.max(scalar_from_i32(clip_x0));
                    let x1 = x1.min(scalar_from_i32(clip_x1));
                    if x0 >= x1 {
                        continue;
                    }
                    let px_start = floor_to_i32(x0).max(clip_x0);
                    let px_end = ceil_to_i32(x1).min(clip_x1);
                    for x in px_start..px_end {
                        let l = scalar_from_i32(x).max(x0);
                        let r = scalar_from_i32(x + 1).min(x1);
                        coverage[usize::try_from(x - clip_x0).unwrap_or(0)] +=
                            (r - l).max(0.0) * 0.25;
                    }
                }
            }

            for (i, &cov) in coverage.iter().enumerate() {
                let mut cov = cov.min(1.0);
                if inverse {
                    cov = 1.0 - cov;
                }
                if cov > 0.0 {
                    let x = clip_x0 + i32::try_from(i).unwrap_or(i32::MAX);
                    let cov8 = f32_to_u8_sat(cov * 255.0);
                    self.blit_pixel_cov(x, y, cov8, &source, blend_mode);
                }
            }
        }
    }
}

/// Build the stroke outline for `path` from the paint's stroke parameters
/// (width, cap, join, miter limit), delegating to `skia-rs-path`'s
/// `stroke_to_fill`.
fn stroke_outline(path: &Path, paint: &Paint) -> Option<Path> {
    use skia_rs_path::{StrokeCap, StrokeJoin, StrokeParams};

    let cap = match paint.stroke_cap() {
        skia_rs_paint::StrokeCap::Butt => StrokeCap::Butt,
        skia_rs_paint::StrokeCap::Round => StrokeCap::Round,
        skia_rs_paint::StrokeCap::Square => StrokeCap::Square,
    };
    let join = match paint.stroke_join() {
        skia_rs_paint::StrokeJoin::Miter => StrokeJoin::Miter,
        skia_rs_paint::StrokeJoin::Round => StrokeJoin::Round,
        skia_rs_paint::StrokeJoin::Bevel => StrokeJoin::Bevel,
    };
    let params = StrokeParams::new(paint.stroke_width())
        .with_cap(cap)
        .with_join(join)
        .with_miter_limit(paint.stroke_miter());
    skia_rs_path::stroke_to_fill(path, &params)
}

/// Scan-convert a device-space `path` into a [`Region`], clipped to
/// `clip_bounds`.
///
/// Mirrors `SkRegion::setPath`: each scanline is sampled at the pixel center,
/// span x-intersections round to nearest, and inverse fill types produce the
/// complement of the path within `clip_bounds`. Used by the clip stack so
/// non-AA path clips scan-convert the actual path rather than its bounds.
pub(crate) fn path_to_region(path: &Path, clip_bounds: &IRect) -> Region {
    use skia_rs_core::RegionOp;
    use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, round_to_i32, scalar_from_i32};

    let fill_type = path.fill_type();
    let inverse = matches!(
        fill_type,
        FillType::InverseWinding | FillType::InverseEvenOdd
    );

    let mut region = Region::new();
    let device = Rect::new(
        scalar_from_i32(clip_bounds.left),
        scalar_from_i32(clip_bounds.top),
        scalar_from_i32(clip_bounds.right),
        scalar_from_i32(clip_bounds.bottom),
    );
    let edges = collect_edges(path, &Matrix::IDENTITY, Some(&device));
    if !edges.is_empty() {
        let mut get = GlobalEdgeTable::new(edges);
        if let Some(y_start) = get.y_min() {
            let y_end = get.y_max();
            let y_min = floor_to_i32(y_start);
            let y_max = ceil_to_i32(y_end);

            let mut aet = ActiveEdgeTable::new();
            for y in y_min..y_max {
                let scanline = scalar_from_i32(y) + 0.5;
                aet.add_edges(get.get_new_edges_at(scanline), scanline);
                aet.remove_inactive(scanline);

                if !aet.is_empty() && y >= clip_bounds.top && y < clip_bounds.bottom {
                    aet.sort_by_x();
                    for (x0, x1) in aet.get_spans(fill_type) {
                        let xs = round_to_i32(x0).max(clip_bounds.left);
                        let xe = round_to_i32(x1).min(clip_bounds.right);
                        if xs < xe {
                            region.op_rect(IRect::new(xs, y, xe, y + 1), RegionOp::Union);
                        }
                    }
                }

                aet.step_all();
            }
        }
    }

    if inverse {
        let mut complement = Region::from_rect(*clip_bounds);
        complement.op_region(&region, RegionOp::Difference);
        return complement;
    }
    region
}

/// An edge for scanline rasterization with winding direction.
///
/// Edges are oriented from `y_min` to `y_max`, and the winding direction
/// is used for non-zero fill rule calculation.
#[derive(Debug, Clone)]
struct Edge {
    /// Minimum y coordinate (top of edge).
    y_min: f32,
    /// Maximum y coordinate (bottom of edge).
    y_max: f32,
    /// X coordinate at `y_min`.
    x_at_y_min: f32,
    /// Inverse slope (dx/dy) for efficient x calculation.
    inv_slope: f32,
    /// Winding direction: +1 for downward edges, -1 for upward edges.
    /// Used for non-zero fill rule.
    winding: i32,
}

impl Edge {
    /// Create a new edge from two points.
    ///
    /// Returns `None` for horizontal edges (no contribution to fill).
    fn new(p0: Point, p1: Point) -> Option<Self> {
        let dy = p1.y - p0.y;
        if dy.abs() < 0.001 {
            return None; // Horizontal edge
        }

        // Determine winding direction based on original edge direction
        let winding = if dy > 0.0 { 1 } else { -1 };

        // Orient edge so y_min < y_max
        let (top, bottom) = if p0.y < p1.y { (p0, p1) } else { (p1, p0) };

        let dy = bottom.y - top.y;
        let dx = bottom.x - top.x;

        Some(Self {
            y_min: top.y,
            y_max: bottom.y,
            x_at_y_min: top.x,
            inv_slope: dx / dy,
            winding,
        })
    }

    /// Calculate x intersection at a given scanline y.
    #[inline]
    fn x_at(&self, y: f32) -> f32 {
        (y - self.y_min) * self.inv_slope + self.x_at_y_min
    }

    /// Check if this edge is active at the given scanline.
    ///
    /// Note: This method is available for direct edge queries but is not used
    /// by the optimized AET algorithm which tracks edges through the GET.
    #[inline]
    #[allow(dead_code)]
    fn is_active_at(&self, y: f32) -> bool {
        y >= self.y_min && y < self.y_max
    }
}

/// Active edge entry for the Active Edge Table.
///
/// Contains the current x-intercept and a reference to the edge.
#[derive(Debug, Clone)]
struct ActiveEdge {
    /// Current x-intercept at the current scanline.
    x: f32,
    /// Inverse slope for incremental updates.
    inv_slope: f32,
    /// Winding direction.
    winding: i32,
    /// Maximum y coordinate (for removal).
    y_max: f32,
}

impl ActiveEdge {
    /// Create a new active edge from an Edge at a given scanline.
    fn from_edge(edge: &Edge, y: f32) -> Self {
        Self {
            x: edge.x_at(y),
            inv_slope: edge.inv_slope,
            winding: edge.winding,
            y_max: edge.y_max,
        }
    }

    /// Update x-intercept for the next scanline.
    #[inline]
    fn step(&mut self) {
        self.x += self.inv_slope;
    }

    /// Check if this edge is still active at the given y.
    #[inline]
    fn is_active_at(&self, y: f32) -> bool {
        y < self.y_max
    }
}

/// Global Edge Table - edges sorted by `y_min` for efficient scanline processing.
struct GlobalEdgeTable {
    /// Edges sorted by `y_min`.
    edges: Vec<Edge>,
    /// Current index into the edge list.
    current_index: usize,
}

impl GlobalEdgeTable {
    /// Create a new GET from a list of edges.
    fn new(mut edges: Vec<Edge>) -> Self {
        // Sort edges by y_min (primary), then by x_at_y_min (secondary)
        edges.sort_by(|a, b| {
            a.y_min
                .partial_cmp(&b.y_min)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.x_at_y_min
                        .partial_cmp(&b.x_at_y_min)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        Self {
            edges,
            current_index: 0,
        }
    }

    /// Get the minimum y coordinate where edges start.
    fn y_min(&self) -> Option<f32> {
        self.edges.first().map(|e| e.y_min)
    }

    /// Get the maximum y coordinate where edges end.
    fn y_max(&self) -> f32 {
        self.edges
            .iter()
            .map(|e| e.y_max)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Get all edges that become active at the given scanline.
    fn get_new_edges_at(&mut self, y: f32) -> impl Iterator<Item = &Edge> {
        let start = self.current_index;
        while self.current_index < self.edges.len() && self.edges[self.current_index].y_min <= y {
            self.current_index += 1;
        }
        self.edges[start..self.current_index].iter()
    }
}

/// Active Edge Table - maintains edges intersecting the current scanline.
struct ActiveEdgeTable {
    /// Active edges, sorted by x-intercept.
    edges: Vec<ActiveEdge>,
}

impl ActiveEdgeTable {
    /// Create a new empty AET.
    const fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Add new edges that become active at the given scanline.
    fn add_edges<'a>(&mut self, new_edges: impl Iterator<Item = &'a Edge>, y: f32) {
        for edge in new_edges {
            self.edges.push(ActiveEdge::from_edge(edge, y));
        }
    }

    /// Remove edges that are no longer active at the given scanline.
    fn remove_inactive(&mut self, y: f32) {
        self.edges.retain(|e| e.is_active_at(y));
    }

    /// Sort edges by x-intercept using insertion sort.
    ///
    /// Insertion sort is optimal here because the list is nearly sorted
    /// (edges only move slightly between scanlines).
    fn sort_by_x(&mut self) {
        // Insertion sort - optimal for nearly sorted data
        for i in 1..self.edges.len() {
            let mut j = i;
            while j > 0 && self.edges[j - 1].x > self.edges[j].x {
                self.edges.swap(j - 1, j);
                j -= 1;
            }
        }
    }

    /// Step all edges to the next scanline.
    fn step_all(&mut self) {
        for edge in &mut self.edges {
            edge.step();
        }
    }

    /// Get span pairs for filling using the specified fill rule.
    fn get_spans(&self, fill_type: FillType) -> Vec<(f32, f32)> {
        let mut spans = Vec::new();

        match fill_type {
            FillType::Winding | FillType::InverseWinding => {
                // Non-zero winding rule
                let mut winding = 0i32;
                let mut span_start: Option<f32> = None;

                for edge in &self.edges {
                    let was_inside = winding != 0;
                    winding += edge.winding;
                    let is_inside = winding != 0;

                    if !was_inside && is_inside {
                        span_start = Some(edge.x);
                    } else if was_inside
                        && !is_inside
                        && let Some(start) = span_start
                    {
                        spans.push((start, edge.x));
                        span_start = None;
                    }
                }
            }
            FillType::EvenOdd | FillType::InverseEvenOdd => {
                // Even-odd rule - fill between alternating pairs
                let mut inside = false;
                let mut span_start: Option<f32> = None;

                for edge in &self.edges {
                    inside = !inside;
                    if inside {
                        span_start = Some(edge.x);
                    } else if let Some(start) = span_start {
                        spans.push((start, edge.x));
                        span_start = None;
                    }
                }
            }
        }

        spans
    }

    /// Check if the AET is empty.
    const fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Quadratic Bezier point at parameter `t` (`mt` = 1 - t):
/// `p0*mt^2 + 2*p1*mt*t + p2*t^2`.
fn quad_point(t: Scalar, mt: Scalar, p0: Point, p1: Point, p2: Point) -> Point {
    Point::new(
        mt * mt * p0.x + 2.0 * mt * t * p1.x + t * t * p2.x,
        mt * mt * p0.y + 2.0 * mt * t * p1.y + t * t * p2.y,
    )
}

/// Cubic Bezier point at parameter `t` (`mt` = 1 - t):
/// `p0*mt^3 + 3*p1*mt^2*t + 3*p2*mt*t^2 + p3*t^3`.
fn cubic_point(t: Scalar, mt: Scalar, p0: Point, p1: Point, p2: Point, p3: Point) -> Point {
    let mt2 = mt * mt;
    let t2 = t * t;
    Point::new(
        mt2 * mt * p0.x + 3.0 * mt2 * t * p1.x + 3.0 * mt * t2 * p2.x + t2 * t * p3.x,
        mt2 * mt * p0.y + 3.0 * mt2 * t * p1.y + 3.0 * mt * t2 * p2.y + t2 * t * p3.y,
    )
}

/// The point at `t` on the conic `[start, ctrl, end]` with `weight`, in its
/// exact rational form.
fn conic_point(start: Point, ctrl: Point, end: Point, weight: Scalar, t: Scalar) -> Point {
    let mt = 1.0 - t;
    let (a, b, c) = (mt * mt, 2.0 * weight * t * mt, t * t);
    let denom = a + b + c;
    Point::new(
        (start.x * a + ctrl.x * b + end.x * c) / denom,
        (start.y * a + ctrl.y * b + end.y * c) / denom,
    )
}

/// Flatten one curve segment into device-space edges, starting from the
/// already-mapped `current` point, and return the new current point.
///
/// `eval` gives the curve's local-space point at `t` in `[0, 1]` and `end`
/// is its exact local end point (see [`flatten_curve_device_error`]).
/// `hull` is the curve's local control polygon; its last point is the end.
fn push_curve_edges(
    edges: &mut Vec<Edge>,
    scratch: &mut Vec<Point>,
    mut current: Point,
    eval: impl Fn(Scalar) -> Point,
    matrix: &Matrix,
    hull: &[Point],
    device: Option<&Rect>,
) -> Point {
    let end = hull[hull.len() - 1];
    flatten_curve_culled(scratch, eval, end, matrix, hull, device);
    for &p in scratch.iter() {
        let p = matrix.map_point(p);
        if let Some(edge) = Edge::new(current, p) {
            edges.push(edge);
        }
        current = p;
    }
    current
}

/// Flatten a curve into local-space polyline points (excluding its start),
/// replacing the contents of `out` (a reusable buffer, so hot paths do not
/// allocate per segment), subdividing until every piece deviates from the
/// true curve by at most [`CONIC_FLATTEN_TOLERANCE`] *device* pixels under
/// `matrix` (checked at the quarter, half and three-quarter points of each
/// piece, so inflected cubics are subdivided too).
///
/// `eval` gives the local-space point at `t` in `[0, 1]`; `end` is the exact
/// local end point. The curve is evaluated in local space and the error is
/// measured after mapping through the full matrix, which is correct for any
/// CTM, including perspective (where mapping control points alone would draw
/// the wrong curve, and would change a conic's weight). Subdivision depth is
/// capped at 2^12 segments per curve, comparable to Skia's edge builders
/// (up to 32 quads per conic, each up to 64 lines), which keeps the bound
/// for curves with device radii up to ~1e6 px; only curves that need the
/// depth pay for it.
fn flatten_curve_device_error(
    out: &mut Vec<Point>,
    eval: impl Fn(Scalar) -> Point,
    end: Point,
    matrix: &Matrix,
) {
    const MAX_DEPTH: u32 = 12;

    out.clear();
    // Iterative depth-first subdivision over [t0, t1] intervals, in curve
    // order. Each pop pushes at most two entries one level deeper, so the
    // stack never holds more than MAX_DEPTH + 1 entries: a fixed array.
    let mut stack = [(0.0_f32, 1.0_f32, 0_u32); MAX_DEPTH as usize + 1];
    let mut len = 1;
    while len > 0 {
        len -= 1;
        let (t0, t1, depth) = stack[len];
        let tm = f32::midpoint(t0, t1);
        let d0 = matrix.map_point(eval(t0));
        let d1 = matrix.map_point(eval(t1));
        // Compare the curve with the chord at t = 1/4, 1/2 and 3/4 of the
        // interval. The midpoint alone is blind to inflections: a symmetric
        // S-shaped cubic passes exactly through its chord's midpoint while
        // bulging to both sides of it.
        let mut max_dev_sq: Scalar = 0.0;
        for f in [0.25, 0.5, 0.75] {
            let p = matrix.map_point(eval((t1 - t0) * f + t0));
            let (cx, cy) = ((d1.x - d0.x) * f + d0.x, (d1.y - d0.y) * f + d0.y);
            let (dx, dy) = (p.x - cx, p.y - cy);
            max_dev_sq = max_dev_sq.max(dx * dx + dy * dy);
        }
        let within = max_dev_sq <= CONIC_FLATTEN_TOLERANCE * CONIC_FLATTEN_TOLERANCE;
        if depth >= MAX_DEPTH || within || !max_dev_sq.is_finite() {
            out.push(if t1 >= 1.0 { end } else { eval(t1) });
        } else {
            // Push the second half first so the first half is processed
            // next, keeping the output in curve order.
            stack[len] = (tm, t1, depth + 1);
            stack[len + 1] = (t0, tm, depth + 1);
            len += 2;
        }
    }
}

/// [`flatten_curve_device_error`] for a curve with local control points
/// `hull`, except that a curve whose mapped hull lies entirely beyond one
/// side of `device` becomes a single chord. The curve stays inside its hull,
/// so every crossing it would make is on the same off-device side as the
/// chord's: the device's coverage is unchanged, and huge off-screen curves
/// cost nothing (Skia clips curves to the device before building edges).
fn flatten_curve_culled(
    out: &mut Vec<Point>,
    eval: impl Fn(Scalar) -> Point,
    end: Point,
    matrix: &Matrix,
    hull: &[Point],
    device: Option<&Rect>,
) {
    if let Some(device) = device
        && !matrix.has_perspective()
    {
        let mapped = hull.iter().map(|p| matrix.map_point(*p));
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (
            Scalar::INFINITY,
            Scalar::INFINITY,
            Scalar::NEG_INFINITY,
            Scalar::NEG_INFINITY,
        );
        for q in mapped {
            min_x = min_x.min(q.x);
            min_y = min_y.min(q.y);
            max_x = max_x.max(q.x);
            max_y = max_y.max(q.y);
        }
        let off = max_x < device.left - 1.0
            || min_x > device.right + 1.0
            || max_y < device.top - 1.0
            || min_y > device.bottom + 1.0;
        if off {
            out.clear();
            out.push(end);
            return;
        }
    }
    flatten_curve_device_error(out, eval, end, matrix);
}

/// `path` with every curve flattened in local space so that, after mapping
/// through `matrix`, the polyline stays within the device-pixel flattening
/// error (the same bound fills use). Lets a path be mapped point-by-point
/// under perspective, where mapping control points would bend curves wrong.
pub(crate) fn flatten_for_matrix(path: &Path, matrix: &Matrix) -> Path {
    let mut out = PathBuilder::new();
    out.fill_type(path.fill_type());
    let mut scratch: Vec<Point> = Vec::new();
    let mut current = Point::zero();
    let mut start = Point::zero();
    let emit = |out: &mut PathBuilder, points: &[Point]| {
        for q in points {
            out.line_to(q.x, q.y);
        }
    };
    for element in path {
        match element {
            PathElement::Move(p) => {
                out.move_to(p.x, p.y);
                (current, start) = (p, p);
            }
            PathElement::Line(p) => {
                out.line_to(p.x, p.y);
                current = p;
            }
            PathElement::Quad(c1, p) => {
                let s0 = current;
                flatten_curve_device_error(
                    &mut scratch,
                    |t| quad_point(t, 1.0 - t, s0, c1, p),
                    p,
                    matrix,
                );
                emit(&mut out, &scratch);
                current = p;
            }
            PathElement::Conic(c1, p, weight) => {
                let s0 = current;
                flatten_curve_device_error(
                    &mut scratch,
                    |t| conic_point(s0, c1, p, weight, t),
                    p,
                    matrix,
                );
                emit(&mut out, &scratch);
                current = p;
            }
            PathElement::Cubic(c1, c2, p) => {
                let s0 = current;
                flatten_curve_device_error(
                    &mut scratch,
                    |t| cubic_point(t, 1.0 - t, s0, c1, c2, p),
                    p,
                    matrix,
                );
                emit(&mut out, &scratch);
                current = p;
            }
            PathElement::Close => {
                out.close();
                current = start;
            }
        }
    }
    out.build()
}

/// Skia's `SkPathPriv::kW0PlaneDistance`: perspective geometry is clipped
/// to `w > 1/16384`.
const W0_PLANE_DISTANCE: Scalar = 1.0 / 16384.0;

/// Segments a curve crossing the w plane is split into before clipping.
const PERSPECTIVE_CLIP_STEPS: u16 = 32;

/// Port of `SkPathPriv::PerspectiveClip` for fills: `None` when `matrix` has
/// no perspective or the whole path is in front of the `w = 1/16384` plane;
/// otherwise the path clipped to that half-plane in local space (empty when
/// it lies entirely behind). Every contour is treated as closed (a fill) and
/// clipped Sutherland-Hodgman style; curves wholly in front of the plane
/// are kept exactly, and curves that cross it are flattened first (Skia's
/// edge clipper chops them exactly).
pub(crate) fn perspective_clip(path: &Path, matrix: &Matrix) -> Option<Path> {
    if !matrix.has_perspective() {
        return None;
    }
    let [.., persp_x, persp_y, persp_w] = matrix.values;
    let plane_offset = persp_w - W0_PLANE_DISTANCE;
    let w = |p: Point| persp_x * p.x + persp_y * p.y + plane_offset;
    let bounds = path.bounds();
    let corners = [
        Point::new(bounds.left, bounds.top),
        Point::new(bounds.right, bounds.top),
        Point::new(bounds.left, bounds.bottom),
        Point::new(bounds.right, bounds.bottom),
    ];
    let in_front = corners.iter().filter(|p| w(**p) > 0.0).count();
    if in_front == corners.len() {
        return None;
    }
    let mut out = PathBuilder::new();
    out.fill_type(path.fill_type());
    if in_front == 0 {
        return Some(out.build());
    }

    let mut clipper = PlaneClipper {
        out: &mut out,
        w: &w,
        started: false,
        pending: None,
    };
    let (mut start, mut current) = (Point::zero(), Point::zero());
    // A path without a leading move starts at the origin (as collect_edges
    // treats it).
    clipper.begin(Point::zero());
    for element in path {
        match element {
            PathElement::Move(p) => {
                clipper.close_contour(current, start);
                (start, current) = (p, p);
                clipper.begin(p);
            }
            PathElement::Line(p) => {
                clipper.line(current, p);
                current = p;
            }
            PathElement::Quad(c1, p) => {
                let s0 = current;
                clipper.curve(
                    &[s0, c1, p],
                    |t| quad_point(t, 1.0 - t, s0, c1, p),
                    |o| {
                        o.quad_to(c1.x, c1.y, p.x, p.y);
                    },
                );
                current = p;
            }
            PathElement::Conic(c1, p, weight) => {
                let s0 = current;
                clipper.curve(
                    &[s0, c1, p],
                    |t| conic_point(s0, c1, p, weight, t),
                    |o| {
                        o.conic_to(c1.x, c1.y, p.x, p.y, weight);
                    },
                );
                current = p;
            }
            PathElement::Cubic(c1, c2, p) => {
                let s0 = current;
                clipper.curve(
                    &[s0, c1, c2, p],
                    |t| cubic_point(t, 1.0 - t, s0, c1, c2, p),
                    |o| {
                        o.cubic_to(c1.x, c1.y, c2.x, c2.y, p.x, p.y);
                    },
                );
                current = p;
            }
            PathElement::Close => {
                clipper.close_contour(current, start);
                current = start;
                clipper.begin(start);
            }
        }
    }
    clipper.close_contour(current, start);
    Some(out.build())
}

/// The part of the local segment `p0`-`p1` in front of the perspective
/// `w = 1/16384` plane (the segment itself without perspective); `None`
/// when it lies entirely behind.
fn clip_to_w_plane(matrix: &Matrix, p0: Point, p1: Point) -> Option<(Point, Point)> {
    if !matrix.has_perspective() {
        return Some((p0, p1));
    }
    let [.., persp_x, persp_y, persp_w] = matrix.values;
    let plane_offset = persp_w - W0_PLANE_DISTANCE;
    let w = |p: Point| persp_x * p.x + persp_y * p.y + plane_offset;
    let (w0, w1) = (w(p0), w(p1));
    let crossing = || {
        let t = w0 / (w0 - w1);
        Point::new(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t)
    };
    match (w0 > 0.0, w1 > 0.0) {
        (true, true) => Some((p0, p1)),
        (true, false) => Some((p0, crossing())),
        (false, true) => Some((crossing(), p1)),
        (false, false) => None,
    }
}

/// Sutherland-Hodgman clipping of closed contours against `w(p) > 0`.
struct PlaneClipper<'a, W: Fn(Point) -> Scalar> {
    out: &'a mut PathBuilder,
    w: &'a W,
    /// Whether the current output contour has its `move_to` yet.
    started: bool,
    /// The contour's start point, when in front of the plane, emitted with
    /// the first segment (so a bare move, e.g. after a `Close`, adds no
    /// empty contour).
    pending: Option<Point>,
}

impl<W: Fn(Point) -> Scalar> PlaneClipper<'_, W> {
    fn begin(&mut self, p: Point) {
        self.started = false;
        self.pending = ((self.w)(p) > 0.0).then_some(p);
    }

    /// Open the output contour at the pending start point, if any.
    fn flush(&mut self) {
        if !self.started
            && let Some(start) = self.pending.take()
        {
            self.out.move_to(start.x, start.y);
            self.started = true;
        }
    }

    fn emit(&mut self, p: Point) {
        self.flush();
        if self.started {
            self.out.line_to(p.x, p.y);
        } else {
            self.out.move_to(p.x, p.y);
            self.started = true;
        }
    }

    /// The point on `a`-`b` where `w` crosses zero.
    fn crossing(&self, a: Point, b: Point) -> Point {
        let (wa, wb) = ((self.w)(a), (self.w)(b));
        let t = wa / (wa - wb);
        Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
    }

    fn line(&mut self, a: Point, b: Point) {
        let (ia, ib) = ((self.w)(a) > 0.0, (self.w)(b) > 0.0);
        match (ia, ib) {
            (true, true) => self.emit(b),
            (true, false) => {
                let i = self.crossing(a, b);
                self.emit(i);
            }
            (false, true) => {
                let i = self.crossing(a, b);
                self.emit(i);
                self.emit(b);
            }
            (false, false) => {}
        }
    }

    /// A curve with control polygon `points` (first = current point): kept
    /// verbatim (via `keep`) when every control point is in front of the
    /// plane (the curve lies in their hull), else flattened and clipped.
    fn curve(
        &mut self,
        points: &[Point],
        eval: impl Fn(Scalar) -> Point,
        keep: impl FnOnce(&mut PathBuilder),
    ) {
        if points.iter().all(|p| (self.w)(*p) > 0.0) {
            self.flush();
            if self.started {
                keep(self.out);
                return;
            }
        }
        let mut prev = points[0];
        for i in 1..=PERSPECTIVE_CLIP_STEPS {
            let t = Scalar::from(i) / Scalar::from(PERSPECTIVE_CLIP_STEPS);
            let next = eval(t);
            self.line(prev, next);
            prev = next;
        }
    }

    fn close_contour(&mut self, current: Point, start: Point) {
        if current != start {
            self.line(current, start);
        }
        if self.started {
            self.out.close();
        }
        self.started = false;
        self.pending = None;
    }
}

/// Liang-Barsky clip of the segment `a`-`b` to `(left, top, right, bottom)`,
/// computed in f64 so endpoints far outside the device keep the visible
/// part precise. `None` when the segment is outside or not finite.
#[allow(
    clippy::cast_possible_truncation,
    reason = "clipped coordinates lie within the device bounds, so they fit f32"
)]
fn clip_segment(a: Point, b: Point, bounds: (f64, f64, f64, f64)) -> Option<(Point, Point)> {
    let (left, top, right, bottom) = bounds;
    let (ax, ay, bx, by) = (
        f64::from(a.x),
        f64::from(a.y),
        f64::from(b.x),
        f64::from(b.y),
    );
    if !(ax.is_finite() && ay.is_finite() && bx.is_finite() && by.is_finite()) {
        return None;
    }
    let (dx, dy) = (bx - ax, by - ay);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, q) in [
        (-dx, ax - left),
        (dx, right - ax),
        (-dy, ay - top),
        (dy, bottom - ay),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    if t0 > t1 {
        return None;
    }
    let at = |t: f64| Point::new((ax + t * dx) as Scalar, (ay + t * dy) as Scalar);
    // Unclipped ends are returned exactly.
    Some((
        if t0 > 0.0 { at(t0) } else { a },
        if t1 < 1.0 { at(t1) } else { b },
    ))
}

/// Cap extension for hairline segments (Skia's `extend_pts`), applied in
/// device space and mapped back to the local space the walker draws in.
struct HairlineCaps {
    matrix: Matrix,
    inverse: Matrix,
    outset: Scalar,
}

impl HairlineCaps {
    /// Extend a segment's start (`at_start`) and/or end (`at_end`) by the cap
    /// outset along its device-space tangent. Control points that coincide
    /// with the end point move with it; a fully degenerate segment moves its
    /// end point horizontally by the outset (+x at the start, -x at the end):
    /// Skia sets the tangent to (1, 0) / (-1, 0) and then adds
    /// `tangent * capOutset`, as for any other tangent.
    fn extend(&self, pts: &mut [Point], at_start: bool, at_end: bool) {
        if !(at_start || at_end) {
            return;
        }
        let mut dev: [Point; 4] = [Point::zero(); 4];
        let n = pts.len();
        for (d, p) in dev.iter_mut().zip(pts.iter()) {
            *d = self.matrix.map_point(*p);
        }
        let dev = &mut dev[..n];
        let outset = self.outset;
        let push = |p: &mut Point, (tx, ty): (Scalar, Scalar)| {
            p.x += tx * outset;
            p.y += ty * outset;
        };
        let unit = |from: Point, to: Point| {
            let (dx, dy) = (from.x - to.x, from.y - to.y);
            let len = dx.hypot(dy);
            (dx / len, dy / len)
        };
        if at_start {
            match (1..n).find(|&k| dev[k] != dev[0]) {
                Some(k) => {
                    let tangent = unit(dev[0], dev[k]);
                    for p in &mut dev[..k] {
                        push(p, tangent);
                    }
                }
                None => push(&mut dev[0], (1.0, 0.0)),
            }
        }
        if at_end {
            let last = n - 1;
            match (0..last).rev().find(|&k| dev[k] != dev[last]) {
                Some(k) => {
                    let tangent = unit(dev[last], dev[k]);
                    for p in &mut dev[k + 1..] {
                        push(p, tangent);
                    }
                }
                None => push(&mut dev[last], (-1.0, 0.0)),
            }
        }
        for (p, d) in pts.iter_mut().zip(dev.iter()) {
            *p = self.inverse.map_point(*d);
        }
    }
}

/// Whether the contour whose verbs (after its `Move`) are `rest` ends with a
/// `Close` (Skia's `is_next_contour_closed`).
fn contour_is_closed(rest: &[Verb]) -> bool {
    rest.iter()
        .find(|v| matches!(v, Verb::Move | Verb::Close))
        .is_some_and(|v| *v == Verb::Close)
}

/// Maximum distance, in device pixels, between a conic and the polyline it is
/// flattened to (Skia's path flattening tolerance).
pub(crate) const CONIC_FLATTEN_TOLERANCE: Scalar = 0.25;

/// Collect edges from a path.
fn collect_edges(path: &Path, matrix: &Matrix, device: Option<&Rect>) -> Vec<Edge> {
    let mut edges = Vec::new();
    // A path without a leading move starts at the local origin, so the
    // device-space start is the *mapped* origin (matching the local one).
    let mut current = matrix.map_point(Point::zero());
    let mut contour_start = current;
    // Local-space counterparts, needed to flatten conics before a
    // perspective projection.
    let mut local_current = Point::zero();
    let mut local_contour_start = Point::zero();
    // Reused by every curve in the path (no per-segment allocation).
    let mut curve_scratch: Vec<Point> = Vec::new();

    for element in path {
        let local_start = local_current;
        local_current = match element {
            PathElement::Move(p) => {
                local_contour_start = p;
                p
            }
            PathElement::Line(p)
            | PathElement::Quad(_, p)
            | PathElement::Conic(_, p, _)
            | PathElement::Cubic(_, _, p) => p,
            PathElement::Close => local_contour_start,
        };
        match element {
            PathElement::Move(p) => {
                // Filling treats every open contour as closed (as Skia does):
                // add the implicit closing edge before starting a new one.
                if let Some(edge) = Edge::new(current, contour_start) {
                    edges.push(edge);
                }
                current = matrix.map_point(p);
                contour_start = current;
            }
            PathElement::Line(p) => {
                let end = matrix.map_point(p);
                if let Some(edge) = Edge::new(current, end) {
                    edges.push(edge);
                }
                current = end;
            }
            PathElement::Conic(ctrl, end, weight) => {
                current = push_curve_edges(
                    &mut edges,
                    &mut curve_scratch,
                    current,
                    |t| conic_point(local_start, ctrl, end, weight, t),
                    matrix,
                    &[local_start, ctrl, end],
                    device,
                );
            }
            // Every curve is evaluated in local space and flattened against
            // a device-pixel error bound: correct under any CTM (mapping
            // control points is wrong under perspective) and as fine as
            // the curve's on-screen size needs.
            PathElement::Quad(ctrl, end) => {
                current = push_curve_edges(
                    &mut edges,
                    &mut curve_scratch,
                    current,
                    |t| quad_point(t, 1.0 - t, local_start, ctrl, end),
                    matrix,
                    &[local_start, ctrl, end],
                    device,
                );
            }
            PathElement::Cubic(c1, c2, end) => {
                current = push_curve_edges(
                    &mut edges,
                    &mut curve_scratch,
                    current,
                    |t| cubic_point(t, 1.0 - t, local_start, c1, c2, end),
                    matrix,
                    &[local_start, c1, c2, end],
                    device,
                );
            }
            PathElement::Close => {
                if let Some(edge) = Edge::new(current, contour_start) {
                    edges.push(edge);
                }
                current = contour_start;
            }
        }
    }
    // Implicitly close the last contour (a Close already made this a
    // zero-length edge, which Edge::new rejects).
    if let Some(edge) = Edge::new(current, contour_start) {
        edges.push(edge);
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huge_curves_stay_within_the_device_flattening_tolerance() {
        // A quarter circle of device radius 1e5: every chord's sagitta must
        // stay within CONIC_FLATTEN_TOLERANCE (256 segments gave ~0.47 px).
        let radius = 1.0e5_f32;
        let matrix = Matrix::scale(radius, radius);
        let (start, ctrl, end) = (
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        );
        let weight = std::f32::consts::FRAC_1_SQRT_2;
        let mut out = Vec::new();
        flatten_curve_device_error(
            &mut out,
            |t| conic_point(start, ctrl, end, weight, t),
            end,
            &matrix,
        );
        let mut prev = matrix.map_point(start);
        let mut worst = 0.0_f64;
        for q in &out {
            let next = matrix.map_point(*q);
            let (mx, my) = (
                f64::midpoint(f64::from(prev.x), f64::from(next.x)),
                f64::midpoint(f64::from(prev.y), f64::from(next.y)),
            );
            worst = worst.max(f64::from(radius) - mx.hypot(my));
            prev = next;
        }
        assert!(
            worst <= 0.3,
            "max sagitta {worst} px over {} segments",
            out.len()
        );
    }

    #[test]
    fn perspective_clip_adds_no_empty_contours() {
        // Three closed rects straddling the w plane: every output contour
        // must have segments (no bare move+close left after each Close).
        let mut b = PathBuilder::new();
        for i in 0..3_u8 {
            let x = f32::from(i) * 60.0;
            b.add_rect(&Rect::new(x, 0.0, x + 50.0, 150.0));
        }
        let matrix = Matrix {
            values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.01, 1.0],
        };
        let clipped = perspective_clip(&b.build(), &matrix).expect("straddles the plane");
        let verbs = clipped.verbs();
        for pair in verbs.windows(2) {
            assert!(
                !(pair[0] == Verb::Move && pair[1] == Verb::Close),
                "empty contour in {verbs:?}"
            );
        }
        let contours = verbs.iter().filter(|v| **v == Verb::Move).count();
        assert_eq!(contours, 3);
    }

    #[test]
    fn test_pixel_buffer_new() {
        let buffer = PixelBuffer::new(100, 100);
        assert_eq!(buffer.width, 100);
        assert_eq!(buffer.height, 100);
        assert_eq!(buffer.pixels.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_pixel_buffer_clear() {
        let mut buffer = PixelBuffer::new(10, 10);
        buffer.clear(Color::from_argb(255, 255, 0, 0));

        let pixel = buffer.get_pixel(5, 5).unwrap();
        assert_eq!(pixel.red(), 255);
        assert_eq!(pixel.green(), 0);
        assert_eq!(pixel.blue(), 0);
        assert_eq!(pixel.alpha(), 255);
    }

    #[test]
    fn test_pixel_buffer_set_get() {
        let mut buffer = PixelBuffer::new(10, 10);
        buffer.set_pixel(5, 5, Color::from_argb(255, 0, 255, 0));

        let pixel = buffer.get_pixel(5, 5).unwrap();
        assert_eq!(pixel.green(), 255);
    }

    #[test]
    fn test_rasterizer_draw_rect() {
        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 255, 255, 255));

        let mut rasterizer = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        paint.set_style(Style::Fill);

        rasterizer.fill_rect(&Rect::from_xywh(10.0, 10.0, 50.0, 50.0), &paint);

        // Check a pixel inside the rect
        let pixel = buffer.get_pixel(25, 25).unwrap();
        assert_eq!(pixel.red(), 255);
        assert_eq!(pixel.green(), 0);
    }

    #[test]
    fn test_blend_src_over() {
        // blend_colors operates on PREMULTIPLIED inputs and returns premul.
        // Premul half-red over premul opaque blue.
        let src = premultiply_color(Color::from_argb(128, 255, 0, 0));
        let dst = Color::from_argb(255, 0, 0, 255);
        let result = blend_colors(src, dst, BlendMode::SrcOver);

        // Semi-transparent red over blue: result stays opaque, red present,
        // blue attenuated by (1 - 0.5).
        assert_eq!(result.alpha(), 255);
        assert!(result.red() > 100);
        assert!(result.blue() > 100);
    }

    #[test]
    fn test_premul_storage_translucent_fill() {
        // A translucent SrcOver fill over a transparent buffer must store
        // PREMULTIPLIED bytes (SkSurface_Raster with AlphaType::Premul).
        // Old straight-alpha storage kept red==255; premul stores red==alpha.
        let mut buffer = PixelBuffer::new(4, 4);
        // transparent background (all zero, premul transparent)
        let mut rasterizer = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(128, 255, 0, 0));
        paint.set_style(Style::Fill);
        rasterizer.fill_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &paint);

        let offset = buffer.stride + 4;
        let r = buffer.pixels[offset];
        let a = buffer.pixels[offset + 3];
        assert_eq!(a, 128, "alpha stored");
        // Premultiplied red == mul_div_255_round(255,128) == 128, NOT 255.
        assert_eq!(r, 128, "red must be premultiplied");
    }

    #[test]
    fn test_blend_colors_delegates_all_modes() {
        // blend_colors must delegate non-Porter-Duff modes to paint's
        // BlendMode::apply on premultiplied values (no silent SrcOver fallback).
        let src = premultiply_color(Color::from_argb(200, 100, 150, 50));
        let dst = premultiply_color(Color::from_argb(180, 60, 90, 200));
        for mode in [
            BlendMode::Multiply,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Luminosity,
        ] {
            let s = skia_rs_core::Color4f::from_color(src);
            let d = skia_rs_core::Color4f::from_color(dst);
            let expected = mode.apply(s, d).to_color();
            let got = blend_colors(src, dst, mode);
            assert_eq!(got, expected, "mode {mode:?} must match paint apply");
        }
    }

    #[test]
    fn test_porter_duff_src_atop_premul() {
        // SrcATop on premul: out = src*da + dst*(1-sa).
        let src = premultiply_color(Color::from_argb(128, 255, 0, 0));
        let dst = premultiply_color(Color::from_argb(255, 0, 0, 255));
        let got = blend_colors(src, dst, BlendMode::SrcATop);
        // alpha == dst alpha for SrcATop
        assert_eq!(got.alpha(), 255);
    }

    // ============ Active Edge Table Tests ============

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test: edge endpoints are exact literals copied straight through Edge::new"
    )]
    fn test_edge_creation() {
        // Horizontal edge should return None
        let p0 = Point::new(0.0, 10.0);
        let p1 = Point::new(100.0, 10.0);
        assert!(Edge::new(p0, p1).is_none());

        // Downward edge (positive winding)
        let p0 = Point::new(0.0, 0.0);
        let p1 = Point::new(10.0, 100.0);
        let edge = Edge::new(p0, p1).unwrap();
        assert_eq!(edge.winding, 1);
        assert_eq!(edge.y_min, 0.0);
        assert_eq!(edge.y_max, 100.0);

        // Upward edge (negative winding)
        let p0 = Point::new(10.0, 100.0);
        let p1 = Point::new(0.0, 0.0);
        let edge = Edge::new(p0, p1).unwrap();
        assert_eq!(edge.winding, -1);
        assert_eq!(edge.y_min, 0.0);
        assert_eq!(edge.y_max, 100.0);
    }

    #[test]
    fn test_edge_x_at() {
        let p0 = Point::new(0.0, 0.0);
        let p1 = Point::new(100.0, 100.0);
        let edge = Edge::new(p0, p1).unwrap();

        // 45-degree line: x should equal y
        assert!((edge.x_at(0.0) - 0.0).abs() < 0.001);
        assert!((edge.x_at(50.0) - 50.0).abs() < 0.001);
        assert!((edge.x_at(100.0) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_active_edge_step() {
        let p0 = Point::new(0.0, 0.0);
        let p1 = Point::new(100.0, 100.0);
        let edge = Edge::new(p0, p1).unwrap();

        let mut active = ActiveEdge::from_edge(&edge, 0.5);
        let initial_x = active.x;
        active.step();
        // After stepping, x should increase by inv_slope
        assert!((active.x - initial_x - 1.0).abs() < 0.001);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test: y_min values are exact literals copied straight through Edge::new"
    )]
    fn test_global_edge_table_ordering() {
        let edges = vec![
            Edge::new(Point::new(0.0, 50.0), Point::new(10.0, 100.0)).unwrap(),
            Edge::new(Point::new(0.0, 0.0), Point::new(10.0, 50.0)).unwrap(),
            Edge::new(Point::new(0.0, 25.0), Point::new(10.0, 75.0)).unwrap(),
        ];

        let get = GlobalEdgeTable::new(edges);

        // Edges should be sorted by y_min
        assert_eq!(get.edges[0].y_min, 0.0);
        assert_eq!(get.edges[1].y_min, 25.0);
        assert_eq!(get.edges[2].y_min, 50.0);
    }

    #[test]
    fn test_active_edge_table_spans_even_odd() {
        let mut aet = ActiveEdgeTable::new();

        // Simulate a square: 4 vertical edges
        let left_edge = Edge::new(Point::new(10.0, 0.0), Point::new(10.0, 100.0)).unwrap();
        let right_edge = Edge::new(Point::new(50.0, 0.0), Point::new(50.0, 100.0)).unwrap();

        aet.add_edges([&left_edge, &right_edge].into_iter(), 50.0);
        aet.sort_by_x();

        let spans = aet.get_spans(FillType::EvenOdd);
        assert_eq!(spans.len(), 1);
        assert!((spans[0].0 - 10.0).abs() < 0.001);
        assert!((spans[0].1 - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_active_edge_table_spans_nonzero() {
        let mut aet = ActiveEdgeTable::new();

        // Create a proper polygon contour - a square traversed clockwise
        // Left edge goes down (winding +1), right edge goes up (winding -1)
        let left_edge = Edge::new(Point::new(10.0, 0.0), Point::new(10.0, 100.0)).unwrap();
        // Reverse the right edge so it goes up (from bottom to top)
        let right_edge = Edge::new(Point::new(50.0, 100.0), Point::new(50.0, 0.0)).unwrap();

        aet.add_edges([&left_edge, &right_edge].into_iter(), 50.0);
        aet.sort_by_x();

        let spans = aet.get_spans(FillType::Winding);

        // Left edge winding +1, right edge winding -1
        // At scanline: winding goes 0 -> +1 -> 0
        // Should produce one span from left to right
        assert_eq!(spans.len(), 1);
        assert!((spans[0].0 - 10.0).abs() < 0.001);
        assert!((spans[0].1 - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_fill_triangle_path() {
        use skia_rs_path::PathBuilder;

        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 255, 255, 255));

        let mut rasterizer = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        paint.set_style(Style::Fill);

        // Create a triangle path
        let mut builder = PathBuilder::new();
        builder
            .move_to(50.0, 10.0)
            .line_to(90.0, 90.0)
            .line_to(10.0, 90.0)
            .close();
        let path = builder.build();

        rasterizer.draw_path(&path, &paint);

        // Check a pixel inside the triangle (centroid-ish)
        let pixel = buffer.get_pixel(50, 60).unwrap();
        assert_eq!(pixel.red(), 255, "Triangle should be filled at center");

        // Check a pixel outside the triangle
        let pixel = buffer.get_pixel(10, 10).unwrap();
        assert_eq!(pixel.red(), 255, "Outside should remain white (background)");
        assert_eq!(pixel.green(), 255);
    }

    #[test]
    fn test_fill_complex_polygon() {
        use skia_rs_path::PathBuilder;

        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 0, 0, 0));

        let mut rasterizer = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 0, 255, 0));
        paint.set_style(Style::Fill);

        // Create a star-like shape (self-intersecting)
        let mut builder = PathBuilder::new();
        builder
            .move_to(50.0, 10.0)
            .line_to(61.0, 40.0)
            .line_to(90.0, 40.0)
            .line_to(68.0, 58.0)
            .line_to(79.0, 90.0)
            .line_to(50.0, 70.0)
            .line_to(21.0, 90.0)
            .line_to(32.0, 58.0)
            .line_to(10.0, 40.0)
            .line_to(39.0, 40.0)
            .close();
        let path = builder.build();

        rasterizer.draw_path(&path, &paint);

        // The path should have some filled pixels
        // Check center region
        let pixel = buffer.get_pixel(50, 50).unwrap();
        // With even-odd rule, center of star might not be filled
        // With non-zero (default), it should be filled
        assert_eq!(pixel.green(), 255, "Star center should be filled");
    }

    // ============ Conformance-audit regression tests (Task 4) ============

    #[test]
    fn test_fill_rect_rotated_ctm_scan_converts_quad() {
        // Under a rotated CTM, fill_rect must fill the mapped QUAD, not the
        // axis-aligned bounding box of the quad.
        let mut buffer = PixelBuffer::new(100, 100);
        let mut r = Rasterizer::new(&mut buffer);
        // Rotate 45 degrees about (50, 50).
        let rot = Matrix::translate(50.0, 50.0)
            .concat(&Matrix::rotate(std::f32::consts::FRAC_PI_4))
            .concat(&Matrix::translate(-50.0, -50.0));
        r.set_matrix(&rot);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        r.fill_rect(&Rect::from_xywh(20.0, 20.0, 60.0, 60.0), &paint);

        // Center is inside the rotated quad.
        assert_eq!(buffer.get_pixel(50, 50).unwrap().red(), 255);
        // The AABB corner (25, 25) is OUTSIDE the rotated quad (the quad is a
        // diamond centered at 50,50 with vertices ~(50, 7.5)...).
        assert_eq!(
            buffer.get_pixel(25, 25).unwrap().alpha(),
            0,
            "AABB corner outside the rotated quad must not be filled"
        );
        // A diamond vertex region is inside.
        assert_eq!(buffer.get_pixel(50, 15).unwrap().red(), 255);
    }

    #[test]
    fn test_stroke_rect_honors_stroke_width() {
        // Stroke width 10 centered on the rect edge: a band [15, 25] around
        // the x=20 edge must be filled; the rect center must not be.
        let mut buffer = PixelBuffer::new(100, 100);
        let mut r = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 0, 0, 255));
        paint.set_style(Style::Stroke);
        paint.set_stroke_width(10.0);
        r.draw_rect(&Rect::from_xywh(20.0, 20.0, 60.0, 60.0), &paint);

        // 3px inside the stroke band on the left edge.
        assert_eq!(buffer.get_pixel(17, 50).unwrap().blue(), 255);
        assert_eq!(buffer.get_pixel(23, 50).unwrap().blue(), 255);
        // Center: untouched.
        assert_eq!(buffer.get_pixel(50, 50).unwrap().alpha(), 0);
        // Far outside: untouched.
        assert_eq!(buffer.get_pixel(5, 50).unwrap().alpha(), 0);
    }

    #[test]
    fn test_fill_path_honors_shader_in_local_space_with_clip() {
        use skia_rs_paint::shader::{TileMode, shaders};
        use skia_rs_path::PathBuilder;

        // Horizontal gradient black -> white over local x in [0, 10].
        // CTM translates by (50, 0): device x=50 is local x=0 (black end),
        // device x=59 is local x=9 (near-white). Sampling in DEVICE space
        // would read far off the gradient end instead.
        let shader = shaders::linear_gradient(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            vec![
                skia_rs_core::Color4f::new(0.0, 0.0, 0.0, 1.0),
                skia_rs_core::Color4f::new(1.0, 1.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );

        let mut buffer = PixelBuffer::new(100, 100);
        let mut r = Rasterizer::new(&mut buffer);
        r.set_matrix(&Matrix::translate(50.0, 0.0));
        // Clip: only y in [0, 5).
        r.clip_rect_aa(&Rect::from_xywh(0.0, 0.0, 100.0, 5.0), false);

        let mut paint = Paint::new();
        paint.set_shader(Some(shader));
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0));
        let path = b.build();
        r.draw_path(&path, &paint);

        // Local x=0.5 at device x=50 -> near-black.
        let dark = buffer.get_pixel(50, 2).unwrap();
        assert!(
            dark.red() < 40,
            "local-space start should be dark: {dark:?}"
        );
        // Local x=9.5 at device x=59 -> near-white.
        let bright = buffer.get_pixel(59, 2).unwrap();
        assert!(
            bright.red() > 215,
            "local-space end should be bright: {bright:?}"
        );
        // The clip must be respected by the shader fill.
        assert_eq!(
            buffer.get_pixel(55, 8).unwrap().alpha(),
            0,
            "shader fill must respect the clip"
        );
    }

    #[test]
    fn test_fill_path_aa_routed_and_respects_clip_and_y_max() {
        use skia_rs_path::PathBuilder;

        // An AA paint must produce partial-coverage edge pixels, must not
        // paint below the path's bottom (edges past y_max), and must respect
        // the clip.
        let mut buffer = PixelBuffer::new(60, 60);
        let mut r = Rasterizer::new(&mut buffer);
        r.clip_rect_aa(&Rect::from_xywh(0.0, 0.0, 60.0, 40.0), false);

        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        paint.set_anti_alias(true);
        // Triangle with a diagonal edge.
        let mut b = PathBuilder::new();
        b.move_to(10.0, 10.0)
            .line_to(50.0, 10.0)
            .line_to(10.0, 50.0)
            .close();
        let path = b.build();
        r.draw_path(&path, &paint);

        // Interior: fully covered.
        assert_eq!(buffer.get_pixel(15, 15).unwrap().red(), 255);
        // Diagonal edge pixel: partial coverage (alpha strictly between).
        let edge = buffer.get_pixel(30, 29).unwrap();
        assert!(
            edge.alpha() > 0 && edge.alpha() < 255,
            "diagonal edge should be anti-aliased, got alpha {}",
            edge.alpha()
        );
        // Below the clip (y >= 40): nothing, even though the path reaches y=50.
        for y in 41..50 {
            assert_eq!(
                buffer.get_pixel(11, y).unwrap().alpha(),
                0,
                "AA fill must respect the clip at y={y}"
            );
        }
        // Beyond the path bottom (y >= 50): nothing (no stale edges).
        assert_eq!(buffer.get_pixel(11, 52).unwrap().alpha(), 0);
    }

    #[test]
    fn test_inverse_fill_types_fill_complement_within_clip() {
        use skia_rs_path::PathBuilder;

        let mut buffer = PixelBuffer::new(40, 40);
        let mut r = Rasterizer::new(&mut buffer);
        r.clip_rect_aa(&Rect::from_xywh(0.0, 0.0, 30.0, 30.0), false);

        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 0, 255, 0));
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::from_xywh(10.0, 10.0, 10.0, 10.0));
        let mut path = b.build();
        path.set_fill_type(FillType::InverseWinding);
        r.draw_path(&path, &paint);

        // Inside the rect: NOT filled (complement).
        assert_eq!(buffer.get_pixel(15, 15).unwrap().alpha(), 0);
        // Outside the rect but inside the clip: filled.
        assert_eq!(buffer.get_pixel(5, 5).unwrap().green(), 255);
        assert_eq!(buffer.get_pixel(25, 25).unwrap().green(), 255);
        // Outside the clip: not filled.
        assert_eq!(buffer.get_pixel(35, 35).unwrap().alpha(), 0);
    }

    #[test]
    fn test_fill_circle_translucent_no_double_blend() {
        // Every pixel covered by a translucent circle fill must be blended
        // exactly once: uniform value across the interior.
        let mut buffer = PixelBuffer::new(60, 60);
        let mut r = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(128, 255, 0, 0));
        r.fill_circle(Point::new(30.0, 30.0), 20.0, &paint);

        // Premul 50% red over transparent = (128, 0, 0, 128) everywhere inside.
        let center = buffer.get_pixel(30, 30).unwrap();
        assert_eq!(center.alpha(), 128);
        assert_eq!(center.red(), 128);
        // Sample many interior pixels: all identical (no double-blended rows).
        for y in 15..45 {
            for x in 15..45 {
                let dx = skia_rs_core::cast::scalar_from_i32(x) - 30.0;
                let dy = skia_rs_core::cast::scalar_from_i32(y) - 30.0;
                if dx.hypot(dy) < 18.0 {
                    let p = buffer.get_pixel(x, y).unwrap();
                    assert_eq!(
                        (p.red(), p.alpha()),
                        (128, 128),
                        "double-blended pixel at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_circle_under_nonuniform_scale_becomes_ellipse() {
        // scale(1, 2): a circle of radius 10 at (30, 15) maps to an ellipse
        // with vertical semi-axis 20 centered at (30, 30).
        let mut buffer = PixelBuffer::new(60, 60);
        let mut r = Rasterizer::new(&mut buffer);
        r.set_matrix(&Matrix::scale(1.0, 2.0));
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        r.draw_circle(Point::new(30.0, 15.0), 10.0, &paint);

        // Vertical extremes of the ellipse (y = 30 +/- ~19) are covered.
        assert_eq!(buffer.get_pixel(30, 12).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(30, 48).unwrap().red(), 255);
        // A circle of radius 10 (or 20) would NOT cover (39, 30)+(30,12)
        // simultaneously; check horizontal extent stays ~10.
        assert_eq!(buffer.get_pixel(38, 30).unwrap().red(), 255);
        assert_eq!(
            buffer.get_pixel(45, 30).unwrap().alpha(),
            0,
            "horizontal semi-axis must stay ~10"
        );
    }

    #[test]
    fn test_stroke_and_fill_translucent_no_double_blend() {
        // StrokeAndFill must not double-blend the region where fill and
        // stroke overlap: one uniform value across fill+stroke geometry.
        let mut buffer = PixelBuffer::new(60, 60);
        let mut r = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(128, 0, 0, 255));
        paint.set_style(Style::StrokeAndFill);
        paint.set_stroke_width(6.0);
        r.draw_rect(&Rect::from_xywh(20.0, 20.0, 20.0, 20.0), &paint);

        // Points: rect interior, on the rect edge (overlap zone), and inside
        // the outset band. All must be single-blended: (0,0,128,128) premul.
        for &(x, y) in &[(30, 30), (20, 30), (18, 30), (30, 20), (30, 42)] {
            let p = buffer.get_pixel(x, y).unwrap();
            assert_eq!(
                (p.blue(), p.alpha()),
                (128, 128),
                "double blend at ({x}, {y}): {p:?}"
            );
        }
        // Outside the outset band: untouched.
        assert_eq!(buffer.get_pixel(10, 30).unwrap().alpha(), 0);
    }

    #[test]
    fn test_winding_number_calculation() {
        // Test that the winding rule correctly handles overlapping regions
        use skia_rs_path::PathBuilder;

        let mut buffer = PixelBuffer::new(100, 100);

        // Create two overlapping squares - with non-zero winding, overlap is filled
        let mut path_builder = PathBuilder::new();

        // First square (clockwise)
        path_builder
            .move_to(20.0, 20.0)
            .line_to(60.0, 20.0)
            .line_to(60.0, 60.0)
            .line_to(20.0, 60.0)
            .close();

        // Second square (also clockwise, overlapping)
        path_builder
            .move_to(40.0, 40.0)
            .line_to(80.0, 40.0)
            .line_to(80.0, 80.0)
            .line_to(40.0, 80.0)
            .close();

        let mut path = path_builder.build();
        path.set_fill_type(FillType::Winding);

        buffer.clear(Color::from_argb(255, 255, 255, 255));
        let mut rasterizer = Rasterizer::new(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));

        rasterizer.fill_path(&path, &paint);

        // With non-zero winding, the overlap region should be filled
        let overlap_pixel = buffer.get_pixel(50, 50).unwrap();
        assert_eq!(overlap_pixel.red(), 255, "Overlap should be filled");
    }
}
