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

use skia_rs_core::premultiply_color;
use skia_rs_core::{Color, Color4f, IRect, Matrix, Point, Rect, Region, Scalar};
use skia_rs_paint::{BlendMode, Paint, Style};
use skia_rs_path::PathBuilder;

use crate::simd::mul_div_255_round;
use skia_rs_path::{FillType, Path, PathElement};

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
        for chunk in self.pixels.chunks_exact_mut(4) {
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

        // `src` is premultiplied: coverage attenuates all four channels.
        let cov = skia_rs_core::cast::f32_to_u8_sat(coverage.min(1.0) * 255.0);
        let src_with_coverage = apply_coverage(src, cov);

        let dst = self.get_pixel(x, y).unwrap_or(Color::from_argb(0, 0, 0, 0));
        let blended = blend_colors(src_with_coverage, dst, blend_mode);
        self.set_pixel(x, y, blended);
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
        let (dx, dy) = (x + self.origin.0, y + self.origin.1);
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
        let c = apply_coverage(source_color_at(source, x, y), combined);
        self.buffer.blend_pixel(x, y, c, blend_mode);
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
        use skia_rs_core::cast::round_to_i32;
        let transformed = self.matrix.map_point(point);
        let x = round_to_i32(transformed.x);
        let y = round_to_i32(transformed.y);

        let coverage = self.get_clip_coverage(x, y);
        if coverage > 0 {
            let color = premultiply_color(paint.color32());
            if coverage == 255 {
                self.buffer.blend_pixel(x, y, color, paint.blend_mode());
            } else {
                // Apply clip coverage to the premultiplied color (all channels).
                let adjusted_color = apply_coverage(color, coverage);
                self.buffer
                    .blend_pixel(x, y, adjusted_color, paint.blend_mode());
            }
        }
    }

    /// Draw a line using Bresenham's algorithm (aliased) or Wu's algorithm (anti-aliased).
    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        if paint.is_anti_alias() {
            self.draw_line_aa(p0, p1, paint);
        } else {
            self.draw_line_aliased(p0, p1, paint);
        }
    }

    /// Draw line without anti-aliasing (Bresenham).
    fn draw_line_aliased(&mut self, p0: Point, p1: Point, paint: &Paint) {
        use skia_rs_core::cast::round_to_i32;
        let t0 = self.matrix.map_point(p0);
        let t1 = self.matrix.map_point(p1);

        let mut x0 = round_to_i32(t0.x);
        let mut y0 = round_to_i32(t0.y);
        let x1 = round_to_i32(t1.x);
        let y1 = round_to_i32(t1.y);

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let color = premultiply_color(paint.color32());
        let blend_mode = paint.blend_mode();

        loop {
            let coverage = self.get_clip_coverage(x0, y0);
            if coverage > 0 {
                if coverage == 255 {
                    self.buffer.blend_pixel(x0, y0, color, blend_mode);
                } else {
                    let adjusted = apply_coverage(color, coverage);
                    self.buffer.blend_pixel(x0, y0, adjusted, blend_mode);
                }
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Draw line with anti-aliasing using Wu's algorithm.
    fn draw_line_aa(&mut self, p0: Point, p1: Point, paint: &Paint) {
        use skia_rs_core::cast::floor_to_i32;
        let t0 = self.matrix.map_point(p0);
        let t1 = self.matrix.map_point(p1);

        let mut x0 = t0.x;
        let mut y0 = t0.y;
        let mut x1 = t1.x;
        let mut y1 = t1.y;

        let color = premultiply_color(paint.color32());
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
        let gradient = if dx.abs() < 0.0001 { 1.0 } else { dy / dx };

        // Handle first endpoint
        let xend = x0.round();
        let yend = y0 + gradient * (xend - x0);
        let xgap = 1.0 - (x0 + 0.5).fract();
        let xpxl1 = floor_to_i32(xend);
        let ypxl1 = floor_to_i32(yend);

        if steep {
            self.plot_aa(ypxl1, xpxl1, (1.0 - yend.fract()) * xgap, color, blend_mode);
            self.plot_aa(ypxl1 + 1, xpxl1, yend.fract() * xgap, color, blend_mode);
        } else {
            self.plot_aa(xpxl1, ypxl1, (1.0 - yend.fract()) * xgap, color, blend_mode);
            self.plot_aa(xpxl1, ypxl1 + 1, yend.fract() * xgap, color, blend_mode);
        }

        let mut intery = yend + gradient;

        // Handle second endpoint
        let xend = x1.round();
        let yend = y1 + gradient * (xend - x1);
        let xgap = (x1 + 0.5).fract();
        let xpxl2 = floor_to_i32(xend);
        let ypxl2 = floor_to_i32(yend);

        if steep {
            self.plot_aa(ypxl2, xpxl2, (1.0 - yend.fract()) * xgap, color, blend_mode);
            self.plot_aa(ypxl2 + 1, xpxl2, yend.fract() * xgap, color, blend_mode);
        } else {
            self.plot_aa(xpxl2, ypxl2, (1.0 - yend.fract()) * xgap, color, blend_mode);
            self.plot_aa(xpxl2, ypxl2 + 1, yend.fract() * xgap, color, blend_mode);
        }

        // Main loop
        if steep {
            for x in (xpxl1 + 1)..xpxl2 {
                let y = floor_to_i32(intery);
                self.plot_aa(y, x, 1.0 - intery.fract(), color, blend_mode);
                self.plot_aa(y + 1, x, intery.fract(), color, blend_mode);
                intery += gradient;
            }
        } else {
            for x in (xpxl1 + 1)..xpxl2 {
                let y = floor_to_i32(intery);
                self.plot_aa(x, y, 1.0 - intery.fract(), color, blend_mode);
                self.plot_aa(x, y + 1, intery.fract(), color, blend_mode);
                intery += gradient;
            }
        }
    }

    /// Plot a pixel with coverage for anti-aliasing.
    #[inline]
    fn plot_aa(&mut self, x: i32, y: i32, coverage: f32, color: Color, blend_mode: BlendMode) {
        let clip_coverage = self.get_clip_coverage(x, y);
        if clip_coverage > 0 {
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
            for x in start..=end {
                let coverage = self.get_clip_coverage(x, y);
                if coverage > 0 {
                    if coverage == 255 {
                        self.buffer.blend_pixel(x, y, color, blend_mode);
                    } else {
                        let adjusted = apply_coverage(color, coverage);
                        self.buffer.blend_pixel(x, y, adjusted, blend_mode);
                    }
                }
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
        if !self.matrix_is_axis_aligned() || paint.is_anti_alias() {
            let mut b = PathBuilder::new();
            b.add_rect(rect);
            self.fill_path(&b.build(), paint);
            return;
        }

        let transformed = self.matrix.map_rect(rect);

        let x0 = round_to_i32(transformed.left);
        let y0 = round_to_i32(transformed.top);
        let x1 = round_to_i32(transformed.right);
        let y1 = round_to_i32(transformed.bottom);

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
        let x0 = round_to_i32(rect.left) - self.origin.0;
        let y0 = round_to_i32(rect.top) - self.origin.1;
        let x1 = round_to_i32(rect.right) - self.origin.0;
        let y1 = round_to_i32(rect.bottom) - self.origin.1;

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

        let tl = Point::new(rect.left, rect.top);
        let tr = Point::new(rect.right, rect.top);
        let bl = Point::new(rect.left, rect.bottom);
        let br = Point::new(rect.right, rect.bottom);

        self.draw_line(tl, tr, paint);
        self.draw_line(tr, br, paint);
        self.draw_line(br, bl, paint);
        self.draw_line(bl, tl, paint);
    }

    /// Draw a rectangle (filled or stroked based on paint style).
    pub fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
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

        let y0 = floor_to_i32(tc.y - r);
        let y1 = ceil_to_i32(tc.y + r);
        for y in y0..y1 {
            let dy = scalar_from_i32(y) + 0.5 - tc.y;
            if dy.abs() > r {
                continue;
            }
            let half = r.mul_add(r, -(dy * dy)).sqrt();
            let x0 = round_to_i32(tc.x - half);
            let x1 = round_to_i32(tc.x + half);
            self.blit_span(x0, x1, y, &source, blend_mode);
        }
    }

    /// Draw a stroked circle.
    pub fn stroke_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        use skia_rs_core::cast::round_to_i32;
        let tc = self.matrix.map_point(center);
        let cx = round_to_i32(tc.x);
        let cy = round_to_i32(tc.y);
        let r = round_to_i32(radius * self.matrix.scale_x().abs());

        let color = premultiply_color(paint.color32());
        let blend_mode = paint.blend_mode();

        let mut x = 0;
        let mut y = r;
        let mut d = 1 - r;

        while x <= y {
            // Plot pixels in all 8 octants
            self.buffer.blend_pixel(cx + x, cy + y, color, blend_mode);
            self.buffer.blend_pixel(cx - x, cy + y, color, blend_mode);
            self.buffer.blend_pixel(cx + x, cy - y, color, blend_mode);
            self.buffer.blend_pixel(cx - x, cy - y, color, blend_mode);
            self.buffer.blend_pixel(cx + y, cy + x, color, blend_mode);
            self.buffer.blend_pixel(cx - y, cy + x, color, blend_mode);
            self.buffer.blend_pixel(cx + y, cy - x, color, blend_mode);
            self.buffer.blend_pixel(cx - y, cy - x, color, blend_mode);

            x += 1;
            if d < 0 {
                d += 2 * x + 1;
            } else {
                y -= 1;
                d += 2 * (x - y) + 1;
            }
        }
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

        let color = premultiply_color(paint.color32());
        let blend_mode = paint.blend_mode();

        // Calculate bounding box
        let min_x = floor_to_i32(cx - r - 1.0);
        let max_x = ceil_to_i32(cx + r + 1.0);
        let min_y = floor_to_i32(cy - r - 1.0);
        let max_y = ceil_to_i32(cy + r + 1.0);

        match paint.style() {
            Style::Fill => {
                // For each pixel in bounding box
                for py in min_y..=max_y {
                    for px in min_x..=max_x {
                        // Calculate distance from pixel center to circle center
                        let dx = scalar_from_i32(px) + 0.5 - cx;
                        let dy = scalar_from_i32(py) + 0.5 - cy;
                        let dist_sq = dx.mul_add(dx, dy * dy);

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
                            self.plot_aa(px, py, coverage, color, blend_mode);
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
                        let dist = dx.mul_add(dx, dy * dy).sqrt();

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
                            self.plot_aa(px, py, coverage, color, blend_mode);
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

    /// Stroke a path as a hairline (width 0) by walking its flattened
    /// segments with the line rasterizer.
    #[allow(
        clippy::similar_names,
        reason = "t/mt (parameter and its complement) and per-segment prev_t/prev_mt are the standard Bezier-flattening parameter names"
    )]
    fn stroke_path_hairline(&mut self, path: &Path, paint: &Paint) {
        use skia_rs_core::cast::scalar_from_i32;

        let mut current = Point::zero();
        let mut contour_start = Point::zero();

        for element in path {
            match element {
                PathElement::Move(p) => {
                    current = p;
                    contour_start = p;
                }
                PathElement::Line(p) => {
                    self.draw_line(current, p, paint);
                    current = p;
                }
                PathElement::Quad(ctrl, end) => {
                    // Approximate with lines
                    let steps = 16;
                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let mt = 1.0 - t;
                        let p = quad_point(t, mt, current, ctrl, end);
                        self.draw_line(
                            if i == 1 {
                                current
                            } else {
                                let pt = scalar_from_i32(i - 1) / scalar_from_i32(steps);
                                let pmt = 1.0 - pt;
                                quad_point(pt, pmt, current, ctrl, end)
                            },
                            p,
                            paint,
                        );
                    }
                    current = end;
                }
                PathElement::Conic(ctrl, end, _w) => {
                    // Approximate as quad for simplicity
                    let steps = 16;
                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let mt = 1.0 - t;
                        let p = quad_point(t, mt, current, ctrl, end);
                        let prev_t = scalar_from_i32(i - 1) / scalar_from_i32(steps);
                        let prev_mt = 1.0 - prev_t;
                        let prev = quad_point(prev_t, prev_mt, current, ctrl, end);
                        self.draw_line(prev, p, paint);
                    }
                    current = end;
                }
                PathElement::Cubic(c1, c2, end) => {
                    // Approximate with lines
                    let steps = 24;
                    let mut prev = current;
                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let mt = 1.0 - t;
                        let p = cubic_point(t, mt, current, c1, c2, end);
                        self.draw_line(prev, p, paint);
                        prev = p;
                    }
                    current = end;
                }
                PathElement::Close => {
                    if current != contour_start {
                        self.draw_line(current, contour_start, paint);
                    }
                    current = contour_start;
                }
            }
        }
    }

    /// Fill a path, routing anti-aliased paints through [`fill_path_aa`]
    /// (`paint.isAntiAlias()` selects the AA blitter in `SkDraw::drawPath`).
    fn fill_path(&mut self, path: &Path, paint: &Paint) {
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

        let edges = collect_edges(path, &self.matrix);
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

        let edges = collect_edges(path, &self.matrix);
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
                        coverage[usize::try_from(x - clip_x0).unwrap_or(0)] = (r - l)
                            .max(0.0)
                            .mul_add(0.25, coverage[usize::try_from(x - clip_x0).unwrap_or(0)]);
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
    let edges = collect_edges(path, &Matrix::IDENTITY);
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
        (y - self.y_min).mul_add(self.inv_slope, self.x_at_y_min)
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
                    } else if was_inside && !is_inside {
                        if let Some(start) = span_start {
                            spans.push((start, edge.x));
                            span_start = None;
                        }
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
    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Quadratic Bezier point at parameter `t` (`mt` = 1 - t), as separable
/// `mul_add` chains: `p0*mt^2 + 2*p1*mt*t + p2*t^2`.
fn quad_point(t: Scalar, mt: Scalar, p0: Point, p1: Point, p2: Point) -> Point {
    Point::new(
        (t * t).mul_add(p2.x, (2.0 * mt * t).mul_add(p1.x, mt * mt * p0.x)),
        (t * t).mul_add(p2.y, (2.0 * mt * t).mul_add(p1.y, mt * mt * p0.y)),
    )
}

/// Cubic Bezier point at parameter `t` (`mt` = 1 - t), as separable `mul_add`
/// chains: `p0*mt^3 + 3*p1*mt^2*t + 3*p2*mt*t^2 + p3*t^3`.
fn cubic_point(t: Scalar, mt: Scalar, p0: Point, p1: Point, p2: Point, p3: Point) -> Point {
    let mt2 = mt * mt;
    let t2 = t * t;
    Point::new(
        (t2 * t).mul_add(
            p3.x,
            (3.0 * mt * t2).mul_add(p2.x, (mt2 * mt).mul_add(p0.x, 3.0 * mt2 * t * p1.x)),
        ),
        (t2 * t).mul_add(
            p3.y,
            (3.0 * mt * t2).mul_add(p2.y, (mt2 * mt).mul_add(p0.y, 3.0 * mt2 * t * p1.y)),
        ),
    )
}

/// Collect edges from a path.
#[allow(
    clippy::similar_names,
    reason = "t/mt (Bezier parameter and its complement) is the standard flattening naming"
)]
fn collect_edges(path: &Path, matrix: &Matrix) -> Vec<Edge> {
    use skia_rs_core::cast::scalar_from_i32;

    let mut edges = Vec::new();
    let mut current = Point::zero();
    let mut contour_start = Point::zero();

    for element in path {
        match element {
            PathElement::Move(p) => {
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
            PathElement::Quad(ctrl, end) => {
                let ctrl = matrix.map_point(ctrl);
                let end = matrix.map_point(end);
                // Flatten to lines
                let steps = 8;
                let start = current;
                for i in 1..=steps {
                    let t = scalar_from_i32(i) / scalar_from_i32(steps);
                    let mt = 1.0 - t;
                    let p = quad_point(t, mt, start, ctrl, end);
                    if let Some(edge) = Edge::new(current, p) {
                        edges.push(edge);
                    }
                    current = p;
                }
            }
            PathElement::Conic(ctrl, end, _w) => {
                let ctrl = matrix.map_point(ctrl);
                let end = matrix.map_point(end);
                let steps = 8;
                let start = current;
                for i in 1..=steps {
                    let t = scalar_from_i32(i) / scalar_from_i32(steps);
                    let mt = 1.0 - t;
                    let p = quad_point(t, mt, start, ctrl, end);
                    if let Some(edge) = Edge::new(current, p) {
                        edges.push(edge);
                    }
                    current = p;
                }
            }
            PathElement::Cubic(c1, c2, end) => {
                let c1 = matrix.map_point(c1);
                let c2 = matrix.map_point(c2);
                let end = matrix.map_point(end);
                let steps = 12;
                let start = current;
                for i in 1..=steps {
                    let t = scalar_from_i32(i) / scalar_from_i32(steps);
                    let mt = 1.0 - t;
                    let p = cubic_point(t, mt, start, c1, c2, end);
                    if let Some(edge) = Edge::new(current, p) {
                        edges.push(edge);
                    }
                    current = p;
                }
            }
            PathElement::Close => {
                if let Some(edge) = Edge::new(current, contour_start) {
                    edges.push(edge);
                }
                current = contour_start;
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

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
