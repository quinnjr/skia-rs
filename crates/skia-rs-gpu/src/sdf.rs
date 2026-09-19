//! Signed Distance Field (SDF) rendering for GPU text and shapes.
//!
//! This module provides utilities for generating and rendering SDFs,
//! which enable resolution-independent rendering of text and vector shapes.

use crate::cast_util::{scalar_from_u32, scalar_from_usize};
use skia_rs_core::cast::f32_to_u8_sat;
use skia_rs_core::{Point, Rect};

/// SDF generation configuration.
#[derive(Debug, Clone)]
pub struct SdfConfig {
    /// Output texture size.
    pub size: u32,
    /// Padding around the shape.
    pub padding: u32,
    /// Spread (distance field radius in pixels).
    pub spread: f32,
    /// Scale factor for generating the SDF.
    pub scale: f32,
}

impl Default for SdfConfig {
    fn default() -> Self {
        Self {
            size: 64,
            padding: 4,
            spread: 8.0,
            scale: 1.0,
        }
    }
}

impl SdfConfig {
    /// Create a configuration for high-resolution SDF.
    #[must_use]
    pub const fn high_res() -> Self {
        Self {
            size: 128,
            padding: 8,
            spread: 16.0,
            scale: 1.0,
        }
    }

    /// Create a configuration for compact SDF.
    #[must_use]
    pub const fn compact() -> Self {
        Self {
            size: 32,
            padding: 2,
            spread: 4.0,
            scale: 1.0,
        }
    }
}

/// SDF render parameters for shader use.
#[derive(Debug, Clone, Copy)]
pub struct SdfRenderParams {
    /// Smoothing factor (typically 0.25 / spread).
    pub smoothing: f32,
    /// Outline width (0 = no outline).
    pub outline_width: f32,
    /// Soft edge factor.
    pub soft_edge: f32,
    /// Distance threshold for rendering.
    pub threshold: f32,
}

impl Default for SdfRenderParams {
    fn default() -> Self {
        Self {
            smoothing: 0.25 / 8.0,
            outline_width: 0.0,
            soft_edge: 0.0,
            threshold: 0.5,
        }
    }
}

impl SdfRenderParams {
    /// Create parameters for crisp rendering.
    #[must_use]
    pub fn crisp(spread: f32) -> Self {
        Self {
            smoothing: 0.1 / spread,
            outline_width: 0.0,
            soft_edge: 0.0,
            threshold: 0.5,
        }
    }

    /// Create parameters for soft rendering.
    #[must_use]
    pub fn soft(spread: f32) -> Self {
        Self {
            smoothing: 0.5 / spread,
            outline_width: 0.0,
            soft_edge: 0.2,
            threshold: 0.5,
        }
    }

    /// Create parameters with outline.
    #[must_use]
    pub fn with_outline(spread: f32, outline_width: f32) -> Self {
        Self {
            smoothing: 0.25 / spread,
            outline_width,
            soft_edge: 0.0,
            threshold: 0.5,
        }
    }
}

/// Generate a signed distance field from a binary mask.
///
/// Uses Felzenszwalb & Huttenlocher's O(n) distance transform algorithm
/// (Felzenszwalb, 2012) rather than the previous O(n²·spread²) brute-force
/// dead-reckoning search. The algorithm computes the exact Euclidean
/// squared distance transform in linear time via a lower-envelope
/// computation on each row, then each column.
///
/// For a 256×256 mask with spread 16 the old implementation touched ~17M
/// pixels per direction (quadratic in spread). The new implementation
/// touches 64k pixels per pass regardless of spread.
#[must_use]
pub fn generate_sdf_from_mask(mask: &[u8], width: u32, height: u32, spread: f32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let inf = f32::MAX / 4.0;

    // Distance from each pixel to the nearest "off" pixel (outside region).
    let mut d_out = vec![inf; w * h];
    // Distance to the nearest "on" pixel.
    let mut d_in = vec![inf; w * h];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let is_inside = mask[idx] > 127;
            if is_inside {
                d_out[idx] = inf;
                d_in[idx] = 0.0;
            } else {
                d_out[idx] = 0.0;
                d_in[idx] = inf;
            }
        }
    }

    distance_transform_2d(&mut d_out, w, h);
    distance_transform_2d(&mut d_in, w, h);

    let mut sdf = vec![0.0f32; w * h];
    for i in 0..(w * h) {
        // d_out[i]: squared distance from i to nearest "outside" pixel (a
        //          pixel whose mask byte was 0). Non-zero iff i is inside.
        // d_in[i]:  squared distance from i to nearest "inside" pixel.
        //          Non-zero iff i is outside.
        //
        // Signed distance:
        //   inside pixel  → negative, magnitude = distance to nearest outside
        //   outside pixel → positive, magnitude = distance to nearest inside
        let out_d = d_out[i].max(0.0).sqrt();
        let in_d = d_in[i].max(0.0).sqrt();
        // Distances above are measured to the nearest opposite-region pixel
        // *center*. The mask edge lies between texels — roughly half a texel
        // closer than that center — so subtract 0.5 to measure to the edge
        // itself, matching SkDistanceFieldGen (which models the boundary
        // between texels, `distance = 0.5 - alpha`). This places the two
        // texels straddling an edge at -0.5 (inside) and +0.5 (outside),
        // with the true edge at 0 exactly between them.
        let signed = if mask[i] > 127 {
            -(out_d - 0.5)
        } else {
            in_d - 0.5
        };
        // Clamp to +/- spread so the mapping to texture texels is stable.
        sdf[i] = signed.clamp(-spread, spread);
    }
    sdf
}

/// Felzenszwalb–Huttenlocher 2D squared-distance transform.
///
/// Runs the 1D transform first on each row, then on each column of the
/// intermediate. Input `f` should start with 0 for source pixels and
/// infinity elsewhere; output is the squared Euclidean distance from each
/// pixel to the nearest source pixel.
fn distance_transform_2d(f: &mut [f32], w: usize, h: usize) {
    // Row pass.
    let mut row = vec![0.0f32; w];
    for y in 0..h {
        for x in 0..w {
            row[x] = f[y * w + x];
        }
        distance_transform_1d(&mut row);
        for x in 0..w {
            f[y * w + x] = row[x];
        }
    }

    // Column pass.
    let mut col = vec![0.0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = f[y * w + x];
        }
        distance_transform_1d(&mut col);
        for y in 0..h {
            f[y * w + x] = col[y];
        }
    }
}

/// 1D lower-envelope distance transform (Felzenszwalb 2012).
///
/// Canonical two-pass algorithm: sweep left-to-right building the lower
/// envelope of parabolas `(i - q)² + f[q]`, then evaluate the envelope at
/// each integer point. Runs in O(n).
fn distance_transform_1d(f: &mut [f32]) {
    let len = f.len();
    if len == 0 {
        return;
    }
    let big = f32::MAX / 4.0;

    // Read-only copy of the input.
    let input: Vec<f32> = f.to_vec();

    // Find first non-infinite index; if the entire input is infinite, the
    // result is also infinite (no source pixel anywhere).
    let Some(first_finite) = input.iter().position(|&value| value < big) else {
        for value in f.iter_mut() {
            *value = big;
        }
        return;
    };

    // Indices of parabolas in the lower envelope, and the boundaries
    // between them. `env_top` is the index of the rightmost parabola in the
    // envelope.
    let mut envelope_vertex = vec![0usize; len];
    let mut envelope_bound = vec![0.0f32; len + 1];
    let mut env_top: isize = 0;
    envelope_vertex[0] = first_finite;
    envelope_bound[0] = f32::NEG_INFINITY;
    envelope_bound[1] = f32::INFINITY;

    for parab in (first_finite + 1)..len {
        if input[parab] >= big {
            continue;
        }
        // Intersection of parabola `parab` with parabola envelope_vertex[env_top].
        loop {
            let top_idx = index_from_isize(env_top);
            let vertex = envelope_vertex[top_idx];
            let f_vertex = input[vertex];
            let f_parab = input[parab];
            let denom = 2.0 * (scalar_from_usize(parab) - scalar_from_usize(vertex));
            let s = ((f_parab + scalar_from_usize(parab * parab))
                - (f_vertex + scalar_from_usize(vertex * vertex)))
                / denom;
            if s <= envelope_bound[top_idx] {
                if env_top == 0 {
                    // Replace the single parabola in the envelope.
                    envelope_vertex[0] = parab;
                    envelope_bound[0] = f32::NEG_INFINITY;
                    envelope_bound[1] = f32::INFINITY;
                    break;
                }
                env_top -= 1;
            } else {
                env_top += 1;
                let new_top = index_from_isize(env_top);
                envelope_vertex[new_top] = parab;
                envelope_bound[new_top] = s;
                envelope_bound[new_top + 1] = f32::INFINITY;
                break;
            }
        }
    }

    // Evaluate the lower envelope at each integer location.
    let mut cursor: isize = 0;
    for (out_idx, dst) in f.iter_mut().enumerate() {
        while cursor < env_top
            && envelope_bound[index_from_isize(cursor) + 1] < scalar_from_usize(out_idx)
        {
            cursor += 1;
        }
        let vertex = envelope_vertex[index_from_isize(cursor)];
        let f_vertex = input[vertex];
        if f_vertex >= big {
            *dst = big;
        } else {
            let d = scalar_from_usize(out_idx) - scalar_from_usize(vertex);
            *dst = d.mul_add(d, f_vertex);
        }
    }
}

/// Convert a non-negative envelope index to `usize`.
///
/// # Panics
/// Panics if `k` is negative, which cannot happen: `env_top`/`cursor` only
/// ever decrease below zero via the `env_top == 0` early-return guard above.
#[inline]
fn index_from_isize(k: isize) -> usize {
    usize::try_from(k).expect("envelope index must be non-negative")
}

/// Convert SDF to normalized texture data (0-255).
///
/// Matches `SkDistanceFieldGen::pack_distance_field_val`: **inside is high**
/// (> 128) and outside is low, with the zero level (the edge) at 128. Skia
/// negates the signed distance before packing, so an inside pixel (negative
/// signed distance in this crate's convention) maps above 128. The positive
/// side is scaled by 127/128 to avoid overflowing 255, and the byte is
/// rounded to nearest.
#[must_use]
pub fn sdf_to_texture(sdf: &[f32], spread: f32) -> Vec<u8> {
    let mag = spread.max(1e-6);
    sdf.iter()
        .map(|&d| {
            // Negate: inside (d < 0) -> positive -> high byte. Clamp the
            // positive range to mag*127/128 (Skia) so it never rounds to 256.
            let shifted = (-d).clamp(-mag, mag * 127.0 / 128.0) + mag;
            f32_to_u8_sat(shifted / (2.0 * mag) * 256.0)
        })
        .collect()
}

/// Sample SDF at a point with bilinear filtering.
#[must_use]
pub fn sample_sdf_bilinear(sdf: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 {
    let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
    let height_i32 = i32::try_from(height).unwrap_or(i32::MAX);
    let x0 =
        u32::try_from(skia_rs_core::cast::floor_to_i32(x).clamp(0, width_i32 - 1)).unwrap_or(0);
    let y0 =
        u32::try_from(skia_rs_core::cast::floor_to_i32(y).clamp(0, height_i32 - 1)).unwrap_or(0);
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x.floor();
    let fy = y - y.floor();

    let d00 = sdf[(y0 * width + x0) as usize];
    let d10 = sdf[(y0 * width + x1) as usize];
    let d01 = sdf[(y1 * width + x0) as usize];
    let d11 = sdf[(y1 * width + x1) as usize];

    let d0 = d00.mul_add(1.0 - fx, d10 * fx);
    let d1 = d01.mul_add(1.0 - fx, d11 * fx);

    d0 * (1.0 - fy) + d1 * fy
}

/// Generate SDF for a circle.
#[must_use]
pub fn generate_circle_sdf(size: u32, radius: f32) -> Vec<f32> {
    let mut sdf = Vec::with_capacity((size * size) as usize);
    let center = scalar_from_u32(size) * 0.5;

    for y in 0..size {
        for x in 0..size {
            let dx = scalar_from_u32(x) + 0.5 - center;
            let dy = scalar_from_u32(y) + 0.5 - center;
            let dist = dx.hypot(dy) - radius;
            sdf.push(dist);
        }
    }

    sdf
}

/// Generate SDF for a rounded rectangle.
#[must_use]
pub fn generate_rounded_rect_sdf(size: u32, rect: Rect, radius: f32) -> Vec<f32> {
    let mut sdf = Vec::with_capacity((size * size) as usize);

    for y in 0..size {
        for x in 0..size {
            let px = scalar_from_u32(x) + 0.5;
            let py = scalar_from_u32(y) + 0.5;

            // Distance to rounded rectangle
            let dist = sdf_rounded_rect(px, py, &rect, radius);
            sdf.push(dist);
        }
    }

    sdf
}

/// Calculate SDF for a rounded rectangle at a point.
fn sdf_rounded_rect(x: f32, y: f32, rect: &Rect, radius: f32) -> f32 {
    let center = rect.center();
    let cx = center.x;
    let cy = center.y;
    let hw = rect.width().mul_add(0.5, -radius);
    let hh = rect.height().mul_add(0.5, -radius);

    let dx = (x - cx).abs() - hw;
    let dy = (y - cy).abs() - hh;

    let outside_dist = dx.max(0.0).hypot(dy.max(0.0));
    let inside_dist = dx.max(dy).min(0.0);

    outside_dist + inside_dist - radius
}

/// Multi-channel SDF data (for improved quality).
#[derive(Debug, Clone)]
pub struct MsdfData {
    /// Red channel (SDF).
    pub r: Vec<f32>,
    /// Green channel (SDF).
    pub g: Vec<f32>,
    /// Blue channel (SDF).
    pub b: Vec<f32>,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl MsdfData {
    /// Create empty MSDF data.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            r: vec![0.0; size],
            g: vec![0.0; size],
            b: vec![0.0; size],
            width,
            height,
        }
    }

    /// Convert to RGB texture data.
    #[must_use]
    pub fn to_texture(&self, spread: f32) -> Vec<u8> {
        let size = (self.width * self.height) as usize;
        let mut data = Vec::with_capacity(size * 3);

        for i in 0..size {
            let r = f32::midpoint(self.r[i] / spread, 1.0).clamp(0.0, 1.0);
            let g = f32::midpoint(self.g[i] / spread, 1.0).clamp(0.0, 1.0);
            let b = f32::midpoint(self.b[i] / spread, 1.0).clamp(0.0, 1.0);

            data.push(f32_to_u8_sat(r * 255.0));
            data.push(f32_to_u8_sat(g * 255.0));
            data.push(f32_to_u8_sat(b * 255.0));
        }

        data
    }

    /// Sample median value for MSDF rendering.
    #[must_use]
    #[allow(
        clippy::many_single_char_names,
        reason = "x/y position and r/g/b channel names are the standard convention"
    )]
    pub fn sample_median(&self, x: f32, y: f32) -> f32 {
        let r = sample_sdf_bilinear(&self.r, self.width, self.height, x, y);
        let g = sample_sdf_bilinear(&self.g, self.width, self.height, x, y);
        let b = sample_sdf_bilinear(&self.b, self.width, self.height, x, y);

        // Median of three
        r.max(g.min(b)).min(g.max(b))
    }
}

/// SDF glyph metrics.
#[derive(Debug, Clone, Copy)]
pub struct SdfGlyphMetrics {
    /// UV coordinates in atlas [u0, v0, u1, v1].
    pub uv: [f32; 4],
    /// Offset from baseline.
    pub offset: Point,
    /// Size in pixels.
    pub size: [f32; 2],
    /// Advance width.
    pub advance: f32,
    /// SDF spread used.
    pub spread: f32,
}

/// SDF text rendering batch.
#[derive(Debug, Clone)]
pub struct SdfTextBatch {
    /// Instances to render.
    pub instances: Vec<SdfTextInstance>,
    /// Render parameters.
    pub params: SdfRenderParams,
}

/// SDF text instance.
#[derive(Debug, Clone, Copy)]
pub struct SdfTextInstance {
    /// Position on screen.
    pub position: Point,
    /// UV coordinates.
    pub uv: [f32; 4],
    /// Size.
    pub size: [f32; 2],
    /// Color (RGBA).
    pub color: [f32; 4],
    /// Scale factor.
    pub scale: f32,
}

impl SdfTextBatch {
    /// Create a new batch.
    #[must_use]
    pub const fn new(params: SdfRenderParams) -> Self {
        Self {
            instances: Vec::new(),
            params,
        }
    }

    /// Add a glyph instance.
    pub fn add_glyph(
        &mut self,
        metrics: &SdfGlyphMetrics,
        position: Point,
        scale: f32,
        color: [f32; 4],
    ) {
        self.instances.push(SdfTextInstance {
            position: Point::new(
                metrics.offset.x.mul_add(scale, position.x),
                metrics.offset.y.mul_add(scale, position.y),
            ),
            uv: metrics.uv,
            size: [metrics.size[0] * scale, metrics.size[1] * scale],
            color,
            scale,
        });
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Get instance count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Clear the batch.
    pub fn clear(&mut self) {
        self.instances.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_util::u32_from_usize;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_sdf_config() {
        let config = SdfConfig::default();
        assert_eq!(config.size, 64);
        assert_eq!(config.spread, 8.0);

        let high = SdfConfig::high_res();
        assert!(high.size > config.size);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_sdf_render_params() {
        let params = SdfRenderParams::default();
        assert!(params.smoothing > 0.0);
        assert_eq!(params.outline_width, 0.0);

        let outline = SdfRenderParams::with_outline(8.0, 2.0);
        assert_eq!(outline.outline_width, 2.0);
    }

    #[test]
    fn test_generate_circle_sdf() {
        let sdf = generate_circle_sdf(32, 10.0);
        assert_eq!(sdf.len(), 32 * 32);

        // Center should be inside (negative)
        let center_idx = 16 * 32 + 16;
        assert!(sdf[center_idx] < 0.0);

        // Corner should be outside (positive)
        assert!(sdf[0] > 0.0);
    }

    #[test]
    fn test_generate_rounded_rect_sdf() {
        let rect = Rect::from_xywh(4.0, 4.0, 24.0, 24.0);
        let sdf = generate_rounded_rect_sdf(32, rect, 4.0);
        assert_eq!(sdf.len(), 32 * 32);

        // Center should be inside
        let center_idx = 16 * 32 + 16;
        assert!(sdf[center_idx] < 0.0);
    }

    #[test]
    fn test_sdf_to_texture_inside_is_high() {
        // Per SkDistanceFieldGen: inside (negative signed distance) packs to
        // HIGH bytes (>128); the edge (0) is at 128; outside is low.
        let sdf = vec![-8.0, 0.0, 8.0];
        let texture = sdf_to_texture(&sdf, 8.0);

        assert_eq!(texture.len(), 3);
        assert_eq!(texture[0], 255, "fully inside -> high");
        assert_eq!(texture[1], 128, "edge -> 128");
        assert_eq!(texture[2], 0, "fully outside -> low");
        // Monotonic: inside strictly greater than outside.
        assert!(texture[0] > texture[1] && texture[1] > texture[2]);
    }

    #[test]
    fn test_sample_sdf_bilinear() {
        let sdf = vec![0.0, 1.0, 2.0, 3.0];
        let val = sample_sdf_bilinear(&sdf, 2, 2, 0.5, 0.5);
        // Should be average of all four corners
        assert!((val - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_msdf_data() {
        let mut msdf = MsdfData::new(32, 32);
        assert_eq!(msdf.r.len(), 32 * 32);

        // Set some values
        msdf.r[0] = 1.0;
        msdf.g[0] = 2.0;
        msdf.b[0] = 3.0;

        let median = msdf.sample_median(0.0, 0.0);
        assert!((median - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_generate_sdf_from_mask_basic() {
        // 8x8 mask with a 4x4 filled square in the centre.
        let w = 8;
        let h = 8;
        let mut mask = vec![0u8; w * h];
        for y in 2..6 {
            for x in 2..6 {
                mask[y * w + x] = 255;
            }
        }

        let sdf = generate_sdf_from_mask(&mask, u32_from_usize(w), u32_from_usize(h), 10.0);
        assert_eq!(sdf.len(), 64);

        // Centre of the square should be interior (negative).
        let centre = sdf[3 * w + 3];
        assert!(centre < 0.0, "centre should be inside (negative)");

        // Corner of the canvas should be exterior (positive).
        let corner = sdf[0];
        assert!(corner > 0.0, "canvas corner should be outside");

        // A pixel at the boundary (row 2, column 2 is inside; row 1, col 2
        // is outside and adjacent) — its distance should be near 1.
        let just_outside = sdf[w + 2];
        assert!(just_outside > 0.0 && just_outside < 2.0);
    }

    #[test]
    fn test_distance_transform_1d_exact() {
        // A 1D array with a single source at index 3 should produce squared
        // distance equal to (i - 3)^2 for each i.
        let inf = f32::MAX / 4.0;
        let mut f = vec![inf; 7];
        f[3] = 0.0;
        distance_transform_1d(&mut f);
        for (i, &value) in f.iter().enumerate() {
            let signed_i = i32::try_from(i).unwrap_or(0) - 3;
            let expected = scalar_from_u32(u32::try_from(signed_i * signed_i).unwrap_or(0));
            assert!(
                (value - expected).abs() < 1e-4,
                "idx {i}: {value} vs {expected}"
            );
        }
    }

    #[test]
    fn test_distance_transform_1d_multiple_sources() {
        // Two sources at ends: each pixel's distance is to the nearest one.
        let inf = f32::MAX / 4.0;
        let mut f = vec![inf; 7];
        f[0] = 0.0;
        f[6] = 0.0;
        distance_transform_1d(&mut f);
        // pixel 3 is equidistant (3 each way) → 9.
        assert!((f[3] - 9.0).abs() < 1e-4);
        // pixel 2 is 2 from 0 → 4.
        assert!((f[2] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_sdf_generation_scales_linearly() {
        // Coarse regression guard: generating a reasonably large SDF must
        // return in finite time. The old brute-force algorithm with spread
        // 100 on a 64x64 mask performed ~4M distance checks per pixel; the
        // F&H transform performs at most ~n per dimension.
        //
        // We don't time the call directly (too flaky for CI); we assert
        // that the result is self-consistent for a single-pixel source:
        // every pixel's distance is its Euclidean distance to that source,
        // up to the spread clamp.
        let w = 32;
        let h = 32;
        let mut mask = vec![0u8; w * h];
        mask[16 * w + 16] = 255; // single interior pixel

        let sdf = generate_sdf_from_mask(&mask, u32_from_usize(w), u32_from_usize(h), 100.0);
        // The inside pixel has nearest outside at distance 1; with the
        // half-texel edge correction the signed distance is -(1 - 0.5) = -0.5.
        let inside = sdf[16 * w + 16];
        assert!((inside + 0.5).abs() < 1e-4, "expected -0.5, got {inside}");
        // A pixel three away (outside): distance-to-inside 3 -> 3 - 0.5 = 2.5.
        let three_away = sdf[16 * w + 19];
        assert!(
            (three_away - 2.5).abs() < 1e-4,
            "expected ~2.5, got {three_away}"
        );
    }

    #[test]
    fn test_sdf_edge_offset_straddles_zero() {
        // Regression: the two texels straddling a mask edge must sit at
        // -0.5 (inside) and +0.5 (outside), i.e. the edge is at 0 between
        // them (half-texel offset correction).
        let w = 8;
        let h = 8;
        let mut mask = vec![0u8; w * h];
        for y in 0..h {
            for x in 4..w {
                mask[y * w + x] = 255; // right half filled
            }
        }
        let sdf = generate_sdf_from_mask(&mask, u32_from_usize(w), u32_from_usize(h), 10.0);
        // Column 4 is the first inside column; column 3 is the last outside.
        let inside_edge = sdf[3 * w + 4];
        let outside_edge = sdf[3 * w + 3];
        assert!(
            (inside_edge + 0.5).abs() < 1e-4,
            "inside edge ~ -0.5, got {inside_edge}"
        );
        assert!(
            (outside_edge - 0.5).abs() < 1e-4,
            "outside edge ~ +0.5, got {outside_edge}"
        );
    }

    #[test]
    fn test_generate_sdf_from_mask_packs_inside_high() {
        // End-to-end: a filled region must pack to HIGH texels, background low.
        let w = 8;
        let h = 8;
        let mut mask = vec![0u8; w * h];
        for y in 2..6 {
            for x in 2..6 {
                mask[y * w + x] = 255;
            }
        }
        let sdf = generate_sdf_from_mask(&mask, u32_from_usize(w), u32_from_usize(h), 8.0);
        let tex = sdf_to_texture(&sdf, 8.0);
        assert!(tex[3 * w + 3] > 128, "inside texel must be high");
        assert!(tex[0] < 128, "outside texel must be low");
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_sdf_text_batch() {
        let params = SdfRenderParams::default();
        let mut batch = SdfTextBatch::new(params);

        assert!(batch.is_empty());

        let metrics = SdfGlyphMetrics {
            uv: [0.0, 0.0, 0.1, 0.1],
            offset: Point::new(0.0, -10.0),
            size: [16.0, 20.0],
            advance: 10.0,
            spread: 8.0,
        };

        batch.add_glyph(
            &metrics,
            Point::new(100.0, 100.0),
            2.0,
            [1.0, 1.0, 1.0, 1.0],
        );

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.instances[0].size, [32.0, 40.0]); // Scaled by 2
    }
}
