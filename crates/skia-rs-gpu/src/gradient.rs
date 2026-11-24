//! Gradient texture generation for GPU rendering.
//!
//! This module provides utilities for converting gradient definitions
//! into textures suitable for GPU sampling.

use crate::cast_util::{scalar_from_u32, usize_from_scalar_sat};
use skia_rs_core::Color4f;
use skia_rs_core::cast::f32_to_u8_sat;

/// Gradient stop.
#[derive(Debug, Clone, Copy)]
pub struct GradientStop {
    /// Position (0.0 to 1.0).
    pub position: f32,
    /// Color at this position.
    pub color: Color4f,
}

impl GradientStop {
    /// Create a new gradient stop.
    #[must_use]
    pub const fn new(position: f32, color: Color4f) -> Self {
        Self {
            position: position.clamp(0.0, 1.0),
            color,
        }
    }
}

/// Gradient type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientType {
    /// Linear gradient.
    Linear,
    /// Radial gradient.
    Radial,
    /// Sweep (angular) gradient.
    Sweep,
    /// Two-point conical gradient.
    TwoPointConical,
}

/// Tile mode for gradient edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradientTileMode {
    /// Clamp to edge colors.
    #[default]
    Clamp,
    /// Repeat the gradient.
    Repeat,
    /// Mirror the gradient.
    Mirror,
    /// Decal mode (transparent outside).
    Decal,
}

/// Configuration for gradient texture generation.
#[derive(Debug, Clone)]
pub struct GradientTextureConfig {
    /// Texture width.
    pub width: u32,
    /// Texture height (1 for 1D gradients).
    pub height: u32,
    /// Use sRGB color space.
    pub srgb: bool,
    /// Premultiply alpha.
    pub premultiply: bool,
    /// Generate mipmaps.
    pub mipmaps: bool,
}

impl Default for GradientTextureConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 1,
            srgb: true,
            premultiply: true,
            mipmaps: false,
        }
    }
}

/// Encode a sampled gradient color into an RGBA8 texel.
///
/// Order matters and follows Skia: the transfer function (sRGB) is applied
/// **before** premultiplication, so a premultiplied sRGB target stores
/// `srgb(c) * a` — not `srgb(c * a)`. Premultiplying in the linear domain and
/// then encoding (the previous behavior) darkens partially-transparent
/// gradient stops. Byte conversion rounds to nearest.
fn encode_texel(color: Color4f, config: &GradientTextureConfig) -> [u8; 4] {
    let (mut r, mut g, mut b) = (color.r, color.g, color.b);
    if config.srgb {
        r = linear_to_srgb(r);
        g = linear_to_srgb(g);
        b = linear_to_srgb(b);
    }
    let a = color.a;
    if config.premultiply {
        r *= a;
        g *= a;
        b *= a;
    }
    [
        f32_to_u8_sat(r * 255.0),
        f32_to_u8_sat(g * 255.0),
        f32_to_u8_sat(b * 255.0),
        f32_to_u8_sat(a * 255.0),
    ]
}

/// Generate a 1D gradient texture.
#[must_use]
pub fn generate_gradient_texture_1d(
    stops: &[GradientStop],
    tile_mode: GradientTileMode,
    config: &GradientTextureConfig,
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(config.width as usize * 4);

    // Ensure stops are sorted
    let mut sorted_stops = stops.to_vec();
    sorted_stops.sort_by(|a, b| a.position.total_cmp(&b.position));

    // Ensure we have at least 2 stops
    if sorted_stops.is_empty() {
        sorted_stops.push(GradientStop::new(0.0, Color4f::black()));
        sorted_stops.push(GradientStop::new(1.0, Color4f::black()));
    } else if sorted_stops.len() == 1 {
        let color = sorted_stops[0].color;
        sorted_stops.clear();
        sorted_stops.push(GradientStop::new(0.0, color));
        sorted_stops.push(GradientStop::new(1.0, color));
    }

    // Add implicit stops at 0 and 1 if needed
    if sorted_stops[0].position > 0.0 {
        sorted_stops.insert(0, GradientStop::new(0.0, sorted_stops[0].color));
    }
    if let Some(last) = sorted_stops.last()
        && last.position < 1.0
    {
        let last_color = last.color;
        sorted_stops.push(GradientStop::new(1.0, last_color));
    }

    for x in 0..config.width {
        // Sample at the texel *center* (half-texel offset) so the ramp maps
        // across [0, 1). For Repeat this makes the final texel hold the
        // t->1- color rather than wrapping (via rem_euclid) back to the
        // first stop, which produced a visible seam at the tile boundary.
        let mut t = (scalar_from_u32(x) + 0.5) / scalar_from_u32(config.width);

        // Apply tile mode
        t = apply_tile_mode(t, tile_mode);

        // Find surrounding stops
        let color = sample_gradient(&sorted_stops, t);

        pixels.extend_from_slice(&encode_texel(color, config));
    }

    pixels
}

/// Generate a 2D radial gradient texture.
#[must_use]
pub fn generate_radial_gradient_texture(
    stops: &[GradientStop],
    tile_mode: GradientTileMode,
    config: &GradientTextureConfig,
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((config.width * config.height) as usize * 4);

    // Ensure stops are sorted
    let mut sorted_stops = stops.to_vec();
    sorted_stops.sort_by(|a, b| a.position.total_cmp(&b.position));

    if sorted_stops.is_empty() {
        sorted_stops.push(GradientStop::new(0.0, Color4f::black()));
        sorted_stops.push(GradientStop::new(1.0, Color4f::black()));
    }

    let center_x = scalar_from_u32(config.width) * 0.5;
    let center_y = scalar_from_u32(config.height) * 0.5;
    let max_radius = center_x.hypot(center_y);

    for y in 0..config.height {
        for x in 0..config.width {
            let dx = scalar_from_u32(x) - center_x;
            let dy = scalar_from_u32(y) - center_y;
            let mut t = dx.hypot(dy) / max_radius;

            t = apply_tile_mode(t, tile_mode);

            let color = sample_gradient(&sorted_stops, t);
            pixels.extend_from_slice(&encode_texel(color, config));
        }
    }

    pixels
}

/// Generate a sweep gradient texture.
#[must_use]
pub fn generate_sweep_gradient_texture(
    stops: &[GradientStop],
    tile_mode: GradientTileMode,
    config: &GradientTextureConfig,
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((config.width * config.height) as usize * 4);

    let mut sorted_stops = stops.to_vec();
    sorted_stops.sort_by(|a, b| a.position.total_cmp(&b.position));

    if sorted_stops.is_empty() {
        sorted_stops.push(GradientStop::new(0.0, Color4f::black()));
        sorted_stops.push(GradientStop::new(1.0, Color4f::black()));
    }

    let center_x = scalar_from_u32(config.width) * 0.5;
    let center_y = scalar_from_u32(config.height) * 0.5;

    for y in 0..config.height {
        for x in 0..config.width {
            let dx = scalar_from_u32(x) - center_x;
            let dy = scalar_from_u32(y) - center_y;

            // Sweep angle origin is the +x axis, increasing clockwise in the
            // y-down device space, matching Skia's `xy_to_unit_angle`
            // (SkRasterPipeline_opts.h). Normalize to [0, 1): at the +x axis
            // (dy=0, dx>0) atan2 is 0, so t=0 maps to the first stop.
            let two_pi = 2.0 * std::f32::consts::PI;
            let mut t = (dy.atan2(dx) / two_pi).rem_euclid(1.0);
            t = apply_tile_mode(t, tile_mode);

            let color = sample_gradient(&sorted_stops, t);
            pixels.extend_from_slice(&encode_texel(color, config));
        }
    }

    pixels
}

/// Apply tile mode to a gradient position.
fn apply_tile_mode(t: f32, mode: GradientTileMode) -> f32 {
    match mode {
        GradientTileMode::Clamp => t.clamp(0.0, 1.0),
        GradientTileMode::Repeat => t.rem_euclid(1.0),
        GradientTileMode::Mirror => {
            let cycle = t.rem_euclid(2.0);
            if cycle > 1.0 { 2.0 - cycle } else { cycle }
        }
        GradientTileMode::Decal => {
            if (0.0..=1.0).contains(&t) {
                t
            } else {
                -1.0 // Signal for transparent
            }
        }
    }
}

/// Sample gradient at position t.
fn sample_gradient(stops: &[GradientStop], t: f32) -> Color4f {
    if t < 0.0 {
        return Color4f::transparent();
    }

    // Find surrounding stops
    let mut lower_idx = 0;
    for (i, stop) in stops.iter().enumerate() {
        if stop.position <= t {
            lower_idx = i;
        } else {
            break;
        }
    }

    let upper_idx = (lower_idx + 1).min(stops.len() - 1);

    if lower_idx == upper_idx {
        return stops[lower_idx].color;
    }

    let lower = &stops[lower_idx];
    let upper = &stops[upper_idx];

    // Interpolate
    let range = upper.position - lower.position;
    let blend = if range > 0.0 {
        (t - lower.position) / range
    } else {
        0.0
    };

    Color4f::new(
        (upper.color.r - lower.color.r).mul_add(blend, lower.color.r),
        (upper.color.g - lower.color.g).mul_add(blend, lower.color.g),
        (upper.color.b - lower.color.b).mul_add(blend, lower.color.b),
        (upper.color.a - lower.color.a).mul_add(blend, lower.color.a),
    )
}

/// Convert linear color component to sRGB.
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055f32.mul_add(linear.powf(1.0 / 2.4), -0.055)
    }
}

/// Convert sRGB color component to linear.
#[must_use]
pub fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Gradient lookup table for shader use.
#[derive(Debug, Clone)]
pub struct GradientLUT {
    /// RGBA data.
    pub data: Vec<u8>,
    /// Width of the lookup table.
    pub width: u32,
}

impl GradientLUT {
    /// Create a new gradient LUT from stops.
    #[must_use]
    pub fn from_stops(stops: &[GradientStop], width: u32, tile_mode: GradientTileMode) -> Self {
        let config = GradientTextureConfig {
            width,
            height: 1,
            srgb: false, // Keep linear for shader use
            premultiply: true,
            mipmaps: false,
        };

        let data = generate_gradient_texture_1d(stops, tile_mode, &config);
        Self { data, width }
    }

    /// Sample the LUT at position t.
    #[must_use]
    pub fn sample(&self, t: f32) -> Color4f {
        let t = t.clamp(0.0, 1.0);
        let x = usize_from_scalar_sat((t * scalar_from_u32(self.width - 1)).round());
        let idx = x * 4;

        if idx + 3 < self.data.len() {
            Color4f::new(
                f32::from(self.data[idx]) / 255.0,
                f32::from(self.data[idx + 1]) / 255.0,
                f32::from(self.data[idx + 2]) / 255.0,
                f32::from(self.data[idx + 3]) / 255.0,
            )
        } else {
            Color4f::transparent()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_gradient_stop() {
        let stop = GradientStop::new(0.5, Color4f::from_rgb(1.0, 0.0, 0.0));
        assert_eq!(stop.position, 0.5);
        assert_eq!(stop.color.r, 1.0);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_gradient_stop_clamping() {
        let stop = GradientStop::new(1.5, Color4f::from_rgb(1.0, 0.0, 0.0));
        assert_eq!(stop.position, 1.0);

        let stop = GradientStop::new(-0.5, Color4f::from_rgb(1.0, 0.0, 0.0));
        assert_eq!(stop.position, 0.0);
    }

    #[test]
    fn test_gradient_sort_with_nan_position_does_not_panic() {
        // GradientStop::new() clamps NaN to 0.0, so construct directly to
        // bypass the constructor and exercise the sort with a NaN field.
        let stops = vec![
            GradientStop {
                position: 0.5,
                color: Color4f::from_rgb(1.0, 0.0, 0.0),
            },
            GradientStop {
                position: f32::NAN,
                color: Color4f::from_rgb(0.0, 1.0, 0.0),
            },
            GradientStop {
                position: 0.2,
                color: Color4f::from_rgb(0.0, 0.0, 1.0),
            },
        ];

        let config = GradientTextureConfig {
            width: 4,
            height: 4,
            ..Default::default()
        };

        // Should not panic despite the NaN position.
        let _ = generate_gradient_texture_1d(&stops, GradientTileMode::Clamp, &config);
        let _ = generate_radial_gradient_texture(&stops, GradientTileMode::Clamp, &config);
        let _ = generate_sweep_gradient_texture(&stops, GradientTileMode::Clamp, &config);
    }

    #[test]
    fn test_generate_gradient_1d() {
        let stops = vec![
            GradientStop::new(0.0, Color4f::from_rgb(1.0, 0.0, 0.0)),
            GradientStop::new(1.0, Color4f::from_rgb(0.0, 0.0, 1.0)),
        ];

        let config = GradientTextureConfig {
            width: 256,
            height: 1,
            srgb: false,
            premultiply: false,
            mipmaps: false,
        };

        let pixels = generate_gradient_texture_1d(&stops, GradientTileMode::Clamp, &config);
        assert_eq!(pixels.len(), 256 * 4);

        // First pixel should be red
        assert!(pixels[0] > 200); // R
        assert!(pixels[2] < 50); // B

        // Last pixel should be blue
        let last = (255 * 4) as usize;
        assert!(pixels[last] < 50); // R
        assert!(pixels[last + 2] > 200); // B
    }

    #[test]
    fn test_tile_mode_repeat() {
        let t = apply_tile_mode(1.5, GradientTileMode::Repeat);
        assert!((t - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_tile_mode_mirror() {
        let t = apply_tile_mode(1.5, GradientTileMode::Mirror);
        assert!((t - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_linear_srgb_conversion() {
        let linear = 0.5;
        let srgb = linear_to_srgb(linear);
        let back = srgb_to_linear(srgb);
        assert!((linear - back).abs() < 0.001);
    }

    #[test]
    fn test_premultiply_after_srgb_encoding() {
        // Regression: premul must happen AFTER transfer encoding
        // (store = srgb(c)*a), not before (srgb(c*a)). For solid red at
        // alpha 0.5: srgb(1.0)*0.5 = 0.5 -> 128, whereas the old order gave
        // srgb(0.5) ~ 0.735 -> ~188.
        let stops = vec![
            GradientStop::new(0.0, Color4f::new(1.0, 0.0, 0.0, 0.5)),
            GradientStop::new(1.0, Color4f::new(1.0, 0.0, 0.0, 0.5)),
        ];
        let config = GradientTextureConfig {
            width: 4,
            height: 1,
            srgb: true,
            premultiply: true,
            mipmaps: false,
        };
        let px = generate_gradient_texture_1d(&stops, GradientTileMode::Clamp, &config);
        assert!(
            (i32::from(px[0]) - 128).abs() <= 1,
            "premul-after-encode R should be ~128, got {}",
            px[0]
        );
        assert_eq!(px[3], 128, "alpha ~0.5 -> 128");
    }

    #[test]
    fn test_sweep_t_zero_at_positive_x() {
        // Regression: sweep t=0 must fall on the +x axis (first stop),
        // per Skia's xy_to_unit_angle.
        let stops = vec![
            GradientStop::new(0.0, Color4f::new(1.0, 0.0, 0.0, 1.0)), // red at t=0
            GradientStop::new(0.5, Color4f::new(0.0, 1.0, 0.0, 1.0)),
            GradientStop::new(1.0, Color4f::new(1.0, 0.0, 0.0, 1.0)),
        ];
        let config = GradientTextureConfig {
            width: 64,
            height: 64,
            srgb: false,
            premultiply: false,
            mipmaps: false,
        };
        let px = generate_sweep_gradient_texture(&stops, GradientTileMode::Repeat, &config);
        // Pixel to the right of center (+x axis): row 32, col 63.
        let idx = (32 * 64 + 63) * 4;
        assert!(
            px[idx] > 200,
            "+x should be near first stop (red), R={}",
            px[idx]
        );
        assert!(
            px[idx + 1] < 60,
            "+x should not be green, G={}",
            px[idx + 1]
        );
    }

    #[test]
    fn test_repeat_lut_no_wraparound_seam() {
        // Regression: for Repeat, the final LUT texel must hold the t->1-
        // color, not wrap back to the first stop (a seam).
        let stops = vec![
            GradientStop::new(0.0, Color4f::new(1.0, 0.0, 0.0, 1.0)),
            GradientStop::new(1.0, Color4f::new(0.0, 0.0, 1.0, 1.0)),
        ];
        let config = GradientTextureConfig {
            width: 64,
            height: 1,
            srgb: false,
            premultiply: false,
            mipmaps: false,
        };
        let px = generate_gradient_texture_1d(&stops, GradientTileMode::Repeat, &config);
        let last = (63 * 4) as usize;
        // Last texel should be near the last stop (blue), not the first (red).
        assert!(
            px[last + 2] > 200,
            "last texel B should be high, got {}",
            px[last + 2]
        );
        assert!(
            px[last] < 60,
            "last texel R should be low (no seam), got {}",
            px[last]
        );
    }

    #[test]
    fn test_gradient_lut() {
        let stops = vec![
            GradientStop::new(0.0, Color4f::from_rgb(1.0, 0.0, 0.0)),
            GradientStop::new(1.0, Color4f::from_rgb(0.0, 0.0, 1.0)),
        ];

        let lut = GradientLUT::from_stops(&stops, 256, GradientTileMode::Clamp);

        let start = lut.sample(0.0);
        assert!(start.r > 0.9);

        let end = lut.sample(1.0);
        assert!(end.b > 0.9);

        let mid = lut.sample(0.5);
        assert!(mid.r > 0.3 && mid.r < 0.7);
        assert!(mid.b > 0.3 && mid.b < 0.7);
    }
}
