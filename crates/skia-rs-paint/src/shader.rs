//! Shader types for gradients, images, and patterns.
//!
//! This module provides Skia-compatible shader implementations for:
//! - Solid colors
//! - Linear gradients
//! - Radial gradients
//! - Sweep (angular) gradients
//! - Two-point conical gradients
//! - Image shaders
//! - Blend shaders

use bitflags::bitflags;
use skia_rs_core::cast::{floor_to_i32, saturate_to_i32, scalar_from_i32};
use skia_rs_core::{Color4f, Matrix, Point, Rect, Scalar};
use std::sync::Arc;

// =============================================================================
// Shader Kind Discriminants (for serialization)
// =============================================================================

pub(crate) const SHADER_KIND_COLOR: u8 = 1;
pub(crate) const SHADER_KIND_LINEAR_GRADIENT: u8 = 2;
pub(crate) const SHADER_KIND_RADIAL_GRADIENT: u8 = 3;
pub(crate) const SHADER_KIND_SWEEP_GRADIENT: u8 = 4;
pub(crate) const SHADER_KIND_TWO_POINT_CONICAL: u8 = 5;
pub(crate) const SHADER_KIND_IMAGE: u8 = 6;
pub(crate) const SHADER_KIND_PERLIN_NOISE: u8 = 7;
pub(crate) const SHADER_KIND_BLEND: u8 = 8;
pub(crate) const SHADER_KIND_LOCAL_MATRIX: u8 = 9;
pub(crate) const SHADER_KIND_COMPOSE: u8 = 10;
pub(crate) const SHADER_KIND_EMPTY: u8 = 11;

// =============================================================================
// Helper Functions for Serialization
// =============================================================================

/// Serialize a child shader (recursively).
///
/// Format: [1 byte present] [4 bytes length] [data]
/// If the shader cannot be serialized, writes 0 for absent.
fn serialize_child_shader(buf: &mut Vec<u8>, shader: &ShaderRef) {
    if let Some(child_data) = shader.serialize() {
        buf.push(1);
        let len = u32::try_from(child_data.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&child_data);
    } else {
        buf.push(0);
    }
}

/// Serialize common gradient data (colors, positions, `tile_mode`, flags, `local_matrix`).
fn serialize_gradient_common(
    buf: &mut Vec<u8>,
    colors: &[Color4f],
    positions: Option<&[Scalar]>,
    tile_mode: TileMode,
    flags: GradientFlags,
    local_matrix: Option<&Matrix>,
) {
    // Tile mode (1 byte)
    buf.push(tile_mode as u8);

    // Flags (4 bytes)
    buf.extend_from_slice(&flags.bits().to_le_bytes());

    // Colors count (4 bytes)
    let colors_len = u32::try_from(colors.len()).unwrap_or(u32::MAX);
    buf.extend_from_slice(&colors_len.to_le_bytes());
    for color in colors {
        buf.extend_from_slice(&color.r.to_le_bytes());
        buf.extend_from_slice(&color.g.to_le_bytes());
        buf.extend_from_slice(&color.b.to_le_bytes());
        buf.extend_from_slice(&color.a.to_le_bytes());
    }

    // Positions (1 byte present flag + data if present)
    if let Some(positions) = positions {
        buf.push(1);
        for pos in positions {
            buf.extend_from_slice(&pos.to_le_bytes());
        }
    } else {
        buf.push(0);
    }

    // Local matrix (1 byte present flag + 36 bytes if present)
    if let Some(matrix) = local_matrix {
        buf.push(1);
        for i in 0..9 {
            buf.extend_from_slice(&matrix.values[i].to_le_bytes());
        }
    } else {
        buf.push(0);
    }
}

// =============================================================================
// Helper Functions for Gradient Sampling
// =============================================================================

/// Apply tile mode to a t value (typically 0-1 range).
#[inline]
fn apply_tile_mode(t: Scalar, mode: TileMode) -> Scalar {
    match mode {
        TileMode::Clamp => t.clamp(0.0, 1.0),
        TileMode::Repeat => t.rem_euclid(1.0),
        TileMode::Mirror => {
            let t = t.rem_euclid(2.0);
            if t > 1.0 { 2.0 - t } else { t }
        }
        TileMode::Decal => t, // Colors outside 0-1 handled separately
    }
}

/// Interpolate a gradient color at position t.
///
/// Positions are handled like `SkGradientBaseShader`: each explicit stop is
/// pinned monotonic into `[prev, 1]` (`SkTPin`), a first stop greater than 0
/// implies an extra stop at t=0 carrying the first color
/// (`fFirstStopIsImplicit`), and a last stop below 1 implies an extra stop at
/// t=1 carrying the last color (`fLastStopIsImplicit`).
///
/// Interpolation happens in the color space of `colors` as given — callers
/// choose premultiplied or straight stops.
#[inline]
fn interpolate_gradient_color(
    colors: &[Color4f],
    positions: Option<&[Scalar]>,
    t: Scalar,
) -> Color4f {
    if colors.is_empty() {
        return Color4f::transparent();
    }
    if colors.len() == 1 {
        return colors[0];
    }

    // Handle out-of-bounds for decal mode
    if !(0.0..=1.0).contains(&t) {
        return Color4f::transparent();
    }

    if let Some(pos) = positions.filter(|p| p.len() == colors.len()) {
        // Pin positions monotonic into [0, 1] (SkTPin semantics).
        let mut pinned: Vec<Scalar> = Vec::with_capacity(pos.len());
        let mut prev: Scalar = 0.0;
        for &p in pos {
            let cur = if p.is_nan() { prev } else { p.clamp(prev, 1.0) };
            pinned.push(cur);
            prev = cur;
        }

        // Implicit stop at t=0: everything below the first explicit stop
        // takes the first color.
        if t <= pinned[0] {
            return colors[0];
        }
        // Find the segment containing t.
        for i in 0..pinned.len() - 1 {
            if t <= pinned[i + 1] {
                let segment_t = if pinned[i + 1] > pinned[i] {
                    (t - pinned[i]) / (pinned[i + 1] - pinned[i])
                } else {
                    0.0
                };
                return colors[i].lerp(&colors[i + 1], segment_t);
            }
        }
        // Implicit stop at t=1: everything past the last explicit stop
        // takes the last color.
        return colors[colors.len() - 1];
    }

    // Uniform positions
    let stops = i32::try_from(colors.len() - 1).unwrap_or(i32::MAX);
    let scaled = t * scalar_from_i32(stops);
    let idx_i32 = saturate_to_i32(scaled.floor());
    let idx = usize::try_from(idx_i32).unwrap_or(0);
    let frac = scaled - scalar_from_i32(idx_i32);

    if idx >= colors.len() - 1 {
        colors[colors.len() - 1]
    } else {
        colors[idx].lerp(&colors[idx + 1], frac)
    }
}

/// Interpolate a gradient color at position t, honoring flags.
///
/// Returns a **premultiplied** color, per the `Shader::sample` contract.
/// With `INTERPOLATE_PREMUL` the stops are premultiplied before
/// interpolation (Skia's `kInterpolateColorsInPremul`); otherwise stops are
/// interpolated in straight (unpremultiplied) space — Skia's default — and
/// the result is premultiplied on output.
#[inline]
fn interpolate_gradient_color_with_flags(
    colors: &[Color4f],
    positions: Option<&[Scalar]>,
    t: Scalar,
    flags: GradientFlags,
) -> Color4f {
    if flags.contains(GradientFlags::INTERPOLATE_PREMUL) {
        if colors.is_empty() {
            return Color4f::transparent();
        }
        if colors.len() == 1 {
            return colors[0].premul();
        }

        let premul: Vec<Color4f> = colors.iter().map(skia_rs_core::Color4f::premul).collect();
        // Already premultiplied — return as-is.
        interpolate_gradient_color(&premul, positions, t)
    } else {
        interpolate_gradient_color(colors, positions, t).premul()
    }
}

// =============================================================================
// Helper Functions for Image Sampling
// =============================================================================

/// Map a coordinate into the image range [0, size) using the given tile mode.
///
/// Returns NaN if the coordinate is outside the range under Decal mode —
/// callers should treat NaN results as transparent pixels.
fn apply_image_tile(coord: Scalar, size: Scalar, mode: TileMode) -> Scalar {
    if size <= 0.0 {
        return Scalar::NAN;
    }
    match mode {
        TileMode::Clamp => coord.clamp(0.0, size - 1e-4),
        TileMode::Repeat => {
            let m = coord.rem_euclid(size);
            if m < 0.0 { m + size } else { m }
        }
        TileMode::Mirror => {
            let n = coord.rem_euclid(2.0 * size);
            if n < size {
                n
            } else {
                2.0f32.mul_add(size, -n) - 1e-4
            }
        }
        TileMode::Decal => {
            if coord < 0.0 || coord >= size {
                Scalar::NAN
            } else {
                coord
            }
        }
    }
}

/// Read a single RGBA8 pixel at integer coordinates.
///
/// Returns premultiplied Color4f in [0, 1]. For non-RGBA8 color types,
/// the read is simplified (baseline support).
#[allow(
    clippy::many_single_char_names,
    reason = "r/g/b/a are the conventional names for color channel components"
)]
fn read_pixel_rgba8(
    pixels: &[u8],
    info: &skia_rs_core::pixel::ImageInfo,
    x: i32,
    y: i32,
) -> Color4f {
    use skia_rs_core::color::ColorType;
    let bpp = info.bytes_per_pixel();
    let row_bytes = info.min_row_bytes();
    let x = usize::try_from(x).unwrap_or(usize::MAX);
    let y = usize::try_from(y).unwrap_or(usize::MAX);
    let offset = y * row_bytes + x * bpp;
    if offset + bpp > pixels.len() {
        return Color4f::new(0.0, 0.0, 0.0, 0.0);
    }

    match info.color_type {
        ColorType::Rgba8888 => {
            let r = f32::from(pixels[offset]) / 255.0;
            let g = f32::from(pixels[offset + 1]) / 255.0;
            let b = f32::from(pixels[offset + 2]) / 255.0;
            let a = f32::from(pixels[offset + 3]) / 255.0;
            Color4f::new(r, g, b, a)
        }
        ColorType::Bgra8888 => {
            let b = f32::from(pixels[offset]) / 255.0;
            let g = f32::from(pixels[offset + 1]) / 255.0;
            let r = f32::from(pixels[offset + 2]) / 255.0;
            let a = f32::from(pixels[offset + 3]) / 255.0;
            Color4f::new(r, g, b, a)
        }
        ColorType::Rgb888x => {
            let r = f32::from(pixels[offset]) / 255.0;
            let g = f32::from(pixels[offset + 1]) / 255.0;
            let b = f32::from(pixels[offset + 2]) / 255.0;
            Color4f::new(r, g, b, 1.0)
        }
        ColorType::Alpha8 => {
            // Alpha-only sources decode as black with alpha (0,0,0,a);
            // upstream colorizes by the paint color at draw time.
            let a = f32::from(pixels[offset]) / 255.0;
            Color4f::new(0.0, 0.0, 0.0, a)
        }
        ColorType::Gray8 => {
            let g = f32::from(pixels[offset]) / 255.0;
            Color4f::new(g, g, g, 1.0)
        }
        _ => Color4f::new(0.0, 0.0, 0.0, 0.0),
    }
}

/// Tile mode for shaders.
///
/// Determines how a shader handles coordinates outside its bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TileMode {
    /// Clamp to edge color.
    #[default]
    Clamp = 0,
    /// Repeat the pattern.
    Repeat,
    /// Mirror the pattern.
    Mirror,
    /// Extend with transparent (coordinates outside bounds are transparent).
    Decal,
}

bitflags! {
    /// Flags controlling gradient behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct GradientFlags: u32 {
        /// Interpolate colors in premultiplied space.
        const INTERPOLATE_PREMUL = 1 << 0;
    }
}

/// A shader that generates colors for drawing.
///
/// Corresponds to Skia's `SkShader`.
pub trait Shader: Send + Sync + std::fmt::Debug {
    /// Get the local matrix.
    fn local_matrix(&self) -> Option<&Matrix>;

    /// Check if this shader is opaque.
    fn is_opaque(&self) -> bool;

    /// Returns the kind of shader for debugging.
    fn shader_kind(&self) -> ShaderKind;

    /// Sample the shader at a given point.
    ///
    /// The point is in the shader's local coordinate space (after applying
    /// any local matrix).
    ///
    /// **Contract: the returned color is premultiplied.** This matches
    /// Skia's raster pipeline, where shader stages emit premultiplied
    /// colors that feed directly into blend stages
    /// (`skia/src/opts/SkRasterPipeline_opts.h`). All shaders in this crate
    /// (solid colors, gradients, images, blends, runtime effects) follow
    /// this convention, and `BlendMode::apply` expects premultiplied
    /// inputs. Consumers that need straight alpha must unpremultiply.
    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        // Default implementation returns transparent
        let _ = (x, y);
        Color4f::transparent()
    }

    /// Serialize this shader to bytes.
    ///
    /// Returns `None` for shader types that do not support serialization
    /// (e.g., runtime shaders with `SkSL` source). Built-in shaders override
    /// this to return `Some(bytes)`.
    fn serialize(&self) -> Option<Vec<u8>> {
        None
    }

    /// Concrete-type downcast hook.
    ///
    /// Returns `Some(self as &dyn Any)` for shaders that support exact
    /// parameter extraction (the built-in gradients), enabling consumers such
    /// as the GPU paint bridge to read real geometry — endpoints, radius,
    /// angles, color stops — rather than probing via [`Shader::sample`].
    /// Defaults to `None`; a `None` result means the caller must fall back to
    /// sampling. Kept as an explicit accessor (rather than an `Any`
    /// supertrait) to stay within the crate's MSRV, which predates trait
    /// upcasting.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// Kind of shader (for debugging/inspection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    /// Solid color shader.
    Color,
    /// Linear gradient shader.
    LinearGradient,
    /// Radial gradient shader.
    RadialGradient,
    /// Sweep/angular gradient shader.
    SweepGradient,
    /// Two-point conical gradient shader.
    TwoPointConicalGradient,
    /// Image shader.
    Image,
    /// Blend shader (combines two shaders).
    Blend,
    /// Perlin noise shader.
    PerlinNoise,
    /// Local matrix wrapper shader.
    LocalMatrix,
    /// Composed shader (chain of shaders).
    Compose,
    /// Empty/null shader.
    Empty,
}

/// A solid color shader.
///
/// Corresponds to Skia's `SkColorShader`.
#[derive(Debug, Clone)]
pub struct ColorShader {
    color: Color4f,
}

impl ColorShader {
    /// Create a new solid color shader.
    #[inline]
    #[must_use]
    pub const fn new(color: Color4f) -> Self {
        Self { color }
    }

    /// Get the color.
    #[inline]
    #[must_use]
    pub const fn color(&self) -> Color4f {
        self.color
    }
}

impl Shader for ColorShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        None
    }

    fn is_opaque(&self) -> bool {
        self.color.a >= 1.0
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::Color
    }

    fn sample(&self, _x: Scalar, _y: Scalar) -> Color4f {
        // sample() returns premultiplied color; the stored color is straight.
        self.color.premul()
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_COLOR);
        buf.extend_from_slice(&self.color.r.to_le_bytes());
        buf.extend_from_slice(&self.color.g.to_le_bytes());
        buf.extend_from_slice(&self.color.b.to_le_bytes());
        buf.extend_from_slice(&self.color.a.to_le_bytes());
        Some(buf)
    }
}

/// Linear gradient shader.
///
/// Corresponds to Skia's `SkGradientShader::MakeLinear`.
#[derive(Debug, Clone)]
pub struct LinearGradient {
    start: Point,
    end: Point,
    colors: Vec<Color4f>,
    positions: Option<Vec<Scalar>>,
    tile_mode: TileMode,
    flags: GradientFlags,
    local_matrix: Option<Matrix>,
}

impl LinearGradient {
    /// Create a new linear gradient.
    #[must_use]
    pub const fn new(
        start: Point,
        end: Point,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            start,
            end,
            colors,
            positions,
            tile_mode,
            flags: GradientFlags::empty(),
            local_matrix: None,
        }
    }

    /// Set the local matrix.
    #[must_use]
    pub const fn with_local_matrix(mut self, matrix: Matrix) -> Self {
        self.local_matrix = Some(matrix);
        self
    }

    /// Set gradient flags.
    #[must_use]
    pub const fn with_flags(mut self, flags: GradientFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Get the start point.
    #[inline]
    #[must_use]
    pub const fn start(&self) -> Point {
        self.start
    }

    /// Get the end point.
    #[inline]
    #[must_use]
    pub const fn end(&self) -> Point {
        self.end
    }

    /// Get the colors.
    #[inline]
    #[must_use]
    pub fn colors(&self) -> &[Color4f] {
        &self.colors
    }

    /// Get the positions.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> Option<&[Scalar]> {
        self.positions.as_deref()
    }

    /// Get the tile mode.
    #[inline]
    #[must_use]
    pub const fn tile_mode(&self) -> TileMode {
        self.tile_mode
    }
}

impl Shader for LinearGradient {
    fn local_matrix(&self) -> Option<&Matrix> {
        self.local_matrix.as_ref()
    }

    fn is_opaque(&self) -> bool {
        self.colors.iter().all(|c| c.a >= 1.0)
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::LinearGradient
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        // Calculate the projection of the point onto the gradient line
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let len_sq = dx.mul_add(dx, dy * dy);

        if len_sq < 1e-10 {
            // Degenerate gradient (start == end)
            return self
                .colors
                .first()
                .copied()
                .unwrap_or(Color4f::transparent())
                .premul();
        }

        // Project point onto gradient line
        let px = x - self.start.x;
        let py = y - self.start.y;
        let mut t = px.mul_add(dx, py * dy) / len_sq;

        // Apply tile mode
        t = apply_tile_mode(t, self.tile_mode);

        // Interpolate color
        interpolate_gradient_color_with_flags(
            &self.colors,
            self.positions.as_deref(),
            t,
            self.flags,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_LINEAR_GRADIENT);

        // Start point (8 bytes)
        buf.extend_from_slice(&self.start.x.to_le_bytes());
        buf.extend_from_slice(&self.start.y.to_le_bytes());

        // End point (8 bytes)
        buf.extend_from_slice(&self.end.x.to_le_bytes());
        buf.extend_from_slice(&self.end.y.to_le_bytes());

        // Common gradient data
        serialize_gradient_common(
            &mut buf,
            &self.colors,
            self.positions.as_deref(),
            self.tile_mode,
            self.flags,
            self.local_matrix.as_ref(),
        );

        Some(buf)
    }
}

/// Radial gradient shader.
///
/// Corresponds to Skia's `SkGradientShader::MakeRadial`.
#[derive(Debug, Clone)]
pub struct RadialGradient {
    center: Point,
    radius: Scalar,
    colors: Vec<Color4f>,
    positions: Option<Vec<Scalar>>,
    tile_mode: TileMode,
    flags: GradientFlags,
    local_matrix: Option<Matrix>,
}

impl RadialGradient {
    /// Create a new radial gradient.
    #[must_use]
    pub const fn new(
        center: Point,
        radius: Scalar,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            center,
            radius,
            colors,
            positions,
            tile_mode,
            flags: GradientFlags::empty(),
            local_matrix: None,
        }
    }

    /// Set the local matrix.
    #[must_use]
    pub const fn with_local_matrix(mut self, matrix: Matrix) -> Self {
        self.local_matrix = Some(matrix);
        self
    }

    /// Set gradient flags.
    #[must_use]
    pub const fn with_flags(mut self, flags: GradientFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Get the center point.
    #[inline]
    #[must_use]
    pub const fn center(&self) -> Point {
        self.center
    }

    /// Get the radius.
    #[inline]
    #[must_use]
    pub const fn radius(&self) -> Scalar {
        self.radius
    }

    /// Get the colors.
    #[inline]
    #[must_use]
    pub fn colors(&self) -> &[Color4f] {
        &self.colors
    }

    /// Get the positions.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> Option<&[Scalar]> {
        self.positions.as_deref()
    }

    /// Get the tile mode.
    #[inline]
    #[must_use]
    pub const fn tile_mode(&self) -> TileMode {
        self.tile_mode
    }
}

impl Shader for RadialGradient {
    fn local_matrix(&self) -> Option<&Matrix> {
        self.local_matrix.as_ref()
    }

    fn is_opaque(&self) -> bool {
        self.colors.iter().all(|c| c.a >= 1.0)
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::RadialGradient
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        if self.radius <= 0.0 {
            return self
                .colors
                .first()
                .copied()
                .unwrap_or(Color4f::transparent())
                .premul();
        }

        // Calculate distance from center
        let dx = x - self.center.x;
        let dy = y - self.center.y;
        let dist = dx.hypot(dy);
        let mut t = dist / self.radius;

        // Apply tile mode
        t = apply_tile_mode(t, self.tile_mode);

        // Interpolate color
        interpolate_gradient_color_with_flags(
            &self.colors,
            self.positions.as_deref(),
            t,
            self.flags,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_RADIAL_GRADIENT);

        // Center point (8 bytes)
        buf.extend_from_slice(&self.center.x.to_le_bytes());
        buf.extend_from_slice(&self.center.y.to_le_bytes());

        // Radius (4 bytes)
        buf.extend_from_slice(&self.radius.to_le_bytes());

        // Common gradient data
        serialize_gradient_common(
            &mut buf,
            &self.colors,
            self.positions.as_deref(),
            self.tile_mode,
            self.flags,
            self.local_matrix.as_ref(),
        );

        Some(buf)
    }
}

/// Sweep (angular) gradient shader.
///
/// Corresponds to Skia's `SkGradientShader::MakeSweep`.
#[derive(Debug, Clone)]
pub struct SweepGradient {
    center: Point,
    start_angle: Scalar,
    end_angle: Scalar,
    colors: Vec<Color4f>,
    positions: Option<Vec<Scalar>>,
    tile_mode: TileMode,
    flags: GradientFlags,
    local_matrix: Option<Matrix>,
}

impl SweepGradient {
    /// Create a new sweep gradient.
    ///
    /// Angles are in degrees, with 0 pointing right and increasing clockwise.
    #[must_use]
    pub const fn new(
        center: Point,
        start_angle: Scalar,
        end_angle: Scalar,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            center,
            start_angle,
            end_angle,
            colors,
            positions,
            tile_mode,
            flags: GradientFlags::empty(),
            local_matrix: None,
        }
    }

    /// Create a full sweep gradient (0-360 degrees).
    #[must_use]
    pub const fn new_full(
        center: Point,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
    ) -> Self {
        Self::new(center, 0.0, 360.0, colors, positions, TileMode::Clamp)
    }

    /// Set the local matrix.
    #[must_use]
    pub const fn with_local_matrix(mut self, matrix: Matrix) -> Self {
        self.local_matrix = Some(matrix);
        self
    }

    /// Set gradient flags.
    #[must_use]
    pub const fn with_flags(mut self, flags: GradientFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Get the center point.
    #[inline]
    #[must_use]
    pub const fn center(&self) -> Point {
        self.center
    }

    /// Get the start angle in degrees.
    #[inline]
    #[must_use]
    pub const fn start_angle(&self) -> Scalar {
        self.start_angle
    }

    /// Get the end angle in degrees.
    #[inline]
    #[must_use]
    pub const fn end_angle(&self) -> Scalar {
        self.end_angle
    }

    /// Get the colors.
    #[inline]
    #[must_use]
    pub fn colors(&self) -> &[Color4f] {
        &self.colors
    }

    /// Get the positions.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> Option<&[Scalar]> {
        self.positions.as_deref()
    }
}

impl Shader for SweepGradient {
    fn local_matrix(&self) -> Option<&Matrix> {
        self.local_matrix.as_ref()
    }

    fn is_opaque(&self) -> bool {
        self.colors.iter().all(|c| c.a >= 1.0)
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::SweepGradient
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        // Calculate angle from center
        let dx = x - self.center.x;
        let dy = y - self.center.y;

        // atan2 returns [-PI, PI], convert to [0, 360]
        let mut angle = dy.atan2(dx).to_degrees();
        if angle < 0.0 {
            angle += 360.0;
        }

        // Map angle to t value
        let sweep = self.end_angle - self.start_angle;
        let mut t = if sweep.abs() < 1e-10 {
            0.0
        } else {
            (angle - self.start_angle) / sweep
        };

        // Apply tile mode
        t = apply_tile_mode(t, self.tile_mode);

        // Interpolate color
        interpolate_gradient_color_with_flags(
            &self.colors,
            self.positions.as_deref(),
            t,
            self.flags,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_SWEEP_GRADIENT);

        // Center point (8 bytes)
        buf.extend_from_slice(&self.center.x.to_le_bytes());
        buf.extend_from_slice(&self.center.y.to_le_bytes());

        // Start and end angles (8 bytes)
        buf.extend_from_slice(&self.start_angle.to_le_bytes());
        buf.extend_from_slice(&self.end_angle.to_le_bytes());

        // Common gradient data
        serialize_gradient_common(
            &mut buf,
            &self.colors,
            self.positions.as_deref(),
            self.tile_mode,
            self.flags,
            self.local_matrix.as_ref(),
        );

        Some(buf)
    }
}

/// Two-point conical gradient shader.
///
/// Creates a gradient between two circles defined by center and radius.
/// Corresponds to Skia's `SkGradientShader::MakeTwoPointConical`.
#[derive(Debug, Clone)]
pub struct TwoPointConicalGradient {
    start_center: Point,
    start_radius: Scalar,
    end_center: Point,
    end_radius: Scalar,
    colors: Vec<Color4f>,
    positions: Option<Vec<Scalar>>,
    tile_mode: TileMode,
    flags: GradientFlags,
    local_matrix: Option<Matrix>,
}

impl TwoPointConicalGradient {
    /// Create a new two-point conical gradient.
    #[must_use]
    pub const fn new(
        start_center: Point,
        start_radius: Scalar,
        end_center: Point,
        end_radius: Scalar,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> Self {
        Self {
            start_center,
            start_radius,
            end_center,
            end_radius,
            colors,
            positions,
            tile_mode,
            flags: GradientFlags::empty(),
            local_matrix: None,
        }
    }

    /// Set the local matrix.
    #[must_use]
    pub const fn with_local_matrix(mut self, matrix: Matrix) -> Self {
        self.local_matrix = Some(matrix);
        self
    }

    /// Set gradient flags.
    #[must_use]
    pub const fn with_flags(mut self, flags: GradientFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Get the start center.
    #[inline]
    #[must_use]
    pub const fn start_center(&self) -> Point {
        self.start_center
    }

    /// Get the start radius.
    #[inline]
    #[must_use]
    pub const fn start_radius(&self) -> Scalar {
        self.start_radius
    }

    /// Get the end center.
    #[inline]
    #[must_use]
    pub const fn end_center(&self) -> Point {
        self.end_center
    }

    /// Get the end radius.
    #[inline]
    #[must_use]
    pub const fn end_radius(&self) -> Scalar {
        self.end_radius
    }

    /// Get the colors.
    #[inline]
    #[must_use]
    pub fn colors(&self) -> &[Color4f] {
        &self.colors
    }

    /// Get the positions.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> Option<&[Scalar]> {
        self.positions.as_deref()
    }

    /// Get the tile mode.
    #[inline]
    #[must_use]
    pub const fn tile_mode(&self) -> TileMode {
        self.tile_mode
    }
}

impl Shader for TwoPointConicalGradient {
    fn local_matrix(&self) -> Option<&Matrix> {
        self.local_matrix.as_ref()
    }

    fn is_opaque(&self) -> bool {
        self.colors.iter().all(|c| c.a >= 1.0)
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::TwoPointConicalGradient
    }

    #[allow(
        clippy::many_single_char_names,
        reason = "faithful port of Skia's two-point-conical quadratic solve; a/b/c/d/dr/e/t0/t1 mirror the derivation in the comment above"
    )]
    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        // Two-point conical gradient: solve quadratic for parameter t.
        // Point P at (x, y). Circles C0 = start_center (radius r0) and
        // C1 = end_center (radius r1).
        //
        // We need t such that P lies on the interpolated circle:
        //   center(t) = C0 + t * (C1 - C0)
        //   radius(t) = r0 + t * (r1 - r0)
        //   |P - center(t)| = radius(t)
        //
        // Let d = C1 - C0, dr = r1 - r0, e = P - C0.
        // Expanding: |e - t*d|^2 = (r0 + t*dr)^2
        //   e.e - 2t(e.d) + t^2(d.d) = r0^2 + 2t*r0*dr + t^2*dr^2
        //   t^2(d.d - dr^2) - 2t(e.d + r0*dr) + (e.e - r0^2) = 0
        //
        // Quadratic: A*t^2 + B*t + C = 0 with
        //   A = d.d - dr^2
        //   B = -2*(e.d + r0*dr)
        //   C = e.e - r0^2
        //
        // Pick the larger root (Skia convention for non-degenerate case).

        let dx = self.end_center.x - self.start_center.x;
        let dy = self.end_center.y - self.start_center.y;
        let dr = self.end_radius - self.start_radius;

        let ex = x - self.start_center.x;
        let ey = y - self.start_center.y;

        let a = dr.mul_add(-dr, dx.mul_add(dx, dy * dy));
        let b = -2.0 * self.start_radius.mul_add(dr, ex.mul_add(dx, ey * dy));
        let c = self
            .start_radius
            .mul_add(-self.start_radius, ex.mul_add(ex, ey * ey));

        let t_opt = if a.abs() < 1e-7 {
            // Linear case (d.d == dr^2): B*t + C = 0
            if b.abs() < 1e-7 { None } else { Some(-c / b) }
        } else {
            let disc = (4.0 * a).mul_add(-c, b * b);
            if disc < 0.0 {
                None
            } else {
                let sqrt_disc = disc.sqrt();
                // Two roots represent two circles the point could lie on.
                // Skia's well-behaved (non-swapped) case takes the +sqrt,
                // i.e. the LARGER t; the smaller root is only used when the
                // larger one yields a negative interpolated radius (the
                // swapped/negative-focal case).
                let t0 = (-b + sqrt_disc) / (2.0 * a);
                let t1 = (-b - sqrt_disc) / (2.0 * a);
                let t_larger = t0.max(t1);
                let t_smaller = t0.min(t1);
                let r_at = |t: Scalar| t.mul_add(dr, self.start_radius);
                if r_at(t_larger) >= 0.0 {
                    Some(t_larger)
                } else if r_at(t_smaller) >= 0.0 {
                    Some(t_smaller)
                } else {
                    None
                }
            }
        };

        let Some(t) = t_opt else {
            return Color4f::transparent();
        };

        // Apply tile mode to t, then interpolate colors (honoring flags,
        // e.g. INTERPOLATE_PREMUL, like the other gradient kinds).
        let t_tiled = apply_tile_mode(t, self.tile_mode);
        interpolate_gradient_color_with_flags(
            &self.colors,
            self.positions.as_deref(),
            t_tiled,
            self.flags,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_TWO_POINT_CONICAL);

        // Start center and radius (12 bytes)
        buf.extend_from_slice(&self.start_center.x.to_le_bytes());
        buf.extend_from_slice(&self.start_center.y.to_le_bytes());
        buf.extend_from_slice(&self.start_radius.to_le_bytes());

        // End center and radius (12 bytes)
        buf.extend_from_slice(&self.end_center.x.to_le_bytes());
        buf.extend_from_slice(&self.end_center.y.to_le_bytes());
        buf.extend_from_slice(&self.end_radius.to_le_bytes());

        // Common gradient data
        serialize_gradient_common(
            &mut buf,
            &self.colors,
            self.positions.as_deref(),
            self.tile_mode,
            self.flags,
            self.local_matrix.as_ref(),
        );

        Some(buf)
    }
}

/// Image shader that tiles an image.
///
/// Corresponds to Skia's `SkImageShader`.
#[derive(Debug, Clone)]
pub struct ImageShader {
    /// Image bounds (width, height).
    bounds: Rect,
    /// Tile mode for X axis.
    tile_mode_x: TileMode,
    /// Tile mode for Y axis.
    tile_mode_y: TileMode,
    /// Sampling options.
    sampling: SamplingOptions,
    /// Local matrix.
    local_matrix: Option<Matrix>,
    /// Owned pixel data (RGBA8 premultiplied, row-major).
    /// None if the shader was created without pixels (metadata-only).
    pixels: Option<Arc<Vec<u8>>>,
    /// Image info describing pixel layout.
    image_info: Option<skia_rs_core::pixel::ImageInfo>,
}

/// Sampling options for image shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SamplingOptions {
    /// Filter mode.
    pub filter: FilterMode,
    /// Mipmap mode.
    pub mipmap: MipmapMode,
}

/// Filter mode for image sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum FilterMode {
    /// Nearest neighbor sampling.
    #[default]
    Nearest = 0,
    /// Bilinear interpolation.
    Linear,
}

/// Mipmap mode for image sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum MipmapMode {
    /// No mipmapping.
    #[default]
    None = 0,
    /// Nearest mipmap level.
    Nearest,
    /// Linear interpolation between mipmap levels.
    Linear,
}

impl SamplingOptions {
    /// Nearest neighbor sampling (no filtering).
    pub const NEAREST: Self = Self {
        filter: FilterMode::Nearest,
        mipmap: MipmapMode::None,
    };

    /// Bilinear filtering.
    pub const LINEAR: Self = Self {
        filter: FilterMode::Linear,
        mipmap: MipmapMode::None,
    };

    /// Trilinear filtering (linear with linear mipmap).
    pub const TRILINEAR: Self = Self {
        filter: FilterMode::Linear,
        mipmap: MipmapMode::Linear,
    };
}

impl ImageShader {
    /// Create a new image shader.
    #[must_use]
    pub const fn new(
        bounds: Rect,
        tile_mode_x: TileMode,
        tile_mode_y: TileMode,
        sampling: SamplingOptions,
    ) -> Self {
        Self {
            bounds,
            tile_mode_x,
            tile_mode_y,
            sampling,
            local_matrix: None,
            pixels: None,
            image_info: None,
        }
    }

    /// Create an image shader with the same tile mode for both axes.
    #[must_use]
    pub const fn with_tile_mode(
        bounds: Rect,
        tile_mode: TileMode,
        sampling: SamplingOptions,
    ) -> Self {
        Self::new(bounds, tile_mode, tile_mode, sampling)
    }

    /// Create an `ImageShader` with owned pixel data.
    ///
    /// Pixels are treated as premultiplied RGBA8 in the sRGB color space.
    /// The `image_info` should describe the width, height, color type, and
    /// row bytes of the pixel buffer.
    #[must_use]
    pub const fn with_pixels(
        pixels: Arc<Vec<u8>>,
        image_info: skia_rs_core::pixel::ImageInfo,
        tile_mode_x: TileMode,
        tile_mode_y: TileMode,
        sampling: SamplingOptions,
    ) -> Self {
        let bounds = Rect::from_xywh(
            0.0,
            0.0,
            scalar_from_i32(image_info.width()),
            scalar_from_i32(image_info.height()),
        );
        Self {
            bounds,
            tile_mode_x,
            tile_mode_y,
            sampling,
            local_matrix: None,
            pixels: Some(pixels),
            image_info: Some(image_info),
        }
    }

    /// Set the local matrix.
    #[must_use]
    pub const fn with_local_matrix(mut self, matrix: Matrix) -> Self {
        self.local_matrix = Some(matrix);
        self
    }

    /// Get the image bounds.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Get the X tile mode.
    #[inline]
    #[must_use]
    pub const fn tile_mode_x(&self) -> TileMode {
        self.tile_mode_x
    }

    /// Get the Y tile mode.
    #[inline]
    #[must_use]
    pub const fn tile_mode_y(&self) -> TileMode {
        self.tile_mode_y
    }

    /// Get the sampling options.
    #[inline]
    #[must_use]
    pub const fn sampling(&self) -> SamplingOptions {
        self.sampling
    }
}

impl Shader for ImageShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        self.local_matrix.as_ref()
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        let (pixels, info) = match (&self.pixels, &self.image_info) {
            (Some(p), Some(i)) => (p.as_ref(), i),
            _ => return Color4f::new(0.0, 0.0, 0.0, 0.0),
        };

        let width = info.width();
        let height = info.height();
        if width <= 0 || height <= 0 {
            return Color4f::new(0.0, 0.0, 0.0, 0.0);
        }

        match self.sampling.filter {
            FilterMode::Nearest => {
                // Apply tile mode to map (x, y) into [0, width) x [0, height).
                let sx = apply_image_tile(x, scalar_from_i32(width), self.tile_mode_x);
                let sy = apply_image_tile(y, scalar_from_i32(height), self.tile_mode_y);

                if !sx.is_finite() || !sy.is_finite() || sx < 0.0 || sy < 0.0 {
                    return Color4f::new(0.0, 0.0, 0.0, 0.0);
                }

                // Nearest-neighbor sampling: round down to integer pixel.
                let ix = floor_to_i32(sx).clamp(0, width - 1);
                let iy = floor_to_i32(sy).clamp(0, height - 1);

                read_pixel_rgba8(pixels, info, ix, iy)
            }
            FilterMode::Linear => {
                // Bilinear filtering: texel centers sit at integer + 0.5, so
                // shift by 0.5 and blend the four surrounding texels. Each
                // neighbor coordinate is mapped through the tile mode; Decal
                // neighbors outside the image contribute transparent.
                let fx = x - 0.5;
                let fy = y - 0.5;
                let x0 = fx.floor();
                let y0 = fy.floor();
                let wx = fx - x0;
                let wy = fy - y0;

                let texel = |tx: Scalar, ty: Scalar| -> Color4f {
                    let sx = apply_image_tile(tx, scalar_from_i32(width), self.tile_mode_x);
                    let sy = apply_image_tile(ty, scalar_from_i32(height), self.tile_mode_y);
                    if !sx.is_finite() || !sy.is_finite() || sx < 0.0 || sy < 0.0 {
                        return Color4f::new(0.0, 0.0, 0.0, 0.0);
                    }
                    let ix = floor_to_i32(sx).clamp(0, width - 1);
                    let iy = floor_to_i32(sy).clamp(0, height - 1);
                    read_pixel_rgba8(pixels, info, ix, iy)
                };

                let c00 = texel(x0, y0);
                let c10 = texel(x0 + 1.0, y0);
                let c01 = texel(x0, y0 + 1.0);
                let c11 = texel(x0 + 1.0, y0 + 1.0);

                let lerp = |a: f32, b: f32, t: f32| (b - a).mul_add(t, a);
                let bilerp = |c00: f32, c10: f32, c01: f32, c11: f32| {
                    lerp(lerp(c00, c10, wx), lerp(c01, c11, wx), wy)
                };
                Color4f::new(
                    bilerp(c00.r, c10.r, c01.r, c11.r),
                    bilerp(c00.g, c10.g, c01.g, c11.g),
                    bilerp(c00.b, c10.b, c01.b, c11.b),
                    bilerp(c00.a, c10.a, c01.a, c11.a),
                )
            }
        }
    }

    fn is_opaque(&self) -> bool {
        // Image shaders are generally not assumed to be opaque
        // without analyzing the actual image data
        false
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::Image
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_IMAGE);

        // Bounds (16 bytes)
        buf.extend_from_slice(&self.bounds.left.to_le_bytes());
        buf.extend_from_slice(&self.bounds.top.to_le_bytes());
        buf.extend_from_slice(&self.bounds.right.to_le_bytes());
        buf.extend_from_slice(&self.bounds.bottom.to_le_bytes());

        // Tile modes (2 bytes)
        buf.push(self.tile_mode_x as u8);
        buf.push(self.tile_mode_y as u8);

        // Sampling options (2 bytes: filter + mipmap)
        buf.push(self.sampling.filter as u8);
        buf.push(self.sampling.mipmap as u8);

        // Local matrix (1 byte present flag + 36 bytes if present)
        if let Some(matrix) = &self.local_matrix {
            buf.push(1);
            for i in 0..9 {
                buf.extend_from_slice(&matrix.values[i].to_le_bytes());
            }
        } else {
            buf.push(0);
        }

        // Image info and pixel data (if present)
        if let (Some(pixels), Some(info)) = (&self.pixels, &self.image_info) {
            buf.push(1); // present

            // ImageInfo fields
            let img_w = u32::try_from(info.width()).unwrap_or(0);
            let img_h = u32::try_from(info.height()).unwrap_or(0);
            buf.extend_from_slice(&img_w.to_le_bytes());
            buf.extend_from_slice(&img_h.to_le_bytes());
            buf.push(info.color_type as u8);
            buf.push(info.alpha_type as u8);

            // Pixel data length + data
            let pixels_len = u32::try_from(pixels.len()).unwrap_or(u32::MAX);
            buf.extend_from_slice(&pixels_len.to_le_bytes());
            buf.extend_from_slice(pixels);
        } else {
            buf.push(0); // absent
        }

        Some(buf)
    }
}

/// Blend shader that combines two shaders.
///
/// Corresponds to Skia's `SkShaders::Blend`.
#[derive(Debug)]
pub struct BlendShader {
    blend_mode: crate::BlendMode,
    dst: ShaderRef,
    src: ShaderRef,
}

impl BlendShader {
    /// Create a new blend shader.
    pub fn new(blend_mode: crate::BlendMode, dst: ShaderRef, src: ShaderRef) -> Self {
        Self {
            blend_mode,
            dst,
            src,
        }
    }

    /// Get the blend mode.
    #[inline]
    #[must_use]
    pub const fn blend_mode(&self) -> crate::BlendMode {
        self.blend_mode
    }

    /// Get the destination shader.
    #[inline]
    #[must_use]
    pub fn dst(&self) -> &ShaderRef {
        &self.dst
    }

    /// Get the source shader.
    #[inline]
    #[must_use]
    pub fn src(&self) -> &ShaderRef {
        &self.src
    }
}

impl Shader for BlendShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        None
    }

    fn is_opaque(&self) -> bool {
        // Blend shader opacity depends on the blend mode and child shaders
        false
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::Blend
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        let src = self.src.sample(x, y);
        let dst = self.dst.sample(x, y);
        self.blend_mode.apply(src, dst)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_BLEND);

        // Blend mode (1 byte)
        buf.push(self.blend_mode as u8);

        // Src shader (recursive)
        serialize_child_shader(&mut buf, &self.src);

        // Dst shader (recursive)
        serialize_child_shader(&mut buf, &self.dst);

        Some(buf)
    }
}

/// Perlin noise generator based on Ken Perlin's classic algorithm.
///
/// Produces smoothly-varying pseudo-random values in the range roughly [-1, 1]
/// for noise, or [0, 1] for turbulence (which uses `abs()`). The classic Perlin
/// noise uses a 256-entry permutation table and interpolates gradients at
/// integer lattice points.
#[derive(Debug, Clone)]
struct PerlinNoiseGenerator {
    /// Permutation table seeded from the shader seed.
    perm: [u16; 512],
    /// Gradient lookup table. Each gradient is an 8-valued direction on a
    /// unit square (Skia style: simpler than classic Perlin gradients).
    grad_x: [f32; 256],
    grad_y: [f32; 256],
}

impl PerlinNoiseGenerator {
    fn new(seed: u32) -> Self {
        // Linear-congruential generator for reproducibility.
        let mut state = if seed == 0 { 0xdead_beef } else { seed };
        let mut rand_u8 = || -> u8 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) & 0xff) as u8
        };

        let mut perm_table = [0u16; 256];
        for (i, slot) in perm_table.iter_mut().enumerate() {
            *slot = u16::try_from(i).unwrap_or(u16::MAX);
        }
        // Fisher-Yates shuffle
        for i in (1..256).rev() {
            let j = (rand_u8() as usize) % (i + 1);
            perm_table.swap(i, j);
        }

        let mut perm = [0u16; 512];
        for i in 0..512 {
            perm[i] = perm_table[i & 255];
        }

        // Gradient table: unit vectors at 8 directions
        let mut grad_x = [0.0f32; 256];
        let mut grad_y = [0.0f32; 256];
        for (i, (gx, gy)) in grad_x.iter_mut().zip(grad_y.iter_mut()).enumerate() {
            let angle =
                scalar_from_i32(i32::try_from(i).unwrap_or(0)) * std::f32::consts::TAU / 256.0;
            *gx = angle.cos();
            *gy = angle.sin();
        }

        Self {
            perm,
            grad_x,
            grad_y,
        }
    }

    /// Evaluate classic 2D Perlin noise at (x, y). Returns approximately [-1, 1].
    fn noise_2d(&self, x: f32, y: f32) -> f32 {
        let xi = floor_to_i32(x);
        let yi = floor_to_i32(y);
        let xf = x - scalar_from_i32(xi);
        let yf = y - scalar_from_i32(yi);

        let u = fade(xf);
        let v = fade(yf);

        let xi0 = usize::try_from(xi & 255).unwrap_or(0);
        let yi0 = usize::try_from(yi & 255).unwrap_or(0);
        let xi1 = usize::try_from((xi + 1) & 255).unwrap_or(0);
        let yi1 = usize::try_from((yi + 1) & 255).unwrap_or(0);

        let g00 = self.perm[(self.perm[xi0] as usize + yi0) & 511] as usize;
        let g10 = self.perm[(self.perm[xi1] as usize + yi0) & 511] as usize;
        let g01 = self.perm[(self.perm[xi0] as usize + yi1) & 511] as usize;
        let g11 = self.perm[(self.perm[xi1] as usize + yi1) & 511] as usize;

        let d00 = self.grad_x[g00].mul_add(xf, self.grad_y[g00] * yf);
        let d10 = self.grad_x[g10].mul_add(xf - 1.0, self.grad_y[g10] * yf);
        let d01 = self.grad_x[g01].mul_add(xf, self.grad_y[g01] * (yf - 1.0));
        let d11 = self.grad_x[g11].mul_add(xf - 1.0, self.grad_y[g11] * (yf - 1.0));

        let ix0 = lerp(d00, d10, u);
        let ix1 = lerp(d01, d11, u);
        lerp(ix0, ix1, v)
    }

    /// Fractal (summed octaves with decreasing amplitude).
    fn fractal_2d(&self, x: f32, y: f32, octaves: u32) -> f32 {
        let mut total = 0.0_f32;
        let mut freq = 1.0_f32;
        let mut amp = 1.0_f32;
        let mut max_val = 0.0_f32;
        for _ in 0..octaves {
            total = self.noise_2d(x * freq, y * freq).mul_add(amp, total);
            max_val += amp;
            freq *= 2.0;
            amp *= 0.5;
        }
        if max_val > 0.0 { total / max_val } else { 0.0 }
    }

    /// Turbulence (absolute value of octave sum).
    fn turbulence_2d(&self, x: f32, y: f32, octaves: u32) -> f32 {
        let mut total = 0.0_f32;
        let mut freq = 1.0_f32;
        let mut amp = 1.0_f32;
        let mut max_val = 0.0_f32;
        for _ in 0..octaves {
            total = self.noise_2d(x * freq, y * freq).abs().mul_add(amp, total);
            max_val += amp;
            freq *= 2.0;
            amp *= 0.5;
        }
        if max_val > 0.0 {
            (total / max_val).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[inline]
fn fade(t: f32) -> f32 {
    // Ken Perlin's improved fade: 6t^5 - 15t^4 + 10t^3
    t * t * t * t.mul_add(t.mul_add(6.0, -15.0), 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

/// Perlin noise shader.
///
/// Corresponds to Skia's `SkPerlinNoiseShader`.
#[derive(Debug, Clone)]
pub struct PerlinNoiseShader {
    noise_type: NoiseType,
    base_frequency_x: Scalar,
    base_frequency_y: Scalar,
    num_octaves: i32,
    seed: Scalar,
    tile_size: Option<(i32, i32)>,
    #[cfg_attr(test, allow(dead_code))]
    generator: std::sync::OnceLock<PerlinNoiseGenerator>,
}

/// Type of Perlin noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Fractal noise (smoother).
    FractalNoise,
    /// Turbulence (more chaotic).
    Turbulence,
}

impl PerlinNoiseShader {
    /// Create a fractal noise shader.
    #[must_use]
    pub const fn fractal_noise(
        base_frequency_x: Scalar,
        base_frequency_y: Scalar,
        num_octaves: i32,
        seed: Scalar,
    ) -> Self {
        Self {
            noise_type: NoiseType::FractalNoise,
            base_frequency_x,
            base_frequency_y,
            num_octaves,
            seed,
            tile_size: None,
            generator: std::sync::OnceLock::new(),
        }
    }

    /// Create a turbulence shader.
    #[must_use]
    pub const fn turbulence(
        base_frequency_x: Scalar,
        base_frequency_y: Scalar,
        num_octaves: i32,
        seed: Scalar,
    ) -> Self {
        Self {
            noise_type: NoiseType::Turbulence,
            base_frequency_x,
            base_frequency_y,
            num_octaves,
            seed,
            tile_size: None,
            generator: std::sync::OnceLock::new(),
        }
    }

    /// Set the tile size for seamless tiling.
    #[must_use]
    pub const fn with_tile_size(mut self, width: i32, height: i32) -> Self {
        self.tile_size = Some((width, height));
        self
    }

    /// Get the noise type.
    #[inline]
    pub const fn noise_type(&self) -> NoiseType {
        self.noise_type
    }

    /// Get the base frequency X.
    #[inline]
    pub const fn base_frequency_x(&self) -> Scalar {
        self.base_frequency_x
    }

    /// Get the base frequency Y.
    #[inline]
    pub const fn base_frequency_y(&self) -> Scalar {
        self.base_frequency_y
    }

    /// Get the number of octaves.
    #[inline]
    pub const fn num_octaves(&self) -> i32 {
        self.num_octaves
    }

    /// Get the seed.
    #[inline]
    pub const fn seed(&self) -> Scalar {
        self.seed
    }
}

impl Shader for PerlinNoiseShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        None
    }

    fn is_opaque(&self) -> bool {
        true
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::PerlinNoise
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "seed is only used to deterministically initialize the noise LCG; the float bit pattern is saturated into u32, no core cast helper covers f32->u32 and the exact numeric value carries no meaning beyond reproducibility"
    )]
    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        let generator = self
            .generator
            .get_or_init(|| PerlinNoiseGenerator::new(self.seed as u32));
        let fx = x * self.base_frequency_x;
        let fy = y * self.base_frequency_y;
        let octaves = u32::try_from(self.num_octaves.max(1)).unwrap_or(1);

        let value = match self.noise_type {
            NoiseType::FractalNoise => {
                // Map fractal noise from [-1, 1] to [0, 1]
                generator
                    .fractal_2d(fx, fy, octaves)
                    .mul_add(0.5, 0.5)
                    .clamp(0.0, 1.0)
            }
            NoiseType::Turbulence => generator.turbulence_2d(fx, fy, octaves),
        };

        // Return grayscale with full alpha. Skia uses separate per-channel noise
        // in production, but grayscale is an acceptable baseline.
        Color4f::new(value, value, value, 1.0)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_PERLIN_NOISE);

        // Noise type (1 byte)
        buf.push(match self.noise_type {
            NoiseType::FractalNoise => 0,
            NoiseType::Turbulence => 1,
        });

        // Base frequencies (8 bytes)
        buf.extend_from_slice(&self.base_frequency_x.to_le_bytes());
        buf.extend_from_slice(&self.base_frequency_y.to_le_bytes());

        // Number of octaves (4 bytes)
        buf.extend_from_slice(&self.num_octaves.to_le_bytes());

        // Seed (4 bytes)
        buf.extend_from_slice(&self.seed.to_le_bytes());

        // Tile size (1 byte present flag + 8 bytes if present)
        if let Some((width, height)) = self.tile_size {
            buf.push(1);
            buf.extend_from_slice(&width.to_le_bytes());
            buf.extend_from_slice(&height.to_le_bytes());
        } else {
            buf.push(0);
        }

        Some(buf)
    }
}

/// A boxed shader type.
pub type ShaderRef = Arc<dyn Shader>;

/// Local matrix wrapper shader.
///
/// Wraps another shader with a local transformation matrix.
/// Corresponds to Skia's `SkLocalMatrixShader`.
#[derive(Debug)]
pub struct LocalMatrixShader {
    inner: ShaderRef,
    matrix: Matrix,
}

impl LocalMatrixShader {
    /// Create a new local matrix shader.
    pub fn new(inner: ShaderRef, matrix: Matrix) -> Self {
        Self { inner, matrix }
    }

    /// Get the inner shader.
    #[inline]
    #[must_use]
    pub fn inner(&self) -> &ShaderRef {
        &self.inner
    }

    /// Get the matrix.
    #[inline]
    #[must_use]
    pub const fn matrix(&self) -> &Matrix {
        &self.matrix
    }
}

impl Shader for LocalMatrixShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        Some(&self.matrix)
    }

    fn is_opaque(&self) -> bool {
        self.inner.is_opaque()
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::LocalMatrix
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        // Apply the inverse of the local matrix to map destination coords
        // back into the inner shader's coordinate space. If the matrix is
        // not invertible, fall back to sampling the inner shader at the
        // untransformed coordinates.
        let inv = self.matrix.invert();
        let (sx, sy) = inv.map_or((x, y), |m| {
            let p = m.map_point(Point::new(x, y));
            (p.x, p.y)
        });
        self.inner.sample(sx, sy)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_LOCAL_MATRIX);

        // Matrix (36 bytes)
        for i in 0..9 {
            buf.extend_from_slice(&self.matrix.values[i].to_le_bytes());
        }

        // Inner shader (recursive)
        serialize_child_shader(&mut buf, &self.inner);

        Some(buf)
    }
}

/// Compose shader that chains two shaders together.
///
/// The compose shader applies the inner shader, then the outer shader.
#[derive(Debug)]
pub struct ComposeShader {
    outer: ShaderRef,
    inner: ShaderRef,
    blend_mode: crate::BlendMode,
}

impl ComposeShader {
    /// Create a new compose shader.
    pub fn new(outer: ShaderRef, inner: ShaderRef, blend_mode: crate::BlendMode) -> Self {
        Self {
            outer,
            inner,
            blend_mode,
        }
    }

    /// Get the outer shader.
    #[inline]
    #[must_use]
    pub fn outer(&self) -> &ShaderRef {
        &self.outer
    }

    /// Get the inner shader.
    #[inline]
    #[must_use]
    pub fn inner(&self) -> &ShaderRef {
        &self.inner
    }

    /// Get the blend mode.
    #[inline]
    #[must_use]
    pub const fn blend_mode(&self) -> crate::BlendMode {
        self.blend_mode
    }
}

impl Shader for ComposeShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        None
    }

    fn is_opaque(&self) -> bool {
        self.outer.is_opaque() && self.inner.is_opaque()
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::Compose
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        let dst = self.inner.sample(x, y);
        let src = self.outer.sample(x, y);
        self.blend_mode.apply(src, dst)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(SHADER_KIND_COMPOSE);

        // Blend mode (1 byte)
        buf.push(self.blend_mode as u8);

        // Outer shader (recursive)
        serialize_child_shader(&mut buf, &self.outer);

        // Inner shader (recursive)
        serialize_child_shader(&mut buf, &self.inner);

        Some(buf)
    }
}

/// Empty shader that produces transparent pixels.
#[derive(Debug, Clone, Default)]
pub struct EmptyShader;

impl EmptyShader {
    /// Create an empty shader.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Shader for EmptyShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        None
    }

    fn is_opaque(&self) -> bool {
        false
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::Empty
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        Some(vec![SHADER_KIND_EMPTY])
    }
}

// =============================================================================
// Deserialization
// =============================================================================

/// Deserialize a shader from bytes.
///
/// Returns `None` if the data is invalid or the shader type is unsupported.
pub(crate) fn deserialize_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    if *offset >= bytes.len() {
        return None;
    }
    let kind = bytes[*offset];
    *offset += 1;

    match kind {
        SHADER_KIND_COLOR => deserialize_color_shader(bytes, offset),
        SHADER_KIND_LINEAR_GRADIENT => deserialize_linear_gradient(bytes, offset),
        SHADER_KIND_RADIAL_GRADIENT => deserialize_radial_gradient(bytes, offset),
        SHADER_KIND_SWEEP_GRADIENT => deserialize_sweep_gradient(bytes, offset),
        SHADER_KIND_TWO_POINT_CONICAL => deserialize_two_point_conical(bytes, offset),
        SHADER_KIND_IMAGE => deserialize_image_shader(bytes, offset),
        SHADER_KIND_PERLIN_NOISE => deserialize_perlin_noise(bytes, offset),
        SHADER_KIND_BLEND => deserialize_blend_shader(bytes, offset),
        SHADER_KIND_LOCAL_MATRIX => deserialize_local_matrix_shader(bytes, offset),
        SHADER_KIND_COMPOSE => deserialize_compose_shader(bytes, offset),
        SHADER_KIND_EMPTY => Some(Arc::new(EmptyShader::new())),
        _ => None, // Unknown shader kind
    }
}

/// Helper to read a child shader (recursively).
fn deserialize_child_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    if *offset >= bytes.len() {
        return None;
    }
    let present = bytes[*offset];
    *offset += 1;

    if present == 0 {
        return None;
    }

    // Read length
    if *offset + 4 > bytes.len() {
        return None;
    }
    let len = u32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]) as usize;
    *offset += 4;

    if *offset + len > bytes.len() {
        return None;
    }

    let child_bytes = &bytes[*offset..*offset + len];
    *offset += len;

    let mut child_offset = 0;
    deserialize_shader(child_bytes, &mut child_offset)
}

/// Helper to deserialize common gradient data.
/// Colors, optional explicit positions, tile mode, flags, and optional
/// local matrix, as decoded from common gradient serialization data.
type GradientCommonData = (
    Vec<Color4f>,
    Option<Vec<Scalar>>,
    TileMode,
    GradientFlags,
    Option<Matrix>,
);

#[allow(
    clippy::too_many_lines,
    reason = "sequentially decodes each serialized gradient field in the same order serialize_gradient_common wrote them; splitting would obscure that pairing"
)]
fn deserialize_gradient_common(bytes: &[u8], offset: &mut usize) -> Option<GradientCommonData> {
    // Tile mode (1 byte)
    if *offset >= bytes.len() {
        return None;
    }
    let tile_mode = match bytes[*offset] {
        0 => TileMode::Clamp,
        1 => TileMode::Repeat,
        2 => TileMode::Mirror,
        3 => TileMode::Decal,
        _ => return None,
    };
    *offset += 1;

    // Flags (4 bytes)
    if *offset + 4 > bytes.len() {
        return None;
    }
    let flags = GradientFlags::from_bits_truncate(u32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]));
    *offset += 4;

    // Colors count (4 bytes)
    if *offset + 4 > bytes.len() {
        return None;
    }
    let color_count = u32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]) as usize;
    *offset += 4;

    // Colors (16 bytes each)
    if *offset + color_count * 16 > bytes.len() {
        return None;
    }
    let mut colors = Vec::with_capacity(color_count);
    for _ in 0..color_count {
        let r = f32::from_le_bytes([
            bytes[*offset],
            bytes[*offset + 1],
            bytes[*offset + 2],
            bytes[*offset + 3],
        ]);
        let g = f32::from_le_bytes([
            bytes[*offset + 4],
            bytes[*offset + 5],
            bytes[*offset + 6],
            bytes[*offset + 7],
        ]);
        let b = f32::from_le_bytes([
            bytes[*offset + 8],
            bytes[*offset + 9],
            bytes[*offset + 10],
            bytes[*offset + 11],
        ]);
        let a = f32::from_le_bytes([
            bytes[*offset + 12],
            bytes[*offset + 13],
            bytes[*offset + 14],
            bytes[*offset + 15],
        ]);
        colors.push(Color4f::new(r, g, b, a));
        *offset += 16;
    }

    // Positions (1 byte present flag + data)
    if *offset >= bytes.len() {
        return None;
    }
    let positions_present = bytes[*offset];
    *offset += 1;
    let positions = if positions_present == 1 {
        if *offset + color_count * 4 > bytes.len() {
            return None;
        }
        let mut pos_vec = Vec::with_capacity(color_count);
        for _ in 0..color_count {
            let pos = f32::from_le_bytes([
                bytes[*offset],
                bytes[*offset + 1],
                bytes[*offset + 2],
                bytes[*offset + 3],
            ]);
            pos_vec.push(pos);
            *offset += 4;
        }
        Some(pos_vec)
    } else {
        None
    };

    // Local matrix (1 byte present flag + 36 bytes)
    if *offset >= bytes.len() {
        return None;
    }
    let matrix_present = bytes[*offset];
    *offset += 1;
    let local_matrix = if matrix_present == 1 {
        if *offset + 36 > bytes.len() {
            return None;
        }
        let mut m = Matrix::identity();
        for i in 0..9 {
            m.values[i] = f32::from_le_bytes([
                bytes[*offset],
                bytes[*offset + 1],
                bytes[*offset + 2],
                bytes[*offset + 3],
            ]);
            *offset += 4;
        }
        Some(m)
    } else {
        None
    };

    Some((colors, positions, tile_mode, flags, local_matrix))
}

fn deserialize_color_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    if *offset + 16 > bytes.len() {
        return None;
    }
    let r = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let g = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    let b = f32::from_le_bytes([
        bytes[*offset + 8],
        bytes[*offset + 9],
        bytes[*offset + 10],
        bytes[*offset + 11],
    ]);
    let a = f32::from_le_bytes([
        bytes[*offset + 12],
        bytes[*offset + 13],
        bytes[*offset + 14],
        bytes[*offset + 15],
    ]);
    *offset += 16;
    Some(Arc::new(ColorShader::new(Color4f::new(r, g, b, a))))
}

fn deserialize_linear_gradient(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Start point (8 bytes)
    if *offset + 8 > bytes.len() {
        return None;
    }
    let start_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let start_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;

    // End point (8 bytes)
    if *offset + 8 > bytes.len() {
        return None;
    }
    let end_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let end_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;

    // Common gradient data
    let (colors, positions, tile_mode, flags, local_matrix) =
        deserialize_gradient_common(bytes, offset)?;

    let mut gradient = LinearGradient::new(
        Point::new(start_x, start_y),
        Point::new(end_x, end_y),
        colors,
        positions,
        tile_mode,
    )
    .with_flags(flags);

    if let Some(matrix) = local_matrix {
        gradient = gradient.with_local_matrix(matrix);
    }

    Some(Arc::new(gradient))
}

fn deserialize_radial_gradient(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Center point (8 bytes)
    if *offset + 8 > bytes.len() {
        return None;
    }
    let center_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let center_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;

    // Radius (4 bytes)
    if *offset + 4 > bytes.len() {
        return None;
    }
    let radius = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    *offset += 4;

    // Common gradient data
    let (colors, positions, tile_mode, flags, local_matrix) =
        deserialize_gradient_common(bytes, offset)?;

    let mut gradient = RadialGradient::new(
        Point::new(center_x, center_y),
        radius,
        colors,
        positions,
        tile_mode,
    )
    .with_flags(flags);

    if let Some(matrix) = local_matrix {
        gradient = gradient.with_local_matrix(matrix);
    }

    Some(Arc::new(gradient))
}

fn deserialize_sweep_gradient(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Center point (8 bytes)
    if *offset + 8 > bytes.len() {
        return None;
    }
    let center_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let center_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;

    // Start and end angles (8 bytes)
    if *offset + 8 > bytes.len() {
        return None;
    }
    let start_angle = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let end_angle = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;

    // Common gradient data
    let (colors, positions, tile_mode, flags, local_matrix) =
        deserialize_gradient_common(bytes, offset)?;

    let mut gradient = SweepGradient::new(
        Point::new(center_x, center_y),
        start_angle,
        end_angle,
        colors,
        positions,
        tile_mode,
    )
    .with_flags(flags);

    if let Some(matrix) = local_matrix {
        gradient = gradient.with_local_matrix(matrix);
    }

    Some(Arc::new(gradient))
}

fn deserialize_two_point_conical(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Start center and radius (12 bytes)
    if *offset + 12 > bytes.len() {
        return None;
    }
    let start_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let start_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    let start_radius = f32::from_le_bytes([
        bytes[*offset + 8],
        bytes[*offset + 9],
        bytes[*offset + 10],
        bytes[*offset + 11],
    ]);
    *offset += 12;

    // End center and radius (12 bytes)
    if *offset + 12 > bytes.len() {
        return None;
    }
    let end_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let end_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    let end_radius = f32::from_le_bytes([
        bytes[*offset + 8],
        bytes[*offset + 9],
        bytes[*offset + 10],
        bytes[*offset + 11],
    ]);
    *offset += 12;

    // Common gradient data
    let (colors, positions, tile_mode, flags, local_matrix) =
        deserialize_gradient_common(bytes, offset)?;

    let mut gradient = TwoPointConicalGradient::new(
        Point::new(start_x, start_y),
        start_radius,
        Point::new(end_x, end_y),
        end_radius,
        colors,
        positions,
        tile_mode,
    )
    .with_flags(flags);

    if let Some(matrix) = local_matrix {
        gradient = gradient.with_local_matrix(matrix);
    }

    Some(Arc::new(gradient))
}

#[allow(
    clippy::too_many_lines,
    reason = "sequentially decodes each serialized ImageShader field in wire order; splitting would obscure that pairing with serialize()"
)]
fn deserialize_image_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Bounds (16 bytes)
    if *offset + 16 > bytes.len() {
        return None;
    }
    let left = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let top = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    let right = f32::from_le_bytes([
        bytes[*offset + 8],
        bytes[*offset + 9],
        bytes[*offset + 10],
        bytes[*offset + 11],
    ]);
    let bottom = f32::from_le_bytes([
        bytes[*offset + 12],
        bytes[*offset + 13],
        bytes[*offset + 14],
        bytes[*offset + 15],
    ]);
    *offset += 16;

    // Tile modes (2 bytes)
    if *offset + 2 > bytes.len() {
        return None;
    }
    let tile_mode_x = match bytes[*offset] {
        0 => TileMode::Clamp,
        1 => TileMode::Repeat,
        2 => TileMode::Mirror,
        3 => TileMode::Decal,
        _ => return None,
    };
    let tile_mode_y = match bytes[*offset + 1] {
        0 => TileMode::Clamp,
        1 => TileMode::Repeat,
        2 => TileMode::Mirror,
        3 => TileMode::Decal,
        _ => return None,
    };
    *offset += 2;

    // Sampling options (2 bytes)
    if *offset + 2 > bytes.len() {
        return None;
    }
    let filter = match bytes[*offset] {
        0 => FilterMode::Nearest,
        1 => FilterMode::Linear,
        _ => return None,
    };
    let mipmap = match bytes[*offset + 1] {
        0 => MipmapMode::None,
        1 => MipmapMode::Nearest,
        2 => MipmapMode::Linear,
        _ => return None,
    };
    *offset += 2;

    // Local matrix (1 byte present flag + 36 bytes)
    if *offset >= bytes.len() {
        return None;
    }
    let matrix_present = bytes[*offset];
    *offset += 1;
    let local_matrix = if matrix_present == 1 {
        if *offset + 36 > bytes.len() {
            return None;
        }
        let mut m = Matrix::identity();
        for i in 0..9 {
            m.values[i] = f32::from_le_bytes([
                bytes[*offset],
                bytes[*offset + 1],
                bytes[*offset + 2],
                bytes[*offset + 3],
            ]);
            *offset += 4;
        }
        Some(m)
    } else {
        None
    };

    // Image info and pixel data (if present)
    if *offset >= bytes.len() {
        return None;
    }
    let image_present = bytes[*offset];
    *offset += 1;

    let (pixels, image_info) = if image_present == 1 {
        // ImageInfo fields
        if *offset + 6 > bytes.len() {
            return None;
        }
        let width = i32::from_le_bytes([
            bytes[*offset],
            bytes[*offset + 1],
            bytes[*offset + 2],
            bytes[*offset + 3],
        ]);
        let height = i32::from_le_bytes([
            bytes[*offset + 4],
            bytes[*offset + 5],
            bytes[*offset + 6],
            bytes[*offset + 7],
        ]);
        let color_type = match bytes[*offset + 8] {
            0 => skia_rs_core::color::ColorType::Unknown,
            1 => skia_rs_core::color::ColorType::Alpha8,
            2 => skia_rs_core::color::ColorType::Rgb565,
            3 => skia_rs_core::color::ColorType::Argb4444,
            4 => skia_rs_core::color::ColorType::Rgba8888,
            5 => skia_rs_core::color::ColorType::Rgb888x,
            6 => skia_rs_core::color::ColorType::Bgra8888,
            7 => skia_rs_core::color::ColorType::Rgba1010102,
            8 => skia_rs_core::color::ColorType::Bgra1010102,
            9 => skia_rs_core::color::ColorType::Rgb101010x,
            10 => skia_rs_core::color::ColorType::Bgr101010x,
            11 => skia_rs_core::color::ColorType::Gray8,
            12 => skia_rs_core::color::ColorType::RgbaF16,
            13 => skia_rs_core::color::ColorType::RgbaF16Norm,
            14 => skia_rs_core::color::ColorType::RgbaF32,
            15 => skia_rs_core::color::ColorType::R8Unorm,
            16 => skia_rs_core::color::ColorType::A16Float,
            17 => skia_rs_core::color::ColorType::R16G16Float,
            18 => skia_rs_core::color::ColorType::A16Unorm,
            19 => skia_rs_core::color::ColorType::R16G16Unorm,
            20 => skia_rs_core::color::ColorType::R16G16B16A16Unorm,
            21 => skia_rs_core::color::ColorType::Srgba8888,
            22 => skia_rs_core::color::ColorType::R8Unorm2,
            _ => return None,
        };
        let alpha_type = match bytes[*offset + 9] {
            0 => skia_rs_core::color::AlphaType::Unknown,
            1 => skia_rs_core::color::AlphaType::Opaque,
            2 => skia_rs_core::color::AlphaType::Premul,
            3 => skia_rs_core::color::AlphaType::Unpremul,
            _ => return None,
        };
        *offset += 10;

        // Pixel data length + data
        if *offset + 4 > bytes.len() {
            return None;
        }
        let pixel_len = u32::from_le_bytes([
            bytes[*offset],
            bytes[*offset + 1],
            bytes[*offset + 2],
            bytes[*offset + 3],
        ]) as usize;
        *offset += 4;

        if *offset + pixel_len > bytes.len() {
            return None;
        }
        let pixel_data = bytes[*offset..*offset + pixel_len].to_vec();
        *offset += pixel_len;

        let info =
            skia_rs_core::pixel::ImageInfo::new(width, height, color_type, alpha_type).ok()?;

        (Some(Arc::new(pixel_data)), Some(info))
    } else {
        (None, None)
    };

    Some(Arc::new(ImageShader {
        bounds: Rect::from_xywh(left, top, right - left, bottom - top),
        tile_mode_x,
        tile_mode_y,
        sampling: SamplingOptions { filter, mipmap },
        local_matrix,
        pixels,
        image_info,
    }))
}

fn deserialize_perlin_noise(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Noise type (1 byte)
    if *offset >= bytes.len() {
        return None;
    }
    let noise_type = match bytes[*offset] {
        0 => NoiseType::FractalNoise,
        1 => NoiseType::Turbulence,
        _ => return None,
    };
    *offset += 1;

    // Base frequencies (8 bytes)
    if *offset + 8 > bytes.len() {
        return None;
    }
    let base_frequency_x = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    let base_frequency_y = f32::from_le_bytes([
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;

    // Number of octaves (4 bytes)
    if *offset + 4 > bytes.len() {
        return None;
    }
    let num_octaves = i32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    *offset += 4;

    // Seed (4 bytes)
    if *offset + 4 > bytes.len() {
        return None;
    }
    let seed = f32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    *offset += 4;

    // Tile size (1 byte present flag + 8 bytes)
    if *offset >= bytes.len() {
        return None;
    }
    let tile_present = bytes[*offset];
    *offset += 1;
    let tile_size = if tile_present == 1 {
        if *offset + 8 > bytes.len() {
            return None;
        }
        let width = i32::from_le_bytes([
            bytes[*offset],
            bytes[*offset + 1],
            bytes[*offset + 2],
            bytes[*offset + 3],
        ]);
        let height = i32::from_le_bytes([
            bytes[*offset + 4],
            bytes[*offset + 5],
            bytes[*offset + 6],
            bytes[*offset + 7],
        ]);
        *offset += 8;
        Some((width, height))
    } else {
        None
    };

    let mut shader = PerlinNoiseShader {
        noise_type,
        base_frequency_x,
        base_frequency_y,
        num_octaves,
        seed,
        tile_size: None,
        generator: std::sync::OnceLock::new(),
    };

    if let Some((w, h)) = tile_size {
        shader = shader.with_tile_size(w, h);
    }

    Some(Arc::new(shader))
}

fn deserialize_blend_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Blend mode (1 byte)
    if *offset >= bytes.len() {
        return None;
    }
    let blend_mode = crate::BlendMode::from_u8(bytes[*offset])?;
    *offset += 1;

    // Src shader (recursive)
    let src = deserialize_child_shader(bytes, offset)?;

    // Dst shader (recursive)
    let dst = deserialize_child_shader(bytes, offset)?;

    Some(Arc::new(BlendShader::new(blend_mode, dst, src)))
}

fn deserialize_local_matrix_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Matrix (36 bytes)
    if *offset + 36 > bytes.len() {
        return None;
    }
    let mut matrix = Matrix::identity();
    for i in 0..9 {
        matrix.values[i] = f32::from_le_bytes([
            bytes[*offset],
            bytes[*offset + 1],
            bytes[*offset + 2],
            bytes[*offset + 3],
        ]);
        *offset += 4;
    }

    // Inner shader (recursive)
    let inner = deserialize_child_shader(bytes, offset)?;

    Some(Arc::new(LocalMatrixShader::new(inner, matrix)))
}

fn deserialize_compose_shader(bytes: &[u8], offset: &mut usize) -> Option<ShaderRef> {
    // Blend mode (1 byte)
    if *offset >= bytes.len() {
        return None;
    }
    let blend_mode = crate::BlendMode::from_u8(bytes[*offset])?;
    *offset += 1;

    // Outer shader (recursive)
    let outer = deserialize_child_shader(bytes, offset)?;

    // Inner shader (recursive)
    let inner = deserialize_child_shader(bytes, offset)?;

    Some(Arc::new(ComposeShader::new(outer, inner, blend_mode)))
}

/// Convenience functions for creating shaders.
pub mod shaders {
    use super::{
        Arc, BlendShader, Color4f, ColorShader, ComposeShader, EmptyShader, LinearGradient,
        LocalMatrixShader, Matrix, PerlinNoiseShader, Point, RadialGradient, Scalar, ShaderRef,
        SweepGradient, TileMode, TwoPointConicalGradient,
    };

    /// Create a solid color shader.
    #[must_use]
    pub fn color(color: Color4f) -> ShaderRef {
        Arc::new(ColorShader::new(color))
    }

    /// Create a linear gradient shader.
    #[must_use]
    pub fn linear_gradient(
        start: Point,
        end: Point,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> ShaderRef {
        Arc::new(LinearGradient::new(
            start, end, colors, positions, tile_mode,
        ))
    }

    /// Create a radial gradient shader.
    #[must_use]
    pub fn radial_gradient(
        center: Point,
        radius: Scalar,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> ShaderRef {
        Arc::new(RadialGradient::new(
            center, radius, colors, positions, tile_mode,
        ))
    }

    /// Create a sweep gradient shader.
    #[must_use]
    pub fn sweep_gradient(
        center: Point,
        start_angle: Scalar,
        end_angle: Scalar,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> ShaderRef {
        Arc::new(SweepGradient::new(
            center,
            start_angle,
            end_angle,
            colors,
            positions,
            tile_mode,
        ))
    }

    /// Create a two-point conical gradient shader.
    #[must_use]
    pub fn two_point_conical_gradient(
        start_center: Point,
        start_radius: Scalar,
        end_center: Point,
        end_radius: Scalar,
        colors: Vec<Color4f>,
        positions: Option<Vec<Scalar>>,
        tile_mode: TileMode,
    ) -> ShaderRef {
        Arc::new(TwoPointConicalGradient::new(
            start_center,
            start_radius,
            end_center,
            end_radius,
            colors,
            positions,
            tile_mode,
        ))
    }

    /// Create a blend shader.
    pub fn blend(blend_mode: crate::BlendMode, dst: ShaderRef, src: ShaderRef) -> ShaderRef {
        Arc::new(BlendShader::new(blend_mode, dst, src))
    }

    /// Create a fractal noise shader.
    #[must_use]
    pub fn fractal_noise(
        base_frequency_x: Scalar,
        base_frequency_y: Scalar,
        num_octaves: i32,
        seed: Scalar,
    ) -> ShaderRef {
        Arc::new(PerlinNoiseShader::fractal_noise(
            base_frequency_x,
            base_frequency_y,
            num_octaves,
            seed,
        ))
    }

    /// Create a turbulence shader.
    #[must_use]
    pub fn turbulence(
        base_frequency_x: Scalar,
        base_frequency_y: Scalar,
        num_octaves: i32,
        seed: Scalar,
    ) -> ShaderRef {
        Arc::new(PerlinNoiseShader::turbulence(
            base_frequency_x,
            base_frequency_y,
            num_octaves,
            seed,
        ))
    }

    /// Wrap a shader with a local matrix transformation.
    pub fn with_local_matrix(shader: ShaderRef, matrix: Matrix) -> ShaderRef {
        Arc::new(LocalMatrixShader::new(shader, matrix))
    }

    /// Compose two shaders together.
    pub fn compose(outer: ShaderRef, inner: ShaderRef, blend_mode: crate::BlendMode) -> ShaderRef {
        Arc::new(ComposeShader::new(outer, inner, blend_mode))
    }

    /// Create an empty (transparent) shader.
    #[must_use]
    pub fn empty() -> ShaderRef {
        Arc::new(EmptyShader::new())
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert exact expected values (e.g. transparent alpha == 0.0), not tolerance comparisons"
)]
mod tests {
    use super::*;

    #[test]
    fn test_color_shader() {
        let shader = ColorShader::new(Color4f::new(1.0, 0.0, 0.0, 1.0));
        assert!(shader.is_opaque());
        assert_eq!(shader.shader_kind(), ShaderKind::Color);
    }

    #[test]
    fn test_linear_gradient() {
        let colors = vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ];
        let shader = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            colors,
            None,
            TileMode::Clamp,
        );
        assert!(shader.is_opaque());
        assert_eq!(shader.shader_kind(), ShaderKind::LinearGradient);
    }

    #[test]
    fn test_gradient_with_transparency() {
        let colors = vec![
            Color4f::new(1.0, 0.0, 0.0, 0.5),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ];
        let shader = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            colors,
            None,
            TileMode::Clamp,
        );
        assert!(!shader.is_opaque());
    }

    #[test]
    fn test_shader_convenience_functions() {
        let color = shaders::color(Color4f::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(color.shader_kind(), ShaderKind::Color);

        let linear = shaders::linear_gradient(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            vec![Color4f::new(1.0, 0.0, 0.0, 1.0)],
            None,
            TileMode::Clamp,
        );
        assert_eq!(linear.shader_kind(), ShaderKind::LinearGradient);
    }

    #[test]
    fn test_local_matrix_shader_translates() {
        let base: Arc<dyn Shader> = Arc::new(ColorShader::new(Color4f::new(1.0, 0.0, 0.0, 1.0)));
        // LocalMatrixShader with translate(50, 0) should still return red
        // regardless of query point (ColorShader ignores coords).
        let matrix = Matrix::translate(50.0, 0.0);
        let wrapped = LocalMatrixShader::new(base, matrix);
        let c = wrapped.sample(0.0, 0.0);
        assert!((c.r - 1.0).abs() < 1e-5);
        assert!((c.g - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_shader_uses_blend_mode() {
        let red: Arc<dyn Shader> = Arc::new(ColorShader::new(Color4f::new(1.0, 0.0, 0.0, 1.0)));
        let green: Arc<dyn Shader> = Arc::new(ColorShader::new(Color4f::new(0.0, 1.0, 0.0, 1.0)));
        let blend = BlendShader::new(crate::BlendMode::Src, red.clone(), green.clone());
        let c = blend.sample(0.0, 0.0);
        // Src mode returns src; with BlendShader(mode, dst, src), src is green
        assert!((c.g - 1.0).abs() < 1e-5, "Expected green component = 1.0");
        assert!((c.r - 0.0).abs() < 1e-5, "Expected red component = 0.0");
    }

    #[test]
    fn test_compose_shader_blends_children() {
        let solid: Arc<dyn Shader> = Arc::new(ColorShader::new(Color4f::new(1.0, 1.0, 1.0, 1.0)));
        let half: Arc<dyn Shader> = Arc::new(ColorShader::new(Color4f::new(0.5, 0.5, 0.5, 1.0)));
        // ComposeShader(outer, inner, mode) blends them
        let compose = ComposeShader::new(solid.clone(), half.clone(), crate::BlendMode::SrcOver);
        let c = compose.sample(0.0, 0.0);
        assert!(
            c.a > 0.5,
            "ComposeShader should not produce transparent output"
        );
    }

    #[test]
    fn test_two_point_conical_gradient_interpolates() {
        // Two circles: inner at origin (r=10), outer at (100, 0) (r=30).
        // Interpolate red -> blue.
        let gradient = TwoPointConicalGradient::new(
            Point::new(0.0, 0.0),
            10.0,
            Point::new(100.0, 0.0),
            30.0,
            vec![
                Color4f::new(1.0, 0.0, 0.0, 1.0), // red at t=0 (inner circle)
                Color4f::new(0.0, 0.0, 1.0, 1.0), // blue at t=1 (outer circle)
            ],
            None, // positions default
            TileMode::Clamp,
        );

        // The point (10, 0) lies on two interpolated circles: t=0 (the inner
        // circle) and t=0.25 (center (25,0), radius 15). Per the gradient
        // spec (and Skia), the LARGEST t with r(t) >= 0 wins, so the color is
        // lerp(red, blue, 0.25).
        let c_inner = gradient.sample(10.0, 0.0);
        assert!(
            (c_inner.r - 0.75).abs() < 1e-3,
            "point on both cones takes the larger t (0.25), got r={}",
            c_inner.r
        );

        // A point on the outer circle (at end_center + end_radius in x direction)
        // should be blue (t=1).
        let c_outer = gradient.sample(130.0, 0.0);
        assert!(
            c_outer.b > 0.9,
            "outer circle point should be blue, got b={}",
            c_outer.b
        );

        // Midpoint should show a mix (purple-ish)
        let c_mid = gradient.sample(50.0, 0.0);
        assert!(
            c_mid.r > 0.1 && c_mid.b > 0.1,
            "midpoint should mix red and blue, got r={} b={}",
            c_mid.r,
            c_mid.b
        );
    }

    #[test]
    fn test_two_point_conical_gradient_outside_returns_clamped() {
        // Outside both circles with TileMode::Clamp should clamp to edge color
        let gradient = TwoPointConicalGradient::new(
            Point::new(0.0, 0.0),
            10.0,
            Point::new(50.0, 0.0),
            20.0,
            vec![
                Color4f::new(1.0, 0.0, 0.0, 1.0),
                Color4f::new(0.0, 1.0, 0.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );

        // Far off-axis
        let c = gradient.sample(200.0, 200.0);
        // Should be bounded, not NaN
        assert!(c.r.is_finite() && c.g.is_finite() && c.b.is_finite() && c.a.is_finite());
    }

    #[test]
    fn test_perlin_fractal_noise_in_range() {
        let shader = PerlinNoiseShader::fractal_noise(0.1, 0.1, 4, 42.0);
        for x in 0..10 {
            for y in 0..10 {
                let c = shader.sample(scalar_from_i32(x), scalar_from_i32(y));
                assert!(c.r >= 0.0 && c.r <= 1.0, "r out of range: {}", c.r);
                assert!(c.a >= 0.99, "alpha should be 1: {}", c.a);
            }
        }
    }

    #[test]
    fn test_perlin_turbulence_in_range() {
        let shader = PerlinNoiseShader::turbulence(0.1, 0.1, 4, 42.0);
        for x in 0..10 {
            for y in 0..10 {
                let c = shader.sample(scalar_from_i32(x), scalar_from_i32(y));
                assert!(c.r >= 0.0 && c.r <= 1.0);
            }
        }
    }

    #[test]
    fn test_perlin_different_seeds_produce_different_noise() {
        let a = PerlinNoiseShader::fractal_noise(0.5, 0.5, 2, 1.0);
        let b = PerlinNoiseShader::fractal_noise(0.5, 0.5, 2, 999.0);
        // At least one of a few sample points should differ
        let mut differ = false;
        for k in 0..10 {
            let ra = a.sample(scalar_from_i32(k), scalar_from_i32(k)).r;
            let rb = b.sample(scalar_from_i32(k), scalar_from_i32(k)).r;
            if (ra - rb).abs() > 1e-4 {
                differ = true;
                break;
            }
        }
        assert!(differ, "Different seeds should produce different output");
    }

    #[test]
    fn test_perlin_deterministic_per_seed() {
        let shader = PerlinNoiseShader::fractal_noise(0.3, 0.3, 3, 12345.0);
        let c1 = shader.sample(10.0, 10.0);
        let c2 = shader.sample(10.0, 10.0);
        assert!((c1.r - c2.r).abs() < 1e-5);
    }

    #[test]
    fn test_image_shader_samples_rgba8_pixel() {
        use skia_rs_core::color::AlphaType;
        use skia_rs_core::color::ColorType;
        use skia_rs_core::pixel::ImageInfo;
        // 2x2 image: red, green, blue, white
        let pixels: Arc<Vec<u8>> = Arc::new(vec![
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 255, 255, // (1,1) white
        ]);
        let info = ImageInfo::new(2, 2, ColorType::Rgba8888, AlphaType::Premul).unwrap();
        let shader = ImageShader::with_pixels(
            pixels,
            info,
            TileMode::Clamp,
            TileMode::Clamp,
            SamplingOptions::default(),
        );

        // Sample exact pixel positions
        let c_00 = shader.sample(0.5, 0.5);
        assert!(
            (c_00.r - 1.0).abs() < 1e-3,
            "expected red at (0.5, 0.5), got {c_00:?}"
        );

        let c_11 = shader.sample(1.5, 1.5);
        assert!(
            (c_11.r - 1.0).abs() < 1e-3 && (c_11.g - 1.0).abs() < 1e-3,
            "expected white at (1.5, 1.5), got {c_11:?}"
        );
    }

    #[test]
    fn test_image_shader_without_pixels_returns_transparent() {
        // Using the existing constructor (no pixels), sample should return transparent
        let shader = ImageShader::new(
            Rect::from_xywh(0.0, 0.0, 100.0, 100.0),
            TileMode::Clamp,
            TileMode::Clamp,
            SamplingOptions::default(),
        );
        let c = shader.sample(50.0, 50.0);
        assert_eq!(c.a, 0.0, "expected transparent for shader without pixels");
    }

    #[test]
    fn test_image_shader_clamp_tile_mode() {
        use skia_rs_core::color::AlphaType;
        use skia_rs_core::color::ColorType;
        use skia_rs_core::pixel::ImageInfo;
        let pixels: Arc<Vec<u8>> = Arc::new(vec![0, 0, 0, 255, 255, 255, 255, 255]);
        let info = ImageInfo::new(2, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();
        let shader = ImageShader::with_pixels(
            pixels,
            info,
            TileMode::Clamp,
            TileMode::Clamp,
            SamplingOptions::default(),
        );

        // Sampling far outside should clamp to nearest edge pixel
        let c_right = shader.sample(100.0, 0.5);
        assert!(
            (c_right.r - 1.0).abs() < 1e-3,
            "clamp should return white from right edge, got {c_right:?}"
        );
    }

    #[test]
    fn test_image_shader_repeat_tile_mode() {
        use skia_rs_core::color::AlphaType;
        use skia_rs_core::color::ColorType;
        use skia_rs_core::pixel::ImageInfo;
        let pixels: Arc<Vec<u8>> = Arc::new(vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
        ]);
        let info = ImageInfo::new(2, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();
        let shader = ImageShader::with_pixels(
            pixels,
            info,
            TileMode::Repeat,
            TileMode::Repeat,
            SamplingOptions::default(),
        );

        // Sampling beyond width should wrap
        let c = shader.sample(2.5, 0.5); // Should wrap to 0.5
        assert!(
            (c.r - 1.0).abs() < 1e-3,
            "repeat should wrap to red pixel, got {c:?}"
        );
    }

    #[test]
    fn test_image_shader_decal_tile_mode() {
        use skia_rs_core::color::AlphaType;
        use skia_rs_core::color::ColorType;
        use skia_rs_core::pixel::ImageInfo;
        let pixels: Arc<Vec<u8>> = Arc::new(vec![255, 0, 0, 255]);
        let info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();
        let shader = ImageShader::with_pixels(
            pixels,
            info,
            TileMode::Decal,
            TileMode::Decal,
            SamplingOptions::default(),
        );

        // Inside bounds should return color
        let c_in = shader.sample(0.5, 0.5);
        assert!((c_in.r - 1.0).abs() < 1e-3, "inside should return red");

        // Outside bounds should return transparent
        let c_out = shader.sample(2.0, 0.5);
        assert_eq!(c_out.a, 0.0, "decal outside bounds should be transparent");
    }

    #[test]
    fn test_gradient_flags_interpolate_premul() {
        // Create a gradient from opaque red to transparent
        let colors = vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 0.0, 0.0),
        ];

        // Without premul flag
        let grad_normal = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            colors.clone(),
            None,
            TileMode::Clamp,
        );

        // With premul flag
        let grad_premul = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            colors,
            None,
            TileMode::Clamp,
        )
        .with_flags(GradientFlags::INTERPOLATE_PREMUL);

        // Sample at midpoint (t=0.5). sample() returns premultiplied color.
        let c_normal = grad_normal.sample(5.0, 0.0);
        let c_premul = grad_premul.sample(5.0, 0.0);

        // Without the flag, interpolation happens in unpremul space:
        // straight (0.5, 0, 0, 0.5), premultiplied on output -> r = 0.25.
        assert!(
            (c_normal.r - 0.25).abs() < 1e-4,
            "unpremul interpolation midpoint premul r, got {}",
            c_normal.r
        );
        assert!((c_normal.a - 0.5).abs() < 1e-4);

        // With the flag, stops are premultiplied before interpolation:
        // lerp((1,0,0,1), (0,0,0,0)) = (0.5, 0, 0, 0.5), already premul.
        assert!(
            (c_premul.r - 0.5).abs() < 1e-4,
            "premul interpolation midpoint r, got {}",
            c_premul.r
        );
        assert!((c_premul.a - 0.5).abs() < 1e-4);
    }

    // --- Conformance regression tests (Task 3) ---

    #[test]
    fn test_gradient_t_below_first_stop_returns_first_color() {
        // SkGradientBaseShader: pos[0] > 0 means an implicit stop at t=0
        // carrying the first color (fFirstStopIsImplicit).
        let colors = vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ];
        let grad = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            colors,
            Some(vec![0.5, 1.0]),
            TileMode::Clamp,
        );
        let c = grad.sample(25.0, 0.0); // t = 0.25 < first stop 0.5
        assert!(
            (c.r - 1.0).abs() < 1e-4 && c.b.abs() < 1e-4,
            "t below first explicit stop must return the first color, got {c:?}"
        );
    }

    #[test]
    fn test_gradient_positions_pinned_monotonic() {
        // Non-monotonic positions are pinned per SkTPin(pos[i], prev, 1):
        // [0.0, 0.8, 0.5] becomes [0.0, 0.8, 0.8].
        let colors = vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 1.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ];
        let grad = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            colors,
            Some(vec![0.0, 0.8, 0.5]),
            TileMode::Clamp,
        );
        // t = 0.9 is past the pinned last stop (0.8) -> last color.
        let c = grad.sample(90.0, 0.0);
        assert!(
            (c.b - 1.0).abs() < 1e-4,
            "t past pinned last stop must return last color, got {c:?}"
        );
        // t = 0.4 lies mid-segment [0.0, 0.8] -> lerp(red, green, 0.5).
        let c = grad.sample(40.0, 0.0);
        assert!(
            (c.r - 0.5).abs() < 1e-4 && (c.g - 0.5).abs() < 1e-4,
            "pinned positions should keep first segment intact, got {c:?}"
        );
    }

    #[test]
    fn test_gradient_position_above_one_pinned() {
        // Position 2.0 pins to 1.0.
        let colors = vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ];
        let grad = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            colors,
            Some(vec![0.0, 2.0]),
            TileMode::Clamp,
        );
        let c = grad.sample(50.0, 0.0); // t = 0.5 -> halfway
        assert!(
            (c.r - 0.5).abs() < 1e-4 && (c.b - 0.5).abs() < 1e-4,
            "position above 1 must be pinned to 1, got {c:?}"
        );
    }

    #[test]
    fn test_two_point_conical_picks_larger_root() {
        // C0 = (0,0) r0 = 0.1, C1 = (1,0) r1 = 0.2, P = (0.5, 0).
        // Quadratic roots: t = 2/3 and t = 4/11; both have r(t) >= 0.
        // Upstream well-behaved case takes the larger t (== +sqrt root).
        let grad = TwoPointConicalGradient::new(
            Point::new(0.0, 0.0),
            0.1,
            Point::new(1.0, 0.0),
            0.2,
            vec![
                Color4f::new(0.0, 0.0, 0.0, 1.0),
                Color4f::new(1.0, 1.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let c = grad.sample(0.5, 0.0);
        assert!(
            (c.r - 2.0 / 3.0).abs() < 1e-3,
            "conical must pick the larger valid root (t = 2/3), got r = {}",
            c.r
        );
    }

    #[test]
    fn test_two_point_conical_honors_premul_flag() {
        let grad = TwoPointConicalGradient::new(
            Point::new(0.0, 0.0),
            0.0,
            Point::new(10.0, 0.0),
            10.0,
            vec![
                Color4f::new(1.0, 0.0, 0.0, 1.0),
                Color4f::new(0.0, 0.0, 1.0, 0.0),
            ],
            None,
            TileMode::Clamp,
        )
        .with_flags(GradientFlags::INTERPOLATE_PREMUL);
        // Point (5, 0) -> t = 0.25 (focal gradient: |5 - 10t| = 10t).
        // Premul stops: (1,0,0,1) and (0,0,0,0); lerp at 0.25 gives
        // (0.75, 0, 0, 0.75). Blue must not leak in premul interpolation
        // (without the flag the straight-space lerp would leak b = 0.1875).
        let c = grad.sample(5.0, 0.0);
        assert!(
            (c.r - 0.75).abs() < 1e-3 && c.b.abs() < 1e-3 && (c.a - 0.75).abs() < 1e-3,
            "conical must interpolate premultiplied stops with the flag, got {c:?}"
        );
    }

    #[test]
    fn test_color_shader_sample_is_premultiplied() {
        let shader = ColorShader::new(Color4f::new(1.0, 0.5, 0.0, 0.5));
        let c = shader.sample(0.0, 0.0);
        assert!((c.r - 0.5).abs() < 1e-5, "premul r, got {}", c.r);
        assert!((c.g - 0.25).abs() < 1e-5, "premul g, got {}", c.g);
        assert!((c.a - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_blend_shader_children_are_premul_for_blend() {
        // src: red at 50% alpha; dst: opaque green. SrcOver on premul values:
        // r = 0.5, g = 1*(1-0.5) = 0.5, a = 1.
        let src: ShaderRef = Arc::new(ColorShader::new(Color4f::new(1.0, 0.0, 0.0, 0.5)));
        let dst: ShaderRef = Arc::new(ColorShader::new(Color4f::new(0.0, 1.0, 0.0, 1.0)));
        let blend = BlendShader::new(crate::BlendMode::SrcOver, dst, src);
        let c = blend.sample(0.0, 0.0);
        assert!((c.r - 0.5).abs() < 1e-4, "premul srcover r, got {}", c.r);
        assert!((c.g - 0.5).abs() < 1e-4, "premul srcover g, got {}", c.g);
        assert!((c.a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compose_shader_children_are_premul_for_blend() {
        let outer: ShaderRef = Arc::new(ColorShader::new(Color4f::new(1.0, 0.0, 0.0, 0.5)));
        let inner: ShaderRef = Arc::new(ColorShader::new(Color4f::new(0.0, 1.0, 0.0, 1.0)));
        let compose = ComposeShader::new(outer, inner, crate::BlendMode::SrcOver);
        let c = compose.sample(0.0, 0.0);
        assert!((c.r - 0.5).abs() < 1e-4, "premul srcover r, got {}", c.r);
        assert!((c.g - 0.5).abs() < 1e-4, "premul srcover g, got {}", c.g);
    }

    #[test]
    fn test_alpha8_samples_as_black_with_alpha() {
        // Alpha-only sources decode as (0,0,0,a) — black, not white.
        use skia_rs_core::color::{AlphaType, ColorType};
        use skia_rs_core::pixel::ImageInfo;

        let info = ImageInfo::new(1, 1, ColorType::Alpha8, AlphaType::Premul).unwrap();
        let pixels = Arc::new(vec![128u8]);
        let shader = ImageShader::with_pixels(
            pixels,
            info,
            TileMode::Clamp,
            TileMode::Clamp,
            SamplingOptions::NEAREST,
        );
        let c = shader.sample(0.5, 0.5);
        assert!(c.r.abs() < 1e-4, "alpha8 r must be 0, got {}", c.r);
        assert!(c.g.abs() < 1e-4, "alpha8 g must be 0, got {}", c.g);
        assert!(c.b.abs() < 1e-4, "alpha8 b must be 0, got {}", c.b);
        assert!((c.a - 128.0 / 255.0).abs() < 1e-4, "alpha8 a, got {}", c.a);
    }

    #[test]
    fn test_image_shader_bilinear_filtering() {
        use skia_rs_core::color::{AlphaType, ColorType};
        use skia_rs_core::pixel::ImageInfo;

        // 2x1 image: black texel then white texel.
        let info = ImageInfo::new(2, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();
        let pixels = Arc::new(vec![
            0u8, 0, 0, 255, // black
            255, 255, 255, 255, // white
        ]);
        let shader = ImageShader::with_pixels(
            pixels,
            info,
            TileMode::Clamp,
            TileMode::Clamp,
            SamplingOptions::LINEAR,
        );
        // x = 1.0 is exactly between the two texel centers (0.5 and 1.5):
        // bilinear result is 50% gray.
        let c = shader.sample(1.0, 0.5);
        assert!(
            (c.r - 0.5).abs() < 2.0 / 255.0,
            "bilinear midpoint should be 0.5 gray, got {}",
            c.r
        );
        // At a texel center filtering must return the texel itself.
        let c0 = shader.sample(0.5, 0.5);
        assert!(c0.r < 1e-3, "texel center must be exact, got {}", c0.r);
    }
}
