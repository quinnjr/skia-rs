//! Paint structure for drawing configuration.

use crate::blend::BlendMode;
use crate::filter::{ColorFilterRef, ImageFilterRef, MaskFilterRef};
use crate::shader::ShaderRef;
use skia_rs_core::cast::f32_to_u8_sat;
use skia_rs_core::{Color, Color4f, Scalar};
use skia_rs_path::PathEffectRef;

/// Convert a float color component in [0, 1] to a byte, rounding to
/// nearest (Skia's `SkColor4f::toSkColor` semantics — 0.5 maps to 128).
#[inline]
fn color_component_to_byte(v: Scalar) -> u8 {
    f32_to_u8_sat(v * 255.0)
}

/// Paint style (fill, stroke, or both).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum Style {
    /// Fill the shape.
    #[default]
    Fill = 0,
    /// Stroke the outline.
    Stroke,
    /// Both fill and stroke.
    StrokeAndFill,
}

/// Stroke cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum StrokeCap {
    /// Flat cap.
    #[default]
    Butt = 0,
    /// Round cap.
    Round,
    /// Square cap (extends by half stroke width).
    Square,
}

/// Stroke join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum StrokeJoin {
    /// Miter join.
    #[default]
    Miter = 0,
    /// Round join.
    Round,
    /// Bevel join.
    Bevel,
}

/// Paint configuration for drawing operations.
#[derive(Debug, Clone)]
pub struct Paint {
    /// Fill color.
    color: Color4f,
    /// Shader for complex fills (gradients, images, etc.).
    shader: Option<ShaderRef>,
    /// Mask filter for blur, emboss, etc.
    mask_filter: Option<MaskFilterRef>,
    /// Color filter for color transformations.
    color_filter: Option<ColorFilterRef>,
    /// Image filter for effects like blur, drop shadow, etc.
    image_filter: Option<ImageFilterRef>,
    /// Path effect (dash, trim, corner, etc.).
    path_effect: Option<PathEffectRef>,
    /// Blend mode.
    blend_mode: BlendMode,
    /// Style (fill/stroke).
    style: Style,
    /// Stroke width.
    stroke_width: Scalar,
    /// Stroke miter limit.
    stroke_miter: Scalar,
    /// Stroke cap.
    stroke_cap: StrokeCap,
    /// Stroke join.
    stroke_join: StrokeJoin,
    /// Anti-aliasing enabled.
    anti_alias: bool,
    /// Dithering enabled.
    dither: bool,
}

impl Default for Paint {
    fn default() -> Self {
        Self {
            color: Color4f::new(0.0, 0.0, 0.0, 1.0),
            shader: None,
            mask_filter: None,
            color_filter: None,
            image_filter: None,
            path_effect: None,
            blend_mode: BlendMode::SrcOver,
            style: Style::Fill,
            // SkPaint.cpp: fWidth{0} — a zero stroke width means hairline.
            stroke_width: 0.0,
            stroke_miter: 4.0,
            stroke_cap: StrokeCap::Butt,
            stroke_join: StrokeJoin::Miter,
            // SkPaint defaults to no anti-aliasing.
            anti_alias: false,
            dither: false,
        }
    }
}

impl Paint {
    /// Create a new paint with default settings.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the color as Color4f.
    #[inline]
    #[must_use]
    pub const fn color(&self) -> Color4f {
        self.color
    }

    /// Get the color as 32-bit Color.
    ///
    /// Float components are rounded to the nearest byte (Skia's
    /// `SkColor4f::toSkColor` rounding), not truncated.
    #[inline]
    #[must_use]
    pub fn color32(&self) -> Color {
        Color::from_argb(
            color_component_to_byte(self.color.a),
            color_component_to_byte(self.color.r),
            color_component_to_byte(self.color.g),
            color_component_to_byte(self.color.b),
        )
    }

    /// Set the color from Color4f.
    #[inline]
    pub const fn set_color(&mut self, color: Color4f) -> &mut Self {
        self.color = color;
        self
    }

    /// Set the color from 32-bit Color.
    #[inline]
    pub fn set_color32(&mut self, color: Color) -> &mut Self {
        self.color = color.to_color4f();
        self
    }

    /// Set ARGB components.
    #[inline]
    pub fn set_argb(&mut self, a: u8, r: u8, g: u8, b: u8) -> &mut Self {
        self.set_color32(Color::from_argb(a, r, g, b))
    }

    /// Get the alpha value (0.0-1.0).
    #[inline]
    #[must_use]
    pub const fn alpha(&self) -> Scalar {
        self.color.a
    }

    /// Set the alpha value (0.0-1.0).
    #[inline]
    pub const fn set_alpha(&mut self, alpha: Scalar) -> &mut Self {
        self.color.a = alpha;
        self
    }

    /// Get the blend mode.
    #[inline]
    #[must_use]
    pub const fn blend_mode(&self) -> BlendMode {
        self.blend_mode
    }

    /// Set the blend mode.
    #[inline]
    pub const fn set_blend_mode(&mut self, mode: BlendMode) -> &mut Self {
        self.blend_mode = mode;
        self
    }

    /// Get the style.
    #[inline]
    #[must_use]
    pub const fn style(&self) -> Style {
        self.style
    }

    /// Set the style.
    #[inline]
    pub const fn set_style(&mut self, style: Style) -> &mut Self {
        self.style = style;
        self
    }

    /// Get the stroke width.
    #[inline]
    #[must_use]
    pub const fn stroke_width(&self) -> Scalar {
        self.stroke_width
    }

    /// Set the stroke width.
    #[inline]
    pub const fn set_stroke_width(&mut self, width: Scalar) -> &mut Self {
        self.stroke_width = width.max(0.0);
        self
    }

    /// Get the stroke miter limit.
    #[inline]
    #[must_use]
    pub const fn stroke_miter(&self) -> Scalar {
        self.stroke_miter
    }

    /// Set the stroke miter limit.
    #[inline]
    pub const fn set_stroke_miter(&mut self, miter: Scalar) -> &mut Self {
        self.stroke_miter = miter.max(0.0);
        self
    }

    /// Get the stroke cap.
    #[inline]
    #[must_use]
    pub const fn stroke_cap(&self) -> StrokeCap {
        self.stroke_cap
    }

    /// Set the stroke cap.
    #[inline]
    pub const fn set_stroke_cap(&mut self, cap: StrokeCap) -> &mut Self {
        self.stroke_cap = cap;
        self
    }

    /// Get the stroke join.
    #[inline]
    #[must_use]
    pub const fn stroke_join(&self) -> StrokeJoin {
        self.stroke_join
    }

    /// Set the stroke join.
    #[inline]
    pub const fn set_stroke_join(&mut self, join: StrokeJoin) -> &mut Self {
        self.stroke_join = join;
        self
    }

    /// Get the shader.
    #[inline]
    #[must_use]
    pub fn shader(&self) -> Option<&ShaderRef> {
        self.shader.as_ref()
    }

    /// Set the shader.
    #[inline]
    pub fn set_shader(&mut self, shader: Option<ShaderRef>) -> &mut Self {
        self.shader = shader;
        self
    }

    /// Check if the paint has a shader.
    #[inline]
    #[must_use]
    pub fn has_shader(&self) -> bool {
        self.shader.is_some()
    }

    /// Get the mask filter.
    #[inline]
    #[must_use]
    pub fn mask_filter(&self) -> Option<&MaskFilterRef> {
        self.mask_filter.as_ref()
    }

    /// Set the mask filter.
    #[inline]
    pub fn set_mask_filter(&mut self, filter: Option<MaskFilterRef>) -> &mut Self {
        self.mask_filter = filter;
        self
    }

    /// Check if the paint has a mask filter.
    #[inline]
    #[must_use]
    pub fn has_mask_filter(&self) -> bool {
        self.mask_filter.is_some()
    }

    /// Get the color filter.
    #[inline]
    #[must_use]
    pub fn color_filter(&self) -> Option<&ColorFilterRef> {
        self.color_filter.as_ref()
    }

    /// Set the color filter.
    #[inline]
    pub fn set_color_filter(&mut self, filter: Option<ColorFilterRef>) -> &mut Self {
        self.color_filter = filter;
        self
    }

    /// Check if the paint has a color filter.
    #[inline]
    #[must_use]
    pub fn has_color_filter(&self) -> bool {
        self.color_filter.is_some()
    }

    /// Get the image filter.
    #[inline]
    #[must_use]
    pub fn image_filter(&self) -> Option<&ImageFilterRef> {
        self.image_filter.as_ref()
    }

    /// Set the image filter.
    #[inline]
    pub fn set_image_filter(&mut self, filter: Option<ImageFilterRef>) -> &mut Self {
        self.image_filter = filter;
        self
    }

    /// Check if the paint has an image filter.
    #[inline]
    #[must_use]
    pub fn has_image_filter(&self) -> bool {
        self.image_filter.is_some()
    }

    /// Get the path effect.
    #[inline]
    #[must_use]
    pub fn path_effect(&self) -> Option<&PathEffectRef> {
        self.path_effect.as_ref()
    }

    /// Set the path effect.
    #[inline]
    pub fn set_path_effect(&mut self, effect: Option<PathEffectRef>) -> &mut Self {
        self.path_effect = effect;
        self
    }

    /// Check if the paint has a path effect.
    #[inline]
    #[must_use]
    pub fn has_path_effect(&self) -> bool {
        self.path_effect.is_some()
    }

    /// Check if anti-aliasing is enabled.
    #[inline]
    #[must_use]
    pub const fn is_anti_alias(&self) -> bool {
        self.anti_alias
    }

    /// Set anti-aliasing.
    #[inline]
    pub const fn set_anti_alias(&mut self, aa: bool) -> &mut Self {
        self.anti_alias = aa;
        self
    }

    /// Check if dithering is enabled.
    #[inline]
    #[must_use]
    pub const fn is_dither(&self) -> bool {
        self.dither
    }

    /// Set dithering.
    #[inline]
    pub const fn set_dither(&mut self, dither: bool) -> &mut Self {
        self.dither = dither;
        self
    }

    /// Alias for `is_anti_alias` (Skia compatibility).
    #[inline]
    #[must_use]
    pub const fn anti_alias(&self) -> bool {
        self.anti_alias
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize the paint to bytes.
    ///
    /// Format:
    /// - 4 bytes: RGBA color (each component 0-255)
    /// - 1 byte: blend mode
    /// - 1 byte: style
    /// - 4 bytes: stroke width (f32 little-endian)
    /// - 4 bytes: stroke miter (f32 little-endian)
    /// - 1 byte: stroke cap
    /// - 1 byte: stroke join
    /// - 1 byte: flags (bit 0: `anti_alias`, bit 1: dither)
    /// - \[optional trailing sections for shader, `mask_filter`, `color_filter`, `image_filter`\]
    ///
    /// Each optional section is: \[1 byte present flag\] \[4 bytes length\] \[data\].
    /// If a shader/filter does not support serialization (e.g., runtime shaders), the
    /// present flag is 0 and that field is omitted from deserialization.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(17);

        // Color (4 bytes, rounded to nearest)
        data.push(color_component_to_byte(self.color.r));
        data.push(color_component_to_byte(self.color.g));
        data.push(color_component_to_byte(self.color.b));
        data.push(color_component_to_byte(self.color.a));

        // Blend mode (1 byte)
        data.push(self.blend_mode as u8);

        // Style (1 byte)
        data.push(self.style as u8);

        // Stroke width (4 bytes)
        data.extend_from_slice(&self.stroke_width.to_le_bytes());

        // Stroke miter (4 bytes)
        data.extend_from_slice(&self.stroke_miter.to_le_bytes());

        // Stroke cap (1 byte)
        data.push(self.stroke_cap as u8);

        // Stroke join (1 byte)
        data.push(self.stroke_join as u8);

        // Flags (1 byte)
        let mut flags = 0u8;
        if self.anti_alias {
            flags |= 0x01;
        }
        if self.dither {
            flags |= 0x02;
        }
        data.push(flags);

        // Shader section
        // A section whose length does not fit the u32 prefix is absent.
        match self
            .shader
            .as_ref()
            .and_then(|s| s.serialize())
            .and_then(|d| Some((u32::try_from(d.len()).ok()?, d)))
        {
            Some((len, shader_data)) => {
                data.push(1); // present
                data.extend_from_slice(&len.to_le_bytes());
                data.extend_from_slice(&shader_data);
            }
            None => data.push(0), // absent
        }

        // MaskFilter section
        // A section whose length does not fit the u32 prefix is absent.
        match self
            .mask_filter
            .as_ref()
            .and_then(|f| f.serialize())
            .and_then(|d| Some((u32::try_from(d.len()).ok()?, d)))
        {
            Some((len, filter_data)) => {
                data.push(1);
                data.extend_from_slice(&len.to_le_bytes());
                data.extend_from_slice(&filter_data);
            }
            None => data.push(0),
        }

        // ColorFilter section
        // A section whose length does not fit the u32 prefix is absent.
        match self
            .color_filter
            .as_ref()
            .and_then(|f| f.serialize())
            .and_then(|d| Some((u32::try_from(d.len()).ok()?, d)))
        {
            Some((len, filter_data)) => {
                data.push(1);
                data.extend_from_slice(&len.to_le_bytes());
                data.extend_from_slice(&filter_data);
            }
            None => data.push(0),
        }

        // ImageFilter section
        // A section whose length does not fit the u32 prefix is absent.
        match self
            .image_filter
            .as_ref()
            .and_then(|f| f.serialize())
            .and_then(|d| Some((u32::try_from(d.len()).ok()?, d)))
        {
            Some((len, filter_data)) => {
                data.push(1);
                data.extend_from_slice(&len.to_le_bytes());
                data.extend_from_slice(&filter_data);
            }
            None => data.push(0),
        }

        data
    }

    /// Read an optional length-prefixed byte section at `*offset`, advancing
    /// `*offset` past it.
    ///
    /// Returns `Ok(None)` if the section is absent (present-flag is 0 or
    /// there is no more data), `Ok(Some(bytes))` if present and well formed,
    /// and `Err(())` if the data is truncated/malformed.
    fn read_optional_section<'a>(
        data: &'a [u8],
        offset: &mut usize,
    ) -> Result<Option<&'a [u8]>, ()> {
        if *offset >= data.len() {
            return Ok(None);
        }
        // 0 is absent and 1 present; any other flag is malformed (reading
        // on would misalign every later section).
        let present = match data[*offset] {
            0 => false,
            1 => true,
            _ => return Err(()),
        };
        *offset += 1;
        if !present {
            return Ok(None);
        }
        if *offset + 4 > data.len() {
            return Err(());
        }
        let len = u32::from_le_bytes([
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ]) as usize;
        *offset += 4;
        if len > data.len() - *offset {
            return Err(());
        }
        let bytes = &data[*offset..*offset + len];
        *offset += len;
        Ok(Some(bytes))
    }

    /// Deserialize a paint from bytes.
    ///
    /// Returns `None` if the data is invalid or too short.
    ///
    /// Shaders and filters are deserialized from optional trailing sections.
    /// Runtime shaders/filters (SkSL-based) do not serialize and will be `None`
    /// after deserialization.
    #[must_use]
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 17 {
            return None;
        }

        let mut offset = 0;

        // Color
        let r = f32::from(data[offset]) / 255.0;
        let g = f32::from(data[offset + 1]) / 255.0;
        let b = f32::from(data[offset + 2]) / 255.0;
        let a = f32::from(data[offset + 3]) / 255.0;
        let color = Color4f::new(r, g, b, a);
        offset += 4;

        // Blend mode
        let blend_mode = BlendMode::from_u8(data[offset])?;
        offset += 1;

        // Style
        let style = match data[offset] {
            0 => Style::Fill,
            1 => Style::Stroke,
            2 => Style::StrokeAndFill,
            _ => return None,
        };
        offset += 1;

        // Stroke width
        let stroke_width = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Stroke miter
        let stroke_miter = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Stroke cap
        let stroke_cap = match data[offset] {
            0 => StrokeCap::Butt,
            1 => StrokeCap::Round,
            2 => StrokeCap::Square,
            _ => return None,
        };
        offset += 1;

        // Stroke join
        let stroke_join = match data[offset] {
            0 => StrokeJoin::Miter,
            1 => StrokeJoin::Round,
            2 => StrokeJoin::Bevel,
            _ => return None,
        };
        offset += 1;

        // Flags
        let flags = data[offset];
        let anti_alias = (flags & 0x01) != 0;
        let dither = (flags & 0x02) != 0;
        offset += 1;

        // Optional trailing sections (shader, mask_filter, color_filter, image_filter)
        let shader = Self::read_optional_section(data, &mut offset)
            .ok()?
            .and_then(|shader_bytes| {
                let mut shader_offset = 0;
                crate::shader::deserialize_shader(shader_bytes, &mut shader_offset)
            });

        let mask_filter = Self::read_optional_section(data, &mut offset)
            .ok()?
            .and_then(|filter_bytes| {
                let mut filter_offset = 0;
                crate::filter::deserialize_mask_filter(filter_bytes, &mut filter_offset)
            });

        let color_filter = Self::read_optional_section(data, &mut offset)
            .ok()?
            .and_then(|filter_bytes| {
                let mut filter_offset = 0;
                crate::filter::deserialize_color_filter(filter_bytes, &mut filter_offset)
            });

        let image_filter = Self::read_optional_section(data, &mut offset)
            .ok()?
            .and_then(|filter_bytes| {
                let mut filter_offset = 0;
                crate::filter::deserialize_image_filter(filter_bytes, &mut filter_offset)
            });

        Some(Self {
            color,
            shader,
            mask_filter,
            color_filter,
            image_filter,
            path_effect: None,
            blend_mode,
            style,
            stroke_width,
            stroke_miter,
            stroke_cap,
            stroke_join,
            anti_alias,
            dither,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact default value check (hairline stroke width is exactly 0.0), not a computed comparison"
    )]
    fn test_paint_defaults_match_skpaint() {
        // SkPaint.cpp: fWidth{0} (hairline) and fAntiAlias false by default.
        let paint = Paint::new();
        assert_eq!(
            paint.stroke_width(),
            0.0,
            "default stroke width is hairline (0)"
        );
        assert!(!paint.is_anti_alias(), "anti-aliasing is off by default");
    }

    #[test]
    fn test_color32_rounds_float_components() {
        // Float->byte conversion must round to nearest, not truncate:
        // 0.5 * 255 = 127.5 -> 128.
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(0.5, 0.5, 0.5, 0.5));
        let c = paint.color32();
        assert_eq!(c.red(), 128, "r was {}", c.red());
        assert_eq!(c.green(), 128);
        assert_eq!(c.blue(), 128);
        assert_eq!(c.alpha(), 128);
    }

    #[test]
    fn test_serialize_rounds_color_bytes() {
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(0.5, 0.5, 0.5, 0.5));
        let data = paint.serialize();
        // First four bytes are RGBA.
        assert_eq!(&data[0..4], &[128, 128, 128, 128]);
    }

    #[test]
    fn test_paint_serialization() {
        let mut paint = Paint::new();
        paint
            .set_color(Color4f::new(1.0, 0.5, 0.25, 0.75))
            .set_blend_mode(BlendMode::Multiply)
            .set_style(Style::Stroke)
            .set_stroke_width(2.5)
            .set_stroke_miter(8.0)
            .set_stroke_cap(StrokeCap::Round)
            .set_stroke_join(StrokeJoin::Bevel)
            .set_anti_alias(false)
            .set_dither(true);

        let serialized = paint.serialize();
        let deserialized = Paint::deserialize(&serialized).unwrap();

        // Check round-trip
        assert!((deserialized.color().r - paint.color().r).abs() < 0.01);
        assert!((deserialized.color().g - paint.color().g).abs() < 0.01);
        assert!((deserialized.color().b - paint.color().b).abs() < 0.01);
        assert!((deserialized.color().a - paint.color().a).abs() < 0.01);
        assert_eq!(deserialized.blend_mode(), paint.blend_mode());
        assert_eq!(deserialized.style(), paint.style());
        assert!((deserialized.stroke_width() - paint.stroke_width()).abs() < 0.001);
        assert!((deserialized.stroke_miter() - paint.stroke_miter()).abs() < 0.001);
        assert_eq!(deserialized.stroke_cap(), paint.stroke_cap());
        assert_eq!(deserialized.stroke_join(), paint.stroke_join());
        assert_eq!(deserialized.is_anti_alias(), paint.is_anti_alias());
        assert_eq!(deserialized.is_dither(), paint.is_dither());
    }

    #[test]
    fn test_paint_deserialize_invalid() {
        // Too short
        assert!(Paint::deserialize(&[0; 10]).is_none());

        // Invalid blend mode
        let mut data = Paint::new().serialize();
        data[4] = 255;
        assert!(Paint::deserialize(&data).is_none());
    }

    #[test]
    fn test_paint_mask_filter() {
        use crate::filter::{BlurMaskFilter, BlurStyle};
        use std::sync::Arc;

        let mut paint = Paint::new();
        assert!(paint.mask_filter().is_none());
        assert!(!paint.has_mask_filter());

        let filter: MaskFilterRef = Arc::new(BlurMaskFilter::new(BlurStyle::Normal, 5.0));
        paint.set_mask_filter(Some(filter));
        assert!(paint.mask_filter().is_some());
        assert!(paint.has_mask_filter());

        paint.set_mask_filter(None);
        assert!(paint.mask_filter().is_none());
        assert!(!paint.has_mask_filter());
    }

    #[test]
    fn test_paint_color_filter() {
        use crate::filter::ColorMatrixFilter;
        use std::sync::Arc;

        let mut paint = Paint::new();
        assert!(paint.color_filter().is_none());
        assert!(!paint.has_color_filter());

        let filter: ColorFilterRef = Arc::new(ColorMatrixFilter::identity());
        paint.set_color_filter(Some(filter));
        assert!(paint.color_filter().is_some());
        assert!(paint.has_color_filter());

        paint.set_color_filter(None);
        assert!(paint.color_filter().is_none());
        assert!(!paint.has_color_filter());
    }

    #[test]
    fn test_paint_image_filter() {
        use crate::filter::BlurImageFilter;
        use crate::shader::TileMode;
        use std::sync::Arc;

        let mut paint = Paint::new();
        assert!(paint.image_filter().is_none());
        assert!(!paint.has_image_filter());

        let filter: ImageFilterRef = Arc::new(BlurImageFilter::new(5.0, 5.0, TileMode::Clamp));
        paint.set_image_filter(Some(filter));
        assert!(paint.image_filter().is_some());
        assert!(paint.has_image_filter());

        paint.set_image_filter(None);
        assert!(paint.image_filter().is_none());
        assert!(!paint.has_image_filter());
    }

    #[test]
    fn test_paint_serialize_preserves_color_shader() {
        use crate::shader::ColorShader;
        use std::sync::Arc;

        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(ColorShader::new(Color4f::new(
            0.5, 0.25, 0.75, 1.0,
        )))));

        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();

        assert!(restored.shader().is_some());
        // Verify by sampling - should be the color we set
        let c = restored.shader().unwrap().sample(0.0, 0.0);
        assert!((c.r - 0.5).abs() < 1e-3);
        assert!((c.g - 0.25).abs() < 1e-3);
        assert!((c.b - 0.75).abs() < 1e-3);
        assert!((c.a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_paint_serialize_preserves_linear_gradient() {
        use crate::shader::{LinearGradient, TileMode};
        use skia_rs_core::Point;
        use std::sync::Arc;

        let mut paint = Paint::new();
        let gradient = LinearGradient::new(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            vec![
                Color4f::new(1.0, 0.0, 0.0, 1.0),
                Color4f::new(0.0, 0.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        paint.set_shader(Some(Arc::new(gradient)));

        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();

        assert!(restored.shader().is_some());
        // Sample at start and end to verify gradient
        let c_start = restored.shader().unwrap().sample(0.0, 0.0);
        assert!((c_start.r - 1.0).abs() < 1e-3);
        assert!((c_start.b - 0.0).abs() < 1e-3);

        let c_end = restored.shader().unwrap().sample(100.0, 0.0);
        assert!((c_end.r - 0.0).abs() < 1e-3);
        assert!((c_end.b - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_paint_serialize_no_shader_backward_compat() {
        // A Paint without a shader should still round-trip
        let paint = Paint::new();
        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();
        assert!(restored.shader().is_none());
    }

    #[test]
    fn test_paint_serialize_old_format_still_deserializes() {
        // Old format (17 bytes) should deserialize correctly
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(0.8, 0.6, 0.4, 1.0));

        let bytes = paint.serialize();
        // Take only the first 17 bytes (original format)
        let old_bytes = &bytes[0..17];
        let restored = Paint::deserialize(old_bytes).unwrap();

        // Should deserialize but shader will be None
        assert!(restored.shader().is_none());
        assert!((restored.color().r - 0.8).abs() < 0.01);
        assert!((restored.color().g - 0.6).abs() < 0.01);
        assert!((restored.color().b - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_paint_serialize_preserves_mask_filter() {
        use crate::filter::{BlurMaskFilter, BlurStyle};
        use std::sync::Arc;

        let mut paint = Paint::new();
        paint.set_mask_filter(Some(Arc::new(BlurMaskFilter::new(BlurStyle::Normal, 4.5))));
        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();
        assert!(restored.mask_filter().is_some());
        assert_eq!(restored.mask_filter().unwrap().blur_radius(), Some(4.5));
    }

    #[test]
    fn test_paint_serialize_preserves_color_filter() {
        use crate::filter::ColorMatrixFilter;
        use std::sync::Arc;

        let mut paint = Paint::new();
        let matrix = ColorMatrixFilter::identity();
        paint.set_color_filter(Some(Arc::new(matrix)));
        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();
        assert!(restored.color_filter().is_some());
    }

    #[test]
    fn test_paint_serialize_preserves_image_filter() {
        use crate::filter::BlurImageFilter;
        use std::sync::Arc;

        let mut paint = Paint::new();
        let filter = BlurImageFilter::new(3.0, 3.0, crate::shader::TileMode::Clamp);
        paint.set_image_filter(Some(Arc::new(filter)));
        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();
        assert!(restored.image_filter().is_some());
    }

    #[test]
    fn test_paint_serialize_all_effects_round_trip() {
        use crate::filter::{BlurImageFilter, BlurMaskFilter, BlurStyle, ColorMatrixFilter};
        use crate::shader::ColorShader;
        use std::sync::Arc;

        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(ColorShader::new(Color4f::new(
            0.1, 0.2, 0.3, 1.0,
        )))));
        paint.set_mask_filter(Some(Arc::new(BlurMaskFilter::new(BlurStyle::Normal, 2.0))));
        paint.set_color_filter(Some(Arc::new(ColorMatrixFilter::identity())));
        paint.set_image_filter(Some(Arc::new(BlurImageFilter::new(
            1.5,
            1.5,
            crate::shader::TileMode::Clamp,
        ))));

        let bytes = paint.serialize();
        let restored = Paint::deserialize(&bytes).unwrap();
        assert!(restored.shader().is_some());
        assert!(restored.mask_filter().is_some());
        assert!(restored.color_filter().is_some());
        assert!(restored.image_filter().is_some());
    }

    #[test]
    fn test_paint_serialize_all_blend_modes() {
        use crate::blend::BlendMode;
        for i in 0..29u8 {
            if let Some(mode) = BlendMode::from_u8(i) {
                let mut paint = Paint::new();
                paint.set_blend_mode(mode);
                let bytes = paint.serialize();
                let restored = Paint::deserialize(&bytes).unwrap();
                assert_eq!(
                    restored.blend_mode(),
                    mode,
                    "round-trip failed for {mode:?}"
                );
            }
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact byte round-trip check for serialize/deserialize, not a computed comparison"
    )]
    fn test_paint_serialize_extreme_stroke_widths() {
        let mut paint = Paint::new();
        for w in &[0.0, 0.001, 100.0, 10_000.0, f32::MIN_POSITIVE, f32::EPSILON] {
            paint.set_stroke_width(*w);
            let bytes = paint.serialize();
            let restored = Paint::deserialize(&bytes).unwrap();
            assert_eq!(
                restored.stroke_width(),
                *w,
                "stroke width {w} failed round-trip"
            );
        }
    }

    #[test]
    fn test_paint_deserialize_truncated_buffer_is_safe() {
        // Truncated at various lengths should return None, not panic
        let paint = Paint::new();
        let bytes = paint.serialize();
        for cutoff in 0..bytes.len() {
            let truncated = &bytes[..cutoff];
            let _ = Paint::deserialize(truncated); // must not panic
        }
    }

    #[test]
    fn test_paint_deserialize_trailing_garbage_is_ignored() {
        let paint = Paint::new();
        let mut bytes = paint.serialize();
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        // Should still deserialize successfully (trailing bytes are simply unread)
        // OR return error safely — either is acceptable, must not panic
        let _ = Paint::deserialize(&bytes);
    }
}

// Property-based tests for T-7
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        #[allow(
            clippy::float_cmp,
            reason = "exact byte round-trip check for serialize/deserialize, not a computed comparison"
        )]
        fn test_paint_stroke_width_round_trip(width in 0.0_f32..1e6) {
            let mut paint = Paint::new();
            paint.set_stroke_width(width);
            let bytes = paint.serialize();
            let restored = Paint::deserialize(&bytes).unwrap();
            prop_assert_eq!(restored.stroke_width(), width);
        }

        #[test]
        fn test_blend_mode_apply_never_nan(
            sr in 0.0_f32..=1.0, sg in 0.0_f32..=1.0, sb in 0.0_f32..=1.0, sa in 0.0_f32..=1.0,
            dr in 0.0_f32..=1.0, dg in 0.0_f32..=1.0, db in 0.0_f32..=1.0, da in 0.0_f32..=1.0,
            mode_u8 in 0u8..29
        ) {
            use crate::blend::BlendMode;
            if let Some(mode) = BlendMode::from_u8(mode_u8) {
                let src = Color4f::new(sr, sg, sb, sa);
                let dst = Color4f::new(dr, dg, db, da);
                let result = mode.apply(src, dst);
                prop_assert!(result.r.is_finite());
                prop_assert!(result.g.is_finite());
                prop_assert!(result.b.is_finite());
                prop_assert!(result.a.is_finite());
            }
        }
    }
}
