//! Layer types for Lottie animations.
//!
//! This module handles different layer types:
//! - Precomposition
//! - Solid
//! - Image
//! - Null
//! - Shape
//! - Text

use crate::keyframe::AnimatedProperty;
use crate::mask::Mask;
use crate::model::LayerModel;
use crate::shapes::Shape;
use crate::transform::Transform;
use skia_rs_core::{Color, Scalar};
use skia_rs_paint::BlendMode;

/// Layer type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Precomposition layer (nested composition).
    Precomp = 0,
    /// Solid color layer.
    Solid = 1,
    /// Image layer.
    Image = 2,
    /// Null layer (invisible, for parenting).
    Null = 3,
    /// Shape layer.
    Shape = 4,
    /// Text layer.
    Text = 5,
    /// Audio layer.
    Audio = 6,
    /// Video placeholder.
    VideoPlaceholder = 7,
    /// Image sequence.
    ImageSequence = 8,
    /// Video layer.
    Video = 9,
    /// Image placeholder.
    ImagePlaceholder = 10,
    /// Guide layer.
    Guide = 11,
    /// Adjustment layer.
    Adjustment = 12,
    /// Camera layer.
    Camera = 13,
    /// Light layer.
    Light = 14,
    /// Unknown layer type.
    Unknown = 255,
}

impl From<i32> for LayerType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Precomp,
            1 => Self::Solid,
            2 => Self::Image,
            3 => Self::Null,
            4 => Self::Shape,
            5 => Self::Text,
            6 => Self::Audio,
            7 => Self::VideoPlaceholder,
            8 => Self::ImageSequence,
            9 => Self::Video,
            10 => Self::ImagePlaceholder,
            11 => Self::Guide,
            12 => Self::Adjustment,
            13 => Self::Camera,
            14 => Self::Light,
            _ => Self::Unknown,
        }
    }
}

/// A layer in the animation.
#[derive(Debug, Clone)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent Lottie/Bodymovin JSON field (ao/ddd/hd/td); grouping them into a bitset or sub-struct would diverge from the 1:1 schema mapping without adding clarity"
)]
pub struct Layer {
    /// Layer name.
    pub name: String,
    /// Layer index.
    pub index: i32,
    /// Parent layer index.
    pub parent: Option<i32>,
    /// Layer type.
    pub layer_type: LayerType,
    /// In point (start frame).
    pub in_point: Scalar,
    /// Out point (end frame).
    pub out_point: Scalar,
    /// Start time offset.
    pub start_time: Scalar,
    /// Layer transform.
    pub transform: Transform,
    /// Auto-orient along path.
    pub auto_orient: bool,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Is 3D layer.
    pub is_3d: bool,
    /// Is hidden.
    pub hidden: bool,
    /// Layer content.
    pub content: LayerContent,
    /// Masks.
    pub masks: Vec<Mask>,
    /// Track matte type (set on the consumer layer).
    pub matte_mode: Option<MatteMode>,
    /// Explicit track matte source layer index (`tp`); `None` means "the
    /// previous layer in the array" (legacy assets).
    pub matte_layer: Option<i32>,
    /// Legacy "track matte source" hidden flag (`td`). When set (and `hd`
    /// is absent/false), this layer is excluded from the main render list
    /// but its content is still available as a matte input.
    pub matte_source_hidden: bool,
    /// Time stretch factor.
    pub time_stretch: Scalar,
    /// Time remapping.
    pub time_remap: Option<AnimatedProperty>,
}

/// Layer content variants.
#[derive(Debug, Clone)]
pub enum LayerContent {
    /// No content (null layer).
    None,
    /// Precomposition content.
    Precomp(PrecompContent),
    /// Solid color content.
    Solid(SolidContent),
    /// Image content.
    Image(ImageContent),
    /// Shape content.
    Shape(ShapeContent),
    /// Text content.
    Text(TextContent),
}

/// Precomposition content.
#[derive(Debug, Clone)]
pub struct PrecompContent {
    /// Reference ID to asset.
    pub ref_id: String,
    /// Time remapping.
    pub time_remap: Option<AnimatedProperty>,
}

/// Solid color content.
#[derive(Debug, Clone)]
pub struct SolidContent {
    /// Solid color.
    pub color: Color,
    /// Width.
    pub width: Scalar,
    /// Height.
    pub height: Scalar,
}

/// Image content.
#[derive(Debug, Clone)]
pub struct ImageContent {
    /// Reference ID to asset.
    pub ref_id: String,
}

/// Shape content.
#[derive(Debug, Clone)]
pub struct ShapeContent {
    /// Shapes.
    pub shapes: Vec<Shape>,
}

/// Text content.
#[derive(Debug, Clone)]
pub struct TextContent {
    /// Text document.
    pub document: TextDocument,
    /// Path to follow.
    pub path: Option<TextPath>,
    /// More options.
    pub more_options: TextMoreOptions,
}

/// Text document.
#[derive(Debug, Clone, Default)]
pub struct TextDocument {
    /// Text string.
    pub text: String,
    /// Font size.
    pub font_size: Scalar,
    /// Font family.
    pub font_family: String,
    /// Fill color.
    pub fill_color: Option<Color>,
    /// Stroke color.
    pub stroke_color: Option<Color>,
    /// Stroke width.
    pub stroke_width: Scalar,
    /// Justification (0=left, 1=right, 2=center).
    pub justification: i32,
    /// Tracking.
    pub tracking: Scalar,
    /// Line height.
    pub line_height: Scalar,
}

/// Text path options.
#[derive(Debug, Clone, Default)]
pub struct TextPath {
    /// Path mask index.
    pub mask_index: i32,
    /// First margin.
    pub first_margin: Scalar,
    /// Last margin.
    pub last_margin: Scalar,
    /// Force alignment.
    pub force_alignment: bool,
    /// Perpendicular to path.
    pub perpendicular: bool,
}

/// Additional text options.
#[derive(Debug, Clone, Default)]
pub struct TextMoreOptions {
    /// Anchor point grouping.
    pub anchor_point_grouping: i32,
    /// Grouping alignment.
    pub grouping_alignment: [Scalar; 2],
}

/// Track matte mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatteMode {
    /// No matte.
    None = 0,
    /// Alpha matte.
    Alpha = 1,
    /// Inverted alpha matte.
    AlphaInverted = 2,
    /// Luma matte.
    Luma = 3,
    /// Inverted luma matte.
    LumaInverted = 4,
}

impl From<i32> for MatteMode {
    fn from(value: i32) -> Self {
        match value {
            1 => Self::Alpha,
            2 => Self::AlphaInverted,
            3 => Self::Luma,
            4 => Self::LumaInverted,
            _ => Self::None,
        }
    }
}

impl Layer {
    /// Parse from Lottie layer model.
    #[allow(
        clippy::too_many_lines,
        reason = "mirrors upstream skottie layer-content dispatch across precomp/solid/shape/text/image/null kinds; splitting would fragment a single logical match"
    )]
    pub fn from_lottie(model: &LayerModel) -> Self {
        let layer_type = LayerType::from(model.layer_type);

        let transform = model
            .transform
            .as_ref()
            .map(Transform::from_lottie)
            .unwrap_or_default();

        let content = match layer_type {
            LayerType::Precomp => LayerContent::Precomp(PrecompContent {
                ref_id: model.ref_id.clone().unwrap_or_default(),
                time_remap: None,
            }),
            LayerType::Solid => {
                let color = model
                    .solid_color
                    .as_ref()
                    .map_or(Color::WHITE, |s| parse_color_string(s));
                LayerContent::Solid(SolidContent {
                    color,
                    width: model.solid_width.unwrap_or(100.0),
                    height: model.solid_height.unwrap_or(100.0),
                })
            }
            LayerType::Image => LayerContent::Image(ImageContent {
                ref_id: model.ref_id.clone().unwrap_or_default(),
            }),
            LayerType::Shape => {
                let shapes: Vec<Shape> =
                    model.shapes.iter().filter_map(Shape::from_lottie).collect();
                LayerContent::Shape(ShapeContent { shapes })
            }
            LayerType::Text => {
                let doc = model
                    .text
                    .as_ref()
                    .map_or_else(TextDocument::default, |text_data| {
                        text_data
                            .document
                            .keyframes
                            .first()
                            .map(|kf| TextDocument {
                                text: kf.data.text.clone(),
                                font_size: kf.data.size,
                                font_family: kf.data.font.clone(),
                                fill_color: kf.data.fill_color.as_ref().map(|c| {
                                    Color::from_rgb(
                                        skia_rs_core::cast::f32_to_u8_sat(
                                            c.first().copied().unwrap_or(0.0) * 255.0,
                                        ),
                                        skia_rs_core::cast::f32_to_u8_sat(
                                            c.get(1).copied().unwrap_or(0.0) * 255.0,
                                        ),
                                        skia_rs_core::cast::f32_to_u8_sat(
                                            c.get(2).copied().unwrap_or(0.0) * 255.0,
                                        ),
                                    )
                                }),
                                stroke_color: kf.data.stroke_color.as_ref().map(|c| {
                                    Color::from_rgb(
                                        skia_rs_core::cast::f32_to_u8_sat(
                                            c.first().copied().unwrap_or(0.0) * 255.0,
                                        ),
                                        skia_rs_core::cast::f32_to_u8_sat(
                                            c.get(1).copied().unwrap_or(0.0) * 255.0,
                                        ),
                                        skia_rs_core::cast::f32_to_u8_sat(
                                            c.get(2).copied().unwrap_or(0.0) * 255.0,
                                        ),
                                    )
                                }),
                                stroke_width: kf.data.stroke_width.unwrap_or(0.0),
                                justification: kf.data.justify.unwrap_or(0),
                                tracking: kf.data.tracking.unwrap_or(0.0),
                                line_height: kf.data.line_height.unwrap_or(0.0),
                            })
                            .unwrap_or_default()
                    });
                LayerContent::Text(TextContent {
                    document: doc,
                    path: None,
                    more_options: TextMoreOptions::default(),
                })
            }
            _ => LayerContent::None,
        };

        let masks = model.masks.iter().map(Mask::from_lottie).collect();

        let blend_mode = match model.blend_mode {
            1 => BlendMode::Multiply,
            2 => BlendMode::Screen,
            3 => BlendMode::Overlay,
            4 => BlendMode::Darken,
            5 => BlendMode::Lighten,
            6 => BlendMode::ColorDodge,
            7 => BlendMode::ColorBurn,
            8 => BlendMode::HardLight,
            9 => BlendMode::SoftLight,
            10 => BlendMode::Difference,
            11 => BlendMode::Exclusion,
            12 => BlendMode::Hue,
            13 => BlendMode::Saturation,
            14 => BlendMode::Color,
            15 => BlendMode::Luminosity,
            _ => BlendMode::SrcOver,
        };

        Self {
            name: model.name.clone(),
            index: model.index,
            parent: model.parent,
            layer_type,
            in_point: model.in_point,
            out_point: model.out_point,
            start_time: model.start_time,
            transform,
            auto_orient: model.auto_orient != 0,
            blend_mode,
            is_3d: model.is_3d != 0,
            hidden: model.hidden,
            content,
            masks,
            matte_mode: model.track_matte_type.map(MatteMode::from),
            matte_layer: model.track_matte_parent,
            matte_source_hidden: model.track_matte_source_hidden,
            // "sr"/"tm" are only meaningful for precomp layers (upstream
            // `Layer.cpp`/`PrecompLayer.cpp` only reads them there); other
            // layer types simply carry through the defaults.
            time_stretch: model.stretch.unwrap_or(1.0),
            time_remap: model.time_remap.as_ref().map(AnimatedProperty::from_lottie),
        }
    }

    /// Check if this layer is visible at a specific frame.
    #[must_use]
    pub fn is_visible_at(&self, frame: Scalar) -> bool {
        !self.hidden && frame >= self.in_point && frame < self.out_point
    }

    /// Get the local (content) frame for a precomp layer's *nested*
    /// composition, at a given (unadjusted) comp frame.
    ///
    /// Per upstream `attachPrecompLayer` (`PrecompLayer.cpp`), only the
    /// precomp's content time is remapped — the precomp layer's own
    /// transform/opacity/masks evaluate at the unadjusted comp frame.
    /// `tm` (if present) overrides the linear `(t - st) / sr` mapping.
    #[must_use]
    pub fn precomp_content_frame(&self, frame: Scalar, frame_rate: Scalar) -> Scalar {
        self.time_remap.as_ref().map_or_else(
            || {
                let stretch = if self.time_stretch == 0.0 {
                    1.0
                } else {
                    self.time_stretch
                };
                (frame - self.start_time) / stretch
            },
            |remap| {
                let seconds = remap.value_at(frame).as_scalar().unwrap_or(0.0);
                seconds * frame_rate
            },
        )
    }

    /// Get the opacity at a specific frame.
    #[must_use]
    pub fn opacity_at(&self, frame: Scalar) -> Scalar {
        self.transform.opacity_at(frame)
    }

    /// Get the transform matrix at a specific frame.
    #[must_use]
    pub fn matrix_at(&self, frame: Scalar) -> skia_rs_core::Matrix {
        self.transform.matrix_at(frame)
    }

    /// Check if this layer has masks.
    #[must_use]
    pub const fn has_masks(&self) -> bool {
        !self.masks.is_empty()
    }

    /// Check if this is a matte layer.
    #[must_use]
    pub fn is_matte_layer(&self) -> bool {
        self.matte_mode.is_some() && self.matte_mode != Some(MatteMode::None)
    }
}

/// Parse a hex color string to Color.
fn parse_color_string(s: &str) -> Color {
    let s = s.trim_start_matches('#');

    if s.len() >= 6 {
        let r = u8::from_str_radix(&s[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&s[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&s[4..6], 16).unwrap_or(0);
        Color::from_rgb(r, g, b)
    } else {
        Color::WHITE
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert exact keyframe/interpolation output values, not tolerances"
)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_type_conversion() {
        assert_eq!(LayerType::from(0), LayerType::Precomp);
        assert_eq!(LayerType::from(4), LayerType::Shape);
        assert_eq!(LayerType::from(99), LayerType::Unknown);
    }

    #[test]
    fn test_matte_mode_conversion() {
        assert_eq!(MatteMode::from(1), MatteMode::Alpha);
        assert_eq!(MatteMode::from(3), MatteMode::Luma);
        assert_eq!(MatteMode::from(99), MatteMode::None);
    }

    #[test]
    fn test_parse_color_string() {
        let color = parse_color_string("#FF0000");
        assert_eq!(color.red(), 255);
        assert_eq!(color.green(), 0);
        assert_eq!(color.blue(), 0);
    }

    #[test]
    fn test_layer_visibility() {
        let layer = Layer {
            name: "test".to_string(),
            index: 1,
            parent: None,
            layer_type: LayerType::Shape,
            in_point: 10.0,
            out_point: 50.0,
            start_time: 0.0,
            transform: Transform::default(),
            auto_orient: false,
            blend_mode: BlendMode::SrcOver,
            is_3d: false,
            hidden: false,
            content: LayerContent::None,
            masks: Vec::new(),
            matte_mode: None,
            matte_layer: None,
            matte_source_hidden: false,
            time_stretch: 1.0,
            time_remap: None,
        };

        assert!(!layer.is_visible_at(5.0));
        assert!(layer.is_visible_at(10.0));
        assert!(layer.is_visible_at(30.0));
        assert!(!layer.is_visible_at(50.0));
    }

    #[test]
    fn test_precomp_stretch_and_start_time_parsed_and_applied() {
        let json = r#"{
            "ty":0, "nm":"precomp", "ind":1, "ip":0, "op":100, "st":10, "sr":2,
            "refId":"comp_0"
        }"#;
        let model: crate::model::LayerModel = serde_json::from_str(json).unwrap();
        let layer = Layer::from_lottie(&model);

        assert_eq!(layer.time_stretch, 2.0);
        // (frame - st) / sr
        assert_eq!(layer.precomp_content_frame(30.0, 30.0), (30.0 - 10.0) / 2.0);
    }

    #[test]
    fn test_precomp_time_remap_overrides_linear_mapping() {
        let json = r#"{
            "ty":0, "nm":"precomp", "ind":1, "ip":0, "op":100, "st":10, "sr":2,
            "refId":"comp_0", "tm":{"a":0,"k":1.5}
        }"#;
        let model: crate::model::LayerModel = serde_json::from_str(json).unwrap();
        let layer = Layer::from_lottie(&model);

        // tm is in seconds; content frame = tm * fps, ignoring st/sr.
        assert_eq!(layer.precomp_content_frame(30.0, 30.0), 45.0);
    }

    #[test]
    fn test_sr_st_do_not_affect_own_opacity_or_transform_frame() {
        // A non-precomp layer's own transform/opacity are evaluated at the
        // raw comp frame - "sr"/"st" must not shift them (only precomp
        // *content* is remapped).
        let json = r#"{
            "ty":4, "nm":"shape", "ind":1, "ip":0, "op":100, "st":10, "sr":2,
            "ks": {"o": {"a":1,"k":[{"t":0,"s":[0]},{"t":10,"s":[100]}]}}
        }"#;
        let model: crate::model::LayerModel = serde_json::from_str(json).unwrap();
        let layer = Layer::from_lottie(&model);

        // At comp frame 10, opacity should be 100 (raw frame), not shifted
        // by "st" as if it were (10-10)/2=0.
        assert_eq!(layer.opacity_at(10.0), 1.0);
    }
}
