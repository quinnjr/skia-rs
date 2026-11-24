//! Transform animations for Lottie layers.
//!
//! This module handles animated transforms including:
//! - Position (with optional separated X/Y)
//! - Anchor point
//! - Scale
//! - Rotation
//! - Opacity
//! - Skew

use crate::keyframe::{AnimatedProperty, KeyframeValue};
use crate::model::{ShapeModel, TransformModel};
use skia_rs_core::{Matrix, Scalar};

/// AE control limit for skew angle (degrees), per upstream
/// `TransformAdapter2D::totalMatrix` (`Transform.cpp`).
const MAX_SKEW_ANGLE: Scalar = 85.0;

/// Animated transform for a layer or shape.
#[derive(Debug, Clone)]
pub struct Transform {
    /// Anchor point.
    pub anchor: AnimatedProperty,
    /// Position.
    pub position: AnimatedProperty,
    /// Position X (if separated).
    pub position_x: Option<AnimatedProperty>,
    /// Position Y (if separated).
    pub position_y: Option<AnimatedProperty>,
    /// Scale (percentage, 100 = 1.0).
    pub scale: AnimatedProperty,
    /// Rotation (degrees).
    pub rotation: AnimatedProperty,
    /// Opacity (0-100).
    pub opacity: AnimatedProperty,
    /// Skew (degrees).
    pub skew: Option<AnimatedProperty>,
    /// Skew axis (degrees).
    pub skew_axis: Option<AnimatedProperty>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            anchor: AnimatedProperty::static_value(KeyframeValue::Vec2([0.0, 0.0])),
            position: AnimatedProperty::static_value(KeyframeValue::Vec2([0.0, 0.0])),
            position_x: None,
            position_y: None,
            scale: AnimatedProperty::static_value(KeyframeValue::Vec2([100.0, 100.0])),
            rotation: AnimatedProperty::static_value(KeyframeValue::Scalar(0.0)),
            opacity: AnimatedProperty::static_value(KeyframeValue::Scalar(100.0)),
            skew: None,
            skew_axis: None,
        }
    }
}

impl Transform {
    /// Create a new identity transform.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse from Lottie transform model.
    #[must_use]
    pub fn from_lottie(model: &TransformModel) -> Self {
        let mut transform = Self::new();

        if let Some(ref anchor) = model.anchor {
            transform.anchor = AnimatedProperty::from_lottie(anchor);
        }

        if let Some(ref position) = model.position {
            transform.position = AnimatedProperty::from_lottie(position);
        }

        if let Some(ref px) = model.position_x {
            transform.position_x = Some(AnimatedProperty::from_lottie(px));
        }

        if let Some(ref py) = model.position_y {
            transform.position_y = Some(AnimatedProperty::from_lottie(py));
        }

        if let Some(ref scale) = model.scale {
            transform.scale = AnimatedProperty::from_lottie(scale);
        }

        if let Some(ref rotation) = model.rotation {
            transform.rotation = AnimatedProperty::from_lottie(rotation);
        }

        if let Some(ref opacity) = model.opacity {
            transform.opacity = AnimatedProperty::from_lottie(opacity);
        }

        if let Some(ref skew) = model.skew {
            transform.skew = Some(AnimatedProperty::from_lottie(skew));
        }

        if let Some(ref skew_axis) = model.skew_axis {
            transform.skew_axis = Some(AnimatedProperty::from_lottie(skew_axis));
        }

        transform
    }

    /// Parse a group/shape "tr" transform item. `ShapeModel` fields are
    /// heavily overloaded across shape types (same JSON keys mean
    /// different things per `"ty"`); for `"tr"` items:
    /// `a`=anchor, `p`=position, `s`=scale (%), `r`=rotation (degrees),
    /// `o`=opacity (%), `sk`=skew (degrees), `sa`=skew axis (degrees).
    #[must_use]
    pub fn from_shape_lottie(model: &ShapeModel) -> Self {
        let mut transform = Self::new();

        if let Some(ref anchor) = model.transform {
            transform.anchor = AnimatedProperty::from_lottie(anchor);
        }
        if let Some(ref position) = model.position {
            transform.position = AnimatedProperty::from_lottie(position);
        }
        if let Some(ref scale) = model.size {
            transform.scale = AnimatedProperty::from_lottie(scale);
        }
        if let Some(ref rotation) = model.roundness {
            transform.rotation = AnimatedProperty::from_lottie(rotation);
        }
        if let Some(ref opacity) = model.opacity {
            transform.opacity = AnimatedProperty::from_lottie(opacity);
        }
        if let Some(ref skew) = model.skew {
            transform.skew = Some(AnimatedProperty::from_lottie(skew));
        }
        if let Some(ref skew_axis) = model.skew_axis {
            transform.skew_axis = Some(AnimatedProperty::from_lottie(skew_axis));
        }

        transform
    }

    /// Check if this transform is animated.
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.anchor.is_animated()
            || self.position.is_animated()
            || self
                .position_x
                .as_ref()
                .is_some_and(super::keyframe::AnimatedProperty::is_animated)
            || self
                .position_y
                .as_ref()
                .is_some_and(super::keyframe::AnimatedProperty::is_animated)
            || self.scale.is_animated()
            || self.rotation.is_animated()
            || self.opacity.is_animated()
            || self
                .skew
                .as_ref()
                .is_some_and(super::keyframe::AnimatedProperty::is_animated)
            || self
                .skew_axis
                .as_ref()
                .is_some_and(super::keyframe::AnimatedProperty::is_animated)
    }

    /// Get the position at a specific frame.
    #[must_use]
    pub fn position_at(&self, frame: Scalar) -> [Scalar; 2] {
        if self.position_x.is_some() || self.position_y.is_some() {
            // Separated position
            let x = self
                .position_x
                .as_ref()
                .map_or(0.0, |p| p.value_at(frame).as_scalar().unwrap_or(0.0));
            let y = self
                .position_y
                .as_ref()
                .map_or(0.0, |p| p.value_at(frame).as_scalar().unwrap_or(0.0));
            [x, y]
        } else {
            self.position
                .value_at(frame)
                .as_vec2()
                .unwrap_or([0.0, 0.0])
        }
    }

    /// Get the anchor point at a specific frame.
    #[must_use]
    pub fn anchor_at(&self, frame: Scalar) -> [Scalar; 2] {
        self.anchor.value_at(frame).as_vec2().unwrap_or([0.0, 0.0])
    }

    /// Get the scale at a specific frame (as factor, not percentage).
    #[must_use]
    pub fn scale_at(&self, frame: Scalar) -> [Scalar; 2] {
        let scale = self
            .scale
            .value_at(frame)
            .as_vec2()
            .unwrap_or([100.0, 100.0]);
        [scale[0] / 100.0, scale[1] / 100.0]
    }

    /// Get the rotation at a specific frame (in radians).
    #[must_use]
    pub fn rotation_at(&self, frame: Scalar) -> Scalar {
        let degrees = self.rotation.value_at(frame).as_scalar().unwrap_or(0.0);
        degrees.to_radians()
    }

    /// Get the opacity at a specific frame (0.0 - 1.0).
    #[must_use]
    pub fn opacity_at(&self, frame: Scalar) -> Scalar {
        let opacity = self.opacity.value_at(frame).as_scalar().unwrap_or(100.0);
        (opacity / 100.0).clamp(0.0, 1.0)
    }

    /// Get the skew at a specific frame (in radians).
    ///
    /// Per upstream `TransformAdapter2D::totalMatrix` (`Transform.cpp`),
    /// the skew angle is pinned to `[-85, 85]` degrees and negated before
    /// conversion to radians.
    #[must_use]
    pub fn skew_at(&self, frame: Scalar) -> Option<Scalar> {
        self.skew.as_ref().map(|s| {
            let degrees = s.value_at(frame).as_scalar().unwrap_or(0.0);
            let pinned = degrees.clamp(-MAX_SKEW_ANGLE, MAX_SKEW_ANGLE);
            -pinned.to_radians()
        })
    }

    /// Get the skew axis at a specific frame (in radians).
    #[must_use]
    pub fn skew_axis_at(&self, frame: Scalar) -> Option<Scalar> {
        self.skew_axis.as_ref().map(|s| {
            let degrees = s.value_at(frame).as_scalar().unwrap_or(0.0);
            degrees.to_radians()
        })
    }

    /// Compute the transformation matrix at a specific frame.
    #[must_use]
    pub fn matrix_at(&self, frame: Scalar) -> Matrix {
        let position = self.position_at(frame);
        let anchor = self.anchor_at(frame);
        let scale = self.scale_at(frame);
        let rotation = self.rotation_at(frame);

        // Build matrix: translate(position) * rotate * scale * translate(-anchor)
        let mut matrix = Matrix::IDENTITY;

        // Translate to position
        matrix = matrix.concat(&Matrix::translate(position[0], position[1]));

        // Rotate
        if rotation != 0.0 {
            matrix = matrix.concat(&Matrix::rotate(rotation));
        }

        // Skew (if present)
        if let Some(skew) = self.skew_at(frame)
            && skew != 0.0
        {
            let skew_axis = self.skew_axis_at(frame).unwrap_or(0.0);
            // Rotate to skew axis, apply skew, rotate back
            if skew_axis != 0.0 {
                matrix = matrix.concat(&Matrix::rotate(skew_axis));
            }
            matrix = matrix.concat(&Matrix::skew(skew.tan(), 0.0));
            if skew_axis != 0.0 {
                matrix = matrix.concat(&Matrix::rotate(-skew_axis));
            }
        }

        // Scale
        matrix = matrix.concat(&Matrix::scale(scale[0], scale[1]));

        // Translate by negative anchor
        matrix = matrix.concat(&Matrix::translate(-anchor[0], -anchor[1]));

        matrix
    }
}

/// Transform snapshot at a specific frame.
#[derive(Debug, Clone, Copy)]
pub struct TransformSnapshot {
    /// Position.
    pub position: [Scalar; 2],
    /// Anchor point.
    pub anchor: [Scalar; 2],
    /// Scale.
    pub scale: [Scalar; 2],
    /// Rotation (radians).
    pub rotation: Scalar,
    /// Opacity (0-1).
    pub opacity: Scalar,
    /// Skew (radians).
    pub skew: Option<Scalar>,
    /// Skew axis (radians).
    pub skew_axis: Option<Scalar>,
}

impl TransformSnapshot {
    /// Create a snapshot from a transform at a specific frame.
    #[must_use]
    pub fn from_transform(transform: &Transform, frame: Scalar) -> Self {
        Self {
            position: transform.position_at(frame),
            anchor: transform.anchor_at(frame),
            scale: transform.scale_at(frame),
            rotation: transform.rotation_at(frame),
            opacity: transform.opacity_at(frame),
            skew: transform.skew_at(frame),
            skew_axis: transform.skew_axis_at(frame),
        }
    }

    /// Compute the matrix.
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        let mut matrix = Matrix::IDENTITY;

        // Translate to position
        matrix = matrix.concat(&Matrix::translate(self.position[0], self.position[1]));

        // Rotate
        if self.rotation != 0.0 {
            matrix = matrix.concat(&Matrix::rotate(self.rotation));
        }

        // Skew
        if let Some(skew) = self.skew
            && skew != 0.0
        {
            let skew_axis = self.skew_axis.unwrap_or(0.0);
            if skew_axis != 0.0 {
                matrix = matrix.concat(&Matrix::rotate(skew_axis));
            }
            matrix = matrix.concat(&Matrix::skew(skew.tan(), 0.0));
            if skew_axis != 0.0 {
                matrix = matrix.concat(&Matrix::rotate(-skew_axis));
            }
        }

        // Scale
        matrix = matrix.concat(&Matrix::scale(self.scale[0], self.scale[1]));

        // Translate by negative anchor
        matrix = matrix.concat(&Matrix::translate(-self.anchor[0], -self.anchor[1]));

        matrix
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
    fn test_identity_transform() {
        let transform = Transform::new();
        let matrix = transform.matrix_at(0.0);

        // Should be close to identity
        assert!((matrix.values[0] - 1.0).abs() < 0.001);
        assert!((matrix.values[4] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_position() {
        let mut transform = Transform::new();
        transform.position = AnimatedProperty::static_value(KeyframeValue::Vec2([100.0, 50.0]));

        let pos = transform.position_at(0.0);
        assert_eq!(pos, [100.0, 50.0]);
    }

    #[test]
    fn test_scale_percentage() {
        let mut transform = Transform::new();
        transform.scale = AnimatedProperty::static_value(KeyframeValue::Vec2([50.0, 200.0]));

        let scale = transform.scale_at(0.0);
        assert_eq!(scale, [0.5, 2.0]);
    }

    #[test]
    fn test_rotation_to_radians() {
        let mut transform = Transform::new();
        transform.rotation = AnimatedProperty::static_value(KeyframeValue::Scalar(90.0));

        let rotation = transform.rotation_at(0.0);
        assert!((rotation - std::f32::consts::PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn test_opacity_clamping() {
        let mut transform = Transform::new();
        transform.opacity = AnimatedProperty::static_value(KeyframeValue::Scalar(150.0));

        let opacity = transform.opacity_at(0.0);
        assert_eq!(opacity, 1.0); // Clamped to 1.0
    }

    #[test]
    fn test_skew_is_negated_and_pinned() {
        let mut transform = Transform::new();
        transform.skew = Some(AnimatedProperty::static_value(KeyframeValue::Scalar(30.0)));

        let skew = transform.skew_at(0.0).unwrap();
        assert!((skew - (-30.0f32.to_radians())).abs() < 0.0001);

        // AE pins skew to +/-85 degrees.
        transform.skew = Some(AnimatedProperty::static_value(KeyframeValue::Scalar(200.0)));
        let skew = transform.skew_at(0.0).unwrap();
        assert!((skew - (-85.0f32.to_radians())).abs() < 0.0001);
    }

    #[test]
    fn test_from_shape_lottie_parses_tr_transform() {
        use crate::model::ShapeModel;

        let json = r#"{
            "ty":"tr",
            "a":{"a":0,"k":[5,5,0]},
            "p":{"a":0,"k":[50,60,0]},
            "s":{"a":0,"k":[50,50]},
            "r":{"a":0,"k":45},
            "o":{"a":0,"k":50}
        }"#;
        let model: ShapeModel = serde_json::from_str(json).unwrap();
        let transform = Transform::from_shape_lottie(&model);

        assert_eq!(transform.position_at(0.0), [50.0, 60.0]);
        assert_eq!(transform.anchor_at(0.0), [5.0, 5.0]);
        assert_eq!(transform.scale_at(0.0), [0.5, 0.5]);
        assert!((transform.rotation_at(0.0) - std::f32::consts::FRAC_PI_4).abs() < 0.001);
        assert_eq!(transform.opacity_at(0.0), 0.5);
    }

    #[test]
    fn test_is_animated() {
        let transform = Transform::new();
        assert!(!transform.is_animated()); // Static transform

        // Add animated property
        let mut animated = Transform::new();
        animated
            .position
            .add_keyframe(crate::keyframe::Keyframe::new(
                0.0,
                KeyframeValue::Vec2([0.0, 0.0]),
            ));
        animated
            .position
            .add_keyframe(crate::keyframe::Keyframe::new(
                10.0,
                KeyframeValue::Vec2([100.0, 100.0]),
            ));
        assert!(animated.is_animated());
    }
}
