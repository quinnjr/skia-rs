//! Shape layers for Lottie animations.
//!
//! This module handles shape layer elements:
//! - Rectangle
//! - Ellipse
//! - Path
//! - Polystar (star/polygon)
//! - Fill
//! - Stroke
//! - Gradient fill/stroke
//! - Group
//! - Trim paths
//! - Merge paths
//! - Repeater

#![allow(
    clippy::suboptimal_flops,
    reason = "faithful port of Skia's fma-free reference arithmetic; mul_add's fused rounding diverges from Skia"
)]

use crate::keyframe::{AnimatedProperty, KeyframeValue, PathData};
use crate::model::ShapeModel;
use crate::transform::Transform;
use skia_rs_core::{Color4f, Scalar};
use skia_rs_paint::{StrokeCap, StrokeJoin};
use skia_rs_path::{Path, PathBuilder};

/// Shape element types.
#[derive(Debug, Clone)]
pub enum Shape {
    /// Group of shapes.
    Group(ShapeGroup),
    /// Rectangle.
    Rectangle(RectangleShape),
    /// Ellipse.
    Ellipse(EllipseShape),
    /// Path.
    Path(PathShape),
    /// Polystar (star or polygon).
    Polystar(PolystarShape),
    /// Fill.
    Fill(FillShape),
    /// Stroke.
    Stroke(StrokeShape),
    /// Gradient fill.
    GradientFill(GradientFillShape),
    /// Gradient stroke.
    GradientStroke(GradientStrokeShape),
    /// Trim paths.
    TrimPath(TrimPathShape),
    /// Merge paths.
    MergePaths(MergePathsShape),
    /// Round corners.
    RoundCorners(RoundCornersShape),
    /// Repeater.
    Repeater(RepeaterShape),
    /// Transform.
    Transform(ShapeTransform),
}

impl Shape {
    /// Parse from Lottie shape model.
    #[must_use]
    pub fn from_lottie(model: &ShapeModel) -> Option<Self> {
        if model.hidden {
            return None;
        }

        match model.shape_type.as_str() {
            "gr" => Some(Self::Group(ShapeGroup::from_lottie(model))),
            "rc" => Some(Self::Rectangle(RectangleShape::from_lottie(model))),
            "el" => Some(Self::Ellipse(EllipseShape::from_lottie(model))),
            "sh" => Some(Self::Path(PathShape::from_lottie(model))),
            "sr" => Some(Self::Polystar(PolystarShape::from_lottie(model))),
            "fl" => Some(Self::Fill(FillShape::from_lottie(model))),
            "st" => Some(Self::Stroke(StrokeShape::from_lottie(model))),
            "gf" => Some(Self::GradientFill(GradientFillShape::from_lottie(model))),
            "gs" => Some(Self::GradientStroke(GradientStrokeShape::from_lottie(
                model,
            ))),
            "tm" => Some(Self::TrimPath(TrimPathShape::from_lottie(model))),
            "mm" => Some(Self::MergePaths(MergePathsShape::from_lottie(model))),
            "rd" => Some(Self::RoundCorners(RoundCornersShape::from_lottie(model))),
            "rp" => Some(Self::Repeater(RepeaterShape::from_lottie(model))),
            "tr" => Some(Self::Transform(ShapeTransform::from_lottie(model))),
            _ => None, // Unknown shape type
        }
    }
}

/// Group of shapes.
#[derive(Debug, Clone)]
pub struct ShapeGroup {
    /// Group name.
    pub name: String,
    /// Child shapes.
    pub shapes: Vec<Shape>,
    /// Group transform (usually the last "tr" item).
    pub transform: Option<Transform>,
}

impl ShapeGroup {
    /// Create a new empty group.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            shapes: Vec::new(),
            transform: None,
        }
    }

    /// Parse from Lottie model.
    #[must_use]
    pub fn from_lottie(model: &ShapeModel) -> Self {
        let mut group = Self::new(&model.name);

        for item in &model.items {
            if item.shape_type == "tr" {
                group.transform = Some(Transform::from_shape_lottie(item));
            } else if let Some(shape) = Shape::from_lottie(item) {
                group.shapes.push(shape);
            }
        }

        group
    }

    /// Build paths for this group at a specific frame.
    #[must_use]
    pub fn build_paths(&self, frame: Scalar) -> Vec<Path> {
        let mut paths = Vec::new();

        for shape in &self.shapes {
            match shape {
                Shape::Rectangle(rect) => {
                    if let Some(path) = rect.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Ellipse(ellipse) => {
                    if let Some(path) = ellipse.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Path(path_shape) => {
                    if let Some(path) = path_shape.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Polystar(star) => {
                    if let Some(path) = star.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Group(sub_group) => {
                    paths.extend(sub_group.build_paths(frame));
                }
                _ => {}
            }
        }

        paths
    }
}

/// Rectangle shape.
#[derive(Debug, Clone)]
pub struct RectangleShape {
    /// Shape name.
    pub name: String,
    /// Position.
    pub position: AnimatedProperty,
    /// Size.
    pub size: AnimatedProperty,
    /// Corner roundness.
    pub roundness: AnimatedProperty,
    /// Direction (1=clockwise, 3=counter-clockwise).
    pub direction: i32,
}

impl RectangleShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            position: model
                .position
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            size: model
                .size
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            roundness: model
                .roundness
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            direction: model
                .direction
                .as_ref()
                .and_then(super::model::DirectionOrDash::as_direction)
                .unwrap_or(1),
        }
    }

    /// Build a path at a specific frame.
    #[must_use]
    pub fn to_path(&self, frame: Scalar) -> Option<Path> {
        // Circular-arc corner constant (matches SkRRect's cubic-bezier
        // approximation of a quarter circle), not the coarser quadratic
        // approximation.
        const KAPPA: Scalar = 0.552_284_8;

        let pos = self
            .position
            .value_at(frame)
            .as_vec2()
            .unwrap_or([0.0, 0.0]);
        let size = self.size.value_at(frame).as_vec2().unwrap_or([0.0, 0.0]);
        let roundness = self.roundness.value_at(frame).as_scalar().unwrap_or(0.0);

        if size[0] <= 0.0 || size[1] <= 0.0 {
            return None;
        }

        let half_w = size[0] / 2.0;
        let half_h = size[1] / 2.0;
        let left = pos[0] - half_w;
        let top = pos[1] - half_h;

        let mut builder = PathBuilder::new();
        let right = left + size[0];
        let bottom = top + size[1];
        // "d": 3 => CCW, everything else (default 1) => CW, matching
        // upstream `RectangleGeometryAdapter` (Rectangle.cpp).
        let ccw = self.direction == 3;

        if roundness > 0.0 {
            let r = roundness.min(half_w).min(half_h);
            let k = r * KAPPA;

            builder.move_to(left + r, top);
            if ccw {
                builder.cubic_to(left + r - k, top, left, top + r - k, left, top + r);
                builder.line_to(left, bottom - r);
                builder.cubic_to(left, bottom - r + k, left + r - k, bottom, left + r, bottom);
                builder.line_to(right - r, bottom);
                builder.cubic_to(
                    right - r + k,
                    bottom,
                    right,
                    bottom - r + k,
                    right,
                    bottom - r,
                );
                builder.line_to(right, top + r);
                builder.cubic_to(right, top + r - k, right - r + k, top, right - r, top);
            } else {
                builder.line_to(right - r, top);
                builder.cubic_to(right - r + k, top, right, top + r - k, right, top + r);
                builder.line_to(right, bottom - r);
                builder.cubic_to(
                    right,
                    bottom - r + k,
                    right - r + k,
                    bottom,
                    right - r,
                    bottom,
                );
                builder.line_to(left + r, bottom);
                builder.cubic_to(left + r - k, bottom, left, bottom - r + k, left, bottom - r);
                builder.line_to(left, top + r);
                builder.cubic_to(left, top + r - k, left + r - k, top, left + r, top);
            }
        } else {
            builder.move_to(left, top);
            if ccw {
                builder.line_to(left, bottom);
                builder.line_to(right, bottom);
                builder.line_to(right, top);
            } else {
                builder.line_to(right, top);
                builder.line_to(right, bottom);
                builder.line_to(left, bottom);
            }
        }

        builder.close();
        Some(builder.build())
    }
}

/// Ellipse shape.
#[derive(Debug, Clone)]
pub struct EllipseShape {
    /// Shape name.
    pub name: String,
    /// Position.
    pub position: AnimatedProperty,
    /// Size.
    pub size: AnimatedProperty,
    /// Direction.
    pub direction: i32,
}

impl EllipseShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            position: model
                .position
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            size: model
                .size
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            direction: model
                .direction
                .as_ref()
                .and_then(super::model::DirectionOrDash::as_direction)
                .unwrap_or(1),
        }
    }

    /// Build a path at a specific frame.
    #[must_use]
    pub fn to_path(&self, frame: Scalar) -> Option<Path> {
        let pos = self
            .position
            .value_at(frame)
            .as_vec2()
            .unwrap_or([0.0, 0.0]);
        let size = self.size.value_at(frame).as_vec2().unwrap_or([0.0, 0.0]);

        if size[0] <= 0.0 || size[1] <= 0.0 {
            return None;
        }

        let rx = size[0] / 2.0;
        let ry = size[1] / 2.0;
        let cx = pos[0];
        let cy = pos[1];

        // Approximate ellipse with 4 cubic bezier curves
        let k = 0.552_284_8; // (4/3) * tan(π/8)
        let kx = rx * k;
        let ky = ry * k;

        let mut builder = PathBuilder::new();
        builder.move_to(cx + rx, cy);
        builder.cubic_to(cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry);
        builder.cubic_to(cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy);
        builder.cubic_to(cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry);
        builder.cubic_to(cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy);
        builder.close();

        Some(builder.build())
    }
}

/// Path shape (bezier path).
#[derive(Debug, Clone)]
pub struct PathShape {
    /// Shape name.
    pub name: String,
    /// Path data.
    pub path: AnimatedProperty,
    /// Direction.
    pub direction: i32,
}

impl PathShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            path: model
                .path
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            direction: model
                .direction
                .as_ref()
                .and_then(super::model::DirectionOrDash::as_direction)
                .unwrap_or(1),
        }
    }

    /// Build a path at a specific frame.
    #[must_use]
    pub fn to_path(&self, frame: Scalar) -> Option<Path> {
        let value = self.path.value_at(frame);

        match value {
            KeyframeValue::Path(path_data) => Some(path_data_to_path(&path_data)),
            _ => None,
        }
    }
}

/// Convert `PathData` to skia Path.
fn path_data_to_path(data: &PathData) -> Path {
    let mut builder = PathBuilder::new();

    if data.vertices.is_empty() {
        return builder.build();
    }

    let n = data.vertices.len();
    builder.move_to(data.vertices[0][0], data.vertices[0][1]);

    for i in 1..n {
        let prev = i - 1;
        let out_t = data.out_tangents.get(prev).copied().unwrap_or([0.0, 0.0]);
        let in_t = data.in_tangents.get(i).copied().unwrap_or([0.0, 0.0]);

        let c1 = [
            data.vertices[prev][0] + out_t[0],
            data.vertices[prev][1] + out_t[1],
        ];
        let c2 = [data.vertices[i][0] + in_t[0], data.vertices[i][1] + in_t[1]];

        if out_t == [0.0, 0.0] && in_t == [0.0, 0.0] {
            builder.line_to(data.vertices[i][0], data.vertices[i][1]);
        } else {
            builder.cubic_to(
                c1[0],
                c1[1],
                c2[0],
                c2[1],
                data.vertices[i][0],
                data.vertices[i][1],
            );
        }
    }

    if data.closed && n > 1 {
        let last = n - 1;
        let out_t = data.out_tangents.get(last).copied().unwrap_or([0.0, 0.0]);
        let in_t = data.in_tangents.first().copied().unwrap_or([0.0, 0.0]);

        let c1 = [
            data.vertices[last][0] + out_t[0],
            data.vertices[last][1] + out_t[1],
        ];
        let c2 = [data.vertices[0][0] + in_t[0], data.vertices[0][1] + in_t[1]];

        if !(out_t == [0.0, 0.0] && in_t == [0.0, 0.0]) {
            builder.cubic_to(
                c1[0],
                c1[1],
                c2[0],
                c2[1],
                data.vertices[0][0],
                data.vertices[0][1],
            );
        }
        builder.close();
    }

    builder.build()
}

/// Polystar shape (star or polygon).
#[derive(Debug, Clone)]
pub struct PolystarShape {
    /// Shape name.
    pub name: String,
    /// Position.
    pub position: AnimatedProperty,
    /// Number of points.
    pub points: AnimatedProperty,
    /// Outer radius.
    pub outer_radius: AnimatedProperty,
    /// Inner radius (for stars).
    pub inner_radius: AnimatedProperty,
    /// Outer roundness.
    pub outer_roundness: AnimatedProperty,
    /// Inner roundness (for stars).
    pub inner_roundness: AnimatedProperty,
    /// Star type (1=star, 2=polygon).
    pub star_type: i32,
    /// Rotation.
    pub rotation: AnimatedProperty,
    /// Direction.
    pub direction: i32,
}

impl PolystarShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            position: model
                .position
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            points: model
                .points
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            outer_radius: model
                .outer_radius
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            inner_radius: model
                .inner_radius
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            outer_roundness: AnimatedProperty::default(),
            inner_roundness: AnimatedProperty::default(),
            star_type: model.star_type.unwrap_or(1),
            rotation: model
                .roundness
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            direction: model
                .direction
                .as_ref()
                .and_then(super::model::DirectionOrDash::as_direction)
                .unwrap_or(1),
        }
    }

    /// Build a path at a specific frame.
    #[must_use]
    pub fn to_path(&self, frame: Scalar) -> Option<Path> {
        let pos = self
            .position
            .value_at(frame)
            .as_vec2()
            .unwrap_or([0.0, 0.0]);
        let points = self.points.value_at(frame).as_scalar().unwrap_or(5.0);
        let outer_r = self
            .outer_radius
            .value_at(frame)
            .as_scalar()
            .unwrap_or(100.0);
        let inner_r = self
            .inner_radius
            .value_at(frame)
            .as_scalar()
            .unwrap_or(50.0);
        let rotation = self.rotation.value_at(frame).as_scalar().unwrap_or(0.0);

        let n = skia_rs_core::cast::round_to_i32(points);
        if n < 3 {
            return None;
        }

        let mut builder = PathBuilder::new();
        let rot_rad = (rotation - 90.0).to_radians();

        let is_star = self.star_type == 1;
        let step_count = if is_star { n * 2 } else { n };
        let angle_step = std::f32::consts::TAU / skia_rs_core::cast::scalar_from_i32(step_count);

        for i in 0..step_count {
            let angle = rot_rad + angle_step * skia_rs_core::cast::scalar_from_i32(i);
            let radius = if is_star && i % 2 == 1 {
                inner_r
            } else {
                outer_r
            };

            let x = angle.cos().mul_add(radius, pos[0]);
            let y = angle.sin().mul_add(radius, pos[1]);

            if i == 0 {
                builder.move_to(x, y);
            } else {
                builder.line_to(x, y);
            }
        }

        builder.close();
        Some(builder.build())
    }
}

/// Fill shape.
#[derive(Debug, Clone)]
pub struct FillShape {
    /// Shape name.
    pub name: String,
    /// Color.
    pub color: AnimatedProperty,
    /// Opacity (0-100).
    pub opacity: AnimatedProperty,
    /// Fill rule (1=non-zero, 2=even-odd).
    pub fill_rule: i32,
}

impl FillShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            color: model
                .color
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            opacity: model.opacity.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(100.0)),
                AnimatedProperty::from_lottie,
            ),
            // "fl" shapes reuse the "r" key (roundness on "rc" shapes) for
            // fill rule: 1 = non-zero (default), 2 = even-odd.
            fill_rule: model
                .roundness
                .as_ref()
                .map(|v| AnimatedProperty::from_lottie(v).value_at(0.0))
                .and_then(|v| v.as_scalar())
                .map_or(1, skia_rs_core::cast::round_to_i32),
        }
    }

    /// Get the color at a specific frame.
    #[must_use]
    pub fn color_at(&self, frame: Scalar) -> Color4f {
        let c = self
            .color
            .value_at(frame)
            .as_color()
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let opacity = self.opacity.value_at(frame).as_scalar().unwrap_or(100.0) / 100.0;
        Color4f::new(c[0], c[1], c[2], c[3] * opacity)
    }

    /// Get the `skia_rs_path::FillType` for this fill's rule.
    #[must_use]
    pub const fn path_fill_type(&self) -> skia_rs_path::FillType {
        if self.fill_rule == 2 {
            skia_rs_path::FillType::EvenOdd
        } else {
            skia_rs_path::FillType::Winding
        }
    }
}

/// Stroke shape.
#[derive(Debug, Clone)]
pub struct StrokeShape {
    /// Shape name.
    pub name: String,
    /// Color.
    pub color: AnimatedProperty,
    /// Opacity (0-100).
    pub opacity: AnimatedProperty,
    /// Stroke width.
    pub width: AnimatedProperty,
    /// Line cap.
    pub line_cap: StrokeCap,
    /// Line join.
    pub line_join: StrokeJoin,
    /// Miter limit.
    pub miter_limit: Scalar,
    /// Dash pattern.
    pub dashes: Vec<AnimatedProperty>,
    /// Dash offset.
    pub dash_offset: AnimatedProperty,
}

impl StrokeShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        let line_cap = match model.line_cap.unwrap_or(2) {
            1 => StrokeCap::Butt,
            3 => StrokeCap::Square,
            _ => StrokeCap::Round,
        };

        let line_join = match model.line_join.unwrap_or(2) {
            1 => StrokeJoin::Miter,
            3 => StrokeJoin::Bevel,
            _ => StrokeJoin::Round,
        };

        Self {
            name: model.name.clone(),
            color: model
                .color
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            opacity: model.opacity.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(100.0)),
                AnimatedProperty::from_lottie,
            ),
            width: model.stroke_width.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(1.0)),
                AnimatedProperty::from_lottie,
            ),
            line_cap,
            line_join,
            miter_limit: model.miter_limit.unwrap_or(4.0),
            dashes: Self::parse_dashes(model),
            dash_offset: Self::parse_dash_offset(model),
        }
    }

    /// Parse the dash intervals from the "d" array (excluding the trailing
    /// offset element), matching upstream `DashAdapter`
    /// (`FillStroke.cpp`): an arbitrary number of on/off intervals followed
    /// by a single trailing offset entry.
    fn parse_dashes(model: &ShapeModel) -> Vec<AnimatedProperty> {
        let Some(elems) = model.direction.as_ref().and_then(|d| d.as_dashes()) else {
            return Vec::new();
        };
        if elems.len() <= 1 {
            return Vec::new();
        }
        elems[..elems.len() - 1]
            .iter()
            .map(|e| AnimatedProperty::from_lottie(&e.value))
            .collect()
    }

    /// Parse the trailing dash offset element.
    fn parse_dash_offset(model: &ShapeModel) -> AnimatedProperty {
        let Some(elems) = model.direction.as_ref().and_then(|d| d.as_dashes()) else {
            return AnimatedProperty::default();
        };
        match elems.last() {
            Some(last) if elems.len() > 1 => AnimatedProperty::from_lottie(&last.value),
            _ => AnimatedProperty::default(),
        }
    }

    /// Get the color at a specific frame.
    #[must_use]
    pub fn color_at(&self, frame: Scalar) -> Color4f {
        let c = self
            .color
            .value_at(frame)
            .as_color()
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let opacity = self.opacity.value_at(frame).as_scalar().unwrap_or(100.0) / 100.0;
        Color4f::new(c[0], c[1], c[2], c[3] * opacity)
    }

    /// Get the stroke width at a specific frame.
    #[must_use]
    pub fn width_at(&self, frame: Scalar) -> Scalar {
        self.width.value_at(frame).as_scalar().unwrap_or(1.0)
    }

    /// Get the resolved dash intervals (on/off pairs) at a frame, if dashed.
    #[must_use]
    pub fn dash_intervals_at(&self, frame: Scalar) -> Option<Vec<Scalar>> {
        if self.dashes.is_empty() {
            return None;
        }
        Some(
            self.dashes
                .iter()
                .map(|d| d.value_at(frame).as_scalar().unwrap_or(0.0))
                .collect(),
        )
    }

    /// Get the dash phase/offset at a frame.
    #[must_use]
    pub fn dash_offset_at(&self, frame: Scalar) -> Scalar {
        self.dash_offset.value_at(frame).as_scalar().unwrap_or(0.0)
    }
}

/// Gradient fill shape.
#[derive(Debug, Clone)]
pub struct GradientFillShape {
    /// Shape name.
    pub name: String,
    /// Gradient type (1=linear, 2=radial).
    pub gradient_type: i32,
    /// Start point.
    pub start_point: AnimatedProperty,
    /// End point.
    pub end_point: AnimatedProperty,
    /// Gradient colors and stops.
    pub colors: AnimatedProperty,
    /// Number of colors.
    pub color_count: i32,
    /// Opacity (0-100).
    pub opacity: AnimatedProperty,
    /// Radial gradient highlight length (-100..100, percent of start->end).
    pub highlight_length: AnimatedProperty,
    /// Radial gradient highlight angle (degrees).
    pub highlight_angle: AnimatedProperty,
}

impl GradientFillShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            gradient_type: model.gradient_type.unwrap_or(1),
            start_point: model
                .gradient_start
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            end_point: model
                .gradient_end
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            colors: model
                .gradient_colors
                .as_ref()
                .map(|gc| AnimatedProperty::from_lottie(&gc.colors))
                .unwrap_or_default(),
            color_count: model.gradient_colors.as_ref().map_or(2, |gc| gc.count),
            opacity: model.opacity.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(100.0)),
                AnimatedProperty::from_lottie,
            ),
            highlight_length: model
                .gradient_highlight_length
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            // "a" is reused for group anchor elsewhere; on "gf"/"gs" shapes
            // it's the radial gradient highlight angle.
            highlight_angle: model
                .transform
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
        }
    }

    /// Resolve the color stops at a frame.
    #[must_use]
    pub fn stops_at(&self, frame: Scalar) -> Vec<(Scalar, Color4f)> {
        resolve_gradient_stops(&self.colors, self.color_count, frame)
    }
}

/// Gradient stroke shape.
#[derive(Debug, Clone)]
pub struct GradientStrokeShape {
    /// Shape name.
    pub name: String,
    /// Gradient type (1=linear, 2=radial).
    pub gradient_type: i32,
    /// Start point.
    pub start_point: AnimatedProperty,
    /// End point.
    pub end_point: AnimatedProperty,
    /// Gradient colors and stops.
    pub colors: AnimatedProperty,
    /// Number of colors.
    pub color_count: i32,
    /// Opacity (0-100).
    pub opacity: AnimatedProperty,
    /// Stroke width.
    pub width: AnimatedProperty,
    /// Line cap.
    pub line_cap: StrokeCap,
    /// Line join.
    pub line_join: StrokeJoin,
    /// Radial gradient highlight length (-100..100, percent of start->end).
    pub highlight_length: AnimatedProperty,
    /// Radial gradient highlight angle (degrees).
    pub highlight_angle: AnimatedProperty,
}

impl GradientStrokeShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        let line_cap = match model.line_cap.unwrap_or(2) {
            1 => StrokeCap::Butt,
            3 => StrokeCap::Square,
            _ => StrokeCap::Round,
        };
        let line_join = match model.line_join.unwrap_or(2) {
            1 => StrokeJoin::Miter,
            3 => StrokeJoin::Bevel,
            _ => StrokeJoin::Round,
        };

        Self {
            name: model.name.clone(),
            gradient_type: model.gradient_type.unwrap_or(1),
            start_point: model
                .gradient_start
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            end_point: model
                .gradient_end
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            colors: model
                .gradient_colors
                .as_ref()
                .map(|gc| AnimatedProperty::from_lottie(&gc.colors))
                .unwrap_or_default(),
            color_count: model.gradient_colors.as_ref().map_or(2, |gc| gc.count),
            opacity: model.opacity.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(100.0)),
                AnimatedProperty::from_lottie,
            ),
            width: model.stroke_width.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(1.0)),
                AnimatedProperty::from_lottie,
            ),
            line_cap,
            line_join,
            highlight_length: model
                .gradient_highlight_length
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            highlight_angle: model
                .transform
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
        }
    }

    /// Resolve the color stops at a frame.
    #[must_use]
    pub fn stops_at(&self, frame: Scalar) -> Vec<(Scalar, Color4f)> {
        resolve_gradient_stops(&self.colors, self.color_count, frame)
    }
}

/// Resolve a Lottie gradient's consolidated stop table into `(t, color)`
/// pairs. The raw table holds `color_count` color records (`t,r,g,b`)
/// followed by an implicit number of opacity records (`t,a`); this merges
/// the two channel streams by position, matching (approximately) upstream
/// `GradientAdapter::onSync` (`Gradient.cpp`).
fn resolve_gradient_stops(
    colors: &AnimatedProperty,
    color_count: i32,
    frame: Scalar,
) -> Vec<(Scalar, Color4f)> {
    let value = colors.value_at(frame);
    let raw: Vec<Scalar> = value
        .as_float_array()
        .map(<[f32]>::to_vec)
        .or_else(|| value.as_color().map(|c| c.to_vec()))
        .unwrap_or_default();

    let c_count = usize::try_from(color_count).unwrap_or(0);
    let c_size = (c_count * 4).min(raw.len());
    let color_stops: Vec<(Scalar, [Scalar; 3])> = (0..c_count)
        .filter(|i| i * 4 + 3 < c_size)
        .map(|i| {
            let base = i * 4;
            (raw[base], [raw[base + 1], raw[base + 2], raw[base + 3]])
        })
        .collect();

    let opacity_raw = &raw[c_size..];
    let o_count = opacity_raw.len() / 2;
    let opacity_stops: Vec<(Scalar, Scalar)> = (0..o_count)
        .map(|i| (opacity_raw[i * 2], opacity_raw[i * 2 + 1]))
        .collect();

    if color_stops.is_empty() && opacity_stops.is_empty() {
        return Vec::new();
    }

    let sample_color = |t: Scalar| -> [Scalar; 3] {
        if color_stops.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        if t <= color_stops[0].0 {
            return color_stops[0].1;
        }
        for w in color_stops.windows(2) {
            if t <= w[1].0 {
                let f = if w[1].0 > w[0].0 {
                    (t - w[0].0) / (w[1].0 - w[0].0)
                } else {
                    0.0
                };
                return [
                    (w[1].1[0] - w[0].1[0]).mul_add(f, w[0].1[0]),
                    (w[1].1[1] - w[0].1[1]).mul_add(f, w[0].1[1]),
                    (w[1].1[2] - w[0].1[2]).mul_add(f, w[0].1[2]),
                ];
            }
        }
        color_stops.last().unwrap().1
    };
    let sample_opacity = |t: Scalar| -> Scalar {
        if opacity_stops.is_empty() {
            return 1.0;
        }
        if t <= opacity_stops[0].0 {
            return opacity_stops[0].1;
        }
        for w in opacity_stops.windows(2) {
            if t <= w[1].0 {
                let f = if w[1].0 > w[0].0 {
                    (t - w[0].0) / (w[1].0 - w[0].0)
                } else {
                    0.0
                };
                return (w[1].1 - w[0].1).mul_add(f, w[0].1);
            }
        }
        opacity_stops.last().unwrap().1
    };

    let mut positions: Vec<Scalar> = color_stops
        .iter()
        .map(|s| s.0)
        .chain(opacity_stops.iter().map(|s| s.0))
        .collect();
    // `total_cmp` (rather than `partial_cmp().unwrap()`) tolerates a NaN
    // offset reachable via degenerate keyframe interpolation instead of
    // panicking.
    positions.sort_by(f32::total_cmp);
    positions.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if positions.is_empty() {
        positions = vec![0.0, 1.0];
    }

    positions
        .into_iter()
        .map(|t| {
            let c = sample_color(t);
            let a = sample_opacity(t);
            (t, Color4f::new(c[0], c[1], c[2], a))
        })
        .collect()
}

/// Trim paths shape.
#[derive(Debug, Clone)]
pub struct TrimPathShape {
    /// Shape name.
    pub name: String,
    /// Start (0-100%).
    pub start: AnimatedProperty,
    /// End (0-100%).
    pub end: AnimatedProperty,
    /// Offset (degrees).
    pub offset: AnimatedProperty,
    /// Trim mode (1=simultaneously, 2=individually).
    pub mode: i32,
}

impl TrimPathShape {
    /// Parse from Lottie model.
    ///
    /// "tm" shapes reuse the generic `ShapeModel` fields: `size` ("s") is
    /// trim start, the dedicated `trim_end` ("e") field is trim end, and
    /// `opacity` ("o") is the offset (degrees) — matching how the shared
    /// struct is overloaded by shape type across this module.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            start: model
                .size
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            end: model.trim_end.as_ref().map_or_else(
                || AnimatedProperty::static_value(KeyframeValue::Scalar(100.0)),
                AnimatedProperty::from_lottie,
            ),
            offset: model
                .opacity
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
            mode: model.trim_mode.unwrap_or(1),
        }
    }

    /// Get the resolved `(start, end, inverted)` trim interval at a frame,
    /// both in `0..=1`, matching upstream `TrimEffectAdapter::onSync`
    /// (`modules/skottie/src/layers/shapelayer/TrimPaths.cpp`).
    #[must_use]
    pub fn resolved_at(&self, frame: Scalar) -> (Scalar, Scalar, bool) {
        let start = self.start.value_at(frame).as_scalar().unwrap_or(0.0) / 100.0;
        let end = self.end.value_at(frame).as_scalar().unwrap_or(100.0) / 100.0;
        let offset = self.offset.value_at(frame).as_scalar().unwrap_or(0.0) / 360.0;

        let mut start_t = start.min(end) + offset;
        let mut stop_t = start.max(end) + offset;
        let mut inverted = false;

        if stop_t - start_t < 1.0 {
            start_t -= start_t.floor();
            stop_t -= stop_t.floor();
            if start_t > stop_t {
                std::mem::swap(&mut start_t, &mut stop_t);
                inverted = true;
            }
        } else {
            start_t = 0.0;
            stop_t = 1.0;
        }

        (start_t, stop_t, inverted)
    }
}

/// Merge paths shape.
#[derive(Debug, Clone)]
pub struct MergePathsShape {
    /// Shape name.
    pub name: String,
    /// Merge mode (1=merge, 2=add, 3=subtract, 4=intersect, 5=exclude).
    pub mode: i32,
}

impl MergePathsShape {
    /// Parse from Lottie model.
    #[must_use]
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            mode: 1,
        }
    }
}

/// Round corners shape.
#[derive(Debug, Clone)]
pub struct RoundCornersShape {
    /// Shape name.
    pub name: String,
    /// Radius.
    pub radius: AnimatedProperty,
}

impl RoundCornersShape {
    /// Parse from Lottie model.
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            radius: model
                .roundness
                .as_ref()
                .map(AnimatedProperty::from_lottie)
                .unwrap_or_default(),
        }
    }
}

/// Repeater shape.
#[derive(Debug, Clone)]
pub struct RepeaterShape {
    /// Shape name.
    pub name: String,
    /// Number of copies.
    pub copies: AnimatedProperty,
    /// Offset.
    pub offset: AnimatedProperty,
    /// Transform for each copy.
    pub transform: Option<Transform>,
}

impl RepeaterShape {
    /// Parse from Lottie model.
    #[must_use]
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            copies: AnimatedProperty::static_value(KeyframeValue::Scalar(3.0)),
            offset: AnimatedProperty::default(),
            transform: None,
        }
    }
}

/// Shape transform (at end of group).
#[derive(Debug, Clone)]
pub struct ShapeTransform {
    /// Shape name.
    pub name: String,
    /// Transform.
    pub transform: Transform,
}

impl ShapeTransform {
    /// Parse from Lottie model.
    #[must_use]
    pub fn from_lottie(model: &ShapeModel) -> Self {
        Self {
            name: model.name.clone(),
            transform: Transform::from_shape_lottie(model),
        }
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
    fn test_resolve_gradient_stops_with_nan_offset_does_not_panic() {
        // A degenerate stop table with a NaN offset (reachable via
        // degenerate keyframe interpolation) must not panic when the
        // consolidated position list is sorted.
        let colors = AnimatedProperty::static_value(KeyframeValue::FloatArray(vec![
            f32::NAN,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]));
        let stops = resolve_gradient_stops(&colors, 2, 0.0);
        assert_eq!(stops.len(), 2);
    }

    #[test]
    fn test_rectangle_path() {
        let rect = RectangleShape {
            name: "test".to_string(),
            position: AnimatedProperty::static_value(KeyframeValue::Vec2([50.0, 50.0])),
            size: AnimatedProperty::static_value(KeyframeValue::Vec2([100.0, 80.0])),
            roundness: AnimatedProperty::static_value(KeyframeValue::Scalar(0.0)),
            direction: 1,
        };

        let path = rect.to_path(0.0).unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_ellipse_path() {
        let ellipse = EllipseShape {
            name: "test".to_string(),
            position: AnimatedProperty::static_value(KeyframeValue::Vec2([100.0, 100.0])),
            size: AnimatedProperty::static_value(KeyframeValue::Vec2([50.0, 30.0])),
            direction: 1,
        };

        let path = ellipse.to_path(0.0).unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_fill_color() {
        let fill = FillShape {
            name: "test".to_string(),
            color: AnimatedProperty::static_value(KeyframeValue::Color([1.0, 0.0, 0.0, 1.0])),
            opacity: AnimatedProperty::static_value(KeyframeValue::Scalar(50.0)),
            fill_rule: 1,
        };

        let color = fill.color_at(0.0);
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.0);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 0.5); // 50% opacity
    }

    #[test]
    fn test_fill_rule_even_odd_parses_from_r_field() {
        let json = r#"{"ty":"fl","r":2,"c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}"#;
        let model: ShapeModel = serde_json::from_str(json).unwrap();
        let fill = FillShape::from_lottie(&model);
        assert_eq!(fill.fill_rule, 2);
        assert_eq!(fill.path_fill_type(), skia_rs_path::FillType::EvenOdd);
    }

    #[test]
    fn test_stroke_dash_array_parses_intervals_and_offset() {
        // "d": [dash, gap, offset] - two intervals + trailing offset.
        let json = r#"{"ty":"st","c":{"a":0,"k":[0,0,0,1]},"w":{"a":0,"k":2},
            "d":[
                {"n":"d","nm":"dash","v":{"a":0,"k":10}},
                {"n":"g","nm":"gap","v":{"a":0,"k":5}},
                {"n":"o","nm":"offset","v":{"a":0,"k":3}}
            ]}"#;
        let model: ShapeModel = serde_json::from_str(json).unwrap();
        let stroke = StrokeShape::from_lottie(&model);

        let intervals = stroke.dash_intervals_at(0.0).unwrap();
        assert_eq!(intervals, vec![10.0, 5.0]);
        assert_eq!(stroke.dash_offset_at(0.0), 3.0);
    }

    #[test]
    fn test_rectangle_direction_field_not_confused_with_dash() {
        // "d": 3 on a rectangle is a direction, not a dash array.
        let json = r#"{"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[10,10]},"d":3}"#;
        let model: ShapeModel = serde_json::from_str(json).unwrap();
        let rect = RectangleShape::from_lottie(&model);
        assert_eq!(rect.direction, 3);
    }

    #[test]
    fn test_gradient_stops_resolve_color_and_opacity() {
        // 2 color stops (t,r,g,b) + 1 opacity stop (t,a): red@0, blue@1, alpha 0.5@0.
        let raw = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5];
        let colors = AnimatedProperty::static_value(KeyframeValue::FloatArray(raw));
        let stops = resolve_gradient_stops(&colors, 2, 0.0);

        assert!(stops.len() >= 2);
        assert_eq!(stops[0].0, 0.0);
        assert_eq!(stops[0].1.r, 1.0);
        assert_eq!(stops[0].1.a, 0.5);
        assert_eq!(stops.last().unwrap().1.b, 1.0);
    }

    #[test]
    fn test_rounded_rect_uses_cubic_arcs() {
        let rect = RectangleShape {
            name: "test".to_string(),
            position: AnimatedProperty::static_value(KeyframeValue::Vec2([50.0, 50.0])),
            size: AnimatedProperty::static_value(KeyframeValue::Vec2([100.0, 80.0])),
            roundness: AnimatedProperty::static_value(KeyframeValue::Scalar(10.0)),
            direction: 1,
        };
        let path = rect.to_path(0.0).unwrap();
        // Corners should be built from cubic curves, not quads.
        let has_cubic = path
            .iter()
            .any(|e| matches!(e, skia_rs_path::PathElement::Cubic(..)));
        let has_quad = path
            .iter()
            .any(|e| matches!(e, skia_rs_path::PathElement::Quad(..)));
        assert!(has_cubic);
        assert!(!has_quad);
    }
}
