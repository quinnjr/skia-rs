//! Path utility functions.
//!
//! This module provides utility functions for path manipulation,
//! including stroke-to-fill conversion.

use crate::flatten::{flatten_conic_adaptive, flatten_cubic_adaptive, flatten_quad_adaptive};
use crate::{Path, PathBuilder, PathElement};
use skia_rs_core::cast::{ceil_to_i32, scalar_from_i32};
use skia_rs_core::{Point, Scalar};

/// Tolerance for flattening curves to polylines before stroking.
const STROKE_FLATTEN_TOLERANCE: Scalar = 0.25;

/// Stroke cap style for stroke-to-fill conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum StrokeCap {
    /// Flat cap - no extension beyond the endpoint.
    #[default]
    Butt = 0,
    /// Round cap - semicircle at each endpoint.
    Round,
    /// Square cap - extends by half the stroke width.
    Square,
}

/// Stroke join style for stroke-to-fill conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum StrokeJoin {
    /// Miter join - sharp corners.
    #[default]
    Miter = 0,
    /// Round join - rounded corners.
    Round,
    /// Bevel join - flat corners.
    Bevel,
}

/// Parameters for stroke-to-fill conversion.
#[derive(Debug, Clone)]
pub struct StrokeParams {
    /// Stroke width.
    pub width: Scalar,
    /// Stroke cap style.
    pub cap: StrokeCap,
    /// Stroke join style.
    pub join: StrokeJoin,
    /// Miter limit (for miter joins).
    pub miter_limit: Scalar,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self {
            width: 1.0,
            cap: StrokeCap::Butt,
            join: StrokeJoin::Miter,
            miter_limit: 4.0,
        }
    }
}

impl StrokeParams {
    /// Create new stroke parameters.
    #[must_use]
    pub fn new(width: Scalar) -> Self {
        Self {
            width,
            ..Default::default()
        }
    }

    /// Set the stroke cap.
    #[must_use]
    pub const fn with_cap(mut self, cap: StrokeCap) -> Self {
        self.cap = cap;
        self
    }

    /// Set the stroke join.
    #[must_use]
    pub const fn with_join(mut self, join: StrokeJoin) -> Self {
        self.join = join;
        self
    }

    /// Set the miter limit.
    #[must_use]
    pub const fn with_miter_limit(mut self, limit: Scalar) -> Self {
        self.miter_limit = limit;
        self
    }
}

/// Convert a stroked path to a filled path.
///
/// This creates an outline around the input path that, when filled,
/// would produce the same visual result as stroking the original path.
///
/// # Arguments
/// * `path` - The input path to stroke.
/// * `params` - Stroke parameters (width, cap, join, miter limit).
///
/// # Returns
/// The stroked path as a fillable outline, or `None` if the path is empty.
pub fn stroke_to_fill(path: &Path, params: &StrokeParams) -> Option<Path> {
    if path.is_empty() || params.width <= 0.0 {
        return None;
    }

    let half_width = params.width / 2.0;
    let mut builder = PathBuilder::new();

    // Collect path elements into contours, tracking closed state per contour
    let mut contours: Vec<(Vec<Point>, bool)> = Vec::new();
    let mut current_contour: Vec<Point> = Vec::new();
    let mut current_closed = false;

    for element in path {
        match element {
            PathElement::Move(p) => {
                if !current_contour.is_empty() {
                    contours.push((std::mem::take(&mut current_contour), current_closed));
                }
                current_contour.push(p);
                current_closed = false;
            }
            PathElement::Line(p) => {
                current_contour.push(p);
            }
            PathElement::Quad(ctrl, end) => {
                // Flatten quadratic to lines with error-driven subdivision.
                if let Some(&start) = current_contour.last() {
                    flatten_quad_adaptive(
                        &mut current_contour,
                        start,
                        ctrl,
                        end,
                        STROKE_FLATTEN_TOLERANCE,
                    );
                }
            }
            PathElement::Cubic(ctrl1, ctrl2, end) => {
                // Flatten cubic to lines with error-driven subdivision.
                if let Some(&start) = current_contour.last() {
                    flatten_cubic_adaptive(
                        &mut current_contour,
                        start,
                        ctrl1,
                        ctrl2,
                        end,
                        STROKE_FLATTEN_TOLERANCE,
                    );
                }
            }
            PathElement::Conic(ctrl, end, weight) => {
                // Flatten conic via its exact rational form (not a broken
                // single-quad approximation), with error-driven subdivision.
                if let Some(&start) = current_contour.last() {
                    flatten_conic_adaptive(
                        &mut current_contour,
                        start,
                        ctrl,
                        end,
                        weight,
                        STROKE_FLATTEN_TOLERANCE,
                    );
                }
            }
            PathElement::Close => {
                current_closed = true;
            }
        }
    }

    if !current_contour.is_empty() {
        contours.push((current_contour, current_closed));
    }

    // Process each contour using its own closed state
    for (contour, is_closed) in &contours {
        stroke_contour(&mut builder, contour, *is_closed, half_width, params);
    }

    Some(builder.build())
}

/// Unit normal (left of travel direction) of the segment `a -> b`.
#[inline]
fn seg_normal(a: Point, b: Point) -> Point {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = dx.hypot(dy);
    if len > 0.0 {
        Point::new(-dy / len, dx / len)
    } else {
        Point::new(0.0, 1.0)
    }
}

/// The offset points wrapping around `vertex` at a 180-degree turn (incoming
/// normal `n1`, outgoing `n2` = -`n1`): a semicircle for round joins, else
/// a straight bevel (a miter is past any limit at 180 degrees).
fn fold_wrap(
    vertex: Point,
    n1: Point,
    n2: Point,
    half_width: Scalar,
    join: StrokeJoin,
) -> Vec<Point> {
    if join == StrokeJoin::Round {
        // Sweep from n1 through the incoming direction (n1 turned -90
        // degrees) to n2 = -n1.
        let start = n1.y.atan2(n1.x);
        (0..=8_u8)
            .map(|k| {
                let a = start - std::f32::consts::PI * Scalar::from(k) / 8.0;
                Point::new(
                    a.cos().mul_add(half_width, vertex.x),
                    a.sin().mul_add(half_width, vertex.y),
                )
            })
            .collect()
    } else {
        vec![
            Point::new(
                n1.x.mul_add(half_width, vertex.x),
                n1.y.mul_add(half_width, vertex.y),
            ),
            Point::new(
                n2.x.mul_add(half_width, vertex.x),
                n2.y.mul_add(half_width, vertex.y),
            ),
        ]
    }
}
/// Append the offset points for the join at `vertex` between incoming normal
/// `n1` and outgoing normal `n2` to the `left` (+normal) and `right` (-normal)
/// offset rings.
fn add_join(
    left: &mut Vec<Point>,
    right: &mut Vec<Point>,
    vertex: Point,
    n1: Point,
    n2: Point,
    half_width: Scalar,
    params: &StrokeParams,
) {
    let avg = Point::new(n1.x + n2.x, n1.y + n2.y);
    let avg_len = avg.length();

    if avg_len <= 0.001 {
        // A 180-degree turn: the stroke wraps around the vertex, as Skia's
        // joiner draws on the outer side (a semicircle for round joins; a
        // miter is past any limit here, so it bevels straight across). Both
        // rings get the wrap, the right one reversed, so a contour folding
        // back on itself (e.g. a closed two-point polygon) fills instead of
        // its two rings cancelling.
        let wrap = fold_wrap(vertex, n1, n2, half_width, params.join);
        left.extend(wrap.iter().copied());
        right.extend(wrap.iter().rev().copied());
        return;
    }

    let scale = half_width / avg_len;
    let offset = Point::new(avg.x * scale, avg.y * scale);

    match params.join {
        StrokeJoin::Miter => {
            let miter_len = 1.0 / (avg_len / 2.0);
            if miter_len <= params.miter_limit {
                left.push(Point::new(
                    offset.x.mul_add(miter_len, vertex.x),
                    offset.y.mul_add(miter_len, vertex.y),
                ));
                right.push(Point::new(
                    offset.x.mul_add(-miter_len, vertex.x),
                    offset.y.mul_add(-miter_len, vertex.y),
                ));
            } else {
                // Fall back to bevel.
                left.push(Point::new(
                    n1.x.mul_add(half_width, vertex.x),
                    n1.y.mul_add(half_width, vertex.y),
                ));
                left.push(Point::new(
                    n2.x.mul_add(half_width, vertex.x),
                    n2.y.mul_add(half_width, vertex.y),
                ));
                right.push(Point::new(
                    n1.x.mul_add(-half_width, vertex.x),
                    n1.y.mul_add(-half_width, vertex.y),
                ));
                right.push(Point::new(
                    n2.x.mul_add(-half_width, vertex.x),
                    n2.y.mul_add(-half_width, vertex.y),
                ));
            }
        }
        StrokeJoin::Bevel => {
            left.push(Point::new(
                n1.x.mul_add(half_width, vertex.x),
                n1.y.mul_add(half_width, vertex.y),
            ));
            left.push(Point::new(
                n2.x.mul_add(half_width, vertex.x),
                n2.y.mul_add(half_width, vertex.y),
            ));
            right.push(Point::new(
                n1.x.mul_add(-half_width, vertex.x),
                n1.y.mul_add(-half_width, vertex.y),
            ));
            right.push(Point::new(
                n2.x.mul_add(-half_width, vertex.x),
                n2.y.mul_add(-half_width, vertex.y),
            ));
        }
        StrokeJoin::Round => {
            let start_angle = (n1.y * half_width).atan2(n1.x * half_width);
            let end_angle = (n2.y * half_width).atan2(n2.x * half_width);
            let mut delta = end_angle - start_angle;
            if delta > std::f32::consts::PI {
                delta -= std::f32::consts::TAU;
            } else if delta < -std::f32::consts::PI {
                delta += std::f32::consts::TAU;
            }
            let n_segs = ceil_to_i32(delta.abs() / std::f32::consts::FRAC_PI_4).max(4);
            for k in 0..=n_segs {
                let t = scalar_from_i32(k) / scalar_from_i32(n_segs);
                let a = delta.mul_add(t, start_angle);
                left.push(Point::new(
                    a.cos().mul_add(half_width, vertex.x),
                    a.sin().mul_add(half_width, vertex.y),
                ));
                right.push(Point::new(
                    a.cos().mul_add(-half_width, vertex.x),
                    a.sin().mul_add(-half_width, vertex.y),
                ));
            }
        }
    }
}

/// Emit an offset ring as a closed contour, optionally reversed.
fn emit_ring(builder: &mut PathBuilder, ring: &[Point], reversed: bool) {
    if ring.is_empty() {
        return;
    }
    if reversed {
        let last = ring.len() - 1;
        builder.move_to(ring[last].x, ring[last].y);
        for p in ring[..last].iter().rev() {
            builder.line_to(p.x, p.y);
        }
    } else {
        builder.move_to(ring[0].x, ring[0].y);
        for p in &ring[1..] {
            builder.line_to(p.x, p.y);
        }
    }
    builder.close();
}

fn stroke_contour(
    builder: &mut PathBuilder,
    points: &[Point],
    is_closed: bool,
    half_width: Scalar,
    params: &StrokeParams,
) {
    // Drop consecutive duplicate points so segments are well-defined.
    let mut pts: Vec<Point> = Vec::with_capacity(points.len());
    for &p in points {
        if pts.last().is_none_or(|q: &Point| *q != p) {
            pts.push(p);
        }
    }
    // For closed contours, a trailing point equal to the start is redundant.
    if is_closed && pts.len() >= 2 && pts.first() == pts.last() {
        pts.pop();
    }

    if is_closed {
        stroke_closed(builder, &pts, half_width, params);
    } else {
        stroke_open(builder, &pts, half_width, params);
    }
}

/// Stroke a closed contour: strokes every segment including the closing edge,
/// joins at every vertex (including the start/end vertex), and emits the outer
/// ring forward and the inner ring reversed so winding cancels and the result
/// renders as a frame. Mirrors `SkPathStroker::close` + `reversePathTo`.
fn stroke_closed(
    builder: &mut PathBuilder,
    pts: &[Point],
    half_width: Scalar,
    params: &StrokeParams,
) {
    let n = pts.len();
    if n < 2 {
        return;
    }

    // Segment normals, including the closing segment pts[n-1] -> pts[0].
    let normals: Vec<Point> = (0..n)
        .map(|i| seg_normal(pts[i], pts[(i + 1) % n]))
        .collect();

    let mut left: Vec<Point> = Vec::with_capacity(n);
    let mut right: Vec<Point> = Vec::with_capacity(n);

    for i in 0..n {
        let n1 = normals[(i + n - 1) % n]; // incoming segment normal
        let n2 = normals[i]; // outgoing segment normal
        add_join(&mut left, &mut right, pts[i], n1, n2, half_width, params);
    }

    // Outer ring forward, inner ring reversed => opposite winding => frame.
    emit_ring(builder, &left, false);
    emit_ring(builder, &right, true);
}

/// Stroke an open contour into a single filled outline with end caps.
fn stroke_open(
    builder: &mut PathBuilder,
    pts: &[Point],
    half_width: Scalar,
    params: &StrokeParams,
) {
    let n = pts.len();
    if n < 2 {
        // Zero-length contour: round/square caps still paint a dot.
        if n == 1 {
            emit_cap_dot(builder, pts[0], half_width, params.cap);
        }
        return;
    }

    let normals: Vec<Point> = (0..n - 1).map(|i| seg_normal(pts[i], pts[i + 1])).collect();

    let mut left: Vec<Point> = Vec::with_capacity(n);
    let mut right: Vec<Point> = Vec::with_capacity(n);

    // First point.
    let first_normal = normals[0];
    left.push(Point::new(
        first_normal.x.mul_add(half_width, pts[0].x),
        first_normal.y.mul_add(half_width, pts[0].y),
    ));
    right.push(Point::new(
        first_normal.x.mul_add(-half_width, pts[0].x),
        first_normal.y.mul_add(-half_width, pts[0].y),
    ));

    for i in 1..n - 1 {
        add_join(
            &mut left,
            &mut right,
            pts[i],
            normals[i - 1],
            normals[i],
            half_width,
            params,
        );
    }

    // Last point.
    let last_normal = normals[normals.len() - 1];
    left.push(Point::new(
        last_normal.x.mul_add(half_width, pts[n - 1].x),
        last_normal.y.mul_add(half_width, pts[n - 1].y),
    ));
    right.push(Point::new(
        last_normal.x.mul_add(-half_width, pts[n - 1].x),
        last_normal.y.mul_add(-half_width, pts[n - 1].y),
    ));

    builder.move_to(left[0].x, left[0].y);
    add_cap(builder, pts[0], normals[0], half_width, params.cap, true);
    for p in &left {
        builder.line_to(p.x, p.y);
    }
    add_cap(
        builder,
        pts[n - 1],
        last_normal,
        half_width,
        params.cap,
        false,
    );
    for p in right.iter().rev() {
        builder.line_to(p.x, p.y);
    }
    builder.close();
}

/// Emit a cap-shaped dot for a zero-length contour (Round -> disc, Square ->
/// square, Butt -> nothing), matching Skia's zero-length-segment handling.
fn emit_cap_dot(builder: &mut PathBuilder, center: Point, half_width: Scalar, cap: StrokeCap) {
    match cap {
        StrokeCap::Butt => {}
        StrokeCap::Square => {
            builder.move_to(center.x - half_width, center.y - half_width);
            builder.line_to(center.x + half_width, center.y - half_width);
            builder.line_to(center.x + half_width, center.y + half_width);
            builder.line_to(center.x - half_width, center.y + half_width);
            builder.close();
        }
        StrokeCap::Round => {
            let steps: i32 = 16;
            builder.move_to(center.x + half_width, center.y);
            for i in 1..steps {
                let a = (scalar_from_i32(i) / scalar_from_i32(steps)) * std::f32::consts::TAU;
                builder.line_to(
                    a.cos().mul_add(half_width, center.x),
                    a.sin().mul_add(half_width, center.y),
                );
            }
            builder.close();
        }
    }
}

fn add_cap(
    builder: &mut PathBuilder,
    center: Point,
    normal: Point,
    half_width: Scalar,
    cap: StrokeCap,
    is_start: bool,
) {
    match cap {
        StrokeCap::Butt => {
            // No extension
        }
        StrokeCap::Square => {
            // Extend by half_width in the direction perpendicular to normal
            let dir = if is_start {
                Point::new(-normal.y, normal.x)
            } else {
                Point::new(normal.y, -normal.x)
            };
            let ext = Point::new(dir.x * half_width, dir.y * half_width);
            builder.line_to(
                normal.x.mul_add(half_width, center.x) + ext.x,
                normal.y.mul_add(half_width, center.y) + ext.y,
            );
            builder.line_to(
                normal.x.mul_add(-half_width, center.x) + ext.x,
                normal.y.mul_add(-half_width, center.y) + ext.y,
            );
        }
        StrokeCap::Round => {
            // Approximate semicircle with line segments
            let steps: i32 = 8;
            let start_angle = if is_start {
                normal.y.atan2(normal.x)
            } else {
                (-normal.y).atan2(-normal.x)
            };

            for i in 0..=steps {
                let t = scalar_from_i32(i) / scalar_from_i32(steps);
                let angle = t.mul_add(std::f32::consts::PI, start_angle);
                let x = angle.cos().mul_add(half_width, center.x);
                let y = angle.sin().mul_add(half_width, center.y);
                builder.line_to(x, y);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FillType;
    use skia_rs_core::Rect;

    #[test]
    fn test_stroked_rect_is_frame_with_empty_middle() {
        // Stroking a rect and filling the result with the winding rule must
        // yield a frame: the band under the original edges is filled and the
        // interior (and exterior) is empty. Verified via contains().
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
        let path = b.build();

        let params = StrokeParams::new(2.0); // half-width 1
        let mut stroked = stroke_to_fill(&path, &params).unwrap();
        stroked.set_fill_type(FillType::Winding);

        // A point on the stroked frame (near the left edge at x=0) is filled.
        assert!(
            stroked.contains(Point::new(0.0, 50.0)),
            "point on the stroked edge should be filled"
        );
        // The middle of the rect is empty (inner and outer rings cancel).
        assert!(
            !stroked.contains(Point::new(50.0, 50.0)),
            "the middle of a stroked rect must be empty, not a filled slab"
        );
        // A point well outside the outer ring is empty.
        assert!(
            !stroked.contains(Point::new(200.0, 200.0)),
            "outside the stroke must be empty"
        );
    }

    #[test]
    fn test_stroked_closed_triangle_is_frame() {
        // A closed triangle stroked then winding-filled must be hollow.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(100.0, 0.0);
        b.line_to(50.0, 100.0);
        b.close();
        let path = b.build();

        let params = StrokeParams::new(4.0);
        let mut stroked = stroke_to_fill(&path, &params).unwrap();
        stroked.set_fill_type(FillType::Winding);

        // Centroid is well inside -> must be empty (frame, not filled slab).
        assert!(
            !stroked.contains(Point::new(50.0, 33.0)),
            "interior of stroked closed triangle must be empty"
        );
    }

    #[test]
    fn test_stroke_to_fill_line() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();

        let params = StrokeParams::new(10.0);
        let stroked = stroke_to_fill(&path, &params).unwrap();

        assert!(!stroked.is_empty());
    }

    #[test]
    fn test_stroke_to_fill_triangle() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        builder.line_to(50.0, 100.0);
        builder.close();
        let path = builder.build();

        let params = StrokeParams::new(5.0).with_join(StrokeJoin::Round);
        let stroked = stroke_to_fill(&path, &params).unwrap();

        assert!(!stroked.is_empty());
    }

    #[test]
    fn test_stroke_to_fill_multi_contour_mixed_closed_open() {
        // Path with two contours: first closed (triangle), second open (line).
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0);
        builder.line_to(5.0, 10.0);
        builder.close();
        builder.move_to(20.0, 0.0);
        builder.line_to(30.0, 0.0);
        let path = builder.build();

        let params = StrokeParams::new(2.0);
        let result = stroke_to_fill(&path, &params);
        assert!(
            result.is_some(),
            "stroke_to_fill should succeed for valid input"
        );
        let stroked = result.unwrap();
        assert!(
            stroked.iter().count() > 0,
            "stroked path should not be empty"
        );
    }

    #[test]
    fn test_stroke_params() {
        let params = StrokeParams::new(2.0)
            .with_cap(StrokeCap::Round)
            .with_join(StrokeJoin::Bevel)
            .with_miter_limit(10.0);

        #[allow(
            clippy::float_cmp,
            reason = "exact test assertion, values round-trip literals"
        )]
        {
            assert_eq!(params.width, 2.0);
        }
        assert_eq!(params.cap, StrokeCap::Round);
        assert_eq!(params.join, StrokeJoin::Bevel);
        #[allow(
            clippy::float_cmp,
            reason = "exact test assertion, values round-trip literals"
        )]
        {
            assert_eq!(params.miter_limit, 10.0);
        }
    }

    #[test]
    fn test_stroke_to_fill_round_join_generates_arc() {
        // Two perpendicular lines meeting at right angle.
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(50.0, 0.0);
        builder.line_to(50.0, 50.0);
        let path = builder.build();

        let params = StrokeParams::new(10.0).with_join(StrokeJoin::Round);
        let result = stroke_to_fill(&path, &params);
        assert!(result.is_some());

        let stroked = result.unwrap();
        let count = stroked.iter().count();
        // Round join should produce more verbs than a straight-line join.
        // For 90-degree turn with default segment count, expect at least 8+ extra verbs
        // beyond the basic 5-6 a miter would emit.
        assert!(
            count > 10,
            "Round join should generate arc segments, got {count} verbs"
        );
    }
}
