//! Path builder for constructing paths.

#![allow(
    clippy::suboptimal_flops,
    reason = "faithful port of Skia's fma-free reference arithmetic; mul_add's fused rounding diverges from Skia"
)]

use crate::{FillType, Path, Verb};
use skia_rs_core::cast::{ceil_to_i32, scalar_from_i32};
use skia_rs_core::{Point, Rect, Scalar};

/// sqrt(2)/2, the conic weight for a 90-degree circular arc.
const OVAL_CONIC_WEIGHT: Scalar = std::f32::consts::FRAC_1_SQRT_2;

/// Builder for constructing paths.
#[derive(Debug, Clone, Default)]
pub struct PathBuilder {
    path: Path,
    last_move: Option<Point>,
}

impl PathBuilder {
    /// Create a new path builder.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a path builder with specified fill type.
    #[inline]
    #[must_use]
    pub fn with_fill_type(fill_type: FillType) -> Self {
        let mut builder = Self::new();
        builder.path.fill_type = fill_type;
        builder
    }

    /// Set the fill type.
    #[inline]
    pub const fn fill_type(&mut self, fill_type: FillType) -> &mut Self {
        self.path.fill_type = fill_type;
        self
    }

    /// Move to a point.
    ///
    /// Consecutive `move_to` calls overwrite the pending Move point instead of
    /// appending a redundant verb (upstream `SkPathBuilder::moveTo`).
    #[inline]
    pub fn move_to(&mut self, x: Scalar, y: Scalar) -> &mut Self {
        let p = Point::new(x, y);
        if self.path.verbs.last() == Some(&Verb::Move) {
            if let Some(last) = self.path.points.last_mut() {
                *last = p;
            }
        } else {
            self.path.verbs.push(Verb::Move);
            self.path.points.push(p);
        }
        self.last_move = Some(p);
        self.path.bounds = None;
        self
    }

    /// Line to a point.
    #[inline]
    pub fn line_to(&mut self, x: Scalar, y: Scalar) -> &mut Self {
        self.ensure_move();
        self.path.verbs.push(Verb::Line);
        self.path.points.push(Point::new(x, y));
        self.path.bounds = None;
        self
    }

    /// Quadratic bezier curve.
    #[inline]
    pub fn quad_to(&mut self, x1: Scalar, y1: Scalar, x2: Scalar, y2: Scalar) -> &mut Self {
        self.ensure_move();
        self.path.verbs.push(Verb::Quad);
        self.path.points.push(Point::new(x1, y1));
        self.path.points.push(Point::new(x2, y2));
        self.path.bounds = None;
        self
    }

    /// Conic curve (weighted quadratic).
    #[inline]
    pub fn conic_to(
        &mut self,
        x1: Scalar,
        y1: Scalar,
        x2: Scalar,
        y2: Scalar,
        w: Scalar,
    ) -> &mut Self {
        self.ensure_move();
        self.path.verbs.push(Verb::Conic);
        self.path.points.push(Point::new(x1, y1));
        self.path.points.push(Point::new(x2, y2));
        self.path.conic_weights.push(w);
        self.path.bounds = None;
        self
    }

    /// Cubic bezier curve.
    #[inline]
    pub fn cubic_to(
        &mut self,
        x1: Scalar,
        y1: Scalar,
        x2: Scalar,
        y2: Scalar,
        x3: Scalar,
        y3: Scalar,
    ) -> &mut Self {
        self.ensure_move();
        self.path.verbs.push(Verb::Cubic);
        self.path.points.push(Point::new(x1, y1));
        self.path.points.push(Point::new(x2, y2));
        self.path.points.push(Point::new(x3, y3));
        self.path.bounds = None;
        self
    }

    /// Close the current contour.
    ///
    /// A repeated `close()` is a no-op (upstream `SkPathBuilder::close`).
    #[inline]
    pub fn close(&mut self) -> &mut Self {
        if let Some(&last) = self.path.verbs.last()
            && last != Verb::Close
        {
            self.path.verbs.push(Verb::Close);
        }
        self
    }

    /// Add a rectangle.
    pub fn add_rect(&mut self, rect: &Rect) -> &mut Self {
        self.move_to(rect.left, rect.top)
            .line_to(rect.right, rect.top)
            .line_to(rect.right, rect.bottom)
            .line_to(rect.left, rect.bottom)
            .close()
    }

    /// Add an oval inscribed in the rectangle.
    ///
    /// Uses four quarter-circle conics of weight √2/2 (upstream
    /// `gFourQuarterCircleConics`), CW starting at the 3-o'clock point.
    pub fn add_oval(&mut self, rect: &Rect) -> &mut Self {
        let cx = f32::midpoint(rect.left, rect.right);
        let cy = f32::midpoint(rect.top, rect.bottom);
        let rx = rect.width() / 2.0;
        let ry = rect.height() / 2.0;
        let w = OVAL_CONIC_WEIGHT;

        self.move_to(cx + rx, cy)
            .conic_to(cx + rx, cy + ry, cx, cy + ry, w)
            .conic_to(cx - rx, cy + ry, cx - rx, cy, w)
            .conic_to(cx - rx, cy - ry, cx, cy - ry, w)
            .conic_to(cx + rx, cy - ry, cx + rx, cy, w)
            .close()
    }

    /// Add a circle.
    pub fn add_circle(&mut self, cx: Scalar, cy: Scalar, radius: Scalar) -> &mut Self {
        self.add_oval(&Rect::new(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
        ))
    }

    /// Add a rounded rectangle, like Skia's
    /// `SkPath::addRoundRect(rect, rx, ry, SkPathDirection::kCW)`:
    /// `SkRRect::setRectXY` followed by `addRRect` with its default start
    /// index 6.
    ///
    /// - Negative radii add nothing.
    /// - Non-finite radii are treated as zero.
    /// - Radii larger than half the rect are scaled down together, clamped
    ///   to the half-dimensions, and kept circular if they were equal.
    /// - With zero radii (or an empty rect) the result is a rectangle
    ///   starting at the bottom-left corner (`addRect` start index 3).
    /// - When the radii cover the whole rect the result is an oval starting
    ///   at the left-middle point (`addOval` start index 3).
    /// - Otherwise the contour starts at `(left, bottom - ry)`.
    ///
    /// All contours run clockwise and corners are exact quarter-ellipse
    /// conics (weight `sqrt(2)/2`), so dashes and trims line up with Skia.
    pub fn add_round_rect(&mut self, rect: &Rect, rx: Scalar, ry: Scalar) -> &mut Self {
        const W: Scalar = std::f32::consts::FRAC_1_SQRT_2;

        if rx < 0.0 || ry < 0.0 {
            return self;
        }

        // SkRRect::setRectXY (on the sorted rect).
        let (l, r) = (rect.left.min(rect.right), rect.left.max(rect.right));
        let (t, b) = (rect.top.min(rect.bottom), rect.top.max(rect.bottom));
        let (width, height) = (r - l, b - t);
        let (half_w, half_h) = (f32::midpoint(-l, r), f32::midpoint(-t, b));
        let (mut rx, mut ry) = if rx.is_finite() && ry.is_finite() {
            (rx, ry)
        } else {
            (0.0, 0.0)
        };
        #[allow(
            clippy::float_cmp,
            reason = "SkRRect::setRectXY's exact wasCircular test"
        )]
        let was_circular = rx == ry;
        if width < rx + rx || height < ry + ry {
            let scale = (width / (rx + rx)).min(height / (ry + ry));
            rx = (rx * scale).min(half_w);
            ry = (ry * scale).min(half_h);
            if was_circular {
                rx = rx.min(ry);
                ry = rx;
            }
        }

        if rx <= 0.0 || ry <= 0.0 || width <= 0.0 || height <= 0.0 {
            // Rect-type rrect: addRect with start index (6 + 1) / 2 = 3.
            return self
                .move_to(l, b)
                .line_to(l, t)
                .line_to(r, t)
                .line_to(r, b)
                .close();
        }

        if rx >= half_w && ry >= half_h {
            // Oval-type rrect: addOval with start index 6 / 2 = 3.
            let (cx, cy) = (f32::midpoint(l, r), f32::midpoint(t, b));
            return self
                .move_to(l, cy)
                .conic_to(l, t, cx, t, W)
                .conic_to(r, t, r, cy, W)
                .conic_to(r, b, cx, b, W)
                .conic_to(l, b, l, cy, W)
                .close();
        }

        self.move_to(l, b - ry)
            .line_to(l, t + ry)
            .conic_to(l, t, l + rx, t, W)
            .line_to(r - rx, t)
            .conic_to(r, t, r, t + ry, W)
            .line_to(r, b - ry)
            .conic_to(r, b, r - rx, b, W)
            .line_to(l + rx, b)
            .conic_to(l, b, l, b - ry, W)
            .close()
    }

    /// Add an arc as a new contour.
    ///
    /// The arc is inscribed in the oval bounded by `oval`, starting at `start_angle`
    /// and sweeping `sweep_angle` degrees. Angles are measured in degrees, with
    /// 0 at the 3 o'clock position, increasing clockwise.
    pub fn add_arc(&mut self, oval: &Rect, start_angle: Scalar, sweep_angle: Scalar) -> &mut Self {
        if oval.is_empty() || sweep_angle == 0.0 {
            return self;
        }

        // For a full sweep we can use the oval shortcut only when the start
        // angle is a multiple of 90 degrees (upstream SkPathBuilder::addArc).
        // Otherwise we honor the start angle by drawing the actual arc. The oval
        // helper only emits CW geometry, so the shortcut is limited to sweep > 0.
        if sweep_angle >= 360.0 || sweep_angle <= -360.0 {
            let start_over_90 = start_angle / 90.0;
            let start_over_90i = start_over_90.round();
            if (start_over_90 - start_over_90i).abs() < 1e-4 && sweep_angle > 0.0 {
                return self.add_oval(oval);
            }
        }

        let cx = oval.center().x;
        let cy = oval.center().y;
        let rx = oval.width() / 2.0;
        let ry = oval.height() / 2.0;

        // Convert to radians
        let start_rad = start_angle.to_radians();
        let sweep_rad = sweep_angle.to_radians();

        // Starting point honors the start angle; direction follows the sweep sign.
        let start_x = cx + rx * start_rad.cos();
        let start_y = cy + ry * start_rad.sin();
        self.move_to(start_x, start_y);

        // Add arc segments (approximate with cubics), no x-axis rotation.
        self.add_arc_to_impl(cx, cy, rx, ry, start_rad, sweep_rad, None);

        self
    }

    /// Arc to a point using radii and rotation.
    ///
    /// This matches the SVG arc command semantics.
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful port of the SVG elliptical-arc command's fixed parameter list (rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y)"
    )]
    pub fn arc_to(
        &mut self,
        rx: Scalar,
        ry: Scalar,
        x_axis_rotate: Scalar,
        large_arc: bool,
        sweep: bool,
        x: Scalar,
        y: Scalar,
    ) -> &mut Self {
        self.ensure_move();

        // Get current point
        let current = self.current_point();
        #[allow(
            clippy::float_cmp,
            reason = "exact degenerate-endpoint check mirrors upstream SkPath::ArcTo's bitwise comparison"
        )]
        if current.x == x && current.y == y {
            return self;
        }

        // Handle degenerate cases
        if rx == 0.0 || ry == 0.0 {
            return self.line_to(x, y);
        }

        // Convert to center parameterization and add cubics
        self.svg_arc_to_cubics(
            current.x,
            current.y,
            rx.abs(),
            ry.abs(),
            x_axis_rotate,
            large_arc,
            sweep,
            x,
            y,
        );

        self
    }

    /// Add relative line.
    pub fn r_line_to(&mut self, dx: Scalar, dy: Scalar) -> &mut Self {
        let current = self.current_point();
        self.line_to(current.x + dx, current.y + dy)
    }

    /// Add relative move.
    pub fn r_move_to(&mut self, dx: Scalar, dy: Scalar) -> &mut Self {
        let current = self.current_point();
        self.move_to(current.x + dx, current.y + dy)
    }

    /// Add relative quadratic.
    pub fn r_quad_to(&mut self, dx1: Scalar, dy1: Scalar, dx2: Scalar, dy2: Scalar) -> &mut Self {
        let current = self.current_point();
        self.quad_to(
            current.x + dx1,
            current.y + dy1,
            current.x + dx2,
            current.y + dy2,
        )
    }

    /// Add relative cubic.
    pub fn r_cubic_to(
        &mut self,
        dx1: Scalar,
        dy1: Scalar,
        dx2: Scalar,
        dy2: Scalar,
        dx3: Scalar,
        dy3: Scalar,
    ) -> &mut Self {
        let current = self.current_point();
        self.cubic_to(
            current.x + dx1,
            current.y + dy1,
            current.x + dx2,
            current.y + dy2,
            current.x + dx3,
            current.y + dy3,
        )
    }

    /// Add a line from the path.
    pub fn add_line(&mut self, p0: Point, p1: Point) -> &mut Self {
        self.move_to(p0.x, p0.y).line_to(p1.x, p1.y)
    }

    /// Add a polygon.
    pub fn add_polygon(&mut self, points: &[Point], close: bool) -> &mut Self {
        if points.is_empty() {
            return self;
        }

        self.move_to(points[0].x, points[0].y);
        for p in &points[1..] {
            self.line_to(p.x, p.y);
        }
        if close {
            self.close();
        }
        self
    }

    /// Add another path to this builder.
    pub fn add_path(&mut self, path: &Path) -> &mut Self {
        for element in path {
            match element {
                crate::PathElement::Move(p) => {
                    self.move_to(p.x, p.y);
                }
                crate::PathElement::Line(p) => {
                    self.line_to(p.x, p.y);
                }
                crate::PathElement::Quad(p1, p2) => {
                    self.quad_to(p1.x, p1.y, p2.x, p2.y);
                }
                crate::PathElement::Conic(p1, p2, w) => {
                    self.conic_to(p1.x, p1.y, p2.x, p2.y, w);
                }
                crate::PathElement::Cubic(p1, p2, p3) => {
                    self.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                }
                crate::PathElement::Close => {
                    self.close();
                }
            }
        }
        self
    }

    /// Build the path.
    #[inline]
    pub fn build(self) -> Path {
        self.path
    }

    /// Get the current point.
    ///
    /// After a `close()`, the current point returns to the subpath's initial
    /// (last move) point, matching Skia's `case 'Z': c = first`.
    pub fn current_point(&self) -> Point {
        if self.path.verbs.last() == Some(&Verb::Close)
            && let Some(mv) = self.last_move
        {
            return mv;
        }
        self.path.points.last().copied().unwrap_or(Point::zero())
    }

    /// Ensure we have an open contour to draw into.
    ///
    /// Ported from `SkPathBuilder::ensureMove`: when the path is empty, start at
    /// the origin; when the last verb is `Close`, inject a `moveTo` at the last
    /// move point so the drawing op starts a new contour there.
    fn ensure_move(&mut self) {
        match self.path.verbs.last() {
            None => {
                self.move_to(0.0, 0.0);
            }
            Some(&Verb::Close) => {
                let mv = self.last_move.unwrap_or(Point::zero());
                self.move_to(mv.x, mv.y);
            }
            _ => {}
        }
    }

    /// Internal helper to add arc segments as cubic beziers.
    ///
    /// The optional `rot` matrix (a rotation about the ellipse center) is
    /// applied to every emitted point, used to realize the SVG arc's
    /// x-axis-rotation.
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful port of Skia's addArcSegments helper geometry parameter list (center, radii, angles, rotation)"
    )]
    fn add_arc_to_impl(
        &mut self,
        cx: Scalar,
        cy: Scalar,
        rx: Scalar,
        ry: Scalar,
        start_angle: Scalar,
        sweep_angle: Scalar,
        rot: Option<&skia_rs_core::Matrix>,
    ) {
        // Break arc into segments of at most 90 degrees
        let num_segments = ceil_to_i32(sweep_angle.abs() / std::f32::consts::FRAC_PI_2).max(1);
        let segment_angle = sweep_angle / scalar_from_i32(num_segments);

        let mut angle = start_angle;
        for _ in 0..num_segments {
            let end_angle = angle + segment_angle;
            self.add_arc_segment(cx, cy, rx, ry, angle, end_angle, rot);
            angle = end_angle;
        }
    }

    /// Add a single arc segment (at most 90 degrees) as a cubic bezier.
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful port of Skia's addArcSegment helper geometry parameter list (center, radii, angles, rotation)"
    )]
    fn add_arc_segment(
        &mut self,
        cx: Scalar,
        cy: Scalar,
        rx: Scalar,
        ry: Scalar,
        start_angle: Scalar,
        end_angle: Scalar,
        rot: Option<&skia_rs_core::Matrix>,
    ) {
        let sweep = end_angle - start_angle;
        // A zero (or degenerate) sweep would divide by zero and produce NaN
        // control points; skip it (upstream early-returns on zero sweep).
        if sweep == 0.0 {
            return;
        }
        let half_sweep = sweep / 2.0;
        let sin_half = half_sweep.sin();
        if sin_half == 0.0 {
            return;
        }

        // Control point distance factor
        let k = (4.0 / 3.0) * (1.0 - half_sweep.cos()) / sin_half;

        let (sin_start, cos_start) = start_angle.sin_cos();
        let (sin_end, cos_end) = end_angle.sin_cos();

        let x0 = rx.mul_add(cos_start, cx);
        let y0 = ry.mul_add(sin_start, cy);
        let mut p1 = Point::new(
            (k * rx).mul_add(-sin_start, x0),
            (k * ry).mul_add(cos_start, y0),
        );
        let x3 = rx.mul_add(cos_end, cx);
        let y3 = ry.mul_add(sin_end, cy);
        let mut p3 = Point::new(x3, y3);
        let mut p2 = Point::new(
            (k * rx).mul_add(sin_end, x3),
            (k * ry).mul_add(-cos_end, y3),
        );

        if let Some(m) = rot {
            p1 = m.map_point(p1);
            p2 = m.map_point(p2);
            p3 = m.map_point(p3);
        }

        self.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
    }

    /// Convert SVG arc to cubic bezier segments.
    #[allow(
        clippy::too_many_arguments,
        reason = "faithful port of the W3C SVG arc-to-cubic conversion algorithm's fixed parameter list"
    )]
    fn svg_arc_to_cubics(
        &mut self,
        x1: Scalar,
        y1: Scalar,
        mut rx: Scalar,
        mut ry: Scalar,
        phi: Scalar,
        large_arc: bool,
        sweep: bool,
        x2: Scalar,
        y2: Scalar,
    ) {
        // Based on W3C SVG arc implementation notes
        let phi_rad = phi.to_radians();
        let (sin_phi, cos_phi) = phi_rad.sin_cos();

        // Step 1: Compute (x1', y1')
        let dx = (x1 - x2) / 2.0;
        let dy = (y1 - y2) / 2.0;
        let x1p = cos_phi.mul_add(dx, sin_phi * dy);
        let y1p = (-sin_phi).mul_add(dx, cos_phi * dy);

        // Scale radii if needed
        let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
        if lambda > 1.0 {
            let sqrt_lambda = lambda.sqrt();
            rx *= sqrt_lambda;
            ry *= sqrt_lambda;
        }

        // Step 2: Compute (cx', cy')
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let x1p2 = x1p * x1p;
        let y1p2 = y1p * y1p;

        let mut sq = (ry2.mul_add(-x1p2, rx2.mul_add(ry2, -(rx2 * y1p2)))
            / rx2.mul_add(y1p2, ry2 * x1p2))
        .max(0.0);
        sq = sq.sqrt();
        if large_arc == sweep {
            sq = -sq;
        }

        let cxp = sq * rx * y1p / ry;
        let cyp = -sq * ry * x1p / rx;

        // Step 3: Compute (cx, cy)
        let cx = cos_phi.mul_add(cxp, -(sin_phi * cyp)) + f32::midpoint(x1, x2);
        let cy = sin_phi.mul_add(cxp, cos_phi * cyp) + f32::midpoint(y1, y2);

        // Step 4: Compute angles
        let theta1 = angle_between(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry);
        let mut dtheta = angle_between(
            (x1p - cxp) / rx,
            (y1p - cyp) / ry,
            (-x1p - cxp) / rx,
            (-y1p - cyp) / ry,
        );

        if !sweep && dtheta > 0.0 {
            dtheta -= std::f32::consts::TAU;
        } else if sweep && dtheta < 0.0 {
            dtheta += std::f32::consts::TAU;
        }

        // The center parameterization above is computed in the ellipse-aligned
        // frame; the emitted geometry must be rotated by phi about the center so
        // it lies on the actual x-axis-rotated ellipse (SVG 2 B.2.3). Rotating
        // the control points is an exact affine map of the axis-aligned arc.
        let rot = skia_rs_core::Matrix::rotate_around(phi_rad, Point::new(cx, cy));
        self.add_arc_to_impl(cx, cy, rx, ry, theta1, dtheta, Some(&rot));
    }
}

/// Compute angle between two vectors.
fn angle_between(ux: Scalar, uy: Scalar, vx: Scalar, vy: Scalar) -> Scalar {
    let n = ux.hypot(uy) * vx.hypot(vy);
    if n == 0.0 {
        return 0.0;
    }
    let c = ux.mul_add(vx, uy * vy) / n;
    let s = ux.mul_add(vy, -(uy * vx));
    s.atan2(c.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PathElement, Verb};

    #[test]
    fn add_round_rect_matches_skia_add_rrect_contour() {
        // SkPath::addRRect(rrect, kCW) starts at index 6, (left, bottom - ry),
        // runs up the left edge, and alternates line / conic corner.
        let mut b = PathBuilder::new();
        b.add_round_rect(&Rect::from_xywh(10.0, 10.0, 100.0, 50.0), 10.0, 8.0);
        let path = b.build();

        let elements: Vec<PathElement> = path.iter().collect();
        assert_eq!(elements[0], PathElement::Move(Point::new(10.0, 52.0)));
        assert_eq!(elements[1], PathElement::Line(Point::new(10.0, 18.0)));
        let w = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(
            elements[2],
            PathElement::Conic(Point::new(10.0, 10.0), Point::new(20.0, 10.0), w)
        );
        let verbs: Vec<Verb> = path.verbs().to_vec();
        assert_eq!(
            verbs,
            [
                Verb::Move,
                Verb::Line,
                Verb::Conic,
                Verb::Line,
                Verb::Conic,
                Verb::Line,
                Verb::Conic,
                Verb::Line,
                Verb::Conic,
                Verb::Close,
            ]
        );
    }

    #[test]
    fn add_round_rect_with_full_radii_is_an_oval_from_the_left() {
        // An oval-type rrect becomes addOval(start index 3): four conics from
        // the left-middle point, clockwise, with no zero-length edges.
        let mut b = PathBuilder::new();
        b.add_round_rect(&Rect::from_xywh(0.0, 0.0, 40.0, 40.0), 20.0, 20.0);
        let path = b.build();
        assert_eq!(
            path.iter().next(),
            Some(PathElement::Move(Point::new(0.0, 20.0)))
        );
        assert_eq!(
            path.verbs(),
            [
                Verb::Move,
                Verb::Conic,
                Verb::Conic,
                Verb::Conic,
                Verb::Conic,
                Verb::Close,
            ]
        );
    }

    #[test]
    fn add_round_rect_handles_degenerate_radii_like_skia() {
        let rect = Rect::from_xywh(0.0, 0.0, 40.0, 20.0);
        let first = |rx: Scalar, ry: Scalar| {
            let mut b = PathBuilder::new();
            b.add_round_rect(&rect, rx, ry);
            b.build().iter().next()
        };

        // Negative radii add nothing (SkPath::addRoundRect).
        assert_eq!(first(-1.0, 5.0), None);

        // Non-finite radii become 0: a rect from the bottom-left corner,
        // with finite coordinates.
        for (rx, ry) in [
            (Scalar::INFINITY, 5.0),
            (Scalar::NAN, 5.0),
            (5.0, Scalar::INFINITY),
        ] {
            let mut b = PathBuilder::new();
            b.add_round_rect(&rect, rx, ry);
            let path = b.build();
            assert_eq!(
                path.iter().next(),
                Some(PathElement::Move(Point::new(0.0, 20.0))),
                "rx={rx} ry={ry}"
            );
            assert!(path.bounds().left.is_finite() && path.bounds().right.is_finite());
            assert_eq!(path.verbs()[1..4], [Verb::Line, Verb::Line, Verb::Line]);
        }
    }

    #[test]
    fn add_round_rect_scales_oversized_radii_uniformly() {
        // SkRRect::setRectXY scales both radii by the same factor: a 40x20
        // rect with radii (40, 40) gets min(40/80, 20/80) = 0.25 -> (10, 10).
        let mut b = PathBuilder::new();
        b.add_round_rect(&Rect::from_xywh(0.0, 0.0, 40.0, 20.0), 40.0, 40.0);
        let path = b.build();
        let first = path.iter().next();
        assert_eq!(first, Some(PathElement::Move(Point::new(0.0, 10.0))));
    }

    #[test]
    fn test_line_after_close_injects_move() {
        // After close(), a line_to must start a new contour at the last move point.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.close();
        b.line_to(20.0, 20.0);
        let path = b.build();
        assert_eq!(
            path.verbs(),
            &[Verb::Move, Verb::Line, Verb::Close, Verb::Move, Verb::Line]
        );
        // The injected move returns to the subpath initial point (0,0).
        let move_idx = 2; // point index of the injected Move
        assert_eq!(path.points()[move_idx], Point::new(0.0, 0.0));
    }

    #[test]
    fn test_repeated_close_is_noop() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.close();
        b.close();
        b.close();
        let path = b.build();
        assert_eq!(
            path.verbs().iter().filter(|v| **v == Verb::Close).count(),
            1
        );
    }

    #[test]
    fn test_consecutive_move_to_overwrites() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.move_to(5.0, 5.0);
        b.line_to(10.0, 10.0);
        let path = b.build();
        assert_eq!(path.verbs(), &[Verb::Move, Verb::Line]);
        assert_eq!(path.points()[0], Point::new(5.0, 5.0));
    }

    #[test]
    fn test_current_point_after_close_is_subpath_start() {
        let mut b = PathBuilder::new();
        b.move_to(3.0, 4.0);
        b.line_to(10.0, 0.0);
        b.close();
        assert_eq!(b.current_point(), Point::new(3.0, 4.0));
    }

    #[test]
    fn test_add_oval_uses_conics() {
        let mut b = PathBuilder::new();
        b.add_oval(&Rect::new(0.0, 0.0, 100.0, 50.0));
        let path = b.build();
        assert_eq!(
            path.verbs(),
            &[
                Verb::Move,
                Verb::Conic,
                Verb::Conic,
                Verb::Conic,
                Verb::Conic,
                Verb::Close
            ]
        );
        assert!(path.is_oval());
    }

    #[test]
    fn test_add_arc_zero_sweep_is_finite() {
        let mut b = PathBuilder::new();
        b.add_arc(&Rect::new(0.0, 0.0, 100.0, 100.0), 0.0, 0.0);
        let path = b.build();
        // Zero sweep must not produce NaN control points.
        for p in path.points() {
            assert!(
                p.x.is_finite() && p.y.is_finite(),
                "arc produced non-finite point"
            );
        }
    }

    #[test]
    fn test_add_arc_full_circle_honors_start_angle() {
        // Full circle starting at 45 degrees: start point must honor startAngle,
        // not snap to the 3-o'clock oval start.
        let oval = Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut b = PathBuilder::new();
        b.add_arc(&oval, 45.0, 360.0);
        let path = b.build();
        let start = path.points()[0];
        let (cx, cy) = (50.0, 50.0);
        let r = 50.0;
        let expected = Point::new(
            cx + r * 45.0_f32.to_radians().cos(),
            cy + r * 45.0_f32.to_radians().sin(),
        );
        assert!(
            (start.x - expected.x).abs() < 0.5 && (start.y - expected.y).abs() < 0.5,
            "full-circle arc start {start:?} should honor 45deg start {expected:?}"
        );
    }

    #[test]
    fn test_svg_arc_rotation_endpoint_lands_on_target() {
        // A rotated elliptical arc must land exactly on the target endpoint.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.arc_to(50.0, 20.0, 45.0, false, true, 40.0, 30.0);
        let path = b.build();
        let last = path.last_point().unwrap();
        assert!(
            (last.x - 40.0).abs() < 0.1 && (last.y - 30.0).abs() < 0.1,
            "rotated arc endpoint {last:?} must land on target (40, 30)"
        );
    }

    #[test]
    fn test_svg_arc_rotation_not_axis_aligned() {
        // With x-axis-rotation, intermediate geometry must differ from an
        // axis-aligned ellipse arc (the rotation must actually be applied).
        let mut rotated = PathBuilder::new();
        rotated.move_to(0.0, 0.0);
        rotated.arc_to(50.0, 20.0, 60.0, false, true, 40.0, 30.0);
        let rp = rotated.build();

        let mut plain = PathBuilder::new();
        plain.move_to(0.0, 0.0);
        plain.arc_to(50.0, 20.0, 0.0, false, true, 40.0, 30.0);
        let pp = plain.build();

        // Some interior control point must differ noticeably.
        let differ = rp
            .points()
            .iter()
            .zip(pp.points().iter())
            .any(|(a, b)| (a.x - b.x).abs() > 1.0 || (a.y - b.y).abs() > 1.0);
        assert!(differ, "rotation must change the emitted geometry");
    }
}
