//! Adaptive curve flattening utilities.
//!
//! Convert Bezier curves (quadratic, cubic, conic) into polylines with
//! adaptive tolerance based on chord-midpoint deviation. Used by `PathMeasure`,
//! `DashEffect`, `stroke_to_fill`, and boolean ops.

use skia_rs_core::{Point, Scalar};

const MAX_SUBDIVISION_DEPTH: u32 = 12;

/// Flatten a quadratic Bezier curve into line segments with given tolerance.
///
/// Appends points (excluding `start`) to the output vec. Subdivides until
/// the chord-midpoint deviation is below the tolerance.
pub fn flatten_quad_adaptive(
    output: &mut Vec<Point>,
    start: Point,
    ctrl: Point,
    end: Point,
    tolerance: Scalar,
) {
    flatten_quad_recursive(output, start, ctrl, end, tolerance, 0);
}

fn flatten_quad_recursive(
    output: &mut Vec<Point>,
    start: Point,
    ctrl: Point,
    end: Point,
    tolerance: Scalar,
    depth: u32,
) {
    let chord_mid = Point::new((start.x + end.x) * 0.5, (start.y + end.y) * 0.5);
    let dx = ctrl.x - chord_mid.x;
    let dy = ctrl.y - chord_mid.y;
    let dev_sq = dx * dx + dy * dy;
    let tol_sq = tolerance * tolerance;

    if depth >= MAX_SUBDIVISION_DEPTH || dev_sq < tol_sq * 4.0 {
        output.push(end);
        return;
    }

    let m1 = Point::new((start.x + ctrl.x) * 0.5, (start.y + ctrl.y) * 0.5);
    let m2 = Point::new((ctrl.x + end.x) * 0.5, (ctrl.y + end.y) * 0.5);
    let m12 = Point::new((m1.x + m2.x) * 0.5, (m1.y + m2.y) * 0.5);

    flatten_quad_recursive(output, start, m1, m12, tolerance, depth + 1);
    flatten_quad_recursive(output, m12, m2, end, tolerance, depth + 1);
}

/// Flatten a cubic Bezier curve into line segments with given tolerance.
///
/// Appends points (excluding `start`) to the output vec.
pub fn flatten_cubic_adaptive(
    output: &mut Vec<Point>,
    start: Point,
    ctrl1: Point,
    ctrl2: Point,
    end: Point,
    tolerance: Scalar,
) {
    flatten_cubic_recursive(output, start, ctrl1, ctrl2, end, tolerance, 0);
}

fn flatten_cubic_recursive(
    output: &mut Vec<Point>,
    start: Point,
    ctrl1: Point,
    ctrl2: Point,
    end: Point,
    tolerance: Scalar,
    depth: u32,
) {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let chord_len_sq = dx.mul_add(dx, dy * dy);

    let dev_of = |c: Point| -> Scalar {
        if chord_len_sq > 1e-9 {
            let t = (c.x - start.x).mul_add(dx, (c.y - start.y) * dy) / chord_len_sq;
            let proj_x = t.mul_add(dx, start.x);
            let proj_y = t.mul_add(dy, start.y);
            let pdx = c.x - proj_x;
            let pdy = c.y - proj_y;
            pdx.mul_add(pdx, pdy * pdy)
        } else {
            let pdx = c.x - start.x;
            let pdy = c.y - start.y;
            pdx.mul_add(pdx, pdy * pdy)
        }
    };

    let max_dev = dev_of(ctrl1).max(dev_of(ctrl2));
    let tol_sq = tolerance * tolerance;

    if depth >= MAX_SUBDIVISION_DEPTH || max_dev < tol_sq {
        output.push(end);
        return;
    }

    let m1 = Point::new((start.x + ctrl1.x) * 0.5, (start.y + ctrl1.y) * 0.5);
    let m2 = Point::new((ctrl1.x + ctrl2.x) * 0.5, (ctrl1.y + ctrl2.y) * 0.5);
    let m3 = Point::new((ctrl2.x + end.x) * 0.5, (ctrl2.y + end.y) * 0.5);
    let m12 = Point::new((m1.x + m2.x) * 0.5, (m1.y + m2.y) * 0.5);
    let m23 = Point::new((m2.x + m3.x) * 0.5, (m2.y + m3.y) * 0.5);
    let mid = Point::new((m12.x + m23.x) * 0.5, (m12.y + m23.y) * 0.5);

    flatten_cubic_recursive(output, start, m1, m12, mid, tolerance, depth + 1);
    flatten_cubic_recursive(output, mid, m23, m3, end, tolerance, depth + 1);
}

/// Flatten a conic (rational quadratic Bezier) into line segments.
///
/// Uses the rational form:
/// B(t) = (start*(1-t)² + 2*ctrl*w*t*(1-t) + end*t²) /
///        ((1-t)² + 2*w*t*(1-t) + t²)
pub fn flatten_conic_adaptive(
    output: &mut Vec<Point>,
    start: Point,
    ctrl: Point,
    end: Point,
    weight: Scalar,
    tolerance: Scalar,
) {
    flatten_conic_recursive(output, start, ctrl, end, weight, 0.0, 1.0, tolerance, 0);
}

fn eval_conic(start: Point, ctrl: Point, end: Point, w: Scalar, t: Scalar) -> Point {
    let mt = 1.0 - t;
    let denom = t.mul_add(t, (2.0 * w * t).mul_add(mt, mt * mt));
    let inv_denom = 1.0 / denom;
    Point::new(
        (end.x * t).mul_add(t, (start.x * mt).mul_add(mt, 2.0 * ctrl.x * w * t * mt)) * inv_denom,
        (end.y * t).mul_add(t, (start.y * mt).mul_add(mt, 2.0 * ctrl.y * w * t * mt)) * inv_denom,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "faithful port of the recursive conic flattening subdivision helper's parameter list (endpoints, weight, t-range, tolerance, depth)"
)]
fn flatten_conic_recursive(
    output: &mut Vec<Point>,
    start: Point,
    ctrl: Point,
    end: Point,
    weight: Scalar,
    t0: Scalar,
    t1: Scalar,
    tolerance: Scalar,
    depth: u32,
) {
    let p0 = eval_conic(start, ctrl, end, weight, t0);
    let p1 = eval_conic(start, ctrl, end, weight, t1);
    let tm = (t0 + t1) * 0.5;
    let pm = eval_conic(start, ctrl, end, weight, tm);
    let chord_mid = Point::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5);
    let dx = pm.x - chord_mid.x;
    let dy = pm.y - chord_mid.y;
    let dev_sq = dx * dx + dy * dy;
    let tol_sq = tolerance * tolerance;

    if depth >= MAX_SUBDIVISION_DEPTH || dev_sq < tol_sq {
        output.push(p1);
        return;
    }

    flatten_conic_recursive(
        output,
        start,
        ctrl,
        end,
        weight,
        t0,
        tm,
        tolerance,
        depth + 1,
    );
    flatten_conic_recursive(
        output,
        start,
        ctrl,
        end,
        weight,
        tm,
        t1,
        tolerance,
        depth + 1,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_quad_simple() {
        let mut output = Vec::new();
        flatten_quad_adaptive(
            &mut output,
            Point::new(0.0, 0.0),
            Point::new(50.0, 50.0),
            Point::new(100.0, 0.0),
            0.5,
        );
        assert!(!output.is_empty());
        assert_eq!(*output.last().unwrap(), Point::new(100.0, 0.0));
    }

    #[test]
    fn test_flatten_quad_straight_line() {
        let mut output = Vec::new();
        flatten_quad_adaptive(
            &mut output,
            Point::new(0.0, 0.0),
            Point::new(50.0, 0.0),
            Point::new(100.0, 0.0),
            0.5,
        );
        assert_eq!(
            output.len(),
            1,
            "Straight line should produce only endpoint"
        );
    }

    #[test]
    fn test_flatten_cubic_simple() {
        let mut output = Vec::new();
        flatten_cubic_adaptive(
            &mut output,
            Point::new(0.0, 0.0),
            Point::new(33.0, 50.0),
            Point::new(66.0, 50.0),
            Point::new(100.0, 0.0),
            0.5,
        );
        assert!(!output.is_empty());
        assert_eq!(*output.last().unwrap(), Point::new(100.0, 0.0));
    }

    #[test]
    fn test_flatten_conic_quarter_circle() {
        let mut output = Vec::new();
        flatten_conic_adaptive(
            &mut output,
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            std::f32::consts::FRAC_1_SQRT_2,
            0.01,
        );
        for p in &output {
            let r = p.x.hypot(p.y);
            assert!(
                (r - 1.0).abs() < 0.05,
                "Point not on unit circle: ({}, {})",
                p.x,
                p.y
            );
        }
        assert!(
            output.len() >= 4,
            "Expected subdivision, got {} points",
            output.len()
        );
    }
}
