//! Path boolean operations (union, intersect, difference, xor).
//!
//! This module implements boolean operations on paths by linearizing
//! each path into a polygon soup and dispatching to the `geo` crate's
//! sweep-line [`BooleanOps`] implementation, which correctly handles
//! concave polygons, holes, partial overlaps, and self-intersecting
//! input.
//!
//! [`BooleanOps`]: geo::BooleanOps

use crate::{Path, PathBuilder, PathElement};
use geo::{BooleanOps, Coord, LineString, MultiPolygon, Polygon as GeoPolygon};
use skia_rs_core::{Point, Rect, Scalar};

/// Operation type for path boolean operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PathOp {
    /// Subtract the second path from the first.
    Difference = 0,
    /// Intersect the two paths.
    Intersect,
    /// Union the two paths.
    Union,
    /// XOR the two paths (areas in one but not both).
    Xor,
    /// Reverse difference (subtract first from second).
    ReverseDifference,
}

/// Perform a boolean operation on two paths.
///
/// # Arguments
/// * `path1` - The first path
/// * `path2` - The second path
/// * `op` - The operation to perform
///
/// # Returns
/// The resulting path, or `None` if the operation fails.
///
/// # Algorithm
///
/// Both paths are linearized (Bezier and conic curves flattened to
/// polylines at a 0.5-unit tolerance), then the requested boolean
/// operation is performed via the [`geo`] crate's [`BooleanOps`]
/// sweep-line implementation. The resulting `MultiPolygon` is
/// converted back to a [`Path`].
///
/// Concave polygons, holes, and partial overlaps all produce
/// correct results. Self-intersecting input is handled by `geo`'s
/// robust intersection resolution. The previously-documented
/// limitations around Sutherland-Hodgman clipping and trivial-only
/// difference/xor no longer apply.
///
/// [`BooleanOps`]: geo::BooleanOps
pub fn op(path1: &Path, path2: &Path, op: PathOp) -> Option<Path> {
    Some(PathOps::new(path1, path2, op).compute())
}

/// Simplify a path by resolving self-intersections and overlaps.
///
/// Always runs the boolean-ops machinery (a self-union through geo's
/// sweep-line) rather than short-circuiting on the empty operand, so
/// self-intersecting or overlapping input is properly resolved.
pub fn simplify(path: &Path) -> Option<Path> {
    if path.is_empty() {
        return Some(Path::new());
    }
    let polys = path_to_polygons(path, 0.5);
    let simplified = polygon_union(&polys, &[]);
    Some(polygons_to_path(&simplified))
}

/// Internal path operations implementation.
struct PathOps<'a> {
    path1: &'a Path,
    path2: &'a Path,
    op: PathOp,
}

impl<'a> PathOps<'a> {
    const fn new(path1: &'a Path, path2: &'a Path, op: PathOp) -> Self {
        Self { path1, path2, op }
    }

    fn compute(&self) -> Path {
        // Handle empty paths
        if self.path1.is_empty() && self.path2.is_empty() {
            return Path::new();
        }

        if self.path1.is_empty() {
            return match self.op {
                PathOp::Union | PathOp::ReverseDifference | PathOp::Xor => self.path2.clone(),
                PathOp::Difference | PathOp::Intersect => Path::new(),
            };
        }

        if self.path2.is_empty() {
            return match self.op {
                PathOp::Union | PathOp::Difference | PathOp::Xor => self.path1.clone(),
                PathOp::Intersect | PathOp::ReverseDifference => Path::new(),
            };
        }

        // Check if bounding boxes intersect
        let bounds1 = self.path1.bounds();
        let bounds2 = self.path2.bounds();

        if !bounds_intersect(&bounds1, &bounds2) {
            return match self.op {
                PathOp::Union => {
                    // Combine both paths
                    let mut builder = PathBuilder::new();
                    Self::add_path_to_builder(&mut builder, self.path1);
                    Self::add_path_to_builder(&mut builder, self.path2);
                    builder.build()
                }
                PathOp::Intersect => Path::new(),
                PathOp::Difference => self.path1.clone(),
                PathOp::ReverseDifference => self.path2.clone(),
                PathOp::Xor => {
                    let mut builder = PathBuilder::new();
                    Self::add_path_to_builder(&mut builder, self.path1);
                    Self::add_path_to_builder(&mut builder, self.path2);
                    builder.build()
                }
            };
        }

        // For complex cases, use polygon-based operations
        self.compute_polygon_ops()
    }

    fn add_path_to_builder(builder: &mut PathBuilder, path: &Path) {
        for elem in path {
            match elem {
                PathElement::Move(p) => {
                    builder.move_to(p.x, p.y);
                }
                PathElement::Line(p) => {
                    builder.line_to(p.x, p.y);
                }
                PathElement::Quad(c, p) => {
                    builder.quad_to(c.x, c.y, p.x, p.y);
                }
                PathElement::Conic(c, p, w) => {
                    builder.conic_to(c.x, c.y, p.x, p.y, w);
                }
                PathElement::Cubic(c1, c2, p) => {
                    builder.cubic_to(c1.x, c1.y, c2.x, c2.y, p.x, p.y);
                }
                PathElement::Close => {
                    builder.close();
                }
            }
        }
    }

    fn compute_polygon_ops(&self) -> Path {
        // Convert paths to polygons (linearize curves)
        let polys1 = path_to_polygons(self.path1, 0.5);
        let polys2 = path_to_polygons(self.path2, 0.5);

        // Perform the boolean operation
        let result_polys = match self.op {
            PathOp::Union => polygon_union(&polys1, &polys2),
            PathOp::Intersect => polygon_intersect(&polys1, &polys2),
            PathOp::Difference => polygon_difference(&polys1, &polys2),
            PathOp::ReverseDifference => polygon_difference(&polys2, &polys1),
            PathOp::Xor => polygon_xor(&polys1, &polys2),
        };

        // Convert result back to path
        polygons_to_path(&result_polys)
    }
}

fn bounds_intersect(a: &Rect, b: &Rect) -> bool {
    a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top
}

/// A simple polygon represented as a list of points.
#[derive(Debug, Clone)]
struct Polygon {
    points: Vec<Point>,
    is_hole: bool,
}

impl Polygon {
    const fn new() -> Self {
        Self {
            points: Vec::new(),
            is_hole: false,
        }
    }

    fn add_point(&mut self, p: Point) {
        self.points.push(p);
    }

    fn is_empty(&self) -> bool {
        self.points.len() < 3
    }

    fn signed_area(&self) -> Scalar {
        if self.points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            area = self.points[i].x.mul_add(self.points[j].y, area);
            area = self.points[j].x.mul_add(-self.points[i].y, area);
        }
        area / 2.0
    }

    #[cfg(test)]
    fn contains_point(&self, p: Point) -> bool {
        if self.points.len() < 3 {
            return false;
        }

        let mut winding = 0;
        let n = self.points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let p1 = self.points[i];
            let p2 = self.points[j];

            if p1.y <= p.y {
                if p2.y > p.y {
                    // Upward crossing
                    if is_left(p1, p2, p) > 0.0 {
                        winding += 1;
                    }
                }
            } else if p2.y <= p.y {
                // Downward crossing
                if is_left(p1, p2, p) < 0.0 {
                    winding -= 1;
                }
            }
        }

        winding != 0
    }
}

#[cfg(test)]
fn is_left(p0: Point, p1: Point, p2: Point) -> Scalar {
    (p1.x - p0.x).mul_add(p2.y - p0.y, -((p2.x - p0.x) * (p1.y - p0.y)))
}

/// Convert a path to a list of polygons.
fn path_to_polygons(path: &Path, tolerance: Scalar) -> Vec<Polygon> {
    let mut polygons = Vec::new();
    let mut current_poly = Polygon::new();
    let mut current_point = Point::new(0.0, 0.0);
    let mut first_point = Point::new(0.0, 0.0);

    for elem in path {
        match elem {
            PathElement::Move(p) => {
                if !current_poly.is_empty() {
                    polygons.push(current_poly);
                }
                current_poly = Polygon::new();
                current_poly.add_point(p);
                current_point = p;
                first_point = p;
            }
            PathElement::Line(p) => {
                current_poly.add_point(p);
                current_point = p;
            }
            PathElement::Quad(c, p) => {
                // Linearize quadratic bezier
                linearize_quad(&mut current_poly, current_point, c, p, tolerance);
                current_point = p;
            }
            PathElement::Conic(c, p, w) => {
                let pts = linearize_conic(current_point, c, p, w, tolerance);
                current_poly.points.extend_from_slice(&pts[1..]);
                current_point = p;
            }
            PathElement::Cubic(c1, c2, p) => {
                // Linearize cubic bezier
                linearize_cubic(&mut current_poly, current_point, c1, c2, p, tolerance);
                current_point = p;
            }
            PathElement::Close => {
                if !current_poly.is_empty() {
                    // Determine if this is a hole based on winding
                    current_poly.is_hole = current_poly.signed_area() < 0.0;
                    polygons.push(current_poly);
                }
                current_poly = Polygon::new();
                current_point = first_point;
            }
        }
    }

    if !current_poly.is_empty() {
        current_poly.is_hole = current_poly.signed_area() < 0.0;
        polygons.push(current_poly);
    }

    polygons
}

fn linearize_quad(poly: &mut Polygon, p0: Point, p1: Point, p2: Point, tolerance: Scalar) {
    // Check if curve is flat enough
    let d = distance_to_line(p1, p0, p2);
    if d < tolerance {
        poly.add_point(p2);
    } else {
        // Subdivide
        let q0 = p0.lerp(p1, 0.5);
        let q1 = p1.lerp(p2, 0.5);
        let r = q0.lerp(q1, 0.5);

        linearize_quad(poly, p0, q0, r, tolerance);
        linearize_quad(poly, r, q1, p2, tolerance);
    }
}

/// Linearize a conic (rational quadratic Bezier) into line segments.
fn linearize_conic(
    start: Point,
    ctrl: Point,
    end: Point,
    weight: Scalar,
    tolerance: Scalar,
) -> Vec<Point> {
    let mut output = vec![start];
    crate::flatten::flatten_conic_adaptive(&mut output, start, ctrl, end, weight, tolerance);
    output
}

fn linearize_cubic(
    poly: &mut Polygon,
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    tolerance: Scalar,
) {
    // Check if curve is flat enough
    let d1 = distance_to_line(p1, p0, p3);
    let d2 = distance_to_line(p2, p0, p3);
    if d1.max(d2) < tolerance {
        poly.add_point(p3);
    } else {
        // Subdivide using de Casteljau's algorithm
        let q0 = p0.lerp(p1, 0.5);
        let q1 = p1.lerp(p2, 0.5);
        let q2 = p2.lerp(p3, 0.5);
        let r0 = q0.lerp(q1, 0.5);
        let r1 = q1.lerp(q2, 0.5);
        let s = r0.lerp(r1, 0.5);

        linearize_cubic(poly, p0, q0, r0, s, tolerance);
        linearize_cubic(poly, s, r1, q2, p3, tolerance);
    }
}

fn distance_to_line(p: Point, line_start: Point, line_end: Point) -> Scalar {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let len_sq = dx.mul_add(dx, dy * dy);

    if len_sq < 1e-10 {
        return p.distance(&line_start);
    }

    let cross = (p.x - line_start.x).mul_add(dy, -((p.y - line_start.y) * dx));
    cross.abs() / len_sq.sqrt()
}

/// Convert our internal polygon list into a `geo::MultiPolygon`.
///
/// Our [`Polygon`] type carries an `is_hole` flag determined from the
/// signed area after flattening. Shells (non-holes) open a new
/// [`geo::Polygon`]; subsequent holes are attached to the most-recent
/// shell until the next shell is seen. This matches how
/// [`path_to_polygons`] emits polygons — each `Close` pushes one
/// polygon, with hole-ness derived from winding direction.
fn skia_polygons_to_geo(polys: &[Polygon]) -> MultiPolygon<f64> {
    let mut shells: Vec<GeoPolygon<f64>> = Vec::new();
    let mut pending_shell: Option<Vec<Coord<f64>>> = None;
    let mut pending_holes: Vec<LineString<f64>> = Vec::new();

    for poly in polys {
        if poly.is_empty() {
            continue;
        }
        let coords: Vec<Coord<f64>> = poly
            .points
            .iter()
            .map(|p| Coord {
                x: f64::from(p.x),
                y: f64::from(p.y),
            })
            .collect();

        if poly.is_hole {
            pending_holes.push(LineString::new(coords));
        } else {
            if let Some(shell) = pending_shell.take() {
                shells.push(GeoPolygon::new(
                    LineString::new(shell),
                    std::mem::take(&mut pending_holes),
                ));
            }
            pending_shell = Some(coords);
        }
    }

    if let Some(shell) = pending_shell {
        shells.push(GeoPolygon::new(LineString::new(shell), pending_holes));
    } else if !pending_holes.is_empty() {
        // Orphan holes without a shell — treat each as a shell with
        // its natural winding; geo will normalize as needed during
        // bool ops. This avoids losing geometry when input has no
        // explicit outer contour.
        for hole in pending_holes {
            shells.push(GeoPolygon::new(hole, Vec::new()));
        }
    }

    MultiPolygon(shells)
}

/// Convert a `geo::MultiPolygon` back into our internal polygon list,
/// emitting each exterior ring as a shell followed by its interior
/// rings as holes. The caller's [`polygons_to_path`] writes each
/// polygon as a separate subpath, so the ordering here matters:
/// shells come first, then the matching holes, and `is_hole` is set
/// correctly so downstream consumers can distinguish them.
#[allow(
    clippy::cast_possible_truncation,
    reason = "narrowing f64 (geo crate's coordinate type) back to Scalar (f32); path coordinates are f32 throughout, so this is an unavoidable boundary conversion at the boolean-ops adapter"
)]
fn geo_to_skia_polygons(mp: &MultiPolygon<f64>) -> Vec<Polygon> {
    let mut out = Vec::new();
    for poly in mp.iter() {
        let exterior: Vec<Point> = poly
            .exterior()
            .coords()
            .map(|c| Point::new(c.x as f32, c.y as f32))
            .collect();
        if exterior.len() >= 3 {
            out.push(Polygon {
                points: exterior,
                is_hole: false,
            });
        }
        for hole in poly.interiors() {
            let hole_pts: Vec<Point> = hole
                .coords()
                .map(|c| Point::new(c.x as f32, c.y as f32))
                .collect();
            if hole_pts.len() >= 3 {
                out.push(Polygon {
                    points: hole_pts,
                    is_hole: true,
                });
            }
        }
    }
    out
}

/// Union of two polygon sets via geo's sweep-line boolean ops.
fn polygon_union(polys1: &[Polygon], polys2: &[Polygon]) -> Vec<Polygon> {
    let a = skia_polygons_to_geo(polys1);
    let b = skia_polygons_to_geo(polys2);
    geo_to_skia_polygons(&a.union(&b))
}

/// Intersection of two polygon sets via geo's sweep-line boolean ops.
fn polygon_intersect(polys1: &[Polygon], polys2: &[Polygon]) -> Vec<Polygon> {
    let a = skia_polygons_to_geo(polys1);
    let b = skia_polygons_to_geo(polys2);
    geo_to_skia_polygons(&a.intersection(&b))
}

/// Difference of two polygon sets (polys1 - polys2) via geo's
/// sweep-line boolean ops.
fn polygon_difference(polys1: &[Polygon], polys2: &[Polygon]) -> Vec<Polygon> {
    let a = skia_polygons_to_geo(polys1);
    let b = skia_polygons_to_geo(polys2);
    geo_to_skia_polygons(&a.difference(&b))
}

/// Symmetric difference (XOR) via geo's sweep-line boolean ops.
fn polygon_xor(polys1: &[Polygon], polys2: &[Polygon]) -> Vec<Polygon> {
    let a = skia_polygons_to_geo(polys1);
    let b = skia_polygons_to_geo(polys2);
    geo_to_skia_polygons(&a.xor(&b))
}

/// Convert polygons back to a path.
fn polygons_to_path(polygons: &[Polygon]) -> Path {
    let mut builder = PathBuilder::new();

    for poly in polygons {
        if poly.points.len() < 3 {
            continue;
        }

        builder.move_to(poly.points[0].x, poly.points[0].y);
        for p in &poly.points[1..] {
            builder.line_to(p.x, p.y);
        }
        builder.close();
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_paths() {
        let empty = Path::new();
        let result = op(&empty, &empty, PathOp::Union);
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_simplify_resolves_self_intersection() {
        // A bowtie/figure-eight self-intersects at (5,5). simplify() must run
        // the ops machinery to resolve it, not short-circuit and return the
        // input unchanged.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 10.0);
        b.line_to(10.0, 0.0);
        b.line_to(0.0, 10.0);
        b.close();
        let bowtie = b.build();

        let result = simplify(&bowtie).expect("simplify should succeed");
        assert!(!result.is_empty(), "simplified bowtie must not be empty");
        // The machinery ran: the resolved geometry differs from the raw input.
        assert_ne!(result, bowtie, "simplify must resolve, not clone the input");
        // A point well inside the right lobe stays filled.
        assert!(
            result.contains(Point::new(8.0, 5.0)),
            "interior of a lobe should remain filled after simplify"
        );
    }

    #[test]
    fn test_simplify_empty_is_empty() {
        let result = simplify(&Path::new()).expect("simplify of empty is Some");
        assert!(result.is_empty());
    }

    #[test]
    fn test_union_non_overlapping() {
        let mut builder1 = PathBuilder::new();
        builder1.add_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0));
        let path1 = builder1.build();

        let mut builder2 = PathBuilder::new();
        builder2.add_rect(&Rect::from_xywh(20.0, 0.0, 10.0, 10.0));
        let path2 = builder2.build();

        let result = op(&path1, &path2, PathOp::Union);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_intersect_non_overlapping() {
        let mut builder1 = PathBuilder::new();
        builder1.add_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0));
        let path1 = builder1.build();

        let mut builder2 = PathBuilder::new();
        builder2.add_rect(&Rect::from_xywh(20.0, 0.0, 10.0, 10.0));
        let path2 = builder2.build();

        let result = op(&path1, &path2, PathOp::Intersect);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_path_to_polygons_handles_conic_weight() {
        // Quarter circle: M (1,0), conic to (1,1) end (0,1) with weight sqrt(2)/2.
        // Close back through the origin so the polygon has enough points to be kept.
        // Linearized points along the arc should lie on the unit circle.
        let mut builder = PathBuilder::new();
        builder.move_to(1.0, 0.0);
        builder.conic_to(1.0, 1.0, 0.0, 1.0, std::f32::consts::FRAC_1_SQRT_2);
        builder.line_to(0.0, 0.0);
        builder.close();
        let path = builder.build();

        // Use a tight tolerance so the arc gets subdivided finely.
        let polygons = path_to_polygons(&path, 0.01);
        assert!(!polygons.is_empty(), "Should produce at least one polygon");

        // Filter to points on the arc portion (exclude origin and axes endpoints
        // that are trivially correct). Check every point that is NOT (0,0):
        // arc points should be near the unit circle, axis points are at radius 1
        // already, and the origin is at radius 0 — skip it.
        let polygon = &polygons[0];
        let mut arc_point_count = 0;
        for p in &polygon.points {
            let dist_sq = p.x.mul_add(p.x, p.y * p.y);
            if dist_sq < 0.5 {
                // Origin — skip
                continue;
            }
            arc_point_count += 1;
            assert!(
                (dist_sq - 1.0).abs() < 0.05,
                "Point ({}, {}) should be near unit circle (dist² = {})",
                p.x,
                p.y,
                dist_sq
            );
        }
        // Verify we actually got subdivided arc points, not just endpoints.
        assert!(
            arc_point_count >= 4,
            "Expected at least 4 arc points from subdivision, got {arc_point_count}"
        );
    }

    #[test]
    fn test_polygon_contains_point() {
        let mut poly = Polygon::new();
        poly.add_point(Point::new(0.0, 0.0));
        poly.add_point(Point::new(10.0, 0.0));
        poly.add_point(Point::new(10.0, 10.0));
        poly.add_point(Point::new(0.0, 10.0));

        assert!(poly.contains_point(Point::new(5.0, 5.0)));
        assert!(!poly.contains_point(Point::new(15.0, 5.0)));
    }

    /// Walk every shell polygon in `polys` and test whether `probe`
    /// lies inside any of them (ignoring holes). Used by the
    /// correctness tests below to check the output of a boolean op.
    fn polys_contain_probe(polys: &[Polygon], probe: Point) -> bool {
        polys.iter().any(|p| !p.is_hole && p.contains_point(probe))
    }

    /// Build a square subpath from top-left corner + side length.
    fn rect_path(x: f32, y: f32, side: f32) -> Path {
        let mut b = PathBuilder::new();
        b.move_to(x, y);
        b.line_to(x + side, y);
        b.line_to(x + side, y + side);
        b.line_to(x, y + side);
        b.close();
        b.build()
    }

    // GAP-C4 regression: partial-overlap difference must carve out the
    // intersection, not return the unmodified subject.
    #[test]
    fn test_difference_partial_overlap() {
        let path_a = rect_path(0.0, 0.0, 20.0);
        let path_b = rect_path(10.0, 10.0, 20.0);

        let result = op(&path_a, &path_b, PathOp::Difference).expect("op returns Some");
        let polys = path_to_polygons(&result, 0.5);

        assert!(
            !polys.is_empty(),
            "A - B of partially overlapping rects must not be empty"
        );
        // (5, 5) is in A only — belongs in the result.
        assert!(
            polys_contain_probe(&polys, Point::new(5.0, 5.0)),
            "point in A \\ B (A only) should be in result"
        );
        // (15, 15) is in both A and B — must be carved out.
        assert!(
            !polys_contain_probe(&polys, Point::new(15.0, 15.0)),
            "point in A ∩ B should NOT be in A \\ B"
        );
        // (25, 25) is in B only — must not appear in A \ B.
        assert!(
            !polys_contain_probe(&polys, Point::new(25.0, 25.0)),
            "point only in B should not be in A \\ B"
        );
    }

    // GAP-C5 regression: intersection on a concave subject must not
    // degrade to the Sutherland-Hodgman answer.
    #[test]
    fn test_intersect_concave_polygons() {
        // L-shape: width 20, height 20, with a 10x10 corner cut out
        // of the top-right, leaving the horizontal arm y in [0, 10]
        // and the vertical arm x in [0, 10].
        let mut l = PathBuilder::new();
        l.move_to(0.0, 0.0);
        l.line_to(20.0, 0.0);
        l.line_to(20.0, 10.0);
        l.line_to(10.0, 10.0);
        l.line_to(10.0, 20.0);
        l.line_to(0.0, 20.0);
        l.close();
        let path_l = l.build();

        // Rect 5..25 in both axes — overlaps both arms of the L and
        // the cut-out corner.
        let path_r = rect_path(5.0, 5.0, 20.0);

        let result = op(&path_l, &path_r, PathOp::Intersect).expect("op returns Some");
        let polys = path_to_polygons(&result, 0.5);

        assert!(!polys.is_empty(), "L ∩ rect must not be empty");
        // (15, 7) is inside the L (horizontal arm, y<10) and inside
        // the rect (x>5, y>5) — should be in result.
        assert!(
            polys_contain_probe(&polys, Point::new(15.0, 7.0)),
            "point in both L and rect should be in L ∩ rect"
        );
        // (7, 15) is inside the L (vertical arm, x<10) and inside
        // the rect — should be in result.
        assert!(
            polys_contain_probe(&polys, Point::new(7.0, 15.0)),
            "point in vertical arm and rect should be in L ∩ rect"
        );
        // (15, 15) is in the cut-out corner of the L (OUTSIDE L)
        // but inside the rect — must NOT be in result. This is the
        // case Sutherland-Hodgman would get wrong.
        assert!(
            !polys_contain_probe(&polys, Point::new(15.0, 15.0)),
            "point in L's cut-out corner must not appear in L ∩ rect"
        );
    }

    // GAP-C4 regression: partial-overlap xor must produce a ring-
    // like region around the overlap, not the concatenation of the
    // two inputs.
    #[test]
    fn test_xor_partial_overlap() {
        let path_a = rect_path(0.0, 0.0, 20.0);
        let path_b = rect_path(10.0, 10.0, 20.0);

        let result = op(&path_a, &path_b, PathOp::Xor).expect("op returns Some");
        let polys = path_to_polygons(&result, 0.5);

        assert!(!polys.is_empty(), "A XOR B must not be empty");
        // (15, 15) is in both — must be excluded.
        assert!(
            !polys_contain_probe(&polys, Point::new(15.0, 15.0)),
            "A ∩ B region must not be in A XOR B"
        );
        // (5, 5) is in A only.
        assert!(
            polys_contain_probe(&polys, Point::new(5.0, 5.0)),
            "A-only region must be in A XOR B"
        );
        // (25, 25) is in B only.
        assert!(
            polys_contain_probe(&polys, Point::new(25.0, 25.0)),
            "B-only region must be in A XOR B"
        );
    }

    // GAP-C4 regression: ReverseDifference must actually subtract.
    #[test]
    fn test_reverse_difference_partial_overlap() {
        let path_a = rect_path(0.0, 0.0, 20.0);
        let path_b = rect_path(10.0, 10.0, 20.0);

        let result = op(&path_a, &path_b, PathOp::ReverseDifference).expect("op returns Some");
        let polys = path_to_polygons(&result, 0.5);

        // ReverseDifference is B \ A.
        // (25, 25) is in B only — must be in result.
        assert!(
            polys_contain_probe(&polys, Point::new(25.0, 25.0)),
            "point in B only should be in B \\ A"
        );
        // (15, 15) is in both — must be carved out.
        assert!(
            !polys_contain_probe(&polys, Point::new(15.0, 15.0)),
            "overlap region should not appear in B \\ A"
        );
        // (5, 5) is in A only — must be excluded.
        assert!(
            !polys_contain_probe(&polys, Point::new(5.0, 5.0)),
            "point only in A should not be in B \\ A"
        );
    }
}
