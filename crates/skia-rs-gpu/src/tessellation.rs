//! Path tessellation for GPU rendering.
//!
//! This module provides algorithms for converting vector paths into triangle meshes
//! suitable for GPU rendering.

use crate::cast_util::{scalar_from_u32, scalar_from_usize, u32_from_usize};
use skia_rs_core::{Matrix, Point, Rect, Scalar};
use skia_rs_path::{Path, PathElement};

/// A vertex in a tessellated mesh.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct TessVertex {
    /// Position.
    pub position: [f32; 2],
    /// UV coordinates (for texturing/gradients).
    pub uv: [f32; 2],
}

impl TessVertex {
    /// Create a new vertex.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
        }
    }

    /// Create a vertex from a point.
    #[inline]
    #[must_use]
    pub const fn from_point(p: Point) -> Self {
        Self {
            position: [p.x, p.y],
            uv: [0.0, 0.0],
        }
    }
}

/// Index type for tessellated meshes.
pub type TessIndex = u32;

/// A tessellated mesh ready for GPU rendering.
#[derive(Debug, Clone, Default)]
pub struct TessMesh {
    /// Vertices.
    pub vertices: Vec<TessVertex>,
    /// Indices (triangle list).
    pub indices: Vec<TessIndex>,
}

impl TessMesh {
    /// Create a new empty mesh.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mesh with preallocated capacity.
    #[must_use]
    pub fn with_capacity(vertex_capacity: usize, index_capacity: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertex_capacity),
            indices: Vec::with_capacity(index_capacity),
        }
    }

    /// Clear the mesh.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Add a vertex and return its index.
    pub fn add_vertex(&mut self, vertex: TessVertex) -> TessIndex {
        let idx = u32_from_usize(self.vertices.len());
        self.vertices.push(vertex);
        idx
    }

    /// Add a triangle by indices.
    pub fn add_triangle(&mut self, a: TessIndex, b: TessIndex, c: TessIndex) {
        self.indices.push(a);
        self.indices.push(b);
        self.indices.push(c);
    }

    /// Merge another mesh into this one.
    pub fn merge(&mut self, other: &Self) {
        let base_index = u32_from_usize(self.vertices.len());
        self.vertices.extend_from_slice(&other.vertices);
        self.indices
            .extend(other.indices.iter().map(|i| i + base_index));
    }
}

/// Tessellation quality settings.
#[derive(Debug, Clone, Copy)]
pub struct TessQuality {
    /// Maximum distance from curve to approximating line segment.
    pub tolerance: Scalar,
    /// Maximum number of subdivisions for curves.
    pub max_subdivisions: u32,
}

impl Default for TessQuality {
    fn default() -> Self {
        Self {
            tolerance: 0.25,
            max_subdivisions: 10,
        }
    }
}

impl TessQuality {
    /// Low quality (fast).
    pub const LOW: Self = Self {
        tolerance: 1.0,
        max_subdivisions: 5,
    };

    /// Medium quality.
    pub const MEDIUM: Self = Self {
        tolerance: 0.5,
        max_subdivisions: 8,
    };

    /// High quality.
    pub const HIGH: Self = Self {
        tolerance: 0.25,
        max_subdivisions: 10,
    };

    /// Very high quality (slow).
    pub const VERY_HIGH: Self = Self {
        tolerance: 0.1,
        max_subdivisions: 15,
    };
}

/// Upper bound on the number of segments a single curve is flattened into.
///
/// Matches Skia's `GrPathUtils::kMaxPointsPerCurve` (`1 << 10`). Curve
/// subdivision is tolerance-driven; this only guards against pathological
/// inputs (huge magnification) rather than being the normal limiter.
pub const MAX_POINTS_PER_CURVE: u32 = 1 << 10;

/// Compute the maximum scale factor (largest singular value of the 2x2
/// linear part) of a view matrix — the Skia `SkMatrix::getMaxScale`
/// equivalent used by `GrPathUtils::scaleToleranceToSrc`.
#[must_use]
#[allow(
    clippy::many_single_char_names,
    reason = "a/b/c/d are the 2x2 linear-part matrix entries; single-letter names match the standard matrix notation"
)]
pub fn matrix_max_scale(m: &Matrix) -> Scalar {
    let a = m.scale_x();
    let b = m.skew_y();
    let c = m.skew_x();
    let d = m.scale_y();
    // Largest singular value of [[a, c], [b, d]].
    let aa = d.mul_add(d, c.mul_add(c, a.mul_add(a, b * b)));
    let det = a.mul_add(d, -(b * c));
    let disc = aa.mul_add(aa, -(4.0 * det * det)).max(0.0).sqrt();
    (0.5 * (aa + disc)).max(0.0).sqrt()
}

/// Stroke join style (how consecutive segments meet at a vertex).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeJoin {
    /// Sharp corner extended to a point, limited by the miter limit.
    #[default]
    Miter,
    /// Flat corner (chamfer).
    Bevel,
    /// Rounded corner (arc of radius half-width).
    Round,
}

/// Stroke cap style (how open contour ends are terminated).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeCap {
    /// Squared off exactly at the endpoint.
    #[default]
    Butt,
    /// Squared off half a stroke width beyond the endpoint.
    Square,
    /// Rounded (semicircle of radius half-width).
    Round,
}

/// Stroke styling: width, join, cap, and miter limit.
#[derive(Debug, Clone, Copy)]
pub struct StrokeStyle {
    /// Full stroke width.
    pub width: Scalar,
    /// Join style at interior vertices.
    pub join: StrokeJoin,
    /// Cap style at open contour ends.
    pub cap: StrokeCap,
    /// Miter limit: max ratio of miter length to half-width before a miter
    /// join falls back to bevel. Skia's default is 4.
    pub miter_limit: Scalar,
}

impl StrokeStyle {
    /// Create a style with the given width and Skia defaults (miter joins,
    /// butt caps, miter limit 4).
    #[must_use]
    pub const fn new(width: Scalar) -> Self {
        Self {
            width,
            join: StrokeJoin::Miter,
            cap: StrokeCap::Butt,
            miter_limit: 4.0,
        }
    }

    /// Builder: set the join style.
    #[must_use]
    pub const fn with_join(mut self, join: StrokeJoin) -> Self {
        self.join = join;
        self
    }

    /// Builder: set the cap style.
    #[must_use]
    pub const fn with_cap(mut self, cap: StrokeCap) -> Self {
        self.cap = cap;
        self
    }

    /// Builder: set the miter limit.
    #[must_use]
    pub const fn with_miter_limit(mut self, limit: Scalar) -> Self {
        self.miter_limit = limit;
        self
    }
}

/// Path tessellator.
pub struct PathTessellator {
    quality: TessQuality,
    /// Flattened points from current contour.
    contour_points: Vec<Point>,
    /// Max view-matrix scale used to convert the device-space tolerance to
    /// source space (`GrPathUtils::scaleToleranceToSrc`). 1.0 = identity.
    device_scale: Scalar,
}

impl PathTessellator {
    /// Create a new tessellator with default quality.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality: TessQuality::default(),
            contour_points: Vec::new(),
            device_scale: 1.0,
        }
    }

    /// Create a new tessellator with specified quality.
    #[must_use]
    pub const fn with_quality(quality: TessQuality) -> Self {
        Self {
            quality,
            contour_points: Vec::new(),
            device_scale: 1.0,
        }
    }

    /// Set the view matrix used for device-space flattening. Curves are
    /// flattened to a tolerance measured in *device* pixels: the source-space
    /// tolerance shrinks as the matrix magnifies, so a path drawn 10x larger
    /// gets ~10x finer subdivision (`GrPathUtils::scaleToleranceToSrc`).
    pub fn set_view_matrix(&mut self, matrix: &Matrix) {
        self.device_scale = matrix_max_scale(matrix).max(1e-4);
    }

    /// Source-space flattening tolerance: device tolerance divided by the
    /// view-matrix max scale, clamped to a small positive minimum.
    fn src_tolerance(&self) -> Scalar {
        (self.quality.tolerance / self.device_scale).max(1e-4)
    }

    /// Seed a fresh contour at `start` when the point list is empty.
    ///
    /// Implements `SkPath`'s post-`Close` rule: a drawing verb (line/curve)
    /// that follows a `Close` without an intervening `Move` begins a new
    /// contour at the previous contour's start point (the last move point).
    /// Without this the new contour would silently drop its first vertex.
    fn ensure_contour(&mut self, start: Point) {
        if self.contour_points.is_empty() {
            self.contour_points.push(start);
        }
    }

    /// Tessellate a path for filling.
    pub fn tessellate_fill(&mut self, path: &Path) -> TessMesh {
        let mut mesh = TessMesh::new();
        self.contour_points.clear();

        let mut current_point = Point::zero();
        let mut contour_start = Point::zero();

        for element in path {
            match element {
                PathElement::Move(p) => {
                    self.flush_contour(&mut mesh);
                    current_point = p;
                    contour_start = p;
                    self.contour_points.push(p);
                }
                PathElement::Line(p) => {
                    self.ensure_contour(contour_start);
                    self.contour_points.push(p);
                    current_point = p;
                }
                PathElement::Quad(ctrl, end) => {
                    self.ensure_contour(contour_start);
                    self.flatten_quad(current_point, ctrl, end);
                    current_point = end;
                }
                PathElement::Conic(ctrl, end, weight) => {
                    self.ensure_contour(contour_start);
                    self.flatten_conic(current_point, ctrl, end, weight);
                    current_point = end;
                }
                PathElement::Cubic(ctrl1, ctrl2, end) => {
                    self.ensure_contour(contour_start);
                    self.flatten_cubic(current_point, ctrl1, ctrl2, end);
                    current_point = end;
                }
                PathElement::Close => {
                    if current_point != contour_start {
                        self.contour_points.push(contour_start);
                    }
                    self.flush_contour(&mut mesh);
                    // Per SkPath, the next contour (if any drawing verb
                    // follows without a Move) starts at the close point.
                    current_point = contour_start;
                }
            }
        }

        // Flush any remaining contour
        self.flush_contour(&mut mesh);

        mesh
    }

    /// Flatten a path into a list of contours (each a polyline of points),
    /// applying the same curve flattening and post-`Close` seeding as
    /// [`Self::tessellate_fill`]. Used for fill-strategy classification.
    fn flatten_contours(&mut self, path: &Path) -> Vec<Vec<Point>> {
        let mut contours: Vec<Vec<Point>> = Vec::new();
        self.contour_points.clear();
        let mut current = Point::zero();
        let mut start = Point::zero();

        for element in path {
            match element {
                PathElement::Move(p) => {
                    if self.contour_points.len() >= 3 {
                        contours.push(std::mem::take(&mut self.contour_points));
                    } else {
                        self.contour_points.clear();
                    }
                    current = p;
                    start = p;
                    self.contour_points.push(p);
                }
                PathElement::Line(p) => {
                    self.ensure_contour(start);
                    self.contour_points.push(p);
                    current = p;
                }
                PathElement::Quad(ctrl, end) => {
                    self.ensure_contour(start);
                    self.flatten_quad(current, ctrl, end);
                    current = end;
                }
                PathElement::Conic(ctrl, end, weight) => {
                    self.ensure_contour(start);
                    self.flatten_conic(current, ctrl, end, weight);
                    current = end;
                }
                PathElement::Cubic(ctrl1, ctrl2, end) => {
                    self.ensure_contour(start);
                    self.flatten_cubic(current, ctrl1, ctrl2, end);
                    current = end;
                }
                PathElement::Close => {
                    if self.contour_points.len() >= 3 {
                        contours.push(std::mem::take(&mut self.contour_points));
                    } else {
                        self.contour_points.clear();
                    }
                    current = start;
                }
            }
        }
        if self.contour_points.len() >= 3 {
            contours.push(std::mem::take(&mut self.contour_points));
        }
        self.contour_points.clear();
        contours
    }

    /// Classify how a path's fill should be rasterized.
    ///
    /// Direct triangulation only produces a correct fill for a *single convex
    /// contour*. Multi-contour paths (holes) and non-convex contours require
    /// the winding/even-odd fill rule, which ear-clipping per contour cannot
    /// express (it would fill holes solid). Those must be routed through
    /// stencil-then-cover, so this returns [`FillStrategy::StencilCover`] for
    /// them. Matches the brief's minimum conforming approach.
    pub fn classify_fill(&mut self, path: &Path) -> FillStrategy {
        let contours = self.flatten_contours(path);
        if contours.len() == 1 && polygon_is_convex(&contours[0]) {
            FillStrategy::DirectConvex
        } else {
            FillStrategy::StencilCover
        }
    }

    /// Tessellate a fill only when it is a single convex contour, for which
    /// direct triangulation is correct. Returns `None` when the path must be
    /// routed through stencil-then-cover (see [`Self::classify_fill`]).
    pub fn tessellate_fill_convex(&mut self, path: &Path) -> Option<TessMesh> {
        match self.classify_fill(path) {
            FillStrategy::DirectConvex => Some(self.tessellate_fill(path)),
            FillStrategy::StencilCover => None,
        }
    }

    /// Tessellate a path for stroking with the default style (miter joins,
    /// butt caps, miter limit 4).
    pub fn tessellate_stroke(&mut self, path: &Path, stroke_width: Scalar) -> TessMesh {
        self.tessellate_stroke_styled(path, &StrokeStyle::new(stroke_width))
    }

    /// Tessellate a path for stroking with an explicit [`StrokeStyle`]
    /// (join, cap, miter limit).
    pub fn tessellate_stroke_styled(&mut self, path: &Path, style: &StrokeStyle) -> TessMesh {
        let mut mesh = TessMesh::new();
        self.contour_points.clear();

        let mut current_point = Point::zero();
        let mut contour_start = Point::zero();

        for element in path {
            match element {
                PathElement::Move(p) => {
                    self.flush_stroke_contour(&mut mesh, style, false);
                    current_point = p;
                    contour_start = p;
                    self.contour_points.push(p);
                }
                PathElement::Line(p) => {
                    self.ensure_contour(contour_start);
                    self.contour_points.push(p);
                    current_point = p;
                }
                PathElement::Quad(ctrl, end) => {
                    self.ensure_contour(contour_start);
                    self.flatten_quad(current_point, ctrl, end);
                    current_point = end;
                }
                PathElement::Conic(ctrl, end, weight) => {
                    self.ensure_contour(contour_start);
                    self.flatten_conic(current_point, ctrl, end, weight);
                    current_point = end;
                }
                PathElement::Cubic(ctrl1, ctrl2, end) => {
                    self.ensure_contour(contour_start);
                    self.flatten_cubic(current_point, ctrl1, ctrl2, end);
                    current_point = end;
                }
                PathElement::Close => {
                    if current_point != contour_start {
                        self.contour_points.push(contour_start);
                    }
                    self.flush_stroke_contour(&mut mesh, style, true);
                    current_point = contour_start;
                }
            }
        }

        // Flush any remaining contour
        self.flush_stroke_contour(&mut mesh, style, false);

        mesh
    }

    /// Flatten a quadratic bezier curve.
    fn flatten_quad(&mut self, p0: Point, p1: Point, p2: Point) {
        let steps = self.quad_subdivisions(p0, p1, p2);
        for i in 1..=steps {
            let t = scalar_from_u32(i) / scalar_from_u32(steps);
            let p = Self::eval_quad(p0, p1, p2, t);
            self.contour_points.push(p);
        }
    }

    /// Flatten a conic curve.
    fn flatten_conic(&mut self, p0: Point, p1: Point, p2: Point, w: Scalar) {
        // For simplicity, treat conics as quadratics when w ≈ 1
        if (w - 1.0).abs() < 0.001 {
            self.flatten_quad(p0, p1, p2);
            return;
        }

        // Adaptively choose step count based on curve deviation.
        //
        // Like the quadratic heuristic, use the max distance from the
        // control point to the chord as a proxy for curvature. Multiply
        // by a weight-dependent factor: conics with large (or small)
        // weights deviate further from the chord than w = 1 quadratics
        // for the same control-point placement, so scale the deviation
        // up as the weight departs from 1.
        let base_d = Self::point_to_line_distance(p1, p0, p2);
        let weight_amp = 1.0 + (w - 1.0).abs().min(4.0);
        let d = base_d * weight_amp;
        let steps = crate::cast_util::u32_from_scalar_sat((d / self.src_tolerance()).sqrt().ceil())
            .clamp(2, MAX_POINTS_PER_CURVE);
        for i in 1..=steps {
            let t = scalar_from_u32(i) / scalar_from_u32(steps);
            let p = Self::eval_conic(p0, p1, p2, w, t);
            self.contour_points.push(p);
        }
    }

    /// Flatten a cubic bezier curve.
    fn flatten_cubic(&mut self, p0: Point, p1: Point, p2: Point, p3: Point) {
        let steps = self.cubic_subdivisions(p0, p1, p2, p3);
        for i in 1..=steps {
            let t = scalar_from_u32(i) / scalar_from_u32(steps);
            let p = Self::eval_cubic(p0, p1, p2, p3, t);
            self.contour_points.push(p);
        }
    }

    /// Calculate number of subdivisions for quadratic curve.
    ///
    /// Tolerance-driven (device-space): the count scales with the square root
    /// of the curve's deviation over the source-space tolerance, bounded only
    /// by `MAX_POINTS_PER_CURVE`. The old fixed `max_subdivisions` cap made
    /// magnified curves visibly faceted; it no longer limits curve flattening.
    fn quad_subdivisions(&self, p0: Point, p1: Point, p2: Point) -> u32 {
        let d = Self::point_to_line_distance(p1, p0, p2);
        crate::cast_util::u32_from_scalar_sat((d / self.src_tolerance()).sqrt().ceil())
            .clamp(1, MAX_POINTS_PER_CURVE)
    }

    /// Calculate number of subdivisions for cubic curve.
    fn cubic_subdivisions(&self, p0: Point, p1: Point, p2: Point, p3: Point) -> u32 {
        let d1 = Self::point_to_line_distance(p1, p0, p3);
        let d2 = Self::point_to_line_distance(p2, p0, p3);
        let d = d1.max(d2);
        crate::cast_util::u32_from_scalar_sat((d / self.src_tolerance()).sqrt().ceil())
            .clamp(1, MAX_POINTS_PER_CURVE)
    }

    /// Evaluate quadratic bezier at t.
    fn eval_quad(p0: Point, p1: Point, p2: Point, t: Scalar) -> Point {
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let t2 = t * t;
        Point::new(
            t2.mul_add(p2.x, (2.0 * mt * t).mul_add(p1.x, mt2 * p0.x)),
            t2.mul_add(p2.y, (2.0 * mt * t).mul_add(p1.y, mt2 * p0.y)),
        )
    }

    /// Evaluate conic at t.
    fn eval_conic(p0: Point, p1: Point, p2: Point, w: Scalar, t: Scalar) -> Point {
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let t2 = t * t;
        let wt = 2.0 * w * mt * t;
        let denom = mt2 + wt + t2;
        Point::new(
            t2.mul_add(p2.x, mt2 * p0.x + wt * p1.x) / denom,
            t2.mul_add(p2.y, mt2 * p0.y + wt * p1.y) / denom,
        )
    }

    /// Evaluate cubic bezier at t.
    fn eval_cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: Scalar) -> Point {
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let t2 = t * t;
        let t3 = t2 * t;
        Point::new(
            t3.mul_add(
                p3.x,
                (3.0 * mt * t2).mul_add(p2.x, (3.0 * mt2 * t).mul_add(p1.x, mt3 * p0.x)),
            ),
            t3.mul_add(
                p3.y,
                (3.0 * mt * t2).mul_add(p2.y, (3.0 * mt2 * t).mul_add(p1.y, mt3 * p0.y)),
            ),
        )
    }

    /// Calculate distance from point to line.
    fn point_to_line_distance(p: Point, line_start: Point, line_end: Point) -> Scalar {
        let dx = line_end.x - line_start.x;
        let dy = line_end.y - line_start.y;
        let len_sq = dx.mul_add(dx, dy * dy);
        if len_sq < 1e-10 {
            return (p.x - line_start.x).hypot(p.y - line_start.y);
        }
        let num = (p.x - line_start.x)
            .mul_add(dy, -((p.y - line_start.y) * dx))
            .abs();
        num / len_sq.sqrt()
    }

    /// Flush current contour for fill tessellation.
    ///
    /// Uses a proper ear-clipping triangulation that handles both convex and
    /// concave polygons. The classical O(n²) ear-clip algorithm (Meisters 1975):
    /// repeatedly find a triangle formed by three consecutive vertices that
    /// is (a) wound in the polygon's direction and (b) contains no other
    /// vertex, emit it, remove the middle vertex, and continue. This is
    /// correct for any simple polygon, including concave shapes like letter
    /// outlines. For self-intersecting polygons the output is still a valid
    /// triangulation of the positive-area regions under the winding rule.
    fn flush_contour(&mut self, mesh: &mut TessMesh) {
        if self.contour_points.len() < 3 {
            self.contour_points.clear();
            return;
        }

        // Drop a trailing duplicate of the first point (added by PathElement::Close)
        // so the winding/ear tests don't see a zero-length final edge.
        if self.contour_points.len() > 3 {
            let first = self.contour_points[0];
            let last = *self.contour_points.last().unwrap();
            if (first.x - last.x).abs() < 1e-6 && (first.y - last.y).abs() < 1e-6 {
                self.contour_points.pop();
            }
        }

        if self.contour_points.len() < 3 {
            self.contour_points.clear();
            return;
        }

        // Push vertices into the mesh up front; ear clipping emits indices.
        let base_idx = u32_from_usize(mesh.vertices.len());
        let vertices: Vec<TessVertex> = self
            .contour_points
            .iter()
            .map(|p| TessVertex::from_point(*p))
            .collect();
        mesh.vertices.extend(vertices);

        ear_clip_triangulate(&self.contour_points, base_idx, mesh);
        self.contour_points.clear();
    }

    /// Flush current contour for stroke tessellation.
    ///
    /// Strokes each segment as an offset quad, then fills joins and end caps
    /// per the [`StrokeStyle`]. Miter joins extend the outer corner by
    /// `half_width / cos(theta/2)` and fall back to a bevel when that exceeds
    /// the miter limit; caps honor butt/square/round on open contours. The
    /// triangles are coverage-correct (they may overlap in the join interior,
    /// which is harmless for opaque/stencil rasterization).
    fn flush_stroke_contour(&mut self, mesh: &mut TessMesh, style: &StrokeStyle, closed: bool) {
        let pts = std::mem::take(&mut self.contour_points);
        stroke_contour(&pts, style, closed, mesh);
    }
}

/// Stroke a flattened contour into triangles.
fn stroke_contour(raw: &[Point], style: &StrokeStyle, closed: bool, mesh: &mut TessMesh) {
    let hw = style.width * 0.5;
    if hw <= 0.0 {
        return;
    }

    // Drop consecutive duplicate points so segment directions are well-defined.
    let mut pts: Vec<Point> = Vec::with_capacity(raw.len());
    for &p in raw {
        if pts.last().is_none_or(|q| dist2(*q, p) > 1e-12) {
            pts.push(p);
        }
    }
    // For a closed contour, drop a trailing duplicate of the first point.
    if closed && pts.len() > 1 && dist2(pts[0], *pts.last().unwrap()) <= 1e-12 {
        pts.pop();
    }
    let n = pts.len();
    if n < 2 {
        return;
    }

    // Unit direction and left-normal for each segment.
    let seg_count = if closed { n } else { n - 1 };
    let mut dirs = Vec::with_capacity(seg_count);
    for s in 0..seg_count {
        let a = pts[s];
        let b = pts[(s + 1) % n];
        dirs.push(unit(sub(b, a)));
    }

    // Body quad for each segment.
    for s in 0..seg_count {
        let a = pts[s];
        let b = pts[(s + 1) % n];
        let nrm = left_normal(dirs[s]);
        emit_quad(
            mesh,
            offset(a, nrm, hw),
            offset(a, nrm, -hw),
            offset(b, nrm, -hw),
            offset(b, nrm, hw),
        );
    }

    // Joins at interior vertices (and the wrap vertex for closed contours).
    let join_start = usize::from(!closed);
    for (v, &p) in pts.iter().enumerate().skip(join_start) {
        // Incoming segment ends at vertex v; outgoing starts at v.
        let in_seg = (v + seg_count - 1) % seg_count;
        let out_seg = v % seg_count;
        if !closed && (v == 0 || v >= seg_count) {
            continue;
        }
        emit_join(mesh, p, dirs[in_seg], dirs[out_seg], hw, style);
    }

    // Caps on open contours.
    if !closed {
        // Start cap: outward tangent points backward along the first segment.
        emit_cap(mesh, pts[0], neg(dirs[0]), hw, style.cap);
        // End cap: outward tangent points forward along the last segment.
        emit_cap(mesh, pts[n - 1], dirs[seg_count - 1], hw, style.cap);
    }
}

#[inline]
fn sub(a: Point, b: Point) -> Point {
    Point::new(a.x - b.x, a.y - b.y)
}
#[inline]
fn neg(a: Point) -> Point {
    Point::new(-a.x, -a.y)
}
#[inline]
fn dist2(a: Point, b: Point) -> Scalar {
    let d = sub(a, b);
    d.x.mul_add(d.x, d.y * d.y)
}
#[inline]
fn unit(v: Point) -> Point {
    let len = v.x.hypot(v.y);
    if len < 1e-10 {
        Point::new(0.0, 0.0)
    } else {
        Point::new(v.x / len, v.y / len)
    }
}
#[inline]
fn left_normal(dir: Point) -> Point {
    Point::new(-dir.y, dir.x)
}
#[inline]
fn offset(p: Point, dir: Point, amt: Scalar) -> Point {
    Point::new(dir.x.mul_add(amt, p.x), dir.y.mul_add(amt, p.y))
}

/// Wrap an angle (in radians) to `(-PI, PI]` in a single step, avoiding an
/// iterative `while` loop with a float comparison in its condition.
#[inline]
fn wrap_to_pi(angle: Scalar) -> Scalar {
    let two_pi = 2.0 * std::f32::consts::PI;
    angle - two_pi * (angle / two_pi).round()
}

/// Emit two triangles for quad a-b-c-d (in order).
fn emit_quad(mesh: &mut TessMesh, a: Point, b: Point, c: Point, d: Point) {
    let ia = mesh.add_vertex(TessVertex::from_point(a));
    let ib = mesh.add_vertex(TessVertex::from_point(b));
    let ic = mesh.add_vertex(TessVertex::from_point(c));
    let id = mesh.add_vertex(TessVertex::from_point(d));
    mesh.add_triangle(ia, ib, ic);
    mesh.add_triangle(ia, ic, id);
}

fn emit_tri(mesh: &mut TessMesh, a: Point, b: Point, c: Point) {
    let ia = mesh.add_vertex(TessVertex::from_point(a));
    let ib = mesh.add_vertex(TessVertex::from_point(b));
    let ic = mesh.add_vertex(TessVertex::from_point(c));
    mesh.add_triangle(ia, ib, ic);
}

/// Fill the join at vertex `v` between two segment directions.
fn emit_join(
    mesh: &mut TessMesh,
    v: Point,
    d_in: Point,
    d_out: Point,
    hw: Scalar,
    style: &StrokeStyle,
) {
    let n0 = left_normal(d_in);
    let n1 = left_normal(d_out);
    let turn = d_in.x.mul_add(d_out.y, -(d_in.y * d_out.x)); // z of cross(d_in, d_out)
    if turn.abs() < 1e-9 {
        return; // straight — segment quads already meet flush
    }

    // Bevel triangles fill the wedge between the two segment ends on both
    // offset sides (the inner side lies inside the overlap; harmless).
    emit_tri(mesh, v, offset(v, n0, hw), offset(v, n1, hw));
    emit_tri(mesh, v, offset(v, n0, -hw), offset(v, n1, -hw));

    match style.join {
        StrokeJoin::Bevel | StrokeJoin::Round => {
            if style.join == StrokeJoin::Round {
                emit_round_fan(mesh, v, n0, n1, hw);
            }
        }
        StrokeJoin::Miter => {
            // Miter length ratio = 1 / cos(theta/2), where the half-angle is
            // between the bisector and a segment normal.
            let m = unit(Point::new(n0.x + n1.x, n0.y + n1.y));
            let cos_half = m.x.mul_add(n0.x, m.y * n0.y);
            if cos_half.abs() > 1e-4 {
                let ratio = 1.0 / cos_half.abs();
                if ratio <= style.miter_limit {
                    // Extend the outer corner to the miter tip on both sides;
                    // the inner tip is inside the body union (harmless).
                    let tip_pos = offset(v, m, hw * ratio);
                    let tip_neg = offset(v, m, -hw * ratio);
                    emit_tri(mesh, offset(v, n0, hw), tip_pos, offset(v, n1, hw));
                    emit_tri(mesh, offset(v, n0, -hw), tip_neg, offset(v, n1, -hw));
                }
                // else: exceeds miter limit -> bevel only (already emitted).
            }
        }
    }
}

/// Fill a round join/cap as a triangle fan from `v` sweeping between the two
/// offset normals at radius `hw`.
fn emit_round_fan(mesh: &mut TessMesh, v: Point, n0: Point, n1: Point, hw: Scalar) {
    let a0 = n0.y.atan2(n0.x);
    let a1 = n1.y.atan2(n1.x);
    // Sweep the shorter way around.
    let delta = wrap_to_pi(a1 - a0);
    let steps =
        crate::cast_util::u32_from_scalar_sat((delta.abs() / (std::f32::consts::PI / 8.0)).ceil())
            .max(1);
    for i in 0..steps {
        let t0 = delta.mul_add(scalar_from_u32(i) / scalar_from_u32(steps), a0);
        let t1 = delta.mul_add(scalar_from_u32(i + 1) / scalar_from_u32(steps), a0);
        let p0 = Point::new(t0.cos().mul_add(hw, v.x), t0.sin().mul_add(hw, v.y));
        let p1 = Point::new(t1.cos().mul_add(hw, v.x), t1.sin().mul_add(hw, v.y));
        emit_tri(mesh, v, p0, p1);
        // Mirror the fan to the opposite side for symmetric round joins.
        let p0m = Point::new(t0.cos().mul_add(-hw, v.x), t0.sin().mul_add(-hw, v.y));
        let p1m = Point::new(t1.cos().mul_add(-hw, v.x), t1.sin().mul_add(-hw, v.y));
        emit_tri(mesh, v, p1m, p0m);
    }
}

/// Emit an end cap at endpoint `p` whose outward tangent is `out_dir`.
fn emit_cap(mesh: &mut TessMesh, p: Point, out_dir: Point, hw: Scalar, cap: StrokeCap) {
    let nrm = left_normal(out_dir);
    match cap {
        StrokeCap::Butt => {}
        StrokeCap::Square => {
            // Extend the stroke rectangle by half_width along the outward
            // tangent.
            let a = offset(p, nrm, hw);
            let b = offset(p, nrm, -hw);
            let a2 = offset(a, out_dir, hw);
            let b2 = offset(b, out_dir, hw);
            emit_quad(mesh, a, b, b2, a2);
        }
        StrokeCap::Round => {
            // Semicircle of radius half_width around the endpoint, sweeping
            // from +normal to -normal through the outward tangent.
            let start = nrm.y.atan2(nrm.x);
            let out_ang = out_dir.y.atan2(out_dir.x);
            // Sweep 180 degrees toward the outward direction.
            let dir_sign = {
                let d = wrap_to_pi(out_ang - start);
                if d >= 0.0 { 1.0 } else { -1.0 }
            };
            let steps = 8u32;
            for i in 0..steps {
                let t0 = (dir_sign * std::f32::consts::PI)
                    .mul_add(scalar_from_u32(i) / scalar_from_u32(steps), start);
                let t1 = (dir_sign * std::f32::consts::PI)
                    .mul_add(scalar_from_u32(i + 1) / scalar_from_u32(steps), start);
                let p0 = Point::new(t0.cos().mul_add(hw, p.x), t0.sin().mul_add(hw, p.y));
                let p1 = Point::new(t1.cos().mul_add(hw, p.x), t1.sin().mul_add(hw, p.y));
                emit_tri(mesh, p, p0, p1);
            }
        }
    }
}

impl Default for PathTessellator {
    fn default() -> Self {
        Self::new()
    }
}

/// How a path's fill should be rasterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillStrategy {
    /// A single convex contour: direct triangulation is correct and cheap.
    DirectConvex,
    /// Multiple contours and/or a non-convex contour: must be routed through
    /// stencil-then-cover so the fill rule and holes are honored.
    StencilCover,
}

/// Test whether a polygon (given as a closed point ring) is convex.
///
/// A polygon is convex iff every consecutive turn has the same sign.
/// Collinear (zero-cross) triples are ignored. Fewer than 3 points is not a
/// fillable polygon and returns false.
#[allow(
    clippy::many_single_char_names,
    reason = "a/b/c are triangle vertices in the standard geometry naming convention"
)]
fn polygon_is_convex(points: &[Point]) -> bool {
    let n = points.len();
    if n < 3 {
        return false;
    }
    let mut sign = 0i32;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        let c = points[(i + 2) % n];
        let cross = triangle_cross(a, b, c);
        if cross.abs() < 1e-9 {
            continue; // collinear — does not affect convexity
        }
        let s = if cross > 0.0 { 1 } else { -1 };
        if sign == 0 {
            sign = s;
        } else if s != sign {
            return false;
        }
    }
    true
}

/// Compute signed area of a polygon (positive = counter-clockwise).
fn polygon_signed_area(points: &[Point]) -> Scalar {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area: Scalar = 0.0;
    for i in 0..n {
        let p = points[i];
        let q = points[(i + 1) % n];
        area += p.x.mul_add(q.y, -(q.x * p.y));
    }
    area * 0.5
}

/// Twice the signed area of triangle abc (sign gives orientation).
#[inline]
fn triangle_cross(a: Point, b: Point, c: Point) -> Scalar {
    (b.x - a.x).mul_add(c.y - a.y, -((b.y - a.y) * (c.x - a.x)))
}

/// Test whether point `p` lies inside triangle `abc` (with a tolerant
/// boundary that excludes the triangle's own vertices).
fn point_in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool {
    let d1 = triangle_cross(p, a, b);
    let d2 = triangle_cross(p, b, c);
    let d3 = triangle_cross(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Ear-clip triangulation for an arbitrary simple polygon.
///
/// Writes triangles into `mesh` as indices offset by `base_idx`. The
/// algorithm is robust for concave polygons; it degenerates gracefully
/// (emits a fan) when no ear can be found, which only happens for
/// pathological self-intersecting input.
#[allow(
    clippy::many_single_char_names,
    reason = "a/b/c and ia/ib/ic are triangle vertices/indices in the standard geometry naming convention"
)]
#[allow(
    clippy::too_many_lines,
    reason = "ear-clipping is a single cohesive algorithm; splitting it up would obscure the control flow"
)]
fn ear_clip_triangulate(points: &[Point], base_idx: TessIndex, mesh: &mut TessMesh) {
    let n = points.len();
    if n < 3 {
        return;
    }

    // Determine polygon orientation. Ears are the convex vertices relative
    // to the polygon's winding, so we flip the sense for clockwise input.
    let ccw = polygon_signed_area(points) >= 0.0;

    // Working list of polygon-index references. We remove indices as we
    // clip ears, but keep `points` untouched.
    let mut remaining: Vec<usize> = (0..n).collect();

    // Safety guard against infinite loops for self-intersecting input:
    // bound total iterations at ~2n² — finite, correct on simple polygons.
    let max_iters = 2 * n * n + 4;
    let mut iters = 0;

    while remaining.len() > 3 {
        iters += 1;
        if iters > max_iters {
            // Fall back to a fan over the remaining vertices. This matches
            // the original behaviour for truly degenerate input and keeps
            // the tessellator total — we never emit nothing.
            break;
        }

        let m = remaining.len();
        let mut ear_pos: Option<usize> = None;

        for i in 0..m {
            let ia = remaining[(i + m - 1) % m];
            let ib = remaining[i];
            let ic = remaining[(i + 1) % m];
            let a = points[ia];
            let b = points[ib];
            let c = points[ic];

            // Reflex check: ear must be convex in the polygon's winding.
            let cross = triangle_cross(a, b, c);
            let is_convex = if ccw { cross > 0.0 } else { cross < 0.0 };
            if !is_convex {
                continue;
            }

            // Ear check: no other polygon vertex lies inside triangle abc.
            let mut is_ear = true;
            for &other in &remaining {
                if other == ia || other == ib || other == ic {
                    continue;
                }
                if point_in_triangle(points[other], a, b, c) {
                    is_ear = false;
                    break;
                }
            }

            if is_ear {
                ear_pos = Some(i);
                break;
            }
        }

        match ear_pos {
            Some(i) => {
                let m = remaining.len();
                let ia = remaining[(i + m - 1) % m];
                let ib = remaining[i];
                let ic = remaining[(i + 1) % m];
                // Emit triangle in the polygon's winding order so the
                // front-face rule stays consistent with the source path.
                if ccw {
                    mesh.add_triangle(
                        base_idx + u32_from_usize(ia),
                        base_idx + u32_from_usize(ib),
                        base_idx + u32_from_usize(ic),
                    );
                } else {
                    mesh.add_triangle(
                        base_idx + u32_from_usize(ia),
                        base_idx + u32_from_usize(ic),
                        base_idx + u32_from_usize(ib),
                    );
                }
                remaining.remove(i);
            }
            None => {
                // No ear found. For a simple polygon this cannot happen, so
                // the input is self-intersecting. Fall back so we still
                // produce *some* triangulation rather than dropping the
                // contour silently.
                break;
            }
        }
    }

    if remaining.len() == 3 {
        let ia = remaining[0];
        let ib = remaining[1];
        let ic = remaining[2];
        let a = points[ia];
        let b = points[ib];
        let c = points[ic];
        let cross = triangle_cross(a, b, c);
        if cross.abs() > 1e-12 {
            if ccw == (cross > 0.0) {
                mesh.add_triangle(
                    base_idx + u32_from_usize(ia),
                    base_idx + u32_from_usize(ib),
                    base_idx + u32_from_usize(ic),
                );
            } else {
                mesh.add_triangle(
                    base_idx + u32_from_usize(ia),
                    base_idx + u32_from_usize(ic),
                    base_idx + u32_from_usize(ib),
                );
            }
        }
    } else if remaining.len() > 3 {
        // Fan fallback for degenerate input: preserves some coverage rather
        // than yielding an empty mesh.
        let ia = remaining[0];
        for w in remaining.windows(2).skip(1) {
            let ib = w[0];
            let ic = w[1];
            mesh.add_triangle(
                base_idx + u32_from_usize(ia),
                base_idx + u32_from_usize(ib),
                base_idx + u32_from_usize(ic),
            );
        }
    }
}

/// Tessellate a rectangle.
#[must_use]
pub fn tessellate_rect(rect: Rect) -> TessMesh {
    let mut mesh = TessMesh::with_capacity(4, 6);

    let v0 = mesh.add_vertex(TessVertex::new(rect.left, rect.top, 0.0, 0.0));
    let v1 = mesh.add_vertex(TessVertex::new(rect.right, rect.top, 1.0, 0.0));
    let v2 = mesh.add_vertex(TessVertex::new(rect.right, rect.bottom, 1.0, 1.0));
    let v3 = mesh.add_vertex(TessVertex::new(rect.left, rect.bottom, 0.0, 1.0));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v2, v3);

    mesh
}

/// Tessellate a rounded rectangle.
#[must_use]
#[allow(
    clippy::many_single_char_names,
    reason = "x/y/u/v are the standard position/UV coordinate names used throughout this module"
)]
pub fn tessellate_rounded_rect(rect: Rect, radius: Scalar, quality: TessQuality) -> TessMesh {
    let mut mesh = TessMesh::new();

    let r = radius.min(rect.width() * 0.5).min(rect.height() * 0.5);
    if r < 0.001 {
        return tessellate_rect(rect);
    }

    // Calculate number of segments for corners
    let segments = crate::cast_util::usize_from_scalar_sat(
        (std::f32::consts::PI * r / quality.tolerance).ceil(),
    )
    .clamp(4, MAX_POINTS_PER_CURVE as usize);

    let center = rect.center();
    let center_idx = mesh.add_vertex(TessVertex::new(center.x, center.y, 0.5, 0.5));

    let mut edge_vertices = Vec::new();

    // Top-left corner
    for i in 0..=segments {
        let angle = (scalar_from_usize(i) / scalar_from_usize(segments))
            .mul_add(std::f32::consts::FRAC_PI_2, std::f32::consts::PI);
        let x = rect.left + r + r * angle.cos();
        let y = rect.top + r + r * angle.sin();
        let u = (x - rect.left) / rect.width();
        let v = (y - rect.top) / rect.height();
        edge_vertices.push(mesh.add_vertex(TessVertex::new(x, y, u, v)));
    }

    // Top-right corner
    for i in 0..=segments {
        let angle = std::f32::consts::PI.mul_add(
            1.5,
            (scalar_from_usize(i) / scalar_from_usize(segments)) * std::f32::consts::FRAC_PI_2,
        );
        let x = rect.right - r + r * angle.cos();
        let y = rect.top + r + r * angle.sin();
        let u = (x - rect.left) / rect.width();
        let v = (y - rect.top) / rect.height();
        edge_vertices.push(mesh.add_vertex(TessVertex::new(x, y, u, v)));
    }

    // Bottom-right corner
    for i in 0..=segments {
        let angle =
            (scalar_from_usize(i) / scalar_from_usize(segments)) * std::f32::consts::FRAC_PI_2;
        let x = rect.right - r + r * angle.cos();
        let y = rect.bottom - r + r * angle.sin();
        let u = (x - rect.left) / rect.width();
        let v = (y - rect.top) / rect.height();
        edge_vertices.push(mesh.add_vertex(TessVertex::new(x, y, u, v)));
    }

    // Bottom-left corner
    for i in 0..=segments {
        let angle = (scalar_from_usize(i) / scalar_from_usize(segments))
            .mul_add(std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        let x = rect.left + r + r * angle.cos();
        let y = rect.bottom - r + r * angle.sin();
        let u = (x - rect.left) / rect.width();
        let v = (y - rect.top) / rect.height();
        edge_vertices.push(mesh.add_vertex(TessVertex::new(x, y, u, v)));
    }

    // Create triangles from center to edge
    let n = edge_vertices.len();
    for i in 0..n {
        let next = (i + 1) % n;
        mesh.add_triangle(center_idx, edge_vertices[i], edge_vertices[next]);
    }

    mesh
}

/// Tessellate a circle.
#[must_use]
pub fn tessellate_circle(center: Point, radius: Scalar, quality: TessQuality) -> TessMesh {
    let mut mesh = TessMesh::new();

    let segments = crate::cast_util::usize_from_scalar_sat(
        (2.0 * std::f32::consts::PI * radius / quality.tolerance).ceil(),
    )
    .clamp(8, MAX_POINTS_PER_CURVE as usize);

    let center_idx = mesh.add_vertex(TessVertex::new(center.x, center.y, 0.5, 0.5));

    let mut edge_vertices = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle =
            (scalar_from_usize(i) / scalar_from_usize(segments)) * 2.0 * std::f32::consts::PI;
        let x = radius.mul_add(angle.cos(), center.x);
        let y = radius.mul_add(angle.sin(), center.y);
        let u = 0.5f32.mul_add(angle.cos(), 0.5);
        let v = 0.5f32.mul_add(angle.sin(), 0.5);
        edge_vertices.push(mesh.add_vertex(TessVertex::new(x, y, u, v)));
    }

    for i in 0..segments {
        let next = (i + 1) % segments;
        mesh.add_triangle(center_idx, edge_vertices[i], edge_vertices[next]);
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use skia_rs_path::PathBuilder;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_tess_vertex() {
        let v = TessVertex::new(1.0, 2.0, 0.5, 0.5);
        assert_eq!(v.position, [1.0, 2.0]);
        assert_eq!(v.uv, [0.5, 0.5]);
    }

    #[test]
    fn test_tess_mesh() {
        let mut mesh = TessMesh::new();
        assert!(mesh.is_empty());

        let v0 = mesh.add_vertex(TessVertex::new(0.0, 0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(TessVertex::new(1.0, 0.0, 1.0, 0.0));
        let v2 = mesh.add_vertex(TessVertex::new(0.5, 1.0, 0.5, 1.0));
        mesh.add_triangle(v0, v1, v2);

        assert!(!mesh.is_empty());
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
    }

    #[test]
    fn test_tessellate_rect() {
        let rect = Rect::from_xywh(0.0, 0.0, 100.0, 50.0);
        let mesh = tessellate_rect(rect);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_tessellate_circle() {
        let mesh = tessellate_circle(Point::new(50.0, 50.0), 25.0, TessQuality::MEDIUM);
        assert!(mesh.vertices.len() > 8);
        assert!(mesh.triangle_count() >= 8);
    }

    #[test]
    fn test_tessellate_rounded_rect() {
        let rect = Rect::from_xywh(0.0, 0.0, 100.0, 50.0);
        let mesh = tessellate_rounded_rect(rect, 10.0, TessQuality::MEDIUM);
        assert!(mesh.vertices.len() > 4);
        assert!(mesh.triangle_count() > 2);
    }

    #[test]
    fn test_path_tessellator_fill() {
        let mut tessellator = PathTessellator::new();
        let mut builder = PathBuilder::new();
        builder
            .move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0)
            .line_to(0.0, 100.0)
            .close();
        let path = builder.build();

        let mesh = tessellator.tessellate_fill(&path);
        assert!(!mesh.is_empty());
        // 5 vertices: 4 corners + 1 close point (returns to start)
        assert!(mesh.vertices.len() >= 4);
        assert!(mesh.triangle_count() >= 2);
    }

    #[test]
    fn test_path_tessellator_stroke() {
        let mut tessellator = PathTessellator::new();
        let mut builder = PathBuilder::new();
        builder
            .move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0);
        let path = builder.build();

        let mesh = tessellator.tessellate_stroke(&path, 2.0);
        assert!(!mesh.is_empty());
        assert!(mesh.vertices.len() >= 6);
    }

    #[test]
    fn test_quality_presets() {
        const { assert!(TessQuality::LOW.tolerance > TessQuality::HIGH.tolerance) };
        const { assert!(TessQuality::LOW.max_subdivisions < TessQuality::HIGH.max_subdivisions) };
    }

    fn rect_path() -> Path {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(10.0, 10.0)
            .line_to(0.0, 10.0)
            .close();
        b.build()
    }

    #[test]
    fn test_classify_fill_single_convex_direct() {
        let mut t = PathTessellator::new();
        assert_eq!(t.classify_fill(&rect_path()), FillStrategy::DirectConvex);
        assert!(t.tessellate_fill_convex(&rect_path()).is_some());
    }

    #[test]
    fn test_classify_fill_multi_contour_stencil() {
        // Two contours (outer + hole) must route to stencil-cover so the hole
        // stays a hole rather than being filled solid by per-contour ear-clip.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .line_to(30.0, 0.0)
            .line_to(30.0, 30.0)
            .line_to(0.0, 30.0)
            .close()
            .move_to(10.0, 10.0)
            .line_to(20.0, 10.0)
            .line_to(20.0, 20.0)
            .line_to(10.0, 20.0)
            .close();
        let path = b.build();
        let mut t = PathTessellator::new();
        assert_eq!(t.classify_fill(&path), FillStrategy::StencilCover);
        assert!(t.tessellate_fill_convex(&path).is_none());
    }

    #[test]
    fn test_classify_fill_concave_stencil() {
        // A single non-convex (L-shaped) contour must route to stencil-cover.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .line_to(30.0, 0.0)
            .line_to(30.0, 10.0)
            .line_to(10.0, 10.0)
            .line_to(10.0, 30.0)
            .line_to(0.0, 30.0)
            .close();
        let path = b.build();
        let mut t = PathTessellator::new();
        assert_eq!(t.classify_fill(&path), FillStrategy::StencilCover);
    }

    #[test]
    fn test_device_scale_increases_curve_subdivision() {
        // Regression: device-space tolerance. Magnifying the view matrix must
        // yield finer curve flattening (more vertices), not a fixed cap.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .cubic_to(0.0, 100.0, 100.0, 100.0, 100.0, 0.0);
        let path = b.build();

        let mut t1 = PathTessellator::new();
        let n1 = t1.tessellate_fill(&path).vertices.len();

        let mut t10 = PathTessellator::new();
        t10.set_view_matrix(&Matrix::scale(10.0, 10.0));
        let n10 = t10.tessellate_fill(&path).vertices.len();

        assert!(
            n10 > n1,
            "magnified path should subdivide finer: {n1} vs {n10}"
        );
    }

    #[test]
    fn test_line_after_close_starts_at_last_move() {
        // Regression: a Line following Close without a Move begins a new
        // contour at the last move point. The second contour must therefore
        // be a full triangle, not a dropped-first-vertex degenerate.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(10.0, 10.0)
            .close()
            // No move_to here: per SkPath, these lines start a new contour at
            // (0,0) (the last move point).
            .line_to(0.0, 20.0)
            .line_to(20.0, 20.0);
        let path = b.build();

        let mut t = PathTessellator::new();
        let contours = t.flatten_contours(&path);
        assert_eq!(contours.len(), 2, "expected two contours");
        // The second contour must begin at the last move point (0,0), not at
        // its first explicit line vertex.
        assert_eq!(contours[1][0], Point::new(0.0, 0.0));
    }

    #[test]
    fn test_stroke_miter_vs_bevel_geometry() {
        // A sharp right-angle corner: the miter join extends the outer corner
        // beyond the bevel, so the miter mesh reaches farther from the vertex.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 10.0).line_to(10.0, 10.0).line_to(10.0, 0.0);
        let path = b.build();

        let mut t = PathTessellator::new();
        let miter =
            t.tessellate_stroke_styled(&path, &StrokeStyle::new(4.0).with_join(StrokeJoin::Miter));
        let mut t2 = PathTessellator::new();
        let bevel =
            t2.tessellate_stroke_styled(&path, &StrokeStyle::new(4.0).with_join(StrokeJoin::Bevel));

        // Measure how far the outer corner extends along the join's outer
        // bisector ((1,1)/sqrt2) from the corner (10,10). Segment endpoints
        // project negatively, so this isolates the join geometry.
        let corner = Point::new(10.0, 10.0);
        let bx = std::f32::consts::FRAC_1_SQRT_2;
        let reach = |m: &TessMesh| {
            m.vertices
                .iter()
                .map(|v| (v.position[0] - corner.x).mul_add(bx, (v.position[1] - corner.y) * bx))
                .fold(f32::MIN, f32::max)
        };
        assert!(
            reach(&miter) > reach(&bevel) + 1e-3,
            "miter must reach farther than bevel at a sharp corner: {} vs {}",
            reach(&miter),
            reach(&bevel)
        );
    }

    #[test]
    fn test_stroke_square_cap_adds_geometry() {
        // A square cap extends the ends, producing more geometry than butt.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(10.0, 0.0);
        let path = b.build();

        let mut t = PathTessellator::new();
        let butt =
            t.tessellate_stroke_styled(&path, &StrokeStyle::new(4.0).with_cap(StrokeCap::Butt));
        let mut t2 = PathTessellator::new();
        let square =
            t2.tessellate_stroke_styled(&path, &StrokeStyle::new(4.0).with_cap(StrokeCap::Square));
        assert!(square.triangle_count() > butt.triangle_count());

        // Square cap reaches at least half_width beyond the end (x = 12).
        let max_x = square
            .vertices
            .iter()
            .map(|v| v.position[0])
            .fold(0.0f32, f32::max);
        assert!(
            max_x >= 11.9,
            "square cap should extend past endpoint, got {max_x}"
        );
    }

    #[test]
    fn test_polygon_signed_area() {
        // CCW square
        let ccw = [
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        assert!(polygon_signed_area(&ccw) > 0.0);
        // CW square
        let cw = [
            Point::new(0.0, 0.0),
            Point::new(0.0, 10.0),
            Point::new(10.0, 10.0),
            Point::new(10.0, 0.0),
        ];
        assert!(polygon_signed_area(&cw) < 0.0);
    }

    #[test]
    fn test_ear_clip_concave_l_shape() {
        // L-shaped hexagon (classic concave test): one vertex is reflex.
        // Outer contour (CCW): 6 vertices, 1 reflex → ear clip must emit
        // exactly 4 triangles, and no triangle may straddle the notch.
        //
        //  +----+
        //  |    |
        //  |    +--+
        //  |       |
        //  +-------+
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(20.0, 0.0),
            Point::new(20.0, 10.0),
            Point::new(10.0, 10.0),
            Point::new(10.0, 20.0),
            Point::new(0.0, 20.0),
        ];
        let mut mesh = TessMesh::new();
        for p in &pts {
            mesh.add_vertex(TessVertex::from_point(*p));
        }
        ear_clip_triangulate(&pts, 0, &mut mesh);
        assert_eq!(mesh.triangle_count(), 4, "L-shape must emit n-2 triangles");

        // Sanity: none of the emitted triangles should cover the notch.
        // The notch is at x in (10, 20), y in (10, 20) — the point (15, 15)
        // must NOT lie inside any triangle we emitted.
        let notch = Point::new(15.0, 15.0);
        for tri in mesh.indices.chunks(3) {
            let a = pts[tri[0] as usize];
            let b = pts[tri[1] as usize];
            let c = pts[tri[2] as usize];
            assert!(
                !point_in_triangle(notch, a, b, c),
                "triangle ({a:?},{b:?},{c:?}) covers the concave notch"
            );
        }
    }

    #[test]
    fn test_ear_clip_u_shape_concave() {
        // U-shape: classic non-convex. Fan triangulation would emit
        // triangles that cross the opening; ear clipping must not.
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(30.0, 0.0),
            Point::new(30.0, 30.0),
            Point::new(20.0, 30.0),
            Point::new(20.0, 10.0),
            Point::new(10.0, 10.0),
            Point::new(10.0, 30.0),
            Point::new(0.0, 30.0),
        ];
        let mut mesh = TessMesh::new();
        for p in &pts {
            mesh.add_vertex(TessVertex::from_point(*p));
        }
        ear_clip_triangulate(&pts, 0, &mut mesh);
        // 8 vertices → 6 triangles.
        assert_eq!(mesh.triangle_count(), 6);

        // Point in the middle of the opening must NOT be covered.
        let hole = Point::new(15.0, 20.0);
        for tri in mesh.indices.chunks(3) {
            let a = pts[tri[0] as usize];
            let b = pts[tri[1] as usize];
            let c = pts[tri[2] as usize];
            assert!(!point_in_triangle(hole, a, b, c));
        }
    }

    #[test]
    fn test_ear_clip_clockwise_input() {
        // Same L-shape but in CW order — the algorithm must still triangulate
        // it without choking on the flipped orientation.
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(0.0, 20.0),
            Point::new(10.0, 20.0),
            Point::new(10.0, 10.0),
            Point::new(20.0, 10.0),
            Point::new(20.0, 0.0),
        ];
        let mut mesh = TessMesh::new();
        for p in &pts {
            mesh.add_vertex(TessVertex::from_point(*p));
        }
        ear_clip_triangulate(&pts, 0, &mut mesh);
        assert_eq!(mesh.triangle_count(), 4);
    }

    #[test]
    fn test_path_tessellator_concave_fill() {
        // Full integration test: concave L-path through the tessellator.
        let mut tessellator = PathTessellator::new();
        let mut builder = PathBuilder::new();
        builder
            .move_to(0.0, 0.0)
            .line_to(20.0, 0.0)
            .line_to(20.0, 10.0)
            .line_to(10.0, 10.0)
            .line_to(10.0, 20.0)
            .line_to(0.0, 20.0)
            .close();
        let path = builder.build();

        let mesh = tessellator.tessellate_fill(&path);
        // Six polygon vertices → four triangles.
        assert_eq!(mesh.triangle_count(), 4);
    }

    #[test]
    fn test_conic_adaptive_subdivision() {
        // Near-unit weight should degrade to the quadratic path (few steps
        // for a gentle curve). Large-weight conic should emit more steps.
        let mut t = PathTessellator::with_quality(TessQuality {
            tolerance: 0.25,
            max_subdivisions: 64,
        });
        t.contour_points.push(Point::new(0.0, 0.0));
        t.flatten_conic(
            Point::new(0.0, 0.0),
            Point::new(5.0, 5.0),
            Point::new(10.0, 0.0),
            3.0,
        );
        let heavy = t.contour_points.len();
        t.contour_points.clear();

        t.contour_points.push(Point::new(0.0, 0.0));
        t.flatten_conic(
            Point::new(0.0, 0.0),
            Point::new(5.0, 0.01),
            Point::new(10.0, 0.0),
            3.0,
        );
        let gentle = t.contour_points.len();

        assert!(
            heavy > gentle,
            "sharp conic ({heavy} steps) should emit more segments than a gentle one ({gentle} steps)"
        );
    }
}
