//! Path data structure and iteration.

use skia_rs_core::cast::scalar_from_i32;
use skia_rs_core::{Point, Rect, Scalar};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU8, Ordering};

/// Path fill type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum FillType {
    /// Non-zero winding rule.
    #[default]
    Winding = 0,
    /// Even-odd rule.
    EvenOdd,
    /// Inverse non-zero winding.
    InverseWinding,
    /// Inverse even-odd.
    InverseEvenOdd,
}

impl FillType {
    /// Check if this is an inverse fill type.
    #[inline]
    #[must_use]
    pub const fn is_inverse(&self) -> bool {
        matches!(self, Self::InverseWinding | Self::InverseEvenOdd)
    }

    /// Convert to the inverse fill type.
    #[inline]
    #[must_use]
    pub const fn inverse(&self) -> Self {
        match self {
            Self::Winding => Self::InverseWinding,
            Self::EvenOdd => Self::InverseEvenOdd,
            Self::InverseWinding => Self::Winding,
            Self::InverseEvenOdd => Self::EvenOdd,
        }
    }
}

/// Path verb (command type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Verb {
    /// Move to a point.
    Move = 0,
    /// Line to a point.
    Line,
    /// Quadratic bezier.
    Quad,
    /// Conic (weighted quadratic).
    Conic,
    /// Cubic bezier.
    Cubic,
    /// Close the current contour.
    Close,
}

impl Verb {
    /// Number of points consumed by this verb.
    #[inline]
    #[must_use]
    pub const fn point_count(&self) -> usize {
        match self {
            Self::Move | Self::Line => 1,
            Self::Quad | Self::Conic => 2,
            Self::Cubic => 3,
            Self::Close => 0,
        }
    }
}

/// Path direction (clockwise or counter-clockwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PathDirection {
    /// Clockwise direction.
    #[default]
    CW = 0,
    /// Counter-clockwise direction.
    CCW,
}

/// Path convexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PathConvexity {
    /// Unknown convexity.
    #[default]
    Unknown = 0,
    /// Path is convex.
    Convex = 1,
    /// Path is concave.
    Concave = 2,
}

impl PathConvexity {
    const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Convex,
            2 => Self::Concave,
            _ => Self::Unknown,
        }
    }
}

/// A 2D geometric path.
#[derive(Debug)]
pub struct Path {
    /// Path verbs.
    pub(crate) verbs: SmallVec<[Verb; 16]>,
    /// Path points.
    pub(crate) points: SmallVec<[Point; 32]>,
    /// Conic weights.
    pub(crate) conic_weights: SmallVec<[Scalar; 4]>,
    /// Fill type.
    pub(crate) fill_type: FillType,
    /// Cached bounds (lazily computed).
    pub(crate) bounds: Option<Rect>,
    /// Cached convexity (stored as u8 for Send+Sync via `AtomicU8`).
    pub(crate) convexity: AtomicU8,
}

impl Default for Path {
    fn default() -> Self {
        Self {
            verbs: SmallVec::new(),
            points: SmallVec::new(),
            conic_weights: SmallVec::new(),
            fill_type: FillType::default(),
            bounds: None,
            convexity: AtomicU8::new(PathConvexity::Unknown as u8),
        }
    }
}

impl Clone for Path {
    fn clone(&self) -> Self {
        Self {
            verbs: self.verbs.clone(),
            points: self.points.clone(),
            conic_weights: self.conic_weights.clone(),
            fill_type: self.fill_type,
            bounds: self.bounds,
            convexity: AtomicU8::new(self.convexity.load(Ordering::Relaxed)),
        }
    }
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.verbs == other.verbs
            && self.points == other.points
            && self.conic_weights == other.conic_weights
            && self.fill_type == other.fill_type
    }
}

#[inline]
const fn axis_of(p: Point, axis: usize) -> Scalar {
    if axis == 0 { p.x } else { p.y }
}

#[inline]
fn record_axis_bound(
    axis: usize,
    val: Scalar,
    min_x: &mut Scalar,
    max_x: &mut Scalar,
    min_y: &mut Scalar,
    max_y: &mut Scalar,
) {
    if axis == 0 {
        if val < *min_x {
            *min_x = val;
        }
        if val > *max_x {
            *max_x = val;
        }
    } else {
        if val < *min_y {
            *min_y = val;
        }
        if val > *max_y {
            *max_y = val;
        }
    }
}

impl Path {
    /// Create a new empty path.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the fill type.
    #[inline]
    pub const fn fill_type(&self) -> FillType {
        self.fill_type
    }

    /// Set the fill type.
    #[inline]
    pub const fn set_fill_type(&mut self, fill_type: FillType) {
        self.fill_type = fill_type;
    }

    /// Check if the path is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.verbs.is_empty()
    }

    /// Get the number of verbs.
    #[inline]
    pub fn verb_count(&self) -> usize {
        self.verbs.len()
    }

    /// Get the number of points.
    #[inline]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Get the bounds of the path.
    pub fn bounds(&self) -> Rect {
        if let Some(bounds) = self.bounds {
            return bounds;
        }

        if self.points.is_empty() {
            return Rect::EMPTY;
        }

        // Track finiteness like SkPath::isFinite / computeBounds: any non-finite
        // coordinate yields empty bounds (upstream accumulates min/max and checks
        // the accumulator for finiteness at the end).
        let mut min_x = self.points[0].x;
        let mut min_y = self.points[0].y;
        let mut max_x = min_x;
        let mut max_y = min_y;

        for p in &self.points[1..] {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
            return Rect::EMPTY;
        }

        Rect::new(min_x, min_y, max_x, max_y)
    }

    /// Clear the path.
    #[inline]
    pub fn reset(&mut self) {
        self.verbs.clear();
        self.points.clear();
        self.conic_weights.clear();
        self.bounds = None;
    }

    /// Iterate over the path elements.
    pub const fn iter(&self) -> PathIter<'_> {
        PathIter {
            path: self,
            verb_index: 0,
            point_index: 0,
            weight_index: 0,
        }
    }

    /// Get the verbs slice.
    #[inline]
    pub fn verbs(&self) -> &[Verb] {
        &self.verbs
    }

    /// Get the points slice.
    #[inline]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Get the last point in the path.
    #[inline]
    pub fn last_point(&self) -> Option<Point> {
        self.points.last().copied()
    }

    /// Get the number of contours in the path.
    pub fn contour_count(&self) -> usize {
        self.verbs.iter().filter(|v| **v == Verb::Move).count()
    }

    /// Check if the path is closed.
    pub fn is_closed(&self) -> bool {
        self.verbs.last() == Some(&Verb::Close)
    }

    /// Check if the path represents a line.
    pub fn is_line(&self) -> bool {
        self.verbs.len() == 2 && self.verbs[0] == Verb::Move && self.verbs[1] == Verb::Line
    }

    /// Check if the path represents a rectangle.
    ///
    /// Ported from `SkPathPriv::IsRectContour` (single contour, `allowPartial =
    /// false`) with the `trivial_rect` fast-path: accepts the crate's own
    /// `add_rect` output (Move + 3 Lines + Close) and rejects non-rectangular
    /// horizontal/vertical staircases by counting direction changes and
    /// validating closure.
    pub fn is_rect(&self) -> Option<Rect> {
        if self.verbs.len() < 4 || self.points.len() < 4 {
            return None;
        }
        // Only line segments are allowed (kLine segment mask).
        for v in &self.verbs {
            if matches!(v, Verb::Quad | Verb::Conic | Verb::Cubic) {
                return None;
            }
        }

        if let Some(r) = trivial_rect(&self.points, &self.verbs) {
            return Some(r);
        }
        is_rect_contour(&self.points, &self.verbs)
    }

    /// Returns true if this path is a simple oval (ellipse).
    ///
    /// Verifies both verb structure (Move, 4 curves, Close) AND that the
    /// curve endpoints lie on the cardinal points of the bounding ellipse
    /// (left, right, top, bottom). This rejects 4-cubic paths that share
    /// the verb pattern but have arbitrary geometry.
    pub fn is_oval(&self) -> bool {
        let elements: Vec<_> = self.iter().collect();
        if elements.len() != 6 {
            return false;
        }

        let PathElement::Move(start) = elements[0] else {
            return false;
        };
        if !matches!(elements[5], PathElement::Close) {
            return false;
        }

        let all_cubic = elements[1..5]
            .iter()
            .all(|e| matches!(e, PathElement::Cubic(_, _, _)));
        let all_conic = elements[1..5]
            .iter()
            .all(|e| matches!(e, PathElement::Conic(_, _, _)));
        if !all_cubic && !all_conic {
            return false;
        }

        let bounds = self.bounds();
        let cx = (bounds.left + bounds.right) * 0.5;
        let cy = (bounds.top + bounds.bottom) * 0.5;
        if bounds.right - bounds.left <= 0.0 || bounds.bottom - bounds.top <= 0.0 {
            return false;
        }

        let tolerance = ((bounds.right - bounds.left) + (bounds.bottom - bounds.top)) * 1e-4;
        let on_cardinal = |p: Point| -> bool {
            let on_h = (p.y - cy).abs() < tolerance
                && ((p.x - bounds.left).abs() < tolerance
                    || (p.x - bounds.right).abs() < tolerance);
            let on_v = (p.x - cx).abs() < tolerance
                && ((p.y - bounds.top).abs() < tolerance
                    || (p.y - bounds.bottom).abs() < tolerance);
            on_h || on_v
        };

        if !on_cardinal(start) {
            return false;
        }

        for elem in &elements[1..5] {
            let (PathElement::Cubic(_, _, end) | PathElement::Conic(_, end, _)) = *elem else {
                return false;
            };
            if !on_cardinal(end) {
                return false;
            }
        }

        true
    }

    /// Get the convexity of the path.
    ///
    /// Ported from `SkPathPriv::ComputeConvexity`: per-contour and verb-aware.
    /// Any second contour is Concave; convexity along a single contour is
    /// determined by cross-product direction changes (including the closing
    /// edges) via the `Convexicator`, with no fixed magnitude threshold.
    pub fn convexity(&self) -> PathConvexity {
        let cached = PathConvexity::from_u8(self.convexity.load(Ordering::Relaxed));
        if cached != PathConvexity::Unknown {
            return cached;
        }

        let result = self.compute_convexity();
        self.convexity.store(result as u8, Ordering::Relaxed);
        result
    }

    fn compute_convexity(&self) -> PathConvexity {
        // Count trailing Move verbs to trim (like trim_trailing_moves).
        let mut vb_count = self.verbs.len();
        while vb_count > 0 && self.verbs[vb_count - 1] == Verb::Move {
            vb_count -= 1;
        }
        if vb_count == 0 {
            return PathConvexity::Convex; // degenerate
        }

        // Quick concave test by counting direction-sign changes.
        if convex::is_concave_by_sign(&self.points) {
            return PathConvexity::Concave;
        }

        let mut contour_count = 0;
        let mut needs_close = false;
        let mut state = convex::Convexicator::new();

        for elem in self.iter().take(vb_count) {
            let is_move = matches!(elem, PathElement::Move(_));

            if contour_count == 0 {
                if let PathElement::Move(p) = elem {
                    state.set_move_pt(p);
                } else {
                    contour_count += 1;
                    needs_close = true;
                }
            }

            if contour_count == 1 {
                match elem {
                    PathElement::Close | PathElement::Move(_) => {
                        if !state.close() {
                            return PathConvexity::Concave;
                        }
                        needs_close = false;
                        contour_count += 1;
                    }
                    PathElement::Line(p) => {
                        if !state.add_pt(p) {
                            return PathConvexity::Concave;
                        }
                    }
                    PathElement::Quad(c, e) | PathElement::Conic(c, e, _) => {
                        if !state.add_pt(c) || !state.add_pt(e) {
                            return PathConvexity::Concave;
                        }
                    }
                    PathElement::Cubic(c1, c2, e) => {
                        if !state.add_pt(c1) || !state.add_pt(c2) || !state.add_pt(e) {
                            return PathConvexity::Concave;
                        }
                    }
                }
            } else if contour_count >= 2 && !is_move {
                // The first contour has closed; any drawing verb means multiple
                // contours, which cannot be convex.
                return PathConvexity::Concave;
            }
        }

        if needs_close && !state.close() {
            return PathConvexity::Concave;
        }

        match state.first_direction() {
            convex::FirstDir::Unknown => {
                if state.reversals() >= 3 {
                    PathConvexity::Concave
                } else {
                    PathConvexity::Convex // kConvex_Degenerate
                }
            }
            _ => PathConvexity::Convex,
        }
    }

    /// Check if the path is convex.
    #[inline]
    pub fn is_convex(&self) -> bool {
        self.convexity() == PathConvexity::Convex
    }

    /// Get the first direction of the path.
    ///
    /// Ported from `SkPathPriv::ComputeFirstDirection`: loops over all contours
    /// and keeps the cross product of the contour that contains the global
    /// y-max (bottom-most point in y-down device space); `cross > 0 => CW`.
    pub fn direction(&self) -> Option<PathDirection> {
        if self.points.is_empty() {
            return None;
        }
        let bounds = self.bounds();
        if bounds == Rect::EMPTY {
            return None;
        }

        // initialize with our logical y-min (top)
        let mut ymax = bounds.top;
        let mut ymax_cross = 0.0f32;

        for (start, end) in self.contour_point_ranges() {
            let pts = &self.points[start..end];
            let n = pts.len();
            if n < 3 {
                continue;
            }
            let index = find_max_y(pts);
            if pts[index].y < ymax {
                continue;
            }

            // If there is more than 1 distinct point at the y-max, take the
            // x-min and x-max of them and subtract to compute the direction.
            #[allow(
                clippy::float_cmp,
                reason = "exact y equality mirrors upstream SkPathPriv::ComputeFirstDirection's bitwise comparison"
            )]
            let same_y = pts[(index + 1) % n].y == pts[index].y;
            let cross = if same_y {
                let (min_index, max_index) = find_min_max_x_at_y(pts, index);
                if min_index == max_index {
                    try_crossprod(pts, index)
                } else {
                    let min_i32 = i32::try_from(min_index).unwrap_or(i32::MAX);
                    let max_i32 = i32::try_from(max_index).unwrap_or(i32::MAX);
                    scalar_from_i32(min_i32) - scalar_from_i32(max_i32)
                }
            } else {
                try_crossprod(pts, index)
            };

            if cross != 0.0 {
                ymax = pts[index].y;
                ymax_cross = cross;
            }
        }

        if ymax_cross == 0.0 {
            None
        } else if ymax_cross > 0.0 {
            Some(PathDirection::CW)
        } else {
            Some(PathDirection::CCW)
        }
    }

    /// Return `(start, end)` point-index ranges, one per contour (Move..next Move).
    fn contour_point_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut pt = 0usize;
        let mut contour_start: Option<usize> = None;
        for &v in &self.verbs {
            match v {
                Verb::Move => {
                    if let Some(s) = contour_start.take() {
                        ranges.push((s, pt));
                    }
                    contour_start = Some(pt);
                    pt += 1;
                }
                Verb::Line => pt += 1,
                Verb::Quad | Verb::Conic => pt += 2,
                Verb::Cubic => pt += 3,
                Verb::Close => {}
            }
        }
        if let Some(s) = contour_start {
            ranges.push((s, pt));
        }
        ranges
    }

    /// Reverse the path direction.
    ///
    /// Ported from `SkPathPriv::ReverseAddPath`: walks contours back-to-front,
    /// emitting `Move` first, points in reverse order with per-verb point-count
    /// handling, and preserving one `Close` per originally-closed contour.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "faithful port of SkPathPriv::ReverseAddPath's signed back-to-front index walk; converting to unsigned arithmetic risks changing underflow/panic behavior"
    )]
    pub fn reverse(&mut self) {
        if self.verbs.is_empty() {
            return;
        }

        let mut nv: SmallVec<[Verb; 16]> = SmallVec::new();
        let mut np: SmallVec<[Point; 32]> = SmallVec::new();
        let mut nw: SmallVec<[Scalar; 4]> = SmallVec::new();

        let mut p: isize = self.points.len() as isize;
        let mut wi: isize = self.conic_weights.len() as isize;
        let mut need_move = true;
        let mut need_close = false;

        let mut vi = self.verbs.len();
        while vi > 0 {
            vi -= 1;
            let v = self.verbs[vi];
            let n = v.point_count() as isize;

            if need_move {
                p -= 1;
                let mp = self.points[p as usize];
                nv.push(Verb::Move);
                np.push(mp);
                need_move = false;
            }
            p -= n;
            match v {
                Verb::Move => {
                    if need_close {
                        nv.push(Verb::Close);
                        need_close = false;
                    }
                    need_move = true;
                    p += 1;
                }
                Verb::Line => {
                    nv.push(Verb::Line);
                    np.push(self.points[p as usize]);
                }
                Verb::Quad => {
                    nv.push(Verb::Quad);
                    np.push(self.points[(p + 1) as usize]);
                    np.push(self.points[p as usize]);
                }
                Verb::Conic => {
                    wi -= 1;
                    nv.push(Verb::Conic);
                    np.push(self.points[(p + 1) as usize]);
                    np.push(self.points[p as usize]);
                    nw.push(self.conic_weights[wi as usize]);
                }
                Verb::Cubic => {
                    nv.push(Verb::Cubic);
                    np.push(self.points[(p + 2) as usize]);
                    np.push(self.points[(p + 1) as usize]);
                    np.push(self.points[p as usize]);
                }
                Verb::Close => {
                    need_close = true;
                }
            }
        }
        if need_close {
            nv.push(Verb::Close);
        }

        self.verbs = nv;
        self.points = np;
        self.conic_weights = nw;
        self.bounds = None;
        self.convexity
            .store(PathConvexity::Unknown as u8, Ordering::Relaxed);
    }

    /// Transform the path by a matrix.
    pub fn transform(&mut self, matrix: &skia_rs_core::Matrix) {
        for point in &mut self.points {
            *point = matrix.map_point(*point);
        }
        self.bounds = None;
        self.convexity
            .store(PathConvexity::Unknown as u8, Ordering::Relaxed);
    }

    /// Create a transformed copy of the path.
    #[must_use]
    pub fn transformed(&self, matrix: &skia_rs_core::Matrix) -> Self {
        let mut result = self.clone();
        result.transform(matrix);
        result
    }

    /// Offset the path by (dx, dy).
    pub fn offset(&mut self, dx: Scalar, dy: Scalar) {
        for point in &mut self.points {
            point.x += dx;
            point.y += dy;
        }
        if let Some(ref mut bounds) = self.bounds {
            bounds.left += dx;
            bounds.right += dx;
            bounds.top += dy;
            bounds.bottom += dy;
        }
    }

    /// Check if a point is inside the path (using the fill rule).
    ///
    /// Ported from `SkPathPriv::Contains`: accumulates **signed** winding
    /// (+1/-1 per crossing direction) via `winding_line` on each edge, treats
    /// every contour as implicitly closed for hit-testing (like
    /// `SkPathEdgeIter`), uses inclusive bounds for the early-out, and XORs the
    /// result with the inverse-fill flag.
    pub fn contains(&self, point: Point) -> bool {
        use crate::flatten::{
            flatten_conic_adaptive, flatten_cubic_adaptive, flatten_quad_adaptive,
        };
        const TOL: Scalar = 0.1;

        let is_inverse = self.fill_type.is_inverse();
        if self.is_empty() {
            return is_inverse;
        }

        let bounds = self.bounds();
        if bounds == Rect::EMPTY || !contains_inclusive(&bounds, point) {
            return is_inverse;
        }

        let x = point.x;
        let y = point.y;
        let mut w = 0i32;
        let mut on_curve_count = 0i32;

        let mut current = Point::zero();
        let mut contour_start = Point::zero();
        let mut needs_close_line = false;
        let mut pts: Vec<Point> = Vec::with_capacity(32);

        for element in self {
            match element {
                PathElement::Move(p) => {
                    if needs_close_line {
                        w += winding_line(current, contour_start, x, y, &mut on_curve_count);
                        needs_close_line = false;
                    }
                    current = p;
                    contour_start = p;
                }
                PathElement::Line(end) => {
                    w += winding_line(current, end, x, y, &mut on_curve_count);
                    current = end;
                    needs_close_line = true;
                }
                PathElement::Quad(ctrl, end) => {
                    pts.clear();
                    flatten_quad_adaptive(&mut pts, current, ctrl, end, TOL);
                    let mut prev = current;
                    for pt in &pts {
                        w += winding_line(prev, *pt, x, y, &mut on_curve_count);
                        prev = *pt;
                    }
                    current = end;
                    needs_close_line = true;
                }
                PathElement::Conic(ctrl, end, weight) => {
                    pts.clear();
                    flatten_conic_adaptive(&mut pts, current, ctrl, end, weight, TOL);
                    let mut prev = current;
                    for pt in &pts {
                        w += winding_line(prev, *pt, x, y, &mut on_curve_count);
                        prev = *pt;
                    }
                    current = end;
                    needs_close_line = true;
                }
                PathElement::Cubic(c1, c2, end) => {
                    pts.clear();
                    flatten_cubic_adaptive(&mut pts, current, c1, c2, end, TOL);
                    let mut prev = current;
                    for pt in &pts {
                        w += winding_line(prev, *pt, x, y, &mut on_curve_count);
                        prev = *pt;
                    }
                    current = end;
                    needs_close_line = true;
                }
                PathElement::Close => {
                    if needs_close_line {
                        w += winding_line(current, contour_start, x, y, &mut on_curve_count);
                        needs_close_line = false;
                    }
                    current = contour_start;
                }
            }
        }
        if needs_close_line {
            w += winding_line(current, contour_start, x, y, &mut on_curve_count);
        }

        let even_odd_fill = matches!(self.fill_type, FillType::EvenOdd | FillType::InverseEvenOdd);
        if even_odd_fill {
            w &= 1;
        }
        if w != 0 {
            return !is_inverse;
        }
        if on_curve_count <= 1 {
            return (on_curve_count != 0) ^ is_inverse;
        }
        if (on_curve_count & 1) != 0 || even_odd_fill {
            return ((on_curve_count & 1) != 0) ^ is_inverse;
        }
        // Winding fill with the point touching an even number of curves: upstream
        // resolves coincidence via tangent comparison. With flattened curves an
        // exact even on-curve hit is essentially unreachable; treat as a boundary
        // touch (inside for non-inverse fills).
        !is_inverse
    }

    /// Returns the tight bounding rectangle of this path.
    ///
    /// Unlike `bounds()`, this computes the bounds from the actual curve
    /// extents, not the control-point bounding box. For cubic and quadratic
    /// Bezier curves with control points outside the actual curve range,
    /// `tight_bounds()` may be significantly smaller than `bounds()`.
    ///
    /// Conics fall back to control-polygon bounds (exact extrema for rational
    /// curves would require solving a quartic).
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "faithful port of Skia's per-verb curve-extrema tight-bounds computation; short names (s, e, cv, c1v, c2v, a, b, cc) mirror the algebraic derivation"
    )]
    pub fn tight_bounds(&self) -> Rect {
        if self.verbs.is_empty() {
            return Rect::EMPTY;
        }

        let mut min_x = Scalar::INFINITY;
        let mut min_y = Scalar::INFINITY;
        let mut max_x = Scalar::NEG_INFINITY;
        let mut max_y = Scalar::NEG_INFINITY;

        let include = |p: Point,
                       min_x: &mut Scalar,
                       min_y: &mut Scalar,
                       max_x: &mut Scalar,
                       max_y: &mut Scalar| {
            if p.x < *min_x {
                *min_x = p.x;
            }
            if p.y < *min_y {
                *min_y = p.y;
            }
            if p.x > *max_x {
                *max_x = p.x;
            }
            if p.y > *max_y {
                *max_y = p.y;
            }
        };

        let mut current = Point::new(0.0, 0.0);

        for elem in self {
            match elem {
                PathElement::Move(p) | PathElement::Line(p) => {
                    include(p, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    current = p;
                }
                PathElement::Quad(c, p) => {
                    include(current, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    include(p, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    // Quadratic extremum per axis: t = (start - ctrl) / (start - 2*ctrl + end)
                    for axis in 0..2 {
                        let s = axis_of(current, axis);
                        let cv = axis_of(c, axis);
                        let e = axis_of(p, axis);
                        let denom = 2.0f32.mul_add(-cv, s) + e;
                        if denom.abs() > 1e-9 {
                            let t = (s - cv) / denom;
                            if t > 0.0 && t < 1.0 {
                                let mt = 1.0 - t;
                                let val =
                                    (t * t).mul_add(e, (mt * mt).mul_add(s, 2.0 * mt * t * cv));
                                record_axis_bound(
                                    axis, val, &mut min_x, &mut max_x, &mut min_y, &mut max_y,
                                );
                            }
                        }
                    }
                    current = p;
                }
                PathElement::Cubic(c1, c2, p) => {
                    include(current, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    include(p, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    // Cubic extrema per axis: solve 3*a*t^2 + 2*b*t + c = 0 where
                    // B'(t) = 3*(1-t)^2*(c1-s) + 6*(1-t)*t*(c2-c1) + 3*t^2*(e-c2)
                    // Expanding into quadratic: a*t^2 + b*t + cc
                    for axis in 0..2 {
                        let s = axis_of(current, axis);
                        let c1v = axis_of(c1, axis);
                        let c2v = axis_of(c2, axis);
                        let e = axis_of(p, axis);
                        let a = 3.0 * (3.0f32.mul_add(c1v, 3.0f32.mul_add(-c2v, e)) - s);
                        let b = 6.0 * (2.0f32.mul_add(-c1v, c2v) + s);
                        let cc = 3.0 * (c1v - s);

                        let mut roots: [Scalar; 2] = [Scalar::NAN, Scalar::NAN];
                        let mut n_roots = 0;

                        if a.abs() < 1e-9 {
                            // Linear: b*t + cc = 0
                            if b.abs() > 1e-9 {
                                let t = -cc / b;
                                roots[0] = t;
                                n_roots = 1;
                            }
                        } else {
                            let disc = (4.0 * a).mul_add(-cc, b * b);
                            if disc >= 0.0 {
                                let sqrt_disc = disc.sqrt();
                                roots[0] = (-b + sqrt_disc) / (2.0 * a);
                                roots[1] = (-b - sqrt_disc) / (2.0 * a);
                                n_roots = 2;
                            }
                        }

                        for &t in &roots[..n_roots] {
                            if t.is_finite() && t > 0.0 && t < 1.0 {
                                let mt = 1.0 - t;
                                let val = (t * t * t).mul_add(
                                    e,
                                    (3.0 * mt * t * t).mul_add(
                                        c2v,
                                        (mt * mt * mt).mul_add(s, 3.0 * mt * mt * t * c1v),
                                    ),
                                );
                                record_axis_bound(
                                    axis, val, &mut min_x, &mut max_x, &mut min_y, &mut max_y,
                                );
                            }
                        }
                    }
                    current = p;
                }
                PathElement::Conic(c, p, _w) => {
                    // Rational extrema require quartic solve; use control polygon
                    // as a correct (if looser) upper bound.
                    include(current, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    include(c, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    include(p, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                    current = p;
                }
                PathElement::Close => {}
            }
        }

        if min_x == Scalar::INFINITY {
            return Rect::EMPTY;
        }
        Rect::new(min_x, min_y, max_x, max_y)
    }

    /// Get the total length of the path.
    pub fn length(&self) -> Scalar {
        use crate::flatten::{
            flatten_conic_adaptive, flatten_cubic_adaptive, flatten_quad_adaptive,
        };

        const TOL: Scalar = 0.25;
        let mut total = 0.0;
        let mut current = Point::zero();
        let mut contour_start = Point::zero();
        let mut pts: Vec<Point> = Vec::with_capacity(32);

        for element in self {
            match element {
                PathElement::Move(p) => {
                    current = p;
                    contour_start = p;
                }
                PathElement::Line(end) => {
                    total += current.distance(&end);
                    current = end;
                }
                PathElement::Quad(ctrl, end) => {
                    pts.clear();
                    flatten_quad_adaptive(&mut pts, current, ctrl, end, TOL);
                    let mut prev = current;
                    for pt in &pts {
                        total += prev.distance(pt);
                        prev = *pt;
                    }
                    current = end;
                }
                PathElement::Conic(ctrl, end, w) => {
                    pts.clear();
                    flatten_conic_adaptive(&mut pts, current, ctrl, end, w, TOL);
                    let mut prev = current;
                    for pt in &pts {
                        total += prev.distance(pt);
                        prev = *pt;
                    }
                    current = end;
                }
                PathElement::Cubic(c1, c2, end) => {
                    pts.clear();
                    flatten_cubic_adaptive(&mut pts, current, c1, c2, end, TOL);
                    let mut prev = current;
                    for pt in &pts {
                        total += prev.distance(pt);
                        prev = *pt;
                    }
                    current = end;
                }
                PathElement::Close => {
                    total += current.distance(&contour_start);
                    current = contour_start;
                }
            }
        }

        total
    }
}

/// Returns the first pt with the maximum Y coordinate (ported from `find_max_y`).
fn find_max_y(pts: &[Point]) -> usize {
    let mut max = pts[0].y;
    let mut first_index = 0;
    for (i, p) in pts.iter().enumerate().skip(1) {
        if p.y > max {
            max = p.y;
            first_index = i;
        }
    }
    first_index
}

/// Find a point different from `pts[index]`, moving by `inc` (ported from `find_diff_pt`).
fn find_diff_pt(pts: &[Point], index: usize, inc: usize) -> usize {
    let n = pts.len();
    let mut i = index;
    loop {
        i = (i + inc) % n;
        if i == index {
            break;
        }
        if pts[index] != pts[i] {
            break;
        }
    }
    i
}

/// From `index` moving forward, find the xmin and xmax of contiguous same-Y points.
/// Returns `(min_index, max_index)` (ported from `find_min_max_x_at_y`).
fn find_min_max_x_at_y(pts: &[Point], index: usize) -> (usize, usize) {
    let y = pts[index].y;
    let mut min = pts[index].x;
    let mut max = min;
    let mut min_index = index;
    let mut max_index = index;
    #[allow(
        clippy::float_cmp,
        reason = "exact y equality mirrors upstream find_min_max_x_at_y's bitwise comparison"
    )]
    for (i, p) in pts.iter().enumerate().skip(index + 1) {
        if p.y != y {
            break;
        }
        let x = p.x;
        if x < min {
            min = x;
            min_index = i;
        } else if x > max {
            max = x;
            max_index = i;
        }
    }
    (min_index, max_index)
}

/// cross product of (p1 - p0) and (p2 - p0) (ported from `cross_prod`, f64 promotion).
#[allow(
    clippy::cast_possible_truncation,
    reason = "f64 promotion is deliberately used to recover precision near-zero, then narrowed back to Scalar (f32); this is the whole point of the promotion and matches upstream cross_prod"
)]
fn cross_prod(p0: Point, p1: Point, p2: Point) -> Scalar {
    let cross = (p1.x - p0.x).mul_add(p2.y - p0.y, -((p1.y - p0.y) * (p2.x - p0.x)));
    if cross == 0.0 {
        let p0x = f64::from(p0.x);
        let p0y = f64::from(p0.y);
        let p1x = f64::from(p1.x);
        let p1y = f64::from(p1.y);
        let p2x = f64::from(p2.x);
        let p2y = f64::from(p2.y);
        return (p1x - p0x).mul_add(p2y - p0y, -((p1y - p0y) * (p2x - p0x))) as Scalar;
    }
    cross
}

/// The `TRY_CROSSPROD` path from `ComputeFirstDirection`.
#[allow(
    clippy::float_cmp,
    reason = "exact equality mirrors upstream TRY_CROSSPROD's bitwise comparisons"
)]
fn try_crossprod(pts: &[Point], index: usize) -> Scalar {
    let n = pts.len();
    let prev = find_diff_pt(pts, index, n - 1);
    if prev == index {
        return 0.0;
    }
    let next = find_diff_pt(pts, index, 1);
    let mut cross = cross_prod(pts[prev], pts[index], pts[next]);
    if cross == 0.0 && pts[prev].y == pts[index].y && pts[next].y == pts[index].y {
        cross = pts[index].x - pts[next].x;
    }
    cross
}

/// Ported from `rect_make_dir`: encode an axis-aligned edge direction (0..3).
#[inline]
fn rect_make_dir(dx: Scalar, dy: Scalar) -> i32 {
    i32::from(dx != 0.0) | (i32::from(dx > 0.0 || dy > 0.0) << 1)
}

/// Build a sorted rect spanning two corner points.
#[inline]
const fn rect_from_corners(a: Point, b: Point) -> Rect {
    Rect::new(a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y))
}

/// Fast-path port of `trivial_rect`: exactly Move, Line, Line, Line, Close
/// over 4 points forming three right corners.
fn trivial_rect(pts: &[Point], vbs: &[Verb]) -> Option<Rect> {
    const TRIVIAL: [Verb; 5] = [Verb::Move, Verb::Line, Verb::Line, Verb::Line, Verb::Close];
    if pts.len() != 4 || vbs.len() != TRIVIAL.len() || vbs != TRIVIAL {
        return None;
    }
    let v0 = Point::new(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    let v1 = Point::new(pts[2].x - pts[1].x, pts[2].y - pts[1].y);
    let v2 = Point::new(pts[3].x - pts[2].x, pts[3].y - pts[2].y);
    let v3 = Point::new(pts[0].x - pts[3].x, pts[0].y - pts[3].y);

    // One vector axis-aligned, and each following vector orthogonal to the last.
    let axis_aligned = |a: &Point| (a.x == 0.0) ^ (a.y == 0.0);
    let orthogonal =
        |a: &Point, b: &Point| ((a.x == 0.0) ^ (b.x == 0.0)) && ((a.y == 0.0) ^ (b.y == 0.0));

    if !(axis_aligned(&v0) && orthogonal(&v0, &v1) && orthogonal(&v1, &v2) && orthogonal(&v2, &v3))
    {
        return None;
    }
    Some(rect_from_corners(pts[0], pts[2]))
}

/// General port of `SkPathPriv::IsRectContour` for a single contour
/// (`allowPartial = false`).
#[allow(
    clippy::too_many_lines,
    reason = "faithful port of SkPathPriv::IsRectContour's edge-direction state machine"
)]
fn is_rect_contour(points: &[Point], verbs: &[Verb]) -> Option<Rect> {
    let verb_cnt = verbs.len();
    let mut pi = 0usize;
    let mut curr_verb = 0usize;

    let mut corners = 0usize;
    let mut line_start = Point::zero();
    let mut first_pt = Point::zero();
    let mut last_pt = Point::zero();
    let mut first_corner = Point::zero();
    let mut third_corner = Point::zero();
    let mut directions = [-1i32; 5];
    let mut closed_or_moved = false;
    let mut auto_close = false;

    while curr_verb < verb_cnt {
        let verb = verbs[curr_verb];
        match verb {
            Verb::Close | Verb::Line => {
                if verb == Verb::Close {
                    auto_close = true;
                } else {
                    last_pt = points[pi];
                }
                let line_end = if verb == Verb::Close {
                    first_pt
                } else {
                    let p = points[pi];
                    pi += 1;
                    p
                };
                let dx = line_end.x - line_start.x;
                let dy = line_end.y - line_start.y;
                if dx != 0.0 && dy != 0.0 {
                    return None; // diagonal
                }
                if !line_end.is_finite() || !line_start.is_finite() {
                    return None; // infinity or NaN
                }
                if line_start == line_end {
                    // single point on side is OK
                } else {
                    let next_direction = rect_make_dir(dx, dy);
                    if corners == 0 {
                        directions[0] = next_direction;
                        corners = 1;
                        closed_or_moved = false;
                        line_start = line_end;
                    } else if closed_or_moved {
                        return None; // closed followed by a line
                    } else if auto_close && next_direction == directions[0] {
                        // colinear with first
                    } else {
                        closed_or_moved = auto_close;
                        if directions[corners - 1] == next_direction {
                            if corners == 3 && verb == Verb::Line {
                                third_corner = line_end;
                            }
                        } else {
                            directions[corners] = next_direction;
                            corners += 1;
                            match corners {
                                2 => first_corner = line_start,
                                3 => {
                                    if (directions[0] ^ directions[2]) != 2 {
                                        return None;
                                    }
                                    third_corner = line_end;
                                }
                                4 => {
                                    if (directions[1] ^ directions[3]) != 2 {
                                        return None;
                                    }
                                }
                                _ => return None, // too many direction changes
                            }
                        }
                        line_start = line_end;
                    }
                }
            }
            Verb::Quad | Verb::Conic | Verb::Cubic => return None,
            Verb::Move => {
                if corners == 0 {
                    first_pt = points[pi];
                } else {
                    let cx = first_pt.x - last_pt.x;
                    let cy = first_pt.y - last_pt.y;
                    if cx != 0.0 && cy != 0.0 {
                        return None; // diagonal
                    }
                }
                line_start = points[pi];
                pi += 1;
                closed_or_moved = true;
            }
        }
        curr_verb += 1;
    }

    if !(3..=4).contains(&corners) {
        return None;
    }
    let cx = first_pt.x - last_pt.x;
    let cy = first_pt.y - last_pt.y;
    if cx != 0.0 && cy != 0.0 {
        return None;
    }
    Some(rect_from_corners(first_corner, third_corner))
}

/// Inclusive point-in-rect test (ported from `contains_inclusive`).
#[inline]
fn contains_inclusive(r: &Rect, p: Point) -> bool {
    r.left <= p.x && p.x <= r.right && r.top <= p.y && p.y <= r.bottom
}

/// True if `b` lies between `a` and `c` (inclusive), ported from Skia's `between`.
#[inline]
fn between(a: Scalar, b: Scalar, c: Scalar) -> bool {
    (a - b) * (c - b) <= 0.0
}

#[inline]
fn sign_as_int(x: Scalar) -> i32 {
    if x < 0.0 { -1 } else { i32::from(x > 0.0) }
}

/// Ported from `checkOnCurve`.
#[inline]
#[allow(
    clippy::float_cmp,
    reason = "exact equality mirrors upstream checkOnCurve's bitwise comparisons"
)]
fn check_on_curve(x: Scalar, y: Scalar, start: Point, end: Point) -> bool {
    if start.y == end.y {
        between(start.x, x, end.x) && x != end.x
    } else {
        x == start.x && y == start.y
    }
}

/// Signed winding contribution of a line edge for the point (x, y).
///
/// Ported from `winding_line` in `SkPathPriv.cpp`: returns +1/-1 for a crossing
/// to the right of the point (sign encodes the edge's y-direction), 0 otherwise,
/// and increments `on_curve_count` when the point lies on the edge.
#[allow(
    clippy::float_cmp,
    reason = "exact equality mirrors upstream winding_line's bitwise comparisons"
)]
fn winding_line(a: Point, b: Point, x: Scalar, y: Scalar, on_curve_count: &mut i32) -> i32 {
    let x0 = a.x;
    let mut y0 = a.y;
    let x1 = b.x;
    let mut y1 = b.y;

    let dy = y1 - y0;
    let (mut dir, swapped) = if y0 > y1 { (-1, true) } else { (1, false) };
    if swapped {
        std::mem::swap(&mut y0, &mut y1);
    }
    if y < y0 || y > y1 {
        return 0;
    }
    if check_on_curve(x, y, a, b) {
        *on_curve_count += 1;
        return 0;
    }
    if y == y1 {
        return 0;
    }
    let cross = (x1 - x0).mul_add(y - a.y, -(dy * (x - x0)));

    if cross == 0.0 {
        if x != x1 || y != b.y {
            *on_curve_count += 1;
        }
        dir = 0;
    } else if sign_as_int(cross) == dir {
        dir = 0;
    }
    dir
}

/// Convexity computation, ported from the `Convexicator` in `SkPathPriv.cpp`.
mod convex {
    use super::sign_as_int;
    use skia_rs_core::Point;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FirstDir {
        Cw,
        Ccw,
        Unknown,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DirChange {
        Left,
        Right,
        Straight,
        Backwards,
        Unknown,
        Invalid,
    }

    #[inline]
    fn vsub(a: Point, b: Point) -> Point {
        Point::new(a.x - b.x, a.y - b.y)
    }

    /// Quick concave test: counts direction-sign changes around the contour.
    /// Ported from `Convexicator::IsConcaveBySign`.
    #[allow(
        clippy::similar_names,
        reason = "dxes/dyes and last_sx/last_sy mirror upstream Convexicator field names"
    )]
    pub fn is_concave_by_sign(pts: &[Point]) -> bool {
        let count = pts.len();
        if count <= 3 {
            return false;
        }
        let first_pt = pts[0];
        let mut curr_pt = pts[0];
        let mut dxes = 0i32;
        let mut dyes = 0i32;
        let mut last_sx = 2i32;
        let mut last_sy = 2i32;

        let process = |next: Point,
                       curr: &mut Point,
                       dxes: &mut i32,
                       dyes: &mut i32,
                       last_sx: &mut i32,
                       last_sy: &mut i32|
         -> Option<bool> {
            let vx = next.x - curr.x;
            let vy = next.y - curr.y;
            if vx != 0.0 || vy != 0.0 {
                if !vx.is_finite() || !vy.is_finite() {
                    return Some(true);
                }
                let sx = i32::from(vx < 0.0);
                let sy = i32::from(vy < 0.0);
                *dxes += i32::from(sx != *last_sx);
                *dyes += i32::from(sy != *last_sy);
                if *dxes > 3 || *dyes > 3 {
                    return Some(true);
                }
                *last_sx = sx;
                *last_sy = sy;
            }
            *curr = next;
            None
        };

        for &p in &pts[1..count] {
            if let Some(r) = process(
                p,
                &mut curr_pt,
                &mut dxes,
                &mut dyes,
                &mut last_sx,
                &mut last_sy,
            ) {
                return r;
            }
        }
        // closing edge back to the first point
        if let Some(r) = process(
            first_pt,
            &mut curr_pt,
            &mut dxes,
            &mut dyes,
            &mut last_sx,
            &mut last_sy,
        ) {
            return r;
        }
        false
    }

    /// Only valid for a single contour.
    pub struct Convexicator {
        first_pt: Point,
        first_vec: Point,
        last_pt: Point,
        last_vec: Point,
        expected_dir: DirChange,
        first_direction: FirstDir,
        reversals: i32,
    }

    impl Convexicator {
        pub const fn new() -> Self {
            Self {
                first_pt: Point::zero(),
                first_vec: Point::zero(),
                last_pt: Point::zero(),
                last_vec: Point::zero(),
                expected_dir: DirChange::Invalid,
                first_direction: FirstDir::Unknown,
                reversals: 0,
            }
        }

        pub const fn first_direction(&self) -> FirstDir {
            self.first_direction
        }

        pub const fn reversals(&self) -> i32 {
            self.reversals
        }

        pub const fn set_move_pt(&mut self, pt: Point) {
            self.first_pt = pt;
            self.last_pt = pt;
            self.expected_dir = DirChange::Invalid;
        }

        pub fn add_pt(&mut self, pt: Point) -> bool {
            if self.last_pt == pt {
                return true;
            }
            if self.first_pt == self.last_pt
                && self.expected_dir == DirChange::Invalid
                && self.last_vec.is_zero()
            {
                self.last_vec = vsub(pt, self.last_pt);
                self.first_vec = self.last_vec;
            } else if !self.add_vec(vsub(pt, self.last_pt)) {
                return false;
            }
            self.last_pt = pt;
            true
        }

        pub fn close(&mut self) -> bool {
            let fp = self.first_pt;
            let fv = self.first_vec;
            self.add_pt(fp) && self.add_vec(fv)
        }

        fn direction_change(&self, cur_vec: Point) -> DirChange {
            let cross = self.last_vec.cross(&cur_vec);
            if !cross.is_finite() {
                return DirChange::Unknown;
            }
            if cross == 0.0 {
                return if self.last_vec.dot(&cur_vec) < 0.0 {
                    DirChange::Backwards
                } else {
                    DirChange::Straight
                };
            }
            if sign_as_int(cross) == 1 {
                DirChange::Right
            } else {
                DirChange::Left
            }
        }

        fn add_vec(&mut self, cur_vec: Point) -> bool {
            let dir = self.direction_change(cur_vec);
            match dir {
                DirChange::Left | DirChange::Right => {
                    if self.expected_dir == DirChange::Invalid {
                        self.expected_dir = dir;
                        self.first_direction = if dir == DirChange::Right {
                            FirstDir::Cw
                        } else {
                            FirstDir::Ccw
                        };
                    } else if dir != self.expected_dir {
                        self.first_direction = FirstDir::Unknown;
                        return false;
                    }
                    self.last_vec = cur_vec;
                }
                DirChange::Straight => {}
                DirChange::Backwards => {
                    // allow the path to reverse direction twice
                    self.last_vec = cur_vec;
                    self.reversals += 1;
                    return self.reversals < 3;
                }
                DirChange::Unknown => {
                    // Non-finite cross product: cannot determine convexity.
                    return false;
                }
                DirChange::Invalid => unreachable!("invalid direction change flag"),
            }
            true
        }
    }
}

/// A path element from iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathElement {
    /// Move to point.
    Move(Point),
    /// Line to point.
    Line(Point),
    /// Quadratic bezier (control, end).
    Quad(Point, Point),
    /// Conic (control, end, weight).
    Conic(Point, Point, Scalar),
    /// Cubic bezier (control1, control2, end).
    Cubic(Point, Point, Point),
    /// Close the path.
    Close,
}

impl<'a> IntoIterator for &'a Path {
    type Item = PathElement;
    type IntoIter = PathIter<'a>;

    fn into_iter(self) -> PathIter<'a> {
        self.iter()
    }
}

/// Iterator over path elements.
pub struct PathIter<'a> {
    path: &'a Path,
    verb_index: usize,
    point_index: usize,
    weight_index: usize,
}

impl Iterator for PathIter<'_> {
    type Item = PathElement;

    fn next(&mut self) -> Option<Self::Item> {
        if self.verb_index >= self.path.verbs.len() {
            return None;
        }

        let verb = self.path.verbs[self.verb_index];
        self.verb_index += 1;

        let element = match verb {
            Verb::Move => {
                let p = self.path.points[self.point_index];
                self.point_index += 1;
                PathElement::Move(p)
            }
            Verb::Line => {
                let p = self.path.points[self.point_index];
                self.point_index += 1;
                PathElement::Line(p)
            }
            Verb::Quad => {
                let p1 = self.path.points[self.point_index];
                let p2 = self.path.points[self.point_index + 1];
                self.point_index += 2;
                PathElement::Quad(p1, p2)
            }
            Verb::Conic => {
                let p1 = self.path.points[self.point_index];
                let p2 = self.path.points[self.point_index + 1];
                let w = self.path.conic_weights[self.weight_index];
                self.point_index += 2;
                self.weight_index += 1;
                PathElement::Conic(p1, p2, w)
            }
            Verb::Cubic => {
                let p1 = self.path.points[self.point_index];
                let p2 = self.path.points[self.point_index + 1];
                let p3 = self.path.points[self.point_index + 2];
                self.point_index += 3;
                PathElement::Cubic(p1, p2, p3)
            }
            Verb::Close => PathElement::Close,
        };

        Some(element)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PathBuilder;

    #[test]
    fn test_is_oval_true_for_actual_oval() {
        let mut builder = PathBuilder::new();
        builder.add_oval(&Rect::new(0.0, 0.0, 100.0, 50.0));
        let path = builder.build();
        assert!(path.is_oval(), "add_oval result should report is_oval=true");
    }

    #[test]
    fn test_is_oval_false_for_random_cubics() {
        // 4 cubics with arbitrary control points — should NOT be detected.
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.cubic_to(10.0, 0.0, 20.0, 5.0, 30.0, 10.0);
        builder.cubic_to(40.0, 15.0, 50.0, 20.0, 60.0, 25.0);
        builder.cubic_to(70.0, 30.0, 80.0, 35.0, 90.0, 40.0);
        builder.cubic_to(95.0, 45.0, 100.0, 47.0, 0.0, 0.0);
        builder.close();
        let path = builder.build();
        assert!(
            !path.is_oval(),
            "Random 4-cubic path should not be detected as oval"
        );
    }

    #[test]
    fn test_path_convexity_returns_consistent_result() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0);
        builder.line_to(10.0, 10.0);
        builder.line_to(0.0, 10.0);
        builder.close();
        let path = builder.build();

        let c1 = path.convexity();
        let c2 = path.convexity();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_tight_bounds_smaller_than_bounds_for_curves() {
        // A cubic Bezier where control points extend far beyond the actual curve.
        // The "loose" bounds (bounds()) include control points; tight should be tighter.
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.cubic_to(50.0, 100.0, 50.0, -100.0, 100.0, 0.0);
        let path = builder.build();

        let loose = path.bounds();
        let tight = path.tight_bounds();

        // Loose bounds include y=±100 control points
        assert!(
            loose.top <= -99.0 || loose.bottom >= 99.0,
            "Loose bounds should include control points (top={}, bottom={})",
            loose.top,
            loose.bottom
        );

        // Tight bounds should be strictly tighter on at least one of top/bottom
        assert!(
            tight.top > loose.top || tight.bottom < loose.bottom,
            "Tight bounds should be tighter than loose for this off-axis cubic"
        );

        // For this symmetric S-cubic, actual extrema are approximately ±19.245
        assert!(
            tight.bottom < 30.0 && tight.top > -30.0,
            "Tight bounds should reflect actual curve range (got top={}, bottom={})",
            tight.top,
            tight.bottom
        );
    }

    #[test]
    fn test_tight_bounds_same_as_bounds_for_lines() {
        // For line-only paths, tight_bounds should equal bounds
        let mut builder = PathBuilder::new();
        builder.move_to(10.0, 20.0);
        builder.line_to(30.0, 40.0);
        let path = builder.build();

        let loose = path.bounds();
        let tight = path.tight_bounds();
        assert!((loose.left - tight.left).abs() < 1e-4);
        assert!((loose.right - tight.right).abs() < 1e-4);
        assert!((loose.top - tight.top).abs() < 1e-4);
        assert!((loose.bottom - tight.bottom).abs() < 1e-4);
    }

    #[test]
    fn test_length_quarter_circle_close_to_pi_over_2() {
        // Quarter-circle arc as a conic with weight sqrt(2)/2.
        // Radius 1, circumference of full circle = 2π, quarter = π/2 ≈ 1.5708.
        let mut builder = PathBuilder::new();
        builder.move_to(1.0, 0.0);
        builder.conic_to(1.0, 1.0, 0.0, 1.0, std::f32::consts::FRAC_1_SQRT_2);
        let path = builder.build();
        let len = path.length();
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(
            (len - expected).abs() < 0.05,
            "expected ~π/2 = {expected}, got {len}"
        );
    }

    #[test]
    fn test_length_tight_cubic_less_than_control_polygon() {
        // Straight-line cubic: all 4 control points collinear. Length = chord length.
        // Control polygon length = 3x chord length. Old impl would return 3x.
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.cubic_to(1.0, 0.0, 2.0, 0.0, 3.0, 0.0);
        let path = builder.build();
        let len = path.length();
        assert!((len - 3.0).abs() < 0.1, "expected 3.0, got {len}");
    }

    #[test]
    fn test_direction_add_rect_is_cw() {
        // add_rect emits CW geometry in y-down device space; must report CW.
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        let path = b.build();
        assert_eq!(path.direction(), Some(PathDirection::CW));
    }

    #[test]
    fn test_is_rect_accepts_add_rect_output() {
        // add_rect emits Move + 3 Lines + Close (the 4th edge is the close).
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::new(1.0, 2.0, 5.0, 8.0));
        let path = b.build();
        let r = path
            .is_rect()
            .expect("add_rect output must be recognized as a rect");
        assert!((r.left - 1.0).abs() < 1e-4 && (r.top - 2.0).abs() < 1e-4);
        assert!((r.right - 5.0).abs() < 1e-4 && (r.bottom - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_is_rect_rejects_hv_staircase() {
        // A horizontal/vertical staircase has too many direction changes.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.line_to(10.0, 10.0);
        b.line_to(20.0, 10.0);
        b.line_to(20.0, 20.0);
        b.line_to(0.0, 20.0);
        b.close();
        let path = b.build();
        assert_eq!(path.is_rect(), None, "staircase must not be a rect");
    }

    #[test]
    fn test_convexity_two_contours_is_concave() {
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        b.add_rect(&Rect::new(20.0, 20.0, 30.0, 30.0));
        let path = b.build();
        assert_eq!(path.convexity(), PathConvexity::Concave);
    }

    #[test]
    fn test_convexity_single_rect_is_convex() {
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        let path = b.build();
        assert_eq!(path.convexity(), PathConvexity::Convex);
    }

    #[test]
    fn test_convexity_concave_l_shape() {
        // An L-shape is concave.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(20.0, 0.0);
        b.line_to(20.0, 10.0);
        b.line_to(10.0, 10.0);
        b.line_to(10.0, 20.0);
        b.line_to(0.0, 20.0);
        b.close();
        let path = b.build();
        assert_eq!(path.convexity(), PathConvexity::Concave);
    }

    #[test]
    fn test_reverse_produces_valid_verb_stream() {
        // Reversing must not produce a Move at the end or doubled/spurious Close.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.line_to(10.0, 10.0);
        b.close();
        let mut path = b.build();
        path.reverse();
        let verbs = path.verbs();
        assert_eq!(verbs[0], Verb::Move, "reversed path must start with Move");
        // Move must never appear except as first verb of a contour, never trailing.
        assert_ne!(*verbs.last().unwrap(), Verb::Move, "must not end with Move");
        // Exactly one Close for the single closed contour.
        let closes = verbs.iter().filter(|v| **v == Verb::Close).count();
        assert_eq!(closes, 1, "exactly one Close preserved");
        // The reversed closed triangle still contains its centroid.
        assert!(path.contains(Point::new(6.6, 3.3)));
    }

    #[test]
    fn test_bounds_non_finite_is_empty() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(Scalar::INFINITY, 10.0);
        let path = b.build();
        assert_eq!(
            path.bounds(),
            Rect::EMPTY,
            "non-finite path has empty bounds"
        );
    }

    #[test]
    fn test_contains_signed_winding_self_overlap() {
        // Even-odd vs winding differ on a self-overlapping figure-eight-like path.
        // Two nested CCW+CW squares: winding cancels in the inner region.
        let mut b = PathBuilder::new();
        // outer CW
        b.move_to(0.0, 0.0);
        b.line_to(30.0, 0.0);
        b.line_to(30.0, 30.0);
        b.line_to(0.0, 30.0);
        b.close();
        // inner, same CW winding -> winding rule keeps interior filled
        b.move_to(10.0, 10.0);
        b.line_to(20.0, 10.0);
        b.line_to(20.0, 20.0);
        b.line_to(10.0, 20.0);
        b.close();
        let mut path = b.build();
        path.set_fill_type(FillType::Winding);
        // both wound same way -> center has winding 2 -> filled
        assert!(path.contains(Point::new(15.0, 15.0)));
        path.set_fill_type(FillType::EvenOdd);
        // even-odd -> center crosses two boundaries -> empty (hole)
        assert!(!path.contains(Point::new(15.0, 15.0)));
    }

    #[test]
    fn test_contains_inverse_fill_outside_bounds() {
        let mut b = PathBuilder::new();
        b.add_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        let mut path = b.build();
        path.set_fill_type(FillType::InverseWinding);
        // Point outside bounds is contained for inverse fill.
        assert!(path.contains(Point::new(100.0, 100.0)));
        // Point inside the rect is NOT contained for inverse fill.
        assert!(!path.contains(Point::new(5.0, 5.0)));
    }

    #[test]
    fn test_contains_implicit_close_unclosed_triangle() {
        // Triangle without explicit Close must still hit-test as closed.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.line_to(5.0, 10.0);
        // no close()
        let path = b.build();
        assert!(
            path.contains(Point::new(5.0, 3.0)),
            "implicit closing edge fills the triangle"
        );
    }

    #[test]
    fn test_contains_conic_honors_weight() {
        // A quarter-circle conic + closing lines forms a filled quarter-disk.
        // A point inside the arc but outside the control triangle should still be contained.
        let mut builder = PathBuilder::new();
        builder.move_to(1.0, 0.0);
        builder.conic_to(1.0, 1.0, 0.0, 1.0, std::f32::consts::FRAC_1_SQRT_2);
        builder.line_to(0.0, 0.0);
        builder.close();
        let path = builder.build();

        // Point at (0.7, 0.3) is inside the quarter-circle (r = sqrt(0.49 + 0.09) ≈ 0.76 < 1)
        assert!(
            path.contains(Point::new(0.7, 0.3)),
            "point inside quarter-disk should be contained"
        );

        // Point at (0.9, 0.9) is outside the arc (r = sqrt(0.81 + 0.81) ≈ 1.27 > 1)
        assert!(
            !path.contains(Point::new(0.9, 0.9)),
            "point outside quarter-disk should not be contained"
        );
    }
}
