//! Path effects (dash, corner, discrete, etc.).
//!
//! Path effects modify how a path is stroked or filled. They can be applied
//! to create dashed lines, rounded corners, jittery edges, and more.

use crate::{
    FillType, Path, PathBuilder, PathElement,
    flatten::{flatten_conic_adaptive, flatten_cubic_adaptive, flatten_quad_adaptive},
};
use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, scalar_from_i32};
use skia_rs_core::{Point, Scalar};
use std::sync::Arc;

/// The stroke parameters a path effect can read and rewrite while filtering.
///
/// Corresponds to Skia's `SkStrokeRec` (style and width only; cap, join and
/// miter stay on the paint). Effects inspect it (e.g. dashing refuses filled
/// geometry) and change it (e.g. `Sk1DPathEffect` switches to fill), and the
/// caller draws the result with whatever the record says afterwards, even
/// when the effect itself failed (`skpathutils::FillPathWithPaint`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrokeRec {
    /// Fill the geometry.
    Fill,
    /// Stroke with a zero-width (1 device pixel) hairline.
    Hairline,
    /// Stroke with the given width (> 0).
    Stroke(Scalar),
    /// Fill and stroke with the given width (> 0).
    StrokeAndFill(Scalar),
}

impl StrokeRec {
    /// True for [`Fill`](Self::Fill) (`SkStrokeRec::isFillStyle`).
    #[must_use]
    pub const fn is_fill_style(&self) -> bool {
        matches!(self, Self::Fill)
    }

    /// Switch to filling (`SkStrokeRec::setFillStyle`).
    pub const fn set_fill_style(&mut self) {
        *self = Self::Fill;
    }

    /// Switch to stroking at `width`; zero means hairline
    /// (`SkStrokeRec::setStrokeStyle(width, false)`).
    pub const fn set_stroke_style(&mut self, width: Scalar) {
        *self = if width == 0.0 {
            Self::Hairline
        } else {
            Self::Stroke(width)
        };
    }
}

/// A path effect that modifies how a path is stroked or filled.
///
/// Corresponds to Skia's `SkPathEffect`.
pub trait PathEffect: Send + Sync + std::fmt::Debug {
    /// Apply the effect to a path.
    ///
    /// Mirrors `SkPathEffect::filterPath`'s two outcomes:
    /// - `Some(path)`: the effect applied and `path` is what gets drawn. An
    ///   *empty* path means "draw nothing" (e.g. a 2D tiling whose source
    ///   has empty bounds, or a 1D stamp along a zero-length path).
    /// - `None`: the effect could not apply (`filterPath` returned false),
    ///   so callers draw the source path unchanged.
    fn apply(&self, path: &Path) -> Option<Path>;

    /// Apply the effect the way Skia's `SkPathEffect::filterPath` does:
    /// reading and updating the stroke record `rec`, which the caller then
    /// uses to draw the result (or the source, if this returns `None`).
    ///
    /// The default applies [`apply`](Self::apply) and leaves `rec` alone.
    fn filter_path(&self, path: &Path, rec: &mut StrokeRec) -> Option<Path> {
        let _ = rec;
        self.apply(path)
    }

    /// [`filter_path`](Self::filter_path) given the area that can be seen,
    /// in the path's coordinate space (Skia's `cullRect` argument to
    /// `filterPath`), and the device resolution scale (`res_scale`, the
    /// CTM's maximum scale, Skia's `SkStrokeRec::getResScale`): effects that
    /// flatten curves do so finely enough for that scale. Effects may skip
    /// output entirely outside `cull`; the default ignores both.
    ///
    /// Callers must outset `cull` for whatever the result will be drawn
    /// with: the stroke's half-width times the miter limit for miter joins,
    /// plus any anti-aliasing fringe (the canvas does this). Effects only
    /// add margins for geometry they create themselves, and assume Skia's
    /// default miter limit (4) where they need one.
    fn filter_path_culled(
        &self,
        path: &Path,
        rec: &mut StrokeRec,
        cull: Option<&skia_rs_core::Rect>,
        res_scale: Scalar,
    ) -> Option<Path> {
        let _ = (cull, res_scale);
        self.filter_path(path, rec)
    }

    /// Whether this effect's culling only drops output Skia's own culling
    /// would drop, so its result under a cull is still exactly what an
    /// outer effect (e.g. in a [`ComposeEffect`]) should measure. Dashing
    /// culls like `SkDashPath` (phase-preserving, as Skia passes the cull
    /// to an inner dash); the 2D effects' culling is an extra optimization
    /// Skia does not do. The default (`false`) is safe for any effect.
    fn cull_is_exact(&self) -> bool {
        false
    }

    /// Get the kind of path effect for debugging.
    fn effect_kind(&self) -> PathEffectKind;
}

/// Kind of path effect (for debugging/inspection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathEffectKind {
    /// Dash effect.
    Dash,
    /// Corner rounding effect.
    Corner,
    /// Discrete/jitter effect.
    Discrete,
    /// 1D path effect (stamps a path along another).
    Path1D,
    /// 2D path effect (tiles a path in 2D).
    Path2D,
    /// Line 2D effect.
    Line2D,
    /// Trim effect.
    Trim,
    /// Composed effect (one after another).
    Compose,
    /// Sum effect (both applied, results combined).
    Sum,
}

/// A boxed path effect.
pub type PathEffectRef = Arc<dyn PathEffect>;

// =============================================================================
// Dash Effect
// =============================================================================

/// Convert a path with curves to a path containing only lines,
/// using adaptive tolerance for curve flattening. Used by `DashEffect`
/// so dash intervals can transition mid-curve.
fn flatten_path_to_lines(path: &Path, tolerance: Scalar) -> Path {
    let mut builder = PathBuilder::new();
    let mut current = Point::new(0.0, 0.0);
    let mut contour_start = Point::new(0.0, 0.0);
    let mut points: Vec<Point> = Vec::with_capacity(16);

    for elem in path {
        match elem {
            PathElement::Move(p) => {
                builder.move_to(p.x, p.y);
                current = p;
                contour_start = p;
            }
            PathElement::Line(p) => {
                builder.line_to(p.x, p.y);
                current = p;
            }
            PathElement::Quad(c, e) => {
                points.clear();
                flatten_quad_adaptive(&mut points, current, c, e, tolerance);
                for p in &points {
                    builder.line_to(p.x, p.y);
                }
                current = e;
            }
            PathElement::Cubic(c1, c2, e) => {
                points.clear();
                flatten_cubic_adaptive(&mut points, current, c1, c2, e, tolerance);
                for p in &points {
                    builder.line_to(p.x, p.y);
                }
                current = e;
            }
            PathElement::Conic(c, e, w) => {
                points.clear();
                flatten_conic_adaptive(&mut points, current, c, e, w, tolerance);
                for p in &points {
                    builder.line_to(p.x, p.y);
                }
                current = e;
            }
            PathElement::Close => {
                builder.close();
                current = contour_start;
            }
        }
    }

    builder.build()
}

/// Dash effect that creates dashed/dotted lines.
///
/// Corresponds to Skia's `SkDashPathEffect`.
#[derive(Debug, Clone)]
pub struct DashEffect {
    /// Dash intervals (on, off, on, off, ...).
    intervals: Vec<Scalar>,
    /// Phase offset into the dash pattern.
    phase: Scalar,
    /// Sum of all intervals (cached).
    interval_sum: Scalar,
}

impl DashEffect {
    /// Create a new dash effect.
    ///
    /// `intervals` must have an even number (at least two) of entries:
    /// on/off pairs. Like `SkDashPathEffect::Make` (`ValidDashPath`), an
    /// odd count gives `None`; callers with SVG semantics, where an odd
    /// `stroke-dasharray` repeats, duplicate it first.
    #[must_use]
    pub fn new(intervals: Vec<Scalar>, phase: Scalar) -> Option<Self> {
        if intervals.len() < 2 || !intervals.len().is_multiple_of(2) {
            return None;
        }

        // Validate intervals are positive
        for &interval in &intervals {
            if interval < 0.0 {
                return None;
            }
        }

        // SkDashPath::ValidDashPath: a positive total and a finite phase and
        // total (a NaN/infinite phase would never find its start interval).
        let interval_sum: Scalar = intervals.iter().sum();
        if !(interval_sum > 0.0 && interval_sum.is_finite() && phase.is_finite()) {
            return None;
        }

        Some(Self {
            intervals,
            phase,
            interval_sum,
        })
    }

    /// Create a simple dash pattern (dash length, gap length).
    #[must_use]
    pub fn simple(dash: Scalar, gap: Scalar) -> Option<Self> {
        Self::new(vec![dash, gap], 0.0)
    }

    /// Create a dotted pattern.
    #[must_use]
    pub fn dotted(dot_size: Scalar, gap: Scalar) -> Option<Self> {
        Self::new(vec![dot_size, gap], 0.0)
    }

    /// Get the intervals.
    #[must_use]
    pub fn intervals(&self) -> &[Scalar] {
        &self.intervals
    }

    /// Get the phase.
    #[must_use]
    pub const fn phase(&self) -> Scalar {
        self.phase
    }
}

impl DashEffect {
    /// Dash a single line segment `from -> to`, advancing the dash state
    /// (`is_on`, `interval_idx`, `remaining_in_interval`) across it. Used for
    /// both explicit line segments and the synthesized closing edge so the dash
    /// distance advances continuously (upstream `SkDashPath::InternalFilter`).
    /// `SkDashPath::InternalFilter` gives up (the effect fails) once the dash
    /// count, accumulated over the path's contours, exceeds `kMaxDashCount`
    /// (which also covers non-finite lengths).
    ///
    /// `flat` is the already-flattened source (lines only), so this is one
    /// cheap pass over its segments rather than a second measurement.
    fn dash_count_is_bounded(&self, flat: &Path) -> bool {
        const MAX_DASH_COUNT: Scalar = 1_000_000.0;
        let on_intervals =
            scalar_from_i32(i32::try_from(self.intervals.len() / 2).unwrap_or(i32::MAX));
        let (mut start, mut current) = (Point::zero(), Point::zero());
        let mut length: Scalar = 0.0;
        for element in flat {
            match element {
                PathElement::Move(p) => {
                    (start, current) = (p, p);
                }
                PathElement::Line(p)
                | PathElement::Quad(_, p)
                | PathElement::Conic(_, p, _)
                | PathElement::Cubic(_, _, p) => {
                    length += current.distance(&p);
                    current = p;
                }
                PathElement::Close => {
                    length += current.distance(&start);
                    current = start;
                }
            }
        }
        let dash_count = length * on_intervals / self.interval_sum;
        dash_count.is_finite() && dash_count <= MAX_DASH_COUNT
    }

    /// `SkDashPath::InternalFilter`'s initial join for culled rects: when
    /// `source` is a closed rect, the dash starts "on", and it ends inside an
    /// "on" interval (or at the start of an "off" one), append a tiny right
    /// angle at the first corner to `culled`, so the first and last dashes
    /// are stroked with a join instead of two butt ends.
    fn add_rect_start_join(&self, source: &Path, culled: Path) -> Path {
        const TINY_OFFSET: Scalar = 1.0 / 4096.0; // SK_ScalarNearlyZero
        if source.is_rect().is_none() || !source.is_closed() {
            return culled;
        }
        let (start_phase, index, _) = self.dash_parameters();
        if !index.is_multiple_of(2) {
            return culled;
        }
        let points = source.points();
        let n = points.len();
        let perimeter: Scalar = (0..n)
            .map(|i| points[i].distance(&points[(i + 1) % n]))
            .sum();
        let mut end_phase = (perimeter + start_phase) % self.interval_sum;
        let mut end_index = 0;
        loop {
            if end_phase <= self.intervals[end_index] {
                break;
            }
            end_phase -= self.intervals[end_index];
            end_index += 1;
            if end_index == self.intervals.len() {
                end_phase = 0.0;
                break;
            }
        }
        if end_index.is_multiple_of(2) != (end_phase > 0.0) {
            return culled;
        }
        let mid = points[0];
        let Some(&last) = points.iter().rev().find(|p| **p != mid) else {
            return culled;
        };
        let Some(&next) = points.iter().find(|p| **p != mid) else {
            return culled;
        };
        let mut b = PathBuilder::new();
        b.add_path(&culled);
        let (vx, vy) = (
            (mid.x - last.x) * TINY_OFFSET,
            (mid.y - last.y) * TINY_OFFSET,
        );
        b.move_to(mid.x - vx, mid.y - vy).line_to(mid.x, mid.y);
        let (wx, wy) = (
            (mid.x - next.x) * TINY_OFFSET,
            (mid.y - next.y) * TINY_OFFSET,
        );
        b.line_to(mid.x - wx, mid.y - wy);
        b.build()
    }

    /// `SkDashPath::CalcDashParameters`: the phase normalized into
    /// `[0, interval_sum)` (a negative phase is flipped), the index of the
    /// interval it falls in, and how much of that interval remains. A phase
    /// exactly at an interval's end moves on only if the interval is
    /// non-empty, and rounding that runs past the last interval falls back
    /// to the start of the first, as in Skia.
    #[allow(clippy::float_cmp, reason = "SkDashPath's exact phase == gap test")]
    fn dash_parameters(&self) -> (Scalar, usize, Scalar) {
        let len = self.interval_sum;
        let mut phase = self.phase;
        if phase < 0.0 {
            phase = -phase;
            if phase > len {
                phase %= len;
            }
            phase = len - phase;
            if phase == len {
                phase = 0.0;
            }
        } else if phase >= len {
            phase %= len;
        }
        let adjusted = phase;
        for (index, &gap) in self.intervals.iter().enumerate() {
            if phase > gap || (phase == gap && gap != 0.0) {
                phase -= gap;
            } else {
                return (adjusted, index, gap - phase);
            }
        }
        (adjusted, 0, self.intervals[0])
    }

    fn dash_line(
        &self,
        dashes: &mut Vec<Vec<Point>>,
        from: Point,
        to: Point,
        is_on: &mut bool,
        interval_idx: &mut usize,
        remaining_in_interval: &mut Scalar,
    ) {
        let segment_length = from.distance(&to);
        // A non-finite length would never finish the walk (apply rejects
        // such paths first; this keeps the loop safe on its own).
        if !(segment_length.is_finite() && segment_length > 0.0) {
            return;
        }
        // Walk by distance along the segment, as SkDashPath does. (Stepping
        // a parametric t by interval / length loses tiny intervals once t is
        // near 1.) apply bounds the dash count, so each full interval cycle
        // still moves `pos` forward in f32.
        let point_at = |d: Scalar| {
            let t = d / segment_length;
            Point::new(
                (to.x - from.x).mul_add(t, from.x),
                (to.y - from.y).mul_add(t, from.y),
            )
        };
        let mut pos: Scalar = 0.0;
        loop {
            let remaining_segment = segment_length - pos;
            if *remaining_in_interval >= remaining_segment {
                if *is_on {
                    push_dash_point(dashes, to);
                }
                *remaining_in_interval -= remaining_segment;
                return;
            }
            pos += *remaining_in_interval;
            let mid = point_at(pos);
            if *is_on {
                push_dash_point(dashes, mid);
            }
            *interval_idx = (*interval_idx + 1) % self.intervals.len();
            *is_on = (*interval_idx).is_multiple_of(2);
            *remaining_in_interval = self.intervals[*interval_idx];
            if *is_on {
                dashes.push(vec![mid]);
            }
        }
    }
}

impl PathEffect for DashEffect {
    fn filter_path(&self, path: &Path, rec: &mut StrokeRec) -> Option<Path> {
        self.filter_path_culled(path, rec, None, 1.0)
    }

    /// As [`filter_path`](PathEffect::filter_path), flattening curves for
    /// `res_scale` (`SkDashPath` measures with the stroke record's `resScale`),
    /// so dashes on zoomed-in curves stay smooth.
    fn filter_path_culled(
        &self,
        path: &Path,
        rec: &mut StrokeRec,
        cull: Option<&skia_rs_core::Rect>,
        res_scale: Scalar,
    ) -> Option<Path> {
        // SkDashPath::InternalFilter does nothing for geometry that wants to
        // be filled (Fill or StrokeAndFill): the source is drawn unchanged.
        if matches!(rec, StrokeRec::Fill | StrokeRec::StrokeAndFill(_)) {
            return None;
        }
        // SkDashPath's cull_path: a line or rect source is first trimmed to
        // the cull (keeping the dash phase), so only the visible part counts
        // against kMaxDashCount.
        let culled = cull.and_then(|c| cull_dash_source(path, c, *rec, self.interval_sum));
        if culled.as_ref().is_some_and(Path::is_empty) {
            // Entirely outside the cull: nothing visible to dash.
            return Some(Path::new());
        }
        // A culled closed rect is dashed as open edges; if the pattern is
        // "on" both where it starts and where it ends, SkDashPath adds a
        // tiny right angle at the start corner so the stroke joins there.
        let culled = culled.map(|c| self.add_rect_start_join(path, c));
        let path = culled.as_ref().unwrap_or(path);
        let scale = if res_scale.is_finite() && res_scale > 0.0 {
            res_scale
        } else {
            1.0
        };
        self.dash(path, DASH_FLATTEN_TOLERANCE / scale)
    }

    fn apply(&self, path: &Path) -> Option<Path> {
        self.dash(path, DASH_FLATTEN_TOLERANCE)
    }

    fn cull_is_exact(&self) -> bool {
        true
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Dash
    }
}

/// `SkDashPath`'s `cull_path`: a line, or the edges of a rect, trimmed to
/// `cull` outset for the stroke. Trimmed ends move by whole interval lengths
/// so the dash pattern is unchanged (`clip_line`). `None` when the source is
/// neither (or a diagonal line); an empty path when it lies entirely outside
/// the cull. (Skia dashes such an off-screen source in full; skipping it
/// draws the same pixels without the work.)
#[allow(
    clippy::float_cmp,
    reason = "exact axis-alignment test, as in clip_line"
)]
fn cull_dash_source(
    path: &Path,
    cull: &skia_rs_core::Rect,
    rec: StrokeRec,
    interval_len: Scalar,
) -> Option<Path> {
    // outset_for_stroke: half the width (1 for hairlines), times the miter
    // limit for miter joins; the join is not in the record, so allow for
    // Skia's default miter limit (4).
    let radius = match rec {
        StrokeRec::Stroke(w) | StrokeRec::StrokeAndFill(w) if w > 0.0 => w / 2.0,
        _ => 1.0,
    } * 4.0;
    let bounds = cull.inset(-radius, -radius);
    let mut builder = PathBuilder::new();
    if path.is_line() {
        let pts = path.points();
        let (mut a, mut b) = (pts[0], pts[1]);
        // Beyond Skia's clip_line (which trims only along the line's own
        // axis): a horizontal or vertical line outside the cull across its
        // axis is off-screen too.
        let off_across = (a.y == b.y && (a.y < bounds.top || a.y > bounds.bottom))
            || (a.x == b.x && (a.x < bounds.left || a.x > bounds.right));
        if off_across {
            return Some(Path::new());
        }
        if !clip_dash_line(&mut a, &mut b, &bounds, interval_len, 0.0) {
            // A horizontal/vertical line that missed the cull is off-screen;
            // a diagonal one cannot be trimmed here.
            let axis_aligned = a.x == b.x || a.y == b.y;
            return axis_aligned.then(Path::new);
        }
        builder.move_to(a.x, a.y).line_to(b.x, b.y);
        return Some(builder.build());
    }
    path.is_rect()?;
    // The rect's edges, including the closing one, culled separately with
    // the unculled length so far keeping the phase.
    let mut edges = Vec::with_capacity(4);
    let (mut start, mut current) = (Point::zero(), Point::zero());
    for element in path {
        match element {
            PathElement::Move(p) => (start, current) = (p, p),
            PathElement::Line(p) => {
                edges.push((current, p));
                current = p;
            }
            PathElement::Close => {
                if current != start {
                    edges.push((current, start));
                }
                current = start;
            }
            _ => return None,
        }
    }
    let mut accum = 0.0_f64;
    let mut last: Option<Point> = None;
    for (a, b) in edges {
        let (mut ca, mut cb) = (a, b);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the prior phase is below one interval length"
        )]
        let prior = (accum % f64::from(interval_len)) as Scalar;
        if clip_dash_line(&mut ca, &mut cb, &bounds, interval_len, prior) {
            if last != Some(ca) {
                builder.move_to(ca.x, ca.y);
            }
            builder.line_to(cb.x, cb.y);
            last = Some(cb);
        }
        accum += f64::from((b.x - a.x + b.y - a.y).abs());
    }
    // An empty result means every edge missed the cull: off-screen.
    Some(builder.build())
}

/// `SkDashPath`'s `clip_line` for a horizontal or vertical line: trim it to
/// `bounds` along its axis, moving each trimmed end by a whole number of
/// `interval_len` (and `prior_phase` for rect edges) so the dash pattern is
/// unchanged. False for a diagonal line or one outside the bounds.
#[allow(clippy::float_cmp, reason = "SkDashPath tests the exact deltas")]
fn clip_dash_line(
    p0: &mut Point,
    p1: &mut Point,
    bounds: &skia_rs_core::Rect,
    interval_len: Scalar,
    prior_phase: Scalar,
) -> bool {
    let (dx, dy) = (p1.x - p0.x, p1.y - p0.y);
    if dx != 0.0 && dy != 0.0 {
        return false;
    }
    let vertical = dy != 0.0;
    let axis = |p: &Point| if vertical { p.y } else { p.x };
    let (mut min, mut max) = (axis(p0), axis(p1));
    let swapped = max < min;
    if swapped {
        std::mem::swap(&mut min, &mut max);
    }
    let (left_top, right_bottom) = if vertical {
        (bounds.top, bounds.bottom)
    } else {
        (bounds.left, bounds.right)
    };
    if max < left_top || min > right_bottom {
        return false;
    }
    if min < left_top {
        min = left_top - (left_top - min) % interval_len;
        if !swapped {
            min -= prior_phase;
        }
    }
    if max > right_bottom {
        max = right_bottom + (max - right_bottom) % interval_len;
        if swapped {
            max += prior_phase;
        }
    }
    if swapped {
        std::mem::swap(&mut min, &mut max);
    }
    if vertical {
        (p0.y, p1.y) = (min, max);
    } else {
        (p0.x, p1.x) = (min, max);
    }
    true
}

/// Extend the current dash with `p` (starting one if there is none).
fn push_dash_point(dashes: &mut Vec<Vec<Point>>, p: Point) {
    match dashes.last_mut() {
        Some(dash) => dash.push(p),
        None => dashes.push(vec![p]),
    }
}

/// Emit a contour's dashes (polylines) into `builder`. With `join_ends`
/// (a closed contour that is "on" where it starts and where it ends), the
/// first dash continues the last one, as `SkDashPath`'s `skipFirstSegment`
/// does, so the stroke joins across the start point.
fn emit_dashes(builder: &mut PathBuilder, dashes: &mut Vec<Vec<Point>>, join_ends: bool) {
    if join_ends && dashes.len() >= 2 {
        let first = dashes.remove(0);
        if let Some(last) = dashes.last_mut() {
            last.extend(first.into_iter().skip(1));
        }
    }
    for dash in dashes.drain(..) {
        if dash.len() >= 2 {
            builder.move_to(dash[0].x, dash[0].y);
            for q in &dash[1..] {
                builder.line_to(q.x, q.y);
            }
        }
    }
}

/// Curve-flattening tolerance for dashing, in local units at a resolution
/// scale of 1 (divided by the scale otherwise).
const DASH_FLATTEN_TOLERANCE: Scalar = 0.5;

impl DashEffect {
    /// Dash `path`, flattening curves to within `tolerance` local units.
    #[allow(
        clippy::too_many_lines,
        reason = "faithful port of SkDashPath::InternalFilter's per-verb dash state machine"
    )]
    fn dash(&self, path: &Path, tolerance: Scalar) -> Option<Path> {
        if path.is_empty() {
            return None;
        }

        // Check the dash-count bound on a coarse flattening first, so a path
        // that fails it is not first flattened at a fine (zoomed) tolerance.
        let coarse_tolerance = DASH_FLATTEN_TOLERANCE.max(tolerance);
        let coarse = flatten_path_to_lines(path, coarse_tolerance);
        if !self.dash_count_is_bounded(&coarse) {
            return None;
        }
        // Pre-flatten curves to lines so dash logic can transition mid-curve.
        let flattened = if tolerance < coarse_tolerance {
            flatten_path_to_lines(path, tolerance)
        } else {
            coarse
        };
        let path = &flattened;

        // Each contour's dashes are collected as polylines first, so a closed
        // contour can join its last dash to its first (SkDashPath skips the
        // first dash of a closed contour and appends it after the closing
        // edge, continuing the last dash when that is still "on").
        let mut builder = PathBuilder::new();
        let mut dashes: Vec<Vec<Point>> = Vec::new();
        let mut current_pos = Point::zero();
        let mut contour_start = Point::zero();

        // SkDashPath::CalcDashParameters: where in the pattern each contour
        // starts.
        let (_, start_index, start_length) = self.dash_parameters();
        let starts_on = start_index % 2 == 0;
        let mut interval_idx = start_index;
        let mut is_on = starts_on;
        let mut remaining_in_interval = start_length;

        for element in path {
            match element {
                PathElement::Move(p) => {
                    emit_dashes(&mut builder, &mut dashes, false);
                    current_pos = p;
                    contour_start = p;
                    // Reset dash state for the new contour.
                    interval_idx = start_index;
                    is_on = starts_on;
                    remaining_in_interval = start_length;
                    if is_on {
                        dashes.push(vec![p]);
                    }
                }
                // (The source is flattened to lines above, so curves only
                // appear here defensively, as their chords.)
                PathElement::Line(end)
                | PathElement::Quad(_, end)
                | PathElement::Conic(_, end, _)
                | PathElement::Cubic(_, _, end) => {
                    self.dash_line(
                        &mut dashes,
                        current_pos,
                        end,
                        &mut is_on,
                        &mut interval_idx,
                        &mut remaining_in_interval,
                    );
                    current_pos = end;
                }
                PathElement::Close => {
                    // Dash the closing edge continuously (advancing the dash
                    // distance across it), not solid/dropped.
                    if current_pos != contour_start {
                        self.dash_line(
                            &mut dashes,
                            current_pos,
                            contour_start,
                            &mut is_on,
                            &mut interval_idx,
                            &mut remaining_in_interval,
                        );
                    }
                    // Join the last dash to the first across the start when
                    // the pattern is "on" at both ends.
                    emit_dashes(&mut builder, &mut dashes, starts_on && is_on);
                    current_pos = contour_start;
                    interval_idx = start_index;
                    is_on = starts_on;
                    remaining_in_interval = start_length;
                    if is_on {
                        dashes.push(vec![contour_start]);
                    }
                }
            }
        }
        emit_dashes(&mut builder, &mut dashes, false);

        Some(builder.build())
    }
}

// =============================================================================
// Corner Effect
// =============================================================================

/// Corner path effect that rounds corners.
///
/// Corresponds to Skia's `SkCornerPathEffect`.
#[derive(Debug, Clone, Copy)]
pub struct CornerEffect {
    /// Radius for corner rounding.
    radius: Scalar,
}

impl CornerEffect {
    /// Create a new corner effect.
    #[must_use]
    pub fn new(radius: Scalar) -> Option<Self> {
        if radius <= 0.0 {
            return None;
        }
        Some(Self { radius })
    }

    /// Get the radius.
    #[must_use]
    pub const fn radius(&self) -> Scalar {
        self.radius
    }
}

/// Which verb kind was last processed (for the corner-effect state machine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CornerVerb {
    None,
    Move,
    Line,
    Curve,
    Close,
}

/// Compute the corner step vector for the segment `a -> b`.
///
/// Returns `true` if a straight middle segment should be drawn (i.e. the
/// segment is longer than `2 * radius`), matching `ComputeStep`.
fn corner_compute_step(a: Point, b: Point, radius: Scalar, step: &mut Point) -> bool {
    let dist = a.distance(&b);
    let mut s = Point::new(b.x - a.x, b.y - a.y);
    if dist <= radius * 2.0 {
        s.x *= 0.5;
        s.y *= 0.5;
        *step = s;
        false
    } else {
        let f = radius / dist;
        s.x *= f;
        s.y *= f;
        *step = s;
        true
    }
}

/// Whether the contour starting at `move_idx` is closed (contains a `Close`
/// before the next `Move`).
fn corner_contour_is_closed(elements: &[PathElement], move_idx: usize) -> bool {
    for e in &elements[move_idx + 1..] {
        match e {
            PathElement::Move(_) => return false,
            PathElement::Close => return true,
            _ => {}
        }
    }
    false
}

impl PathEffect for CornerEffect {
    #[allow(
        clippy::too_many_lines,
        reason = "faithful port of SkCornerPathEffectImpl::onFilterPath's per-verb corner-rounding state machine"
    )]
    fn apply(&self, path: &Path) -> Option<Path> {
        if path.is_empty() || self.radius <= 0.0 {
            return None;
        }

        // Faithful port of SkCornerPathEffectImpl::onFilterPath: rounds every
        // corner including the joint between the last and first segments of a
        // closed contour, and starts a closed contour's output at the first
        // segment's rounded start (moveTo + step).
        let radius = self.radius;
        let mut dst = PathBuilder::new();
        let elements: Vec<PathElement> = path.iter().collect();

        let mut prev_verb = CornerVerb::None;
        let mut move_to = Point::zero();
        let mut last_corner = Point::zero();
        let mut first_step = Point::zero();
        let mut step = Point::zero();
        let mut prev_is_valid = true;
        let mut current = Point::zero();

        for (i, elem) in elements.iter().enumerate() {
            let cur_verb;
            match *elem {
                PathElement::Move(p) => {
                    // Close out the previous (open) contour.
                    if prev_verb == CornerVerb::Line {
                        dst.line_to(last_corner.x, last_corner.y);
                    }
                    if corner_contour_is_closed(&elements, i) {
                        move_to = p;
                        prev_is_valid = false;
                    } else {
                        dst.move_to(p.x, p.y);
                        prev_is_valid = true;
                    }
                    current = p;
                    cur_verb = CornerVerb::Move;
                }
                PathElement::Line(end) => {
                    let a = current;
                    let b = end;
                    let draw_segment = corner_compute_step(a, b, radius, &mut step);
                    if prev_is_valid {
                        dst.quad_to(a.x, a.y, a.x + step.x, a.y + step.y);
                    } else {
                        dst.move_to(move_to.x + step.x, move_to.y + step.y);
                        prev_is_valid = true;
                    }
                    if draw_segment {
                        dst.line_to(b.x - step.x, b.y - step.y);
                    }
                    last_corner = b;
                    current = end;
                    cur_verb = CornerVerb::Line;
                }
                PathElement::Quad(ctrl, end) => {
                    if !prev_is_valid {
                        dst.move_to(current.x, current.y);
                        prev_is_valid = true;
                    }
                    dst.quad_to(ctrl.x, ctrl.y, end.x, end.y);
                    last_corner = end;
                    first_step = Point::zero();
                    current = end;
                    cur_verb = CornerVerb::Curve;
                }
                PathElement::Conic(ctrl, end, w) => {
                    if !prev_is_valid {
                        dst.move_to(current.x, current.y);
                        prev_is_valid = true;
                    }
                    dst.conic_to(ctrl.x, ctrl.y, end.x, end.y, w);
                    last_corner = end;
                    first_step = Point::zero();
                    current = end;
                    cur_verb = CornerVerb::Curve;
                }
                PathElement::Cubic(c1, c2, end) => {
                    if !prev_is_valid {
                        dst.move_to(current.x, current.y);
                        prev_is_valid = true;
                    }
                    dst.cubic_to(c1.x, c1.y, c2.x, c2.y, end.x, end.y);
                    last_corner = end;
                    first_step = Point::zero();
                    current = end;
                    cur_verb = CornerVerb::Curve;
                }
                PathElement::Close => {
                    if first_step.x != 0.0 || first_step.y != 0.0 {
                        dst.quad_to(
                            last_corner.x,
                            last_corner.y,
                            last_corner.x + first_step.x,
                            last_corner.y + first_step.y,
                        );
                    }
                    dst.close();
                    prev_is_valid = false;
                    current = move_to;
                    cur_verb = CornerVerb::Close;
                }
            }

            // Capture the first segment's step so the closing joint can be
            // rounded back toward the contour start.
            if prev_verb == CornerVerb::Move {
                first_step = step;
            }
            prev_verb = cur_verb;
        }

        if prev_is_valid {
            dst.line_to(last_corner.x, last_corner.y);
        }

        Some(dst.build())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Corner
    }
}

// =============================================================================
// Discrete Effect
// =============================================================================

/// Discrete path effect that adds random displacement (jitter).
///
/// Corresponds to Skia's `SkDiscretePathEffect`.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteEffect {
    /// Segment length for subdividing the path.
    seg_length: Scalar,
    /// Maximum deviation from the original path.
    deviation: Scalar,
    /// Random seed.
    seed: u32,
}

impl DiscreteEffect {
    /// Create a new discrete effect.
    #[must_use]
    ///
    /// Like `SkDiscretePathEffect::Make`, non-finite parameters or a
    /// segment length at or below `SK_ScalarNearlyZero` give `None`.
    pub fn new(seg_length: Scalar, deviation: Scalar, seed: u32) -> Option<Self> {
        const SCALAR_NEARLY_ZERO: Scalar = 1.0 / 4096.0;
        if !(seg_length.is_finite() && deviation.is_finite()) || seg_length <= SCALAR_NEARLY_ZERO {
            return None;
        }
        Some(Self {
            seg_length,
            deviation,
            seed,
        })
    }

    /// Get the segment length.
    #[must_use]
    pub const fn seg_length(&self) -> Scalar {
        self.seg_length
    }

    /// Get the deviation.
    #[must_use]
    pub const fn deviation(&self) -> Scalar {
        self.deviation
    }

    /// Simple pseudo-random number generator.
    fn random(seed: u32) -> Scalar {
        // Simple LCG
        let n = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let masked =
            u16::try_from((n >> 16) & 0x7FFF).expect("masked to 15 bits, always fits in u16");
        (Scalar::from(masked) / 32767.0).mul_add(2.0, -1.0)
    }
}

/// `SkDiscretePathEffect`'s `kMaxReasonableIterations`: the most segments
/// one contour is split into.
const MAX_DISCRETE_SEGMENTS: i32 = 100_000;

/// The point at `t` on the conic `p0, p1, p2` with weight `w` (the exact
/// rational form, so quarter-circle conics stay circular).
#[allow(
    clippy::suboptimal_flops,
    reason = "unfused rational form, like the rasterizer's conic evaluation"
)]
fn conic_eval(p0: Point, p1: Point, p2: Point, weight: Scalar, t: Scalar) -> Point {
    let mt = 1.0 - t;
    let (k0, k1, k2) = (mt * mt, 2.0 * weight * mt * t, t * t);
    let denom = k0 + k1 + k2;
    Point::new(
        (k0 * p0.x + k1 * p1.x + k2 * p2.x) / denom,
        (k0 * p0.y + k1 * p1.y + k2 * p2.y) / denom,
    )
}

/// Approximate arc length of a conic (16-segment polyline).
fn conic_length(p0: Point, p1: Point, p2: Point, w: Scalar) -> Scalar {
    let mut length = 0.0;
    let mut prev = p0;
    for i in 1..=16_u8 {
        let next = conic_eval(p0, p1, p2, w, Scalar::from(i) / 16.0);
        length += prev.distance(&next);
        prev = next;
    }
    length
}

impl PathEffect for DiscreteEffect {
    fn apply(&self, path: &Path) -> Option<Path> {
        if path.is_empty() {
            return None;
        }

        let mut builder = PathBuilder::new();
        let mut current_pos = Point::zero();
        let mut seed = self.seed;
        // SkDiscretePathEffect caps each contour at kMaxReasonableIterations
        // segments, so a long contour or a tiny segment length cannot
        // allocate without bound.
        let mut contour_budget = MAX_DISCRETE_SEGMENTS;
        let take_segments = |wanted: i32, budget: &mut i32| {
            let n = wanted.clamp(1, (*budget).max(1));
            *budget = budget.saturating_sub(n);
            n
        };

        for element in path {
            match element {
                PathElement::Move(p) => {
                    let dx = Self::random(seed) * self.deviation;
                    seed = seed.wrapping_add(1);
                    let dy = Self::random(seed) * self.deviation;
                    seed = seed.wrapping_add(1);
                    builder.move_to(p.x + dx, p.y + dy);
                    current_pos = p;
                    contour_budget = MAX_DISCRETE_SEGMENTS;
                }
                PathElement::Line(end) => {
                    let length = current_pos.distance(&end);
                    let num_segments =
                        take_segments(ceil_to_i32(length / self.seg_length), &mut contour_budget);

                    for i in 1..=num_segments {
                        let t = scalar_from_i32(i) / scalar_from_i32(num_segments);
                        let x = (end.x - current_pos.x).mul_add(t, current_pos.x);
                        let y = (end.y - current_pos.y).mul_add(t, current_pos.y);

                        let dx = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);
                        let dy = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);

                        builder.line_to(x + dx, y + dy);
                    }
                    current_pos = end;
                }
                PathElement::Close => {
                    builder.close();
                }
                // For curves, subdivide into lines first
                PathElement::Quad(ctrl, end) => {
                    let steps = take_segments(
                        ceil_to_i32(quadratic_length(current_pos, ctrl, end) / self.seg_length)
                            .max(4),
                        &mut contour_budget,
                    );

                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let p = quadratic_point(current_pos, ctrl, end, t);

                        let dx = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);
                        let dy = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);

                        builder.line_to(p.x + dx, p.y + dy);
                    }
                    current_pos = end;
                }
                PathElement::Conic(ctrl, end, w) => {
                    let steps = take_segments(
                        ceil_to_i32(conic_length(current_pos, ctrl, end, w) / self.seg_length)
                            .max(4),
                        &mut contour_budget,
                    );

                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let p = conic_eval(current_pos, ctrl, end, w, t);

                        let dx = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);
                        let dy = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);

                        builder.line_to(p.x + dx, p.y + dy);
                    }
                    current_pos = end;
                }
                PathElement::Cubic(ctrl1, ctrl2, end) => {
                    let steps = take_segments(
                        ceil_to_i32(cubic_length(current_pos, ctrl1, ctrl2, end) / self.seg_length)
                            .max(4),
                        &mut contour_budget,
                    );

                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let p = cubic_point(current_pos, ctrl1, ctrl2, end, t);

                        let dx = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);
                        let dy = Self::random(seed) * self.deviation;
                        seed = seed.wrapping_add(1);

                        builder.line_to(p.x + dx, p.y + dy);
                    }
                    current_pos = end;
                }
            }
        }

        Some(builder.build())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Discrete
    }
}

// =============================================================================
// Trim Effect
// =============================================================================

/// Trim effect that shows only a portion of the path.
///
/// Corresponds to Skia's `SkTrimPathEffect`.
#[derive(Debug, Clone, Copy)]
pub struct TrimEffect {
    /// Start of the visible portion (0.0 - 1.0).
    start: Scalar,
    /// End of the visible portion (0.0 - 1.0).
    end: Scalar,
    /// Trim mode.
    mode: TrimMode,
}

/// Trim mode for path effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TrimMode {
    /// Normal trim (from start to end).
    #[default]
    Normal,
    /// Inverted trim (everything except start to end).
    Inverted,
}

impl TrimEffect {
    /// Create a new trim effect.
    #[must_use]
    pub fn new(start: Scalar, end: Scalar, mode: TrimMode) -> Option<Self> {
        if !(0.0..=1.0).contains(&start) || !(0.0..=1.0).contains(&end) {
            return None;
        }
        Some(Self { start, end, mode })
    }

    /// Get the start position.
    #[must_use]
    pub const fn start(&self) -> Scalar {
        self.start
    }

    /// Get the end position.
    #[must_use]
    pub const fn end(&self) -> Scalar {
        self.end
    }

    /// Get the trim mode.
    #[must_use]
    pub const fn mode(&self) -> TrimMode {
        self.mode
    }
}

impl PathEffect for TrimEffect {
    fn apply(&self, path: &Path) -> Option<Path> {
        let measure = crate::measure::PathMeasure::new(path);
        let total = measure.length();
        if total <= 0.0 {
            // SkTrimPE adds no segments for a zero-length source and still
            // succeeds, so nothing is drawn (not even a capped dot).
            return Some(Path::new());
        }

        let start_dist = self.start * total;
        let end_dist = self.end * total;

        match self.mode {
            TrimMode::Normal => {
                if start_dist >= end_dist {
                    return Some(Path::new());
                }
                // A degenerate segment draws nothing; `None` would make the
                // caller draw the untrimmed source instead.
                Some(
                    measure
                        .get_segment(start_dist, end_dist)
                        .unwrap_or_default(),
                )
            }
            TrimMode::Inverted => {
                if start_dist >= end_dist {
                    return Some(path.clone());
                }
                let mut builder = PathBuilder::new();
                if start_dist > 0.0
                    && let Some(before) = measure.get_segment(0.0, start_dist)
                {
                    for elem in &before {
                        match elem {
                            PathElement::Move(p) => {
                                builder.move_to(p.x, p.y);
                            }
                            PathElement::Line(p) => {
                                builder.line_to(p.x, p.y);
                            }
                            PathElement::Quad(p1, p2) => {
                                builder.quad_to(p1.x, p1.y, p2.x, p2.y);
                            }
                            PathElement::Cubic(p1, p2, p3) => {
                                builder.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                            }
                            PathElement::Conic(p1, p2, w) => {
                                builder.conic_to(p1.x, p1.y, p2.x, p2.y, w);
                            }
                            PathElement::Close => {
                                builder.close();
                            }
                        }
                    }
                }
                if end_dist < total
                    && let Some(after) = measure.get_segment(end_dist, total)
                {
                    for elem in &after {
                        match elem {
                            PathElement::Move(p) => {
                                builder.move_to(p.x, p.y);
                            }
                            PathElement::Line(p) => {
                                builder.line_to(p.x, p.y);
                            }
                            PathElement::Quad(p1, p2) => {
                                builder.quad_to(p1.x, p1.y, p2.x, p2.y);
                            }
                            PathElement::Cubic(p1, p2, p3) => {
                                builder.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                            }
                            PathElement::Conic(p1, p2, w) => {
                                builder.conic_to(p1.x, p1.y, p2.x, p2.y, w);
                            }
                            PathElement::Close => {
                                builder.close();
                            }
                        }
                    }
                }
                Some(builder.build())
            }
        }
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Trim
    }
}

// =============================================================================
// Compose Effect
// =============================================================================

/// Compose effect that applies one effect, then another.
///
/// Corresponds to Skia's `SkPathEffect::MakeCompose`.
#[derive(Debug)]
pub struct ComposeEffect {
    outer: PathEffectRef,
    inner: PathEffectRef,
}

impl ComposeEffect {
    /// Create a composed effect: outer(inner(path)).
    pub fn new(outer: PathEffectRef, inner: PathEffectRef) -> Self {
        Self { outer, inner }
    }
}

impl PathEffect for ComposeEffect {
    fn filter_path(&self, path: &Path, rec: &mut StrokeRec) -> Option<Path> {
        // SkComposePathEffect: inner then outer on one SkStrokeRec; if the
        // inner effect fails, the outer one still runs on the source.
        let intermediate = self.inner.filter_path(path, rec);
        self.outer
            .filter_path(intermediate.as_ref().unwrap_or(path), rec)
    }

    fn filter_path_culled(
        &self,
        path: &Path,
        rec: &mut StrokeRec,
        cull: Option<&skia_rs_core::Rect>,
        res_scale: Scalar,
    ) -> Option<Path> {
        // The outer effect measures the inner one's output (dash phase,
        // trim lengths, stamp spacing). Skia passes the cull to both, which
        // is exact for an inner dash (SkDashPath trims phase-preservingly,
        // and must, to stay under kMaxDashCount); an inner effect whose
        // culling Skia would not do gets no cull, so the outer one sees its
        // full geometry. (The resolution scale always passes through: it
        // only refines flattening.)
        let inner_cull = cull.filter(|_| self.inner.cull_is_exact());
        let intermediate = self
            .inner
            .filter_path_culled(path, rec, inner_cull, res_scale);
        self.outer
            .filter_path_culled(intermediate.as_ref().unwrap_or(path), rec, cull, res_scale)
    }

    fn apply(&self, path: &Path) -> Option<Path> {
        let intermediate = self.inner.apply(path);
        self.outer.apply(intermediate.as_ref().unwrap_or(path))
    }

    fn cull_is_exact(&self) -> bool {
        self.inner.cull_is_exact() && self.outer.cull_is_exact()
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Compose
    }
}

// =============================================================================
// Sum Effect
// =============================================================================

/// Sum effect that applies both effects and combines the results.
///
/// Corresponds to Skia's `SkPathEffect::MakeSum`.
#[derive(Debug)]
pub struct SumEffect {
    first: PathEffectRef,
    second: PathEffectRef,
}

impl SumEffect {
    /// Create a sum effect.
    pub fn new(first: PathEffectRef, second: PathEffectRef) -> Self {
        Self { first, second }
    }
}

impl PathEffect for SumEffect {
    fn filter_path(&self, path: &Path, rec: &mut StrokeRec) -> Option<Path> {
        // SkSumPathEffect always runs both, in order, on one SkStrokeRec and
        // appends whatever each produced.
        let path1 = self.first.filter_path(path, rec);
        let path2 = self.second.filter_path(path, rec);
        Self::combine(path1, path2)
    }

    fn filter_path_culled(
        &self,
        path: &Path,
        rec: &mut StrokeRec,
        cull: Option<&skia_rs_core::Rect>,
        res_scale: Scalar,
    ) -> Option<Path> {
        // Both outputs are drawn with the record as `second` leaves it. If
        // `second` changes it (e.g. Line2D sets a stroke), `first` culled
        // for the wrong stroke, so run it again without culling.
        let rec_before = *rec;
        let path1 = self.first.filter_path_culled(path, rec, cull, res_scale);
        let rec_after_first = *rec;
        let path2 = self.second.filter_path_culled(path, rec, cull, res_scale);
        let path1 = if cull.is_some() && *rec != rec_after_first {
            let mut fresh = rec_before;
            self.first
                .filter_path_culled(path, &mut fresh, None, res_scale)
        } else {
            path1
        };
        Self::combine(path1, path2)
    }

    fn apply(&self, path: &Path) -> Option<Path> {
        Self::combine(self.first.apply(path), self.second.apply(path))
    }

    fn cull_is_exact(&self) -> bool {
        self.first.cull_is_exact() && self.second.cull_is_exact()
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Sum
    }
}

impl SumEffect {
    fn combine(path1: Option<Path>, path2: Option<Path>) -> Option<Path> {
        match (path1, path2) {
            (Some(p1), Some(p2)) => {
                // Combine the paths
                let mut builder = PathBuilder::new();
                builder.add_path(&p1);
                builder.add_path(&p2);
                Some(builder.build())
            }
            (Some(p), None) | (None, Some(p)) => Some(p),
            (None, None) => None,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn quadratic_point(p0: Point, p1: Point, p2: Point, t: Scalar) -> Point {
    let mt = 1.0 - t;
    Point::new(
        (t * t).mul_add(p2.x, (mt * mt).mul_add(p0.x, 2.0 * mt * t * p1.x)),
        (t * t).mul_add(p2.y, (mt * mt).mul_add(p0.y, 2.0 * mt * t * p1.y)),
    )
}

fn cubic_point(p0: Point, p1: Point, p2: Point, p3: Point, t: Scalar) -> Point {
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let t2 = t * t;
    Point::new(
        (t2 * t).mul_add(
            p3.x,
            (3.0 * mt * t2).mul_add(p2.x, (mt2 * mt).mul_add(p0.x, 3.0 * mt2 * t * p1.x)),
        ),
        (t2 * t).mul_add(
            p3.y,
            (3.0 * mt * t2).mul_add(p2.y, (mt2 * mt).mul_add(p0.y, 3.0 * mt2 * t * p1.y)),
        ),
    )
}

fn quadratic_length(p0: Point, p1: Point, p2: Point) -> Scalar {
    // Approximate with chord + control polygon
    let chord = p0.distance(&p2);
    let polygon = p0.distance(&p1) + p1.distance(&p2);
    f32::midpoint(chord, polygon)
}

fn cubic_length(p0: Point, p1: Point, p2: Point, p3: Point) -> Scalar {
    // Approximate with chord + control polygon
    let chord = p0.distance(&p3);
    let polygon = p0.distance(&p1) + p1.distance(&p2) + p2.distance(&p3);
    f32::midpoint(chord, polygon)
}

// =============================================================================
// Path1D Effect
// =============================================================================

/// Style for how the stamped path is placed along the input path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Path1DStyle {
    /// Translate the path shape.
    #[default]
    Translate,
    /// Rotate the path shape to follow the path direction.
    Rotate,
    /// Morph the path shape to follow the path curvature.
    Morph,
}

/// 1D path effect that stamps a path along another path.
///
/// Corresponds to Skia's `SkPath1DPathEffect`.
#[derive(Debug, Clone)]
pub struct Path1DEffect {
    /// The path to stamp.
    path: Path,
    /// Distance between stamps (advance).
    advance: Scalar,
    /// Initial phase offset.
    phase: Scalar,
    /// The distance along each contour of the first stamp: `phase` cleaned
    /// up as `SkPath1DPathEffect`'s constructor does.
    initial_offset: Scalar,
    /// How to place the stamped path.
    style: Path1DStyle,
}

impl Path1DEffect {
    /// Create a new 1D path effect.
    ///
    /// - `path`: The path to stamp along the input path.
    /// - `advance`: Distance between each stamp.
    /// - `phase`: Initial offset along the path.
    /// - `style`: How to orient the stamped path.
    ///
    /// Like `SkPath1DPathEffect::Make`, a non-positive or non-finite
    /// `advance`, a non-finite `phase` or an empty `path` gives `None`. The
    /// phase is cleaned up as Skia's constructor does (PostScript's
    /// interpretation): a positive phase becomes `advance - phase mod
    /// advance`, a negative one `-phase mod advance`, so stamps start that
    /// far into each contour.
    pub fn new(path: Path, advance: Scalar, phase: Scalar, style: Path1DStyle) -> Option<Self> {
        if !(advance.is_finite() && advance > 0.0 && phase.is_finite()) || path.is_empty() {
            return None;
        }
        let requested_phase = phase;
        let mut phase = phase;
        if phase < 0.0 {
            phase = -phase;
            if phase > advance {
                phase %= advance;
            }
        } else {
            if phase > advance {
                phase %= advance;
            }
            phase = advance - phase;
        }
        if phase >= advance {
            phase = 0.0;
        }
        Some(Self {
            path,
            advance,
            phase: requested_phase,
            initial_offset: phase,
            style,
        })
    }

    /// Get the stamped path.
    pub const fn path(&self) -> &Path {
        &self.path
    }

    /// Get the advance distance.
    pub const fn advance(&self) -> Scalar {
        self.advance
    }

    /// Get the phase.
    pub const fn phase(&self) -> Scalar {
        self.phase
    }

    /// Get the style.
    pub const fn style(&self) -> Path1DStyle {
        self.style
    }
}

/// The most points [`Path1DEffect`] emits (stamps times the stamp's point
/// count, over all contours); further stamps are dropped. A guard against
/// enormous path lengths and heavy stamps.
const MAX_PATH1D_POINTS: usize = 4_194_304;

/// `Sk1DPathEffect`'s per-contour stamp limit (`MAX_REASONABLE_ITERATIONS`):
/// exceeding it fails the effect.
const MAX_REASONABLE_ITERATIONS: u32 = 100_000;

/// `SkPath1DPathEffect`'s `morphpath`: bend `stamp` along contour `contour`
/// starting at `start_distance`. Each point `(sx, sy)` moves to the contour
/// point at `start_distance + sx`, offset by `sy` along the normal there (`morphpoints`); lines
/// become quads through their midpoint so they can bend. A point whose
/// position lookup fails (a NaN distance) drops its verb, as in Skia, where
/// `morphpoints` returns false and `morphpath` skips the verb.
fn morph_stamp(
    dst: &mut PathBuilder,
    stamp: &Path,
    measure: &crate::PathMeasure,
    contour: usize,
    start_distance: Scalar,
) {
    let morph = |p: Point| {
        let (pos, tangent) = measure.get_pos_tan_in_contour(contour, start_distance + p.x)?;
        // The normal offset: tangent rotated a quarter turn, scaled by y.
        let (nx, ny) = (-tangent.y * p.y, tangent.x * p.y);
        Some(Point::new(pos.x + nx, pos.y + ny))
    };
    let mut current = Point::zero();
    let mut start = Point::zero();
    for element in stamp {
        match element {
            PathElement::Move(p) => {
                if let Some(q) = morph(p) {
                    dst.move_to(q.x, q.y);
                }
                (current, start) = (p, p);
            }
            PathElement::Line(p) => {
                let mid = Point::new(f32::midpoint(current.x, p.x), f32::midpoint(current.y, p.y));
                if let (Some(c), Some(e)) = (morph(mid), morph(p)) {
                    dst.quad_to(c.x, c.y, e.x, e.y);
                }
                current = p;
            }
            PathElement::Quad(c, p) => {
                if let (Some(c), Some(e)) = (morph(c), morph(p)) {
                    dst.quad_to(c.x, c.y, e.x, e.y);
                }
                current = p;
            }
            PathElement::Conic(c, p, w) => {
                if let (Some(c), Some(e)) = (morph(c), morph(p)) {
                    dst.conic_to(c.x, c.y, e.x, e.y, w);
                }
                current = p;
            }
            PathElement::Cubic(c1, c2, p) => {
                if let (Some(a), Some(b), Some(e)) = (morph(c1), morph(c2), morph(p)) {
                    dst.cubic_to(a.x, a.y, b.x, b.y, e.x, e.y);
                }
                current = p;
            }
            PathElement::Close => {
                dst.close();
                current = start;
            }
        }
    }
}

impl PathEffect for Path1DEffect {
    fn filter_path(&self, path: &Path, rec: &mut StrokeRec) -> Option<Path> {
        // SkPath1DPathEffect::onFilterPath: rec->setFillStyle(), then filter.
        rec.set_fill_style();
        self.apply(path)
    }

    fn apply(&self, src_path: &Path) -> Option<Path> {
        use crate::PathMeasure;

        // Sk1DPathEffect::onFilterPath succeeds with nothing for a source
        // without contours (an inverse-filled empty path must not flood
        // the clip), so there is no early failure here.
        let measure = PathMeasure::new(src_path);
        // Output guard: at most MAX_PATH1D_POINTS points (stamps times the
        // stamp's point count); past it the stamps are truncated, like the
        // 2D effects, rather than failing (which would fill the source).
        // Sk1DPathEffect fails when a contour needs more than
        // MAX_REASONABLE_ITERATIONS stamps (or has a non-finite length).
        // Decide that first, so the point budget below never turns Skia's
        // failure into a truncated success.
        // Count by stepping `distance` exactly as the walk below (and Skia)
        // does, so rounding cannot make the two disagree at the limit; a
        // direct estimate settles every contour clearly under or over it
        // without stepping. "Clearly" is 2%: summing `advance` in f32 up to
        // 1e5 times drifts by at most ~0.6% (1e5 * 2^-24 relative error per
        // add), so only estimates within 2% of the limit need the walk.
        let limit = scalar_from_i32(i32::try_from(MAX_REASONABLE_ITERATIONS).unwrap_or(i32::MAX));
        for i in 0..measure.contour_count() {
            let length = measure.contour_length(i).unwrap_or(0.0);
            let estimate = ((length - self.initial_offset) / self.advance).ceil();
            if !estimate.is_finite() || estimate > limit * 1.02 {
                return None;
            }
            if estimate < limit * 0.98 {
                continue;
            }
            let mut distance = self.initial_offset;
            let mut stamps = 0_u32;
            loop {
                if distance >= length {
                    break;
                }
                if stamps == MAX_REASONABLE_ITERATIONS {
                    return None;
                }
                stamps += 1;
                distance += self.advance;
            }
        }
        // Morph turns each stamp line into a quad (one extra control point).
        let morph_extra = if self.style == Path1DStyle::Morph {
            self.path
                .iter()
                .filter(|e| matches!(e, PathElement::Line(_)))
                .count()
        } else {
            0
        };
        let stamp_points = (self.path.points().len() + morph_extra).max(1);
        let mut stamps_left = (MAX_PATH1D_POINTS / stamp_points).max(1);

        // Sk1DPathEffect::onFilterPath: each contour restarts at the phase,
        // and more than MAX_REASONABLE_ITERATIONS stamps on one contour
        // fails the effect. A path with no length succeeds with nothing.
        let mut builder = PathBuilder::new();
        for i in 0..measure.contour_count() {
            let length = measure.contour_length(i).unwrap_or(0.0);
            let mut distance = self.initial_offset;
            let mut governor = MAX_REASONABLE_ITERATIONS;
            loop {
                if distance >= length {
                    break;
                }
                if governor == 0 {
                    return None;
                }
                if stamps_left == 0 {
                    return Some(builder.build());
                }
                governor -= 1;
                stamps_left -= 1;
                if self.style == Path1DStyle::Morph {
                    morph_stamp(&mut builder, &self.path, &measure, i, distance);
                } else if let Some((pos, tangent)) = measure.get_pos_tan_in_contour(i, distance) {
                    let transform = match self.style {
                        Path1DStyle::Translate => skia_rs_core::Matrix::translate(pos.x, pos.y),
                        Path1DStyle::Rotate | Path1DStyle::Morph => {
                            let angle = tangent.y.atan2(tangent.x);
                            // First rotate, then translate
                            let rotation = skia_rs_core::Matrix::rotate(angle);
                            let translation = skia_rs_core::Matrix::translate(pos.x, pos.y);
                            translation.concat(&rotation)
                        }
                    };
                    builder.add_path(&self.path.transformed(&transform));
                }
                distance += self.advance;
            }
        }
        Some(builder.build())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Path1D
    }
}

// =============================================================================
// Path2D Effect
// =============================================================================

/// 2D path effect that tiles a path in a 2D pattern.
///
/// Corresponds to Skia's `SkPath2DPathEffect`.
#[derive(Debug, Clone)]
pub struct Path2DEffect {
    /// The transformation matrix for the tile.
    matrix: skia_rs_core::Matrix,
    /// The path to tile.
    path: Path,
}

impl Path2DEffect {
    /// Create a new 2D path effect.
    ///
    /// - `matrix`: The matrix defining the tiling pattern.
    /// - `path`: The path to tile.
    pub fn new(matrix: skia_rs_core::Matrix, path: Path) -> Option<Self> {
        if path.is_empty() {
            return None;
        }
        Some(Self { matrix, path })
    }

    /// Get the matrix.
    pub const fn matrix(&self) -> &skia_rs_core::Matrix {
        &self.matrix
    }

    /// Get the tiled path.
    pub const fn path(&self) -> &Path {
        &self.path
    }
}

impl Path2DEffect {
    /// `Sk2DPathEffect` tiling: a stamp at every lattice cell covered by the
    /// source (limited to `cull` when given).
    fn tile(&self, src_path: &Path, cull: Option<&skia_rs_core::Rect>) -> Option<Path> {
        // Sk2DPathEffect: a singular matrix fails the effect (the source is
        // drawn unchanged). An empty source succeeds with nothing (its
        // lattice rect is empty), so an inverse-filled empty path does not
        // flood the clip.
        let inverse = self.matrix.invert()?;
        // A stamp placed at `loc` is visible when `loc + stamp` meets the
        // cull: widen the cull by the stamp's extent.
        let b = self.path.bounds();
        let cull = cull.map(|c| {
            skia_rs_core::Rect::new(
                c.left - b.right,
                c.top - b.bottom,
                c.right - b.left,
                c.bottom - b.top,
            )
        });
        let mut spans = lattice_spans(src_path, &inverse, cull.as_ref());
        // Stamps cost per cell, so clip span columns to the cull here
        // (lattice_spans culls rows only).
        if let Some((cull_left, _, cull_right, _)) = lattice_cull(&inverse, cull.as_ref()) {
            spans = spans
                .into_iter()
                .filter_map(|(y, first, count)| {
                    let start = first.max(cull_left);
                    let end = first.saturating_add(count).min(cull_right);
                    (start < end).then(|| (y, start, end - start))
                })
                .collect();
        }

        // Output budget: keep every stamp when the total fits. Otherwise
        // share the budget across rows by need (water-filling: rows that
        // need little keep everything, the rest share an equal cap), so a
        // truncated pattern still spans the source. With more rows than
        // stamps, every `stride`-th row gets one stamp.
        let budget = (MAX_LATTICE_POINTS / self.path.points().len().max(1)).max(1);
        let mut row_need: Vec<(i32, usize)> = Vec::new();
        for &(y, _, count) in &spans {
            let count = usize::try_from(count).unwrap_or(0);
            match row_need.last_mut() {
                Some((row, need)) if *row == y => *need = need.saturating_add(count),
                _ => row_need.push((y, count)),
            }
        }
        let stride = row_need.len().div_ceil(budget).max(1);
        let kept: Vec<(i32, usize)> = row_need.into_iter().step_by(stride).collect();
        let row_cap = water_fill_cap(kept.iter().map(|&(_, need)| need), budget);
        let kept_rows: std::collections::HashSet<i32> = kept.iter().map(|&(y, _)| y).collect();

        // SkPath2DPathEffect::next: the stamp is *translated* to each covered
        // cell's center mapped through the lattice matrix.
        let mut builder = PathBuilder::new();
        let (mut row, mut used) = (None, 0_usize);
        for (y, left, count) in spans {
            if !kept_rows.contains(&y) {
                continue;
            }
            if row != Some(y) {
                (row, used) = (Some(y), 0);
            }
            let take = usize::try_from(count).unwrap_or(0).min(row_cap - used);
            used += take;
            for x in (left..).take(take) {
                let loc = self.matrix.map_point(Point::new(
                    scalar_from_i32(x) + 0.5,
                    scalar_from_i32(y) + 0.5,
                ));
                builder.add_path(
                    &self
                        .path
                        .transformed(&skia_rs_core::Matrix::translate(loc.x, loc.y)),
                );
            }
        }
        Some(builder.build())
    }
}

impl PathEffect for Path2DEffect {
    fn apply(&self, src_path: &Path) -> Option<Path> {
        self.tile(src_path, None)
    }

    fn filter_path_culled(
        &self,
        path: &Path,
        rec: &mut StrokeRec,
        cull: Option<&skia_rs_core::Rect>,
        res_scale: Scalar,
    ) -> Option<Path> {
        // The stamps are drawn with whatever stroke the record holds, which
        // an earlier effect (a Sum's first child, a Compose's inner effect)
        // may have set. Its miter joins reach width/2 * the paint's miter
        // limit, which this effect cannot see, so a stroked record is not
        // culled (the point budget still bounds the output). Fills and
        // hairlines reach at most a pixel past the stamp.
        let _ = res_scale;
        let cull = match *rec {
            StrokeRec::Stroke(_) | StrokeRec::StrokeAndFill(_) => None,
            StrokeRec::Fill | StrokeRec::Hairline => cull.map(|c| c.inset(-1.0, -1.0)),
        };
        self.tile(path, cull.as_ref())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Path2D
    }
}

// =============================================================================
// Line2D Effect
// =============================================================================

/// Resource guards for the 2D effects against pathological lattices: at
/// most this many lattice rows are scanned (sampled evenly over the source
/// when it has more, so the pattern still covers it), columns are limited
/// to [`MAX_LATTICE_COLUMNS`] (only to keep counts in `i32`), and
/// [`Line2DEffect`] emits at most half that many lines, sampled evenly, and
/// [`Path2DEffect`] emits at most [`MAX_LATTICE_POINTS`] points, budgeted
/// evenly across the covered rows (each row keeps its leftmost cells; with
/// more rows than stamps, rows are sampled evenly). The
/// scanned lattice rect is first limited to the visible (cull) area, when
/// the caller provides one. Beyond these limits the pattern is truncated
/// (Skia has no limit outside fuzzer builds, where it caps spans at 100
/// cells).
const MAX_LATTICE_EXTENT: i64 = 1 << 16;
/// See [`MAX_LATTICE_EXTENT`]: the column limit, which only keeps span
/// counts in `i32` (columns do not add to the scan cost).
const MAX_LATTICE_COLUMNS: i64 = 1 << 30;
/// See [`MAX_LATTICE_EXTENT`]: the most points [`Path2DEffect`] emits
/// (stamps times the stamp's point count).
const MAX_LATTICE_POINTS: usize = 1 << 21;

/// The lattice-space edges of `mapped`, flattened, with every contour
/// implicitly closed, oriented downward as `(top, bottom, winding)` and
/// sorted by their top. Horizontal edges are kept; they never cross a row
/// center, so the scan skips them.
fn lattice_edges(mapped: &Path) -> Vec<(Point, Point, i32)> {
    let flat = flatten_path_to_lines(mapped, 0.1);
    let mut edges: Vec<(Point, Point)> = Vec::new();
    let (mut start, mut current) = (Point::zero(), Point::zero());
    for element in &flat {
        match element {
            PathElement::Move(p) => {
                if current != start {
                    edges.push((current, start));
                }
                start = p;
                current = p;
            }
            PathElement::Line(p)
            | PathElement::Quad(_, p)
            | PathElement::Conic(_, p, _)
            | PathElement::Cubic(_, _, p) => {
                edges.push((current, p));
                current = p;
            }
            PathElement::Close => {
                if current != start {
                    edges.push((current, start));
                }
                current = start;
            }
        }
    }
    if current != start {
        edges.push((current, start));
    }

    let mut edges: Vec<(Point, Point, i32)> = edges
        .into_iter()
        .map(|(a, b)| if a.y < b.y { (a, b, 1) } else { (b, a, -1) })
        .collect();
    edges.sort_by(|p, q| p.0.y.total_cmp(&q.0.y));
    edges
}

/// The inside intervals of one lattice row from its sorted `crossings`,
/// per the fill rule; inverse fills take the complement within
/// `[left, right)`.
fn row_intervals(
    crossings: &[(Scalar, i32)],
    even_odd: bool,
    inverse_fill: bool,
    left: i32,
    right: i32,
) -> Vec<(Scalar, Scalar)> {
    let mut intervals: Vec<(Scalar, Scalar)> = Vec::new();
    let mut winding = 0;
    let mut entered: Option<Scalar> = None;
    for &(x, dir) in crossings {
        winding += dir;
        let inside = if even_odd {
            winding % 2 != 0
        } else {
            winding != 0
        };
        match (inside, entered) {
            (true, None) => entered = Some(x),
            (false, Some(x0)) => {
                intervals.push((x0, x));
                entered = None;
            }
            _ => {}
        }
    }
    if inverse_fill {
        // The complement of the inside intervals within the lattice rect.
        let mut complement = Vec::new();
        let mut cursor = scalar_from_i32(left);
        for &(x0, x1) in &intervals {
            complement.push((cursor, x0));
            cursor = x1;
        }
        complement.push((cursor, scalar_from_i32(right)));
        intervals = complement;
    }
    intervals
}

/// The cull rect in lattice cells, `(left, top, right, bottom)` half-open,
/// outset by a cell (a cell's output can reach past its center by up to a
/// cell). `None` when there is no usable cull: none given, non-finite, or a
/// perspective inverse, which can map the cull's corners across the horizon
/// (w <= 0) where their bounding box no longer covers the visible cells.
fn lattice_cull(
    inverse: &skia_rs_core::Matrix,
    cull: Option<&skia_rs_core::Rect>,
) -> Option<(i32, i32, i32, i32)> {
    let cull = cull.filter(|c| c.is_finite() && !inverse.has_perspective())?;
    let c = inverse.map_rect(cull);
    Some((
        floor_to_i32(c.left).saturating_sub(1),
        floor_to_i32(c.top).saturating_sub(1),
        ceil_to_i32(c.right).saturating_add(1),
        ceil_to_i32(c.bottom).saturating_add(1),
    ))
}

/// The largest per-row cap `c` such that `sum(min(need, c)) <= budget`
/// (`usize::MAX` when everything fits): rows needing less than `c` keep all
/// of their cells and the remainder is shared equally.
fn water_fill_cap(needs: impl Iterator<Item = usize>, budget: usize) -> usize {
    let mut needs: Vec<usize> = needs.collect();
    // Saturating: on 32-bit targets the total can exceed usize.
    if needs.iter().fold(0_usize, |sum, &n| sum.saturating_add(n)) <= budget {
        return usize::MAX;
    }
    needs.sort_unstable();
    let mut remaining = budget;
    let mut rows_left = needs.len();
    for need in needs {
        let fair = remaining / rows_left;
        if need > fair {
            return fair;
        }
        remaining -= need;
        rows_left -= 1;
    }
    usize::MAX
}

/// The lattice cells covered by `src`, as `(row, first_column, count)`
/// spans, following `Sk2DPathEffect::filterPath`: the source is mapped into
/// lattice space by `inverse`, its bounds are rounded to an integer rect, and
/// the mapped path is scan-converted like `SkRegion::setPath` (non-AA, a cell
/// is covered when its center `(x + 0.5, y + 0.5)` is inside, respecting the
/// fill type). A source with no area covers no cells.
///
/// Rows and columns beyond [`MAX_LATTICE_EXTENT`] are not scanned.
#[allow(
    clippy::suboptimal_flops,
    reason = "unfused arithmetic like SkRegion's edge walk; mul_add rounds differently"
)]
fn lattice_spans(
    src: &Path,
    inverse: &skia_rs_core::Matrix,
    cull: Option<&skia_rs_core::Rect>,
) -> Vec<(i32, i32, i32)> {
    if src.is_empty() {
        return Vec::new();
    }
    let mapped = src.transformed(inverse);
    let b = mapped.bounds();
    // SkRect::round (sk_float_round2int: floor(x + 0.5)).
    let round = |v: Scalar| floor_to_i32(v + 0.5);
    let (left, top, right, bottom) = (round(b.left), round(b.top), round(b.right), round(b.bottom));
    // Limit to the cull area in lattice space, outset by a cell (a cell's
    // output can reach past its center by up to a cell). Cells outside it
    // produce nothing visible.
    // Only rows are culled here: spans keep their true extent, so callers
    // that judge a span by its length (Line2D's `count > 1`) see the whole
    // run, and callers that pay per cell clip columns themselves.
    let (top, bottom) = match lattice_cull(inverse, cull) {
        Some((_, cull_top, _, cull_bottom)) => (top.max(cull_top), bottom.min(cull_bottom)),
        None => (top, bottom),
    };
    if left >= right || top >= bottom {
        return Vec::new();
    }
    // Measure in i64 (the rounded edges saturate, so `right - left` could
    // overflow i32). Columns are clamped to keep counts in i32; rows beyond
    // the scan limit are sampled with a stride instead of cut off.
    let right = i32::try_from(
        (i64::from(right) - i64::from(left)).min(MAX_LATTICE_COLUMNS) + i64::from(left),
    )
    .unwrap_or(right);
    let height = i64::from(bottom) - i64::from(top);
    let row_stride = usize::try_from((height + MAX_LATTICE_EXTENT - 1) / MAX_LATTICE_EXTENT)
        .unwrap_or(usize::MAX)
        .max(1);

    let edges = lattice_edges(&mapped);

    let fill_type = mapped.fill_type();
    let even_odd = matches!(fill_type, FillType::EvenOdd | FillType::InverseEvenOdd);
    let inverse_fill = matches!(
        fill_type,
        FillType::InverseWinding | FillType::InverseEvenOdd
    );

    // Active-edge scan: edges enter the active list when the row reaches
    // their top and leave once below, so each row only visits the edges
    // that cross it.
    let mut next_edge = 0;
    let mut active: Vec<(Point, Point, i32)> = Vec::new();

    let mut spans = Vec::new();
    let mut crossings: Vec<(Scalar, i32)> = Vec::new();
    let mut row_spans: Vec<(i32, i32)> = Vec::new();
    for y in (top..bottom).step_by(row_stride) {
        let yc = scalar_from_i32(y) + 0.5;
        while next_edge < edges.len() && edges[next_edge].0.y <= yc {
            active.push(edges[next_edge]);
            next_edge += 1;
        }
        active.retain(|(_, hi, _)| yc < hi.y);
        if active.is_empty() && next_edge == edges.len() && !inverse_fill {
            break;
        }
        crossings.clear();
        for &(lo, hi, dir) in &active {
            if lo.y <= yc {
                let x = (yc - lo.y) / (hi.y - lo.y) * (hi.x - lo.x) + lo.x;
                crossings.push((x, dir));
            }
        }
        crossings.sort_by(|p, q| p.0.total_cmp(&q.0));

        let intervals = row_intervals(&crossings, even_odd, inverse_fill, left, right);

        // A cell x is covered when its center x + 0.5 lies in [x0, x1).
        // Touching or overlapping runs merge into one span, as an SkRegion
        // row does (two shapes sharing an edge can split the winding there).
        row_spans.clear();
        for (x0, x1) in intervals {
            let first = ceil_to_i32(x0 - 0.5).max(left);
            let end = ceil_to_i32(x1 - 0.5).min(right);
            if first < end {
                row_spans.push((first, end));
            }
        }
        row_spans.sort_unstable();
        let mut merged: Option<(i32, i32)> = None;
        for &(first, end) in &row_spans {
            merged = match merged {
                Some((m0, m1)) if first <= m1 => Some((m0, m1.max(end))),
                Some((m0, m1)) => {
                    spans.push((y, m0, m1 - m0));
                    Some((first, end))
                }
                None => Some((first, end)),
            };
        }
        if let Some((m0, m1)) = merged {
            spans.push((y, m0, m1 - m0));
        }
    }
    spans
}

/// 2D line effect that fills a region with parallel lines.
///
/// Corresponds to Skia's `SkLine2DPathEffect`.
#[derive(Debug, Clone, Copy)]
pub struct Line2DEffect {
    /// Line width.
    width: Scalar,
    /// The transformation matrix for the line pattern.
    matrix: skia_rs_core::Matrix,
}

impl Line2DEffect {
    /// Create a new 2D line effect.
    ///
    /// - `width`: The width of the lines.
    /// - `matrix`: The matrix defining the line pattern orientation and spacing.
    #[must_use]
    ///
    /// Like `SkLine2DPathEffect::Make`, a width of 0 is a hairline hatch and
    /// a negative or NaN width gives `None`.
    pub fn new(width: Scalar, matrix: skia_rs_core::Matrix) -> Option<Self> {
        if width.is_nan() || width < 0.0 {
            return None;
        }
        Some(Self { width, matrix })
    }

    /// Get the line width.
    #[must_use]
    pub const fn width(&self) -> Scalar {
        self.width
    }

    /// Get the matrix.
    #[must_use]
    pub const fn matrix(&self) -> &skia_rs_core::Matrix {
        &self.matrix
    }
}

impl Line2DEffect {
    /// The hatch centerlines (`SkLine2DPathEffect::nextSpan`): one line per
    /// span of more than one covered lattice cell, from the first cell's
    /// center to one past the last, mapped through the lattice matrix.
    /// `None` when the lattice matrix is singular.
    fn hatch_lines(
        &self,
        src_path: &Path,
        cull: Option<&skia_rs_core::Rect>,
    ) -> Option<Vec<(Point, Point)>> {
        // As in Sk2DPathEffect, only a singular matrix fails; an empty
        // source succeeds with no lines.
        let inverse = self.matrix.invert()?;
        let spans: Vec<_> = lattice_spans(src_path, &inverse, cull)
            .into_iter()
            .filter(|&(_, _, count)| count > 1)
            .collect();
        // Output guard: at most MAX_LATTICE_POINTS / 2 lines, sampled
        // evenly over the rows when there are more.
        let stride = spans.len().div_ceil(MAX_LATTICE_POINTS / 2).max(1);
        let lines = spans
            .into_iter()
            .step_by(stride)
            .map(|(v, u, count)| {
                let y = scalar_from_i32(v) + 0.5;
                (
                    self.matrix
                        .map_point(Point::new(scalar_from_i32(u) + 0.5, y)),
                    self.matrix
                        .map_point(Point::new(scalar_from_i32(u + count) + 0.5, y)),
                )
            })
            .collect();
        Some(lines)
    }
}

impl PathEffect for Line2DEffect {
    /// Skia's `SkLine2DPathEffect::onFilterPath`: the open hatch centerlines,
    /// with the record set to stroke them at the effect's width
    /// (`rec->setStrokeStyle(fWidth)`). A failure leaves the record alone.
    fn filter_path(&self, path: &Path, rec: &mut StrokeRec) -> Option<Path> {
        self.filter_path_culled(path, rec, None, 1.0)
    }

    /// As [`filter_path`](PathEffect::filter_path), skipping hatch lines
    /// outside `cull` (widened by the line width).
    fn filter_path_culled(
        &self,
        path: &Path,
        rec: &mut StrokeRec,
        cull: Option<&skia_rs_core::Rect>,
        res_scale: Scalar,
    ) -> Option<Path> {
        let _ = res_scale;
        let cull = cull.map(|c| c.inset(-self.width, -self.width));
        let lines = self.hatch_lines(path, cull.as_ref())?;
        rec.set_stroke_style(self.width);
        let mut builder = PathBuilder::new();
        for (p0, p1) in lines {
            builder.move_to(p0.x, p0.y).line_to(p1.x, p1.y);
        }
        Some(builder.build())
    }

    /// The hatch as fillable geometry: each centerline from
    /// [`filter_path`](PathEffect::filter_path) outlined as a butt-ended band
    /// of the effect's width. A zero-width (hairline) hatch has no area, so
    /// it returns the open centerlines themselves: they draw only when
    /// stroked as hairlines, which `filter_path`'s stroke record does.
    fn apply(&self, src_path: &Path) -> Option<Path> {
        let lines = self.hatch_lines(src_path, None)?;
        if self.width == 0.0 {
            let mut builder = PathBuilder::new();
            for (p0, p1) in lines {
                builder.move_to(p0.x, p0.y).line_to(p1.x, p1.y);
            }
            return Some(builder.build());
        }
        let half_width = self.width / 2.0;
        // Bands cost four points each (hatch_lines budgets two per line):
        // sample evenly to stay within MAX_LATTICE_POINTS.
        let stride = lines.len().div_ceil(MAX_LATTICE_POINTS / 4).max(1);
        let mut builder = PathBuilder::new();
        for (p0, p1) in lines.into_iter().step_by(stride) {
            let (dx, dy) = (p1.x - p0.x, p1.y - p0.y);
            let len = dx.hypot(dy);
            if len > 0.0 && half_width > 0.0 {
                let (nx, ny) = (-dy / len * half_width, dx / len * half_width);
                builder.move_to(p0.x + nx, p0.y + ny);
                builder.line_to(p1.x + nx, p1.y + ny);
                builder.line_to(p1.x - nx, p1.y - ny);
                builder.line_to(p0.x - nx, p0.y - ny);
                builder.close();
            }
        }
        Some(builder.build())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Line2D
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a dash path effect.
#[must_use]
pub fn make_dash(intervals: Vec<Scalar>, phase: Scalar) -> Option<PathEffectRef> {
    DashEffect::new(intervals, phase).map(|e| Arc::new(e) as PathEffectRef)
}

/// Create a corner path effect.
#[must_use]
pub fn make_corner(radius: Scalar) -> Option<PathEffectRef> {
    CornerEffect::new(radius).map(|e| Arc::new(e) as PathEffectRef)
}

/// Create a discrete path effect.
#[must_use]
pub fn make_discrete(seg_length: Scalar, deviation: Scalar, seed: u32) -> Option<PathEffectRef> {
    DiscreteEffect::new(seg_length, deviation, seed).map(|e| Arc::new(e) as PathEffectRef)
}

/// Create a trim path effect.
#[must_use]
pub fn make_trim(start: Scalar, end: Scalar, mode: TrimMode) -> Option<PathEffectRef> {
    TrimEffect::new(start, end, mode).map(|e| Arc::new(e) as PathEffectRef)
}

/// Compose two path effects.
pub fn make_compose(outer: PathEffectRef, inner: PathEffectRef) -> PathEffectRef {
    Arc::new(ComposeEffect::new(outer, inner))
}

/// Sum two path effects.
pub fn make_sum(first: PathEffectRef, second: PathEffectRef) -> PathEffectRef {
    Arc::new(SumEffect::new(first, second))
}

/// Create a 1D path effect.
pub fn make_path_1d(
    path: Path,
    advance: Scalar,
    phase: Scalar,
    style: Path1DStyle,
) -> Option<PathEffectRef> {
    Path1DEffect::new(path, advance, phase, style).map(|e| Arc::new(e) as PathEffectRef)
}

/// Create a 2D path effect.
pub fn make_path_2d(matrix: skia_rs_core::Matrix, path: Path) -> Option<PathEffectRef> {
    Path2DEffect::new(matrix, path).map(|e| Arc::new(e) as PathEffectRef)
}

/// Create a 2D line effect.
#[must_use]
pub fn make_line_2d(width: Scalar, matrix: skia_rs_core::Matrix) -> Option<PathEffectRef> {
    Line2DEffect::new(width, matrix).map(|e| Arc::new(e) as PathEffectRef)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A horizontal line: non-empty path whose bounds have zero height.
    fn flat_line() -> Path {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 10.0).line_to(50.0, 10.0);
        b.build()
    }

    /// A non-empty path with zero length.
    fn zero_length() -> Path {
        let mut b = PathBuilder::new();
        b.move_to(5.0, 5.0).line_to(5.0, 5.0);
        b.build()
    }

    fn square(size: Scalar) -> Path {
        let mut b = PathBuilder::new();
        b.add_rect(&skia_rs_core::Rect::from_xywh(0.0, 0.0, size, size));
        b.build()
    }

    #[test]
    fn effects_that_succeed_with_nothing_return_empty_not_none() {
        // `None` makes callers draw the source unchanged; these cases succeed
        // in Skia with an empty result, so nothing must be drawn.
        let tile = Path2DEffect::new(skia_rs_core::Matrix::scale(4.0, 4.0), square(2.0)).unwrap();
        let produced = tile.apply(&flat_line()).expect("2D effect applies");
        assert!(produced.is_empty(), "empty-bounds source tiles to nothing");

        let lines = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap();
        let produced = lines.apply(&flat_line()).expect("line 2D effect applies");
        assert!(
            produced.is_empty(),
            "empty-bounds source hatches to nothing"
        );

        let stamp = Path1DEffect::new(square(2.0), 5.0, 0.0, Path1DStyle::Translate).unwrap();
        let produced = stamp.apply(&zero_length()).expect("1D effect applies");
        assert!(produced.is_empty(), "zero-length source gets no stamps");
    }

    #[test]
    fn two_d_effects_only_cover_cells_inside_the_source() {
        // Sk2DPathEffect scan-converts the source into lattice cells
        // (SkRegion::setPath at cell centers): a zero-area source covers no
        // cells, so nothing is drawn even under a rotated lattice...
        let rotated = skia_rs_core::Matrix::rotate(std::f32::consts::FRAC_PI_4)
            .concat(&skia_rs_core::Matrix::scale(4.0, 4.0));
        let hatch = Line2DEffect::new(1.0, rotated).unwrap();
        assert!(hatch.apply(&flat_line()).expect("applies").is_empty());

        // ...while a source with area is hatched under the same lattice.
        assert!(!hatch.apply(&square(40.0)).expect("applies").is_empty());

        // Path2D stamps exactly the covered cells: a 20x20 square under a
        // 4x4 lattice covers 5 x 5 cell centers.
        let mut stamp = PathBuilder::new();
        stamp
            .move_to(0.0, 0.0)
            .line_to(1.0, 0.0)
            .line_to(0.0, 1.0)
            .close();
        let stamp = stamp.build();
        let tile = Path2DEffect::new(skia_rs_core::Matrix::scale(4.0, 4.0), stamp).unwrap();
        let produced = tile.apply(&square(20.0)).expect("applies");
        let contours = produced
            .iter()
            .filter(|e| matches!(e, PathElement::Move(_)))
            .count();
        assert_eq!(contours, 25);
    }
    #[test]
    fn huge_lattices_are_truncated_not_failed_or_overflowed() {
        // A 2100 x 2100 source under a unit lattice (4.4M cells) still
        // hatches (it used to fail, drawing the source as a solid fill).
        let unit = Line2DEffect::new(1.0, skia_rs_core::Matrix::IDENTITY).unwrap();
        assert!(!unit.apply(&square(2100.0)).expect("applies").is_empty());

        // Lattice-space bounds beyond i32 (a 3e6-wide source under a 0.001
        // lattice) must not overflow when measuring the lattice rect.
        let tiny = skia_rs_core::Matrix::scale(0.001, 0.001);
        let mut b = PathBuilder::new();
        b.add_rect(&skia_rs_core::Rect::from_xywh(-1.5e6, 0.0, 3e6, 10.0));
        let wide = b.build();
        assert!(Line2DEffect::new(1.0, tiny).unwrap().apply(&wide).is_some());
        let dot = square(0.0001);
        assert!(Path2DEffect::new(tiny, dot).unwrap().apply(&wide).is_some());
    }

    #[test]
    fn line2d_apply_is_fillable_and_filter_path_strokes_centerlines() {
        let hatch = Line2DEffect::new(2.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap();
        let closes = |p: &Path| p.iter().filter(|e| matches!(e, PathElement::Close)).count();

        // apply: closed bands that draw when filled.
        let bands = hatch.apply(&square(40.0)).expect("applies");
        assert!(closes(&bands) > 0, "apply returns fillable bands");

        // filter_path: open centerlines, stroked at the effect width.
        let mut rec = StrokeRec::Fill;
        let lines = hatch.filter_path(&square(40.0), &mut rec).expect("applies");
        assert_eq!(closes(&lines), 0);
        assert!(!lines.is_empty());
        assert_eq!(rec, StrokeRec::Stroke(2.0));
    }

    #[test]
    fn line2d_merges_runs_split_by_a_shared_edge() {
        // Two same-direction rects sharing the edge x = 6: under a 4x4
        // lattice the left one covers only cell 0 and the right one cells
        // 1..=9. An SkRegion row merges them into one run, so each hatch
        // line starts at cell 0's center (x = 2), not at cell 1's (x = 6).
        let mut b = PathBuilder::new();
        b.add_rect(&skia_rs_core::Rect::new(0.0, 0.0, 6.0, 40.0));
        b.add_rect(&skia_rs_core::Rect::new(6.0, 0.0, 40.0, 40.0));
        let source = b.build();
        let hatch = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap();
        let mut rec = StrokeRec::Fill;
        let lines = hatch.filter_path(&source, &mut rec).expect("applies");
        let min_x = lines
            .points()
            .iter()
            .map(|p| p.x)
            .fold(Scalar::INFINITY, Scalar::min);
        assert!((min_x - 2.0).abs() < 1e-4, "hatch starts at x = {min_x}");
    }

    #[test]
    fn two_d_effects_succeed_with_nothing_on_an_empty_source() {
        // Sk2DPathEffect::onFilterPath succeeds for an empty source (and
        // Line2D still sets the stroke record), so an inverse-filled empty
        // path is not drawn unchanged (which would flood the clip).
        let mut b = PathBuilder::new();
        b.fill_type(FillType::InverseWinding);
        let empty = b.build();
        let lattice = skia_rs_core::Matrix::scale(4.0, 4.0);

        let hatch = Line2DEffect::new(1.0, lattice).unwrap();
        let mut rec = StrokeRec::Fill;
        let lines = hatch.filter_path(&empty, &mut rec).expect("succeeds");
        assert!(lines.is_empty());
        assert_eq!(rec, StrokeRec::Stroke(1.0));

        let tile = Path2DEffect::new(lattice, square(1.0)).unwrap();
        assert!(tile.apply(&empty).expect("succeeds").is_empty());
    }

    #[test]
    fn dash_and_1d_effects_fail_on_unbounded_output() {
        // SkDashPath: more than kMaxDashCount (1e6) dashes per contour fails.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(1e8, 0.0);
        let long = b.build();
        let dash = DashEffect::new(vec![5.0, 5.0], 0.0).unwrap();
        assert!(dash.apply(&long).is_none(), "1e7 dashes: gives up");
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(1000.0, 0.0);
        assert!(dash.apply(&b.build()).is_some(), "100 dashes: fine");

        // Path1D: a non-finite or enormous stamp count fails too.
        let stamp = square(1.0);
        let one_d = Path1DEffect::new(stamp, 1.0, 0.0, Path1DStyle::Translate).unwrap();
        // Over Skia's 100,000 stamps per contour: fails like Skia.
        assert!(one_d.apply(&long).is_none(), "1e8 stamps: gives up");
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(3e38, 3e38);
        assert!(
            one_d.apply(&b.build()).is_none(),
            "infinite length: gives up"
        );
    }

    #[test]
    fn path2d_output_is_bounded_by_points() {
        // A 13-point stamp over ~1M covered cells stays within the point
        // budget instead of emitting 13M points.
        let mut b = PathBuilder::new();
        b.add_circle(0.0, 0.0, 0.5);
        let circle = b.build();
        let per_stamp = circle.points().len();
        let tile = Path2DEffect::new(skia_rs_core::Matrix::scale(2.0, 2.0), circle).unwrap();
        let produced = tile.apply(&square(2000.0)).expect("applies");
        assert!(produced.points().len() <= MAX_LATTICE_POINTS + per_stamp);
        assert!(!produced.is_empty());
    }

    #[test]
    fn two_d_effects_skip_cells_outside_the_cull() {
        let hatch = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap();
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 40.0, 40.0);
        let mut rec = StrokeRec::Fill;
        let lines = hatch
            .filter_path_culled(&square(4000.0), &mut rec, Some(&cull), 1.0)
            .expect("applies");
        // Only rows near the cull remain (each still spans the source).
        let max_y = lines.points().iter().map(|p| p.y).fold(0.0, Scalar::max);
        assert!(!lines.is_empty() && max_y < 60.0, "max y = {max_y}");
    }

    #[test]
    fn heavy_stamps_respect_the_point_budgets() {
        // A 10k-point stamp: Path2D over many rows must stay within the point
        // budget (sampling rows), and Path1D must fail rather than emit
        // ~1e9 points.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        for i in 1..10_000 {
            b.line_to(scalar_from_i32(i % 7) * 0.01, scalar_from_i32(i % 5) * 0.01);
        }
        let heavy = b.build();
        let per_stamp = heavy.points().len();

        let tile =
            Path2DEffect::new(skia_rs_core::Matrix::scale(1.0, 0.01), heavy.clone()).unwrap();
        let produced = tile.apply(&square(100.0)).expect("applies");
        assert!(!produced.is_empty());
        assert!(produced.points().len() <= MAX_LATTICE_POINTS + per_stamp);

        let one_d = Path1DEffect::new(heavy, 1.0, 0.0, Path1DStyle::Translate).unwrap();
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(5e4, 0.0);
        let stamped = one_d.apply(&b.build()).expect("truncates, not fails");
        assert!(!stamped.is_empty());
        assert!(stamped.points().len() <= MAX_PATH1D_POINTS);
    }

    #[test]
    fn path1d_cleans_up_phase_and_restarts_per_contour_like_skia() {
        let stamp = square(1.0);
        let starts = |phase: Scalar, src: &Path| {
            let effect = Path1DEffect::new(stamp.clone(), 10.0, phase, Path1DStyle::Translate)
                .expect("valid");
            let out = effect.apply(src).expect("applies");
            out.iter()
                .filter_map(|e| match e {
                    PathElement::Move(p) => Some((p.x, p.y)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(25.0, 0.0);
        let line = b.build();
        // SkPath1DPathEffect: phase 3 -> offset 7; phase -3 -> offset 3.
        assert_eq!(starts(3.0, &line), [(7.0, 0.0), (17.0, 0.0)]);
        assert_eq!(starts(-3.0, &line), [(3.0, 0.0), (13.0, 0.0), (23.0, 0.0)]);

        // Each contour restarts at the phase (offset 0 for phase 0... which
        // Skia maps to advance - 0 = advance -> 0).
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(15.0, 0.0);
        b.move_to(0.0, 50.0).line_to(15.0, 50.0);
        assert_eq!(
            starts(0.0, &b.build()),
            [(0.0, 0.0), (10.0, 0.0), (0.0, 50.0), (10.0, 50.0)]
        );

        // Non-finite phase or advance is rejected (SkPath1DPathEffect::Make).
        assert!(
            Path1DEffect::new(stamp.clone(), 1.0, Scalar::NAN, Path1DStyle::Translate).is_none()
        );
        assert!(Path1DEffect::new(stamp, Scalar::INFINITY, 0.0, Path1DStyle::Translate).is_none());
    }

    #[test]
    fn dash_count_accumulates_across_contours() {
        // 20 contours of 600k dashes each: each is under kMaxDashCount but
        // the total (12M) is over, so SkDashPath gives up.
        let mut b = PathBuilder::new();
        for i in 0..20 {
            let y = scalar_from_i32(i);
            b.move_to(0.0, y).line_to(6e6, y);
        }
        let dash = DashEffect::new(vec![5.0, 5.0], 0.0).unwrap();
        assert!(dash.apply(&b.build()).is_none());
    }

    #[test]
    fn line2d_output_is_bounded() {
        // Many thin contours under a fine lattice: the line count stays
        // within the budget.
        let mut b = PathBuilder::new();
        for i in 0..400 {
            let x = scalar_from_i32(i) * 2.0;
            b.add_rect(&skia_rs_core::Rect::new(x, 0.0, x + 1.0, 100.0));
        }
        let hatch = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(0.02, 0.02)).unwrap();
        let mut rec = StrokeRec::Fill;
        let lines = hatch.filter_path(&b.build(), &mut rec).expect("applies");
        assert!(!lines.is_empty());
        assert!(lines.points().len() <= MAX_LATTICE_POINTS);
    }

    #[test]
    fn path1d_succeeds_with_nothing_on_an_empty_source_and_keeps_its_phase() {
        let mut b = PathBuilder::new();
        b.fill_type(FillType::InverseWinding);
        let empty = b.build();
        let effect = Path1DEffect::new(square(1.0), 10.0, 3.0, Path1DStyle::Translate).unwrap();
        let mut rec = StrokeRec::Fill;
        let out = effect.filter_path(&empty, &mut rec).expect("succeeds");
        assert!(out.is_empty());

        // The getter returns the caller's phase, so rebuilding is lossless.
        assert!((effect.phase() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn line2d_apply_bands_stay_within_the_point_budget() {
        let mut b = PathBuilder::new();
        for i in 0..400 {
            let x = scalar_from_i32(i) * 2.0;
            b.add_rect(&skia_rs_core::Rect::new(x, 0.0, x + 1.0, 100.0));
        }
        let hatch = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(0.02, 0.02)).unwrap();
        let bands = hatch.apply(&b.build()).expect("applies");
        assert!(!bands.is_empty());
        assert!(bands.points().len() <= MAX_LATTICE_POINTS);
    }

    #[test]
    fn path2d_keeps_every_stamp_when_the_total_fits() {
        // An L: a 1-cell bar 2000 rows tall plus a 10000-cell base row.
        // ~12000 stamps fit the budget, so the base row is complete.
        let mut b = PathBuilder::new();
        b.add_rect(&skia_rs_core::Rect::new(0.0, 0.0, 1.0, 2000.0));
        b.add_rect(&skia_rs_core::Rect::new(0.0, 2000.0, 10000.0, 2001.0));
        let tile = Path2DEffect::new(skia_rs_core::Matrix::IDENTITY, square(0.5)).unwrap();
        let produced = tile.apply(&b.build()).expect("applies");
        let stamps = produced
            .iter()
            .filter(|e| matches!(e, PathElement::Move(_)))
            .count();
        assert_eq!(stamps, 2000 + 10000);
        assert_eq!(water_fill_cap([1, 1, 50].into_iter(), 10), 8);
        assert_eq!(water_fill_cap([1, 1, 5].into_iter(), 10), usize::MAX);
    }

    #[test]
    fn compose_does_not_cull_the_inner_effect() {
        // A dash over a Line2D hatch must see the full hatch lines: culling
        // the inner effect would move every dash with the clip.
        let hatch: PathEffectRef =
            Arc::new(Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap());
        let dash: PathEffectRef = Arc::new(DashEffect::new(vec![3.0, 3.0], 0.0).unwrap());
        let composed = ComposeEffect::new(dash, hatch);
        let source = square(400.0);
        let cull = skia_rs_core::Rect::new(200.0, 0.0, 220.0, 400.0);
        let mut rec = StrokeRec::Fill;
        let culled = composed
            .filter_path_culled(&source, &mut rec, Some(&cull), 1.0)
            .expect("applies");
        let mut rec = StrokeRec::Fill;
        let full = composed.filter_path(&source, &mut rec).expect("applies");
        assert_eq!(culled.points(), full.points());
    }

    #[test]
    fn path1d_fails_like_skia_before_applying_the_point_budget() {
        // A 100-point stamp every 1.0 along 150,000 units needs 150,000
        // stamps: over Skia's 100,000 per contour, so the effect fails even
        // though the point budget alone would have truncated it.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        for i in 1..100 {
            b.line_to(scalar_from_i32(i % 3) * 0.1, scalar_from_i32(i % 2) * 0.1);
        }
        let stamp = b.build();
        let effect = Path1DEffect::new(stamp, 1.0, 0.0, Path1DStyle::Translate).unwrap();
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(150_000.0, 0.0);
        assert!(effect.apply(&b.build()).is_none());
    }

    #[test]
    fn tall_lattices_are_sampled_over_the_whole_source() {
        // 100,000 lattice rows (over the 65,536-row scan limit) with no cull:
        // rows are sampled evenly, so the hatch still reaches the bottom of
        // the source instead of stopping at y = 655.
        let mut b = PathBuilder::new();
        b.add_rect(&skia_rs_core::Rect::new(0.0, 0.0, 10.0, 1000.0));
        let hatch = Line2DEffect::new(0.005, skia_rs_core::Matrix::scale(0.01, 0.01)).unwrap();
        let mut rec = StrokeRec::Fill;
        let lines = hatch.filter_path(&b.build(), &mut rec).expect("applies");
        let max_y = lines.points().iter().map(|p| p.y).fold(0.0, Scalar::max);
        assert!(max_y > 990.0, "hatch reaches y = {max_y}");
    }

    #[test]
    fn path1d_morph_bends_stamps_along_the_path() {
        // A straight 10-unit bar morphed along a circle of radius 50 must
        // follow the circle: its far end lies on the circle (distance 50
        // from the center), not on the tangent line (sqrt(50^2 + 10^2)).
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(10.0, 0.0);
        let bar = b.build();
        let mut b = PathBuilder::new();
        b.add_circle(0.0, 0.0, 50.0);
        let circle = b.build();
        let effect = Path1DEffect::new(bar, 1000.0, 0.0, Path1DStyle::Morph).unwrap();
        let out = effect.apply(&circle).expect("applies");
        let end = *out.points().last().expect("a stamp");
        let r = end.x.hypot(end.y);
        assert!((r - 50.0).abs() < 0.5, "end at radius {r}");
        assert!(
            out.iter().any(|e| matches!(e, PathElement::Quad(..))),
            "lines become quads"
        );
    }

    #[test]
    fn line2d_width_validation_matches_skia() {
        let m = skia_rs_core::Matrix::scale(4.0, 4.0);
        assert!(Line2DEffect::new(Scalar::NAN, m).is_none());
        assert!(Line2DEffect::new(-1.0, m).is_none());
        // Width 0 is a hairline hatch.
        let hairline = Line2DEffect::new(0.0, m).expect("hairline hatch");
        let mut rec = StrokeRec::Fill;
        hairline
            .filter_path(&square(40.0), &mut rec)
            .expect("applies");
        assert_eq!(rec, StrokeRec::Hairline);
    }

    #[test]
    fn path2d_cull_accounts_for_a_stroke_set_by_an_earlier_effect() {
        // In a Sum, Line2D sets the record to Stroke(20) before Path2D runs:
        // a stamp outside the cull can still show its stroke (and miter
        // spikes, at the paint's unknown limit) inside it, so nothing is
        // culled. A stamp 200 units right of the cull survives.
        let line: PathEffectRef =
            Arc::new(Line2DEffect::new(20.0, skia_rs_core::Matrix::scale(50.0, 50.0)).unwrap());
        let tile: PathEffectRef = Arc::new(
            Path2DEffect::new(skia_rs_core::Matrix::scale(4.0, 4.0), square(1.0)).unwrap(),
        );
        let sum = SumEffect::new(line, tile.clone());
        let source = square(400.0);
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut rec = StrokeRec::Fill;
        let culled = sum
            .filter_path_culled(&source, &mut rec, Some(&cull), 1.0)
            .expect("applies");
        // 4-unit cells: the cull's one-cell margin ends near x = 102, but a
        // stamp at x = 106..110 still shows its 20-wide stroke inside.
        assert!(culled.points().iter().any(|p| p.x > 105.0 && p.x < 111.0));
        // x = 302 is a 4-unit cell center, where only stamps (never the
        // Line2D hatch, whose ends sit at odd multiples of 25) have points.
        assert!(culled.points().iter().any(|p| (p.x - 302.0).abs() < 0.01));
    }

    #[test]
    fn effects_reject_non_finite_parameters_like_skia() {
        assert!(DashEffect::new(vec![5.0, 5.0], Scalar::NAN).is_none());
        assert!(DashEffect::new(vec![5.0, 5.0], Scalar::INFINITY).is_none());
        assert!(DashEffect::new(vec![Scalar::INFINITY, 5.0], 0.0).is_none());
        assert!(DiscreteEffect::new(Scalar::NAN, 1.0, 0).is_none());
        assert!(DiscreteEffect::new(1.0, Scalar::INFINITY, 0).is_none());
        assert!(
            DiscreteEffect::new(1e-5, 1.0, 0).is_none(),
            "below nearly-zero"
        );
    }

    #[test]
    fn discrete_segments_are_capped_per_contour() {
        let discrete = DiscreteEffect::new(1e-3, 0.5, 7).unwrap();
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).line_to(1e6, 0.0);
        let out = discrete.apply(&b.build()).expect("applies");
        // 1e9 segments requested; at most MAX_DISCRETE_SEGMENTS emitted.
        let max = usize::try_from(MAX_DISCRETE_SEGMENTS).unwrap() + 1;
        assert!(out.points().len() <= max, "{} points", out.points().len());
    }

    #[test]
    fn perspective_lattices_are_not_culled() {
        // With a perspective lattice matrix the cull's mapped box is
        // meaningless, so culled output equals the unculled output.
        let lattice = skia_rs_core::Matrix {
            values: [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.01, 0.0, 1.0],
        };
        let hatch = Line2DEffect::new(1.0, lattice).unwrap();
        let source = square(300.0);
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 100.0, 100.0);
        let (mut a, mut b) = (StrokeRec::Fill, StrokeRec::Fill);
        let culled = hatch
            .filter_path_culled(&source, &mut a, Some(&cull), 1.0)
            .expect("applies");
        let full = hatch.filter_path(&source, &mut b).expect("applies");
        assert_eq!(culled.points(), full.points());
    }

    #[test]
    fn morph_stamps_respect_the_point_budget() {
        // A 4-line square morphed costs 9 points per stamp, not 5.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .line_to(0.1, 0.0)
            .line_to(0.1, 0.1)
            .line_to(0.0, 0.1)
            .line_to(0.0, 0.0);
        let effect = Path1DEffect::new(b.build(), 0.12, 0.0, Path1DStyle::Morph).unwrap();
        // Six contours of ~91,700 stamps each (under Skia's 100,000 per
        // contour): ~550k stamps, over the budget at 9 points per stamp
        // (~466k) but not at the 5 the budget used to assume (~839k).
        let mut b = PathBuilder::new();
        for i in 0..6 {
            let y = scalar_from_i32(i) * 10.0;
            b.move_to(0.0, y).line_to(11_000.0, y);
        }
        let out = effect.apply(&b.build()).expect("applies");
        assert!(
            out.points().len() <= MAX_PATH1D_POINTS + 9,
            "{}",
            out.points().len()
        );
    }

    #[test]
    fn zero_width_line2d_apply_returns_its_centerlines() {
        let hatch = Line2DEffect::new(0.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap();
        let lines = hatch.apply(&square(40.0)).expect("applies");
        assert!(!lines.is_empty(), "the hairline hatch is not lost");
    }

    #[test]
    fn line2d_keeps_lines_whose_span_ends_just_inside_the_cull() {
        // 20-unit cells; the source covers cells 0..=4 (a 5-cell span whose
        // line runs to x = 110). The cull starts at x = 103, i.e. lattice
        // column 4 after the one-cell outset. Clipping the span to the cull
        // first would leave one cell and drop the line; it must stay.
        let hatch = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(20.0, 20.0)).unwrap();
        let mut b = PathBuilder::new();
        b.add_rect(&skia_rs_core::Rect::new(0.0, 0.0, 110.0, 200.0));
        let cull = skia_rs_core::Rect::new(103.0, 0.0, 300.0, 200.0);
        let mut rec = StrokeRec::Fill;
        let lines = hatch
            .filter_path_culled(&b.build(), &mut rec, Some(&cull), 1.0)
            .expect("applies");
        assert!(
            lines.points().iter().any(|p| p.x > 105.0),
            "hatch reaches into the cull"
        );
    }

    #[test]
    fn sum_recomputes_first_unculled_when_second_sets_a_stroke() {
        // Sum(Path2D, Line2D): Line2D (second) sets Stroke(8) after Path2D
        // (first) culled for a fill. Both draw with the stroke, so Path2D's
        // stamps outside the cull must come back.
        let tile: PathEffectRef = Arc::new(
            Path2DEffect::new(skia_rs_core::Matrix::scale(4.0, 4.0), square(1.0)).unwrap(),
        );
        let line: PathEffectRef =
            Arc::new(Line2DEffect::new(8.0, skia_rs_core::Matrix::scale(50.0, 50.0)).unwrap());
        let sum = SumEffect::new(tile, line);
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut rec = StrokeRec::Fill;
        let out = sum
            .filter_path_culled(&square(400.0), &mut rec, Some(&cull), 1.0)
            .expect("applies");
        assert_eq!(rec, StrokeRec::Stroke(8.0));
        // x = 302: a stamp-only coordinate (4-unit cell center) far outside.
        assert!(out.points().iter().any(|p| (p.x - 302.0).abs() < 0.01));
    }

    #[test]
    fn dashes_flatten_curves_for_the_resolution_scale() {
        // Long dashes on a radius-5 circle: with res_scale 20 the chords
        // must be short enough to look smooth at 20x (sagitta <= 0.5 px).
        let mut b = PathBuilder::new();
        b.add_circle(0.0, 0.0, 5.0);
        let circle = b.build();
        let dash = DashEffect::new(vec![20.0, 1.0], 0.0).unwrap();
        let longest = |res_scale: Scalar| {
            let mut rec = StrokeRec::Stroke(1.0);
            let out = dash
                .filter_path_culled(&circle, &mut rec, None, res_scale)
                .expect("applies");
            let (mut current, mut max) = (Point::zero(), 0.0_f32);
            for e in &out {
                match e {
                    PathElement::Move(p) => current = p,
                    PathElement::Line(p) => {
                        max = max.max(current.distance(&p));
                        current = p;
                    }
                    _ => {}
                }
            }
            max
        };
        assert!(longest(20.0) <= 1.1, "zoomed chords: {}", longest(20.0));
        assert!(
            longest(1.0) > 2.0,
            "unzoomed chords stay coarse: {}",
            longest(1.0)
        );
    }

    #[test]
    fn dashed_long_lines_are_trimmed_to_the_cull_before_counting() {
        // 2e7 long at intervals [2, 2]: 5e6 dashes over the whole line, but
        // only the part near a 100x100 cull counts (SkDashPath's cull_path).
        let dash = DashEffect::new(vec![2.0, 2.0], 0.0).unwrap();
        let mut b = PathBuilder::new();
        b.move_to(-1e7, 50.0).line_to(1e7, 50.0);
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut rec = StrokeRec::Stroke(1.0);
        let out = dash
            .filter_path_culled(&b.build(), &mut rec, Some(&cull), 1.0)
            .expect("dashes the visible part instead of failing");
        assert!(!out.is_empty());
        // Phase kept: a dash still starts on a multiple of the interval.
        let first = out.points()[0].x;
        let periods = first / 4.0;
        assert!(
            (periods - periods.round()).abs() < 1e-3,
            "first dash at {first}"
        );
        // The unculled line still exceeds the cap and fails, as in Skia.
        let mut rec = StrokeRec::Stroke(1.0);
        let mut b = PathBuilder::new();
        b.move_to(-1e7, 50.0).line_to(1e7, 50.0);
        assert!(dash.filter_path(&b.build(), &mut rec).is_none());
    }

    #[test]
    fn discrete_follows_conics_exactly() {
        // Zero deviation reproduces the source: a circle of conics stays at
        // radius 40 (evaluated as quads it bulges to about 44 at 45 degrees).
        let discrete = DiscreteEffect::new(5.0, 0.0, 3).unwrap();
        let mut b = PathBuilder::new();
        b.add_circle(0.0, 0.0, 40.0);
        let out = discrete.apply(&b.build()).expect("applies");
        let max_r = out
            .points()
            .iter()
            .map(|p| p.x.hypot(p.y))
            .fold(0.0, Scalar::max);
        assert!((max_r - 40.0).abs() < 0.05, "max radius {max_r}");
    }

    #[test]
    fn off_screen_dashed_lines_are_skipped() {
        // A long dashed line entirely outside the cull draws nothing and is
        // not dashed along its whole (off-screen) length.
        let dash = DashEffect::new(vec![2.0, 2.0], 0.0).unwrap();
        let mut b = PathBuilder::new();
        b.move_to(0.0, -5000.0).line_to(1e6, -5000.0);
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut rec = StrokeRec::Stroke(1.0);
        let out = dash
            .filter_path_culled(&b.build(), &mut rec, Some(&cull), 1.0)
            .expect("succeeds");
        assert!(out.is_empty(), "{} points", out.points().len());
    }

    #[test]
    fn path1d_limit_matches_the_f32_walk_near_100k_stamps() {
        // advance 0.1 over 9999 units: the estimate (99,990) is under the
        // limit, but 1e5 f32 additions of 0.1 drift short of 9999, so the
        // walk (Skia's governor) needs more than 100,000 stamps and fails.
        // The effect must agree with that walk.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        for i in 1..42 {
            b.line_to(scalar_from_i32(i % 2) * 0.01, scalar_from_i32(i % 3) * 0.01);
        }
        let effect = Path1DEffect::new(b.build(), 0.1, 0.0, Path1DStyle::Translate).unwrap();
        let length = 9999.0_f32;
        let (mut distance, mut stamps) = (effect.initial_offset, 0_u32);
        loop {
            if distance >= length {
                break;
            }
            stamps += 1;
            distance += 0.1;
        }
        let mut line = PathBuilder::new();
        line.move_to(0.0, 0.0).line_to(length, 0.0);
        let result = effect.apply(&line.build());
        assert_eq!(
            result.is_none(),
            stamps > MAX_REASONABLE_ITERATIONS,
            "walk needs {stamps} stamps"
        );
    }

    #[test]
    fn dash_parameters_match_skia_calc_dash_parameters() {
        let params = |intervals: Vec<Scalar>, phase: Scalar| {
            DashEffect::new(intervals, phase).unwrap().dash_parameters()
        };
        // A zero-length "on" interval at the phase is where the dash starts
        // (index 0), not skipped.
        assert_eq!(params(vec![0.0, 10.0], 0.0), (0.0, 0, 0.0));
        // Phase exactly at a non-empty interval's end moves on.
        assert_eq!(params(vec![10.0, 5.0], 10.0), (10.0, 1, 5.0));
        // A phase equal to the total length wraps to 0 (phase >= len is
        // reduced mod len), even with a zero-length tail interval.
        assert_eq!(params(vec![10.0, 0.0], 10.0), (0.0, 0, 10.0));
        // Negative phases are flipped into the pattern.
        assert_eq!(params(vec![10.0, 5.0], -3.0), (12.0, 1, 3.0));
    }

    #[test]
    fn closed_contours_join_the_last_dash_to_the_first() {
        // A closed 10x10 square (perimeter 40) with [8, 3]: 40 = 3 * 11 + 7,
        // so the last dash (33..40) runs into the start corner while \"on\"
        // and must continue the first dash (0..8) instead of ending there.
        let dash = DashEffect::new(vec![8.0, 3.0], 0.0).unwrap();
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(10.0, 10.0)
            .line_to(0.0, 10.0)
            .close();
        let out = dash.apply(&b.build()).expect("applies");
        let moves = out
            .iter()
            .filter(|e| matches!(e, PathElement::Move(_)))
            .count();
        // Dashes: 0..8, 11..19, 22..30, 33..40+0..8 joined: 3 subpaths.
        assert_eq!(moves, 3);
        // The joined dash passes through the start corner as an interior
        // vertex (no subpath starts at the origin).
        let starts: Vec<Point> = out
            .iter()
            .filter_map(|e| match e {
                PathElement::Move(p) => Some(p),
                _ => None,
            })
            .collect();
        assert!(starts.iter().all(|p| p.x.hypot(p.y) > 1e-3), "{starts:?}");
    }

    #[test]
    fn compose_passes_the_cull_to_an_inner_dash() {
        // Skia gives an inner dash the cull, which trims a very long line to
        // the visible part before kMaxDashCount is checked. Without it the
        // dash fails and the outer effect gets the undashed line.
        let dash: PathEffectRef = Arc::new(DashEffect::new(vec![1.0, 1.0], 0.0).unwrap());
        let corner: PathEffectRef = Arc::new(CornerEffect::new(2.0).unwrap());
        let composed = ComposeEffect::new(corner, dash);
        let mut b = PathBuilder::new();
        b.move_to(-5e7, 50.0).line_to(5e7, 50.0);
        let cull = skia_rs_core::Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut rec = StrokeRec::Stroke(1.0);
        let out = composed
            .filter_path_culled(&b.build(), &mut rec, Some(&cull), 1.0)
            .expect("applies");
        let dashes = out
            .iter()
            .filter(|e| matches!(e, PathElement::Move(_)))
            .count();
        assert!(
            dashes > 10,
            "the visible part is dashed ({dashes} subpaths)"
        );

        // 2D effects' culling is not exact, so they still get no cull as an
        // inner effect (checked by compose_does_not_cull_the_inner_effect).
        let hatch = Line2DEffect::new(1.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap();
        assert!(!hatch.cull_is_exact());
    }

    #[test]
    fn singular_2d_matrix_fails_before_the_empty_bounds_check() {
        // Sk2DPathEffect returns false for a non-invertible matrix first, so
        // the source is drawn unchanged even when its bounds are empty.
        let singular = skia_rs_core::Matrix::scale(0.0, 0.0);
        let tile = Path2DEffect::new(singular, square(2.0)).unwrap();
        assert!(tile.apply(&flat_line()).is_none());
        let lines = Line2DEffect::new(1.0, singular).unwrap();
        assert!(lines.apply(&flat_line()).is_none());
    }

    #[test]
    fn trim_of_zero_length_source_draws_nothing() {
        let trim = TrimEffect::new(0.2, 0.8, TrimMode::Normal).unwrap();
        let produced = trim.apply(&zero_length()).expect("trim applies");
        assert!(
            produced.is_empty(),
            "no capped dot for a zero-length source"
        );
    }

    #[test]
    fn filter_path_updates_the_stroke_rec_like_skia() {
        let stamp: PathEffectRef =
            Arc::new(Path1DEffect::new(square(2.0), 5.0, 0.0, Path1DStyle::Translate).unwrap());
        let hatch: PathEffectRef =
            Arc::new(Line2DEffect::new(3.0, skia_rs_core::Matrix::scale(4.0, 4.0)).unwrap());
        let bad_hatch: PathEffectRef =
            Arc::new(Line2DEffect::new(3.0, skia_rs_core::Matrix::scale(0.0, 0.0)).unwrap());
        let dash: PathEffectRef = Arc::new(DashEffect::new(vec![4.0, 2.0], 0.0).unwrap());

        // Path1D switches to fill; Line2D strokes at its own width.
        let mut rec = StrokeRec::Hairline;
        assert!(stamp.filter_path(&flat_line(), &mut rec).is_some());
        assert_eq!(rec, StrokeRec::Fill);
        let mut rec = StrokeRec::Fill;
        assert!(hatch.filter_path(&square(20.0), &mut rec).is_some());
        assert_eq!(rec, StrokeRec::Stroke(3.0));

        // A failed Line2D leaves the record untouched.
        let mut rec = StrokeRec::Fill;
        assert!(bad_hatch.filter_path(&square(20.0), &mut rec).is_none());
        assert_eq!(rec, StrokeRec::Fill);

        // Dash refuses filled geometry and dashes strokes.
        let mut rec = StrokeRec::Fill;
        assert!(dash.filter_path(&flat_line(), &mut rec).is_none());
        let mut rec = StrokeRec::StrokeAndFill(2.0);
        assert!(dash.filter_path(&flat_line(), &mut rec).is_none());
        let mut rec = StrokeRec::Stroke(2.0);
        assert!(dash.filter_path(&flat_line(), &mut rec).is_some());

        // Sum: a failing child must not restyle the other child's output.
        let sum = SumEffect::new(Arc::clone(&stamp), Arc::clone(&bad_hatch));
        let mut rec = StrokeRec::Hairline;
        assert!(sum.filter_path(&flat_line(), &mut rec).is_some());
        assert_eq!(rec, StrokeRec::Fill, "stamps stay filled");

        // Compose: a failing inner effect leaves the outer to run on the
        // source (SkComposePathEffect), instead of failing the whole effect.
        let compose = ComposeEffect::new(Arc::clone(&dash), Arc::clone(&bad_hatch));
        let mut rec = StrokeRec::Stroke(2.0);
        let dashed = compose
            .filter_path(&flat_line(), &mut rec)
            .expect("outer dash still runs on the source");
        assert!(
            dashed.verb_count() > flat_line().verb_count(),
            "source got dashed"
        );
        assert!(compose.apply(&flat_line()).is_some());
    }

    #[test]
    fn effects_still_produce_geometry_for_real_sources() {
        let tile = Path2DEffect::new(skia_rs_core::Matrix::scale(4.0, 4.0), square(2.0)).unwrap();
        assert!(!tile.apply(&square(20.0)).unwrap().is_empty());

        let stamp = Path1DEffect::new(square(2.0), 5.0, 0.0, Path1DStyle::Translate).unwrap();
        assert!(!stamp.apply(&flat_line()).unwrap().is_empty());
    }

    #[test]
    fn test_dash_effect() {
        let dash = DashEffect::new(vec![10.0, 5.0], 0.0).unwrap();
        assert_eq!(dash.intervals().len(), 2);
        #[allow(
            clippy::float_cmp,
            reason = "exact test assertion, value round-trips a literal"
        )]
        {
            assert_eq!(dash.phase(), 0.0);
        }
    }

    #[test]
    fn test_dash_odd_intervals() {
        // SkDashPathEffect::Make rejects an odd interval count.
        assert!(DashEffect::new(vec![10.0, 5.0, 3.0], 0.0).is_none());
        assert!(DashEffect::new(vec![10.0], 0.0).is_none());
    }

    #[test]
    fn test_dash_closing_segment_is_dashed_continuously() {
        use crate::PathMeasure;
        // A closed 10x10 square (perimeter 40) dashed [5 on, 5 off] must be
        // dashed continuously across the closing edge, giving 4 on-segments of
        // length 5 each => total drawn length 20.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.line_to(10.0, 10.0);
        b.line_to(0.0, 10.0);
        b.close();
        let square = b.build();

        let effect = DashEffect::new(vec![5.0, 5.0], 0.0).unwrap();
        let dashed = effect.apply(&square).unwrap();
        let drawn = PathMeasure::new(&dashed).length();
        assert!(
            (drawn - 20.0).abs() < 1.0,
            "closing edge must be dashed continuously: expected ~20 drawn units, got {drawn}"
        );
    }

    #[test]
    fn test_dash_effect_on_curve_produces_dashes_along_curve() {
        // A long quadratic curve with a 10-on/10-off dash.
        // Dash transitions should occur along the curve, not just at endpoints.
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.quad_to(50.0, 100.0, 100.0, 0.0);
        let path = builder.build();

        let effect = DashEffect::new(vec![10.0, 10.0], 0.0).unwrap();
        let result = effect.apply(&path).unwrap();

        // Each dash starts with a Move. On a curved path with length ~140,
        // we should get multiple dashes (not just one at start and one at end).
        let moves = result
            .iter()
            .filter(|e| matches!(e, PathElement::Move(_)))
            .count();
        assert!(
            moves >= 3,
            "Expected at least 3 dashes on curved path (curve length ~140, dash period 20), got {moves}"
        );
    }

    #[test]
    fn test_corner_effect() {
        let corner = CornerEffect::new(5.0).unwrap();
        #[allow(
            clippy::float_cmp,
            reason = "exact test assertion, value round-trips a literal"
        )]
        {
            assert_eq!(corner.radius(), 5.0);
        }
    }

    #[test]
    fn test_corner_effect_rounds_closed_contour_joint() {
        // A closed square with all four corners (including the closing joint
        // between the last and first segments) must be rounded => 4 quads.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(40.0, 0.0);
        b.line_to(40.0, 40.0);
        b.line_to(0.0, 40.0);
        b.line_to(0.0, 0.0);
        b.close();
        let square = b.build();

        let corner = CornerEffect::new(5.0).unwrap();
        let rounded = corner.apply(&square).unwrap();
        let quads = rounded
            .iter()
            .filter(|e| matches!(e, PathElement::Quad(_, _)))
            .count();
        assert_eq!(
            quads, 4,
            "all four corners of a closed contour must be rounded"
        );
        assert!(rounded.is_closed(), "rounded closed contour stays closed");
    }

    #[test]
    fn test_discrete_effect() {
        let discrete = DiscreteEffect::new(10.0, 5.0, 42).unwrap();
        #[allow(
            clippy::float_cmp,
            reason = "exact test assertion, values round-trip literals"
        )]
        {
            assert_eq!(discrete.seg_length(), 10.0);
            assert_eq!(discrete.deviation(), 5.0);
        }
    }

    #[test]
    fn test_make_functions() {
        assert!(make_dash(vec![10.0, 5.0], 0.0).is_some());
        assert!(make_corner(5.0).is_some());
        assert!(make_discrete(10.0, 5.0, 0).is_some());
        assert!(make_trim(0.0, 1.0, TrimMode::Normal).is_some());
    }

    #[test]
    fn test_compose_effect() {
        let dash = make_dash(vec![10.0, 5.0], 0.0).unwrap();
        let corner = make_corner(5.0).unwrap();
        let composed = make_compose(dash, corner);
        assert_eq!(composed.effect_kind(), PathEffectKind::Compose);
    }

    #[test]
    fn test_trim_effect_normal_extracts_subsegment() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();

        let effect = TrimEffect::new(0.25, 0.75, TrimMode::Normal).unwrap();
        let result = effect.apply(&path).unwrap();

        let bounds = result.bounds();
        assert!(
            (bounds.left - 25.0).abs() < 0.5,
            "Trimmed left should be ~25, got {}",
            bounds.left
        );
        assert!(
            (bounds.right - 75.0).abs() < 0.5,
            "Trimmed right should be ~75, got {}",
            bounds.right
        );
    }

    #[test]
    fn test_trim_effect_inverted_keeps_outside_range() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();

        let effect = TrimEffect::new(0.25, 0.75, TrimMode::Inverted).unwrap();
        let result = effect.apply(&path).unwrap();

        let bounds = result.bounds();
        assert!((bounds.left - 0.0).abs() < 0.5);
        assert!((bounds.right - 100.0).abs() < 0.5);
        // Should have at least 2 contours (before 25 and after 75)
        assert_eq!(
            result.contour_count(),
            2,
            "Inverted trim should produce two contours"
        );
    }
}
