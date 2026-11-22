//! Path effects (dash, corner, discrete, etc.).
//!
//! Path effects modify how a path is stroked or filled. They can be applied
//! to create dashed lines, rounded corners, jittery edges, and more.

use crate::{
    Path, PathBuilder, PathElement,
    flatten::{flatten_conic_adaptive, flatten_cubic_adaptive, flatten_quad_adaptive},
};
use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, scalar_from_i32};
use skia_rs_core::{Point, Scalar};
use std::sync::Arc;

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
    /// `intervals` must have an even number of entries (on/off pairs).
    /// If odd, the pattern is duplicated to make it even.
    #[must_use]
    pub fn new(intervals: Vec<Scalar>, phase: Scalar) -> Option<Self> {
        if intervals.is_empty() {
            return None;
        }

        // Ensure even number of intervals
        let intervals = if intervals.len().is_multiple_of(2) {
            intervals
        } else {
            let mut doubled = intervals.clone();
            doubled.extend(intervals.iter().copied());
            doubled
        };

        // Validate intervals are positive
        for &interval in &intervals {
            if interval < 0.0 {
                return None;
            }
        }

        let interval_sum: Scalar = intervals.iter().sum();
        if interval_sum <= 0.0 {
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
    fn dash_line(
        &self,
        builder: &mut PathBuilder,
        from: Point,
        to: Point,
        is_on: &mut bool,
        interval_idx: &mut usize,
        remaining_in_interval: &mut Scalar,
    ) {
        let segment_length = from.distance(&to);
        if segment_length <= 0.0 {
            return;
        }
        let mut t = 0.0;
        loop {
            if t >= 1.0 {
                break;
            }
            let remaining_segment = segment_length * (1.0 - t);

            if *remaining_in_interval >= remaining_segment {
                if *is_on {
                    builder.line_to(to.x, to.y);
                }
                *remaining_in_interval -= remaining_segment;
                t = 1.0;
            } else {
                let dt = *remaining_in_interval / segment_length;
                let mid = Point::new(
                    (to.x - from.x).mul_add(t + dt, from.x),
                    (to.y - from.y).mul_add(t + dt, from.y),
                );
                if *is_on {
                    builder.line_to(mid.x, mid.y);
                }
                *interval_idx = (*interval_idx + 1) % self.intervals.len();
                *is_on = (*interval_idx).is_multiple_of(2);
                *remaining_in_interval = self.intervals[*interval_idx];
                if *is_on {
                    builder.move_to(mid.x, mid.y);
                }
                t += dt;
            }
        }
    }
}

impl PathEffect for DashEffect {
    #[allow(
        clippy::too_many_lines,
        reason = "faithful port of SkDashPath::InternalFilter's per-verb dash state machine"
    )]
    fn apply(&self, path: &Path) -> Option<Path> {
        if path.is_empty() {
            return None;
        }

        // Pre-flatten curves to lines so dash logic can transition mid-curve.
        let flattened = flatten_path_to_lines(path, 0.5);
        let path = &flattened;

        let mut builder = PathBuilder::new();
        let mut current_pos = Point::zero();
        let mut contour_start = Point::zero();
        #[allow(unused_assignments)]
        let mut _distance_along_contour = 0.0_f32;

        // Adjust phase to be within the interval sum
        let mut phase = self.phase % self.interval_sum;
        if phase < 0.0 {
            phase += self.interval_sum;
        }

        // Find starting interval index and offset
        let (mut interval_idx, interval_offset) = {
            let mut accumulated = 0.0;
            let mut idx = 0;
            loop {
                if accumulated + self.intervals[idx] > phase {
                    break;
                }
                accumulated += self.intervals[idx];
                idx = (idx + 1) % self.intervals.len();
            }
            (idx, phase - accumulated)
        };

        let mut is_on = interval_idx % 2 == 0;
        let mut remaining_in_interval = self.intervals[interval_idx] - interval_offset;

        for element in path {
            match element {
                PathElement::Move(p) => {
                    current_pos = p;
                    contour_start = p;
                    _distance_along_contour = 0.0;
                    // Reset dash state for new contour
                    let mut accumulated = 0.0;
                    interval_idx = 0;
                    loop {
                        if accumulated + self.intervals[interval_idx] > phase {
                            break;
                        }
                        accumulated += self.intervals[interval_idx];
                        interval_idx = (interval_idx + 1) % self.intervals.len();
                    }
                    is_on = interval_idx % 2 == 0;
                    remaining_in_interval = self.intervals[interval_idx] - (phase - accumulated);
                    if is_on {
                        builder.move_to(p.x, p.y);
                    }
                }
                PathElement::Line(end) => {
                    self.dash_line(
                        &mut builder,
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
                            &mut builder,
                            current_pos,
                            contour_start,
                            &mut is_on,
                            &mut interval_idx,
                            &mut remaining_in_interval,
                        );
                    }
                    current_pos = contour_start;
                }
                // For curves, we approximate with lines (simplified)
                PathElement::Quad(ctrl, end) => {
                    // Subdivide and treat as lines
                    let steps: i32 = 8;
                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let p = quadratic_point(current_pos, ctrl, end, t);
                        if is_on {
                            builder.line_to(p.x, p.y);
                        }
                    }
                    current_pos = end;
                }
                PathElement::Conic(ctrl, end, _weight) => {
                    // Approximate as quad
                    let steps: i32 = 8;
                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let p = quadratic_point(current_pos, ctrl, end, t);
                        if is_on {
                            builder.line_to(p.x, p.y);
                        }
                    }
                    current_pos = end;
                }
                PathElement::Cubic(ctrl1, ctrl2, end) => {
                    let steps: i32 = 12;
                    for i in 1..=steps {
                        let t = scalar_from_i32(i) / scalar_from_i32(steps);
                        let p = cubic_point(current_pos, ctrl1, ctrl2, end, t);
                        if is_on {
                            builder.line_to(p.x, p.y);
                        }
                    }
                    current_pos = end;
                }
            }
        }

        Some(builder.build())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Dash
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
    pub fn new(seg_length: Scalar, deviation: Scalar, seed: u32) -> Option<Self> {
        if seg_length <= 0.0 {
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

impl PathEffect for DiscreteEffect {
    fn apply(&self, path: &Path) -> Option<Path> {
        if path.is_empty() {
            return None;
        }

        let mut builder = PathBuilder::new();
        let mut current_pos = Point::zero();
        let mut seed = self.seed;

        for element in path {
            match element {
                PathElement::Move(p) => {
                    let dx = Self::random(seed) * self.deviation;
                    seed = seed.wrapping_add(1);
                    let dy = Self::random(seed) * self.deviation;
                    seed = seed.wrapping_add(1);
                    builder.move_to(p.x + dx, p.y + dy);
                    current_pos = p;
                }
                PathElement::Line(end) => {
                    let length = current_pos.distance(&end);
                    let num_segments = ceil_to_i32(length / self.seg_length).max(1);

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
                    let steps =
                        ceil_to_i32(quadratic_length(current_pos, ctrl, end) / self.seg_length)
                            .max(4);

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
                PathElement::Conic(ctrl, end, _w) => {
                    let steps =
                        ceil_to_i32(quadratic_length(current_pos, ctrl, end) / self.seg_length)
                            .max(4);

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
                PathElement::Cubic(ctrl1, ctrl2, end) => {
                    let steps =
                        ceil_to_i32(cubic_length(current_pos, ctrl1, ctrl2, end) / self.seg_length)
                            .max(4);

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
            return Some(path.clone());
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
    fn apply(&self, path: &Path) -> Option<Path> {
        let intermediate = self.inner.apply(path)?;
        self.outer.apply(&intermediate)
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
    fn apply(&self, path: &Path) -> Option<Path> {
        let path1 = self.first.apply(path);
        let path2 = self.second.apply(path);

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

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Sum
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
    pub fn new(path: Path, advance: Scalar, phase: Scalar, style: Path1DStyle) -> Option<Self> {
        if advance <= 0.0 || path.is_empty() {
            return None;
        }
        Some(Self {
            path,
            advance,
            phase,
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

impl PathEffect for Path1DEffect {
    fn apply(&self, src_path: &Path) -> Option<Path> {
        use crate::PathMeasure;

        if src_path.is_empty() {
            return None;
        }

        let mut builder = PathBuilder::new();
        let measure = PathMeasure::new(src_path);
        let length = measure.length();

        if length <= 0.0 {
            // Sk1DPathEffect still succeeds here: no stamps, nothing drawn.
            return Some(Path::new());
        }

        let mut distance = self.phase;
        loop {
            if distance >= length {
                break;
            }
            // Get position and tangent at this distance
            let pos = measure.get_point_at(distance);
            let tangent = measure.get_tangent_at(distance);

            if let (Some(pos), Some(tangent)) = (pos, tangent) {
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

                // Transform and add the stamped path
                let transformed = self.path.transformed(&transform);
                builder.add_path(&transformed);
            }
            distance += self.advance;
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

impl PathEffect for Path2DEffect {
    fn apply(&self, src_path: &Path) -> Option<Path> {
        if src_path.is_empty() {
            return None;
        }

        let bounds = src_path.bounds();
        if bounds.is_empty() {
            // Sk2DPathEffect::filterPath succeeds with an empty result here
            // (no tiles), so nothing is drawn rather than the source.
            return Some(Path::new());
        }
        let mut builder = PathBuilder::new();

        // Compute the inverse matrix to find tile positions
        let inverse = self.matrix.invert()?;

        // Transform bounds to tile space
        let transformed_bounds = inverse.map_rect(&bounds);

        // Compute tile range
        let start_x = floor_to_i32(transformed_bounds.left) - 1;
        let end_x = ceil_to_i32(transformed_bounds.right) + 1;
        let start_y = floor_to_i32(transformed_bounds.top) - 1;
        let end_y = ceil_to_i32(transformed_bounds.bottom) + 1;

        // Limit iterations for safety
        let max_tiles = 1000;
        let mut tile_count = 0;

        for y in start_y..=end_y {
            for x in start_x..=end_x {
                if tile_count >= max_tiles {
                    break;
                }

                // Transform tile to world space
                let tile_offset =
                    skia_rs_core::Matrix::translate(scalar_from_i32(x), scalar_from_i32(y));
                let tile_matrix = self.matrix.concat(&tile_offset);
                let transformed_path = self.path.transformed(&tile_matrix);

                builder.add_path(&transformed_path);
                tile_count += 1;
            }
        }

        Some(builder.build())
    }

    fn effect_kind(&self) -> PathEffectKind {
        PathEffectKind::Path2D
    }
}

// =============================================================================
// Line2D Effect
// =============================================================================

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
    pub fn new(width: Scalar, matrix: skia_rs_core::Matrix) -> Option<Self> {
        if width <= 0.0 {
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

impl PathEffect for Line2DEffect {
    fn apply(&self, src_path: &Path) -> Option<Path> {
        if src_path.is_empty() {
            return None;
        }

        let bounds = src_path.bounds();
        if bounds.is_empty() {
            // Sk2DPathEffect::filterPath succeeds with an empty result here
            // (no tiles), so nothing is drawn rather than the source.
            return Some(Path::new());
        }
        let mut builder = PathBuilder::new();

        // Compute the inverse matrix
        let inverse = self.matrix.invert()?;

        // Transform bounds to line space
        let transformed_bounds = inverse.map_rect(&bounds);

        // Generate horizontal lines in transformed space
        let start_y = floor_to_i32(transformed_bounds.top) - 1;
        let end_y = ceil_to_i32(transformed_bounds.bottom) + 1;

        // Limit iterations for safety
        let max_lines = 500;

        for (line_count, y) in (start_y..=end_y).enumerate() {
            if line_count >= max_lines {
                break;
            }

            // Create a line from left to right
            let y_pos = scalar_from_i32(y);
            let p0 = Point::new(transformed_bounds.left - 10.0, y_pos);
            let p1 = Point::new(transformed_bounds.right + 10.0, y_pos);

            // Transform back to world space
            let world_p0 = self.matrix.map_point(p0);
            let world_p1 = self.matrix.map_point(p1);

            // Create a rectangle for the line (with width)
            let half_width = self.width / 2.0;
            let dx = world_p1.x - world_p0.x;
            let dy = world_p1.y - world_p0.y;
            let len = dx.hypot(dy);

            if len > 0.0 {
                let nx = -dy / len * half_width;
                let ny = dx / len * half_width;

                builder.move_to(world_p0.x + nx, world_p0.y + ny);
                builder.line_to(world_p1.x + nx, world_p1.y + ny);
                builder.line_to(world_p1.x - nx, world_p1.y - ny);
                builder.line_to(world_p0.x - nx, world_p0.y - ny);
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
        let dash = DashEffect::new(vec![10.0, 5.0, 3.0], 0.0).unwrap();
        // Should be doubled to make even
        assert_eq!(dash.intervals().len(), 6);
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
