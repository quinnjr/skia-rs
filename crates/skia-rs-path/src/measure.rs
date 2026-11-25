//! Path measurement and traversal.

use crate::flatten::{flatten_conic_adaptive, flatten_cubic_adaptive, flatten_quad_adaptive};
use crate::{Path, PathElement};
use skia_rs_core::{Matrix, Point, Scalar};

/// `a + (b - a) * t`, unfused like Skia's interpolation (`mul_add` rounds
/// once and differs from Skia in the last bit).
#[inline]
#[allow(
    clippy::suboptimal_flops,
    reason = "unfused like Skia; mul_add rounds differently"
)]
pub(crate) fn lerp(a: Scalar, b: Scalar, t: Scalar) -> Scalar {
    (b - a) * t + a
}

/// Tolerance used when flattening curves for length measurement.
const FLATTEN_TOLERANCE: Scalar = 0.25;

/// A line segment with cumulative length up to its endpoint.
#[derive(Debug, Clone)]
struct Segment {
    start: Point,
    end: Point,
    length: Scalar,
    /// Cumulative length within the contour up to and including this segment.
    cumulative: Scalar,
}

#[derive(Debug)]
struct Contour {
    segments: Vec<Segment>,
    length: Scalar,
}

/// Measures the length of a path and allows querying points along it.
#[derive(Debug)]
pub struct PathMeasure {
    contours: Vec<Contour>,
    contour_lengths: Vec<Scalar>,
    total_length: Scalar,
}

impl PathMeasure {
    /// Create a new path measure.
    pub fn new(path: &Path) -> Self {
        let mut measure = Self {
            contours: Vec::new(),
            contour_lengths: Vec::new(),
            total_length: 0.0,
        };
        measure.compute_lengths(path);
        measure
    }

    /// Get the total length of the path.
    #[inline]
    #[must_use]
    pub const fn length(&self) -> Scalar {
        self.total_length
    }

    /// Get the number of contours.
    #[inline]
    #[must_use]
    pub const fn contour_count(&self) -> usize {
        self.contour_lengths.len()
    }

    /// Get the length of a specific contour.
    #[must_use]
    pub fn contour_length(&self, index: usize) -> Option<Scalar> {
        self.contour_lengths.get(index).copied()
    }

    /// Get a point at a distance along the path.
    ///
    /// The distance is pinned to `[0, length]` (like `SkContourMeasure::getPosTan`);
    /// `None` is only returned for NaN input or an empty path.
    #[must_use]
    pub fn get_point_at(&self, distance: Scalar) -> Option<Point> {
        if distance.is_nan() || self.total_length <= 0.0 {
            return None;
        }
        let distance = distance.clamp(0.0, self.total_length);
        let (contour, offset) = self.locate(distance)?;
        let seg = Self::segment_at(contour, offset);
        let seg_start_cum = seg.cumulative - seg.length;
        let t = if seg.length > 0.0 {
            ((offset - seg_start_cum) / seg.length).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some(Point::new(
            lerp(seg.start.x, seg.end.x, t),
            lerp(seg.start.y, seg.end.y, t),
        ))
    }

    /// Position and unit tangent at `distance` along contour `index` alone
    /// (pinned to `[0, contour length]`), like `SkContourMeasure::getPosTan`
    /// on that contour. `None` for a NaN distance or a missing contour.
    ///
    /// # Examples
    /// ```
    /// use skia_rs_path::{PathBuilder, PathMeasure};
    ///
    /// let mut b = PathBuilder::new();
    /// b.move_to(0.0, 0.0).line_to(10.0, 0.0); // contour 0
    /// b.move_to(0.0, 5.0).line_to(0.0, 25.0); // contour 1
    /// let measure = PathMeasure::new(&b.build());
    ///
    /// // Distances are measured within the contour, from its start.
    /// let (pos, tangent) = measure.get_pos_tan_in_contour(1, 5.0).unwrap();
    /// assert_eq!((pos.x, pos.y), (0.0, 10.0));
    /// assert_eq!((tangent.x, tangent.y), (0.0, 1.0));
    ///
    /// // Past the end is pinned to it; a missing contour is `None`.
    /// let (end, _) = measure.get_pos_tan_in_contour(0, 99.0).unwrap();
    /// assert_eq!((end.x, end.y), (10.0, 0.0));
    /// assert!(measure.get_pos_tan_in_contour(2, 0.0).is_none());
    /// ```
    #[must_use]
    pub fn get_pos_tan_in_contour(&self, index: usize, distance: Scalar) -> Option<(Point, Point)> {
        let contour = self.contours.get(index)?;
        if distance.is_nan() || contour.length <= 0.0 {
            return None;
        }
        let offset = distance.clamp(0.0, contour.length);
        let seg = Self::segment_at(contour, offset);
        let seg_start_cum = seg.cumulative - seg.length;
        let t = if seg.length > 0.0 {
            ((offset - seg_start_cum) / seg.length).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (dx, dy) = (seg.end.x - seg.start.x, seg.end.y - seg.start.y);
        let pos = Point::new(
            lerp(seg.start.x, seg.end.x, t),
            lerp(seg.start.y, seg.end.y, t),
        );
        let len = dx.hypot(dy);
        let tangent = if len > 0.0 {
            Point::new(dx / len, dy / len)
        } else {
            Point::new(1.0, 0.0)
        };
        Some((pos, tangent))
    }

    /// Get the tangent (unit direction vector) at a distance along the path.
    ///
    /// The distance is pinned to `[0, length]` (like `SkContourMeasure::getPosTan`);
    /// `None` is only returned for NaN input or an empty path.
    #[must_use]
    pub fn get_tangent_at(&self, distance: Scalar) -> Option<Point> {
        if distance.is_nan() || self.total_length <= 0.0 {
            return None;
        }
        let distance = distance.clamp(0.0, self.total_length);
        let (contour, offset) = self.locate(distance)?;
        let seg = Self::segment_at(contour, offset);
        let dx = seg.end.x - seg.start.x;
        let dy = seg.end.y - seg.start.y;
        let len = dx.hypot(dy);
        if len > 0.0 {
            Some(Point::new(dx / len, dy / len))
        } else {
            Some(Point::new(1.0, 0.0))
        }
    }

    /// Get the transformation matrix at a distance along the path.
    ///
    /// The returned matrix maps the local origin to the position at the
    /// given distance, and the local x-axis to the path's tangent
    /// direction at that distance. Useful for placing text or stamps
    /// along a path.
    #[must_use]
    pub fn get_matrix_at(&self, distance: Scalar) -> Option<Matrix> {
        let position = self.get_point_at(distance)?;
        let tangent = self.get_tangent_at(distance)?;
        Some(Matrix {
            values: [
                tangent.x, -tangent.y, position.x, tangent.y, tangent.x, position.y, 0.0, 0.0, 1.0,
            ],
        })
    }

    /// Get a segment of the path between the given distances.
    ///
    /// Returns a new Path containing the portion from `start` to `end`,
    /// constructed from line segments (curves in the source path are
    /// flattened during length computation).
    #[must_use]
    pub fn get_segment(&self, start: Scalar, end: Scalar) -> Option<Path> {
        // Pin start/stop into the legal range like SkContourMeasure::getSegment:
        // clamp start up to 0 and stop down to length; reject only NaN or an
        // inverted (start > stop) range.
        let start = start.max(0.0);
        let end = end.min(self.total_length);
        // Reject NaN or an inverted (start >= end) range. `partial_cmp` is used
        // instead of negating `start < end` so NaN (which fails every ordered
        // comparison) is handled explicitly rather than via De Morgan negation.
        if start.partial_cmp(&end) != Some(std::cmp::Ordering::Less) {
            return None;
        }

        let mut builder = crate::PathBuilder::new();
        let mut acc = 0.0;
        let mut started_any = false;

        for contour in &self.contours {
            let c_start = acc;
            let c_end = acc + contour.length;
            acc = c_end;

            if c_end <= start {
                continue;
            }
            if c_start >= end {
                break;
            }

            let local_seg_start = (start.max(c_start) - c_start).max(0.0);
            let local_seg_end = (end.min(c_end) - c_start).min(contour.length);

            let first_pt = interpolate_at(contour, local_seg_start);
            builder.move_to(first_pt.x, first_pt.y);
            started_any = true;

            for seg in &contour.segments {
                let s_start = seg.cumulative - seg.length;
                let s_end = seg.cumulative;
                if s_end <= local_seg_start {
                    continue;
                }
                if s_start >= local_seg_end {
                    break;
                }
                let t1 = if s_end > local_seg_end {
                    ((local_seg_end - s_start) / seg.length).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let p = Point::new(
                    lerp(seg.start.x, seg.end.x, t1),
                    lerp(seg.start.y, seg.end.y, t1),
                );
                builder.line_to(p.x, p.y);
            }
        }

        if started_any {
            Some(builder.build())
        } else {
            None
        }
    }

    /// Locate the contour containing `distance` and return the offset within it.
    fn locate(&self, distance: Scalar) -> Option<(&Contour, Scalar)> {
        let mut remaining = distance;
        for c in &self.contours {
            if remaining <= c.length {
                return Some((c, remaining));
            }
            remaining -= c.length;
        }
        // At exact total_length, return last contour at its end
        self.contours.last().map(|c| (c, c.length))
    }

    /// Find the segment within a contour that contains the given offset.
    fn segment_at(contour: &Contour, offset: Scalar) -> &Segment {
        let idx = match contour.segments.binary_search_by(|s| {
            s.cumulative
                .partial_cmp(&offset)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => i.min(contour.segments.len().saturating_sub(1)),
        };
        &contour.segments[idx]
    }

    #[allow(
        clippy::too_many_lines,
        reason = "faithful port of SkContourMeasureIter's per-verb contour length/segment accumulation"
    )]
    fn compute_lengths(&mut self, path: &Path) {
        let mut current_contour: Option<Contour> = None;
        let mut current_pt = Point::new(0.0, 0.0);
        let mut contour_start = Point::new(0.0, 0.0);
        let mut points: Vec<Point> = Vec::with_capacity(16);

        let push_segment = |contour: &mut Contour, start: Point, end: Point| {
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let len = dx.hypot(dy);
            if len > 0.0 {
                contour.length += len;
                contour.segments.push(Segment {
                    start,
                    end,
                    length: len,
                    cumulative: contour.length,
                });
            }
        };

        for elem in path {
            match elem {
                PathElement::Move(p) => {
                    if let Some(c) = current_contour.take()
                        && c.length > 0.0
                    {
                        self.contour_lengths.push(c.length);
                        self.total_length += c.length;
                        self.contours.push(c);
                    }
                    current_contour = Some(Contour {
                        segments: Vec::new(),
                        length: 0.0,
                    });
                    current_pt = p;
                    contour_start = p;
                }
                PathElement::Line(p) => {
                    if let Some(c) = current_contour.as_mut() {
                        push_segment(c, current_pt, p);
                    }
                    current_pt = p;
                }
                PathElement::Quad(ctrl, end) => {
                    if let Some(c) = current_contour.as_mut() {
                        points.clear();
                        flatten_quad_adaptive(
                            &mut points,
                            current_pt,
                            ctrl,
                            end,
                            FLATTEN_TOLERANCE,
                        );
                        let mut prev = current_pt;
                        for p in &points {
                            push_segment(c, prev, *p);
                            prev = *p;
                        }
                    }
                    current_pt = end;
                }
                PathElement::Cubic(c1, c2, end) => {
                    if let Some(c) = current_contour.as_mut() {
                        points.clear();
                        flatten_cubic_adaptive(
                            &mut points,
                            current_pt,
                            c1,
                            c2,
                            end,
                            FLATTEN_TOLERANCE,
                        );
                        let mut prev = current_pt;
                        for p in &points {
                            push_segment(c, prev, *p);
                            prev = *p;
                        }
                    }
                    current_pt = end;
                }
                PathElement::Conic(ctrl, end, w) => {
                    if let Some(c) = current_contour.as_mut() {
                        points.clear();
                        flatten_conic_adaptive(
                            &mut points,
                            current_pt,
                            ctrl,
                            end,
                            w,
                            FLATTEN_TOLERANCE,
                        );
                        let mut prev = current_pt;
                        for p in &points {
                            push_segment(c, prev, *p);
                            prev = *p;
                        }
                    }
                    current_pt = end;
                }
                PathElement::Close => {
                    if let Some(c) = current_contour.as_mut()
                        && current_pt != contour_start
                    {
                        push_segment(c, current_pt, contour_start);
                    }
                    current_pt = contour_start;
                }
            }
        }

        if let Some(c) = current_contour.take()
            && c.length > 0.0
        {
            self.contour_lengths.push(c.length);
            self.total_length += c.length;
            self.contours.push(c);
        }
    }
}

fn interpolate_at(contour: &Contour, offset: Scalar) -> Point {
    let seg = PathMeasure::segment_at(contour, offset);
    let seg_start_cum = seg.cumulative - seg.length;
    let t = if seg.length > 0.0 {
        ((offset - seg_start_cum) / seg.length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Point::new(
        lerp(seg.start.x, seg.end.x, t),
        lerp(seg.start.y, seg.end.y, t),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PathBuilder;

    #[test]
    fn test_path_measure_empty_path() {
        let path = PathBuilder::new().build();
        let measure = PathMeasure::new(&path);
        #[allow(
            clippy::float_cmp,
            reason = "exact test assertion, value round-trips a literal"
        )]
        {
            assert_eq!(measure.length(), 0.0);
        }
        assert_eq!(measure.contour_count(), 0);
    }

    #[test]
    fn test_path_measure_single_line() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!((measure.length() - 100.0).abs() < 0.01);
        assert_eq!(measure.contour_count(), 1);
    }

    #[test]
    fn test_get_point_at_pins_out_of_range() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        // Negative distance pins to the start.
        let p = measure.get_point_at(-10.0).expect("pinned start");
        assert!((p.x - 0.0).abs() < 0.01 && (p.y - 0.0).abs() < 0.01);
        // Distance past the end pins to the end.
        let p = measure.get_point_at(1000.0).expect("pinned end");
        assert!((p.x - 100.0).abs() < 0.01 && (p.y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_get_tangent_at_pins_out_of_range() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!(measure.get_tangent_at(-5.0).is_some());
        assert!(measure.get_tangent_at(500.0).is_some());
    }

    #[test]
    fn test_get_segment_pins_start_stop() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        // Out-of-range start/stop are clamped rather than rejected.
        let seg = measure.get_segment(-10.0, 1000.0).expect("clamped segment");
        assert!(!seg.is_empty());
    }

    #[test]
    fn test_path_measure_diagonal_line() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(3.0, 4.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!((measure.length() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_path_measure_multiple_lines() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0);
        builder.line_to(10.0, 10.0);
        builder.line_to(0.0, 10.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!((measure.length() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_path_measure_quadratic_curve() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.quad_to(50.0, 0.0, 100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!((measure.length() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_path_measure_multi_contour() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0);
        builder.move_to(20.0, 0.0);
        builder.line_to(50.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!((measure.length() - 40.0).abs() < 0.01);
        assert_eq!(measure.contour_count(), 2);
        assert!((measure.contour_length(0).unwrap() - 10.0).abs() < 0.01);
        assert!((measure.contour_length(1).unwrap() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_path_measure_closed_contour() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0);
        builder.line_to(10.0, 10.0);
        builder.line_to(0.0, 10.0);
        builder.close();
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        // Close adds the segment from (0,10) back to (0,0), length 10
        assert!((measure.length() - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_get_point_at_start() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let p = measure.get_point_at(0.0).unwrap();
        assert!((p.x - 0.0).abs() < 0.01);
        assert!((p.y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_get_point_at_midpoint() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let p = measure.get_point_at(50.0).unwrap();
        assert!((p.x - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_get_point_at_end() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let p = measure.get_point_at(100.0).unwrap();
        assert!((p.x - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_get_point_at_out_of_range() {
        // Skia pins out-of-range distances to [0, length] and returns a point;
        // only NaN / empty paths yield None.
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let start = measure.get_point_at(-1.0).expect("pinned to start");
        assert!((start.x - 0.0).abs() < 0.01);
        let end = measure.get_point_at(101.0).expect("pinned to end");
        assert!((end.x - 100.0).abs() < 0.01);
        assert!(measure.get_point_at(Scalar::NAN).is_none());
    }

    #[test]
    fn test_get_point_at_multi_contour() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0); // contour 1: 0..10
        builder.move_to(20.0, 0.0);
        builder.line_to(50.0, 0.0); // contour 2: 10..40
        let path = builder.build();
        let measure = PathMeasure::new(&path);

        let p = measure.get_point_at(5.0).unwrap();
        assert!((p.x - 5.0).abs() < 0.01);

        let p = measure.get_point_at(25.0).unwrap();
        assert!((p.x - 35.0).abs() < 0.01);
    }

    #[test]
    fn test_get_tangent_horizontal_line() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let t = measure.get_tangent_at(50.0).unwrap();
        assert!((t.x - 1.0).abs() < 0.01);
        assert!((t.y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_get_tangent_diagonal() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(3.0, 4.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let t = measure.get_tangent_at(2.5).unwrap();
        assert!((t.x - 0.6).abs() < 0.01);
        assert!((t.y - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_get_matrix_horizontal_line() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        let m = measure.get_matrix_at(50.0).unwrap();
        let mapped = m.map_point(Point::new(0.0, 0.0));
        assert!((mapped.x - 50.0).abs() < 0.01);
        assert!((mapped.y - 0.0).abs() < 0.01);
        let mapped_tangent = m.map_point(Point::new(1.0, 0.0));
        assert!((mapped_tangent.x - 51.0).abs() < 0.01);
    }

    #[test]
    fn test_get_segment_subset_of_line() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);

        let segment = measure.get_segment(25.0, 75.0).unwrap();
        let bounds = segment.bounds();
        assert!((bounds.left - 25.0).abs() < 0.5);
        assert!((bounds.right - 75.0).abs() < 0.5);
    }

    #[test]
    fn test_get_segment_invalid_range() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);

        // Inverted range is still rejected.
        assert!(measure.get_segment(75.0, 25.0).is_none());
        // Out-of-range endpoints are pinned like SkContourMeasure::getSegment.
        assert!(measure.get_segment(-1.0, 50.0).is_some());
        assert!(measure.get_segment(50.0, 200.0).is_some());
    }

    #[test]
    fn test_get_matrix_at() {
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();
        let measure = PathMeasure::new(&path);
        assert!(measure.get_matrix_at(50.0).is_some());
    }
}
