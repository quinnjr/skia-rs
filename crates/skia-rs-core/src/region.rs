//! Region operations for complex clipping.
//!
//! This module provides Skia-compatible region types for representing
//! complex clip areas composed of multiple rectangles.

use crate::geometry::{IRect, Rect};

/// Operation type for combining regions.
///
/// Corresponds to Skia's `SkRegion::Op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum RegionOp {
    /// Subtract the second region from the first.
    Difference = 0,
    /// Intersect the two regions.
    Intersect,
    /// Union the two regions.
    Union,
    /// XOR the two regions (areas in one but not both).
    Xor,
    /// Reverse difference (subtract first from second).
    ReverseDifference,
    /// Replace with the second region.
    Replace,
}

/// A region composed of rectangles.
///
/// Represents a complex area that can be the result of multiple boolean
/// operations on rectangles. Used for efficient clipping operations.
///
/// Corresponds to Skia's `SkRegion`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Region {
    /// The rectangles composing this region (in scanline order).
    rects: Vec<IRect>,
}

impl Region {
    /// Create an empty region.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { rects: Vec::new() }
    }

    /// Create a region from a single rectangle.
    #[must_use]
    pub fn from_rect(rect: IRect) -> Self {
        if rect.is_empty() {
            return Self::new();
        }
        Self { rects: vec![rect] }
    }

    /// Create a region from a floating-point rectangle (rounds outward).
    #[must_use]
    pub fn from_rect_f(rect: &Rect) -> Self {
        Self::from_rect(rect.round_out())
    }

    /// Returns true if the region is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }

    /// Returns true if the region is a single rectangle.
    #[inline]
    #[must_use]
    pub const fn is_rect(&self) -> bool {
        self.rects.len() == 1
    }

    /// Returns true if the region is complex (more than one rectangle).
    #[inline]
    #[must_use]
    pub const fn is_complex(&self) -> bool {
        self.rects.len() > 1
    }

    /// Returns the bounding rectangle of this region.
    #[inline]
    #[must_use]
    pub fn bounds(&self) -> IRect {
        if self.rects.is_empty() {
            return IRect::empty();
        }
        let mut bounds = self.rects[0];
        for rect in &self.rects[1..] {
            bounds = bounds.union(rect);
        }
        bounds
    }

    /// Returns the number of rectangles in the region.
    #[inline]
    #[must_use]
    pub const fn rect_count(&self) -> usize {
        self.rects.len()
    }

    /// Returns an iterator over the rectangles in the region.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = IRect> + '_ {
        self.rects.iter().copied()
    }

    /// Returns the rectangles composing this region.
    #[inline]
    #[must_use]
    pub fn rects(&self) -> &[IRect] {
        &self.rects
    }

    /// Clear the region to empty.
    pub fn set_empty(&mut self) {
        self.rects.clear();
    }

    /// Set the region to a single rectangle.
    pub fn set_rect(&mut self, rect: IRect) -> bool {
        if rect.is_empty() {
            self.set_empty();
            return false;
        }
        self.rects.clear();
        self.rects.push(rect);
        true
    }

    /// Set the region to another region.
    pub fn set_region(&mut self, other: &Self) -> bool {
        self.rects.clone_from(&other.rects);
        !self.is_empty()
    }

    /// Returns true if the point is contained in the region.
    #[must_use]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        if !self.bounds().contains(x, y) {
            return false;
        }
        for rect in &self.rects {
            if rect.contains(x, y) {
                return true;
            }
        }
        false
    }

    /// Returns true if the rectangle is completely contained in the region.
    #[must_use]
    pub fn contains_rect(&self, rect: &IRect) -> bool {
        if rect.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }

        // Quick bounds check
        let bounds = self.bounds();
        if let Some(intersection) = bounds.intersect(rect) {
            if intersection != *rect {
                return false;
            }
        } else {
            return false;
        }

        // For a single rectangle, simple containment check
        if self.is_rect() {
            return true;
        }

        // For complex regions: verify each y-band of the query rect is
        // x-covered by the union of region rects active in that band.
        let query_top = rect.top;
        let query_bottom = rect.bottom;

        // Collect y-edges from region rects that overlap the query range
        let mut y_edges: Vec<i32> = vec![query_top, query_bottom];
        for r in &self.rects {
            if r.top < query_bottom && r.bottom > query_top {
                if r.top > query_top && r.top < query_bottom {
                    y_edges.push(r.top);
                }
                if r.bottom > query_top && r.bottom < query_bottom {
                    y_edges.push(r.bottom);
                }
            }
        }
        y_edges.sort_unstable();
        y_edges.dedup();

        // Process each y-band
        let mut x_intervals: Vec<(i32, i32)> = Vec::with_capacity(self.rects.len());
        for window in y_edges.windows(2) {
            let band_top = window[0];
            let band_bottom = window[1];

            x_intervals.clear();
            x_intervals.extend(
                self.rects
                    .iter()
                    .filter(|r| r.top <= band_top && r.bottom >= band_bottom)
                    .map(|r| (r.left, r.right)),
            );

            if x_intervals.is_empty() {
                return false;
            }

            x_intervals.sort_unstable_by_key(|&(l, _)| l);

            let mut covered_right = rect.left;
            for (l, r) in x_intervals.iter().copied() {
                if l > covered_right {
                    return false; // Gap in coverage
                }
                covered_right = covered_right.max(r);
                if covered_right >= rect.right {
                    break;
                }
            }
            if covered_right < rect.right {
                return false;
            }
        }

        true
    }

    /// Returns true if this region intersects with a rectangle.
    #[must_use]
    pub fn intersects_rect(&self, rect: &IRect) -> bool {
        if self.is_empty() || rect.is_empty() {
            return false;
        }
        if self.bounds().intersect(rect).is_none() {
            return false;
        }
        for r in &self.rects {
            if r.intersect(rect).is_some() {
                return true;
            }
        }
        false
    }

    /// Returns true if this region intersects with another region.
    #[must_use]
    pub fn intersects_region(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        if self.bounds().intersect(&other.bounds()).is_none() {
            return false;
        }
        for r1 in &self.rects {
            for r2 in &other.rects {
                if r1.intersect(r2).is_some() {
                    return true;
                }
            }
        }
        false
    }

    /// Translate the region by (dx, dy).
    pub fn translate(&mut self, dx: i32, dy: i32) {
        for rect in &mut self.rects {
            *rect = rect.offset(dx, dy);
        }
    }

    /// Returns a translated copy of this region.
    #[must_use]
    pub fn translated(&self, dx: i32, dy: i32) -> Self {
        let mut result = self.clone();
        result.translate(dx, dy);
        result
    }

    /// Combine this region with a rectangle using the specified operation.
    pub fn op_rect(&mut self, rect: IRect, op: RegionOp) -> bool {
        let other = Self::from_rect(rect);
        self.op_region(&other, op)
    }

    /// Combine this region with another region using the specified operation.
    pub fn op_region(&mut self, other: &Self, op: RegionOp) -> bool {
        match op {
            RegionOp::Replace => self.set_region(other),
            RegionOp::Intersect => self.intersect(other),
            RegionOp::Union => self.union(other),
            RegionOp::Xor => self.xor(other),
            RegionOp::Difference => self.difference(other),
            RegionOp::ReverseDifference => {
                let mut temp = other.clone();
                temp.difference(self);
                *self = temp;
                !self.is_empty()
            }
        }
    }

    /// Intersect this region with another.
    fn intersect(&mut self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            self.set_empty();
            return false;
        }

        // Quick bounds check
        if self.bounds().intersect(&other.bounds()).is_none() {
            self.set_empty();
            return false;
        }

        let mut result_rects = Vec::new();
        for r1 in &self.rects {
            for r2 in &other.rects {
                if let Some(intersection) = r1.intersect(r2) {
                    result_rects.push(intersection);
                }
            }
        }

        // Canonicalize to strict scanline form so PartialEq, is_rect(),
        // rect_count() and iteration order match SkRegion semantics.
        self.rects = canonicalize_rects(&result_rects);
        !self.is_empty()
    }

    /// Union this region with another.
    fn union(&mut self, other: &Self) -> bool {
        if other.is_empty() {
            return !self.is_empty();
        }
        if self.is_empty() {
            return self.set_region(other);
        }

        // Combine all rects, then canonicalize to scanline form so overlapping
        // and adjacent rects are merged.  Without this the rect list grows
        // without bound on repeated unions, making all O(n) region ops slow.
        let mut all_rects = std::mem::take(&mut self.rects);
        all_rects.extend(other.rects.iter().copied());

        self.rects = canonicalize_rects(&all_rects);
        !self.is_empty()
    }

    /// XOR this region with another.
    fn xor(&mut self, other: &Self) -> bool {
        // XOR = (A - B) + (B - A)
        let mut a_minus_b = self.clone();
        a_minus_b.difference(other);

        let mut b_minus_a = other.clone();
        b_minus_a.difference(self);

        a_minus_b.union(&b_minus_a);
        *self = a_minus_b;
        !self.is_empty()
    }

    /// Subtract another region from this one.
    fn difference(&mut self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return !self.is_empty();
        }

        // Quick bounds check
        if self.bounds().intersect(&other.bounds()).is_none() {
            return !self.is_empty();
        }

        // For each rectangle in self, subtract all rectangles from other
        let mut result_rects = Vec::new();
        for rect in &self.rects {
            let mut fragments = vec![*rect];
            for other_rect in &other.rects {
                let mut new_fragments = Vec::new();
                for frag in fragments {
                    new_fragments.extend(subtract_rect(&frag, other_rect));
                }
                fragments = new_fragments;
            }
            result_rects.extend(fragments);
        }

        // Canonicalize to strict scanline form so PartialEq, is_rect(),
        // rect_count() and iteration order match SkRegion semantics.
        self.rects = canonicalize_rects(&result_rects);
        !self.is_empty()
    }
}

/// Subtract rect2 from rect1, returning the resulting fragments.
fn subtract_rect(rect1: &IRect, rect2: &IRect) -> Vec<IRect> {
    if rect1.intersect(rect2).is_none() {
        return vec![*rect1];
    }

    let mut result = Vec::new();

    // Top fragment
    if rect1.top < rect2.top {
        let top = IRect::new(rect1.left, rect1.top, rect1.right, rect2.top);
        if !top.is_empty() {
            result.push(top);
        }
    }

    // Bottom fragment
    if rect1.bottom > rect2.bottom {
        let bottom = IRect::new(rect1.left, rect2.bottom, rect1.right, rect1.bottom);
        if !bottom.is_empty() {
            result.push(bottom);
        }
    }

    // Left fragment (in the middle band)
    let middle_top = rect1.top.max(rect2.top);
    let middle_bottom = rect1.bottom.min(rect2.bottom);
    if middle_top < middle_bottom {
        if rect1.left < rect2.left {
            let left = IRect::new(rect1.left, middle_top, rect2.left, middle_bottom);
            if !left.is_empty() {
                result.push(left);
            }
        }

        // Right fragment (in the middle band)
        if rect1.right > rect2.right {
            let right = IRect::new(rect2.right, middle_top, rect1.right, middle_bottom);
            if !right.is_empty() {
                result.push(right);
            }
        }
    }

    result
}

/// A horizontal scanline band: `(top, bottom, x-intervals)`, where each
/// x-interval is an inclusive-left/exclusive-right `(left, right)` pair.
type Band = (i32, i32, Vec<(i32, i32)>);

/// Canonicalize a list of rectangles into scanline form.
///
/// Returns a minimal list of non-overlapping rectangles whose union equals
/// the union of the input rectangles.  The algorithm works in three passes:
///
/// 1. Collect all unique horizontal y-edges from the input rects.
/// 2. For each consecutive pair of y-edges (a "band"), collect the
///    x-intervals contributed by every input rect that spans the full band
///    height, then merge overlapping/adjacent intervals into a sorted,
///    non-overlapping list.
/// 3. Attempt to extend the previous output band downward when the current
///    band has an identical x-interval list, eliminating redundant splits.
///
/// The result is equivalent to Skia's canonical `SkRegion` scanline format.
fn canonicalize_rects(input: &[IRect]) -> Vec<IRect> {
    if input.is_empty() {
        return Vec::new();
    }

    // Collect all unique y-edges, skipping empty rects.
    let mut y_edges: Vec<i32> = Vec::with_capacity(input.len() * 2);
    for r in input {
        if r.is_empty() {
            continue;
        }
        y_edges.push(r.top);
        y_edges.push(r.bottom);
    }
    if y_edges.is_empty() {
        return Vec::new();
    }
    y_edges.sort_unstable();
    y_edges.dedup();

    let mut result: Vec<IRect> = Vec::new();
    let mut prev_band: Option<Band> = None;

    // Reusable buffers - allocate once, clear per-band.
    let mut intervals: Vec<(i32, i32)> = Vec::with_capacity(input.len());
    let mut merged: Vec<(i32, i32)> = Vec::with_capacity(input.len());

    for window in y_edges.windows(2) {
        let band_top = window[0];
        let band_bottom = window[1];

        intervals.clear();
        intervals.extend(
            input
                .iter()
                .filter(|r| !r.is_empty() && r.top <= band_top && r.bottom >= band_bottom)
                .map(|r| (r.left, r.right)),
        );

        if intervals.is_empty() {
            if let Some((top, bottom, ints)) = prev_band.take() {
                for (l, r) in ints {
                    result.push(IRect::new(l, top, r, bottom));
                }
            }
            continue;
        }

        intervals.sort_unstable_by_key(|&(l, _)| l);
        merged.clear();
        let mut cur = intervals[0];
        for &(l, r) in &intervals[1..] {
            if l <= cur.1 {
                if r > cur.1 {
                    cur.1 = r;
                }
            } else {
                merged.push(cur);
                cur = (l, r);
            }
        }
        merged.push(cur);

        // Try to extend the previous band downward when x-intervals match.
        match prev_band.take() {
            Some((prev_top, prev_bottom, ref prev_intervals))
                if prev_bottom == band_top && prev_intervals == &merged =>
            {
                prev_band = Some((prev_top, band_bottom, std::mem::take(&mut merged)));
            }
            Some((prev_top, prev_bottom, prev_intervals)) => {
                for (l, r) in prev_intervals {
                    result.push(IRect::new(l, prev_top, r, prev_bottom));
                }
                prev_band = Some((band_top, band_bottom, std::mem::take(&mut merged)));
            }
            None => {
                prev_band = Some((band_top, band_bottom, std::mem::take(&mut merged)));
            }
        }
    }

    if let Some((top, bottom, intervals)) = prev_band {
        for (l, r) in intervals {
            result.push(IRect::new(l, top, r, bottom));
        }
    }

    result
}

/// Iterator over the rectangles in a region.
pub struct RegionIter<'a> {
    region: &'a Region,
    index: usize,
}

impl<'a> Iterator for RegionIter<'a> {
    type Item = &'a IRect;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.region.rects.len() {
            let rect = &self.region.rects[self.index];
            self.index += 1;
            Some(rect)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.region.rects.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RegionIter<'_> {}

impl<'a> IntoIterator for &'a Region {
    type Item = &'a IRect;
    type IntoIter = RegionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        RegionIter {
            region: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_region() {
        let region = Region::new();
        assert!(region.is_empty());
        assert!(!region.is_rect());
        assert!(!region.is_complex());
    }

    #[test]
    fn test_rect_region() {
        let rect = IRect::new(10, 20, 100, 200);
        let region = Region::from_rect(rect);
        assert!(!region.is_empty());
        assert!(region.is_rect());
        assert!(!region.is_complex());
        assert_eq!(region.bounds(), rect);
    }

    #[test]
    fn test_contains_point() {
        let region = Region::from_rect(IRect::new(0, 0, 100, 100));
        assert!(region.contains(50, 50));
        assert!(region.contains(0, 0));
        assert!(!region.contains(100, 100)); // Exclusive
        assert!(!region.contains(-1, 50));
    }

    #[test]
    fn test_translate() {
        let mut region = Region::from_rect(IRect::new(0, 0, 100, 100));
        region.translate(50, 50);
        assert_eq!(region.bounds(), IRect::new(50, 50, 150, 150));
    }

    #[test]
    fn test_intersect() {
        let mut region = Region::from_rect(IRect::new(0, 0, 100, 100));
        region.op_rect(IRect::new(50, 50, 150, 150), RegionOp::Intersect);
        assert_eq!(region.bounds(), IRect::new(50, 50, 100, 100));
    }

    #[test]
    fn test_union() {
        let mut region = Region::from_rect(IRect::new(0, 0, 50, 50));
        region.op_rect(IRect::new(50, 50, 100, 100), RegionOp::Union);
        assert_eq!(region.bounds(), IRect::new(0, 0, 100, 100));
        assert!(region.is_complex());
    }

    #[test]
    fn test_difference() {
        let mut region = Region::from_rect(IRect::new(0, 0, 100, 100));
        region.op_rect(IRect::new(25, 25, 75, 75), RegionOp::Difference);
        assert!(!region.is_empty());
        assert!(region.is_complex());
        // Should have 4 fragments around the hole
        assert_eq!(region.rect_count(), 4);
    }

    #[test]
    fn test_region_contains_rect_spanning_two_components() {
        // Build a complex region from two adjacent rectangles
        // [0,0,10,10] and [10,0,20,10] - together they form [0,0,20,10]
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let other = Region::from_rect(IRect::new(10, 0, 20, 10));
        region.op_region(&other, RegionOp::Union);

        // A query rect [5,2,15,8] is contained by the union but not by
        // either individual rect
        let query = IRect::new(5, 2, 15, 8);
        assert!(
            region.contains_rect(&query),
            "Region should contain rect spanning multiple components"
        );
    }

    #[test]
    fn test_region_contains_rect_truly_outside() {
        let region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let query = IRect::new(20, 20, 30, 30);
        assert!(!region.contains_rect(&query));
    }

    #[test]
    fn test_region_contains_rect_partial_overlap_not_contained() {
        let region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let query = IRect::new(5, 5, 15, 15); // half outside
        assert!(!region.contains_rect(&query));
    }

    #[test]
    fn test_region_union_merges_adjacent_horizontal() {
        // [0,0,10,10] union [10,0,20,10] should produce a single rect [0,0,20,10]
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let other = Region::from_rect(IRect::new(10, 0, 20, 10));
        region.op_region(&other, RegionOp::Union);

        let rects = region.rects();
        assert_eq!(rects.len(), 1, "Adjacent horizontal rects should merge");
        assert_eq!(rects[0], IRect::new(0, 0, 20, 10));
    }

    #[test]
    fn test_region_union_merges_overlapping() {
        // [0,0,10,10] union [5,0,15,10] should produce [0,0,15,10]
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let other = Region::from_rect(IRect::new(5, 0, 15, 10));
        region.op_region(&other, RegionOp::Union);

        let rects = region.rects();
        assert_eq!(rects.len(), 1, "Overlapping rects should merge");
        assert_eq!(rects[0], IRect::new(0, 0, 15, 10));
    }

    #[test]
    fn test_region_union_disjoint_keeps_both() {
        // [0,0,10,10] union [20,20,30,30] should keep both
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let other = Region::from_rect(IRect::new(20, 20, 30, 30));
        region.op_region(&other, RegionOp::Union);

        let rects = region.rects();
        assert_eq!(rects.len(), 2, "Disjoint rects should not merge");
    }

    #[test]
    fn test_region_union_repeated_does_not_grow_unboundedly() {
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        let other = Region::from_rect(IRect::new(0, 0, 10, 10));

        for _ in 0..100 {
            region.op_region(&other, RegionOp::Union);
        }

        let rects = region.rects();
        assert_eq!(
            rects.len(),
            1,
            "Repeated union with same rect should stay at 1"
        );
    }

    #[test]
    fn test_region_union_repeated_disjoint_grows_linearly() {
        let mut region = Region::new();
        for i in 0..50 {
            let other = Region::from_rect(IRect::new(i * 20, 0, i * 20 + 10, 10));
            region.op_region(&other, RegionOp::Union);
        }
        let rects = region.rects();
        assert_eq!(rects.len(), 50, "50 disjoint rects should remain 50 rects");
    }

    // --- Conformance regression tests (Task 1) ---

    #[test]
    fn test_difference_canonicalizes_scanline_order() {
        let mut region = Region::from_rect(IRect::new(0, 0, 100, 100));
        region.op_rect(IRect::new(25, 25, 75, 75), RegionOp::Difference);

        // Iteration must be in strict top-to-bottom scanline order.
        let rects: Vec<_> = region.iter().collect();
        assert_eq!(rects.len(), 4);
        for w in rects.windows(2) {
            assert!(w[0].top <= w[1].top, "rects must be in scanline order");
        }

        // The same frame built purely from unions must compare equal, proving
        // difference canonicalizes to the same representation.
        let mut frame = Region::from_rect(IRect::new(0, 0, 100, 25));
        frame.op_rect(IRect::new(0, 25, 25, 75), RegionOp::Union);
        frame.op_rect(IRect::new(75, 25, 100, 75), RegionOp::Union);
        frame.op_rect(IRect::new(0, 75, 100, 100), RegionOp::Union);
        assert_eq!(region, frame, "difference must canonicalize like union");
    }

    #[test]
    fn test_intersect_canonicalizes_scanline_order() {
        // A complex region (two vertically disjoint rects) intersected with a
        // spanning rect must yield canonical, scanline-ordered output.
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        region.op_rect(IRect::new(0, 20, 10, 30), RegionOp::Union);
        region.op_rect(IRect::new(0, 0, 10, 30), RegionOp::Intersect);

        let rects: Vec<_> = region.iter().collect();
        assert_eq!(
            rects,
            vec![IRect::new(0, 0, 10, 10), IRect::new(0, 20, 10, 30)]
        );
    }

    #[test]
    fn test_intersect_merges_to_single_rect() {
        // Two adjacent rects whose intersection with a band should merge into a
        // single canonical rect (is_rect true, rect_count 1).
        let mut region = Region::from_rect(IRect::new(0, 0, 10, 10));
        region.op_rect(IRect::new(0, 10, 10, 20), RegionOp::Union);
        // Region is now the single merged rect [0,0,10,20].
        region.op_rect(IRect::new(0, 0, 10, 20), RegionOp::Intersect);
        assert!(region.is_rect(), "intersection should be a single rect");
        assert_eq!(region.rect_count(), 1);
        assert_eq!(region.bounds(), IRect::new(0, 0, 10, 20));
    }
}
