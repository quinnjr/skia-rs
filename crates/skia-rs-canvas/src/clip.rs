//! Advanced clipping operations for the rasterizer.
//!
//! This module provides:
//! - **Anti-aliased clipping** using coverage masks
//! - **Region-based clipping** for complex clip shapes
//! - **Clip stack** for save/restore semantics
//!
//! ## Anti-Aliased Clipping
//!
//! Anti-aliased clips use a coverage mask to store per-pixel alpha values,
//! allowing smooth clip edges without jagged artifacts. The coverage is
//! computed by rasterizing the clip path with sub-pixel precision.
//!
//! ## Region-Based Clipping
//!
//! Region clips use `skia_rs_core::Region` to represent complex clip areas
//! composed of multiple rectangles. This is efficient for non-anti-aliased
//! clips with complex shapes.

use skia_rs_core::{IRect, Point, Rect, Region, RegionOp};
use skia_rs_path::Path;

/// A coverage mask for anti-aliased clipping.
///
/// Stores per-pixel coverage values (0-255) where 0 is fully clipped
/// and 255 is fully visible. Intermediate values provide smooth edges.
#[derive(Debug, Clone)]
pub struct ClipMask {
    /// Width in pixels.
    width: i32,
    /// Height in pixels.
    height: i32,
    /// Coverage data (one byte per pixel).
    coverage: Vec<u8>,
    /// Bounds of the mask in device coordinates.
    bounds: IRect,
}

impl ClipMask {
    /// Create a new clip mask filled with the given coverage value.
    #[must_use]
    pub fn new(width: i32, height: i32, initial_coverage: u8) -> Self {
        let size = usize::try_from(width * height).unwrap_or(0);
        Self {
            width,
            height,
            coverage: vec![initial_coverage; size],
            bounds: IRect::new(0, 0, width, height),
        }
    }

    /// Create a clip mask from a rectangle with anti-aliased edges.
    #[must_use]
    pub fn from_rect_aa(rect: &Rect, device_bounds: &IRect) -> Self {
        use skia_rs_core::cast::scalar_from_i32;

        let width = device_bounds.width();
        let height = device_bounds.height();
        let mut mask = Self::new(width, height, 0);

        // Rasterize the rectangle with sub-pixel coverage
        let left = rect.left - scalar_from_i32(device_bounds.left);
        let top = rect.top - scalar_from_i32(device_bounds.top);
        let right = rect.right - scalar_from_i32(device_bounds.left);
        let bottom = rect.bottom - scalar_from_i32(device_bounds.top);

        for y in 0..height {
            for x in 0..width {
                let px = scalar_from_i32(x);
                let py = scalar_from_i32(y);

                // Calculate coverage for this pixel
                let coverage = compute_rect_coverage(px, py, left, top, right, bottom);
                mask.set_coverage(x, y, coverage);
            }
        }

        mask.bounds = *device_bounds;
        mask
    }

    /// Create a clip mask from a path with anti-aliased edges.
    #[must_use]
    pub fn from_path_aa(path: &Path, device_bounds: &IRect) -> Self {
        let width = device_bounds.width();
        let height = device_bounds.height();
        let mut mask = Self::new(width, height, 0);
        mask.coverage = crate::raster::path_coverage_aa(path, device_bounds);
        mask.bounds = *device_bounds;
        mask
    }

    /// Get the coverage value at (x, y) in local coordinates.
    #[inline]
    #[must_use]
    pub fn get_coverage(&self, x: i32, y: i32) -> u8 {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 0;
        }
        self.coverage[usize::try_from(y * self.width + x).unwrap_or(0)]
    }

    /// Get the coverage value at device coordinates.
    #[inline]
    #[must_use]
    pub fn get_coverage_device(&self, x: i32, y: i32) -> u8 {
        let lx = x - self.bounds.left;
        let ly = y - self.bounds.top;
        self.get_coverage(lx, ly)
    }

    /// Set the coverage value at (x, y).
    #[inline]
    pub fn set_coverage(&mut self, x: i32, y: i32, coverage: u8) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            self.coverage[usize::try_from(y * self.width + x).unwrap_or(0)] = coverage;
        }
    }

    /// Returns the bounds of this mask.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> IRect {
        self.bounds
    }

    /// Returns the width of this mask.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.width
    }

    /// Returns the height of this mask.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.height
    }

    /// Intersect this mask with another mask.
    pub fn intersect(&mut self, other: &Self) {
        // Find intersection bounds
        let Some(intersection) = self.bounds.intersect(&other.bounds) else {
            // No intersection - clear the mask
            self.coverage.fill(0);
            return;
        };

        // For pixels in the intersection, multiply coverage
        for y in intersection.top..intersection.bottom {
            for x in intersection.left..intersection.right {
                let self_cov = u32::from(self.get_coverage_device(x, y));
                let other_cov = u32::from(other.get_coverage_device(x, y));
                let combined = u8::try_from((self_cov * other_cov) / 255).unwrap_or(255);

                let lx = x - self.bounds.left;
                let ly = y - self.bounds.top;
                self.set_coverage(lx, ly, combined);
            }
        }

        // Clear pixels outside the intersection
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x + self.bounds.left;
                let dy = y + self.bounds.top;
                if dx < intersection.left
                    || dx >= intersection.right
                    || dy < intersection.top
                    || dy >= intersection.bottom
                {
                    self.coverage[usize::try_from(y * self.width + x).unwrap_or(0)] = 0;
                }
            }
        }
    }

    /// Apply a rectangular clip to this mask.
    pub fn clip_rect(&mut self, rect: &IRect) {
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x + self.bounds.left;
                let dy = y + self.bounds.top;
                if !rect.contains(dx, dy) {
                    self.coverage[usize::try_from(y * self.width + x).unwrap_or(0)] = 0;
                }
            }
        }
    }
}

/// Compute rectangle coverage for a pixel.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "x_coverage/y_coverage are clamped to [0,1], so coverage*255 is provably in [0,255]; truncation (not rounding) is the intended AA coverage semantics"
)]
fn compute_rect_coverage(px: f32, py: f32, left: f32, top: f32, right: f32, bottom: f32) -> u8 {
    // Calculate how much of the pixel is inside the rectangle
    let x_coverage = (right.min(px + 1.0) - left.max(px)).clamp(0.0, 1.0);
    let y_coverage = (bottom.min(py + 1.0) - top.max(py)).clamp(0.0, 1.0);
    let coverage = x_coverage * y_coverage;
    (coverage * 255.0) as u8
}

/// Represents a clip state that can be either rectangular, regional, or masked.
#[derive(Debug, Clone)]
pub enum ClipState {
    /// Simple rectangular clip (fast path).
    Rect(Rect),
    /// Region-based clip (multiple rectangles).
    Region(Region),
    /// Anti-aliased clip with coverage mask.
    Mask(ClipMask),
    /// Combined region and mask clip.
    RegionAndMask(Region, ClipMask),
}

impl ClipState {
    /// Create a clip state from a rectangle.
    #[must_use]
    pub const fn from_rect(rect: Rect) -> Self {
        Self::Rect(rect)
    }

    /// Create a clip state from a region.
    #[must_use]
    pub const fn from_region(region: Region) -> Self {
        Self::Region(region)
    }

    /// Create an anti-aliased clip state from a rectangle.
    #[must_use]
    pub fn from_rect_aa(rect: &Rect, device_bounds: &IRect) -> Self {
        Self::Mask(ClipMask::from_rect_aa(rect, device_bounds))
    }

    /// Create an anti-aliased clip state from a path.
    #[must_use]
    pub fn from_path_aa(path: &Path, device_bounds: &IRect) -> Self {
        Self::Mask(ClipMask::from_path_aa(path, device_bounds))
    }

    /// Get the bounding rectangle of this clip.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        use skia_rs_core::cast::scalar_from_i32;

        match self {
            Self::Rect(r) => *r,
            Self::Region(r) | Self::RegionAndMask(r, _) => {
                let b = r.bounds();
                Rect::new(
                    scalar_from_i32(b.left),
                    scalar_from_i32(b.top),
                    scalar_from_i32(b.right),
                    scalar_from_i32(b.bottom),
                )
            }
            Self::Mask(m) => {
                let b = m.bounds();
                Rect::new(
                    scalar_from_i32(b.left),
                    scalar_from_i32(b.top),
                    scalar_from_i32(b.right),
                    scalar_from_i32(b.bottom),
                )
            }
        }
    }

    /// Check if a point is inside the clip.
    ///
    /// Containment tests sample the **pixel center** (x + 0.5, y + 0.5),
    /// matching Skia's non-AA coverage rule.
    #[must_use]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        use skia_rs_core::cast::scalar_from_i32;

        match self {
            Self::Rect(r) => r.contains(Point::new(
                scalar_from_i32(x) + 0.5,
                scalar_from_i32(y) + 0.5,
            )),
            Self::Region(r) => r.contains(x, y),
            Self::Mask(m) => m.get_coverage_device(x, y) > 0,
            Self::RegionAndMask(r, m) => r.contains(x, y) && m.get_coverage_device(x, y) > 0,
        }
    }

    /// Get the coverage at a point (0-255).
    #[must_use]
    pub fn get_coverage(&self, x: i32, y: i32) -> u8 {
        use skia_rs_core::cast::scalar_from_i32;

        match self {
            Self::Rect(r) => {
                if r.contains(Point::new(
                    scalar_from_i32(x) + 0.5,
                    scalar_from_i32(y) + 0.5,
                )) {
                    255
                } else {
                    0
                }
            }
            Self::Region(r) => {
                if r.contains(x, y) {
                    255
                } else {
                    0
                }
            }
            Self::Mask(m) => m.get_coverage_device(x, y),
            Self::RegionAndMask(r, m) => {
                if r.contains(x, y) {
                    m.get_coverage_device(x, y)
                } else {
                    0
                }
            }
        }
    }

    /// Check if this is an anti-aliased clip.
    #[must_use]
    pub const fn is_anti_aliased(&self) -> bool {
        matches!(self, Self::Mask(_) | Self::RegionAndMask(_, _))
    }

    /// Intersect this clip with a rectangle.
    ///
    /// Non-AA clip rects round to nearest (Skia `rect.round()`), not
    /// round-out.
    pub fn intersect_rect(&mut self, rect: &Rect) {
        match self {
            Self::Rect(r) => {
                if let Some(intersection) = r.intersect(rect) {
                    *r = intersection;
                } else {
                    *r = Rect::EMPTY;
                }
            }
            Self::Region(r) => {
                r.op_rect(rect.round(), skia_rs_core::RegionOp::Intersect);
            }
            Self::Mask(m) => {
                m.clip_rect(&rect.round());
            }
            Self::RegionAndMask(r, m) => {
                let irect = rect.round();
                r.op_rect(irect, skia_rs_core::RegionOp::Intersect);
                m.clip_rect(&irect);
            }
        }
    }

    /// Intersect this clip with a region.
    pub fn intersect_region(&mut self, region: &Region) {
        match self {
            Self::Rect(r) => {
                // Non-AA conversion rounds to nearest (rect.round()).
                let mut new_region = Region::from_rect(r.round());
                new_region.op_region(region, skia_rs_core::RegionOp::Intersect);
                *self = Self::Region(new_region);
            }
            Self::Region(r) => {
                r.op_region(region, skia_rs_core::RegionOp::Intersect);
            }
            Self::Mask(m) => {
                // Convert region to mask intersection
                for y in 0..m.height {
                    for x in 0..m.width {
                        let dx = x + m.bounds.left;
                        let dy = y + m.bounds.top;
                        if !region.contains(dx, dy) {
                            m.set_coverage(x, y, 0);
                        }
                    }
                }
            }
            Self::RegionAndMask(r, m) => {
                r.op_region(region, skia_rs_core::RegionOp::Intersect);
                // Also update mask
                for y in 0..m.height {
                    for x in 0..m.width {
                        let dx = x + m.bounds.left;
                        let dy = y + m.bounds.top;
                        if !region.contains(dx, dy) {
                            m.set_coverage(x, y, 0);
                        }
                    }
                }
            }
        }
    }

    /// Subtract a rectangle from the current clip (difference operation).
    pub fn difference_rect(&mut self, rect: &Rect) {
        match self {
            Self::Rect(r) => {
                // Convert to region for difference operation
                let mut region = Region::from_rect(r.round());
                region.op_rect(rect.round(), skia_rs_core::RegionOp::Difference);
                *self = Self::Region(region);
            }
            Self::Region(r) => {
                r.op_rect(rect.round(), skia_rs_core::RegionOp::Difference);
            }
            Self::Mask(m) => {
                // Zero out coverage in the rect area
                let irect = rect.round();
                for y in 0..m.height {
                    for x in 0..m.width {
                        let dx = x + m.bounds.left;
                        let dy = y + m.bounds.top;
                        if irect.contains(dx, dy) {
                            m.set_coverage(x, y, 0);
                        }
                    }
                }
            }
            Self::RegionAndMask(r, m) => {
                r.op_rect(rect.round(), skia_rs_core::RegionOp::Difference);
                // Also zero out coverage in the rect area
                let irect = rect.round();
                for y in 0..m.height {
                    for x in 0..m.width {
                        let dx = x + m.bounds.left;
                        let dy = y + m.bounds.top;
                        if irect.contains(dx, dy) {
                            m.set_coverage(x, y, 0);
                        }
                    }
                }
            }
        }
    }
}

/// A stack of clip states for save/restore semantics.
#[derive(Debug, Clone)]
pub struct ClipStack {
    /// Stack of saved clip states.
    stack: Vec<ClipState>,
    /// Current clip state.
    current: ClipState,
}

impl ClipStack {
    /// Create a new clip stack with the given device bounds.
    #[must_use]
    pub const fn new(device_bounds: &Rect) -> Self {
        Self {
            stack: Vec::new(),
            current: ClipState::Rect(*device_bounds),
        }
    }

    /// Save the current clip state.
    pub fn save(&mut self) {
        self.stack.push(self.current.clone());
    }

    /// Restore the previous clip state.
    pub fn restore(&mut self) {
        if let Some(state) = self.stack.pop() {
            self.current = state;
        }
    }

    /// Get the current save count.
    #[must_use]
    pub fn save_count(&self) -> usize {
        self.stack.len()
    }

    /// Restore to a specific save count.
    pub fn restore_to_count(&mut self, count: usize) {
        while self.stack.len() > count {
            self.restore();
        }
    }

    /// Get the current clip state.
    #[must_use]
    pub const fn current(&self) -> &ClipState {
        &self.current
    }

    /// Get the current clip bounds.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        self.current.bounds()
    }

    /// Check if a point is inside the current clip.
    #[must_use]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        self.current.contains(x, y)
    }

    /// Get the coverage at a point.
    #[must_use]
    pub fn get_coverage(&self, x: i32, y: i32) -> u8 {
        self.current.get_coverage(x, y)
    }

    /// Intersect the current clip with a rectangle (non-anti-aliased).
    ///
    /// The rect is rounded to nearest integer edges first (Skia
    /// `rect.round()` for non-AA clips), so subsequent pixel-center
    /// containment tests match Skia's BW clip behavior.
    pub fn clip_rect(&mut self, rect: &Rect) {
        use skia_rs_core::cast::scalar_from_i32;

        let ir = rect.round();
        let rounded = Rect::new(
            scalar_from_i32(ir.left),
            scalar_from_i32(ir.top),
            scalar_from_i32(ir.right),
            scalar_from_i32(ir.bottom),
        );
        self.current.intersect_rect(&rounded);
    }

    /// Apply a clip operation with a rectangle.
    pub fn clip_rect_op(&mut self, rect: &Rect, op: super::ClipOp) {
        use super::ClipOp;
        match op {
            ClipOp::Intersect => self.current.intersect_rect(rect),
            ClipOp::Difference => self.current.difference_rect(rect),
        }
    }

    /// Intersect the current clip with a rectangle (anti-aliased).
    pub fn clip_rect_aa(&mut self, rect: &Rect, device_bounds: &IRect) {
        let mask = ClipMask::from_rect_aa(rect, device_bounds);
        match &mut self.current {
            ClipState::Rect(r) => {
                let mut new_mask = mask;
                new_mask.clip_rect(&r.round_out());
                self.current = ClipState::Mask(new_mask);
            }
            ClipState::Region(r) => {
                let mut new_mask = mask;
                for y in 0..new_mask.height {
                    for x in 0..new_mask.width {
                        let dx = x + new_mask.bounds.left;
                        let dy = y + new_mask.bounds.top;
                        if !r.contains(dx, dy) {
                            new_mask.set_coverage(x, y, 0);
                        }
                    }
                }
                self.current = ClipState::Mask(new_mask);
            }
            ClipState::Mask(m) => {
                m.intersect(&mask);
            }
            ClipState::RegionAndMask(r, m) => {
                use skia_rs_core::cast::{ceil_to_i32, floor_to_i32};

                // Intersect both the region and the mask
                let rounded_rect = IRect::new(
                    floor_to_i32(rect.left),
                    floor_to_i32(rect.top),
                    ceil_to_i32(rect.right),
                    ceil_to_i32(rect.bottom),
                );
                r.op_rect(rounded_rect, RegionOp::Intersect);
                m.intersect(&mask);
            }
        }
    }

    /// Intersect the current clip with a region.
    pub fn clip_region(&mut self, region: &Region) {
        self.current.intersect_region(region);
    }

    /// Apply a clip operation with a rectangle, optionally anti-aliased.
    pub fn clip_rect_with_op(
        &mut self,
        rect: &Rect,
        op: super::ClipOp,
        device_bounds: &IRect,
        anti_alias: bool,
    ) {
        use super::ClipOp;
        if anti_alias {
            match op {
                ClipOp::Intersect => self.clip_rect_aa(rect, device_bounds),
                ClipOp::Difference => {
                    // Subtract ONLY the rect: the existing clip is retained
                    // and coverage inside the rect is zeroed (anti-aliased at
                    // its edges). No pre-intersection with the rect.
                    let sub = ClipMask::from_rect_aa(rect, device_bounds);
                    self.subtract_coverage(&sub, device_bounds);
                }
            }
        } else {
            self.clip_rect_op(rect, op);
        }
    }

    /// Subtract `sub`'s coverage from the current clip, leaving everything
    /// outside `sub` untouched: `new = current * (255 - sub) / 255` per pixel
    /// (rounded). Works uniformly for every [`ClipState`] variant; the result
    /// is a [`ClipState::Mask`] spanning `device_bounds`.
    fn subtract_coverage(&mut self, sub: &ClipMask, device_bounds: &IRect) {
        use crate::simd::mul_div_255_round;

        let width = device_bounds.width();
        let height = device_bounds.height();
        let mut new_mask = ClipMask::new(width, height, 0);
        new_mask.bounds = *device_bounds;
        for y in 0..height {
            for x in 0..width {
                let dx = x + device_bounds.left;
                let dy = y + device_bounds.top;
                let cur = self.current.get_coverage(dx, dy);
                if cur == 0 {
                    continue;
                }
                let s = u32::from(sub.get_coverage_device(dx, dy));
                new_mask.set_coverage(x, y, mul_div_255_round(u32::from(cur), 255 - s));
            }
        }
        self.current = ClipState::Mask(new_mask);
    }

    /// Subtract a device-space region from the current clip in every
    /// [`ClipState`] variant (non-AA difference).
    fn subtract_region(&mut self, sub: &Region) {
        use skia_rs_core::RegionOp;
        match &mut self.current {
            ClipState::Rect(r) => {
                let mut cur = Region::from_rect(r.round());
                cur.op_region(sub, RegionOp::Difference);
                self.current = ClipState::Region(cur);
            }
            ClipState::Region(r) => {
                r.op_region(sub, RegionOp::Difference);
            }
            ClipState::Mask(m) => {
                for y in 0..m.height {
                    for x in 0..m.width {
                        let dx = x + m.bounds.left;
                        let dy = y + m.bounds.top;
                        if sub.contains(dx, dy) {
                            m.set_coverage(x, y, 0);
                        }
                    }
                }
            }
            ClipState::RegionAndMask(r, m) => {
                r.op_region(sub, RegionOp::Difference);
                for y in 0..m.height {
                    for x in 0..m.width {
                        let dx = x + m.bounds.left;
                        let dy = y + m.bounds.top;
                        if sub.contains(dx, dy) {
                            m.set_coverage(x, y, 0);
                        }
                    }
                }
            }
        }
    }

    /// Intersect the current clip with a path.
    pub fn clip_path(&mut self, path: &Path, device_bounds: &IRect, anti_alias: bool) {
        if path.is_empty() {
            self.current.intersect_rect(&Rect::EMPTY);
            return;
        }

        if anti_alias {
            let mask = ClipMask::from_path_aa(path, device_bounds);
            match &mut self.current {
                ClipState::Rect(r) => {
                    let mut new_mask = mask;
                    new_mask.clip_rect(&r.round_out());
                    self.current = ClipState::Mask(new_mask);
                }
                ClipState::Region(r) => {
                    self.current = ClipState::RegionAndMask(r.clone(), mask);
                }
                ClipState::Mask(m) | ClipState::RegionAndMask(_, m) => {
                    m.intersect(&mask);
                }
            }
        } else {
            // Non-AA path clip: scan-convert the actual path into a region
            // (SkRegion::setPath), not merely its bounds.
            let region = crate::raster::path_to_region(path, device_bounds);
            self.current.intersect_region(&region);
        }
    }

    /// Intersect the current clip with a rounded rectangle.
    pub fn clip_round_rect(
        &mut self,
        rect: &Rect,
        rx: skia_rs_core::Scalar,
        ry: skia_rs_core::Scalar,
        device_bounds: &IRect,
        anti_alias: bool,
    ) {
        let mut builder = skia_rs_path::PathBuilder::new();
        builder.add_round_rect(rect, rx, ry);
        let path = builder.build();
        self.clip_path(&path, device_bounds, anti_alias);
    }

    /// Apply a clip operation with a path, optionally anti-aliased.
    pub fn clip_path_with_op(
        &mut self,
        path: &Path,
        device_bounds: &IRect,
        op: super::ClipOp,
        anti_alias: bool,
    ) {
        use super::ClipOp;
        match op {
            ClipOp::Intersect => self.clip_path(path, device_bounds, anti_alias),
            ClipOp::Difference => {
                // Rasterize the ACTUAL path coverage and subtract it from the
                // existing clip; works in every state combination.
                if anti_alias {
                    let sub = ClipMask::from_path_aa(path, device_bounds);
                    self.subtract_coverage(&sub, device_bounds);
                } else {
                    let sub = crate::raster::path_to_region(path, device_bounds);
                    self.subtract_region(&sub);
                }
            }
        }
    }

    /// Check if the current clip is anti-aliased.
    #[must_use]
    pub const fn is_anti_aliased(&self) -> bool {
        self.current.is_anti_aliased()
    }

    /// Reset the clip to device bounds.
    pub fn reset(&mut self, device_bounds: &Rect) {
        self.stack.clear();
        self.current = ClipState::Rect(*device_bounds);
    }

    /// Check if a rectangle is completely outside the current clip.
    #[must_use]
    pub fn quick_reject(&self, rect: &Rect) -> bool {
        let clip_bounds = self.bounds();
        !clip_bounds.intersects(rect)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_mask_new() {
        let mask = ClipMask::new(100, 100, 255);
        assert_eq!(mask.width(), 100);
        assert_eq!(mask.height(), 100);
        assert_eq!(mask.get_coverage(50, 50), 255);
    }

    #[test]
    fn test_clip_mask_from_rect_aa() {
        let rect = Rect::new(10.5, 10.5, 90.5, 90.5);
        let bounds = IRect::new(0, 0, 100, 100);
        let mask = ClipMask::from_rect_aa(&rect, &bounds);

        // Fully inside should be 255
        assert_eq!(mask.get_coverage(50, 50), 255);

        // Fully outside should be 0
        assert_eq!(mask.get_coverage(0, 0), 0);
        assert_eq!(mask.get_coverage(99, 99), 0);

        // Edge pixels should have partial coverage
        let edge_coverage = mask.get_coverage(10, 50);
        assert!(edge_coverage > 0 && edge_coverage < 255);
    }

    #[test]
    fn test_clip_state_rect() {
        let state = ClipState::from_rect(Rect::new(10.0, 10.0, 90.0, 90.0));

        assert!(state.contains(50, 50));
        assert!(!state.contains(0, 0));
        assert_eq!(state.get_coverage(50, 50), 255);
        assert_eq!(state.get_coverage(0, 0), 0);
    }

    #[test]
    fn test_clip_state_region() {
        let mut region = Region::new();
        region.set_rect(IRect::new(10, 10, 50, 50));
        region.op_rect(IRect::new(50, 50, 90, 90), skia_rs_core::RegionOp::Union);

        let state = ClipState::from_region(region);

        assert!(state.contains(25, 25)); // In first rect
        assert!(state.contains(75, 75)); // In second rect
        assert!(!state.contains(25, 75)); // Not in either
    }

    #[test]
    fn test_clip_stack_save_restore() {
        let mut stack = ClipStack::new(&Rect::new(0.0, 0.0, 100.0, 100.0));

        assert_eq!(stack.save_count(), 0);

        stack.save();
        assert_eq!(stack.save_count(), 1);

        stack.clip_rect(&Rect::new(10.0, 10.0, 50.0, 50.0));
        assert!(stack.contains(25, 25));
        assert!(!stack.contains(75, 75));

        stack.restore();
        assert_eq!(stack.save_count(), 0);
        assert!(stack.contains(75, 75)); // Original clip restored
    }

    #[test]
    fn test_clip_stack_nested() {
        let mut stack = ClipStack::new(&Rect::new(0.0, 0.0, 100.0, 100.0));

        stack.save();
        stack.clip_rect(&Rect::new(0.0, 0.0, 50.0, 100.0));

        stack.save();
        stack.clip_rect(&Rect::new(0.0, 0.0, 100.0, 50.0));

        // Should be clipped to intersection: (0,0) - (50,50)
        assert!(stack.contains(25, 25));
        assert!(!stack.contains(75, 25)); // Outside horizontal clip
        assert!(!stack.contains(25, 75)); // Outside vertical clip

        stack.restore();
        // Back to horizontal clip only
        assert!(stack.contains(25, 75));

        stack.restore();
        // Back to full bounds
        assert!(stack.contains(75, 75));
    }

    #[test]
    fn test_clip_region_integration() {
        let mut stack = ClipStack::new(&Rect::new(0.0, 0.0, 100.0, 100.0));

        let mut region = Region::new();
        region.set_rect(IRect::new(0, 0, 50, 50));
        region.op_rect(IRect::new(50, 50, 100, 100), skia_rs_core::RegionOp::Union);

        stack.clip_region(&region);

        // L-shaped region
        assert!(stack.contains(25, 25)); // Top-left
        assert!(stack.contains(75, 75)); // Bottom-right
        assert!(!stack.contains(75, 25)); // Top-right (not in region)
        assert!(!stack.contains(25, 75)); // Bottom-left (not in region)
    }

    #[test]
    fn test_compute_rect_coverage() {
        // Fully inside
        assert_eq!(compute_rect_coverage(5.0, 5.0, 0.0, 0.0, 10.0, 10.0), 255);

        // Fully outside
        assert_eq!(compute_rect_coverage(15.0, 15.0, 0.0, 0.0, 10.0, 10.0), 0);

        // Partial coverage (50% horizontal)
        let coverage = compute_rect_coverage(9.0, 5.0, 0.0, 0.0, 9.5, 10.0);
        assert!(coverage > 0 && coverage < 255);
    }

    #[test]
    fn test_non_aa_clip_path_scan_converts_actual_path() {
        use skia_rs_path::PathBuilder;

        // A triangle occupying the lower-left half of [10,10]-[90,90]. The
        // non-AA path clip must scan-convert the actual triangle, not its
        // bounding box.
        let device_bounds = IRect::new(0, 0, 100, 100);
        let mut stack = ClipStack::new(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0));

        let mut builder = PathBuilder::new();
        builder
            .move_to(10.0, 10.0)
            .line_to(10.0, 90.0)
            .line_to(90.0, 90.0)
            .close();
        let path = builder.build();
        stack.clip_path(&path, &device_bounds, false);

        // Inside the triangle.
        assert!(stack.contains(20, 80), "inside triangle must remain");
        // Inside the bounds but OUTSIDE the triangle (upper-right corner).
        assert!(
            !stack.contains(80, 20),
            "outside triangle (inside bounds) must be clipped"
        );
        // Outside the bounds entirely.
        assert!(!stack.contains(95, 95));
    }

    #[test]
    fn test_aa_difference_rect_keeps_clip_outside_rect() {
        use crate::ClipOp;

        // AA Difference must subtract ONLY the rect: the clip outside the
        // rect is retained (no pre-intersection with the rect).
        let device_bounds = IRect::new(0, 0, 100, 100);
        let mut stack = ClipStack::new(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        stack.clip_rect_with_op(
            &Rect::from_xywh(20.0, 20.0, 20.0, 20.0),
            ClipOp::Difference,
            &device_bounds,
            true,
        );

        // Inside the subtracted rect: clipped out.
        assert!(!stack.contains(30, 30));
        assert_eq!(stack.get_coverage(30, 30), 0);
        // Outside the subtracted rect: fully retained.
        assert!(stack.contains(70, 70), "clip outside difference rect kept");
        assert_eq!(stack.get_coverage(70, 70), 255);
        assert!(stack.contains(5, 5));
    }

    #[test]
    fn test_clip_path_difference_subtracts_actual_path_non_aa() {
        use crate::ClipOp;
        use skia_rs_path::PathBuilder;

        // Difference with a triangle: only the triangle is subtracted, not
        // its bounding box.
        let device_bounds = IRect::new(0, 0, 100, 100);
        let mut stack = ClipStack::new(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0));

        let mut builder = PathBuilder::new();
        builder
            .move_to(10.0, 10.0)
            .line_to(10.0, 90.0)
            .line_to(90.0, 90.0)
            .close();
        let path = builder.build();
        stack.clip_path_with_op(&path, &device_bounds, ClipOp::Difference, false);

        // Inside the triangle: subtracted.
        assert!(!stack.contains(20, 80));
        // Inside the triangle's bounds but outside the triangle: retained.
        assert!(
            stack.contains(80, 20),
            "bounds-but-not-path region must remain in the clip"
        );
        // Far outside: retained.
        assert!(stack.contains(95, 5));
    }

    #[test]
    fn test_clip_path_difference_subtracts_from_mask_state() {
        use crate::ClipOp;
        use skia_rs_path::PathBuilder;

        // Difference against a Mask state (AA clip already applied) must
        // subtract the path coverage — not silently no-op.
        let device_bounds = IRect::new(0, 0, 100, 100);
        let mut stack = ClipStack::new(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        // Force a Mask state via an AA rect clip.
        stack.clip_rect_aa(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0), &device_bounds);
        assert!(stack.is_anti_aliased());

        let mut builder = PathBuilder::new();
        builder.add_rect(&Rect::from_xywh(40.0, 40.0, 20.0, 20.0));
        let path = builder.build();
        stack.clip_path_with_op(&path, &device_bounds, ClipOp::Difference, false);

        assert!(!stack.contains(50, 50), "path area subtracted from mask");
        assert!(stack.contains(10, 10), "area outside path retained");
    }

    #[test]
    fn test_non_aa_clip_rect_rounds_to_nearest() {
        // Non-AA clip rects round to nearest (Skia rect.round()), not
        // round-out, and containment uses pixel centers.
        // (10.6, 10.6, 20.4, 20.4).round() == (11, 11, 20, 20).
        let mut stack = ClipStack::new(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        stack.clip_rect(&Rect::new(10.6, 10.6, 20.4, 20.4));

        assert!(!stack.contains(10, 15), "x=10 is left of rounded clip");
        assert!(stack.contains(11, 15), "x=11 is the first included column");
        assert!(stack.contains(19, 15), "x=19 is the last included column");
        assert!(!stack.contains(20, 15), "x=20 rounds out of the clip");
    }

    #[test]
    fn test_clip_rect_aa_preserves_region_on_region_and_mask() {
        use skia_rs_path::PathBuilder;

        // Create a clip stack with a RegionAndMask state and verify that
        // clip_rect_aa intersects both the region and the mask components
        let device_bounds = Rect::from_xywh(0.0, 0.0, 100.0, 100.0);
        let device_bounds_i = IRect::new(0, 0, 100, 100);
        let mut stack = ClipStack::new(&device_bounds);

        // Set up an initial rect clip
        let initial_rect = Rect::from_xywh(10.0, 10.0, 50.0, 50.0);
        stack.clip_rect(&initial_rect);

        // Apply a path clip to force RegionAndMask state
        let mut builder = PathBuilder::new();
        builder.add_rect(&Rect::from_xywh(15.0, 15.0, 40.0, 40.0));
        let path = builder.build();
        stack.clip_path(&path, &device_bounds_i, true);

        // Now apply an AA rect clip
        let aa_rect = Rect::from_xywh(20.0, 20.0, 30.0, 30.0);
        stack.clip_rect_aa(&aa_rect, &device_bounds_i);

        // Test points that should be clipped based on the region intersection
        // Point outside the AA rect but inside the mask should be clipped
        assert!(
            !stack.contains(5, 5),
            "Point outside AA rect should be clipped"
        );

        // Point inside the AA rect intersection should be included
        assert!(
            stack.contains(25, 25),
            "Point inside intersection should be included"
        );

        // Point outside the AA rect should be clipped even if mask would include it
        assert!(
            !stack.contains(55, 55),
            "Point outside AA rect should be clipped"
        );
    }
}
