//! Geometric primitives: points, sizes, rectangles, and matrices.
//!
//! This module provides Skia-compatible geometry types.

// The point/vector/matrix/cubic math here is a faithful port of Skia's
// fma-free reference arithmetic; rewriting `a * b + c` as `a.mul_add(b, c)`
// changes the rounding (single fused rounding) and diverges from Skia, so the
// nursery `suboptimal_flops` lint is allowed module-wide.
#![allow(
    clippy::suboptimal_flops,
    reason = "faithful port of Skia's fma-free reference arithmetic; mul_add's fused rounding diverges from Skia"
)]

use crate::Scalar;
use bytemuck::{Pod, Zeroable};

// Saturating float→int conversion helpers now live in `crate::cast`
// (`round_to_i32` / `floor_to_i32` / `ceil_to_i32`), shared across the
// workspace.
use crate::cast::{ceil_to_i32, floor_to_i32, round_to_i32, scalar_from_i32};

/// IEEE float divide that yields ±∞/NaN on divide-by-zero instead of trapping,
/// matching Skia's `sk_ieee_float_divide`.
#[inline]
fn ieee_float_divide(numer: Scalar, denom: Scalar) -> Scalar {
    numer / denom
}

/// Returns a copy of `rect` with left ≤ right and top ≤ bottom, matching Skia's
/// `SkRect::makeSorted`.
#[inline]
const fn sorted_rect(rect: Rect) -> Rect {
    Rect {
        left: rect.left.min(rect.right),
        top: rect.top.min(rect.bottom),
        right: rect.left.max(rect.right),
        bottom: rect.top.max(rect.bottom),
    }
}

// =============================================================================
// Point Types
// =============================================================================

/// A point with integer coordinates.
///
/// Equivalent to Skia's `SkIPoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct IPoint {
    /// X coordinate.
    pub x: i32,
    /// Y coordinate.
    pub y: i32,
}

impl IPoint {
    /// Creates a new point.
    #[inline]
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Returns the origin (0, 0).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Returns true if both coordinates are zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0
    }

    /// Negates both coordinates.
    #[inline]
    #[must_use]
    pub const fn negate(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }

    /// Offsets the point by (dx, dy).
    #[inline]
    #[must_use]
    pub const fn offset(&self, dx: i32, dy: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

/// A point with floating-point coordinates.
///
/// Equivalent to Skia's `SkPoint` / `SkVector`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Point {
    /// X coordinate.
    pub x: Scalar,
    /// Y coordinate.
    pub y: Scalar,
}

impl Point {
    /// Creates a new point.
    #[inline]
    #[must_use]
    pub const fn new(x: Scalar, y: Scalar) -> Self {
        Self { x, y }
    }

    /// Returns the origin (0, 0).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Returns true if both coordinates are zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }

    /// Returns true if either coordinate is NaN or infinite.
    #[inline]
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Negates both coordinates.
    #[inline]
    #[must_use]
    pub fn negate(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }

    /// Offsets the point by (dx, dy).
    #[inline]
    #[must_use]
    pub fn offset(&self, dx: Scalar, dy: Scalar) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }

    /// Returns the length of the vector from origin to this point.
    #[inline]
    #[must_use]
    pub fn length(&self) -> Scalar {
        self.x.hypot(self.y)
    }

    /// Returns the squared length (avoids sqrt).
    #[inline]
    #[must_use]
    pub fn length_squared(&self) -> Scalar {
        self.x * self.x + self.y * self.y
    }

    /// Returns a normalized (unit length) vector, or zero if length is zero.
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            Self::zero()
        }
    }

    /// Dot product with another point/vector.
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> Scalar {
        self.x * other.x + self.y * other.y
    }

    /// Cross product (returns the z-component of the 3D cross product).
    #[inline]
    #[must_use]
    pub fn cross(&self, other: &Self) -> Scalar {
        self.x * other.y - self.y * other.x
    }

    /// Returns the distance to another point.
    #[inline]
    #[must_use]
    pub fn distance(&self, other: &Self) -> Scalar {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx.hypot(dy)
    }

    /// Scales the point by a factor.
    #[inline]
    #[must_use]
    pub fn scale(&self, factor: Scalar) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }

    /// Linear interpolation between this point and another.
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: Self, t: Scalar) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl From<IPoint> for Point {
    #[inline]
    fn from(p: IPoint) -> Self {
        Self {
            x: scalar_from_i32(p.x),
            y: scalar_from_i32(p.y),
        }
    }
}

// Operator implementations for Point
impl std::ops::Add for Point {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::AddAssign for Point {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Point {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::SubAssign for Point {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl std::ops::Mul<Scalar> for Point {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::MulAssign<Scalar> for Point {
    #[inline]
    fn mul_assign(&mut self, rhs: Scalar) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl std::ops::Div<Scalar> for Point {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl std::ops::Neg for Point {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// A 3D point with floating-point coordinates.
///
/// Equivalent to Skia's `SkPoint3`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Point3 {
    /// X coordinate.
    pub x: Scalar,
    /// Y coordinate.
    pub y: Scalar,
    /// Z coordinate.
    pub z: Scalar,
}

impl Point3 {
    /// Creates a new 3D point.
    #[inline]
    #[must_use]
    pub const fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
        Self { x, y, z }
    }

    /// Returns the origin (0, 0, 0).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Returns the length of the vector.
    #[inline]
    #[must_use]
    pub fn length(&self) -> Scalar {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Dot product with another 3D point/vector.
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> Scalar {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[inline]
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

// =============================================================================
// Size Types
// =============================================================================

/// A size with integer dimensions.
///
/// Equivalent to Skia's `SkISize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct ISize {
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
}

impl ISize {
    /// Creates a new size.
    #[inline]
    #[must_use]
    pub const fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    /// Returns an empty size (0, 0).
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
        }
    }

    /// Returns true if width or height is <= 0.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// Returns the area (width * height).
    #[inline]
    #[must_use]
    pub const fn area(&self) -> i64 {
        self.width as i64 * self.height as i64
    }
}

/// A size with floating-point dimensions.
///
/// Equivalent to Skia's `SkSize`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Size {
    /// Width.
    pub width: Scalar,
    /// Height.
    pub height: Scalar,
}

impl Size {
    /// Creates a new size.
    #[inline]
    #[must_use]
    pub const fn new(width: Scalar, height: Scalar) -> Self {
        Self { width, height }
    }

    /// Returns an empty size (0, 0).
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            width: 0.0,
            height: 0.0,
        }
    }

    /// Returns true if width or height is <= 0.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Returns the area (width * height).
    #[inline]
    #[must_use]
    pub fn area(&self) -> Scalar {
        self.width * self.height
    }

    /// Converts to integer size by truncating.
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "documented truncating Size→ISize conversion; Rust's saturating float cast is the intended behavior"
    )]
    pub const fn to_isize(&self) -> ISize {
        ISize {
            width: self.width as i32,
            height: self.height as i32,
        }
    }

    /// Converts to integer size by rounding.
    ///
    /// Uses Skia's `SkScalarRoundToInt` (round half toward +∞, saturating).
    #[inline]
    #[must_use]
    pub fn to_isize_round(&self) -> ISize {
        ISize {
            width: round_to_i32(self.width),
            height: round_to_i32(self.height),
        }
    }
}

impl From<ISize> for Size {
    #[inline]
    fn from(s: ISize) -> Self {
        Self {
            width: scalar_from_i32(s.width),
            height: scalar_from_i32(s.height),
        }
    }
}

// =============================================================================
// Rectangle Types
// =============================================================================

/// A rectangle with integer coordinates.
///
/// Equivalent to Skia's `SkIRect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct IRect {
    /// Left edge.
    pub left: i32,
    /// Top edge.
    pub top: i32,
    /// Right edge.
    pub right: i32,
    /// Bottom edge.
    pub bottom: i32,
}

impl IRect {
    /// Creates a new rectangle from edges.
    #[inline]
    #[must_use]
    pub const fn new(left: i32, top: i32, right: i32, bottom: i32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Creates a rectangle from origin and size.
    ///
    /// Right/bottom edges are computed with saturating addition, matching
    /// Skia's `SkIRect::MakeXYWH` (`Sk32_sat_add`).
    #[inline]
    #[must_use]
    pub const fn from_xywh(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            left: x,
            top: y,
            right: x.saturating_add(width),
            bottom: y.saturating_add(height),
        }
    }

    /// Creates a rectangle from size (origin at 0,0).
    #[inline]
    #[must_use]
    pub const fn from_size(size: ISize) -> Self {
        Self {
            left: 0,
            top: 0,
            right: size.width,
            bottom: size.height,
        }
    }

    /// Returns an empty rectangle.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        }
    }

    /// Returns the width.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.right - self.left
    }

    /// Returns the height.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.bottom - self.top
    }

    /// Returns the size.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> ISize {
        ISize {
            width: self.width(),
            height: self.height(),
        }
    }

    /// Returns true if the rectangle has zero or negative area.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    /// Returns true if the point is inside the rectangle.
    #[inline]
    #[must_use]
    pub const fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    /// Returns the intersection of two rectangles, or None if they don't intersect.
    #[inline]
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right);
        let bottom = self.bottom.min(other.bottom);

        if left < right && top < bottom {
            Some(Self {
                left,
                top,
                right,
                bottom,
            })
        } else {
            None
        }
    }

    /// Returns the union (bounding box) of two rectangles.
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    /// Offsets the rectangle by (dx, dy).
    #[inline]
    #[must_use]
    pub const fn offset(&self, dx: i32, dy: i32) -> Self {
        Self {
            left: self.left + dx,
            top: self.top + dy,
            right: self.right + dx,
            bottom: self.bottom + dy,
        }
    }

    /// Convert to a floating-point Rect.
    #[inline]
    #[must_use]
    pub const fn to_rect(&self) -> Rect {
        Rect::new(
            scalar_from_i32(self.left),
            scalar_from_i32(self.top),
            scalar_from_i32(self.right),
            scalar_from_i32(self.bottom),
        )
    }

    /// Insets the rectangle by (dx, dy) on each side.
    #[inline]
    #[must_use]
    pub const fn inset(&self, dx: i32, dy: i32) -> Self {
        Self {
            left: self.left + dx,
            top: self.top + dy,
            right: self.right - dx,
            bottom: self.bottom - dy,
        }
    }
}

/// A rectangle with floating-point coordinates.
///
/// Equivalent to Skia's `SkRect`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Rect {
    /// Left edge.
    pub left: Scalar,
    /// Top edge.
    pub top: Scalar,
    /// Right edge.
    pub right: Scalar,
    /// Bottom edge.
    pub bottom: Scalar,
}

impl Rect {
    /// Empty rectangle constant.
    pub const EMPTY: Self = Self {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    /// Creates a new rectangle from edges.
    #[inline]
    #[must_use]
    pub const fn new(left: Scalar, top: Scalar, right: Scalar, bottom: Scalar) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Creates a rectangle from origin and size.
    #[inline]
    #[must_use]
    pub const fn from_xywh(x: Scalar, y: Scalar, width: Scalar, height: Scalar) -> Self {
        Self {
            left: x,
            top: y,
            right: x + width,
            bottom: y + height,
        }
    }

    /// Creates a rectangle from size (origin at 0,0).
    #[inline]
    #[must_use]
    pub const fn from_size(size: Size) -> Self {
        Self {
            left: 0.0,
            top: 0.0,
            right: size.width,
            bottom: size.height,
        }
    }

    /// Creates a rectangle from center and half-width/half-height.
    #[inline]
    #[must_use]
    pub fn from_center(center: Point, half_width: Scalar, half_height: Scalar) -> Self {
        Self {
            left: center.x - half_width,
            top: center.y - half_height,
            right: center.x + half_width,
            bottom: center.y + half_height,
        }
    }

    /// Returns an empty rectangle.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// Returns the width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> Scalar {
        self.right - self.left
    }

    /// Returns the height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> Scalar {
        self.bottom - self.top
    }

    /// Returns the size.
    #[inline]
    #[must_use]
    pub fn size(&self) -> Size {
        Size {
            width: self.width(),
            height: self.height(),
        }
    }

    /// Returns the center point.
    #[inline]
    #[must_use]
    pub const fn center(&self) -> Point {
        Point {
            x: f32::midpoint(self.left, self.right),
            y: f32::midpoint(self.top, self.bottom),
        }
    }

    /// Returns true if the rectangle has zero or negative area.
    ///
    /// Written as the negation of a non-empty rect (matching Skia's
    /// `SkRect::isEmpty`) so any NaN coordinate reports empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !(self.left < self.right && self.top < self.bottom)
    }

    /// Returns true if all coordinates are finite.
    #[inline]
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.left.is_finite()
            && self.top.is_finite()
            && self.right.is_finite()
            && self.bottom.is_finite()
    }

    /// Returns true if the point (x, y) is inside the rectangle.
    #[inline]
    #[must_use]
    pub fn contains_xy(&self, x: Scalar, y: Scalar) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    /// Returns true if the point is inside the rectangle.
    #[inline]
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        self.contains_xy(point.x, point.y)
    }

    /// Returns true if this rectangle contains the other rectangle.
    ///
    /// Returns false if either rectangle is empty, matching Skia's
    /// `SkRect::contains(const SkRect&)`.
    #[inline]
    #[must_use]
    pub fn contains_rect(&self, other: &Self) -> bool {
        !other.is_empty()
            && !self.is_empty()
            && self.left <= other.left
            && self.top <= other.top
            && self.right >= other.right
            && self.bottom >= other.bottom
    }

    /// Returns the intersection of two rectangles, or None if they don't intersect.
    #[inline]
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right);
        let bottom = self.bottom.min(other.bottom);

        if left < right && top < bottom {
            Some(Self {
                left,
                top,
                right,
                bottom,
            })
        } else {
            None
        }
    }

    /// Returns true if this rectangle intersects with another.
    #[inline]
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.left < other.right
            && other.left < self.right
            && self.top < other.bottom
            && other.top < self.bottom
    }

    /// Returns the union (bounding box) of two rectangles.
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    /// Alias for union - joins two rectangles into their bounding box.
    #[inline]
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        self.union(other)
    }

    /// Offsets the rectangle by (dx, dy).
    #[inline]
    #[must_use]
    pub fn offset(&self, dx: Scalar, dy: Scalar) -> Self {
        Self {
            left: self.left + dx,
            top: self.top + dy,
            right: self.right + dx,
            bottom: self.bottom + dy,
        }
    }

    /// Insets the rectangle by (dx, dy) on each side.
    #[inline]
    #[must_use]
    pub fn inset(&self, dx: Scalar, dy: Scalar) -> Self {
        Self {
            left: self.left + dx,
            top: self.top + dy,
            right: self.right - dx,
            bottom: self.bottom - dy,
        }
    }

    /// Rounds to the smallest enclosing integer rectangle.
    ///
    /// Uses Skia's saturating float→int casts (`SkRect::roundOut`).
    #[inline]
    #[must_use]
    pub fn round_out(&self) -> IRect {
        IRect {
            left: floor_to_i32(self.left),
            top: floor_to_i32(self.top),
            right: ceil_to_i32(self.right),
            bottom: ceil_to_i32(self.bottom),
        }
    }

    /// Rounds to the largest enclosed integer rectangle.
    ///
    /// Uses Skia's saturating float→int casts (`SkRect::roundIn`).
    #[inline]
    #[must_use]
    pub fn round_in(&self) -> IRect {
        IRect {
            left: ceil_to_i32(self.left),
            top: ceil_to_i32(self.top),
            right: floor_to_i32(self.right),
            bottom: floor_to_i32(self.bottom),
        }
    }

    /// Rounds to nearest integer rectangle.
    ///
    /// Uses Skia's `round_to_i32` (round half toward +∞, saturating), so
    /// NaN and out-of-range coordinates saturate rather than becoming zero.
    #[inline]
    #[must_use]
    pub fn round(&self) -> IRect {
        IRect {
            left: round_to_i32(self.left),
            top: round_to_i32(self.top),
            right: round_to_i32(self.right),
            bottom: round_to_i32(self.bottom),
        }
    }
}

impl From<IRect> for Rect {
    #[inline]
    fn from(r: IRect) -> Self {
        Self {
            left: scalar_from_i32(r.left),
            top: scalar_from_i32(r.top),
            right: scalar_from_i32(r.right),
            bottom: scalar_from_i32(r.bottom),
        }
    }
}

// =============================================================================
// Rounded Rectangle
// =============================================================================

/// A rectangle with rounded corners.
///
/// Equivalent to Skia's `SkRRect`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RRect {
    /// The bounding rectangle.
    pub rect: Rect,
    /// Corner radii: [top-left, top-right, bottom-right, bottom-left].
    /// Each corner has (x-radius, y-radius).
    pub radii: [Point; 4],
}

/// Corner indices for `RRect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Corner {
    /// Top-left corner.
    TopLeft = 0,
    /// Top-right corner.
    TopRight = 1,
    /// Bottom-right corner.
    BottomRight = 2,
    /// Bottom-left corner.
    BottomLeft = 3,
}

impl RRect {
    /// Creates a rounded rectangle with the same x/y radius for all corners.
    ///
    /// Matches Skia's `SkRRect::setRectXY`:
    /// - the rect is sorted and, if empty or non-finite, yields a square-corner
    ///   (or empty) rect rather than panicking;
    /// - oversized radii are scaled by the single factor
    ///   `min(w / (2·xRad), h / (2·yRad))`, preserving aspect ratio;
    /// - if **either** radius is ≤ 0, both become zero (square corners).
    #[must_use]
    pub fn from_rect_xy(rect: Rect, mut x_rad: Scalar, mut y_rad: Scalar) -> Self {
        // initializeRect: reject non-finite, sort, empty-handle.
        if !rect.is_finite() {
            return Self::default();
        }
        let rect = sorted_rect(rect);
        if rect.is_empty() {
            return Self::from_rect(rect);
        }

        // Devolve into a simple rect if the radii aren't finite.
        if !(x_rad.is_finite() && y_rad.is_finite()) {
            x_rad = 0.0;
            y_rad = 0.0;
        }

        // Proportionally scale down radii that don't fit, preserving aspect.
        let w = rect.width();
        let h = rect.height();
        if w < x_rad + x_rad || h < y_rad + y_rad {
            // At most one divide is by zero, and neither numerator is zero.
            let scale =
                ieee_float_divide(w, x_rad + x_rad).min(ieee_float_divide(h, y_rad + y_rad));
            x_rad *= scale;
            y_rad *= scale;
        }

        // If either radius collapsed to zero (or below), all corners are square.
        if x_rad <= 0.0 || y_rad <= 0.0 {
            return Self::from_rect(rect);
        }

        let radius = Point::new(x_rad, y_rad);
        Self {
            rect,
            radii: [radius, radius, radius, radius],
        }
    }

    /// Creates a rounded rectangle with a uniform radius.
    #[inline]
    #[must_use]
    pub fn from_rect_radius(rect: Rect, radius: Scalar) -> Self {
        Self::from_rect_xy(rect, radius, radius)
    }

    /// Creates a simple (non-rounded) rectangle.
    #[inline]
    #[must_use]
    pub const fn from_rect(rect: Rect) -> Self {
        Self {
            rect,
            radii: [Point::zero(); 4],
        }
    }

    /// Creates an oval inscribed in the rectangle.
    ///
    /// Matches Skia's `SkRRect::setOval`: the rect is sorted first, and the
    /// radii are half of the sorted dimensions (so inverted rects don't
    /// misbehave).
    #[inline]
    #[must_use]
    pub fn from_oval(rect: Rect) -> Self {
        if !rect.is_finite() {
            return Self::default();
        }
        let rect = sorted_rect(rect);
        let x_rad = rect.width() * 0.5;
        let y_rad = rect.height() * 0.5;
        Self::from_rect_xy(rect, x_rad, y_rad)
    }

    /// Returns the bounding rectangle.
    #[inline]
    #[must_use]
    pub const fn rect(&self) -> &Rect {
        &self.rect
    }

    /// Returns the radius for a specific corner.
    #[inline]
    #[must_use]
    pub const fn radius(&self, corner: Corner) -> Point {
        self.radii[corner as usize]
    }

    /// Returns true if all corners have zero radius.
    #[inline]
    #[must_use]
    pub fn is_rect(&self) -> bool {
        self.radii.iter().all(|r| r.x == 0.0 && r.y == 0.0)
    }

    /// Returns true if this is an oval (all corners have the same radius equal to half the dimensions).
    #[inline]
    #[must_use]
    pub fn is_oval(&self) -> bool {
        let x_rad = self.rect.width() * 0.5;
        let y_rad = self.rect.height() * 0.5;
        self.radii
            .iter()
            .all(|r| (r.x - x_rad).abs() < 1e-6 && (r.y - y_rad).abs() < 1e-6)
    }

    /// Returns true if all corners have the same radius.
    #[inline]
    #[must_use]
    pub fn is_simple(&self) -> bool {
        let first = self.radii[0];
        self.radii.iter().all(|r| *r == first)
    }
}

// =============================================================================
// Matrix (3x3)
// =============================================================================

/// A 3x3 transformation matrix.
///
/// Equivalent to Skia's `SkMatrix`.
///
/// The matrix is stored in row-major order:
/// ```text
/// | scale_x  skew_x   trans_x |
/// | skew_y   scale_y  trans_y |
/// | persp_0  persp_1  persp_2 |
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix {
    /// Matrix values in row-major order.
    pub values: [Scalar; 9],
}

impl Default for Matrix {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix {
    /// The identity matrix constant.
    pub const IDENTITY: Self = Self {
        values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    /// Index constants for matrix elements.
    pub const SCALE_X: usize = 0;
    /// Skew X index.
    pub const SKEW_X: usize = 1;
    /// Translate X index.
    pub const TRANS_X: usize = 2;
    /// Skew Y index.
    pub const SKEW_Y: usize = 3;
    /// Scale Y index.
    pub const SCALE_Y: usize = 4;
    /// Translate Y index.
    pub const TRANS_Y: usize = 5;
    /// Perspective 0 index.
    pub const PERSP_0: usize = 6;
    /// Perspective 1 index.
    pub const PERSP_1: usize = 7;
    /// Perspective 2 index.
    pub const PERSP_2: usize = 8;

    /// Creates the identity matrix.
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Creates a translation matrix.
    #[inline]
    #[must_use]
    pub const fn translate(dx: Scalar, dy: Scalar) -> Self {
        Self {
            values: [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a scale matrix.
    #[inline]
    #[must_use]
    pub const fn scale(sx: Scalar, sy: Scalar) -> Self {
        Self {
            values: [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a rotation matrix (angle in radians).
    #[inline]
    #[must_use]
    pub fn rotate(radians: Scalar) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            values: [cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a rotation matrix around a pivot point.
    #[inline]
    #[must_use]
    pub fn rotate_around(radians: Scalar, pivot: Point) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            values: [
                cos,
                -sin,
                pivot.x - pivot.x * cos + pivot.y * sin,
                sin,
                cos,
                pivot.y - pivot.x * sin - pivot.y * cos,
                0.0,
                0.0,
                1.0,
            ],
        }
    }

    /// Creates a skew matrix.
    ///
    /// Takes raw skew factors (not angles), matching Skia's `SkMatrix::MakeSkew`.
    /// To skew by an angle, pass `angle.tan()` as the parameter.
    #[inline]
    #[must_use]
    pub const fn skew(kx: Scalar, ky: Scalar) -> Self {
        Self {
            values: [1.0, kx, 0.0, ky, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Returns true if this is the identity matrix.
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::identity()
    }

    /// Returns true if the matrix only contains translation.
    #[inline]
    #[must_use]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison is intentional and matches Skia's exact SkScalar comparison"
    )]
    pub fn is_translate(&self) -> bool {
        self.values[Self::SCALE_X] == 1.0
            && self.values[Self::SKEW_X] == 0.0
            && self.values[Self::SKEW_Y] == 0.0
            && self.values[Self::SCALE_Y] == 1.0
            && self.values[Self::PERSP_0] == 0.0
            && self.values[Self::PERSP_1] == 0.0
            && self.values[Self::PERSP_2] == 1.0
    }

    /// Returns true if the matrix only contains scale and translation.
    #[inline]
    #[must_use]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison is intentional and matches Skia's exact SkScalar comparison"
    )]
    pub fn is_scale_translate(&self) -> bool {
        self.values[Self::SKEW_X] == 0.0
            && self.values[Self::SKEW_Y] == 0.0
            && self.values[Self::PERSP_0] == 0.0
            && self.values[Self::PERSP_1] == 0.0
            && self.values[Self::PERSP_2] == 1.0
    }

    /// Returns the translation component.
    #[inline]
    #[must_use]
    pub const fn translation(&self) -> Point {
        Point {
            x: self.values[Self::TRANS_X],
            y: self.values[Self::TRANS_Y],
        }
    }

    /// Returns the X scale factor.
    #[inline]
    #[must_use]
    pub const fn scale_x(&self) -> Scalar {
        self.values[Self::SCALE_X]
    }

    /// Returns the Y scale factor.
    #[inline]
    #[must_use]
    pub const fn scale_y(&self) -> Scalar {
        self.values[Self::SCALE_Y]
    }

    /// Returns the X skew factor.
    #[inline]
    #[must_use]
    pub const fn skew_x(&self) -> Scalar {
        self.values[Self::SKEW_X]
    }

    /// Returns the Y skew factor.
    #[inline]
    #[must_use]
    pub const fn skew_y(&self) -> Scalar {
        self.values[Self::SKEW_Y]
    }

    /// Concatenates this matrix with another (self * other).
    #[inline]
    #[must_use]
    pub fn concat(&self, other: &Self) -> Self {
        let a = &self.values;
        let b = &other.values;
        Self {
            values: [
                a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
                a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
                a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
                a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
                a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
                a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
                a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
                a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
                a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
            ],
        }
    }

    /// Transforms a point by this matrix.
    ///
    /// For perspective transformations (when the bottom row is not [0, 0, 1]),
    /// the result is the projected point. If the perspective division would
    /// produce w == 0, returns (0, 0) to avoid producing NaN or infinity.
    #[inline]
    #[must_use]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison is intentional and matches Skia's exact SkScalar comparison"
    )]
    pub fn map_point(&self, point: Point) -> Point {
        let m = &self.values;
        let x = m[0] * point.x + m[1] * point.y + m[2];
        let y = m[3] * point.x + m[4] * point.y + m[5];

        // Handle perspective
        if m[6] != 0.0 || m[7] != 0.0 || m[8] != 1.0 {
            let w = m[6] * point.x + m[7] * point.y + m[8];
            if w == 0.0 {
                Point { x: 0.0, y: 0.0 }
            } else {
                let w_inv = 1.0 / w;
                Point {
                    x: x * w_inv,
                    y: y * w_inv,
                }
            }
        } else {
            Point { x, y }
        }
    }

    /// Transforms a rectangle by this matrix (returns bounding box of transformed corners).
    #[inline]
    #[must_use]
    pub fn map_rect(&self, rect: &Rect) -> Rect {
        let corners = [
            self.map_point(Point::new(rect.left, rect.top)),
            self.map_point(Point::new(rect.right, rect.top)),
            self.map_point(Point::new(rect.right, rect.bottom)),
            self.map_point(Point::new(rect.left, rect.bottom)),
        ];

        let mut min_x = corners[0].x;
        let mut min_y = corners[0].y;
        let mut max_x = corners[0].x;
        let mut max_y = corners[0].y;

        for p in &corners[1..] {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        Rect::new(min_x, min_y, max_x, max_y)
    }

    /// Computes the determinant.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> Scalar {
        let m = &self.values;
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    /// Computes the inverse matrix, or None if singular.
    ///
    /// Matches Skia's `sk_inv_determinant` (`src/core/SkMatrix.cpp`): the
    /// determinant is computed in double precision and only rejected when it
    /// falls at or below `SK_ScalarNearlyZero` **cubed** (`(1/4096)³ ≈
    /// 1.45e-11`), since the determinant scales with the cube of the matrix
    /// members. The computed inverse is additionally rejected if any element is
    /// non-finite (`SkMatrix::invert` checks `inv.isFinite()`).
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "deliberate f64→f32 narrowing that mirrors Skia's determinant precision handling"
    )]
    pub fn invert(&self) -> Option<Self> {
        // SK_ScalarNearlyZero == 1/4096; the tolerance is that value cubed.
        const NEARLY_ZERO: f64 = 1.0 / 4096.0;
        const THRESHOLD: f64 = NEARLY_ZERO * NEARLY_ZERO * NEARLY_ZERO;
        let m = &self.values;

        // Compute the determinant in f64 to match Skia's precision.
        let det = f64::from(m[0])
            * (f64::from(m[4]) * f64::from(m[8]) - f64::from(m[5]) * f64::from(m[7]))
            - f64::from(m[1])
                * (f64::from(m[3]) * f64::from(m[8]) - f64::from(m[5]) * f64::from(m[6]))
            + f64::from(m[2])
                * (f64::from(m[3]) * f64::from(m[7]) - f64::from(m[4]) * f64::from(m[6]));

        // Skia compares the f32-narrowed determinant against the tolerance.
        if f64::from((det as f32).abs()) <= THRESHOLD {
            return None;
        }

        let inv_det = (1.0 / det) as Scalar;

        let inv = Self {
            values: [
                (m[4] * m[8] - m[5] * m[7]) * inv_det,
                (m[2] * m[7] - m[1] * m[8]) * inv_det,
                (m[1] * m[5] - m[2] * m[4]) * inv_det,
                (m[5] * m[6] - m[3] * m[8]) * inv_det,
                (m[0] * m[8] - m[2] * m[6]) * inv_det,
                (m[2] * m[3] - m[0] * m[5]) * inv_det,
                (m[3] * m[7] - m[4] * m[6]) * inv_det,
                (m[1] * m[6] - m[0] * m[7]) * inv_det,
                (m[0] * m[4] - m[1] * m[3]) * inv_det,
            ],
        };

        // Reject a non-finite inverse, matching SkMatrix::invert's isFinite check.
        if inv.values.iter().all(|v| v.is_finite()) {
            Some(inv)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        reason = "tests assert exact expected f32 results; exact equality is intentional here"
    )]
    use super::*;

    #[test]
    fn test_point_operations() {
        let p1 = Point::new(3.0, 4.0);
        assert!((p1.length() - 5.0).abs() < 1e-6);

        let p2 = Point::new(1.0, 2.0);
        assert!((p1.dot(&p2) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_rect_intersection() {
        let r1 = Rect::new(0.0, 0.0, 10.0, 10.0);
        let r2 = Rect::new(5.0, 5.0, 15.0, 15.0);
        let intersection = r1.intersect(&r2).unwrap();
        assert_eq!(intersection, Rect::new(5.0, 5.0, 10.0, 10.0));
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity();
        let p = Point::new(5.0, 7.0);
        let result = m.map_point(p);
        assert_eq!(result, p);
    }

    #[test]
    fn test_matrix_translate() {
        let m = Matrix::translate(10.0, 20.0);
        let p = Point::new(5.0, 7.0);
        let result = m.map_point(p);
        assert_eq!(result, Point::new(15.0, 27.0));
    }

    #[test]
    fn test_matrix_inverse() {
        let m = Matrix::translate(10.0, 20.0);
        let inv = m.invert().unwrap();
        let result = m.concat(&inv);
        assert!((result.values[0] - 1.0).abs() < 1e-6);
        assert!((result.values[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_map_point_perspective_zero_w() {
        // Matrix that produces w=0 for the origin point.
        // perspective row: w = x + y + 0 = 0 at (0, 0).
        let matrix = Matrix {
            values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        };
        let point = matrix.map_point(Point::new(0.0, 0.0));
        // Should return (0, 0) instead of NaN/inf
        assert!(point.x.is_finite(), "x should be finite, got {}", point.x);
        assert!(point.y.is_finite(), "y should be finite, got {}", point.y);
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
    }

    #[test]
    fn test_matrix_skew_raw_factors() {
        // Skew with kx=0.5, ky=0.0 should produce a matrix where
        // a point (0, 1) maps to (0.5, 1) - x is shifted by 0.5*y
        let matrix = Matrix::skew(0.5, 0.0);
        let point = matrix.map_point(Point::new(0.0, 1.0));
        assert!(
            (point.x - 0.5).abs() < 1e-6,
            "Expected x=0.5, got {}",
            point.x
        );
        assert!(
            (point.y - 1.0).abs() < 1e-6,
            "Expected y=1.0, got {}",
            point.y
        );

        // Skew matrix values should match: [1, 0.5, 0, 0, 1, 0, 0, 0, 1]
        let m = matrix.values;
        assert_eq!(m[0], 1.0);
        assert!((m[1] - 0.5).abs() < 1e-6);
        assert_eq!(m[2], 0.0);
        assert_eq!(m[3], 0.0);
        assert_eq!(m[4], 1.0);
        assert_eq!(m[5], 0.0);
    }

    #[test]
    fn test_matrix_map_point_perspective_normal() {
        // Verify normal perspective division still works.
        // w = 2 for all points (persp_2 = 2, persp_0/1 = 0).
        let matrix = Matrix {
            values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        };
        let point = matrix.map_point(Point::new(4.0, 6.0));
        // x/w = 4/2 = 2, y/w = 6/2 = 3
        assert_eq!(point.x, 2.0);
        assert_eq!(point.y, 3.0);
    }

    #[test]
    fn test_rrect_from_rect_xy_clamps_radii() {
        let rect = Rect::new(0.0, 0.0, 100.0, 50.0);

        // Radii larger than half-dimensions should be clamped
        let rounded = RRect::from_rect_xy(rect, 60.0, 30.0);
        assert_eq!(rounded.radii[0].x, 50.0, "rx should clamp to width/2 = 50");
        assert_eq!(rounded.radii[0].y, 25.0, "ry should clamp to height/2 = 25");
    }

    #[test]
    fn test_rrect_from_rect_xy_negative_radii_clamped_to_zero() {
        let rect = Rect::new(0.0, 0.0, 100.0, 50.0);

        let rounded = RRect::from_rect_xy(rect, -5.0, -10.0);
        assert_eq!(rounded.radii[0].x, 0.0);
        assert_eq!(rounded.radii[0].y, 0.0);
    }

    #[test]
    fn test_rrect_from_rect_xy_normal_radii_unchanged() {
        let rect = Rect::new(0.0, 0.0, 100.0, 50.0);

        let rounded = RRect::from_rect_xy(rect, 10.0, 5.0);
        assert_eq!(rounded.radii[0].x, 10.0);
        assert_eq!(rounded.radii[0].y, 5.0);
    }

    #[test]
    fn test_matrix_invert_uses_skia_threshold() {
        // Skia rejects only when |det| <= (1/4096)^3 ≈ 1.45e-11. A det of 1e-6
        // is far above that, so this matrix must invert (the old f32 threshold
        // of EPSILON*256 wrongly rejected it).
        let small = Matrix {
            values: [
                1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, // det = 1e-6
                0.0, 0.0, 1.0,
            ],
        };
        let inv = small.invert().expect("det 1e-6 is above Skia's threshold");
        assert!((inv.values[0] - 1000.0).abs() < 1.0);

        // A determinant below the cubed nearly-zero constant is singular.
        let singular = Matrix {
            values: [
                1e-5, 0.0, 0.0, 0.0, 1e-7, 0.0, // det = 1e-12 < 1.45e-11
                0.0, 0.0, 1.0,
            ],
        };
        assert!(
            singular.invert().is_none(),
            "det 1e-12 is below Skia's (1/4096)^3 threshold"
        );

        // Well above threshold - should invert successfully.
        let invertible = Matrix {
            values: [
                0.1, 0.0, 0.0, 0.0, 0.1, 0.0, // det = 0.01
                0.0, 0.0, 1.0,
            ],
        };
        assert!(
            invertible.invert().is_some(),
            "Matrix with determinant 0.01 should be invertible"
        );
    }

    // --- Conformance regression tests (Task 1) ---

    #[test]
    fn test_rect_is_empty_nan() {
        // A rect with a NaN coordinate must report empty (Skia SkRect::isEmpty).
        let nan_rect = Rect::new(0.0, 0.0, Scalar::NAN, 10.0);
        assert!(nan_rect.is_empty(), "NaN rect must be empty");
        // ...and must not survive a union with a real rect.
        let real = Rect::new(1.0, 1.0, 5.0, 5.0);
        assert_eq!(nan_rect.union(&real), real);
        assert_eq!(real.union(&nan_rect), real);
        // Ordinary non-empty and empty rects still behave.
        assert!(!Rect::new(0.0, 0.0, 2.0, 2.0).is_empty());
        assert!(Rect::new(2.0, 0.0, 2.0, 2.0).is_empty());
    }

    #[test]
    fn test_rect_contains_rect_empty() {
        let outer = Rect::new(0.0, 0.0, 10.0, 10.0);
        // An empty inner rect is never contained.
        let empty_inner = Rect::new(5.0, 5.0, 5.0, 5.0);
        assert!(!outer.contains_rect(&empty_inner));
        // An empty outer rect contains nothing.
        let empty_outer = Rect::new(5.0, 5.0, 5.0, 5.0);
        assert!(!empty_outer.contains_rect(&Rect::new(5.0, 5.0, 6.0, 6.0)));
        // Genuine containment still works.
        assert!(outer.contains_rect(&Rect::new(1.0, 1.0, 9.0, 9.0)));
    }

    #[test]
    fn test_matrix_invert_small_scale() {
        // scale(0.005, 0.005) has determinant 2.5e-5, far above (1/4096)^3.
        let m = Matrix::scale(0.005, 0.005);
        let inv = m.invert().expect("small-scale matrix must invert");
        // Inverse of a 0.005 scale is a 200x scale.
        assert!((inv.values[0] - 200.0).abs() < 1e-2);
        assert!((inv.values[4] - 200.0).abs() < 1e-2);
    }

    #[test]
    fn test_matrix_invert_singular() {
        // A truly singular matrix (zero scale) must not invert.
        let m = Matrix::scale(0.0, 0.0);
        assert!(m.invert().is_none());
    }

    #[test]
    fn test_rect_round_saturates_nan_and_range() {
        // NaN saturates to the max s32-in-float rather than becoming 0.
        let r = Rect::new(Scalar::NAN, 0.5, 2.5, 3.5);
        let ir = r.round();
        assert_eq!(ir.left, 2_147_483_520);
        // 0.5 rounds toward +inf -> 1; 2.5 -> 3; 3.5 -> 4.
        assert_eq!(ir.top, 1);
        assert_eq!(ir.right, 3);
        assert_eq!(ir.bottom, 4);
        // Out-of-range positive saturates.
        let big = Rect::new(0.0, 0.0, 1e30, 1e30);
        assert_eq!(big.round().right, 2_147_483_520);
    }

    #[test]
    fn test_size_to_isize_round_saturates() {
        assert_eq!(Size::new(2.5, 3.5).to_isize_round(), ISize::new(3, 4));
        assert_eq!(
            Size::new(Scalar::NAN, 1.0).to_isize_round().width,
            2_147_483_520
        );
    }

    #[test]
    fn test_irect_from_xywh_saturates() {
        // x + width overflows i32; must saturate, not wrap/panic.
        let r = IRect::from_xywh(i32::MAX - 1, 0, 100, 10);
        assert_eq!(r.right, i32::MAX);
        assert_eq!(r.bottom, 10);
    }

    #[test]
    fn test_rrect_from_rect_xy_inverted_rect() {
        // Inverted rect must be sorted instead of panicking on clamp(max<min).
        let inverted = Rect::new(10.0, 10.0, 0.0, 0.0);
        let rr = RRect::from_rect_xy(inverted, 2.0, 2.0);
        assert_eq!(*rr.rect(), Rect::new(0.0, 0.0, 10.0, 10.0));
        assert_eq!(rr.radius(Corner::TopLeft), Point::new(2.0, 2.0));
    }

    #[test]
    fn test_rrect_from_rect_xy_oversized_radii_preserve_aspect() {
        // rect 40 wide, 20 tall; radii 30x30 don't fit. Scale factor =
        // min(40/60, 20/60) = 1/3, so both radii scale to 10, preserving aspect.
        let rect = Rect::new(0.0, 0.0, 40.0, 20.0);
        let rr = RRect::from_rect_xy(rect, 30.0, 30.0);
        let r = rr.radius(Corner::TopLeft);
        assert!((r.x - 10.0).abs() < 1e-4, "x radius {} != 10", r.x);
        assert!((r.y - 10.0).abs() < 1e-4, "y radius {} != 10", r.y);
    }

    #[test]
    fn test_rrect_from_oval_sorts_inverted_rect() {
        // An inverted oval rect must sort and still produce a proper oval,
        // not a square-cornered rect.
        let inverted = Rect::new(20.0, 10.0, 0.0, 0.0);
        let rr = RRect::from_oval(inverted);
        assert_eq!(*rr.rect(), Rect::new(0.0, 0.0, 20.0, 10.0));
        assert_eq!(rr.radius(Corner::TopLeft), Point::new(10.0, 5.0));
        assert!(rr.is_oval());
    }

    #[test]
    fn test_rrect_from_rect_xy_zero_radius_makes_square() {
        // If either radius <= 0, both corners become square.
        let rect = Rect::new(0.0, 0.0, 40.0, 20.0);
        let rr = RRect::from_rect_xy(rect, 5.0, 0.0);
        assert!(rr.is_rect(), "one zero radius must yield square corners");
    }
}
