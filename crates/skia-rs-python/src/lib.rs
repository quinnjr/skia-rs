//! Python bindings for skia-rs.
//!
//! This crate provides Python bindings using `PyO3`.
//!
//! # Installation
//!
//! ```bash
//! pip install skia-rs
//! # or build from source:
//! maturin develop --release
//! ```
//!
//! # Example
//!
//! ```python
//! import skia_rs
//!
//! # Create a surface
//! surface = skia_rs.Surface(800, 600)
//!
//! # Create a paint
//! paint = skia_rs.Paint()
//! paint.color = 0xFFFF0000  # Red
//! paint.anti_alias = True
//!
//! # Draw
//! surface.clear(0xFFFFFFFF)  # White
//! surface.draw_circle(400, 300, 100, paint)
//!
//! # Save to file
//! surface.save_png("output.png")
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use skia_rs_canvas::Surface as RsSurface;
use skia_rs_core::{Color, Matrix as RsMatrix, Point as RsPoint, Rect as RsRect};
use skia_rs_paint::{Paint as RsPaint, Style as RsStyle};
use skia_rs_path::{Path as RsPath, PathBuilder as RsPathBuilder};

// =============================================================================
// Point
// =============================================================================

/// A 2D point with x and y coordinates.
#[pyclass]
#[derive(Clone, Copy)]
pub struct Point {
    inner: RsPoint,
}

#[pymethods]
impl Point {
    /// Create a new point.
    #[new]
    const fn new(x: f32, y: f32) -> Self {
        Self {
            inner: RsPoint::new(x, y),
        }
    }

    /// X coordinate.
    // pyo3 requires `&self` (not by-value `self`) for pymethods receivers,
    // since the underlying Python object is shared and cannot be moved out
    // of the interpreter; this holds even though `Point` is `Copy`.
    #[getter]
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    const fn x(&self) -> f32 {
        self.inner.x
    }

    #[setter]
    const fn set_x(&mut self, x: f32) {
        self.inner.x = x;
    }

    /// Y coordinate.
    #[getter]
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    const fn y(&self) -> f32 {
        self.inner.y
    }

    #[setter]
    const fn set_y(&mut self, y: f32) {
        self.inner.y = y;
    }

    /// Calculate the length of the vector from origin.
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    fn length(&self) -> f32 {
        self.inner.length()
    }

    /// Normalize the point (unit vector).
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    fn normalize(&self) -> Self {
        Self {
            inner: self.inner.normalize(),
        }
    }

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    fn __repr__(&self) -> String {
        format!("Point({}, {})", self.inner.x, self.inner.y)
    }

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    fn __add__(&self, other: Self) -> Self {
        Self {
            inner: RsPoint::new(self.inner.x + other.inner.x, self.inner.y + other.inner.y),
        }
    }

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    fn __sub__(&self, other: Self) -> Self {
        Self {
            inner: RsPoint::new(self.inner.x - other.inner.x, self.inner.y - other.inner.y),
        }
    }

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "pyo3 pymethods receivers must be `&self`/`&mut self`, not by-value, regardless of Copy"
    )]
    fn __mul__(&self, scalar: f32) -> Self {
        Self {
            inner: RsPoint::new(self.inner.x * scalar, self.inner.y * scalar),
        }
    }
}

// =============================================================================
// Rect
// =============================================================================

/// A rectangle defined by left, top, right, bottom edges.
#[pyclass]
#[derive(Clone, Copy)]
pub struct Rect {
    inner: RsRect,
}

#[pymethods]
impl Rect {
    /// Create a new rectangle from edges.
    #[new]
    const fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            inner: RsRect::new(left, top, right, bottom),
        }
    }

    /// Create a rectangle from position and size.
    #[staticmethod]
    const fn from_xywh(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            inner: RsRect::from_xywh(x, y, width, height),
        }
    }

    /// Create a rectangle from size (at origin).
    #[staticmethod]
    const fn from_wh(width: f32, height: f32) -> Self {
        Self {
            inner: RsRect::from_xywh(0.0, 0.0, width, height),
        }
    }

    #[getter]
    const fn left(&self) -> f32 {
        self.inner.left
    }

    #[getter]
    const fn top(&self) -> f32 {
        self.inner.top
    }

    #[getter]
    const fn right(&self) -> f32 {
        self.inner.right
    }

    #[getter]
    const fn bottom(&self) -> f32 {
        self.inner.bottom
    }

    #[getter]
    fn width(&self) -> f32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> f32 {
        self.inner.height()
    }

    /// Check if the rectangle is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if a point is inside the rectangle.
    fn contains(&self, x: f32, y: f32) -> bool {
        self.inner.contains(RsPoint::new(x, y))
    }

    /// Get the center of the rectangle.
    const fn center(&self) -> Point {
        let c = self.inner.center();
        Point { inner: c }
    }

    /// Expand the rectangle to include a point.
    const fn join(&self, x: f32, y: f32) -> Self {
        Self {
            inner: RsRect::new(
                self.inner.left.min(x),
                self.inner.top.min(y),
                self.inner.right.max(x),
                self.inner.bottom.max(y),
            ),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Rect({}, {}, {}, {})",
            self.inner.left, self.inner.top, self.inner.right, self.inner.bottom
        )
    }
}

// =============================================================================
// Matrix
// =============================================================================

/// A 3x3 transformation matrix.
#[pyclass]
#[derive(Clone)]
pub struct Matrix {
    inner: RsMatrix,
}

#[pymethods]
impl Matrix {
    /// Create an identity matrix.
    #[new]
    const fn new() -> Self {
        Self {
            inner: RsMatrix::IDENTITY,
        }
    }

    /// Create a translation matrix.
    #[staticmethod]
    const fn translate(dx: f32, dy: f32) -> Self {
        Self {
            inner: RsMatrix::translate(dx, dy),
        }
    }

    /// Create a scale matrix.
    #[staticmethod]
    const fn scale(sx: f32, sy: f32) -> Self {
        Self {
            inner: RsMatrix::scale(sx, sy),
        }
    }

    /// Create a rotation matrix (radians).
    #[staticmethod]
    fn rotate(radians: f32) -> Self {
        Self {
            inner: RsMatrix::rotate(radians),
        }
    }

    /// Create a rotation matrix (degrees).
    #[staticmethod]
    fn rotate_deg(degrees: f32) -> Self {
        let radians = degrees.to_radians();
        Self {
            inner: RsMatrix::rotate(radians),
        }
    }

    /// Concatenate with another matrix.
    fn concat(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.concat(&other.inner),
        }
    }

    /// Invert the matrix.
    fn invert(&self) -> Option<Self> {
        self.inner.invert().map(|m| Self { inner: m })
    }

    /// Transform a point.
    fn map_point(&self, x: f32, y: f32) -> Point {
        let p = self.inner.map_point(RsPoint::new(x, y));
        Point { inner: p }
    }

    fn __repr__(&self) -> String {
        format!("Matrix([{:?}])", self.inner.values)
    }
}

// =============================================================================
// Paint
// =============================================================================

/// Paint controls styling for drawing operations.
#[pyclass]
pub struct Paint {
    inner: RsPaint,
}

#[pymethods]
impl Paint {
    /// Create a new paint with default settings.
    #[new]
    fn new() -> Self {
        Self {
            inner: RsPaint::new(),
        }
    }

    /// Color as ARGB integer.
    #[getter]
    fn color(&self) -> u32 {
        self.inner.color32().0
    }

    #[setter]
    fn set_color(&mut self, color: u32) {
        self.inner.set_color32(Color(color));
    }

    /// Set color from RGBA components (0-255).
    fn set_argb(&mut self, a: u8, r: u8, g: u8, b: u8) {
        self.inner.set_color32(Color::from_argb(a, r, g, b));
    }

    /// Style: "fill", "stroke", or `"stroke_and_fill"`.
    #[getter]
    const fn style(&self) -> &'static str {
        match self.inner.style() {
            RsStyle::Fill => "fill",
            RsStyle::Stroke => "stroke",
            RsStyle::StrokeAndFill => "stroke_and_fill",
        }
    }

    #[setter]
    fn set_style(&mut self, style: &str) -> PyResult<()> {
        let s = match style {
            "fill" => RsStyle::Fill,
            "stroke" => RsStyle::Stroke,
            "stroke_and_fill" => RsStyle::StrokeAndFill,
            _ => return Err(PyValueError::new_err("Invalid style")),
        };
        self.inner.set_style(s);
        Ok(())
    }

    /// Stroke width.
    #[getter]
    const fn stroke_width(&self) -> f32 {
        self.inner.stroke_width()
    }

    #[setter]
    const fn set_stroke_width(&mut self, width: f32) {
        self.inner.set_stroke_width(width);
    }

    /// Anti-aliasing enabled.
    #[getter]
    const fn anti_alias(&self) -> bool {
        self.inner.is_anti_alias()
    }

    #[setter]
    const fn set_anti_alias(&mut self, aa: bool) {
        self.inner.set_anti_alias(aa);
    }

    /// Alpha (0-255).
    #[getter]
    fn alpha(&self) -> u8 {
        skia_rs_core::cast::f32_to_u8_sat(self.inner.alpha() * 255.0)
    }

    #[setter]
    fn set_alpha(&mut self, alpha: u8) {
        self.inner.set_alpha(f32::from(alpha) / 255.0);
    }

    fn __repr__(&self) -> String {
        format!(
            "Paint(color=0x{:08X}, style={}, stroke_width={})",
            self.inner.color32().0,
            self.style(),
            self.inner.stroke_width()
        )
    }
}

// =============================================================================
// PathBuilder
// =============================================================================

/// Builder for constructing paths.
#[pyclass]
pub struct PathBuilder {
    inner: RsPathBuilder,
}

#[pymethods]
impl PathBuilder {
    /// Create a new path builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: RsPathBuilder::new(),
        }
    }

    /// Move to a point.
    fn move_to(mut self_: PyRefMut<'_, Self>, x: f32, y: f32) -> PyRefMut<'_, Self> {
        self_.inner.move_to(x, y);
        self_
    }

    /// Line to a point.
    fn line_to(mut self_: PyRefMut<'_, Self>, x: f32, y: f32) -> PyRefMut<'_, Self> {
        self_.inner.line_to(x, y);
        self_
    }

    /// Quadratic bezier curve.
    fn quad_to(
        mut self_: PyRefMut<'_, Self>,
        cx: f32,
        cy: f32,
        x: f32,
        y: f32,
    ) -> PyRefMut<'_, Self> {
        self_.inner.quad_to(cx, cy, x, y);
        self_
    }

    /// Cubic bezier curve.
    fn cubic_to(
        mut self_: PyRefMut<'_, Self>,
        c1x: f32,
        c1y: f32,
        c2x: f32,
        c2y: f32,
        x: f32,
        y: f32,
    ) -> PyRefMut<'_, Self> {
        self_.inner.cubic_to(c1x, c1y, c2x, c2y, x, y);
        self_
    }

    /// Close the current contour.
    fn close(mut self_: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        self_.inner.close();
        self_
    }

    /// Add a rectangle.
    fn add_rect(
        mut self_: PyRefMut<'_, Self>,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    ) -> PyRefMut<'_, Self> {
        self_.inner.add_rect(&RsRect::new(left, top, right, bottom));
        self_
    }

    /// Add an oval inscribed in a rectangle.
    fn add_oval(
        mut self_: PyRefMut<'_, Self>,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    ) -> PyRefMut<'_, Self> {
        self_.inner.add_oval(&RsRect::new(left, top, right, bottom));
        self_
    }

    /// Add a circle.
    fn add_circle(
        mut self_: PyRefMut<'_, Self>,
        cx: f32,
        cy: f32,
        radius: f32,
    ) -> PyRefMut<'_, Self> {
        self_.inner.add_circle(cx, cy, radius);
        self_
    }

    /// Add a rounded rectangle.
    fn add_round_rect(
        mut self_: PyRefMut<'_, Self>,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        rx: f32,
        ry: f32,
    ) -> PyRefMut<'_, Self> {
        self_
            .inner
            .add_round_rect(&RsRect::new(left, top, right, bottom), rx, ry);
        self_
    }

    /// Build the path.
    fn build(&self) -> Path {
        Path {
            inner: self.inner.clone().build(),
        }
    }

    /// Reset the builder.
    fn reset(&mut self) {
        self.inner = RsPathBuilder::new();
    }
}

// =============================================================================
// Path
// =============================================================================

/// An immutable path containing geometry.
#[pyclass]
#[derive(Clone)]
pub struct Path {
    inner: RsPath,
}

#[pymethods]
impl Path {
    /// Create an empty path.
    #[new]
    fn new() -> Self {
        Self {
            inner: RsPath::new(),
        }
    }

    /// Check if the path is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the bounding box.
    fn bounds(&self) -> Rect {
        Rect {
            inner: self.inner.bounds(),
        }
    }

    /// Check if a point is inside the path.
    fn contains(&self, x: f32, y: f32) -> bool {
        self.inner.contains(RsPoint::new(x, y))
    }

    fn __repr__(&self) -> String {
        let bounds = self.inner.bounds();
        format!(
            "Path(bounds=Rect({}, {}, {}, {}))",
            bounds.left, bounds.top, bounds.right, bounds.bottom
        )
    }
}

// =============================================================================
// Surface
// =============================================================================

/// A drawing surface backed by pixels.
#[pyclass]
pub struct Surface {
    inner: RsSurface,
}

#[pymethods]
impl Surface {
    /// Create a new raster surface.
    #[new]
    fn new(width: i32, height: i32) -> PyResult<Self> {
        RsSurface::new_raster_n32_premul(width, height)
            .map(|s| Self { inner: s })
            .ok_or_else(|| PyValueError::new_err("Failed to create surface"))
    }

    /// Width in pixels.
    #[getter]
    const fn width(&self) -> i32 {
        self.inner.width()
    }

    /// Height in pixels.
    #[getter]
    const fn height(&self) -> i32 {
        self.inner.height()
    }

    /// Clear the surface with a color.
    fn clear(&mut self, color: u32) {
        let mut canvas = self.inner.raster_canvas();
        canvas.clear(Color(color));
    }

    /// Draw a rectangle.
    fn draw_rect(&mut self, left: f32, top: f32, right: f32, bottom: f32, paint: &Paint) {
        let mut canvas = self.inner.raster_canvas();
        canvas.draw_rect(&RsRect::new(left, top, right, bottom), &paint.inner);
    }

    /// Draw a circle.
    fn draw_circle(&mut self, cx: f32, cy: f32, radius: f32, paint: &Paint) {
        let mut canvas = self.inner.raster_canvas();
        canvas.draw_circle(RsPoint::new(cx, cy), radius, &paint.inner);
    }

    /// Draw an oval inscribed in a rectangle.
    fn draw_oval(&mut self, left: f32, top: f32, right: f32, bottom: f32, paint: &Paint) {
        let mut canvas = self.inner.raster_canvas();
        canvas.draw_oval(&RsRect::new(left, top, right, bottom), &paint.inner);
    }

    /// Draw a line.
    fn draw_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, paint: &Paint) {
        let mut canvas = self.inner.raster_canvas();
        canvas.draw_line(RsPoint::new(x0, y0), RsPoint::new(x1, y1), &paint.inner);
    }

    /// Draw a path.
    fn draw_path(&mut self, path: &Path, paint: &Paint) {
        let mut canvas = self.inner.raster_canvas();
        canvas.draw_path(&path.inner, &paint.inner);
    }

    /// Draw a point.
    fn draw_point(&mut self, x: f32, y: f32, paint: &Paint) {
        let mut canvas = self.inner.raster_canvas();
        canvas.draw_point(RsPoint::new(x, y), &paint.inner);
    }

    /// Get pixel data as bytes (RGBA).
    fn pixels(&self) -> Vec<u8> {
        let mut pixels = self.inner.pixels().to_vec();
        skia_rs_canvas::simd::unpremultiply_span(&mut pixels);
        pixels
    }

    /// Save to PNG file.
    #[cfg(feature = "png")]
    fn save_png(&self, path: &str) -> PyResult<()> {
        // Would need codec integration
        Err(PyValueError::new_err("PNG saving not yet implemented"))
    }

    fn __repr__(&self) -> String {
        format!("Surface({}x{})", self.inner.width(), self.inner.height())
    }
}

// =============================================================================
// Color utilities
// =============================================================================

/// Create an ARGB color value.
#[pyfunction]
const fn argb(a: u8, r: u8, g: u8, b: u8) -> u32 {
    Color::from_argb(a, r, g, b).0
}

/// Create an RGB color value (fully opaque).
#[pyfunction]
const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    Color::from_rgb(r, g, b).0
}

/// Predefined colors.
#[pyclass]
struct Colors;

#[pymethods]
impl Colors {
    #[classattr]
    const BLACK: u32 = 0xFF00_0000;
    #[classattr]
    const WHITE: u32 = 0xFFFF_FFFF;
    #[classattr]
    const RED: u32 = 0xFFFF_0000;
    #[classattr]
    const GREEN: u32 = 0xFF00_FF00;
    #[classattr]
    const BLUE: u32 = 0xFF00_00FF;
    #[classattr]
    const YELLOW: u32 = 0xFFFF_FF00;
    #[classattr]
    const CYAN: u32 = 0xFF00_FFFF;
    #[classattr]
    const MAGENTA: u32 = 0xFFFF_00FF;
    #[classattr]
    const TRANSPARENT: u32 = 0x0000_0000;
}

// =============================================================================
// Module
// =============================================================================

/// skia-rs Python bindings.
#[pymodule]
fn skia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Rect>()?;
    m.add_class::<Matrix>()?;
    m.add_class::<Paint>()?;
    m.add_class::<PathBuilder>()?;
    m.add_class::<Path>()?;
    m.add_class::<Surface>()?;
    m.add_class::<Colors>()?;
    m.add_function(wrap_pyfunction!(argb, m)?)?;
    m.add_function(wrap_pyfunction!(rgb, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact float equality is intentional here: `Rect::join`'s min/max of
    /// literal f32 values is deterministic bit-for-bit, so an epsilon
    /// comparison would only hide real regressions.
    fn assert_f32_eq(actual: f32, expected: f32) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn rect_join_expands_to_include_point_outside_max_corner() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let joined = rect.join(20.0, 20.0);
        assert_f32_eq(joined.inner.left, 0.0);
        assert_f32_eq(joined.inner.top, 0.0);
        assert_f32_eq(joined.inner.right, 20.0);
        assert_f32_eq(joined.inner.bottom, 20.0);
    }

    #[test]
    fn rect_join_expands_to_include_point_outside_min_corner() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let joined = rect.join(-5.0, -5.0);
        assert_f32_eq(joined.inner.left, -5.0);
        assert_f32_eq(joined.inner.top, -5.0);
        assert_f32_eq(joined.inner.right, 10.0);
        assert_f32_eq(joined.inner.bottom, 10.0);
    }

    #[test]
    fn rect_join_point_already_inside_is_noop() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        let joined = rect.join(5.0, 5.0);
        assert_f32_eq(joined.inner.left, 0.0);
        assert_f32_eq(joined.inner.top, 0.0);
        assert_f32_eq(joined.inner.right, 10.0);
        assert_f32_eq(joined.inner.bottom, 10.0);
    }

    // Alpha bridge: Paint::alpha()/set_alpha() convert between a 0-255 u8
    // and the underlying 0.0-1.0 f32. Exercised here as plain arithmetic
    // (matching the getter/setter bodies) since PyO3 getters/setters
    // require the GIL and can't be called directly in a host test.
    fn alpha_to_u8(alpha: f32) -> u8 {
        skia_rs_core::cast::f32_to_u8_sat(alpha * 255.0)
    }

    fn u8_to_alpha(alpha: u8) -> f32 {
        f32::from(alpha) / 255.0
    }

    #[test]
    fn alpha_round_trip_mid_value() {
        let a = u8_to_alpha(128);
        assert_eq!(alpha_to_u8(a), 128);
    }

    #[test]
    fn alpha_round_trip_zero() {
        let a = u8_to_alpha(0);
        assert_eq!(alpha_to_u8(a), 0);
    }

    #[test]
    fn alpha_round_trip_max() {
        let a = u8_to_alpha(255);
        assert_eq!(alpha_to_u8(a), 255);
    }
}
