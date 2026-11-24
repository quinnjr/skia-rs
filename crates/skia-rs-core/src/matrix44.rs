//! 4x4 transformation matrix for 3D transformations.
//!
//! This module provides a 4x4 matrix type for 3D transformations,
//! corresponding to Skia's `SkM44` / `SkMatrix44`.

#![allow(
    clippy::suboptimal_flops,
    reason = "matrix44 is a faithful port of Skia SkM44; mul_add's fused rounding diverges from Skia's fma-free reference arithmetic"
)]
#![allow(
    clippy::float_cmp,
    reason = "exact comparisons (identity/special-value checks) match Skia's exact SkScalar comparison"
)]

use crate::Scalar;
use crate::geometry::{Matrix, Point, Point3};

/// A 4x4 transformation matrix for 3D transformations.
///
/// Corresponds to Skia's `SkM44`.
///
/// The matrix is stored in column-major order:
/// ```text
/// | m[0]  m[4]  m[8]   m[12] |
/// | m[1]  m[5]  m[9]   m[13] |
/// | m[2]  m[6]  m[10]  m[14] |
/// | m[3]  m[7]  m[11]  m[15] |
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Matrix44 {
    /// Matrix values in column-major order.
    pub values: [Scalar; 16],
}

impl Default for Matrix44 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix44 {
    /// The identity matrix constant.
    pub const IDENTITY: Self = Self {
        values: [
            1.0, 0.0, 0.0, 0.0, // column 0
            0.0, 1.0, 0.0, 0.0, // column 1
            0.0, 0.0, 1.0, 0.0, // column 2
            0.0, 0.0, 0.0, 1.0, // column 3
        ],
    };

    /// Creates a new matrix from column-major values.
    #[inline]
    #[must_use]
    pub const fn from_cols(values: [Scalar; 16]) -> Self {
        Self { values }
    }

    /// Creates a new matrix from row-major values.
    #[must_use]
    pub const fn from_rows(
        r0: [Scalar; 4],
        r1: [Scalar; 4],
        r2: [Scalar; 4],
        r3: [Scalar; 4],
    ) -> Self {
        Self {
            values: [
                r0[0], r1[0], r2[0], r3[0], // column 0
                r0[1], r1[1], r2[1], r3[1], // column 1
                r0[2], r1[2], r2[2], r3[2], // column 2
                r0[3], r1[3], r2[3], r3[3], // column 3
            ],
        }
    }

    /// Creates the identity matrix.
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Creates a translation matrix.
    #[must_use]
    pub const fn translate(x: Scalar, y: Scalar, z: Scalar) -> Self {
        Self {
            values: [
                1.0, 0.0, 0.0, 0.0, // column 0
                0.0, 1.0, 0.0, 0.0, // column 1
                0.0, 0.0, 1.0, 0.0, // column 2
                x, y, z, 1.0, // column 3
            ],
        }
    }

    /// Creates a scale matrix.
    #[must_use]
    pub const fn scale(sx: Scalar, sy: Scalar, sz: Scalar) -> Self {
        Self {
            values: [
                sx, 0.0, 0.0, 0.0, // column 0
                0.0, sy, 0.0, 0.0, // column 1
                0.0, 0.0, sz, 0.0, // column 2
                0.0, 0.0, 0.0, 1.0, // column 3
            ],
        }
    }

    /// Creates a rotation matrix around the X axis.
    #[must_use]
    pub fn rotate_x(radians: Scalar) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            values: [
                1.0, 0.0, 0.0, 0.0, // column 0
                0.0, cos, sin, 0.0, // column 1
                0.0, -sin, cos, 0.0, // column 2
                0.0, 0.0, 0.0, 1.0, // column 3
            ],
        }
    }

    /// Creates a rotation matrix around the Y axis.
    #[must_use]
    pub fn rotate_y(radians: Scalar) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            values: [
                cos, 0.0, -sin, 0.0, // column 0
                0.0, 1.0, 0.0, 0.0, // column 1
                sin, 0.0, cos, 0.0, // column 2
                0.0, 0.0, 0.0, 1.0, // column 3
            ],
        }
    }

    /// Creates a rotation matrix around the Z axis.
    #[must_use]
    pub fn rotate_z(radians: Scalar) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            values: [
                cos, sin, 0.0, 0.0, // column 0
                -sin, cos, 0.0, 0.0, // column 1
                0.0, 0.0, 1.0, 0.0, // column 2
                0.0, 0.0, 0.0, 1.0, // column 3
            ],
        }
    }

    /// Creates a rotation matrix around an arbitrary axis.
    #[must_use]
    pub fn rotate(axis: Point3, radians: Scalar) -> Self {
        let len = axis.length();
        if len == 0.0 {
            return Self::IDENTITY;
        }

        let x = axis.x / len;
        let y = axis.y / len;
        let z = axis.z / len;

        let (sin, cos) = radians.sin_cos();
        let t = 1.0 - cos;

        Self {
            values: [
                t * x * x + cos,
                t * x * y + sin * z,
                t * x * z - sin * y,
                0.0,
                t * x * y - sin * z,
                t * y * y + cos,
                t * y * z + sin * x,
                0.0,
                t * x * z + sin * y,
                t * y * z - sin * x,
                t * z * z + cos,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
    }

    /// Creates a look-at matrix (camera transformation).
    #[must_use]
    pub fn look_at(eye: Point3, center: Point3, up: Point3) -> Self {
        let f = Point3::new(center.x - eye.x, center.y - eye.y, center.z - eye.z);
        let f_len = f.length();
        if f_len == 0.0 {
            return Self::IDENTITY;
        }
        let f = Point3::new(f.x / f_len, f.y / f_len, f.z / f_len);

        let s = f.cross(&up);
        let s_len = s.length();
        if s_len == 0.0 {
            return Self::IDENTITY;
        }
        let s = Point3::new(s.x / s_len, s.y / s_len, s.z / s_len);

        let u = s.cross(&f);

        Self {
            values: [
                s.x,
                u.x,
                -f.x,
                0.0, // column 0
                s.y,
                u.y,
                -f.y,
                0.0, // column 1
                s.z,
                u.z,
                -f.z,
                0.0, // column 2
                -s.dot(&Point3::new(eye.x, eye.y, eye.z)),
                -u.dot(&Point3::new(eye.x, eye.y, eye.z)),
                f.dot(&Point3::new(eye.x, eye.y, eye.z)),
                1.0, // column 3
            ],
        }
    }

    /// Creates a perspective projection matrix.
    #[must_use]
    pub fn perspective(fov_y: Scalar, aspect: Scalar, near: Scalar, far: Scalar) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let nf = 1.0 / (near - far);

        Self {
            values: [
                f / aspect,
                0.0,
                0.0,
                0.0, // column 0
                0.0,
                f,
                0.0,
                0.0, // column 1
                0.0,
                0.0,
                (far + near) * nf,
                -1.0, // column 2
                0.0,
                0.0,
                2.0 * far * near * nf,
                0.0, // column 3
            ],
        }
    }

    /// Creates an orthographic projection matrix.
    #[must_use]
    pub fn ortho(
        left: Scalar,
        right: Scalar,
        bottom: Scalar,
        top: Scalar,
        near: Scalar,
        far: Scalar,
    ) -> Self {
        let dx = right - left;
        let dy = top - bottom;
        let dz = far - near;

        Self {
            values: [
                2.0 / dx,
                0.0,
                0.0,
                0.0, // column 0
                0.0,
                2.0 / dy,
                0.0,
                0.0, // column 1
                0.0,
                0.0,
                -2.0 / dz,
                0.0, // column 2
                -(right + left) / dx,
                -(top + bottom) / dy,
                -(far + near) / dz,
                1.0, // column 3
            ],
        }
    }

    /// Build an orthographic projection matrix, returning `None` if any range is zero.
    ///
    /// This is a checked variant of [`Self::ortho`] that validates preconditions
    /// instead of producing infinity or NaN. Use this when the projection
    /// parameters come from untrusted sources.
    #[must_use]
    pub fn ortho_checked(
        left: Scalar,
        right: Scalar,
        bottom: Scalar,
        top: Scalar,
        near: Scalar,
        far: Scalar,
    ) -> Option<Self> {
        if left == right || bottom == top || near == far {
            return None;
        }
        Some(Self::ortho(left, right, bottom, top, near, far))
    }

    /// Build a perspective projection matrix, returning `None` if `fov_y` or aspect is zero.
    ///
    /// This is a checked variant of [`Self::perspective`] that validates
    /// preconditions instead of producing infinity or NaN.
    #[must_use]
    pub fn perspective_checked(
        fov_y: Scalar,
        aspect: Scalar,
        near: Scalar,
        far: Scalar,
    ) -> Option<Self> {
        if fov_y == 0.0 || near == far || aspect == 0.0 {
            return None;
        }
        Some(Self::perspective(fov_y, aspect, near, far))
    }

    /// Returns true if this is the identity matrix.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Get the value at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if `row >= 4` or `col >= 4`.
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Scalar {
        assert!(row < 4, "row out of bounds: {row} (must be < 4)");
        assert!(col < 4, "col out of bounds: {col} (must be < 4)");
        self.values[col * 4 + row]
    }

    /// Set the value at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if `row >= 4` or `col >= 4`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Scalar) {
        assert!(row < 4, "row out of bounds: {row} (must be < 4)");
        assert!(col < 4, "col out of bounds: {col} (must be < 4)");
        self.values[col * 4 + row] = value;
    }

    /// Get a column as a 4-element array.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= 4`.
    #[inline]
    #[must_use]
    pub fn col(&self, idx: usize) -> [Scalar; 4] {
        assert!(idx < 4, "col index out of bounds: {idx} (must be < 4)");
        let base = idx * 4;
        [
            self.values[base],
            self.values[base + 1],
            self.values[base + 2],
            self.values[base + 3],
        ]
    }

    /// Get a row as a 4-element array.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= 4`.
    #[inline]
    #[must_use]
    pub fn row(&self, idx: usize) -> [Scalar; 4] {
        assert!(idx < 4, "row index out of bounds: {idx} (must be < 4)");
        [
            self.values[idx],
            self.values[idx + 4],
            self.values[idx + 8],
            self.values[idx + 12],
        ]
    }

    /// Concatenates this matrix with another (self * other).
    #[must_use]
    pub fn concat(&self, other: &Self) -> Self {
        let mut result = [0.0; 16];

        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.get(row, k) * other.get(k, col);
                }
                result[col * 4 + row] = sum;
            }
        }

        Self { values: result }
    }

    /// Pre-concatenates this matrix with a translation (`self * T`).
    ///
    /// Matches `SkM44::preTranslate`: the translation is applied in the local
    /// (source) coordinate space, updating the last column.
    #[must_use]
    pub fn pre_translate(&self, x: Scalar, y: Scalar, z: Scalar) -> Self {
        self.concat(&Self::translate(x, y, z))
    }

    /// Post-concatenates this matrix with a translation (`T * self`).
    ///
    /// Matches `SkM44::postTranslate`: the translation is applied in the
    /// destination coordinate space.
    #[must_use]
    pub fn post_translate(&self, x: Scalar, y: Scalar, z: Scalar) -> Self {
        Self::translate(x, y, z).concat(self)
    }

    /// Pre-concatenates this matrix with a scale (`self * S`).
    ///
    /// Matches `SkM44::preScale`.
    #[must_use]
    pub fn pre_scale(&self, sx: Scalar, sy: Scalar, sz: Scalar) -> Self {
        self.concat(&Self::scale(sx, sy, sz))
    }

    /// Post-concatenates this matrix with a scale (`S * self`).
    #[must_use]
    pub fn post_scale(&self, sx: Scalar, sy: Scalar, sz: Scalar) -> Self {
        Self::scale(sx, sy, sz).concat(self)
    }

    /// Transforms a 3D point by this matrix.
    #[must_use]
    pub fn map_point3(&self, point: Point3) -> Point3 {
        let x = self.get(0, 0) * point.x
            + self.get(0, 1) * point.y
            + self.get(0, 2) * point.z
            + self.get(0, 3);
        let y = self.get(1, 0) * point.x
            + self.get(1, 1) * point.y
            + self.get(1, 2) * point.z
            + self.get(1, 3);
        let z = self.get(2, 0) * point.x
            + self.get(2, 1) * point.y
            + self.get(2, 2) * point.z
            + self.get(2, 3);
        let w = self.get(3, 0) * point.x
            + self.get(3, 1) * point.y
            + self.get(3, 2) * point.z
            + self.get(3, 3);

        if w != 0.0 && w != 1.0 {
            Point3::new(x / w, y / w, z / w)
        } else {
            Point3::new(x, y, z)
        }
    }

    /// Transforms a 2D point by this matrix (z=0, w=1).
    #[must_use]
    pub fn map_point(&self, point: Point) -> Point {
        let x = self.get(0, 0) * point.x + self.get(0, 1) * point.y + self.get(0, 3);
        let y = self.get(1, 0) * point.x + self.get(1, 1) * point.y + self.get(1, 3);
        let w = self.get(3, 0) * point.x + self.get(3, 1) * point.y + self.get(3, 3);

        if w != 0.0 && w != 1.0 {
            Point::new(x / w, y / w)
        } else {
            Point::new(x, y)
        }
    }

    /// Computes the determinant of this matrix.
    #[must_use]
    pub fn determinant(&self) -> Scalar {
        let m = &self.values;

        let s0 = m[0] * m[5] - m[4] * m[1];
        let s1 = m[0] * m[9] - m[8] * m[1];
        let s2 = m[0] * m[13] - m[12] * m[1];
        let s3 = m[4] * m[9] - m[8] * m[5];
        let s4 = m[4] * m[13] - m[12] * m[5];
        let s5 = m[8] * m[13] - m[12] * m[9];

        let c5 = m[10] * m[15] - m[14] * m[11];
        let c4 = m[6] * m[15] - m[14] * m[7];
        let c3 = m[6] * m[11] - m[10] * m[7];
        let c2 = m[2] * m[15] - m[14] * m[3];
        let c1 = m[2] * m[11] - m[10] * m[3];
        let c0 = m[2] * m[7] - m[6] * m[3];

        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    }

    /// Computes the inverse of this matrix, or None if singular.
    ///
    /// Matches Skia's `SkInvert4x4Matrix` (`src/core/SkMatrixInvert.cpp`): the
    /// determinant and all cofactors are computed in double precision, and the
    /// matrix is reported singular only when the determinant is exactly zero or
    /// when the computed inverse contains a non-finite element (e.g. `1/det`
    /// overflowing on a denormalized determinant).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "inverse is computed in f64 then narrowed to Scalar (f32), matching Skia's SkInvert4x4Matrix; the f64→f32 narrowing is intentional and has no safe std conversion"
    )]
    #[must_use]
    pub fn invert(&self) -> Option<Self> {
        let m = &self.values;
        // Column-major storage matches SkM44's fMat layout directly.
        let a00 = f64::from(m[0]);
        let a01 = f64::from(m[1]);
        let a02 = f64::from(m[2]);
        let a03 = f64::from(m[3]);
        let a10 = f64::from(m[4]);
        let a11 = f64::from(m[5]);
        let a12 = f64::from(m[6]);
        let a13 = f64::from(m[7]);
        let a20 = f64::from(m[8]);
        let a21 = f64::from(m[9]);
        let a22 = f64::from(m[10]);
        let a23 = f64::from(m[11]);
        let a30 = f64::from(m[12]);
        let a31 = f64::from(m[13]);
        let a32 = f64::from(m[14]);
        let a33 = f64::from(m[15]);

        let mut b00 = a00 * a11 - a01 * a10;
        let mut b01 = a00 * a12 - a02 * a10;
        let mut b02 = a00 * a13 - a03 * a10;
        let mut b03 = a01 * a12 - a02 * a11;
        let mut b04 = a01 * a13 - a03 * a11;
        let mut b05 = a02 * a13 - a03 * a12;
        let mut b06 = a20 * a31 - a21 * a30;
        let mut b07 = a20 * a32 - a22 * a30;
        let mut b08 = a20 * a33 - a23 * a30;
        let mut b09 = a21 * a32 - a22 * a31;
        let mut b10 = a21 * a33 - a23 * a31;
        let mut b11 = a22 * a33 - a23 * a32;

        let determinant = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        if determinant == 0.0 {
            return None;
        }

        let invdet = 1.0 / determinant;
        b00 *= invdet;
        b01 *= invdet;
        b02 *= invdet;
        b03 *= invdet;
        b04 *= invdet;
        b05 *= invdet;
        b06 *= invdet;
        b07 *= invdet;
        b08 *= invdet;
        b09 *= invdet;
        b10 *= invdet;
        b11 *= invdet;

        let out = [
            (a11 * b11 - a12 * b10 + a13 * b09) as Scalar,
            (a02 * b10 - a01 * b11 - a03 * b09) as Scalar,
            (a31 * b05 - a32 * b04 + a33 * b03) as Scalar,
            (a22 * b04 - a21 * b05 - a23 * b03) as Scalar,
            (a12 * b08 - a10 * b11 - a13 * b07) as Scalar,
            (a00 * b11 - a02 * b08 + a03 * b07) as Scalar,
            (a32 * b02 - a30 * b05 - a33 * b01) as Scalar,
            (a20 * b05 - a22 * b02 + a23 * b01) as Scalar,
            (a10 * b10 - a11 * b08 + a13 * b06) as Scalar,
            (a01 * b08 - a00 * b10 - a03 * b06) as Scalar,
            (a30 * b04 - a31 * b02 + a33 * b00) as Scalar,
            (a21 * b02 - a20 * b04 - a23 * b00) as Scalar,
            (a11 * b07 - a10 * b09 - a12 * b06) as Scalar,
            (a00 * b09 - a01 * b07 + a02 * b06) as Scalar,
            (a31 * b01 - a30 * b03 - a32 * b00) as Scalar,
            (a20 * b03 - a21 * b01 + a22 * b00) as Scalar,
        ];

        // Reject a non-finite inverse (overflow on denormalized determinant).
        if out.iter().all(|v| v.is_finite()) {
            Some(Self { values: out })
        } else {
            None
        }
    }

    /// Returns the transpose of this matrix.
    #[must_use]
    pub const fn transpose(&self) -> Self {
        Self {
            values: [
                self.values[0],
                self.values[4],
                self.values[8],
                self.values[12],
                self.values[1],
                self.values[5],
                self.values[9],
                self.values[13],
                self.values[2],
                self.values[6],
                self.values[10],
                self.values[14],
                self.values[3],
                self.values[7],
                self.values[11],
                self.values[15],
            ],
        }
    }

    /// Extracts the upper-left 3x3 as a 2D Matrix (ignoring z and perspective).
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        Matrix {
            values: [
                self.get(0, 0),
                self.get(0, 1),
                self.get(0, 3),
                self.get(1, 0),
                self.get(1, 1),
                self.get(1, 3),
                self.get(3, 0),
                self.get(3, 1),
                self.get(3, 3),
            ],
        }
    }

    /// Creates a 4x4 matrix from a 2D Matrix.
    #[must_use]
    pub const fn from_matrix(m: &Matrix) -> Self {
        Self {
            values: [
                m.values[0],
                m.values[3],
                0.0,
                m.values[6], // column 0
                m.values[1],
                m.values[4],
                0.0,
                m.values[7], // column 1
                0.0,
                0.0,
                1.0,
                0.0, // column 2
                m.values[2],
                m.values[5],
                0.0,
                m.values[8], // column 3
            ],
        }
    }
}

impl std::ops::Mul for Matrix44 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.concat(&rhs)
    }
}

impl std::ops::MulAssign for Matrix44 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.concat(&rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let m = Matrix44::identity();
        assert!(m.is_identity());

        let p = Point3::new(1.0, 2.0, 3.0);
        let result = m.map_point3(p);
        assert!((result.x - p.x).abs() < 1e-6);
        assert!((result.y - p.y).abs() < 1e-6);
        assert!((result.z - p.z).abs() < 1e-6);
    }

    #[test]
    fn test_translate() {
        let m = Matrix44::translate(10.0, 20.0, 30.0);
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = m.map_point3(p);
        assert!((result.x - 11.0).abs() < 1e-6);
        assert!((result.y - 22.0).abs() < 1e-6);
        assert!((result.z - 33.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale() {
        let m = Matrix44::scale(2.0, 3.0, 4.0);
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = m.map_point3(p);
        assert!((result.x - 2.0).abs() < 1e-6);
        assert!((result.y - 6.0).abs() < 1e-6);
        assert!((result.z - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_invert() {
        let m = Matrix44::translate(10.0, 20.0, 30.0);
        let inv = m.invert().unwrap();
        let result = m.concat(&inv);
        assert!((result.values[0] - 1.0).abs() < 1e-6);
        assert!((result.values[5] - 1.0).abs() < 1e-6);
        assert!((result.values[10] - 1.0).abs() < 1e-6);
        assert!((result.values[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_from_matrix() {
        let m3 = Matrix::translate(10.0, 20.0);
        let m4 = Matrix44::from_matrix(&m3);
        let m3_back = m4.to_matrix();

        let p = Point::new(5.0, 7.0);
        let r1 = m3.map_point(p);
        let r2 = m3_back.map_point(p);
        assert!((r1.x - r2.x).abs() < 1e-6);
        assert!((r1.y - r2.y).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "row out of bounds")]
    fn test_matrix44_get_row_out_of_bounds() {
        let m = Matrix44::identity();
        let _ = m.get(4, 0);
    }

    #[test]
    #[should_panic(expected = "col out of bounds")]
    fn test_matrix44_get_col_out_of_bounds() {
        let m = Matrix44::identity();
        let _ = m.get(0, 4);
    }

    #[test]
    fn test_matrix44_ortho_zero_width_returns_none() {
        let m = Matrix44::ortho_checked(5.0, 5.0, 0.0, 10.0, -1.0, 1.0);
        assert!(m.is_none(), "ortho with zero width should return None");
    }

    #[test]
    fn test_matrix44_ortho_zero_height_returns_none() {
        let m = Matrix44::ortho_checked(0.0, 10.0, 5.0, 5.0, -1.0, 1.0);
        assert!(m.is_none(), "ortho with zero height should return None");
    }

    #[test]
    fn test_matrix44_ortho_zero_depth_returns_none() {
        let m = Matrix44::ortho_checked(0.0, 10.0, 0.0, 10.0, 1.0, 1.0);
        assert!(m.is_none(), "ortho with zero depth should return None");
    }

    #[test]
    fn test_matrix44_ortho_normal_returns_some() {
        let m = Matrix44::ortho_checked(0.0, 10.0, 0.0, 10.0, -1.0, 1.0);
        assert!(m.is_some());
    }

    #[test]
    fn test_matrix44_perspective_zero_fov_returns_none() {
        let m = Matrix44::perspective_checked(0.0, 1.0, 0.1, 100.0);
        assert!(m.is_none(), "perspective with zero fov should return None");
    }

    #[test]
    fn test_matrix44_perspective_normal_returns_some() {
        let m = Matrix44::perspective_checked(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        assert!(m.is_some());
    }

    // --- Conformance regression tests (Task 1) ---

    #[test]
    fn test_pre_translate_is_self_times_t() {
        // preTranslate applies the translation in local space: self * T.
        // Starting from a 2x scale, translating (1,0,0) should move the point
        // by 2 in device space (the scale acts on the translation).
        let scale = Matrix44::scale(2.0, 2.0, 2.0);
        let pre = scale.pre_translate(1.0, 0.0, 0.0);
        assert_eq!(pre, scale.concat(&Matrix44::translate(1.0, 0.0, 0.0)));
        // Last column encodes the transformed translation: 2 * 1 = 2.
        assert_eq!(pre.get(0, 3), 2.0);
    }

    #[test]
    fn test_post_translate_is_t_times_self() {
        // postTranslate applies the translation in device space: T * self.
        let scale = Matrix44::scale(2.0, 2.0, 2.0);
        let post = scale.post_translate(1.0, 0.0, 0.0);
        assert_eq!(post, Matrix44::translate(1.0, 0.0, 0.0).concat(&scale));
        // Device-space translation is not scaled: stays 1.
        assert_eq!(post.get(0, 3), 1.0);
    }

    #[test]
    fn test_pre_post_scale_semantics() {
        let t = Matrix44::translate(10.0, 0.0, 0.0);
        // preScale: self * S. Scaling then applying t's translation leaves the
        // translation column untouched (S doesn't move the origin).
        assert_eq!(
            t.pre_scale(2.0, 2.0, 2.0),
            t.concat(&Matrix44::scale(2.0, 2.0, 2.0))
        );
        // postScale: S * self. The translation column gets scaled by 2.
        let post = t.post_scale(2.0, 2.0, 2.0);
        assert_eq!(post, Matrix44::scale(2.0, 2.0, 2.0).concat(&t));
        assert_eq!(post.get(0, 3), 20.0);
    }

    #[test]
    fn test_invert_small_scale() {
        // A tiny uniform scale (det = 1e-9) is invertible in Skia; the old
        // absolute f32 threshold wrongly rejected it.
        let m = Matrix44::scale(1e-3, 1e-3, 1e-3);
        let inv = m.invert().expect("small-scale 4x4 must invert");
        assert!((inv.values[0] - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_invert_singular_returns_none() {
        let m = Matrix44::scale(1.0, 1.0, 0.0);
        assert!(m.invert().is_none(), "zero-scale axis is singular");
    }
}
