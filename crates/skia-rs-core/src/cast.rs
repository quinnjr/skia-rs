//! Numeric cast helpers shared across the skia-rs workspace.
//!
//! The Rust standard library has no safe, saturating float→int conversion: a
//! bare `x as i32` silently saturates but trips the `cast_possible_truncation`
//! family of lints, and `TryFrom` is not implemented for floats. These helpers
//! are the single, documented home for those unavoidable `as` casts. Every
//! other narrowing conversion in the workspace should route through here (or
//! through [`TryFrom`]) instead of writing a bare `as` cast.

use crate::Scalar;

/// Largest s32 value that round-trips exactly through f32 (Skia's
/// `SK_MaxS32FitsInFloat`).
const SK_MAX_S32_FITS_IN_FLOAT: Scalar = 2_147_483_520.0;
/// Smallest (most negative) s32 value that round-trips exactly through f32.
const SK_MIN_S32_FITS_IN_FLOAT: Scalar = -2_147_483_520.0;

/// Saturating float→int cast matching Skia's `sk_float_saturate2int`.
///
/// Out-of-range values clamp to the representable s32 bounds and NaN maps to
/// the maximum, never to zero.
#[inline]
#[must_use]
pub fn saturate_to_i32(x: Scalar) -> i32 {
    // NaN fails the `<` test, so it clamps to the max (matching Skia).
    let x = if x < SK_MAX_S32_FITS_IN_FLOAT {
        x
    } else {
        SK_MAX_S32_FITS_IN_FLOAT
    };
    let x = if x > SK_MIN_S32_FITS_IN_FLOAT {
        x
    } else {
        SK_MIN_S32_FITS_IN_FLOAT
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "saturating float→int, matches Skia sk_float_saturate2int; no safe std conversion exists"
    )]
    {
        x as i32
    }
}

/// `sk_float_round2int`: round half toward +∞ (`floor(x + 0.5)`), then saturate.
#[inline]
#[must_use]
pub fn round_to_i32(x: Scalar) -> i32 {
    saturate_to_i32((x + 0.5).floor())
}

/// `sk_float_floor2int`: floor then saturate.
#[inline]
#[must_use]
pub fn floor_to_i32(x: Scalar) -> i32 {
    saturate_to_i32(x.floor())
}

/// `sk_float_ceil2int`: ceil then saturate.
#[inline]
#[must_use]
pub fn ceil_to_i32(x: Scalar) -> i32 {
    saturate_to_i32(x.ceil())
}

/// Convert an `i32` to a [`Scalar`], matching Skia's `SkIntToScalar`.
///
/// This is a widening cast in bit-width but loses integer precision above
/// `2^24`; there is no lossless `i32`→`f32` conversion in std. Behaviour
/// matches Skia, which likewise casts directly.
#[inline]
#[must_use]
pub const fn scalar_from_i32(x: i32) -> Scalar {
    #[expect(
        clippy::cast_precision_loss,
        reason = "i32→f32 loses precision above 2^24; matches Skia's SkIntToScalar, no lossless std conversion exists"
    )]
    {
        x as Scalar
    }
}

/// Round and clamp a 0–255 range float to a `u8`.
///
/// The value is rounded to the nearest integer and then clamped to `[0, 255]`,
/// matching the inline `(x).round().clamp(0.0, 255.0) as u8` idiom used for
/// 8-bit color/pixel components. NaN maps to `0`.
#[inline]
#[must_use]
pub const fn f32_to_u8_sat(x: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is rounded then clamped to [0,255]; no safe std float→u8 conversion exists"
    )]
    {
        x.round().clamp(0.0, 255.0) as u8
    }
}

/// Round and clamp a 0–65535 range float to a `u16`.
///
/// The value is rounded to the nearest integer and then clamped to
/// `[0, 65535]`, matching the inline idiom used for 16-bit color/pixel
/// components. NaN maps to `0`.
#[inline]
#[must_use]
pub const fn f32_to_u16_sat(x: f32) -> u16 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is rounded then clamped to [0,65535]; no safe std float→u16 conversion exists"
    )]
    {
        x.round().clamp(0.0, 65535.0) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ceil_to_i32, f32_to_u8_sat, f32_to_u16_sat, floor_to_i32, round_to_i32, saturate_to_i32,
    };

    const MAX_FIT: i32 = 2_147_483_520;
    const MIN_FIT: i32 = -2_147_483_520;

    #[test]
    fn saturate_clamps_and_maps_nan_to_max() {
        assert_eq!(saturate_to_i32(0.0), 0);
        assert_eq!(saturate_to_i32(1.9), 1);
        assert_eq!(saturate_to_i32(-1.9), -1);
        assert_eq!(saturate_to_i32(1e30), MAX_FIT);
        assert_eq!(saturate_to_i32(-1e30), MIN_FIT);
        assert_eq!(saturate_to_i32(f32::NAN), MAX_FIT);
    }

    #[test]
    fn round_floor_ceil() {
        assert_eq!(round_to_i32(2.5), 3);
        assert_eq!(round_to_i32(-2.5), -2);
        assert_eq!(floor_to_i32(2.9), 2);
        assert_eq!(ceil_to_i32(2.1), 3);
    }

    #[test]
    fn u8_sat_rounds_and_clamps() {
        assert_eq!(f32_to_u8_sat(-5.0), 0);
        assert_eq!(f32_to_u8_sat(0.4), 0);
        assert_eq!(f32_to_u8_sat(0.5), 1);
        assert_eq!(f32_to_u8_sat(254.6), 255);
        assert_eq!(f32_to_u8_sat(300.0), 255);
        assert_eq!(f32_to_u8_sat(f32::NAN), 0);
    }

    #[test]
    fn u16_sat_rounds_and_clamps() {
        assert_eq!(f32_to_u16_sat(-5.0), 0);
        assert_eq!(f32_to_u16_sat(0.5), 1);
        assert_eq!(f32_to_u16_sat(70000.0), 65535);
    }
}
