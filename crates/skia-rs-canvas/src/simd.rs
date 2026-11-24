//! SIMD-optimized blitting operations.
//!
//! This module provides hardware-accelerated pixel operations using:
//! - **SSE4.2** on `x86/x86_64` (128-bit, 4 pixels at a time)
//! - **AVX2** on `x86/x86_64` (256-bit, 8 pixels at a time)
//! - **NEON** on ARM/AArch64 (128-bit, 4 pixels at a time)
//!
//! The module automatically selects the best available instruction set at runtime.
//!
//! ## Performance
//!
//! SIMD operations can provide 4-8x speedup for batch pixel operations like:
//! - Filling horizontal spans with solid colors
//! - Alpha blending multiple pixels
//! - Premultiplied alpha operations

use skia_rs_core::Color;
pub(crate) use skia_rs_core::mul_div_255_round;

/// Premultiplied `SrcOver` on straight byte channels that are already
/// premultiplied: `out = src + SkMulDiv255Round(dst, 255 - srcA)`, per
/// `SkPMSrcOver` in `skia/src/core/SkBlitRow_D32.cpp`. `src` and `dst`
/// are both premultiplied. The scalar, SSE, AVX2 and NEON span blitters
/// and `blend_colors` all use this identical formula so they agree
/// bit-for-bit.
#[inline]
pub(crate) fn src_over_premul(src: Color, dst: Color) -> Color {
    let inv = 255 - u32::from(src.alpha());
    Color::from_argb(
        src.alpha() + mul_div_255_round(u32::from(dst.alpha()), inv),
        src.red() + mul_div_255_round(u32::from(dst.red()), inv),
        src.green() + mul_div_255_round(u32::from(dst.green()), inv),
        src.blue() + mul_div_255_round(u32::from(dst.blue()), inv),
    )
}

/// SIMD capabilities detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// SSE4.2 support (`x86/x86_64`)
    pub sse42: bool,
    /// AVX2 support (`x86/x86_64`)
    pub avx2: bool,
    /// NEON support (ARM/AArch64)
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime.
    #[inline]
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "only const on non-x86 targets; x86 probes CPU features at runtime"
    )]
    pub fn detect() -> Self {
        Self {
            sse42: Self::has_sse42(),
            avx2: Self::has_avx2(),
            neon: Self::has_neon(),
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn has_sse42() -> bool {
        #[cfg(target_feature = "sse4.2")]
        {
            true
        }
        #[cfg(not(target_feature = "sse4.2"))]
        {
            is_x86_feature_detected!("sse4.2")
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    const fn has_sse42() -> bool {
        false
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn has_avx2() -> bool {
        #[cfg(target_feature = "avx2")]
        {
            true
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            is_x86_feature_detected!("avx2")
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    const fn has_avx2() -> bool {
        false
    }

    #[cfg(target_arch = "aarch64")]
    const fn has_neon() -> bool {
        // NEON is mandatory on AArch64
        true
    }

    #[cfg(target_arch = "arm")]
    fn has_neon() -> bool {
        #[cfg(target_feature = "neon")]
        {
            true
        }
        #[cfg(not(target_feature = "neon"))]
        {
            false
        }
    }

    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    const fn has_neon() -> bool {
        false
    }

    /// Returns the best available SIMD width in pixels.
    #[inline]
    #[must_use]
    pub const fn best_width(&self) -> usize {
        if self.avx2 {
            8 // AVX2: 256 bits = 8 x 32-bit pixels
        } else if self.sse42 || self.neon {
            4 // SSE4.2/NEON: 128 bits = 4 x 32-bit pixels
        } else {
            1 // Scalar fallback
        }
    }
}

/// Global SIMD capabilities, lazily initialized.
static SIMD_CAPS: std::sync::OnceLock<SimdCapabilities> = std::sync::OnceLock::new();

/// Get the detected SIMD capabilities.
#[inline]
pub fn simd_capabilities() -> &'static SimdCapabilities {
    SIMD_CAPS.get_or_init(SimdCapabilities::detect)
}

// ============================================================================
// SIMD-optimized fill operations
// ============================================================================

/// Fill a span of pixels with a solid color (SIMD-optimized).
///
/// This is the hot path for solid color fills - optimized for both
/// opaque and semi-transparent colors.
#[inline]
pub fn fill_span_solid(dst: &mut [u8], color: Color) {
    let len = dst.len() / 4;
    if len == 0 {
        return;
    }

    // For opaque colors, use simple memset-style fill
    if color.alpha() == 255 {
        fill_span_opaque(dst, color);
        return;
    }

    // For transparent source, nothing to do
    if color.alpha() == 0 {
        return;
    }

    let caps = simd_capabilities();

    // Use SIMD for semi-transparent blending
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if caps.avx2 && len >= 8 {
            // SAFETY: We've verified AVX2 support and have enough data
            unsafe { fill_span_blend_avx2(dst, color) };
            return;
        }
        // No dedicated SSE4.2 blend path: the AVX2 path above (or the scalar
        // fallback below) covers x86/x86_64. A prior SSE4.1 implementation
        // was removed because it applied alpha uniformly rather than
        // per-channel, producing incorrect color values for semi-transparent
        // blends.
    }

    #[cfg(target_arch = "aarch64")]
    {
        if caps.neon && len >= 4 {
            // SAFETY: NEON is always available on AArch64
            unsafe { fill_span_blend_neon(dst, color) };
            return;
        }
    }

    // Scalar fallback
    fill_span_blend_scalar(dst, color);
}

/// Fill span with opaque color (no blending needed).
#[inline]
fn fill_span_opaque(dst: &mut [u8], color: Color) {
    let pattern = [color.red(), color.green(), color.blue(), color.alpha()];
    for chunk in dst.as_chunks_mut::<4>().0 {
        chunk.copy_from_slice(&pattern);
    }
}

/// Scalar fallback for premultiplied `SrcOver` span fill.
///
/// `src` is a premultiplied color. Result: `out = src + SkMulDiv255Round(dst,
/// 255 - srcA)` per channel — the canonical `SkPMSrcOver`.
fn fill_span_blend_scalar(dst: &mut [u8], src: Color) {
    let sa = u32::from(src.alpha());
    let sr = u32::from(src.red());
    let sg = u32::from(src.green());
    let sb = u32::from(src.blue());
    let inv_sa = 255 - sa;

    for chunk in dst.as_chunks_mut::<4>().0 {
        chunk[0] = (sr + u32::from(mul_div_255_round(u32::from(chunk[0]), inv_sa))).min(255) as u8;
        chunk[1] = (sg + u32::from(mul_div_255_round(u32::from(chunk[1]), inv_sa))).min(255) as u8;
        chunk[2] = (sb + u32::from(mul_div_255_round(u32::from(chunk[2]), inv_sa))).min(255) as u8;
        chunk[3] = (sa + u32::from(mul_div_255_round(u32::from(chunk[3]), inv_sa))).min(255) as u8;
    }
}

// ============================================================================
// x86/x86_64 AVX2 Implementation
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_span_blend_avx2(dst: &mut [u8], src: Color) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_loadu_si256, _mm256_mullo_epi32,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_srli_epi32,
        _mm256_storeu_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_loadu_si256, _mm256_mullo_epi32,
        _mm256_or_si256, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_srli_epi32,
        _mm256_storeu_si256,
    };

    let len = dst.len() / 4;
    let chunks = len / 8;
    let remainder_start = chunks * 32;

    let inv_alpha = i32::from(255 - src.alpha());

    let ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 32;
        // SAFETY: `offset` stays within `dst` because `chunks = len / 8` and
        // each iteration advances by 32 B (8 pixels x 4 B), matching `len`.
        let dst_ptr = unsafe { ptr.add(offset) };

        // Load 8 pixels (32 bytes).
        // SAFETY: `dst_ptr` points to at least 32 valid, initialized bytes
        // within `dst` (see offset bound above); alignment is not required
        // for `_mm256_loadu_si256`.
        #[allow(
            clippy::cast_ptr_alignment,
            reason = "_mm256_loadu_si256 is the unaligned-load intrinsic by design; the __m256i-typed pointer is never dereferenced except by this intrinsic"
        )]
        let dst_pixels = unsafe { _mm256_loadu_si256(dst_ptr.cast::<__m256i>()) };

        // Extract channels using masks
        let mask_r = _mm256_set1_epi32(i32::from_ne_bytes(0x0000_00FF_u32.to_ne_bytes()));
        let mask_g = _mm256_set1_epi32(i32::from_ne_bytes(0x0000_FF00_u32.to_ne_bytes()));
        let mask_b = _mm256_set1_epi32(i32::from_ne_bytes(0x00FF_0000_u32.to_ne_bytes()));
        let mask_a = _mm256_set1_epi32(i32::from_ne_bytes(0xFF00_0000_u32.to_ne_bytes()));

        let dst_r = _mm256_and_si256(dst_pixels, mask_r);
        let dst_g = _mm256_srli_epi32(_mm256_and_si256(dst_pixels, mask_g), 8);
        let dst_b = _mm256_srli_epi32(_mm256_and_si256(dst_pixels, mask_b), 16);
        let dst_a = _mm256_srli_epi32(_mm256_and_si256(dst_pixels, mask_a), 24);

        let src_r = _mm256_set1_epi32(i32::from(src.red()));
        let src_g = _mm256_set1_epi32(i32::from(src.green()));
        let src_b = _mm256_set1_epi32(i32::from(src.blue()));
        let src_a = _mm256_set1_epi32(i32::from(src.alpha()));

        // Premultiplied SrcOver with SkMulDiv255Round on the dst term, matching
        // the scalar/NEON paths bit-for-bit:
        //   prod = dst * inv_alpha + 128
        //   res  = (prod + (prod >> 8)) >> 8
        //   out  = src + res              (src is premultiplied)
        let inv_vec = _mm256_set1_epi32(inv_alpha);
        let c128 = _mm256_set1_epi32(128);
        let blend = |s: __m256i, d: __m256i| -> __m256i {
            let prod = _mm256_add_epi32(_mm256_mullo_epi32(d, inv_vec), c128);
            let res = _mm256_srli_epi32(_mm256_add_epi32(prod, _mm256_srli_epi32(prod, 8)), 8);
            _mm256_add_epi32(s, res)
        };

        let result_r = blend(src_r, dst_r);
        let result_g = blend(src_g, dst_g);
        let result_b = blend(src_b, dst_b);
        let result_a = blend(src_a, dst_a);

        // Recombine channels
        let rg = _mm256_or_si256(result_r, _mm256_slli_epi32(result_g, 8));
        let ba = _mm256_or_si256(
            _mm256_slli_epi32(result_b, 16),
            _mm256_slli_epi32(result_a, 24),
        );
        let result = _mm256_or_si256(rg, ba);

        // Store result.
        // SAFETY: same 32-byte bound as the load above; alignment is not
        // required for `_mm256_storeu_si256`.
        #[allow(
            clippy::cast_ptr_alignment,
            reason = "_mm256_storeu_si256 is the unaligned-store intrinsic by design; the __m256i-typed pointer is never dereferenced except by this intrinsic"
        )]
        unsafe {
            _mm256_storeu_si256(dst_ptr.cast::<__m256i>(), result);
        }
    }

    // Handle remainder
    if remainder_start < dst.len() {
        fill_span_blend_scalar(&mut dst[remainder_start..], src);
    }
}

// ============================================================================
// ARM NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_span_blend_neon(dst: &mut [u8], src: Color) {
    use std::arch::aarch64::{
        uint8x8_t, uint8x8x4_t, uint16x8_t, vaddq_u16, vdupq_n_u16, vld4_u8, vmovl_u8, vmovn_u16,
        vmulq_u16, vshrq_n_u16, vst4_u8,
    };

    let len = dst.len() / 4;
    // Process 8 pixels per iteration. `vld4_u8`/`vst4_u8` each touch exactly
    // 32 bytes (4 channels x 8 lanes), so the load, the store, and the pointer
    // advance are all 32 bytes — no out-of-bounds reads/writes and no pixel is
    // blended twice. (The previous code loaded 64 B via `vld4q_u8`, stored
    // 32 B via `vst4_u8`, and advanced only 16 B.)
    let chunks = len / 8;
    let remainder_start = chunks * 32;

    // Prepare premultiplied-source vectors (broadcast to 16-bit lanes).
    // Register-only NEON intrinsics are safe inside this `target_feature`
    // function; only the pointer loads/stores below need `unsafe`.
    let src_r = vdupq_n_u16(u16::from(src.red()));
    let src_g = vdupq_n_u16(u16::from(src.green()));
    let src_b = vdupq_n_u16(u16::from(src.blue()));
    let src_a = vdupq_n_u16(u16::from(src.alpha()));
    let inv_alpha = vdupq_n_u16(u16::from(255 - src.alpha()));
    let c128 = vdupq_n_u16(128);

    let ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 32;
        let dst_ptr = unsafe { ptr.add(offset) };

        // Load 8 pixels, deinterleaved into separate R, G, B, A (8 lanes each).
        let dst_rgba = unsafe { vld4_u8(dst_ptr) };

        // Widen each channel to 16-bit for arithmetic.
        let dst_r = vmovl_u8(dst_rgba.0);
        let dst_g = vmovl_u8(dst_rgba.1);
        let dst_b = vmovl_u8(dst_rgba.2);
        let dst_a = vmovl_u8(dst_rgba.3);

        // Premultiplied SrcOver with SkMulDiv255Round on the dst term:
        //   prod = dst * inv_alpha + 128
        //   res  = (prod + (prod >> 8)) >> 8
        //   out  = src + res              (src is premultiplied)
        // Identical to the scalar and AVX2 paths.
        let blend = |s: uint16x8_t, d: uint16x8_t| -> uint8x8_t {
            let prod = vaddq_u16(vmulq_u16(d, inv_alpha), c128);
            let res = vshrq_n_u16(vaddq_u16(prod, vshrq_n_u16(prod, 8)), 8);
            vmovn_u16(vaddq_u16(s, res))
        };

        let result_r = blend(src_r, dst_r);
        let result_g = blend(src_g, dst_g);
        let result_b = blend(src_b, dst_b);
        let result_a = blend(src_a, dst_a);

        // Interleave and store exactly the 8 pixels (32 bytes) we loaded.
        let result = uint8x8x4_t(result_r, result_g, result_b, result_a);
        unsafe { vst4_u8(dst_ptr, result) };
    }

    // Scalar tail for the remaining < 8 pixels.
    if remainder_start < dst.len() {
        fill_span_blend_scalar(&mut dst[remainder_start..], src);
    }
}

// ============================================================================
// Batch pixel blending
// ============================================================================

/// Blend multiple source pixels onto destination pixels (SIMD-optimized).
///
/// Both `src` and `dst` must have the same length (multiple of 4 bytes).
#[inline]
pub fn blend_pixels_src_over(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len() % 4, 0);

    // For now, use scalar implementation
    // SIMD version would require more complex per-pixel alpha handling
    blend_pixels_src_over_scalar(dst, src);
}

/// Scalar fallback for pixel blending.
fn blend_pixels_src_over_scalar(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<4>().0)
    {
        let sa = u32::from(s[3]);
        if sa == 0 {
            continue; // Fully transparent source
        }
        if sa == 255 {
            d.copy_from_slice(s); // Fully opaque source
            continue;
        }

        let inv_sa = 255 - sa;
        d[0] = ((u32::from(s[0]) * 255 + u32::from(d[0]) * inv_sa) / 255).min(255) as u8;
        d[1] = ((u32::from(s[1]) * 255 + u32::from(d[1]) * inv_sa) / 255).min(255) as u8;
        d[2] = ((u32::from(s[2]) * 255 + u32::from(d[2]) * inv_sa) / 255).min(255) as u8;
        d[3] = ((sa * 255 + u32::from(d[3]) * inv_sa) / 255).min(255) as u8;
    }
}

// ============================================================================
// Premultiply/Unpremultiply operations
// ============================================================================

/// Premultiply alpha for a span of pixels (in-place).
#[inline]
pub fn premultiply_span(pixels: &mut [u8]) {
    for chunk in pixels.as_chunks_mut::<4>().0 {
        let a = u32::from(chunk[3]);
        if a == 255 {
            continue; // Already effectively premultiplied
        }
        if a == 0 {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
            continue;
        }
        chunk[0] = mul_div_255_round(u32::from(chunk[0]), a);
        chunk[1] = mul_div_255_round(u32::from(chunk[1]), a);
        chunk[2] = mul_div_255_round(u32::from(chunk[2]), a);
    }
}

/// Unpremultiply alpha for a span of pixels (in-place).
#[inline]
pub fn unpremultiply_span(pixels: &mut [u8]) {
    for chunk in pixels.as_chunks_mut::<4>().0 {
        let a = u32::from(chunk[3]);
        if a == 0 || a == 255 {
            continue;
        }
        // Rounded unpremultiply: (c * 255 + a/2) / a.
        chunk[0] = ((u32::from(chunk[0]) * 255 + a / 2) / a).min(255) as u8;
        chunk[1] = ((u32::from(chunk[1]) * 255 + a / 2) / a).min(255) as u8;
        chunk[2] = ((u32::from(chunk[2]) * 255 + a / 2) / a).min(255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let caps = simd_capabilities();
        println!("SIMD Capabilities: {caps:?}");
        println!("Best width: {} pixels", caps.best_width());

        // Should detect at least scalar (width 1)
        assert!(caps.best_width() >= 1);
    }

    #[test]
    fn test_fill_span_opaque() {
        let mut dst = vec![0u8; 16]; // 4 pixels
        let color = Color::from_argb(255, 255, 128, 64);

        fill_span_solid(&mut dst, color);

        // Check all pixels are filled correctly
        for chunk in dst.as_chunks::<4>().0 {
            assert_eq!(chunk[0], 255); // R
            assert_eq!(chunk[1], 128); // G
            assert_eq!(chunk[2], 64); // B
            assert_eq!(chunk[3], 255); // A
        }
    }

    #[test]
    fn test_fill_span_transparent() {
        let mut dst = vec![100u8; 16];
        let color = Color::from_argb(0, 255, 128, 64);

        fill_span_solid(&mut dst, color);

        // Should remain unchanged (transparent source)
        for byte in &dst {
            assert_eq!(*byte, 100);
        }
    }

    #[test]
    fn test_fill_span_blend_scalar() {
        let mut dst = vec![128u8; 16]; // Gray premultiplied background
        // Premultiplied 50% red = (128, 128, 0, 0).
        let color = Color::from_argb(128, 128, 0, 0);

        fill_span_blend_scalar(&mut dst, color);

        // Premul SrcOver: out = src + round(dst * (255 - 128)).
        // Red: 128 + round(128*127/255)=128+64=192 (increase).
        // Green/Blue: 0 + round(128*127/255)=64 (decrease from 128).
        for chunk in dst.as_chunks::<4>().0 {
            assert_eq!(chunk[0], 192, "Red premul SrcOver");
            assert_eq!(chunk[1], 64, "Green premul SrcOver");
            assert_eq!(chunk[2], 64, "Blue premul SrcOver");
        }
    }

    #[test]
    fn test_blend_pixels_src_over() {
        let mut dst = vec![100u8; 16];
        let src = vec![200u8; 16];

        blend_pixels_src_over(&mut dst, &src);

        // With alpha=200, result should be mostly source
        for chunk in dst.as_chunks::<4>().0 {
            assert!(chunk[0] > 150); // Should be close to 200
        }
    }

    #[test]
    fn test_premultiply_span() {
        let mut pixels = vec![200, 100, 50, 128, 255, 255, 255, 255, 100, 100, 100, 0];

        premultiply_span(&mut pixels);

        // First pixel: 50% alpha
        assert_eq!(pixels[0], 100); // 200 * 128 / 255 ≈ 100
        assert_eq!(pixels[1], 50); // 100 * 128 / 255 ≈ 50
        assert_eq!(pixels[2], 25); // 50 * 128 / 255 ≈ 25
        assert_eq!(pixels[3], 128); // Alpha unchanged

        // Second pixel: opaque - unchanged
        assert_eq!(pixels[4], 255);
        assert_eq!(pixels[5], 255);
        assert_eq!(pixels[6], 255);
        assert_eq!(pixels[7], 255);

        // Third pixel: transparent - RGB zeroed
        assert_eq!(pixels[8], 0);
        assert_eq!(pixels[9], 0);
        assert_eq!(pixels[10], 0);
        assert_eq!(pixels[11], 0);
    }

    #[test]
    fn test_unpremultiply_span() {
        let mut pixels = vec![100, 50, 25, 128]; // Premultiplied 50% alpha

        unpremultiply_span(&mut pixels);

        // Should recover original values (approximately)
        assert!((i32::from(pixels[0]) - 200).abs() <= 2);
        assert!((i32::from(pixels[1]) - 100).abs() <= 2);
        assert!((i32::from(pixels[2]) - 50).abs() <= 2);
        assert_eq!(pixels[3], 128);
    }

    #[test]
    fn test_fill_span_solid_various_sizes() {
        // Test with various buffer sizes to exercise SIMD and scalar paths
        for num_pixels in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
            let mut dst = vec![50u8; num_pixels * 4];
            let color = Color::from_argb(200, 100, 150, 200);

            fill_span_solid(&mut dst, color);

            // Verify all pixels are blended correctly
            for (i, chunk) in dst.as_chunks::<4>().0.iter().enumerate() {
                assert!(
                    chunk[0] > 50 && chunk[0] < 200,
                    "Pixel {} R incorrect: {}",
                    i,
                    chunk[0]
                );
            }
        }
    }

    #[test]
    fn test_fill_span_blend_half_alpha_correctness() {
        // Premultiplied SrcOver: out = src + SkMulDiv255Round(dst, 255 - srcA).
        // src is premultiplied half-alpha white = (128,128,128,128). Over black:
        //   out = 128 + round(0 * 127 / 255) = 128 for every channel.
        // The scalar, AVX2 and NEON paths all use this SAME rounded formula, so
        // they must agree BIT-EXACTLY (no ±1 tolerance).
        let mut dst_scalar = vec![0u8; 64]; // 16 RGBA pixels, all black
        let mut dst_simd = vec![0u8; 64];
        let src = Color::from_argb(128, 128, 128, 128); // premultiplied half-white

        fill_span_blend_scalar(&mut dst_scalar, src);
        fill_span_solid(&mut dst_simd, src);

        for chunk in dst_scalar.as_chunks::<4>().0 {
            assert_eq!(chunk, &[128, 128, 128, 128], "premul SrcOver over black");
        }
        assert_eq!(dst_scalar, dst_simd, "SIMD and scalar must be bit-exact");
    }

    /// Differential test: the SIMD `fill_span_solid`, the scalar
    /// `fill_span_blend_scalar`, and `raster::blend_colors(SrcOver)` must all
    /// produce identical bytes for premultiplied `SrcOver` over arbitrary
    /// destinations and span lengths.
    #[test]
    fn test_simd_scalar_blend_colors_agree_bit_exact() {
        use skia_rs_paint::BlendMode;

        // Deterministic pseudo-random destinations.
        let mut seed: u32 = 0x1234_5678;
        let mut rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        };

        // A few premultiplied sources (r,g,b <= a).
        let sources = [
            Color::from_argb(128, 128, 64, 32),
            Color::from_argb(200, 150, 100, 50),
            Color::from_argb(1, 1, 0, 0),
            Color::from_argb(254, 200, 200, 200),
        ];

        for &src in &sources {
            for len in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64] {
                let base: Vec<u8> = (0..len * 4).map(|_| rng()).collect();

                let mut simd = base.clone();
                fill_span_solid(&mut simd, src);

                let mut scalar = base.clone();
                fill_span_blend_scalar(&mut scalar, src);

                assert_eq!(simd, scalar, "SIMD vs scalar, src {src:?}, len {len}");

                // And against blend_colors used by the per-pixel path.
                for (i, chunk) in base.as_chunks::<4>().0.iter().enumerate() {
                    let dst = Color::from_argb(chunk[3], chunk[0], chunk[1], chunk[2]);
                    let expect = crate::raster::blend_colors_for_test(src, dst, BlendMode::SrcOver);
                    let off = i * 4;
                    assert_eq!(
                        [simd[off], simd[off + 1], simd[off + 2], simd[off + 3]],
                        [expect.red(), expect.green(), expect.blue(), expect.alpha()],
                        "blend_colors vs SIMD at pixel {i}"
                    );
                }
            }
        }
    }

    /// NEON must match the scalar path bit-exactly across odd span lengths and
    /// must never read or write out of bounds. Runs only on aarch64 (where the
    /// NEON path is compiled); the logic is also fixed by inspection so an
    /// aarch64 CI run is needed to exercise it.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_odd_lengths() {
        let caps = simd_capabilities();
        if !caps.neon {
            return;
        }
        let src = Color::from_argb(150, 120, 80, 40); // premultiplied
        for len in [1usize, 2, 3, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 100] {
            // Pad the buffer with guard bytes to catch OOB writes.
            let guard = 8;
            let mut buf = vec![0xABu8; len * 4 + guard * 2];
            // Byte index modulo 256, so the pattern wraps deterministically.
            for (b, i) in buf.iter_mut().zip((0..=u8::MAX).cycle()) {
                *b = i.wrapping_mul(7).wrapping_add(3);
            }
            let mut expected = buf.clone();

            let span = guard..guard + len * 4;
            unsafe { fill_span_blend_neon(&mut buf[span.clone()], src) };
            fill_span_blend_scalar(&mut expected[span], src);

            assert_eq!(buf, expected, "NEON vs scalar mismatch/OOB at len {len}");
        }
    }
}
