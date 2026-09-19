//! Pixel formats and image storage.

use crate::cast::f32_to_u8_sat;
use crate::color::{AlphaType, ColorSpace, ColorType, mul_div_255_round};
use crate::geometry::{IRect, ISize};
use bitflags::bitflags;
use thiserror::Error;

// =============================================================================
// Errors
// =============================================================================

/// Errors related to pixel operations.
#[derive(Debug, Error)]
pub enum PixelError {
    /// Invalid image dimensions.
    #[error("invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Width.
        width: i32,
        /// Height.
        height: i32,
    },

    /// Row bytes too small for the width.
    #[error("row bytes {row_bytes} too small for width {width} with {bpp} bytes per pixel")]
    RowBytesTooSmall {
        /// Provided row bytes.
        row_bytes: usize,
        /// Image width.
        width: i32,
        /// Bytes per pixel.
        bpp: usize,
    },

    /// Buffer too small.
    #[error("buffer size {actual} too small, need {required}")]
    BufferTooSmall {
        /// Required size.
        required: usize,
        /// Actual size.
        actual: usize,
    },

    /// Unsupported color type.
    #[error("unsupported color type")]
    UnsupportedColorType,
}

// =============================================================================
// Image Info
// =============================================================================

/// Describes the properties of pixel data.
///
/// Equivalent to Skia's `SkImageInfo`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageInfo {
    /// Image dimensions.
    pub dimensions: ISize,
    /// Pixel format.
    pub color_type: ColorType,
    /// Alpha interpretation.
    pub alpha_type: AlphaType,
    /// Color space (optional).
    pub color_space: Option<ColorSpace>,
}

impl ImageInfo {
    /// Creates a new image info with the specified properties.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new(
        width: i32,
        height: i32,
        color_type: ColorType,
        alpha_type: AlphaType,
    ) -> Result<Self, PixelError> {
        // Skia's SkImageInfo::Make accepts zero dimensions as a legal empty
        // info sentinel; only negative dimensions are rejected.
        if width < 0 || height < 0 {
            return Err(PixelError::InvalidDimensions { width, height });
        }
        Ok(Self {
            dimensions: ISize::new(width, height),
            color_type,
            alpha_type,
            color_space: None,
        })
    }

    /// Creates image info with sRGB color space.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub fn new_srgb(
        width: i32,
        height: i32,
        color_type: ColorType,
        alpha_type: AlphaType,
    ) -> Result<Self, PixelError> {
        let mut info = Self::new(width, height, color_type, alpha_type)?;
        info.color_space = Some(ColorSpace::srgb());
        Ok(info)
    }

    /// Creates RGBA 8888 image info.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new_rgba8888(
        width: i32,
        height: i32,
        alpha_type: AlphaType,
    ) -> Result<Self, PixelError> {
        Self::new(width, height, ColorType::Rgba8888, alpha_type)
    }

    /// Creates BGRA 8888 image info (native on little-endian).
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new_bgra8888(
        width: i32,
        height: i32,
        alpha_type: AlphaType,
    ) -> Result<Self, PixelError> {
        Self::new(width, height, ColorType::Bgra8888, alpha_type)
    }

    /// Creates native 32-bit image info.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new_n32(
        width: i32,
        height: i32,
        alpha_type: AlphaType,
    ) -> Result<Self, PixelError> {
        Self::new(width, height, ColorType::n32(), alpha_type)
    }

    /// Creates opaque native 32-bit image info.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new_n32_opaque(width: i32, height: i32) -> Result<Self, PixelError> {
        Self::new_n32(width, height, AlphaType::Opaque)
    }

    /// Creates premultiplied native 32-bit image info.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new_n32_premul(width: i32, height: i32) -> Result<Self, PixelError> {
        Self::new_n32(width, height, AlphaType::Premul)
    }

    /// Creates alpha-only 8-bit image info.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is
    /// negative.
    #[inline]
    pub const fn new_alpha8(width: i32, height: i32) -> Result<Self, PixelError> {
        Self::new(width, height, ColorType::Alpha8, AlphaType::Premul)
    }

    /// Returns the image width.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.dimensions.width
    }

    /// Returns the image height.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.dimensions.height
    }

    /// Returns true if the image has zero area.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.dimensions.is_empty()
    }

    /// Returns bytes per pixel.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> usize {
        self.color_type.bytes_per_pixel()
    }

    /// Returns the minimum row bytes for this image.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "ImageInfo width is validated non-negative in ImageInfo::new; const fn cannot use TryFrom"
    )]
    pub const fn min_row_bytes(&self) -> usize {
        self.width() as usize * self.bytes_per_pixel()
    }

    /// Computes the byte size for the given row bytes.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "height is checked > 0 above, so the cast is non-negative; const fn cannot use TryFrom"
    )]
    pub const fn compute_byte_size(&self, row_bytes: usize) -> usize {
        if self.height() <= 0 {
            return 0;
        }
        // Last row doesn't need full row_bytes, just the actual pixels
        let last_row = self.min_row_bytes();
        let other_rows = row_bytes * (self.height() as usize - 1);
        other_rows + last_row
    }

    /// Returns the byte size using minimum row bytes.
    #[inline]
    #[must_use]
    pub const fn min_byte_size(&self) -> usize {
        self.compute_byte_size(self.min_row_bytes())
    }

    /// Returns the bounds as an `IRect`.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> IRect {
        IRect::from_size(self.dimensions)
    }

    /// Returns true if the alpha type is opaque.
    #[inline]
    #[must_use]
    pub const fn is_opaque(&self) -> bool {
        self.alpha_type.is_opaque()
    }

    /// Returns a new image info with the specified alpha type.
    #[inline]
    #[must_use]
    pub fn with_alpha_type(&self, alpha_type: AlphaType) -> Self {
        Self {
            alpha_type,
            ..self.clone()
        }
    }

    /// Returns a new image info with the specified color type.
    #[inline]
    #[must_use]
    pub fn with_color_type(&self, color_type: ColorType) -> Self {
        Self {
            color_type,
            ..self.clone()
        }
    }

    /// Returns a new image info with the specified color space.
    #[inline]
    #[must_use]
    pub fn with_color_space(&self, color_space: Option<ColorSpace>) -> Self {
        Self {
            color_space,
            ..self.clone()
        }
    }

    /// Returns a new image info with the specified dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::InvalidDimensions`] if `width` or `height` is not
    /// positive.
    #[inline]
    pub fn with_dimensions(&self, width: i32, height: i32) -> Result<Self, PixelError> {
        if width <= 0 || height <= 0 {
            return Err(PixelError::InvalidDimensions { width, height });
        }
        Ok(Self {
            dimensions: ISize::new(width, height),
            ..self.clone()
        })
    }

    /// Validates that `row_bytes` is sufficient for this image.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::RowBytesTooSmall`] if `row_bytes` is less than
    /// [`min_row_bytes`](Self::min_row_bytes).
    #[inline]
    pub const fn validate_row_bytes(&self, row_bytes: usize) -> Result<(), PixelError> {
        let min = self.min_row_bytes();
        if row_bytes < min {
            Err(PixelError::RowBytesTooSmall {
                row_bytes,
                width: self.width(),
                bpp: self.bytes_per_pixel(),
            })
        } else {
            Ok(())
        }
    }
}

impl Default for ImageInfo {
    fn default() -> Self {
        Self {
            dimensions: ISize::empty(),
            color_type: ColorType::Unknown,
            alpha_type: AlphaType::Unknown,
            color_space: None,
        }
    }
}

// =============================================================================
// Pixmap (read-only pixel access)
// =============================================================================

/// Read-only access to pixel data.
///
/// Equivalent to Skia's `SkPixmap`.
#[derive(Debug)]
pub struct Pixmap<'a> {
    /// Image properties.
    pub info: ImageInfo,
    /// Pixel data.
    pixels: &'a [u8],
    /// Bytes per row (may include padding).
    row_bytes: usize,
}

impl<'a> Pixmap<'a> {
    /// Creates a new pixmap wrapping existing pixel data.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::RowBytesTooSmall`] if `row_bytes` is too small for
    /// the image width, or [`PixelError::BufferTooSmall`] if `pixels` is shorter
    /// than the computed byte size.
    pub fn new(info: ImageInfo, pixels: &'a [u8], row_bytes: usize) -> Result<Self, PixelError> {
        info.validate_row_bytes(row_bytes)?;

        let required = info.compute_byte_size(row_bytes);
        if pixels.len() < required {
            return Err(PixelError::BufferTooSmall {
                required,
                actual: pixels.len(),
            });
        }

        Ok(Self {
            info,
            pixels,
            row_bytes,
        })
    }

    /// Returns the image info.
    #[inline]
    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Returns the raw pixel data.
    #[inline]
    #[must_use]
    pub const fn pixels(&self) -> &[u8] {
        self.pixels
    }

    /// Returns the row bytes.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Returns the image width.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.info.width()
    }

    /// Returns the image height.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.info.height()
    }

    /// Returns a pointer to the start of a row.
    #[inline]
    #[must_use]
    pub fn row(&self, y: i32) -> Option<&[u8]> {
        if y < 0 || y >= self.height() {
            return None;
        }
        let offset = usize::try_from(y).unwrap_or(0) * self.row_bytes;
        let end = offset + self.info.min_row_bytes();
        Some(&self.pixels[offset..end])
    }

    /// Returns the address of a specific pixel.
    #[inline]
    #[must_use]
    pub fn pixel_addr(&self, x: i32, y: i32) -> Option<&[u8]> {
        if x < 0 || x >= self.width() || y < 0 || y >= self.height() {
            return None;
        }
        let bpp = self.info.bytes_per_pixel();
        let offset = usize::try_from(y).unwrap_or(0) * self.row_bytes
            + usize::try_from(x).unwrap_or(0) * bpp;
        Some(&self.pixels[offset..offset + bpp])
    }
}

// =============================================================================
// Bitmap (mutable pixel storage)
// =============================================================================

/// Mutable pixel storage.
///
/// Equivalent to Skia's `SkBitmap`.
#[derive(Debug, Clone)]
pub struct Bitmap {
    /// Image properties.
    info: ImageInfo,
    /// Owned pixel data.
    pixels: Vec<u8>,
    /// Bytes per row.
    row_bytes: usize,
}

impl Bitmap {
    /// Creates a new empty bitmap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            info: ImageInfo::default(),
            pixels: Vec::new(),
            row_bytes: 0,
        }
    }

    /// Allocates a bitmap with the specified properties.
    ///
    /// # Errors
    ///
    /// Currently always succeeds, but returns [`Result`] for forward
    /// compatibility with allocation-failure reporting.
    pub fn allocate(info: ImageInfo) -> Result<Self, PixelError> {
        let row_bytes = info.min_row_bytes();
        let size = info.compute_byte_size(row_bytes);
        let pixels = vec![0u8; size];

        Ok(Self {
            info,
            pixels,
            row_bytes,
        })
    }

    /// Allocates a bitmap with custom row bytes.
    ///
    /// # Errors
    ///
    /// Returns [`PixelError::RowBytesTooSmall`] if `row_bytes` is too small for
    /// the image width.
    pub fn allocate_with_row_bytes(info: ImageInfo, row_bytes: usize) -> Result<Self, PixelError> {
        info.validate_row_bytes(row_bytes)?;
        let size = info.compute_byte_size(row_bytes);
        let pixels = vec![0u8; size];

        Ok(Self {
            info,
            pixels,
            row_bytes,
        })
    }

    /// Returns the image info.
    #[inline]
    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Returns the raw pixel data.
    #[inline]
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Returns mutable pixel data.
    #[inline]
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Returns the row bytes.
    #[inline]
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Returns the image width.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.info.width()
    }

    /// Returns the image height.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.info.height()
    }

    /// Returns true if the bitmap has no pixels.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Returns a pointer to the start of a row.
    #[inline]
    #[must_use]
    pub fn row(&self, y: i32) -> Option<&[u8]> {
        if y < 0 || y >= self.height() {
            return None;
        }
        let offset = usize::try_from(y).unwrap_or(0) * self.row_bytes;
        let end = offset + self.info.min_row_bytes();
        Some(&self.pixels[offset..end])
    }

    /// Returns a mutable pointer to the start of a row.
    #[inline]
    pub fn row_mut(&mut self, y: i32) -> Option<&mut [u8]> {
        if y < 0 || y >= self.height() {
            return None;
        }
        let offset = usize::try_from(y).unwrap_or(0) * self.row_bytes;
        let min_row = self.info.min_row_bytes();
        let end = offset + min_row;
        Some(&mut self.pixels[offset..end])
    }

    /// Returns a read-only pixmap view.
    #[inline]
    #[must_use]
    pub fn as_pixmap(&self) -> Pixmap<'_> {
        Pixmap {
            info: self.info.clone(),
            pixels: &self.pixels,
            row_bytes: self.row_bytes,
        }
    }

    /// Fills the bitmap with zeros.
    #[inline]
    pub fn erase(&mut self) {
        self.pixels.fill(0);
    }

    /// Fills the bitmap with a specific byte value.
    #[inline]
    pub fn fill(&mut self, value: u8) {
        self.pixels.fill(value);
    }
}

impl Default for Bitmap {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Surface Properties
// =============================================================================

bitflags! {
    /// Surface properties flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct SurfacePropsFlags: u32 {
        /// Use device-independent fonts.
        const USE_DEVICE_INDEPENDENT_FONTS = 1 << 0;
    }
}

/// Pixel geometry for LCD text rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PixelGeometry {
    /// Unknown pixel geometry.
    #[default]
    Unknown = 0,
    /// Horizontal RGB subpixels.
    RgbH,
    /// Horizontal BGR subpixels.
    BgrH,
    /// Vertical RGB subpixels.
    RgbV,
    /// Vertical BGR subpixels.
    BgrV,
}

/// Properties of a surface (for text rendering hints).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SurfaceProps {
    /// Flags.
    pub flags: SurfacePropsFlags,
    /// Pixel geometry.
    pub pixel_geometry: PixelGeometry,
}

impl SurfaceProps {
    /// Create new surface properties.
    #[inline]
    #[must_use]
    pub const fn new(flags: SurfacePropsFlags, pixel_geometry: PixelGeometry) -> Self {
        Self {
            flags,
            pixel_geometry,
        }
    }
}

// =============================================================================
// Pixel Format Conversion
// =============================================================================

/// Convert pixels between color types.
///
/// This handles common pixel format conversions used in graphics applications.
///
/// # Errors
///
/// Returns [`PixelError::InvalidDimensions`] if the source and destination
/// dimensions differ, [`PixelError::RowBytesTooSmall`] if either `row_bytes`
/// is too small, or [`PixelError::BufferTooSmall`] if either buffer is shorter
/// than its computed byte size.
pub fn convert_pixels(
    src: &[u8],
    src_info: &ImageInfo,
    src_row_bytes: usize,
    dst: &mut [u8],
    dst_info: &ImageInfo,
    dst_row_bytes: usize,
) -> Result<(), PixelError> {
    // Validate dimensions match
    if src_info.width() != dst_info.width() || src_info.height() != dst_info.height() {
        return Err(PixelError::InvalidDimensions {
            width: dst_info.width(),
            height: dst_info.height(),
        });
    }

    src_info.validate_row_bytes(src_row_bytes)?;
    dst_info.validate_row_bytes(dst_row_bytes)?;

    let required_src = src_info.compute_byte_size(src_row_bytes);
    if src.len() < required_src {
        return Err(PixelError::BufferTooSmall {
            required: required_src,
            actual: src.len(),
        });
    }

    let required_dst = dst_info.compute_byte_size(dst_row_bytes);
    if dst.len() < required_dst {
        return Err(PixelError::BufferTooSmall {
            required: required_dst,
            actual: dst.len(),
        });
    }

    let width = usize::try_from(src_info.width()).unwrap_or(0);
    let height = usize::try_from(src_info.height()).unwrap_or(0);

    // Compute alpha conversion mode once; apply per-row for cache locality.
    let alpha_conv = AlphaConversion::from_alpha_types(
        src_info.alpha_type,
        dst_info.alpha_type,
        dst_info.color_type.has_alpha(),
    );

    for y in 0..height {
        let src_row_start = y * src_row_bytes;
        let dst_row_start = y * dst_row_bytes;

        convert_row(
            &src[src_row_start..],
            src_info.color_type,
            &mut dst[dst_row_start..],
            dst_info.color_type,
            width,
            alpha_conv,
        )?;
    }

    Ok(())
}

/// How to convert alpha during pixel conversion.
#[derive(Clone, Copy, Debug)]
enum AlphaConversion {
    None,
    Premultiply,
    Unpremultiply,
}

impl AlphaConversion {
    fn from_alpha_types(src: AlphaType, dst: AlphaType, color_has_alpha: bool) -> Self {
        if !color_has_alpha || src == dst {
            return Self::None;
        }
        match (src, dst) {
            (AlphaType::Unpremul, AlphaType::Premul) => Self::Premultiply,
            (AlphaType::Premul, AlphaType::Unpremul) => Self::Unpremultiply,
            _ => Self::None,
        }
    }
}

/// Convert a single row of pixels, applying alpha conversion in the same pass.
#[allow(
    clippy::too_many_lines,
    reason = "faithful port of Skia's per-format conversion switch; splitting reduces fidelity"
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "BT.709 luma is a faithful port of Skia's fma-free reference arithmetic; mul_add's fused rounding diverges from Skia"
)]
fn convert_row(
    src: &[u8],
    src_type: ColorType,
    dst: &mut [u8],
    dst_type: ColorType,
    width: usize,
    alpha_conv: AlphaConversion,
) -> Result<(), PixelError> {
    use ColorType::{
        Alpha8, Bgra8888, Gray8, Rgb565, Rgb888, Rgba8888, RgbaF16, RgbaF16Norm, RgbaF32,
    };

    // Same format - just copy
    if src_type == dst_type {
        let bpp = src_type.bytes_per_pixel();
        let row_len = width * bpp;
        dst[..row_len].copy_from_slice(&src[..row_len]);
        apply_alpha_conversion(&mut dst[..row_len], dst_type, width, alpha_conv);
        return Ok(());
    }

    match (src_type, dst_type) {
        // RGBA8888 <-> BGRA8888
        (Rgba8888, Bgra8888) | (Bgra8888, Rgba8888) => {
            for i in 0..width {
                let si = i * 4;
                dst[si] = src[si + 2]; // R <-> B
                dst[si + 1] = src[si + 1]; // G
                dst[si + 2] = src[si]; // B <-> R
                dst[si + 3] = src[si + 3]; // A
            }
        }

        // RGB888 -> RGBA8888
        (Rgb888, Rgba8888) => {
            for i in 0..width {
                let si = i * 3;
                let di = i * 4;
                dst[di] = src[si];
                dst[di + 1] = src[si + 1];
                dst[di + 2] = src[si + 2];
                dst[di + 3] = 255;
            }
        }

        // RGBA8888 -> RGB888 (drop alpha)
        (Rgba8888, Rgb888) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 3;
                dst[di] = src[si];
                dst[di + 1] = src[si + 1];
                dst[di + 2] = src[si + 2];
            }
        }

        // RGB565 -> RGBA8888
        (Rgb565, Rgba8888) => {
            for i in 0..width {
                let si = i * 2;
                let di = i * 4;
                let pixel = u16::from_le_bytes([src[si], src[si + 1]]);
                let r = ((pixel >> 11) & 0x1F) as u8;
                let g = ((pixel >> 5) & 0x3F) as u8;
                let b = (pixel & 0x1F) as u8;
                // Replicate high bits into the low bits so full-scale maps to
                // 255, matching Skia's SkR16ToR32 / SkG16ToG32 / SkB16ToB32.
                dst[di] = (r << 3) | (r >> 2); // R (5 bits -> 8 bits)
                dst[di + 1] = (g << 2) | (g >> 4); // G (6 bits -> 8 bits)
                dst[di + 2] = (b << 3) | (b >> 2); // B (5 bits -> 8 bits)
                dst[di + 3] = 255;
            }
        }

        // RGBA8888 -> RGB565
        (Rgba8888, Rgb565) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 2;
                let r = u16::from(src[si] >> 3);
                let g = u16::from(src[si + 1] >> 2);
                let b = u16::from(src[si + 2] >> 3);
                let pixel = (r << 11) | (g << 5) | b;
                let bytes = pixel.to_le_bytes();
                dst[di] = bytes[0];
                dst[di + 1] = bytes[1];
            }
        }

        // Gray8 -> RGBA8888
        (Gray8, Rgba8888) => {
            for (&gray, out) in src.iter().zip(dst.chunks_exact_mut(4)).take(width) {
                out[0] = gray;
                out[1] = gray;
                out[2] = gray;
                out[3] = 255;
            }
        }

        // RGBA8888 -> Gray8 (luminance)
        (Rgba8888, Gray8) => {
            for (chunk, out) in src.chunks_exact(4).zip(dst.iter_mut()).take(width) {
                let r = f32::from(chunk[0]);
                let g = f32::from(chunk[1]);
                let b = f32::from(chunk[2]);
                // ITU-R BT.709 luma coefficients, matching Skia's
                // bt709_luminance_or_luma_to_alpha raster-pipeline stage.
                *out = f32_to_u8_sat(0.2126 * r + 0.7152 * g + 0.0722 * b);
            }
        }

        // Alpha8 -> RGBA8888 (white with alpha)
        (Alpha8, Rgba8888) => {
            for (&alpha, out) in src.iter().zip(dst.chunks_exact_mut(4)).take(width) {
                out[0] = 255;
                out[1] = 255;
                out[2] = 255;
                out[3] = alpha;
            }
        }

        // RGBA8888 -> Alpha8 (extract alpha)
        (Rgba8888, Alpha8) => {
            for i in 0..width {
                dst[i] = src[i * 4 + 3];
            }
        }

        // BGRA8888 -> RGB888
        (Bgra8888, Rgb888) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 3;
                dst[di] = src[si + 2]; // R
                dst[di + 1] = src[si + 1]; // G
                dst[di + 2] = src[si]; // B
            }
        }

        // RGB888 -> BGRA8888
        (Rgb888, Bgra8888) => {
            for i in 0..width {
                let si = i * 3;
                let di = i * 4;
                dst[di] = src[si + 2]; // B
                dst[di + 1] = src[si + 1]; // G
                dst[di + 2] = src[si]; // R
                dst[di + 3] = 255;
            }
        }

        // ---- F16 (IEEE 754 binary16) ----
        (Rgba8888, RgbaF16 | RgbaF16Norm) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 8;
                for c in 0..4 {
                    let bytes = f16_to_bytes(f32::from(src[si + c]) / 255.0);
                    dst[di + c * 2] = bytes[0];
                    dst[di + c * 2 + 1] = bytes[1];
                }
            }
        }
        (RgbaF16 | RgbaF16Norm, Rgba8888) => {
            for i in 0..width {
                let si = i * 8;
                let di = i * 4;
                for c in 0..4 {
                    let v = f16_from_bytes([src[si + c * 2], src[si + c * 2 + 1]]);
                    dst[di + c] = f32_to_u8_sat(v * 255.0);
                }
            }
        }
        (Bgra8888, RgbaF16 | RgbaF16Norm) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 8;
                let channels = [src[si + 2], src[si + 1], src[si], src[si + 3]];
                for c in 0..4 {
                    let bytes = f16_to_bytes(f32::from(channels[c]) / 255.0);
                    dst[di + c * 2] = bytes[0];
                    dst[di + c * 2 + 1] = bytes[1];
                }
            }
        }
        (RgbaF16 | RgbaF16Norm, Bgra8888) => {
            for i in 0..width {
                let si = i * 8;
                let di = i * 4;
                let r = f16_from_bytes([src[si], src[si + 1]]);
                let g = f16_from_bytes([src[si + 2], src[si + 3]]);
                let b = f16_from_bytes([src[si + 4], src[si + 5]]);
                let a = f16_from_bytes([src[si + 6], src[si + 7]]);
                dst[di] = f32_to_u8_sat(b * 255.0);
                dst[di + 1] = f32_to_u8_sat(g * 255.0);
                dst[di + 2] = f32_to_u8_sat(r * 255.0);
                dst[di + 3] = f32_to_u8_sat(a * 255.0);
            }
        }

        // ---- F32 (IEEE 754 binary32) ----
        (Rgba8888, RgbaF32) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 16;
                for c in 0..4 {
                    let v = (f32::from(src[si + c]) / 255.0).to_le_bytes();
                    dst[di + c * 4..di + c * 4 + 4].copy_from_slice(&v);
                }
            }
        }
        (RgbaF32, Rgba8888) => {
            for i in 0..width {
                let si = i * 16;
                let di = i * 4;
                for c in 0..4 {
                    let v = f32_from_le(&src[si + c * 4..si + c * 4 + 4]);
                    dst[di + c] = f32_to_u8_sat(v * 255.0);
                }
            }
        }
        (Bgra8888, RgbaF32) => {
            for i in 0..width {
                let si = i * 4;
                let di = i * 16;
                let channels = [src[si + 2], src[si + 1], src[si], src[si + 3]];
                for c in 0..4 {
                    let v = (f32::from(channels[c]) / 255.0).to_le_bytes();
                    dst[di + c * 4..di + c * 4 + 4].copy_from_slice(&v);
                }
            }
        }
        (RgbaF32, Bgra8888) => {
            for i in 0..width {
                let si = i * 16;
                let di = i * 4;
                let r = f32_from_le(&src[si..si + 4]);
                let g = f32_from_le(&src[si + 4..si + 8]);
                let b = f32_from_le(&src[si + 8..si + 12]);
                let a = f32_from_le(&src[si + 12..si + 16]);
                dst[di] = f32_to_u8_sat(b * 255.0);
                dst[di + 1] = f32_to_u8_sat(g * 255.0);
                dst[di + 2] = f32_to_u8_sat(r * 255.0);
                dst[di + 3] = f32_to_u8_sat(a * 255.0);
            }
        }

        // ---- F16 <-> F32 ----
        (RgbaF16 | RgbaF16Norm, RgbaF32) => {
            for i in 0..width {
                let si = i * 8;
                let di = i * 16;
                for c in 0..4 {
                    let v = f16_from_bytes([src[si + c * 2], src[si + c * 2 + 1]]);
                    dst[di + c * 4..di + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        (RgbaF32, RgbaF16 | RgbaF16Norm) => {
            for i in 0..width {
                let si = i * 16;
                let di = i * 8;
                for c in 0..4 {
                    let v = f32_from_le(&src[si + c * 4..si + c * 4 + 4]);
                    let bytes = f16_to_bytes(v);
                    dst[di + c * 2] = bytes[0];
                    dst[di + c * 2 + 1] = bytes[1];
                }
            }
        }

        _ => return Err(PixelError::UnsupportedColorType),
    }

    apply_alpha_conversion(
        &mut dst[..width * dst_type.bytes_per_pixel()],
        dst_type,
        width,
        alpha_conv,
    );

    Ok(())
}

/// Apply alpha premultiply / unpremultiply to a row, dispatching on the
/// destination color type.
///
/// Each color type is handled according to its actual byte/bit layout — the
/// generic 4-byte-RGBA path is only used for byte-aligned RGBA/BGRA formats.
/// Packed formats (`Argb4444`, the `1010102` variants, `R16G16B16A16Unorm`) are
/// unpacked to normalized floats, converted with rounding, and repacked so the
/// operation never corrupts pixels. Alpha-only formats have no color channels
/// and are left unchanged.
#[inline]
#[allow(
    clippy::match_same_arms,
    reason = "explicit no-op arm documents that alpha-only formats have no color channels, distinct from the unsupported catch-all"
)]
fn apply_alpha_conversion(
    row: &mut [u8],
    color_type: ColorType,
    width: usize,
    alpha_conv: AlphaConversion,
) {
    use ColorType::{
        A16Float, A16Unorm, Alpha8, Argb4444, Bgra8888, Bgra1010102, R16G16B16A16Unorm, Rgba8888,
        Rgba1010102, RgbaF16, RgbaF16Norm, RgbaF32, Srgba8888,
    };
    let premul = match alpha_conv {
        AlphaConversion::None => return,
        AlphaConversion::Premultiply => true,
        AlphaConversion::Unpremultiply => false,
    };
    match color_type {
        // Byte-aligned RGBA/BGRA: color channels are bytes 0..3, alpha byte 3.
        Rgba8888 | Bgra8888 | Srgba8888 => {
            if premul {
                premultiply_in_place(row);
            } else {
                unpremultiply_in_place(row);
            }
        }
        RgbaF16 | RgbaF16Norm => {
            if premul {
                premultiply_in_place_f16(row, width);
            } else {
                unpremultiply_in_place_f16(row, width);
            }
        }
        RgbaF32 => {
            if premul {
                premultiply_in_place_f32(row, width);
            } else {
                unpremultiply_in_place_f32(row, width);
            }
        }
        // Alpha-only formats have no color channels to (un)premultiply.
        Alpha8 | A16Unorm | A16Float => {}
        Argb4444 => convert_alpha_4444(row, width, premul),
        // Both 1010102 orderings keep alpha in the top 2 bits and all three
        // 10-bit lanes are color, so the conversion is order-independent.
        Rgba1010102 | Bgra1010102 => convert_alpha_1010102(row, width, premul),
        R16G16B16A16Unorm => convert_alpha_16161616(row, width, premul),
        // Any remaining type either has no alpha or isn't produced here.
        _ => {}
    }
}

/// Applies premultiply (`rgb *= a`) or unpremultiply (`rgb /= a`) to normalized
/// color channels, matching the premul semantics of the raster pipeline.
#[inline]
fn apply_alpha_norm(r: f32, g: f32, b: f32, a: f32, premul: bool) -> (f32, f32, f32) {
    if premul {
        (r * a, g * a, b * a)
    } else if a > 0.0 {
        ((r / a).min(1.0), (g / a).min(1.0), (b / a).min(1.0))
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Rounds a normalized value in `[0, 1]` to an integer in `[0, max]`, matching
/// Skia's `to_unorm` (round to nearest).
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "faithful port of Skia to_unorm: clamp to [0,max], explicit *max+0.5 round-half-up (mul_add would diverge from Skia), then saturating cast (no safe std float->uint conversion exists)"
)]
fn to_unorm_round(v: f32, max: f32) -> u32 {
    (v.clamp(0.0, 1.0) * max + 0.5) as u32
}

/// Per-pixel alpha conversion for `Argb4444` (16-bit `rrrr gggg bbbb aaaa`).
#[allow(
    clippy::many_single_char_names,
    reason = "r, g, b, a channel names intentionally match Skia's C++ source"
)]
fn convert_alpha_4444(row: &mut [u8], width: usize, premul: bool) {
    for i in 0..width {
        let off = i * 2;
        let v = u16::from_le_bytes([row[off], row[off + 1]]);
        let a4 = v & 0xF;
        let r = f32::from((v >> 12) & 0xF) / 15.0;
        let g = f32::from((v >> 8) & 0xF) / 15.0;
        let b = f32::from((v >> 4) & 0xF) / 15.0;
        let a = f32::from(a4) / 15.0;
        let (r, g, b) = apply_alpha_norm(r, g, b, a, premul);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "four 4-bit lanes are packed into 16 bits, which always fits u16"
        )]
        let packed = ((to_unorm_round(r, 15.0) << 12)
            | (to_unorm_round(g, 15.0) << 8)
            | (to_unorm_round(b, 15.0) << 4)
            | u32::from(a4)) as u16;
        row[off..off + 2].copy_from_slice(&packed.to_le_bytes());
    }
}

/// Per-pixel alpha conversion for the `1010102` formats (10-bit color lanes,
/// 2-bit alpha in the top bits).
#[expect(
    clippy::cast_precision_loss,
    reason = "10-bit and 2-bit masked values are < 2^24 and convert to f32 exactly"
)]
fn convert_alpha_1010102(row: &mut [u8], width: usize, premul: bool) {
    for i in 0..width {
        let off = i * 4;
        let v = u32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]);
        let a2 = (v >> 30) & 0x3;
        let c0 = (v & 0x3FF) as f32 / 1023.0;
        let c1 = ((v >> 10) & 0x3FF) as f32 / 1023.0;
        let c2 = ((v >> 20) & 0x3FF) as f32 / 1023.0;
        let a = a2 as f32 / 3.0;
        let (c0, c1, c2) = apply_alpha_norm(c0, c1, c2, a, premul);
        let packed = to_unorm_round(c0, 1023.0)
            | (to_unorm_round(c1, 1023.0) << 10)
            | (to_unorm_round(c2, 1023.0) << 20)
            | (a2 << 30);
        row[off..off + 4].copy_from_slice(&packed.to_le_bytes());
    }
}

/// Per-pixel alpha conversion for `R16G16B16A16Unorm` (four little-endian
/// 16-bit unorm channels, R G B A).
#[expect(
    clippy::cast_possible_truncation,
    reason = "to_unorm_round with max=65535 yields a value that always fits u16"
)]
fn convert_alpha_16161616(row: &mut [u8], width: usize, premul: bool) {
    for i in 0..width {
        let off = i * 8;
        let r = f32::from(u16::from_le_bytes([row[off], row[off + 1]])) / 65535.0;
        let g = f32::from(u16::from_le_bytes([row[off + 2], row[off + 3]])) / 65535.0;
        let b = f32::from(u16::from_le_bytes([row[off + 4], row[off + 5]])) / 65535.0;
        let a_raw = u16::from_le_bytes([row[off + 6], row[off + 7]]);
        let a = f32::from(a_raw) / 65535.0;
        let (r, g, b) = apply_alpha_norm(r, g, b, a, premul);
        row[off..off + 2].copy_from_slice(&(to_unorm_round(r, 65535.0) as u16).to_le_bytes());
        row[off + 2..off + 4].copy_from_slice(&(to_unorm_round(g, 65535.0) as u16).to_le_bytes());
        row[off + 4..off + 6].copy_from_slice(&(to_unorm_round(b, 65535.0) as u16).to_le_bytes());
        // Alpha channel (bytes 6..8) is left unchanged.
    }
}

// =============================================================================
// IEEE 754 binary16 (F16) <-> f32 conversion
//
// Skia stores RGBA F16 as 4 consecutive little-endian half-floats per pixel.
// We do straight IEEE conversion (not storage-only, so subnormals and ±Inf
// round correctly).
// =============================================================================

/// Decode a single IEEE 754 binary16 from little-endian bytes into f32.
#[inline]
#[allow(
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    reason = "const fn cannot use From/TryFrom; casts are width-safe operations on IEEE-754 binary16 bit fields"
)]
const fn f16_from_bytes(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = (bits >> 15) as u32 & 0x1;
    let exp = (bits >> 10) as u32 & 0x1f;
    let mant = bits as u32 & 0x3ff;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — normalize into f32 representation.
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = m & 0x3ff;
            let exp32 = (e + 127) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Encode an f32 as IEEE 754 binary16, little-endian. NaN/Inf preserved.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "IEEE-754 binary16 encoding; every cast operates on values masked or bounded to their target bit width"
)]
fn f16_to_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    let result: u16 = if exp32 == 0xff {
        // Inf / NaN
        let m = (mant >> 13) as u16;
        (sign << 15) | (0x1f << 10) | if mant != 0 { m.max(1) } else { 0 }
    } else if exp32 == 0 {
        // Zero / subnormal f32 -> zero f16
        sign << 15
    } else {
        let new_exp = exp32 - 127 + 15;
        if new_exp >= 0x1f {
            // Overflow -> Inf
            (sign << 15) | (0x1f << 10)
        } else if new_exp <= 0 {
            // Underflow -> subnormal or zero
            if new_exp < -10 {
                sign << 15
            } else {
                let shift = 14 - new_exp;
                let mant_with_implicit = mant | 0x80_0000;
                let shifted = mant_with_implicit >> shift;
                // Round-to-nearest-even on the bit just below the kept mantissa.
                let round_bit = (mant_with_implicit >> (shift - 1)) & 0x1;
                let m = (shifted as u16) + round_bit as u16;
                (sign << 15) | m
            }
        } else {
            // Normal f16
            let m = (mant >> 13) as u16;
            let round_bit = ((mant >> 12) & 0x1) as u16;
            let sticky = u16::from((mant & 0xfff) != 0);
            let mut out = (sign << 15) | ((new_exp as u16) << 10) | m;
            // Round-to-nearest-even
            if round_bit == 1 && (sticky == 1 || (m & 0x1) == 1) {
                out = out.wrapping_add(1);
            }
            out
        }
    };

    result.to_le_bytes()
}

#[inline]
const fn f32_from_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn premultiply_in_place_f16(row: &mut [u8], width: usize) {
    for i in 0..width {
        let off = i * 8;
        let a = f16_from_bytes([row[off + 6], row[off + 7]]);
        if a >= 1.0 {
            continue;
        }
        let r = f16_from_bytes([row[off], row[off + 1]]) * a;
        let g = f16_from_bytes([row[off + 2], row[off + 3]]) * a;
        let b = f16_from_bytes([row[off + 4], row[off + 5]]) * a;
        let rb = f16_to_bytes(r);
        let gb = f16_to_bytes(g);
        let bb = f16_to_bytes(b);
        row[off] = rb[0];
        row[off + 1] = rb[1];
        row[off + 2] = gb[0];
        row[off + 3] = gb[1];
        row[off + 4] = bb[0];
        row[off + 5] = bb[1];
    }
}

fn unpremultiply_in_place_f16(row: &mut [u8], width: usize) {
    for i in 0..width {
        let off = i * 8;
        let a = f16_from_bytes([row[off + 6], row[off + 7]]);
        if a <= 0.0 {
            row[off..off + 6].fill(0);
            continue;
        }
        if a >= 1.0 {
            continue;
        }
        let inv = 1.0 / a;
        let r = f16_from_bytes([row[off], row[off + 1]]) * inv;
        let g = f16_from_bytes([row[off + 2], row[off + 3]]) * inv;
        let b = f16_from_bytes([row[off + 4], row[off + 5]]) * inv;
        let rb = f16_to_bytes(r);
        let gb = f16_to_bytes(g);
        let bb = f16_to_bytes(b);
        row[off] = rb[0];
        row[off + 1] = rb[1];
        row[off + 2] = gb[0];
        row[off + 3] = gb[1];
        row[off + 4] = bb[0];
        row[off + 5] = bb[1];
    }
}

fn premultiply_in_place_f32(row: &mut [u8], width: usize) {
    for i in 0..width {
        let off = i * 16;
        let a = f32_from_le(&row[off + 12..off + 16]);
        if a >= 1.0 {
            continue;
        }
        for c in 0..3 {
            let co = off + c * 4;
            let v = f32_from_le(&row[co..co + 4]) * a;
            row[co..co + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

fn unpremultiply_in_place_f32(row: &mut [u8], width: usize) {
    for i in 0..width {
        let off = i * 16;
        let a = f32_from_le(&row[off + 12..off + 16]);
        if a <= 0.0 {
            row[off..off + 12].fill(0);
            continue;
        }
        if a >= 1.0 {
            continue;
        }
        let inv = 1.0 / a;
        for c in 0..3 {
            let co = off + c * 4;
            let v = f32_from_le(&row[co..co + 4]) * inv;
            row[co..co + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Swizzle RGBA to BGRA (or vice versa) in place.
///
/// This is a fast path for the common case of converting between
/// RGBA and BGRA formats.
#[inline]
pub fn swizzle_rb_in_place(pixels: &mut [u8]) {
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }
}

/// Convert premultiplied alpha to unpremultiplied in place.
///
/// Uses Skia's rounded `SkUnPreMultiply::ApplyScale` semantics (round to
/// nearest), matching the rounding already used by `premultiply_in_place`.
pub fn unpremultiply_in_place(pixels: &mut [u8]) {
    for chunk in pixels.chunks_exact_mut(4) {
        let a = chunk[3];
        if a == 0 {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
        } else if a < 255 {
            let scale = 255.0 / f32::from(a);
            chunk[0] = f32_to_u8_sat(f32::from(chunk[0]) * scale);
            chunk[1] = f32_to_u8_sat(f32::from(chunk[1]) * scale);
            chunk[2] = f32_to_u8_sat(f32::from(chunk[2]) * scale);
        }
    }
}

/// Convert unpremultiplied alpha to premultiplied in place.
///
/// Uses Skia's rounded `SkMulDiv255Round` semantics (round to nearest), so
/// e.g. `r=3, a=128` yields 2, not 1.
pub fn premultiply_in_place(pixels: &mut [u8]) {
    for chunk in pixels.chunks_exact_mut(4) {
        let a = u32::from(chunk[3]);
        if a == 0 {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
        } else if a < 255 {
            chunk[0] = mul_div_255_round(u32::from(chunk[0]), a);
            chunk[1] = mul_div_255_round(u32::from(chunk[1]), a);
            chunk[2] = mul_div_255_round(u32::from(chunk[2]), a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_info() {
        let info = ImageInfo::new_n32_premul(100, 200).unwrap();
        assert_eq!(info.width(), 100);
        assert_eq!(info.height(), 200);
        assert_eq!(info.bytes_per_pixel(), 4);
        assert_eq!(info.min_row_bytes(), 400);
    }

    #[test]
    fn test_bitmap_allocate() {
        let info = ImageInfo::new_rgba8888(10, 10, AlphaType::Premul).unwrap();
        let bitmap = Bitmap::allocate(info).unwrap();
        assert_eq!(bitmap.width(), 10);
        assert_eq!(bitmap.height(), 10);
        assert_eq!(bitmap.pixels().len(), 400);
    }

    #[test]
    fn test_pixmap() {
        let data = vec![0u8; 400];
        let info = ImageInfo::new_rgba8888(10, 10, AlphaType::Premul).unwrap();
        let pixmap = Pixmap::new(info, &data, 40).unwrap();
        assert_eq!(pixmap.width(), 10);
        assert_eq!(pixmap.height(), 10);
    }

    #[test]
    fn test_rgba_bgra_conversion() {
        let src_info = ImageInfo::new_rgba8888(2, 1, AlphaType::Premul).unwrap();
        let dst_info = ImageInfo::new_bgra8888(2, 1, AlphaType::Premul).unwrap();

        let src = [255, 128, 64, 255, 100, 150, 200, 128];
        let mut dst = [0u8; 8];

        convert_pixels(&src, &src_info, 8, &mut dst, &dst_info, 8).unwrap();

        assert_eq!(dst[0], 64); // B from R position
        assert_eq!(dst[1], 128); // G
        assert_eq!(dst[2], 255); // R from B position
        assert_eq!(dst[3], 255); // A
    }

    #[test]
    fn test_swizzle_in_place() {
        let mut pixels = [255, 128, 64, 255, 100, 150, 200, 128];
        swizzle_rb_in_place(&mut pixels);

        assert_eq!(pixels[0], 64); // R <-> B swapped
        assert_eq!(pixels[2], 255); // R <-> B swapped
    }

    #[test]
    fn test_premultiply_round_trip() {
        let mut pixels = [200, 100, 50, 128];
        premultiply_in_place(&mut pixels);

        // After premultiply: r = 200 * 128/255 ≈ 100
        assert!(pixels[0] > 90 && pixels[0] < 110);

        unpremultiply_in_place(&mut pixels);

        // Should be close to original (with some precision loss)
        assert!(pixels[0] > 190 && pixels[0] < 210);
    }

    #[test]
    fn test_convert_pixels_unpremul_to_premul() {
        // Source: 1 unpremul pixel: red=255, green=128, blue=64, alpha=128
        let src = vec![255u8, 128, 64, 128];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();

        // Destination: 1 premul pixel
        let mut dst = vec![0u8; 4];
        let dst_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();

        let result = convert_pixels(&src, &src_info, 4, &mut dst, &dst_info, 4);
        assert!(result.is_ok());

        // Premul: r' = r * a / 255 = 255 * 128 / 255 = 128
        //         g' = 128 * 128 / 255 = 64
        //         b' = 64 * 128 / 255 = 32
        //         a' = 128 (unchanged)
        assert_eq!(dst[0], 128, "red should be premultiplied");
        assert_eq!(dst[1], 64, "green should be premultiplied");
        assert_eq!(dst[2], 32, "blue should be premultiplied");
        assert_eq!(dst[3], 128, "alpha unchanged");
    }

    #[test]
    fn test_convert_pixels_premul_to_unpremul() {
        // Source: 1 premul pixel: r=128, g=64, b=32, a=128
        // (was r=255, g=128, b=64 unpremul)
        let src = vec![128u8, 64, 32, 128];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();

        let mut dst = vec![0u8; 4];
        let dst_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();

        let result = convert_pixels(&src, &src_info, 4, &mut dst, &dst_info, 4);
        assert!(result.is_ok());

        // Unpremul: r' = r * 255 / a = 128 * 255 / 128 = 255
        //           g' = 64 * 255 / 128 = 127 (rounding)
        //           b' = 32 * 255 / 128 = 63
        //           a' = 128 (unchanged)
        assert_eq!(dst[0], 255);
        assert!(
            (i32::from(dst[1]) - 127).abs() <= 1,
            "green ~127, got {}",
            dst[1]
        );
        assert!(
            (i32::from(dst[2]) - 63).abs() <= 1,
            "blue ~63, got {}",
            dst[2]
        );
        assert_eq!(dst[3], 128);
    }

    #[test]
    fn test_convert_pixels_same_alpha_type_no_op() {
        // Same alpha type should not modify pixel values (regression check)
        let src = vec![100u8, 150, 200, 128];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();

        let mut dst = vec![0u8; 4];
        let dst_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul).unwrap();

        let result = convert_pixels(&src, &src_info, 4, &mut dst, &dst_info, 4);
        assert!(result.is_ok());

        assert_eq!(dst, src.as_slice());
    }

    #[test]
    fn test_f16_codec_roundtrip_selected_values() {
        // Spot-check the half-float encode/decode pair for a handful of
        // representative values: zero, small normal, 1.0, values inside
        // [0, 1], and sign.
        for &v in &[0.0f32, 0.25, 0.5, 1.0, -1.0, 0.003_921_569] {
            let bytes = f16_to_bytes(v);
            let back = f16_from_bytes(bytes);
            assert!(
                (back - v).abs() <= v.abs().mul_add(1e-3, 1e-3),
                "f16 roundtrip of {v} got {back}"
            );
        }
    }

    #[test]
    fn test_convert_pixels_rgba8_to_f32() {
        let src = vec![255u8, 128, 0, 255];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();
        let dst_info = ImageInfo::new(1, 1, ColorType::RgbaF32, AlphaType::Unpremul).unwrap();
        let mut dst = vec![0u8; 16];
        convert_pixels(&src, &src_info, 4, &mut dst, &dst_info, 16).unwrap();
        let r = f32_from_le(&dst[0..4]);
        let g = f32_from_le(&dst[4..8]);
        let b = f32_from_le(&dst[8..12]);
        let a = f32_from_le(&dst[12..16]);
        assert!((r - 1.0).abs() < 1e-3);
        assert!((g - 128.0 / 255.0).abs() < 1e-3);
        assert!((b - 0.0).abs() < 1e-3);
        assert!((a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_convert_pixels_f32_to_rgba8_clamps() {
        // F32 values outside [0,1] must clamp.
        let mut src = vec![0u8; 16];
        src[0..4].copy_from_slice(&2.0f32.to_le_bytes());
        src[4..8].copy_from_slice(&0.5f32.to_le_bytes());
        src[8..12].copy_from_slice(&(-0.1f32).to_le_bytes());
        src[12..16].copy_from_slice(&1.0f32.to_le_bytes());
        let src_info = ImageInfo::new(1, 1, ColorType::RgbaF32, AlphaType::Unpremul).unwrap();
        let dst_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();
        let mut dst = vec![0u8; 4];
        convert_pixels(&src, &src_info, 16, &mut dst, &dst_info, 4).unwrap();
        assert_eq!(dst, &[255, 128, 0, 255]);
    }

    #[test]
    fn test_convert_pixels_rgba8_to_f16() {
        let src = vec![255u8, 128, 0, 255];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();
        let dst_info = ImageInfo::new(1, 1, ColorType::RgbaF16, AlphaType::Unpremul).unwrap();
        let mut dst = vec![0u8; 8];
        convert_pixels(&src, &src_info, 4, &mut dst, &dst_info, 8).unwrap();
        assert!((f16_from_bytes([dst[0], dst[1]]) - 1.0).abs() < 1e-2);
        assert!((f16_from_bytes([dst[2], dst[3]]) - 128.0 / 255.0).abs() < 1e-2);
        assert!((f16_from_bytes([dst[4], dst[5]]) - 0.0).abs() < 1e-2);
        assert!((f16_from_bytes([dst[6], dst[7]]) - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_convert_pixels_f16_to_rgba8_roundtrip() {
        // RGBA8 -> F16 -> RGBA8 should come back close to the input.
        let src = vec![200u8, 100, 50, 255];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();
        let mid_info = ImageInfo::new(1, 1, ColorType::RgbaF16, AlphaType::Unpremul).unwrap();
        let dst_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();
        let mut mid = vec![0u8; 8];
        let mut dst = vec![0u8; 4];
        convert_pixels(&src, &src_info, 4, &mut mid, &mid_info, 8).unwrap();
        convert_pixels(&mid, &mid_info, 8, &mut dst, &dst_info, 4).unwrap();
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!(
                (i32::from(*a) - i32::from(*b)).abs() <= 1,
                "mismatch {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_convert_pixels_f32_premultiply_unpremul() {
        // Convert RGBA8 unpremul -> F32 premul, verify premultiply happened.
        let src = vec![255u8, 128, 0, 128];
        let src_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul).unwrap();
        let dst_info = ImageInfo::new(1, 1, ColorType::RgbaF32, AlphaType::Premul).unwrap();
        let mut dst = vec![0u8; 16];
        convert_pixels(&src, &src_info, 4, &mut dst, &dst_info, 16).unwrap();
        let r = f32_from_le(&dst[0..4]);
        let a = f32_from_le(&dst[12..16]);
        // r should be 1.0 * (128/255) ≈ 0.502
        assert!((r - 128.0 / 255.0).abs() < 1e-2, "premul red = {r}");
        assert!((a - 128.0 / 255.0).abs() < 1e-3, "alpha = {a}");
    }

    // --- Conformance regression tests (Task 1) ---

    #[test]
    fn test_image_info_accepts_zero_dimensions() {
        // SkImageInfo::Make(0, 0, ...) is a legal empty info.
        let info = ImageInfo::new(0, 0, ColorType::Rgba8888, AlphaType::Premul);
        assert!(info.is_ok(), "zero dimensions must be legal");
        let info = info.unwrap();
        assert_eq!(info.width(), 0);
        assert_eq!(info.height(), 0);
        // Negatives are still rejected.
        assert!(ImageInfo::new(-1, 4, ColorType::Rgba8888, AlphaType::Premul).is_err());
        assert!(ImageInfo::new(4, -1, ColorType::Rgba8888, AlphaType::Premul).is_err());
    }

    #[test]
    fn test_rgb565_to_rgba8888_replicates_low_bits() {
        // Full-scale 565 white (0xFFFF) must map to 255,255,255, not 248,252,248.
        let src = 0xFFFFu16.to_le_bytes();
        let mut dst = [0u8; 4];
        convert_row(
            &src,
            ColorType::Rgb565,
            &mut dst,
            ColorType::Rgba8888,
            1,
            AlphaConversion::None,
        )
        .unwrap();
        assert_eq!(dst, [255, 255, 255, 255], "565 white must be full white");

        // Full-scale red channel (31) -> 255.
        let red565 = (0x1Fu16 << 11).to_le_bytes();
        let mut dst = [0u8; 4];
        convert_row(
            &red565,
            ColorType::Rgb565,
            &mut dst,
            ColorType::Rgba8888,
            1,
            AlphaConversion::None,
        )
        .unwrap();
        assert_eq!(dst, [255, 0, 0, 255]);
    }

    #[test]
    fn test_rgba8888_to_gray8_uses_bt709() {
        // Pure green -> 0.7152 * 255 ≈ 182.
        let src = [0u8, 255, 0, 255];
        let mut dst = [0u8; 1];
        convert_row(
            &src,
            ColorType::Rgba8888,
            &mut dst,
            ColorType::Gray8,
            1,
            AlphaConversion::None,
        )
        .unwrap();
        assert_eq!(dst[0], crate::cast::f32_to_u8_sat(0.7152 * 255.0f32));
    }

    #[test]
    fn test_premultiply_in_place_rounds() {
        // r=3, a=128 must round to 2, not truncate to 1.
        let mut pixels = [3u8, 3, 3, 128];
        premultiply_in_place(&mut pixels);
        assert_eq!(pixels, [2, 2, 2, 128]);
    }

    // Helper: run a same-type premul then unpremul round-trip on an opaque
    // pixel; alpha is full so colors must be preserved exactly.
    fn assert_same_type_opaque_roundtrip(ct: ColorType, row: &[u8]) {
        let bpp = ct.bytes_per_pixel();
        let width = row.len() / bpp;
        let mut premul = row.to_vec();
        apply_alpha_conversion(&mut premul, ct, width, AlphaConversion::Premultiply);
        assert_eq!(
            premul, row,
            "{ct:?}: premultiplying an opaque pixel must not change it"
        );
    }

    #[test]
    fn test_apply_alpha_conversion_argb4444_no_corruption() {
        // Opaque 4444 pixel: R=F, G=A, B=5, A=F -> 0xFA5F.
        let px = 0xFA5Fu16.to_le_bytes();
        assert_same_type_opaque_roundtrip(ColorType::Argb4444, &px);

        // Half-alpha premultiply then unpremultiply should stay in-format and
        // keep alpha unchanged.
        let px = 0x8888u16.to_le_bytes(); // R=8 G=8 B=8 A=8
        let mut buf = px.to_vec();
        apply_alpha_conversion(
            &mut buf,
            ColorType::Argb4444,
            1,
            AlphaConversion::Premultiply,
        );
        let v = u16::from_le_bytes([buf[0], buf[1]]);
        assert_eq!(v & 0xF, 0x8, "alpha nibble must be unchanged");
        // Premul color = round(8/15 * 8/15 * 15) = round(4.27) = 4.
        assert_eq!((v >> 12) & 0xF, 4);
    }

    #[test]
    fn test_apply_alpha_conversion_1010102_no_corruption() {
        // Opaque RGBA1010102 pixel: max color, alpha=3.
        let opaque =
            (0x3FFu32 | (0x3FFu32 << 10) | (0x3FFu32 << 20) | (0x3u32 << 30)).to_le_bytes();
        assert_same_type_opaque_roundtrip(ColorType::Rgba1010102, &opaque);
        assert_same_type_opaque_roundtrip(ColorType::Bgra1010102, &opaque);
    }

    #[test]
    fn test_apply_alpha_conversion_16161616_no_corruption() {
        // Opaque R16G16B16A16 pixel.
        let mut px = Vec::new();
        for &c in &[40000u16, 20000, 10000, 65535] {
            px.extend_from_slice(&c.to_le_bytes());
        }
        assert_same_type_opaque_roundtrip(ColorType::R16G16B16A16Unorm, &px);
    }

    #[test]
    fn test_apply_alpha_conversion_alpha_only_is_noop() {
        // Alpha-only formats have no color channels; conversion is a no-op.
        let mut a8 = [10u8, 128, 200, 255];
        apply_alpha_conversion(&mut a8, ColorType::Alpha8, 4, AlphaConversion::Premultiply);
        assert_eq!(a8, [10, 128, 200, 255]);

        let mut a16 = 30000u16.to_le_bytes();
        apply_alpha_conversion(
            &mut a16,
            ColorType::A16Unorm,
            1,
            AlphaConversion::Unpremultiply,
        );
        assert_eq!(u16::from_le_bytes(a16), 30000);
    }
}
