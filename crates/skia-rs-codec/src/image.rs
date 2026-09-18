//! Image type for immutable pixel data.
//!
//! Images represent immutable pixel data that can be drawn to a canvas.

use skia_rs_core::cast::{floor_to_i32, scalar_from_i32};
use skia_rs_core::{AlphaType, ColorSpace, ColorType, Rect, Scalar};
use std::sync::Arc;

/// Convert a pixel-loop index to a [`Scalar`] for resampling math.
///
/// Routes through [`scalar_from_i32`] (Skia's documented lossy `i32->f32`
/// widening) rather than a bare `as` cast; saturates to `i32::MAX` for the
/// (practically unreachable, since a buffer that large could not have been
/// allocated) case of an index beyond `i32::MAX`.
#[inline]
fn usize_to_scalar(v: usize) -> Scalar {
    scalar_from_i32(i32::try_from(v).unwrap_or(i32::MAX))
}

/// Round a non-negative [`Scalar`] pixel coordinate down to a `usize` index,
/// saturating (never panicking) on out-of-range input.
#[inline]
fn scalar_to_usize_floor(v: Scalar) -> usize {
    usize::try_from(floor_to_i32(v)).unwrap_or(0)
}

/// Sampling mode for image resampling operations.
///
/// Corresponds to a subset of Skia's `SkSamplingOptions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplingOptions {
    /// Nearest-neighbor sampling: copy the closest source pixel. Fast,
    /// blocky, correct for pixel art and thumbnails.
    #[default]
    Nearest,
    /// Bilinear filtering: blend the four surrounding source pixels. Good
    /// for photographs and gradient-heavy content.
    Linear,
    /// Mitchell-Netravali cubic filter (B=C=1/3). Matches Skia's
    /// `SkMitchellSampling`. Sharper than bilinear with modest ringing.
    Mitchell,
    /// Catmull-Rom cubic filter (B=0, C=1/2). Matches Skia's
    /// `SkCubicSampling` with the Catmull-Rom preset. Sharper than Mitchell
    /// with more ringing.
    CatmullRom,
    /// Lanczos-3 windowed sinc filter (a=3). Highest quality for photo
    /// downscaling; 6-tap per axis.
    Lanczos3,
}

/// Simplified image info for codec use (avoids Result-based construction).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageInfo {
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
    /// Color type.
    pub color_type: ColorType,
    /// Alpha type.
    pub alpha_type: AlphaType,
    /// Color space.
    pub color_space: Option<ColorSpace>,
}

impl ImageInfo {
    /// Create a new image info.
    #[must_use]
    pub const fn new(
        width: i32,
        height: i32,
        color_type: ColorType,
        alpha_type: AlphaType,
    ) -> Self {
        Self {
            width,
            height,
            color_type,
            alpha_type,
            color_space: None,
        }
    }

    /// Get the width.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.width
    }

    /// Get the height.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.height
    }

    /// Get the color type.
    #[inline]
    #[must_use]
    pub const fn color_type(&self) -> ColorType {
        self.color_type
    }

    /// Get the alpha type.
    #[inline]
    #[must_use]
    pub const fn alpha_type(&self) -> AlphaType {
        self.alpha_type
    }

    /// Get the color space.
    #[inline]
    #[must_use]
    pub const fn color_space(&self) -> Option<&ColorSpace> {
        self.color_space.as_ref()
    }

    /// Returns true if dimensions are zero or negative.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// Returns true if alpha is opaque.
    #[inline]
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        self.alpha_type == AlphaType::Opaque
    }

    /// Bytes per pixel.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> usize {
        self.color_type.bytes_per_pixel()
    }

    /// Minimum row bytes.
    #[inline]
    #[must_use]
    pub fn min_row_bytes(&self) -> usize {
        let Ok(width) = usize::try_from(self.width) else {
            return 0;
        };
        width * self.bytes_per_pixel()
    }

    /// Compute byte size for given row bytes.
    #[inline]
    #[must_use]
    pub fn compute_byte_size(&self, row_bytes: usize) -> usize {
        let Ok(height) = usize::try_from(self.height) else {
            return 0;
        };
        if height == 0 {
            return 0;
        }
        let last_row = self.min_row_bytes();
        let other_rows = row_bytes * (height - 1);
        other_rows + last_row
    }
}

/// An immutable image.
///
/// Images are the immutable counterpart to Bitmap. Once created, an Image's
/// pixels cannot be changed. Images can be created from:
/// - Pixel data (raster images)
/// - GPU textures
/// - Other images (subsets, scaling)
/// - Encoded data (PNG, JPEG, etc.)
///
/// Corresponds to Skia's `SkImage`.
#[derive(Clone)]
pub struct Image {
    inner: Arc<ImageData>,
}

struct ImageData {
    info: ImageInfo,
    pixels: Vec<u8>,
    row_bytes: usize,
    unique_id: u64,
}

/// Allocate the next process-wide unique image ID.
///
/// Skia's `SkImage::uniqueID()` is a stable value drawn from a monotonic
/// counter, never a pointer address: pointer reuse after free would collide
/// two logically distinct images, breaking caches keyed on the ID.
fn next_image_id() -> u64 {
    static ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("color_type", &self.color_type())
            .field("alpha_type", &self.alpha_type())
            .finish()
    }
}

impl Image {
    /// Create an image from raw pixel data.
    ///
    /// The pixels are copied into the image.
    #[must_use]
    pub fn from_raster_data(info: &ImageInfo, pixels: &[u8], row_bytes: usize) -> Option<Self> {
        if info.is_empty() {
            return None;
        }

        let expected_size = info.compute_byte_size(row_bytes);
        if pixels.len() < expected_size {
            return None;
        }

        Some(Self {
            inner: Arc::new(ImageData {
                info: info.clone(),
                pixels: pixels[..expected_size].to_vec(),
                row_bytes,
                unique_id: next_image_id(),
            }),
        })
    }

    /// Create an image from owned pixel data.
    #[must_use]
    pub fn from_raster_data_owned(
        info: ImageInfo,
        pixels: Vec<u8>,
        row_bytes: usize,
    ) -> Option<Self> {
        if info.is_empty() {
            return None;
        }

        let expected_size = info.compute_byte_size(row_bytes);
        if pixels.len() < expected_size {
            return None;
        }

        Some(Self {
            inner: Arc::new(ImageData {
                info,
                pixels,
                row_bytes,
                unique_id: next_image_id(),
            }),
        })
    }

    /// Create a new RGBA image filled with a color.
    #[must_use]
    pub fn from_color(width: i32, height: i32, color: u32) -> Option<Self> {
        if width <= 0 || height <= 0 {
            return None;
        }

        let info = ImageInfo::new(width, height, ColorType::Rgba8888, AlphaType::Premul);
        let width_usize = usize::try_from(width).ok()?;
        let height_usize = usize::try_from(height).ok()?;
        let row_bytes = width_usize * 4;
        let mut pixels = vec![0u8; height_usize * row_bytes];

        // Split color into RGBA bytes
        let r = u8::try_from((color >> 16) & 0xFF).unwrap_or(0);
        let g = u8::try_from((color >> 8) & 0xFF).unwrap_or(0);
        let b = u8::try_from(color & 0xFF).unwrap_or(0);
        let a = u8::try_from((color >> 24) & 0xFF).unwrap_or(0);

        for y in 0..height_usize {
            for x in 0..width_usize {
                let offset = y * row_bytes + x * 4;
                pixels[offset] = r;
                pixels[offset + 1] = g;
                pixels[offset + 2] = b;
                pixels[offset + 3] = a;
            }
        }

        Self::from_raster_data_owned(info, pixels, row_bytes)
    }

    /// Get the image width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> i32 {
        self.inner.info.width()
    }

    /// Get the image height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> i32 {
        self.inner.info.height()
    }

    /// Get the image dimensions as (width, height).
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> (i32, i32) {
        (self.width(), self.height())
    }

    /// Get the image bounds as a rectangle.
    #[inline]
    #[must_use]
    pub fn bounds(&self) -> Rect {
        Rect::from_xywh(
            0.0,
            0.0,
            skia_rs_core::cast::scalar_from_i32(self.width()),
            skia_rs_core::cast::scalar_from_i32(self.height()),
        )
    }

    /// Get the image info.
    #[inline]
    #[must_use]
    pub fn info(&self) -> &ImageInfo {
        &self.inner.info
    }

    /// Get the color type.
    #[inline]
    #[must_use]
    pub fn color_type(&self) -> ColorType {
        self.inner.info.color_type()
    }

    /// Get the alpha type.
    #[inline]
    #[must_use]
    pub fn alpha_type(&self) -> AlphaType {
        self.inner.info.alpha_type()
    }

    /// Get the color space.
    #[inline]
    #[must_use]
    pub fn color_space(&self) -> Option<&ColorSpace> {
        self.inner.info.color_space()
    }

    /// Returns true if the image is opaque.
    #[inline]
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        self.inner.info.is_opaque()
    }

    /// Get the row bytes (stride).
    #[inline]
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.inner.row_bytes
    }

    /// Get the unique ID for this image.
    ///
    /// The ID comes from a process-wide monotonic counter assigned at
    /// construction (mirroring `SkImage::uniqueID()`), so distinct images
    /// always compare unequal even if a later allocation reuses a freed
    /// image's memory. Cloning an `Image` shares the same ID.
    #[inline]
    #[must_use]
    pub fn unique_id(&self) -> usize {
        usize::try_from(self.inner.unique_id).unwrap_or(usize::MAX)
    }

    /// Read pixels from the image into a buffer.
    pub fn read_pixels(
        &self,
        dst_info: &ImageInfo,
        dst_pixels: &mut [u8],
        dst_row_bytes: usize,
        src_x: i32,
        src_y: i32,
    ) -> bool {
        // Validate bounds
        if src_x < 0 || src_y < 0 || dst_info.is_empty() {
            return false;
        }
        if src_x + dst_info.width() > self.width() {
            return false;
        }
        if src_y + dst_info.height() > self.height() {
            return false;
        }
        let (Ok(src_x), Ok(src_y)) = (usize::try_from(src_x), usize::try_from(src_y)) else {
            return false;
        };
        let (Ok(dst_width), Ok(dst_height)) = (
            usize::try_from(dst_info.width()),
            usize::try_from(dst_info.height()),
        ) else {
            return false;
        };

        let dst_size = dst_info.compute_byte_size(dst_row_bytes);
        if dst_pixels.len() < dst_size {
            return false;
        }

        // Simple copy for matching formats
        if dst_info.color_type() == self.color_type() && dst_info.alpha_type() == self.alpha_type()
        {
            let bytes_per_pixel = self.color_type().bytes_per_pixel();
            let src_row_bytes = self.inner.row_bytes;

            for y in 0..dst_height {
                let src_offset = (src_y + y) * src_row_bytes + src_x * bytes_per_pixel;
                let dst_offset = y * dst_row_bytes;
                let copy_len = dst_width * bytes_per_pixel;

                dst_pixels[dst_offset..dst_offset + copy_len]
                    .copy_from_slice(&self.inner.pixels[src_offset..src_offset + copy_len]);
            }

            return true;
        }

        // Format conversion: delegate to skia_rs_core::convert_pixels, which handles
        // color-type and alpha-type conversion in one pass. To support sub-region
        // reads, we point the source slice at the subset's first byte and keep the
        // original row stride — convert_pixels walks rows at `src_row_bytes` each.
        let src_bpp = self.color_type().bytes_per_pixel();
        let src_row_bytes = self.inner.row_bytes;
        let src_start = src_y * src_row_bytes + src_x * src_bpp;

        // Build source info describing the subset (same color/alpha type as self).
        let Ok(src_core_info) = skia_rs_core::ImageInfo::new(
            dst_info.width(),
            dst_info.height(),
            self.color_type(),
            self.alpha_type(),
        ) else {
            return false;
        };
        let Ok(dst_core_info) = skia_rs_core::ImageInfo::new(
            dst_info.width(),
            dst_info.height(),
            dst_info.color_type(),
            dst_info.alpha_type(),
        ) else {
            return false;
        };

        // Bounds-check the source slice: we need src_start + (height-1)*row_bytes + min_row_bytes.
        let src_required = src_core_info.compute_byte_size(src_row_bytes);
        if src_start + src_required > self.inner.pixels.len() {
            return false;
        }

        skia_rs_core::convert_pixels(
            &self.inner.pixels[src_start..],
            &src_core_info,
            src_row_bytes,
            dst_pixels,
            &dst_core_info,
            dst_row_bytes,
        )
        .is_ok()
    }

    /// Read a single pixel at (x, y).
    #[must_use]
    pub fn read_pixel(&self, x: i32, y: i32) -> Option<skia_rs_core::Color4f> {
        if x < 0 || x >= self.width() || y < 0 || y >= self.height() {
            return None;
        }
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return None;
        };

        let bytes_per_pixel = self.color_type().bytes_per_pixel();
        let offset = y * self.inner.row_bytes + x * bytes_per_pixel;

        match self.color_type() {
            ColorType::Rgba8888 => {
                let red = f32::from(self.inner.pixels[offset]) / 255.0;
                let green = f32::from(self.inner.pixels[offset + 1]) / 255.0;
                let blue = f32::from(self.inner.pixels[offset + 2]) / 255.0;
                let alpha = f32::from(self.inner.pixels[offset + 3]) / 255.0;
                Some(skia_rs_core::Color4f::new(red, green, blue, alpha))
            }
            ColorType::Bgra8888 => {
                let blue = f32::from(self.inner.pixels[offset]) / 255.0;
                let green = f32::from(self.inner.pixels[offset + 1]) / 255.0;
                let red = f32::from(self.inner.pixels[offset + 2]) / 255.0;
                let alpha = f32::from(self.inner.pixels[offset + 3]) / 255.0;
                Some(skia_rs_core::Color4f::new(red, green, blue, alpha))
            }
            ColorType::Alpha8 => {
                let alpha = f32::from(self.inner.pixels[offset]) / 255.0;
                Some(skia_rs_core::Color4f::new(0.0, 0.0, 0.0, alpha))
            }
            ColorType::Gray8 => {
                let gray = f32::from(self.inner.pixels[offset]) / 255.0;
                Some(skia_rs_core::Color4f::new(gray, gray, gray, 1.0))
            }
            _ => None,
        }
    }

    /// Get direct access to the pixel data (if available).
    #[must_use]
    pub fn peek_pixels(&self) -> Option<&[u8]> {
        Some(&self.inner.pixels)
    }

    /// Create a subset of this image.
    #[must_use]
    pub fn make_subset(&self, subset: &Rect) -> Option<Self> {
        use skia_rs_core::cast::saturate_to_i32;
        let x = saturate_to_i32(subset.left);
        let y = saturate_to_i32(subset.top);
        let w = saturate_to_i32(subset.width());
        let h = saturate_to_i32(subset.height());

        if x < 0 || y < 0 || w <= 0 || h <= 0 {
            return None;
        }
        if x + w > self.width() || y + h > self.height() {
            return None;
        }
        let (Ok(x), Ok(y), Ok(w_usize), Ok(h_usize)) = (
            usize::try_from(x),
            usize::try_from(y),
            usize::try_from(w),
            usize::try_from(h),
        ) else {
            return None;
        };

        let new_info = ImageInfo::new(w, h, self.color_type(), self.alpha_type());
        let bytes_per_pixel = self.color_type().bytes_per_pixel();
        let new_row_bytes = w_usize * bytes_per_pixel;
        let mut new_pixels = vec![0u8; h_usize * new_row_bytes];

        for row in 0..h_usize {
            let src_offset = (y + row) * self.inner.row_bytes + x * bytes_per_pixel;
            let dst_offset = row * new_row_bytes;

            new_pixels[dst_offset..dst_offset + new_row_bytes]
                .copy_from_slice(&self.inner.pixels[src_offset..src_offset + new_row_bytes]);
        }

        Self::from_raster_data_owned(new_info, new_pixels, new_row_bytes)
    }

    /// Create a scaled version of this image using nearest-neighbor
    /// sampling.
    ///
    /// For quality-sensitive scaling (photographs, downscaling), prefer
    /// [`Image::make_scaled_with`] with [`SamplingOptions::Linear`].
    #[must_use]
    pub fn make_scaled(&self, width: i32, height: i32) -> Option<Self> {
        self.make_scaled_with(width, height, SamplingOptions::Nearest)
    }

    /// Create a scaled version of this image with explicit sampling
    /// options.
    ///
    /// - [`SamplingOptions::Nearest`]: fastest, appropriate for pixel art
    ///   and thumbnails, visibly blocky when upscaled.
    /// - [`SamplingOptions::Linear`]: bilinear filtering, adequate for
    ///   photographs.
    /// - [`SamplingOptions::Mitchell`] / [`SamplingOptions::CatmullRom`]:
    ///   cubic (4-tap per axis) filters; sharper than linear with a small
    ///   cost bump.
    /// - [`SamplingOptions::Lanczos3`]: 6-tap per axis windowed sinc, best
    ///   quality for photographic downscaling.
    #[must_use]
    pub fn make_scaled_with(
        &self,
        width: i32,
        height: i32,
        sampling: SamplingOptions,
    ) -> Option<Self> {
        if width <= 0 || height <= 0 {
            return None;
        }
        let (Ok(width_usize), Ok(height_usize)) = (usize::try_from(width), usize::try_from(height))
        else {
            return None;
        };
        // Any live `Image` has positive dimensions (enforced at construction
        // by `ImageInfo::is_empty()`), so these never hit the fallback.
        let src_w = usize::try_from(self.width()).unwrap_or(0);
        let src_h = usize::try_from(self.height()).unwrap_or(0);
        if src_w == 0 || src_h == 0 {
            return None;
        }

        let new_info = ImageInfo::new(width, height, self.color_type(), self.alpha_type());
        let bytes_per_pixel = self.color_type().bytes_per_pixel();
        let new_row_bytes = width_usize * bytes_per_pixel;
        let mut new_pixels = vec![0u8; height_usize * new_row_bytes];

        let src_row_bytes = self.inner.row_bytes;

        // Filtered resampling (Linear / cubic / Lanczos) blends neighboring
        // pixels, which is only correct in *premultiplied* space — otherwise
        // color bleeds out of fully transparent texels. If the source is
        // unpremultiplied RGBA/BGRA, premultiply a working copy, filter it,
        // then unpremultiply the result. Opaque and already-premultiplied
        // sources filter in place. Nearest-neighbor copies whole pixels and
        // needs no premultiplication.
        let filters = !matches!(sampling, SamplingOptions::Nearest);
        let needs_premul = filters
            && bytes_per_pixel == 4
            && self.alpha_type() == AlphaType::Unpremul
            && matches!(self.color_type(), ColorType::Rgba8888 | ColorType::Bgra8888);
        let premul_storage;
        let src_pixels: &[u8] = if needs_premul {
            let mut v = self.inner.pixels.clone();
            skia_rs_core::premultiply_in_place(&mut v);
            premul_storage = v;
            &premul_storage
        } else {
            &self.inner.pixels
        };

        match sampling {
            SamplingOptions::Nearest => {
                resample_nearest(
                    src_pixels,
                    self.width(),
                    self.height(),
                    src_row_bytes,
                    bytes_per_pixel,
                    &mut new_pixels,
                    width,
                    height,
                    new_row_bytes,
                );
            }
            SamplingOptions::Linear => {
                resample_linear(
                    src_pixels,
                    src_w,
                    src_h,
                    src_row_bytes,
                    bytes_per_pixel,
                    &mut new_pixels,
                    width_usize,
                    height_usize,
                    new_row_bytes,
                );
            }
            SamplingOptions::Mitchell | SamplingOptions::CatmullRom | SamplingOptions::Lanczos3 => {
                resample_separable(
                    src_pixels,
                    src_w,
                    src_h,
                    src_row_bytes,
                    bytes_per_pixel,
                    &mut new_pixels,
                    width_usize,
                    height_usize,
                    new_row_bytes,
                    sampling,
                );
            }
        }

        // Undo the premultiplication applied above so the result keeps the
        // source's unpremultiplied alpha type.
        if needs_premul {
            skia_rs_core::unpremultiply_in_place(&mut new_pixels);
        }

        Self::from_raster_data_owned(new_info, new_pixels, new_row_bytes)
    }

    /// Apply an image filter to this image and return a new filtered image.
    ///
    /// The filter is applied in-place on an RGBA8 copy of the image. If the
    /// source image is not already `Rgba8888/Premul`, the pixels are
    /// converted via [`skia_rs_core::convert_pixels`] before filtering and
    /// the returned image reports `Rgba8888/Premul`.
    ///
    /// Multi-input filters (such as [`skia_rs_paint::MergeImageFilter`] and
    /// [`skia_rs_paint::DisplacementMapImageFilter`]'s displacement input)
    /// cannot be expressed through the single-buffer `apply(pixels)` API and
    /// will return the unfiltered copy — the paint crate's filter trait has
    /// a documented limitation there that will be lifted when the
    /// multi-input API lands.
    ///
    /// Returns `None` if pixel extraction or conversion fails.
    pub fn make_with_filter(&self, filter: &dyn skia_rs_paint::ImageFilter) -> Option<Self> {
        let width = self.width();
        let height = self.height();
        if width <= 0 || height <= 0 {
            return None;
        }
        let (Ok(width_usize), Ok(height_usize)) = (usize::try_from(width), usize::try_from(height))
        else {
            return None;
        };

        // Build a fresh RGBA8 Premul buffer, converting if necessary.
        let rgba_row_bytes = width_usize * 4;
        let mut rgba = vec![0u8; height_usize * rgba_row_bytes];
        let dst_info = ImageInfo::new(width, height, ColorType::Rgba8888, AlphaType::Premul);
        if !self.read_pixels(&dst_info, &mut rgba, rgba_row_bytes, 0, 0) {
            return None;
        }

        // Apply the filter in place.
        filter.apply(&mut rgba, width_usize, height_usize);

        Self::from_raster_data_owned(dst_info, rgba, rgba_row_bytes)
    }
}

// =============================================================================
// Separable resamplers (Mitchell, Catmull-Rom, Lanczos3)
//
// Two-pass approach: horizontal then vertical. Each pass builds per-destination
// pixel (src-index, weight) lists once and reuses them across every row/column,
// which is dramatically cheaper than recomputing per output pixel.
// =============================================================================

#[inline]
fn cubic_bc(t: f32, b: f32, c: f32) -> f32 {
    // Mitchell-Netravali family:
    //   f(t) = 1/6 * [ (12-9B-6C)|t|^3 + (-18+12B+6C)|t|^2 + (6-2B) ] for |t|<1
    //          1/6 * [ (-B-6C)|t|^3 + (6B+30C)|t|^2 + (-12B-48C)|t| + (8B+24C) ]
    //          for 1<=|t|<2
    //          0 otherwise.
    let t = t.abs();
    if t < 1.0 {
        let t2 = t * t;
        let t3 = t2 * t;
        (6.0f32
            .mul_add(-c, 9.0f32.mul_add(-b, 12.0))
            .mul_add(t3, 6.0f32.mul_add(c, 12.0f32.mul_add(b, -18.0)) * t2)
            + 2.0f32.mul_add(-b, 6.0))
            / 6.0
    } else if t < 2.0 {
        let t2 = t * t;
        let t3 = t2 * t;
        ((-12.0f32).mul_add(b, -(48.0 * c)).mul_add(
            t,
            6.0f32
                .mul_add(-c, -b)
                .mul_add(t3, 6.0f32.mul_add(b, 30.0 * c) * t2),
        ) + 8.0f32.mul_add(b, 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

#[inline]
fn sinc_pi(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

#[inline]
fn lanczos3(t: f32) -> f32 {
    let t = t.abs();
    if t >= 3.0 {
        0.0
    } else {
        sinc_pi(t) * sinc_pi(t / 3.0)
    }
}

/// Kernel radius in source units. For upscales the kernel is evaluated at
/// unit spacing; for downscales the kernel is stretched to `scale` to avoid
/// aliasing (standard box-widening trick).
#[inline]
fn kernel_weight(sampling: SamplingOptions, t: f32) -> f32 {
    match sampling {
        SamplingOptions::Mitchell => cubic_bc(t, 1.0 / 3.0, 1.0 / 3.0),
        SamplingOptions::CatmullRom => cubic_bc(t, 0.0, 0.5),
        SamplingOptions::Lanczos3 => lanczos3(t),
        // Handled elsewhere.
        SamplingOptions::Nearest | SamplingOptions::Linear => 0.0,
    }
}

#[inline]
const fn kernel_radius(sampling: SamplingOptions) -> f32 {
    match sampling {
        SamplingOptions::Mitchell | SamplingOptions::CatmullRom => 2.0,
        SamplingOptions::Lanczos3 => 3.0,
        SamplingOptions::Nearest | SamplingOptions::Linear => 1.0,
    }
}

/// Per-output-pixel list of (src-sample-index, weight) pairs.
struct Taps {
    // Flattened: for output i, tap_count indices start at i*max_taps.
    indices: Vec<i32>,
    weights: Vec<f32>,
    tap_count: usize,
}

fn build_taps(src_len: usize, dst_len: usize, sampling: SamplingOptions) -> Taps {
    let scale = usize_to_scalar(src_len) / usize_to_scalar(dst_len);
    // When downscaling, stretch the kernel by `scale` so the anti-aliasing
    // footprint matches the reduction ratio. When upscaling, leave it alone.
    let filter_scale = if scale > 1.0 { scale } else { 1.0 };
    let support = kernel_radius(sampling) * filter_scale;
    let tap_count = usize::try_from(skia_rs_core::cast::ceil_to_i32(support)).unwrap_or(0) * 2;
    let max_idx = i32::try_from(src_len).unwrap_or(i32::MAX) - 1;

    let mut indices = vec![0i32; dst_len * tap_count];
    let mut weights = vec![0f32; dst_len * tap_count];

    for i in 0..dst_len {
        let center = (usize_to_scalar(i) + 0.5).mul_add(scale, -0.5);
        let left = floor_to_i32(center - support + 0.5);
        let mut sum = 0.0f32;
        for k in 0..tap_count {
            let sx = left + i32::try_from(k).unwrap_or(i32::MAX);
            let t = (scalar_from_i32(sx) - center) / filter_scale;
            let w = kernel_weight(sampling, t);
            let clamped = sx.clamp(0, max_idx);
            indices[i * tap_count + k] = clamped;
            weights[i * tap_count + k] = w;
            sum += w;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for w in &mut weights[i * tap_count..(i + 1) * tap_count] {
                *w *= inv;
            }
        }
    }

    Taps {
        indices,
        weights,
        tap_count,
    }
}

/// Nearest-neighbor resampling: sample pixel centers via `src =
/// floor((dst + 0.5) * scale)`. This aligns sample points with Skia's
/// nearest-neighbor mapping so a 1:1 scale reproduces the source exactly
/// and up/downscaling stays centered.
#[allow(clippy::too_many_arguments)]
fn resample_nearest(
    src_pixels: &[u8],
    src_w: i32,
    src_h: i32,
    src_row_bytes: usize,
    bytes_per_pixel: usize,
    new_pixels: &mut [u8],
    width: i32,
    height: i32,
    new_row_bytes: usize,
) {
    let x_scale = scalar_from_i32(src_w) / scalar_from_i32(width);
    let y_scale = scalar_from_i32(src_h) / scalar_from_i32(height);
    let src_w = usize::try_from(src_w).unwrap_or(0);
    let src_h = usize::try_from(src_h).unwrap_or(0);
    let width = usize::try_from(width).unwrap_or(0);
    let height = usize::try_from(height).unwrap_or(0);
    if src_w == 0 || src_h == 0 {
        return;
    }

    for dst_y in 0..height {
        let src_y = scalar_to_usize_floor((usize_to_scalar(dst_y) + 0.5) * y_scale).min(src_h - 1);
        for dst_x in 0..width {
            let src_x =
                scalar_to_usize_floor((usize_to_scalar(dst_x) + 0.5) * x_scale).min(src_w - 1);

            let src_offset = src_y * src_row_bytes + src_x * bytes_per_pixel;
            let dst_offset = dst_y * new_row_bytes + dst_x * bytes_per_pixel;

            new_pixels[dst_offset..dst_offset + bytes_per_pixel]
                .copy_from_slice(&src_pixels[src_offset..src_offset + bytes_per_pixel]);
        }
    }
}

/// Bilinear resampling: sample at pixel centers via
/// `src_x = (dst_x + 0.5) * src_w/dst_w - 0.5`. The `-0.5` aligns pixel
/// centers so a 1:1 scale reproduces the source exactly.
#[allow(clippy::too_many_arguments)]
fn resample_linear(
    src_pixels: &[u8],
    src_w: usize,
    src_h: usize,
    src_row_bytes: usize,
    bytes_per_pixel: usize,
    new_pixels: &mut [u8],
    width: usize,
    height: usize,
    new_row_bytes: usize,
) {
    let x_scale = usize_to_scalar(src_w) / usize_to_scalar(width);
    let y_scale = usize_to_scalar(src_h) / usize_to_scalar(height);
    let max_x = src_w.saturating_sub(1);
    let max_y = src_h.saturating_sub(1);

    for dst_y in 0..height {
        let fy = (usize_to_scalar(dst_y) + 0.5)
            .mul_add(y_scale, -0.5)
            .max(0.0);
        let y0 = scalar_to_usize_floor(fy).min(max_y);
        let y1 = (y0 + 1).min(max_y);
        let ty = fy - usize_to_scalar(y0);

        for dst_x in 0..width {
            let fx = (usize_to_scalar(dst_x) + 0.5)
                .mul_add(x_scale, -0.5)
                .max(0.0);
            let x0 = scalar_to_usize_floor(fx).min(max_x);
            let x1 = (x0 + 1).min(max_x);
            let tx = fx - usize_to_scalar(x0);

            let p00 = y0 * src_row_bytes + x0 * bytes_per_pixel;
            let p01 = y0 * src_row_bytes + x1 * bytes_per_pixel;
            let p10 = y1 * src_row_bytes + x0 * bytes_per_pixel;
            let p11 = y1 * src_row_bytes + x1 * bytes_per_pixel;
            let dst_offset = dst_y * new_row_bytes + dst_x * bytes_per_pixel;

            for i in 0..bytes_per_pixel {
                let c00 = f32::from(src_pixels[p00 + i]);
                let c01 = f32::from(src_pixels[p01 + i]);
                let c10 = f32::from(src_pixels[p10 + i]);
                let c11 = f32::from(src_pixels[p11 + i]);

                let top = c00.mul_add(1.0 - tx, c01 * tx);
                let bot = c10.mul_add(1.0 - tx, c11 * tx);
                let v = top * (1.0 - ty) + bot * ty;

                new_pixels[dst_offset + i] = skia_rs_core::cast::f32_to_u8_sat(v);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resample_separable(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    src_row_bytes: usize,
    bpp: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    dst_row_bytes: usize,
    sampling: SamplingOptions,
) {
    // Horizontal pass: src_w columns -> dst_w columns, same src_h rows.
    let h_taps = build_taps(src_w, dst_w, sampling);
    let intermediate_row_bytes = dst_w * bpp;
    let mut intermediate = vec![0f32; src_h * dst_w * bpp];

    for y in 0..src_h {
        let src_row = &src[y * src_row_bytes..y * src_row_bytes + src_w * bpp];
        let dst_row = &mut intermediate[y * dst_w * bpp..y * dst_w * bpp + dst_w * bpp];
        for x in 0..dst_w {
            let base = x * h_taps.tap_count;
            let mut acc = [0f32; 4];
            for k in 0..h_taps.tap_count {
                let sx = usize::try_from(h_taps.indices[base + k]).unwrap_or(0);
                let w = h_taps.weights[base + k];
                let p = &src_row[sx * bpp..sx * bpp + bpp];
                for c in 0..bpp {
                    acc[c] = f32::from(p[c]).mul_add(w, acc[c]);
                }
            }
            for c in 0..bpp {
                dst_row[x * bpp + c] = acc[c];
            }
        }
    }

    // Vertical pass: src_h rows -> dst_h rows, dst_w columns.
    let v_taps = build_taps(src_h, dst_h, sampling);

    for y in 0..dst_h {
        let base = y * v_taps.tap_count;
        let dst_row = &mut dst[y * dst_row_bytes..y * dst_row_bytes + intermediate_row_bytes];
        for x in 0..dst_w {
            let mut acc = [0f32; 4];
            for k in 0..v_taps.tap_count {
                let sy = usize::try_from(v_taps.indices[base + k]).unwrap_or(0);
                let w = v_taps.weights[base + k];
                let p = &intermediate[sy * dst_w * bpp + x * bpp..sy * dst_w * bpp + x * bpp + bpp];
                for c in 0..bpp {
                    acc[c] = p[c].mul_add(w, acc[c]);
                }
            }
            for c in 0..bpp {
                dst_row[x * bpp + c] = skia_rs_core::cast::f32_to_u8_sat(acc[c]);
            }
        }
    }
}

/// A reference to an image (shared ownership).
pub type ImageRef = Arc<Image>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_from_color() {
        let image = Image::from_color(100, 100, 0xFF_FF0000).unwrap();
        assert_eq!(image.width(), 100);
        assert_eq!(image.height(), 100);
        assert_eq!(image.color_type(), ColorType::Rgba8888);
    }

    #[test]
    fn test_image_from_raster() {
        let info = ImageInfo::new(10, 10, ColorType::Rgba8888, AlphaType::Premul);
        let pixels = vec![0u8; 10 * 10 * 4];
        let image = Image::from_raster_data(&info, &pixels, 10 * 4).unwrap();
        assert_eq!(image.dimensions(), (10, 10));
    }

    #[test]
    fn test_image_subset() {
        let image = Image::from_color(100, 100, 0xFF_FF0000).unwrap();
        let subset = image
            .make_subset(&Rect::from_xywh(25.0, 25.0, 50.0, 50.0))
            .unwrap();
        assert_eq!(subset.dimensions(), (50, 50));
    }

    #[test]
    fn test_image_scaled() {
        let image = Image::from_color(100, 100, 0xFF_FF0000).unwrap();
        let scaled = image.make_scaled(50, 50).unwrap();
        assert_eq!(scaled.dimensions(), (50, 50));
    }

    #[test]
    fn test_image_scaled_linear_blends_neighbors() {
        // 2x1 source: black / white. Scale up 4x wide with linear sampling
        // and expect intermediate gray values in the middle.
        let info = ImageInfo::new(2, 1, ColorType::Rgba8888, AlphaType::Premul);
        let src = vec![
            0, 0, 0, 255, // black
            255, 255, 255, 255, // white
        ];
        let image = Image::from_raster_data(&info, &src, 8).unwrap();

        let up = image
            .make_scaled_with(4, 1, SamplingOptions::Linear)
            .unwrap();
        assert_eq!(up.dimensions(), (4, 1));

        // Nearest sampling would produce [black, black, white, white].
        // Linear sampling should give [black, ~blend, ~blend, white].
        let p0 = up.read_pixel(0, 0).unwrap();
        let p1 = up.read_pixel(1, 0).unwrap();
        let p2 = up.read_pixel(2, 0).unwrap();
        let p3 = up.read_pixel(3, 0).unwrap();

        // End points remain (approximately) black and white.
        assert!(p0.r < 0.1);
        assert!(p3.r > 0.9);
        // Interior samples are non-extreme greys.
        assert!(p1.r > 0.0 && p1.r < 1.0);
        assert!(p2.r > 0.0 && p2.r < 1.0);
        // Monotonic: p1.r < p2.r (brightness increases left-to-right).
        assert!(p1.r < p2.r);
    }

    #[test]
    fn test_image_scaled_nearest_preserves_pixels() {
        // 2x1 source: black / white. Double the width with nearest
        // sampling — each source pixel should appear exactly twice.
        let info = ImageInfo::new(2, 1, ColorType::Rgba8888, AlphaType::Premul);
        let src = vec![0, 0, 0, 255, 255, 255, 255, 255];
        let image = Image::from_raster_data(&info, &src, 8).unwrap();

        let up = image
            .make_scaled_with(4, 1, SamplingOptions::Nearest)
            .unwrap();
        let p0 = up.read_pixel(0, 0).unwrap();
        let p1 = up.read_pixel(1, 0).unwrap();
        let p2 = up.read_pixel(2, 0).unwrap();
        let p3 = up.read_pixel(3, 0).unwrap();
        assert!(p0.r < 0.01 && p1.r < 0.01);
        assert!(p2.r > 0.99 && p3.r > 0.99);
    }

    #[test]
    fn test_image_scaled_mitchell_monotone_upscale() {
        // Ramp 4x1 -> 16x1. Mitchell should remain monotonic across the
        // ramp (no reversals) and preserve endpoints approximately.
        let info = ImageInfo::new(4, 1, ColorType::Rgba8888, AlphaType::Premul);
        let mut src = Vec::with_capacity(16);
        for &g in &[0u8, 85, 170, 255] {
            src.extend_from_slice(&[g, g, g, 255]);
        }
        let image = Image::from_raster_data(&info, &src, 16).unwrap();
        let up = image
            .make_scaled_with(16, 1, SamplingOptions::Mitchell)
            .unwrap();
        assert_eq!(up.dimensions(), (16, 1));

        let mut last = -1.0f32;
        let mut strictly_increasing = 0;
        for x in 0..16 {
            let p = up.read_pixel(x, 0).unwrap();
            // Allow small ringing tolerance below 0 / above 1 still monotone.
            if p.r > last {
                strictly_increasing += 1;
            }
            last = p.r;
        }
        // Mitchell rings mildly; require at least mostly-increasing behavior.
        assert!(
            strictly_increasing >= 13,
            "got {strictly_increasing} monotone steps"
        );
    }

    #[test]
    fn test_image_scaled_catmullrom_preserves_endpoints() {
        // A uniform image downscaled with Catmull-Rom should stay uniform.
        let info = ImageInfo::new(8, 8, ColorType::Rgba8888, AlphaType::Premul);
        let src = vec![200u8; 8 * 8 * 4];
        let image = Image::from_raster_data(&info, &src, 32).unwrap();
        let down = image
            .make_scaled_with(4, 4, SamplingOptions::CatmullRom)
            .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let p = down.read_pixel(x, y).unwrap();
                // Uniform source => uniform output (within rounding).
                assert!((p.r - 200.0 / 255.0).abs() < 0.02, "{},{} = {}", x, y, p.r);
            }
        }
    }

    #[test]
    fn test_image_scaled_lanczos3_downscale_reduces_aliasing() {
        // High-frequency source (1-pixel stripes). Lanczos should produce
        // a much more uniform grey than nearest-neighbor when downscaled
        // to a single row where stripes vanish.
        let info = ImageInfo::new(16, 1, ColorType::Rgba8888, AlphaType::Premul);
        let mut src = Vec::with_capacity(64);
        for x in 0..16 {
            let g = if x & 1 == 0 { 0u8 } else { 255 };
            src.extend_from_slice(&[g, g, g, 255]);
        }
        let image = Image::from_raster_data(&info, &src, 64).unwrap();
        let down = image
            .make_scaled_with(4, 1, SamplingOptions::Lanczos3)
            .unwrap();
        // Each 4-pixel group contains 2 black + 2 white so the average is 127.5.
        // Lanczos should land near grey (with some kernel asymmetry tolerance).
        for x in 0..4 {
            let p = down.read_pixel(x, 0).unwrap();
            let grey = p.r * 255.0;
            assert!(
                (grey - 127.5).abs() < 35.0,
                "Lanczos downscale x={x} got {grey}",
            );
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact integer-valued dimensions round-trip exactly through f32; comparison is intentionally exact"
    )]
    fn test_image_bounds() {
        let image = Image::from_color(100, 200, 0xFF_000000).unwrap();
        let bounds = image.bounds();
        assert_eq!(bounds.width(), 100.0);
        assert_eq!(bounds.height(), 200.0);
    }

    #[test]
    fn test_read_pixels_rgba_to_bgra_converts() {
        // Source image: 2x2 RGBA, each pixel a distinct R,G,B,A tuple.
        let info = ImageInfo::new(2, 2, ColorType::Rgba8888, AlphaType::Premul);
        let src = vec![
            10, 20, 30, 255, 40, 50, 60, 255, // row 0
            70, 80, 90, 255, 100, 110, 120, 255, // row 1
        ];
        let image = Image::from_raster_data(&info, &src, 8).unwrap();

        let dst_info = ImageInfo::new(2, 2, ColorType::Bgra8888, AlphaType::Premul);
        let mut dst = vec![0u8; 16];
        assert!(image.read_pixels(&dst_info, &mut dst, 8, 0, 0));

        // Pixel 0 should be (B=30, G=20, R=10, A=255).
        assert_eq!(dst[0..4], [30, 20, 10, 255]);
        assert_eq!(dst[4..8], [60, 50, 40, 255]);
        assert_eq!(dst[8..12], [90, 80, 70, 255]);
        assert_eq!(dst[12..16], [120, 110, 100, 255]);
    }

    #[test]
    fn test_read_pixels_subset_with_conversion() {
        // 4x4 RGBA with unique values per pixel.
        let info = ImageInfo::new(4, 4, ColorType::Rgba8888, AlphaType::Premul);
        let mut src = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            for x in 0..4 {
                let off = (y * 4 + x) * 4;
                src[off] = u8::try_from(x * 10 + y).unwrap();
                src[off + 1] = u8::try_from(x * 10 + y + 1).unwrap();
                src[off + 2] = u8::try_from(x * 10 + y + 2).unwrap();
                src[off + 3] = 255;
            }
        }
        let image = Image::from_raster_data(&info, &src, 16).unwrap();

        // Read a 2x2 subset at offset (1, 1) as BGRA.
        let dst_info = ImageInfo::new(2, 2, ColorType::Bgra8888, AlphaType::Premul);
        let mut dst = vec![0u8; 16];
        assert!(image.read_pixels(&dst_info, &mut dst, 8, 1, 1));

        // Expected: subset(1,1) = src pixel (1,1) = (r=11, g=12, b=13, a=255)
        // BGRA form = (13, 12, 11, 255).
        assert_eq!(dst[0..4], [13, 12, 11, 255]);
        // subset(1,2) = src (2,1) = (r=21, g=22, b=23) => BGRA (23, 22, 21, 255).
        assert_eq!(dst[4..8], [23, 22, 21, 255]);
        // subset(2,1) = src (1,2) = (r=12, g=13, b=14) => BGRA (14, 13, 12, 255).
        assert_eq!(dst[8..12], [14, 13, 12, 255]);
        assert_eq!(dst[12..16], [24, 23, 22, 255]);
    }

    #[test]
    fn test_read_pixels_alpha_type_conversion() {
        // Source image: RGBA Unpremul (50, 100, 150, 128).
        // After premultiplication: c * a/255 = 25, 50, 75.
        let info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Unpremul);
        let src = vec![50, 100, 150, 128];
        let image = Image::from_raster_data(&info, &src, 4).unwrap();

        let dst_info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul);
        let mut dst = vec![0u8; 4];
        assert!(image.read_pixels(&dst_info, &mut dst, 4, 0, 0));

        assert_eq!(dst[0], 25);
        assert_eq!(dst[1], 50);
        assert_eq!(dst[2], 75);
        assert_eq!(dst[3], 128);
    }

    #[test]
    fn test_make_with_filter_offset_shifts_pixels() {
        // A red pixel in a 3x3 black field.
        let info = ImageInfo::new(3, 3, ColorType::Rgba8888, AlphaType::Premul);
        let mut pixels = vec![0u8; 3 * 3 * 4];
        // Set center pixel (1,1) to red.
        let center_off = (3 + 1) * 4;
        pixels[center_off] = 255;
        pixels[center_off + 3] = 255;
        let image = Image::from_raster_data(&info, &pixels, 12).unwrap();

        // Offset by +1 in both axes: red should land at (2, 2).
        let filter = skia_rs_paint::OffsetImageFilter::new(1.0, 1.0, None);
        let filtered = image.make_with_filter(&filter).unwrap();
        assert_eq!(filtered.dimensions(), (3, 3));

        let color = filtered.read_pixel(2, 2).unwrap();
        assert!((color.r - 1.0).abs() < 1e-3);

        // Original center (1, 1) should now be black (clamp-to-edge fill
        // comes from outside the left/top edge, which is also black).
        let center = filtered.read_pixel(1, 1).unwrap();
        assert!(center.r < 0.01);
    }

    #[test]
    fn test_read_pixels_rejects_out_of_bounds() {
        let image = Image::from_color(4, 4, 0xFF_FF0000).unwrap();
        let dst_info = ImageInfo::new(4, 4, ColorType::Bgra8888, AlphaType::Premul);
        let mut dst = vec![0u8; 64];
        // Negative src_x.
        assert!(!image.read_pixels(&dst_info, &mut dst, 16, -1, 0));
        // Subset exceeds image width.
        assert!(!image.read_pixels(&dst_info, &mut dst, 16, 1, 0));
    }

    /// Unique IDs come from a monotonic counter, not a pointer: two distinct
    /// images always differ, and a clone shares the source's ID.
    #[test]
    fn test_unique_id_is_monotonic_not_pointer() {
        let a = Image::from_color(2, 2, 0xFF_FF0000).unwrap();
        let b = Image::from_color(2, 2, 0xFF_FF0000).unwrap();
        assert_ne!(a.unique_id(), b.unique_id(), "distinct images differ");
        assert_eq!(a.clone().unique_id(), a.unique_id(), "clone shares ID");

        // Drop an image and allocate another; a pointer-based ID could
        // recycle the freed address and collide. A monotonic counter cannot.
        let id_a = a.unique_id();
        drop(a);
        let c = Image::from_color(2, 2, 0xFF_FF0000).unwrap();
        assert_ne!(c.unique_id(), id_a);
    }

    /// Nearest-neighbor sampling uses pixel centers
    /// (`floor((dst + 0.5) * scale)`). Downscaling a 4x1 source
    /// [10,20,30,40] to 2x1 must pick the *center* source pixels: column 0
    /// maps to `floor(0.5 * 2) = 1` (value 20) and column 1 to
    /// `floor(1.5 * 2) = 3` (value 40). The old top-left mapping
    /// (`floor(dst * scale)`) would instead pick indices 0 and 2 (10 and 30).
    #[test]
    fn test_nearest_samples_pixel_centers() {
        let info = ImageInfo::new(4, 1, ColorType::Rgba8888, AlphaType::Opaque);
        let pixels = vec![
            10, 10, 10, 255, //
            20, 20, 20, 255, //
            30, 30, 30, 255, //
            40, 40, 40, 255,
        ];
        let src = Image::from_raster_data_owned(info, pixels, 16).unwrap();
        let scaled = src
            .make_scaled_with(2, 1, SamplingOptions::Nearest)
            .unwrap();
        let px = scaled.peek_pixels().unwrap();
        assert_eq!(px[0], 20, "col 0 -> src index 1");
        assert_eq!(px[4], 40, "col 1 -> src index 3");
    }

    /// Filtered scaling must operate in premultiplied space: a fully
    /// transparent texel's (arbitrary) color must not bleed into an adjacent
    /// opaque texel. Source: [opaque red | transparent green(garbage)].
    /// Bilinear downscale-to-average must keep the surviving color pure red,
    /// not a red/green mix.
    #[test]
    fn test_linear_filters_in_premul_space() {
        let info = ImageInfo::new(2, 1, ColorType::Rgba8888, AlphaType::Unpremul);
        // Left: opaque red. Right: transparent but with green stored in RGB
        // (as could occur from an unpremultiplied source).
        let pixels = vec![255, 0, 0, 255, 0, 255, 0, 0];
        let src = Image::from_raster_data_owned(info, pixels, 8).unwrap();
        // Downscale to 1x1: the single output blends both texels. In premul
        // space the transparent texel contributes no color, so the result is
        // red at half alpha; unpremultiplied that is pure red.
        let scaled = src.make_scaled_with(1, 1, SamplingOptions::Linear).unwrap();
        let px = scaled.peek_pixels().unwrap();
        assert_eq!(px[1], 0, "no green bleed from transparent texel");
        assert_eq!(px[2], 0, "blue stays zero");
        assert!(px[0] > 200, "red survives, got {}", px[0]);
    }
}
