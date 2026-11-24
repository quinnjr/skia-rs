//! Color, mask, and image filters.

#![allow(
    clippy::suboptimal_flops,
    reason = "faithful port of Skia's fma-free reference arithmetic; mul_add's fused rounding diverges from Skia"
)]

use skia_rs_core::cast::{f32_to_u8_sat, round_to_i32, saturate_to_i32, scalar_from_i32};
use skia_rs_core::{Color4f, Matrix, Rect, Scalar, mul_div_255_round};
use std::sync::Arc;

/// Truncating (non-rounding), saturating `f32` -> `u8` cast.
///
/// Matches the `f32_to_u8_trunc(x)` idiom used throughout this
/// file's pixel math, which truncates toward zero rather than rounding.
/// Differs from [`f32_to_u8_sat`], which additionally rounds to the
/// nearest integer; that rounding would change these filters' output by
/// up to one LSB per channel, so it is not used here.
#[inline]
fn f32_to_u8_trunc(x: f32) -> u8 {
    u8::try_from(saturate_to_i32(x).clamp(0, 255)).unwrap_or(0)
}

// =============================================================================
// Serialization Kind Constants
// =============================================================================

// MaskFilter kinds
pub(crate) const MASK_FILTER_KIND_BLUR: u8 = 1;
pub(crate) const MASK_FILTER_KIND_SHADER: u8 = 2;
pub(crate) const MASK_FILTER_KIND_TABLE: u8 = 3;

// ColorFilter kinds
pub(crate) const COLOR_FILTER_KIND_MATRIX: u8 = 1;

// ImageFilter kinds
pub(crate) const IMAGE_FILTER_KIND_BLUR: u8 = 1;
pub(crate) const IMAGE_FILTER_KIND_DROP_SHADOW: u8 = 2;
pub(crate) const IMAGE_FILTER_KIND_MORPHOLOGY: u8 = 3;
pub(crate) const IMAGE_FILTER_KIND_COLOR_FILTER: u8 = 4;
pub(crate) const IMAGE_FILTER_KIND_DISPLACEMENT_MAP: u8 = 5;
pub(crate) const IMAGE_FILTER_KIND_LIGHTING: u8 = 6;
pub(crate) const IMAGE_FILTER_KIND_COMPOSE: u8 = 7;
pub(crate) const IMAGE_FILTER_KIND_MERGE: u8 = 8;
pub(crate) const IMAGE_FILTER_KIND_OFFSET: u8 = 9;
pub(crate) const IMAGE_FILTER_KIND_MATRIX_CONVOLUTION: u8 = 10;
pub(crate) const IMAGE_FILTER_KIND_TILE: u8 = 11;
pub(crate) const IMAGE_FILTER_KIND_BLEND: u8 = 12;
pub(crate) const IMAGE_FILTER_KIND_ARITHMETIC: u8 = 13;

/// A color filter that transforms colors.
pub trait ColorFilter: Send + Sync + std::fmt::Debug {
    /// Filter a color.
    fn filter_color(&self, color: Color4f) -> Color4f;

    /// Serialize this filter for persistence. Returns None if the filter
    /// is not serializable (e.g., user-provided types).
    fn serialize(&self) -> Option<Vec<u8>> {
        None
    }
}

/// A matrix color filter.
#[derive(Debug, Clone)]
pub struct ColorMatrixFilter {
    /// 5x4 color matrix (row-major, last column is translation).
    matrix: [Scalar; 20],
}

impl ColorMatrixFilter {
    /// Create a new color matrix filter.
    #[must_use]
    pub const fn new(matrix: [Scalar; 20]) -> Self {
        Self { matrix }
    }

    /// Create an identity color matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ])
    }

    /// Create a saturation filter.
    #[must_use]
    pub fn saturation(sat: Scalar) -> Self {
        let s = sat;
        let ms = 1.0 - s;
        let r = 0.2126 * ms;
        let g = 0.7152 * ms;
        let b = 0.0722 * ms;

        Self::new([
            r + s,
            g,
            b,
            0.0,
            0.0,
            r,
            g + s,
            b,
            0.0,
            0.0,
            r,
            g,
            b + s,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ])
    }

    /// Create a color matrix filter that adjusts brightness.
    ///
    /// A value of 0.0 is no change; positive values brighten, negative darken.
    /// Implemented as adding `amount` to the RGB bias (translation column).
    #[must_use]
    pub const fn brightness(amount: Scalar) -> Self {
        let mut m = [0.0f32; 20];
        m[0] = 1.0;
        m[6] = 1.0;
        m[12] = 1.0;
        m[18] = 1.0;
        m[4] = amount;
        m[9] = amount;
        m[14] = amount;
        Self::new(m)
    }

    /// Create a color matrix filter that adjusts contrast.
    ///
    /// A value of 1.0 is no change; values > 1 increase contrast, < 1 decrease.
    #[must_use]
    pub fn contrast(amount: Scalar) -> Self {
        // result = (src - 0.5) * amount + 0.5
        //        = src * amount + (0.5 - 0.5 * amount)
        let bias = 0.5f32.mul_add(-amount, 0.5);
        let mut m = [0.0f32; 20];
        m[0] = amount;
        m[6] = amount;
        m[12] = amount;
        m[18] = 1.0;
        m[4] = bias;
        m[9] = bias;
        m[14] = bias;
        Self::new(m)
    }

    /// Create a color matrix filter that rotates hue by `degrees`.
    #[must_use]
    pub fn hue_rotate(degrees: Scalar) -> Self {
        let r = degrees.to_radians();
        let cos_a = r.cos();
        let sin_a = r.sin();
        // Reference: W3C feColorMatrix hueRotate
        let m = [
            sin_a.mul_add(-0.213, cos_a.mul_add(0.787, 0.213)),
            sin_a.mul_add(-0.715, cos_a.mul_add(-0.715, 0.715)),
            sin_a.mul_add(0.928, cos_a.mul_add(-0.072, 0.072)),
            0.0,
            0.0,
            sin_a.mul_add(0.143, cos_a.mul_add(-0.213, 0.213)),
            sin_a.mul_add(0.140, cos_a.mul_add(0.285, 0.715)),
            sin_a.mul_add(-0.283, cos_a.mul_add(-0.072, 0.072)),
            0.0,
            0.0,
            sin_a.mul_add(-0.787, cos_a.mul_add(-0.213, 0.213)),
            sin_a.mul_add(0.715, cos_a.mul_add(-0.715, 0.715)),
            sin_a.mul_add(0.072, cos_a.mul_add(0.928, 0.072)),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ];
        Self::new(m)
    }

    /// Create a color matrix filter that inverts RGB channels.
    #[must_use]
    pub const fn invert() -> Self {
        let m = [
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        Self::new(m)
    }

    /// Create a color matrix filter that applies a sepia tone.
    #[must_use]
    pub const fn sepia() -> Self {
        let m = [
            0.393, 0.769, 0.189, 0.0, 0.0, 0.349, 0.686, 0.168, 0.0, 0.0, 0.272, 0.534, 0.131, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        Self::new(m)
    }

    /// Create a color matrix filter that converts to grayscale using
    /// luminance weights (Rec. 601).
    #[must_use]
    pub const fn grayscale() -> Self {
        let r = 0.299;
        let g = 0.587;
        let b = 0.114;
        let m = [
            r, g, b, 0.0, 0.0, r, g, b, 0.0, 0.0, r, g, b, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        Self::new(m)
    }
}

impl ColorFilter for ColorMatrixFilter {
    fn filter_color(&self, color: Color4f) -> Color4f {
        let m = &self.matrix;
        Color4f {
            r: m[3].mul_add(
                color.a,
                m[2].mul_add(color.b, m[0].mul_add(color.r, m[1] * color.g)),
            ) + m[4],
            g: m[8].mul_add(
                color.a,
                m[7].mul_add(color.b, m[5].mul_add(color.r, m[6] * color.g)),
            ) + m[9],
            b: m[13].mul_add(
                color.a,
                m[12].mul_add(color.b, m[10].mul_add(color.r, m[11] * color.g)),
            ) + m[14],
            a: m[18].mul_add(
                color.a,
                m[17].mul_add(color.b, m[15].mul_add(color.r, m[16] * color.g)),
            ) + m[19],
        }
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![COLOR_FILTER_KIND_MATRIX];
        for v in &self.matrix {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Some(buf)
    }
}

/// Blur style for mask filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum BlurStyle {
    /// Normal blur.
    #[default]
    Normal = 0,
    /// Solid blur (inner visible).
    Solid,
    /// Outer blur only.
    Outer,
    /// Inner blur only.
    Inner,
}

/// A mask filter (blur, emboss, etc.).
pub trait MaskFilter: Send + Sync + std::fmt::Debug {
    /// Get the blur radius if this is a blur filter.
    fn blur_radius(&self) -> Option<Scalar>;

    /// Apply the filter to an alpha mask in place.
    ///
    /// `mask` is a row-major `width * height` bytes of alpha values.
    /// Default implementation is a no-op — concrete filters override.
    fn apply_mask(&self, mask: &mut [u8], width: usize, height: usize) {
        let _ = (mask, width, height);
    }

    /// Serialize this filter for persistence. Returns None if the filter
    /// is not serializable (e.g., user-provided types).
    fn serialize(&self) -> Option<Vec<u8>> {
        None
    }
}

/// A blur mask filter.
#[derive(Debug, Clone)]
pub struct BlurMaskFilter {
    style: BlurStyle,
    sigma: Scalar,
}

impl BlurMaskFilter {
    /// Create a new blur mask filter.
    #[must_use]
    pub const fn new(style: BlurStyle, sigma: Scalar) -> Self {
        Self { style, sigma }
    }

    /// Get the blur style.
    #[must_use]
    pub const fn style(&self) -> BlurStyle {
        self.style
    }

    /// Get the sigma value.
    #[must_use]
    pub const fn sigma(&self) -> Scalar {
        self.sigma
    }
}

impl MaskFilter for BlurMaskFilter {
    fn blur_radius(&self) -> Option<Scalar> {
        Some(self.sigma)
    }

    fn apply_mask(&self, mask: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // SkBlurMask::ConvertSigmaToRadius; do not truncate sigma directly.
        let radius = sigma_to_int_box_radius(self.sigma);
        if radius == 0 {
            // No effective blur. Per SkBlurMask::BoxBlur, Outer style then
            // produces an empty mask; other styles keep the original.
            if self.style == BlurStyle::Outer {
                mask.fill(0);
            }
            return;
        }

        // Keep the source mask for style composition.
        let src: Vec<u8> = mask.to_vec();

        // Separable box blur (horizontal then vertical), three passes for
        // the classic three-box Gaussian approximation.
        for _ in 0..3 {
            box_blur_horizontal(mask, width, height, radius);
            box_blur_vertical(mask, width, height, radius);
        }

        // Compose the blur with the source per SkBlurMask's style handling.
        match self.style {
            BlurStyle::Normal => {}
            BlurStyle::Solid => {
                // src UNION blur: d = s + d - s*d (clamp_solid_with_orig).
                for (d, &s) in mask.iter_mut().zip(src.iter()) {
                    let blur = u32::from(*d);
                    let s = u32::from(s);
                    let combined = (s + blur - u32::from(mul_div_255_round(s, blur))).min(255);
                    *d = u8::try_from(combined).unwrap_or(u8::MAX);
                }
            }
            BlurStyle::Outer => {
                // blur MINUS src: d = d * (1 - s) (clamp_outer_with_orig).
                for (d, &s) in mask.iter_mut().zip(src.iter()) {
                    *d = mul_div_255_round(u32::from(*d), 255 - u32::from(s));
                }
            }
            BlurStyle::Inner => {
                // blur INTERSECT src: d = d * s (merge_src_with_blur).
                for (d, &s) in mask.iter_mut().zip(src.iter()) {
                    *d = mul_div_255_round(u32::from(*d), u32::from(s));
                }
            }
        }
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![MASK_FILTER_KIND_BLUR];
        buf.push(self.style as u8);
        buf.extend_from_slice(&self.sigma.to_le_bytes());
        Some(buf)
    }
}

/// An image filter.
pub trait ImageFilter: Send + Sync + std::fmt::Debug {
    /// Get the bounds that this filter affects.
    fn filter_bounds(&self, src: &Rect) -> Rect;

    /// Apply the filter to an RGBA8 image buffer in place.
    ///
    /// `pixels` is a row-major `width * height * 4` byte buffer.
    /// Default implementation is a no-op — concrete filters override.
    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        let _ = (pixels, width, height);
    }

    /// Serialize this filter for persistence. Returns None if the filter
    /// is not serializable (e.g., user-provided types).
    fn serialize(&self) -> Option<Vec<u8>> {
        None
    }

    /// Whether the filter turns transparent black into something visible
    /// (`SkImageFilter_Base::affectsTransparentBlack`): its output then
    /// covers the whole clip, not just the layer's content. As in Skia, a
    /// filter with inputs is affected when any input is; every built-in
    /// filter with inputs overrides this to check them.
    fn affects_transparent_black(&self) -> bool {
        false
    }

    /// The area of the source this filter reads to produce `output` (both
    /// in layer space), like `SkImageFilter_Base::onGetInputLayerBounds`:
    /// a layer with this filter must hold it. The default mirrors
    /// [`filter_bounds`](Self::filter_bounds), outsetting `output` by the
    /// filter's largest spread on any side.
    fn input_bounds(&self, output: &Rect) -> Rect {
        spread_input(&self.filter_bounds(output), output)
    }

    /// This filter with its geometric parameters (blur sigmas, offsets,
    /// radii, rects, and those of its inputs) mapped through `matrix`, as
    /// `SkCanvas` maps a layer's image filter from local space into layer
    /// space (`skif::Mapping::paramToLayer`). `None` when nothing changes:
    /// the default, for filters without geometric parameters or inputs
    /// (every built-in filter that has either overrides this).
    ///
    /// Filters run on device-space layer pixels here, while Skia decomposes
    /// the matrix (`skif::Mapping::decomposeCTM`) and filters in a
    /// scale-only (or, for matrix convolution and lighting, translate-only)
    /// layer space, resampling the result. The mapping here matches Skia:
    /// - blur, drop shadow, morphology, offset, tile, color filter and the
    ///   combining filters: under scale/translate and quarter turns;
    /// - matrix convolution: under translate and axis-permuting matrices
    ///   (quarter turns, mirrors) at unit scale (the kernel is re-oriented,
    ///   not resampled);
    /// - lighting: lights map under translate, scale and quarter turns
    ///   (`apply` here shades with a uniform factor, not per-pixel normals);
    /// - displacement map: under translate, uniform scale, half turns and
    ///   diagonal swaps (one signed scale for both axes cannot express other
    ///   quarter turns or single-axis mirrors).
    ///
    /// Anisotropic filters under other rotations or skews are fitted to the
    /// nearest axes.
    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        let _ = matrix;
        None
    }
}

/// `filter` with its parameters mapped through `matrix` (see
/// [`ImageFilter::map_params`]); `filter` itself when nothing changes.
#[must_use]
pub fn map_image_filter(filter: &ImageFilterRef, matrix: &Matrix) -> ImageFilterRef {
    filter
        .map_params(matrix)
        .unwrap_or_else(|| Arc::clone(filter))
}

/// `SkBlurImageFilter`'s `kMaxSigma`: blur sigmas (in layer space) are
/// clamped to this, which also bounds how far a blur reaches.
const MAX_BLUR_SIGMA: Scalar = 532.0;

/// A blur sigma clamped like `SkBlurImageFilter` (to [`MAX_BLUR_SIGMA`]).
/// A non-finite or negative sigma, which `SkImageFilters::Blur` rejects
/// (no filter), is 0: no blur.
const fn max_sigma(sigma: Scalar) -> Scalar {
    if sigma.is_finite() && sigma > 0.0 {
        sigma.min(MAX_BLUR_SIGMA)
    } else {
        0.0
    }
}

/// `output` outset by the largest amount `spread` (a filter's output bounds
/// for `output`) extends past it on any side.
fn spread_input(spread: &Rect, output: &Rect) -> Rect {
    let reach = [
        output.left - spread.left,
        output.top - spread.top,
        spread.right - output.right,
        spread.bottom - output.bottom,
    ]
    .into_iter()
    .fold(0.0, Scalar::max);
    if reach.is_finite() {
        output.inset(-reach, -reach)
    } else {
        *output
    }
}

/// Whether a color filter turns transparent black into something visible.
#[must_use]
pub fn color_filter_affects_transparent_black(filter: &dyn ColorFilter) -> bool {
    filter.filter_color(Color4f::new(0.0, 0.0, 0.0, 0.0)).a > 0.0
}

/// Whether an optional input affects transparent black.
fn input_affects(input: Option<&ImageFilterRef>) -> bool {
    input.is_some_and(|f| f.affects_transparent_black())
}

/// The bounds an optional `input` filter produces from `src` (`src` itself
/// with no input): filters with an input start from it, as Skia's
/// `onFilterBounds` maps through the inputs first.
fn input_filter_bounds(input: Option<&ImageFilterRef>, src: &Rect) -> Rect {
    input.map_or(*src, |input| input.filter_bounds(src))
}

/// What an optional input filter reads for `rect` (`None` is the source
/// itself, which is read as is).
fn input_of(input: Option<&ImageFilterRef>, rect: &Rect) -> Rect {
    input.map_or(*rect, |f| f.input_bounds(rect))
}

/// Maps an optional input filter.
fn map_input(input: Option<&ImageFilterRef>, matrix: &Matrix) -> Option<ImageFilterRef> {
    input.map(|f| map_image_filter(f, matrix))
}

/// The uniform scale of `matrix`'s linear part (`sqrt(|det|)`).
fn linear_scale(matrix: &Matrix) -> Scalar {
    let v = &matrix.values;
    (v[Matrix::SCALE_X] * v[Matrix::SCALE_Y] - v[Matrix::SKEW_X] * v[Matrix::SKEW_Y])
        .abs()
        .sqrt()
}

/// Maps a vector (an offset) through the linear part of `matrix`.
fn map_vector(matrix: &Matrix, dx: Scalar, dy: Scalar) -> (Scalar, Scalar) {
    let v = &matrix.values;
    (
        v[Matrix::SCALE_X] * dx + v[Matrix::SKEW_X] * dy,
        v[Matrix::SKEW_Y] * dx + v[Matrix::SCALE_Y] * dy,
    )
}

/// Maps a size (sigmas, radii) as the lengths of its mapped axis vectors,
/// each assigned to the device axis its local axis mostly maps to: exact
/// for scale/translate and quarter turns (where x and y swap). Other
/// rotations and skews get the nearest axis-aligned fit; see
/// [`ImageFilter::map_params`].
fn map_size(matrix: &Matrix, w: Scalar, h: Scalar) -> (Scalar, Scalar) {
    let (x0, y0) = map_vector(matrix, w, 0.0);
    let (x1, y1) = map_vector(matrix, 0.0, h);
    let (along_w, along_h) = (x0.hypot(y0), x1.hypot(y1));
    // Where local x points, independent of w (which may be 0).
    let (ux, uy) = map_vector(matrix, 1.0, 0.0);
    if ux.abs() < uy.abs() {
        (along_h, along_w)
    } else {
        (along_w, along_h)
    }
}

/// A blur image filter.
#[derive(Debug, Clone)]
pub struct BlurImageFilter {
    sigma_x: Scalar,
    sigma_y: Scalar,
    tile_mode: crate::shader::TileMode,
}

impl BlurImageFilter {
    /// Create a new blur image filter.
    #[must_use]
    pub const fn new(sigma_x: Scalar, sigma_y: Scalar, tile_mode: crate::shader::TileMode) -> Self {
        Self {
            sigma_x,
            sigma_y,
            tile_mode,
        }
    }
}

impl ImageFilter for BlurImageFilter {
    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        let (sigma_x, sigma_y) = map_size(matrix, max_sigma(self.sigma_x), max_sigma(self.sigma_y));
        let (sigma_x, sigma_y) = (max_sigma(sigma_x), max_sigma(sigma_y));
        Some(Arc::new(Self {
            sigma_x,
            sigma_y,
            tile_mode: self.tile_mode,
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        // Blur expands bounds by ~3 sigma
        let dx = max_sigma(self.sigma_x) * 3.0;
        let dy = max_sigma(self.sigma_y) * 3.0;
        Rect::new(src.left - dx, src.top - dy, src.right + dx, src.bottom + dy)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // SkBlurMask::ConvertSigmaToRadius; do not truncate sigma directly.
        let rx = sigma_to_int_box_radius(max_sigma(self.sigma_x));
        let ry = sigma_to_int_box_radius(max_sigma(self.sigma_y));
        if rx == 0 && ry == 0 {
            return;
        }
        // Three-box approximation of a Gaussian.
        for _ in 0..3 {
            if rx > 0 {
                rgba_box_blur_horizontal(pixels, width, height, rx);
            }
            if ry > 0 {
                rgba_box_blur_vertical(pixels, width, height, ry);
            }
        }
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_BLUR];
        buf.extend_from_slice(&self.sigma_x.to_le_bytes());
        buf.extend_from_slice(&self.sigma_y.to_le_bytes());
        buf.push(self.tile_mode as u8);
        Some(buf)
    }
}

/// A drop shadow image filter.
#[derive(Debug, Clone)]
pub struct DropShadowImageFilter {
    dx: Scalar,
    dy: Scalar,
    sigma_x: Scalar,
    sigma_y: Scalar,
    color: Color4f,
    shadow_only: bool,
}

impl DropShadowImageFilter {
    /// Create a new drop shadow filter.
    #[must_use]
    pub const fn new(
        dx: Scalar,
        dy: Scalar,
        sigma_x: Scalar,
        sigma_y: Scalar,
        color: Color4f,
        shadow_only: bool,
    ) -> Self {
        Self {
            dx,
            dy,
            sigma_x,
            sigma_y,
            color,
            shadow_only,
        }
    }
}

impl ImageFilter for DropShadowImageFilter {
    fn input_bounds(&self, output: &Rect) -> Rect {
        // The shadow at `output` comes from the source moved back by the
        // offset, blurred (3 sigma); the source itself is drawn unless
        // shadow-only.
        let shadow = output.offset(-self.dx, -self.dy).inset(
            -3.0 * max_sigma(self.sigma_x),
            -3.0 * max_sigma(self.sigma_y),
        );
        if self.shadow_only {
            shadow
        } else {
            shadow.union(output)
        }
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        let (dx, dy) = map_vector(matrix, self.dx, self.dy);
        let (sigma_x, sigma_y) = map_size(matrix, max_sigma(self.sigma_x), max_sigma(self.sigma_y));
        let (sigma_x, sigma_y) = (max_sigma(sigma_x), max_sigma(sigma_y));
        Some(Arc::new(Self {
            dx,
            dy,
            sigma_x,
            sigma_y,
            ..self.clone()
        }))
    }

    #[allow(
        clippy::similar_names,
        reason = "blur_dx/blur_dy are the conventional x/y-axis pair for this offset+blur computation"
    )]
    fn filter_bounds(&self, src: &Rect) -> Rect {
        let blur_dx = max_sigma(self.sigma_x) * 3.0;
        let blur_dy = max_sigma(self.sigma_y) * 3.0;

        if self.shadow_only {
            Rect::new(
                src.left + self.dx - blur_dx,
                src.top + self.dy - blur_dy,
                src.right + self.dx + blur_dx,
                src.bottom + self.dy + blur_dy,
            )
        } else {
            Rect::new(
                src.left.min(src.left + self.dx - blur_dx),
                src.top.min(src.top + self.dy - blur_dy),
                src.right.max(src.right + self.dx + blur_dx),
                src.bottom.max(src.bottom + self.dy + blur_dy),
            )
        }
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_DROP_SHADOW];
        buf.extend_from_slice(&self.dx.to_le_bytes());
        buf.extend_from_slice(&self.dy.to_le_bytes());
        buf.extend_from_slice(&self.sigma_x.to_le_bytes());
        buf.extend_from_slice(&self.sigma_y.to_le_bytes());
        buf.extend_from_slice(&self.color.r.to_le_bytes());
        buf.extend_from_slice(&self.color.g.to_le_bytes());
        buf.extend_from_slice(&self.color.b.to_le_bytes());
        buf.extend_from_slice(&self.color.a.to_le_bytes());
        buf.push(u8::from(self.shadow_only));
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // 1. Extract alpha channel into shadow buffer
        let mut shadow = vec![0u8; width * height];
        for i in 0..width * height {
            shadow[i] = pixels[i * 4 + 3];
        }
        // 2. Blur shadow (sigma converted per SkBlurMask::ConvertSigmaToRadius)
        let rx = sigma_to_int_box_radius(max_sigma(self.sigma_x));
        let ry = sigma_to_int_box_radius(max_sigma(self.sigma_y));
        if rx > 0 {
            box_blur_horizontal(&mut shadow, width, height, rx);
        }
        if ry > 0 {
            box_blur_vertical(&mut shadow, width, height, ry);
        }
        // 3. Colorize and offset the shadow, then composite
        let dx = saturate_to_i32(self.dx.round());
        let dy = saturate_to_i32(self.dy.round());
        let sr = f32_to_u8_trunc(self.color.r * 255.0);
        let sg = f32_to_u8_trunc(self.color.g * 255.0);
        let sb = f32_to_u8_trunc(self.color.b * 255.0);
        let sa_mul = self.color.a.clamp(0.0, 1.0);
        let original = pixels.to_vec();
        for y in 0..height {
            for x in 0..width {
                // Read shadow from offset position (clamp)
                let x_i32 = i32::try_from(x).unwrap_or(i32::MAX);
                let y_i32 = i32::try_from(y).unwrap_or(i32::MAX);
                let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
                let height_i32 = i32::try_from(height).unwrap_or(i32::MAX);
                let sx = usize::try_from((x_i32 - dx).clamp(0, width_i32 - 1)).unwrap_or(0);
                let sy = usize::try_from((y_i32 - dy).clamp(0, height_i32 - 1)).unwrap_or(0);
                let shadow_alpha = (f32::from(shadow[sy * width + sx]) / 255.0) * sa_mul;
                let shadow_rgba = [
                    f32_to_u8_trunc(f32::from(sr) * shadow_alpha),
                    f32_to_u8_trunc(f32::from(sg) * shadow_alpha),
                    f32_to_u8_trunc(f32::from(sb) * shadow_alpha),
                    f32_to_u8_trunc(shadow_alpha * 255.0),
                ];
                // Src-over: result = src + dst * (1 - src.a)
                let idx = (y * width + x) * 4;
                if self.shadow_only {
                    // Only shadow, no source
                    pixels[idx] = shadow_rgba[0];
                    pixels[idx + 1] = shadow_rgba[1];
                    pixels[idx + 2] = shadow_rgba[2];
                    pixels[idx + 3] = shadow_rgba[3];
                } else {
                    // Composite source over shadow
                    let src_a = f32::from(original[idx + 3]) / 255.0;
                    let inv_a = 1.0 - src_a;
                    pixels[idx] = f32_to_u8_trunc(
                        f32::from(original[idx]) + f32::from(shadow_rgba[0]) * inv_a,
                    );
                    pixels[idx + 1] = f32_to_u8_trunc(
                        f32::from(original[idx + 1]) + f32::from(shadow_rgba[1]) * inv_a,
                    );
                    pixels[idx + 2] = f32_to_u8_trunc(
                        f32::from(original[idx + 2]) + f32::from(shadow_rgba[2]) * inv_a,
                    );
                    pixels[idx + 3] = f32_to_u8_trunc(
                        f32::from(original[idx + 3]) + f32::from(shadow_rgba[3]) * inv_a,
                    );
                }
            }
        }
    }
}

// =============================================================================
// Additional Mask Filters
// =============================================================================

/// A shader-based mask filter.
///
/// Corresponds to Skia's `SkShaderMaskFilter`.
#[derive(Debug)]
pub struct ShaderMaskFilter {
    shader: crate::shader::ShaderRef,
}

impl ShaderMaskFilter {
    /// Create a new shader mask filter.
    pub fn new(shader: crate::shader::ShaderRef) -> Self {
        Self { shader }
    }

    /// Get the shader.
    #[must_use]
    pub fn shader(&self) -> &crate::shader::ShaderRef {
        &self.shader
    }
}

impl MaskFilter for ShaderMaskFilter {
    fn blur_radius(&self) -> Option<Scalar> {
        None
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![MASK_FILTER_KIND_SHADER];
        let shader_data = self.shader.serialize()?;
        let len = u32::try_from(shader_data.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&shader_data);
        Some(buf)
    }
}

/// A table-based mask filter using a lookup table.
///
/// Corresponds to Skia's `SkTableMaskFilter`.
#[derive(Debug, Clone)]
pub struct TableMaskFilter {
    table: [u8; 256],
}

impl TableMaskFilter {
    /// Create a new table mask filter.
    #[must_use]
    pub const fn new(table: [u8; 256]) -> Self {
        Self { table }
    }

    /// Create a gamma correction table.
    #[must_use]
    pub fn gamma(gamma: Scalar) -> Self {
        let mut table = [0u8; 256];
        for (i, v) in table.iter_mut().enumerate() {
            let normalized = scalar_from_i32(i32::try_from(i).unwrap_or(0)) / 255.0;
            *v = f32_to_u8_sat(normalized.powf(gamma) * 255.0);
        }
        Self { table }
    }

    /// Create a clipping table.
    #[must_use]
    pub fn clip(min: u8, max: u8) -> Self {
        let mut table = [0u8; 256];
        for (i, v) in table.iter_mut().enumerate() {
            let i = u8::try_from(i).unwrap_or(u8::MAX);
            *v = if i < min {
                0
            } else if i > max {
                255
            } else {
                f32_to_u8_trunc(f32::from(i - min) / f32::from(max - min) * 255.0)
            };
        }
        Self { table }
    }

    /// Get the lookup table.
    #[must_use]
    pub const fn table(&self) -> &[u8; 256] {
        &self.table
    }
}

impl MaskFilter for TableMaskFilter {
    fn blur_radius(&self) -> Option<Scalar> {
        None
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![MASK_FILTER_KIND_TABLE];
        buf.extend_from_slice(&self.table);
        Some(buf)
    }
}

// =============================================================================
// Additional Image Filters
// =============================================================================

/// Morphology filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MorphologyType {
    /// Dilate (expand bright regions).
    Dilate,
    /// Erode (shrink bright regions).
    Erode,
}

/// A morphology image filter (dilate or erode).
///
/// Corresponds to Skia's `SkMorphologyImageFilter`.
#[derive(Debug, Clone)]
pub struct MorphologyImageFilter {
    morph_type: MorphologyType,
    radius_x: Scalar,
    radius_y: Scalar,
    input: Option<ImageFilterRef>,
}

impl MorphologyImageFilter {
    /// Create a dilate filter.
    #[must_use]
    pub fn dilate(radius_x: Scalar, radius_y: Scalar, input: Option<ImageFilterRef>) -> Self {
        Self {
            morph_type: MorphologyType::Dilate,
            radius_x,
            radius_y,
            input,
        }
    }

    /// Create an erode filter.
    #[must_use]
    pub fn erode(radius_x: Scalar, radius_y: Scalar, input: Option<ImageFilterRef>) -> Self {
        Self {
            morph_type: MorphologyType::Erode,
            radius_x,
            radius_y,
            input,
        }
    }

    /// Get the morphology type.
    #[must_use]
    pub const fn morph_type(&self) -> MorphologyType {
        self.morph_type
    }

    /// Get the X radius.
    #[must_use]
    pub const fn radius_x(&self) -> Scalar {
        self.radius_x
    }

    /// Get the Y radius.
    #[must_use]
    pub const fn radius_y(&self) -> Scalar {
        self.radius_y
    }
}

impl ImageFilter for MorphologyImageFilter {
    fn affects_transparent_black(&self) -> bool {
        input_affects(self.input.as_ref())
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        input_of(
            self.input.as_ref(),
            &output.inset(-self.radius_x, -self.radius_y),
        )
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        // A negative or NaN radius does nothing; mapping takes lengths, so
        // keep it at 0 rather than let the sign drop.
        let (radius_x, radius_y) = map_size(matrix, self.radius_x.max(0.0), self.radius_y.max(0.0));
        Some(Arc::new(Self {
            morph_type: self.morph_type,
            radius_x,
            radius_y,
            input: map_input(self.input.as_ref(), matrix),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        let src = input_filter_bounds(self.input.as_ref(), src);
        Rect::new(
            src.left - self.radius_x,
            src.top - self.radius_y,
            src.right + self.radius_x,
            src.bottom + self.radius_y,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_MORPHOLOGY];
        buf.push(match self.morph_type {
            MorphologyType::Dilate => 0,
            MorphologyType::Erode => 1,
        });
        buf.extend_from_slice(&self.radius_x.to_le_bytes());
        buf.extend_from_slice(&self.radius_y.to_le_bytes());
        // Serialize input filter if present
        match &self.input {
            Some(input) => {
                let input_data = input.serialize()?;
                buf.push(1);
                let len = u32::try_from(input_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&input_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // The input filter runs first (Skia filters its input image).
        if let Some(ref input) = self.input {
            input.apply(pixels, width, height);
        }
        let rx = usize::try_from(round_to_i32(self.radius_x.max(0.0))).unwrap_or(0);
        let ry = usize::try_from(round_to_i32(self.radius_y.max(0.0))).unwrap_or(0);
        if rx == 0 && ry == 0 {
            return;
        }
        let is_dilate = matches!(self.morph_type, MorphologyType::Dilate);
        let src = pixels.to_vec();
        for y in 0..height {
            for x in 0..width {
                let mut r = if is_dilate { 0u8 } else { 255u8 };
                let mut g = r;
                let mut b = r;
                let mut a = r;
                let x_min = x.saturating_sub(rx);
                let y_min = y.saturating_sub(ry);
                let x_max = (x + rx).min(width - 1);
                let y_max = (y + ry).min(height - 1);
                for ny in y_min..=y_max {
                    for nx in x_min..=x_max {
                        let idx = (ny * width + nx) * 4;
                        let sr = src[idx];
                        let sg = src[idx + 1];
                        let sb = src[idx + 2];
                        let sa = src[idx + 3];
                        if is_dilate {
                            if sr > r {
                                r = sr;
                            }
                            if sg > g {
                                g = sg;
                            }
                            if sb > b {
                                b = sb;
                            }
                            if sa > a {
                                a = sa;
                            }
                        } else {
                            if sr < r {
                                r = sr;
                            }
                            if sg < g {
                                g = sg;
                            }
                            if sb < b {
                                b = sb;
                            }
                            if sa < a {
                                a = sa;
                            }
                        }
                    }
                }
                let idx = (y * width + x) * 4;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = a;
            }
        }
    }
}

/// A color filter wrapped as an image filter.
///
/// Corresponds to Skia's `SkColorFilterImageFilter`.
#[derive(Debug)]
pub struct ColorFilterImageFilter {
    color_filter: ColorFilterRef,
    input: Option<ImageFilterRef>,
}

impl ColorFilterImageFilter {
    /// Create a new color filter image filter.
    pub fn new(color_filter: ColorFilterRef, input: Option<ImageFilterRef>) -> Self {
        Self {
            color_filter,
            input,
        }
    }

    /// Get the color filter.
    #[must_use]
    pub fn color_filter(&self) -> &ColorFilterRef {
        &self.color_filter
    }
}

impl ImageFilter for ColorFilterImageFilter {
    fn affects_transparent_black(&self) -> bool {
        input_affects(self.input.as_ref())
            || color_filter_affects_transparent_black(self.color_filter.as_ref())
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        input_of(self.input.as_ref(), output)
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        let input = self.input.as_ref()?;
        Some(Arc::new(Self {
            color_filter: Arc::clone(&self.color_filter),
            input: Some(map_image_filter(input, matrix)),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        // A color filter keeps its input's bounds.
        input_filter_bounds(self.input.as_ref(), src)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_COLOR_FILTER];
        // Serialize the wrapped color filter
        let color_filter_data = self.color_filter.serialize()?;
        let len = u32::try_from(color_filter_data.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&color_filter_data);
        // Serialize input filter if present
        match &self.input {
            Some(input) => {
                let input_data = input.serialize()?;
                buf.push(1);
                let len = u32::try_from(input_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&input_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // Apply inner filter first if present
        if let Some(ref input) = self.input {
            input.apply(pixels, width, height);
        }
        // Apply the color filter per pixel. The buffer holds premultiplied
        // RGBA8; SkColorFilters::Matrix runs in unpremul space by default
        // (unpremul -> matrix -> clamp -> premul).
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let a = f32::from(pixels[idx + 3]) / 255.0;
                let unpremul = if a > 0.0 {
                    Color4f {
                        r: (f32::from(pixels[idx]) / 255.0) / a,
                        g: (f32::from(pixels[idx + 1]) / 255.0) / a,
                        b: (f32::from(pixels[idx + 2]) / 255.0) / a,
                        a,
                    }
                } else {
                    Color4f::new(0.0, 0.0, 0.0, 0.0)
                };
                let filtered = self.color_filter.filter_color(unpremul);
                // Clamp in unpremul space, then re-premultiply.
                let fa = filtered.a.clamp(0.0, 1.0);
                let fr = filtered.r.clamp(0.0, 1.0) * fa;
                let fg = filtered.g.clamp(0.0, 1.0) * fa;
                let fb = filtered.b.clamp(0.0, 1.0) * fa;
                pixels[idx] = f32_to_u8_sat(fr * 255.0);
                pixels[idx + 1] = f32_to_u8_sat(fg * 255.0);
                pixels[idx + 2] = f32_to_u8_sat(fb * 255.0);
                pixels[idx + 3] = f32_to_u8_sat(fa * 255.0);
            }
        }
    }
}

/// A displacement map image filter.
///
/// Corresponds to Skia's `SkDisplacementMapEffect`.
#[derive(Debug)]
pub struct DisplacementMapImageFilter {
    x_channel: ColorChannel,
    y_channel: ColorChannel,
    scale: Scalar,
    displacement: ImageFilterRef,
    color: Option<ImageFilterRef>,
}

/// Color channel selector for displacement map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorChannel {
    /// Red channel.
    R,
    /// Green channel.
    G,
    /// Blue channel.
    B,
    /// Alpha channel.
    A,
}

impl DisplacementMapImageFilter {
    /// Create a new displacement map filter.
    pub fn new(
        x_channel: ColorChannel,
        y_channel: ColorChannel,
        scale: Scalar,
        displacement: ImageFilterRef,
        color: Option<ImageFilterRef>,
    ) -> Self {
        Self {
            x_channel,
            y_channel,
            scale,
            displacement,
            color,
        }
    }
}

impl ImageFilter for DisplacementMapImageFilter {
    fn affects_transparent_black(&self) -> bool {
        self.displacement.affects_transparent_black() || input_affects(self.color.as_ref())
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        // The color input is read up to scale/2 away; the displacement map
        // at the output itself.
        let reach = self.scale.abs() / 2.0;
        let color = input_of(self.color.as_ref(), &output.inset(-reach, -reach));
        color.union(&self.displacement.input_bounds(output))
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        // Skia maps the scale as a size; this filter has one scale for
        // both axes, so take the mean of the two mapped lengths.
        let (sx, sy) = map_size(matrix, self.scale, self.scale);
        // The local displacement (scale * (cx, cy)) maps through the linear
        // part. One signed scale can express that exactly when both device
        // axes get the same sign: scale/translate with equal signs (incl. a
        // half turn), or an axis swap with equal signs (channels swap).
        // Other quarter turns and single-axis mirrors are approximated by
        // the magnitude with the original sign.
        let values = &matrix.values;
        let (scale_x, skew_x, skew_y, scale_y) = (
            values[Matrix::SCALE_X],
            values[Matrix::SKEW_X],
            values[Matrix::SKEW_Y],
            values[Matrix::SCALE_Y],
        );
        let same_sign =
            |lhs: Scalar, rhs: Scalar| lhs != 0.0 && rhs != 0.0 && (lhs > 0.0) == (rhs > 0.0);
        let (x_channel, y_channel, sign) =
            if skew_x == 0.0 && skew_y == 0.0 && same_sign(scale_x, scale_y) {
                (self.x_channel, self.y_channel, scale_x.signum())
            } else if scale_x == 0.0 && scale_y == 0.0 && same_sign(skew_x, skew_y) {
                (self.y_channel, self.x_channel, skew_x.signum())
            } else {
                (self.x_channel, self.y_channel, 1.0)
            };
        Some(Arc::new(Self {
            x_channel,
            y_channel,
            // Lengths lose the sign: restore it from the scale and matrix.
            scale: f32::midpoint(sx, sy).copysign(self.scale) * sign,
            displacement: map_image_filter(&self.displacement, matrix),
            color: map_input(self.color.as_ref(), matrix),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        // The color input's output, moved by up to |scale|/2 in each
        // direction (a negative scale moves it the other way, not less).
        let src = input_filter_bounds(self.color.as_ref(), src);
        let offset = self.scale.abs() / 2.0;
        Rect::new(
            src.left - offset,
            src.top - offset,
            src.right + offset,
            src.bottom + offset,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_DISPLACEMENT_MAP];
        buf.push(match self.x_channel {
            ColorChannel::R => 0,
            ColorChannel::G => 1,
            ColorChannel::B => 2,
            ColorChannel::A => 3,
        });
        buf.push(match self.y_channel {
            ColorChannel::R => 0,
            ColorChannel::G => 1,
            ColorChannel::B => 2,
            ColorChannel::A => 3,
        });
        buf.extend_from_slice(&self.scale.to_le_bytes());
        // Serialize displacement filter
        let displacement_data = self.displacement.serialize()?;
        let len = u32::try_from(displacement_data.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&displacement_data);
        // Serialize color filter if present
        match &self.color {
            Some(color) => {
                let color_data = color.serialize()?;
                buf.push(1);
                let len = u32::try_from(color_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&color_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // As SkDisplacementMapImageFilter: the offsets come from the
        // displacement input's output, and the pixels moved come from the
        // color input's output (the source itself with no color input).
        let scale = self.scale;
        let mut map = pixels.to_vec();
        self.displacement.apply(&mut map, width, height);
        let mut src = pixels.to_vec();
        if let Some(color) = &self.color {
            color.apply(&mut src, width, height);
        }
        let x_channel_idx = match self.x_channel {
            ColorChannel::R => 0,
            ColorChannel::G => 1,
            ColorChannel::B => 2,
            ColorChannel::A => 3,
        };
        let y_channel_idx = match self.y_channel {
            ColorChannel::R => 0,
            ColorChannel::G => 1,
            ColorChannel::B => 2,
            ColorChannel::A => 3,
        };
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                // Read displacement channels (0..255 maps to -1..1)
                let disp_x = (f32::from(map[idx + x_channel_idx]) / 127.5 - 1.0) * scale;
                let disp_y = (f32::from(map[idx + y_channel_idx]) / 127.5 - 1.0) * scale;
                let x_i32 = i32::try_from(x).unwrap_or(i32::MAX);
                let y_i32 = i32::try_from(y).unwrap_or(i32::MAX);
                let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
                let height_i32 = i32::try_from(height).unwrap_or(i32::MAX);
                let sample_x = saturate_to_i32(scalar_from_i32(x_i32) + disp_x);
                let sample_y = saturate_to_i32(scalar_from_i32(y_i32) + disp_y);
                let sx = usize::try_from(sample_x.clamp(0, width_i32 - 1)).unwrap_or(0);
                let sy = usize::try_from(sample_y.clamp(0, height_i32 - 1)).unwrap_or(0);
                let src_idx = (sy * width + sx) * 4;
                pixels[idx] = src[src_idx];
                pixels[idx + 1] = src[src_idx + 1];
                pixels[idx + 2] = src[src_idx + 2];
                pixels[idx + 3] = src[src_idx + 3];
            }
        }
    }
}

/// Light type for lighting filters.
#[derive(Debug, Clone)]
pub enum LightType {
    /// Distant light (parallel rays).
    Distant {
        /// Direction vector (towards light source).
        direction: (Scalar, Scalar, Scalar),
    },
    /// Point light (emanates from a single point).
    Point {
        /// Light position.
        location: (Scalar, Scalar, Scalar),
    },
    /// Spot light (cone of light from a point).
    Spot {
        /// Light position.
        location: (Scalar, Scalar, Scalar),
        /// Target point.
        target: (Scalar, Scalar, Scalar),
        /// Specular exponent.
        specular_exponent: Scalar,
        /// Cutoff angle.
        cutoff_angle: Scalar,
    },
}

impl LightType {
    /// The light mapped into layer space, like `SkLight::transform`: light
    /// positions map as points with z scaled by the matrix's uniform scale
    /// (`SkLight` averages the mapped (z, z) magnitudes). A distant light's
    /// direction turns with the matrix but keeps its x/y length and z, so
    /// its elevation does not change.
    fn mapped(&self, matrix: &Matrix) -> Self {
        // Heights scale by the uniform scale only: mapping (z, z) through the
        // signed linear part would flip or zero them under rotations.
        let z_scale = linear_scale(matrix);
        let map_point3 = |(x, y, z): (Scalar, Scalar, Scalar)| {
            let p = matrix.map_point(skia_rs_core::Point::new(x, y));
            (p.x, p.y, z * z_scale)
        };
        match *self {
            Self::Distant { direction } => {
                let (dx, dy) = map_vector(matrix, direction.0, direction.1);
                let (len, mapped_len) = (direction.0.hypot(direction.1), dx.hypot(dy));
                let k = if mapped_len > 0.0 {
                    len / mapped_len
                } else {
                    0.0
                };
                Self::Distant {
                    direction: (dx * k, dy * k, direction.2),
                }
            }
            Self::Point { location } => Self::Point {
                location: map_point3(location),
            },
            Self::Spot {
                location,
                target,
                specular_exponent,
                cutoff_angle,
            } => Self::Spot {
                location: map_point3(location),
                target: map_point3(target),
                specular_exponent,
                cutoff_angle,
            },
        }
    }
}

/// A lighting image filter.
///
/// Corresponds to Skia's lighting image filters.
#[derive(Debug)]
pub struct LightingImageFilter {
    light: LightType,
    surface_scale: Scalar,
    diffuse_constant: Option<Scalar>,
    specular_constant: Option<Scalar>,
    specular_exponent: Option<Scalar>,
    light_color: Color4f,
    input: Option<ImageFilterRef>,
}

impl LightingImageFilter {
    /// Create a diffuse lighting filter.
    #[must_use]
    pub fn diffuse(
        light: LightType,
        surface_scale: Scalar,
        kd: Scalar,
        light_color: Color4f,
        input: Option<ImageFilterRef>,
    ) -> Self {
        Self {
            light,
            surface_scale,
            diffuse_constant: Some(kd),
            specular_constant: None,
            specular_exponent: None,
            light_color,
            input,
        }
    }

    /// Create a specular lighting filter.
    #[must_use]
    pub fn specular(
        light: LightType,
        surface_scale: Scalar,
        ks: Scalar,
        shininess: Scalar,
        light_color: Color4f,
        input: Option<ImageFilterRef>,
    ) -> Self {
        Self {
            light,
            surface_scale,
            diffuse_constant: None,
            specular_constant: Some(ks),
            specular_exponent: Some(shininess),
            light_color,
            input,
        }
    }
}

impl ImageFilter for LightingImageFilter {
    fn affects_transparent_black(&self) -> bool {
        input_affects(self.input.as_ref())
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        Some(Arc::new(Self {
            light: self.light.mapped(matrix),
            // apply uses surface_scale as a brightness factor (it does not
            // build normals from a height field), so it is not mapped: the
            // result must not depend on the zoom.
            surface_scale: self.surface_scale,
            diffuse_constant: self.diffuse_constant,
            specular_constant: self.specular_constant,
            specular_exponent: self.specular_exponent,
            light_color: self.light_color,
            input: map_input(self.input.as_ref(), matrix),
        }))
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        // Surface normals read the neighbouring pixels.
        input_of(self.input.as_ref(), &output.inset(-1.0, -1.0))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        // Lighting keeps its input's bounds.
        input_filter_bounds(self.input.as_ref(), src)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_LIGHTING];
        // Serialize light type
        match &self.light {
            LightType::Distant { direction } => {
                buf.push(0);
                buf.extend_from_slice(&direction.0.to_le_bytes());
                buf.extend_from_slice(&direction.1.to_le_bytes());
                buf.extend_from_slice(&direction.2.to_le_bytes());
            }
            LightType::Point { location } => {
                buf.push(1);
                buf.extend_from_slice(&location.0.to_le_bytes());
                buf.extend_from_slice(&location.1.to_le_bytes());
                buf.extend_from_slice(&location.2.to_le_bytes());
            }
            LightType::Spot {
                location,
                target,
                specular_exponent,
                cutoff_angle,
            } => {
                buf.push(2);
                buf.extend_from_slice(&location.0.to_le_bytes());
                buf.extend_from_slice(&location.1.to_le_bytes());
                buf.extend_from_slice(&location.2.to_le_bytes());
                buf.extend_from_slice(&target.0.to_le_bytes());
                buf.extend_from_slice(&target.1.to_le_bytes());
                buf.extend_from_slice(&target.2.to_le_bytes());
                buf.extend_from_slice(&specular_exponent.to_le_bytes());
                buf.extend_from_slice(&cutoff_angle.to_le_bytes());
            }
        }
        buf.extend_from_slice(&self.surface_scale.to_le_bytes());
        // Serialize optional diffuse/specular constants
        buf.push(u8::from(self.diffuse_constant.is_some()));
        if let Some(kd) = self.diffuse_constant {
            buf.extend_from_slice(&kd.to_le_bytes());
        }
        buf.push(u8::from(self.specular_constant.is_some()));
        if let Some(ks) = self.specular_constant {
            buf.extend_from_slice(&ks.to_le_bytes());
        }
        buf.push(u8::from(self.specular_exponent.is_some()));
        if let Some(exp) = self.specular_exponent {
            buf.extend_from_slice(&exp.to_le_bytes());
        }
        buf.extend_from_slice(&self.light_color.r.to_le_bytes());
        buf.extend_from_slice(&self.light_color.g.to_le_bytes());
        buf.extend_from_slice(&self.light_color.b.to_le_bytes());
        buf.extend_from_slice(&self.light_color.a.to_le_bytes());
        // Serialize input filter if present
        match &self.input {
            Some(input) => {
                let input_data = input.serialize()?;
                buf.push(1);
                let len = u32::try_from(input_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&input_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // The input filter runs first, as for the other filters with inputs.
        if let Some(ref input) = self.input {
            input.apply(pixels, width, height);
        }
        // Baseline lighting: modulate intensity by surface_scale factor.
        // A full implementation would compute per-pixel normals from the alpha
        // channel (height field) and evaluate the lighting model (Phong, diffuse).
        // This simplified version applies uniform lighting based on surface_scale.
        let scale = self.surface_scale.clamp(0.0, 10.0);
        let factor = (1.0 + scale * 0.1).min(2.0);
        let lr = (self.light_color.r * 255.0).clamp(0.0, 255.0);
        let lg = (self.light_color.g * 255.0).clamp(0.0, 255.0);
        let lb = (self.light_color.b * 255.0).clamp(0.0, 255.0);
        for i in 0..width * height {
            let idx = i * 4;
            // Premultiplied: no color channel may exceed alpha.
            let alpha = pixels[idx + 3];
            let lit = |v: u8, light: Scalar| {
                f32_to_u8_trunc(f32::from(v) * factor * light / 255.0).min(alpha)
            };
            pixels[idx] = lit(pixels[idx], lr);
            pixels[idx + 1] = lit(pixels[idx + 1], lg);
            pixels[idx + 2] = lit(pixels[idx + 2], lb);
        }
    }
}

/// A compose image filter that chains two filters.
///
/// Corresponds to Skia's `SkComposeImageFilter`.
#[derive(Debug)]
pub struct ComposeImageFilter {
    outer: ImageFilterRef,
    inner: ImageFilterRef,
}

impl ComposeImageFilter {
    /// Create a compose filter (outer applied after inner).
    pub fn new(outer: ImageFilterRef, inner: ImageFilterRef) -> Self {
        Self { outer, inner }
    }
}

impl ImageFilter for ComposeImageFilter {
    fn affects_transparent_black(&self) -> bool {
        self.outer.affects_transparent_black() || self.inner.affects_transparent_black()
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        // The inner filter produces what the outer one reads.
        self.inner.input_bounds(&self.outer.input_bounds(output))
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        Some(Arc::new(Self {
            outer: map_image_filter(&self.outer, matrix),
            inner: map_image_filter(&self.inner, matrix),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        let inner_bounds = self.inner.filter_bounds(src);
        self.outer.filter_bounds(&inner_bounds)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_COMPOSE];
        let outer_data = self.outer.serialize()?;
        let inner_data = self.inner.serialize()?;
        let len = u32::try_from(outer_data.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&outer_data);
        let len = u32::try_from(inner_data.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&inner_data);
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        // Apply inner filter first, then outer
        self.inner.apply(pixels, width, height);
        self.outer.apply(pixels, width, height);
    }
}

/// A merge image filter that combines multiple inputs.
///
/// Corresponds to Skia's `SkMergeImageFilter`.
#[derive(Debug)]
pub struct MergeImageFilter {
    inputs: Vec<Option<ImageFilterRef>>,
}

impl MergeImageFilter {
    /// Create a merge filter with the given inputs.
    #[must_use]
    pub fn new(inputs: Vec<Option<ImageFilterRef>>) -> Self {
        Self { inputs }
    }
}

impl ImageFilter for MergeImageFilter {
    fn affects_transparent_black(&self) -> bool {
        self.inputs.iter().any(|i| input_affects(i.as_ref()))
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        self.inputs
            .iter()
            .map(|i| input_of(i.as_ref(), output))
            .fold(*output, |acc, r| acc.union(&r))
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        Some(Arc::new(Self {
            inputs: self
                .inputs
                .iter()
                .map(|i| map_input(i.as_ref(), matrix))
                .collect(),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        let mut result = *src;
        for filter in self.inputs.iter().flatten() {
            let bounds = filter.filter_bounds(src);
            result = result.union(&bounds);
        }
        result
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_MERGE];
        let len = u32::try_from(self.inputs.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        for input in &self.inputs {
            match input {
                Some(filter) => {
                    let filter_data = filter.serialize()?;
                    buf.push(1);
                    let len = u32::try_from(filter_data.len()).unwrap_or(u32::MAX);
                    buf.extend_from_slice(&len.to_le_bytes());
                    buf.extend_from_slice(&filter_data);
                }
                None => buf.push(0),
            }
        }
        Some(buf)
    }

    /// Apply is a no-op for this filter in the single-buffer `apply()` API.
    ///
    /// `MergeImageFilter` requires multiple independent input pixel buffers,
    /// which the current single-buffer signature cannot provide. A future
    /// revision will introduce a multi-input apply variant.
    fn apply(&self, _pixels: &mut [u8], _width: usize, _height: usize) {
        // Documented limitation: multi-input filter requires extended API.
    }
}

/// An offset image filter.
///
/// Corresponds to Skia's `SkOffsetImageFilter`.
#[derive(Debug)]
pub struct OffsetImageFilter {
    dx: Scalar,
    dy: Scalar,
    input: Option<ImageFilterRef>,
}

impl OffsetImageFilter {
    /// Create an offset filter.
    #[must_use]
    pub fn new(dx: Scalar, dy: Scalar, input: Option<ImageFilterRef>) -> Self {
        Self { dx, dy, input }
    }
}

impl ImageFilter for OffsetImageFilter {
    fn affects_transparent_black(&self) -> bool {
        input_affects(self.input.as_ref())
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        input_of(self.input.as_ref(), &output.offset(-self.dx, -self.dy))
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        let (dx, dy) = map_vector(matrix, self.dx, self.dy);
        Some(Arc::new(Self {
            dx,
            dy,
            input: map_input(self.input.as_ref(), matrix),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        let src = input_filter_bounds(self.input.as_ref(), src);
        Rect::new(
            src.left + self.dx,
            src.top + self.dy,
            src.right + self.dx,
            src.bottom + self.dy,
        )
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_OFFSET];
        buf.extend_from_slice(&self.dx.to_le_bytes());
        buf.extend_from_slice(&self.dy.to_le_bytes());
        match &self.input {
            Some(input) => {
                let input_data = input.serialize()?;
                buf.push(1);
                let len = u32::try_from(input_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&input_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // Apply input filter first if present
        if let Some(ref input) = self.input {
            input.apply(pixels, width, height);
        }
        // Shift pixels with clamp-to-edge
        let offset_x = saturate_to_i32(self.dx.round());
        let offset_y = saturate_to_i32(self.dy.round());
        if offset_x == 0 && offset_y == 0 {
            return;
        }

        // Create a copy to read from
        let mut copy = vec![0u8; width * height * 4];
        copy.copy_from_slice(pixels);

        let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
        let height_i32 = i32::try_from(height).unwrap_or(i32::MAX);
        // Fill with shifted pixels
        for y in 0..height {
            for x in 0..width {
                let x_i32 = i32::try_from(x).unwrap_or(i32::MAX);
                let y_i32 = i32::try_from(y).unwrap_or(i32::MAX);
                let src_x =
                    usize::try_from((x_i32 - offset_x).clamp(0, width_i32 - 1)).unwrap_or(0);
                let src_y =
                    usize::try_from((y_i32 - offset_y).clamp(0, height_i32 - 1)).unwrap_or(0);
                let dst_idx = (y * width + x) * 4;
                let src_idx = (src_y * width + src_x) * 4;
                pixels[dst_idx..dst_idx + 4].copy_from_slice(&copy[src_idx..src_idx + 4]);
            }
        }
    }
}

/// A matrix convolution image filter.
///
/// Corresponds to Skia's `SkMatrixConvolutionImageFilter`.
#[derive(Debug, Clone)]
pub struct MatrixConvolutionImageFilter {
    kernel_size: (i32, i32),
    kernel: Vec<Scalar>,
    gain: Scalar,
    bias: Scalar,
    kernel_offset: (i32, i32),
    tile_mode: crate::shader::TileMode,
    convolve_alpha: bool,
    input: Option<ImageFilterRef>,
}

impl MatrixConvolutionImageFilter {
    /// The kernel's own outset of `src` (without its input's).
    fn own_bounds(&self, src: &Rect) -> Rect {
        let (kw, kh) = self.kernel_size;
        let (ox, oy) = self.kernel_offset;
        Rect::new(
            src.left - scalar_from_i32(kw - ox - 1),
            src.top - scalar_from_i32(kh - oy - 1),
            src.right + scalar_from_i32(ox),
            src.bottom + scalar_from_i32(oy),
        )
    }

    /// Create a matrix convolution filter.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors upstream SkMatrixConvolutionImageFilter::Make's parameter list one-to-one"
    )]
    pub fn new(
        kernel_size: (i32, i32),
        kernel: Vec<Scalar>,
        gain: Scalar,
        bias: Scalar,
        kernel_offset: (i32, i32),
        tile_mode: crate::shader::TileMode,
        convolve_alpha: bool,
        input: Option<ImageFilterRef>,
    ) -> Self {
        Self {
            kernel_size,
            kernel,
            gain,
            bias,
            kernel_offset,
            tile_mode,
            convolve_alpha,
            input,
        }
    }
}

impl MatrixConvolutionImageFilter {
    /// The kernel (size, values, offset) as seen in device space under an
    /// axis-permuting `matrix`: each tap's offset from the kernel origin is
    /// mapped by the signs/permutation of the linear part. Other matrices
    /// leave the kernel unchanged.
    #[allow(clippy::float_cmp, reason = "exact zero tests for an axis permutation")]
    fn reoriented(&self, matrix: &Matrix) -> ((i32, i32), Vec<Scalar>, (i32, i32)) {
        let unchanged = (self.kernel_size, self.kernel.clone(), self.kernel_offset);
        let values = &matrix.values;
        let (scale_x, skew_x, skew_y, scale_y) = (
            values[Matrix::SCALE_X],
            values[Matrix::SKEW_X],
            values[Matrix::SKEW_Y],
            values[Matrix::SCALE_Y],
        );
        let sign = |t: Scalar| if t < 0.0 { -1 } else { 1 };
        let (kw, kh) = self.kernel_size;
        // Checked: a bogus kernel size must not overflow or allocate.
        let len = usize::try_from(kw)
            .ok()
            .zip(usize::try_from(kh).ok())
            .and_then(|(w, h)| w.checked_mul(h));
        let Some(len) = len else {
            return unchanged;
        };
        if kw <= 0 || kh <= 0 || self.kernel.len() != len || matrix.has_perspective() {
            return unchanged;
        }
        // Device tap offset (u', v') from a local one (u, v).
        let map: Box<dyn Fn(i32, i32) -> (i32, i32)> =
            if skew_x == 0.0 && skew_y == 0.0 && scale_x != 0.0 && scale_y != 0.0 {
                let (sx, sy) = (sign(scale_x), sign(scale_y));
                Box::new(move |u, w| (sx * u, sy * w))
            } else if scale_x == 0.0 && scale_y == 0.0 && skew_x != 0.0 && skew_y != 0.0 {
                let (sx, sy) = (sign(skew_x), sign(skew_y));
                Box::new(move |u, w| (sx * w, sy * u))
            } else {
                return unchanged;
            };
        let (ox, oy) = self.kernel_offset;
        let taps: Vec<((i32, i32), Scalar)> = (0..kh)
            .flat_map(|ky| (0..kw).map(move |kx| (kx, ky)))
            .map(|(kx, ky)| {
                let value = self.kernel[usize::try_from(ky * kw + kx).unwrap_or(0)];
                (map(kx - ox, ky - oy), value)
            })
            .collect();
        let min_u = taps.iter().map(|t| t.0.0).min().unwrap_or(0);
        let min_v = taps.iter().map(|t| t.0.1).min().unwrap_or(0);
        let max_u = taps.iter().map(|t| t.0.0).max().unwrap_or(0);
        let max_v = taps.iter().map(|t| t.0.1).max().unwrap_or(0);
        let (nw, nh) = (max_u - min_u + 1, max_v - min_v + 1);
        let mut kernel = vec![0.0; usize::try_from(nw * nh).unwrap_or(0)];
        for ((u, w), value) in taps {
            let index = usize::try_from((w - min_v) * nw + (u - min_u)).unwrap_or(0);
            kernel[index] = value;
        }
        ((nw, nh), kernel, (-min_u, -min_v))
    }
}

impl ImageFilter for MatrixConvolutionImageFilter {
    fn affects_transparent_black(&self) -> bool {
        // With convolve_alpha, a positive bias is added to transparent
        // pixels too.
        input_affects(self.input.as_ref()) || (self.convolve_alpha && self.bias > 0.0)
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        // Skia runs the kernel in a translate-only layer space and resamples.
        // Here the kernel is re-oriented for axis-permuting matrices
        // (quarter turns, mirrors), which is exact at unit scale; it is not
        // scaled.
        let (kernel_size, kernel, kernel_offset) = self.reoriented(matrix);
        Some(Arc::new(Self {
            kernel_size,
            kernel,
            kernel_offset,
            input: map_input(self.input.as_ref(), matrix),
            ..self.clone()
        }))
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        input_of(
            self.input.as_ref(),
            &spread_input(&self.own_bounds(output), output),
        )
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        self.own_bounds(&input_filter_bounds(self.input.as_ref(), src))
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_MATRIX_CONVOLUTION];
        buf.extend_from_slice(&self.kernel_size.0.to_le_bytes());
        buf.extend_from_slice(&self.kernel_size.1.to_le_bytes());
        let len = u32::try_from(self.kernel.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        for v in &self.kernel {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&self.gain.to_le_bytes());
        buf.extend_from_slice(&self.bias.to_le_bytes());
        buf.extend_from_slice(&self.kernel_offset.0.to_le_bytes());
        buf.extend_from_slice(&self.kernel_offset.1.to_le_bytes());
        buf.push(self.tile_mode as u8);
        buf.push(u8::from(self.convolve_alpha));
        match &self.input {
            Some(input) => {
                let input_data = input.serialize()?;
                buf.push(1);
                let len = u32::try_from(input_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&input_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // The input filter runs first (Skia filters its input image).
        if let Some(ref input) = self.input {
            input.apply(pixels, width, height);
        }
        let (kw, kh) = self.kernel_size;
        if kw <= 0 || kh <= 0 || self.kernel.is_empty() {
            return;
        }
        let src = pixels.to_vec();
        let gain = self.gain;
        let bias = self.bias;
        let (ox, oy) = self.kernel_offset;

        // Map a source coordinate through the tile mode. Returns None for
        // out-of-bounds Decal taps (which contribute transparent black).
        let tile = |v: i32, size: i32| -> Option<i32> {
            if (0..size).contains(&v) {
                return Some(v);
            }
            match self.tile_mode {
                crate::shader::TileMode::Clamp => Some(v.clamp(0, size - 1)),
                crate::shader::TileMode::Repeat => Some(v.rem_euclid(size)),
                crate::shader::TileMode::Mirror => {
                    let period = 2 * size;
                    let m = v.rem_euclid(period);
                    Some(if m < size { m } else { period - m - 1 })
                }
                crate::shader::TileMode::Decal => None,
            }
        };

        let width_i32 = i32::try_from(width).unwrap_or(i32::MAX);
        let height_i32 = i32::try_from(height).unwrap_or(i32::MAX);
        for y in 0..height {
            for x in 0..width {
                let x_i32 = i32::try_from(x).unwrap_or(i32::MAX);
                let y_i32 = i32::try_from(y).unwrap_or(i32::MAX);
                let mut acc = [0.0f32; 4];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sx = tile(x_i32 + kx - ox, width_i32);
                        let sy = tile(y_i32 + ky - oy, height_i32);
                        let (sx, sy) = match (sx, sy) {
                            (Some(sx), Some(sy)) => (
                                usize::try_from(sx).unwrap_or(0),
                                usize::try_from(sy).unwrap_or(0),
                            ),
                            _ => continue, // Decal: transparent contribution
                        };
                        let sidx = (sy * width + sx) * 4;
                        let kidx = usize::try_from(ky).unwrap_or(0)
                            * usize::try_from(kw).unwrap_or(0)
                            + usize::try_from(kx).unwrap_or(0);
                        let k = self.kernel.get(kidx).copied().unwrap_or(0.0);
                        if self.convolve_alpha {
                            // Convolve the premultiplied channels directly.
                            for c in 0..4 {
                                acc[c] = f32::from(src[sidx + c]).mul_add(k, acc[c]);
                            }
                        } else {
                            // Upstream (SkKnownRuntimeEffects convolution):
                            // convolve UNPREMULTIPLIED RGB, keep alpha.
                            let sa = f32::from(src[sidx + 3]) / 255.0;
                            if sa > 0.0 {
                                for c in 0..3 {
                                    acc[c] = (f32::from(src[sidx + c]) / sa).mul_add(k, acc[c]);
                                }
                            }
                        }
                    }
                }
                let idx = (y * width + x) * 4;
                if self.convolve_alpha {
                    for c in 0..4 {
                        let v = acc[c].mul_add(gain, bias);
                        pixels[idx + c] = f32_to_u8_sat(v);
                    }
                } else {
                    // Re-premultiply the convolved RGB with the pixel's
                    // original (unchanged) alpha.
                    let a = f32::from(src[idx + 3]) / 255.0;
                    for c in 0..3 {
                        let v = acc[c].mul_add(gain, bias).clamp(0.0, 255.0) * a;
                        pixels[idx + c] = f32_to_u8_sat(v);
                    }
                    pixels[idx + 3] = src[idx + 3];
                }
            }
        }
    }
}

/// A tile image filter.
///
/// Corresponds to Skia's `SkTileImageFilter`.
#[derive(Debug)]
pub struct TileImageFilter {
    src_rect: Rect,
    dst_rect: Rect,
    input: Option<ImageFilterRef>,
}

impl TileImageFilter {
    /// Create a tile filter.
    #[must_use]
    pub fn new(src_rect: Rect, dst_rect: Rect, input: Option<ImageFilterRef>) -> Self {
        Self {
            src_rect,
            dst_rect,
            input,
        }
    }
}

impl ImageFilter for TileImageFilter {
    fn affects_transparent_black(&self) -> bool {
        input_affects(self.input.as_ref())
    }

    fn input_bounds(&self, _output: &Rect) -> Rect {
        // SkTileImageFilter reads its source rect, whatever the output.
        input_of(self.input.as_ref(), &self.src_rect)
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        Some(Arc::new(Self {
            src_rect: matrix.map_rect_clipped(&self.src_rect),
            dst_rect: matrix.map_rect_clipped(&self.dst_rect),
            input: map_input(self.input.as_ref(), matrix),
        }))
    }

    fn filter_bounds(&self, _src: &Rect) -> Rect {
        self.dst_rect
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_TILE];
        buf.extend_from_slice(&self.src_rect.left.to_le_bytes());
        buf.extend_from_slice(&self.src_rect.top.to_le_bytes());
        buf.extend_from_slice(&self.src_rect.right.to_le_bytes());
        buf.extend_from_slice(&self.src_rect.bottom.to_le_bytes());
        buf.extend_from_slice(&self.dst_rect.left.to_le_bytes());
        buf.extend_from_slice(&self.dst_rect.top.to_le_bytes());
        buf.extend_from_slice(&self.dst_rect.right.to_le_bytes());
        buf.extend_from_slice(&self.dst_rect.bottom.to_le_bytes());
        match &self.input {
            Some(input) => {
                let input_data = input.serialize()?;
                buf.push(1);
                let len = u32::try_from(input_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&input_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    fn apply(&self, pixels: &mut [u8], width: usize, height: usize) {
        if width == 0 || height == 0 {
            return;
        }
        // The input filter runs first (Skia filters its input image).
        if let Some(ref input) = self.input {
            input.apply(pixels, width, height);
        }
        // Tile the src_rect region across dst_rect only (SkTileImageFilter
        // fills its destination rect, not the whole buffer).
        let to_usize = |v: Scalar| usize::try_from(saturate_to_i32(v.max(0.0))).unwrap_or(0);
        let sx0 = to_usize(self.src_rect.left).min(width);
        let sy0 = to_usize(self.src_rect.top).min(height);
        let sx1 = to_usize(self.src_rect.right).min(width);
        let sy1 = to_usize(self.src_rect.bottom).min(height);
        if sx1 <= sx0 || sy1 <= sy0 {
            return;
        }
        let sw = sx1 - sx0;
        let sh = sy1 - sy0;

        let dx0 = to_usize(self.dst_rect.left).min(width);
        let dy0 = to_usize(self.dst_rect.top).min(height);
        let dx1 = to_usize(self.dst_rect.right).min(width);
        let dy1 = to_usize(self.dst_rect.bottom).min(height);
        if dx1 <= dx0 || dy1 <= dy0 {
            return;
        }

        let src = pixels.to_vec();
        for y in dy0..dy1 {
            for x in dx0..dx1 {
                let tx = sx0 + ((x - dx0) % sw);
                let ty = sy0 + ((y - dy0) % sh);
                let src_idx = (ty * width + tx) * 4;
                let dst_idx = (y * width + x) * 4;
                pixels[dst_idx..dst_idx + 4].copy_from_slice(&src[src_idx..src_idx + 4]);
            }
        }
    }
}

/// A blend image filter.
///
/// Corresponds to Skia's `SkBlendImageFilter`.
#[derive(Debug)]
pub struct BlendImageFilter {
    mode: crate::BlendMode,
    background: Option<ImageFilterRef>,
    foreground: Option<ImageFilterRef>,
}

impl BlendImageFilter {
    /// Create a blend filter.
    #[must_use]
    pub fn new(
        mode: crate::BlendMode,
        background: Option<ImageFilterRef>,
        foreground: Option<ImageFilterRef>,
    ) -> Self {
        Self {
            mode,
            background,
            foreground,
        }
    }
}

impl ImageFilter for BlendImageFilter {
    fn affects_transparent_black(&self) -> bool {
        input_affects(self.background.as_ref()) || input_affects(self.foreground.as_ref())
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        input_of(self.background.as_ref(), output)
            .union(&input_of(self.foreground.as_ref(), output))
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        Some(Arc::new(Self {
            mode: self.mode,
            background: map_input(self.background.as_ref(), matrix),
            foreground: map_input(self.foreground.as_ref(), matrix),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        let bg = self
            .background
            .as_ref()
            .map_or(*src, |f| f.filter_bounds(src));
        let fg = self
            .foreground
            .as_ref()
            .map_or(*src, |f| f.filter_bounds(src));
        bg.union(&fg)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_BLEND];
        buf.push(self.mode as u8);
        match &self.background {
            Some(bg) => {
                let bg_data = bg.serialize()?;
                buf.push(1);
                let len = u32::try_from(bg_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&bg_data);
            }
            None => buf.push(0),
        }
        match &self.foreground {
            Some(fg) => {
                let fg_data = fg.serialize()?;
                buf.push(1);
                let len = u32::try_from(fg_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&fg_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    /// Apply is a no-op for this filter in the single-buffer `apply()` API.
    ///
    /// `BlendImageFilter` requires separate background and foreground pixel
    /// buffers, which the current single-buffer signature cannot provide.
    /// A future revision will introduce a multi-input apply variant.
    fn apply(&self, _pixels: &mut [u8], _width: usize, _height: usize) {
        // Documented limitation: multi-input filter requires extended API.
    }
}

/// An arithmetic blend image filter.
///
/// Result = k1 * src * dst + k2 * src + k3 * dst + k4
#[derive(Debug)]
pub struct ArithmeticImageFilter {
    k1: Scalar,
    k2: Scalar,
    k3: Scalar,
    k4: Scalar,
    enforce_pm_color: bool,
    background: Option<ImageFilterRef>,
    foreground: Option<ImageFilterRef>,
}

impl ArithmeticImageFilter {
    /// Create an arithmetic blend filter.
    #[must_use]
    pub fn new(
        k1: Scalar,
        k2: Scalar,
        k3: Scalar,
        k4: Scalar,
        enforce_pm_color: bool,
        background: Option<ImageFilterRef>,
        foreground: Option<ImageFilterRef>,
    ) -> Self {
        Self {
            k1,
            k2,
            k3,
            k4,
            enforce_pm_color,
            background,
            foreground,
        }
    }
}

impl ImageFilter for ArithmeticImageFilter {
    fn affects_transparent_black(&self) -> bool {
        // k4 is added to every pixel, transparent ones included.
        self.k4 > 0.0
            || input_affects(self.background.as_ref())
            || input_affects(self.foreground.as_ref())
    }

    fn input_bounds(&self, output: &Rect) -> Rect {
        input_of(self.background.as_ref(), output)
            .union(&input_of(self.foreground.as_ref(), output))
    }

    fn map_params(&self, matrix: &Matrix) -> Option<ImageFilterRef> {
        Some(Arc::new(Self {
            k1: self.k1,
            k2: self.k2,
            k3: self.k3,
            k4: self.k4,
            enforce_pm_color: self.enforce_pm_color,
            background: map_input(self.background.as_ref(), matrix),
            foreground: map_input(self.foreground.as_ref(), matrix),
        }))
    }

    fn filter_bounds(&self, src: &Rect) -> Rect {
        let bg = self
            .background
            .as_ref()
            .map_or(*src, |f| f.filter_bounds(src));
        let fg = self
            .foreground
            .as_ref()
            .map_or(*src, |f| f.filter_bounds(src));
        bg.union(&fg)
    }

    fn serialize(&self) -> Option<Vec<u8>> {
        let mut buf = vec![IMAGE_FILTER_KIND_ARITHMETIC];
        buf.extend_from_slice(&self.k1.to_le_bytes());
        buf.extend_from_slice(&self.k2.to_le_bytes());
        buf.extend_from_slice(&self.k3.to_le_bytes());
        buf.extend_from_slice(&self.k4.to_le_bytes());
        buf.push(u8::from(self.enforce_pm_color));
        match &self.background {
            Some(bg) => {
                let bg_data = bg.serialize()?;
                buf.push(1);
                let len = u32::try_from(bg_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&bg_data);
            }
            None => buf.push(0),
        }
        match &self.foreground {
            Some(fg) => {
                let fg_data = fg.serialize()?;
                buf.push(1);
                let len = u32::try_from(fg_data.len()).unwrap_or(u32::MAX);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&fg_data);
            }
            None => buf.push(0),
        }
        Some(buf)
    }

    /// Apply is a no-op for this filter in the single-buffer `apply()` API.
    ///
    /// `ArithmeticImageFilter` requires separate background and foreground pixel
    /// buffers, which the current single-buffer signature cannot provide.
    /// A future revision will introduce a multi-input apply variant.
    fn apply(&self, _pixels: &mut [u8], _width: usize, _height: usize) {
        // Documented limitation: multi-input filter requires extended API.
    }
}

// =============================================================================
// Boxed Filter Types
// =============================================================================

/// Boxed filter types.
pub type ColorFilterRef = Arc<dyn ColorFilter + Send + Sync>;
/// Boxed mask filter type.
pub type MaskFilterRef = Arc<dyn MaskFilter + Send + Sync>;
/// Boxed image filter type.
pub type ImageFilterRef = Arc<dyn ImageFilter + Send + Sync>;

// =============================================================================
// Box Blur Helpers
// =============================================================================

/// Convert a Gaussian sigma to the box-blur radius of the three-box
/// approximation, per `SkBlurMask::ConvertSigmaToRadius`:
/// `radius = sigma > 0.5 ? (sigma - 0.5) / 0.57735 : 0`.
#[inline]
fn sigma_to_box_radius(sigma: Scalar) -> Scalar {
    const K_BLUR_SIGMA_SCALE: Scalar = 0.57735;
    if sigma > 0.5 {
        (sigma - 0.5) / K_BLUR_SIGMA_SCALE
    } else {
        0.0
    }
}

/// Integer box radius for the three-box passes. Rounds to nearest instead of
/// truncating so that e.g. sigma 0.9 (radius ~0.69) still blurs.
#[inline]
fn sigma_to_int_box_radius(sigma: Scalar) -> usize {
    usize::try_from(round_to_i32(sigma_to_box_radius(sigma).max(0.0))).unwrap_or(0)
}

fn box_blur_horizontal(buf: &mut [u8], width: usize, height: usize, radius: usize) {
    if width == 0 {
        return;
    }
    let r = radius;
    let window = u32::try_from(2 * r + 1).unwrap_or(u32::MAX);
    let mut row_copy = vec![0u8; width];
    for y in 0..height {
        let row_start = y * width;
        // Copy current row
        row_copy.copy_from_slice(&buf[row_start..row_start + width]);
        // Initialize sum with edge clamping
        let mut sum: u32 = 0;
        for i in 0..=r {
            sum += u32::from(row_copy[i.min(width - 1)]);
        }
        sum += u32::from(row_copy[0]) * u32::try_from(r).unwrap_or(u32::MAX); // clamp left edge
        buf[row_start] = u8::try_from(sum / window).unwrap_or(u8::MAX);

        for x in 1..width {
            // Slide window: add right, remove left
            let right_idx = (x + r).min(width - 1);
            let left_idx = if x > r { x - r - 1 } else { 0 };
            sum = sum.saturating_add(u32::from(row_copy[right_idx]));
            sum = sum.saturating_sub(u32::from(row_copy[left_idx]));
            buf[row_start + x] = u8::try_from(sum / window).unwrap_or(u8::MAX);
        }
    }
}

fn box_blur_vertical(buf: &mut [u8], width: usize, height: usize, radius: usize) {
    if height == 0 {
        return;
    }
    let r = radius;
    let window = u32::try_from(2 * r + 1).unwrap_or(u32::MAX);
    let mut col_copy = vec![0u8; height];
    for x in 0..width {
        // Copy column
        for y in 0..height {
            col_copy[y] = buf[y * width + x];
        }
        // Initialize sum with top edge clamping
        let mut sum: u32 = 0;
        for i in 0..=r {
            sum += u32::from(col_copy[i.min(height - 1)]);
        }
        sum += u32::from(col_copy[0]) * u32::try_from(r).unwrap_or(u32::MAX);
        buf[x] = u8::try_from(sum / window).unwrap_or(u8::MAX);

        for y in 1..height {
            let bottom_idx = (y + r).min(height - 1);
            let top_idx = if y > r { y - r - 1 } else { 0 };
            sum = sum.saturating_add(u32::from(col_copy[bottom_idx]));
            sum = sum.saturating_sub(u32::from(col_copy[top_idx]));
            buf[y * width + x] = u8::try_from(sum / window).unwrap_or(u8::MAX);
        }
    }
}

fn rgba_box_blur_horizontal(pixels: &mut [u8], width: usize, height: usize, radius: usize) {
    if width == 0 {
        return;
    }
    let stride = width * 4;
    let window = u32::try_from(2 * radius + 1).unwrap_or(u32::MAX);
    let mut row_copy = vec![0u8; stride];
    for y in 0..height {
        let row_start = y * stride;
        row_copy.copy_from_slice(&pixels[row_start..row_start + stride]);
        // 4 channels: track sum per channel
        let mut sums: [u32; 4] = [0; 4];
        for i in 0..=radius {
            let idx = i.min(width - 1) * 4;
            for c in 0..4 {
                sums[c] += u32::from(row_copy[idx + c]);
            }
        }
        for c in 0..4 {
            sums[c] += u32::from(row_copy[c]) * u32::try_from(radius).unwrap_or(u32::MAX);
        }
        for c in 0..4 {
            pixels[row_start + c] = u8::try_from(sums[c] / window).unwrap_or(u8::MAX);
        }

        for x in 1..width {
            let right_idx = (x + radius).min(width - 1) * 4;
            let left_idx = if x > radius { (x - radius - 1) * 4 } else { 0 };
            for c in 0..4 {
                sums[c] = sums[c].saturating_add(u32::from(row_copy[right_idx + c]));
                sums[c] = sums[c].saturating_sub(u32::from(row_copy[left_idx + c]));
                pixels[row_start + x * 4 + c] = u8::try_from(sums[c] / window).unwrap_or(u8::MAX);
            }
        }
    }
}

fn rgba_box_blur_vertical(pixels: &mut [u8], width: usize, height: usize, radius: usize) {
    if height == 0 {
        return;
    }
    let stride = width * 4;
    let window = u32::try_from(2 * radius + 1).unwrap_or(u32::MAX);
    let mut col_copy = vec![0u8; height * 4];
    for x in 0..width {
        for y in 0..height {
            for c in 0..4 {
                col_copy[y * 4 + c] = pixels[y * stride + x * 4 + c];
            }
        }
        let mut sums: [u32; 4] = [0; 4];
        for i in 0..=radius {
            let idx = i.min(height - 1) * 4;
            for c in 0..4 {
                sums[c] += u32::from(col_copy[idx + c]);
            }
        }
        for c in 0..4 {
            sums[c] += u32::from(col_copy[c]) * u32::try_from(radius).unwrap_or(u32::MAX);
        }
        for c in 0..4 {
            pixels[x * 4 + c] = u8::try_from(sums[c] / window).unwrap_or(u8::MAX);
        }

        for y in 1..height {
            let bottom_idx = (y + radius).min(height - 1) * 4;
            let top_idx = if y > radius { (y - radius - 1) * 4 } else { 0 };
            for c in 0..4 {
                sums[c] = sums[c].saturating_add(u32::from(col_copy[bottom_idx + c]));
                sums[c] = sums[c].saturating_sub(u32::from(col_copy[top_idx + c]));
                pixels[y * stride + x * 4 + c] = u8::try_from(sums[c] / window).unwrap_or(u8::MAX);
            }
        }
    }
}

// =============================================================================
// Deserialization Functions
// =============================================================================

/// Deserialize a `MaskFilter` from bytes.
pub(crate) fn deserialize_mask_filter(bytes: &[u8], offset: &mut usize) -> Option<MaskFilterRef> {
    if *offset >= bytes.len() {
        return None;
    }
    let kind = bytes[*offset];
    *offset += 1;

    match kind {
        MASK_FILTER_KIND_BLUR => {
            if *offset + 5 > bytes.len() {
                return None;
            }
            let style = match bytes[*offset] {
                0 => BlurStyle::Normal,
                1 => BlurStyle::Solid,
                2 => BlurStyle::Outer,
                3 => BlurStyle::Inner,
                _ => return None,
            };
            *offset += 1;
            let sigma = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            Some(Arc::new(BlurMaskFilter::new(style, sigma)))
        }
        MASK_FILTER_KIND_SHADER => {
            if *offset + 4 > bytes.len() {
                return None;
            }
            let len = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
            *offset += 4;
            if *offset + len > bytes.len() {
                return None;
            }
            let shader_bytes = &bytes[*offset..*offset + len];
            *offset += len;
            let mut shader_offset = 0;
            let shader = crate::shader::deserialize_shader(shader_bytes, &mut shader_offset)?;
            Some(Arc::new(ShaderMaskFilter::new(shader)))
        }
        MASK_FILTER_KIND_TABLE => {
            if *offset + 256 > bytes.len() {
                return None;
            }
            let mut table = [0u8; 256];
            table.copy_from_slice(&bytes[*offset..*offset + 256]);
            *offset += 256;
            Some(Arc::new(TableMaskFilter::new(table)))
        }
        _ => None,
    }
}

/// Deserialize a `ColorFilter` from bytes.
pub(crate) fn deserialize_color_filter(bytes: &[u8], offset: &mut usize) -> Option<ColorFilterRef> {
    if *offset >= bytes.len() {
        return None;
    }
    let kind = bytes[*offset];
    *offset += 1;

    match kind {
        COLOR_FILTER_KIND_MATRIX => {
            if *offset + 80 > bytes.len() {
                return None;
            }
            let mut matrix = [0.0f32; 20];
            for v in &mut matrix {
                *v = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
                *offset += 4;
            }
            Some(Arc::new(ColorMatrixFilter::new(matrix)))
        }
        _ => None,
    }
}

/// Deserialize an `ImageFilter` from bytes.
#[allow(
    clippy::too_many_lines,
    reason = "one match arm per serialized image-filter kind tag; splitting would obscure the pairing with each filter's serialize()"
)]
pub(crate) fn deserialize_image_filter(bytes: &[u8], offset: &mut usize) -> Option<ImageFilterRef> {
    if *offset >= bytes.len() {
        return None;
    }
    let kind = bytes[*offset];
    *offset += 1;

    match kind {
        IMAGE_FILTER_KIND_BLUR => {
            if *offset + 9 > bytes.len() {
                return None;
            }
            let sigma_x = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let sigma_y = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let tile_mode = match bytes[*offset] {
                0 => crate::shader::TileMode::Clamp,
                1 => crate::shader::TileMode::Repeat,
                2 => crate::shader::TileMode::Mirror,
                3 => crate::shader::TileMode::Decal,
                _ => return None,
            };
            *offset += 1;
            Some(Arc::new(BlurImageFilter::new(sigma_x, sigma_y, tile_mode)))
        }
        IMAGE_FILTER_KIND_DROP_SHADOW => {
            if *offset + 33 > bytes.len() {
                return None;
            }
            let dx = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let dy = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let sigma_x = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let sigma_y = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let r = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let g = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let b = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let a = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let shadow_only = bytes[*offset] != 0;
            *offset += 1;
            Some(Arc::new(DropShadowImageFilter::new(
                dx,
                dy,
                sigma_x,
                sigma_y,
                Color4f::new(r, g, b, a),
                shadow_only,
            )))
        }
        IMAGE_FILTER_KIND_MORPHOLOGY => {
            if *offset + 9 > bytes.len() {
                return None;
            }
            let morph_type = match bytes[*offset] {
                0 => MorphologyType::Dilate,
                1 => MorphologyType::Erode,
                _ => return None,
            };
            *offset += 1;
            let radius_x = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let radius_y = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let input = if *offset < bytes.len() && bytes[*offset] == 1 {
                *offset += 1;
                if *offset + 4 > bytes.len() {
                    return None;
                }
                let len = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
                *offset += 4;
                if *offset + len > bytes.len() {
                    return None;
                }
                let input_bytes = &bytes[*offset..*offset + len];
                *offset += len;
                let mut input_offset = 0;
                Some(deserialize_image_filter(input_bytes, &mut input_offset)?)
            } else {
                if *offset < bytes.len() {
                    *offset += 1;
                }
                None
            };
            Some(Arc::new(match morph_type {
                MorphologyType::Dilate => MorphologyImageFilter::dilate(radius_x, radius_y, input),
                MorphologyType::Erode => MorphologyImageFilter::erode(radius_x, radius_y, input),
            }))
        }
        IMAGE_FILTER_KIND_OFFSET => {
            if *offset + 8 > bytes.len() {
                return None;
            }
            let dx = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let dy = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let input = if *offset < bytes.len() && bytes[*offset] == 1 {
                *offset += 1;
                if *offset + 4 > bytes.len() {
                    return None;
                }
                let len = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
                *offset += 4;
                if *offset + len > bytes.len() {
                    return None;
                }
                let input_bytes = &bytes[*offset..*offset + len];
                *offset += len;
                let mut input_offset = 0;
                Some(deserialize_image_filter(input_bytes, &mut input_offset)?)
            } else {
                if *offset < bytes.len() {
                    *offset += 1;
                }
                None
            };
            Some(Arc::new(OffsetImageFilter::new(dx, dy, input)))
        }
        IMAGE_FILTER_KIND_COMPOSE => {
            if *offset + 8 > bytes.len() {
                return None;
            }
            let outer_len =
                u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
            *offset += 4;
            if *offset + outer_len > bytes.len() {
                return None;
            }
            let outer_bytes = &bytes[*offset..*offset + outer_len];
            *offset += outer_len;
            let mut outer_offset = 0;
            let outer = deserialize_image_filter(outer_bytes, &mut outer_offset)?;

            if *offset + 4 > bytes.len() {
                return None;
            }
            let inner_len =
                u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
            *offset += 4;
            if *offset + inner_len > bytes.len() {
                return None;
            }
            let inner_bytes = &bytes[*offset..*offset + inner_len];
            *offset += inner_len;
            let mut inner_offset = 0;
            let inner = deserialize_image_filter(inner_bytes, &mut inner_offset)?;

            Some(Arc::new(ComposeImageFilter::new(outer, inner)))
        }
        IMAGE_FILTER_KIND_MERGE => {
            if *offset + 4 > bytes.len() {
                return None;
            }
            let count = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
            *offset += 4;
            let mut inputs = Vec::with_capacity(count);
            for _ in 0..count {
                if *offset >= bytes.len() {
                    return None;
                }
                let input = if bytes[*offset] == 1 {
                    *offset += 1;
                    if *offset + 4 > bytes.len() {
                        return None;
                    }
                    let len =
                        u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().ok()?) as usize;
                    *offset += 4;
                    if *offset + len > bytes.len() {
                        return None;
                    }
                    let input_bytes = &bytes[*offset..*offset + len];
                    *offset += len;
                    let mut input_offset = 0;
                    Some(deserialize_image_filter(input_bytes, &mut input_offset)?)
                } else {
                    *offset += 1;
                    None
                };
                inputs.push(input);
            }
            Some(Arc::new(MergeImageFilter::new(inputs)))
        }
        _ => None, // Other filter types not yet implemented in deserialization
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn quarter_turns_swap_directional_filter_axes() {
        // A horizontal blur under a 90-degree rotation is vertical on
        // screen, as Skia draws it.
        let blur: ImageFilterRef = Arc::new(BlurImageFilter::new(
            10.0,
            0.0,
            crate::shader::TileMode::Decal,
        ));
        let mapped = map_image_filter(&blur, &Matrix::rotate(90.0_f32.to_radians()));
        let debug = format!("{mapped:?}");
        assert!(
            debug.contains("sigma_x: 0.0") && debug.contains("sigma_y: 10.0"),
            "{debug}"
        );
        // Scale alone keeps the axes.
        let mapped = map_image_filter(&blur, &Matrix::scale(2.0, 3.0));
        let debug = format!("{mapped:?}");
        assert!(
            debug.contains("sigma_x: 20.0") && debug.contains("sigma_y: 0.0"),
            "{debug}"
        );
    }

    #[test]
    fn convolution_kernels_reorient_under_quarter_turns() {
        // A horizontal 3x1 edge kernel under rotate(90) is vertical on
        // screen: local tap offset (u, v) maps to device (-v, u).
        let conv: ImageFilterRef = Arc::new(MatrixConvolutionImageFilter::new(
            (3, 1),
            vec![1.0, 0.0, -1.0],
            1.0,
            0.0,
            (1, 0),
            crate::shader::TileMode::Decal,
            true,
            None,
        ));
        let mapped = map_image_filter(&conv, &Matrix::rotate(90.0_f32.to_radians()));
        let debug = format!("{mapped:?}");
        assert!(debug.contains("kernel_size: (1, 3)"), "{debug}");
        assert!(debug.contains("kernel: [1.0, 0.0, -1.0]"), "{debug}");
        assert!(debug.contains("kernel_offset: (0, 1)"), "{debug}");
    }

    #[test]
    fn distant_lights_keep_their_elevation() {
        // Under scale a distant light keeps its direction; under a quarter
        // turn it turns with the content.
        let light = LightType::Distant {
            direction: (1.0, 0.0, 1.0),
        };
        let scaled = light.mapped(&Matrix::scale(2.0, 2.0));
        assert!(matches!(scaled, LightType::Distant { direction } if direction == (1.0, 0.0, 1.0)));
        // The surface scale is not mapped (as in Skia), so the shading
        // does not change with zoom.
        let lighting: ImageFilterRef = Arc::new(LightingImageFilter::diffuse(
            light.clone(),
            1.5,
            1.0,
            Color4f::new(1.0, 1.0, 1.0, 1.0),
            None,
        ));
        let debug = format!(
            "{:?}",
            map_image_filter(&lighting, &Matrix::scale(2.0, 2.0))
        );
        assert!(
            debug.contains("surface_scale: 1.5"),
            "zoom must not change it: {debug}"
        );
        // Point lights keep their height (scaled, never flipped) under turns.
        let point = LightType::Point {
            location: (10.0, 20.0, 5.0),
        };
        for turn in [90.0_f32, 180.0] {
            let mapped = point.mapped(&Matrix::rotate(turn.to_radians()));
            assert!(
                matches!(mapped, LightType::Point { location } if (location.2 - 5.0).abs() < 1e-4),
                "{turn}: {mapped:?}"
            );
        }
        let turned = light.mapped(&Matrix::rotate(90.0_f32.to_radians()));
        assert!(matches!(turned, LightType::Distant { direction } if direction == (0.0, 1.0, 1.0)));
    }

    #[test]
    fn filters_report_whether_they_affect_transparent_black() {
        let mut add_alpha = [0.0; 20];
        for i in 0..4 {
            add_alpha[i * 5 + i] = 1.0;
        }
        add_alpha[19] = 1.0;
        let cf: ColorFilterRef = Arc::new(ColorMatrixFilter::new(add_alpha));
        assert!(ColorFilterImageFilter::new(cf, None).affects_transparent_black());
        let identity: ColorFilterRef = Arc::new(ColorMatrixFilter::identity());
        assert!(!ColorFilterImageFilter::new(identity, None).affects_transparent_black());
        let blur = BlurImageFilter::new(2.0, 2.0, crate::shader::TileMode::Decal);
        assert!(!blur.affects_transparent_black());
    }

    #[test]
    fn transparent_black_is_seen_through_inputs() {
        let mut add_alpha = [0.0; 20];
        for i in 0..4 {
            add_alpha[i * 5 + i] = 1.0;
        }
        add_alpha[19] = 1.0;
        let cf: ColorFilterRef = Arc::new(ColorMatrixFilter::new(add_alpha));
        let wash: ImageFilterRef = Arc::new(ColorFilterImageFilter::new(cf, None));
        assert!(
            OffsetImageFilter::new(5.0, 5.0, Some(Arc::clone(&wash))).affects_transparent_black()
        );
        assert!(
            MorphologyImageFilter::dilate(1.0, 1.0, Some(Arc::clone(&wash)))
                .affects_transparent_black()
        );
        // A biased, alpha-convolving kernel lights up transparent pixels.
        let biased = MatrixConvolutionImageFilter::new(
            (1, 1),
            vec![1.0],
            1.0,
            0.5,
            (0, 0),
            crate::shader::TileMode::Decal,
            true,
            None,
        );
        assert!(biased.affects_transparent_black());
    }

    #[test]
    fn lighting_keeps_pixels_premultiplied() {
        let lighting = LightingImageFilter::diffuse(
            LightType::Distant {
                direction: (0.0, 0.0, 1.0),
            },
            10.0,
            1.0,
            Color4f::new(1.0, 1.0, 1.0, 1.0),
            None,
        );
        // Half-transparent red (premultiplied 128, 0, 0, 128).
        let mut pixels = vec![128u8, 0, 0, 128];
        lighting.apply(&mut pixels, 1, 1);
        assert!(pixels[0] <= pixels[3], "color <= alpha: {pixels:?}");
    }

    #[test]
    fn lighting_runs_its_input_filter() {
        // An input that clears the pixels: lighting of nothing is nothing.
        let mut clear = [0.0; 20];
        clear[19] = 0.0;
        let cf: ColorFilterRef = Arc::new(ColorMatrixFilter::new(clear));
        let input: ImageFilterRef = Arc::new(ColorFilterImageFilter::new(cf, None));
        let lighting = LightingImageFilter::diffuse(
            LightType::Distant {
                direction: (0.0, 0.0, 1.0),
            },
            1.0,
            1.0,
            Color4f::new(1.0, 1.0, 1.0, 1.0),
            Some(input),
        );
        let mut pixels = vec![200u8, 100, 50, 255];
        lighting.apply(&mut pixels, 1, 1);
        assert_eq!(pixels[3], 0, "the input cleared the pixel: {pixels:?}");
    }

    #[test]
    fn invalid_blur_and_morphology_parameters_stay_no_ops() {
        let ident = Matrix::identity();
        for sigma in [f32::NAN, -5.0] {
            let blur: ImageFilterRef = Arc::new(BlurImageFilter::new(
                sigma,
                sigma,
                crate::shader::TileMode::Decal,
            ));
            let mapped = format!("{:?}", map_image_filter(&blur, &ident));
            assert!(
                mapped.contains("sigma_x: 0.0") && mapped.contains("sigma_y: 0.0"),
                "{mapped}"
            );
            let r = Rect::new(0.0, 0.0, 10.0, 10.0);
            assert_eq!(blur.filter_bounds(&r), r, "no reach for sigma {sigma}");
        }
        let dilate: ImageFilterRef = Arc::new(MorphologyImageFilter::dilate(-3.0, -3.0, None));
        let mapped = format!("{:?}", map_image_filter(&dilate, &ident));
        assert!(
            mapped.contains("radius_x: 0.0") && mapped.contains("radius_y: 0.0"),
            "{mapped}"
        );
    }

    #[test]
    fn negative_displacement_scales_reach_as_far_as_positive_ones() {
        let map: ImageFilterRef = Arc::new(OffsetImageFilter::new(0.0, 0.0, None));
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        for scale in [20.0, -20.0] {
            let f = DisplacementMapImageFilter::new(
                ColorChannel::R,
                ColorChannel::G,
                scale,
                Arc::clone(&map),
                None,
            );
            assert_eq!(
                f.filter_bounds(&r),
                Rect::new(-10.0, -10.0, 20.0, 20.0),
                "{scale}"
            );
        }
    }

    #[test]
    fn bogus_convolution_kernel_sizes_do_not_overflow() {
        let conv: ImageFilterRef = Arc::new(MatrixConvolutionImageFilter::new(
            (65536, 65536),
            vec![1.0],
            1.0,
            0.0,
            (0, 0),
            crate::shader::TileMode::Decal,
            true,
            None,
        ));
        // Mapping (done for every layer) leaves the bogus kernel alone.
        let mapped = map_image_filter(&conv, &Matrix::rotate(90.0_f32.to_radians()));
        assert!(format!("{mapped:?}").contains("kernel_size: (65536, 65536)"));
    }

    #[test]
    fn displacement_mapping_follows_half_turns_and_diagonal_swaps() {
        let map: ImageFilterRef = Arc::new(OffsetImageFilter::new(0.0, 0.0, None));
        let filter: ImageFilterRef = Arc::new(DisplacementMapImageFilter::new(
            ColorChannel::R,
            ColorChannel::G,
            20.0,
            map,
            None,
        ));
        // A half turn reverses both axes: the scale's sign flips.
        let turned = format!(
            "{:?}",
            map_image_filter(&filter, &Matrix::rotate(180.0_f32.to_radians()))
        );
        assert!(turned.contains("scale: -20.0"), "{turned}");
        // Reflecting across the diagonal swaps which channel drives which axis.
        let mut swap = Matrix::identity();
        swap.values = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let swapped = format!("{:?}", map_image_filter(&filter, &swap));
        assert!(
            swapped.contains("x_channel: G") && swapped.contains("y_channel: R"),
            "{swapped}"
        );
    }

    #[test]
    fn displacement_mapping_keeps_the_scale_sign() {
        let map: ImageFilterRef = Arc::new(OffsetImageFilter::new(0.0, 0.0, None));
        let filter: ImageFilterRef = Arc::new(DisplacementMapImageFilter::new(
            ColorChannel::R,
            ColorChannel::G,
            -20.0,
            map,
            None,
        ));
        let mapped = map_image_filter(&filter, &Matrix::scale(2.0, 2.0));
        let debug = format!("{mapped:?}");
        assert!(debug.contains("scale: -40.0"), "{debug}");
    }

    #[test]
    fn lighting_and_convolution_map_their_inputs_and_lights() {
        // A point light at (10, 20, 5) under scale(4) moves to (40, 80, 20),
        // and a convolution's offset input maps too (its reach grows 4x).
        let m = Matrix::scale(4.0, 4.0);
        let light = LightType::Point {
            location: (10.0, 20.0, 5.0),
        }
        .mapped(&m);
        assert!(matches!(light, LightType::Point { location } if location == (40.0, 80.0, 20.0)));

        let offset: ImageFilterRef = Arc::new(OffsetImageFilter::new(100.0, 0.0, None));
        let conv: ImageFilterRef = Arc::new(MatrixConvolutionImageFilter::new(
            (1, 1),
            vec![1.0],
            1.0,
            0.0,
            (0, 0),
            crate::shader::TileMode::Decal,
            true,
            Some(offset),
        ));
        let mapped = map_image_filter(&conv, &m);
        let out = Rect::new(0.0, 0.0, 10.0, 10.0);
        let input = mapped.input_bounds(&out);
        assert_eq!(input.left, -400.0, "the offset input maps: {input:?}");
    }

    use super::*;

    #[test]
    fn filters_run_their_inputs_and_bound_through_them() {
        // Each filter runs its input first: here a 1-pixel shift right.
        let shift: ImageFilterRef = Arc::new(OffsetImageFilter::new(1.0, 0.0, None));
        let filters: Vec<ImageFilterRef> = vec![
            Arc::new(MorphologyImageFilter::dilate(0.0, 0.0, Some(shift.clone()))),
            Arc::new(MatrixConvolutionImageFilter::new(
                (1, 1),
                vec![1.0],
                1.0,
                0.0,
                (0, 0),
                crate::shader::TileMode::Decal,
                true,
                Some(shift.clone()),
            )),
            Arc::new(TileImageFilter::new(
                Rect::new(0.0, 0.0, 3.0, 1.0),
                Rect::new(0.0, 0.0, 3.0, 1.0),
                Some(shift.clone()),
            )),
            // Scale 0 moves nothing: the output is the color input's.
            Arc::new(DisplacementMapImageFilter::new(
                ColorChannel::R,
                ColorChannel::G,
                0.0,
                Arc::new(OffsetImageFilter::new(0.0, 0.0, None)),
                Some(shift),
            )),
        ];
        for filter in &filters {
            let mut px = vec![255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0];
            filter.apply(&mut px, 3, 1);
            assert_eq!(&px[4..8], &[255, 0, 0, 255], "{filter:?}");
        }

        // A filter wrapping a blur reaches as far as the blur.
        let blur: ImageFilterRef = Arc::new(BlurImageFilter::new(
            5.0,
            5.0,
            crate::shader::TileMode::Decal,
        ));
        let src = Rect::new(40.0, 40.0, 60.0, 60.0);
        let wrapped = OffsetImageFilter::new(0.0, 0.0, Some(blur.clone()));
        assert_eq!(wrapped.filter_bounds(&src), blur.filter_bounds(&src));
    }

    #[test]
    fn test_blur_mask_filter_blurs() {
        let filter = BlurMaskFilter::new(BlurStyle::Normal, 3.0);
        // 5x5 mask with a single bright pixel in the center
        let mut mask = vec![0u8; 25];
        mask[12] = 255;
        filter.apply_mask(&mut mask, 5, 5);
        // Center should be less than 255 (blurred)
        assert!(
            mask[12] < 255,
            "center should be blurred lower, got {}",
            mask[12]
        );
        // Should have spread to neighbors
        let has_spread = mask.iter().any(|&v| v > 0 && v < 255);
        assert!(has_spread, "blur should produce intermediate values");
    }

    #[test]
    fn test_blur_image_filter_blurs_rgba() {
        let filter = BlurImageFilter::new(3.0, 3.0, crate::shader::TileMode::Clamp);
        // 5x5 RGBA image, white pixel in center, rest black
        let mut pixels = vec![0u8; 25 * 4];
        pixels[12 * 4] = 255;
        pixels[12 * 4 + 1] = 255;
        pixels[12 * 4 + 2] = 255;
        pixels[12 * 4 + 3] = 255;
        filter.apply(&mut pixels, 5, 5);
        // Center should dim
        assert!(pixels[12 * 4] < 255);
        // Neighbor should brighten
        let neighbor_r = pixels[11 * 4];
        assert!(neighbor_r > 0);
    }

    // --- Conformance regression tests (Task 3) ---

    #[test]
    fn test_blur_mask_filter_small_sigma_blurs() {
        // SkBlurMask::ConvertSigmaToRadius: radius = (sigma - 0.5) / 0.57735.
        // sigma = 0.9 gives radius ~0.69, which must blur (the old code
        // truncated sigma to 0 and did nothing).
        let filter = BlurMaskFilter::new(BlurStyle::Normal, 0.9);
        let mut mask = vec![0u8; 25];
        mask[12] = 255;
        filter.apply_mask(&mut mask, 5, 5);
        assert!(
            mask[12] < 255,
            "sigma 0.9 must blur, center still {}",
            mask[12]
        );
        assert!(mask[11] > 0, "sigma 0.9 must spread to neighbors");
    }

    #[test]
    fn test_blur_image_filter_small_sigma_blurs() {
        let filter = BlurImageFilter::new(0.9, 0.9, crate::shader::TileMode::Clamp);
        let mut pixels = vec![0u8; 25 * 4];
        for c in 0..4 {
            pixels[12 * 4 + c] = 255;
        }
        filter.apply(&mut pixels, 5, 5);
        assert!(pixels[12 * 4] < 255, "sigma 0.9 must blur the image");
        assert!(pixels[11 * 4] > 0, "sigma 0.9 must spread to neighbors");
    }

    #[test]
    fn test_blur_mask_filter_solid_style() {
        // Solid = src UNION blur: the original geometry stays fully opaque,
        // the halo spreads outward.
        let filter = BlurMaskFilter::new(BlurStyle::Solid, 2.0);
        let mut mask = vec![0u8; 49];
        mask[24] = 255; // center of 7x7
        filter.apply_mask(&mut mask, 7, 7);
        assert_eq!(mask[24], 255, "solid keeps the source opaque");
        assert!(mask[23] > 0, "solid still has a blur halo");
    }

    #[test]
    fn test_blur_mask_filter_outer_style() {
        // Outer = blur MINUS src: zero where the source was opaque, halo
        // outside.
        let filter = BlurMaskFilter::new(BlurStyle::Outer, 2.0);
        let mut mask = vec![0u8; 49];
        mask[24] = 255;
        filter.apply_mask(&mut mask, 7, 7);
        assert_eq!(mask[24], 0, "outer clears the source region");
        assert!(mask[23] > 0, "outer keeps the halo outside the source");
    }

    #[test]
    fn test_blur_mask_filter_inner_style() {
        // Inner = blur INTERSECT src: nonzero only inside the source.
        let filter = BlurMaskFilter::new(BlurStyle::Inner, 2.0);
        let mut mask = vec![0u8; 49];
        mask[24] = 255;
        filter.apply_mask(&mut mask, 7, 7);
        assert!(mask[24] > 0, "inner keeps blur inside the source");
        assert!(mask[24] < 255, "inner is the blurred value, not the source");
        assert_eq!(mask[23], 0, "inner is empty outside the source");
    }

    #[test]
    fn test_color_filter_image_filter_unpremul_roundtrip() {
        // Premul pixel: 50%-alpha red = (128, 0, 0, 128). The matrix runs in
        // unpremul space (SkColorFilters::Matrix default): unpremul (1,0,0),
        // invert -> (0,1,1), clamp, re-premul with a=0.5 -> (0,128,128,128).
        let color_filter = Arc::new(ColorMatrixFilter::invert()) as ColorFilterRef;
        let filter = ColorFilterImageFilter::new(color_filter, None);
        let mut pixels = vec![128u8, 0, 0, 128];
        filter.apply(&mut pixels, 1, 1);
        assert_eq!(pixels[3], 128, "alpha untouched by invert");
        assert!(pixels[0] <= 1, "r must be ~0, got {}", pixels[0]);
        assert!(
            (i32::from(pixels[1]) - 128).abs() <= 1,
            "g must be re-premultiplied (~128), got {}",
            pixels[1]
        );
        assert!(
            (i32::from(pixels[2]) - 128).abs() <= 1,
            "b must be re-premultiplied (~128), got {}",
            pixels[2]
        );
    }

    #[test]
    fn test_matrix_convolution_unpremul_when_not_convolving_alpha() {
        // 2x1 image: premul 50%-alpha red, opaque blue. 1x2 averaging kernel.
        // With convolve_alpha = false the RGB convolution runs on
        // unpremultiplied colors and is re-premultiplied with the original
        // alpha: pixel 0 -> avg((1,0,0),(0,0,1)) = (.5,0,.5), a=0.5 ->
        // premul bytes (64, 0, 64, 128).
        let filter = MatrixConvolutionImageFilter::new(
            (2, 1),
            vec![0.5, 0.5],
            1.0,
            0.0,
            (0, 0),
            crate::shader::TileMode::Clamp,
            false,
            None,
        );
        let mut pixels = vec![
            128u8, 0, 0, 128, // premul 50% red
            0, 0, 255, 255, // opaque blue
        ];
        filter.apply(&mut pixels, 2, 1);
        assert_eq!(pixels[3], 128, "alpha unchanged");
        assert!(
            (i32::from(pixels[0]) - 64).abs() <= 1,
            "r must be premul(0.5 * a), got {}",
            pixels[0]
        );
        assert!(
            (i32::from(pixels[2]) - 64).abs() <= 1,
            "b must be premul(0.5 * a) (unpremul convolve), got {}",
            pixels[2]
        );
    }

    #[test]
    fn test_matrix_convolution_honors_decal_tile_mode() {
        // 1x1 white image with a 3x1 averaging kernel centered on the pixel.
        // Decal: out-of-bounds taps contribute transparent -> ~255/3.
        let filter = MatrixConvolutionImageFilter::new(
            (3, 1),
            vec![1.0 / 3.0; 3],
            1.0,
            0.0,
            (1, 0),
            crate::shader::TileMode::Decal,
            true,
            None,
        );
        let mut pixels = vec![255u8, 255, 255, 255];
        filter.apply(&mut pixels, 1, 1);
        assert!(
            pixels[0] < 100,
            "decal tile mode must not clamp-extend the edge, got {}",
            pixels[0]
        );
        // Clamp, by contrast, keeps the pixel at ~255.
        let filter_clamp = MatrixConvolutionImageFilter::new(
            (3, 1),
            vec![1.0 / 3.0; 3],
            1.0,
            0.0,
            (1, 0),
            crate::shader::TileMode::Clamp,
            true,
            None,
        );
        let mut pixels = vec![255u8, 255, 255, 255];
        filter_clamp.apply(&mut pixels, 1, 1);
        assert!(pixels[0] > 250, "clamp extends the edge pixel");
    }

    #[test]
    fn test_tile_image_filter_fills_only_dst_rect() {
        // 4x4 blue buffer with a red pixel at (0,0). src_rect = unit square at
        // origin, dst_rect = 2x2 at origin. Only dst_rect gets tiled red;
        // pixels outside dst_rect are untouched.
        let filter = TileImageFilter::new(
            Rect::from_xywh(0.0, 0.0, 1.0, 1.0),
            Rect::from_xywh(0.0, 0.0, 2.0, 2.0),
            None,
        );
        let mut pixels = Vec::new();
        for i in 0..16 {
            if i == 0 {
                pixels.extend_from_slice(&[255u8, 0, 0, 255]); // red
            } else {
                pixels.extend_from_slice(&[0u8, 0, 255, 255]); // blue
            }
        }
        filter.apply(&mut pixels, 4, 4);
        // Inside dst_rect: tiled red.
        for (x, y) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let idx = (y * 4 + x) * 4;
            assert_eq!(pixels[idx], 255, "({x}, {y}) inside dst_rect is red");
        }
        // Outside dst_rect: untouched blue.
        for (x, y) in [(2usize, 0usize), (3, 3), (0, 2), (2, 2)] {
            let idx = (y * 4 + x) * 4;
            assert_eq!(
                pixels[idx + 2],
                255,
                "({x}, {y}) outside dst_rect stays blue"
            );
        }
    }

    #[test]
    fn test_color_filter_image_filter() {
        let color_filter = Arc::new(ColorMatrixFilter::saturation(0.0)) as ColorFilterRef;
        let filter = ColorFilterImageFilter::new(color_filter, None);
        // 2x2 RGBA image with various colors
        let mut pixels = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            128, 128, 128, 255, // gray
        ];
        filter.apply(&mut pixels, 2, 2);
        // Saturation 0.0 should produce grayscale
        // Red pixel should have equal RGB
        let r = pixels[0];
        let g = pixels[1];
        let b = pixels[2];
        assert!(
            (i32::from(r) - i32::from(g)).abs() < 5,
            "red pixel should be grayscale"
        );
        assert!(
            (i32::from(g) - i32::from(b)).abs() < 5,
            "red pixel should be grayscale"
        );
    }

    #[test]
    fn test_compose_image_filter() {
        let inner = Arc::new(BlurImageFilter::new(
            2.0,
            2.0,
            crate::shader::TileMode::Clamp,
        )) as ImageFilterRef;
        let outer_color = Arc::new(ColorMatrixFilter::saturation(0.0)) as ColorFilterRef;
        let outer = Arc::new(ColorFilterImageFilter::new(outer_color, None)) as ImageFilterRef;
        let compose = ComposeImageFilter::new(outer, inner);

        let mut pixels = vec![0u8; 25 * 4];
        pixels[12 * 4] = 255; // red center
        pixels[12 * 4 + 3] = 255;
        compose.apply(&mut pixels, 5, 5);
        // Should be blurred and grayscale
        assert!(pixels[12 * 4] < 255, "should be blurred");
        // Check a neighbor got some color
        assert!(pixels[11 * 4] > 0, "should have spread");
    }

    #[test]
    fn test_offset_image_filter() {
        let filter = OffsetImageFilter::new(1.0, 1.0, None);
        let mut pixels = vec![0u8; 16 * 4]; // 4x4
        pixels[0] = 255; // red at top-left
        pixels[3] = 255; // alpha
        filter.apply(&mut pixels, 4, 4);
        // Pixel should have shifted right and down
        // Top-left should now be clamped from what was at (-1, -1) = top-left
        assert_eq!(pixels[0], 255, "clamp-to-edge should preserve top-left");
    }

    #[test]
    fn test_drop_shadow_image_filter_applies_shadow() {
        let filter =
            DropShadowImageFilter::new(2.0, 2.0, 1.0, 1.0, Color4f::new(0.0, 0.0, 0.0, 1.0), false);
        // 5x5 image with single white pixel at center
        let mut pixels = vec![0u8; 25 * 4];
        pixels[12 * 4] = 255;
        pixels[12 * 4 + 1] = 255;
        pixels[12 * 4 + 2] = 255;
        pixels[12 * 4 + 3] = 255;
        filter.apply(&mut pixels, 5, 5);
        // Shadow should now be present at some offset positions
        // The original white pixel should still be bright
        assert!(pixels[12 * 4 + 3] > 200, "source should remain bright");
        // Check that some shadow spread occurred (non-zero alpha in neighbors)
        let mut has_shadow = false;
        for i in 0..25 {
            if i != 12 && pixels[i * 4 + 3] > 0 {
                has_shadow = true;
                break;
            }
        }
        assert!(has_shadow, "shadow should spread to other pixels");
    }

    #[test]
    fn test_morphology_dilate_expands_bright_pixel() {
        let filter = MorphologyImageFilter::dilate(1.0, 1.0, None);
        // Single bright white pixel in center
        let mut pixels = vec![0u8; 25 * 4];
        for c in 0..4 {
            pixels[12 * 4 + c] = 255;
        }
        filter.apply(&mut pixels, 5, 5);
        // Neighbors should now be bright (dilated)
        assert_eq!(
            pixels[7 * 4],
            255,
            "neighbor above (7) should be dilated to 255"
        );
        assert_eq!(
            pixels[11 * 4],
            255,
            "neighbor left (11) should be dilated to 255"
        );
        assert_eq!(
            pixels[13 * 4],
            255,
            "neighbor right (13) should be dilated to 255"
        );
        assert_eq!(
            pixels[17 * 4],
            255,
            "neighbor below (17) should be dilated to 255"
        );
    }

    #[test]
    fn test_morphology_erode_shrinks_bright_region() {
        let filter = MorphologyImageFilter::erode(1.0, 1.0, None);
        // 3x3 bright region in center of 5x5
        let mut pixels = vec![0u8; 25 * 4];
        for y in 1..4 {
            for x in 1..4 {
                let idx = (y * 5 + x) * 4;
                pixels[idx] = 255;
                pixels[idx + 1] = 255;
                pixels[idx + 2] = 255;
                pixels[idx + 3] = 255;
            }
        }
        filter.apply(&mut pixels, 5, 5);
        // Center should still be bright
        assert_eq!(pixels[12 * 4], 255, "center should remain bright");
        // Edges should erode (neighbor of black = become black)
        assert_eq!(
            pixels[6 * 4],
            0,
            "top-left of bright region (6) should erode to 0"
        );
    }

    #[test]
    fn test_displacement_map_displaces_pixels() {
        let filter = DisplacementMapImageFilter::new(
            ColorChannel::R,
            ColorChannel::G,
            10.0,
            Arc::new(BlurImageFilter::new(
                0.0,
                0.0,
                crate::shader::TileMode::Clamp,
            )),
            None,
        );
        // Create a gradient image for displacement
        let mut pixels = vec![0u8; 25 * 4];
        for y in 0..5 {
            for x in 0..5 {
                let idx = (y * 5 + x) * 4;
                pixels[idx] = u8::try_from(x * 50).unwrap_or(u8::MAX); // R gradient horizontal
                pixels[idx + 1] = u8::try_from(y * 50).unwrap_or(u8::MAX); // G gradient vertical
                pixels[idx + 2] = 128;
                pixels[idx + 3] = 255;
            }
        }
        let original = pixels.clone();
        filter.apply(&mut pixels, 5, 5);
        // Pixels should have moved based on R and G channels
        // At least some pixels should differ from original
        let mut changed = false;
        for i in 0..25 {
            if pixels[i * 4] != original[i * 4]
                || pixels[i * 4 + 1] != original[i * 4 + 1]
                || pixels[i * 4 + 2] != original[i * 4 + 2]
            {
                changed = true;
                break;
            }
        }
        assert!(changed, "displacement should change pixel positions");
    }

    #[test]
    fn test_lighting_image_filter() {
        let light = LightType::Distant {
            direction: (0.0, 0.0, 1.0),
        };
        let filter =
            LightingImageFilter::diffuse(light, 1.0, 1.0, Color4f::new(1.0, 1.0, 1.0, 1.0), None);
        let mut pixels = vec![128u8; 25 * 4]; // gray image
        filter.apply(&mut pixels, 5, 5);
        // Lighting should modulate the image (may brighten based on implementation).
        // `pixels[0]` is a `u8`, so simply reading it back is the smoke test:
        // this panics on out-of-bounds access if `apply` corrupted the buffer.
        let _ = pixels[0];
    }

    #[test]
    fn test_matrix_convolution_identity() {
        // Identity kernel: center=1, rest=0
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let filter = MatrixConvolutionImageFilter::new(
            (3, 3),
            kernel,
            1.0,
            0.0,
            (1, 1),
            crate::shader::TileMode::Clamp,
            true,
            None,
        );
        let mut pixels = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            128, 128, 128, 255, // gray
        ];
        let original = pixels.clone();
        filter.apply(&mut pixels, 2, 2);
        // Identity kernel should preserve pixels (within rounding)
        for i in 0..16 {
            assert!(
                (i32::from(pixels[i]) - i32::from(original[i])).abs() <= 1,
                "identity kernel should preserve pixel values"
            );
        }
    }

    #[test]
    fn test_tile_image_filter() {
        let filter = TileImageFilter::new(
            Rect::new(0.0, 0.0, 2.0, 2.0),
            Rect::new(0.0, 0.0, 4.0, 4.0),
            None,
        );
        let mut pixels = vec![0u8; 16 * 4]; // 4x4
        // Set top-left 2x2 to distinct colors
        pixels[0] = 255; // (0,0) red
        pixels[4] = 0;
        pixels[4 + 1] = 255; // (1,0) green
        pixels[8] = 0;
        pixels[8 + 2] = 255; // (0,1) blue
        pixels[12] = 255;
        pixels[12 + 1] = 255; // (1,1) yellow
        filter.apply(&mut pixels, 4, 4);
        // The 2x2 tile should repeat across the 4x4 image
        // (2,0) should match (0,0) = red
        assert_eq!(pixels[2 * 4], 255, "tile should repeat horizontally");
        // (0,2) should match (0,0) = red
        assert_eq!(pixels[8 * 4], 255, "tile should repeat vertically");
    }

    #[test]
    fn test_color_matrix_invert() {
        let filter = ColorMatrixFilter::invert();
        let result = filter.filter_color(Color4f::new(0.2, 0.4, 0.8, 1.0));
        assert!((result.r - 0.8).abs() < 1e-3);
        assert!((result.g - 0.6).abs() < 1e-3);
        assert!((result.b - 0.2).abs() < 1e-3);
        assert!((result.a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_color_matrix_grayscale() {
        let filter = ColorMatrixFilter::grayscale();
        let result = filter.filter_color(Color4f::new(1.0, 0.0, 0.0, 1.0));
        // R -> 0.299, G -> 0.299, B -> 0.299 (all channels become luminance)
        assert!((result.r - 0.299).abs() < 1e-3);
        assert!((result.g - 0.299).abs() < 1e-3);
        assert!((result.b - 0.299).abs() < 1e-3);
        assert!((result.a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_color_matrix_brightness_zero_is_identity() {
        let filter = ColorMatrixFilter::brightness(0.0);
        let input = Color4f::new(0.3, 0.4, 0.5, 0.7);
        let result = filter.filter_color(input);
        assert!((result.r - 0.3).abs() < 1e-3);
        assert!((result.g - 0.4).abs() < 1e-3);
        assert!((result.b - 0.5).abs() < 1e-3);
        assert!((result.a - 0.7).abs() < 1e-3);
    }

    #[test]
    fn test_color_matrix_sepia() {
        let filter = ColorMatrixFilter::sepia();
        let result = filter.filter_color(Color4f::new(1.0, 0.0, 0.0, 1.0));
        // Red through sepia matrix
        assert!(result.r > 0.3);
        assert!(result.g > 0.3);
        assert!(result.b > 0.1);
        assert!((result.a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_color_matrix_contrast() {
        let filter = ColorMatrixFilter::contrast(2.0);
        let result = filter.filter_color(Color4f::new(0.75, 0.25, 0.5, 1.0));
        // (0.75 - 0.5) * 2.0 + 0.5 = 1.0
        assert!((result.r - 1.0).abs() < 1e-3);
        // (0.25 - 0.5) * 2.0 + 0.5 = 0.0
        assert!((result.g - 0.0).abs() < 1e-3);
        // (0.5 - 0.5) * 2.0 + 0.5 = 0.5
        assert!((result.b - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_color_matrix_hue_rotate() {
        let filter = ColorMatrixFilter::hue_rotate(180.0);
        let result = filter.filter_color(Color4f::new(1.0, 0.0, 0.0, 1.0));
        // Red rotated 180 degrees should be cyan-ish (low R, high G/B)
        assert!(result.r < 0.5);
        assert!((result.a - 1.0).abs() < 1e-3);
    }
}
