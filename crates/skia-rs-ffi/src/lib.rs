//! C FFI bindings for skia-rs.
//!
//! This crate provides C-compatible bindings for use from other languages.
//! It exposes a C API that mirrors the Skia C API for drop-in compatibility.
//!
//! # Safety
//!
//! All FFI functions are inherently unsafe. Callers must ensure:
//! - Pointers are valid and non-null (unless explicitly documented otherwise)
//! - Proper lifetime management (using the appropriate `_unref` functions)
//! - Thread safety requirements are followed (see below)
//!
//! Every exported `sk_*` function wraps its body in `catch_panic` /
//! `catch_panic_void` so that a Rust panic can never unwind into C. If a
//! panic is caught the function returns a default value (null pointer, zero,
//! `false`, identity matrix, etc.) and sets a thread-visible flag readable
//! via [`sk_last_call_panicked`].
//!
//! # Reference Counting
//!
//! Objects follow Skia's reference counting model:
//! - Objects are created with a reference count of 1
//! - `sk_*_ref()` increments the reference count
//! - `sk_*_unref()` decrements the reference count and frees when it reaches 0
//! - Use `sk_refcnt_get_count()` to query the current count
//!
//! Reference counting operations (`ref`/`unref`) are **thread-safe** and use
//! atomic operations internally. The generic [`sk_refcnt_get_count`] /
//! [`sk_refcnt_is_unique`] entry points perform a magic-tag check *after*
//! reading the first 4 bytes at the pointer, so passing a pointer to a
//! *live, non-refcounted* allocation of at least 4 bytes returns 0 instead
//! of a bogus refcount. This is a best-effort heuristic, not a memory-safety
//! guarantee: `ptr` must still be null or a pointer to a valid allocation of
//! at least `size_of::<RefCountedHeader>()` bytes — passing a dangling,
//! freed, unaligned, or too-short pointer is still undefined behavior, same
//! as any other FFI entry point in this crate.
//!
//! # ABI Initialization
//!
//! Callers **must** invoke [`sk_init`] once at startup, passing the major /
//! minor version the client was compiled against. `sk_init` returns `false`
//! if the versions are incompatible; the caller should then abort rather
//! than calling any other `sk_*` function.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::missing_safety_doc)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types)] // FFI types follow C naming conventions

pub mod abi;

use std::ffi::{c_char, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

// =============================================================================
// Panic Catching Infrastructure
// =============================================================================

/// Global flag indicating if the last FFI call panicked.
///
/// The flag is read-and-clear. Callers query it via [`sk_last_call_panicked`]
/// immediately after an FFI call to detect panic recovery.
static LAST_PANIC: AtomicBool = AtomicBool::new(false);

/// Check if the last FFI call panicked.
///
/// Returns true if a panic occurred, false otherwise.
/// Reading this flag clears it.
#[unsafe(no_mangle)]
pub extern "C" fn sk_last_call_panicked() -> bool {
    LAST_PANIC.swap(false, Ordering::SeqCst)
}

/// Catch panics and return a default value if one occurs.
#[inline]
fn catch_panic<T: Default, F: FnOnce() -> T>(f: F) -> T {
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        LAST_PANIC.store(true, Ordering::SeqCst);
        T::default()
    })
}

/// Catch panics in void-returning functions.
#[inline]
fn catch_panic_void<F: FnOnce()>(f: F) {
    if panic::catch_unwind(AssertUnwindSafe(f)).is_err() {
        LAST_PANIC.store(true, Ordering::SeqCst);
    }
}

// =============================================================================
// Reference Counting Infrastructure
// =============================================================================

/// Magic tag embedded at the head of every [`RefCounted<T>`] instance.
///
/// The generic [`sk_refcnt_get_count`] / [`sk_refcnt_is_unique`] entry points
/// validate this tag before dereferencing the refcount field. That lets them
/// reject bogus pointers (e.g. a `Box<Paint>` that was created outside the
/// refcount wrapper) without reading arbitrary memory.
const REFCOUNT_TAG: u32 = 0x534B_5231; // "SKR1"

/// Reference counted wrapper for FFI objects.
///
/// This provides Skia-compatible reference counting semantics:
/// - Created with refcount of 1
/// - `ref()` increments
/// - `unref()` decrements and frees when 0
///
/// Layout (fixed, `repr(C)`):
///
/// | offset | bytes | field       |
/// |--------|-------|-------------|
/// | 0      | 4     | `tag`       |
/// | 4      | 4     | `refcnt`    |
/// | 8      | ..    | `value`     |
///
/// The refcount lives at offset 4 (not 0) so that callers who rely on the
/// "all refcounted types start with `AtomicU32`" idiom get a hard type-check
/// via the tag instead of a silent misread.
///
/// cbindgen:no-export
#[repr(C)]
pub struct RefCounted<T> {
    /// Magic tag — must equal [`REFCOUNT_TAG`].
    tag: u32,
    /// Reference count.
    refcnt: AtomicU32,
    /// The wrapped value.
    value: T,
}

impl<T> RefCounted<T> {
    /// Create a new reference counted object with refcount of 1.
    pub fn new(value: T) -> *mut Self {
        Box::into_raw(Box::new(Self {
            tag: REFCOUNT_TAG,
            refcnt: AtomicU32::new(1),
            value,
        }))
    }

    /// Increment the reference count.
    ///
    /// # Safety
    /// Pointer must be valid and non-null.
    pub unsafe fn ref_ptr(ptr: *mut Self) {
        if let Some(rc) = ptr.as_ref() {
            debug_assert_eq!(rc.tag, REFCOUNT_TAG, "sk refcount tag corrupted");
            rc.refcnt.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement the reference count and free if it reaches 0.
    ///
    /// # Safety
    /// Pointer must be valid and non-null.
    /// Returns true if the object was freed.
    pub unsafe fn unref_ptr(ptr: *mut Self) -> bool {
        if ptr.is_null() {
            return false;
        }

        let rc = &*ptr;
        debug_assert_eq!(rc.tag, REFCOUNT_TAG, "sk refcount tag corrupted");
        // Use AcqRel to ensure proper synchronization
        if rc.refcnt.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last reference, drop the box
            drop(Box::from_raw(ptr));
            true
        } else {
            false
        }
    }

    /// Get the current reference count.
    ///
    /// # Safety
    /// Pointer must be valid and non-null.
    pub unsafe fn get_count(ptr: *const Self) -> u32 {
        ptr.as_ref()
            .map_or(0, |rc| rc.refcnt.load(Ordering::Relaxed))
    }

    /// Check if this is the only reference.
    ///
    /// # Safety
    /// Pointer must be valid and non-null.
    pub unsafe fn is_unique(ptr: *const Self) -> bool {
        Self::get_count(ptr) == 1
    }

    /// Get a reference to the inner value.
    ///
    /// # Safety
    /// Pointer must be valid and non-null.
    pub unsafe fn get_ref<'a>(ptr: *const Self) -> Option<&'a T> {
        ptr.as_ref().map(|rc| &rc.value)
    }

    /// Get a mutable reference to the inner value.
    ///
    /// # Safety
    /// Pointer must be valid and non-null.
    /// Caller must ensure exclusive access.
    pub unsafe fn get_mut<'a>(ptr: *mut Self) -> Option<&'a mut T> {
        ptr.as_mut().map(|rc| &mut rc.value)
    }
}

// =============================================================================
// Reference Counting C API
// =============================================================================

/// Opaque reference counted object type.
pub type sk_refcnt_t = c_void;

/// Tag-validating view of the shared refcount header.
///
/// Every [`RefCounted<T>`] starts with `{ tag: u32, refcnt: AtomicU32 }`.
/// The generic entry points below match this layout to peek at the refcount
/// without knowing `T`.
#[repr(C)]
struct RefCountedHeader {
    tag: u32,
    refcnt: AtomicU32,
}

/// Get the reference count of an object.
///
/// Returns 0 if the pointer is null or fails the tag check. The tag check
/// is a best-effort heuristic against passing a non-refcounted pointer, not
/// a memory-safety guarantee: `ptr` must still be null or point to a valid
/// allocation of at least `size_of::<RefCountedHeader>()` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_refcnt_get_count(ptr: *const sk_refcnt_t) -> u32 {
    catch_panic(|| {
        if ptr.is_null() {
            return 0;
        }
        let header = &*ptr.cast::<RefCountedHeader>();
        if header.tag != REFCOUNT_TAG {
            return 0;
        }
        header.refcnt.load(Ordering::Relaxed)
    })
}

/// Check if an object has only one reference (is unique).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_refcnt_is_unique(ptr: *const sk_refcnt_t) -> bool {
    sk_refcnt_get_count(ptr) == 1
}

// Re-export types for FFI
use skia_rs_canvas::{ClipOp, PictureRef, PixelBuffer, Surface};
use skia_rs_core::{
    AlphaType, Color, Color4f, ColorSpace, ColorType, IPoint, IRect, ISize, IccProfile, ImageInfo,
    Matrix, Matrix44, Point, Rect, Region, RegionOp, Scalar, Size,
};
use skia_rs_paint::{
    BlendMode, BlurMaskFilter, BlurStyle, ColorFilterRef, ColorMatrixFilter, ImageFilterRef,
    LinearGradient, MaskFilterRef, Paint, RadialGradient, ShaderRef, Style, SweepGradient,
    TileMode,
};
use skia_rs_path::{
    DashEffect, FillType, Path, PathBuilder, PathEffectRef, TrimEffect, TrimMode, Verb,
};
use skia_rs_text::{Font, TextBlob, Typeface, TypefaceRef};

// =============================================================================
// Type Definitions
// =============================================================================

/// Clip operation for canvas clipping functions.
///
/// Numeric values match upstream `SkClipOp` (`include/core/SkClipOp.h`):
/// `kDifference = 0`, `kIntersect = 1`. These constants are exposed to C as
/// plain `uint32_t` (see [`sk_clip_op_t`]) rather than a Rust-repr enum
/// parameter, because an out-of-range value received directly into a Rust
/// `#[repr(u32)] enum` parameter is undefined behavior; functions that take
/// a clip op decode the raw integer via `decode_clip_op` instead.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkClipOp {
    /// Subtract the new region from the existing clip (difference).
    Difference = 0,
    /// Intersect the existing clip with the new region.
    Intersect = 1,
}

/// C ABI type for [`SkClipOp`].
///
/// Functions taking a clip op accept this raw `uint32_t` and decode it via
/// `decode_clip_op`, rejecting out-of-range values instead of constructing
/// an invalid Rust enum from C.
pub type sk_clip_op_t = u32;

/// Decode a raw `sk_clip_op_t` into [`ClipOp`]. Returns `None` for any value
/// other than the two defined by upstream `SkClipOp`.
const fn decode_clip_op(v: sk_clip_op_t) -> Option<ClipOp> {
    match v {
        0 => Some(ClipOp::Difference),
        1 => Some(ClipOp::Intersect),
        _ => None,
    }
}

/// Region operation for region manipulation functions.
///
/// Numeric values match upstream `SkRegion::Op` (`include/core/SkRegion.h`).
/// Exposed to C as a raw `uint32_t` (see [`sk_region_op_t`]) and decoded via
/// `decode_region_op`, which rejects out-of-range values.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkRegionOp {
    /// Subtract the operand from the target.
    Difference = 0,
    /// Intersect target and operand.
    Intersect = 1,
    /// Union target and operand.
    Union = 2,
    /// Exclusive-or target and operand.
    Xor = 3,
    /// Subtract target from operand (reverse difference).
    ReverseDifference = 4,
    /// Replace target with operand.
    Replace = 5,
}

/// C ABI type for [`SkRegionOp`]; see `decode_region_op`.
pub type sk_region_op_t = u32;

/// Decode a raw `sk_region_op_t` into [`RegionOp`]. Returns `None` for any
/// out-of-range value.
const fn decode_region_op(v: sk_region_op_t) -> Option<RegionOp> {
    match v {
        0 => Some(RegionOp::Difference),
        1 => Some(RegionOp::Intersect),
        2 => Some(RegionOp::Union),
        3 => Some(RegionOp::Xor),
        4 => Some(RegionOp::ReverseDifference),
        5 => Some(RegionOp::Replace),
        _ => None,
    }
}

/// Trim path effect mode.
///
/// Numeric values match upstream `SkTrimPathEffect::Mode`
/// (`include/effects/SkTrimPathEffect.h`): `kNormal = 0`, `kInverted = 1`.
/// Exposed to C as a raw `uint32_t` (see [`sk_trim_mode_t`]) and decoded via
/// `decode_trim_mode`, which rejects out-of-range values.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkTrimMode {
    /// Normal trim: keep segment from start to end.
    Normal = 0,
    /// Inverted trim: discard segment from start to end.
    Inverted = 1,
}

/// C ABI type for [`SkTrimMode`]; see `decode_trim_mode`.
pub type sk_trim_mode_t = u32;

/// Decode a raw `sk_trim_mode_t` into [`TrimMode`]. Returns `None` for any
/// out-of-range value.
const fn decode_trim_mode(v: sk_trim_mode_t) -> Option<TrimMode> {
    match v {
        0 => Some(TrimMode::Normal),
        1 => Some(TrimMode::Inverted),
        _ => None,
    }
}

/// C-compatible point structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_point_t {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

/// C-compatible integer point structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_ipoint_t {
    /// X coordinate.
    pub x: i32,
    /// Y coordinate.
    pub y: i32,
}

/// C-compatible size structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_size_t {
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

/// C-compatible integer size structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_isize_t {
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
}

/// C-compatible rectangle structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_rect_t {
    /// Left edge.
    pub left: f32,
    /// Top edge.
    pub top: f32,
    /// Right edge.
    pub right: f32,
    /// Bottom edge.
    pub bottom: f32,
}

/// C-compatible integer rectangle structure.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_irect_t {
    /// Left edge.
    pub left: i32,
    /// Top edge.
    pub top: i32,
    /// Right edge.
    pub right: i32,
    /// Bottom edge.
    pub bottom: i32,
}

/// C-compatible matrix structure.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct sk_matrix_t {
    /// Matrix values (row-major).
    pub values: [f32; 9],
}

impl Default for sk_matrix_t {
    fn default() -> Self {
        Self {
            values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// C-compatible image info structure.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct sk_imageinfo_t {
    /// Width.
    pub width: i32,
    /// Height.
    pub height: i32,
    /// Color type.
    pub color_type: u32,
    /// Alpha type.
    pub alpha_type: u32,
}

/// C-compatible 4-component color (float RGBA).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sk_color4f_t {
    /// Red.
    pub r: f32,
    /// Green.
    pub g: f32,
    /// Blue.
    pub b: f32,
    /// Alpha.
    pub a: f32,
}

/// C-compatible color (ARGB packed).
pub type sk_color_t = u32;

// =============================================================================
// Conversion helpers
// =============================================================================

impl From<Point> for sk_point_t {
    fn from(p: Point) -> Self {
        Self { x: p.x, y: p.y }
    }
}

impl From<sk_point_t> for Point {
    fn from(p: sk_point_t) -> Self {
        Self::new(p.x, p.y)
    }
}

impl From<Rect> for sk_rect_t {
    fn from(r: Rect) -> Self {
        Self {
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
        }
    }
}

impl From<sk_rect_t> for Rect {
    fn from(r: sk_rect_t) -> Self {
        Self::new(r.left, r.top, r.right, r.bottom)
    }
}

impl From<Matrix> for sk_matrix_t {
    fn from(m: Matrix) -> Self {
        Self { values: m.values }
    }
}

impl From<sk_matrix_t> for Matrix {
    fn from(m: sk_matrix_t) -> Self {
        Self { values: m.values }
    }
}

impl From<Color4f> for sk_color4f_t {
    fn from(c: Color4f) -> Self {
        Self {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }
}

impl From<sk_color4f_t> for Color4f {
    fn from(c: sk_color4f_t) -> Self {
        Self::new(c.r, c.g, c.b, c.a)
    }
}

/// Decode a raw `sk_color_type_t` into [`ColorType`], matching upstream
/// `SkColorType` numbering (`include/core/SkColorType.h`): `kUnknown = 0`,
/// `kAlpha_8 = 1`, `kRGB_565 = 2`, `kARGB_4444 = 3`, `kRGBA_8888 = 4`,
/// `kRGB_888x = 5`, `kBGRA_8888 = 6`, `kRGBA_1010102 = 7`,
/// `kBGRA_1010102 = 8`, `kRGB_101010x = 9`, `kBGR_101010x = 10`,
/// `kGray_8 = 14`, `kRGBA_F16Norm = 15`, `kRGBA_F16 = 16`,
/// `kRGBA_F32 = 18`, `kSRGBA_8888 = 26`, `kR8_unorm = 27`. Any value that
/// upstream defines but this crate's [`ColorType`] has no representation
/// for (and any wholly unknown value) returns `None` — never a silent
/// fallback to RGBA.
const fn decode_color_type(v: u32) -> Option<ColorType> {
    match v {
        0 => Some(ColorType::Unknown),
        1 => Some(ColorType::Alpha8),
        2 => Some(ColorType::Rgb565),
        3 => Some(ColorType::Argb4444),
        4 => Some(ColorType::Rgba8888),
        5 => Some(ColorType::Rgb888x),
        6 => Some(ColorType::Bgra8888),
        7 => Some(ColorType::Rgba1010102),
        8 => Some(ColorType::Bgra1010102),
        9 => Some(ColorType::Rgb101010x),
        10 => Some(ColorType::Bgr101010x),
        14 => Some(ColorType::Gray8),
        15 => Some(ColorType::RgbaF16Norm),
        16 => Some(ColorType::RgbaF16),
        18 => Some(ColorType::RgbaF32),
        26 => Some(ColorType::Srgba8888),
        // `ColorType::R8Unorm` (15) is mislabeled "8-bit palette index" in
        // skia-rs-core; `R8Unorm2` (22) is the true single-red-channel
        // format matching upstream `kR8_unorm`.
        27 => Some(ColorType::R8Unorm2),
        _ => None,
    }
}

/// Decode a raw alpha type value into [`AlphaType`]. Returns `None` for any
/// out-of-range value instead of silently coercing it.
const fn decode_alpha_type(v: u32) -> Option<AlphaType> {
    match v {
        0 => Some(AlphaType::Unknown),
        1 => Some(AlphaType::Opaque),
        2 => Some(AlphaType::Premul),
        3 => Some(AlphaType::Unpremul),
        _ => None,
    }
}

/// Decode a raw tile mode value into [`TileMode`]. Returns `None` for any
/// out-of-range value instead of silently coercing it.
const fn decode_tile_mode(v: u32) -> Option<TileMode> {
    match v {
        0 => Some(TileMode::Clamp),
        1 => Some(TileMode::Repeat),
        2 => Some(TileMode::Mirror),
        3 => Some(TileMode::Decal),
        _ => None,
    }
}

// =============================================================================
// ABI initialization
// =============================================================================

static INIT_OK: AtomicBool = AtomicBool::new(false);

/// Initialize the library and verify ABI compatibility.
///
/// C clients must call this once, before any other `sk_*` function, passing
/// the major/minor version they were compiled against. Returns `true` if the
/// linked library is compatible. If `false`, the caller must not call any
/// other `sk_*` function — behavior is undefined.
///
/// Calling `sk_init` twice with compatible values is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn sk_init(major: u32, minor: u32) -> bool {
    if abi::sk_abi_is_compatible(major, minor) {
        INIT_OK.store(true, Ordering::SeqCst);
        true
    } else {
        false
    }
}

/// Check whether [`sk_init`] has been called successfully.
#[unsafe(no_mangle)]
pub extern "C" fn sk_is_initialized() -> bool {
    INIT_OK.load(Ordering::SeqCst)
}

// =============================================================================
// Surface API (Reference Counted)
// =============================================================================

/// Reference counted surface type.
pub type sk_surface_t = RefCounted<Surface>;

/// Create a new raster surface.
///
/// Returns a surface with refcount of 1, or null on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_new_raster(width: i32, height: i32) -> *mut sk_surface_t {
    catch_panic(|| {
        Surface::new_raster_n32_premul(width, height).map_or(ptr::null_mut(), RefCounted::new)
    })
}

/// Create a raster surface with specific image info.
///
/// Returns a surface with refcount of 1, or null on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_new_raster_with_info(
    info: *const sk_imageinfo_t,
) -> *mut sk_surface_t {
    catch_panic(|| {
        if info.is_null() {
            return ptr::null_mut();
        }

        let info = &*info;
        let Some(color_type) = decode_color_type(info.color_type) else {
            return ptr::null_mut();
        };
        let Some(alpha_type) = decode_alpha_type(info.alpha_type) else {
            return ptr::null_mut();
        };
        let Ok(img_info) = ImageInfo::new(info.width, info.height, color_type, alpha_type) else {
            return ptr::null_mut();
        };

        Surface::new_raster(&img_info, None).map_or(ptr::null_mut(), RefCounted::new)
    })
}

/// Increment the reference count of a surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_ref(surface: *mut sk_surface_t) {
    catch_panic_void(|| RefCounted::ref_ptr(surface));
}

/// Decrement the reference count of a surface.
///
/// Frees the surface when the count reaches 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_unref(surface: *mut sk_surface_t) {
    catch_panic_void(|| {
        if RefCounted::unref_ptr(surface) {
            // The surface was freed — drop any stale lock entry so its
            // address can be safely reused by a future allocation.
            locked_surface_set().remove(&(surface as usize));
        }
    });
}

/// Get the reference count of a surface.
///
/// A canvas locked with [`sk_surface_lock_canvas`] holds one reference
/// until [`sk_canvas_release`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_get_refcnt(surface: *const sk_surface_t) -> u32 {
    catch_panic(|| RefCounted::get_count(surface))
}

/// Check if the surface has only one reference.
///
/// A locked canvas holds a reference, so this is false while one is
/// locked.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_is_unique(surface: *const sk_surface_t) -> bool {
    catch_panic(|| RefCounted::is_unique(surface))
}

/// Get the width of a surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_get_width(surface: *const sk_surface_t) -> i32 {
    catch_panic(|| RefCounted::get_ref(surface).map_or(0, skia_rs_canvas::Surface::width))
}

/// Get the height of a surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_get_height(surface: *const sk_surface_t) -> i32 {
    catch_panic(|| RefCounted::get_ref(surface).map_or(0, skia_rs_canvas::Surface::height))
}

/// Get the pixel data from a surface (unsynchronized borrow).
///
/// The bytes are always **premultiplied**, regardless of the surface's
/// declared alpha type — the underlying buffer physically stores premul
/// pixels, and a borrowed pointer cannot be converted in place without
/// mutating the surface's own storage. Surfaces created with
/// `alpha_type = Unpremul` are rejected (returns `false`); use
/// [`sk_surface_read_pixels`] instead, which copies out and unpremultiplies
/// as needed.
///
/// **Lifetime:** the pointer returned via `out_pixels` is only valid for as
/// long as `surface` remains allocated AND is not mutated. Any subsequent
/// drawing call on this surface (e.g. [`sk_surface_draw_rect`]) may move
/// or resize the underlying buffer, invalidating the pointer. Callers that
/// need a stable snapshot should copy out via [`sk_surface_read_pixels`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_peek_pixels(
    surface: *const sk_surface_t,
    out_pixels: *mut *const u8,
    out_row_bytes: *mut usize,
) -> bool {
    catch_panic(|| {
        if out_pixels.is_null() || out_row_bytes.is_null() {
            return false;
        }

        if let Some(s) = RefCounted::get_ref(surface) {
            if s.info().alpha_type == AlphaType::Unpremul {
                // The borrowed buffer is physically premultiplied; we
                // cannot hand back unpremultiplied bytes without copying.
                return false;
            }
            *out_pixels = s.pixels().as_ptr();
            *out_row_bytes = s.row_bytes();
            true
        } else {
            false
        }
    })
}

/// Copy the pixel data from a surface into a caller-owned buffer.
///
/// Safer than [`sk_surface_peek_pixels`] — the caller owns the destination
/// buffer and does not need to track the surface's lifetime.
///
/// The bytes are converted to match the surface's declared alpha type: for
/// surfaces created with `alpha_type = Unpremul`, the copied pixels are
/// unpremultiplied before being written to `dst`; otherwise the raw
/// premultiplied bytes are copied as-is.
///
/// Returns the number of bytes written, or 0 on failure (null surface,
/// null/undersized destination buffer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_read_pixels(
    surface: *const sk_surface_t,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    catch_panic(|| {
        if dst.is_null() {
            return 0;
        }
        let Some(s) = RefCounted::get_ref(surface) else {
            return 0;
        };
        let src = s.pixels();
        if dst_len < src.len() {
            return 0;
        }
        ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        if s.info().alpha_type == AlphaType::Unpremul {
            let dst_slice = slice::from_raw_parts_mut(dst, src.len());
            skia_rs_core::unpremultiply_in_place(dst_slice);
        }
        src.len()
    })
}

// =============================================================================
// Paint API (Reference Counted)
// =============================================================================

/// Reference counted paint type.
pub type sk_paint_t = RefCounted<Paint>;

/// Create a new paint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_new() -> *mut sk_paint_t {
    catch_panic(|| RefCounted::new(Paint::new()))
}

/// Clone a paint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_clone(paint: *const sk_paint_t) -> *mut sk_paint_t {
    catch_panic(|| {
        RefCounted::get_ref(paint).map_or(ptr::null_mut(), |p| RefCounted::new(p.clone()))
    })
}

/// Increment the reference count of a paint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_ref(paint: *mut sk_paint_t) {
    catch_panic_void(|| RefCounted::ref_ptr(paint));
}

/// Decrement the reference count of a paint (alias for unref).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_delete(paint: *mut sk_paint_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(paint);
    });
}

/// Decrement the reference count of a paint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_unref(paint: *mut sk_paint_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(paint);
    });
}

/// Get the reference count of a paint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_get_refcnt(paint: *const sk_paint_t) -> u32 {
    catch_panic(|| RefCounted::get_count(paint))
}

/// Set the paint color.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_color(paint: *mut sk_paint_t, color: sk_color_t) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            p.set_color32(Color(color));
        }
    });
}

/// Get the paint color.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_get_color(paint: *const sk_paint_t) -> sk_color_t {
    catch_panic(|| RefCounted::get_ref(paint).map_or(0, |p| p.color32().0))
}

/// Set the paint color as float RGBA.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_color4f(paint: *mut sk_paint_t, color: sk_color4f_t) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            p.set_color(color.into());
        }
    });
}

/// Get the paint color as float RGBA.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_get_color4f(paint: *const sk_paint_t) -> sk_color4f_t {
    catch_panic(|| {
        RefCounted::get_ref(paint).map_or_else(sk_color4f_t::default, |p| p.color().into())
    })
}

/// Set the paint style.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_style(paint: *mut sk_paint_t, style: u32) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            let style = match style {
                1 => Style::Stroke,
                2 => Style::StrokeAndFill,
                _ => Style::Fill,
            };
            p.set_style(style);
        }
    });
}

/// Set the stroke width.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_stroke_width(paint: *mut sk_paint_t, width: f32) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            p.set_stroke_width(width);
        }
    });
}

/// Get the stroke width.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_get_stroke_width(paint: *const sk_paint_t) -> f32 {
    catch_panic(|| RefCounted::get_ref(paint).map_or(0.0, skia_rs_paint::Paint::stroke_width))
}

/// Set anti-alias.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_antialias(paint: *mut sk_paint_t, aa: bool) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            p.set_anti_alias(aa);
        }
    });
}

/// Check if anti-alias is enabled.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_is_antialias(paint: *const sk_paint_t) -> bool {
    catch_panic(|| RefCounted::get_ref(paint).is_some_and(skia_rs_paint::Paint::is_anti_alias))
}

/// C ABI type for the paint's blend mode. Values follow upstream
/// `SkBlendMode` numbering (`include/core/SkBlendMode.h`), all 29 modes
/// (`kClear = 0` .. `kLuminosity = 28`); see [`BlendMode::from_u8`].
pub type sk_blend_mode_t = u32;

/// Set the paint's blend mode.
///
/// `mode` follows upstream `SkBlendMode` (0-28); see [`BlendMode::from_u8`].
/// Returns false (leaving the paint's blend mode unchanged) if `paint` is
/// null or `mode` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_blend_mode(
    paint: *mut sk_paint_t,
    mode: sk_blend_mode_t,
) -> bool {
    catch_panic(|| {
        let Some(p) = RefCounted::get_mut(paint) else {
            return false;
        };
        let Some(mode) = u8::try_from(mode).ok().and_then(BlendMode::from_u8) else {
            return false;
        };
        p.set_blend_mode(mode);
        true
    })
}

/// Get the paint's blend mode as a raw `sk_blend_mode_t` (upstream
/// `SkBlendMode` numbering, 0-28). Returns `SrcOver` (3) if `paint` is null,
/// matching `SkPaint`'s default blend mode.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_get_blend_mode(paint: *const sk_paint_t) -> sk_blend_mode_t {
    catch_panic(|| {
        RefCounted::get_ref(paint).map_or(BlendMode::SrcOver as u32, |p| p.blend_mode() as u32)
    })
}

/// Attach a shader to the paint. Passing null clears it.
///
/// The paint takes a new reference to the shader; the caller retains their
/// own reference and must still unref it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_shader(paint: *mut sk_paint_t, shader: *const sk_shader_t) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            if shader.is_null() {
                p.set_shader(None);
            } else if let Some(s) = RefCounted::get_ref(shader) {
                p.set_shader(Some(s.clone()));
            }
        }
    });
}

/// Attach a color filter to the paint. Passing null clears it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_color_filter(
    paint: *mut sk_paint_t,
    filter: *const sk_colorfilter_t,
) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            if filter.is_null() {
                p.set_color_filter(None);
            } else if let Some(f) = RefCounted::get_ref(filter) {
                p.set_color_filter(Some(f.clone()));
            }
        }
    });
}

/// Attach a mask filter to the paint. Passing null clears it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_mask_filter(
    paint: *mut sk_paint_t,
    filter: *const sk_maskfilter_t,
) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            if filter.is_null() {
                p.set_mask_filter(None);
            } else if let Some(f) = RefCounted::get_ref(filter) {
                p.set_mask_filter(Some(f.clone()));
            }
        }
    });
}

/// Attach an image filter to the paint. Passing null clears it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_image_filter(
    paint: *mut sk_paint_t,
    filter: *const sk_imagefilter_t,
) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            if filter.is_null() {
                p.set_image_filter(None);
            } else if let Some(f) = RefCounted::get_ref(filter) {
                p.set_image_filter(Some(f.clone()));
            }
        }
    });
}

// =============================================================================
// Path API (Reference Counted)
// =============================================================================

/// Reference counted path type.
pub type sk_path_t = RefCounted<Path>;

/// Create a new path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_new() -> *mut sk_path_t {
    catch_panic(|| RefCounted::new(Path::new()))
}

/// Clone a path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_clone(path: *const sk_path_t) -> *mut sk_path_t {
    catch_panic(|| {
        RefCounted::get_ref(path).map_or(ptr::null_mut(), |p| RefCounted::new(p.clone()))
    })
}

/// Increment the reference count of a path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_ref(path: *mut sk_path_t) {
    catch_panic_void(|| RefCounted::ref_ptr(path));
}

/// Decrement the reference count of a path (alias for unref).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_delete(path: *mut sk_path_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(path);
    });
}

/// Decrement the reference count of a path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_unref(path: *mut sk_path_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(path);
    });
}

/// Get the reference count of a path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_get_refcnt(path: *const sk_path_t) -> u32 {
    catch_panic(|| RefCounted::get_count(path))
}

/// Get the path bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_get_bounds(path: *const sk_path_t, bounds: *mut sk_rect_t) {
    catch_panic_void(|| {
        if let (Some(p), Some(b)) = (RefCounted::get_ref(path), bounds.as_mut()) {
            let rect = p.bounds();
            *b = rect.into();
        }
    });
}

/// Check if path is empty.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_is_empty(path: *const sk_path_t) -> bool {
    catch_panic(|| RefCounted::get_ref(path).is_none_or(skia_rs_path::Path::is_empty))
}

/// Get the fill type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_get_filltype(path: *const sk_path_t) -> u32 {
    catch_panic(|| {
        RefCounted::get_ref(path).map_or(0, |p| match p.fill_type() {
            FillType::Winding => 0,
            FillType::EvenOdd => 1,
            FillType::InverseWinding => 2,
            FillType::InverseEvenOdd => 3,
        })
    })
}

/// Set the fill type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_set_filltype(path: *mut sk_path_t, fill_type: u32) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(path) {
            let ft = match fill_type {
                1 => FillType::EvenOdd,
                2 => FillType::InverseWinding,
                3 => FillType::InverseEvenOdd,
                _ => FillType::Winding,
            };
            p.set_fill_type(ft);
        }
    });
}

/// Check if path contains a point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_contains(path: *const sk_path_t, x: f32, y: f32) -> bool {
    catch_panic(|| RefCounted::get_ref(path).is_some_and(|p| p.contains(Point::new(x, y))))
}

// =============================================================================
// Path Iteration API
// =============================================================================

/// Stateful iterator over a path's verbs.
///
/// Create with [`sk_path_iter_new`], advance with [`sk_path_iter_next`],
/// destroy with [`sk_path_iter_delete`]. The iterator holds a clone of the
/// path's verbs/points — the source path can be modified or dropped after
/// iteration begins without invalidating the iterator.
pub struct sk_path_iter_t {
    verbs: Vec<Verb>,
    points: Vec<Point>,
    weights: Vec<Scalar>,
    verb_index: usize,
    point_index: usize,
    weight_index: usize,
}

/// ABI-stable verb codes emitted by [`sk_path_iter_next`].
pub const SK_PATH_VERB_MOVE: u32 = 0;
/// ABI-stable verb code for a line segment.
pub const SK_PATH_VERB_LINE: u32 = 1;
/// ABI-stable verb code for a quadratic bezier.
pub const SK_PATH_VERB_QUAD: u32 = 2;
/// ABI-stable verb code for a conic section.
pub const SK_PATH_VERB_CONIC: u32 = 3;
/// ABI-stable verb code for a cubic bezier.
pub const SK_PATH_VERB_CUBIC: u32 = 4;
/// ABI-stable verb code for a contour close.
pub const SK_PATH_VERB_CLOSE: u32 = 5;
/// Sentinel verb code returned once the iterator is exhausted.
pub const SK_PATH_VERB_DONE: u32 = 6;

/// Create an iterator over a path's verbs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_iter_new(path: *const sk_path_t) -> *mut sk_path_iter_t {
    catch_panic(|| {
        let Some(p) = RefCounted::get_ref(path) else {
            return ptr::null_mut();
        };
        // Collect conic weights (if any) via the public iterator — the
        // raw `conic_weights` field isn't publicly exposed on Path.
        let weights: Vec<Scalar> = if p.verbs().contains(&Verb::Conic) {
            p.iter()
                .filter_map(|el| match el {
                    skia_rs_path::PathElement::Conic(_, _, w) => Some(w),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };

        Box::into_raw(Box::new(sk_path_iter_t {
            verbs: p.verbs().to_vec(),
            points: p.points().to_vec(),
            weights,
            verb_index: 0,
            point_index: 0,
            weight_index: 0,
        }))
    })
}

/// Destroy a path iterator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_iter_delete(iter: *mut sk_path_iter_t) {
    catch_panic_void(|| {
        if !iter.is_null() {
            drop(Box::from_raw(iter));
        }
    });
}

/// Advance the iterator.
///
/// Fills up to four points in `out_points` (caller provides at least 4
/// slots) and the conic weight (if applicable) in `out_weight`. Returns the
/// verb code; `SK_PATH_VERB_DONE` indicates exhaustion.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_path_iter_next(
    iter: *mut sk_path_iter_t,
    out_points: *mut sk_point_t,
    out_weight: *mut f32,
) -> u32 {
    catch_panic(|| {
        let Some(it) = iter.as_mut() else {
            return SK_PATH_VERB_DONE;
        };
        if it.verb_index >= it.verbs.len() {
            return SK_PATH_VERB_DONE;
        }
        let verb = it.verbs[it.verb_index];
        it.verb_index += 1;

        let write = |i: usize, p: Point| {
            if !out_points.is_null() {
                *out_points.add(i) = p.into();
            }
        };

        let consume = verb.point_count();
        if it.point_index + consume > it.points.len() {
            return SK_PATH_VERB_DONE;
        }
        match verb {
            Verb::Move => {
                write(0, it.points[it.point_index]);
                it.point_index += 1;
                SK_PATH_VERB_MOVE
            }
            Verb::Line => {
                write(0, it.points[it.point_index]);
                it.point_index += 1;
                SK_PATH_VERB_LINE
            }
            Verb::Quad => {
                write(0, it.points[it.point_index]);
                write(1, it.points[it.point_index + 1]);
                it.point_index += 2;
                SK_PATH_VERB_QUAD
            }
            Verb::Conic => {
                write(0, it.points[it.point_index]);
                write(1, it.points[it.point_index + 1]);
                let w = it.weights.get(it.weight_index).copied().unwrap_or(1.0);
                if !out_weight.is_null() {
                    *out_weight = w;
                }
                it.point_index += 2;
                it.weight_index += 1;
                SK_PATH_VERB_CONIC
            }
            Verb::Cubic => {
                write(0, it.points[it.point_index]);
                write(1, it.points[it.point_index + 1]);
                write(2, it.points[it.point_index + 2]);
                it.point_index += 3;
                SK_PATH_VERB_CUBIC
            }
            Verb::Close => SK_PATH_VERB_CLOSE,
        }
    })
}

// =============================================================================
// Path Builder API (Reference Counted)
// =============================================================================

/// Reference counted path builder type.
pub type sk_pathbuilder_t = RefCounted<PathBuilder>;

/// Create a new path builder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_new() -> *mut sk_pathbuilder_t {
    catch_panic(|| RefCounted::new(PathBuilder::new()))
}

/// Increment the reference count of a path builder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_ref(builder: *mut sk_pathbuilder_t) {
    catch_panic_void(|| RefCounted::ref_ptr(builder));
}

/// Decrement the reference count of a path builder (alias for unref).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_delete(builder: *mut sk_pathbuilder_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(builder);
    });
}

/// Decrement the reference count of a path builder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_unref(builder: *mut sk_pathbuilder_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(builder);
    });
}

/// Move to a point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_move_to(builder: *mut sk_pathbuilder_t, x: f32, y: f32) {
    catch_panic_void(|| {
        if let Some(b) = RefCounted::get_mut(builder) {
            b.move_to(x, y);
        }
    });
}

/// Line to a point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_line_to(builder: *mut sk_pathbuilder_t, x: f32, y: f32) {
    catch_panic_void(|| {
        if let Some(b) = RefCounted::get_mut(builder) {
            b.line_to(x, y);
        }
    });
}

/// Quadratic bezier to a point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_quad_to(
    builder: *mut sk_pathbuilder_t,
    cx: f32,
    cy: f32,
    x: f32,
    y: f32,
) {
    catch_panic_void(|| {
        if let Some(b) = RefCounted::get_mut(builder) {
            b.quad_to(cx, cy, x, y);
        }
    });
}

/// Cubic bezier to a point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_cubic_to(
    builder: *mut sk_pathbuilder_t,
    c1x: f32,
    c1y: f32,
    c2x: f32,
    c2y: f32,
    x: f32,
    y: f32,
) {
    catch_panic_void(|| {
        if let Some(b) = RefCounted::get_mut(builder) {
            b.cubic_to(c1x, c1y, c2x, c2y, x, y);
        }
    });
}

/// Close the path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_close(builder: *mut sk_pathbuilder_t) {
    catch_panic_void(|| {
        if let Some(b) = RefCounted::get_mut(builder) {
            b.close();
        }
    });
}

/// Add a rectangle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_add_rect(
    builder: *mut sk_pathbuilder_t,
    rect: *const sk_rect_t,
) {
    catch_panic_void(|| {
        if let (Some(b), Some(r)) = (RefCounted::get_mut(builder), rect.as_ref()) {
            b.add_rect(&Rect::from(*r));
        }
    });
}

/// Add an oval.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_add_oval(
    builder: *mut sk_pathbuilder_t,
    rect: *const sk_rect_t,
) {
    catch_panic_void(|| {
        if let (Some(b), Some(r)) = (RefCounted::get_mut(builder), rect.as_ref()) {
            b.add_oval(&Rect::from(*r));
        }
    });
}

/// Add a circle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_add_circle(
    builder: *mut sk_pathbuilder_t,
    cx: f32,
    cy: f32,
    radius: f32,
) {
    catch_panic_void(|| {
        if let Some(b) = RefCounted::get_mut(builder) {
            b.add_circle(cx, cy, radius);
        }
    });
}

/// Build the path and reset the builder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_detach(builder: *mut sk_pathbuilder_t) -> *mut sk_path_t {
    catch_panic(|| {
        RefCounted::get_mut(builder).map_or(ptr::null_mut(), |b| {
            let path = std::mem::replace(b, PathBuilder::new()).build();
            RefCounted::new(path)
        })
    })
}

/// Build the path without resetting the builder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_pathbuilder_snapshot(
    builder: *const sk_pathbuilder_t,
) -> *mut sk_path_t {
    catch_panic(|| {
        RefCounted::get_ref(builder).map_or(ptr::null_mut(), |b| RefCounted::new(b.clone().build()))
    })
}

// =============================================================================
// Matrix API
// =============================================================================

/// Set matrix to identity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_set_identity(matrix: *mut sk_matrix_t) {
    catch_panic_void(|| {
        if let Some(m) = matrix.as_mut() {
            *m = sk_matrix_t::default();
        }
    });
}

/// Set matrix to translate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_set_translate(matrix: *mut sk_matrix_t, dx: f32, dy: f32) {
    catch_panic_void(|| {
        if let Some(m) = matrix.as_mut() {
            *m = Matrix::translate(dx, dy).into();
        }
    });
}

/// Set matrix to scale.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_set_scale(matrix: *mut sk_matrix_t, sx: f32, sy: f32) {
    catch_panic_void(|| {
        if let Some(m) = matrix.as_mut() {
            *m = Matrix::scale(sx, sy).into();
        }
    });
}

/// Set matrix to rotate (degrees).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_set_rotate(matrix: *mut sk_matrix_t, degrees: f32) {
    catch_panic_void(|| {
        if let Some(m) = matrix.as_mut() {
            let radians = degrees.to_radians();
            *m = Matrix::rotate(radians).into();
        }
    });
}

/// Concatenate two matrices: `*result = a * b`.
///
/// `result` may alias `a` and/or `b` (in-place concat is a supported usage,
/// mirroring `SkMatrix::preConcat`/`postConcat` call patterns). Both inputs
/// are copied into owned locals via [`ptr::read`] *before* anything is
/// written through `result`, so no `&`/`&mut` reference ever aliases the
/// same memory — forming `result.as_mut()` while `a`/`b` were still live
/// `&`-refs into the same allocation would be immediate UB when `result`
/// aliases an input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_concat(
    result: *mut sk_matrix_t,
    a: *const sk_matrix_t,
    b: *const sk_matrix_t,
) {
    catch_panic_void(|| {
        if result.is_null() || a.is_null() || b.is_null() {
            return;
        }
        let ma: Matrix = ptr::read(a).into();
        let mb: Matrix = ptr::read(b).into();
        let out: sk_matrix_t = ma.concat(&mb).into();
        ptr::write(result, out);
    });
}

/// Map a point through a matrix.
///
/// `result` may alias `point` (in-place mapping). `matrix`/`point` are read
/// via [`ptr::read`] before writing through `result`; see
/// [`sk_matrix_concat`] for why that ordering matters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_map_point(
    matrix: *const sk_matrix_t,
    point: *const sk_point_t,
    result: *mut sk_point_t,
) {
    catch_panic_void(|| {
        if matrix.is_null() || point.is_null() || result.is_null() {
            return;
        }
        let mat: Matrix = ptr::read(matrix).into();
        let pt: Point = ptr::read(point).into();
        let out: sk_point_t = mat.map_point(pt).into();
        ptr::write(result, out);
    });
}

/// Invert a matrix into `result`. Returns true on success, false if the
/// matrix is singular (non-invertible); on failure `result` is unchanged.
///
/// `result` may alias `matrix` (in-place inversion); `matrix` is read via
/// [`ptr::read`] before writing through `result`; see [`sk_matrix_concat`]
/// for why that ordering matters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_invert(
    matrix: *const sk_matrix_t,
    result: *mut sk_matrix_t,
) -> bool {
    catch_panic(|| {
        if matrix.is_null() || result.is_null() {
            return false;
        }
        let mat: Matrix = ptr::read(matrix).into();
        mat.invert().is_some_and(|inv| {
            let out: sk_matrix_t = inv.into();
            ptr::write(result, out);
            true
        })
    })
}

/// Return true if the matrix is the identity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_is_identity(matrix: *const sk_matrix_t) -> bool {
    catch_panic(|| {
        let Some(m) = matrix.as_ref() else {
            return false;
        };
        let mat: Matrix = (*m).into();
        mat.is_identity()
    })
}

/// Return the determinant of the matrix.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix_determinant(matrix: *const sk_matrix_t) -> f32 {
    catch_panic(|| {
        let Some(m) = matrix.as_ref() else {
            return 0.0;
        };
        let mat: Matrix = (*m).into();
        mat.determinant()
    })
}

// =============================================================================
// Utility functions
// =============================================================================

/// Get the library version as a NUL-terminated static string.
///
/// Sourced from the crate's `Cargo.toml` version (`CARGO_PKG_VERSION`) at
/// compile time, so it always matches the shipped `skia-rs-ffi` release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_version() -> *const c_char {
    static VERSION: &str = concat!("skia-rs ", env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr().cast::<c_char>()
}

/// Check if the library is available.
#[unsafe(no_mangle)]
pub const unsafe extern "C" fn sk_is_available() -> bool {
    true
}

// =============================================================================
// Surface drawing helpers (simplified path; prefer sk_canvas_* for parity)
// =============================================================================

/// The [`locked_surfaces`] set, usable even if another thread panicked
/// while holding its mutex (the set stays consistent: every update is a
/// single insert or remove), so a stale entry can never outlive its surface.
fn locked_surface_set() -> std::sync::MutexGuard<'static, std::collections::HashSet<usize>> {
    locked_surfaces()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Global set of surface pointer addresses currently locked via
/// [`sk_surface_lock_canvas`]. Backs the documented lock contract: only one
/// canvas may be held per surface at a time, and the `sk_surface_draw_*` /
/// [`sk_surface_clear`] convenience helpers (which draw directly on the
/// surface, bypassing the locked canvas) are rejected while a lock is held.
/// Access it through [`locked_surface_set`].
fn locked_surfaces() -> &'static std::sync::Mutex<std::collections::HashSet<usize>> {
    static LOCKED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<usize>>> =
        std::sync::OnceLock::new();
    LOCKED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// Returns true if `surface` currently has an outstanding lock from
/// [`sk_surface_lock_canvas`].
unsafe fn is_surface_locked(surface: *const sk_surface_t) -> bool {
    !surface.is_null() && locked_surface_set().contains(&(surface as usize))
}

/// Clear a surface with a color. No-op while the surface is locked (see
/// [`sk_surface_lock_canvas`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_clear(surface: *mut sk_surface_t, color: sk_color_t) {
    catch_panic_void(|| {
        if is_surface_locked(surface) {
            return;
        }
        if let Some(s) = RefCounted::get_mut(surface) {
            let mut canvas = s.raster_canvas();
            canvas.clear(Color(color));
        }
    });
}

/// Draw a rect on a surface. No-op while the surface is locked (see
/// [`sk_surface_lock_canvas`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_draw_rect(
    surface: *mut sk_surface_t,
    rect: *const sk_rect_t,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        if is_surface_locked(surface) {
            return;
        }
        if let (Some(s), Some(r), Some(p)) = (
            RefCounted::get_mut(surface),
            rect.as_ref(),
            RefCounted::get_ref(paint),
        ) {
            let mut canvas = s.raster_canvas();
            canvas.draw_rect(&Rect::from(*r), p);
        }
    });
}

/// Draw a circle on a surface. No-op while the surface is locked (see
/// [`sk_surface_lock_canvas`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_draw_circle(
    surface: *mut sk_surface_t,
    cx: f32,
    cy: f32,
    radius: f32,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        if is_surface_locked(surface) {
            return;
        }
        if let (Some(s), Some(p)) = (RefCounted::get_mut(surface), RefCounted::get_ref(paint)) {
            let mut canvas = s.raster_canvas();
            canvas.draw_circle(Point::new(cx, cy), radius, p);
        }
    });
}

/// Draw a path on a surface. No-op while the surface is locked (see
/// [`sk_surface_lock_canvas`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_draw_path(
    surface: *mut sk_surface_t,
    path: *const sk_path_t,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        if is_surface_locked(surface) {
            return;
        }
        if let (Some(s), Some(path), Some(p)) = (
            RefCounted::get_mut(surface),
            RefCounted::get_ref(path),
            RefCounted::get_ref(paint),
        ) {
            let mut canvas = s.raster_canvas();
            canvas.draw_path(path, p);
        }
    });
}

/// Draw a line on a surface. No-op while the surface is locked (see
/// [`sk_surface_lock_canvas`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_draw_line(
    surface: *mut sk_surface_t,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        if is_surface_locked(surface) {
            return;
        }
        if let (Some(s), Some(p)) = (RefCounted::get_mut(surface), RefCounted::get_ref(paint)) {
            let mut canvas = s.raster_canvas();
            canvas.draw_line(Point::new(x0, y0), Point::new(x1, y1), p);
        }
    });
}

// =============================================================================
// Canvas API (borrowed from a surface)
// =============================================================================

/// Opaque canvas handle.
///
/// A canvas is an explicit raster drawing context borrowed from a surface.
/// Call [`sk_surface_lock_canvas`] to acquire one, issue draw/transform/clip
/// calls, then release with [`sk_canvas_release`]. The canvas retains
/// exclusive mutable access to the backing pixel buffer while locked.
pub struct sk_canvas_t {
    /// Non-null while locked; dropping this box releases the borrow.
    buffer: *mut PixelBuffer,
    width: i32,
    height: i32,
    state: CanvasState,
    /// Identity (pointer address) of the [`sk_surface_t`] this canvas was
    /// locked from, used to release the entry in [`locked_surfaces`] on
    /// [`sk_canvas_release`]. 0 for canvases not tied to a surface lock.
    locked_surface: usize,
}

/// Clip entry retained across the FFI canvas's transient [`Canvas<'_>`] instances.
///
/// The FFI canvas rebuilds a new [`skia_rs_canvas::Canvas`] for every draw call,
/// so any clip state must be persisted here and replayed before each draw.
/// Each entry keeps the matrix it was set under: like `SkCanvas`, a clip is
/// fixed in device space when it is made, so later transforms do not move it.
#[derive(Clone)]
enum ClipEntry {
    Rect {
        rect: Rect,
        op: ClipOp,
        anti_alias: bool,
        matrix: Matrix,
    },
    Path {
        path: Box<Path>,
        op: ClipOp,
        anti_alias: bool,
        matrix: Matrix,
    },
}

/// An offscreen layer pushed by [`sk_canvas_save_layer`]: draws go into its
/// pixels until the matching restore composites them onto `parent`.
struct FfiLayer {
    /// The layer's premultiplied pixels (owned; from `Box::into_raw`, null
    /// once taken).
    pixels: *mut PixelBuffer,
    /// The layer's top-left in device space.
    origin: (i32, i32),
    /// The area draws into the layer may cover, in device space (the
    /// `bounds` hint limits content, not the filtered result).
    content: skia_rs_core::IRect,
    /// The draw target to return to on restore.
    parent: *mut PixelBuffer,
    /// The layer paint (alpha, blend mode, color and image filters).
    paint: Option<Paint>,
}

impl FfiLayer {
    /// Takes ownership of the layer's pixels.
    fn take_pixels(&mut self) -> Option<PixelBuffer> {
        let pixels = std::mem::replace(&mut self.pixels, ptr::null_mut());
        // SAFETY: `pixels` came from `Box::into_raw` and is taken only once.
        (!pixels.is_null()).then(|| *unsafe { Box::from_raw(pixels) })
    }
}

impl Drop for FfiLayer {
    fn drop(&mut self) {
        drop(self.take_pixels());
    }
}

/// Snapshot of a [`CanvasState`] pushed by [`sk_canvas_save`] or
/// [`sk_canvas_save_layer`] (which also carries the layer).
struct CanvasSaveFrame {
    matrix: Matrix,
    clip_len: usize,
    layer: Option<FfiLayer>,
}

struct CanvasState {
    save_count: usize,
    matrix: Matrix,
    stack: Vec<CanvasSaveFrame>,
    /// Cumulative clip stack; new entries are pushed on clip calls and
    /// truncated back to [`CanvasSaveFrame::clip_len`] on restore.
    clips: Vec<ClipEntry>,
}

impl sk_canvas_t {
    unsafe fn canvas(&mut self) -> skia_rs_canvas::Canvas<'_> {
        let buf = &mut *self.buffer;
        skia_rs_canvas::Canvas::new_raster(buf)
    }

    /// Build a fresh [`Canvas<'_>`] with the FFI state's transform and clip
    /// stack pre-applied. All draw functions go through this to ensure the
    /// transient canvas sees the same clip/matrix the caller has configured.
    ///
    /// The clip list is cloned because `self.canvas()` takes `&mut self`
    /// and the transient `Canvas<'_>` returned borrows the pixel buffer
    /// exclusively. Clip-rect entries are cheap; clip-path entries clone
    /// the `Path` (which holds `Vec<Verb>` + `Vec<Point>`) — usually small.
    ///
    /// Inside a layer only the clips set since the layer was pushed apply
    /// (Skia's layer device starts wide open; the outer clip applies when
    /// the layer is composited), and device coordinates are shifted to the
    /// layer's origin.
    unsafe fn canvas_with_state(&mut self) -> skia_rs_canvas::Canvas<'_> {
        use skia_rs_core::cast::scalar_from_i32;

        let (clip_base, (ox, oy)) = self.target();
        let base = Matrix::translate(-scalar_from_i32(ox), -scalar_from_i32(oy));
        let transform = base.concat(&self.state.matrix);
        let clips = self.state.clips[clip_base.min(self.state.clips.len())..].to_vec();
        let content = self
            .state
            .stack
            .iter()
            .rev()
            .find_map(|f| f.layer.as_ref().map(|l| l.content));
        let mut c = self.canvas();
        if let Some(content) = content {
            c.set_matrix(&base);
            c.clip_rect(
                &Rect::new(
                    scalar_from_i32(content.left),
                    scalar_from_i32(content.top),
                    scalar_from_i32(content.right),
                    scalar_from_i32(content.bottom),
                ),
                ClipOp::Intersect,
                false,
            );
        }
        for entry in &clips {
            match entry {
                ClipEntry::Rect {
                    rect,
                    op,
                    anti_alias,
                    matrix,
                } => {
                    c.set_matrix(&base.concat(matrix));
                    c.clip_rect(rect, *op, *anti_alias);
                }
                ClipEntry::Path {
                    path,
                    op,
                    anti_alias,
                    matrix,
                } => {
                    c.set_matrix(&base.concat(matrix));
                    c.clip_path(path.as_ref(), *op, *anti_alias);
                }
            }
        }
        c.set_matrix(&transform);
        c
    }

    /// The current draw target: the index of the first clip that applies
    /// to it, and its origin in device space (the innermost layer's, or
    /// the surface's).
    fn target(&self) -> (usize, (i32, i32)) {
        self.state
            .stack
            .iter()
            .rev()
            .find_map(|f| f.layer.as_ref().map(|l| (f.clip_len, l.origin)))
            .unwrap_or((0, (0, 0)))
    }

    /// Pops one save; a layer is composited onto its parent through the
    /// parent's clip, like `SkCanvas::internalRestore`.
    unsafe fn restore_one(&mut self) {
        let Some(mut frame) = self.state.stack.pop() else {
            return;
        };
        self.state.matrix = frame.matrix;
        self.state.clips.truncate(frame.clip_len);
        self.state.save_count = self.state.save_count.saturating_sub(1);
        if let Some(mut layer) = frame.layer.take() {
            self.buffer = layer.parent;
            let Some(pixels) = layer.take_pixels() else {
                return;
            };
            let (_, (px, py)) = self.target();
            let origin = (layer.origin.0 - px, layer.origin.1 - py);
            self.canvas_with_state()
                .composite_layer_pixels(pixels, origin, layer.paint.as_ref());
        }
    }
}

/// Acquire a canvas borrowed from the given surface.
///
/// The returned canvas is valid until [`sk_canvas_release`] is called. It
/// holds a reference to `surface` until then, so unreffing the surface
/// while it is locked does not free it under the canvas. Only one canvas may be held per surface at
/// a time — attempting to lock twice returns null. While a surface is
/// locked, the `sk_surface_draw_*`/`sk_surface_clear` convenience helpers
/// (which bypass the locked canvas and draw directly on the surface) are
/// rejected (no-op) to avoid two independent mutable views of the same
/// pixel buffer.
///
/// # Safety
/// `surface` must be null or a live surface handle. The returned canvas must
/// be released exactly once with [`sk_canvas_release`] and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_surface_lock_canvas(surface: *mut sk_surface_t) -> *mut sk_canvas_t {
    catch_panic(|| {
        if surface.is_null() {
            return ptr::null_mut();
        }
        let key = surface as usize;
        if !locked_surface_set().insert(key) {
            // Already locked.
            return ptr::null_mut();
        }
        let Some(s) = RefCounted::get_mut(surface) else {
            locked_surface_set().remove(&key);
            return ptr::null_mut();
        };
        let width = s.width();
        let height = s.height();
        let buffer: *mut PixelBuffer = s.pixel_buffer_mut();
        // Keep the surface alive while locked: the canvas draws into its
        // buffer, and release may composite open layers into it.
        RefCounted::ref_ptr(surface);
        Box::into_raw(Box::new(sk_canvas_t {
            buffer,
            width,
            height,
            state: CanvasState {
                save_count: 0,
                matrix: Matrix::identity(),
                stack: Vec::new(),
                clips: Vec::new(),
            },
            locked_surface: key,
        }))
    })
}

/// Release a canvas acquired via [`sk_surface_lock_canvas`].
///
/// Like `~SkCanvas`, open saves are unwound first, compositing any open
/// layer into the surface.
///
/// # Safety
/// `canvas` must be null or a canvas from [`sk_surface_lock_canvas`] that
/// has not been released; it must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_release(canvas: *mut sk_canvas_t) {
    catch_panic_void(|| {
        if !canvas.is_null() {
            // Unlocks the surface and drops the canvas's reference to it
            // even if compositing below panics. Declared before the canvas
            // so the canvas (which points into the surface's pixels) drops
            // first.
            let _unlock = SurfaceUnlock((*canvas).locked_surface);
            let mut c = Box::from_raw(canvas);
            // Like ~SkCanvas: unwind open saves, compositing open layers.
            while !c.state.stack.is_empty() {
                c.restore_one();
            }
        }
    });
}

/// Releases a locked surface on drop: removes its [`locked_surfaces`] entry
/// and the reference [`sk_surface_lock_canvas`] took. 0 means no surface.
struct SurfaceUnlock(usize);

impl Drop for SurfaceUnlock {
    fn drop(&mut self) {
        if self.0 == 0 {
            return;
        }
        // Remove the entry even if another thread poisoned the mutex: a
        // stale address would make a later surface there look locked.
        locked_surface_set().remove(&self.0);
        // SAFETY: the lock took a reference, so the surface is alive.
        unsafe { RefCounted::unref_ptr(self.0 as *mut sk_surface_t) };
    }
}

/// Get the width of the canvas.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_get_width(canvas: *const sk_canvas_t) -> i32 {
    catch_panic(|| canvas.as_ref().map_or(0, |c| c.width))
}

/// Get the height of the canvas.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_get_height(canvas: *const sk_canvas_t) -> i32 {
    catch_panic(|| canvas.as_ref().map_or(0, |c| c.height))
}

/// Save the current transformation matrix and clip stack.
///
/// Returns the same value as `SkCanvas::save()`: the save count *before*
/// this call in Skia's 1-based counting (1 for the first save, 2 for a
/// nested one, ...). Returns 0 if `canvas` is null.
///
/// # Safety
/// `canvas` must be null or a live canvas handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_save(canvas: *mut sk_canvas_t) -> i32 {
    catch_panic(|| {
        let Some(c) = canvas.as_mut() else {
            return 0;
        };
        let frame = CanvasSaveFrame {
            matrix: c.state.matrix,
            clip_len: c.state.clips.len(),
            layer: None,
        };
        c.state.stack.push(frame);
        c.state.save_count += 1;
        i32::try_from(c.state.save_count).unwrap_or(i32::MAX)
    })
}

/// Save a layer, like `SkCanvas::saveLayer`.
///
/// Later draws go into an offscreen layer until the matching
/// [`sk_canvas_restore`] (or [`sk_canvas_release`]), which composites it
/// through the clip in effect here using `paint`'s alpha, blend mode, color
/// filter and image filter. `bounds` (local coordinates), if given, clips
/// what is drawn into the layer, as in Skia; the image filter's result may
/// still spread past it, up to the clip. Both may be null. Inside the layer
/// only clips set after this call (and `bounds`) apply, as in Skia.
///
/// Returns the same value as `SkCanvas::saveLayer()` (and
/// [`sk_canvas_save`]): the save count *before* this call in Skia's
/// 1-based counting (1 for the first save, 2 for a nested one, ...).
/// Returns 0 if `canvas` is null.
///
/// # Safety
/// `canvas` must be null or a live locked canvas; `bounds` must be null or
/// point to a valid `sk_rect_t`; `paint` must be null or a live paint
/// handle. The paint is copied, so it may be released afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_save_layer(
    canvas: *mut sk_canvas_t,
    bounds: *const sk_rect_t,
    paint: *const sk_paint_t,
) -> i32 {
    catch_panic(|| {
        use skia_rs_core::cast::scalar_from_i32;

        let Some(c) = canvas.as_mut() else {
            return 0;
        };
        let bounds = bounds.as_ref().map(|b| Rect::from(*b));
        let paint = RefCounted::get_ref(paint).cloned();
        let (_, (ox, oy)) = c.target();
        // The surface, in the coordinates of the current target's pixels:
        // bounds nested filtered layers by the real device, not the parent
        // layer.
        let device = Rect::new(
            scalar_from_i32(-ox),
            scalar_from_i32(-oy),
            scalar_from_i32(c.width - ox),
            scalar_from_i32(c.height - oy),
        );
        let (rect, content, paint) = {
            let inner = c.canvas_with_state();
            let rec = skia_rs_canvas::SaveLayerRec {
                bounds: bounds.as_ref(),
                paint: paint.as_ref(),
                flags: skia_rs_canvas::SaveLayerFlags::NONE,
            };
            let (rect, content) = inner.layer_bounds_and_content_within(&rec, &device);
            let paint = inner.layer_paint(paint.as_ref(), &rect);
            (rect, content, paint)
        };
        let (w, h) = (rect.width().max(0), rect.height().max(0));
        let pixels = Box::into_raw(Box::new(PixelBuffer::new(w, h)));
        let layer = FfiLayer {
            pixels,
            origin: (rect.left + ox, rect.top + oy),
            content: skia_rs_core::IRect::new(
                content.left + ox,
                content.top + oy,
                content.right + ox,
                content.bottom + oy,
            ),
            parent: c.buffer,
            paint,
        };
        c.state.stack.push(CanvasSaveFrame {
            matrix: c.state.matrix,
            clip_len: c.state.clips.len(),
            layer: Some(layer),
        });
        c.buffer = pixels;
        c.state.save_count += 1;
        i32::try_from(c.state.save_count).unwrap_or(i32::MAX)
    })
}

/// Restore the most recently saved matrix and clip, like `SkCanvas::restore`.
///
/// If that save was a [`sk_canvas_save_layer`], its
/// offscreen layer is composited into the layer or surface below, through
/// the outer clip, with the layer paint's alpha, blend mode, color filter
/// and image filter. Does nothing when there is no save to restore.
///
/// # Safety
/// `canvas` must be null or a canvas from [`sk_surface_lock_canvas`] that
/// has not been released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_restore(canvas: *mut sk_canvas_t) {
    catch_panic_void(|| {
        let Some(c) = canvas.as_mut() else {
            return;
        };
        c.restore_one();
    });
}

/// Concatenate a matrix onto the canvas's current transform.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_concat(canvas: *mut sk_canvas_t, matrix: *const sk_matrix_t) {
    catch_panic_void(|| {
        let Some(c) = canvas.as_mut() else { return };
        let Some(m) = matrix.as_ref() else { return };
        let mat: Matrix = (*m).into();
        c.state.matrix = c.state.matrix.concat(&mat);
    });
}

/// Translate the canvas coordinate system.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_translate(canvas: *mut sk_canvas_t, dx: f32, dy: f32) {
    catch_panic_void(|| {
        if let Some(c) = canvas.as_mut() {
            c.state.matrix = c.state.matrix.concat(&Matrix::translate(dx, dy));
        }
    });
}

/// Scale the canvas coordinate system.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_scale(canvas: *mut sk_canvas_t, sx: f32, sy: f32) {
    catch_panic_void(|| {
        if let Some(c) = canvas.as_mut() {
            c.state.matrix = c.state.matrix.concat(&Matrix::scale(sx, sy));
        }
    });
}

/// Rotate the canvas coordinate system (degrees).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_rotate(canvas: *mut sk_canvas_t, degrees: f32) {
    catch_panic_void(|| {
        if let Some(c) = canvas.as_mut() {
            let r = degrees.to_radians();
            c.state.matrix = c.state.matrix.concat(&Matrix::rotate(r));
        }
    });
}

/// Clear the canvas with a color.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_clear(canvas: *mut sk_canvas_t, color: sk_color_t) {
    catch_panic_void(|| {
        let Some(c) = canvas.as_mut() else { return };
        // Route through `canvas_with_state` (not the bare `canvas()`) so
        // `clear` respects the canvas's current clip stack, matching
        // upstream `SkCanvas::clear` semantics.
        let mut inner = c.canvas_with_state();
        inner.clear(Color(color));
    });
}

/// Draw a rect on the canvas, honoring the canvas's current transform and
/// clip stack.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_draw_rect(
    canvas: *mut sk_canvas_t,
    rect: *const sk_rect_t,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        let Some(c) = canvas.as_mut() else { return };
        let Some(r) = rect.as_ref() else { return };
        // Copy paint out of the ref-counted cell so it doesn't alias `c`
        // while we mutably borrow the canvas.
        let Some(p) = RefCounted::get_ref(paint).cloned() else {
            return;
        };
        let mut inner = c.canvas_with_state();
        inner.draw_rect(&Rect::from(*r), &p);
    });
}

/// Draw a path on the canvas, honoring the canvas's current transform and
/// clip stack.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_draw_path(
    canvas: *mut sk_canvas_t,
    path: *const sk_path_t,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        let Some(c) = canvas.as_mut() else { return };
        let Some(owned_path) = RefCounted::get_ref(path).cloned() else {
            return;
        };
        let Some(p) = RefCounted::get_ref(paint).cloned() else {
            return;
        };
        let mut inner = c.canvas_with_state();
        inner.draw_path(&owned_path, &p);
    });
}

/// Intersect or subtract a rectangle from the clip.
///
/// `op` follows upstream `SkClipOp`: 0 = Difference, 1 = Intersect (see
/// `decode_clip_op`). Returns false if the canvas handle is null or `op`
/// is out of range; otherwise true.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_clip_rect(
    canvas: *mut sk_canvas_t,
    rect: *const sk_rect_t,
    op: sk_clip_op_t,
    anti_alias: bool,
) -> bool {
    catch_panic(|| {
        let Some(c) = canvas.as_mut() else {
            return false;
        };
        let Some(r) = rect.as_ref() else {
            return false;
        };
        let Some(clip_op) = decode_clip_op(op) else {
            return false;
        };
        c.state.clips.push(ClipEntry::Rect {
            rect: Rect::from(*r),
            op: clip_op,
            anti_alias,
            matrix: c.state.matrix,
        });
        true
    })
}

/// Intersect or subtract a path from the clip.
///
/// `op` follows upstream `SkClipOp`; see `decode_clip_op`. Returns false
/// if `op` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_clip_path(
    canvas: *mut sk_canvas_t,
    path: *const sk_path_t,
    op: sk_clip_op_t,
    anti_alias: bool,
) -> bool {
    catch_panic(|| {
        let Some(c) = canvas.as_mut() else {
            return false;
        };
        let Some(p) = RefCounted::get_ref(path) else {
            return false;
        };
        let Some(clip_op) = decode_clip_op(op) else {
            return false;
        };
        c.state.clips.push(ClipEntry::Path {
            path: Box::new(p.clone()),
            op: clip_op,
            anti_alias,
            matrix: c.state.matrix,
        });
        true
    })
}

/// Fill `out` with the device-space integer clip bounds of this canvas.
///
/// Returns true on success. With no clips `out` is the device bounds
/// `[0, 0, width, height]` (inside a layer, the layer's content bounds:
/// its `bounds` hint, if one was given); an empty clip gives
/// `[0, 0, 0, 0]`, as `SkCanvas::getDeviceClipBounds`.
///
/// # Safety
/// `canvas` must be null or a live canvas handle, and `out` must be null or
/// point to writable memory for one `sk_irect_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_get_clip_ibounds(
    canvas: *mut sk_canvas_t,
    out: *mut sk_irect_t,
) -> bool {
    catch_panic(|| {
        let Some(c) = canvas.as_mut() else {
            return false;
        };
        let Some(dst) = out.as_mut() else {
            return false;
        };
        // Inside a layer the transient canvas is in the layer's pixels:
        // shift back to device space, as getDeviceClipBounds reports.
        let (_, (ox, oy)) = c.target();
        let inner = c.canvas_with_state();
        let bounds = inner.clip_bounds();
        // An empty clip is (0, 0, 0, 0), as getDeviceClipBounds reports it,
        // not a zero-size rect at the layer origin.
        *dst = if bounds.is_empty() {
            sk_irect_t {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            }
        } else {
            sk_irect_t {
                left: skia_rs_core::cast::floor_to_i32(bounds.left) + ox,
                top: skia_rs_core::cast::floor_to_i32(bounds.top) + oy,
                right: skia_rs_core::cast::ceil_to_i32(bounds.right) + ox,
                bottom: skia_rs_core::cast::ceil_to_i32(bounds.bottom) + oy,
            }
        };
        true
    })
}

// =============================================================================
// Image API (Reference Counted)
// =============================================================================

/// Reference counted image type.
pub type sk_image_t = RefCounted<skia_rs_codec::Image>;

/// Decode an encoded image (PNG/JPEG/…) from a byte buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_from_encoded(data: *const u8, len: usize) -> *mut sk_image_t {
    catch_panic(|| {
        if data.is_null() || len == 0 {
            return ptr::null_mut();
        }
        let bytes = slice::from_raw_parts(data, len);
        skia_rs_codec::decode_image(bytes).map_or(ptr::null_mut(), RefCounted::new)
    })
}

/// Create an image filled with a single color.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_from_color(
    width: i32,
    height: i32,
    color: sk_color_t,
) -> *mut sk_image_t {
    catch_panic(|| {
        skia_rs_codec::Image::from_color(width, height, color)
            .map_or(ptr::null_mut(), RefCounted::new)
    })
}

/// Increment the reference count of an image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_ref(image: *mut sk_image_t) {
    catch_panic_void(|| RefCounted::ref_ptr(image));
}

/// Decrement the reference count of an image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_unref(image: *mut sk_image_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(image);
    });
}

/// Get the reference count of an image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_get_refcnt(image: *const sk_image_t) -> u32 {
    catch_panic(|| RefCounted::get_count(image))
}

/// Get the image width.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_get_width(image: *const sk_image_t) -> i32 {
    catch_panic(|| RefCounted::get_ref(image).map_or(0, skia_rs_canvas::Image::width))
}

/// Get the image height.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_get_height(image: *const sk_image_t) -> i32 {
    catch_panic(|| RefCounted::get_ref(image).map_or(0, skia_rs_canvas::Image::height))
}

/// Encode an image as PNG into a caller-allocated buffer.
///
/// Returns the number of bytes written. If `dst` is null or `dst_len` is 0,
/// returns the required buffer size without writing. Returns 0 on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_image_encode_png(
    image: *const sk_image_t,
    dst: *mut u8,
    dst_len: usize,
) -> usize {
    catch_panic(|| {
        use skia_rs_codec::ImageEncoder;
        let Some(img) = RefCounted::get_ref(image) else {
            return 0;
        };
        let encoder = skia_rs_codec::PngEncoder::new();
        let Ok(bytes) = encoder.encode_bytes(img) else {
            return 0;
        };
        if dst.is_null() || dst_len == 0 {
            return bytes.len();
        }
        if dst_len < bytes.len() {
            return 0;
        }
        ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        bytes.len()
    })
}

// =============================================================================
// Typeface / Font API (Reference Counted)
// =============================================================================

/// Reference counted typeface type.
pub type sk_typeface_t = RefCounted<TypefaceRef>;

/// Reference counted font type (typeface + size + options).
pub type sk_font_t = RefCounted<Font>;

/// Create the default typeface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_default() -> *mut sk_typeface_t {
    catch_panic(|| RefCounted::new(Arc::new(Typeface::default_typeface())))
}

/// Load a typeface from raw font file data (TTF/OTF).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_from_data(data: *const u8, len: usize) -> *mut sk_typeface_t {
    catch_panic(|| {
        if data.is_null() || len == 0 {
            return ptr::null_mut();
        }
        let buf = slice::from_raw_parts(data, len).to_vec();
        Typeface::from_data(buf).map_or(ptr::null_mut(), |t| RefCounted::new(Arc::new(t)))
    })
}

/// Increment typeface refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_ref(tf: *mut sk_typeface_t) {
    catch_panic_void(|| RefCounted::ref_ptr(tf));
}

/// Decrement typeface refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_unref(tf: *mut sk_typeface_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(tf);
    });
}

/// Get typeface refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_get_refcnt(tf: *const sk_typeface_t) -> u32 {
    catch_panic(|| RefCounted::get_count(tf))
}

/// Get the typeface's units-per-em.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_units_per_em(tf: *const sk_typeface_t) -> u16 {
    catch_panic(|| RefCounted::get_ref(tf).map_or(0, |t| t.units_per_em()))
}

/// Get the typeface's glyph count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_typeface_glyph_count(tf: *const sk_typeface_t) -> u16 {
    catch_panic(|| RefCounted::get_ref(tf).map_or(0, |t| t.glyph_count()))
}

/// Create a font from a typeface and size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_font_new(tf: *const sk_typeface_t, size: f32) -> *mut sk_font_t {
    catch_panic(|| {
        let Some(t) = RefCounted::get_ref(tf) else {
            return ptr::null_mut();
        };
        RefCounted::new(Font::new(t.clone(), size))
    })
}

/// Increment font refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_font_ref(font: *mut sk_font_t) {
    catch_panic_void(|| RefCounted::ref_ptr(font));
}

/// Decrement font refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_font_unref(font: *mut sk_font_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(font);
    });
}

/// Get font size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_font_get_size(font: *const sk_font_t) -> f32 {
    catch_panic(|| RefCounted::get_ref(font).map_or(0.0, skia_rs_text::Font::size))
}

/// Measure a UTF-8 text string at the font's current size. Returns the
/// advance width in pixels; negative return indicates an invalid font or
/// invalid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_font_measure_text(
    font: *const sk_font_t,
    text: *const c_char,
    text_len: usize,
) -> f32 {
    catch_panic(|| {
        if text.is_null() {
            return -1.0;
        }
        let Some(f) = RefCounted::get_ref(font) else {
            return -1.0;
        };
        let slice = slice::from_raw_parts(text.cast::<u8>(), text_len);
        let Ok(s) = std::str::from_utf8(slice) else {
            return -1.0;
        };
        f.measure_text(s)
    })
}

// =============================================================================
// Shader API (Reference Counted)
// =============================================================================

/// Reference counted shader type.
pub type sk_shader_t = RefCounted<ShaderRef>;

/// Increment shader refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_shader_ref(shader: *mut sk_shader_t) {
    catch_panic_void(|| RefCounted::ref_ptr(shader));
}

/// Decrement shader refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_shader_unref(shader: *mut sk_shader_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(shader);
    });
}

/// Get shader refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_shader_get_refcnt(shader: *const sk_shader_t) -> u32 {
    catch_panic(|| RefCounted::get_count(shader))
}

/// Create a linear gradient shader.
///
/// `colors_len` must equal `positions_len` (or `positions` may be null).
/// `tile_mode` maps to [`TileMode`]: 0=Clamp, 1=Repeat, 2=Mirror, 3=Decal.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_shader_new_linear_gradient(
    start: sk_point_t,
    end: sk_point_t,
    colors: *const sk_color4f_t,
    positions: *const f32,
    count: usize,
    tile_mode: u32,
) -> *mut sk_shader_t {
    catch_panic(|| {
        if colors.is_null() || count == 0 {
            return ptr::null_mut();
        }
        let colors_slice = slice::from_raw_parts(colors, count);
        let colors_vec: Vec<Color4f> = colors_slice.iter().copied().map(Into::into).collect();
        let positions_vec = if positions.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(positions, count).to_vec())
        };
        let Some(tile_mode) = decode_tile_mode(tile_mode) else {
            return ptr::null_mut();
        };
        let grad = LinearGradient::new(
            start.into(),
            end.into(),
            colors_vec,
            positions_vec,
            tile_mode,
        );
        let shader: ShaderRef = Arc::new(grad);
        RefCounted::new(shader)
    })
}

/// Create a radial gradient shader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_shader_new_radial_gradient(
    center: sk_point_t,
    radius: f32,
    colors: *const sk_color4f_t,
    positions: *const f32,
    count: usize,
    tile_mode: u32,
) -> *mut sk_shader_t {
    catch_panic(|| {
        if colors.is_null() || count == 0 {
            return ptr::null_mut();
        }
        let colors_slice = slice::from_raw_parts(colors, count);
        let colors_vec: Vec<Color4f> = colors_slice.iter().copied().map(Into::into).collect();
        let positions_vec = if positions.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(positions, count).to_vec())
        };
        let Some(tile_mode) = decode_tile_mode(tile_mode) else {
            return ptr::null_mut();
        };
        let grad = RadialGradient::new(center.into(), radius, colors_vec, positions_vec, tile_mode);
        let shader: ShaderRef = Arc::new(grad);
        RefCounted::new(shader)
    })
}

/// Create a sweep gradient shader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_shader_new_sweep_gradient(
    center: sk_point_t,
    start_angle: f32,
    end_angle: f32,
    colors: *const sk_color4f_t,
    positions: *const f32,
    count: usize,
) -> *mut sk_shader_t {
    catch_panic(|| {
        if colors.is_null() || count == 0 {
            return ptr::null_mut();
        }
        let colors_slice = slice::from_raw_parts(colors, count);
        let colors_vec: Vec<Color4f> = colors_slice.iter().copied().map(Into::into).collect();
        let positions_vec = if positions.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(positions, count).to_vec())
        };
        let grad = SweepGradient::new(
            center.into(),
            start_angle,
            end_angle,
            colors_vec,
            positions_vec,
            TileMode::Clamp,
        );
        let shader: ShaderRef = Arc::new(grad);
        RefCounted::new(shader)
    })
}

// =============================================================================
// Color Filter API (Reference Counted)
// =============================================================================

/// Reference counted color filter.
pub type sk_colorfilter_t = RefCounted<ColorFilterRef>;

/// Increment color filter refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorfilter_ref(f: *mut sk_colorfilter_t) {
    catch_panic_void(|| RefCounted::ref_ptr(f));
}

/// Decrement color filter refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorfilter_unref(f: *mut sk_colorfilter_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(f);
    });
}

/// Create an identity color-matrix filter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorfilter_new_identity() -> *mut sk_colorfilter_t {
    catch_panic(|| {
        let cf: ColorFilterRef = Arc::new(ColorMatrixFilter::identity());
        RefCounted::new(cf)
    })
}

/// Create a saturation color filter (1.0 = identity, 0.0 = grayscale).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorfilter_new_saturation(amount: f32) -> *mut sk_colorfilter_t {
    catch_panic(|| {
        let cf: ColorFilterRef = Arc::new(ColorMatrixFilter::saturation(amount));
        RefCounted::new(cf)
    })
}

/// Create a color-matrix filter from 20 floats (row-major 4x5).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorfilter_new_matrix(values: *const f32) -> *mut sk_colorfilter_t {
    catch_panic(|| {
        if values.is_null() {
            return ptr::null_mut();
        }
        let mut m = [0.0f32; 20];
        ptr::copy_nonoverlapping(values, m.as_mut_ptr(), 20);
        let cf: ColorFilterRef = Arc::new(ColorMatrixFilter::new(m));
        RefCounted::new(cf)
    })
}

// =============================================================================
// Mask Filter API (Reference Counted)
// =============================================================================

/// Reference counted mask filter.
pub type sk_maskfilter_t = RefCounted<MaskFilterRef>;

/// Increment mask filter refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_maskfilter_ref(f: *mut sk_maskfilter_t) {
    catch_panic_void(|| RefCounted::ref_ptr(f));
}

/// Decrement mask filter refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_maskfilter_unref(f: *mut sk_maskfilter_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(f);
    });
}

/// Create a blur mask filter with the given sigma (pixels).
///
/// `style` follows [`BlurStyle`]: 0=Normal, 1=Solid, 2=Outer, 3=Inner.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_maskfilter_new_blur(style: u32, sigma: f32) -> *mut sk_maskfilter_t {
    catch_panic(|| {
        let s = match style {
            1 => BlurStyle::Solid,
            2 => BlurStyle::Outer,
            3 => BlurStyle::Inner,
            _ => BlurStyle::Normal,
        };
        let mf: MaskFilterRef = Arc::new(BlurMaskFilter::new(s, sigma));
        RefCounted::new(mf)
    })
}

// =============================================================================
// Image Filter API (Reference Counted)
// =============================================================================

/// Reference counted image filter.
pub type sk_imagefilter_t = RefCounted<ImageFilterRef>;

/// Increment image filter refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_imagefilter_ref(f: *mut sk_imagefilter_t) {
    catch_panic_void(|| RefCounted::ref_ptr(f));
}

/// Decrement image filter refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_imagefilter_unref(f: *mut sk_imagefilter_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(f);
    });
}

/// Create a Gaussian blur image filter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_imagefilter_new_blur(
    sigma_x: f32,
    sigma_y: f32,
    tile_mode: u32,
) -> *mut sk_imagefilter_t {
    catch_panic(|| {
        let Some(tile_mode) = decode_tile_mode(tile_mode) else {
            return ptr::null_mut();
        };
        let filter = skia_rs_paint::BlurImageFilter::new(sigma_x, sigma_y, tile_mode);
        let ifref: ImageFilterRef = Arc::new(filter);
        RefCounted::new(ifref)
    })
}

// =============================================================================
// Matrix44 API (4x4 matrix, column-major)
// =============================================================================

/// C-compatible 4x4 matrix structure (column-major).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct sk_matrix44_t {
    /// Matrix values in column-major order (m[col * 4 + row]).
    pub values: [f32; 16],
}

impl Default for sk_matrix44_t {
    fn default() -> Self {
        Self {
            values: Matrix44::IDENTITY.values,
        }
    }
}

impl From<Matrix44> for sk_matrix44_t {
    fn from(m: Matrix44) -> Self {
        Self { values: m.values }
    }
}

impl From<sk_matrix44_t> for Matrix44 {
    fn from(m: sk_matrix44_t) -> Self {
        Self { values: m.values }
    }
}

/// Create an identity [`sk_matrix44_t`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_new() -> sk_matrix44_t {
    catch_panic(sk_matrix44_t::default)
}

/// Construct a [`sk_matrix44_t`] from 16 column-major floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_from_array(values: *const f32) -> sk_matrix44_t {
    catch_panic(|| {
        if values.is_null() {
            return sk_matrix44_t::default();
        }
        let mut out = [0.0f32; 16];
        ptr::copy_nonoverlapping(values, out.as_mut_ptr(), 16);
        sk_matrix44_t { values: out }
    })
}

/// Map a 2D point through a 4x4 matrix (projects through W).
///
/// `result` may alias `point` (in-place mapping); inputs are read via
/// [`ptr::read`] before writing through `result` — see
/// [`sk_matrix_concat`] for why that ordering matters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_map_point(
    matrix: *const sk_matrix44_t,
    point: *const sk_point_t,
    result: *mut sk_point_t,
) {
    catch_panic_void(|| {
        if matrix.is_null() || point.is_null() || result.is_null() {
            return;
        }
        let mat: Matrix44 = ptr::read(matrix).into();
        let pt: Point = ptr::read(point).into();
        let out: sk_point_t = mat.map_point(pt).into();
        ptr::write(result, out);
    });
}

/// Invert a 4x4 matrix. Returns true on success; `result` is unchanged if
/// the matrix is singular.
///
/// `result` may alias `matrix` (in-place inversion); `matrix` is read via
/// [`ptr::read`] before writing through `result` — see
/// [`sk_matrix_concat`] for why that ordering matters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_invert(
    matrix: *const sk_matrix44_t,
    result: *mut sk_matrix44_t,
) -> bool {
    catch_panic(|| {
        if matrix.is_null() || result.is_null() {
            return false;
        }
        let mat: Matrix44 = ptr::read(matrix).into();
        mat.invert().is_some_and(|inv| {
            let out: sk_matrix44_t = inv.into();
            ptr::write(result, out);
            true
        })
    })
}

/// Return true if the matrix is the identity matrix.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_is_identity(matrix: *const sk_matrix44_t) -> bool {
    catch_panic(|| {
        let Some(m) = matrix.as_ref() else {
            return false;
        };
        let mat: Matrix44 = (*m).into();
        mat.is_identity()
    })
}

/// Return the determinant of the matrix.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_determinant(matrix: *const sk_matrix44_t) -> f32 {
    catch_panic(|| {
        let Some(m) = matrix.as_ref() else {
            return 0.0;
        };
        let mat: Matrix44 = (*m).into();
        mat.determinant()
    })
}

/// Concatenate two 4x4 matrices: `*result = a * b`.
///
/// `result` may alias `a` and/or `b` (in-place concat); inputs are read via
/// [`ptr::read`] before writing through `result` — see
/// [`sk_matrix_concat`] for why that ordering matters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_matrix44_concat(
    result: *mut sk_matrix44_t,
    a: *const sk_matrix44_t,
    b: *const sk_matrix44_t,
) {
    catch_panic_void(|| {
        if result.is_null() || a.is_null() || b.is_null() {
            return;
        }
        let ma: Matrix44 = ptr::read(a).into();
        let mb: Matrix44 = ptr::read(b).into();
        let out: sk_matrix44_t = ma.concat(&mb).into();
        ptr::write(result, out);
    });
}

// =============================================================================
// ColorSpace API (Reference Counted)
// =============================================================================

/// Reference counted [`ColorSpace`].
pub type sk_colorspace_t = RefCounted<ColorSpace>;

/// Create a new sRGB color space.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_new_srgb() -> *mut sk_colorspace_t {
    catch_panic(|| RefCounted::new(ColorSpace::srgb()))
}

/// Create a new linear sRGB color space.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_new_srgb_linear() -> *mut sk_colorspace_t {
    catch_panic(|| RefCounted::new(ColorSpace::srgb_linear()))
}

/// Create a new Display P3 color space.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_new_display_p3() -> *mut sk_colorspace_t {
    catch_panic(|| RefCounted::new(ColorSpace::display_p3()))
}

/// Parse a color space from a raw ICC profile byte buffer. Returns null on
/// failure (malformed profile, missing `acsp` magic, etc.).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_from_icc(
    data: *const u8,
    len: usize,
) -> *mut sk_colorspace_t {
    catch_panic(|| {
        if data.is_null() || len == 0 {
            return ptr::null_mut();
        }
        let bytes = slice::from_raw_parts(data, len);
        let Some(profile) = IccProfile::from_bytes(bytes) else {
            return ptr::null_mut();
        };
        RefCounted::new(profile.color_space().clone())
    })
}

/// Return true if this color space is sRGB.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_is_srgb(cs: *const sk_colorspace_t) -> bool {
    catch_panic(|| RefCounted::get_ref(cs).is_some_and(skia_rs_core::ColorSpace::is_srgb))
}

/// Return true if this color space has a linear transfer function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_is_linear(cs: *const sk_colorspace_t) -> bool {
    catch_panic(|| RefCounted::get_ref(cs).is_some_and(skia_rs_core::ColorSpace::is_linear))
}

/// Increment color space refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_ref(cs: *mut sk_colorspace_t) {
    catch_panic_void(|| RefCounted::ref_ptr(cs));
}

/// Decrement color space refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_colorspace_unref(cs: *mut sk_colorspace_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(cs);
    });
}

// =============================================================================
// TextBlob API (Reference Counted)
// =============================================================================

/// Reference counted [`TextBlob`].
pub type sk_textblob_t = RefCounted<TextBlob>;

/// Build a text blob from a NUL-terminated UTF-8 string. `text` and `font`
/// must be valid, non-null pointers; returns null otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_textblob_make_from_text(
    text: *const c_char,
    text_len: usize,
    font: *const sk_font_t,
) -> *mut sk_textblob_t {
    catch_panic(|| {
        if text.is_null() {
            return ptr::null_mut();
        }
        let Some(f) = RefCounted::get_ref(font) else {
            return ptr::null_mut();
        };
        let slice = slice::from_raw_parts(text.cast::<u8>(), text_len);
        let Ok(s) = std::str::from_utf8(slice) else {
            return ptr::null_mut();
        };
        let blob = TextBlob::from_text(s, f, Point::new(0.0, 0.0));
        RefCounted::new(blob)
    })
}

/// Increment text blob refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_textblob_ref(blob: *mut sk_textblob_t) {
    catch_panic_void(|| RefCounted::ref_ptr(blob));
}

/// Decrement text blob refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_textblob_unref(blob: *mut sk_textblob_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(blob);
    });
}

/// Get text blob bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_textblob_get_bounds(
    blob: *const sk_textblob_t,
    out: *mut sk_rect_t,
) -> bool {
    catch_panic(|| {
        if let (Some(b), Some(dst)) = (RefCounted::get_ref(blob), out.as_mut()) {
            *dst = b.bounds().into();
            true
        } else {
            false
        }
    })
}

/// Draw a text blob on a canvas at `(x, y)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_canvas_draw_text_blob(
    canvas: *mut sk_canvas_t,
    blob: *const sk_textblob_t,
    x: f32,
    y: f32,
    paint: *const sk_paint_t,
) -> bool {
    catch_panic(|| {
        let Some(canvas_ref) = canvas.as_mut() else {
            return false;
        };
        let Some(blob_ref) = RefCounted::get_ref(blob).cloned() else {
            return false;
        };
        let Some(paint_ref) = RefCounted::get_ref(paint).cloned() else {
            return false;
        };
        let mut inner = canvas_ref.canvas_with_state();
        inner.draw_text_blob(&blob_ref, x, y, &paint_ref);
        true
    })
}

// =============================================================================
// Picture / PictureRecorder API
// =============================================================================

/// Reference counted [`Picture`](skia_rs_canvas::Picture).
pub type sk_picture_t = RefCounted<PictureRef>;

/// Opaque picture recorder. Not reference counted — single ownership.
///
/// The FFI recorder does not use the workspace [`PictureRecorder`](skia_rs_canvas::PictureRecorder) directly
/// because that type's recording API returns an unboxed mutable reference
/// that cannot cross the FFI boundary cleanly. Instead we track our own
/// [`DrawCommand`](skia_rs_canvas::DrawCommand) stream and wrap it into a [`Picture`](skia_rs_canvas::Picture) at finish time,
/// calling [`Picture::from_parts`](skia_rs_canvas::Picture::from_parts).
pub struct sk_picture_recorder_t {
    /// Accumulated draw commands while recording. Swapped out on finish.
    commands: Vec<skia_rs_canvas::DrawCommand>,
    /// Cull rect recorded by `begin_recording`.
    cull_rect: Rect,
    /// State of the borrowed recording canvas (if any). Only the save stack
    /// is used (to drop unbalanced restores): transforms are recorded as
    /// commands and geometry stays local, so playback applies them; never
    /// bake the matrix or clips here into recorded geometry.
    state: Option<CanvasState>,
    recording: bool,
    /// Liveness flag shared with every [`sk_recording_canvas_t`] handed out
    /// by [`sk_picture_recorder_begin_recording`]. Cleared *before* the
    /// recorder is freed so outstanding recording-canvas handles can detect
    /// use-after-free without ever dereferencing the (now dangling)
    /// `recorder` pointer — see [`resolve_recording_canvas`].
    alive: Arc<AtomicBool>,
}

/// Create a new picture recorder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_recorder_new() -> *mut sk_picture_recorder_t {
    catch_panic(|| {
        Box::into_raw(Box::new(sk_picture_recorder_t {
            commands: Vec::new(),
            cull_rect: Rect::EMPTY,
            state: None,
            recording: false,
            alive: Arc::new(AtomicBool::new(true)),
        }))
    })
}

/// Destroy a picture recorder.
///
/// Any outstanding [`sk_recording_canvas_t`] handles are left dangling by
/// design (the caller is responsible for releasing them), but every
/// `sk_recording_canvas_*` entry point checks the shared liveness flag
/// before touching the recorder, so calling them after this returns errors
/// out safely instead of dereferencing freed memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_recorder_delete(r: *mut sk_picture_recorder_t) {
    catch_panic_void(|| {
        if !r.is_null() {
            (*r).alive.store(false, Ordering::Release);
            drop(Box::from_raw(r));
        }
    });
}

/// Recording canvas handle. Returned by
/// [`sk_picture_recorder_begin_recording`] and accepted by
/// `sk_recording_canvas_*` — it is **not** interchangeable with
/// [`sk_canvas_t`] (which is raster-backed).
pub struct sk_recording_canvas_t {
    /// Points back to the owning recorder. The recorder outlives the
    /// recording canvas under normal use, but callers can delete the
    /// recorder early — [`resolve_recording_canvas`] checks `alive` before
    /// ever dereferencing this pointer, so that case is handled safely
    /// rather than relying on caller discipline.
    recorder: *mut sk_picture_recorder_t,
    /// Cloned from the owning recorder's `alive` flag at `begin_recording`
    /// time. Never dereferences `recorder`.
    alive: Arc<AtomicBool>,
}

/// Resolve a recording-canvas handle to its recorder's `(commands, state)`.
///
/// Returns `None` if the handle is null, the recorder that created it has
/// since been deleted (checked via the shared `alive` flag *before* the
/// `recorder` pointer is ever dereferenced — deleting the recorder frees
/// that memory, so touching it afterwards would be undefined behavior), or
/// the recorder's `state` (only populated between `begin_recording` and
/// `finish_recording`) is absent.
unsafe fn resolve_recording_canvas<'a>(
    canvas: *mut sk_recording_canvas_t,
) -> Option<(
    &'a mut Vec<skia_rs_canvas::DrawCommand>,
    &'a mut CanvasState,
)> {
    let rc = canvas.as_mut()?;
    if !rc.alive.load(Ordering::Acquire) {
        return None;
    }
    let rec = rc.recorder.as_mut()?;
    // Hand out two *disjoint* field borrows (the command list and the canvas
    // state) rather than the whole recorder plus a pointer into it, so no
    // two live `&mut` overlap.
    let sk_picture_recorder_t {
        commands, state, ..
    } = rec;
    Some((commands, state.as_mut()?))
}

/// Begin recording into a fresh picture.
///
/// Returns a recording canvas handle that draw ops append to. The caller
/// drops the handle via [`sk_recording_canvas_release`] but **must** call
/// [`sk_picture_recorder_finish_recording`] to get the recorded picture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_recorder_begin_recording(
    r: *mut sk_picture_recorder_t,
    bounds: *const sk_rect_t,
) -> *mut sk_recording_canvas_t {
    catch_panic(|| {
        let Some(rec) = r.as_mut() else {
            return ptr::null_mut();
        };
        if rec.recording {
            return ptr::null_mut();
        }
        let cull = bounds.as_ref().map_or(Rect::EMPTY, |b| Rect::from(*b));
        rec.commands.clear();
        rec.cull_rect = cull;
        rec.recording = true;
        rec.state = Some(CanvasState {
            save_count: 0,
            matrix: Matrix::identity(),
            stack: Vec::new(),
            clips: Vec::new(),
        });
        let alive = rec.alive.clone();
        Box::into_raw(Box::new(sk_recording_canvas_t { recorder: r, alive }))
    })
}

/// Release a recording canvas handle acquired via
/// [`sk_picture_recorder_begin_recording`]. Does *not* finalize the
/// recording — call [`sk_picture_recorder_finish_recording`] for that.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_recording_canvas_release(canvas: *mut sk_recording_canvas_t) {
    catch_panic_void(|| {
        if !canvas.is_null() {
            drop(Box::from_raw(canvas));
        }
    });
}

/// Finish recording and return the captured picture. Returns null if the
/// recorder was not recording. After this call the recorder can be reused
/// with another [`sk_picture_recorder_begin_recording`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_recorder_finish_recording(
    r: *mut sk_picture_recorder_t,
) -> *mut sk_picture_t {
    catch_panic(|| {
        let Some(rec) = r.as_mut() else {
            return ptr::null_mut();
        };
        if !rec.recording {
            return ptr::null_mut();
        }
        rec.recording = false;
        rec.state = None;
        let commands = std::mem::take(&mut rec.commands);
        let picture = Arc::new(skia_rs_canvas::Picture::from_parts(commands, rec.cull_rect));
        RefCounted::new(picture)
    })
}

/// Push a save onto the recording canvas's state.
///
/// Returns the same value as `SkCanvas::save()`: the save count *before*
/// this call in Skia's 1-based counting (1 for the first save). Returns 0
/// if the handle is null or its recorder was deleted.
///
/// Like `SkPicture`, the recording stores the save and later transform ops
/// as commands and keeps geometry in local coordinates, so playback composes
/// with the target canvas's own matrix and path effects see local geometry.
///
/// # Safety
/// `canvas` must be null or a recording canvas handle from
/// `sk_picture_recorder_begin_recording`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_recording_canvas_save(canvas: *mut sk_recording_canvas_t) -> i32 {
    catch_panic(|| {
        let Some((commands, state)) = resolve_recording_canvas(canvas) else {
            return 0;
        };
        commands.push(skia_rs_canvas::DrawCommand::Save);
        let frame = CanvasSaveFrame {
            matrix: state.matrix,
            clip_len: state.clips.len(),
            layer: None,
        };
        state.stack.push(frame);
        state.save_count += 1;
        i32::try_from(state.save_count).unwrap_or(i32::MAX)
    })
}

/// Pop a save on the recording canvas's state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_recording_canvas_restore(canvas: *mut sk_recording_canvas_t) {
    catch_panic_void(|| {
        let Some((commands, state)) = resolve_recording_canvas(canvas) else {
            return;
        };
        if let Some(frame) = state.stack.pop() {
            commands.push(skia_rs_canvas::DrawCommand::Restore);
            state.matrix = frame.matrix;
            state.clips.truncate(frame.clip_len);
            state.save_count = state.save_count.saturating_sub(1);
        }
    });
}

/// Translate the recording canvas's current transform.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_recording_canvas_translate(
    canvas: *mut sk_recording_canvas_t,
    dx: f32,
    dy: f32,
) {
    catch_panic_void(|| {
        let Some((commands, _)) = resolve_recording_canvas(canvas) else {
            return;
        };
        commands.push(skia_rs_canvas::DrawCommand::Translate { dx, dy });
    });
}

/// Draw a rect into the recording canvas. The rect is recorded in local
/// coordinates; the transform ops recorded before it are replayed on top of
/// the playback canvas's matrix, as with `SkPicture`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_recording_canvas_draw_rect(
    canvas: *mut sk_recording_canvas_t,
    rect: *const sk_rect_t,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        let Some((commands, _)) = resolve_recording_canvas(canvas) else {
            return;
        };
        let Some(r) = rect.as_ref() else { return };
        let Some(p) = RefCounted::get_ref(paint).cloned() else {
            return;
        };
        commands.push(skia_rs_canvas::DrawCommand::DrawRect {
            rect: Rect::from(*r),
            paint: p,
        });
    });
}

/// Draw a path into the recording canvas.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_recording_canvas_draw_path(
    canvas: *mut sk_recording_canvas_t,
    path: *const sk_path_t,
    paint: *const sk_paint_t,
) {
    catch_panic_void(|| {
        let Some((commands, _)) = resolve_recording_canvas(canvas) else {
            return;
        };
        let Some(path_ref) = RefCounted::get_ref(path).cloned() else {
            return;
        };
        let Some(p) = RefCounted::get_ref(paint).cloned() else {
            return;
        };
        commands.push(skia_rs_canvas::DrawCommand::DrawPath {
            path: path_ref,
            paint: p,
        });
    });
}

/// Play a recorded picture back onto a raster canvas.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_playback(
    picture: *const sk_picture_t,
    canvas: *mut sk_canvas_t,
) -> bool {
    catch_panic(|| {
        let Some(pic) = RefCounted::get_ref(picture) else {
            return false;
        };
        let pic_clone = pic.clone();
        let Some(c) = canvas.as_mut() else {
            return false;
        };
        let mut inner = c.canvas_with_state();
        pic_clone.playback(&mut inner);
        true
    })
}

/// Get the picture's cull rect.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_get_cull_rect(
    picture: *const sk_picture_t,
    out: *mut sk_rect_t,
) -> bool {
    catch_panic(|| {
        if let (Some(p), Some(dst)) = (RefCounted::get_ref(picture), out.as_mut()) {
            *dst = p.cull_rect().into();
            true
        } else {
            false
        }
    })
}

/// Get the approximate operation count of a picture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_approximate_op_count(picture: *const sk_picture_t) -> usize {
    catch_panic(|| RefCounted::get_ref(picture).map_or(0, |p| p.approximate_op_count()))
}

/// Increment picture refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_ref(picture: *mut sk_picture_t) {
    catch_panic_void(|| RefCounted::ref_ptr(picture));
}

/// Decrement picture refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_picture_unref(picture: *mut sk_picture_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(picture);
    });
}

// =============================================================================
// Region API (Reference Counted)
// =============================================================================

/// Reference counted [`Region`].
pub type sk_region_t = RefCounted<Region>;

/// Create a new, empty region.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_new() -> *mut sk_region_t {
    catch_panic(|| RefCounted::new(Region::new()))
}

/// Set the region to a single integer rectangle. Returns true if the rect
/// was non-empty (the region now holds it), false if empty (the region is
/// cleared).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_set_rect(
    region: *mut sk_region_t,
    rect: *const sk_irect_t,
) -> bool {
    catch_panic(|| {
        let Some(r) = RefCounted::get_mut(region) else {
            return false;
        };
        let Some(src) = rect.as_ref() else {
            return false;
        };
        r.set_rect(IRect::new(src.left, src.top, src.right, src.bottom))
    })
}

/// Apply a rect op on the region.
///
/// `op` follows upstream `SkRegion::Op`; see `decode_region_op`. Returns
/// false if `op` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_op_rect(
    region: *mut sk_region_t,
    rect: *const sk_irect_t,
    op: sk_region_op_t,
) -> bool {
    catch_panic(|| {
        let Some(r) = RefCounted::get_mut(region) else {
            return false;
        };
        let Some(src) = rect.as_ref() else {
            return false;
        };
        let Some(region_op) = decode_region_op(op) else {
            return false;
        };
        r.op_rect(
            IRect::new(src.left, src.top, src.right, src.bottom),
            region_op,
        )
    })
}

/// Get the region's integer bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_get_bounds(
    region: *const sk_region_t,
    out: *mut sk_irect_t,
) -> bool {
    catch_panic(|| {
        if let (Some(r), Some(dst)) = (RefCounted::get_ref(region), out.as_mut()) {
            let b = r.bounds();
            *dst = sk_irect_t {
                left: b.left,
                top: b.top,
                right: b.right,
                bottom: b.bottom,
            };
            true
        } else {
            false
        }
    })
}

/// Return true if the region is empty.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_is_empty(region: *const sk_region_t) -> bool {
    catch_panic(|| RefCounted::get_ref(region).is_none_or(skia_rs_core::Region::is_empty))
}

/// Return true if the region contains a point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_contains(region: *const sk_region_t, x: i32, y: i32) -> bool {
    catch_panic(|| RefCounted::get_ref(region).is_some_and(|r| r.contains(x, y)))
}

/// Increment region refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_ref(region: *mut sk_region_t) {
    catch_panic_void(|| RefCounted::ref_ptr(region));
}

/// Decrement region refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_region_unref(region: *mut sk_region_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(region);
    });
}

// =============================================================================
// PathEffect API (Reference Counted)
// =============================================================================

/// Reference counted [`PathEffectRef`].
pub type sk_patheffect_t = RefCounted<PathEffectRef>;

/// Create a dash path effect.
///
/// `intervals` must be non-null with an even `count` of at least 2
/// non-negative entries; like `SkDashPathEffect::Make`, an odd `count`
/// returns null.
/// `phase` shifts where the pattern starts. Returns null if the intervals
/// are invalid (empty, negative, or sum to 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_patheffect_new_dash(
    intervals: *const f32,
    count: usize,
    phase: f32,
) -> *mut sk_patheffect_t {
    catch_panic(|| {
        if intervals.is_null() || count == 0 {
            return ptr::null_mut();
        }
        let src = slice::from_raw_parts(intervals, count).to_vec();
        DashEffect::new(src, phase).map_or(ptr::null_mut(), |e| {
            let pe: PathEffectRef = Arc::new(e);
            RefCounted::new(pe)
        })
    })
}

/// Create a trim path effect. `start` and `end` are normalized lengths
/// in [0, 1]. `mode` follows upstream `SkTrimPathEffect::Mode`; see
/// `decode_trim_mode`. Returns null if `mode` is out of range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_patheffect_new_trim(
    start: f32,
    end: f32,
    mode: sk_trim_mode_t,
) -> *mut sk_patheffect_t {
    catch_panic(|| {
        let Some(trim_mode) = decode_trim_mode(mode) else {
            return ptr::null_mut();
        };
        TrimEffect::new(start, end, trim_mode).map_or(ptr::null_mut(), |e| {
            let pe: PathEffectRef = Arc::new(e);
            RefCounted::new(pe)
        })
    })
}

/// Attach a path effect to a paint, or clear it if `effect` is null. The
/// paint takes its own [`Arc`] reference; the caller retains their own.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_paint_set_path_effect(
    paint: *mut sk_paint_t,
    effect: *const sk_patheffect_t,
) {
    catch_panic_void(|| {
        if let Some(p) = RefCounted::get_mut(paint) {
            if effect.is_null() {
                p.set_path_effect(None);
            } else if let Some(e) = RefCounted::get_ref(effect) {
                p.set_path_effect(Some(e.clone()));
            }
        }
    });
}

/// Increment path effect refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_patheffect_ref(effect: *mut sk_patheffect_t) {
    catch_panic_void(|| RefCounted::ref_ptr(effect));
}

/// Decrement path effect refcount.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk_patheffect_unref(effect: *mut sk_patheffect_t) {
    catch_panic_void(|| {
        RefCounted::unref_ptr(effect);
    });
}

// Silence unused import warnings — these types are needed for re-exports and
// conversion glue declared elsewhere in the crate, but rustc can't see them
// used directly.
#[allow(dead_code)]
const fn _unused_deps() {
    let _: Option<IPoint> = None;
    let _: Option<IRect> = None;
    let _: Option<ISize> = None;
    let _: Option<Size> = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_surface_creation() {
        unsafe {
            let surface = sk_surface_new_raster(100, 100);
            assert!(!surface.is_null());
            assert_eq!(sk_surface_get_width(surface), 100);
            assert_eq!(sk_surface_get_height(surface), 100);
            assert_eq!(sk_surface_get_refcnt(surface), 1);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_surface_refcounting() {
        unsafe {
            let surface = sk_surface_new_raster(100, 100);
            assert_eq!(sk_surface_get_refcnt(surface), 1);
            assert!(sk_surface_is_unique(surface));

            sk_surface_ref(surface);
            assert_eq!(sk_surface_get_refcnt(surface), 2);
            assert!(!sk_surface_is_unique(surface));

            sk_surface_unref(surface);
            assert_eq!(sk_surface_get_refcnt(surface), 1);
            assert!(sk_surface_is_unique(surface));

            sk_surface_unref(surface); // Frees it.
        }
    }

    #[test]
    fn test_paint_operations() {
        unsafe {
            let paint = sk_paint_new();
            assert!(!paint.is_null());
            assert_eq!(sk_paint_get_refcnt(paint), 1);

            sk_paint_set_color(paint, 0xFF00_00FF); // Blue
            assert_eq!(sk_paint_get_color(paint), 0xFF00_00FF);

            sk_paint_set_stroke_width(paint, 2.0);
            assert!((sk_paint_get_stroke_width(paint) - 2.0).abs() < f32::EPSILON);

            sk_paint_delete(paint);
        }
    }

    #[test]
    fn test_paint_refcounting() {
        unsafe {
            let paint = sk_paint_new();
            assert_eq!(sk_paint_get_refcnt(paint), 1);

            sk_paint_ref(paint);
            assert_eq!(sk_paint_get_refcnt(paint), 2);

            sk_paint_unref(paint);
            assert_eq!(sk_paint_get_refcnt(paint), 1);

            let paint2 = sk_paint_clone(paint);
            assert_eq!(sk_paint_get_refcnt(paint), 1);
            assert_eq!(sk_paint_get_refcnt(paint2), 1);

            sk_paint_unref(paint);
            sk_paint_unref(paint2);
        }
    }

    #[test]
    fn test_path_builder() {
        unsafe {
            let builder = sk_pathbuilder_new();
            assert!(!builder.is_null());

            sk_pathbuilder_move_to(builder, 0.0, 0.0);
            sk_pathbuilder_line_to(builder, 100.0, 0.0);
            sk_pathbuilder_line_to(builder, 100.0, 100.0);
            sk_pathbuilder_close(builder);

            let path = sk_pathbuilder_detach(builder);
            assert!(!path.is_null());
            assert!(!sk_path_is_empty(path));
            assert_eq!(sk_path_get_refcnt(path), 1);

            let mut bounds = sk_rect_t::default();
            sk_path_get_bounds(path, &raw mut bounds);
            assert!((bounds.left - 0.0).abs() < f32::EPSILON);
            assert!((bounds.right - 100.0).abs() < f32::EPSILON);

            sk_path_delete(path);
            sk_pathbuilder_delete(builder);
        }
    }

    #[test]
    fn test_path_iteration() {
        unsafe {
            let b = sk_pathbuilder_new();
            sk_pathbuilder_move_to(b, 1.0, 2.0);
            sk_pathbuilder_line_to(b, 10.0, 20.0);
            sk_pathbuilder_close(b);
            let path = sk_pathbuilder_detach(b);
            sk_pathbuilder_delete(b);

            let it = sk_path_iter_new(path);
            assert!(!it.is_null());

            let mut pts = [sk_point_t::default(); 4];
            let mut w: f32 = 0.0;

            let v0 = sk_path_iter_next(it, pts.as_mut_ptr(), &raw mut w);
            assert_eq!(v0, SK_PATH_VERB_MOVE);
            assert!((pts[0].x - 1.0).abs() < f32::EPSILON);
            assert!((pts[0].y - 2.0).abs() < f32::EPSILON);

            let v1 = sk_path_iter_next(it, pts.as_mut_ptr(), &raw mut w);
            assert_eq!(v1, SK_PATH_VERB_LINE);
            assert!((pts[0].x - 10.0).abs() < f32::EPSILON);
            assert!((pts[0].y - 20.0).abs() < f32::EPSILON);

            let v2 = sk_path_iter_next(it, pts.as_mut_ptr(), &raw mut w);
            assert_eq!(v2, SK_PATH_VERB_CLOSE);

            let v3 = sk_path_iter_next(it, pts.as_mut_ptr(), &raw mut w);
            assert_eq!(v3, SK_PATH_VERB_DONE);

            sk_path_iter_delete(it);
            sk_path_unref(path);
        }
    }

    #[test]
    fn test_path_refcounting() {
        unsafe {
            let path = sk_path_new();
            assert_eq!(sk_path_get_refcnt(path), 1);

            sk_path_ref(path);
            assert_eq!(sk_path_get_refcnt(path), 2);

            let path2 = sk_path_clone(path);
            assert_eq!(sk_path_get_refcnt(path), 2);
            assert_eq!(sk_path_get_refcnt(path2), 1);

            sk_path_unref(path);
            assert_eq!(sk_path_get_refcnt(path), 1);

            sk_path_unref(path);
            sk_path_unref(path2);
        }
    }

    #[test]
    fn test_matrix_operations() {
        unsafe {
            let mut matrix = sk_matrix_t::default();
            sk_matrix_set_translate(&raw mut matrix, 10.0, 20.0);

            let point = sk_point_t { x: 0.0, y: 0.0 };
            let mut result = sk_point_t::default();
            sk_matrix_map_point(&raw const matrix, &raw const point, &raw mut result);

            assert!((result.x - 10.0).abs() < f32::EPSILON);
            assert!((result.y - 20.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_matrix_invert() {
        unsafe {
            let mut matrix = sk_matrix_t::default();
            sk_matrix_set_scale(&raw mut matrix, 2.0, 4.0);
            let mut inv = sk_matrix_t::default();
            assert!(sk_matrix_invert(&raw const matrix, &raw mut inv));
            // (2,4) scaled then halved/quartered should return origin-offset.
            let p = sk_point_t { x: 2.0, y: 4.0 };
            let mut r = sk_point_t::default();
            sk_matrix_map_point(&raw const inv, &raw const p, &raw mut r);
            assert!((r.x - 1.0).abs() < f32::EPSILON);
            assert!((r.y - 1.0).abs() < f32::EPSILON);

            // Singular matrix should fail.
            let zero = sk_matrix_t { values: [0.0; 9] };
            assert!(!sk_matrix_invert(&raw const zero, &raw mut inv));
        }
    }

    #[test]
    fn test_matrix_is_identity_and_determinant() {
        unsafe {
            let m = sk_matrix_t::default();
            assert!(sk_matrix_is_identity(&raw const m));
            assert!((sk_matrix_determinant(&raw const m) - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_draw_rect() {
        unsafe {
            let surface = sk_surface_new_raster(100, 100);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_0000);
            sk_surface_clear(surface, 0xFFFF_FFFF);

            let rect = sk_rect_t {
                left: 10.0,
                top: 10.0,
                right: 50.0,
                bottom: 50.0,
            };
            sk_surface_draw_rect(surface, &raw const rect, paint);

            sk_paint_delete(paint);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_refcnt_utility() {
        unsafe {
            let surface = sk_surface_new_raster(100, 100);

            let ptr = surface as *const sk_refcnt_t;
            assert_eq!(sk_refcnt_get_count(ptr), 1);
            assert!(sk_refcnt_is_unique(ptr));

            sk_surface_ref(surface);
            assert_eq!(sk_refcnt_get_count(ptr), 2);
            assert!(!sk_refcnt_is_unique(ptr));

            sk_surface_unref(surface);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_refcnt_utility_rejects_untagged_pointer() {
        // Construct a raw AtomicU32 that happens to have the layout of a
        // legacy "refcount at offset 0" but no magic tag. The hardened
        // generic entry points must reject it.
        let fake = Box::into_raw(Box::new([0u32; 4]));
        unsafe {
            let ptr = fake as *const sk_refcnt_t;
            assert_eq!(sk_refcnt_get_count(ptr), 0);
            assert!(!sk_refcnt_is_unique(ptr));
            drop(Box::from_raw(fake));
        }
    }

    #[test]
    fn test_panic_catcher_sets_flag_and_returns_default() {
        // Clear any previous panic state.
        let _ = sk_last_call_panicked();
        assert!(!sk_last_call_panicked());

        // Deliberately trigger a panic inside the wrapper. `catch_panic`
        // returns T::default(), and the flag must come back true.
        let v: u32 = catch_panic(|| panic!("boom"));
        assert_eq!(v, 0);
        assert!(sk_last_call_panicked());

        // Flag auto-clears on read.
        assert!(!sk_last_call_panicked());
    }

    #[test]
    fn test_abi_init() {
        assert!(!abi::sk_abi_is_compatible(99, 0));
        assert!(sk_init(1, 0));
        assert!(sk_is_initialized());
        // A second compatible call still succeeds.
        assert!(sk_init(1, 0));
        // An incompatible call returns false but does not un-initialize.
        assert!(!sk_init(99, 99));
        assert!(sk_is_initialized());
    }

    #[test]
    fn test_cross_thread_refcount_stress() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;

        unsafe {
            let paint = sk_paint_new();
            // Guard: drops to 1 only after all threads finish.
            sk_paint_ref(paint); // Refcount: 2 (one for us, one for threads)

            let n_threads = 8usize;
            let iters = 1_000usize;

            let ptr = paint as usize;
            let done = Arc::new(AtomicUsize::new(0));

            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let d = done.clone();
                    thread::spawn(move || {
                        let p = ptr as *mut sk_paint_t;
                        for _ in 0..iters {
                            sk_paint_ref(p);
                            sk_paint_unref(p);
                        }
                        d.fetch_add(1, Ordering::SeqCst);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(done.load(Ordering::SeqCst), n_threads);
            // After all ref/unref pairs cancel, refcount is 2.
            assert_eq!(sk_paint_get_refcnt(paint), 2);
            sk_paint_unref(paint);
            sk_paint_unref(paint);
        }
    }

    #[test]
    fn test_shader_linear_gradient() {
        unsafe {
            let colors = [
                sk_color4f_t {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                sk_color4f_t {
                    r: 0.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                },
            ];
            let shader = sk_shader_new_linear_gradient(
                sk_point_t { x: 0.0, y: 0.0 },
                sk_point_t { x: 10.0, y: 10.0 },
                colors.as_ptr(),
                ptr::null(),
                2,
                0, // clamp
            );
            assert!(!shader.is_null());
            assert_eq!(sk_shader_get_refcnt(shader), 1);

            let paint = sk_paint_new();
            sk_paint_set_shader(paint, shader);
            // Paint holds its own Arc to the ShaderRef; the refcount of the
            // *wrapper* is unchanged.
            assert_eq!(sk_shader_get_refcnt(shader), 1);

            sk_paint_set_shader(paint, ptr::null());
            sk_paint_unref(paint);
            sk_shader_unref(shader);
        }
    }

    #[test]
    fn test_colorfilter_and_maskfilter() {
        unsafe {
            let cf = sk_colorfilter_new_saturation(0.5);
            assert!(!cf.is_null());
            let mf = sk_maskfilter_new_blur(0, 2.0);
            assert!(!mf.is_null());
            let imf = sk_imagefilter_new_blur(2.0, 2.0, 0);
            assert!(!imf.is_null());

            let paint = sk_paint_new();
            sk_paint_set_color_filter(paint, cf);
            sk_paint_set_mask_filter(paint, mf);
            sk_paint_set_image_filter(paint, imf);
            sk_paint_set_color_filter(paint, ptr::null());
            sk_paint_set_mask_filter(paint, ptr::null());
            sk_paint_set_image_filter(paint, ptr::null());
            sk_paint_unref(paint);

            sk_colorfilter_unref(cf);
            sk_maskfilter_unref(mf);
            sk_imagefilter_unref(imf);
        }
    }

    #[test]
    fn test_image_from_color_and_encode() {
        unsafe {
            let img = sk_image_from_color(8, 8, 0xFF00_FF00);
            assert!(!img.is_null());
            assert_eq!(sk_image_get_width(img), 8);
            assert_eq!(sk_image_get_height(img), 8);

            // Query size.
            let needed = sk_image_encode_png(img, ptr::null_mut(), 0);
            assert!(needed > 0);

            // Encode into a sized buffer.
            let mut buf = vec![0u8; needed];
            let wrote = sk_image_encode_png(img, buf.as_mut_ptr(), buf.len());
            assert_eq!(wrote, needed);
            // PNG magic.
            assert_eq!(&buf[0..4], b"\x89PNG");

            sk_image_unref(img);
        }
    }

    #[test]
    fn test_typeface_and_font() {
        unsafe {
            let tf = sk_typeface_default();
            assert!(!tf.is_null());
            let font = sk_font_new(tf, 14.0);
            assert!(!font.is_null());
            assert!((sk_font_get_size(font) - 14.0).abs() < f32::EPSILON);
            sk_font_unref(font);
            sk_typeface_unref(tf);
        }
    }

    #[test]
    fn test_canvas_lifecycle() {
        unsafe {
            let surface = sk_surface_new_raster(32, 32);
            let canvas = sk_surface_lock_canvas(surface);
            assert!(!canvas.is_null());
            assert_eq!(sk_canvas_get_width(canvas), 32);
            assert_eq!(sk_canvas_get_height(canvas), 32);

            let n0 = sk_canvas_save(canvas);
            assert_eq!(n0, 1);
            sk_canvas_translate(canvas, 10.0, 5.0);
            sk_canvas_scale(canvas, 2.0, 2.0);

            sk_canvas_clear(canvas, 0xFFFF_FFFF);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_0000);
            let rect = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 5.0,
                bottom: 5.0,
            };
            sk_canvas_draw_rect(canvas, &raw const rect, paint);
            sk_paint_unref(paint);

            sk_canvas_restore(canvas);
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_surface_read_pixels_copies() {
        unsafe {
            let surface = sk_surface_new_raster(4, 4);
            sk_surface_clear(surface, 0xFF11_2233);

            let mut buf = vec![0u8; 4 * 4 * 4];
            let wrote = sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            assert_eq!(wrote, buf.len());

            // Too-small buffer fails cleanly.
            let mut small = vec![0u8; 3];
            let wrote_small = sk_surface_read_pixels(surface, small.as_mut_ptr(), small.len());
            assert_eq!(wrote_small, 0);

            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_surface_read_pixels_unpremultiplies_for_unpremul_surface() {
        unsafe {
            // RGBA8888, Unpremul.
            let info = sk_imageinfo_t {
                width: 2,
                height: 2,
                color_type: 4,
                alpha_type: 3,
            };
            let surface = sk_surface_new_raster_with_info(&raw const info);
            assert!(!surface.is_null());

            // Half-transparent red — the surface's internal buffer stores
            // this premultiplied (~128,0,0,128), but a caller reading an
            // Unpremul-typed surface should see it unpremultiplied back to
            // ~255,0,0,128.
            sk_surface_clear(surface, 0x80FF_0000);

            // `peek_pixels` cannot convert a borrowed buffer — must reject
            // Unpremul surfaces.
            let mut px: *const u8 = ptr::null();
            let mut row_bytes: usize = 0;
            assert!(!sk_surface_peek_pixels(
                surface,
                &raw mut px,
                &raw mut row_bytes
            ));

            let mut buf = vec![0u8; 2 * 2 * 4];
            let wrote = sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            assert_eq!(wrote, buf.len());

            // Pixel 0: R close to 255 (unpremultiplied), A close to 128.
            assert!(buf[0] > 250, "expected unpremultiplied red, got {}", buf[0]);
            assert_eq!(buf[1], 0);
            assert_eq!(buf[2], 0);
            assert!((120..=136).contains(&buf[3]), "unexpected alpha {}", buf[3]);

            sk_surface_unref(surface);
            assert!(!sk_last_call_panicked());
        }
    }

    #[test]
    fn test_null_inputs_dont_crash() {
        unsafe {
            // Each of these must either no-op or return a default, never
            // unwind.
            sk_paint_set_color(ptr::null_mut(), 0);
            sk_paint_set_stroke_width(ptr::null_mut(), 1.0);
            sk_paint_unref(ptr::null_mut());
            sk_surface_unref(ptr::null_mut());
            let mut bounds = sk_rect_t::default();
            sk_path_get_bounds(ptr::null(), &raw mut bounds);
            sk_path_unref(ptr::null_mut());
            assert_eq!(
                sk_path_iter_next(ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
                SK_PATH_VERB_DONE
            );
            assert!(!sk_last_call_panicked());
        }
    }

    #[test]
    fn test_counter_sanity() {
        // Ensures the `AtomicUsize` import is used (keeps the test module
        // tidy under -D warnings).
        let c = AtomicUsize::new(0);
        c.fetch_add(1, Ordering::SeqCst);
        assert_eq!(c.load(Ordering::SeqCst), 1);
    }

    // =========================================================================
    // P7I additions — clip / save_layer / Matrix44 / ColorSpace / TextBlob /
    // Picture / Region / PathEffect
    // =========================================================================

    #[test]
    fn test_canvas_clip_rect_intersect_ffi() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            sk_canvas_clear(canvas, 0xFF00_0000);

            // Intersect with a 4x4 inner square.
            let clip_rect = sk_rect_t {
                left: 3.0,
                top: 3.0,
                right: 7.0,
                bottom: 7.0,
            };
            assert!(sk_canvas_clip_rect(
                canvas,
                &raw const clip_rect,
                SkClipOp::Intersect as u32,
                false
            ));

            // Draw a full-canvas white rect — only the inner box survives.
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            let big = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 10.0,
                bottom: 10.0,
            };
            sk_canvas_draw_rect(canvas, &raw const big, paint);
            sk_paint_unref(paint);

            sk_canvas_release(canvas);

            // Check a pixel inside clip and outside clip.
            let mut buf = vec![0u8; 10 * 10 * 4];
            sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            // Pixel at (5,5) — inside clip — should be near white.
            let idx_inside = (5 * 10 + 5) * 4;
            assert!(buf[idx_inside] > 200);
            // Pixel at (0,0) — outside clip — should remain black.
            let idx_outside = 0;
            assert!(buf[idx_outside] < 40);

            sk_surface_unref(surface);
            assert!(!sk_last_call_panicked());
        }
    }

    #[test]
    fn test_sk_canvas_clear_respects_clip() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            // Known background color (opaque blue).
            sk_canvas_clear(canvas, 0xFF00_00FF);

            // Intersect with a 4x4 inner square.
            let clip_rect = sk_rect_t {
                left: 3.0,
                top: 3.0,
                right: 7.0,
                bottom: 7.0,
            };
            assert!(sk_canvas_clip_rect(
                canvas,
                &raw const clip_rect,
                SkClipOp::Intersect as u32,
                false
            ));

            // `clear` must respect the current clip: only the inner box
            // should turn red.
            sk_canvas_clear(canvas, 0xFFFF_0000);

            sk_canvas_release(canvas);

            let mut buf = vec![0u8; 10 * 10 * 4];
            sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            // Pixel at (5,5) — inside clip — should be red.
            let idx_inside = (5 * 10 + 5) * 4;
            assert!(buf[idx_inside] > 200); // R
            assert!(buf[idx_inside + 2] < 40); // B
            // Pixel at (0,0) — outside clip — should remain blue.
            let idx_outside = 0;
            assert!(buf[idx_outside] < 40); // R
            assert!(buf[idx_outside + 2] > 200); // B

            sk_surface_unref(surface);
            assert!(!sk_last_call_panicked());
        }
    }

    #[test]
    fn test_canvas_clip_path_ffi() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);

            let b = sk_pathbuilder_new();
            sk_pathbuilder_add_circle(b, 5.0, 5.0, 3.0);
            let path = sk_pathbuilder_detach(b);
            sk_pathbuilder_delete(b);

            assert!(sk_canvas_clip_path(
                canvas,
                path,
                SkClipOp::Intersect as u32,
                true
            ));

            sk_path_unref(path);
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_clip_rect_rejects_invalid_op() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            let rect = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 5.0,
                bottom: 5.0,
            };
            // A valid op (Difference = 0) should work.
            assert!(sk_canvas_clip_rect(
                canvas,
                &raw const rect,
                SkClipOp::Difference as u32,
                false
            ));
            // An out-of-range raw op value must be rejected, not silently
            // coerced into a Rust enum (which would be UB) or defaulted.
            assert!(!sk_canvas_clip_rect(canvas, &raw const rect, 2, false));
            assert!(!sk_canvas_clip_rect(
                canvas,
                &raw const rect,
                u32::MAX,
                false
            ));
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_clip_path_rejects_invalid_op() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            let b = sk_pathbuilder_new();
            sk_pathbuilder_add_circle(b, 5.0, 5.0, 3.0);
            let path = sk_pathbuilder_detach(b);
            sk_pathbuilder_delete(b);

            assert!(!sk_canvas_clip_path(canvas, path, 99, true));

            sk_path_unref(path);
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_region_op_rect_rejects_invalid_op() {
        unsafe {
            let r = sk_region_new();
            let rect = sk_irect_t {
                left: 0,
                top: 0,
                right: 10,
                bottom: 10,
            };
            assert!(sk_region_set_rect(r, &raw const rect));
            let other = sk_irect_t {
                left: 5,
                top: 5,
                right: 15,
                bottom: 15,
            };
            assert!(!sk_region_op_rect(r, &raw const other, 6));
            assert!(!sk_region_op_rect(r, &raw const other, u32::MAX));
            sk_region_unref(r);
        }
    }

    #[test]
    fn test_patheffect_new_trim_rejects_invalid_mode() {
        unsafe {
            let effect = sk_patheffect_new_trim(0.25, 0.75, 2);
            assert!(effect.is_null());
        }
    }

    #[test]
    fn test_canvas_save_layer_and_restore_ffi() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            let n0 = sk_canvas_save(canvas);
            let n1 = sk_canvas_save_layer(canvas, ptr::null(), ptr::null());
            assert_eq!(n1, n0 + 1);
            sk_canvas_restore(canvas);
            sk_canvas_restore(canvas);
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_clip_ibounds_ffi() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            let mut bounds = sk_irect_t::default();
            assert!(sk_canvas_get_clip_ibounds(canvas, &raw mut bounds));
            assert_eq!(bounds.right, 10);
            assert_eq!(bounds.bottom, 10);

            let inner = sk_rect_t {
                left: 2.0,
                top: 2.0,
                right: 6.0,
                bottom: 6.0,
            };
            sk_canvas_clip_rect(canvas, &raw const inner, SkClipOp::Intersect as u32, false);
            assert!(sk_canvas_get_clip_ibounds(canvas, &raw mut bounds));
            assert!(bounds.left >= 2 && bounds.right <= 6);

            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_matrix44_new_is_identity_ffi() {
        unsafe {
            let m = sk_matrix44_new();
            assert!(sk_matrix44_is_identity(&raw const m));
            assert!((sk_matrix44_determinant(&raw const m) - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_matrix44_from_array_map_invert_ffi() {
        unsafe {
            // scale(2, 2, 2) — column-major.
            let values = [
                2.0, 0.0, 0.0, 0.0, // column 0
                0.0, 2.0, 0.0, 0.0, // column 1
                0.0, 0.0, 2.0, 0.0, // column 2
                0.0, 0.0, 0.0, 1.0, // column 3
            ];
            let m = sk_matrix44_from_array(values.as_ptr());
            assert!(!sk_matrix44_is_identity(&raw const m));

            let pt = sk_point_t { x: 3.0, y: 4.0 };
            let mut out = sk_point_t::default();
            sk_matrix44_map_point(&raw const m, &raw const pt, &raw mut out);
            assert!((out.x - 6.0).abs() < f32::EPSILON);
            assert!((out.y - 8.0).abs() < f32::EPSILON);

            let mut inv = sk_matrix44_new();
            assert!(sk_matrix44_invert(&raw const m, &raw mut inv));
            let mut roundtrip = sk_point_t::default();
            sk_matrix44_map_point(&raw const inv, &raw const out, &raw mut roundtrip);
            assert!((roundtrip.x - pt.x).abs() < 1e-4);
            assert!((roundtrip.y - pt.y).abs() < 1e-4);

            // Singular matrix rejects invert.
            let zero = sk_matrix44_t { values: [0.0; 16] };
            assert!(!sk_matrix44_invert(&raw const zero, &raw mut inv));
        }
    }

    #[test]
    fn test_matrix44_concat_ffi() {
        unsafe {
            // scale(2) * scale(3) = scale(6).
            let s2 = sk_matrix44_from_array(
                [
                    2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ]
                .as_ptr(),
            );
            let s3 = sk_matrix44_from_array(
                [
                    3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ]
                .as_ptr(),
            );
            let mut out = sk_matrix44_new();
            sk_matrix44_concat(&raw mut out, &raw const s2, &raw const s3);
            // map (1,1) -> (6, 6).
            let p = sk_point_t { x: 1.0, y: 1.0 };
            let mut mapped = sk_point_t::default();
            sk_matrix44_map_point(&raw const out, &raw const p, &raw mut mapped);
            assert!((mapped.x - 6.0).abs() < 1e-4);
            assert!((mapped.y - 6.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_colorspace_srgb_ffi() {
        unsafe {
            let cs = sk_colorspace_new_srgb();
            assert!(!cs.is_null());
            assert!(sk_colorspace_is_srgb(cs));
            assert!(!sk_colorspace_is_linear(cs));

            let linear = sk_colorspace_new_srgb_linear();
            assert!(!sk_colorspace_is_srgb(linear));
            assert!(sk_colorspace_is_linear(linear));
            sk_colorspace_unref(linear);

            let p3 = sk_colorspace_new_display_p3();
            assert!(!p3.is_null());
            assert!(!sk_colorspace_is_srgb(p3));
            sk_colorspace_unref(p3);

            sk_colorspace_ref(cs);
            assert_eq!(sk_refcnt_get_count(cs as *const sk_refcnt_t), 2);
            sk_colorspace_unref(cs);
            sk_colorspace_unref(cs);
        }
    }

    #[test]
    fn test_colorspace_from_icc_rejects_bad_data_ffi() {
        unsafe {
            let bogus = [0u8; 8];
            let cs = sk_colorspace_from_icc(bogus.as_ptr(), bogus.len());
            assert!(cs.is_null());
            assert!(!sk_last_call_panicked());
        }
    }

    #[test]
    fn test_textblob_make_and_draw_ffi() {
        unsafe {
            let tf = sk_typeface_default();
            let font = sk_font_new(tf, 16.0);
            let text = b"hi\0";
            let blob = sk_textblob_make_from_text(text.as_ptr().cast::<c_char>(), 2, font);
            assert!(!blob.is_null());

            let mut bounds = sk_rect_t::default();
            assert!(sk_textblob_get_bounds(blob, &raw mut bounds));

            let surface = sk_surface_new_raster(64, 32);
            let canvas = sk_surface_lock_canvas(surface);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFF00_0000);
            assert!(sk_canvas_draw_text_blob(canvas, blob, 2.0, 20.0, paint));
            sk_paint_unref(paint);
            sk_canvas_release(canvas);
            sk_surface_unref(surface);

            sk_textblob_unref(blob);
            sk_font_unref(font);
            sk_typeface_unref(tf);
        }
    }

    #[test]
    fn test_recording_canvas_records_transforms_and_local_geometry() {
        // Like SkPicture: transform ops are recorded as commands and geometry
        // stays local, so path effects see local coordinates at playback.
        unsafe {
            let rec = sk_picture_recorder_new();
            let cull = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 32.0,
                bottom: 32.0,
            };
            let rcanvas = sk_picture_recorder_begin_recording(rec, &raw const cull);
            let paint = sk_paint_new();
            let r = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 4.0,
                bottom: 4.0,
            };
            sk_recording_canvas_save(rcanvas);
            sk_recording_canvas_translate(rcanvas, 8.0, 0.0);
            sk_recording_canvas_draw_rect(rcanvas, &raw const r, paint);
            sk_recording_canvas_restore(rcanvas);

            let commands = &(*rec).commands;
            assert!(matches!(commands[0], skia_rs_canvas::DrawCommand::Save));
            match commands[1] {
                skia_rs_canvas::DrawCommand::Translate { dx, dy } => {
                    assert!((dx - 8.0).abs() < f32::EPSILON && dy.abs() < f32::EPSILON);
                }
                ref other => panic!("expected Translate, got {other:?}"),
            }
            match &commands[2] {
                skia_rs_canvas::DrawCommand::DrawRect { rect, .. } => {
                    assert_eq!(
                        *rect,
                        Rect::from_xywh(0.0, 0.0, 4.0, 4.0),
                        "rect stays local"
                    );
                }
                other => panic!("expected DrawRect, got {other:?}"),
            }
            assert!(matches!(commands[3], skia_rs_canvas::DrawCommand::Restore));

            sk_paint_unref(paint);
            sk_recording_canvas_release(rcanvas);

            // Playback still lands the rect at x = 8..12.
            let picture = sk_picture_recorder_finish_recording(rec);
            let surface = sk_surface_new_raster(32, 32);
            let canvas = sk_surface_lock_canvas(surface);
            sk_canvas_clear(canvas, 0xFFFF_FFFF);
            assert!(sk_picture_playback(picture, canvas));
            sk_canvas_release(canvas);
            let mut buf = vec![0u8; 32 * 32 * 4];
            sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            let px = |x: usize, y: usize| buf[(y * 32 + x) * 4];
            assert!(px(10, 2) < 100, "rect drawn at its translated position");
            assert!(px(2, 2) > 200, "nothing drawn at the untranslated position");

            sk_surface_unref(surface);
            sk_picture_unref(picture);
            sk_picture_recorder_delete(rec);
        }
    }

    #[test]
    fn test_picture_record_and_playback_ffi() {
        unsafe {
            let rec = sk_picture_recorder_new();
            assert!(!rec.is_null());
            let cull = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 16.0,
                bottom: 16.0,
            };
            let rcanvas = sk_picture_recorder_begin_recording(rec, &raw const cull);
            assert!(!rcanvas.is_null());

            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_0000);
            let r = sk_rect_t {
                left: 2.0,
                top: 2.0,
                right: 10.0,
                bottom: 10.0,
            };
            sk_recording_canvas_save(rcanvas);
            sk_recording_canvas_translate(rcanvas, 1.0, 1.0);
            sk_recording_canvas_draw_rect(rcanvas, &raw const r, paint);
            sk_recording_canvas_restore(rcanvas);
            sk_paint_unref(paint);
            sk_recording_canvas_release(rcanvas);

            let picture = sk_picture_recorder_finish_recording(rec);
            assert!(!picture.is_null());
            assert!(sk_picture_approximate_op_count(picture) > 0);

            let mut cull_out = sk_rect_t::default();
            assert!(sk_picture_get_cull_rect(picture, &raw mut cull_out));
            assert!((cull_out.right - 16.0).abs() < f32::EPSILON);

            // Playback onto a fresh canvas.
            let surface = sk_surface_new_raster(16, 16);
            let canvas = sk_surface_lock_canvas(surface);
            sk_canvas_clear(canvas, 0xFFFF_FFFF);
            assert!(sk_picture_playback(picture, canvas));
            sk_canvas_release(canvas);

            let mut buf = vec![0u8; 16 * 16 * 4];
            sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            // Center pixel ought to hold the red fill. Skia pixel layout is
            // BGRA on little-endian RGBA8888 surfaces in this repo; accept
            // either order by looking for a high red/blue somewhere.
            let mut saw_colored = false;
            for px in buf.as_chunks::<4>().0 {
                if px[0] > 150 || px[2] > 150 {
                    saw_colored = true;
                    break;
                }
            }
            assert!(saw_colored, "picture playback produced no colored pixels");

            sk_surface_unref(surface);
            sk_picture_unref(picture);
            sk_picture_recorder_delete(rec);
        }
    }

    #[test]
    fn test_picture_playback_null_inputs_ffi() {
        unsafe {
            assert!(!sk_picture_playback(ptr::null(), ptr::null_mut()));
            assert!(!sk_last_call_panicked());
        }
    }

    #[test]
    fn test_region_set_rect_and_ops_ffi() {
        unsafe {
            let r = sk_region_new();
            assert!(sk_region_is_empty(r));

            let rect = sk_irect_t {
                left: 0,
                top: 0,
                right: 10,
                bottom: 10,
            };
            assert!(sk_region_set_rect(r, &raw const rect));
            assert!(!sk_region_is_empty(r));
            assert!(sk_region_contains(r, 5, 5));
            assert!(!sk_region_contains(r, 15, 15));

            let mut bounds = sk_irect_t::default();
            assert!(sk_region_get_bounds(r, &raw mut bounds));
            assert_eq!(bounds.right, 10);
            assert_eq!(bounds.bottom, 10);

            // Union with another rect.
            let other = sk_irect_t {
                left: 8,
                top: 8,
                right: 20,
                bottom: 20,
            };
            assert!(sk_region_op_rect(
                r,
                &raw const other,
                SkRegionOp::Union as u32
            ));
            assert!(sk_region_get_bounds(r, &raw mut bounds));
            assert_eq!(bounds.right, 20);

            // Test intersect op.
            assert!(sk_region_op_rect(
                r,
                &raw const other,
                SkRegionOp::Intersect as u32
            ));

            sk_region_ref(r);
            assert_eq!(sk_refcnt_get_count(r as *const sk_refcnt_t), 2);
            sk_region_unref(r);
            sk_region_unref(r);
        }
    }

    #[test]
    fn test_patheffect_dash_and_trim_ffi() {
        unsafe {
            // Dash effect.
            let intervals = [4.0f32, 2.0];
            let dash = sk_patheffect_new_dash(intervals.as_ptr(), intervals.len(), 0.0);
            assert!(!dash.is_null());

            // Trim effect.
            let trim = sk_patheffect_new_trim(0.25, 0.75, SkTrimMode::Normal as u32);
            assert!(!trim.is_null());

            // Attaching to a paint.
            let paint = sk_paint_new();
            sk_paint_set_path_effect(paint, dash);
            sk_paint_set_path_effect(paint, ptr::null());
            sk_paint_set_path_effect(paint, trim);
            sk_paint_set_path_effect(paint, ptr::null());
            sk_paint_unref(paint);

            // Test inverted trim mode.
            let inverted = sk_patheffect_new_trim(0.0, 1.0, SkTrimMode::Inverted as u32);
            assert!(!inverted.is_null());
            sk_patheffect_unref(inverted);

            // Rejects empty dash.
            let empty = sk_patheffect_new_dash(ptr::null(), 0, 0.0);
            assert!(empty.is_null());

            sk_patheffect_unref(dash);
            sk_patheffect_unref(trim);
        }
    }

    #[test]
    fn test_patheffect_dash_applies_on_draw_ffi() {
        // Verifies that attaching a dash effect to a paint actually changes
        // rasterized output (proof that Paint::set_path_effect is plumbed
        // through to Canvas::draw_path).
        unsafe {
            let surface = sk_surface_new_raster(32, 4);
            let canvas = sk_surface_lock_canvas(surface);
            sk_canvas_clear(canvas, 0xFF00_0000);

            let b = sk_pathbuilder_new();
            sk_pathbuilder_move_to(b, 0.0, 2.0);
            sk_pathbuilder_line_to(b, 32.0, 2.0);
            let path = sk_pathbuilder_detach(b);
            sk_pathbuilder_delete(b);

            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            sk_paint_set_style(paint, 1); // stroke
            sk_paint_set_stroke_width(paint, 2.0);
            let intervals = [4.0f32, 4.0];
            let dash = sk_patheffect_new_dash(intervals.as_ptr(), intervals.len(), 0.0);
            sk_paint_set_path_effect(paint, dash);

            sk_canvas_draw_path(canvas, path, paint);
            sk_canvas_release(canvas);

            let mut buf = vec![0u8; 32 * 4 * 4];
            sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            // Count white-ish pixels on the center row. A solid stroke would
            // produce ~32 * 2 = ~64 filled pixels; a 4-on/4-off dash should
            // produce roughly half.
            let mut filled = 0;
            for x in 0..32 {
                let i = (2 * 32 + x) * 4; // y=2
                if buf[i] > 128 {
                    filled += 1;
                }
            }
            assert!(
                filled > 4 && filled < 28,
                "dash effect produced {filled} filled pixels on center row — expected partial coverage"
            );

            sk_patheffect_unref(dash);
            sk_paint_unref(paint);
            sk_path_unref(path);
            sk_surface_unref(surface);
        }
    }

    // -- Task 11 conformance-audit regression tests --------------------

    #[test]
    fn test_decode_color_type_matches_upstream_numbering() {
        // include/core/SkColorType.h numbering.
        assert_eq!(decode_color_type(0), Some(ColorType::Unknown));
        assert_eq!(decode_color_type(4), Some(ColorType::Rgba8888));
        assert_eq!(decode_color_type(5), Some(ColorType::Rgb888x));
        assert_eq!(decode_color_type(6), Some(ColorType::Bgra8888));
        assert_eq!(decode_color_type(7), Some(ColorType::Rgba1010102));
        // Wholly unknown / unsupported values are errors, never a silent
        // fallback to RGBA.
        assert_eq!(decode_color_type(255), None);
        assert_eq!(decode_color_type(u32::MAX), None);
    }

    #[test]
    fn test_surface_new_raster_with_info_rejects_unknown_color_type() {
        unsafe {
            let info = sk_imageinfo_t {
                width: 4,
                height: 4,
                color_type: u32::MAX,
                alpha_type: 2, // Premul
            };
            let surface = sk_surface_new_raster_with_info(&raw const info);
            assert!(surface.is_null());
        }
    }

    #[test]
    fn test_surface_new_raster_with_info_bgra8888_is_index_6() {
        // Index 5 (kRGB_888x) must NOT be interpreted as BGRA — the decode
        // table must keep it distinct from index 6 (kBGRA_8888), matching
        // upstream SkColorType numbering.
        assert_eq!(decode_color_type(5), Some(ColorType::Rgb888x));
        assert_eq!(decode_color_type(6), Some(ColorType::Bgra8888));

        // The raster backend now supports BGRA-order 8888 surfaces (stored
        // RGBA internally, presented BGRA on snapshot readback), so a BGRA
        // request creates a surface rather than reporting failure. This is
        // what makes N32 raster surfaces work on Windows (where n32 is BGRA).
        unsafe {
            let info_bgra = sk_imageinfo_t {
                width: 2,
                height: 2,
                color_type: 6,
                alpha_type: 2, // Premul
            };
            let s = sk_surface_new_raster_with_info(&raw const info_bgra);
            assert!(!s.is_null());
            sk_surface_unref(s);
        }
    }

    #[test]
    fn test_blend_mode_decodes_all_29_modes() {
        let decode = |v: u32| u8::try_from(v).ok().and_then(BlendMode::from_u8);
        for v in 0..=28u32 {
            assert!(decode(v).is_some(), "mode {v} should decode");
        }
        assert_eq!(decode(29), None);
        assert_eq!(decode(u32::MAX), None);
    }

    #[test]
    fn test_paint_set_blend_mode_round_trips_and_rejects_invalid() {
        unsafe {
            let paint = sk_paint_new();
            // Default is SrcOver (3).
            assert_eq!(sk_paint_get_blend_mode(paint), BlendMode::SrcOver as u32);

            assert!(sk_paint_set_blend_mode(paint, BlendMode::Multiply as u32));
            assert_eq!(sk_paint_get_blend_mode(paint), BlendMode::Multiply as u32);

            assert!(sk_paint_set_blend_mode(paint, BlendMode::Luminosity as u32));
            assert_eq!(sk_paint_get_blend_mode(paint), BlendMode::Luminosity as u32);

            // Out-of-range value is rejected; blend mode is left unchanged.
            assert!(!sk_paint_set_blend_mode(paint, 29));
            assert_eq!(sk_paint_get_blend_mode(paint), BlendMode::Luminosity as u32);

            sk_paint_unref(paint);
        }
    }

    #[test]
    fn test_sk_version_matches_cargo_pkg_version() {
        unsafe {
            let ptr = sk_version();
            let cstr = std::ffi::CStr::from_ptr(ptr);
            let s = cstr.to_str().unwrap();
            assert!(
                s.contains(env!("CARGO_PKG_VERSION")),
                "sk_version() = {s:?} does not contain CARGO_PKG_VERSION"
            );
        }
    }

    #[test]
    fn test_matrix_concat_in_place_aliasing_is_safe() {
        unsafe {
            let mut m: sk_matrix_t = Matrix::translate(2.0, 3.0).into();
            let scale: sk_matrix_t = Matrix::scale(2.0, 2.0).into();
            // result aliases `a` — must not UB and must produce the correct
            // concatenation (m = m * scale).
            let ptr = &raw mut m;
            sk_matrix_concat(ptr, ptr, &raw const scale);
            let expect: sk_matrix_t = Matrix::translate(2.0, 3.0)
                .concat(&Matrix::scale(2.0, 2.0))
                .into();
            for (a, b) in m.values.iter().zip(expect.values.iter()) {
                assert!((a - b).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_matrix_invert_in_place_aliasing_is_safe() {
        unsafe {
            let mut m: sk_matrix_t = Matrix::scale(2.0, 4.0).into();
            let ptr = &raw mut m;
            assert!(sk_matrix_invert(ptr, ptr));
            let expect: sk_matrix_t = Matrix::scale(2.0, 4.0).invert().unwrap().into();
            for (a, b) in m.values.iter().zip(expect.values.iter()) {
                assert!((a - b).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_matrix_map_point_in_place_aliasing_is_safe() {
        unsafe {
            let m: sk_matrix_t = Matrix::translate(1.0, 1.0).into();
            let mut p = sk_point_t { x: 2.0, y: 3.0 };
            let ptr = &raw mut p;
            sk_matrix_map_point(&raw const m, ptr, ptr);
            assert!((p.x - 3.0).abs() < f32::EPSILON);
            assert!((p.y - 4.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_surface_lock_canvas_rejects_second_lock() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let c1 = sk_surface_lock_canvas(surface);
            assert!(!c1.is_null());

            // A second lock while `c1` is outstanding must fail.
            let c2 = sk_surface_lock_canvas(surface);
            assert!(c2.is_null());

            sk_canvas_release(c1);

            // After releasing, locking again succeeds.
            let c3 = sk_surface_lock_canvas(surface);
            assert!(!c3.is_null());
            sk_canvas_release(c3);

            sk_surface_unref(surface);
        }
    }

    /// Alpha of pixel (x, y) of a 20x20 surface.
    unsafe fn surface_alpha(surface: *mut sk_surface_t, x: usize, y: usize) -> u8 {
        let mut buf = vec![0u8; 20 * 20 * 4];
        unsafe { sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len()) };
        buf[(y * 20 + x) * 4 + 3]
    }

    fn rect(left: f32, top: f32, right: f32, bottom: f32) -> sk_rect_t {
        sk_rect_t {
            left,
            top,
            right,
            bottom,
        }
    }

    #[test]
    fn test_canvas_clips_keep_the_matrix_they_were_set_under() {
        // SkCanvas fixes a clip in device space when it is set; a later
        // translate must not move it.
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let r = rect(0.0, 0.0, 5.0, 5.0);
            assert!(sk_canvas_clip_rect(canvas, &raw const r, 1, false));
            sk_canvas_translate(canvas, 10.0, 10.0);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            let all = rect(-10.0, -10.0, 10.0, 10.0);
            sk_canvas_draw_rect(canvas, &raw const all, paint);
            sk_paint_unref(paint);
            sk_canvas_release(canvas);
            assert_eq!(surface_alpha(surface, 2, 2), 255, "inside the clip");
            assert_eq!(surface_alpha(surface, 12, 12), 0, "the clip did not move");
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_save_layer_composites_with_the_layer_paint() {
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let layer_paint = sk_paint_new();
            sk_paint_set_color(layer_paint, 0x80FF_FFFF);
            sk_canvas_save_layer(canvas, ptr::null(), layer_paint);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            let r = rect(0.0, 0.0, 20.0, 20.0);
            sk_canvas_draw_rect(canvas, &raw const r, paint);
            sk_canvas_restore(canvas);
            sk_paint_unref(paint);
            sk_paint_unref(layer_paint);
            sk_canvas_release(canvas);
            let a = surface_alpha(surface, 5, 5);
            assert!((126..=130).contains(&a), "half-alpha layer: {a}");
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_release_composites_open_layers() {
        // Like ~SkCanvas, releasing with a layer still open composites it
        // instead of dropping its content.
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let layer_paint = sk_paint_new();
            sk_paint_set_color(layer_paint, 0x80FF_FFFF);
            sk_canvas_save_layer(canvas, ptr::null(), layer_paint);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            let r = rect(0.0, 0.0, 20.0, 20.0);
            sk_canvas_draw_rect(canvas, &raw const r, paint);
            sk_paint_unref(paint);
            sk_paint_unref(layer_paint);
            sk_canvas_release(canvas);
            let a = surface_alpha(surface, 5, 5);
            assert!(
                (126..=130).contains(&a),
                "the open layer was composited: {a}"
            );
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_clip_ibounds_are_in_device_space_inside_a_layer() {
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let r = rect(2.0, 3.0, 10.0, 12.0);
            assert!(sk_canvas_clip_rect(canvas, &raw const r, 1, false));
            sk_canvas_save_layer(canvas, ptr::null(), ptr::null());
            let mut out = sk_irect_t {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            assert!(sk_canvas_get_clip_ibounds(canvas, &raw mut out));
            assert_eq!((out.left, out.top, out.right, out.bottom), (2, 3, 10, 12));
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_locked_canvas_keeps_its_surface_alive() {
        // The lock holds a reference, so unreffing the surface while it is
        // locked cannot free the buffer release composites into.
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            assert_eq!(sk_surface_get_refcnt(surface), 2);
            sk_canvas_release(canvas);
            assert_eq!(sk_surface_get_refcnt(surface), 1);
            // Relocking works: the lock entry was removed.
            let again = sk_surface_lock_canvas(surface);
            assert!(!again.is_null());
            sk_canvas_release(again);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_nested_filtered_layers_stay_bounded_by_the_surface() {
        // Like Skia, each filtered layer covers its parent plus its own
        // filter's reach (here capped at the surface size, 20): growth is
        // linear in depth (-60..80 after three), not geometric as when the
        // cap was taken from the parent layer (27x after three).
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let blur = sk_imagefilter_new_blur(1000.0, 1000.0, 0);
            let paint = sk_paint_new();
            sk_paint_set_image_filter(paint, blur);
            for _ in 0..3 {
                sk_canvas_save_layer(canvas, ptr::null(), paint);
            }
            let mut out = sk_irect_t {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            assert!(sk_canvas_get_clip_ibounds(canvas, &raw mut out));
            assert!(
                out.left >= -60 && out.top >= -60 && out.right <= 80 && out.bottom <= 80,
                "layer bounds {:?}",
                (out.left, out.top, out.right, out.bottom)
            );
            sk_canvas_release(canvas);
            sk_paint_unref(paint);
            sk_imagefilter_unref(blur);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_empty_clip_inside_a_layer_is_all_zero() {
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let outer = rect(5.0, 5.0, 15.0, 15.0);
            assert!(sk_canvas_clip_rect(canvas, &raw const outer, 1, false));
            sk_canvas_save_layer(canvas, ptr::null(), ptr::null());
            let miss = rect(100.0, 100.0, 110.0, 110.0);
            assert!(sk_canvas_clip_rect(canvas, &raw const miss, 1, false));
            let mut out = sk_irect_t {
                left: 1,
                top: 1,
                right: 1,
                bottom: 1,
            };
            sk_canvas_get_clip_ibounds(canvas, &raw mut out);
            assert_eq!((out.left, out.top, out.right, out.bottom), (0, 0, 0, 0));
            sk_canvas_release(canvas);
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_canvas_save_layer_hint_limits_content_not_the_filter_output() {
        // Skia's bounds hint clips what is drawn into the layer, but a blur
        // still spreads its result past the hint.
        unsafe {
            let surface = sk_surface_new_raster(20, 20);
            let canvas = sk_surface_lock_canvas(surface);
            let blur = sk_imagefilter_new_blur(1.5, 1.5, 0);
            let layer_paint = sk_paint_new();
            sk_paint_set_image_filter(layer_paint, blur);
            let hint = rect(8.0, 8.0, 12.0, 12.0);
            sk_canvas_save_layer(canvas, &raw const hint, layer_paint);
            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            sk_canvas_draw_rect(canvas, &raw const hint, paint);
            let outside = rect(0.0, 0.0, 3.0, 3.0);
            sk_canvas_draw_rect(canvas, &raw const outside, paint);
            sk_canvas_restore(canvas);
            sk_paint_unref(paint);
            sk_paint_unref(layer_paint);
            sk_imagefilter_unref(blur);
            sk_canvas_release(canvas);
            assert!(
                surface_alpha(surface, 6, 10) > 0,
                "the blur spreads past the hint"
            );
            assert_eq!(
                surface_alpha(surface, 1, 1),
                0,
                "content outside the hint is dropped"
            );
            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_surface_draw_rejected_while_locked() {
        unsafe {
            let surface = sk_surface_new_raster(10, 10);
            let canvas = sk_surface_lock_canvas(surface);
            assert!(!canvas.is_null());

            let paint = sk_paint_new();
            sk_paint_set_color(paint, 0xFFFF_FFFF);
            let rect = sk_rect_t {
                left: 0.0,
                top: 0.0,
                right: 5.0,
                bottom: 5.0,
            };
            // Drawing directly on the surface (bypassing the locked canvas)
            // must be a no-op while a canvas lock is outstanding.
            sk_surface_draw_rect(surface, &raw const rect, paint);
            assert!(!sk_last_call_panicked());

            sk_paint_unref(paint);
            sk_canvas_release(canvas);

            // Verify it was actually a no-op: the pixel buffer is still
            // all-zero (the surface was never cleared/drawn on).
            let mut buf = vec![0xAAu8; 10 * 10 * 4];
            sk_surface_read_pixels(surface, buf.as_mut_ptr(), buf.len());
            assert!(
                buf.iter().all(|&b| b == 0),
                "surface was drawn on while locked"
            );

            sk_surface_unref(surface);
        }
    }

    #[test]
    fn test_picture_recorder_delete_then_use_recording_canvas_is_safe() {
        unsafe {
            let recorder = sk_picture_recorder_new();
            let rc = sk_picture_recorder_begin_recording(recorder, ptr::null());
            assert!(!rc.is_null());

            // Delete the recorder while the recording-canvas handle is
            // still outstanding — a naive raw-pointer implementation would
            // leave `rc` dangling. Every subsequent `sk_recording_canvas_*`
            // call must detect this and return an error/no-op rather than
            // dereferencing freed memory.
            sk_picture_recorder_delete(recorder);

            let save_count = sk_recording_canvas_save(rc);
            assert_eq!(save_count, 0);
            sk_recording_canvas_restore(rc); // must not crash
            sk_recording_canvas_draw_rect(
                rc,
                &sk_rect_t {
                    left: 0.0,
                    top: 0.0,
                    right: 1.0,
                    bottom: 1.0,
                },
                ptr::null(),
            ); // must not crash

            assert!(!sk_last_call_panicked());
            sk_recording_canvas_release(rc);
        }
    }

    #[test]
    fn test_refcnt_get_count_rejects_non_tagged_pointer() {
        unsafe {
            // A live, non-refcounted allocation of sufficient size must
            // fail the tag check and return 0, not a bogus count.
            let junk: [u32; 4] = [0xDEAD_BEEF, 1, 2, 3];
            assert_eq!(sk_refcnt_get_count(junk.as_ptr().cast::<sk_refcnt_t>()), 0);
        }
    }
}
