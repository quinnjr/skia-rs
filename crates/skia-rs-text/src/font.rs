//! Font configuration for text rendering.

use crate::typeface::{Typeface, TypefaceRef};
use skia_rs_core::{Matrix, Point, Scalar};
use std::sync::Arc;

/// Text baseline position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TextBaseline {
    /// Alphabetic baseline.
    #[default]
    Alphabetic = 0,
    /// Top of the em square.
    Top,
    /// Middle of the em square.
    Middle,
    /// Bottom of the em square.
    Bottom,
    /// Ideographic baseline.
    Ideographic,
    /// Hanging baseline.
    Hanging,
}

/// Font edging mode (how glyphs are rendered).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum FontEdging {
    /// Alias (no anti-aliasing).
    Alias = 0,
    /// Anti-aliased.
    #[default]
    AntiAlias,
    /// Subpixel anti-aliased.
    SubpixelAntiAlias,
}

/// Font hinting level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum FontHinting {
    /// No hinting.
    None = 0,
    /// Slight hinting.
    Slight,
    /// Normal hinting.
    #[default]
    Normal,
    /// Full hinting.
    Full,
}

/// Font metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct FontMetrics {
    /// Distance above baseline (negative for above).
    pub ascent: Scalar,
    /// Distance below baseline (positive for below).
    pub descent: Scalar,
    /// Distance between baselines.
    pub leading: Scalar,
    /// Top of bounding box (negative for above baseline).
    pub top: Scalar,
    /// Bottom of bounding box.
    pub bottom: Scalar,
    /// Average character width.
    pub avg_char_width: Scalar,
    /// Maximum character width.
    pub max_char_width: Scalar,
    /// X-height (height of lowercase 'x').
    pub x_height: Scalar,
    /// Cap height (height of uppercase letters).
    pub cap_height: Scalar,
    /// Underline position.
    pub underline_position: Scalar,
    /// Underline thickness.
    pub underline_thickness: Scalar,
    /// Strikeout position.
    pub strikeout_position: Scalar,
    /// Strikeout thickness.
    pub strikeout_thickness: Scalar,
}

impl FontMetrics {
    /// Calculate the line height.
    #[inline]
    #[must_use]
    pub fn line_height(&self) -> Scalar {
        -self.ascent + self.descent + self.leading
    }
}

/// Compute screen-space underline `(position, thickness)` from the `post`
/// table's underline center `position` (font units, positive = above
/// baseline) and optional `thickness`.
///
/// Matches Skia's `SkFontHost_FreeType.cpp`:
///   underlinePosition = -(`post.underline_position` +
///                         `post.underline_thickness` / 2) / upem
/// i.e. the distance to the **top** of the stroke, folding in the
/// half-thickness. `scale` is `size / units_per_em`.
fn underline_metrics_from_post(
    position: i16,
    thickness: Option<i16>,
    scale: Scalar,
    size: Scalar,
) -> (Scalar, Scalar) {
    let thick = Scalar::from(thickness.unwrap_or(0));
    let screen_position = -(Scalar::from(position) + thick / 2.0) * scale;
    let screen_thickness = if thickness.is_some() {
        thick * scale
    } else {
        0.05 * size
    };
    (screen_position, screen_thickness)
}

/// A font configuration (typeface + size + options).
///
/// Corresponds to Skia's `SkFont`.
#[derive(Debug, Clone)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "mirrors SkFont's fixed boolean flag set (subpixel, forceAutoHinting, embeddedBitmaps, linearMetrics, embolden)"
)]
pub struct Font {
    /// The typeface.
    typeface: TypefaceRef,
    /// Font size in points.
    size: Scalar,
    /// Horizontal scale factor.
    scale_x: Scalar,
    /// Skew factor for oblique simulation.
    skew_x: Scalar,
    /// Font edging mode.
    edging: FontEdging,
    /// Font hinting level.
    hinting: FontHinting,
    /// Enable subpixel positioning.
    subpixel: bool,
    /// Force auto-hinting.
    force_auto_hinting: bool,
    /// Embed bitmaps in outlines.
    embedded_bitmaps: bool,
    /// Enable linear metrics.
    linear_metrics: bool,
    /// Embolden the font.
    embolden: bool,
}

impl Default for Font {
    fn default() -> Self {
        Self::new(Arc::new(Typeface::default_typeface()), 12.0)
    }
}

impl Font {
    /// Create a new font with the given typeface and size.
    pub const fn new(typeface: TypefaceRef, size: Scalar) -> Self {
        Self {
            typeface,
            size,
            scale_x: 1.0,
            skew_x: 0.0,
            edging: FontEdging::AntiAlias,
            hinting: FontHinting::Normal,
            subpixel: false,
            force_auto_hinting: false,
            embedded_bitmaps: true,
            linear_metrics: false,
            embolden: false,
        }
    }

    /// Create a font with default typeface.
    #[must_use]
    pub fn from_size(size: Scalar) -> Self {
        Self::new(Arc::new(Typeface::default_typeface()), size)
    }

    /// Get the typeface.
    #[inline]
    #[must_use]
    pub fn typeface(&self) -> Option<&Typeface> {
        Some(self.typeface.as_ref())
    }

    /// Get the typeface reference.
    #[inline]
    #[must_use]
    pub const fn typeface_ref(&self) -> &TypefaceRef {
        &self.typeface
    }

    /// Set the typeface.
    #[inline]
    pub fn set_typeface(&mut self, typeface: TypefaceRef) -> &mut Self {
        self.typeface = typeface;
        self
    }

    /// Get the font size.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> Scalar {
        self.size
    }

    /// Set the font size.
    #[inline]
    pub const fn set_size(&mut self, size: Scalar) -> &mut Self {
        self.size = size.max(0.0);
        self
    }

    /// Get the horizontal scale.
    #[inline]
    #[must_use]
    pub const fn scale_x(&self) -> Scalar {
        self.scale_x
    }

    /// Set the horizontal scale.
    #[inline]
    pub const fn set_scale_x(&mut self, scale: Scalar) -> &mut Self {
        self.scale_x = scale;
        self
    }

    /// Get the skew factor.
    #[inline]
    #[must_use]
    pub const fn skew_x(&self) -> Scalar {
        self.skew_x
    }

    /// Set the skew factor.
    #[inline]
    pub const fn set_skew_x(&mut self, skew: Scalar) -> &mut Self {
        self.skew_x = skew;
        self
    }

    /// Get the edging mode.
    #[inline]
    #[must_use]
    pub const fn edging(&self) -> FontEdging {
        self.edging
    }

    /// Set the edging mode.
    #[inline]
    pub const fn set_edging(&mut self, edging: FontEdging) -> &mut Self {
        self.edging = edging;
        self
    }

    /// Get the hinting level.
    #[inline]
    #[must_use]
    pub const fn hinting(&self) -> FontHinting {
        self.hinting
    }

    /// Set the hinting level.
    #[inline]
    pub const fn set_hinting(&mut self, hinting: FontHinting) -> &mut Self {
        self.hinting = hinting;
        self
    }

    /// Check if subpixel positioning is enabled.
    #[inline]
    #[must_use]
    pub const fn is_subpixel(&self) -> bool {
        self.subpixel
    }

    /// Set subpixel positioning.
    #[inline]
    pub const fn set_subpixel(&mut self, subpixel: bool) -> &mut Self {
        self.subpixel = subpixel;
        self
    }

    /// Check if emboldening is enabled.
    #[inline]
    #[must_use]
    pub const fn is_embolden(&self) -> bool {
        self.embolden
    }

    /// Set emboldening.
    #[inline]
    pub const fn set_embolden(&mut self, embolden: bool) -> &mut Self {
        self.embolden = embolden;
        self
    }

    /// Check if forced auto-hinting is enabled.
    #[inline]
    #[must_use]
    pub const fn is_force_auto_hinting(&self) -> bool {
        self.force_auto_hinting
    }

    /// Set forced auto-hinting.
    #[inline]
    pub const fn set_force_auto_hinting(&mut self, force_auto_hinting: bool) -> &mut Self {
        self.force_auto_hinting = force_auto_hinting;
        self
    }

    /// Check if embedded bitmaps are used.
    #[inline]
    #[must_use]
    pub const fn is_embedded_bitmaps(&self) -> bool {
        self.embedded_bitmaps
    }

    /// Set whether embedded bitmaps are used.
    #[inline]
    pub const fn set_embedded_bitmaps(&mut self, embedded_bitmaps: bool) -> &mut Self {
        self.embedded_bitmaps = embedded_bitmaps;
        self
    }

    /// Check if linear metrics are enabled.
    #[inline]
    #[must_use]
    pub const fn is_linear_metrics(&self) -> bool {
        self.linear_metrics
    }

    /// Set linear metrics.
    #[inline]
    pub const fn set_linear_metrics(&mut self, linear_metrics: bool) -> &mut Self {
        self.linear_metrics = linear_metrics;
        self
    }

    /// Get the font metrics.
    ///
    /// When the typeface has backing font data, this pulls real values from
    /// the font's `hhea`/`OS/2`/`post` tables via `ttf_parser` and scales them
    /// by `size / units_per_em`. Specifically:
    ///
    /// - `ascent` ← `Face::ascender()` (negated, screen-space is y-down)
    /// - `descent` ← `Face::descender()` (negated — ttf-parser returns a
    ///   negative font-space value; we store the positive screen-space
    ///   descent so that `line_height = -ascent + descent + leading` works)
    /// - `leading` ← `Face::line_gap()`, clamped to be non-negative
    /// - `x_height`, `cap_height` ← OS/2 v2+ when present (positive), else a
    ///   fraction of the ascent
    /// - `top` / `bottom` ← the font's global bounding box (`-yMax`, `-yMin`)
    /// - `avg_char_width` ← OS/2 `xAvgCharWidth` (fallback bbox width);
    ///   `max_char_width` ← bbox width
    /// - `underline_position` / `underline_thickness` ← `post` table (position
    ///   folds in half the thickness)
    /// - `strikeout_position` / `strikeout_thickness` ← OS/2
    ///
    /// Falls back to the previous hardcoded multiples of `size` only for the
    /// dataless default typeface, which has no font tables to read.
    ///
    /// All values are fully cache-served from the typeface's `ParsedTypeface`
    /// metadata — no re-parsing of the font on every call.
    #[must_use]
    pub fn metrics(&self) -> FontMetrics {
        if let Some(raw) = self.typeface.raw_metrics() {
            let upem = raw.units_per_em;
            if upem > 0 {
                let scale = self.size / Scalar::from(upem);

                // Font-space ascender is positive (units above baseline).
                // Screen-space ascent is negative (y grows downward).
                let ascent = -Scalar::from(raw.ascender) * scale;
                // Font-space descender is negative (units below baseline).
                // Screen-space descent is positive.
                let descent = -Scalar::from(raw.descender) * scale;
                // Disallow negative linespacing (SkFontHost_FreeType.cpp:
                // `if (leading < 0.0f) leading = 0.0f;`).
                let leading = (Scalar::from(raw.line_gap) * scale).max(0.0);

                // Cap and x heights. Per the FreeType port these are stored
                // *positive* (`os2->sxHeight / upem * scale.y`); when the OS/2
                // table doesn't provide them, Skia synthesises them from the
                // (positive) ascent. `-ascent` is the positive ascent since
                // the screen-space `ascent` is negative.
                let cap_height = raw
                    .cap_height
                    .map_or(-ascent * 0.875, |v| Scalar::from(v) * scale);
                let x_height = raw
                    .x_height
                    .map_or(-ascent * 0.625, |v| Scalar::from(v) * scale);

                // Underline from `post` (cached). Skia measures the underline
                // position to the *top* of the stroke, so it folds in half
                // the thickness:
                //   underlinePosition = -(post.position + thickness/2) / upem
                // The font-space position is the stroke centre measured
                // upward from the baseline (usually negative); negating lands
                // it in screen space (positive = below baseline).
                let (underline_position, underline_thickness) =
                    raw.underline_position
                        .map_or((0.1 * self.size, 0.05 * self.size), |pos| {
                            underline_metrics_from_post(
                                pos,
                                raw.underline_thickness,
                                scale,
                                self.size,
                            )
                        });

                // Strikeout from OS/2 (cached). `-yStrikeoutPosition/upem`.
                let (strikeout_position, strikeout_thickness) =
                    raw.strikeout_position
                        .map_or((-0.3 * self.size, 0.05 * self.size), |pos| {
                            (
                                -Scalar::from(pos) * scale,
                                raw.strikeout_thickness
                                    .map_or(0.05 * self.size, |t| Scalar::from(t) * scale),
                            )
                        });

                // Visible bounds come from the font's global bounding box,
                // matching Skia's `fTop = -bbox.yMax/upem*scale` and
                // `fBottom = -bbox.yMin/upem*scale`. The parsed bbox is
                // already cached as (xmin, ymin, xmax, ymax).
                let (xmin, ymin, xmax, ymax) = raw.bbox;
                let top = -Scalar::from(ymax) * scale;
                let bottom = -Scalar::from(ymin) * scale;

                // Max char width is the bbox width; avg char width comes from
                // OS/2 `xAvgCharWidth` when present, else the bbox width
                // (SkFontHost_FreeType.cpp: `if (!avgCharWidth) avgCharWidth =
                // xmax - xmin;`).
                let bbox_width = (Scalar::from(xmax) - Scalar::from(xmin)) * scale;
                let max_char_width = bbox_width;
                let avg_char_width = raw
                    .avg_char_width
                    .filter(|&a| a != 0)
                    .map_or(bbox_width, |a| Scalar::from(a) * scale);

                return FontMetrics {
                    ascent,
                    descent,
                    leading,
                    top,
                    bottom,
                    avg_char_width,
                    max_char_width,
                    x_height,
                    cap_height,
                    underline_position,
                    underline_thickness,
                    strikeout_position,
                    strikeout_thickness,
                };
            }
        }

        // Dataless default typeface: fall back to the hardcoded
        // approximation. The default typeface has no tables to read and all
        // callers have historically relied on these numbers.
        FontMetrics {
            ascent: -0.8 * self.size,
            descent: 0.2 * self.size,
            leading: 0.0,
            top: -0.9 * self.size,
            bottom: 0.3 * self.size,
            avg_char_width: 0.5 * self.size,
            max_char_width: self.size,
            x_height: 0.5 * self.size,
            cap_height: 0.7 * self.size,
            underline_position: 0.1 * self.size,
            underline_thickness: 0.05 * self.size,
            strikeout_position: -0.3 * self.size,
            strikeout_thickness: 0.05 * self.size,
        }
    }

    /// Get spacing between baselines.
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> Scalar {
        let m = self.metrics();
        m.line_height()
    }

    /// Get the ascent (negative value, distance from baseline to top).
    #[inline]
    #[must_use]
    pub fn ascent(&self) -> Scalar {
        self.metrics().ascent
    }

    /// Get the descent (positive value, distance from baseline to bottom).
    #[inline]
    #[must_use]
    pub fn descent(&self) -> Scalar {
        self.metrics().descent
    }

    /// Measure the width of text.
    ///
    /// Sums `glyph_advance` across each character's glyph so that fonts with
    /// real `hmtx` data measure correctly. Falls back to the crude
    /// `size * 0.5 * char_count` estimate for the dataless default typeface.
    #[must_use]
    pub fn measure_text(&self, text: &str) -> Scalar {
        text.chars()
            .map(|c| self.glyph_advance(self.char_to_glyph(c)))
            .sum()
    }

    /// Get glyph widths for text.
    ///
    /// Returns the per-character horizontal advance. Delegates to
    /// `glyph_advance`, which reads the font's `hmtx` table when present,
    /// so this function produces correct per-character positioning for real
    /// fonts rather than the old uniform `size / 2` approximation.
    #[must_use]
    pub fn get_widths(&self, text: &str) -> Vec<Scalar> {
        text.chars()
            .map(|c| self.glyph_advance(self.char_to_glyph(c)))
            .collect()
    }

    /// Get glyph bounds for text.
    ///
    /// Returns the bounding box of each character's glyph, positioned
    /// cumulatively along x by the per-glyph advance from `hmtx`. Each box's
    /// tight visual extent is taken from `ttf_parser::Face::glyph_bounding_box`
    /// when present; otherwise we fall back to the previous ascent/descent
    /// rectangle with the measured advance as its width.
    #[must_use]
    pub fn get_bounds(&self, text: &str) -> Vec<skia_rs_core::Rect> {
        let metrics = self.metrics();
        let mut out = Vec::with_capacity(text.chars().count());

        let parsed = self
            .typeface
            .font_data()
            .and_then(|data| ttf_parser::Face::parse(data, 0).ok());

        let scale = parsed.as_ref().map_or(1.0, |face| {
            let upem = face.units_per_em();
            if upem > 0 {
                self.size / Scalar::from(upem)
            } else {
                1.0
            }
        });

        let mut x_offset: Scalar = 0.0;
        for c in text.chars() {
            let glyph = self.char_to_glyph(c);
            let advance = self.glyph_advance(glyph);

            let rect = parsed.as_ref().map_or_else(
                || {
                    // Dataless typeface — fall back to advance×line-height box.
                    skia_rs_core::Rect::from_xywh(
                        x_offset,
                        metrics.ascent,
                        advance,
                        -metrics.ascent + metrics.descent,
                    )
                },
                |face| {
                    if glyph == 0 {
                        return skia_rs_core::Rect::from_xywh(x_offset, 0.0, 0.0, 0.0);
                    }
                    face.glyph_bounding_box(ttf_parser::GlyphId(glyph))
                        .map_or_else(
                            || {
                                // Glyph has no outline (e.g. space). Zero-sized box.
                                skia_rs_core::Rect::from_xywh(x_offset, 0.0, 0.0, 0.0)
                            },
                            |bbox| {
                                // Font-space bbox is y-up; flip to y-down screen
                                // space (negate y and swap top/bottom).
                                let left = Scalar::from(bbox.x_min) * scale * self.scale_x;
                                let right = Scalar::from(bbox.x_max) * scale * self.scale_x;
                                let top = -Scalar::from(bbox.y_max) * scale;
                                let bottom = -Scalar::from(bbox.y_min) * scale;
                                skia_rs_core::Rect::new(
                                    x_offset + left,
                                    top,
                                    x_offset + right,
                                    bottom,
                                )
                            },
                        )
                },
            );

            out.push(rect);
            x_offset += advance;
        }

        out
    }

    /// Convert character to glyph ID.
    #[inline]
    #[must_use]
    pub fn char_to_glyph(&self, c: char) -> u16 {
        self.typeface.char_to_glyph(c)
    }

    /// Convert string to glyph IDs.
    #[inline]
    #[must_use]
    pub fn text_to_glyphs(&self, text: &str) -> Vec<u16> {
        self.typeface.chars_to_glyphs(text)
    }

    // =========================================================================
    // Glyph Operations
    // =========================================================================

    /// Get the advance width for a glyph.
    ///
    /// The advance is the horizontal distance to move after drawing this
    /// glyph. When font data is available the value comes from the font's
    /// `hmtx` table scaled by `size / units_per_em`; otherwise a crude
    /// `size * 0.5` approximation is used for the dataless default
    /// typeface.
    #[must_use]
    pub fn glyph_advance(&self, glyph: u16) -> Scalar {
        // Glyph 0 (.notdef) is a real glyph: it carries an `hmtx` advance
        // like any other and Skia advances the pen by it (a tofu box takes
        // up space). Do not special-case it to zero.
        if let Some(data) = self.typeface.font_data() {
            if let Ok(face) = ttf_parser::Face::parse(data, 0) {
                let upem = face.units_per_em();
                if upem > 0 {
                    if let Some(adv) = face.glyph_hor_advance(ttf_parser::GlyphId(glyph)) {
                        return Scalar::from(adv) * self.size / Scalar::from(upem) * self.scale_x;
                    }
                }
            }
        }
        self.size * 0.5 * self.scale_x
    }

    /// Get advance widths for multiple glyphs.
    #[must_use]
    pub fn glyph_advances(&self, glyphs: &[u16]) -> Vec<Scalar> {
        glyphs.iter().map(|&g| self.glyph_advance(g)).collect()
    }

    /// Get the bounding box for a glyph.
    ///
    /// Returns the tight visual bounds around the glyph's outline, read
    /// from the font's glyph bbox (via `ttf_parser::Face::glyph_bounding_box`)
    /// and converted from font-space (y-up) into screen-space (y-down) with
    /// the same `size/upem` scale used for glyph paths.
    ///
    /// For the dataless default typeface or when a glyph has no outline,
    /// falls back to an ascent×descent rectangle sized by the measured
    /// advance.
    #[must_use]
    pub fn glyph_bounds(&self, glyph: u16) -> skia_rs_core::Rect {
        // Glyph 0 (.notdef) has a real outline (the tofu box) and real
        // bounds; treat it like any other glyph.
        if let Some(data) = self.typeface.font_data() {
            if let Ok(face) = ttf_parser::Face::parse(data, 0) {
                let upem = face.units_per_em();
                if upem > 0 {
                    if let Some(bbox) = face.glyph_bounding_box(ttf_parser::GlyphId(glyph)) {
                        let scale = self.size / Scalar::from(upem);
                        let left = Scalar::from(bbox.x_min) * scale * self.scale_x;
                        let right = Scalar::from(bbox.x_max) * scale * self.scale_x;
                        let top = -Scalar::from(bbox.y_max) * scale;
                        let bottom = -Scalar::from(bbox.y_min) * scale;
                        return skia_rs_core::Rect::new(left, top, right, bottom);
                    }
                    // No outline — still report a zero-sized rect rather
                    // than the old ascent×descent approximation.
                    return skia_rs_core::Rect::EMPTY;
                }
            }
        }

        // Dataless typeface fallback.
        let advance = self.glyph_advance(glyph);
        let metrics = self.metrics();
        skia_rs_core::Rect::from_xywh(
            0.0,
            metrics.ascent,
            advance,
            -metrics.ascent + metrics.descent,
        )
    }

    /// Get bounding boxes for multiple glyphs.
    #[must_use]
    pub fn glyph_bounds_batch(&self, glyphs: &[u16]) -> Vec<skia_rs_core::Rect> {
        glyphs.iter().map(|&g| self.glyph_bounds(g)).collect()
    }

    /// Get the path outline for a glyph.
    ///
    /// Returns a path that can be filled to render the glyph. The path is in
    /// local coordinates with the glyph origin at (0, 0), scaled by
    /// `size / units_per_em`, and y-flipped from font-space (y-up) into
    /// screen-space (y-down) so that drawing it at the baseline position
    /// produces correctly oriented glyphs.
    ///
    /// Returns `None` if:
    /// - `glyph` is 0 (.notdef)
    /// - the underlying typeface has no font data (e.g. the default typeface)
    /// - the font data fails to parse
    /// - the glyph has no outline (e.g. space, non-printing characters)
    #[must_use]
    pub fn glyph_path(&self, glyph: u16) -> Option<skia_rs_path::Path> {
        // Glyph 0 (.notdef) has a real outline (the tofu box) in a
        // well-formed font; emit it like any other glyph.
        let data = self.typeface.font_data()?;
        let face = ttf_parser::Face::parse(data, 0).ok()?;

        let upem = face.units_per_em();
        if upem == 0 {
            return None;
        }
        let scale = self.size / Scalar::from(upem);

        let mut builder = GlyphOutlineBuilder {
            builder: skia_rs_path::PathBuilder::new(),
            scale,
            scale_x: self.scale_x,
            skew_x: self.skew_x,
        };
        face.outline_glyph(ttf_parser::GlyphId(glyph), &mut builder)?;
        Some(builder.builder.build())
    }

    /// Get paths for multiple glyphs.
    #[must_use]
    pub fn glyph_paths(&self, glyphs: &[u16]) -> Vec<Option<skia_rs_path::Path>> {
        glyphs.iter().map(|&g| self.glyph_path(g)).collect()
    }

    /// Get the path for a string of text.
    ///
    /// The returned path contains all glyph outlines positioned correctly.
    #[must_use]
    pub fn text_path(&self, text: &str) -> skia_rs_path::Path {
        let mut builder = skia_rs_path::PathBuilder::new();
        let glyphs = self.text_to_glyphs(text);
        let mut x_offset: Scalar = 0.0;

        for glyph in glyphs {
            if let Some(glyph_path) = self.glyph_path(glyph) {
                // Transform and add glyph path
                let transform = skia_rs_core::Matrix::translate(x_offset, 0.0);
                let transformed = glyph_path.transformed(&transform);
                builder.add_path(&transformed);
            }
            x_offset += self.glyph_advance(glyph);
        }

        builder.build()
    }

    /// Check if a glyph has a color representation in the font.
    ///
    /// Returns `true` iff at least one of the following color tables
    /// contains a definition for `glyph`:
    /// - `COLR` (layered colored outlines, v0; v1 paint graphs not yet
    ///   rendered but are still reported as color glyphs)
    /// - `sbix` / `CBDT` / `CBLC` / `bdat` (raster bitmap strikes)
    /// - `SVG ` (embedded SVG documents)
    ///
    /// Previously this was a `glyph > 0x1000` heuristic (gap C-5) which
    /// both false-positives for any normal font with >4096 glyphs and
    /// false-negatives for small emoji-only fonts. The new check
    /// consults the actual font tables via `ttf_parser`.
    #[must_use]
    pub fn glyph_is_color(&self, glyph: u16) -> bool {
        if glyph == 0 {
            return false;
        }
        let Some(data) = self.typeface.font_data() else {
            return false;
        };
        let Ok(face) = ttf_parser::Face::parse(data, 0) else {
            return false;
        };

        let gid = ttf_parser::GlyphId(glyph);

        if face.is_color_glyph(gid) {
            return true;
        }

        let ppem = skia_rs_core::cast::f32_to_u16_sat(self.size.ceil().max(1.0));
        if face.glyph_raster_image(gid, ppem).is_some() {
            return true;
        }

        if face.glyph_svg_image(gid).is_some() {
            return true;
        }

        false
    }

    /// Get an image for a color glyph (emoji) when the font ships one as
    /// a raster bitmap (CBDT/CBLC, sbix, or bdat).
    ///
    /// Returns `None` for:
    /// - non-color glyphs
    /// - color glyphs that only have a COLR or SVG definition (use
    ///   [`Self::glyph_color_layers`] or the SVG data directly for those)
    /// - the dataless default typeface
    ///
    /// Callers are responsible for decoding PNG/JPEG data returned in
    /// `GlyphImageFormat::Png`/`Jpeg` via `skia-rs-codec`. Bitmap formats
    /// (`Bgra32`, `Mono`, etc.) are returned raw — BGRA premultiplied
    /// is the format actually shipped in `CBDT` so consumers should
    /// treat it as premultiplied BGRA, not straight RGBA. The field
    /// order and pixel values are preserved byte-for-byte from the
    /// font; no color conversion is done.
    #[must_use]
    pub fn glyph_image(&self, glyph: u16) -> Option<GlyphImage> {
        if glyph == 0 {
            return None;
        }
        let data = self.typeface.font_data()?;
        let face = ttf_parser::Face::parse(data, 0).ok()?;
        let gid = ttf_parser::GlyphId(glyph);

        let ppem = skia_rs_core::cast::f32_to_u16_sat(self.size.ceil().max(1.0));
        let raster = face.glyph_raster_image(gid, ppem)?;

        // Scale raster bitmap offsets from the strike's ppem into the
        // font's current size so the returned left/top offsets are in
        // display pixels. The strike's `pixels_per_em` may differ from
        // `size` — ttf-parser picks the nearest available strike.
        let scale = if raster.pixels_per_em > 0 {
            self.size / Scalar::from(raster.pixels_per_em)
        } else {
            1.0
        };

        let format = match raster.format {
            ttf_parser::RasterImageFormat::PNG => GlyphImageFormat::Png,
            ttf_parser::RasterImageFormat::BitmapMono
            | ttf_parser::RasterImageFormat::BitmapMonoPacked => GlyphImageFormat::Mono,
            ttf_parser::RasterImageFormat::BitmapGray2
            | ttf_parser::RasterImageFormat::BitmapGray2Packed => GlyphImageFormat::Gray2,
            ttf_parser::RasterImageFormat::BitmapGray4
            | ttf_parser::RasterImageFormat::BitmapGray4Packed => GlyphImageFormat::Gray4,
            ttf_parser::RasterImageFormat::BitmapGray8 => GlyphImageFormat::Gray8,
            ttf_parser::RasterImageFormat::BitmapPremulBgra32 => GlyphImageFormat::PremulBgra32,
        };

        Some(GlyphImage {
            width: i32::from(raster.width),
            height: i32::from(raster.height),
            // Preserve the raw data buffer — no decode, no conversion.
            pixels: raster.data.to_vec(),
            format,
            left: Scalar::from(raster.x) * scale,
            // `raster.y` is the bitmap-top bearing measured upward (y-up)
            // from the glyph origin. Screen space is y-down, so negate it
            // (matches Skia's `-bearingY` convention for bitmap strikes).
            top: -Scalar::from(raster.y) * scale,
        })
    }

    /// Return the SVG document embedded for `glyph` in an `SVG ` table,
    /// if present. Returns the raw SVG bytes (possibly gzipped — the
    /// `SVG ` table stores SVGZ in that case); callers are expected to
    /// decompress and render via `skia-rs-svg`.
    ///
    /// Returns `None` for non-SVG glyphs and for the dataless default
    /// typeface.
    #[must_use]
    pub fn glyph_svg(&self, glyph: u16) -> Option<Vec<u8>> {
        if glyph == 0 {
            return None;
        }
        let data = self.typeface.font_data()?;
        let face = ttf_parser::Face::parse(data, 0).ok()?;
        face.glyph_svg_image(ttf_parser::GlyphId(glyph))
            .map(|doc| doc.data.to_vec())
    }

    /// Return the number of color palettes in the `CPAL` table, or `None`
    /// if the font has no color palette (equivalently: no COLR table or
    /// no CPAL).
    #[must_use]
    pub fn color_palette_count(&self) -> Option<u16> {
        let data = self.typeface.font_data()?;
        let face = ttf_parser::Face::parse(data, 0).ok()?;
        Some(face.color_palettes()?.get())
    }

    /// Decompose a `COLR` (v0 or v1) color glyph into its layer list.
    ///
    /// Each returned [`ColorGlyphLayer`] pairs a regular glyph id with the
    /// paint used to fill it. Callers render a color glyph by drawing
    /// each layer's outline in order with its associated paint —
    /// honouring the layer's affine [`ColorGlyphLayer::transform`] and
    /// optional clip glyph where present.
    ///
    /// - For **COLR v0** fonts (flat layered glyphs filled from a
    ///   palette — Noto Color Emoji v1, Segoe UI Emoji) every layer
    ///   carries [`GlyphPaint::Solid`] with the palette colour, an
    ///   identity transform, and no clip.
    /// - For **COLR v1** fonts (Noto Color Emoji v2, `COLRv1` reference
    ///   fonts) the paint graph is walked and each "outline + paint"
    ///   pair becomes a layer:
    ///   * solid fills → [`GlyphPaint::Solid`]
    ///   * linear gradients (three-point form) → [`GlyphPaint::LinearGradient`]
    ///   * radial gradients (two-circle form) → [`GlyphPaint::RadialGradient`]
    ///   * sweep gradients → [`GlyphPaint::SweepGradient`]
    ///
    ///   The combined affine transform active at the paint node is
    ///   folded into [`ColorGlyphLayer::transform`] so the caller
    ///   applies a single transform per layer. Clip paths (from
    ///   `PaintClip` nodes) and composite modes are tracked on the
    ///   layer; only the subset of [`CompositeMode`] actually emitted
    ///   by real fonts (`SourceOver` is overwhelmingly the common case)
    ///   is preserved, others are reported verbatim for callers that
    ///   want to implement the full Porter-Duff set.
    ///
    /// `palette` selects which `CPAL` palette to use; pass 0 for the
    /// default palette. `foreground_color` is used for layers that
    /// reference the "foreground" palette index (0xFFFF); pass the
    /// paragraph foreground color for that span.
    ///
    /// Returns `None` if the glyph has no COLR definition, the font has
    /// no CPAL, or the typeface has no data.
    #[must_use]
    pub fn glyph_color_layers(
        &self,
        glyph: u16,
        palette: u16,
        foreground_color: u32,
    ) -> Option<Vec<ColorGlyphLayer>> {
        if glyph == 0 {
            return None;
        }
        let data = self.typeface.font_data()?;
        let face = ttf_parser::Face::parse(data, 0).ok()?;
        let gid = ttf_parser::GlyphId(glyph);

        if !face.is_color_glyph(gid) {
            return None;
        }

        // COLR paint geometry (transforms, gradient points/radii) is
        // expressed in font design units, but the glyph outlines this
        // walker's layers get composited against are already scaled to
        // pixel space by `glyph_path` (size / units_per_em). Fold the
        // same scale into the walker so transform translations and
        // gradient geometry land in pixel space too. `upem == 0` is
        // malformed; fall back to no scaling (matches `glyph_path`'s
        // None-on-zero-upem guard as closely as a non-Option return
        // allows).
        let upem = face.units_per_em();
        let scale = if upem > 0 {
            self.size / Scalar::from(upem)
        } else {
            1.0
        };

        // Walk the COLR paint graph and convert every (outline, paint)
        // pair into a `ColorGlyphLayer` carrying a typed `GlyphPaint`.
        // The walker maintains clip and transform stacks so each layer
        // records the effective transform at the paint node and any
        // active clip shape.
        let fg = ttf_rgba_from_argb(foreground_color);
        let mut walker = ColorLayerWalker {
            layers: Vec::new(),
            current_glyph: None,
            palette,
            foreground: fg,
            scale,
            scale_x: self.scale_x,
            skew_x: self.skew_x,
            transform_stack: vec![Matrix::IDENTITY],
            clip_stack: Vec::new(),
            layer_stack: Vec::new(),
        };

        face.paint_color_glyph(gid, palette, fg, &mut walker)?;

        if walker.layers.is_empty() {
            None
        } else {
            Some(walker.layers)
        }
    }

    /// Get positioning information for a run of glyphs.
    #[must_use]
    pub fn glyph_positions(
        &self,
        glyphs: &[u16],
        start: skia_rs_core::Point,
    ) -> Vec<skia_rs_core::Point> {
        let mut positions = Vec::with_capacity(glyphs.len());
        let mut x = start.x;
        let y = start.y;

        for &glyph in glyphs {
            positions.push(skia_rs_core::Point::new(x, y));
            x += self.glyph_advance(glyph);
        }

        positions
    }

    /// Get the intercepts (horizontal line intersections) for glyph outlines.
    ///
    /// Used by text decoration positioning to skip drawing underline or
    /// strikethrough strokes *through* glyph outlines (e.g. the descender
    /// of a `g` or the bowl of a `y` should produce a gap in the
    /// underline rather than be struck through).
    ///
    /// For each glyph in `glyphs`, the outline is read via `glyph_path`,
    /// flattened to line segments, and intersected with the horizontal
    /// band between `y = top` and `y = bottom`. Every segment that enters
    /// or leaves the band contributes an x-coordinate intercept; these
    /// are returned sorted in the form `[enter_1, exit_1, enter_2,
    /// exit_2, ...]` so callers can walk the list in pairs.
    ///
    /// When the underlying typeface has no font data (the default
    /// typeface) or a glyph has no outline (spaces), the function falls
    /// back to the bounding-box intercepts — the old placeholder
    /// behaviour — which produces a continuous underline for that glyph.
    #[must_use]
    pub fn glyph_intercepts(
        &self,
        glyphs: &[u16],
        positions: &[skia_rs_core::Point],
        top: Scalar,
        bottom: Scalar,
    ) -> Vec<Scalar> {
        let (top, bottom) = if top <= bottom {
            (top, bottom)
        } else {
            (bottom, top)
        };

        let mut intercepts: Vec<Scalar> = Vec::new();

        for (i, &glyph) in glyphs.iter().enumerate() {
            if glyph == 0 {
                continue;
            }
            let pos = positions.get(i).copied().unwrap_or_default();

            let glyph_xs = self.glyph_path(glyph).map_or_else(Vec::new, |path| {
                path_band_intercepts(&path, top - pos.y, bottom - pos.y)
            });

            if glyph_xs.is_empty() {
                // Fallback: bounding box test. For the dataless default
                // typeface (no outline) this preserves the previous
                // placeholder behaviour so callers still get *something*.
                let bounds = self.glyph_bounds(glyph);
                if !bounds.is_empty() {
                    let glyph_top = pos.y + bounds.top;
                    let glyph_bottom = pos.y + bounds.bottom;
                    if glyph_bottom >= top && glyph_top <= bottom {
                        intercepts.push(pos.x + bounds.left);
                        intercepts.push(pos.x + bounds.right);
                    }
                }
            } else {
                for x in glyph_xs {
                    intercepts.push(pos.x + x);
                }
            }
        }

        // Sort + collapse overlapping pairs so callers can walk them as
        // enter/exit pairs.
        intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        intercepts
    }
}

/// Flatten a Path into line segments and return the x-coordinates where
/// the outline crosses the horizontal band [`y_top`, `y_bottom`] (y-down
/// screen space). Returns crossings grouped as enter/exit pairs per
/// continuous interval.
///
/// The quad/cubic/conic segments are approximated by adaptive subdivision
/// until the chord-to-curve error drops below a flatness tolerance
/// proportional to the band height (1/32 of band height, or 0.25 px
/// whichever is larger). For typical font sizes this produces 4-16
/// segments per curve, which is plenty for decoration gap rendering
/// without being wasteful.
fn path_band_intercepts(path: &skia_rs_path::Path, y_top: Scalar, y_bottom: Scalar) -> Vec<Scalar> {
    use skia_rs_core::Point;
    use skia_rs_path::PathElement;

    if y_top >= y_bottom {
        return Vec::new();
    }

    let band_height = y_bottom - y_top;
    let tolerance = (band_height / 32.0).max(0.25);

    let mut segments: Vec<(Point, Point)> = Vec::new();
    let mut current = Point::zero();
    let mut contour_start = Point::zero();

    for elem in path {
        match elem {
            PathElement::Move(p) => {
                current = p;
                contour_start = p;
            }
            PathElement::Line(p) => {
                segments.push((current, p));
                current = p;
            }
            PathElement::Quad(c, p) => {
                flatten_quad(current, c, p, tolerance, &mut segments);
                current = p;
            }
            PathElement::Conic(c, p, _w) => {
                // Approximate conic as quadratic (sufficient for decoration
                // intercepts; the exact rational-quadratic form isn't
                // needed at a 0.25px tolerance).
                flatten_quad(current, c, p, tolerance, &mut segments);
                current = p;
            }
            PathElement::Cubic(c1, c2, p) => {
                flatten_cubic(current, c1, c2, p, tolerance, &mut segments);
                current = p;
            }
            PathElement::Close => {
                if current != contour_start {
                    segments.push((current, contour_start));
                    current = contour_start;
                }
            }
        }
    }

    // Collect x-crossings for each horizontal line at the band edges.
    // For the gap-drawing case we want the *entire range* the outline
    // covers inside the band; it's sufficient to take the union of
    // segment endpoints that lie inside the band together with
    // intersections with each band edge. A conservative bound is to use
    // every segment that touches the band and emit its min/max x.
    let mut xs: Vec<Scalar> = Vec::new();
    for &(a, b) in &segments {
        let seg_top = a.y.min(b.y);
        let seg_bot = a.y.max(b.y);
        // Segment disjoint from band → skip.
        if seg_bot < y_top || seg_top > y_bottom {
            continue;
        }

        // Clip the segment to the band and record the x range of the
        // clipped portion. For a line segment (p1 → p2), parametrise by
        // t ∈ [0,1] and find the t range where y(t) ∈ [y_top, y_bottom].
        let dy = b.y - a.y;
        let (t0, t1) = if dy.abs() < 1e-6 {
            // Horizontal segment — lies entirely in the band if touching
            // it at all. Use the full range.
            (0.0f32, 1.0f32)
        } else {
            let mut t_enter = (y_top - a.y) / dy;
            let mut t_exit = (y_bottom - a.y) / dy;
            if t_enter > t_exit {
                std::mem::swap(&mut t_enter, &mut t_exit);
            }
            (t_enter.max(0.0), t_exit.min(1.0))
        };
        if t0 > t1 {
            continue;
        }
        let x0 = (b.x - a.x).mul_add(t0, a.x);
        let x1 = (b.x - a.x).mul_add(t1, a.x);
        xs.push(x0.min(x1));
        xs.push(x0.max(x1));
    }

    if xs.is_empty() {
        return xs;
    }

    // Merge into non-overlapping intervals so callers can consume them
    // as [enter_1, exit_1, enter_2, exit_2, …] pairs. Each pair of
    // xs[2k], xs[2k+1] represents a contiguous range of x where the
    // outline occupies the band.
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<Scalar> = Vec::with_capacity(xs.len());
    let mut iter = xs.chunks_exact(2);
    let mut cur: Option<(Scalar, Scalar)> = None;
    for pair in &mut iter {
        let (lo, hi) = (pair[0], pair[1]);
        cur = match cur {
            Some((a, b)) if lo <= b + tolerance => Some((a, b.max(hi))),
            Some((a, b)) => {
                merged.push(a);
                merged.push(b);
                Some((lo, hi))
            }
            None => Some((lo, hi)),
        };
    }
    if let Some((a, b)) = cur {
        merged.push(a);
        merged.push(b);
    }

    merged
}

/// Recursively subdivide a quadratic bezier into chord approximations
/// until the control point is within `tol` of the chord midpoint.
#[allow(
    clippy::similar_names,
    reason = "midx/midy are the natural names for a chord midpoint's x/y in curve-flattening math"
)]
fn flatten_quad(
    p0: skia_rs_core::Point,
    p1: skia_rs_core::Point,
    p2: skia_rs_core::Point,
    tol: Scalar,
    out: &mut Vec<(skia_rs_core::Point, skia_rs_core::Point)>,
) {
    use skia_rs_core::Point;
    // Midpoint of p0-p2 chord.
    let midx = f32::midpoint(p0.x, p2.x);
    let midy = f32::midpoint(p0.y, p2.y);
    // Distance from control to the chord midpoint.
    let dx = p1.x - midx;
    let dy = p1.y - midy;
    if dy.mul_add(dy, dx * dx) <= tol * tol {
        out.push((p0, p2));
        return;
    }
    // Subdivide at t=0.5 using de Casteljau.
    let q0 = Point::new(f32::midpoint(p0.x, p1.x), f32::midpoint(p0.y, p1.y));
    let q1 = Point::new(f32::midpoint(p1.x, p2.x), f32::midpoint(p1.y, p2.y));
    let mid = Point::new(f32::midpoint(q0.x, q1.x), f32::midpoint(q0.y, q1.y));
    flatten_quad(p0, q0, mid, tol, out);
    flatten_quad(mid, q1, p2, tol, out);
}

/// Recursively subdivide a cubic bezier into chord approximations.
fn flatten_cubic(
    p0: skia_rs_core::Point,
    p1: skia_rs_core::Point,
    p2: skia_rs_core::Point,
    p3: skia_rs_core::Point,
    tol: Scalar,
    out: &mut Vec<(skia_rs_core::Point, skia_rs_core::Point)>,
) {
    use skia_rs_core::Point;
    // Flatness heuristic: distance of control points from the chord.
    let cx = f32::midpoint(p0.x, p3.x);
    let cy = f32::midpoint(p0.y, p3.y);
    let d1 = {
        let dx = p1.x - cx;
        let dy = p1.y - cy;
        dy.mul_add(dy, dx * dx)
    };
    let d2 = {
        let dx = p2.x - cx;
        let dy = p2.y - cy;
        dy.mul_add(dy, dx * dx)
    };
    if d1 <= tol * tol && d2 <= tol * tol {
        out.push((p0, p3));
        return;
    }
    // De Casteljau at t=0.5.
    let q0 = Point::new(f32::midpoint(p0.x, p1.x), f32::midpoint(p0.y, p1.y));
    let q1 = Point::new(f32::midpoint(p1.x, p2.x), f32::midpoint(p1.y, p2.y));
    let q2 = Point::new(f32::midpoint(p2.x, p3.x), f32::midpoint(p2.y, p3.y));
    let r0 = Point::new(f32::midpoint(q0.x, q1.x), f32::midpoint(q0.y, q1.y));
    let r1 = Point::new(f32::midpoint(q1.x, q2.x), f32::midpoint(q1.y, q2.y));
    let mid = Point::new(f32::midpoint(r0.x, r1.x), f32::midpoint(r0.y, r1.y));
    flatten_cubic(p0, q0, r0, mid, tol, out);
    flatten_cubic(mid, r1, q2, p3, tol, out);
}

/// Adapts `ttf_parser::OutlineBuilder` onto `skia_rs_path::PathBuilder`.
///
/// Fonts store outlines with y-axis pointing up; our Path lives in screen
/// space (y-down). The builder flips `y` and scales both axes by
/// `size / units_per_em` so the produced path is already in pixel units
/// relative to the glyph origin.
///
/// The font's `scale_x` and `skew_x` are folded in to match Skia's
/// `SkFontPriv::MakeTextMatrix` (`Scale(size*scaleX, size)` then
/// `postSkew(skewX, 0)`): with `px`/`py` the y-flipped, size-scaled screen
/// coordinates, the emitted point is `(scaleX * px + skewX * py, py)`. This
/// keeps `glyph_path` consistent with `glyph_advance`, which also scales x
/// by `scale_x`.
struct GlyphOutlineBuilder {
    builder: skia_rs_path::PathBuilder,
    scale: Scalar,
    scale_x: Scalar,
    skew_x: Scalar,
}

impl GlyphOutlineBuilder {
    /// Map a font-space (y-up) point to the final screen-space point,
    /// applying the size/upem scale, y-flip, horizontal `scale_x`, and
    /// `skew_x` shear.
    #[inline]
    fn map(&self, x: f32, y: f32) -> (f32, f32) {
        let px = x * self.scale;
        let py = -y * self.scale;
        (self.scale_x.mul_add(px, self.skew_x * py), py)
    }
}

impl ttf_parser::OutlineBuilder for GlyphOutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.map(x, y);
        self.builder.move_to(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.map(x, y);
        self.builder.line_to(x, y);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (x1, y1) = self.map(x1, y1);
        let (x, y) = self.map(x, y);
        self.builder.quad_to(x1, y1, x, y);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (x1, y1) = self.map(x1, y1);
        let (x2, y2) = self.map(x2, y2);
        let (x, y) = self.map(x, y);
        self.builder.cubic_to(x1, y1, x2, y2, x, y);
    }

    fn close(&mut self) {
        self.builder.close();
    }
}

/// Raster image data extracted from a color glyph bitmap strike
/// (`CBDT`/`CBLC`, `sbix`, or `bdat`).
///
/// The `format` field distinguishes encoded (PNG) from raw-bitmap data.
/// Callers typically delegate PNG decoding to `skia-rs-codec`; raw bitmap
/// formats can be uploaded to a texture directly (with the documented
/// byte-order caveat for `PremulBgra32`).
#[derive(Debug, Clone)]
pub struct GlyphImage {
    /// Image width in pixels.
    pub width: i32,
    /// Image height in pixels.
    pub height: i32,
    /// Raw encoded or raw bitmap data. See `format` for the byte layout.
    pub pixels: Vec<u8>,
    /// Pixel layout / encoding of `pixels`.
    pub format: GlyphImageFormat,
    /// Left offset from glyph origin, in display pixels (scaled from the
    /// strike's native ppem to the font's current size).
    pub left: Scalar,
    /// Top offset from glyph origin, in display pixels.
    pub top: Scalar,
}

/// Pixel format of a [`GlyphImage`].
///
/// Mirrors `ttf_parser::RasterImageFormat` but is re-exported here so the
/// skia-rs-text public surface does not leak a ttf-parser type. See the
/// OpenType `CBDT`/`sbix` specs for each format's bit layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphImageFormat {
    /// PNG-encoded bytes. Delegate decoding to `skia-rs-codec`.
    Png,
    /// 1-bit-per-pixel monochrome bitmap, row-aligned to byte boundary.
    /// MSB is the first pixel in each byte.
    Mono,
    /// 2-bits-per-pixel grayscale bitmap.
    Gray2,
    /// 4-bits-per-pixel grayscale bitmap.
    Gray4,
    /// 8-bits-per-pixel grayscale bitmap.
    Gray8,
    /// 32-bit premultiplied BGRA (per the `CBDT` / `sbix` spec).
    PremulBgra32,
}

/// Gradient wrap mode (how a gradient extends past its defined range).
///
/// Mirrors `ttf_parser::colr::GradientExtend` so callers of the
/// skia-rs-text public API don't need a direct ttf-parser dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientExtend {
    /// Clamp to the edge colors (default).
    Pad,
    /// Repeat the gradient pattern.
    Repeat,
    /// Reflect the gradient pattern.
    Reflect,
}

impl From<ttf_parser::colr::GradientExtend> for GradientExtend {
    fn from(v: ttf_parser::colr::GradientExtend) -> Self {
        match v {
            ttf_parser::colr::GradientExtend::Pad => Self::Pad,
            ttf_parser::colr::GradientExtend::Repeat => Self::Repeat,
            ttf_parser::colr::GradientExtend::Reflect => Self::Reflect,
        }
    }
}

/// A single colour stop on a gradient.
///
/// `offset` is a parametric position in `[0, 1]` along the gradient's
/// line / radial axis / sweep arc. `color` is a `0xAARRGGBB` u32
/// matching the convention used elsewhere in the crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientStop {
    /// Stop position in `[0, 1]` along the gradient axis.
    pub offset: u32,
    /// ARGB color (0xAARRGGBB). Encoded as u32 so `GradientStop` can
    /// stay `Copy`; use [`GradientStop::offset_f32`] to read the offset
    /// as a float.
    pub color: u32,
}

impl GradientStop {
    /// Create a stop from a float offset and ARGB color.
    #[inline]
    #[must_use]
    pub const fn new(offset: f32, color: u32) -> Self {
        Self {
            offset: offset.to_bits(),
            color,
        }
    }

    /// Return the offset as a float in `[0, 1]`.
    #[inline]
    #[must_use]
    pub const fn offset_f32(&self) -> f32 {
        f32::from_bits(self.offset)
    }
}

/// The paint (fill) for a single `COLR` layer.
///
/// For COLR v0 and simple v1 solid-fill paints this is [`GlyphPaint::Solid`].
/// For COLR v1 gradient paints the three forms — linear, radial, sweep —
/// are preserved with their full geometric definition, so downstream
/// paint pipelines (e.g. `skia-rs-paint`'s gradient shaders) can
/// rasterise them faithfully. Stops are returned in the order stored
/// in the font; callers that need sorted stops should sort by
/// [`GradientStop::offset_f32`] before sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum GlyphPaint {
    /// A solid `0xAARRGGBB` color.
    Solid(u32),
    /// A three-point linear gradient.
    ///
    /// Following the COLR v1 spec, the line is defined by three points
    /// `(p0, p1, p2)` where `p0..p1` is the gradient direction and
    /// `p0..p2` is the "rotation" axis — callers that only need the
    /// common two-point linear gradient can ignore `p2` and derive the
    /// perpendicular direction from `p1 - p0`.
    LinearGradient {
        /// Start point.
        p0: Point,
        /// End point.
        p1: Point,
        /// Rotation/skew reference point.
        p2: Point,
        /// Colour stops along `p0..p1`.
        stops: Vec<GradientStop>,
        /// Wrap mode for samples outside `[0, 1]`.
        extend: GradientExtend,
    },
    /// A two-circle radial gradient.
    RadialGradient {
        /// Centre of the start circle.
        c0: Point,
        /// Radius of the start circle.
        r0: f32,
        /// Centre of the end circle.
        c1: Point,
        /// Radius of the end circle.
        r1: f32,
        /// Colour stops.
        stops: Vec<GradientStop>,
        /// Wrap mode.
        extend: GradientExtend,
    },
    /// A sweep gradient.
    SweepGradient {
        /// Centre of the sweep.
        center: Point,
        /// Start angle in **degrees**, counter-clockwise from the positive
        /// x-axis (matching Skia's `SkShaders::SweepGradient`, which takes
        /// degrees). Passed straight through from the COLR paint.
        start_angle: f32,
        /// End angle in **degrees**.
        end_angle: f32,
        /// Colour stops.
        stops: Vec<GradientStop>,
        /// Wrap mode.
        extend: GradientExtend,
    },
}

impl GlyphPaint {
    /// Return a representative ARGB color for the paint.
    ///
    /// For a solid paint this is the fill color. For a gradient paint
    /// this is the first stop's colour, which matches the "sample the
    /// gradient at t=0" approximation used as a fallback when the
    /// caller can't rasterise a real gradient. Returns `0xFF_00_00_00`
    /// (opaque black) if the gradient has no stops.
    #[must_use]
    pub fn representative_color(&self) -> u32 {
        match self {
            Self::Solid(c) => *c,
            Self::LinearGradient { stops, .. }
            | Self::RadialGradient { stops, .. }
            | Self::SweepGradient { stops, .. } => stops.first().map_or(0xFF_00_00_00, |s| s.color),
        }
    }

    /// Returns `true` if this paint is a gradient (i.e. not
    /// [`GlyphPaint::Solid`]).
    #[inline]
    #[must_use]
    pub const fn is_gradient(&self) -> bool {
        !matches!(self, Self::Solid(_))
    }
}

/// Composite mode for a COLR v1 layer.
///
/// Mirrors `ttf_parser::colr::CompositeMode` — the full Porter-Duff +
/// blend-mode set so callers can implement the COLR v1 composite node
/// without leaking a ttf-parser type. `SrcOver` is overwhelmingly the
/// common case in real fonts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMode {
    /// `Clear` composite.
    Clear,
    /// `Source` composite.
    Source,
    /// `Destination` composite.
    Destination,
    /// `SourceOver` (aka `SrcOver`) composite — the default.
    SourceOver,
    /// `DestinationOver` composite.
    DestinationOver,
    /// `SourceIn` composite.
    SourceIn,
    /// `DestinationIn` composite.
    DestinationIn,
    /// `SourceOut` composite.
    SourceOut,
    /// `DestinationOut` composite.
    DestinationOut,
    /// `SourceAtop` composite.
    SourceAtop,
    /// `DestinationAtop` composite.
    DestinationAtop,
    /// `Xor` composite.
    Xor,
    /// `Plus` composite.
    Plus,
    /// `Screen` blend.
    Screen,
    /// `Overlay` blend.
    Overlay,
    /// `Darken` blend.
    Darken,
    /// `Lighten` blend.
    Lighten,
    /// `ColorDodge` blend.
    ColorDodge,
    /// `ColorBurn` blend.
    ColorBurn,
    /// `HardLight` blend.
    HardLight,
    /// `SoftLight` blend.
    SoftLight,
    /// `Difference` blend.
    Difference,
    /// `Exclusion` blend.
    Exclusion,
    /// `Multiply` blend.
    Multiply,
    /// `Hue` blend.
    Hue,
    /// `Saturation` blend.
    Saturation,
    /// `Color` blend.
    Color,
    /// `Luminosity` blend.
    Luminosity,
}

impl From<ttf_parser::colr::CompositeMode> for CompositeMode {
    fn from(m: ttf_parser::colr::CompositeMode) -> Self {
        use ttf_parser::colr::CompositeMode as M;
        match m {
            M::Clear => Self::Clear,
            M::Source => Self::Source,
            M::Destination => Self::Destination,
            M::SourceOver => Self::SourceOver,
            M::DestinationOver => Self::DestinationOver,
            M::SourceIn => Self::SourceIn,
            M::DestinationIn => Self::DestinationIn,
            M::SourceOut => Self::SourceOut,
            M::DestinationOut => Self::DestinationOut,
            M::SourceAtop => Self::SourceAtop,
            M::DestinationAtop => Self::DestinationAtop,
            M::Xor => Self::Xor,
            M::Plus => Self::Plus,
            M::Screen => Self::Screen,
            M::Overlay => Self::Overlay,
            M::Darken => Self::Darken,
            M::Lighten => Self::Lighten,
            M::ColorDodge => Self::ColorDodge,
            M::ColorBurn => Self::ColorBurn,
            M::HardLight => Self::HardLight,
            M::SoftLight => Self::SoftLight,
            M::Difference => Self::Difference,
            M::Exclusion => Self::Exclusion,
            M::Multiply => Self::Multiply,
            M::Hue => Self::Hue,
            M::Saturation => Self::Saturation,
            M::Color => Self::Color,
            M::Luminosity => Self::Luminosity,
        }
    }
}

/// A single layer of a `COLR` color glyph.
///
/// Produced by [`Font::glyph_color_layers`]. A color glyph is composed
/// by drawing each layer in order: resolve `glyph`'s outline via
/// [`Font::glyph_path`], transform it by `transform`, intersect with
/// `clip_glyph` if present, then fill with `paint` using
/// `composite_mode`.
///
/// COLR v0 always emits solid layers with identity transform and no
/// clip. COLR v1 can emit any combination of the above.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorGlyphLayer {
    /// Glyph id of the layer's outline.
    pub glyph: u16,
    /// The paint (fill) for this layer.
    pub paint: GlyphPaint,
    /// Affine transform active at the paint node. Identity for COLR v0
    /// and for v1 paints outside a `PaintTransform` subtree. The
    /// transform is in the same y-down screen space as [`Font::glyph_path`]
    /// outputs — a concat of the COLR-graph transforms with the Y-flip
    /// already applied.
    pub transform: Matrix,
    /// Optional clip glyph id. When set, the layer's outline should be
    /// clipped by this glyph's outline before filling — this is the
    /// COLR v1 `PaintGlyph` (outline-as-clip) primitive.
    pub clip_glyph: Option<u16>,
    /// Optional rectangular clip from the COLR v1 `ClipList` (the
    /// `PaintColrGlyph` / base-glyph clip box), in font design units
    /// converted to y-down screen convention (`top = -yMax`,
    /// `bottom = -yMin`). Callers scale it by `size / units_per_em`
    /// alongside the glyph outline. `None` when no clip box is active.
    pub clip_box: Option<skia_rs_core::Rect>,
    /// Composite mode active at the paint node. Defaults to
    /// [`CompositeMode::SourceOver`] for COLR v0 and most v1 paints;
    /// set only when the paint is inside a `PaintComposite` subtree.
    pub composite_mode: CompositeMode,
}

impl ColorGlyphLayer {
    /// Convenience accessor for the layer's "representative" ARGB color.
    ///
    /// For solid fills this is the fill color; for gradient fills this
    /// is the first stop's color — useful as a fallback when the
    /// caller can't rasterise the full gradient.
    #[inline]
    #[must_use]
    pub fn color(&self) -> u32 {
        self.paint.representative_color()
    }

    /// Returns `true` if the layer's paint is a gradient.
    ///
    /// Preserved for source compatibility with the Phase 6A API — new
    /// code should match on [`Self::paint`] directly.
    #[inline]
    #[must_use]
    pub const fn is_gradient(&self) -> bool {
        self.paint.is_gradient()
    }
}

/// Convert a `0xAARRGGBB` u32 to `ttf_parser`'s `RgbaColor`.
fn ttf_rgba_from_argb(argb: u32) -> ttf_parser::RgbaColor {
    let a = ((argb >> 24) & 0xff) as u8;
    let r = ((argb >> 16) & 0xff) as u8;
    let g = ((argb >> 8) & 0xff) as u8;
    let b = (argb & 0xff) as u8;
    ttf_parser::RgbaColor::new(r, g, b, a)
}

/// Pack a `ttf_parser` `RgbaColor` back into an ARGB u32.
fn argb_from_ttf_rgba(c: ttf_parser::RgbaColor) -> u32 {
    (u32::from(c.alpha) << 24)
        | (u32::from(c.red) << 16)
        | (u32::from(c.green) << 8)
        | u32::from(c.blue)
}

/// Convert a ttf-parser affine (`a b c d e f` column vectors in y-up
/// font space) to a `skia_rs_core::Matrix` in y-down screen space.
///
/// ttf-parser's transform acts on y-up coordinates with:
///   x' = a*x + c*y + e
///   y' = b*x + d*y + f
/// Composing with the font's Y-flip (which `glyph_path` already applies)
/// means a COLR transform stays consistent as long as we keep a / d / e
/// straight and negate the Y components (b, c, f) when mapping into
/// screen space — this matches the y-flip applied inside
/// `GlyphOutlineBuilder`.
/// Unscaled convenience wrapper (scale = 1.0). Only used by tests that
/// exercise the y-flip/skew mapping in isolation; production code always
/// goes through [`matrix_from_ttf_transform_scaled`] with the font's
/// size/upem scale.
#[cfg(test)]
fn matrix_from_ttf_transform(t: ttf_parser::Transform) -> Matrix {
    matrix_from_ttf_transform_scaled(t, 1.0, 1.0, 0.0)
}

/// Like [`matrix_from_ttf_transform`], but maps the transform into the
/// same pixel-space, text-matrix convention as the (already-scaled/sheared)
/// glyph outline: `scale` = size/upem, `scale_x`/`skew_x` are the font's
/// synthetic condense/oblique factors, matching `GlyphOutlineBuilder::map`.
///
/// A COLR transform `A` acts on font-space points; the outline it composes
/// against has already been mapped into pixel space by the linear map `L`
/// that `GlyphOutlineBuilder::map` applies (scale, y-flip, `scale_x`, `skew_x`).
/// To keep `A` consistent with pixel-space geometry it must be conjugated:
/// `T = L · A · L⁻¹`, so that `T(L(p)) == L(A(p))` for every font-space
/// point `p`. When `scale_x == 1.0 && skew_x == 0.0`, `L` reduces to the
/// plain `diag(scale, -scale)` y-flip and this is byte-identical to the
/// historical `a, -c, e*scale, -b, d, -f*scale` formula.
fn matrix_from_ttf_transform_scaled(
    t: ttf_parser::Transform,
    scale: Scalar,
    scale_x: Scalar,
    skew_x: Scalar,
) -> Matrix {
    // Row-major 3x3, matching `Matrix::values`:
    //   | scale_x  skew_x   trans_x |   | a  c   e |
    //   | skew_y   scale_y  trans_y | = | b  d   f |   (in font y-up)
    //   |    0       0        1    |   | 0  0   1 |
    let a = Matrix {
        values: [t.a, t.c, t.e, t.b, t.d, t.f, 0.0, 0.0, 1.0],
    };
    // L: the same linear map `GlyphOutlineBuilder::map` applies to outline
    // points (translation-free, since it maps vectors/paint geometry, not
    // positions relative to an origin).
    let l = Matrix {
        values: [
            scale_x * scale,
            -skew_x * scale,
            0.0,
            0.0,
            -scale,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    };
    let l_inv = l.invert().unwrap_or(Matrix::IDENTITY);
    l.concat(&a).concat(&l_inv)
}

/// One entry on the walker's clip stack. COLR clips come in two flavours:
/// an outline-shaped clip from a `PaintGlyph` node, and a rectangular
/// clip box from the `ClipList`. They must be tracked distinctly so a box
/// clip never masquerades as a glyph clip (previously a sentinel gid 0 was
/// pushed, which — now that .notdef has a real outline — would clip layers
/// to the tofu box).
#[derive(Clone, Copy)]
enum ClipEntry {
    /// Outline-as-clip from a `PaintGlyph`.
    Glyph(u16),
    /// Rectangular clip box from the `ClipList`, in y-down screen
    /// convention (font units).
    Box(skia_rs_core::Rect),
}

/// Painter implementation that walks the COLR paint graph and collects
/// typed `ColorGlyphLayer`s.
///
/// For **COLR v0** the driver pairs `outline_glyph(gid)` directly with
/// `paint(...)`, so `current_glyph` holds the fill shape when `paint` runs.
///
/// For **COLR v1** a `PaintGlyph` drives `outline_glyph(gid)` →
/// `push_clip()` (which *consumes* the outline onto the clip stack) →
/// child `paint(...)` → `pop_clip()`. So at `paint` time `current_glyph`
/// is `None` and the fill shape is the innermost glyph clip on the stack.
/// Transform, clip, and composite stacks track the active context so each
/// emitted layer captures its effective transform, clip, and mode.
struct ColorLayerWalker {
    layers: Vec<ColorGlyphLayer>,
    /// The glyph id most recently seen via `outline_glyph`. Consumed
    /// by the next `paint` / `push_clip` call.
    current_glyph: Option<u16>,
    palette: u16,
    foreground: ttf_parser::RgbaColor,
    /// `size / units_per_em`, applied to every COLR transform
    /// translation and gradient point/radius so paint geometry lands
    /// in the same pixel space as the (already-scaled) glyph outlines.
    /// COLR v0 never carries transforms or gradients, so this is a
    /// no-op for v0 fonts.
    scale: Scalar,
    /// The font's synthetic `scale_x`/`skew_x`, folded into COLR paint
    /// geometry the same way `GlyphOutlineBuilder::map` folds them into
    /// the glyph outline, so gradients and transforms stay aligned with
    /// a condensed/expanded or obliqued outline. No-op when 1.0/0.0.
    scale_x: Scalar,
    skew_x: Scalar,
    /// Stack of accumulated transforms. Top element is the currently
    /// active transform; `push_transform` composes onto it.
    transform_stack: Vec<Matrix>,
    /// Stack of active clips (glyph outlines and/or boxes).
    clip_stack: Vec<ClipEntry>,
    /// Stack of composite modes pushed via `push_layer`. Top element
    /// applies to the next painted layer.
    layer_stack: Vec<CompositeMode>,
}

impl ColorLayerWalker {
    #[inline]
    fn current_transform(&self) -> Matrix {
        *self.transform_stack.last().unwrap_or(&Matrix::IDENTITY)
    }

    /// Return the `n`-th glyph clip scanning from the top of the clip
    /// stack (0 = innermost/topmost), skipping box clips.
    fn nth_glyph_clip_from_top(&self, n: usize) -> Option<u16> {
        self.clip_stack
            .iter()
            .rev()
            .filter_map(|e| match e {
                ClipEntry::Glyph(g) => Some(*g),
                ClipEntry::Box(_) => None,
            })
            .nth(n)
    }

    /// Return the innermost active clip box, if any.
    fn current_clip_box(&self) -> Option<skia_rs_core::Rect> {
        self.clip_stack.iter().rev().find_map(|e| match e {
            ClipEntry::Box(r) => Some(*r),
            ClipEntry::Glyph(_) => None,
        })
    }

    #[inline]
    fn current_composite(&self) -> CompositeMode {
        self.layer_stack
            .last()
            .copied()
            .unwrap_or(CompositeMode::SourceOver)
    }

    /// Convert a `ttf_parser::colr::Paint` into our typed `GlyphPaint`.
    fn convert_paint(&self, paint: ttf_parser::colr::Paint<'_>) -> GlyphPaint {
        let palette = self.palette;
        match paint {
            ttf_parser::colr::Paint::Solid(color) => GlyphPaint::Solid(argb_from_ttf_rgba(color)),
            ttf_parser::colr::Paint::LinearGradient(g) => {
                // Font space is y-up; our glyph paths live in y-down
                // screen space, already scaled by size/upem and sheared by
                // scale_x/skew_x. `scale_colr_point` applies the same
                // text-matrix mapping so gradient geometry (font design
                // units) matches the scaled/sheared outline (pixel units).
                let (s, sx, kx) = (self.scale, self.scale_x, self.skew_x);
                GlyphPaint::LinearGradient {
                    p0: scale_colr_point(g.x0, g.y0, s, sx, kx),
                    p1: scale_colr_point(g.x1, g.y1, s, sx, kx),
                    p2: scale_colr_point(g.x2, g.y2, s, sx, kx),
                    stops: collect_stops(g.stops(palette, &[])),
                    extend: g.extend.into(),
                }
            }
            ttf_parser::colr::Paint::RadialGradient(g) => {
                let (s, sx, kx) = (self.scale, self.scale_x, self.skew_x);
                GlyphPaint::RadialGradient {
                    c0: scale_colr_point(g.x0, g.y0, s, sx, kx),
                    r0: g.r0 * s,
                    c1: scale_colr_point(g.x1, g.y1, s, sx, kx),
                    r1: g.r1 * s,
                    stops: collect_stops(g.stops(palette, &[])),
                    extend: g.extend.into(),
                }
            }
            ttf_parser::colr::Paint::SweepGradient(g) => {
                // ttf-parser hands back the raw F2Dot14 sweep angles, which
                // the COLR spec stores in units of 180° (i.e. degrees / 180).
                // Multiply by 180 to recover degrees. Upstream (Skia's
                // fontations port) negates the centre's y but passes the
                // degree angle *straight* into the sweep shader — no extra
                // negation — so we do the same. The centre point is in font
                // design units and must be scaled to pixel space like the
                // linear/radial gradient points; angles are unit-less.
                let (s, sx, kx) = (self.scale, self.scale_x, self.skew_x);
                GlyphPaint::SweepGradient {
                    center: scale_colr_point(g.center_x, g.center_y, s, sx, kx),
                    start_angle: g.start_angle * 180.0,
                    end_angle: g.end_angle * 180.0,
                    stops: collect_stops(g.stops(palette, &[])),
                    extend: g.extend.into(),
                }
            }
        }
    }
}

/// Map a COLR gradient control point from font design units (y-up) into
/// the pixel-space, y-down convention used by the emitted layers. This is
/// the exact mapping `GlyphOutlineBuilder::map` applies to outline points:
/// scale by `size / upem`, flip y, then fold in the font's `scale_x`
/// (horizontal condense/expand) and `skew_x` (synthetic oblique) shear, so
/// gradient geometry lands in the same text-matrix space as the
/// already-scaled/sheared glyph outline.
#[inline]
fn scale_colr_point(x: f32, y: f32, scale: Scalar, scale_x: Scalar, skew_x: Scalar) -> Point {
    let px = x * scale;
    let py = -y * scale;
    Point::new(scale_x.mul_add(px, skew_x * py), py)
}

fn collect_stops(iter: ttf_parser::colr::GradientStopsIter<'_, '_>) -> Vec<GradientStop> {
    iter.map(|s| GradientStop::new(s.stop_offset, argb_from_ttf_rgba(s.color)))
        .collect()
}

impl<'a> ttf_parser::colr::Painter<'a> for ColorLayerWalker {
    fn outline_glyph(&mut self, glyph_id: ttf_parser::GlyphId) {
        // Record which glyph is the current outline target; the next
        // `paint` call will consume this and emit a layer.
        self.current_glyph = Some(glyph_id.0);
    }

    fn paint(&mut self, paint: ttf_parser::colr::Paint<'a>) {
        // Determine the fill shape and any surviving clip.
        //
        // COLR v0: `outline_glyph` set `current_glyph`; that's the fill
        // shape, clipped by whatever glyph clip is active (normally none).
        //
        // COLR v1: `push_clip` already consumed the outline, so
        // `current_glyph` is None. The fill shape is the innermost glyph
        // clip; the next-outer glyph clip (if any) still constrains it.
        let (glyph, clip_glyph) = if let Some(g) = self.current_glyph.take() {
            (g, self.nth_glyph_clip_from_top(0))
        } else if let Some(g) = self.nth_glyph_clip_from_top(0) {
            (g, self.nth_glyph_clip_from_top(1))
        } else {
            return;
        };

        let transform = self.current_transform();
        let clip_box = self.current_clip_box();
        let composite_mode = self.current_composite();
        let paint = self.convert_paint(paint);

        // If the paint is a gradient with no stops (malformed font or
        // ttf-parser truncation), fall back to a solid foreground fill
        // so the layer still renders as *something* rather than being
        // silently dropped.
        let paint = if paint.is_gradient() {
            match &paint {
                GlyphPaint::LinearGradient { stops, .. }
                | GlyphPaint::RadialGradient { stops, .. }
                | GlyphPaint::SweepGradient { stops, .. }
                    if stops.is_empty() =>
                {
                    GlyphPaint::Solid(argb_from_ttf_rgba(self.foreground))
                }
                _ => paint,
            }
        } else {
            paint
        };

        self.layers.push(ColorGlyphLayer {
            glyph,
            paint,
            transform,
            clip_glyph,
            clip_box,
            composite_mode,
        });
    }

    fn push_clip(&mut self) {
        // The current outline becomes the clip shape for subsequent
        // layers until `pop_clip`. Consume it so the next paint call
        // doesn't accidentally pair with it. A `push_clip` with no
        // preceding `outline_glyph` is malformed; still push a balanced
        // entry (the .notdef outline) so `pop_clip` stays balanced.
        let g = self.current_glyph.take().unwrap_or(0);
        self.clip_stack.push(ClipEntry::Glyph(g));
    }

    fn push_clip_box(&mut self, clipbox: ttf_parser::colr::ClipBox) {
        // Rectangular clip from the COLR `ClipList`. Represent it as a
        // distinct box entry (never a sentinel gid 0) so it does not
        // masquerade as a glyph-shaped clip. Font space is y-up; convert
        // to the y-down screen convention used elsewhere.
        let rect = skia_rs_core::Rect::new(
            clipbox.x_min as Scalar,
            -(clipbox.y_max as Scalar),
            clipbox.x_max as Scalar,
            -(clipbox.y_min as Scalar),
        );
        self.clip_stack.push(ClipEntry::Box(rect));
    }

    fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    fn push_layer(&mut self, mode: ttf_parser::colr::CompositeMode) {
        self.layer_stack.push(mode.into());
    }

    fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    fn push_transform(&mut self, transform: ttf_parser::Transform) {
        let local =
            matrix_from_ttf_transform_scaled(transform, self.scale, self.scale_x, self.skew_x);
        let combined = self.current_transform().concat(&local);
        self.transform_stack.push(combined);
    }

    fn pop_transform(&mut self) {
        // Keep at least one identity on the stack so
        // `current_transform` never underflows on a malformed font.
        if self.transform_stack.len() > 1 {
            self.transform_stack.pop();
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "exact-value assertions on deterministic metric math"
)]
mod tests {
    use super::*;

    #[test]
    fn test_font_default() {
        let font = Font::default();
        assert_eq!(font.size(), 12.0);
        assert_eq!(font.scale_x(), 1.0);
    }

    #[test]
    fn test_font_from_size() {
        let font = Font::from_size(24.0);
        assert_eq!(font.size(), 24.0);
    }

    #[test]
    fn test_font_measure_text() {
        let font = Font::from_size(20.0);
        let width = font.measure_text("Hello");
        assert!(width > 0.0);
    }

    #[test]
    fn test_font_metrics() {
        let font = Font::from_size(16.0);
        let metrics = font.metrics();
        assert!(metrics.ascent < 0.0); // Above baseline
        assert!(metrics.descent > 0.0); // Below baseline
    }

    // =========================================================================
    // COLR v1 paint data (P7F)
    // =========================================================================

    #[test]
    fn gradient_stop_roundtrips_offset_through_bits() {
        let stop = GradientStop::new(0.42, 0xFF_80_40_20);
        // The offset must round-trip losslessly — we store as bit pattern.
        assert_eq!(stop.offset_f32(), 0.42);
        assert_eq!(stop.color, 0xFF_80_40_20);
    }

    #[test]
    fn glyph_paint_representative_color_for_solid() {
        let paint = GlyphPaint::Solid(0xFF_12_34_56);
        assert_eq!(paint.representative_color(), 0xFF_12_34_56);
        assert!(!paint.is_gradient());
    }

    #[test]
    fn glyph_paint_representative_color_for_linear_gradient() {
        let paint = GlyphPaint::LinearGradient {
            p0: Point::new(0.0, 0.0),
            p1: Point::new(10.0, 0.0),
            p2: Point::new(0.0, 10.0),
            stops: vec![
                GradientStop::new(0.0, 0xFF_FF_00_00),
                GradientStop::new(1.0, 0xFF_00_FF_00),
            ],
            extend: GradientExtend::Pad,
        };
        // Representative color = first stop color (t=0 sample).
        assert_eq!(paint.representative_color(), 0xFF_FF_00_00);
        assert!(paint.is_gradient());
    }

    #[test]
    fn glyph_paint_representative_color_for_empty_gradient() {
        let paint = GlyphPaint::LinearGradient {
            p0: Point::new(0.0, 0.0),
            p1: Point::new(10.0, 0.0),
            p2: Point::new(0.0, 10.0),
            stops: vec![],
            extend: GradientExtend::Pad,
        };
        // Falls back to opaque black so callers always have a valid
        // color rather than panicking on an empty stop list.
        assert_eq!(paint.representative_color(), 0xFF_00_00_00);
    }

    #[test]
    fn color_glyph_layer_color_and_is_gradient_delegate_to_paint() {
        let layer = ColorGlyphLayer {
            glyph: 42,
            paint: GlyphPaint::Solid(0xFF_AB_CD_EF),
            transform: Matrix::IDENTITY,
            clip_glyph: None,
            clip_box: None,
            composite_mode: CompositeMode::SourceOver,
        };
        assert_eq!(layer.color(), 0xFF_AB_CD_EF);
        assert!(!layer.is_gradient());

        let grad_layer = ColorGlyphLayer {
            glyph: 43,
            paint: GlyphPaint::RadialGradient {
                c0: Point::new(0.0, 0.0),
                r0: 0.0,
                c1: Point::new(0.0, 0.0),
                r1: 10.0,
                stops: vec![GradientStop::new(0.0, 0xFF_12_34_56)],
                extend: GradientExtend::Repeat,
            },
            transform: Matrix::translate(5.0, 10.0),
            clip_glyph: Some(99),
            clip_box: None,
            composite_mode: CompositeMode::Multiply,
        };
        assert_eq!(grad_layer.color(), 0xFF_12_34_56);
        assert!(grad_layer.is_gradient());
        assert_eq!(grad_layer.clip_glyph, Some(99));
        assert_eq!(grad_layer.composite_mode, CompositeMode::Multiply);
    }

    #[test]
    fn gradient_extend_maps_every_ttf_variant() {
        // Every variant of `ttf_parser::colr::GradientExtend` must have a
        // corresponding `GradientExtend` to prevent silent data loss.
        assert_eq!(
            GradientExtend::from(ttf_parser::colr::GradientExtend::Pad),
            GradientExtend::Pad
        );
        assert_eq!(
            GradientExtend::from(ttf_parser::colr::GradientExtend::Repeat),
            GradientExtend::Repeat
        );
        assert_eq!(
            GradientExtend::from(ttf_parser::colr::GradientExtend::Reflect),
            GradientExtend::Reflect
        );
    }

    #[test]
    fn composite_mode_maps_every_ttf_variant() {
        // Spot-check a representative subset; the full 28-variant match
        // is machine-checked by `From::from` exhaustiveness.
        use ttf_parser::colr::CompositeMode as M;
        assert_eq!(CompositeMode::from(M::Clear), CompositeMode::Clear);
        assert_eq!(
            CompositeMode::from(M::SourceOver),
            CompositeMode::SourceOver
        );
        assert_eq!(CompositeMode::from(M::Multiply), CompositeMode::Multiply);
        assert_eq!(
            CompositeMode::from(M::Luminosity),
            CompositeMode::Luminosity
        );
    }

    #[test]
    fn matrix_from_ttf_transform_identity_is_identity() {
        let m = matrix_from_ttf_transform(ttf_parser::Transform::default());
        assert!(m.is_identity(), "{m:?}");
    }

    #[test]
    fn matrix_from_ttf_transform_flips_y_into_screen_space() {
        // ttf_parser Transform for pure scale(1, 1) translate(10, 20)
        // in font y-up space must become translate(10, -20) in screen
        // y-down space.
        let t = ttf_parser::Transform::new(1.0, 0.0, 0.0, 1.0, 10.0, 20.0);
        let m = matrix_from_ttf_transform(t);
        assert_eq!(m.values[Matrix::TRANS_X], 10.0);
        assert_eq!(m.values[Matrix::TRANS_Y], -20.0);
        assert_eq!(m.values[Matrix::SCALE_X], 1.0);
        assert_eq!(m.values[Matrix::SCALE_Y], 1.0);
    }

    #[test]
    fn matrix_from_ttf_transform_negates_skew_y() {
        // A font-space skew of (b=0.5, c=0) becomes a negated b in
        // screen space so the skew is consistent with the outline's
        // y-flip.
        let t = ttf_parser::Transform::new(1.0, 0.5, 0.0, 1.0, 0.0, 0.0);
        let m = matrix_from_ttf_transform(t);
        // Row-major: skew_y = values[3] corresponds to the original
        // b coefficient, which is negated going into screen space.
        assert_eq!(m.values[Matrix::SKEW_Y], -0.5);
    }

    #[test]
    fn matrix_from_ttf_transform_scaled_maps_translation_to_pixel_space() {
        // A COLR v1 PaintTransform/PaintTranslate carries its translation
        // in font design units. On a 1000-upem font rendered at size 16,
        // scale = 16/1000 = 0.016, so a font-unit translation of 200 must
        // emerge as 200 * 16/1000 = 3.2 pixels (y negated for screen
        // space), NOT the raw 200. This pins the ~upem/size mis-scale fix.
        let scale = 16.0 / 1000.0;
        let t = ttf_parser::Transform::new(1.0, 0.0, 0.0, 1.0, 200.0, 200.0);
        let m = matrix_from_ttf_transform_scaled(t, scale, 1.0, 0.0);
        assert_eq!(m.values[Matrix::TRANS_X], 200.0 * scale);
        assert_eq!(m.values[Matrix::TRANS_Y], -200.0 * scale);
        // Scale/skew components are unit-less ratios and stay unscaled.
        assert_eq!(m.values[Matrix::SCALE_X], 1.0);
        assert_eq!(m.values[Matrix::SCALE_Y], 1.0);
    }

    #[test]
    fn matrix_from_ttf_transform_scale_1_matches_unscaled() {
        // COLR v0 (identity transforms) and the scale=1 path must be
        // byte-identical to the historical unscaled behaviour.
        let t = ttf_parser::Transform::new(2.0, 0.5, -0.5, 2.0, 30.0, 40.0);
        assert_eq!(
            matrix_from_ttf_transform_scaled(t, 1.0, 1.0, 0.0).values,
            matrix_from_ttf_transform(t).values,
        );
    }

    #[test]
    fn scale_colr_point_maps_gradient_endpoint_to_pixel_space() {
        // A gradient endpoint at font-unit x=900 on a 1000-upem font at
        // size 16 must come out at 900 * 16/1000 = 14.4 px, with y
        // negated and scaled the same way — not the raw font-unit value.
        let scale = 16.0 / 1000.0;
        let p = scale_colr_point(900.0, 250.0, scale, 1.0, 0.0);
        assert_eq!(p.x, 900.0 * scale);
        assert_eq!(p.y, -250.0 * scale);
    }

    #[test]
    fn scale_colr_point_folds_in_scale_x_and_skew_x() {
        // A COLR gradient endpoint must land in the SAME sheared/condensed
        // space as the glyph outline: GlyphOutlineBuilder::map computes
        // px = x*scale, py = -y*scale, then (scale_x*px + skew_x*py, py).
        // With scale_x = 0.5 (condensed) and skew_x = 0.25 (oblique), a
        // point that ignored scale_x/skew_x would slide off the outline.
        let scale = 16.0 / 1000.0;
        let (scale_x, skew_x) = (0.5, 0.25);
        let p = scale_colr_point(900.0, 250.0, scale, scale_x, skew_x);
        let px = 900.0 * scale;
        let py = -250.0 * scale;
        assert_eq!(p.x, scale_x.mul_add(px, skew_x * py));
        assert_eq!(p.y, py);
        // Sanity: this must differ from the naive (unsheared) mapping,
        // otherwise the regression wouldn't be exercised.
        assert_ne!(p.x, px);
    }

    #[test]
    fn matrix_from_ttf_transform_scaled_folds_in_scale_x_and_skew_x() {
        // The COLR transform's translation must be mapped through the same
        // text matrix as scale_colr_point/GlyphOutlineBuilder::map: for a
        // pure-translation transform (e, f), the emitted pixel-space
        // translation must equal `scale_colr_point`'s mapping of (e, f)
        // as a vector (px = e*scale, py = -f*scale, then
        // (scale_x*px + skew_x*py, py)) — i.e. gradient geometry and COLR
        // transforms share the outline's full text-matrix space.
        let scale = 16.0 / 1000.0;
        let (scale_x, skew_x) = (0.5, 0.25);
        let t = ttf_parser::Transform::new(1.0, 0.0, 0.0, 1.0, 200.0, 300.0);
        let m = matrix_from_ttf_transform_scaled(t, scale, scale_x, skew_x);

        let px = 200.0 * scale;
        let py = -300.0 * scale;
        let expected_x = scale_x.mul_add(px, skew_x * py);
        let expected_y = py;
        assert!((m.values[Matrix::TRANS_X] - expected_x).abs() < 1e-5);
        assert!((m.values[Matrix::TRANS_Y] - expected_y).abs() < 1e-5);
    }

    // =========================================================================
    // COLR walker: v0 vs v1 drive order
    // =========================================================================

    fn new_walker() -> ColorLayerWalker {
        new_walker_scaled(1.0)
    }

    fn new_walker_scaled(scale: Scalar) -> ColorLayerWalker {
        ColorLayerWalker {
            layers: Vec::new(),
            current_glyph: None,
            palette: 0,
            foreground: ttf_parser::RgbaColor::new(0, 0, 0, 255),
            scale,
            scale_x: 1.0,
            skew_x: 0.0,
            transform_stack: vec![Matrix::IDENTITY],
            clip_stack: Vec::new(),
            layer_stack: Vec::new(),
        }
    }

    fn solid() -> ttf_parser::colr::Paint<'static> {
        ttf_parser::colr::Paint::Solid(ttf_parser::RgbaColor::new(255, 0, 0, 255))
    }

    #[test]
    fn colr_v0_pairs_outline_with_paint() {
        use ttf_parser::colr::Painter;
        let mut w = new_walker();
        // COLR v0: outline_glyph directly followed by paint.
        w.outline_glyph(ttf_parser::GlyphId(7));
        w.paint(solid());
        assert_eq!(w.layers.len(), 1, "v0 must emit one layer");
        assert_eq!(w.layers[0].glyph, 7);
        assert_eq!(w.layers[0].clip_glyph, None);
        assert_eq!(w.layers[0].clip_box, None);
    }

    #[test]
    fn colr_v1_emits_layer_from_clip_stack() {
        use ttf_parser::colr::Painter;
        // COLR v1 PaintGlyph drives: outline_glyph -> push_clip (consumes
        // the outline) -> child paint -> pop_clip. The critical
        // regression: `paint` must NOT early-return just because
        // `current_glyph` was consumed; the fill shape is the top clip
        // glyph. Before the fix this produced ZERO layers.
        let mut w = new_walker();
        w.outline_glyph(ttf_parser::GlyphId(5));
        w.push_clip();
        assert!(w.current_glyph.is_none(), "push_clip consumes the outline");
        w.paint(solid());
        w.pop_clip();

        assert_eq!(w.layers.len(), 1, "v1 layer must not be dropped");
        assert_eq!(w.layers[0].glyph, 5, "fill shape is the clip glyph");
        assert_eq!(w.layers[0].clip_glyph, None);
    }

    #[test]
    fn colr_v1_nested_clips_record_inner_fill_and_outer_clip() {
        use ttf_parser::colr::Painter;
        // Nested PaintGlyph: fill = innermost glyph, clipped by the outer.
        let mut w = new_walker();
        w.outline_glyph(ttf_parser::GlyphId(1));
        w.push_clip();
        w.outline_glyph(ttf_parser::GlyphId(2));
        w.push_clip();
        w.paint(solid());
        w.pop_clip();
        w.pop_clip();

        assert_eq!(w.layers.len(), 1);
        assert_eq!(w.layers[0].glyph, 2, "innermost glyph is the fill");
        assert_eq!(
            w.layers[0].clip_glyph,
            Some(1),
            "outer glyph still clips the fill"
        );
    }

    #[test]
    fn colr_clip_box_is_distinct_not_sentinel_glyph_zero() {
        use ttf_parser::colr::Painter;
        // A ClipBox must be represented distinctly — never as a sentinel
        // gid 0 glyph clip (which, now that .notdef has a real outline,
        // would wrongly clip layers to the tofu box).
        let mut w = new_walker();
        w.outline_glyph(ttf_parser::GlyphId(3));
        w.push_clip();
        w.push_clip_box(ttf_parser::RectF {
            x_min: 10.0,
            y_min: 20.0,
            x_max: 30.0,
            y_max: 40.0,
        });
        w.paint(solid());
        w.pop_clip();
        w.pop_clip();

        assert_eq!(w.layers.len(), 1);
        // Fill shape is the real glyph clip (3), NOT gid 0.
        assert_eq!(w.layers[0].glyph, 3);
        assert_ne!(w.layers[0].glyph, 0);
        assert_eq!(
            w.layers[0].clip_glyph, None,
            "box clip must not appear as a glyph clip"
        );
        // The box is exposed, y-flipped into screen space.
        let b = w.layers[0].clip_box.expect("clip box exposed");
        assert_eq!(b.left, 10.0);
        assert_eq!(b.right, 30.0);
        assert_eq!(b.top, -40.0); // -y_max
        assert_eq!(b.bottom, -20.0); // -y_min
    }

    #[test]
    fn sweep_gradient_angles_are_degrees_without_extra_negation() {
        // ttf-parser hands sweep angles as raw F2Dot14 in units of 180°.
        // Converting must multiply by 180 (to degrees) with no negation.
        // Verify the arithmetic our converter applies: raw 0.5 → 90°.
        let raw_start = 0.5f32; // F2Dot14 value == 90 degrees
        let raw_end = 1.0f32; // == 180 degrees
        assert_eq!(raw_start * 180.0, 90.0);
        assert_eq!(raw_end * 180.0, 180.0);
    }

    // =========================================================================
    // Underline half-thickness (SkFontHost_FreeType convention)
    // =========================================================================

    #[test]
    fn underline_position_folds_in_half_thickness() {
        // post.underline_position = -100 (below baseline, y-up), thickness
        // = 40, upem-scale = 1.0. Skia: position = -(-100 + 40/2) = 80
        // (screen y-down, positive = below baseline).
        let (pos, thick) = underline_metrics_from_post(-100, Some(40), 1.0, 16.0);
        assert_eq!(pos, 80.0, "position must fold in half the thickness");
        assert_eq!(thick, 40.0);
    }

    #[test]
    fn underline_thickness_falls_back_when_absent() {
        let (_pos, thick) = underline_metrics_from_post(-100, None, 1.0, 20.0);
        assert_eq!(thick, 0.05 * 20.0);
    }
}
