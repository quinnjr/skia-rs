//! Typeface (font face) abstraction.
//!
//! A typeface represents a specific font file or font family member.

use std::sync::{Arc, OnceLock};

/// Font weight (100-900).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontWeight(pub u16);

impl FontWeight {
    /// Invisible weight.
    pub const INVISIBLE: Self = Self(0);
    /// Thin weight (100).
    pub const THIN: Self = Self(100);
    /// Extra-light weight (200).
    pub const EXTRA_LIGHT: Self = Self(200);
    /// Light weight (300).
    pub const LIGHT: Self = Self(300);
    /// Normal weight (400).
    pub const NORMAL: Self = Self(400);
    /// Medium weight (500).
    pub const MEDIUM: Self = Self(500);
    /// Semi-bold weight (600).
    pub const SEMI_BOLD: Self = Self(600);
    /// Bold weight (700).
    pub const BOLD: Self = Self(700);
    /// Extra-bold weight (800).
    pub const EXTRA_BOLD: Self = Self(800);
    /// Black weight (900).
    pub const BLACK: Self = Self(900);
    /// Extra-black weight (1000).
    pub const EXTRA_BLACK: Self = Self(1000);
}

impl Default for FontWeight {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Font width (condensed to expanded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontWidth(pub u8);

impl FontWidth {
    /// Ultra-condensed (1).
    pub const ULTRA_CONDENSED: Self = Self(1);
    /// Extra-condensed (2).
    pub const EXTRA_CONDENSED: Self = Self(2);
    /// Condensed (3).
    pub const CONDENSED: Self = Self(3);
    /// Semi-condensed (4).
    pub const SEMI_CONDENSED: Self = Self(4);
    /// Normal (5).
    pub const NORMAL: Self = Self(5);
    /// Semi-expanded (6).
    pub const SEMI_EXPANDED: Self = Self(6);
    /// Expanded (7).
    pub const EXPANDED: Self = Self(7);
    /// Extra-expanded (8).
    pub const EXTRA_EXPANDED: Self = Self(8);
    /// Ultra-expanded (9).
    pub const ULTRA_EXPANDED: Self = Self(9);
}

impl Default for FontWidth {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Font slant (upright, italic, oblique).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum FontSlant {
    /// Upright (roman).
    #[default]
    Upright = 0,
    /// Italic.
    Italic,
    /// Oblique.
    Oblique,
}

/// Cached parsed metrics and metadata from a TrueType/OpenType font.
///
/// This avoids re-parsing the font tables for every metrics query.
/// Per-glyph operations (outlines, bounding boxes) still require
/// re-parsing because `ttf_parser::Face` methods borrow from the
/// font data with a lifetime.
#[derive(Debug, Clone)]
struct ParsedTypeface {
    units_per_em: u16,
    ascender: i16,
    descender: i16,
    line_gap: i16,
    cap_height: Option<i16>,
    x_height: Option<i16>,
    italic_angle: f32,
    is_monospaced: bool,
    bbox: (i16, i16, i16, i16), // xmin, ymin, xmax, ymax
    underline_position: Option<i16>,
    underline_thickness: Option<i16>,
    strikeout_position: Option<i16>,
    strikeout_thickness: Option<i16>,
    avg_char_width: Option<i16>,
}

/// Read `xAvgCharWidth` from a raw `OS/2` table blob.
///
/// `xAvgCharWidth` is the first field after the `version` u16, i.e. a
/// signed big-endian i16 at byte offset 2. Returns `None` when the table
/// is too short to contain it (guarding truncated/malformed tables), so
/// callers fall back to a bbox-derived estimate. ttf-parser does not
/// surface this field, hence the manual offset read.
#[inline]
const fn os2_avg_char_width(table: &[u8]) -> Option<i16> {
    if table.len() >= 4 {
        Some(i16::from_be_bytes([table[2], table[3]]))
    } else {
        None
    }
}

/// Raw font metrics extracted from OpenType/TrueType tables.
///
/// These values are in font design units, not scaled to a specific size.
/// Consumers (e.g., PDF font descriptors) can use this to avoid re-parsing
/// the font with `ttf_parser`.
#[derive(Debug, Clone, Copy)]
pub struct RawFontMetrics {
    /// Units per EM (typically 1000 or 2048).
    pub units_per_em: u16,
    /// Typographic ascender (distance above baseline, positive = up).
    pub ascender: i16,
    /// Typographic descender (distance below baseline, negative = down).
    pub descender: i16,
    /// Typographic line gap (additional spacing between lines).
    pub line_gap: i16,
    /// Cap height (height of uppercase letters), if present in OS/2 table.
    pub cap_height: Option<i16>,
    /// x-height (height of lowercase 'x'), if present in OS/2 table.
    pub x_height: Option<i16>,
    /// Italic angle in degrees (0 = upright, positive = clockwise slant).
    pub italic_angle: f32,
    /// Whether this is a monospaced font (from post table).
    pub is_monospaced: bool,
    /// Global bounding box: (xmin, ymin, xmax, ymax) in font units.
    pub bbox: (i16, i16, i16, i16),
    /// Underline center position (font units, positive = above baseline).
    /// From the `post` table.
    pub underline_position: Option<i16>,
    /// Underline stroke thickness (font units). From the `post` table.
    pub underline_thickness: Option<i16>,
    /// Strikeout center position (font units). From the OS/2 table.
    pub strikeout_position: Option<i16>,
    /// Strikeout stroke thickness (font units). From the OS/2 table.
    pub strikeout_thickness: Option<i16>,
    /// Average character advance width (font units), from OS/2
    /// `xAvgCharWidth`. `None` when the font has no OS/2 table.
    pub avg_char_width: Option<i16>,
}

/// Font style combining weight, width, and slant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FontStyle {
    /// Weight.
    pub weight: FontWeight,
    /// Width.
    pub width: FontWidth,
    /// Slant.
    pub slant: FontSlant,
}

impl FontStyle {
    /// Normal style.
    pub const NORMAL: Self = Self {
        weight: FontWeight::NORMAL,
        width: FontWidth::NORMAL,
        slant: FontSlant::Upright,
    };

    /// Bold style.
    pub const BOLD: Self = Self {
        weight: FontWeight::BOLD,
        width: FontWidth::NORMAL,
        slant: FontSlant::Upright,
    };

    /// Italic style.
    pub const ITALIC: Self = Self {
        weight: FontWeight::NORMAL,
        width: FontWidth::NORMAL,
        slant: FontSlant::Italic,
    };

    /// Bold italic style.
    pub const BOLD_ITALIC: Self = Self {
        weight: FontWeight::BOLD,
        width: FontWidth::NORMAL,
        slant: FontSlant::Italic,
    };

    /// Create a new font style.
    #[must_use]
    pub const fn new(weight: FontWeight, width: FontWidth, slant: FontSlant) -> Self {
        Self {
            weight,
            width,
            slant,
        }
    }
}

/// A typeface (font face) represents a specific font.
///
/// Corresponds to Skia's `SkTypeface`.
#[derive(Debug, Clone)]
pub struct Typeface {
    /// Font family name.
    family_name: String,
    /// Font style.
    style: FontStyle,
    /// Unique ID.
    id: u32,
    /// Font data (if loaded from bytes).
    data: Option<Arc<Vec<u8>>>,
    /// Units per EM.
    units_per_em: u16,
    /// Number of glyphs.
    glyph_count: u16,
    /// Cached parsed metadata (initialized on first access).
    parsed: OnceLock<Option<ParsedTypeface>>,
}

impl Typeface {
    /// Create a default typeface.
    pub fn default_typeface() -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

        Self {
            family_name: "sans-serif".to_string(),
            style: FontStyle::NORMAL,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            data: None,
            units_per_em: 2048,
            glyph_count: 256,
            parsed: OnceLock::new(),
        }
    }

    /// Create a typeface from font data.
    ///
    /// The data is parsed with `ttf_parser` to extract real `units_per_em`,
    /// glyph count, family name, and style flags. Returns `None` if the data
    /// is not a valid OpenType/TrueType font (or collection, index 0).
    pub fn from_data(data: Vec<u8>) -> Option<Self> {
        static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

        if data.len() < 12 {
            return None;
        }

        let face = ttf_parser::Face::parse(&data, 0).ok()?;
        let units_per_em = face.units_per_em();
        let glyph_count = face.number_of_glyphs();

        // Pull the first English family name from the name table if present.
        let family_name = face
            .names()
            .into_iter()
            .find(|n| n.name_id == ttf_parser::name_id::FAMILY && n.is_unicode())
            .and_then(|n| n.to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let weight = FontWeight(face.weight().to_number());
        let slant = if face.is_italic() {
            FontSlant::Italic
        } else if face.is_oblique() {
            FontSlant::Oblique
        } else {
            FontSlant::Upright
        };
        let width = FontWidth(u8::try_from(face.width().to_number()).unwrap_or(u8::MAX));

        Some(Self {
            family_name,
            style: FontStyle {
                weight,
                width,
                slant,
            },
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            data: Some(Arc::new(data)),
            units_per_em,
            glyph_count,
            parsed: OnceLock::new(),
        })
    }

    /// Get the family name.
    #[inline]
    pub fn family_name(&self) -> &str {
        &self.family_name
    }

    /// Get the font style.
    #[inline]
    pub const fn style(&self) -> FontStyle {
        self.style
    }

    /// Get the unique ID.
    #[inline]
    pub const fn unique_id(&self) -> u32 {
        self.id
    }

    /// Get units per EM.
    #[inline]
    pub const fn units_per_em(&self) -> u16 {
        self.units_per_em
    }

    /// Get the number of glyphs.
    #[inline]
    pub const fn glyph_count(&self) -> u16 {
        self.glyph_count
    }

    /// Check if this is a bold typeface.
    #[inline]
    pub const fn is_bold(&self) -> bool {
        self.style.weight.0 >= FontWeight::BOLD.0
    }

    /// Check if this is an italic typeface.
    #[inline]
    pub fn is_italic(&self) -> bool {
        self.style.slant != FontSlant::Upright
    }

    /// Check if this typeface has fixed width glyphs.
    ///
    /// Reads the `post` table's `isFixedPitch` flag via `ttf_parser`. Returns
    /// `false` for dataless typefaces (no `post` table is loaded).
    pub fn is_fixed_pitch(&self) -> bool {
        self.parsed().is_some_and(|p| p.is_monospaced)
    }

    /// Get cached parsed metadata if font data is available.
    ///
    /// This parses the font once on first access and caches the result.
    /// Returns `None` if this typeface has no backing font data.
    fn parsed(&self) -> Option<&ParsedTypeface> {
        self.parsed
            .get_or_init(|| {
                let data = self.font_data()?;
                let face = ttf_parser::Face::parse(data, 0).ok()?;
                let (underline_position, underline_thickness) = face
                    .underline_metrics()
                    .map_or((None, None), |lm| (Some(lm.position), Some(lm.thickness)));
                let (strikeout_position, strikeout_thickness) = face
                    .strikeout_metrics()
                    .map_or((None, None), |lm| (Some(lm.position), Some(lm.thickness)));
                Some(ParsedTypeface {
                    units_per_em: face.units_per_em(),
                    ascender: face.ascender(),
                    descender: face.descender(),
                    line_gap: face.line_gap(),
                    cap_height: face.capital_height(),
                    x_height: face.x_height(),
                    italic_angle: face.italic_angle().unwrap_or(0.0),
                    is_monospaced: face.is_monospaced(),
                    bbox: {
                        let r = face.global_bounding_box();
                        (r.x_min, r.y_min, r.x_max, r.y_max)
                    },
                    underline_position,
                    underline_thickness,
                    strikeout_position,
                    strikeout_thickness,
                    // OS/2 `xAvgCharWidth` is a signed i16 at byte offset 2
                    // (immediately after the u16 table version). ttf-parser
                    // does not surface it, so read it from the raw table.
                    avg_char_width: face
                        .raw_face()
                        .table(ttf_parser::Tag::from_bytes(b"OS/2"))
                        .and_then(os2_avg_char_width),
                })
            })
            .as_ref()
    }

    /// Get the glyph ID for a character.
    ///
    /// Uses the font's cmap table via `ttf_parser` when font data is loaded.
    /// Falls back to returning the character code for ASCII when no font
    /// data is available (the default typeface has no backing font).
    pub fn char_to_glyph(&self, c: char) -> u16 {
        if let Some(data) = self.font_data() {
            if let Ok(face) = ttf_parser::Face::parse(data, 0) {
                return face.glyph_index(c).map_or(0, |g| g.0);
            }
        }
        // Fallback for the dataless default typeface — just use the codepoint
        // for ASCII so existing non-data tests continue to produce non-zero
        // glyph IDs.
        if c.is_ascii() { c as u16 } else { 0 }
    }

    /// Get glyph IDs for a string.
    pub fn chars_to_glyphs(&self, chars: &str) -> Vec<u16> {
        chars.chars().map(|c| self.char_to_glyph(c)).collect()
    }

    /// Get direct access to font data (for shaping).
    pub fn font_data(&self) -> Option<&[u8]> {
        self.data.as_ref().map(|d| d.as_slice())
    }

    /// Get raw font metrics in design units.
    ///
    /// Returns `None` if this typeface has no backing font data. These
    /// metrics are cached on first access and do not require re-parsing.
    ///
    /// Consumers that need font-table metadata (PDF font descriptors,
    /// SVG text layout) can use this instead of manually invoking
    /// `ttf_parser::Face::parse()`.
    pub fn raw_metrics(&self) -> Option<RawFontMetrics> {
        let p = self.parsed()?;
        Some(RawFontMetrics {
            units_per_em: p.units_per_em,
            ascender: p.ascender,
            descender: p.descender,
            line_gap: p.line_gap,
            cap_height: p.cap_height,
            x_height: p.x_height,
            italic_angle: p.italic_angle,
            is_monospaced: p.is_monospaced,
            bbox: p.bbox,
            underline_position: p.underline_position,
            underline_thickness: p.underline_thickness,
            strikeout_position: p.strikeout_position,
            strikeout_thickness: p.strikeout_thickness,
            avg_char_width: p.avg_char_width,
        })
    }
}

/// A reference to a typeface (shared ownership).
pub type TypefaceRef = Arc<Typeface>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_style() {
        let normal = FontStyle::NORMAL;
        assert_eq!(normal.weight, FontWeight::NORMAL);
        assert_eq!(normal.slant, FontSlant::Upright);

        let bold = FontStyle::BOLD;
        assert_eq!(bold.weight, FontWeight::BOLD);
    }

    #[test]
    fn test_typeface_default() {
        let tf = Typeface::default_typeface();
        assert_eq!(tf.family_name(), "sans-serif");
        assert!(!tf.is_bold());
        assert!(!tf.is_italic());
    }

    #[test]
    fn test_char_to_glyph() {
        let tf = Typeface::default_typeface();
        assert_eq!(tf.char_to_glyph('A'), 65);
        assert_eq!(tf.char_to_glyph('a'), 97);
    }

    #[test]
    fn os2_avg_char_width_reads_signed_be_i16_at_offset_2() {
        // Synthetic OS/2 blob: version (offset 0..2) then xAvgCharWidth
        // (offset 2..4) big-endian. 0x02F9 = 761. Trailing bytes stand in
        // for the rest of the table and must be ignored.
        let table = [0x00, 0x04, 0x02, 0xF9, 0xAB, 0xCD];
        assert_eq!(os2_avg_char_width(&table), Some(761));
    }

    #[test]
    fn os2_avg_char_width_reads_negative_value() {
        // 0xFF9C big-endian as i16 is -100 — confirms the read is signed.
        let table = [0x00, 0x05, 0xFF, 0x9C];
        assert_eq!(os2_avg_char_width(&table), Some(-100));
    }

    #[test]
    fn os2_avg_char_width_len_guard_rejects_short_table() {
        // Fewer than 4 bytes cannot hold xAvgCharWidth; must yield None
        // rather than panic, so callers use the bbox fallback.
        assert_eq!(os2_avg_char_width(&[0x00, 0x04, 0x02]), None);
        assert_eq!(os2_avg_char_width(&[]), None);
    }
}
