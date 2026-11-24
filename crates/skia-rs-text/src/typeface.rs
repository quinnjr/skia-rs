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

/// Source of [`Typeface::unique_id`] values, shared by every constructor so
/// that IDs never collide between typefaces created through different paths.
static NEXT_TYPEFACE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn next_typeface_id() -> u32 {
    NEXT_TYPEFACE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Whether a font read error is clearly transient (so a miss should not be
/// cached). Anything else (missing, forbidden, a loop, an I/O error) counts
/// as permanent for that candidate.
fn is_transient(error: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    // Too many open files (EMFILE/ENFILE; Windows'
    // ERROR_TOO_MANY_OPEN_FILES), and on Windows a file briefly held by
    // another process (ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION, e.g.
    // antivirus or a font install), which std leaves uncategorized.
    #[cfg(unix)]
    const TRANSIENT_OS_ERRORS: &[i32] = &[23, 24];
    #[cfg(windows)]
    const TRANSIENT_OS_ERRORS: &[i32] = &[4, 32, 33];
    #[cfg(not(any(unix, windows)))]
    const TRANSIENT_OS_ERRORS: &[i32] = &[];
    matches!(
        error.kind(),
        ErrorKind::Interrupted
            | ErrorKind::WouldBlock
            | ErrorKind::TimedOut
            | ErrorKind::ResourceBusy
            | ErrorKind::OutOfMemory
    ) || error
        .raw_os_error()
        .is_some_and(|code| TRANSIENT_OS_ERRORS.contains(&code))
}

/// Well-known locations of each platform's default sans-serif font, in order
/// of preference. Mirrors the families Skia's platform font managers resolve
/// for a `nullptr` family name.
fn system_default_font_candidates() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut paths = Vec::new();
    if cfg!(target_os = "windows") {
        let windir = std::env::var_os("WINDIR")
            .or_else(|| std::env::var_os("SystemRoot"))
            .map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from);
        let fonts = windir.join("Fonts");
        paths.extend(["segoeui.ttf", "arial.ttf", "tahoma.ttf"].map(|f| fonts.join(f)));
    } else if cfg!(target_os = "ios") {
        // iOS system fonts are not at stable paths, and sandboxed apps cannot
        // read them by path; load bundled fonts with Typeface::from_data.
    } else if cfg!(target_os = "macos") {
        paths.extend(
            [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Geneva.ttf",
            ]
            .map(PathBuf::from),
        );
    } else if cfg!(target_os = "android") {
        paths.extend(
            [
                "/system/fonts/Roboto-Regular.ttf",
                "/system/fonts/DroidSans.ttf",
            ]
            .map(PathBuf::from),
        );
    } else {
        // Linux / BSD: the common distro locations of DejaVu, Liberation and
        // Noto Sans (the usual fontconfig `sans-serif` matches).
        paths.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
                "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
            ]
            .map(PathBuf::from),
        );
    }
    paths
}

impl Typeface {
    /// Create a default typeface.
    ///
    /// This is a data-less placeholder: it reports plausible metrics and can
    /// measure text, but it carries no glyph outlines, so drawing text with it
    /// produces no pixels. It never touches the file system, which keeps it
    /// deterministic. Use [`system_default`](Self::system_default) for a real
    /// platform font.
    #[must_use]
    pub fn default_typeface() -> Self {
        Self {
            family_name: "sans-serif".to_string(),
            style: FontStyle::NORMAL,
            id: next_typeface_id(),
            data: None,
            units_per_em: 2048,
            glyph_count: 256,
            parsed: OnceLock::new(),
        }
    }

    /// Load the platform's default sans-serif typeface from the installed
    /// system fonts (`Segoe UI` / `Arial` on Windows, `Helvetica` on macOS,
    /// `DejaVu Sans` / `Liberation Sans` / `Noto Sans` on Linux, `Roboto` on
    /// Android).
    ///
    /// Corresponds to Skia's
    /// `SkFontMgr::legacyMakeTypeface(nullptr, SkFontStyle())`. Unlike
    /// [`default_typeface`](Self::default_typeface),
    /// the result has real glyph outlines and renders visible text. Like the
    /// typeface Skia's font manager caches, it is loaded once per process:
    /// every call returns the same typeface (same [`unique_id`](Self::unique_id)
    /// and shared font bytes), so per-typeface caches keep hitting.
    ///
    /// Returns `None` when none of the well-known font files exists, e.g. in
    /// a minimal container without fonts installed, and always on iOS, where
    /// system fonts cannot be read by path (bundle a font and use
    /// [`from_data`](Self::from_data)). Like Skia's default typeface (set
    /// up once with `SkOnce`), the lookup runs once per process whether it
    /// finds a font or not, so a font-less environment does not re-probe the
    /// candidate paths on every call. A font that loads is always kept, even
    /// a fallback reached because a preferred candidate failed, so every call
    /// returns the same typeface. A miss is not kept when a candidate failed
    /// with a clearly transient error (too many open files, an interrupted,
    /// timed-out or busy read, or on Windows a file briefly held by another
    /// process: a sharing or lock violation): the next call tries again.
    #[must_use]
    pub fn system_default() -> Option<Self> {
        // A hit, or a definite miss, is cached; every call shares it.
        static TYPEFACE: OnceLock<Option<Typeface>> = OnceLock::new();

        if let Some(cached) = TYPEFACE.get() {
            return cached.clone();
        }
        let mut transient_error = false;
        let found =
            system_default_font_candidates()
                .into_iter()
                .find_map(|path| match std::fs::read(path) {
                    Ok(data) => Self::from_data(data),
                    Err(e) => {
                        transient_error |= is_transient(&e);
                        None
                    }
                });
        // A miss after a transient error may clear up: try again next call.
        if found.is_none() && transient_error {
            return None;
        }
        // Parse the font tables before caching: clones share the filled
        // OnceLock, so callers do not each re-parse them.
        if let Some(typeface) = &found {
            let _ = typeface.parsed();
        }
        // If two threads look up concurrently, the first result is kept.
        TYPEFACE.get_or_init(|| found).clone()
    }

    /// Create a typeface from font data.
    ///
    /// The data is parsed with `ttf_parser` to extract real `units_per_em`,
    /// glyph count, family name, and style flags. Returns `None` if the data
    /// is not a valid OpenType/TrueType font (or collection, index 0).
    #[must_use]
    pub fn from_data(data: Vec<u8>) -> Option<Self> {
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
            id: next_typeface_id(),
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
        if let Some(data) = self.font_data()
            && let Ok(face) = ttf_parser::Face::parse(data, 0)
        {
            return face.glyph_index(c).map_or(0, |g| g.0);
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
    #[test]
    fn the_system_default_typeface_shares_its_parsed_tables() {
        // Needs a system font; where there is none there is nothing to share.
        if let Some(typeface) = Typeface::system_default() {
            assert!(typeface.parsed.get().is_some(), "parsed before caching");
        }
    }

    #[test]
    fn only_clearly_transient_read_errors_are_retried() {
        use std::io::{Error, ErrorKind};
        assert!(super::is_transient(&Error::from(ErrorKind::Interrupted)));
        assert!(super::is_transient(&Error::from(ErrorKind::TimedOut)));
        #[cfg(unix)]
        assert!(super::is_transient(&Error::from_raw_os_error(24)));
        #[cfg(windows)]
        for code in [4, 32, 33] {
            assert!(
                super::is_transient(&Error::from_raw_os_error(code)),
                "{code}"
            );
        }
        assert!(!super::is_transient(&Error::from(ErrorKind::NotFound)));
        assert!(!super::is_transient(&Error::from(
            ErrorKind::PermissionDenied
        )));
        assert!(!super::is_transient(&Error::other("I/O error")));
    }

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
    fn test_unique_ids_do_not_collide_across_constructors() {
        let font = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/demo.ttf"
        ))
        .expect("demo.ttf fixture (git lfs pull)");
        let ids = [
            Typeface::default_typeface().unique_id(),
            Typeface::from_data(font.clone()).unwrap().unique_id(),
            Typeface::default_typeface().unique_id(),
            Typeface::from_data(font).unwrap().unique_id(),
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "duplicate typeface ids: {ids:?}");
    }

    #[test]
    fn test_system_default_has_real_glyphs_when_available() {
        // Machines without any of the well-known fonts (minimal containers)
        // legitimately get `None`; everywhere else the typeface must carry
        // real outlines, unlike the data-less `default_typeface`.
        let Some(tf) = Typeface::system_default() else {
            return;
        };
        assert!(tf.glyph_count() > 0);
        assert_ne!(tf.char_to_glyph('A'), 0, "system font must map 'A'");
        // Like Skia's cached default typeface: every call is the same face.
        let again = Typeface::system_default().expect("cached font stays available");
        assert_eq!(tf.unique_id(), again.unique_id());
        assert_eq!(
            tf.font_data().map(<[u8]>::as_ptr),
            again.font_data().map(<[u8]>::as_ptr),
            "font bytes must be shared, not re-read"
        );
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
