//! PDF font embedding support.
//!
//! This module provides font embedding for PDF documents, including:
//! - Type 1 standard fonts (14 built-in)
//! - TrueType font embedding with real metrics from the `hhea` / `OS/2` /
//!   `post` / `hmtx` / `cmap` tables via `ttf-parser`
//! - Accurate per-glyph widths pulled from the font's `hmtx` table
//! - Character usage tracking for `ToUnicode` `CMap` generation
//! - Byte-level `glyf`/`loca` subsetting: the emitted `FontFile2` stream
//!   contains only the glyph outlines actually referenced by drawn text
//!   (plus their composite-glyph dependencies and `.notdef`), while
//!   `cmap`/`hmtx`/`hhea`/`OS/2`/`post`/`name`/`head`/`maxp` are preserved
//!   unchanged so character-code-based access via `WinAnsiEncoding` keeps
//!   working. A six-letter subset prefix is prepended to the `BaseFont` name
//!   per PDF convention.

use skia_rs_core::Scalar;
use skia_rs_core::cast::{floor_to_i32, saturate_to_i32};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

pub(crate) mod subset;

/// Font type for PDF embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfFontType {
    /// Type 1 standard font (14 built-in).
    Type1,
    /// TrueType font.
    TrueType,
    /// OpenType font with CFF outlines.
    OpenTypeCff,
    /// Type 0 (composite) font for CID fonts.
    Type0,
}

/// Standard PDF fonts (the 14 base fonts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StandardFont {
    /// Times Roman.
    TimesRoman,
    /// Times Bold.
    TimesBold,
    /// Times Italic.
    TimesItalic,
    /// Times Bold Italic.
    TimesBoldItalic,
    /// Helvetica.
    Helvetica,
    /// Helvetica Bold.
    HelveticaBold,
    /// Helvetica Oblique.
    HelveticaOblique,
    /// Helvetica Bold Oblique.
    HelveticaBoldOblique,
    /// Courier.
    Courier,
    /// Courier Bold.
    CourierBold,
    /// Courier Oblique.
    CourierOblique,
    /// Courier Bold Oblique.
    CourierBoldOblique,
    /// Symbol.
    Symbol,
    /// Zapf Dingbats.
    ZapfDingbats,
}

impl StandardFont {
    /// Get the PDF base font name.
    #[must_use]
    pub const fn pdf_name(&self) -> &'static str {
        match self {
            Self::TimesRoman => "Times-Roman",
            Self::TimesBold => "Times-Bold",
            Self::TimesItalic => "Times-Italic",
            Self::TimesBoldItalic => "Times-BoldItalic",
            Self::Helvetica => "Helvetica",
            Self::HelveticaBold => "Helvetica-Bold",
            Self::HelveticaOblique => "Helvetica-Oblique",
            Self::HelveticaBoldOblique => "Helvetica-BoldOblique",
            Self::Courier => "Courier",
            Self::CourierBold => "Courier-Bold",
            Self::CourierOblique => "Courier-Oblique",
            Self::CourierBoldOblique => "Courier-BoldOblique",
            Self::Symbol => "Symbol",
            Self::ZapfDingbats => "ZapfDingbats",
        }
    }

    /// Get the encoding, if the font declares one.
    ///
    /// Symbol and `ZapfDingbats` are inherently symbolic fonts whose glyphs
    /// are only reachable through their *built-in* encoding (PDF 32000-1
    /// §9.6.6.2); declaring `/Encoding /StandardEncoding` for them is
    /// wrong (`StandardEncoding` has no entries for their glyph names) and
    /// PDF/A validators flag it. Per spec, symbolic simple fonts should
    /// omit `/Encoding` entirely so the reader uses the font's built-in
    /// one.
    #[must_use]
    pub const fn encoding(&self) -> Option<&'static str> {
        match self {
            Self::Symbol | Self::ZapfDingbats => None,
            _ => Some("WinAnsiEncoding"),
        }
    }
}

/// A PDF font resource.
#[derive(Debug, Clone)]
pub struct PdfFont {
    /// Font type.
    pub font_type: PdfFontType,
    /// Base font name.
    pub base_font: String,
    /// Object ID (assigned when writing).
    pub object_id: Option<u32>,
    /// Font descriptor object ID.
    pub descriptor_id: Option<u32>,
    /// Font encoding. `None` means "use the font's built-in encoding" —
    /// required for symbolic fonts like Symbol/ZapfDingbats, which have no
    /// entries in `StandardEncoding` or `WinAnsiEncoding`.
    pub encoding: Option<String>,
    /// Embedded font data (for TrueType/OpenType).
    pub font_data: Option<Vec<u8>>,
    /// Font flags.
    pub flags: u32,
    /// Italic angle.
    pub italic_angle: Scalar,
    /// Ascender.
    pub ascender: Scalar,
    /// Descender.
    pub descender: Scalar,
    /// Cap height.
    pub cap_height: Scalar,
    /// Stem vertical width.
    pub stem_v: Scalar,
    /// Font bounding box.
    pub bbox: [Scalar; 4],
    /// Character widths — map from character code to advance in PDF glyph
    /// units (1/1000 em). Keyed by the character *code used in the content
    /// stream*, which for TrueType-with-WinAnsi is the Unicode BMP code
    /// point for characters in `WinAnsi` and for Type1 is the standard font's
    /// fixed table.
    pub widths: HashMap<u16, u16>,
    /// First character code.
    pub first_char: u16,
    /// Last character code.
    pub last_char: u16,
    /// Glyph ids used (for future byte-level subsetting).
    pub used_glyphs: Vec<u16>,
    /// Characters used (for `ToUnicode` `CMap` generation).
    pub used_chars: BTreeMap<u32, char>,
    /// `ToUnicode` `CMap`.
    pub to_unicode: Option<String>,
    /// Subset tag (six uppercase ASCII letters) prepended to the `BaseFont`
    /// when the font is actually embedded.
    pub subset_tag: Option<String>,
}

impl PdfFont {
    /// Create a new standard Type 1 font.
    #[must_use]
    pub fn standard(font: StandardFont) -> Self {
        Self {
            font_type: PdfFontType::Type1,
            base_font: font.pdf_name().to_string(),
            object_id: None,
            descriptor_id: None,
            encoding: font.encoding().map(std::string::ToString::to_string),
            font_data: None,
            flags: 0,
            italic_angle: 0.0,
            ascender: 750.0,
            descender: -250.0,
            cap_height: 700.0,
            stem_v: 80.0,
            bbox: [-200.0, -300.0, 1200.0, 1000.0],
            widths: HashMap::new(),
            first_char: 32,
            last_char: 255,
            used_glyphs: Vec::new(),
            used_chars: BTreeMap::new(),
            to_unicode: None,
            subset_tag: None,
        }
    }

    /// Create a TrueType font from font data.
    ///
    /// Parses the font tables to populate accurate metrics and per-glyph
    /// widths, and assigns a deterministic subset tag based on the font
    /// data so the `BaseFont` name matches PDF conventions for embedded
    /// subsets. The embedded `FontFile2` stream is produced at emit time by
    /// the `subset` module: it prunes the `glyf`/`loca` tables to contain
    /// only the used glyphs (plus composite dependencies and `.notdef`)
    /// while preserving every other table — callers do not need to do
    /// anything special; draws via [`PdfCanvas::draw_text`](crate::PdfCanvas::draw_text)
    /// record usage automatically.
    #[must_use]
    pub fn truetype(name: &str, data: Vec<u8>) -> Self {
        let metrics = parse_truetype_metrics(&data);
        let subset_tag = subset_tag_for(&data, name);

        Self {
            font_type: PdfFontType::TrueType,
            base_font: name.to_string(),
            object_id: None,
            descriptor_id: None,
            encoding: Some("WinAnsiEncoding".to_string()),
            font_data: Some(data),
            flags: metrics.flags,
            italic_angle: metrics.italic_angle,
            ascender: metrics.ascender,
            descender: metrics.descender,
            cap_height: metrics.cap_height,
            stem_v: metrics.stem_v,
            bbox: metrics.bbox,
            widths: metrics.widths,
            first_char: metrics.first_char,
            last_char: metrics.last_char,
            used_glyphs: Vec::new(),
            used_chars: BTreeMap::new(),
            to_unicode: None,
            subset_tag: Some(subset_tag),
        }
    }

    /// Create a Type0 (composite) font wrapping TrueType outlines, with a
    /// `CIDFontType2` descendant addressed via `/CIDToGIDMap /Identity`
    /// (PDF 32000-1 §9.7.4). Shares metric parsing with
    /// [`truetype`](Self::truetype); the font dictionary is emitted with
    /// `/Encoding /Identity-H` and a `/DescendantFonts` array (instead of
    /// `/Widths` + `/Encoding`), so a compliant reader interprets the
    /// content stream's text strings as 2-byte CIDs rather than 1-byte
    /// `WinAnsi` codes.
    ///
    /// **Limitation:** only the `/DescendantFonts` structure (widths,
    /// `FontFile2`, `CIDSystemInfo`, etc.) is implemented today. Live
    /// per-glyph CID text drawing — shaping a string into glyph ids and
    /// emitting the matching 2-byte codes plus a CID-keyed `ToUnicode` `CMap` —
    /// is not yet implemented. Calling
    /// [`PdfCanvas::draw_text`](crate::PdfCanvas::draw_text) or
    /// [`PdfCanvas::draw_text_with_font`](crate::PdfCanvas::draw_text_with_font)
    /// against a font returned from this constructor will panic rather
    /// than silently emit invalid 1-byte codes against the `/Identity-H`
    /// encoding. Use [`truetype`](Self::truetype) if you need to draw
    /// live text today.
    #[must_use]
    pub fn truetype_cid(name: &str, data: Vec<u8>) -> Self {
        let metrics = parse_truetype_metrics(&data);
        let subset_tag = subset_tag_for(&data, name);

        Self {
            font_type: PdfFontType::Type0,
            base_font: name.to_string(),
            object_id: None,
            descriptor_id: None,
            encoding: None,
            font_data: Some(data),
            flags: metrics.flags,
            italic_angle: metrics.italic_angle,
            ascender: metrics.ascender,
            descender: metrics.descender,
            cap_height: metrics.cap_height,
            stem_v: metrics.stem_v,
            bbox: metrics.bbox,
            widths: metrics.widths,
            first_char: metrics.first_char,
            last_char: metrics.last_char,
            used_glyphs: Vec::new(),
            used_chars: BTreeMap::new(),
            to_unicode: None,
            subset_tag: Some(subset_tag),
        }
    }

    /// Mark a glyph as used (for subsetting).
    pub fn use_glyph(&mut self, glyph_id: u16) {
        if !self.used_glyphs.contains(&glyph_id) {
            self.used_glyphs.push(glyph_id);
        }
    }

    /// Record that a character was used in drawn text.
    ///
    /// Called by [`crate::PdfCanvas::draw_text`] for every character that
    /// ends up in the content stream. The recorded set is consumed by
    /// [`generate_to_unicode`](Self::generate_to_unicode) to produce a
    /// `ToUnicode` `CMap` that covers the actual characters used.
    pub fn record_char(&mut self, c: char) {
        let code = c as u32;
        self.used_chars.entry(code).or_insert(c);
        // Also flag the glyph as used if we can resolve it from the font.
        if let Some(ref data) = self.font_data
            && let Ok(face) = ttf_parser::Face::parse(data, 0)
            && let Some(gid) = face.glyph_index(c)
        {
            self.use_glyph(gid.0);
        }
    }

    /// Return the `BaseFont` name as it should appear in PDF, including the
    /// `XXXXXX+` subset prefix when one is assigned.
    #[must_use]
    pub fn pdf_base_font(&self) -> String {
        let bare = self.base_font.replace(' ', "");
        match self.subset_tag {
            Some(ref tag) => format!("{tag}+{bare}"),
            None => bare,
        }
    }

    /// Generate the `CIDFontType2` descendant font dictionary referenced
    /// from a `Type0` font's `/DescendantFonts` array (PDF 32000-1 §9.7.4,
    /// §9.7.6).
    ///
    /// `font_descriptor_id` is the object id of this font's
    /// `/FontDescriptor`. `/CIDToGIDMap` is `/Identity` — glyph ids are
    /// used directly as CIDs, matching how [`record_char`](Self::record_char)
    /// resolves glyph ids from the font's `cmap`.
    #[must_use]
    pub fn to_cid_font_dict(&self, id: u32, font_descriptor_id: u32) -> String {
        let mut dict = format!("{id} 0 obj\n<<\n");
        dict.push_str("/Type /Font\n");
        dict.push_str("/Subtype /CIDFontType2\n");
        let _ = writeln!(dict, "/BaseFont /{}", self.pdf_base_font());
        dict.push_str(
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >>\n",
        );
        let _ = writeln!(dict, "/FontDescriptor {font_descriptor_id} 0 R");
        dict.push_str("/CIDToGIDMap /Identity\n");
        dict.push_str("/DW 1000\n");
        let _ = writeln!(dict, "/W {}", self.cid_width_array());
        dict.push_str(">>\nendobj\n");
        dict
    }

    /// Build the `/W` (CID glyph width) array from the glyphs recorded via
    /// [`record_char`](Self::record_char), reading each glyph's advance
    /// from the embedded font's `hmtx` table.
    ///
    /// Uses the verbose `c [w]` per-glyph form (PDF 32000-1 §9.7.4.3) —
    /// less compact than a contiguous-run encoding, but always correct
    /// regardless of how the used-glyph set is distributed.
    fn cid_width_array(&self) -> String {
        let Some(ref data) = self.font_data else {
            return "[]".to_string();
        };
        let Ok(face) = ttf_parser::Face::parse(data, 0) else {
            return "[]".to_string();
        };
        let upem = face.units_per_em();
        if upem == 0 {
            return "[]".to_string();
        }
        let scale = 1000.0 / Scalar::from(upem);

        let mut gids: Vec<u16> = self.used_glyphs.clone();
        gids.sort_unstable();
        gids.dedup();

        let mut out = String::from("[");
        for gid in gids {
            let w = face
                .glyph_hor_advance(ttf_parser::GlyphId(gid))
                .map_or(0, |a| {
                    u32::try_from(floor_to_i32(Scalar::from(a) * scale)).unwrap_or(0)
                });
            let _ = write!(out, " {gid} [{w}]");
        }
        out.push(']');
        out
    }

    /// Generate the font descriptor PDF object.
    #[must_use]
    pub fn to_font_descriptor(&self, id: u32, font_file_id: Option<u32>) -> String {
        let mut dict = format!("{id} 0 obj\n<<\n");
        dict.push_str("/Type /FontDescriptor\n");
        let _ = writeln!(dict, "/FontName /{}", self.pdf_base_font());
        let _ = writeln!(dict, "/Flags {}", self.flags | 32); // Non-symbolic
        let _ = writeln!(
            dict,
            "/FontBBox [{} {} {} {}]",
            saturate_to_i32(self.bbox[0]),
            saturate_to_i32(self.bbox[1]),
            saturate_to_i32(self.bbox[2]),
            saturate_to_i32(self.bbox[3])
        );
        let _ = writeln!(dict, "/ItalicAngle {}", saturate_to_i32(self.italic_angle));
        let _ = writeln!(dict, "/Ascent {}", saturate_to_i32(self.ascender));
        let _ = writeln!(dict, "/Descent {}", saturate_to_i32(self.descender));
        let _ = writeln!(dict, "/CapHeight {}", saturate_to_i32(self.cap_height));
        let _ = writeln!(dict, "/StemV {}", saturate_to_i32(self.stem_v));

        if let Some(file_id) = font_file_id {
            match self.font_type {
                PdfFontType::TrueType | PdfFontType::Type0 => {
                    let _ = writeln!(dict, "/FontFile2 {file_id} 0 R");
                }
                PdfFontType::OpenTypeCff => {
                    let _ = writeln!(dict, "/FontFile3 {file_id} 0 R");
                }
                PdfFontType::Type1 => {}
            }
        }

        dict.push_str(">>\nendobj\n");
        dict
    }

    /// Generate a `ToUnicode` `CMap` that covers every character recorded via
    /// [`record_char`](Self::record_char).
    ///
    /// Simple (Type1/TrueType) fonts are indexed by a *single-byte* code in
    /// the content stream (`WinAnsiEncoding` — see
    /// [`crate::PdfCanvas::draw_text_with_font`]), so per PDF 32000-1
    /// §9.10.3 the `CMap`'s codespace range must be declared as one byte
    /// (`<00> <FF>`) with 2-hex-digit `bfchar` source codes; a 2-byte
    /// codespace against a simple font is a spec violation that breaks
    /// Unicode text extraction in readers. Type0/CID fonts keep the
    /// original 2-byte codespace. If no characters were recorded the `CMap`
    /// falls back to the printable-ASCII range so the font is still
    /// searchable.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: the printable-ASCII fallback range
    /// (32..127) is entirely valid Unicode scalar values.
    #[must_use]
    pub fn generate_to_unicode(&self) -> String {
        let is_simple = !matches!(
            self.font_type,
            PdfFontType::OpenTypeCff | PdfFontType::Type0
        );

        let mut cmap = String::new();
        cmap.push_str("/CIDInit /ProcSet findresource begin\n");
        cmap.push_str("12 dict begin\n");
        cmap.push_str("begincmap\n");
        cmap.push_str("/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n");
        cmap.push_str("/CMapName /Adobe-Identity-UCS def\n");
        cmap.push_str("/CMapType 2 def\n");
        cmap.push_str("1 begincodespacerange\n");
        if is_simple {
            cmap.push_str("<00> <FF>\n");
        } else {
            cmap.push_str("<0000> <FFFF>\n");
        }
        cmap.push_str("endcodespacerange\n");

        // Walk the used-char set. If empty, fall back to printable ASCII.
        let mut entries: Vec<(u32, char)> = if self.used_chars.is_empty() {
            (32u32..127)
                .map(|c| (c, char::from_u32(c).unwrap()))
                .collect()
        } else {
            self.used_chars
                .iter()
                .map(|(&code, &ch)| (code, ch))
                .collect()
        };
        entries.sort_by_key(|&(code, _)| code);

        // For simple fonts the CMap source code is the actual single-byte
        // content-stream code (its WinAnsi byte), which can differ from the
        // character's raw Unicode value (e.g. the C1-remapped 0x80..0x9F
        // block). Characters with no WinAnsi representation can't appear
        // faithfully in a simple-font content stream at all, so they're
        // skipped here rather than emitting a bogus >0xFF "byte".
        let simple_entries: Vec<(u8, char)> = if is_simple {
            entries
                .iter()
                .filter_map(|&(_, ch)| crate::canvas::winansi_byte(ch).map(|b| (b, ch)))
                .collect()
        } else {
            Vec::new()
        };

        // CMap bfchar entries come in blocks of at most 100.
        if is_simple {
            for chunk in simple_entries.chunks(100) {
                let _ = writeln!(cmap, "{} beginbfchar", chunk.len());
                for &(code, ch) in chunk {
                    let mut buf = [0u16; 2];
                    let units = ch.encode_utf16(&mut buf);
                    let target: String = units.iter().fold(String::new(), |mut acc, u| {
                        let _ = write!(acc, "{u:04X}");
                        acc
                    });
                    let _ = writeln!(cmap, "<{code:02X}> <{target}>");
                }
                cmap.push_str("endbfchar\n");
            }
        } else {
            for chunk in entries.chunks(100) {
                let _ = writeln!(cmap, "{} beginbfchar", chunk.len());
                for &(code, ch) in chunk {
                    let src_code = if code > 0xFFFF {
                        // Non-BMP — encode as surrogate pair in the source
                        // as well.
                        let mut buf = [0u16; 2];
                        let slice = ch.encode_utf16(&mut buf);
                        format!("{:04X}{:04X}", slice[0], slice.get(1).copied().unwrap_or(0))
                    } else {
                        format!("{code:04X}")
                    };
                    // Target: UTF-16BE representation (surrogate pair if needed).
                    let mut buf = [0u16; 2];
                    let units = ch.encode_utf16(&mut buf);
                    let target: String = units.iter().fold(String::new(), |mut acc, u| {
                        let _ = write!(acc, "{u:04X}");
                        acc
                    });

                    let _ = writeln!(cmap, "<{src_code}> <{target}>");
                }
                cmap.push_str("endbfchar\n");
            }
        }

        cmap.push_str("endcmap\n");
        cmap.push_str("CMapName currentdict /CMap defineresource pop\n");
        cmap.push_str("end\n");
        cmap.push_str("end\n");

        cmap
    }
}

/// Parsed TrueType metrics.
struct TrueTypeMetrics {
    flags: u32,
    italic_angle: Scalar,
    ascender: Scalar,
    descender: Scalar,
    cap_height: Scalar,
    stem_v: Scalar,
    bbox: [Scalar; 4],
    widths: HashMap<u16, u16>,
    first_char: u16,
    last_char: u16,
}

/// Parse real metrics and widths from TrueType/OpenType font data.
///
/// Reads:
/// - `head` bounding box
/// - `hhea`/`OS/2` ascent, descent, italic angle
/// - `post` italic angle (fallback)
/// - `hmtx` per-glyph advance, converted to PDF glyph units (1/1000 em)
/// - `cmap` character → glyph mapping, used to populate the `widths` map
///   keyed by `WinAnsi` character code
fn parse_truetype_metrics(data: &[u8]) -> TrueTypeMetrics {
    let mut metrics = TrueTypeMetrics {
        flags: 0,
        italic_angle: 0.0,
        ascender: 750.0,
        descender: -250.0,
        cap_height: 700.0,
        stem_v: 80.0,
        bbox: [0.0, -250.0, 1000.0, 750.0],
        widths: HashMap::new(),
        first_char: 32,
        last_char: 126,
    };

    let Ok(face) = ttf_parser::Face::parse(data, 0) else {
        return metrics;
    };

    let upem = face.units_per_em();
    if upem == 0 {
        return metrics;
    }
    // Convert font-units to PDF glyph units (1/1000 em).
    let scale = 1000.0 / Scalar::from(upem);

    metrics.ascender = Scalar::from(face.ascender()) * scale;
    metrics.descender = Scalar::from(face.descender()) * scale;
    metrics.italic_angle = face.italic_angle().unwrap_or(0.0) as Scalar;
    metrics.cap_height = face
        .capital_height()
        .map_or(metrics.ascender * 0.875, |v| Scalar::from(v) * scale);

    let gbox = face.global_bounding_box();
    metrics.bbox = [
        Scalar::from(gbox.x_min) * scale,
        Scalar::from(gbox.y_min) * scale,
        Scalar::from(gbox.x_max) * scale,
        Scalar::from(gbox.y_max) * scale,
    ];

    // Heuristic StemV: a function of x-height works well for most text
    // fonts. Fall back to 80 when no x-height is declared.
    metrics.stem_v = face
        .x_height()
        .map_or(80.0, |xh| (Scalar::from(xh) * scale) * 0.12);

    // Flags bits (PDF 9.8.2):
    //   bit 1 (FixedPitch)  = 1
    //   bit 3 (Symbolic)    = 4
    //   bit 6 (Nonsymbolic) = 32
    //   bit 7 (Italic)      = 64
    let mut flags = 0u32;
    if face.is_monospaced() {
        flags |= 1;
    }
    if face.is_italic() || metrics.italic_angle.abs() > 0.01 {
        flags |= 64;
    }
    // Default to non-symbolic unless the font is clearly symbolic.
    flags |= 32;
    metrics.flags = flags;

    // Populate widths for WinAnsi 32..=255. PDF readers expect Widths[i]
    // (for i in FirstChar..=LastChar) to equal the advance of the glyph
    // assigned to character code i by the encoding. For WinAnsiEncoding
    // Latin-1 characters 32..=255 map to their Unicode code point with a
    // handful of rearrangements in 0x80..=0x9F — we use the BMP mapping
    // for simplicity, which is correct for the overwhelming majority of
    // characters and wrong only for 0x80..0x9F decoratives.
    for code in 32u16..=255u16 {
        let ch = char::from_u32(u32::from(code)).unwrap_or('\0');
        if let Some(gid) = face.glyph_index(ch)
            && let Some(adv) = face.glyph_hor_advance(gid)
        {
            metrics.widths.insert(
                code,
                u16::try_from(floor_to_i32(Scalar::from(adv) * scale)).unwrap_or(0),
            );
        }
    }

    // Ensure FirstChar/LastChar cover any inserted width. Keeping a wider
    // range is valid; PDF just requires Widths.len() == LastChar-FirstChar+1.
    if let Some(&min) = metrics.widths.keys().min() {
        metrics.first_char = min;
    }
    if let Some(&max) = metrics.widths.keys().max() {
        metrics.last_char = max;
    }

    metrics
}

/// Deterministically derive a 6-character subset tag from the font data and
/// name.
///
/// PDF subsets are identified by a six-uppercase-ASCII prefix such as
/// `ABCDEF+FontName`. The tag must be unique within a PDF but is otherwise
/// arbitrary — we derive it from a 64-bit FNV-1a hash so that two calls for
/// the same font produce the same tag (stable across runs).
fn subset_tag_for(data: &[u8], name: &str) -> String {
    // Convert 64 bits into six base-26 letters.
    const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // FNV-1a 64.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let mut tag = [0u8; 6];
    let mut v = h;
    for slot in &mut tag {
        *slot = LETTERS[usize::try_from(v % 26).unwrap_or(0)];
        v /= 26;
    }
    String::from_utf8(tag.to_vec()).unwrap()
}

/// Font manager for PDF documents.
#[derive(Debug, Default)]
pub struct PdfFontManager {
    /// Registered fonts.
    fonts: Vec<PdfFont>,
    /// Font name to index mapping.
    name_to_index: HashMap<String, usize>,
}

impl PdfFontManager {
    /// Create a new font manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a standard font.
    pub fn register_standard(&mut self, font: StandardFont) -> usize {
        let name = font.pdf_name().to_string();
        if let Some(&idx) = self.name_to_index.get(&name) {
            return idx;
        }

        let idx = self.fonts.len();
        self.fonts.push(PdfFont::standard(font));
        self.name_to_index.insert(name, idx);
        idx
    }

    /// Register a TrueType font.
    pub fn register_truetype(&mut self, name: &str, data: Vec<u8>) -> usize {
        if let Some(&idx) = self.name_to_index.get(name) {
            return idx;
        }

        let idx = self.fonts.len();
        self.fonts.push(PdfFont::truetype(name, data));
        self.name_to_index.insert(name.to_string(), idx);
        idx
    }

    /// Register a Type0 (CID-keyed) TrueType font — see
    /// [`PdfFont::truetype_cid`]. Uses a distinct name key (`{name}#cid`)
    /// so the same underlying font data can be registered both as a simple
    /// TrueType font and as a Type0 font without colliding.
    pub fn register_truetype_cid(&mut self, name: &str, data: Vec<u8>) -> usize {
        let key = format!("{name}#cid");
        if let Some(&idx) = self.name_to_index.get(&key) {
            return idx;
        }

        let idx = self.fonts.len();
        self.fonts.push(PdfFont::truetype_cid(name, data));
        self.name_to_index.insert(key, idx);
        idx
    }

    /// Get font by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&PdfFont> {
        self.fonts.get(index)
    }

    /// Get mutable font by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut PdfFont> {
        self.fonts.get_mut(index)
    }

    /// Get font by name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&PdfFont> {
        self.name_to_index
            .get(name)
            .and_then(|&idx| self.fonts.get(idx))
    }

    /// Get all fonts.
    #[must_use]
    pub fn fonts(&self) -> &[PdfFont] {
        &self.fonts
    }

    /// Get mutable fonts.
    pub fn fonts_mut(&mut self) -> &mut [PdfFont] {
        &mut self.fonts
    }

    /// Get number of fonts.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.fonts.len()
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.fonts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_font() {
        let font = PdfFont::standard(StandardFont::Helvetica);
        assert_eq!(font.base_font, "Helvetica");
        assert_eq!(font.font_type, PdfFontType::Type1);
    }

    #[test]
    fn test_font_manager() {
        let mut manager = PdfFontManager::new();

        let idx1 = manager.register_standard(StandardFont::Helvetica);
        let idx2 = manager.register_standard(StandardFont::Helvetica); // Same font

        assert_eq!(idx1, idx2); // Should return same index
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_cid_font_dict() {
        let font = PdfFont::truetype_cid("MyFont", vec![0, 1, 2, 3]);
        let cid_dict = font.to_cid_font_dict(11, 12);
        assert!(cid_dict.contains("/Subtype /CIDFontType2"));
        assert!(cid_dict.contains("/CIDToGIDMap /Identity"));
        assert!(cid_dict.contains("/FontDescriptor 12 0 R"));
        assert!(
            cid_dict.contains(
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >>"
            ),
            "dict: {cid_dict}"
        );
        assert!(cid_dict.contains("/W ["));
    }

    #[test]
    fn test_subset_tag_is_stable_and_valid() {
        let tag = subset_tag_for(b"hello world", "MyFont");
        assert_eq!(tag.len(), 6);
        assert!(tag.chars().all(|c| c.is_ascii_uppercase()));
        // Determinism.
        assert_eq!(tag, subset_tag_for(b"hello world", "MyFont"));
    }

    #[test]
    fn test_to_unicode_uses_recorded_chars() {
        let mut font = PdfFont::standard(StandardFont::Helvetica);
        font.record_char('A');
        font.record_char('é');

        let cmap = font.generate_to_unicode();
        // Simple (Type1) fonts are single-byte-indexed: the codespace must
        // be declared as one byte and bfchar source codes are 2 hex
        // digits, matching the actual content-stream (WinAnsi) code —
        // never the raw 4-digit Unicode value.
        assert!(cmap.contains("<00> <FF>\n"), "cmap: {cmap}");
        // ASCII 'A' → WinAnsi byte 0x41, target UTF-16BE 0041.
        assert!(cmap.contains("<41> <0041>"), "cmap: {cmap}");
        // 'é' → WinAnsi byte 0xE9 (coincides with its Unicode value here).
        assert!(cmap.contains("<E9> <00E9>"), "cmap: {cmap}");
        // We produced exactly 2 entries (beginbfchar count).
        assert!(cmap.contains("2 beginbfchar"));
    }

    #[test]
    fn test_to_unicode_fallback_when_empty() {
        let font = PdfFont::standard(StandardFont::Helvetica);
        let cmap = font.generate_to_unicode();
        // Fallback range is 32..127 — 95 entries, all ASCII so WinAnsi byte
        // equals the printable-ASCII code.
        assert!(cmap.contains("95 beginbfchar"));
        assert!(cmap.contains("<41> <0041>"));
    }

    #[test]
    fn test_truetype_invalid_data_defaults() {
        let font = PdfFont::truetype("Bogus", vec![0, 1, 2, 3]);
        // Defaults.
        assert_eq!(saturate_to_i32(font.ascender), 750);
        assert_eq!(saturate_to_i32(font.descender), -250);
        // Subset tag is still assigned so the BaseFont name is consistent.
        assert!(font.subset_tag.is_some());
        assert!(font.pdf_base_font().contains('+'));
    }

    #[test]
    fn test_record_char_tracks_all_used() {
        let mut font = PdfFont::standard(StandardFont::Helvetica);
        for c in "Hello, world!".chars() {
            font.record_char(c);
        }
        // Unique characters: H, e, l, o, ',', ' ', w, r, d, '!' = 10
        assert_eq!(font.used_chars.len(), 10);
    }
}
