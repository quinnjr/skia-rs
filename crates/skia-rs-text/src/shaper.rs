//! Text shaping using rustybuzz (`HarfBuzz` compatible).
//!
//! Text shaping converts a string of characters into positioned glyphs.

use crate::{Font, Typeface};
use skia_rs_core::Scalar;
use std::sync::Arc;

/// A glyph ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct GlyphId(pub u16);

/// Shaped glyph information.
#[derive(Debug, Clone)]
pub struct ShapedGlyph {
    /// Glyph ID.
    pub glyph_id: GlyphId,
    /// Cluster (character index) this glyph belongs to.
    pub cluster: u32,
    /// X advance.
    pub x_advance: Scalar,
    /// Y advance.
    pub y_advance: Scalar,
    /// X offset from current position.
    pub x_offset: Scalar,
    /// Y offset from current position.
    pub y_offset: Scalar,
}

/// A run of shaped glyphs.
#[derive(Debug, Clone)]
pub struct ShapedRun {
    /// The glyphs in this run.
    pub glyphs: Vec<ShapedGlyph>,
    /// The font used for this run.
    pub font: Font,
    /// The start index in the original text.
    pub start: usize,
    /// The end index in the original text.
    pub end: usize,
    /// Total advance width of this run.
    pub width: Scalar,
}

/// Text direction for shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextDirection {
    /// Left-to-right (default).
    #[default]
    Ltr,
    /// Right-to-left.
    Rtl,
}

/// Script tag for text shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Script(pub [u8; 4]);

impl Script {
    /// Latin script.
    pub const LATIN: Self = Self(*b"Latn");
    /// Arabic script.
    pub const ARABIC: Self = Self(*b"Arab");
    /// Hebrew script.
    pub const HEBREW: Self = Self(*b"Hebr");
    /// Han (Chinese) script.
    pub const HAN: Self = Self(*b"Hani");
    /// Hiragana.
    pub const HIRAGANA: Self = Self(*b"Hira");
    /// Katakana.
    pub const KATAKANA: Self = Self(*b"Kana");
    /// Hangul (Korean).
    pub const HANGUL: Self = Self(*b"Hang");
    /// Common (shared by multiple scripts).
    pub const COMMON: Self = Self(*b"Zyyy");
    /// Unknown/inherited.
    pub const UNKNOWN: Self = Self(*b"Zzzz");
}

impl Default for Script {
    fn default() -> Self {
        Self::COMMON
    }
}

/// Language tag for text shaping.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Language(pub String);

impl Language {
    /// English.
    #[must_use]
    pub fn english() -> Self {
        Self("en".to_string())
    }

    /// Arabic.
    #[must_use]
    pub fn arabic() -> Self {
        Self("ar".to_string())
    }

    /// Chinese (Simplified).
    #[must_use]
    pub fn chinese_simplified() -> Self {
        Self("zh-Hans".to_string())
    }

    /// Japanese.
    #[must_use]
    pub fn japanese() -> Self {
        Self("ja".to_string())
    }
}

/// OpenType features to enable/disable.
#[derive(Debug, Clone, Default)]
pub struct Features {
    features: Vec<(String, bool)>,
}

impl Features {
    /// Create empty feature set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable a feature.
    pub fn enable(&mut self, tag: &str) -> &mut Self {
        self.features.push((tag.to_string(), true));
        self
    }

    /// Disable a feature.
    pub fn disable(&mut self, tag: &str) -> &mut Self {
        self.features.push((tag.to_string(), false));
        self
    }

    /// Enable kerning.
    #[must_use]
    pub fn with_kerning(mut self) -> Self {
        self.enable("kern");
        self
    }

    /// Enable ligatures.
    #[must_use]
    pub fn with_ligatures(mut self) -> Self {
        self.enable("liga");
        self
    }
}

/// Text shaper using rustybuzz.
pub struct Shaper {
    /// Font database for font fallback.
    font_db: Option<Arc<fontdb::Database>>,
}

impl Default for Shaper {
    fn default() -> Self {
        Self::new()
    }
}

impl Shaper {
    /// Create a new shaper.
    #[must_use]
    pub const fn new() -> Self {
        Self { font_db: None }
    }

    /// Create a shaper with a font database for fallback.
    #[must_use]
    pub const fn with_font_db(font_db: Arc<fontdb::Database>) -> Self {
        Self {
            font_db: Some(font_db),
        }
    }

    /// Get the font database used for fallback, if any.
    #[inline]
    #[must_use]
    pub const fn font_db(&self) -> Option<&Arc<fontdb::Database>> {
        self.font_db.as_ref()
    }

    /// Shape text with the given font.
    #[must_use]
    pub fn shape(
        &self,
        text: &str,
        font: &Font,
        direction: TextDirection,
        script: Script,
        language: Option<&Language>,
    ) -> Option<Vec<ShapedRun>> {
        // Get the typeface
        let typeface = font.typeface()?;

        // Try to create a rustybuzz face from the typeface data
        let face = Self::create_face(typeface)?;

        // Create buffer
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(text);

        // Set direction
        buffer.set_direction(match direction {
            TextDirection::Ltr => rustybuzz::Direction::LeftToRight,
            TextDirection::Rtl => rustybuzz::Direction::RightToLeft,
        });

        // Set script
        if let Some(rb_script) = script_to_rustybuzz(script) {
            buffer.set_script(rb_script);
        }

        // Set language
        if let Some(lang) = language
            && let Ok(rb_lang) = lang.0.parse::<rustybuzz::Language>()
        {
            buffer.set_language(rb_lang);
        }

        // Shape the text
        let output = rustybuzz::shape(&face, &[], buffer);

        // Convert to our format. Matches SkShaper_harfbuzz.cpp:
        //   SkScalarFromHBPosX = +(1/65536) * getScaleX()
        //   SkScalarFromHBPosY = -(1/65536)   // HarfBuzz y-up, Skia y-down
        // We work in font units (rustybuzz returns unscaled positions) and
        // convert with `size / upem`, so the x conversion additionally
        // multiplies by `scale_x` and the y conversion is negated.
        let scale = font.size() / skia_rs_core::cast::scalar_from_i32(face.units_per_em());
        let scale_x = scale * font.scale_x();

        let glyphs: Vec<ShapedGlyph> = output
            .glyph_infos()
            .iter()
            .zip(output.glyph_positions().iter())
            .map(|(info, pos)| ShapedGlyph {
                glyph_id: GlyphId(u16::try_from(info.glyph_id).unwrap_or(u16::MAX)),
                cluster: info.cluster,
                x_advance: skia_rs_core::cast::scalar_from_i32(pos.x_advance) * scale_x,
                y_advance: -skia_rs_core::cast::scalar_from_i32(pos.y_advance) * scale,
                x_offset: skia_rs_core::cast::scalar_from_i32(pos.x_offset) * scale_x,
                y_offset: -skia_rs_core::cast::scalar_from_i32(pos.y_offset) * scale,
            })
            .collect();

        let width = glyphs.iter().map(|g| g.x_advance).sum();

        Some(vec![ShapedRun {
            glyphs,
            font: font.clone(),
            start: 0,
            end: text.len(),
            width,
        }])
    }

    /// Shape text with automatic script and direction detection.
    #[must_use]
    pub fn shape_auto(&self, text: &str, font: &Font) -> Option<Vec<ShapedRun>> {
        // Detect direction and script
        let direction = detect_direction(text);
        let script = detect_script(text);

        self.shape(text, font, direction, script, None)
    }

    /// Create a rustybuzz Face from a typeface.
    fn create_face(typeface: &Typeface) -> Option<rustybuzz::Face<'_>> {
        // Try to get font data
        let data = typeface.font_data()?;
        rustybuzz::Face::from_slice(data, 0)
    }
}

/// Detect text direction from content.
fn detect_direction(text: &str) -> TextDirection {
    for ch in text.chars() {
        if is_rtl_char(ch) {
            return TextDirection::Rtl;
        }
        if is_strong_ltr_char(ch) {
            return TextDirection::Ltr;
        }
    }
    TextDirection::Ltr
}

const fn is_rtl_char(ch: char) -> bool {
    matches!(ch,
        '\u{0590}'..='\u{05FF}' | // Hebrew
        '\u{0600}'..='\u{06FF}' | // Arabic
        '\u{0700}'..='\u{074F}' | // Syriac
        '\u{0750}'..='\u{077F}' | // Arabic Supplement
        '\u{08A0}'..='\u{08FF}'   // Arabic Extended-A
    )
}

const fn is_strong_ltr_char(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch.is_ascii_alphabetic()
}

/// Convert our Script to rustybuzz Script.
fn script_to_rustybuzz(script: Script) -> Option<rustybuzz::Script> {
    // rustybuzz uses unicode Script enum
    let tag = ttf_parser::Tag::from_bytes(&script.0);
    rustybuzz::Script::from_iso15924_tag(tag)
}

/// Detect script from content.
fn detect_script(text: &str) -> Script {
    for ch in text.chars() {
        // Arabic
        if matches!(ch, '\u{0600}'..='\u{06FF}' | '\u{0750}'..='\u{077F}') {
            return Script::ARABIC;
        }
        // Hebrew
        if matches!(ch, '\u{0590}'..='\u{05FF}') {
            return Script::HEBREW;
        }
        // CJK
        if matches!(ch, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}') {
            return Script::HAN;
        }
        // Hiragana
        if matches!(ch, '\u{3040}'..='\u{309F}') {
            return Script::HIRAGANA;
        }
        // Katakana
        if matches!(ch, '\u{30A0}'..='\u{30FF}') {
            return Script::KATAKANA;
        }
        // Hangul
        if matches!(ch, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}') {
            return Script::HANGUL;
        }
        // Latin
        if ch.is_ascii_alphabetic() {
            return Script::LATIN;
        }
    }
    Script::COMMON
}

/// Simple paragraph layout.
pub struct ParagraphLayout {
    /// Lines of shaped runs.
    pub lines: Vec<ParagraphLine>,
    /// Total width.
    pub width: Scalar,
    /// Total height.
    pub height: Scalar,
}

/// A line in a paragraph.
#[derive(Debug, Clone)]
pub struct ParagraphLine {
    /// Runs on this line.
    pub runs: Vec<ShapedRun>,
    /// Line width.
    pub width: Scalar,
    /// Baseline offset from top.
    pub baseline: Scalar,
    /// Line height.
    pub height: Scalar,
}

impl ParagraphLayout {
    /// Lay out text within a given width.
    #[must_use]
    pub fn layout(text: &str, font: &Font, max_width: Scalar, shaper: &Shaper) -> Option<Self> {
        let runs = shaper.shape_auto(text, font)?;
        let line_height = font.spacing();
        let ascent = font.ascent();

        let mut lines = Vec::new();
        let mut current_line = ParagraphLine {
            runs: Vec::new(),
            width: 0.0,
            baseline: ascent,
            height: line_height,
        };

        for run in runs {
            // Simple word wrapping: if run doesn't fit, start new line
            if current_line.width + run.width > max_width && current_line.width > 0.0 {
                lines.push(current_line);
                current_line = ParagraphLine {
                    runs: Vec::new(),
                    width: 0.0,
                    baseline: ascent,
                    height: line_height,
                };
            }

            current_line.width += run.width;
            current_line.runs.push(run);
        }

        if !current_line.runs.is_empty() {
            lines.push(current_line);
        }

        let width = lines.iter().map(|l| l.width).fold(0.0f32, f32::max);
        let height =
            skia_rs_core::cast::scalar_from_i32(i32::try_from(lines.len()).unwrap_or(i32::MAX))
                * line_height;

        Some(Self {
            lines,
            width,
            height,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_detection() {
        assert_eq!(detect_direction("Hello"), TextDirection::Ltr);
        assert_eq!(detect_direction("مرحبا"), TextDirection::Rtl);
        assert_eq!(detect_direction("שלום"), TextDirection::Rtl);
    }

    #[test]
    fn test_script_detection() {
        assert_eq!(detect_script("Hello"), Script::LATIN);
        assert_eq!(detect_script("مرحبا"), Script::ARABIC);
        assert_eq!(detect_script("שלום"), Script::HEBREW);
        assert_eq!(detect_script("你好"), Script::HAN);
        assert_eq!(detect_script("こんにちは"), Script::HIRAGANA);
    }
}
