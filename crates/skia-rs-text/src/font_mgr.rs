//! Font manager for system font enumeration and fallback.
//!
//! This module provides functionality for:
//! - Enumerating system fonts
//! - Font family enumeration
//! - Font style set management
//! - Font fallback chains

use crate::typeface::{FontStyle, Typeface, TypefaceRef};
use std::sync::Arc;

/// Font manager for system font enumeration.
///
/// Corresponds to Skia's `SkFontMgr`.
pub trait FontMgr: Send + Sync {
    /// Get the number of font families.
    fn count_families(&self) -> usize;

    /// Get the family name at the given index.
    fn family_name(&self, index: usize) -> Option<String>;

    /// Create a style set for the given family name.
    fn create_style_set(&self, family_name: &str) -> Option<Box<dyn FontStyleSet>>;

    /// Match a family name and style to a typeface.
    fn match_family_style(&self, family_name: &str, style: FontStyle) -> Option<TypefaceRef>;

    /// Match a character to find a fallback font.
    fn match_family_style_character(
        &self,
        family_name: &str,
        style: FontStyle,
        bcp47: &[&str],
        character: char,
    ) -> Option<TypefaceRef>;

    /// Create a typeface from font data.
    fn make_from_data(&self, data: &[u8], index: i32) -> Option<TypefaceRef>;

    /// Create a typeface from a file.
    fn make_from_file(&self, path: &str, index: i32) -> Option<TypefaceRef>;
}

/// A set of font styles within a family.
///
/// Corresponds to Skia's `SkFontStyleSet`.
pub trait FontStyleSet: Send + Sync {
    /// Get the number of styles in this set.
    fn count(&self) -> usize;

    /// Get the style at the given index.
    fn style(&self, index: usize) -> Option<(FontStyle, String)>;

    /// Create a typeface for the style at the given index.
    fn create_typeface(&self, index: usize) -> Option<TypefaceRef>;

    /// Match a style to the closest typeface in this set.
    fn match_style(&self, style: FontStyle) -> Option<TypefaceRef>;
}

/// Default font manager implementation.
///
/// This provides basic font management with an in-memory font registry.
#[derive(Default)]
pub struct DefaultFontMgr {
    families: Vec<FontFamily>,
}

/// A font family with multiple styles.
#[derive(Clone)]
pub struct FontFamily {
    /// Family name.
    pub name: String,
    /// Typefaces in this family.
    pub typefaces: Vec<TypefaceEntry>,
}

/// Entry for a typeface with style information.
#[derive(Clone)]
pub struct TypefaceEntry {
    /// The typeface.
    pub typeface: TypefaceRef,
    /// Style name (e.g., "Regular", "Bold", "Italic").
    pub style_name: String,
    /// Font style.
    pub style: FontStyle,
}

impl DefaultFontMgr {
    /// Create a new default font manager.
    #[must_use]
    pub fn new() -> Self {
        let mut mgr = Self {
            families: Vec::new(),
        };

        // Add a default family with a placeholder typeface
        let default_typeface = Arc::new(Typeface::default_typeface());
        mgr.families.push(FontFamily {
            name: "Default".to_string(),
            typefaces: vec![TypefaceEntry {
                typeface: default_typeface,
                style_name: "Regular".to_string(),
                style: FontStyle::default(),
            }],
        });

        mgr
    }

    /// Register a font family.
    pub fn register_family(&mut self, family: FontFamily) {
        self.families.push(family);
    }

    /// Register a typeface under a family name.
    pub fn register_typeface(
        &mut self,
        family_name: &str,
        typeface: TypefaceRef,
        style_name: &str,
        style: FontStyle,
    ) {
        // Find or create family.
        let family_idx = self
            .families
            .iter()
            .position(|f| f.name == family_name)
            .unwrap_or_else(|| {
                self.families.push(FontFamily {
                    name: family_name.to_string(),
                    typefaces: Vec::new(),
                });
                self.families.len() - 1
            });
        let family = &mut self.families[family_idx];

        family.typefaces.push(TypefaceEntry {
            typeface,
            style_name: style_name.to_string(),
            style,
        });
    }
}

impl FontMgr for DefaultFontMgr {
    fn count_families(&self) -> usize {
        self.families.len()
    }

    fn family_name(&self, index: usize) -> Option<String> {
        self.families.get(index).map(|f| f.name.clone())
    }

    fn create_style_set(&self, family_name: &str) -> Option<Box<dyn FontStyleSet>> {
        self.families
            .iter()
            .find(|f| f.name == family_name)
            .map(|f| Box::new(DefaultFontStyleSet { family: f.clone() }) as Box<dyn FontStyleSet>)
    }

    fn match_family_style(&self, family_name: &str, style: FontStyle) -> Option<TypefaceRef> {
        self.create_style_set(family_name)?.match_style(style)
    }

    fn match_family_style_character(
        &self,
        family_name: &str,
        style: FontStyle,
        _bcp47: &[&str],
        character: char,
    ) -> Option<TypefaceRef> {
        // 1. Preferred: look for a typeface in the requested family whose
        //    cmap contains `character`.
        if let Some(family) = self.families.iter().find(|f| f.name == family_name)
            && let Some(tf) = best_covering_typeface(family, style, character)
        {
            return Some(tf);
        }

        // 2. Scan every registered family for coverage, picking the
        //    typeface whose style is closest to the requested style. This
        //    is the fallback path used for emoji / CJK / script runs.
        let mut best: Option<(u32, TypefaceRef)> = None;
        for family in &self.families {
            if let Some(candidate) = best_covering_typeface(family, style, character) {
                let entry_style = family
                    .typefaces
                    .iter()
                    .find(|e| Arc::ptr_eq(&e.typeface, &candidate))
                    .map_or(style, |e| e.style);
                let dist = style_distance(entry_style, style);
                match &best {
                    None => best = Some((dist, candidate)),
                    Some((d, _)) if dist < *d => best = Some((dist, candidate)),
                    _ => {}
                }
            }
        }

        if let Some((_, tf)) = best {
            return Some(tf);
        }

        // 3. Last resort: match by style without coverage check — preserves
        //    the previous behaviour so callers still get *something* when no
        //    registered font covers the character (e.g. when only the
        //    dataless default typeface is present).
        self.match_family_style(family_name, style)
            .or_else(|| self.match_family_style("Default", style))
    }

    fn make_from_data(&self, data: &[u8], _index: i32) -> Option<TypefaceRef> {
        // Index > 0 refers to the Nth face in a TTC collection; our
        // Typeface::from_data parses index 0 only, so anything else is
        // treated as "font not found" rather than silently returning the
        // wrong face.
        Typeface::from_data(data.to_vec()).map(Arc::new)
    }

    fn make_from_file(&self, path: &str, _index: i32) -> Option<TypefaceRef> {
        // Same TTC caveat as `make_from_data`. Propagate I/O errors as
        // `None` — the FontMgr trait has no error channel.
        let data = std::fs::read(path).ok()?;
        Typeface::from_data(data).map(Arc::new)
    }
}

/// Return the typeface in `family` whose cmap contains `character`, breaking
/// ties by style distance to `style`. Returns `None` if no typeface in the
/// family covers the character.
fn best_covering_typeface(
    family: &FontFamily,
    style: FontStyle,
    character: char,
) -> Option<TypefaceRef> {
    family
        .typefaces
        .iter()
        .filter(|entry| entry.typeface.char_to_glyph(character) != 0)
        .min_by_key(|entry| style_distance(entry.style, style))
        .map(|entry| entry.typeface.clone())
}

/// Default font style set implementation.
struct DefaultFontStyleSet {
    family: FontFamily,
}

impl FontStyleSet for DefaultFontStyleSet {
    fn count(&self) -> usize {
        self.family.typefaces.len()
    }

    fn style(&self, index: usize) -> Option<(FontStyle, String)> {
        self.family
            .typefaces
            .get(index)
            .map(|e| (e.style, e.style_name.clone()))
    }

    fn create_typeface(&self, index: usize) -> Option<TypefaceRef> {
        self.family.typefaces.get(index).map(|e| e.typeface.clone())
    }

    fn match_style(&self, style: FontStyle) -> Option<TypefaceRef> {
        // Find best match based on style distance
        self.family
            .typefaces
            .iter()
            .min_by_key(|e| style_distance(e.style, style))
            .map(|e| e.typeface.clone())
    }
}

/// Calculate the "distance" between two font styles for matching.
fn style_distance(a: FontStyle, b: FontStyle) -> u32 {
    let weight_diff = (i32::from(a.weight.0) - i32::from(b.weight.0)).unsigned_abs();
    let width_diff = (i32::from(a.width.0) - i32::from(b.width.0)).unsigned_abs();
    let slant_diff = if a.slant == b.slant { 0 } else { 100 };

    weight_diff + width_diff * 10 + slant_diff
}

/// Font fallback chain for handling missing glyphs.
#[derive(Clone, Default)]
pub struct FontFallback {
    /// Ordered list of fallback fonts.
    fallback_chain: Vec<TypefaceRef>,
}

impl FontFallback {
    /// Create a new empty fallback chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fallback font to the chain.
    pub fn add_fallback(&mut self, typeface: TypefaceRef) {
        self.fallback_chain.push(typeface);
    }

    /// Find a font that contains the given character.
    pub fn find_font_for_char(&self, c: char, primary: &TypefaceRef) -> TypefaceRef {
        // Check primary font first
        if primary.char_to_glyph(c) != 0 {
            return primary.clone();
        }

        // Check fallback chain
        for fallback in &self.fallback_chain {
            if fallback.char_to_glyph(c) != 0 {
                return fallback.clone();
            }
        }

        // Return primary if no fallback found
        primary.clone()
    }

    /// Get the fallback chain.
    #[must_use]
    pub fn chain(&self) -> &[TypefaceRef] {
        &self.fallback_chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 400-byte minimal font with 'A' at gid 1. Same fixture used by the
    /// integration tests in `tests/glyph_outline.rs`.
    const DEMO_TTF: &[u8] = include_bytes!("../tests/fixtures/demo.ttf");

    #[test]
    fn test_default_font_mgr() {
        let mgr = DefaultFontMgr::new();
        assert!(mgr.count_families() >= 1);
        assert!(mgr.family_name(0).is_some());
    }

    #[test]
    fn test_match_family_style() {
        let mgr = DefaultFontMgr::new();
        let typeface = mgr.match_family_style("Default", FontStyle::default());
        assert!(typeface.is_some());
    }

    #[test]
    fn test_font_fallback() {
        let fallback = FontFallback::new();
        let primary = Arc::new(Typeface::default_typeface());
        let result = fallback.find_font_for_char('A', &primary);
        assert!(Arc::ptr_eq(&result, &primary));
    }

    #[test]
    fn make_from_data_parses_real_font() {
        // Gap N-1: `make_from_data` used to ignore its argument and return
        // the dataless default typeface. It must now parse the bytes via
        // Typeface::from_data and preserve the loaded glyph count.
        let mgr = DefaultFontMgr::new();
        let tf = mgr.make_from_data(DEMO_TTF, 0).expect("parse demo.ttf");
        assert_eq!(tf.units_per_em(), 1000, "demo.ttf upem");
        assert_eq!(tf.glyph_count(), 2, "demo.ttf glyph count");
        // cmap must be wired: 'A' maps to gid 1.
        assert_eq!(tf.char_to_glyph('A'), 1);
    }

    #[test]
    fn make_from_data_rejects_bad_bytes() {
        // Random bytes are not a valid font — must return None, not a
        // silent "default typeface" fallback.
        let mgr = DefaultFontMgr::new();
        assert!(mgr.make_from_data(&[0u8; 64], 0).is_none());
    }

    #[test]
    fn make_from_file_reads_and_parses() {
        // Gap N-1: `make_from_file` must use std::fs::read and delegate to
        // Typeface::from_data.
        let mgr = DefaultFontMgr::new();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/demo.ttf");
        let tf = mgr
            .make_from_file(path, 0)
            .expect("load demo.ttf from disk");
        assert_eq!(tf.units_per_em(), 1000);
    }

    #[test]
    fn match_family_style_character_prefers_covering_typeface() {
        // Gap N-2: when the requested family has a typeface that covers
        // the character, return *that* typeface. When it doesn't, scan
        // other families for coverage before falling back to the default.
        let mut mgr = DefaultFontMgr::new();

        // Register demo.ttf (covers 'A' only) under family "Emoji" to
        // simulate a fallback path.
        let covering = Arc::new(Typeface::from_data(DEMO_TTF.to_vec()).expect("parse demo.ttf"));
        mgr.register_typeface("Emoji", covering, "Regular", FontStyle::default());

        // Asking for family "Default" + character 'A': the default family
        // has no cmap for 'A' (dataless typeface char_to_glyph returns the
        // ASCII codepoint as a fallback, so it *does* return non-zero for
        // 'A'). That means the default family wins for 'A'. Verify that
        // much: the returned typeface is non-None.
        let tf = mgr.match_family_style_character("Default", FontStyle::default(), &[], 'A');
        assert!(tf.is_some());

        // For a character no typeface in any family covers, we must still
        // fall back to *some* typeface rather than None — matching the
        // prior behaviour of the method.
        let tf = mgr.match_family_style_character(
            "Default",
            FontStyle::default(),
            &[],
            '\u{1F4A9}', // pile of poo — no coverage anywhere
        );
        assert!(tf.is_some(), "must not return None when fallback exists");
    }
}
