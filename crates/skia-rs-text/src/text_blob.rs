//! `TextBlob` for positioned text runs.
//!
//! A `TextBlob` contains a sequence of positioned glyphs that can be drawn efficiently.

use crate::font::Font;
use skia_rs_core::{Point, Rect, Scalar};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Monotonic source of `TextBlob` unique ids, mirroring Skia's
/// `SkTextBlob::fUniqueID` (a process-wide atomic counter). Ids start at 1
/// so 0 can stand for "no id".
static NEXT_BLOB_ID: AtomicU32 = AtomicU32::new(1);

#[inline]
fn next_blob_id() -> u32 {
    NEXT_BLOB_ID.fetch_add(1, Ordering::Relaxed)
}

/// A run of glyphs with the same font.
#[derive(Debug, Clone)]
pub struct GlyphRun {
    /// Glyph IDs.
    pub glyphs: Vec<u16>,
    /// Glyph positions (relative to run origin).
    pub positions: Vec<Point>,
    /// The font for this run.
    pub font: Font,
    /// Origin of the run.
    pub origin: Point,
}

impl GlyphRun {
    /// Create a new glyph run.
    #[must_use]
    pub const fn new(font: Font, glyphs: Vec<u16>, positions: Vec<Point>, origin: Point) -> Self {
        Self {
            glyphs,
            positions,
            font,
            origin,
        }
    }

    /// Get the bounds of this run.
    ///
    /// The vertical extent uses the font's conservative bbox-based
    /// `top`/`bottom` metrics (Skia's `fTop`/`fBottom`, derived from the
    /// font's global bounding box), not the tighter ascent/descent — this
    /// matches `SkTextBlob`'s run bounds, which must contain every glyph's
    /// ink. The horizontal extent spans every glyph position with the right
    /// edge extended by the actual per-glyph advance from `hmtx`.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        if self.positions.is_empty() {
            return Rect::EMPTY;
        }

        let metrics = self.font.metrics();
        let top = self.origin.y + metrics.top;
        let bottom = self.origin.y + metrics.bottom;

        let mut left = self.origin.x + self.positions[0].x;
        let mut right = left;

        for (i, pos) in self.positions.iter().enumerate() {
            let x = self.origin.x + pos.x;
            left = left.min(x);
            let advance = self
                .glyphs
                .get(i)
                .map_or_else(|| self.font.size() * 0.5, |&g| self.font.glyph_advance(g));
            right = right.max(x + advance);
        }

        Rect::new(left, top, right, bottom)
    }
}

/// A positioned collection of text runs.
///
/// Corresponds to Skia's `SkTextBlob`.
#[derive(Debug, Clone)]
pub struct TextBlob {
    /// The glyph runs.
    runs: Vec<GlyphRun>,
    /// Cached bounds.
    bounds: Rect,
    /// Process-wide unique id (monotonic), like `SkTextBlob::uniqueID()`.
    unique_id: u32,
}

impl TextBlob {
    /// Create a text blob from runs.
    #[must_use]
    pub fn from_runs(runs: Vec<GlyphRun>) -> Self {
        let bounds = if runs.is_empty() {
            Rect::EMPTY
        } else {
            let mut bounds = runs[0].bounds();
            for run in &runs[1..] {
                bounds = bounds.join(&run.bounds());
            }
            bounds
        };

        Self {
            runs,
            bounds,
            unique_id: next_blob_id(),
        }
    }

    /// Create a simple text blob from text and font.
    ///
    /// Glyphs are positioned by their real per-glyph advances
    /// (`Font::glyph_advance`) so the blob width agrees with
    /// `Font::measure_text` — not the old uniform `size * 0.5` estimate.
    #[must_use]
    pub fn from_text(text: &str, font: &Font, origin: Point) -> Self {
        let glyphs = font.text_to_glyphs(text);
        let mut positions = Vec::with_capacity(glyphs.len());

        let mut x = 0.0;
        for &glyph in &glyphs {
            positions.push(Point::new(x, 0.0));
            x += font.glyph_advance(glyph);
        }

        let run = GlyphRun::new(font.clone(), glyphs, positions, origin);
        Self::from_runs(vec![run])
    }

    /// Get the bounds of the text blob.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Get the glyph runs.
    #[inline]
    #[must_use]
    pub fn runs(&self) -> &[GlyphRun] {
        &self.runs
    }

    /// Get the unique ID.
    ///
    /// Assigned from a process-wide monotonic counter at construction time,
    /// mirroring Skia's `SkTextBlob::uniqueID()`. Stable across moves and
    /// clones-that-rebuild — unlike a pointer address, which is reused as
    /// blobs are freed and reallocated.
    #[inline]
    #[must_use]
    pub const fn unique_id(&self) -> u32 {
        self.unique_id
    }
}

/// A reference to a text blob.
pub type TextBlobRef = Arc<TextBlob>;

/// Builder for creating `TextBlobs`.
pub struct TextBlobBuilder {
    runs: Vec<GlyphRun>,
    current_font: Option<Font>,
    current_glyphs: Vec<u16>,
    current_positions: Vec<Point>,
    current_origin: Point,
}

impl Default for TextBlobBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TextBlobBuilder {
    /// Create a new text blob builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            runs: Vec::new(),
            current_font: None,
            current_glyphs: Vec::new(),
            current_positions: Vec::new(),
            current_origin: Point::zero(),
        }
    }

    /// Allocate a run with horizontal positioning.
    pub fn alloc_run(&mut self, font: &Font, count: usize, x: Scalar, y: Scalar) -> &mut Self {
        self.flush_run();
        self.current_font = Some(font.clone());
        self.current_origin = Point::new(x, y);
        self.current_glyphs.reserve(count);
        self.current_positions.reserve(count);
        self
    }

    /// Allocate a run with full positioning.
    pub fn alloc_run_pos(&mut self, font: &Font, count: usize) -> &mut Self {
        self.flush_run();
        self.current_font = Some(font.clone());
        self.current_origin = Point::zero();
        self.current_glyphs.reserve(count);
        self.current_positions.reserve(count);
        self
    }

    /// Add a glyph with position.
    pub fn add_glyph(&mut self, glyph: u16, pos: Point) -> &mut Self {
        self.current_glyphs.push(glyph);
        self.current_positions.push(pos);
        self
    }

    /// Add text with automatic positioning.
    pub fn add_text(&mut self, text: &str, font: &Font, origin: Point) -> &mut Self {
        self.flush_run();

        let glyphs = font.text_to_glyphs(text);
        let mut positions = Vec::with_capacity(glyphs.len());

        let mut x = 0.0;
        for &glyph in &glyphs {
            positions.push(Point::new(x, 0.0));
            x += font.glyph_advance(glyph);
        }

        let run = GlyphRun::new(font.clone(), glyphs, positions, origin);
        self.runs.push(run);
        self
    }

    /// Add a pre-positioned run of glyphs.
    ///
    /// The positions should be absolute (already include the origin offset).
    pub fn add_positioned_run(
        &mut self,
        font: &Font,
        glyphs: &[u16],
        positions: &[Point],
    ) -> &mut Self {
        self.flush_run();

        if glyphs.is_empty() || glyphs.len() != positions.len() {
            return self;
        }

        let run = GlyphRun::new(
            font.clone(),
            glyphs.to_vec(),
            positions.to_vec(),
            Point::zero(),
        );
        self.runs.push(run);
        self
    }

    /// Flush the current run.
    fn flush_run(&mut self) {
        if let Some(font) = self.current_font.take()
            && !self.current_glyphs.is_empty()
        {
            let run = GlyphRun::new(
                font,
                std::mem::take(&mut self.current_glyphs),
                std::mem::take(&mut self.current_positions),
                self.current_origin,
            );
            self.runs.push(run);
        }
        self.current_glyphs.clear();
        self.current_positions.clear();
    }

    /// Build the text blob.
    #[must_use]
    pub fn build(mut self) -> Option<TextBlob> {
        self.flush_run();

        if self.runs.is_empty() {
            return None;
        }

        Some(TextBlob::from_runs(self.runs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_blob_from_text() {
        let font = Font::from_size(16.0);
        let blob = TextBlob::from_text("Hello", &font, Point::new(0.0, 20.0));

        assert_eq!(blob.runs().len(), 1);
        assert_eq!(blob.runs()[0].glyphs.len(), 5);
    }

    #[test]
    fn test_text_blob_builder() {
        let font = Font::from_size(12.0);
        let mut builder = TextBlobBuilder::new();
        builder.add_text("Hello ", &font, Point::new(0.0, 12.0));
        builder.add_text("World", &font, Point::new(0.0, 24.0));
        let blob = builder.build().unwrap();

        assert_eq!(blob.runs().len(), 2);
    }

    #[test]
    fn test_glyph_run_bounds() {
        let font = Font::from_size(16.0);
        let glyphs = vec![65, 66, 67]; // ABC
        let positions = vec![
            Point::new(0.0, 0.0),
            Point::new(8.0, 0.0),
            Point::new(16.0, 0.0),
        ];
        let run = GlyphRun::new(font, glyphs, positions, Point::new(10.0, 20.0));

        let bounds = run.bounds();
        assert!(bounds.width() > 0.0);
        assert!(bounds.height() > 0.0);
    }
}
