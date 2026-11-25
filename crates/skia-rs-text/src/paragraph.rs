//! Rich text paragraph layout.
//!
//! This module provides Skia Paragraph-style text layout for:
//! - Multi-line text layout
//! - Mixed styles within a paragraph
//! - Line breaking and word wrapping
//! - Hyphenation support
//! - Text alignment and justification

use crate::font::Font;
use crate::shaper::{ShapedGlyph, Shaper, TextDirection};
use crate::text_blob::{TextBlob, TextBlobBuilder};
use skia_rs_core::{Point, Rect, Scalar};

/// Text alignment within a paragraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TextAlign {
    /// Left-aligned text.
    #[default]
    Left = 0,
    /// Right-aligned text.
    Right,
    /// Center-aligned text.
    Center,
    /// Justified text.
    Justify,
    /// Start-aligned (based on text direction).
    Start,
    /// End-aligned (based on text direction).
    End,
}

/// Paragraph style settings.
#[derive(Debug, Clone)]
pub struct ParagraphStyle {
    /// Text alignment.
    pub text_align: TextAlign,
    /// Text direction.
    pub text_direction: TextDirection,
    /// Maximum number of lines (0 = unlimited).
    pub max_lines: usize,
    /// Ellipsis string for truncated text.
    pub ellipsis: Option<String>,
    /// Line height multiplier.
    pub height: Scalar,
    /// Whether to use height strictly.
    pub height_override: bool,
    /// Strutting (minimum line height from font metrics).
    pub strut_enabled: bool,
}

impl Default for ParagraphStyle {
    fn default() -> Self {
        Self {
            text_align: TextAlign::Left,
            text_direction: TextDirection::Ltr,
            max_lines: 0,
            ellipsis: None,
            height: 1.0,
            height_override: false,
            strut_enabled: false,
        }
    }
}

/// Text style for a span of text.
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Font to use.
    pub font: Font,
    /// Foreground color (ARGB).
    pub color: u32,
    /// Background color (ARGB), or 0 for transparent.
    pub background_color: u32,
    /// Decoration (underline, strikethrough, etc.).
    pub decoration: TextDecoration,
    /// Letter spacing.
    pub letter_spacing: Scalar,
    /// Word spacing.
    pub word_spacing: Scalar,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: Font::default(),
            color: 0xFF00_0000, // Black
            background_color: 0,
            decoration: TextDecoration::default(),
            letter_spacing: 0.0,
            word_spacing: 0.0,
        }
    }
}

/// Text decoration settings.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextDecoration {
    /// Underline flag.
    pub underline: bool,
    /// Overline flag.
    pub overline: bool,
    /// Line-through (strikethrough) flag.
    pub line_through: bool,
    /// Decoration color (ARGB), or 0 for text color.
    pub color: u32,
    /// Decoration style.
    pub style: DecorationStyle,
    /// Decoration thickness multiplier.
    pub thickness: Scalar,
}

/// Decoration line style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DecorationStyle {
    /// Solid line.
    #[default]
    Solid = 0,
    /// Double line.
    Double,
    /// Dotted line.
    Dotted,
    /// Dashed line.
    Dashed,
    /// Wavy line.
    Wavy,
}

/// A builder for creating paragraphs.
pub struct ParagraphBuilder {
    style: ParagraphStyle,
    runs: Vec<TextRun>,
    /// Style stack. `push_style` pushes; `pop` removes the top, restoring
    /// the previous style. When empty the default text style applies.
    style_stack: Vec<TextStyle>,
}

/// A run of text with a single style.
#[derive(Debug, Clone)]
struct TextRun {
    text: String,
    style: TextStyle,
}

impl ParagraphBuilder {
    /// Create a new paragraph builder with the given style.
    #[must_use]
    pub const fn new(style: ParagraphStyle) -> Self {
        Self {
            style,
            runs: Vec::new(),
            style_stack: Vec::new(),
        }
    }

    /// The style currently in effect: the top of the stack, or the default
    /// text style when the stack is empty.
    fn current_style(&self) -> TextStyle {
        self.style_stack.last().cloned().unwrap_or_default()
    }

    /// Push a style onto the style stack. It becomes the current style
    /// until a matching [`Self::pop`].
    pub fn push_style(&mut self, style: &TextStyle) -> &mut Self {
        self.style_stack.push(style.clone());
        self
    }

    /// Pop the top style off the stack, restoring the previously pushed
    /// style (or the default style when the stack becomes empty). A pop on
    /// an empty stack is a no-op, matching Skia's `ParagraphBuilder::Pop`.
    pub fn pop(&mut self) -> &mut Self {
        self.style_stack.pop();
        self
    }

    /// Add text with the current style.
    pub fn add_text(&mut self, text: &str) -> &mut Self {
        if !text.is_empty() {
            self.runs.push(TextRun {
                text: text.to_string(),
                style: self.current_style(),
            });
        }
        self
    }

    /// Build the paragraph.
    #[must_use]
    pub fn build(self) -> Paragraph {
        Paragraph {
            style: self.style,
            runs: self.runs,
            lines: Vec::new(),
            width: 0.0,
            height: 0.0,
            laid_out: false,
        }
    }
}

/// A laid-out paragraph of text.
pub struct Paragraph {
    style: ParagraphStyle,
    runs: Vec<TextRun>,
    lines: Vec<TextLine>,
    width: Scalar,
    height: Scalar,
    laid_out: bool,
}

/// A contiguous run of glyphs on a line that share a single `TextStyle`.
///
/// Each fragment carries its own font, color, background, and decoration so
/// that `to_text_blob` (and decoration rendering) can emit the correct
/// per-span styling. Positions are absolute x coordinates from the left
/// edge of the paragraph box, with y measured relative to the line's
/// baseline (y=0 at baseline).
#[derive(Debug, Clone)]
pub struct LineFragment {
    /// Shaped glyph IDs.
    pub glyphs: Vec<u16>,
    /// Absolute x offsets from the paragraph's left edge (before baseline
    /// y is applied).
    pub positions: Vec<Point>,
    /// The text style for this fragment (font, colors, decoration).
    pub style: TextStyle,
    /// Total advance of this fragment.
    pub width: Scalar,
}

/// A line of text in a paragraph.
#[derive(Debug, Clone)]
pub struct TextLine {
    /// Per-style fragments that make up the line (in reading order).
    pub fragments: Vec<LineFragment>,
    /// Baseline Y position relative to the paragraph origin.
    pub baseline: Scalar,
    /// Line bounds (y range covers the whole line box).
    pub bounds: Rect,
    /// Maximum ascent (negative) used by any fragment on this line.
    pub ascent: Scalar,
    /// Maximum descent (positive) used by any fragment on this line.
    pub descent: Scalar,
    /// Total content width (sum of fragment widths before alignment).
    pub width: Scalar,
}

/// A shaped cluster bundled with its style — the atomic unit fed to the
/// line-break loop. A "cluster" is whatever contiguous group of shaped
/// glyphs rustybuzz reports as belonging to a single input codepoint
/// cluster; breaking inside a cluster would corrupt grapheme clusters
/// (ligatures, combining marks, etc.).
struct Cluster {
    /// Shaped glyphs in this cluster.
    glyphs: Vec<ShapedGlyph>,
    /// Total advance (sum of `x_advance` + `letter_spacing` adjustments).
    advance: Scalar,
    /// Style applied to this cluster.
    style: TextStyle,
    /// Whether this cluster is a whitespace (or hard line break) cluster.
    kind: ClusterKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClusterKind {
    /// Visible glyph cluster (letters, punctuation, etc.).
    Glyph,
    /// One or more space characters — acts as a break opportunity.
    Space,
    /// Hard newline — forces a line break; advance is 0.
    Newline,
}

impl Paragraph {
    /// Layout the paragraph to fit within the given width.
    ///
    /// Uses rustybuzz shaping via [`Shaper`] for real OpenType advances,
    /// ligatures, kerning, RTL reordering, and so on. Each input
    /// `TextRun` is shaped independently, then broken into per-cluster
    /// units that participate in word-wrap. The resulting lines preserve
    /// the original per-span [`TextStyle`] so that [`Self::to_text_blob`]
    /// can emit one glyph run per style (preserving font, color, and
    /// decoration per span).
    ///
    /// Word-wrap happens at whitespace cluster boundaries; if a single
    /// word exceeds `width` it is still placed on its own line (no
    /// mid-word breaking, matching the previous behaviour). Hard newlines
    /// (`\n`) always force a line break.
    pub fn layout(&mut self, width: Scalar) {
        self.width = width;
        self.lines.clear();
        self.height = 0.0;

        let shaper = Shaper::new();
        let mut clusters: Vec<Cluster> = Vec::new();

        // Shape every run, flattening into a stream of styled clusters.
        for run in &self.runs {
            Self::shape_run_into_clusters(&shaper, run, &mut clusters);
        }

        // Pack clusters into lines respecting max width and max_lines.
        self.pack_clusters_into_lines(clusters, width);

        // Compute final height from the last line's extent.
        self.height = self.lines.last().map_or(0.0, |l| l.bounds.bottom);
        self.laid_out = true;
    }

    /// Shape a single `TextRun` and append per-cluster atoms to `out`.
    ///
    /// We split on whitespace/newline codepoints *before* shaping so that
    /// each cluster's break-kind is known without re-inspecting the
    /// post-shaping glyph stream (which no longer has a 1:1 mapping to
    /// codepoints for ligated scripts).
    fn shape_run_into_clusters(shaper: &Shaper, run: &TextRun, out: &mut Vec<Cluster>) {
        // Segment the run's text at newlines and runs of whitespace so
        // each segment becomes one cluster. Within visible segments we
        // still shape the whole thing so ligatures/kerning survive.
        let mut segment_start = 0usize;
        let bytes = run.text.as_bytes();
        let mut i = 0usize;
        let push_visible = |out: &mut Vec<Cluster>, seg: &str| {
            if seg.is_empty() {
                return;
            }
            let shape_result = shaper.shape_auto(seg, &run.style.font);
            let glyphs: Vec<ShapedGlyph> = shape_result.map_or_else(
                || fallback_shape(seg, &run.style.font),
                |runs| runs.into_iter().flat_map(|r| r.glyphs).collect(),
            );
            // Cluster advance includes letter_spacing between every glyph.
            // This is what the line-break loop compares against `width`,
            // so it has to agree with `build_line`'s pen-advance loop
            // (which adds x_advance + letter_spacing per glyph).
            let advance: Scalar = glyphs
                .iter()
                .map(|g| g.x_advance + run.style.letter_spacing)
                .sum();
            out.push(Cluster {
                glyphs,
                advance,
                style: run.style.clone(),
                kind: ClusterKind::Glyph,
            });
        };

        while i < bytes.len() {
            // Advance to next char boundary. We know text is utf-8.
            let ch = run.text[i..].chars().next().unwrap();
            let ch_len = ch.len_utf8();

            if ch == '\n' {
                // Flush preceding visible text, then emit a newline cluster.
                push_visible(out, &run.text[segment_start..i]);
                out.push(Cluster {
                    glyphs: Vec::new(),
                    advance: 0.0,
                    style: run.style.clone(),
                    kind: ClusterKind::Newline,
                });
                i += ch_len;
                segment_start = i;
                continue;
            }

            if ch.is_whitespace() {
                // Flush visible text, then consume a run of whitespace.
                push_visible(out, &run.text[segment_start..i]);
                let mut j = i;
                while j < bytes.len() {
                    let wsc = run.text[j..].chars().next().unwrap();
                    if wsc == '\n' || !wsc.is_whitespace() {
                        break;
                    }
                    j += wsc.len_utf8();
                }
                // Shape the whitespace to get real space advances.
                let ws_text = &run.text[i..j];
                let shape_result = shaper.shape_auto(ws_text, &run.style.font);
                let glyphs: Vec<ShapedGlyph> = shape_result.map_or_else(
                    || fallback_shape(ws_text, &run.style.font),
                    |runs| runs.into_iter().flat_map(|r| r.glyphs).collect(),
                );
                // Apply word_spacing once per whitespace cluster, matching
                // the previous implementation's per-space bump.
                let base: Scalar = glyphs.iter().map(|g| g.x_advance).sum();
                let advance = base + run.style.word_spacing;
                out.push(Cluster {
                    glyphs,
                    advance,
                    style: run.style.clone(),
                    kind: ClusterKind::Space,
                });
                i = j;
                segment_start = j;
                continue;
            }

            i += ch_len;
        }

        // Tail — any remaining visible text.
        push_visible(out, &run.text[segment_start..]);
    }

    #[allow(
        clippy::too_many_lines,
        reason = "line-breaking/packing state machine; splitting would fragment cohesive break-decision logic"
    )]
    fn pack_clusters_into_lines(&mut self, clusters: Vec<Cluster>, width: Scalar) {
        let mut pending: Vec<Cluster> = Vec::new();
        let mut pending_advance: Scalar = 0.0;
        let mut y: Scalar = 0.0;

        let max_lines = self.style.max_lines;
        let mut reached_max = false;

        let flush = |this: &mut Self,
                     line_clusters: &mut Vec<Cluster>,
                     content_width: &mut Scalar,
                     y: &mut Scalar,
                     forced_break: bool| {
            if line_clusters.is_empty() && !forced_break {
                return false;
            }
            let line = this.build_line(std::mem::take(line_clusters), *content_width, *y);
            let line_height = line.bounds.height();
            this.lines.push(line);
            *content_width = 0.0;
            *y += line_height;
            max_lines > 0 && this.lines.len() >= max_lines
        };

        for cluster in clusters {
            if reached_max {
                break;
            }

            match cluster.kind {
                ClusterKind::Newline => {
                    reached_max = flush(self, &mut pending, &mut pending_advance, &mut y, true);
                }
                ClusterKind::Space => {
                    // Try to add this space cluster to the current line. If
                    // even the space alone overflows, we emit the current
                    // line and drop the space (it would be trailing ws).
                    if pending_advance + cluster.advance > width && pending_advance > 0.0 {
                        // Wrap: current line is full. Drop the whitespace
                        // cluster (trailing on the wrapped line; never
                        // "starts" the next line — matches standard
                        // soft-wrap behaviour).
                        reached_max =
                            flush(self, &mut pending, &mut pending_advance, &mut y, false);
                        // The space itself is consumed.
                    } else {
                        pending_advance += cluster.advance;
                        pending.push(cluster);
                    }
                }
                ClusterKind::Glyph => {
                    if pending_advance + cluster.advance > width && pending_advance > 0.0 {
                        // Find the last break point — the last space cluster
                        // in `pending`. If there is one, break there; else
                        // this word is longer than the line so we ship the
                        // current line as-is and start fresh with this
                        // cluster (matches prior behaviour).
                        let last_space_ix = pending
                            .iter()
                            .rposition(|c| matches!(c.kind, ClusterKind::Space));
                        if let Some(ix) = last_space_ix {
                            // Split: everything up to `ix` stays on this line,
                            // the space at `ix` is dropped, everything after
                            // plus the new cluster goes to the next line.
                            let after = pending.split_off(ix + 1);
                            // Remove the space at `ix`.
                            pending.pop();
                            let line_width_before: Scalar = pending.iter().map(|c| c.advance).sum();
                            let mut before_adv = line_width_before;
                            reached_max = flush(self, &mut pending, &mut before_adv, &mut y, false);
                            // Now start the next line with `after` + cluster.
                            pending = after;
                            pending_advance = pending.iter().map(|c| c.advance).sum();
                            if !reached_max {
                                pending_advance += cluster.advance;
                                pending.push(cluster);
                            }
                        } else {
                            // No space to break at; ship the current line
                            // and start a new one with this cluster.
                            reached_max =
                                flush(self, &mut pending, &mut pending_advance, &mut y, false);
                            if !reached_max {
                                pending_advance = cluster.advance;
                                pending.push(cluster);
                            }
                        }
                    } else {
                        pending_advance += cluster.advance;
                        pending.push(cluster);
                    }
                }
            }
        }

        if !reached_max && !pending.is_empty() {
            // Flush final line.
            let line = self.build_line(pending, pending_advance, y);
            self.lines.push(line);
        }

        // Apply horizontal alignment in-place.
        let align = self.style.text_align;
        let box_width = self.width;
        let is_rtl = matches!(self.style.text_direction, TextDirection::Rtl);
        for line in &mut self.lines {
            let offset = match align {
                TextAlign::Right => box_width - line.width,
                TextAlign::Center => (box_width - line.width) / 2.0,
                // Justify not yet implemented; falls back to left-aligned.
                TextAlign::Left | TextAlign::Justify => 0.0,
                TextAlign::Start => {
                    if is_rtl {
                        box_width - line.width
                    } else {
                        0.0
                    }
                }
                TextAlign::End => {
                    if is_rtl {
                        0.0
                    } else {
                        box_width - line.width
                    }
                }
            };
            if offset != 0.0 {
                for frag in &mut line.fragments {
                    for p in &mut frag.positions {
                        p.x += offset;
                    }
                }
                line.bounds = line.bounds.offset(offset, 0.0);
            }
        }
    }

    fn build_line(&self, clusters: Vec<Cluster>, content_width: Scalar, y: Scalar) -> TextLine {
        // Derive per-line ascent/descent by taking the max over each
        // cluster's font metrics (mixed-style lines use the tallest
        // fragment's metrics as the line height).
        let mut ascent: Scalar = 0.0;
        let mut descent: Scalar = 0.0;
        let mut leading: Scalar = 0.0;
        for c in &clusters {
            let m = c.style.font.metrics();
            if m.ascent < ascent {
                ascent = m.ascent;
            }
            if m.descent > descent {
                descent = m.descent;
            }
            if m.leading > leading {
                leading = m.leading;
            }
        }
        if clusters.is_empty() {
            // Empty line (e.g. bare newline). Take its height from the
            // paragraph's own style (the first run's font) rather than an
            // unrelated `Font::default()`, so a blank line matches the
            // surrounding text's line height. Falls back to the default
            // font only when the paragraph has no runs at all.
            let m = self
                .runs
                .first()
                .map_or_else(|| Font::default().metrics(), |r| r.style.font.metrics());
            ascent = m.ascent;
            descent = m.descent;
            leading = m.leading;
        }

        let line_height = (-ascent + descent + leading) * self.style.height;
        let baseline = y - ascent;
        let bounds = Rect::from_xywh(0.0, y, self.width, line_height);

        // Merge adjacent clusters with the same style into single fragments
        // so to_text_blob produces one glyph run per style change rather
        // than one per cluster.
        let mut fragments: Vec<LineFragment> = Vec::new();
        let mut cursor_x: Scalar = 0.0;

        for cluster in clusters {
            // Merge into the last fragment when styles match.
            let can_merge = fragments
                .last()
                .is_some_and(|f| styles_eq(&f.style, &cluster.style));

            if !can_merge {
                fragments.push(LineFragment {
                    glyphs: Vec::new(),
                    positions: Vec::new(),
                    style: cluster.style.clone(),
                    width: 0.0,
                });
            }
            let last = fragments.last_mut().unwrap();

            let is_space = matches!(cluster.kind, ClusterKind::Space);
            let letter_spacing = if is_space {
                // Letter spacing is a per-character feature; typographers
                // don't apply it inside a whitespace cluster because the
                // glyph set is just the space glyph. Skipping it here
                // matches how word_spacing is applied separately below.
                0.0
            } else {
                cluster.style.letter_spacing
            };

            // Lay out each shaped glyph at the running cursor. Glyph
            // offsets (x_offset, y_offset) from shaping are applied to
            // the draw position but do not advance the pen. The cursor
            // advances by `x_advance + letter_spacing` per glyph.
            //
            // `y_offset` is already in Skia's y-down convention (the shaper
            // negates HarfBuzz's y-up value), so a raised glyph carries a
            // negative offset. It is stored relative to the baseline and
            // combined with `line.baseline` in `to_text_blob`, so add it
            // directly — do not negate again here.
            for sg in cluster.glyphs {
                last.glyphs.push(sg.glyph_id.0);
                last.positions
                    .push(Point::new(cursor_x + sg.x_offset, sg.y_offset));
                cursor_x += sg.x_advance + letter_spacing;
                last.width += sg.x_advance + letter_spacing;
            }

            // word_spacing is added after every whitespace cluster.
            if is_space {
                cursor_x += cluster.style.word_spacing;
                last.width += cluster.style.word_spacing;
            }
        }

        TextLine {
            fragments,
            baseline,
            bounds,
            ascent,
            descent,
            width: content_width,
        }
    }

    /// Get the laid-out width.
    #[must_use]
    pub fn max_intrinsic_width(&self) -> Scalar {
        self.lines.iter().map(|l| l.width).fold(0.0f32, f32::max)
    }

    /// Get the laid-out height.
    #[must_use]
    pub const fn height(&self) -> Scalar {
        self.height
    }

    /// Get the number of lines.
    #[must_use]
    pub const fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Get the line height for a specific line.
    #[must_use]
    pub fn line_height(&self, line: usize) -> Option<Scalar> {
        self.lines.get(line).map(|l| l.bounds.height())
    }

    /// Get the width of a specific line.
    #[must_use]
    pub fn line_width(&self, line: usize) -> Option<Scalar> {
        self.lines.get(line).map(|l| l.width)
    }

    /// Get access to the laid-out lines (for decoration rendering, hit
    /// testing, etc.).
    #[must_use]
    pub fn lines(&self) -> &[TextLine] {
        &self.lines
    }

    /// Convert the paragraph to a text blob for drawing.
    ///
    /// Emits one glyph run per (line, fragment) so that downstream rendering
    /// can honour per-span font, color, and decoration. Glyph positions are
    /// absolute in the paragraph coordinate space.
    #[must_use]
    pub fn to_text_blob(&self) -> Option<TextBlob> {
        if !self.laid_out || self.lines.is_empty() {
            return None;
        }

        let mut builder = TextBlobBuilder::new();

        for line in &self.lines {
            for frag in &line.fragments {
                if frag.glyphs.is_empty() {
                    continue;
                }
                let positions: Vec<Point> = frag
                    .positions
                    .iter()
                    .map(|p| Point::new(p.x, line.baseline + p.y))
                    .collect();
                builder.add_positioned_run(&frag.style.font, &frag.glyphs, &positions);
            }
        }

        builder.build()
    }

    /// Get the bounding box of the laid-out text.
    #[must_use]
    pub const fn bounds(&self) -> Rect {
        Rect::from_xywh(0.0, 0.0, self.width, self.height)
    }
}

/// Best-effort fallback when `Shaper::shape_auto` returns `None` (e.g. when
/// the font has no backing data — the dataless default typeface). Produces
/// one glyph per character using the typeface's cmap fallback and the
/// font's default glyph advance, so paragraphs built on the default
/// typeface still lay out visibly rather than collapsing to an empty line.
fn fallback_shape(text: &str, font: &Font) -> Vec<ShapedGlyph> {
    use crate::shaper::GlyphId;
    text.chars()
        .enumerate()
        .map(|(i, c)| {
            let gid = font.char_to_glyph(c);
            let adv = font.glyph_advance(gid);
            ShapedGlyph {
                glyph_id: GlyphId(gid),
                cluster: u32::try_from(i).unwrap_or(u32::MAX),
                x_advance: adv,
                y_advance: 0.0,
                x_offset: 0.0,
                y_offset: 0.0,
            }
        })
        .collect()
}

/// Compare two text styles by the fields that affect glyph run emission.
/// Fonts are compared by typeface identity and size/scale/skew so two
/// otherwise-identical styles merge into one run.
fn styles_eq(a: &TextStyle, b: &TextStyle) -> bool {
    let fa = &a.font;
    let fb = &b.font;
    let typefaces_match = std::sync::Arc::ptr_eq(fa.typeface_ref(), fb.typeface_ref());
    typefaces_match
        && fa.size() == fb.size()
        && fa.scale_x() == fb.scale_x()
        && fa.skew_x() == fb.skew_x()
        && a.color == b.color
        && a.background_color == b.background_color
        && decoration_eq(&a.decoration, &b.decoration)
}

fn decoration_eq(a: &TextDecoration, b: &TextDecoration) -> bool {
    a.underline == b.underline
        && a.overline == b.overline
        && a.line_through == b.line_through
        && a.color == b.color
        && a.style == b.style
        && a.thickness == b.thickness
}

// =============================================================================
// Line Breaking
// =============================================================================

/// Line breaker for finding valid break points in text.
pub struct LineBreaker {
    /// Break opportunities (byte offsets).
    breaks: Vec<usize>,
}

impl LineBreaker {
    /// Create a new line breaker for the given text.
    #[must_use]
    pub fn new(text: &str) -> Self {
        let mut breaks = Vec::new();
        let mut last_was_space = false;

        for (i, c) in text.char_indices() {
            // Simple line breaking: break after spaces and hyphens
            if last_was_space && !c.is_whitespace() {
                breaks.push(i);
            }
            if c == '-' {
                breaks.push(i + c.len_utf8());
            }
            last_was_space = c.is_whitespace();
        }

        // Always allow break at end
        breaks.push(text.len());

        Self { breaks }
    }

    /// Get all break opportunities.
    #[must_use]
    pub fn breaks(&self) -> &[usize] {
        &self.breaks
    }

    /// Find the best break point before a given position.
    #[must_use]
    pub fn find_break_before(&self, pos: usize) -> usize {
        self.breaks
            .iter()
            .filter(|&&b| b <= pos)
            .copied()
            .last()
            .unwrap_or(0)
    }
}

// =============================================================================
// Hyphenation
// =============================================================================

/// Simple hyphenation support.
pub struct Hyphenator {
    /// Minimum characters before hyphen.
    min_prefix: usize,
    /// Minimum characters after hyphen.
    min_suffix: usize,
}

impl Default for Hyphenator {
    fn default() -> Self {
        Self {
            min_prefix: 2,
            min_suffix: 3,
        }
    }
}

impl Hyphenator {
    /// Create a new hyphenator.
    #[must_use]
    pub const fn new(min_prefix: usize, min_suffix: usize) -> Self {
        Self {
            min_prefix,
            min_suffix,
        }
    }

    /// Find hyphenation points in a word.
    ///
    /// Returns byte offsets where hyphens can be inserted.
    #[must_use]
    pub fn hyphenate(&self, word: &str) -> Vec<usize> {
        let chars: Vec<char> = word.chars().collect();
        let char_count = chars.len();

        if char_count < self.min_prefix + self.min_suffix {
            return Vec::new();
        }

        let mut points = Vec::new();
        let mut byte_offset = 0;

        for (i, &c) in chars.iter().enumerate() {
            byte_offset += c.len_utf8();

            // Simple rule: allow hyphenation between vowels and consonants
            if i >= self.min_prefix
                && char_count - i > self.min_suffix
                && is_vowel(c) != is_vowel(chars.get(i + 1).copied().unwrap_or('x'))
            {
                points.push(byte_offset);
            }
        }

        points
    }
}

const fn is_vowel(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paragraph_builder() {
        let style = ParagraphStyle::default();
        let mut builder = ParagraphBuilder::new(style);
        builder.add_text("Hello, world!");
        let paragraph = builder.build();
        assert!(!paragraph.laid_out);
    }

    #[test]
    fn test_paragraph_layout() {
        let style = ParagraphStyle::default();
        let mut builder = ParagraphBuilder::new(style);
        builder.add_text("Hello, world! This is a test.");
        let mut paragraph = builder.build();
        paragraph.layout(100.0);

        assert!(paragraph.laid_out);
        assert!(paragraph.height() > 0.0);
    }

    #[test]
    fn test_line_breaker() {
        let breaker = LineBreaker::new("Hello world");
        assert_ne!(breaker.breaks(), []);
    }

    #[test]
    fn test_hyphenator() {
        let hyphenator = Hyphenator::default();
        let points = hyphenator.hyphenate("hyphenation");
        // Should find some hyphenation points in a long word
        assert!(!points.is_empty() || "hyphenation".len() < 5);
    }
}
