//! PDF canvas for drawing.

use crate::document::PdfError;
use crate::font::{PdfFontManager, PdfFontType, StandardFont};
use crate::image::PdfImageManager;
use crate::transparency::{PdfBlendMode, TransparencyManager};
use skia_rs_core::{Color, Matrix, Point, Rect, Scalar};
use skia_rs_paint::{BlendMode, Paint, Style};
use skia_rs_path::{Path, PathElement};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// A canvas that generates PDF content streams.
///
/// The canvas holds `&mut` references to the document-level
/// [`PdfFontManager`], [`PdfImageManager`] and [`TransparencyManager`].
/// Every time the canvas emits a `Tf`, `Do` or `gs` operator it registers
/// the corresponding resource with the manager and remembers its index in
/// one of the `used_*` sets so that `PdfDocument::write_to` can build a
/// correct per-page `/Resources` dictionary.
pub struct PdfCanvas<'a> {
    /// Page width.
    width: Scalar,
    /// Page height.
    height: Scalar,
    /// Object ID (reserved at `begin_page`; filled in when the page is written).
    object_id: u32,
    /// Content stream.
    content: Vec<u8>,
    /// Graphics state stack.
    state_stack: Vec<GraphicsState>,
    /// Shared font manager.
    fonts: &'a mut PdfFontManager,
    /// Shared image manager.
    images: &'a mut PdfImageManager,
    /// Shared transparency manager.
    transparency: &'a mut TransparencyManager,
    /// Set of font indices referenced by this page.
    used_fonts: BTreeSet<usize>,
    /// Set of image indices referenced by this page.
    used_images: BTreeSet<usize>,
    /// Set of `ExtGState` indices referenced by this page.
    used_ext_gstates: BTreeSet<usize>,
    /// Currently selected font index (for `draw_text` without an explicit font).
    current_font: Option<usize>,
}

/// Graphics state.
#[derive(Clone)]
struct GraphicsState {
    /// Current transformation matrix.
    matrix: Matrix,
    /// Current color.
    color: Color,
    /// Line width.
    line_width: Scalar,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            matrix: Matrix::IDENTITY,
            color: Color::BLACK,
            line_width: 1.0,
        }
    }
}

/// Emitted content and per-page resources produced by a [`PdfCanvas`].
///
/// Returned by [`PdfCanvas::finish`]; consumed by
/// [`crate::PdfDocument::end_page`].
pub struct PageContent {
    /// Page width.
    pub width: Scalar,
    /// Page height.
    pub height: Scalar,
    /// Reserved object id for the page object.
    pub object_id: u32,
    /// Raw content-stream bytes.
    pub content: Vec<u8>,
    /// Font manager indices referenced from this page.
    pub used_fonts: Vec<usize>,
    /// Image manager indices referenced from this page.
    pub used_images: Vec<usize>,
    /// `ExtGState` manager indices referenced from this page.
    pub used_ext_gstates: Vec<usize>,
}

impl<'a> PdfCanvas<'a> {
    /// Create a new PDF canvas.
    ///
    /// Called by [`crate::PdfDocument::begin_page`]; external callers should
    /// prefer that entry point.
    pub fn new(
        width: Scalar,
        height: Scalar,
        object_id: u32,
        fonts: &'a mut PdfFontManager,
        images: &'a mut PdfImageManager,
        transparency: &'a mut TransparencyManager,
    ) -> Self {
        let mut canvas = Self {
            width,
            height,
            object_id,
            content: Vec::new(),
            state_stack: vec![GraphicsState::default()],
            fonts,
            images,
            transparency,
            used_fonts: BTreeSet::new(),
            used_images: BTreeSet::new(),
            used_ext_gstates: BTreeSet::new(),
            current_font: None,
        };

        // Set up coordinate system (PDF has origin at bottom-left)
        canvas.write_op(&format!("1 0 0 -1 0 {height} cm\n"));

        canvas
    }

    /// Get the width.
    #[must_use]
    pub const fn width(&self) -> Scalar {
        self.width
    }

    /// Get the height.
    #[must_use]
    pub const fn height(&self) -> Scalar {
        self.height
    }

    /// Get the object ID.
    #[must_use]
    pub const fn object_id(&self) -> u32 {
        self.object_id
    }

    /// Select a font for subsequent [`draw_text`](Self::draw_text) calls.
    ///
    /// `index` must be a valid index into the document's font manager.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range for the document's font manager.
    pub fn set_font(&mut self, index: usize) {
        assert!(
            self.fonts.get(index).is_some(),
            "font index {index} out of range"
        );
        self.used_fonts.insert(index);
        self.current_font = Some(index);
    }

    /// Register a standard font and select it as the current font.
    ///
    /// Returns the manager index — reuse it with [`set_font`](Self::set_font)
    /// or pass it to [`draw_text_with_font`](Self::draw_text_with_font).
    pub fn use_standard_font(&mut self, font: StandardFont) -> usize {
        let idx = self.fonts.register_standard(font);
        self.set_font(idx);
        idx
    }

    /// Finish this canvas, returning the page content and used-resource sets.
    #[must_use]
    pub fn finish(self) -> PageContent {
        PageContent {
            width: self.width,
            height: self.height,
            object_id: self.object_id,
            content: self.content,
            used_fonts: self.used_fonts.into_iter().collect(),
            used_images: self.used_images.into_iter().collect(),
            used_ext_gstates: self.used_ext_gstates.into_iter().collect(),
        }
    }

    /// Write a PDF operation.
    fn write_op(&mut self, op: &str) {
        self.content.extend_from_slice(op.as_bytes());
    }

    /// Get current state.
    fn state(&self) -> &GraphicsState {
        self.state_stack.last().unwrap()
    }

    /// Get mutable current state.
    fn state_mut(&mut self) -> &mut GraphicsState {
        self.state_stack.last_mut().unwrap()
    }

    /// Save graphics state.
    pub fn save(&mut self) {
        let state = self.state().clone();
        self.state_stack.push(state);
        self.write_op("q\n");
    }

    /// Restore graphics state.
    pub fn restore(&mut self) {
        if self.state_stack.len() > 1 {
            self.state_stack.pop();
            self.write_op("Q\n");
        }
    }

    /// Apply a transform.
    pub fn concat(&mut self, matrix: &Matrix) {
        self.state_mut().matrix = self.state().matrix.concat(matrix);
        self.write_op(&format!(
            "{} {} {} {} {} {} cm\n",
            matrix.values[0],
            matrix.values[3],
            matrix.values[1],
            matrix.values[4],
            matrix.values[2],
            matrix.values[5]
        ));
    }

    /// Translate.
    pub fn translate(&mut self, dx: Scalar, dy: Scalar) {
        self.concat(&Matrix::translate(dx, dy));
    }

    /// Scale.
    pub fn scale(&mut self, sx: Scalar, sy: Scalar) {
        self.concat(&Matrix::scale(sx, sy));
    }

    /// Rotate (degrees).
    pub fn rotate(&mut self, degrees: Scalar) {
        let radians = degrees.to_radians();
        self.concat(&Matrix::rotate(radians));
    }

    /// Set the fill color.
    pub fn set_fill_color(&mut self, color: Color) {
        let r = f32::from(color.red()) / 255.0;
        let g = f32::from(color.green()) / 255.0;
        let b = f32::from(color.blue()) / 255.0;
        self.state_mut().color = color;
        self.write_op(&format!("{r:.3} {g:.3} {b:.3} rg\n"));
    }

    /// Set the stroke color.
    pub fn set_stroke_color(&mut self, color: Color) {
        let r = f32::from(color.red()) / 255.0;
        let g = f32::from(color.green()) / 255.0;
        let b = f32::from(color.blue()) / 255.0;
        self.write_op(&format!("{r:.3} {g:.3} {b:.3} RG\n"));
    }

    /// Set the line width.
    pub fn set_line_width(&mut self, width: Scalar) {
        self.state_mut().line_width = width;
        self.write_op(&format!("{width} w\n"));
    }

    /// Draw a rectangle.
    pub fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
        self.apply_paint(paint);
        self.write_op(&format!(
            "{} {} {} {} re ",
            rect.left,
            rect.top,
            rect.width(),
            rect.height()
        ));
        // A single non-self-intersecting rectangle fills identically under
        // nonzero-winding and even-odd, so the plain operators are fine.
        self.stroke_or_fill(paint, false);
    }

    /// Draw a line.
    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        self.apply_paint(paint);
        self.write_op(&format!("{} {} m {} {} l S\n", p0.x, p0.y, p1.x, p1.y));
    }

    /// Draw a circle.
    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        // Approximate circle with bezier curves
        let k = radius * 0.552_284_8; // Magic constant for circle approximation

        self.apply_paint(paint);
        self.write_op(&format!("{} {} m\n", center.x + radius, center.y));

        // Four quadrants
        self.write_op(&format!(
            "{} {} {} {} {} {} c\n",
            center.x + radius,
            center.y + k,
            center.x + k,
            center.y + radius,
            center.x,
            center.y + radius
        ));
        self.write_op(&format!(
            "{} {} {} {} {} {} c\n",
            center.x - k,
            center.y + radius,
            center.x - radius,
            center.y + k,
            center.x - radius,
            center.y
        ));
        self.write_op(&format!(
            "{} {} {} {} {} {} c\n",
            center.x - radius,
            center.y - k,
            center.x - k,
            center.y - radius,
            center.x,
            center.y - radius
        ));
        self.write_op(&format!(
            "{} {} {} {} {} {} c\n",
            center.x + k,
            center.y - radius,
            center.x + radius,
            center.y - k,
            center.x + radius,
            center.y
        ));

        // A circle approximation is a single non-self-intersecting loop.
        self.stroke_or_fill(paint, false);
    }

    /// Draw a path.
    pub fn draw_path(&mut self, path: &Path, paint: &Paint) {
        self.apply_paint(paint);

        let mut current = Point::zero();
        // Subpath start — used to correctly advance `current` on Close.
        let mut subpath_start = Point::zero();

        for element in path {
            match element {
                PathElement::Move(p) => {
                    self.write_op(&format!("{} {} m\n", p.x, p.y));
                    current = p;
                    subpath_start = p;
                }
                PathElement::Line(p) => {
                    self.write_op(&format!("{} {} l\n", p.x, p.y));
                    current = p;
                }
                PathElement::Quad(ctrl, end) => {
                    // Convert quadratic to cubic
                    let c1 = Point::new(
                        (2.0_f32 / 3.0).mul_add(ctrl.x - current.x, current.x),
                        (2.0_f32 / 3.0).mul_add(ctrl.y - current.y, current.y),
                    );
                    let c2 = Point::new(
                        (2.0_f32 / 3.0).mul_add(ctrl.x - end.x, end.x),
                        (2.0_f32 / 3.0).mul_add(ctrl.y - end.y, end.y),
                    );
                    self.write_op(&format!(
                        "{} {} {} {} {} {} c\n",
                        c1.x, c1.y, c2.x, c2.y, end.x, end.y
                    ));
                    current = end;
                }
                PathElement::Conic(ctrl, end, w) => {
                    // A rational quadratic cannot be expressed exactly as a
                    // cubic Bezier, so flatten to line segments using
                    // skia-rs-path's adaptive conic flattener. Using `l`
                    // ops keeps the result visually correct for any weight.
                    let mut points = Vec::new();
                    skia_rs_path::flatten::flatten_conic_adaptive(
                        &mut points,
                        current,
                        ctrl,
                        end,
                        w,
                        0.25, // 0.25 pt tolerance
                    );
                    for p in points {
                        self.write_op(&format!("{} {} l\n", p.x, p.y));
                    }
                    current = end;
                }
                PathElement::Cubic(c1, c2, end) => {
                    self.write_op(&format!(
                        "{} {} {} {} {} {} c\n",
                        c1.x, c1.y, c2.x, c2.y, end.x, end.y
                    ));
                    current = end;
                }
                PathElement::Close => {
                    self.write_op("h\n");
                    current = subpath_start;
                }
            }
        }

        // PDF 32000-1 table 51/60/92: `f*`/`B*`/`W*` select the even-odd
        // fill rule instead of the default nonzero-winding operators.
        let even_odd = matches!(
            path.fill_type(),
            skia_rs_path::FillType::EvenOdd | skia_rs_path::FillType::InverseEvenOdd
        );
        self.stroke_or_fill(paint, even_odd);
    }

    /// Draw text using the currently selected font (see [`set_font`](Self::set_font)).
    ///
    /// Returns `Err(PdfError::Unsupported)` if no font is selected; use
    /// [`draw_text_with_font`](Self::draw_text_with_font) to provide one explicitly.
    ///
    /// # Errors
    ///
    /// Returns `Err(PdfError::Unsupported)` if no font is currently
    /// selected, or if the selected font is a Type0/CID font (see
    /// [`draw_text_with_font`](Self::draw_text_with_font)).
    pub fn draw_text(
        &mut self,
        text: &str,
        x: Scalar,
        y: Scalar,
        font_size: Scalar,
        paint: &Paint,
    ) -> Result<(), PdfError> {
        let font_idx = self.current_font.ok_or_else(|| {
            PdfError::Unsupported(
                "draw_text requires a font: call set_font or use_standard_font first".to_string(),
            )
        })?;
        self.draw_text_with_font(text, x, y, font_size, font_idx, paint)
    }

    /// Draw text with an explicit font.
    ///
    /// Returns `Err(PdfError::Unsupported)` if `font_idx` names a Type0/CID
    /// font (registered via [`PdfFontManager::register_truetype_cid`]);
    /// live per-glyph CID text drawing is not yet implemented (see below).
    ///
    /// # Errors
    ///
    /// Returns `Err(PdfError::Unsupported)` if `font_idx` names a Type0/CID
    /// font.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, debug builds only) if `font_idx` is out
    /// of range — that reflects a caller bug (an index not obtained from
    /// this canvas's font manager), not a recoverable runtime condition.
    pub fn draw_text_with_font(
        &mut self,
        text: &str,
        x: Scalar,
        y: Scalar,
        font_size: Scalar,
        font_idx: usize,
        paint: &Paint,
    ) -> Result<(), PdfError> {
        debug_assert!(
            self.fonts.get(font_idx).is_some(),
            "font index {font_idx} out of range"
        );

        // Type0/CID fonts (see `PdfFontManager::register_truetype_cid`) are
        // emitted with `/Encoding /Identity-H`, which readers interpret as
        // a stream of 2-byte CIDs. This canvas does not yet resolve
        // per-glyph CIDs at draw time (that requires shaping text into
        // glyph ids and emitting 2-byte codes, plus a CID-keyed
        // ToUnicode CMap) — only the simple-font WinAnsi path below is
        // implemented. Silently falling through would emit 1-byte WinAnsi
        // codes against a 2-byte encoding, which readers decode as
        // garbage glyphs. Fail closed instead of producing a corrupt PDF;
        // the `/DescendantFonts` structure can still be registered and
        // emitted (see `PdfFontManager::register_truetype_cid` and
        // `PdfDocument::write_to`) even though live text can't yet be
        // drawn through it.
        if let Some(font) = self.fonts.get(font_idx)
            && matches!(
                font.font_type,
                PdfFontType::Type0 | PdfFontType::OpenTypeCff
            )
        {
            return Err(PdfError::Unsupported(format!(
                "draw_text_with_font: font {font_idx} is a Type0/CID font (registered via \
                     register_truetype_cid, encoded /Identity-H); live per-glyph CID text \
                     drawing is not yet implemented, so drawing through this font would \
                     silently emit invalid 1-byte codes against a 2-byte CID encoding. \
                     Register the text as a simple TrueType font \
                     (PdfFontManager::register_truetype) instead."
            )));
        }

        self.used_fonts.insert(font_idx);

        // Track the characters that were written so the font can produce an
        // accurate ToUnicode CMap at emit time.
        if let Some(font) = self.fonts.get_mut(font_idx) {
            for ch in text.chars() {
                font.record_char(ch);
            }
        }

        self.apply_paint(paint);
        self.write_op("BT\n");
        self.write_op(&format!("/F{} {} Tf\n", font_idx + 1, font_size));
        // The page CTM includes a `1 0 0 -1 0 height` y-flip (see `new`) so
        // that callers can draw in a top-left-origin, y-down space. Glyph
        // outlines in a Type1/TrueType font are defined upright in that
        // same flipped space, so without compensation they render mirrored
        // upside-down. Per upstream `SkPDFDevice::GlyphPositioner`
        // (`skia/src/pdf/SkPDFDevice.cpp`), the fix is to set the text
        // matrix to `1 0 skew -1 x y Tm` — the `-1` in `d` cancels the
        // outer y-flip. We don't support italic skew synthesis here, so
        // skew is always 0.
        self.write_op(&format!("1 0 0 -1 {x} {y} Tm\n"));

        // Emit the text literal. Simple (Type1/TrueType) fonts index into a
        // single-byte encoding (WinAnsiEncoding for everything but Symbol/
        // ZapfDingbats), so the content-stream string must contain
        // single-byte WinAnsi codes — never a UTF-16BE hex string, which is
        // only meaningful against a Type0/CID font's Encoding CMap. ASCII
        // text is already a WinAnsi subset. For non-ASCII text we map each
        // character to its WinAnsi byte where possible; characters outside
        // WinAnsi (true CJK/exotic Unicode) fall back to `?` since this
        // canvas does not yet support switching a run to a Type0 font.
        if text.is_ascii() {
            self.write_op(&format!("({}) Tj\n", escape_pdf_string(text)));
        } else {
            let bytes: Vec<u8> = text
                .chars()
                .map(|c| winansi_byte(c).unwrap_or(b'?'))
                .collect();
            self.write_literal_bytes(&escape_pdf_bytes(&bytes));
            self.write_op(" Tj\n");
        }
        self.write_op("ET\n");
        Ok(())
    }

    /// Write a PDF literal string `(...)` from already-escaped raw bytes.
    ///
    /// Used for WinAnsi-encoded text, where the bytes are not necessarily
    /// valid UTF-8 and so can't go through [`write_op`](Self::write_op)'s
    /// `&str` interface.
    fn write_literal_bytes(&mut self, escaped: &[u8]) {
        self.content.push(b'(');
        self.content.extend_from_slice(escaped);
        self.content.push(b')');
    }

    /// Draw a PDF image registered with the document's image manager.
    ///
    /// The image is painted at the given position at its natural size in
    /// user-space units (1 unit == 1/72 inch). Use [`concat`](Self::concat) or
    /// [`save`](Self::save)/[`restore`](Self::restore) to scale as needed.
    ///
    /// # Panics
    ///
    /// Panics if `image_idx` is out of range for the document's image
    /// manager.
    pub fn draw_image(&mut self, image_idx: usize, x: Scalar, y: Scalar, w: Scalar, h: Scalar) {
        assert!(
            self.images.get(image_idx).is_some(),
            "image index {image_idx} out of range"
        );
        self.used_images.insert(image_idx);

        // Image XObjects are painted onto the unit square (0,0)-(1,1) with
        // sample row 0 (the top of the raster) conventionally placed at
        // v=1. Composing that directly with a naive `w 0 0 h x y cm` places
        // row 0 at the *bottom* of the destination rect, flipping the
        // image vertically. Upstream SkPDFDevice::internalDrawImageRect
        // (`skia/src/pdf/SkPDFDevice.cpp`) counters this with
        // `setScale(1,-1); postTranslate(0,1)` before scaling to the
        // destination size and translating into place; folding that
        // through algebraically for a `(x, y, w, h)` top-left destination
        // rect yields `w 0 0 -h x (y+h) cm`.
        self.write_op("q\n");
        self.write_op(&format!("{} 0 0 {} {} {} cm\n", w, -h, x, y + h));
        self.write_op(&format!("/Im{} Do\n", image_idx + 1));
        self.write_op("Q\n");
    }

    /// Set the fill/stroke alpha for subsequent drawing via an `ExtGState`.
    ///
    /// `alpha` is clamped to `[0, 1]`. Emits a `/GS<n> gs` operator and
    /// registers the state with the document's transparency manager.
    pub fn set_alpha(&mut self, alpha: Scalar) {
        let a = alpha.clamp(0.0, 1.0);
        let idx = self.transparency.get_or_create_alpha_state(a);
        self.used_ext_gstates.insert(idx);
        self.write_op(&format!("/GS{} gs\n", idx + 1));
    }

    /// Set the blend mode for subsequent drawing via an `ExtGState`.
    pub fn set_blend_mode(&mut self, mode: PdfBlendMode) {
        let idx = self.transparency.get_or_create_blend_state(mode);
        self.used_ext_gstates.insert(idx);
        self.write_op(&format!("/GS{} gs\n", idx + 1));
    }

    /// Apply paint settings.
    fn apply_paint(&mut self, paint: &Paint) {
        let color = paint.color32();

        match paint.style() {
            Style::Fill => self.set_fill_color(color),
            Style::Stroke => {
                self.set_stroke_color(color);
                self.set_line_width(paint.stroke_width());
            }
            Style::StrokeAndFill => {
                self.set_fill_color(color);
                self.set_stroke_color(color);
                self.set_line_width(paint.stroke_width());
            }
        }

        // Derive an ExtGState when the paint has a non-opaque alpha or a
        // non-Normal blend mode. We intentionally read the alpha from the
        // raw color channel (not paint.alpha(), which is Color4f.a and
        // therefore already folded into the rg/RG operator) so that partial
        // alpha on a Color32 propagates into /ca /CA.
        let alpha8 = color.alpha();
        let alpha = Scalar::from(alpha8) / 255.0;
        let pdf_blend = PdfBlendMode::from_skia_blend_mode(paint.blend_mode());
        let needs_alpha_gs = alpha < 1.0;
        let needs_blend_gs =
            !matches!(pdf_blend, Some(PdfBlendMode::Normal)) && pdf_blend.is_some();

        if needs_alpha_gs || needs_blend_gs {
            let mut state = crate::transparency::ExtGraphicsState::default();
            if needs_alpha_gs {
                state.fill_alpha = Some(alpha);
                state.stroke_alpha = Some(alpha);
            }
            if needs_blend_gs {
                state.blend_mode = pdf_blend;
            }
            let idx = self.transparency.add_ext_gstate(state);
            self.used_ext_gstates.insert(idx);
            self.write_op(&format!("/GS{} gs\n", idx + 1));
        }

        // Ensure paint's own blend mode (if Normal) doesn't trip us up
        // further — nothing to do. Unused is fine.
        let _ = BlendMode::SrcOver;
    }

    /// Write stroke or fill operator.
    ///
    /// `even_odd` selects the `f*`/`B*` even-odd variants (PDF 32000-1
    /// §8.5.3) instead of the default nonzero-winding `f`/`B`. Stroke-only
    /// painting (`S`) has no fill rule and is unaffected.
    fn stroke_or_fill(&mut self, paint: &Paint, even_odd: bool) {
        match (paint.style(), even_odd) {
            (Style::Fill, false) => self.write_op("f\n"),
            (Style::Fill, true) => self.write_op("f*\n"),
            (Style::Stroke, _) => self.write_op("S\n"),
            (Style::StrokeAndFill, false) => self.write_op("B\n"),
            (Style::StrokeAndFill, true) => self.write_op("B*\n"),
        }
    }
}

/// Escape special characters in a PDF literal string.
///
/// Handles `(`, `)`, `\\`, and non-printable bytes (mapped to octal
/// escapes `\ddd`). Newlines and carriage returns get their dedicated
/// escapes `\n` / `\r` so editors don't rewrap them.
#[must_use]
pub fn escape_pdf_string(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '(' => result.push_str("\\("),
            ')' => result.push_str("\\)"),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\x08' => result.push_str("\\b"),
            '\x0C' => result.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) == 0x7F => {
                // Non-printable: octal escape for each UTF-8 byte.
                let mut buf = [0u8; 4];
                for &byte in c.encode_utf8(&mut buf).as_bytes() {
                    let _ = write!(result, "\\{byte:03o}");
                }
            }
            c => result.push(c),
        }
    }
    result
}

/// Map a Unicode scalar to its single-byte `WinAnsiEncoding` (PDF 32000-1
/// Annex D.2, effectively CP1252) code, if representable.
///
/// Codes `0x20..=0x7E` match ASCII directly; `0xA0..=0xFF` match their
/// Unicode code point (Latin-1 supplement); `0x80..=0x9F` are a fixed
/// CP1252-specific remapping of the C1 control block, spelled out below.
/// Anything else (undefined `WinAnsi` slots, or codepoints outside Latin-1)
/// returns `None`.
#[must_use]
pub fn winansi_byte(c: char) -> Option<u8> {
    match c {
        '\u{20}'..='\u{7E}' => Some(c as u8),
        '\u{20AC}' => Some(0x80), // EURO SIGN
        '\u{201A}' => Some(0x82), // SINGLE LOW-9 QUOTATION MARK
        '\u{0192}' => Some(0x83), // LATIN SMALL LETTER F WITH HOOK
        '\u{201E}' => Some(0x84), // DOUBLE LOW-9 QUOTATION MARK
        '\u{2026}' => Some(0x85), // HORIZONTAL ELLIPSIS
        '\u{2020}' => Some(0x86), // DAGGER
        '\u{2021}' => Some(0x87), // DOUBLE DAGGER
        '\u{02C6}' => Some(0x88), // MODIFIER LETTER CIRCUMFLEX ACCENT
        '\u{2030}' => Some(0x89), // PER MILLE SIGN
        '\u{0160}' => Some(0x8A), // LATIN CAPITAL LETTER S WITH CARON
        '\u{2039}' => Some(0x8B), // SINGLE LEFT-POINTING ANGLE QUOTATION MARK
        '\u{0152}' => Some(0x8C), // LATIN CAPITAL LIGATURE OE
        '\u{017D}' => Some(0x8E), // LATIN CAPITAL LETTER Z WITH CARON
        '\u{2018}' => Some(0x91), // LEFT SINGLE QUOTATION MARK
        '\u{2019}' => Some(0x92), // RIGHT SINGLE QUOTATION MARK
        '\u{201C}' => Some(0x93), // LEFT DOUBLE QUOTATION MARK
        '\u{201D}' => Some(0x94), // RIGHT DOUBLE QUOTATION MARK
        '\u{2022}' => Some(0x95), // BULLET
        '\u{2013}' => Some(0x96), // EN DASH
        '\u{2014}' => Some(0x97), // EM DASH
        '\u{02DC}' => Some(0x98), // SMALL TILDE
        '\u{2122}' => Some(0x99), // TRADE MARK SIGN
        '\u{0161}' => Some(0x9A), // LATIN SMALL LETTER S WITH CARON
        '\u{203A}' => Some(0x9B), // SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
        '\u{0153}' => Some(0x9C), // LATIN SMALL LIGATURE OE
        '\u{017E}' => Some(0x9E), // LATIN SMALL LETTER Z WITH CARON
        '\u{0178}' => Some(0x9F), // LATIN CAPITAL LETTER Y WITH DIAERESIS
        '\u{A0}'..='\u{FF}' => u8::try_from(u32::from(c)).ok(),
        _ => None,
    }
}

/// Escape special bytes in a PDF literal string given raw (already
/// single-byte-encoded, e.g. `WinAnsi`) bytes.
///
/// Byte-oriented counterpart to [`escape_pdf_string`] for text that is not
/// valid UTF-8 on its own (`WinAnsi` bytes `0x80..=0xFF` don't roundtrip
/// through `char`).
#[must_use]
pub fn escape_pdf_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            b'(' => out.extend_from_slice(b"\\("),
            b')' => out.extend_from_slice(b"\\)"),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0C => out.extend_from_slice(b"\\f"),
            b if b < 0x20 || b == 0x7F => {
                out.extend_from_slice(format!("\\{b:03o}").as_bytes());
            }
            b => out.push(b),
        }
    }
    out
}

/// Encode a UTF-8 string as an uppercase hex UTF-16BE literal with a BOM,
/// for use inside a PDF hex-string `<FEFF...>`.
#[must_use]
pub fn utf16be_hex(s: &str) -> String {
    let mut out = String::with_capacity(4 + s.len() * 4);
    // UTF-16BE BOM
    out.push_str("FEFF");
    for unit in s.encode_utf16() {
        let _ = write!(out, "{unit:04X}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::PdfFontManager;
    use crate::image::PdfImageManager;
    use crate::transparency::TransparencyManager;

    fn mk_canvas<'a>(
        fonts: &'a mut PdfFontManager,
        images: &'a mut PdfImageManager,
        transparency: &'a mut TransparencyManager,
    ) -> PdfCanvas<'a> {
        PdfCanvas::new(612.0, 792.0, 1, fonts, images, transparency)
    }

    #[test]
    fn test_pdf_canvas_rect() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();
        let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);

        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 0, 0));

        canvas.draw_rect(&Rect::from_xywh(100.0, 100.0, 200.0, 150.0), &paint);

        let page = canvas.finish();
        let content = String::from_utf8(page.content).unwrap();
        assert!(content.contains("re"));
        assert!(content.contains('f')); // Fill operator
    }

    #[test]
    fn test_pdf_canvas_save_restore() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();
        let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);

        canvas.save();
        canvas.translate(100.0, 100.0);
        canvas.restore();

        let page = canvas.finish();
        let content = String::from_utf8(page.content).unwrap();
        assert!(content.contains('q')); // Save
        assert!(content.contains('Q')); // Restore
    }

    #[test]
    fn test_canvas_draw_text_registers_font() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();

        let page = {
            let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);
            let f = canvas.use_standard_font(StandardFont::Helvetica);
            assert_eq!(f, 0);

            let paint = Paint::new();
            canvas
                .draw_text("Hello", 72.0, 72.0, 12.0, &paint)
                .expect("draw_text should succeed");

            canvas.finish()
        };

        assert_eq!(page.used_fonts, vec![0]);
        let content = String::from_utf8(page.content).unwrap();
        assert!(content.contains("/F1 12 Tf"));
        // The text matrix must counter the page's outer y-flip CTM, per
        // upstream SkPDFDevice::GlyphPositioner, or glyphs render mirrored.
        assert!(
            content.contains("1 0 0 -1 72 72 Tm"),
            "missing compensating text matrix: {content}"
        );
        assert!(content.contains("(Hello) Tj"));
    }

    #[test]
    fn test_canvas_draw_text_non_ascii_uses_winansi_not_utf16be() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();

        let page = {
            let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);
            canvas.use_standard_font(StandardFont::Helvetica);

            let paint = Paint::new();
            canvas
                .draw_text("héllo", 72.0, 72.0, 12.0, &paint)
                .expect("draw_text should succeed");
            canvas.finish()
        };

        // A simple (Type1) font indexes a single-byte encoding, so 'é' must
        // appear as its WinAnsi byte 0xE9 inside a literal string — never
        // as a UTF-16BE hex string, which only makes sense against a
        // Type0/CID font's CMap encoding.
        let expected: &[u8] = b"(h\xE9llo) Tj";
        assert!(
            page.content.windows(expected.len()).any(|w| w == expected),
            "content missing WinAnsi-encoded literal: {:?}",
            String::from_utf8_lossy(&page.content)
        );
        assert!(
            !page.content.windows(5).any(|w| w == b"<FEFF"),
            "must never emit a UTF-16BE hex string against a simple font"
        );
    }

    #[test]
    fn test_canvas_draw_text_unrepresentable_char_falls_back_without_hex() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();

        let page = {
            let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);
            canvas.use_standard_font(StandardFont::Helvetica);
            let paint = Paint::new();
            // '中' is outside WinAnsiEncoding entirely.
            canvas
                .draw_text("a中b", 72.0, 72.0, 12.0, &paint)
                .expect("draw_text should succeed");
            canvas.finish()
        };

        assert!(
            !page.content.windows(5).any(|w| w == b"<FEFF"),
            "must never emit a UTF-16BE hex string against a simple font"
        );
        assert!(
            page.content.windows(7).any(|w| w == b"(a?b) T"),
            "expected literal fallback: {:?}",
            String::from_utf8_lossy(&page.content)
        );
    }

    #[test]
    fn test_canvas_draw_image_records_use() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();

        // Seed an image.
        let idx = images.add_rgb(4, 4, &[0u8; 4 * 4 * 3]);

        let page = {
            let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);
            canvas.draw_image(idx, 10.0, 20.0, 30.0, 40.0);
            canvas.finish()
        };

        assert_eq!(page.used_images, vec![0]);
        let content = String::from_utf8(page.content).unwrap();
        assert!(content.contains("/Im1 Do"));
        // Counter-flip: `w 0 0 -h x (y+h) cm` per SkPDFDevice's
        // setScale(1,-1)/postTranslate(0,1) unit-square correction, so the
        // raster's top row lands at the top of the destination rect.
        assert!(
            content.contains("30 0 0 -40 10 60 cm"),
            "missing counter-flip cm for image placement: {content}"
        );
    }

    #[test]
    fn test_canvas_draw_path_even_odd_uses_star_operator() {
        use skia_rs_path::{FillType, PathBuilder};

        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();

        let mut builder = PathBuilder::with_fill_type(FillType::EvenOdd);
        builder.move_to(0.0, 0.0);
        builder.line_to(10.0, 0.0);
        builder.line_to(10.0, 10.0);
        builder.close();
        let path = builder.build();

        let page = {
            let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);
            let paint = Paint::new();
            canvas.draw_path(&path, &paint);
            canvas.finish()
        };

        let content = String::from_utf8(page.content).unwrap();
        assert!(
            content.contains("f*\n"),
            "even-odd fill must use f*: {content}"
        );
    }

    #[test]
    fn test_canvas_alpha_paint_emits_gs() {
        let mut fonts = PdfFontManager::new();
        let mut images = PdfImageManager::new();
        let mut transparency = TransparencyManager::new();

        let page = {
            let mut canvas = mk_canvas(&mut fonts, &mut images, &mut transparency);
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(128, 255, 0, 0));
            canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0), &paint);
            canvas.finish()
        };

        assert_ne!(page.used_ext_gstates, [] as [usize; 0]);
        let content = String::from_utf8(page.content).unwrap();
        assert!(
            content.contains("/GS1 gs"),
            "content missing ExtGState reference: {content}"
        );
    }

    #[test]
    fn test_escape_pdf_string_controls() {
        assert_eq!(escape_pdf_string("a\nb"), "a\\nb");
        assert_eq!(escape_pdf_string("a\rb"), "a\\rb");
        assert_eq!(escape_pdf_string("a\tb"), "a\\tb");
        assert_eq!(escape_pdf_string("(hi)"), "\\(hi\\)");
        assert_eq!(escape_pdf_string("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn test_utf16be_hex() {
        assert_eq!(utf16be_hex("A"), "FEFF0041");
        assert_eq!(utf16be_hex("é"), "FEFF00E9");
    }
}
