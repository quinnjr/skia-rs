//! PDF document structure.

use crate::canvas::{PageContent, PdfCanvas};
use crate::font::{PdfFont, PdfFontManager, PdfFontType};
use crate::image::PdfImageManager;
use crate::pdfa::{PdfADocument, PdfAError, PdfAFontInfo, PdfALevel, PdfAValidator, XmpMetadata};
use crate::transparency::TransparencyManager;
use skia_rs_core::Scalar;
use std::fmt::Write as _;
use std::io::Write;

/// Error type for PDF document operations.
#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    /// I/O error during write.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// PDF/A validation failure.
    #[error("PDF/A validation failed: {}", _0.join(", "))]
    Validation(Vec<String>),
    /// A requested drawing operation is not yet supported by this backend.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// PDF document metadata.
#[derive(Debug, Clone, Default)]
pub struct PdfMetadata {
    /// Document title.
    pub title: Option<String>,
    /// Document author.
    pub author: Option<String>,
    /// Document subject.
    pub subject: Option<String>,
    /// Document keywords.
    pub keywords: Option<String>,
    /// Creator application.
    pub creator: Option<String>,
    /// Creation date (PDF format).
    pub creation_date: Option<String>,
    /// Modification date (PDF format).
    pub mod_date: Option<String>,
}

/// PDF document builder.
pub struct PdfDocument {
    /// Document metadata.
    metadata: PdfMetadata,
    /// Pages in the document.
    pages: Vec<PdfPage>,
    /// Next object ID.
    next_object_id: u32,
    /// Fonts referenced by any page.
    fonts: PdfFontManager,
    /// Images referenced by any page.
    images: PdfImageManager,
    /// Transparency (ExtGState/SoftMask/Group) resources.
    transparency: TransparencyManager,
    /// PDF/A configuration, if any.
    pdfa: Option<PdfAState>,
}

/// Internal PDF/A state — carries the declared conformance level plus the
/// doc-level validation model kept in sync as resources are written.
struct PdfAState {
    level: PdfALevel,
    doc: PdfADocument,
}

/// A page in the PDF document.
pub struct PdfPage {
    /// Page width in points.
    pub width: Scalar,
    /// Page height in points.
    pub height: Scalar,
    /// Page content stream.
    pub content: Vec<u8>,
    /// Object ID.
    pub object_id: u32,
    /// Font indices used by this page.
    pub used_fonts: Vec<usize>,
    /// Image indices used by this page.
    pub used_images: Vec<usize>,
    /// `ExtGState` indices used by this page.
    pub used_ext_gstates: Vec<usize>,
}

impl Default for PdfDocument {
    fn default() -> Self {
        Self::new()
    }
}

impl PdfDocument {
    /// Create a new PDF document.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PdfMetadata::default(),
            pages: Vec::new(),
            next_object_id: 1,
            fonts: PdfFontManager::new(),
            images: PdfImageManager::new(),
            transparency: TransparencyManager::new(),
            pdfa: None,
        }
    }

    /// Set the document metadata.
    pub fn set_metadata(&mut self, metadata: PdfMetadata) {
        self.metadata = metadata;
    }

    /// Get mutable reference to metadata.
    pub const fn metadata_mut(&mut self) -> &mut PdfMetadata {
        &mut self.metadata
    }

    /// Declare a PDF/A conformance level for this document.
    ///
    /// Creates an internal [`PdfADocument`] validation model, seeds XMP
    /// metadata with a generated document ID, and installs an sRGB output
    /// intent. The model is updated as fonts/images are emitted and
    /// validated by [`validate_pdfa`](Self::validate_pdfa) or — when
    /// non-empty — automatically when [`write_to`](Self::write_to) runs.
    pub fn set_pdfa_conformance(&mut self, level: PdfALevel) {
        let mut doc = PdfADocument::new();
        doc.create_xmp_metadata(level);
        doc.set_srgb_output_intent();
        self.pdfa = Some(PdfAState { level, doc });
    }

    /// Return the active PDF/A level, if any.
    #[must_use]
    pub fn pdfa_level(&self) -> Option<PdfALevel> {
        self.pdfa.as_ref().map(|s| s.level)
    }

    /// Borrow the document's font manager for registering standard / TTF
    /// fonts ahead of time.
    pub const fn fonts_mut(&mut self) -> &mut PdfFontManager {
        &mut self.fonts
    }

    /// Borrow the document's image manager for registering images ahead of
    /// time or linking RGBA image/mask pairs post-hoc.
    pub const fn images_mut(&mut self) -> &mut PdfImageManager {
        &mut self.images
    }

    /// Borrow the document's transparency manager.
    pub const fn transparency_mut(&mut self) -> &mut TransparencyManager {
        &mut self.transparency
    }

    /// Validate the document against its configured PDF/A conformance level.
    ///
    /// Returns `Ok(())` if no level is set (PDF/A validation is opt-in) and
    /// otherwise delegates to [`PdfAValidator`]. This is called
    /// automatically from [`write_to`](Self::write_to) when PDF/A is on.
    ///
    /// # Errors
    ///
    /// Returns the list of [`PdfAError`]s reported by [`PdfAValidator`] if
    /// the document does not conform to its configured PDF/A level.
    pub fn validate_pdfa(&self) -> Result<(), Vec<PdfAError>> {
        let Some(ref state) = self.pdfa else {
            return Ok(());
        };
        let mut validator = PdfAValidator::new(state.level);
        validator.validate(&state.doc)
    }

    /// Allocate a new object ID.
    const fn alloc_object_id(&mut self) -> u32 {
        let id = self.next_object_id;
        self.next_object_id += 1;
        id
    }

    /// Begin a new page.
    ///
    /// The returned [`PdfCanvas`] borrows the document's font/image/
    /// transparency managers; finish it by passing it to
    /// [`end_page`](Self::end_page).
    pub fn begin_page(&mut self, width: Scalar, height: Scalar) -> PdfCanvas<'_> {
        let object_id = self.alloc_object_id();
        PdfCanvas::new(
            width,
            height,
            object_id,
            &mut self.fonts,
            &mut self.images,
            &mut self.transparency,
        )
    }

    /// End the current page and add it to the document.
    ///
    /// Expects a [`PageContent`] produced by [`PdfCanvas::finish`]. Canvas
    /// lifetimes borrow the document's managers, so the canvas must be
    /// finished (releasing those borrows) before the document can be
    /// mutably borrowed again.
    pub fn end_page(&mut self, page: PageContent) {
        let p = PdfPage {
            width: page.width,
            height: page.height,
            content: page.content,
            object_id: page.object_id,
            used_fonts: page.used_fonts,
            used_images: page.used_images,
            used_ext_gstates: page.used_ext_gstates,
        };
        self.pages.push(p);
    }

    /// Get the number of pages.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Write the PDF to a writer.
    ///
    /// The writer emits:
    /// 1. The PDF header (1.4 / 1.7 depending on PDF/A level).
    /// 2. Catalog, pages tree, per-page objects with fully populated
    ///    `/Resources` dictionaries (fonts, images, ext-gstates).
    /// 3. Font dictionary, font descriptor, and (for TrueType) `FontFile2`
    ///    stream for every font referenced from any page.
    /// 4. Image `XObject` for every image referenced from any page.
    /// 5. `ExtGState` dictionary for every transparency state referenced.
    /// 6. For PDF/A-tagged docs: XMP metadata stream and output intent,
    ///    plus a trailer `/ID` entry derived from the document UUID.
    /// 7. The xref table and trailer.
    ///
    /// Validates PDF/A conformance (when configured) before writing and
    /// returns a [`PdfError::Validation`] if validation fails. I/O errors
    /// are returned as [`PdfError::Io`].
    ///
    /// # Errors
    ///
    /// Returns [`PdfError::Validation`] if PDF/A conformance validation
    /// fails, or [`PdfError::Io`] if writing to `writer` fails.
    pub fn write_to<W: Write>(&mut self, writer: &mut W) -> Result<(), PdfError> {
        // Sync PDF/A document model with current state before validating.
        self.sync_pdfa_model();

        if let Err(errors) = self.validate_pdfa() {
            let msgs = errors.iter().map(|e| format!("{e}")).collect();
            return Err(PdfError::Validation(msgs));
        }

        // Now do the actual emission using a dry-run offset tracker.
        let mut ctx = WriteCtx::new(self);
        ctx.emit(writer)?;
        Ok(())
    }

    /// Update the PDF/A validation model from the document's current
    /// fonts, images, and transparency state.
    fn sync_pdfa_model(&mut self) {
        let Some(ref mut state) = self.pdfa else {
            return;
        };

        // Fonts: embedded iff font_data present OR standard.
        for font in self.fonts.fonts() {
            let info = PdfAFontInfo {
                // Standard 14 are implicitly embedded by the viewer.
                is_embedded: matches!(font.font_type, PdfFontType::Type1)
                    || font.font_data.is_some(),
                has_cmap: true, // We always emit a ToUnicode CMap.
                has_widths: !font.widths.is_empty() || matches!(font.font_type, PdfFontType::Type1),
            };
            state.doc.register_font(&font.base_font, info);
        }

        // Colors: anything with DeviceRGB/Gray/CMYK images counts.
        for img in self.images.images() {
            if matches!(
                img.color_space,
                crate::image::PdfColorSpace::DeviceRGB
                    | crate::image::PdfColorSpace::DeviceGray
                    | crate::image::PdfColorSpace::DeviceCMYK
            ) {
                state.doc.features.content.uses_device_colors = true;
            }
        }

        // Transparency: any ExtGState with alpha < 1 or non-Normal blend.
        for gs in self.transparency.ext_gstates() {
            let has_alpha =
                gs.fill_alpha.is_some_and(|a| a < 1.0) || gs.stroke_alpha.is_some_and(|a| a < 1.0);
            let has_blend = gs
                .blend_mode
                .is_some_and(|m| m != crate::transparency::PdfBlendMode::Normal);
            if has_alpha || has_blend {
                state.doc.features.content.uses_transparency = true;
            }
        }
    }

    /// Check if metadata is present.
    const fn has_metadata(&self) -> bool {
        self.metadata.title.is_some()
            || self.metadata.author.is_some()
            || self.metadata.subject.is_some()
            || self.metadata.creator.is_some()
            || self.metadata.keywords.is_some()
    }

    /// Build the info dictionary.
    fn build_info_dict(&self, id: u32) -> String {
        let mut entries = Vec::new();

        if let Some(title) = &self.metadata.title {
            entries.push(format!("/Title {}", pdf_text_string(title)));
        }
        if let Some(author) = &self.metadata.author {
            entries.push(format!("/Author {}", pdf_text_string(author)));
        }
        if let Some(subject) = &self.metadata.subject {
            entries.push(format!("/Subject {}", pdf_text_string(subject)));
        }
        if let Some(creator) = &self.metadata.creator {
            entries.push(format!("/Creator {}", pdf_text_string(creator)));
        }
        if let Some(keywords) = &self.metadata.keywords {
            entries.push(format!("/Keywords {}", pdf_text_string(keywords)));
        }

        entries.push("/Producer (skia-rs)".to_string());

        format!("{} 0 obj\n<< {} >>\nendobj\n", id, entries.join(" "))
    }

    /// Generate PDF bytes.
    ///
    /// Returns an error if PDF/A validation fails or I/O errors occur.
    ///
    /// # Errors
    ///
    /// Returns [`PdfError::Validation`] if PDF/A conformance validation
    /// fails, or [`PdfError::Io`] if writing to the internal buffer fails.
    pub fn to_bytes(&mut self) -> Result<Vec<u8>, PdfError> {
        let mut buffer = Vec::new();
        self.write_to(&mut buffer)?;
        Ok(buffer)
    }
}

/// Emit a PDF text string, picking a PDF-literal form when the text is
/// pure ASCII and a UTF-16BE hex string otherwise.
///
/// Per PDF 7.9.2.2, a Unicode text string must start with the UTF-16BE BOM
/// `FEFF` inside a hex literal.
fn pdf_text_string(s: &str) -> String {
    if s.is_ascii() {
        format!("({})", crate::canvas::escape_pdf_string(s))
    } else {
        format!("<{}>", crate::canvas::utf16be_hex(s))
    }
}

/// Private write-time state for emitting the document.
struct WriteCtx<'a> {
    doc: &'a PdfDocument,
    /// Object id → byte offset within the file.
    offsets: Vec<(u32, u64)>,
    /// Monotonic running offset.
    offset: u64,
    /// Object ID reserved for the catalog.
    catalog_id: u32,
    /// Object ID reserved for the pages tree.
    pages_id: u32,
    /// Object IDs for each font's Font dict.
    font_ids: Vec<u32>,
    /// Object IDs for each font's `FontDescriptor` dict (TrueType only).
    font_desc_ids: Vec<Option<u32>>,
    /// Object IDs for each font's `FontFile2` stream (TrueType/Type0 only).
    font_file_ids: Vec<Option<u32>>,
    /// Object IDs for each font's `ToUnicode` stream.
    font_tounicode_ids: Vec<u32>,
    /// Object IDs for each Type0 font's `CIDFontType2` descendant dict.
    font_descendant_ids: Vec<Option<u32>>,
    /// Object IDs for each image's `XObject`.
    image_ids: Vec<u32>,
    /// Object IDs for each `ExtGState`.
    ext_gstate_ids: Vec<u32>,
    /// Object ID for the XMP metadata stream (PDF/A).
    metadata_id: Option<u32>,
    /// Object ID for the `OutputIntent` dict (PDF/A).
    output_intent_id: Option<u32>,
    /// Object ID for the sRGB ICC profile stream (PDF/A).
    icc_profile_id: Option<u32>,
    /// Object ID for the document info dict.
    info_id: Option<u32>,
    /// Next-available object id.
    next_id: u32,
}

impl<'a> WriteCtx<'a> {
    const fn new(doc: &'a PdfDocument) -> Self {
        Self {
            doc,
            offsets: Vec::new(),
            offset: 0,
            catalog_id: 0,
            pages_id: 0,
            font_ids: Vec::new(),
            font_desc_ids: Vec::new(),
            font_file_ids: Vec::new(),
            font_tounicode_ids: Vec::new(),
            font_descendant_ids: Vec::new(),
            image_ids: Vec::new(),
            ext_gstate_ids: Vec::new(),
            metadata_id: None,
            output_intent_id: None,
            icc_profile_id: None,
            info_id: None,
            next_id: 1,
        }
    }

    const fn alloc(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn write_bytes<W: Write>(&mut self, writer: &mut W, bytes: &[u8]) -> std::io::Result<()> {
        writer.write_all(bytes)?;
        self.offset += bytes.len() as u64;
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "PDF object emitter: splitting would scatter a single serialization sequence across files, harming fidelity/reviewability"
    )]
    fn emit<W: Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        // Pre-allocate object ids for everything we will write, in a stable
        // order that makes forward references in the catalog easy to
        // construct.
        self.catalog_id = self.alloc();
        self.pages_id = self.alloc();

        // Reserve ids for each page + its content stream.
        let page_ids: Vec<(u32, u32)> = (0..self.doc.pages.len())
            .map(|_| (self.alloc(), self.alloc()))
            .collect();

        // Fonts. Both simple TrueType fonts and Type0/CID fonts built on
        // TrueType outlines get a FontDescriptor + embedded FontFile2;
        // Type0 fonts additionally get a CIDFontType2 descendant dict.
        let needs_embed: Vec<bool> = self
            .doc
            .fonts
            .fonts()
            .iter()
            .map(|f| {
                matches!(f.font_type, PdfFontType::TrueType | PdfFontType::Type0)
                    && f.font_data.is_some()
            })
            .collect();
        let is_type0: Vec<bool> = self
            .doc
            .fonts
            .fonts()
            .iter()
            .map(|f| f.font_type == PdfFontType::Type0)
            .collect();
        for (&embed, &type0) in needs_embed.iter().zip(&is_type0) {
            let id = self.alloc();
            self.font_ids.push(id);
            let tu = self.alloc();
            self.font_tounicode_ids.push(tu);
            if embed {
                let desc = self.alloc();
                let file = self.alloc();
                self.font_desc_ids.push(Some(desc));
                self.font_file_ids.push(Some(file));
            } else {
                self.font_desc_ids.push(None);
                self.font_file_ids.push(None);
            }
            if type0 {
                let descendant = self.alloc();
                self.font_descendant_ids.push(Some(descendant));
            } else {
                self.font_descendant_ids.push(None);
            }
        }

        // Images.
        let image_count = self.doc.images.images().len();
        for _ in 0..image_count {
            let id = self.alloc();
            self.image_ids.push(id);
        }

        // ExtGStates.
        let gs_count = self.doc.transparency.ext_gstates().len();
        for _ in 0..gs_count {
            let id = self.alloc();
            self.ext_gstate_ids.push(id);
        }

        // PDF/A auxiliary objects.
        if self.doc.pdfa.is_some() {
            self.metadata_id = Some(self.alloc());
            self.output_intent_id = Some(self.alloc());
            self.icc_profile_id = Some(self.alloc());
        }

        // Info dict.
        if self.doc.has_metadata() {
            self.info_id = Some(self.alloc());
        }

        // ---- PDF header ----
        let version = self
            .doc
            .pdfa
            .as_ref()
            .map_or("1.4", |s| s.level.min_pdf_version());
        let header = format!("%PDF-{version}\n");
        self.write_bytes(writer, header.as_bytes())?;
        self.write_bytes(writer, b"%\xE2\xE3\xCF\xD3\n")?; // binary marker

        // ---- Catalog ----
        self.offsets.push((self.catalog_id, self.offset));
        let mut catalog = format!(
            "{} 0 obj\n<< /Type /Catalog /Pages {} 0 R",
            self.catalog_id, self.pages_id
        );
        if let Some(id) = self.metadata_id {
            let _ = write!(catalog, " /Metadata {id} 0 R");
        }
        if let Some(id) = self.output_intent_id {
            let _ = write!(catalog, " /OutputIntents [{id} 0 R]");
        }
        catalog.push_str(" >>\nendobj\n");
        self.write_bytes(writer, catalog.as_bytes())?;

        // ---- Pages tree ----
        self.offsets.push((self.pages_id, self.offset));
        let kids: String = page_ids
            .iter()
            .map(|(pid, _)| format!("{pid} 0 R"))
            .collect::<Vec<_>>()
            .join(" ");
        let pages = format!(
            "{} 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n",
            self.pages_id,
            kids,
            self.doc.pages.len()
        );
        self.write_bytes(writer, pages.as_bytes())?;

        // ---- Per-page objects ----
        for (i, page) in self.doc.pages.iter().enumerate() {
            let (page_id, content_id) = page_ids[i];

            // Build the /Resources dict for this page.
            let resources = self.build_resources_dict(page);

            self.offsets.push((page_id, self.offset));
            let page_obj = format!(
                "{} 0 obj\n<< /Type /Page /Parent {} 0 R /MediaBox [0 0 {} {}] /Contents {} 0 R /Resources {} >>\nendobj\n",
                page_id, self.pages_id, page.width, page.height, content_id, resources
            );
            self.write_bytes(writer, page_obj.as_bytes())?;

            // Content stream.
            self.offsets.push((content_id, self.offset));
            let content_header = format!(
                "{} 0 obj\n<< /Length {} >>\nstream\n",
                content_id,
                page.content.len()
            );
            self.write_bytes(writer, content_header.as_bytes())?;
            self.write_bytes(writer, &page.content)?;
            self.write_bytes(writer, b"\nendstream\nendobj\n")?;
        }

        // ---- Fonts ----
        for (i, font) in self.doc.fonts.fonts().iter().enumerate() {
            // Font dict with /ToUnicode.
            self.offsets.push((self.font_ids[i], self.offset));
            let font_dict = self.build_font_dict(font, i);
            self.write_bytes(writer, font_dict.as_bytes())?;

            // ToUnicode stream (raw, uncompressed).
            self.offsets.push((self.font_tounicode_ids[i], self.offset));
            let cmap = font.generate_to_unicode();
            let tu = format!(
                "{} 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
                self.font_tounicode_ids[i],
                cmap.len(),
                cmap
            );
            self.write_bytes(writer, tu.as_bytes())?;

            // Font descriptor and FontFile2 for TTF.
            if let (Some(desc_id), Some(file_id)) = (self.font_desc_ids[i], self.font_file_ids[i]) {
                self.offsets.push((desc_id, self.offset));
                let desc = font.to_font_descriptor(desc_id, Some(file_id));
                self.write_bytes(writer, desc.as_bytes())?;

                // FontFile2 — the byte-level-subsetted TrueType bytes,
                // flate-compressed. We prune glyf/loca to just the glyphs
                // the document actually uses (plus .notdef + composite
                // dependencies) while preserving every other table so
                // character-code → glyph resolution through the font's
                // cmap continues to work.
                if let Some(data) = &font.font_data {
                    let subsetted = subset_font_bytes(font, data);
                    self.offsets.push((file_id, self.offset));
                    let compressed = flate_compress(&subsetted);
                    let hdr = format!(
                        "{} 0 obj\n<< /Length {} /Length1 {} /Filter /FlateDecode >>\nstream\n",
                        file_id,
                        compressed.len(),
                        subsetted.len()
                    );
                    self.write_bytes(writer, hdr.as_bytes())?;
                    self.write_bytes(writer, &compressed)?;
                    self.write_bytes(writer, b"\nendstream\nendobj\n")?;
                }

                // CIDFontType2 descendant dict for Type0 fonts (PDF
                // 32000-1 §9.7.4/§9.7.6): CIDSystemInfo, CIDToGIDMap, /W
                // widths, referencing this same FontDescriptor.
                if let Some(descendant_id) = self.font_descendant_ids[i] {
                    self.offsets.push((descendant_id, self.offset));
                    let cid_dict = font.to_cid_font_dict(descendant_id, desc_id);
                    self.write_bytes(writer, cid_dict.as_bytes())?;
                }
            }
        }

        // ---- Images ----
        // `PdfImage::soft_mask_id` holds the *manager index* of the mask
        // (set by `PdfImageManager::add_rgba`); translate to the real
        // object id here so the emitted XObject reference is valid.
        for (i, image) in self.doc.images.images().iter().enumerate() {
            self.offsets.push((self.image_ids[i], self.offset));
            let bytes = if let Some(mask_idx) = image.soft_mask_id {
                let real_id = self
                    .image_ids
                    .get(mask_idx as usize)
                    .copied()
                    .unwrap_or(mask_idx);
                let mut patched = image.clone();
                patched.soft_mask_id = Some(real_id);
                patched.to_pdf_xobject(self.image_ids[i])
            } else {
                image.to_pdf_xobject(self.image_ids[i])
            };
            self.write_bytes(writer, &bytes)?;
        }

        // ---- ExtGStates ----
        for (i, gs) in self.doc.transparency.ext_gstates().iter().enumerate() {
            self.offsets.push((self.ext_gstate_ids[i], self.offset));
            let dict = gs.to_pdf_dict(self.ext_gstate_ids[i]);
            self.write_bytes(writer, dict.as_bytes())?;
        }

        // ---- PDF/A Metadata + OutputIntent ----
        if let Some(ref state) = self.doc.pdfa {
            let xmp = state
                .doc
                .xmp_metadata
                .clone()
                .unwrap_or_else(|| {
                    let mut m = XmpMetadata::new();
                    m.pdfa_level = Some(state.level);
                    m
                })
                .to_xmp();

            let metadata_id = self
                .metadata_id
                .expect("metadata_id must be set when PDF/A is enabled");
            self.offsets.push((metadata_id, self.offset));
            let md = format!(
                "{} 0 obj\n<< /Type /Metadata /Subtype /XML /Length {} >>\nstream\n{}\nendstream\nendobj\n",
                metadata_id,
                xmp.len(),
                xmp
            );
            self.write_bytes(writer, md.as_bytes())?;

            let intent_id = self
                .output_intent_id
                .expect("output_intent_id must be set when PDF/A is enabled");
            let icc_id = self
                .icc_profile_id
                .expect("icc_profile_id must be set when PDF/A is enabled");
            self.offsets.push((intent_id, self.offset));
            let oi = format!(
                "{intent_id} 0 obj\n<< /Type /OutputIntent /S /GTS_PDFA1 /OutputConditionIdentifier (sRGB IEC61966-2.1) /RegistryName (http://www.color.org) /Info (sRGB IEC61966-2.1) /DestOutputProfile {icc_id} 0 R >>\nendobj\n"
            );
            self.write_bytes(writer, oi.as_bytes())?;

            // Real minimal sRGB ICC v2 profile stream, flate-compressed.
            // PDF/A OutputIntents require an actual ICC profile — a
            // placeholder byte string (as this used to embed) is not a
            // valid ICC stream and fails conformance in any validator that
            // parses it, even though the PDF object structure looks right.
            self.offsets.push((icc_id, self.offset));
            let icc_payload = flate_compress(SRGB_ICC_V2_PROFILE);
            let icc_hdr = format!(
                "{} 0 obj\n<< /N 3 /Alternate /DeviceRGB /Length {} /Filter /FlateDecode >>\nstream\n",
                icc_id,
                icc_payload.len()
            );
            self.write_bytes(writer, icc_hdr.as_bytes())?;
            self.write_bytes(writer, &icc_payload)?;
            self.write_bytes(writer, b"\nendstream\nendobj\n")?;
        }

        // ---- Info dict ----
        if let Some(info_id) = self.info_id {
            self.offsets.push((info_id, self.offset));
            let info = self.doc.build_info_dict(info_id);
            self.write_bytes(writer, info.as_bytes())?;
        }

        // ---- xref ----
        let xref_offset = self.offset;
        self.write_bytes(writer, b"xref\n")?;
        let size = self.offsets.len() + 1;
        self.write_bytes(writer, format!("0 {size}\n").as_bytes())?;
        self.write_bytes(writer, b"0000000000 65535 f \n")?;

        let mut sorted = self.offsets.clone();
        sorted.sort_by_key(|(id, _)| *id);

        // Any holes (unlikely with our sequential allocator) get entered as
        // free objects; the producer is strictly append-only so this won't
        // happen in practice — assert in debug.
        for (expected, (id, off)) in (1u32..).zip(sorted.iter()) {
            debug_assert_eq!(*id, expected, "xref hole at {expected}");
            self.write_bytes(writer, format!("{off:010} 00000 n \n").as_bytes())?;
        }

        // ---- Trailer ----
        let id_hex = self
            .doc
            .pdfa
            .as_ref()
            .and_then(|s| s.doc.document_id.as_ref())
            .map_or_else(generate_doc_id, Clone::clone);
        let id_bytes = id_hex.replace('-', "").chars().take(32).collect::<String>();
        let trailer_id_entry = format!("/ID [<{id_bytes}> <{id_bytes}>]");

        self.write_bytes(writer, b"trailer\n")?;
        let trailer = if let Some(info_id) = self.info_id {
            format!(
                "<< /Size {} /Root {} 0 R /Info {} 0 R {} >>\n",
                size, self.catalog_id, info_id, trailer_id_entry
            )
        } else {
            format!(
                "<< /Size {} /Root {} 0 R {} >>\n",
                size, self.catalog_id, trailer_id_entry
            )
        };
        self.write_bytes(writer, trailer.as_bytes())?;

        // startxref
        self.write_bytes(
            writer,
            format!("startxref\n{xref_offset}\n%%EOF\n").as_bytes(),
        )?;

        Ok(())
    }

    fn build_resources_dict(&self, page: &PdfPage) -> String {
        let mut parts = Vec::new();

        // /Font
        if !page.used_fonts.is_empty() {
            let mut entries = Vec::new();
            for &font_idx in &page.used_fonts {
                let id = self.font_ids[font_idx];
                entries.push(format!("/F{} {} 0 R", font_idx + 1, id));
            }
            parts.push(format!("/Font << {} >>", entries.join(" ")));
        }

        // /XObject (images + transparency-group forms)
        if !page.used_images.is_empty() {
            let mut entries = Vec::new();
            for &img_idx in &page.used_images {
                let id = self.image_ids[img_idx];
                entries.push(format!("/Im{} {} 0 R", img_idx + 1, id));
            }
            parts.push(format!("/XObject << {} >>", entries.join(" ")));
        }

        // /ExtGState
        if !page.used_ext_gstates.is_empty() {
            let mut entries = Vec::new();
            for &gs_idx in &page.used_ext_gstates {
                let id = self.ext_gstate_ids[gs_idx];
                entries.push(format!("/GS{} {} 0 R", gs_idx + 1, id));
            }
            parts.push(format!("/ExtGState << {} >>", entries.join(" ")));
        }

        // /ProcSet — always include PDF, Text, ImageC for compatibility.
        parts.push("/ProcSet [/PDF /Text /ImageB /ImageC /ImageI]".to_string());

        format!("<< {} >>", parts.join(" "))
    }

    fn build_font_dict(&self, font: &PdfFont, i: usize) -> String {
        let id = self.font_ids[i];
        let to_unicode_id = self.font_tounicode_ids[i];
        let mut dict = format!("{id} 0 obj\n<<\n");

        match font.font_type {
            PdfFontType::Type1 => {
                dict.push_str("/Type /Font\n");
                dict.push_str("/Subtype /Type1\n");
                let _ = writeln!(dict, "/BaseFont /{}", font.base_font);
                if let Some(ref enc) = font.encoding {
                    let _ = writeln!(dict, "/Encoding /{enc}");
                }
            }
            PdfFontType::TrueType => {
                dict.push_str("/Type /Font\n");
                dict.push_str("/Subtype /TrueType\n");
                let _ = writeln!(dict, "/BaseFont /{}", font.pdf_base_font());
                let _ = writeln!(dict, "/FirstChar {}", font.first_char);
                let _ = writeln!(dict, "/LastChar {}", font.last_char);

                dict.push_str("/Widths [");
                for code in font.first_char..=font.last_char {
                    let w = font.widths.get(&code).copied().unwrap_or(0);
                    let _ = write!(dict, "{w} ");
                }
                dict.push_str("]\n");

                if let Some(desc_id) = self.font_desc_ids[i] {
                    let _ = writeln!(dict, "/FontDescriptor {desc_id} 0 R");
                }
                if let Some(ref enc) = font.encoding {
                    let _ = writeln!(dict, "/Encoding /{enc}");
                }
            }
            PdfFontType::OpenTypeCff | PdfFontType::Type0 => {
                dict.push_str("/Type /Font\n");
                dict.push_str("/Subtype /Type0\n");
                let _ = writeln!(dict, "/BaseFont /{}", font.pdf_base_font());
                dict.push_str("/Encoding /Identity-H\n");
                // PDF 32000-1 §9.7.3: a Type0 font's glyphs come entirely
                // from its descendant CID font — without this array the
                // font has no outlines at all.
                if let Some(desc_id) = self.font_descendant_ids[i] {
                    let _ = writeln!(dict, "/DescendantFonts [{desc_id} 0 R]");
                }
            }
        }

        let _ = writeln!(dict, "/ToUnicode {to_unicode_id} 0 R");
        dict.push_str(">>\nendobj\n");
        dict
    }
}

/// Run the font's used-glyphs set through the byte-level TrueType
/// subsetter. Returns the original bytes unchanged when the subsetter
/// fails (malformed data, unsupported layout), so embedding always
/// succeeds even with exotic fonts.
fn subset_font_bytes(font: &PdfFont, data: &[u8]) -> Vec<u8> {
    let glyphs: std::collections::BTreeSet<u16> = font.used_glyphs.iter().copied().collect();
    crate::font::subset::subset_truetype(data, &glyphs).unwrap_or_else(|_| data.to_vec())
}

/// Flate-compress the given bytes. Used for stream-level compression of
/// TrueType font data and ICC profiles inside PDF streams.
fn flate_compress(data: &[u8]) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use std::io::Write as _;

    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// A real, minimal sRGB ICC v2.1 profile (2576 bytes; "monitor" device
/// class, RGB → XYZ PCS), embedded verbatim in the `OutputIntent`'s
/// `/DestOutputProfile` stream for PDF/A conformance (ISO 19005 requires an
/// actual, parseable ICC profile there — not just a plausible-looking PDF
/// object). Provenance: the "Artifex Software sRGB ICC Profile" already
/// vendored in this workspace at
/// `crates/skia-rs-core/tests/fixtures/srgb.icc` for color-management
/// tests; copied here unmodified as `assets/srgb-v2.icc` so this crate
/// doesn't depend on another crate's test fixtures.
const SRGB_ICC_V2_PROFILE: &[u8] = include_bytes!("../assets/srgb-v2.icc");

/// Produce a 32-char hex document-ID string (used in the trailer `/ID`
/// entry). Derived from the current system time; the trailer `/ID` is
/// allowed to be any byte string of even length so this is enough to let
/// viewers detect concatenated PDFs.
fn generate_doc_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    format!("{:016x}{:016x}", secs, u64::from(nanos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_document_empty() {
        let mut doc = PdfDocument::new();
        let bytes = doc.to_bytes().unwrap();
        assert!(bytes.starts_with(b"%PDF-1.4"));
    }

    #[test]
    fn test_pdf_document_with_page() {
        let mut doc = PdfDocument::new();
        let canvas = doc.begin_page(612.0, 792.0); // Letter size
        let page = canvas.finish();
        doc.end_page(page);

        let bytes = doc.to_bytes().unwrap();
        assert!(bytes.starts_with(b"%PDF-1.4"));
        assert_eq!(doc.page_count(), 1);
    }

    #[test]
    fn test_pdf_metadata() {
        let mut doc = PdfDocument::new();
        doc.metadata_mut().title = Some("Test Document".to_string());
        doc.metadata_mut().author = Some("Test Author".to_string());

        let canvas = doc.begin_page(612.0, 792.0);
        let page = canvas.finish();
        doc.end_page(page);

        let bytes = doc.to_bytes().unwrap();
        let content = String::from_utf8_lossy(&bytes);
        assert!(content.contains("/Title (Test Document)"));
    }

    #[test]
    fn test_document_resources_dict_populated_with_font() {
        use crate::canvas::PdfCanvas;
        use crate::font::StandardFont;
        use skia_rs_paint::Paint;

        let mut doc = PdfDocument::new();
        {
            let mut canvas: PdfCanvas = doc.begin_page(612.0, 792.0);
            canvas.use_standard_font(StandardFont::Helvetica);
            let paint = Paint::new();
            canvas
                .draw_text("Hello, PDF", 72.0, 72.0, 12.0, &paint)
                .expect("draw_text should succeed");
            let page = canvas.finish();
            doc.end_page(page);
        }

        let bytes = doc.to_bytes().unwrap();
        let s = String::from_utf8_lossy(&bytes);
        // The page's /Resources dict must reference the font.
        assert!(
            s.contains("/Font << /F1"),
            "missing /Font in Resources: {s}"
        );
        // The font dict itself must be emitted somewhere in the file.
        assert!(s.contains("/BaseFont /Helvetica"));
        // ToUnicode stream is written.
        assert!(s.contains("/ToUnicode"));
    }

    #[test]
    fn test_document_resources_with_image_and_ext_gstate() {
        use skia_rs_core::{Color, Rect};
        use skia_rs_paint::Paint;

        let mut doc = PdfDocument::new();
        // Register an image on the document.
        let img_idx =
            doc.images_mut()
                .add_rgb(2, 2, &[0, 0, 0, 255, 255, 255, 128, 128, 128, 64, 64, 64]);
        {
            let mut canvas = doc.begin_page(200.0, 200.0);
            canvas.draw_image(img_idx, 10.0, 10.0, 50.0, 50.0);

            // Transparent fill triggers an ExtGState.
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(64, 255, 0, 0));
            canvas.draw_rect(&Rect::from_xywh(80.0, 10.0, 40.0, 40.0), &paint);

            let page = canvas.finish();
            doc.end_page(page);
        }

        let bytes = doc.to_bytes().unwrap();
        let s = String::from_utf8_lossy(&bytes);
        assert!(
            s.contains("/XObject << /Im1"),
            "missing /XObject in Resources: {s}"
        );
        assert!(s.contains("/ExtGState << /GS1"), "missing /ExtGState: {s}");
        assert!(s.contains("/Subtype /Image"));
        assert!(s.contains("/Type /ExtGState"));
    }

    #[test]
    fn test_document_pdfa_sets_xmp_and_output_intent() {
        let mut doc = PdfDocument::new();
        doc.set_pdfa_conformance(PdfALevel::A2b);
        assert_eq!(doc.pdfa_level(), Some(PdfALevel::A2b));

        let canvas = doc.begin_page(612.0, 792.0);
        let page = canvas.finish();
        doc.end_page(page);

        let bytes = doc.to_bytes().unwrap();
        let s = String::from_utf8_lossy(&bytes);

        // Header bumped to 1.7 for PDF/A-2.
        assert!(bytes.starts_with(b"%PDF-1.7"), "wrong PDF version header");
        // XMP metadata stream present.
        assert!(s.contains("/Type /Metadata"));
        assert!(s.contains("pdfaid:part>2"));
        // OutputIntent referenced from catalog.
        assert!(s.contains("/OutputIntents ["));
        assert!(s.contains("/Type /OutputIntent"));
        // /ID in the trailer.
        assert!(s.contains("/ID ["));
    }

    #[test]
    fn test_pdfa_a1_rejects_transparency_on_write() {
        use skia_rs_core::{Color, Rect};
        use skia_rs_paint::Paint;

        let mut doc = PdfDocument::new();
        doc.set_pdfa_conformance(PdfALevel::A1b);
        {
            let mut canvas = doc.begin_page(100.0, 100.0);
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(128, 255, 0, 0));
            canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0), &paint);
            let page = canvas.finish();
            doc.end_page(page);
        }

        let mut buf = Vec::new();
        let err = doc
            .write_to(&mut buf)
            .expect_err("A1b with transparency must fail");
        match err {
            PdfError::Validation(msgs) => {
                let combined = msgs.join(" ");
                assert!(
                    combined.contains("Transparency"),
                    "unexpected error: {combined}"
                );
            }
            PdfError::Io(e) => panic!("expected validation error, got I/O error: {e}"),
            PdfError::Unsupported(msg) => {
                panic!("expected validation error, got unsupported error: {msg}")
            }
        }
    }
}
