//! PDF/A compliance support.
//!
//! This module provides functionality for generating PDF/A compliant documents
//! for long-term archival.
//!
//! # Supported Standards
//!
//! - **PDF/A-1b** (ISO 19005-1) - Basic conformance, visual appearance
//! - **PDF/A-2b** (ISO 19005-2) - Based on PDF 1.7, JPEG2000, transparency
//! - **PDF/A-3b** (ISO 19005-3) - Allows embedded files
//!
//! # Example
//!
//! ```rust,ignore
//! use skia_rs_pdf::{PdfDocument, PdfALevel, PdfAValidator};
//!
//! let mut doc = PdfDocument::new();
//! doc.set_pdfa_conformance(PdfALevel::A1b);
//!
//! // ... add content ...
//!
//! // Validate before saving
//! let validator = PdfAValidator::new(PdfALevel::A1b);
//! if let Err(errors) = validator.validate(&doc) {
//!     for error in errors {
//!         eprintln!("PDF/A violation: {}", error);
//!     }
//! }
//! ```

use std::collections::HashSet;
use std::fmt::Write as _;

// =============================================================================
// PDF/A Conformance Levels
// =============================================================================

/// PDF/A conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PdfALevel {
    /// PDF/A-1a: Full conformance (structure, accessibility)
    A1a,
    /// PDF/A-1b: Basic conformance (visual appearance only)
    A1b,
    /// PDF/A-2a: Full conformance based on PDF 1.7
    A2a,
    /// PDF/A-2b: Basic conformance based on PDF 1.7
    A2b,
    /// PDF/A-2u: Unicode text extraction required
    A2u,
    /// PDF/A-3a: Full conformance with embedded files
    A3a,
    /// PDF/A-3b: Basic conformance with embedded files
    A3b,
    /// PDF/A-3u: Unicode with embedded files
    A3u,
}

impl PdfALevel {
    /// Get the PDF/A part number.
    #[must_use]
    pub const fn part(&self) -> u8 {
        match self {
            Self::A1a | Self::A1b => 1,
            Self::A2a | Self::A2b | Self::A2u => 2,
            Self::A3a | Self::A3b | Self::A3u => 3,
        }
    }

    /// Get the conformance level identifier.
    #[must_use]
    pub const fn conformance(&self) -> &'static str {
        match self {
            Self::A1a | Self::A2a | Self::A3a => "A",
            Self::A1b | Self::A2b | Self::A3b => "B",
            Self::A2u | Self::A3u => "U",
        }
    }

    /// Get the minimum PDF version required.
    #[must_use]
    pub const fn min_pdf_version(&self) -> &'static str {
        match self {
            Self::A1a | Self::A1b => "1.4",
            Self::A2a | Self::A2b | Self::A2u | Self::A3a | Self::A3b | Self::A3u => "1.7",
        }
    }

    /// Check if transparency is allowed.
    #[must_use]
    pub const fn allows_transparency(&self) -> bool {
        !matches!(self, Self::A1a | Self::A1b)
    }

    /// Check if embedded files are allowed.
    #[must_use]
    pub const fn allows_embedded_files(&self) -> bool {
        matches!(self, Self::A3a | Self::A3b | Self::A3u)
    }

    /// Check if JPEG2000 compression is allowed.
    #[must_use]
    pub const fn allows_jpeg2000(&self) -> bool {
        !matches!(self, Self::A1a | Self::A1b)
    }

    /// Check if Unicode text is required.
    #[must_use]
    pub const fn requires_unicode(&self) -> bool {
        matches!(self, Self::A2u | Self::A3u)
    }

    /// Check if structure (tagged PDF) is required.
    #[must_use]
    pub const fn requires_structure(&self) -> bool {
        matches!(self, Self::A1a | Self::A2a | Self::A3a)
    }
}

// =============================================================================
// PDF/A Validation Errors
// =============================================================================

/// PDF/A validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfAError {
    /// Error code.
    pub code: PdfAErrorCode,
    /// Human-readable description.
    pub message: String,
    /// Location in document (if applicable).
    pub location: Option<String>,
}

impl std::fmt::Display for PdfAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref loc) = self.location {
            write!(f, "{:?} at {}: {}", self.code, loc, self.message)
        } else {
            write!(f, "{:?}: {}", self.code, self.message)
        }
    }
}

impl std::error::Error for PdfAError {}

/// PDF/A error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PdfAErrorCode {
    // Metadata errors
    /// Missing XMP metadata
    MissingXmpMetadata,
    /// Invalid XMP metadata
    InvalidXmpMetadata,
    /// Missing document ID
    MissingDocumentId,
    /// Missing PDF/A identification
    MissingPdfaId,
    /// PDF version mismatch
    PdfVersionMismatch,

    // Font errors
    /// Font not embedded
    FontNotEmbedded,
    /// Font missing character mapping
    FontMissingCmap,
    /// Invalid font subset
    InvalidFontSubset,
    /// Font missing glyph widths
    FontMissingWidths,

    // Color errors
    /// Device-dependent color without output intent
    DeviceColorWithoutIntent,
    /// Missing output intent
    MissingOutputIntent,
    /// Invalid ICC profile
    InvalidIccProfile,
    /// Uncalibrated color space
    UncalibratedColorSpace,

    // Image errors
    /// Image compression not allowed
    DisallowedImageCompression,
    /// LZW compression not allowed (PDF/A-1)
    LzwCompressionNotAllowed,
    /// JPEG2000 not allowed (PDF/A-1)
    Jpeg2000NotAllowed,

    // Transparency errors
    /// Transparency not allowed (PDF/A-1)
    TransparencyNotAllowed,
    /// Invalid blend mode
    InvalidBlendMode,

    // Structure errors
    /// Missing document structure (PDF/A-a levels)
    MissingDocumentStructure,
    /// Missing alternative text
    MissingAltText,
    /// Invalid structure element
    InvalidStructureElement,

    // Security errors
    /// Encryption not allowed
    EncryptionNotAllowed,
    /// JavaScript not allowed
    JavaScriptNotAllowed,
    /// Actions not allowed
    ActionsNotAllowed,

    // Embedded file errors
    /// Embedded files not allowed (PDF/A-1, PDF/A-2)
    EmbeddedFilesNotAllowed,
    /// Missing file relationship
    MissingFileRelationship,

    // Other
    /// External content references
    ExternalContentReference,
    /// Audio/video content not allowed
    MultimediaNotAllowed,
}

// =============================================================================
// XMP Metadata
// =============================================================================

/// XMP metadata for PDF/A compliance.
#[derive(Debug, Clone, Default)]
pub struct XmpMetadata {
    /// Document title.
    pub title: Option<String>,
    /// Document author.
    pub author: Option<String>,
    /// Document subject/description.
    pub subject: Option<String>,
    /// Keywords.
    pub keywords: Vec<String>,
    /// Creator application.
    pub creator: Option<String>,
    /// Creation date (ISO 8601).
    pub create_date: Option<String>,
    /// Modification date (ISO 8601).
    pub modify_date: Option<String>,
    /// PDF/A conformance level.
    pub pdfa_level: Option<PdfALevel>,
    /// Document ID (UUID).
    pub document_id: Option<String>,
    /// Instance ID (UUID).
    pub instance_id: Option<String>,
}

impl XmpMetadata {
    /// Create new XMP metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set document title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set document author.
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set PDF/A level.
    #[must_use]
    pub const fn with_pdfa_level(mut self, level: PdfALevel) -> Self {
        self.pdfa_level = Some(level);
        self
    }

    /// Generate XMP packet.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "XMP/RDF packet emitter: a single literal serialization sequence; splitting would harm fidelity/reviewability"
    )]
    pub fn to_xmp(&self) -> String {
        let mut xmp = String::new();

        xmp.push_str(r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>"#);
        xmp.push('\n');
        xmp.push_str(r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">"#);
        xmp.push('\n');
        xmp.push_str(r#"  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#);
        xmp.push('\n');

        // Dublin Core metadata
        xmp.push_str(
            r#"    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">"#,
        );
        xmp.push('\n');

        if let Some(ref title) = self.title {
            let _ = write!(
                xmp,
                r#"      <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{}</rdf:li></rdf:Alt></dc:title>"#,
                escape_xml(title)
            );
            xmp.push('\n');
        }

        if let Some(ref author) = self.author {
            let _ = write!(
                xmp,
                r"      <dc:creator><rdf:Seq><rdf:li>{}</rdf:li></rdf:Seq></dc:creator>",
                escape_xml(author)
            );
            xmp.push('\n');
        }

        if let Some(ref subject) = self.subject {
            let _ = write!(
                xmp,
                r#"      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{}</rdf:li></rdf:Alt></dc:description>"#,
                escape_xml(subject)
            );
            xmp.push('\n');
        }

        xmp.push_str(r"    </rdf:Description>");
        xmp.push('\n');

        // XMP Basic metadata
        xmp.push_str(
            r#"    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">"#,
        );
        xmp.push('\n');

        if let Some(ref creator) = self.creator {
            let _ = write!(
                xmp,
                r"      <xmp:CreatorTool>{}</xmp:CreatorTool>",
                escape_xml(creator)
            );
            xmp.push('\n');
        }

        if let Some(ref create_date) = self.create_date {
            let _ = write!(xmp, r"      <xmp:CreateDate>{create_date}</xmp:CreateDate>");
            xmp.push('\n');
        }

        if let Some(ref modify_date) = self.modify_date {
            let _ = write!(xmp, r"      <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>");
            xmp.push('\n');
        }

        xmp.push_str(r"    </rdf:Description>");
        xmp.push('\n');

        // PDF metadata
        xmp.push_str(
            r#"    <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">"#,
        );
        xmp.push('\n');
        xmp.push_str(r"      <pdf:Producer>skia-rs 0.1.0</pdf:Producer>");
        xmp.push('\n');
        xmp.push_str(r"    </rdf:Description>");
        xmp.push('\n');

        // PDF/A identification
        if let Some(level) = self.pdfa_level {
            xmp.push_str(r#"    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">"#);
            xmp.push('\n');
            let _ = write!(xmp, r"      <pdfaid:part>{}</pdfaid:part>", level.part());
            xmp.push('\n');
            let _ = write!(
                xmp,
                r"      <pdfaid:conformance>{}</pdfaid:conformance>",
                level.conformance()
            );
            xmp.push('\n');
            xmp.push_str(r"    </rdf:Description>");
            xmp.push('\n');
        }

        // XMP Media Management
        if self.document_id.is_some() || self.instance_id.is_some() {
            xmp.push_str(r#"    <rdf:Description rdf:about="" xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">"#);
            xmp.push('\n');

            if let Some(ref doc_id) = self.document_id {
                let _ = write!(
                    xmp,
                    r"      <xmpMM:DocumentID>uuid:{doc_id}</xmpMM:DocumentID>"
                );
                xmp.push('\n');
            }

            if let Some(ref inst_id) = self.instance_id {
                let _ = write!(
                    xmp,
                    r"      <xmpMM:InstanceID>uuid:{inst_id}</xmpMM:InstanceID>"
                );
                xmp.push('\n');
            }

            xmp.push_str(r"    </rdf:Description>");
            xmp.push('\n');
        }

        xmp.push_str(r"  </rdf:RDF>");
        xmp.push('\n');
        xmp.push_str(r"</x:xmpmeta>");
        xmp.push('\n');

        // Padding for in-place updates
        for _ in 0..20 {
            xmp.push_str("                                                                                \n");
        }

        xmp.push_str(r#"<?xpacket end="w"?>"#);

        xmp
    }
}

/// Escape XML special characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// =============================================================================
// Output Intent
// =============================================================================

/// PDF output intent for color management.
#[derive(Debug, Clone)]
pub struct OutputIntent {
    /// Output condition identifier.
    pub output_condition: String,
    /// Output condition identifier type.
    pub output_condition_identifier: String,
    /// Registry name (e.g., "<http://www.color.org>").
    pub registry_name: Option<String>,
    /// Human-readable info.
    pub info: Option<String>,
    /// ICC profile data.
    pub icc_profile: Option<Vec<u8>>,
}

impl OutputIntent {
    /// Create sRGB output intent.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            output_condition: "sRGB IEC61966-2.1".to_string(),
            output_condition_identifier: "sRGB IEC61966-2.1".to_string(),
            registry_name: Some("http://www.color.org".to_string()),
            info: Some("sRGB IEC61966-2.1".to_string()),
            icc_profile: None, // Would contain actual sRGB ICC profile
        }
    }

    /// Create FOGRA39 (coated) output intent for print.
    #[must_use]
    pub fn fogra39() -> Self {
        Self {
            output_condition: "FOGRA39".to_string(),
            output_condition_identifier: "FOGRA39L".to_string(),
            registry_name: Some("http://www.color.org".to_string()),
            info: Some("Coated FOGRA39 (ISO 12647-2:2004)".to_string()),
            icc_profile: None,
        }
    }

    /// Create custom output intent with ICC profile.
    #[must_use]
    pub fn custom(condition: &str, icc_profile: Vec<u8>) -> Self {
        Self {
            output_condition: condition.to_string(),
            output_condition_identifier: condition.to_string(),
            registry_name: Some("http://www.color.org".to_string()),
            info: None,
            icc_profile: Some(icc_profile),
        }
    }
}

// =============================================================================
// PDF/A Validator
// =============================================================================

/// PDF/A compliance validator.
pub struct PdfAValidator {
    level: PdfALevel,
    errors: Vec<PdfAError>,
}

impl PdfAValidator {
    /// Create a new validator for the specified level.
    #[must_use]
    pub const fn new(level: PdfALevel) -> Self {
        Self {
            level,
            errors: Vec::new(),
        }
    }

    /// Validate a document and return errors.
    ///
    /// # Errors
    ///
    /// Returns the accumulated list of [`PdfAError`]s if the document does
    /// not conform to the configured [`PdfALevel`].
    pub fn validate(&mut self, doc: &PdfADocument) -> Result<(), Vec<PdfAError>> {
        self.errors.clear();

        self.check_metadata(doc);
        self.check_fonts(doc);
        self.check_colors(doc);
        self.check_images(doc);
        self.check_transparency(doc);
        self.check_structure(doc);
        self.check_security(doc);
        self.check_embedded_files(doc);

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn check_metadata(&mut self, doc: &PdfADocument) {
        if doc.xmp_metadata.is_none() {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::MissingXmpMetadata,
                message: "XMP metadata stream is required".to_string(),
                location: None,
            });
        }

        if doc.document_id.is_none() {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::MissingDocumentId,
                message: "Document ID is required in trailer".to_string(),
                location: None,
            });
        }

        if let Some(ref xmp) = doc.xmp_metadata
            && xmp.pdfa_level.is_none()
        {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::MissingPdfaId,
                message: "PDF/A identification in XMP metadata is required".to_string(),
                location: None,
            });
        }
    }

    fn check_fonts(&mut self, doc: &PdfADocument) {
        for (name, font) in &doc.fonts {
            if !font.is_embedded {
                self.errors.push(PdfAError {
                    code: PdfAErrorCode::FontNotEmbedded,
                    message: format!("Font '{name}' must be embedded"),
                    location: Some(format!("Font: {name}")),
                });
            }

            if !font.has_cmap {
                self.errors.push(PdfAError {
                    code: PdfAErrorCode::FontMissingCmap,
                    message: format!("Font '{name}' missing ToUnicode CMap"),
                    location: Some(format!("Font: {name}")),
                });
            }

            if !font.has_widths {
                self.errors.push(PdfAError {
                    code: PdfAErrorCode::FontMissingWidths,
                    message: format!("Font '{name}' missing glyph widths"),
                    location: Some(format!("Font: {name}")),
                });
            }
        }
    }

    fn check_colors(&mut self, doc: &PdfADocument) {
        if doc.output_intent.is_none() && doc.features.content.uses_device_colors {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::MissingOutputIntent,
                message: "Output intent required when using device-dependent colors".to_string(),
                location: None,
            });
        }

        for color_space in &doc.uncalibrated_colors {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::UncalibratedColorSpace,
                message: format!("Uncalibrated color space '{color_space}' not allowed"),
                location: Some(format!("ColorSpace: {color_space}")),
            });
        }
    }

    fn check_images(&mut self, doc: &PdfADocument) {
        if !self.level.allows_jpeg2000() && doc.features.compression.uses_jpeg2000 {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::Jpeg2000NotAllowed,
                message: "JPEG2000 compression not allowed in PDF/A-1".to_string(),
                location: None,
            });
        }

        if doc.features.compression.uses_lzw {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::LzwCompressionNotAllowed,
                message: "LZW compression not allowed in PDF/A".to_string(),
                location: None,
            });
        }
    }

    fn check_transparency(&mut self, doc: &PdfADocument) {
        if !self.level.allows_transparency() && doc.features.content.uses_transparency {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::TransparencyNotAllowed,
                message: "Transparency not allowed in PDF/A-1".to_string(),
                location: None,
            });
        }
    }

    fn check_structure(&mut self, doc: &PdfADocument) {
        if self.level.requires_structure() && !doc.features.content.has_structure {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::MissingDocumentStructure,
                message: "Tagged PDF structure required for PDF/A-a conformance".to_string(),
                location: None,
            });
        }
    }

    fn check_security(&mut self, doc: &PdfADocument) {
        if doc.features.security.is_encrypted {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::EncryptionNotAllowed,
                message: "Encryption not allowed in PDF/A".to_string(),
                location: None,
            });
        }

        if doc.features.security.has_javascript {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::JavaScriptNotAllowed,
                message: "JavaScript not allowed in PDF/A".to_string(),
                location: None,
            });
        }
    }

    fn check_embedded_files(&mut self, doc: &PdfADocument) {
        if !self.level.allows_embedded_files() && !doc.embedded_files.is_empty() {
            self.errors.push(PdfAError {
                code: PdfAErrorCode::EmbeddedFilesNotAllowed,
                message: format!("Embedded files not allowed in PDF/A-{}", self.level.part()),
                location: None,
            });
        }

        // PDF/A-3 requires file relationship
        if self.level.allows_embedded_files() {
            for file in &doc.embedded_files {
                if file.relationship.is_none() {
                    self.errors.push(PdfAError {
                        code: PdfAErrorCode::MissingFileRelationship,
                        message: format!("Embedded file '{}' missing AFRelationship", file.name),
                        location: Some(format!("File: {}", file.name)),
                    });
                }
            }
        }
    }
}

// =============================================================================
// PDF/A Document (validation model)
// =============================================================================

/// PDF/A font info for validation.
#[derive(Debug, Clone, Default)]
pub struct PdfAFontInfo {
    /// Font is embedded.
    pub is_embedded: bool,
    /// Has `ToUnicode` `CMap`.
    pub has_cmap: bool,
    /// Has glyph widths.
    pub has_widths: bool,
}

/// Embedded file info.
#[derive(Debug, Clone)]
pub struct EmbeddedFileInfo {
    /// File name.
    pub name: String,
    /// MIME type.
    pub mime_type: Option<String>,
    /// `AFRelationship` (Source, Data, Alternative, etc.)
    pub relationship: Option<String>,
}

/// Compression-related feature flags tracked on a [`PdfADocument`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PdfACompressionFlags {
    /// Uses JPEG2000 compression.
    pub uses_jpeg2000: bool,
    /// Uses LZW compression.
    pub uses_lzw: bool,
}

/// Content-related feature flags tracked on a [`PdfADocument`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PdfAContentFlags {
    /// Uses device-dependent colors (`DeviceRGB`, `DeviceCMYK`, `DeviceGray`).
    pub uses_device_colors: bool,
    /// Uses transparency.
    pub uses_transparency: bool,
    /// Has tagged structure.
    pub has_structure: bool,
}

/// Security-related feature flags tracked on a [`PdfADocument`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PdfASecurityFlags {
    /// Is encrypted.
    pub is_encrypted: bool,
    /// Has JavaScript.
    pub has_javascript: bool,
}

/// Boolean feature flags tracked on a [`PdfADocument`] for validation.
///
/// Grouped into three sub-structs by topic (rather than seven top-level
/// `bool` fields on `PdfADocument`, or one seven-`bool` struct here) to keep
/// the document model within clippy's `struct_excessive_bools` guidance at
/// every level — these are independent, unrelated yes/no facts about the
/// document, not a state machine, so plain bools (rather than enums) remain
/// the clearest representation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PdfAFeatureFlags {
    /// Compression-related flags.
    pub compression: PdfACompressionFlags,
    /// Content-related flags.
    pub content: PdfAContentFlags,
    /// Security-related flags.
    pub security: PdfASecurityFlags,
}

/// PDF/A document model for validation.
#[derive(Debug, Clone, Default)]
pub struct PdfADocument {
    /// XMP metadata.
    pub xmp_metadata: Option<XmpMetadata>,
    /// Document ID.
    pub document_id: Option<String>,
    /// Output intent.
    pub output_intent: Option<OutputIntent>,
    /// Fonts used in document.
    pub fonts: std::collections::HashMap<String, PdfAFontInfo>,
    /// Uncalibrated color spaces used.
    pub uncalibrated_colors: HashSet<String>,
    /// Boolean feature flags (device colors, compression, transparency, etc).
    pub features: PdfAFeatureFlags,
    /// Embedded files.
    pub embedded_files: Vec<EmbeddedFileInfo>,
}

impl PdfADocument {
    /// Create a new PDF/A document model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create XMP metadata with PDF/A identification.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: the `expect` below fires only if the
    /// `xmp_metadata` field set two lines above were somehow cleared
    /// concurrently, which cannot happen through `&mut self`.
    pub fn create_xmp_metadata(&mut self, level: PdfALevel) -> &mut XmpMetadata {
        let doc_id = uuid_v4();
        let inst_id = uuid_v4();

        self.document_id = Some(doc_id.clone());

        self.xmp_metadata = Some(XmpMetadata {
            title: None,
            author: None,
            subject: None,
            keywords: Vec::new(),
            creator: Some("skia-rs".to_string()),
            create_date: Some(iso8601_now()),
            modify_date: Some(iso8601_now()),
            pdfa_level: Some(level),
            document_id: Some(doc_id),
            instance_id: Some(inst_id),
        });

        self.xmp_metadata
            .as_mut()
            .expect("xmp_metadata was just set above")
    }

    /// Set sRGB output intent.
    pub fn set_srgb_output_intent(&mut self) {
        self.output_intent = Some(OutputIntent::srgb());
    }

    /// Register a font.
    pub fn register_font(&mut self, name: &str, info: PdfAFontInfo) {
        self.fonts.insert(name.to_string(), info);
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Generate a UUID v4 — pseudo-random, seeded from both the high-
/// resolution clock *and* a process-wide monotonic counter so that two
/// invocations within the same nanosecond still produce distinct ids.
///
/// Not cryptographically secure (no `getrandom` dependency pulled in for
/// this), but unique under all realistic conditions including rapid-fire
/// document creation on the same thread.
fn uuid_v4() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // Low 64 bits of the nanosecond timestamp — matches the truncating `as
    // u64` cast this replaces (this is a PRNG seed, not a precise duration).
    let nanos = u64::try_from(now.as_nanos() & u128::from(u64::MAX)).unwrap_or(u64::MAX);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

    // Combine the clock and the counter through a splitmix64-style mixer
    // so the output bits look uniformly distributed.
    let mut x = nanos ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;

    let mut y = x
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(counter);
    y = (y ^ (y >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    y = (y ^ (y >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    y ^= y >> 31;

    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        u32::try_from(x >> 32 & 0xFFFF_FFFF).unwrap_or(u32::MAX),
        u16::try_from(x >> 16 & 0xFFFF).unwrap_or(u16::MAX),
        (x & 0x0FFF),
        0x8000 | (u16::try_from(y >> 48 & 0xFFFF).unwrap_or(u16::MAX) & 0x3FFF),
        y & 0xFFFF_FFFF_FFFF
    )
}

/// Get current time in ISO 8601 format.
fn iso8601_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = now.as_secs();

    // Simple date calculation (not accounting for leap seconds, etc.)
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Approximate year/month/day (simplified)
    let mut year = 1970;
    let mut remaining_days = i64::try_from(days).unwrap_or(i64::MAX);

    while remaining_days >= 365 {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days >= days_in_year {
            remaining_days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days_in_month in &month_days {
        if remaining_days >= i64::from(days_in_month) {
            remaining_days -= i64::from(days_in_month);
            month += 1;
        } else {
            break;
        }
    }

    let day = remaining_days + 1;

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

const fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdfa_level() {
        assert_eq!(PdfALevel::A1b.part(), 1);
        assert_eq!(PdfALevel::A2b.part(), 2);
        assert_eq!(PdfALevel::A3b.part(), 3);

        assert!(!PdfALevel::A1b.allows_transparency());
        assert!(PdfALevel::A2b.allows_transparency());

        assert!(!PdfALevel::A2b.allows_embedded_files());
        assert!(PdfALevel::A3b.allows_embedded_files());
    }

    #[test]
    fn test_xmp_generation() {
        let xmp = XmpMetadata::new()
            .with_title("Test Document")
            .with_author("Test Author")
            .with_pdfa_level(PdfALevel::A1b);

        let xml = xmp.to_xmp();
        assert!(xml.contains("Test Document"));
        assert!(xml.contains("Test Author"));
        assert!(xml.contains("pdfaid:part>1"));
        assert!(xml.contains("pdfaid:conformance>B"));
    }

    #[test]
    fn test_validator_missing_metadata() {
        let doc = PdfADocument::new();
        let mut validator = PdfAValidator::new(PdfALevel::A1b);

        let result = validator.validate(&doc);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.code == PdfAErrorCode::MissingXmpMetadata)
        );
    }

    #[test]
    fn test_validator_font_not_embedded() {
        let mut doc = PdfADocument::new();
        doc.create_xmp_metadata(PdfALevel::A1b);
        doc.document_id = Some("test".to_string());

        doc.register_font(
            "Helvetica",
            PdfAFontInfo {
                is_embedded: false,
                has_cmap: true,
                has_widths: true,
            },
        );

        let mut validator = PdfAValidator::new(PdfALevel::A1b);
        let result = validator.validate(&doc);

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.code == PdfAErrorCode::FontNotEmbedded)
        );
    }

    #[test]
    fn test_validator_transparency_a1() {
        let mut doc = PdfADocument::new();
        doc.create_xmp_metadata(PdfALevel::A1b);
        doc.document_id = Some("test".to_string());
        doc.features.content.uses_transparency = true;

        let mut validator = PdfAValidator::new(PdfALevel::A1b);
        let result = validator.validate(&doc);

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.code == PdfAErrorCode::TransparencyNotAllowed)
        );
    }

    #[test]
    fn test_validator_transparency_a2() {
        let mut doc = PdfADocument::new();
        doc.create_xmp_metadata(PdfALevel::A2b);
        doc.document_id = Some("test".to_string());
        doc.features.content.uses_transparency = true;

        let mut validator = PdfAValidator::new(PdfALevel::A2b);
        let result = validator.validate(&doc);

        // Should pass - transparency allowed in PDF/A-2
        assert!(
            result.is_ok()
                || !result
                    .unwrap_err()
                    .iter()
                    .any(|e| e.code == PdfAErrorCode::TransparencyNotAllowed)
        );
    }

    fn base_pdfa_doc(level: PdfALevel) -> PdfADocument {
        let mut doc = PdfADocument::new();
        doc.create_xmp_metadata(level);
        doc.document_id = Some("test".to_string());
        doc
    }

    #[test]
    fn test_validator_missing_document_id() {
        let mut doc = PdfADocument::new();
        doc.create_xmp_metadata(PdfALevel::A1b);
        doc.document_id = None;

        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::MissingDocumentId)
        );
    }

    #[test]
    fn test_validator_missing_pdfa_id() {
        let mut doc = PdfADocument::new();
        // XMP present but with no pdfa_level.
        doc.xmp_metadata = Some(XmpMetadata::new());
        doc.document_id = Some("id".to_string());

        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(errs.iter().any(|e| e.code == PdfAErrorCode::MissingPdfaId));
    }

    #[test]
    fn test_validator_font_missing_cmap() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.register_font(
            "NoCMap",
            PdfAFontInfo {
                is_embedded: true,
                has_cmap: false,
                has_widths: true,
            },
        );
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::FontMissingCmap)
        );
    }

    #[test]
    fn test_validator_font_missing_widths() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.register_font(
            "NoWidths",
            PdfAFontInfo {
                is_embedded: true,
                has_cmap: true,
                has_widths: false,
            },
        );
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::FontMissingWidths)
        );
    }

    #[test]
    fn test_validator_missing_output_intent() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.features.content.uses_device_colors = true;
        // output_intent deliberately None.
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::MissingOutputIntent)
        );
    }

    #[test]
    fn test_validator_uncalibrated_color_space() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.uncalibrated_colors.insert("DeviceN-Foo".to_string());
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::UncalibratedColorSpace)
        );
    }

    #[test]
    fn test_validator_jpeg2000_rejected_in_a1() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.features.compression.uses_jpeg2000 = true;
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::Jpeg2000NotAllowed)
        );
    }

    #[test]
    fn test_validator_jpeg2000_allowed_in_a2() {
        let mut doc = base_pdfa_doc(PdfALevel::A2b);
        doc.features.compression.uses_jpeg2000 = true;
        let mut v = PdfAValidator::new(PdfALevel::A2b);
        let result = v.validate(&doc);
        assert!(
            result.is_ok()
                || !result
                    .unwrap_err()
                    .iter()
                    .any(|e| e.code == PdfAErrorCode::Jpeg2000NotAllowed)
        );
    }

    #[test]
    fn test_validator_lzw_always_rejected() {
        let mut doc = base_pdfa_doc(PdfALevel::A2b);
        doc.features.compression.uses_lzw = true;
        let mut v = PdfAValidator::new(PdfALevel::A2b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::LzwCompressionNotAllowed)
        );
    }

    #[test]
    fn test_validator_structure_required_in_a1a() {
        let doc = base_pdfa_doc(PdfALevel::A1a);
        // has_structure defaults to false.
        let mut v = PdfAValidator::new(PdfALevel::A1a);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::MissingDocumentStructure)
        );
    }

    #[test]
    fn test_validator_encryption_rejected() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.features.security.is_encrypted = true;
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::EncryptionNotAllowed)
        );
    }

    #[test]
    fn test_validator_javascript_rejected() {
        let mut doc = base_pdfa_doc(PdfALevel::A1b);
        doc.features.security.has_javascript = true;
        let mut v = PdfAValidator::new(PdfALevel::A1b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::JavaScriptNotAllowed)
        );
    }

    #[test]
    fn test_validator_embedded_files_rejected_in_a2() {
        let mut doc = base_pdfa_doc(PdfALevel::A2b);
        doc.embedded_files.push(EmbeddedFileInfo {
            name: "x.bin".into(),
            mime_type: None,
            relationship: None,
        });
        let mut v = PdfAValidator::new(PdfALevel::A2b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::EmbeddedFilesNotAllowed)
        );
    }

    #[test]
    fn test_validator_a3_requires_file_relationship() {
        let mut doc = base_pdfa_doc(PdfALevel::A3b);
        doc.embedded_files.push(EmbeddedFileInfo {
            name: "x.bin".into(),
            mime_type: None,
            relationship: None,
        });
        let mut v = PdfAValidator::new(PdfALevel::A3b);
        let errs = v.validate(&doc).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.code == PdfAErrorCode::MissingFileRelationship)
        );
    }

    #[test]
    fn test_validator_a3_accepts_file_with_relationship() {
        let mut doc = base_pdfa_doc(PdfALevel::A3b);
        doc.embedded_files.push(EmbeddedFileInfo {
            name: "x.bin".into(),
            mime_type: Some("application/octet-stream".into()),
            relationship: Some("Data".into()),
        });
        let mut v = PdfAValidator::new(PdfALevel::A3b);
        let result = v.validate(&doc);
        assert!(
            result.is_ok()
                || !result
                    .unwrap_err()
                    .iter()
                    .any(|e| e.code == PdfAErrorCode::MissingFileRelationship)
        );
    }

    #[test]
    fn test_uuid_v4_is_unique_in_tight_loop() {
        use std::collections::HashSet;
        let mut seen: HashSet<String> = HashSet::new();
        for _ in 0..1_000 {
            let id = uuid_v4();
            assert_eq!(id.len(), 36);
            assert!(seen.insert(id), "duplicate UUID in tight loop");
        }
    }
}
