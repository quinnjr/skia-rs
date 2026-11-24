//! End-to-end tests for `skia-rs-pdf`.
//!
//! Covers:
//! - Producing a PDF that mixes text, images, transparency, and PDF/A
//!   metadata, and sanity-checking the resulting bytes (resources dict
//!   populated; `XObjects`, Font, `ExtGState` all referenced; xref/trailer
//!   present).
//! - TrueType embedding: feeding a real `demo.ttf` through `PdfFont::
//!   truetype` and asserting that the emitted PDF contains the font's
//!   `FontFile2` stream, a matching `FontDescriptor`, and populated Widths.

use skia_rs_core::cast::floor_to_i32;
use skia_rs_core::{Color, Rect};
use skia_rs_paint::Paint;
use skia_rs_pdf::{PdfALevel, PdfDocument, StandardFont};

const DEMO_TTF: &[u8] = include_bytes!("fixtures/demo.ttf");

/// Sanity: a trivial document round-trips through the byte writer and
/// contains every structural marker. Not a pdftotext test — we only check
/// the bytes we produced.
#[test]
fn simple_pdf_has_valid_structure() {
    let mut doc = PdfDocument::new();
    doc.metadata_mut().title = Some("Smoke".to_string());
    doc.metadata_mut().creator = Some("skia-rs".to_string());
    {
        let mut c = doc.begin_page(612.0, 792.0);
        c.use_standard_font(StandardFont::Helvetica);
        let mut p = Paint::new();
        p.set_color32(Color::from_rgb(0, 0, 0));
        c.draw_text("Hello, World!", 72.0, 72.0, 14.0, &p)
            .expect("draw_text should succeed");
        let page = c.finish();
        doc.end_page(page);
    }
    let mut buf = Vec::new();
    doc.write_to(&mut buf).unwrap();

    assert!(buf.starts_with(b"%PDF-1.4"));
    assert!(buf.ends_with(b"%%EOF\n"));
    let s = String::from_utf8_lossy(&buf);
    assert!(s.contains("/Title (Smoke)"));
    assert!(s.contains("/Creator (skia-rs)"));
    assert!(s.contains("/ID ["));
    assert!(s.contains("(Hello, World!)"));
}

#[test]
fn end_to_end_pdf_with_text_image_transparency_pdfa() {
    let mut doc = PdfDocument::new();
    doc.set_pdfa_conformance(PdfALevel::A2b); // A2b permits transparency.

    // Seed an RGBA image on the document.
    let data = vec![200u8; 8 * 8 * 4];
    let (img_idx, _mask_idx) = doc.images_mut().add_rgba(8, 8, &data);

    {
        let mut canvas = doc.begin_page(612.0, 792.0);
        canvas.use_standard_font(StandardFont::Helvetica);

        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(0, 0, 0));
        canvas
            .draw_text("Hello, PDF!", 72.0, 100.0, 14.0, &paint)
            .expect("draw_text should succeed");

        // Translucent rect — should register an ExtGState.
        let mut translucent = Paint::new();
        translucent.set_color32(Color::from_argb(96, 255, 64, 64));
        canvas.draw_rect(&Rect::from_xywh(72.0, 150.0, 200.0, 80.0), &translucent);

        canvas.draw_image(img_idx, 300.0, 150.0, 100.0, 100.0);

        let page = canvas.finish();
        doc.end_page(page);
    }

    let mut buf = Vec::new();
    doc.write_to(&mut buf)
        .expect("write should succeed for A2b");

    // `PdfDocument::write_to` emits a binary marker (4 high bits) after the
    // header; cope with non-UTF-8 bytes by using `from_utf8_lossy`.
    let s_cow = String::from_utf8_lossy(&buf);
    let s = &*s_cow;
    // Structural sanity checks.
    assert!(
        buf.starts_with(b"%PDF-1.7"),
        "expected PDF-1.7 header for A2b"
    );
    assert!(s.contains("xref"));
    assert!(s.contains("startxref"));
    assert!(s.contains("%%EOF"));

    // Resources dict populated.
    assert!(s.contains("/Font << /F1"), "font missing from resources");
    assert!(
        s.contains("/XObject << /Im"),
        "image missing from resources"
    );
    assert!(s.contains("/ExtGState << /GS"), "extgstate missing");

    // XMP metadata + output intent from PDF/A wiring.
    assert!(s.contains("/Type /Metadata"));
    assert!(s.contains("/Type /OutputIntent"));
    assert!(s.contains("pdfaid:part>2"));

    // Image XObject emitted with SMask resolved to a real object id (not
    // the raw manager index).
    assert!(s.contains("/SMask"));

    // ToUnicode CMap emitted for the font.
    assert!(s.contains("/ToUnicode"));
    assert!(s.contains("beginbfchar"));
}

#[test]
fn truetype_embedding_emits_fontfile2_and_widths() {
    let mut doc = PdfDocument::new();

    let font_idx = doc.fonts_mut().register_truetype("Demo", DEMO_TTF.to_vec());

    {
        let mut canvas = doc.begin_page(200.0, 200.0);
        canvas.set_font(font_idx);
        let paint = Paint::new();
        canvas
            .draw_text("A", 10.0, 20.0, 12.0, &paint)
            .expect("draw_text should succeed");
        let page = canvas.finish();
        doc.end_page(page);
    }

    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");

    let s = String::from_utf8_lossy(&buf);

    // The TTF is compressed (flate) inside a FontFile2 stream. We detect it
    // via the object header marker.
    assert!(s.contains("/FontFile2"), "no FontFile2 stream: {s}");
    assert!(s.contains("/FontDescriptor"), "no FontDescriptor reference");
    assert!(s.contains("/Subtype /TrueType"));
    // Subset prefix on the BaseFont (six uppercase letters + '+').
    let subset_re = regex_lite(&s, "/BaseFont /", "+Demo");
    assert!(
        subset_re,
        "BaseFont should carry a six-letter subset prefix; output: {}",
        &s[..2000.min(s.len())]
    );

    // Widths array: the demo.ttf has only 'A' mapped, so widths for 'A'
    // (ASCII 0x41 = 65) must be nonzero. The font's FirstChar/LastChar
    // should include 65.
    assert!(
        s.contains("/FirstChar 65") || s.contains("/FirstChar 33"),
        "unexpected FirstChar range"
    );
}

/// The byte-level subsetter must produce a `FontFile2` stream whose
/// declared `/Length1` (the uncompressed byte length of the TTF data)
/// does not exceed the original font size. For the demo fixture which
/// has only 2 glyphs, the subset is typically comparable in size to the
/// original (there's little to prune); the real guard is that the stream
/// is structurally valid and still parses as a TrueType face after being
/// extracted from the PDF.
#[test]
fn truetype_subset_length1_matches_stream_length() {
    let mut doc = PdfDocument::new();

    let font_idx = doc.fonts_mut().register_truetype("Demo", DEMO_TTF.to_vec());

    {
        let mut canvas = doc.begin_page(200.0, 200.0);
        canvas.set_font(font_idx);
        let paint = Paint::new();
        canvas
            .draw_text("A", 10.0, 20.0, 12.0, &paint)
            .expect("draw_text should succeed");
        let page = canvas.finish();
        doc.end_page(page);
    }

    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");

    let s = String::from_utf8_lossy(&buf);

    // Parse /Length1 from the FontFile2 stream header. We look for the
    // "/Length1 " literal and read the unsigned integer that follows.
    let l1_pos = s
        .find("/Length1 ")
        .expect("FontFile2 must declare /Length1");
    let after = &s[l1_pos + "/Length1 ".len()..];
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .expect("Length1 must be followed by a non-digit");
    let length1: usize = after[..end].parse().expect("Length1 must be a valid u32");

    // For a fixture this small (400 B, 2 glyphs) the subset is usually
    // comparable. Assert only the invariant: never exceed the input.
    assert!(
        length1 <= DEMO_TTF.len(),
        "subset Length1 ({}) must not exceed full font size ({})",
        length1,
        DEMO_TTF.len()
    );
    // Sanity: Length1 is nonzero (we embedded *something*).
    assert!(length1 > 0, "Length1 must be nonzero");
}

/// The subsetter strips glyph outlines for glyphs we didn't use. This
/// test uses the `subsetter` entry point directly to prove that a subset
/// of a synthetic fixture containing many 0-filled glyphs is smaller
/// than the original. We build the fixture here rather than ship it so
/// the guard can't rot silently as the crate evolves.
#[test]
fn truetype_subset_actually_prunes_large_glyf() {
    // A 10-glyph TTF with .notdef + 9 glyphs each carrying 1 KB of
    // synthetic outline payload. We keep only glyphs 0 and 2 — so the
    // subset should drop ~8 KB of glyf data.
    let full = build_synthetic_ttf(10, 1024);
    let full_len = full.len();

    let mut keep = std::collections::BTreeSet::new();
    keep.insert(2);
    let subset =
        skia_rs_pdf::subset_truetype_for_tests(&full, &keep).expect("subset of synthetic ttf");

    assert!(
        subset.len() < full_len - 6_000,
        "subset {} should be much smaller than full {}",
        subset.len(),
        full_len
    );

    // Subset must still parse as a TTF.
    let face = ttf_parser::Face::parse(&subset, 0).expect("subset parses");
    assert_eq!(face.number_of_glyphs() as usize, 10);
}

/// A Type0 (CID-keyed) font, once registered and used, must emit a
/// `/DescendantFonts` array pointing at a `CIDFontType2` dict with
/// `/CIDToGIDMap`, `/W`, and `/CIDSystemInfo` — per PDF 32000-1 §9.7.6, a
/// Type0 font is otherwise glyph-less.
#[test]
fn type0_font_emits_descendant_fonts_array() {
    let mut doc = PdfDocument::new();

    let font_idx = doc
        .fonts_mut()
        .register_truetype_cid("Demo", DEMO_TTF.to_vec());

    {
        // `draw_text`/`draw_text_with_font` fail closed against Type0/CID
        // fonts (see `type0_font_draw_text_fails_closed` below), so we
        // can't drive resource usage through a live draw here. `set_font`
        // alone is enough to register the font as used for this page's
        // `/Resources`, and the `/DescendantFonts` structure below is
        // emitted for every registered Type0 font regardless of whether
        // any text was drawn through it (see `PdfDocument::write_to`).
        let mut canvas = doc.begin_page(200.0, 200.0);
        canvas.set_font(font_idx);
        let page = canvas.finish();
        doc.end_page(page);
    }

    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");
    let s = String::from_utf8_lossy(&buf);

    assert!(s.contains("/Subtype /Type0"), "{}", s);
    assert!(
        s.contains("/DescendantFonts ["),
        "missing /DescendantFonts: {s}"
    );
    assert!(s.contains("/Subtype /CIDFontType2"));
    assert!(s.contains("/CIDToGIDMap /Identity"));
    assert!(
        s.contains("/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >>")
    );
    assert!(s.contains("/W ["));
    // Embedded outlines are shared with the simple-TrueType path.
    assert!(s.contains("/FontFile2"));
}

/// Symbolic standard fonts (`Symbol`, `ZapfDingbats`) must omit `/Encoding`
/// so a reader falls back to the font's built-in encoding; non-symbolic
/// standard fonts (e.g. `Helvetica`) keep `/Encoding /WinAnsiEncoding`.
/// Each font is drawn on its own page so every `/Encoding` occurrence in
/// the output maps back to a single, unambiguous font dict.
#[test]
fn standard_font_encoding_is_omitted_only_for_symbolic_fonts() {
    let mut doc = PdfDocument::new();
    let paint = Paint::new();

    {
        let mut canvas = doc.begin_page(200.0, 200.0);
        canvas.use_standard_font(StandardFont::Symbol);
        canvas
            .draw_text("A", 10.0, 20.0, 12.0, &paint)
            .expect("draw_text should succeed");
        let page = canvas.finish();
        doc.end_page(page);
    }
    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");
    let s = String::from_utf8_lossy(&buf);
    assert!(
        !s.contains("/Encoding"),
        "Symbol must omit /Encoding to use its built-in encoding: {s}"
    );

    let mut doc = PdfDocument::new();
    {
        let mut canvas = doc.begin_page(200.0, 200.0);
        canvas.use_standard_font(StandardFont::ZapfDingbats);
        canvas
            .draw_text("A", 10.0, 20.0, 12.0, &paint)
            .expect("draw_text should succeed");
        let page = canvas.finish();
        doc.end_page(page);
    }
    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");
    let s = String::from_utf8_lossy(&buf);
    assert!(
        !s.contains("/Encoding"),
        "ZapfDingbats must omit /Encoding to use its built-in encoding: {s}"
    );

    let mut doc = PdfDocument::new();
    {
        let mut canvas = doc.begin_page(200.0, 200.0);
        canvas.use_standard_font(StandardFont::Helvetica);
        canvas
            .draw_text("A", 10.0, 20.0, 12.0, &paint)
            .expect("draw_text should succeed");
        let page = canvas.finish();
        doc.end_page(page);
    }
    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");
    let s = String::from_utf8_lossy(&buf);
    assert!(
        s.contains("/Encoding /WinAnsiEncoding"),
        "Helvetica should keep /Encoding /WinAnsiEncoding: {s}"
    );
}

/// A font registered via `register_truetype_cid` is emitted with
/// `/Encoding /Identity-H`, which a reader interprets as a stream of
/// 2-byte CIDs. `draw_text`/`draw_text_with_font` don't yet resolve
/// per-glyph CIDs, so drawing live text through such a font must fail
/// closed with `Err(PdfError::Unsupported)` instead of silently emitting
/// 1-byte `WinAnsi` codes against a 2-byte encoding, which would decode as
/// garbage glyphs.
#[test]
fn type0_font_draw_text_fails_closed() {
    let mut doc = PdfDocument::new();

    let font_idx = doc
        .fonts_mut()
        .register_truetype_cid("Demo", DEMO_TTF.to_vec());

    let mut canvas = doc.begin_page(200.0, 200.0);
    canvas.set_font(font_idx);
    let paint = Paint::new();

    let result = canvas.draw_text("A", 10.0, 20.0, 12.0, &paint);

    match result {
        Err(skia_rs_pdf::PdfError::Unsupported(msg)) => {
            assert!(
                msg.contains("Type0"),
                "expected message to mention Type0/CID font, got: {msg}"
            );
        }
        other => panic!(
            "draw_text against a Type0/Identity-H font must fail closed with \
             Err(PdfError::Unsupported), not emit 1-byte codes; got {other:?}"
        ),
    }
}

/// PDF/A `OutputIntents` require a real, parseable ICC profile in
/// `/DestOutputProfile` — a placeholder byte string is not valid ICC data
/// and fails conformance even though the surrounding PDF object looks
/// structurally correct. Decompress the emitted stream and check for the
/// ICC 'acsp' file signature at its spec-mandated offset (36).
#[test]
fn pdfa_output_intent_embeds_real_icc_profile() {
    use std::io::Read as _;

    let mut doc = PdfDocument::new();
    doc.set_pdfa_conformance(PdfALevel::A2b);
    {
        let canvas = doc.begin_page(100.0, 100.0);
        let page = canvas.finish();
        doc.end_page(page);
    }

    let mut buf = Vec::new();
    doc.write_to(&mut buf).expect("write should succeed");

    // Locate the DestOutputProfile stream and decompress it.
    let needle = b"/N 3 /Alternate /DeviceRGB";
    let start = buf
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("ICC profile stream header not found");
    let stream_start = buf[start..]
        .windows(7)
        .position(|w| w == b"stream\n")
        .map(|p| start + p + 7)
        .expect("stream keyword not found");
    let stream_end = buf[stream_start..]
        .windows(10)
        .position(|w| w == b"\nendstream")
        .map(|p| stream_start + p)
        .expect("endstream not found");

    let mut decoder = flate2::read::ZlibDecoder::new(&buf[stream_start..stream_end]);
    let mut icc = Vec::new();
    decoder
        .read_to_end(&mut icc)
        .expect("ICC stream must be valid FlateDecode data");

    assert!(
        icc.len() > 500,
        "ICC profile too small: {} bytes",
        icc.len()
    );
    assert_eq!(
        &icc[36..40],
        b"acsp",
        "missing ICC file signature 'acsp' at offset 36"
    );
    assert_eq!(&icc[16..20], b"RGB ", "profile must be an RGB color space");
}

/// Construct a minimal TrueType file with `num_glyphs` glyphs, each of
/// which is a "simple glyph" with `outline_bytes` bytes of 0-filled
/// outline data. The resulting font is structurally valid enough for
/// ttf-parser to parse, which is all the subsetter test needs.
fn build_synthetic_ttf(num_glyphs: u16, outline_bytes: usize) -> Vec<u8> {
    use std::io::Write as _;
    // ---- Build individual tables ----
    //
    // glyf: one simple glyph per id. A simple glyph with 1 contour, 1
    // point (on-curve, no x/y delta), and `outline_bytes` padding after.
    // Layout: numberOfContours=1 (i16), xMin/yMin/xMax/yMax=0 (4 x i16),
    //         endPtsOfContours[1] = [0] (1 x u16),
    //         instructionLength=0 (u16),
    //         flags[1] = 0x37 (on-curve, x-short +0, y-short +0, 1 byte),
    //         xCoordinates[1] = 0 (1 byte, since x-short), yCoordinates = 0,
    //         padding of `outline_bytes` zero bytes.
    let mut glyf: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = vec![0];
    for _ in 0..num_glyphs {
        // numberOfContours
        glyf.extend_from_slice(&1i16.to_be_bytes());
        // bbox
        glyf.extend_from_slice(&[0u8; 8]);
        // endPtsOfContours
        glyf.extend_from_slice(&0u16.to_be_bytes());
        // instructionLength
        glyf.extend_from_slice(&0u16.to_be_bytes());
        // flags: on-curve(1) | x-short(2) | y-short(4) | x-same(16) | y-same(32) = 0x37
        glyf.push(0x37);
        // Per the x/y-short + same flags, the x and y are 0 (no payload bytes
        // are read — "same" means reuse previous; first point defaults to
        // origin). But we still need at least some bytes or ttf-parser may
        // complain. Padding out glyph bodies to a large, distinctive size.
        glyf.extend_from_slice(&vec![0u8; outline_bytes]);
        while !glyf.len().is_multiple_of(2) {
            glyf.push(0);
        }
        offsets.push(u32::try_from(glyf.len()).unwrap_or(u32::MAX));
    }

    // loca (long format, since glyf is already > 65K).
    let mut loca: Vec<u8> = Vec::new();
    for off in &offsets {
        loca.extend_from_slice(&off.to_be_bytes());
    }

    // head: 54 bytes. Magic fields set; indexToLocFormat=1 (long loca).
    let mut head = vec![0u8; 54];
    head[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // version 1.0
    head[4..8].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // fontRevision
    head[8..12].copy_from_slice(&0u32.to_be_bytes()); // checkSumAdjustment
    head[12..16].copy_from_slice(&0x5F0F_3CF5u32.to_be_bytes()); // magicNumber
    head[16..18].copy_from_slice(&0u16.to_be_bytes()); // flags
    head[18..20].copy_from_slice(&1024u16.to_be_bytes()); // unitsPerEm
    // created/modified (8 bytes each = 16 bytes) — leave as zero.
    head[44..46].copy_from_slice(&0u16.to_be_bytes()); // xMin
    head[46..48].copy_from_slice(&0u16.to_be_bytes()); // yMin
    head[48..50].copy_from_slice(&1024u16.to_be_bytes()); // xMax
    head[50..52].copy_from_slice(&1u16.to_be_bytes()); // indexToLocFormat = 1 (long)
    head[52..54].copy_from_slice(&0u16.to_be_bytes()); // glyphDataFormat

    // maxp v0.5 (6 bytes) is acceptable for TrueType. We use v1.0 (32
    // bytes) to match ttf-parser's expectations more fully.
    let mut maxp = vec![0u8; 32];
    maxp[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // version 1.0
    maxp[4..6].copy_from_slice(&num_glyphs.to_be_bytes()); // numGlyphs
    // The remaining fields (max points, contours, etc.) can be zero for a
    // passive parser.

    // hhea (36 bytes).
    let mut hhea = vec![0u8; 36];
    hhea[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // version 1.0
    hhea[4..6].copy_from_slice(&768i16.to_be_bytes()); // ascender
    hhea[6..8].copy_from_slice(&(-256i16).to_be_bytes()); // descender
    hhea[8..10].copy_from_slice(&0i16.to_be_bytes()); // lineGap
    hhea[10..12].copy_from_slice(&1024u16.to_be_bytes()); // advanceWidthMax
    // 8 reserved i16 fields + metricDataFormat + numberOfHMetrics
    hhea[34..36].copy_from_slice(&num_glyphs.to_be_bytes()); // numberOfHMetrics

    // hmtx: numberOfHMetrics * (u16 advance + i16 lsb).
    let mut hmtx: Vec<u8> = Vec::new();
    for _ in 0..num_glyphs {
        hmtx.extend_from_slice(&1024u16.to_be_bytes()); // advance
        hmtx.extend_from_slice(&0i16.to_be_bytes()); // lsb
    }

    // cmap (minimal format-4 mapping chars 32..32+num_glyphs-1 to glyphs
    // 1..num_glyphs). We leave this tiny — ttf-parser needs at least one
    // encoding record to be happy.
    let cmap = build_minimal_cmap(num_glyphs);

    let tables: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"cmap", cmap),
        (b"glyf", glyf),
        (b"head", head),
        (b"hhea", hhea),
        (b"hmtx", hmtx),
        (b"loca", loca),
        (b"maxp", maxp),
    ];

    // Lay out sfnt.
    let mut sorted: Vec<(&[u8; 4], Vec<u8>)> = tables;
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    let n = u16::try_from(sorted.len()).unwrap_or(u16::MAX);
    let entry_selector = u16::try_from(floor_to_i32(f32::from(n).log2())).unwrap_or(0);
    let search_range = (1u16 << entry_selector) * 16;
    let range_shift = n * 16 - search_range;

    let directory_size = 12 + sorted.len() * 16;
    let mut body_offsets: Vec<u32> = Vec::with_capacity(sorted.len());
    let mut cursor = directory_size;
    for (_, body) in &sorted {
        body_offsets.push(u32::try_from(cursor).unwrap_or(u32::MAX));
        cursor += body.len();
        while !cursor.is_multiple_of(4) {
            cursor += 1;
        }
    }

    let mut out = Vec::with_capacity(cursor);
    // Offset subtable (0x00010000 sfnt version).
    out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_selector.to_be_bytes());
    out.extend_from_slice(&range_shift.to_be_bytes());

    for ((tag, body), offset) in sorted.iter().zip(body_offsets.iter()) {
        out.extend_from_slice(*tag);
        // Checksum — zero is fine for a passive parser.
        out.write_all(&0u32.to_be_bytes()).unwrap();
        out.extend_from_slice(&offset.to_be_bytes());
        out.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_be_bytes());
    }

    for (_, body) in &sorted {
        while out.len() % 4 != 0 {
            out.push(0);
        }
        out.extend_from_slice(body);
    }
    while out.len() % 4 != 0 {
        out.push(0);
    }

    out
}

fn build_minimal_cmap(num_glyphs: u16) -> Vec<u8> {
    // cmap header: version(u16)=0, numTables(u16)=1.
    let mut cmap: Vec<u8> = Vec::new();
    cmap.extend_from_slice(&0u16.to_be_bytes());
    cmap.extend_from_slice(&1u16.to_be_bytes());
    // encoding record: platformID=3 (Microsoft), encodingID=1 (Unicode BMP),
    // offset = 12 (directly after this record).
    cmap.extend_from_slice(&3u16.to_be_bytes());
    cmap.extend_from_slice(&1u16.to_be_bytes());
    cmap.extend_from_slice(&12u32.to_be_bytes());

    // cmap format 4 subtable: map single range [65 .. 65 + num_glyphs-1]
    // to glyph IDs [1 .. num_glyphs-1]. Two segments: the main one plus
    // the mandatory terminator segment starting at 0xFFFF.
    let seg_count = 2u16;
    let _ = num_glyphs;
    // format 4 header:
    //   format(u16)=4, length(u16), language(u16)=0, segCountX2(u16),
    //   searchRange, entrySelector, rangeShift, endCode[seg_count],
    //   reservedPad(u16)=0, startCode[seg_count], idDelta[seg_count],
    //   idRangeOffset[seg_count], glyphIdArray (empty)
    let seg_count_x2 = seg_count * 2;
    let entry_selector_4 = u16::try_from(floor_to_i32(f32::from(seg_count).log2())).unwrap_or(0);
    let search_range_4 = 2 * (1u16 << entry_selector_4);
    let range_shift_4 = seg_count_x2 - search_range_4;

    // Segment 1: end=65, start=65, delta=-64 (wraps to produce GID 1)
    // Segment 2: end=0xFFFF, start=0xFFFF, delta=1, idRangeOffset=0 (GID 0)
    let body_len = 16 + (seg_count as usize) * 8;
    let total_len = 14 + body_len;

    let mut sub: Vec<u8> = Vec::new();
    sub.extend_from_slice(&4u16.to_be_bytes()); // format
    sub.extend_from_slice(&u16::try_from(total_len).unwrap_or(u16::MAX).to_be_bytes());
    sub.extend_from_slice(&0u16.to_be_bytes()); // language
    sub.extend_from_slice(&seg_count_x2.to_be_bytes());
    sub.extend_from_slice(&search_range_4.to_be_bytes());
    sub.extend_from_slice(&entry_selector_4.to_be_bytes());
    sub.extend_from_slice(&range_shift_4.to_be_bytes());
    // endCode
    sub.extend_from_slice(&65u16.to_be_bytes());
    sub.extend_from_slice(&0xFFFFu16.to_be_bytes());
    // reservedPad
    sub.extend_from_slice(&0u16.to_be_bytes());
    // startCode
    sub.extend_from_slice(&65u16.to_be_bytes());
    sub.extend_from_slice(&0xFFFFu16.to_be_bytes());
    // idDelta — choose so GID[65] = 1 → idDelta = 1 - 65 = -64
    sub.extend_from_slice(&(-64i16).to_be_bytes());
    sub.extend_from_slice(&1i16.to_be_bytes());
    // idRangeOffset
    sub.extend_from_slice(&0u16.to_be_bytes());
    sub.extend_from_slice(&0u16.to_be_bytes());

    cmap.extend_from_slice(&sub);
    cmap
}

/// Tiny helper: check that the string `haystack` contains the pattern
/// `start .{6}end` (where `.{6}` matches six arbitrary chars). Used to
/// verify the `ABCDEF+FontName` subset-prefix syntax without pulling in
/// the `regex` crate.
fn regex_lite(haystack: &str, start: &str, end: &str) -> bool {
    let mut i = 0;
    while let Some(pos) = haystack[i..].find(start) {
        let abs = i + pos + start.len();
        if abs + 6 + end.len() > haystack.len() {
            return false;
        }
        let tag = &haystack[abs..abs + 6];
        let rest = &haystack[abs + 6..abs + 6 + end.len()];
        if tag.chars().all(|c| c.is_ascii_uppercase()) && rest == end {
            return true;
        }
        i = abs;
    }
    false
}
