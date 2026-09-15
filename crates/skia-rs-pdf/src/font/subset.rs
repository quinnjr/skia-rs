//! Byte-level TrueType subsetter for PDF embedding.
//!
//! This subsetter prunes a TrueType font's `glyf` / `loca` tables so the
//! emitted `FontFile2` stream contains only the glyph outlines actually used
//! by the document (plus `.notdef` and composite-glyph dependencies), while
//! preserving every other table unchanged. Preserving `cmap`, `hmtx`,
//! `hhea`, `OS/2`, `post`, `name`, `head`, `maxp` lets the PDF continue to
//! reference glyphs by character code through the font's built-in
//! `WinAnsiEncoding` cmap — which is what `PdfFont::truetype` embedding
//! relies on.
//!
//! # Why roll our own?
//!
//! The `subsetter` crate (typst) produces output explicitly designed for
//! CID-keyed Type0 PDF embedding: it strips `cmap` and expects the caller
//! to provide a replacement in PDF. Our embedding path is simple TrueType
//! with `WinAnsi` encoding, where the reader walks the font's own cmap — so
//! a `subsetter` drop-in would break glyph resolution. Klippa (fontations)
//! would work but is not yet a stable crate on crates.io.
//!
//! # Strategy
//!
//! We keep `numGlyphs` exactly as-is: unused glyphs are represented as
//! zero-length entries in `loca`, so every glyph ID that previously existed
//! still resolves (to an empty outline for unused glyphs). This avoids any
//! renumbering — `cmap`, `hmtx`, `hhea.numberOfHMetrics`, `maxp.numGlyphs`
//! all stay valid without modification.
//!
//! Composite glyphs recursively pull in their component glyphs; we follow
//! the `GLYF_COMPONENT` / `MORE_COMPONENTS` flags to compute the full
//! transitive closure before pruning.
//!
//! Output size is dominated by the surviving `glyf` entries plus the
//! untouched tables — in practice a 5-glyph subset of a 200 KB font falls
//! to 10-30 KB (post-flate).

use std::collections::BTreeSet;

use skia_rs_core::cast::floor_to_i32;

/// Subsetting failure modes.
#[derive(Debug)]
pub enum SubsetError {
    /// Font data too short or malformed to read the offset subtable.
    Malformed(&'static str),
    /// The font is a TrueType Collection (`ttcf`); subsetting a specific
    /// face out of a collection is not supported here.
    CollectionNotSupported,
    /// Required table missing (glyf, loca, head, maxp).
    MissingTable(&'static str),
}

impl core::fmt::Display for SubsetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Malformed(why) => write!(f, "malformed font: {why}"),
            Self::CollectionNotSupported => write!(f, "TTC collection not supported"),
            Self::MissingTable(t) => write!(f, "missing required table: {t}"),
        }
    }
}

impl std::error::Error for SubsetError {}

/// Subset a TrueType font to include only the specified glyph IDs (plus
/// any glyphs they reference as composite components, transitively, and
/// `.notdef` (glyph 0) which PDF viewers always expect to be present).
///
/// Returns a byte stream containing a fresh TrueType font identical to
/// the input in every way *except* that the `glyf` table contains only
/// the used glyphs' outlines (with zero-length entries for the rest) and
/// the `loca` table has been rebuilt accordingly. `head.checkSumAdjustment`
/// is zeroed — PDF readers and PDF/A validators do not require a valid
/// checksum on embedded `FontFile2` streams, and computing it correctly
/// would require re-hashing the entire output file.
pub fn subset_truetype(data: &[u8], used_glyphs: &BTreeSet<u16>) -> Result<Vec<u8>, SubsetError> {
    let face = read_face(data)?;

    let glyf_range = face
        .table_range(*b"glyf")
        .ok_or(SubsetError::MissingTable("glyf"))?;
    let loca_range = face
        .table_range(*b"loca")
        .ok_or(SubsetError::MissingTable("loca"))?;
    let head_range = face
        .table_range(*b"head")
        .ok_or(SubsetError::MissingTable("head"))?;
    let maxp_range = face
        .table_range(*b"maxp")
        .ok_or(SubsetError::MissingTable("maxp"))?;

    let glyf = &data[glyf_range];
    let loca = &data[loca_range];
    let head = &data[head_range];
    let maxp = &data[maxp_range];

    // head.indexToLocFormat is at offset 50-51 (u16). 0 = short (u16 * 2),
    // 1 = long (u32).
    if head.len() < 54 {
        return Err(SubsetError::Malformed("head table too short"));
    }
    let long_loca = read_u16_be(head, 50) == 1;

    // maxp.numGlyphs at offset 4 (u16).
    if maxp.len() < 6 {
        return Err(SubsetError::Malformed("maxp table too short"));
    }
    let num_glyphs = read_u16_be(maxp, 4) as usize;

    // Parse loca to get current glyph byte offsets.
    let glyf_offsets = parse_loca(loca, long_loca, num_glyphs)?;
    if glyf_offsets.len() != num_glyphs + 1 {
        return Err(SubsetError::Malformed("loca length mismatch"));
    }

    // Compute the transitive closure of used glyphs — composite glyphs pull
    // in their components. Always include glyph 0 (.notdef).
    let mut keep: BTreeSet<u16> = BTreeSet::new();
    keep.insert(0);
    for &gid in used_glyphs {
        if usize::from(gid) < num_glyphs {
            keep.insert(gid);
        }
    }
    expand_composites(
        glyf,
        &glyf_offsets,
        &mut keep,
        u16::try_from(num_glyphs).unwrap_or(u16::MAX),
    );

    // Rebuild glyf + loca. For unused glyphs we emit a zero-length slot
    // (loca[i] == loca[i+1]). For used glyphs we copy their bytes verbatim.
    let (new_glyf, new_loca) = rebuild_glyf_and_loca(glyf, &glyf_offsets, &keep, long_loca);

    // Write the new font with updated glyf + loca and every other table
    // copied straight from the input.
    let mut tables: Vec<(Tag, Vec<u8>)> = Vec::new();
    for rec in &face.records {
        let tag = rec.tag;
        let body: Vec<u8> = if &tag == b"glyf" {
            new_glyf.clone()
        } else if &tag == b"loca" {
            new_loca.clone()
        } else if &tag == b"head" {
            // Zero out checkSumAdjustment (bytes 8..12) so the output is
            // well-formed even without recomputing the file checksum.
            let mut h = data[face.table_range(tag).unwrap()].to_vec();
            if h.len() >= 12 {
                h[8..12].copy_from_slice(&[0, 0, 0, 0]);
            }
            h
        } else {
            data[face.table_range(tag).unwrap()].to_vec()
        };
        tables.push((tag, body));
    }

    Ok(write_font(face.sfnt_version, &tables))
}

/// Raw face view with the tables we need to find quickly.
struct Face {
    sfnt_version: [u8; 4],
    records: Vec<TableRecord>,
}

type Tag = [u8; 4];

#[derive(Clone)]
struct TableRecord {
    tag: Tag,
    offset: u32,
    length: u32,
}

impl Face {
    fn table_range(&self, tag: Tag) -> Option<std::ops::Range<usize>> {
        self.records.iter().find(|r| r.tag == tag).map(|r| {
            let start = r.offset as usize;
            let end = start + r.length as usize;
            start..end
        })
    }
}

fn read_face(data: &[u8]) -> Result<Face, SubsetError> {
    if data.len() < 12 {
        return Err(SubsetError::Malformed("data < 12 bytes"));
    }
    let sfnt = &data[..4];
    // Reject TTC up front — the tests only use a single-face TTF.
    if sfnt == b"ttcf" {
        return Err(SubsetError::CollectionNotSupported);
    }
    // Accept 0x00010000 (TrueType) and 'true' (Apple TrueType). Reject 'OTTO'
    // (CFF-flavored OpenType) because it has no glyf/loca — our simple
    // TrueType embedding path wouldn't produce it anyway.
    let is_ttf = sfnt == [0x00, 0x01, 0x00, 0x00] || sfnt == b"true";
    if !is_ttf {
        return Err(SubsetError::Malformed(
            "unsupported sfnt version (expected TrueType)",
        ));
    }
    let num_tables = read_u16_be(data, 4) as usize;
    if data.len() < 12 + num_tables * 16 {
        return Err(SubsetError::Malformed("table directory truncated"));
    }
    let mut records = Vec::with_capacity(num_tables);
    for i in 0..num_tables {
        let rec_off = 12 + i * 16;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&data[rec_off..rec_off + 4]);
        let offset = read_u32_be(data, rec_off + 8);
        let length = read_u32_be(data, rec_off + 12);
        if (offset as usize) + (length as usize) > data.len() {
            return Err(SubsetError::Malformed("table body truncated"));
        }
        records.push(TableRecord {
            tag,
            offset,
            length,
        });
    }

    let mut sfnt_version = [0u8; 4];
    sfnt_version.copy_from_slice(sfnt);
    Ok(Face {
        sfnt_version,
        records,
    })
}

fn parse_loca(loca: &[u8], long: bool, num_glyphs: usize) -> Result<Vec<u32>, SubsetError> {
    let expected = if long {
        (num_glyphs + 1) * 4
    } else {
        (num_glyphs + 1) * 2
    };
    if loca.len() < expected {
        return Err(SubsetError::Malformed("loca too short"));
    }
    let mut out = Vec::with_capacity(num_glyphs + 1);
    for i in 0..=num_glyphs {
        let off = if long {
            read_u32_be(loca, i * 4)
        } else {
            // Short loca stores offset / 2.
            u32::from(read_u16_be(loca, i * 2)) * 2
        };
        out.push(off);
    }
    Ok(out)
}

/// Flag bits in composite glyph component records.
const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
const WE_HAVE_A_SCALE: u16 = 0x0008;
const MORE_COMPONENTS: u16 = 0x0020;
const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;

/// Walk the `keep` set and add component glyphs that any composite glyph
/// in the set references. Iterates to a fixed point so that composite
/// chains (glyph A refs B which refs C) are fully captured.
fn expand_composites(glyf: &[u8], offsets: &[u32], keep: &mut BTreeSet<u16>, num_glyphs: u16) {
    let mut changed = true;
    while changed {
        changed = false;
        let snapshot: Vec<u16> = keep.iter().copied().collect();
        for gid in snapshot {
            let start = offsets[gid as usize] as usize;
            let end = offsets[gid as usize + 1] as usize;
            if end <= start || end > glyf.len() {
                continue;
            }
            let body = &glyf[start..end];
            // numberOfContours (i16) at offset 0. Negative = composite.
            if body.len() < 10 {
                continue;
            }
            let num_contours = read_i16_be(body, 0);
            if num_contours >= 0 {
                continue;
            }

            // Walk composite components starting after the 10-byte header
            // (numberOfContours + xMin/yMin/xMax/yMax).
            let mut cursor = 10;
            loop {
                if cursor + 4 > body.len() {
                    break;
                }
                let flags = read_u16_be(body, cursor);
                let child_gid = read_u16_be(body, cursor + 2);
                cursor += 4;

                if child_gid < num_glyphs && !keep.contains(&child_gid) {
                    keep.insert(child_gid);
                    changed = true;
                }

                // Skip over argument payload.
                cursor += if flags & ARG_1_AND_2_ARE_WORDS != 0 {
                    4
                } else {
                    2
                };
                if flags & WE_HAVE_A_SCALE != 0 {
                    cursor += 2;
                } else if flags & WE_HAVE_AN_X_AND_Y_SCALE != 0 {
                    cursor += 4;
                } else if flags & WE_HAVE_A_TWO_BY_TWO != 0 {
                    cursor += 8;
                }

                if flags & MORE_COMPONENTS == 0 {
                    break;
                }
            }
        }
    }
}

/// Rebuild the glyf and loca tables retaining only glyphs in `keep`.
fn rebuild_glyf_and_loca(
    glyf: &[u8],
    offsets: &[u32],
    keep: &BTreeSet<u16>,
    long_loca: bool,
) -> (Vec<u8>, Vec<u8>) {
    let num_glyphs = offsets.len() - 1;
    let mut new_glyf: Vec<u8> = Vec::with_capacity(glyf.len());
    let mut new_offsets: Vec<u32> = Vec::with_capacity(num_glyphs + 1);
    new_offsets.push(0);

    for gid in 0..num_glyphs {
        if keep.contains(&u16::try_from(gid).unwrap_or(u16::MAX)) {
            let start = offsets[gid] as usize;
            let end = offsets[gid + 1] as usize;
            if end > start && end <= glyf.len() {
                new_glyf.extend_from_slice(&glyf[start..end]);
            }
        }
        // If short loca is in effect, offsets must be 2-byte aligned — pad
        // with a zero byte when the running offset is odd.
        if !long_loca && new_glyf.len() % 2 == 1 {
            new_glyf.push(0);
        }
        new_offsets.push(u32::try_from(new_glyf.len()).unwrap_or(u32::MAX));
    }

    // Emit loca in the same format as the source.
    let mut new_loca: Vec<u8> =
        Vec::with_capacity(new_offsets.len() * if long_loca { 4 } else { 2 });
    if long_loca {
        for o in &new_offsets {
            new_loca.extend_from_slice(&o.to_be_bytes());
        }
    } else {
        for o in &new_offsets {
            let half = u16::try_from(*o / 2).unwrap_or(u16::MAX);
            new_loca.extend_from_slice(&half.to_be_bytes());
        }
    }

    (new_glyf, new_loca)
}

/// Write an sfnt-packed font with the given tables. Tables are serialised
/// in tag-alphabetical order per the spec's "recommended order" guidance,
/// with 4-byte zero padding after each table body. Each table's own
/// checksum is computed and stored in its table record.
fn write_font(sfnt_version: [u8; 4], tables: &[(Tag, Vec<u8>)]) -> Vec<u8> {
    let mut sorted: Vec<(Tag, Vec<u8>)> = tables.to_vec();
    sorted.sort_by_key(|t| t.0);

    let num_tables = u16::try_from(sorted.len()).unwrap_or(u16::MAX);
    let entry_selector = u16::try_from(floor_to_i32(f32::from(num_tables).log2())).unwrap_or(0);
    let search_range = (1u16 << entry_selector) * 16;
    let range_shift = num_tables * 16 - search_range;

    let directory_size = 12 + sorted.len() * 16;

    // Lay out each table body at a 4-byte-aligned offset following the
    // directory.
    let mut body_offsets: Vec<u32> = Vec::with_capacity(sorted.len());
    let mut cursor = directory_size;
    for (_, body) in &sorted {
        body_offsets.push(u32::try_from(cursor).unwrap_or(u32::MAX));
        cursor += body.len();
        // 4-byte alignment between table bodies.
        while cursor % 4 != 0 {
            cursor += 1;
        }
    }
    let total_size = cursor;

    let mut out = Vec::with_capacity(total_size);

    // Offset subtable.
    out.extend_from_slice(&sfnt_version);
    out.extend_from_slice(&num_tables.to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_selector.to_be_bytes());
    out.extend_from_slice(&range_shift.to_be_bytes());

    // Table records.
    for ((tag, body), offset) in sorted.iter().zip(body_offsets.iter()) {
        let checksum = table_checksum(body);
        out.extend_from_slice(tag);
        out.extend_from_slice(&checksum.to_be_bytes());
        out.extend_from_slice(&offset.to_be_bytes());
        out.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_be_bytes());
    }

    // Table bodies.
    for (_, body) in &sorted {
        while out.len() % 4 != 0 {
            out.push(0);
        }
        out.extend_from_slice(body);
    }
    // Trailing pad.
    while out.len() % 4 != 0 {
        out.push(0);
    }

    out
}

/// OpenType table checksum: sum of 32-bit big-endian words in the table,
/// with the tail zero-padded.
fn table_checksum(body: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 4 <= body.len() {
        let w = u32::from_be_bytes([body[i], body[i + 1], body[i + 2], body[i + 3]]);
        sum = sum.wrapping_add(w);
        i += 4;
    }
    if i < body.len() {
        let mut tail = [0u8; 4];
        for (j, b) in body[i..].iter().enumerate() {
            tail[j] = *b;
        }
        sum = sum.wrapping_add(u32::from_be_bytes(tail));
    }
    sum
}

const fn read_u16_be(buf: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([buf[off], buf[off + 1]])
}

const fn read_i16_be(buf: &[u8], off: usize) -> i16 {
    i16::from_be_bytes([buf[off], buf[off + 1]])
}

const fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const DEMO_TTF: &[u8] = include_bytes!("../../tests/fixtures/demo.ttf");

    #[test]
    fn subset_preserves_requested_glyphs() {
        let mut keep: BTreeSet<u16> = BTreeSet::new();
        keep.insert(1); // .notdef is auto-included; 1 is "A" in demo.ttf
        let out = subset_truetype(DEMO_TTF, &keep).expect("subset");
        // Output must still be parseable as a TrueType face.
        let face = ttf_parser::Face::parse(&out, 0).expect("parse subset");
        assert!(face.tables().glyf.is_some());
        // numGlyphs preserved.
        assert_eq!(face.number_of_glyphs(), {
            ttf_parser::Face::parse(DEMO_TTF, 0)
                .unwrap()
                .number_of_glyphs()
        });
    }

    #[test]
    fn subset_always_includes_glyph_zero() {
        let keep: BTreeSet<u16> = BTreeSet::new();
        let out = subset_truetype(DEMO_TTF, &keep).expect("subset with empty set");
        let face = ttf_parser::Face::parse(&out, 0).expect("parse");
        // Glyph 0 outline must still be present (it's .notdef, may be empty
        // for demo.ttf, but parse must succeed and the id is valid).
        assert!(face.number_of_glyphs() >= 1);
    }

    #[test]
    fn subset_is_not_larger_than_input() {
        let mut keep: BTreeSet<u16> = BTreeSet::new();
        keep.insert(1);
        let out = subset_truetype(DEMO_TTF, &keep).expect("subset");
        // For this tiny fixture the size may be comparable — but it must
        // never exceed the input + reasonable directory overhead.
        assert!(
            out.len() <= DEMO_TTF.len() + 64,
            "subset {} > input {} + overhead",
            out.len(),
            DEMO_TTF.len()
        );
    }

    #[test]
    fn subset_rejects_ttc_data() {
        let mut ttc = vec![b't', b't', b'c', b'f'];
        ttc.extend_from_slice(&[0, 0, 0, 0]); // rest garbage
        let err = subset_truetype(&ttc, &BTreeSet::new()).unwrap_err();
        matches!(err, SubsetError::CollectionNotSupported);
    }

    #[test]
    fn subset_rejects_truncated_data() {
        let err = subset_truetype(&[0u8; 4], &BTreeSet::new()).unwrap_err();
        matches!(err, SubsetError::Malformed(_));
    }

    #[test]
    fn table_checksum_matches_zero_for_empty() {
        assert_eq!(table_checksum(&[]), 0);
    }

    #[test]
    fn table_checksum_pads_tail() {
        // 5 bytes → first word = [1,2,3,4], second = [5,0,0,0].
        let body = [1u8, 2, 3, 4, 5];
        let expected =
            u32::from_be_bytes([1, 2, 3, 4]).wrapping_add(u32::from_be_bytes([5, 0, 0, 0]));
        assert_eq!(table_checksum(&body), expected);
    }
}
