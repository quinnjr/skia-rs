//! Verifies `Canvas::draw_string` renders actual glyph outlines instead of
//! placeholder rectangles.
//!
//! Uses a 400-byte fixture font copied from the `ttf-parser` crate's own
//! test suite. The font contains a single glyph for 'A' (glyph id 1) with
//! a triangular body — exactly what we need to show that "glyph rendering
//! is not a solid rect".

#![cfg(feature = "text")]

use skia_rs_canvas::{Canvas, raster::PixelBuffer};
use skia_rs_core::Color;
use skia_rs_paint::Paint;
use skia_rs_text::{Font, Typeface};
use std::sync::Arc;

const DEMO_TTF: &[u8] = include_bytes!("fixtures/demo.ttf");

fn demo_font(size: f32) -> Font {
    let tf = Typeface::from_data(DEMO_TTF.to_vec()).expect("parse demo.ttf");
    Font::new(Arc::new(tf), size)
}

#[test]
#[allow(
    clippy::cast_precision_loss,
    reason = "pixel-count ratio in a test; precision is irrelevant"
)]
fn draw_string_renders_non_rectangular_glyph() {
    // Big canvas and big font — a 200pt 'A' gives us plenty of pixels to
    // sample inside vs. outside the triangular body.
    let mut buffer = PixelBuffer::new(400, 400);
    {
        let mut canvas = Canvas::new_raster(&mut buffer);
        let font = demo_font(200.0);
        let mut paint = Paint::new();
        paint.set_color(Color::WHITE.into());
        canvas.draw_string("A", 100.0, 300.0, &font, &paint);
    }

    // Gather stats inside the glyph's approximate bounding box. For demo.ttf
    // at 200pt the 'A' occupies roughly x in [100, 210] and y in [170, 300].
    let (mut painted, mut empty) = (0usize, 0usize);
    for y in 170..300 {
        for x in 100..210 {
            let c = buffer.get_pixel(x, y).unwrap_or(Color::BLACK);
            if c.red() > 100 {
                painted += 1;
            } else {
                empty += 1;
            }
        }
    }

    assert!(painted > 0, "glyph 'A' should produce some painted pixels");
    assert!(
        empty > 0,
        "glyph 'A' must have unfilled pixels within its bbox — a solid \
         rectangle would have zero. (painted={painted}, empty={empty})"
    );
    // Sanity: the 'A' is an outline glyph, neither mostly filled nor empty.
    let total = painted + empty;
    let ratio = painted as f32 / total as f32;
    assert!(
        (0.05..0.95).contains(&ratio),
        "expected a non-trivial mix of painted/empty; got ratio={ratio}"
    );
}

#[test]
fn draw_string_no_op_for_dataless_typeface() {
    // Without font data there is no outline to render. Previously the
    // canvas drew a filled rectangle per glyph; it must now draw nothing.
    let mut buffer = PixelBuffer::new(100, 100);
    {
        let mut canvas = Canvas::new_raster(&mut buffer);
        let font = Font::from_size(24.0); // default, dataless typeface
        let mut paint = Paint::new();
        paint.set_color(Color::WHITE.into());
        canvas.draw_string("Hello", 10.0, 50.0, &font, &paint);
    }

    for y in 0..100 {
        for x in 0..100 {
            let c = buffer.get_pixel(x, y).unwrap_or(Color::BLACK);
            assert_eq!(
                c.red(),
                0,
                "pixel ({x},{y}) should be untouched when no font data is \
                 available, but got red={}",
                c.red()
            );
        }
    }
}

#[test]
fn draw_color_glyph_returns_false_for_outline_font() {
    // demo.ttf has no COLR / CPAL / SVG / CBDT data — every glyph is a
    // pure outline. `draw_color_glyph` must report "not a color glyph"
    // so the caller falls back to the outline path. Regression test for
    // the COLR v1 dispatch entry point.
    let mut buffer = PixelBuffer::new(64, 64);
    let mut canvas = Canvas::new_raster(&mut buffer);
    let font = demo_font(24.0);
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());
    let handled = canvas.draw_color_glyph(1, skia_rs_core::Point::new(10.0, 40.0), &font, &paint);
    assert!(
        !handled,
        "demo.ttf is outline-only; draw_color_glyph must not claim the glyph"
    );
}

#[test]
fn draw_text_blob_renders_non_rectangular_glyph() {
    use skia_rs_core::Point;
    use skia_rs_text::{TextBlob, TextBlobBuilder};

    let font = demo_font(200.0);
    let blob: TextBlob = {
        let mut b = TextBlobBuilder::new();
        b.add_positioned_run(&font, &[1], &[Point::new(100.0, 300.0)]);
        b.build().expect("blob build")
    };

    let mut buffer = PixelBuffer::new(400, 400);
    {
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color(Color::WHITE.into());
        canvas.draw_text_blob(&blob, 0.0, 0.0, &paint);
    }

    let (mut painted, mut empty) = (0usize, 0usize);
    for y in 170..300 {
        for x in 100..210 {
            let c = buffer.get_pixel(x, y).unwrap_or(Color::BLACK);
            if c.red() > 100 {
                painted += 1;
            } else {
                empty += 1;
            }
        }
    }
    assert!(painted > 0, "text_blob should paint some pixels");
    assert!(
        empty > 0,
        "text_blob must have unfilled pixels within its bbox"
    );
}

/// Count pixels with a non-zero red channel in the whole buffer.
fn lit_pixels(buffer: &PixelBuffer, width: i32, height: i32) -> usize {
    let mut lit = 0;
    for y in 0..height {
        for x in 0..width {
            if buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red() > 0 {
                lit += 1;
            }
        }
    }
    lit
}

#[test]
fn embolden_thickens_glyphs_in_every_draw_path() {
    use skia_rs_core::Point;
    use skia_rs_text::TextBlob;

    // Render the same glyph regular and emboldened through each public text
    // entry point; fake bold (Skia's stroke-and-fill outset) must strictly
    // increase coverage, and for a stroked paint it must widen the stroke.
    let render = |embolden: bool, stroke: bool, entry: u8| {
        let mut font = demo_font(120.0);
        font.set_embolden(embolden);
        let mut paint = Paint::new();
        paint.set_color(Color::WHITE.into());
        if stroke {
            paint.set_style(skia_rs_paint::Style::Stroke);
            paint.set_stroke_width(2.0);
        }
        let mut buffer = PixelBuffer::new(300, 300);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            match entry {
                0 => canvas.draw_string("A", 60.0, 220.0, &font, &paint),
                1 => {
                    let blob = TextBlob::from_text("A", &font, Point::zero());
                    canvas.draw_text_blob(&blob, 60.0, 220.0, &paint);
                }
                _ => canvas.draw_glyphs(
                    &[1],
                    &[Point::zero()],
                    Point::new(60.0, 220.0),
                    &font,
                    &paint,
                ),
            }
        }
        lit_pixels(&buffer, 300, 300)
    };

    for entry in 0..3 {
        for stroke in [false, true] {
            let regular = render(false, stroke, entry);
            let bold = render(true, stroke, entry);
            assert!(regular > 0, "entry {entry}: regular glyph must paint");
            assert!(
                bold > regular,
                "entry {entry} stroke={stroke}: embolden must add coverage \
                 (regular={regular}, bold={bold})"
            );
        }
    }
}

#[test]
fn embolden_under_scale_adds_coverage_and_never_double_blends() {
    // Emboldening is done in device space; under a 3x canvas scale both the
    // opaque fast path and the translucent union path must thicken the
    // glyph, and the translucent path must cover each pixel exactly once.
    let render = |embolden: bool, alpha: u8| {
        let mut font = demo_font(40.0);
        font.set_embolden(embolden);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(alpha, 255, 255, 255));
        paint.set_anti_alias(true);
        let mut buffer = PixelBuffer::new(300, 300);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            canvas.scale(3.0, 3.0);
            canvas.draw_string("A", 20.0, 80.0, &font, &paint);
        }
        let mut lit = 0usize;
        let mut max_red = 0u8;
        for y in 0..300 {
            for x in 0..300 {
                let red = buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red();
                max_red = max_red.max(red);
                if red > 0 {
                    lit += 1;
                }
            }
        }
        (lit, max_red)
    };

    for alpha in [255u8, 128] {
        let (regular, regular_max) = render(false, alpha);
        let (bold, bold_max) = render(true, alpha);
        assert!(regular > 0, "alpha {alpha}: regular glyph must paint");
        assert!(
            bold > regular,
            "alpha {alpha}: embolden must add coverage (regular={regular}, bold={bold})"
        );
        assert!(
            bold_max <= regular_max.saturating_add(1),
            "alpha {alpha}: bold glyph must not double-blend the fill/stroke \
             overlap (regular max={regular_max}, bold max={bold_max})"
        );
    }
}

#[test]
fn embolden_under_non_uniform_scale_strokes_in_local_space() {
    // Skia strokes the fake-bold outset in local space, so a non-uniform
    // CTM stretches it. Under scale(4, 1) emboldened text must match a
    // local-space StrokeAndFill of the outline with Skia's outset
    // (40pt -> 40 * 1/32 = 1.25), not a uniform device-space stroke.
    use skia_rs_core::{Matrix, Point};

    let size = 40.0;
    let origin = Point::new(5.0, 60.0);
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());

    let mut bold_font = demo_font(size);
    bold_font.set_embolden(true);
    let mut actual = PixelBuffer::new(300, 100);
    {
        let mut canvas = Canvas::new_raster(&mut actual);
        canvas.scale(4.0, 1.0);
        canvas.draw_glyphs(&[1], &[Point::zero()], origin, &bold_font, &paint);
    }

    let outline = demo_font(size)
        .glyph_path(1)
        .expect("demo.ttf glyph 1 has an outline")
        .transformed(&Matrix::translate(origin.x, origin.y));
    let mut expected_paint = paint.clone();
    expected_paint.set_style(skia_rs_paint::Style::StrokeAndFill);
    expected_paint.set_stroke_width(size / 32.0);
    let mut expected = PixelBuffer::new(300, 100);
    {
        let mut canvas = Canvas::new_raster(&mut expected);
        canvas.scale(4.0, 1.0);
        canvas.draw_path(&outline, &expected_paint);
    }

    assert!(lit_pixels(&actual, 300, 100) > 0);
    for y in 0..100 {
        for x in 0..300 {
            assert_eq!(
                actual.get_pixel(x, y),
                expected.get_pixel(x, y),
                "pixel ({x},{y}) differs from Skia's local-space outset"
            );
        }
    }
}
