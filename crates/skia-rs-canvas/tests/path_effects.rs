//! Path effects on lines and through picture recording.

use skia_rs_canvas::{Canvas, DrawCommand, raster::PixelBuffer};
use skia_rs_core::{Color, Point};
use skia_rs_paint::{Paint, Style};
use skia_rs_path::{DashEffect, PathBuilder};
use std::sync::Arc;

const W: i32 = 120;

/// A boxed drawing routine, for table-driven comparisons.
type Draw<'a> = Box<dyn Fn(&mut Canvas<'_>) + 'a>;
const H: i32 = 20;

fn dashed_paint(intervals: Vec<f32>) -> Paint {
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(4.0);
    paint.set_path_effect(Some(Arc::new(
        DashEffect::new(intervals, 0.0).expect("valid dash"),
    )));
    paint
}

fn lit(buffer: &PixelBuffer, x: i32, y: i32) -> bool {
    buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red() > 100
}

fn row(buffer: &PixelBuffer, y: i32) -> Vec<bool> {
    (0..W).map(|x| lit(buffer, x, y)).collect()
}

#[test]
fn draw_line_applies_dash_effect() {
    // SkCanvas::drawLine is drawPoints(kLines): the paint's path effect
    // applies. A [10, 10] dash must leave unpainted gaps.
    let mut buffer = PixelBuffer::new(W, H);
    {
        let mut canvas = Canvas::new_raster(&mut buffer);
        let paint = dashed_paint(vec![10.0, 10.0]);
        canvas.draw_line(Point::new(0.0, 10.0), Point::new(100.0, 10.0), &paint);
    }
    assert!(lit(&buffer, 5, 10), "first dash should be painted");
    assert!(!lit(&buffer, 15, 10), "first gap should be empty");
    assert!(lit(&buffer, 25, 10), "second dash should be painted");
}

#[test]
fn recorded_dashed_path_plays_back_like_direct_drawing() {
    // An uneven dash: re-dashing an already dashed 10-unit segment would
    // open a gap inside it, so applying the effect twice is detectable.
    let paint = dashed_paint(vec![2.0, 2.0, 10.0, 2.0]);
    let mut b = PathBuilder::new();
    b.move_to(0.0, 10.0).line_to(110.0, 10.0);
    let path = b.build();

    let mut direct = PixelBuffer::new(W, H);
    Canvas::new_raster(&mut direct).draw_path(&path, &paint);

    let mut commands: Vec<DrawCommand> = Vec::new();
    Canvas::new_recording(&mut commands, W, H).draw_path(&path, &paint);
    let mut replayed = PixelBuffer::new(W, H);
    {
        let mut canvas = Canvas::new_raster(&mut replayed);
        for command in &commands {
            command.execute(&mut canvas);
        }
    }

    assert_eq!(
        row(&replayed, 10),
        row(&direct, 10),
        "playback must apply the dash exactly once"
    );
}

#[test]
fn draw_line_fills_path1d_stamps() {
    // Sk1DPathEffect switches the stroke rec to fill: stamping 6x6 squares
    // along a hairline must paint solid squares, not 1px outlines.
    use skia_rs_core::Rect;
    use skia_rs_path::{Path1DEffect, Path1DStyle};

    let mut stamp = PathBuilder::new();
    stamp.add_rect(&Rect::from_xywh(-3.0, -3.0, 6.0, 6.0));
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_path_effect(Some(Arc::new(
        Path1DEffect::new(stamp.build(), 20.0, 0.0, Path1DStyle::Translate).unwrap(),
    )));

    let mut buffer = PixelBuffer::new(W, H);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(10.0, 10.0),
        Point::new(110.0, 10.0),
        &paint,
    );

    // A stamp is centred on (30, 10): its interior must be filled.
    assert!(lit(&buffer, 30, 10), "stamp interior must be filled");
    assert!(lit(&buffer, 29, 9), "stamp interior must be filled");
    // Between stamps nothing is drawn (the source line itself is replaced).
    assert!(!lit(&buffer, 40, 10), "gap between stamps must stay empty");
}

#[cfg(feature = "text")]
#[test]
fn recorded_emboldened_text_replays_per_command_at_the_right_place() {
    // Executing recorded commands one by one on a canvas must give the same
    // result as drawing directly, even with a CTM and fake bold.
    use skia_rs_text::{Font, Typeface};

    const DEMO_TTF: &[u8] = include_bytes!("fixtures/demo.ttf");
    let tf = Typeface::from_data(DEMO_TTF.to_vec()).expect("parse demo.ttf");
    let mut font = Font::new(Arc::new(tf), 30.0);
    font.set_embolden(true);
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());

    let draw = |canvas: &mut Canvas<'_>| {
        canvas.translate(50.0, 0.0);
        canvas.scale(2.0, 2.0);
        canvas.draw_string("A", 5.0, 30.0, &font, &paint);
    };

    let (w, h) = (300, 100);
    let mut direct = PixelBuffer::new(w, h);
    draw(&mut Canvas::new_raster(&mut direct));

    let mut commands: Vec<DrawCommand> = Vec::new();
    draw(&mut Canvas::new_recording(&mut commands, w, h));
    let mut replayed = PixelBuffer::new(w, h);
    {
        let mut canvas = Canvas::new_raster(&mut replayed);
        for command in &commands {
            command.execute(&mut canvas);
        }
    }

    let painted = |b: &PixelBuffer| {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| b.get_pixel(x, y).unwrap_or(Color::BLACK).red() > 100)
            .collect::<Vec<_>>()
    };
    let expected = painted(&direct);
    assert!(!expected.is_empty(), "direct drawing must paint the glyph");
    assert_eq!(
        painted(&replayed),
        expected,
        "replay must not apply the CTM twice"
    );
}

#[test]
fn dash_on_a_filled_paint_fills_the_whole_shape() {
    // SkDashPath::InternalFilter refuses Fill/StrokeAndFill geometry, so a
    // filled rect with a dash effect is filled in full, not as dash slivers.
    use skia_rs_core::Rect;

    let mut paint = dashed_paint(vec![10.0, 5.0]);
    paint.set_style(Style::Fill);
    let mut b = PathBuilder::new();
    b.add_rect(&Rect::from_xywh(10.0, 2.0, 100.0, 16.0));
    let mut via_path = PixelBuffer::new(W, H);
    Canvas::new_raster(&mut via_path).draw_path(&b.build(), &paint);

    for x in 12..108 {
        assert!(
            lit(&via_path, x, 10),
            "filled dashed path must be solid at x={x}"
        );
    }
}

#[test]
fn zero_length_line_draws_cap_dots_like_skia() {
    // A zero-length segment with round or square caps paints a dot of the
    // stroke width; with butt caps it paints nothing.
    use skia_rs_paint::StrokeCap;

    let dot = |cap: StrokeCap| {
        let mut paint = Paint::new();
        paint.set_color(Color::WHITE.into());
        paint.set_style(Style::Stroke);
        paint.set_stroke_width(10.0);
        paint.set_stroke_cap(cap);
        let mut buffer = PixelBuffer::new(W, H);
        Canvas::new_raster(&mut buffer).draw_line(
            Point::new(50.0, 10.0),
            Point::new(50.0, 10.0),
            &paint,
        );
        (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .filter(|&(x, y)| lit(&buffer, x, y))
            .count()
    };

    let round = dot(StrokeCap::Round);
    let square = dot(StrokeCap::Square);
    assert!(
        round >= 60,
        "round cap should paint a ~10px dot, got {round}"
    );
    assert!(
        square >= 90,
        "square cap should paint a ~10x10 dot, got {square}"
    );
    assert!(square > round, "a square dot covers more than a round one");
    assert_eq!(
        dot(StrokeCap::Butt),
        0,
        "butt caps draw nothing for a point"
    );
}

#[test]
fn shapes_apply_path_effects_like_their_paths() {
    // Skia draws rects/ovals/circles/round rects/arcs as paths when the
    // paint has a path effect, so a dashed outline must match the dashed
    // equivalent path exactly.
    use skia_rs_core::Rect;

    const SW: i32 = 120;
    const SH: i32 = 120;
    let paint = dashed_paint(vec![6.0, 4.0]);
    let rect = Rect::from_xywh(10.0, 10.0, 100.0, 80.0);
    let render = |draw: &dyn Fn(&mut Canvas<'_>)| {
        let mut buffer = PixelBuffer::new(SW, SH);
        draw(&mut Canvas::new_raster(&mut buffer));
        (0..SH)
            .flat_map(|y| (0..SW).map(move |x| (x, y)))
            .map(|(x, y)| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red())
            .collect::<Vec<_>>()
    };
    let path_of = |build: &dyn Fn(&mut PathBuilder)| {
        let mut b = PathBuilder::new();
        build(&mut b);
        b.build()
    };

    let rect_path = path_of(&|b| {
        b.add_rect(&rect);
    });
    let oval_path = path_of(&|b| {
        b.add_oval(&rect);
    });
    let circle_path = path_of(&|b| {
        b.add_circle(60.0, 50.0, 40.0);
    });

    let cases: [(&str, Draw<'_>, Draw<'_>); 3] = [
        (
            "rect",
            Box::new(|c| c.draw_rect(&rect, &paint)),
            Box::new(|c| c.draw_path(&rect_path, &paint)),
        ),
        (
            "oval",
            Box::new(|c| c.draw_oval(&rect, &paint)),
            Box::new(|c| c.draw_path(&oval_path, &paint)),
        ),
        (
            "circle",
            Box::new(|c| c.draw_circle(Point::new(60.0, 50.0), 40.0, &paint)),
            Box::new(|c| c.draw_path(&circle_path, &paint)),
        ),
    ];
    for (name, shape, path) in &cases {
        let shape_pixels = render(shape.as_ref());
        let path_pixels = render(path.as_ref());
        assert!(shape_pixels.iter().any(|&r| r > 0), "{name}: nothing drawn");
        assert_eq!(
            shape_pixels, path_pixels,
            "{name}: dash must apply to the shape"
        );
    }

    // Round rects and arcs already build paths: the dash must leave gaps.
    let mut buffer = PixelBuffer::new(SW, SH);
    Canvas::new_raster(&mut buffer).draw_round_rect(&rect, 12.0, 12.0, &paint);
    let top_edge: Vec<bool> = (30..90).map(|x| lit(&buffer, x, 10)).collect();
    assert!(top_edge.iter().any(|&l| l) && top_edge.iter().any(|&l| !l));
}

#[cfg(feature = "text")]
#[test]
fn glyph_path_effects_do_not_depend_on_glyph_position() {
    // Skia applies the paint's path effect to each glyph in glyph space, so
    // a lattice effect looks identical wherever the glyph is drawn.
    use skia_rs_core::Matrix;
    use skia_rs_path::Line2DEffect;
    use skia_rs_text::{Font, Typeface};

    const DEMO_TTF: &[u8] = include_bytes!("fixtures/demo.ttf");
    let font = Font::new(
        Arc::new(Typeface::from_data(DEMO_TTF.to_vec()).expect("parse demo.ttf")),
        80.0,
    );
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());
    paint.set_path_effect(Some(Arc::new(
        Line2DEffect::new(1.0, Matrix::scale(5.0, 5.0)).unwrap(),
    )));

    // Line2D with scale(5, 5) hatches horizontally every 5px, so shift the
    // glyph vertically by 3px: a canvas-anchored hatch would no longer line
    // up with the glyph, while a glyph-space hatch moves with it.
    let (w, h) = (120, 110);
    let render = |y: f32| {
        let mut buffer = PixelBuffer::new(w, h);
        Canvas::new_raster(&mut buffer).draw_glyphs(
            &[1],
            &[Point::zero()],
            Point::new(10.0, y),
            &font,
            &paint,
        );
        buffer
    };
    let at_90 = render(90.0);
    let at_93 = render(93.0);

    let mut painted = 0;
    for y in 0..h - 3 {
        for x in 0..w {
            let a = at_90.get_pixel(x, y).unwrap_or(Color::BLACK);
            let b = at_93.get_pixel(x, y + 3).unwrap_or(Color::BLACK);
            assert_eq!(a, b, "hatch differs at ({x},{y}) after a 3px shift");
            if a.red() > 0 {
                painted += 1;
            }
        }
    }
    assert!(painted > 0, "the hatched glyph must paint something");
}

#[test]
fn effects_on_non_finite_or_overflowing_geometry_terminate() {
    // SkCanvas rejects non-finite geometry, and SkDashPath gives up past
    // kMaxDashCount dashes, so none of these hang or exhaust memory.
    use skia_rs_core::Rect;

    let mut paint = Paint::new();
    paint.set_color32(Color::WHITE);
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(1.0);
    paint.set_path_effect(Some(Arc::new(
        DashEffect::new(vec![5.0, 5.0], 0.0).unwrap(),
    )));
    let mut buffer = PixelBuffer::new(64, 64);
    let mut canvas = Canvas::new_raster(&mut buffer);
    canvas.draw_rect(&Rect::new(0.0, 0.0, f32::INFINITY, 10.0), &paint);
    canvas.draw_oval(&Rect::new(0.0, f32::NAN, 10.0, 10.0), &paint);
    canvas.draw_circle(Point::new(f32::NAN, 0.0), 5.0, &paint);
    // Finite endpoints whose length overflows to infinity.
    canvas.draw_line(Point::new(0.0, 0.0), Point::new(3e38, 3e38), &paint);
    // A finite but enormous dash count.
    canvas.draw_line(Point::new(0.0, 0.0), Point::new(1e9, 0.0), &paint);
}

#[test]
fn line_2d_hatches_the_visible_part_of_a_huge_source() {
    // The lattice is culled to the clip before the size guard applies, so
    // a +-1e6 background still gets hatch lines on screen.
    use skia_rs_core::{Matrix, Rect};

    let mut paint = Paint::new();
    paint.set_color32(Color::WHITE);
    paint.set_path_effect(skia_rs_path::make_line_2d(1.0, Matrix::scale(8.0, 8.0)));
    let mut buffer = PixelBuffer::new(64, 64);
    Canvas::new_raster(&mut buffer).draw_rect(&Rect::new(-1e6, -1e6, 1e6, 1e6), &paint);
    let lit = (0..64).filter(|&y| buffer.get_pixel(32, y).is_some_and(|c| c.red() > 128));
    assert!(lit.count() >= 4, "hatch lines cross the visible column");
}

#[test]
fn round_rects_that_are_ovals_or_rects_dash_like_draw_oval_and_draw_rect() {
    // SkCanvas::onDrawRRect hands oval- and rect-type rrects to drawOval /
    // drawRect, so their dashes start where those start.
    use skia_rs_core::Rect;

    let draw = |f: &dyn Fn(&mut Canvas<'_>, &Paint)| {
        let mut paint = Paint::new();
        paint.set_color32(Color::WHITE);
        paint.set_style(Style::Stroke);
        paint.set_stroke_width(2.0);
        paint.set_path_effect(Some(Arc::new(
            DashEffect::new(vec![7.0, 5.0], 0.0).unwrap(),
        )));
        let mut buffer = PixelBuffer::new(64, 64);
        f(&mut Canvas::new_raster(&mut buffer), &paint);
        (0..64)
            .flat_map(|y| (0..64).map(move |x| (x, y)))
            .map(|(x, y)| buffer.get_pixel(x, y).map_or(0, |c| c.red()))
            .collect::<Vec<_>>()
    };
    let rect = Rect::from_xywh(10.0, 10.0, 40.0, 40.0);
    assert_eq!(
        draw(&|c, p| c.draw_round_rect(&rect, 20.0, 20.0, p)),
        draw(&|c, p| c.draw_oval(&rect, p)),
        "oval-type round rect dashes like draw_oval"
    );
}
