//! Path effects on lines and through picture recording.

use skia_rs_canvas::{Canvas, DrawCommand, raster::PixelBuffer};
use skia_rs_core::{Color, Point};
use skia_rs_paint::{Paint, Style};
use skia_rs_path::{DashEffect, PathBuilder};
use std::sync::Arc;

const W: i32 = 120;
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
