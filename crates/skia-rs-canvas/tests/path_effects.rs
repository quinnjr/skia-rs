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
