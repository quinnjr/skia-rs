//! Conic path segments (circles, ovals, round-rect corners) must be
//! rasterized as true conics, honoring their weight.

use skia_rs_canvas::{Canvas, raster::PixelBuffer};
use skia_rs_core::{Color, Rect};
use skia_rs_paint::{Paint, Style};
use skia_rs_path::PathBuilder;

const SIZE: i32 = 200;

fn white(style: Style) -> Paint {
    let mut paint = Paint::new();
    paint.set_color(Color::WHITE.into());
    paint.set_style(style);
    paint
}

fn lit(buffer: &PixelBuffer, x: i32, y: i32) -> bool {
    buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red() > 100
}

/// The pixel at `dist` from `(cx, cy)` along the 45-degree diagonal.
fn diagonal(cx: f32, cy: f32, dist: f32) -> (i32, i32) {
    let d = dist * std::f32::consts::FRAC_1_SQRT_2;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "pixel coordinates of a small test canvas"
    )]
    ((cx + d) as i32, (cy + d) as i32)
}

#[test]
fn filled_circle_path_is_round_not_quad_bulged() {
    // A weight-sqrt(2)/2 quarter conic flattened as a plain quad reaches
    // ~1.06r on the diagonal; a true circle stays within r.
    let (cx, cy, r) = (100.0, 100.0, 80.0);
    let mut b = PathBuilder::new();
    b.add_circle(cx, cy, r);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_path(&b.build(), &white(Style::Fill));

    let (ix, iy) = diagonal(cx, cy, r - 2.0);
    assert!(lit(&buffer, ix, iy), "inside the circle on the diagonal");
    let (ox, oy) = diagonal(cx, cy, r + 2.0);
    assert!(!lit(&buffer, ox, oy), "the diagonal must not bulge past r");
}

#[test]
fn round_rect_corners_are_circular() {
    // Corner centre (60, 60) with radius 50: the corner arc's diagonal point
    // is 50px from it. A quad-flattened corner bulges ~3px further out.
    let rect = Rect::from_xywh(10.0, 10.0, 180.0, 180.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_round_rect(&rect, 50.0, 50.0, &white(Style::Fill));

    // Diagonal pointing up-left from the corner centre.
    let d = std::f32::consts::FRAC_1_SQRT_2;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "pixel coordinates of a small test canvas"
    )]
    let at = |dist: f32| (dist.mul_add(-d, 60.0) as i32, dist.mul_add(-d, 60.0) as i32);
    let (ix, iy) = at(48.0);
    assert!(lit(&buffer, ix, iy), "inside the rounded corner");
    let (ox, oy) = at(52.0);
    assert!(
        !lit(&buffer, ox, oy),
        "the corner must not bulge past its radius"
    );
}

#[test]
fn hairline_circle_path_stays_on_the_circle() {
    let (cx, cy, r) = (100.0, 100.0, 80.0);
    let mut b = PathBuilder::new();
    b.add_circle(cx, cy, r);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    Canvas::new_raster(&mut buffer).draw_path(&b.build(), &paint);

    // The hairline crosses the diagonal at r, not ~1.06r.
    let hit = (-2..=2).any(|k| {
        #[allow(clippy::cast_precision_loss, reason = "small offsets")]
        let (x, y) = diagonal(cx, cy, r + k as f32);
        lit(&buffer, x, y)
    });
    assert!(
        hit,
        "the hairline must pass within 2px of radius r on the diagonal"
    );
    let (ox, oy) = diagonal(cx, cy, r + 4.0);
    assert!(!lit(&buffer, ox, oy), "no bulged hairline past r");
}
