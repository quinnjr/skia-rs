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

#[test]
fn polygon_points_stroke_one_joined_path() {
    // SkDraw::drawPoints(kPolygon) strokes one open path: the corner gets a
    // miter join (not two butt caps leaving a notch) and the overlap of the
    // two segments is blended once.
    use skia_rs_canvas::PointMode;
    use skia_rs_core::Point;
    use skia_rs_paint::{StrokeCap, StrokeJoin};

    let mut paint = Paint::new();
    paint.set_color32(Color::from_argb(128, 255, 255, 255));
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(10.0);
    paint.set_stroke_cap(StrokeCap::Butt);
    paint.set_stroke_join(StrokeJoin::Miter);

    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_points(
        PointMode::Polygon,
        &[
            Point::new(10.0, 20.0),
            Point::new(100.0, 20.0),
            Point::new(100.0, 110.0),
        ],
        &paint,
    );

    let red = |x, y| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red();
    // Outer corner (x 100..105, y 15..20) is covered only by the miter join.
    assert!(red(103, 17) > 0, "the corner must be mitered, not notched");
    // Where the segments overlap, coverage must not be blended twice.
    let overlap = red(98, 22);
    let single = red(50, 20);
    assert!(single > 0);
    assert!(
        overlap <= single.saturating_add(2),
        "overlap blended twice: {overlap} vs single-coverage {single}"
    );
}

#[test]
fn circle_under_perspective_matches_the_projected_curve() {
    // A perspective map changes a conic's weight, so mapping its control
    // points and keeping w = sqrt(2)/2 draws the wrong curve. The circle
    // must rasterize like a finely flattened polygon projected through the
    // same matrix.
    use skia_rs_core::Matrix;

    let mut persp = Matrix::IDENTITY;
    persp.values[Matrix::PERSP_0] = 0.004;
    persp.values[Matrix::PERSP_1] = 0.002;

    let (cx, cy, r) = (60.0_f32, 60.0_f32, 50.0_f32);
    let mut circle = PathBuilder::new();
    circle.add_circle(cx, cy, r);

    let mut polygon = PathBuilder::new();
    let steps = 720;
    for i in 0..steps {
        #[allow(clippy::cast_precision_loss, reason = "small loop index")]
        let a = (i as f32) / (steps as f32) * std::f32::consts::TAU;
        let (x, y) = (r.mul_add(a.cos(), cx), r.mul_add(a.sin(), cy));
        if i == 0 {
            polygon.move_to(x, y);
        } else {
            polygon.line_to(x, y);
        }
    }
    polygon.close();

    let render = |path: &skia_rs_path::Path| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            canvas.set_matrix(&persp);
            canvas.draw_path(path, &white(Style::Fill));
        }
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| lit(&buffer, x, y))
            .collect::<std::collections::HashSet<_>>()
    };
    let actual = render(&circle.build());
    let expected = render(&polygon.build());
    let differing = actual.symmetric_difference(&expected).count();
    assert!(
        expected.len() > 1000,
        "the projected circle must be visible"
    );
    assert!(
        // ~6 pixels differ with exact flattening, ~40 with the affine
        // weight; allow 0.5% (~16) for AA/edge noise.
        differing * 200 <= expected.len(),
        "{differing} pixels differ from the projected circle ({} expected)",
        expected.len()
    );
}

#[test]
fn magnified_circle_under_perspective_stays_smooth() {
    // A unit circle magnified 80x with a perspective term: the flattening
    // tolerance must hold in device pixels, not local units (0.25 local
    // units would be ~20 device px, a visible polygon).
    use skia_rs_core::Matrix;

    let mut m = Matrix::translate(100.0, 100.0).concat(&Matrix::scale(80.0, 80.0));
    m.values[Matrix::PERSP_0] = 0.05;
    m.values[Matrix::PERSP_1] = 0.03;

    let mut circle = PathBuilder::new();
    circle.add_circle(0.0, 0.0, 1.0);

    let mut polygon = PathBuilder::new();
    let steps = 2000;
    for i in 0..steps {
        #[allow(clippy::cast_precision_loss, reason = "small loop index")]
        let a = (i as f32) / (steps as f32) * std::f32::consts::TAU;
        if i == 0 {
            polygon.move_to(a.cos(), a.sin());
        } else {
            polygon.line_to(a.cos(), a.sin());
        }
    }
    polygon.close();

    let render = |path: &skia_rs_path::Path| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            canvas.set_matrix(&m);
            canvas.draw_path(path, &white(Style::Fill));
        }
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| lit(&buffer, x, y))
            .collect::<std::collections::HashSet<_>>()
    };
    let actual = render(&circle.build());
    let expected = render(&polygon.build());
    let differing = actual.symmetric_difference(&expected).count();
    assert!(
        expected.len() > 5000,
        "the projected circle must be visible"
    );
    assert!(
        differing * 200 <= expected.len(),
        "{differing} pixels differ from the projected circle ({} expected)",
        expected.len()
    );
}
