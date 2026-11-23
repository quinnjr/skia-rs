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
fn polygon_points_stroke_each_segment_separately() {
    // SkDraw::drawPoints(kPolygon) draws every consecutive pair as its own
    // SkPath::Line: with butt caps the outer corner is notched (no join),
    // as in Skia.
    use skia_rs_canvas::PointMode;
    use skia_rs_core::Point;
    use skia_rs_paint::{StrokeCap, StrokeJoin};

    let mut paint = white(Style::Stroke);
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

    // The outer corner square (x 100..105, y 15..20) belongs to neither
    // butt-capped segment.
    assert!(
        !lit(&buffer, 103, 17),
        "the corner must be notched, not joined"
    );
    assert!(lit(&buffer, 50, 20), "first segment drawn");
    assert!(lit(&buffer, 100, 60), "second segment drawn");
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

#[test]
fn cubic_under_perspective_matches_the_projected_curve() {
    // Quads and cubics are also wrong under perspective if only their
    // control points are mapped (a projected Bezier is a rational curve).
    // Fill a long S-shaped cubic across a strong perspective gradient and
    // compare with its exact projected samples: ~8 pixels differ when the
    // curve is evaluated locally and projected, ~500 with mapped controls.
    use skia_rs_core::{Matrix, Point};

    let mut persp = Matrix::IDENTITY;
    persp.values[Matrix::PERSP_0] = 0.006;

    let (p0, p1, p2, p3) = (
        Point::new(0.0, 100.0),
        Point::new(30.0, -150.0),
        Point::new(140.0, 250.0),
        Point::new(170.0, 20.0),
    );
    #[allow(
        clippy::suboptimal_flops,
        reason = "plain Bernstein-form reference evaluation"
    )]
    let at = |t: f32| {
        let mt = 1.0 - t;
        let (a, b, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
        Point::new(
            a * p0.x + b * p1.x + c * p2.x + d * p3.x,
            a * p0.y + b * p1.y + c * p2.y + d * p3.y,
        )
    };

    let mut cubic = PathBuilder::new();
    cubic
        .move_to(p0.x, p0.y)
        .cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
        .line_to(170.0, 190.0)
        .line_to(0.0, 190.0)
        .close();
    let mut polygon = PathBuilder::new();
    polygon.move_to(p0.x, p0.y);
    let steps = 4000;
    for i in 1..=steps {
        #[allow(clippy::cast_precision_loss, reason = "small loop index")]
        let q = at(i as f32 / steps as f32);
        polygon.line_to(q.x, q.y);
    }
    polygon.line_to(170.0, 190.0).line_to(0.0, 190.0).close();

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
    let actual = render(&cubic.build());
    let expected = render(&polygon.build());
    let differing = actual.symmetric_difference(&expected).count();
    assert!(expected.len() > 1000);
    assert!(
        differing * 100 <= expected.len(),
        "{differing} pixels differ from the projected cubic ({} expected)",
        expected.len()
    );
}
#[test]
fn dashed_inverted_rect_matches_the_sorted_rect() {
    // SkCanvas::drawRect sorts the rect, so an inverted rect dashes from
    // the same corner, in the same direction, as the sorted one.
    use skia_rs_path::DashEffect;
    use std::sync::Arc;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(3.0);
    paint.set_path_effect(Some(Arc::new(
        DashEffect::new(vec![10.0, 5.0], 0.0).unwrap(),
    )));

    let render = |rect: Rect| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_rect(&rect, &paint);
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .map(|(x, y)| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red())
            .collect::<Vec<_>>()
    };
    let sorted = render(Rect::new(20.0, 30.0, 170.0, 130.0));
    let inverted = render(Rect::new(170.0, 130.0, 20.0, 30.0));
    assert!(sorted.iter().any(|&v| v > 0));
    assert_eq!(inverted, sorted);
}

#[test]
fn inflected_cubic_is_not_collapsed_to_its_chord() {
    // A symmetric S-shaped cubic passes through its chord's midpoint, so a
    // midpoint-only flatness test would emit a single straight edge.
    use skia_rs_core::Point;

    let (o, s) = (Point::new(20.0, 100.0), 1.6_f32);
    let pt = |x: f32, y: f32| Point::new(x.mul_add(s, o.x), y.mul_add(s, o.y));
    let (p0, p1, p2, p3) = (
        pt(0.0, 0.0),
        pt(0.0, 100.0),
        pt(100.0, -100.0),
        pt(100.0, 0.0),
    );
    let mut path = PathBuilder::new();
    path.move_to(p0.x, p0.y)
        .cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
        .close();
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_path(&path.build(), &white(Style::Fill));

    // B(0.25) lies ~29 * 1.6 = 46px below the chord and B(0.75) as far above
    // it; the lobes between the curve and the chord must be filled.
    assert!(lit(&buffer, 45, 125), "lower lobe of the S must be filled");
    assert!(lit(&buffer, 155, 75), "upper lobe of the S must be filled");
}

#[test]
fn negative_circle_radius_with_a_path_effect_draws_nothing() {
    // SkCanvas::drawCircle clamps a negative radius to 0, with or without a
    // path effect.
    use skia_rs_core::Point;
    use skia_rs_path::{Path1DEffect, Path1DStyle};
    use std::sync::Arc;

    let mut stamp = PathBuilder::new();
    stamp.add_rect(&Rect::from_xywh(-3.0, -3.0, 6.0, 6.0));
    let mut paint = white(Style::Stroke);
    paint.set_path_effect(Some(Arc::new(
        Path1DEffect::new(stamp.build(), 10.0, 0.0, Path1DStyle::Translate).unwrap(),
    )));
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_circle(Point::new(100.0, 100.0), -40.0, &paint);
    let painted = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|&(x, y)| lit(&buffer, x, y))
        .count();
    assert_eq!(painted, 0);
}

#[test]
fn zero_radius_inverted_round_rect_dashes_like_draw_rect() {
    // drawRoundRect with a zero radius is drawRect, which sorts the rect.
    use skia_rs_path::DashEffect;
    use std::sync::Arc;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(3.0);
    paint.set_path_effect(Some(Arc::new(
        DashEffect::new(vec![10.0, 5.0], 0.0).unwrap(),
    )));
    let render = |draw: &dyn Fn(&mut Canvas<'_>)| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        draw(&mut Canvas::new_raster(&mut buffer));
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .map(|(x, y)| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red())
            .collect::<Vec<_>>()
    };
    let inverted = Rect::new(170.0, 130.0, 20.0, 30.0);
    let round = render(&|c| c.draw_round_rect(&inverted, 0.0, 0.0, &paint));
    let rect = render(&|c| c.draw_rect(&inverted, &paint));
    assert!(rect.iter().any(|&v| v > 0));
    assert_eq!(round, rect);
}

#[test]
fn path_without_leading_move_starts_at_the_mapped_origin() {
    // A contour with no leading move starts at the local origin; under a
    // translate its first edge must start at the mapped origin, exactly as
    // with an explicit move_to(0, 0).
    let fill = |explicit_move: bool| {
        let mut b = PathBuilder::new();
        if explicit_move {
            b.move_to(0.0, 0.0);
        }
        b.line_to(60.0, 0.0).line_to(60.0, 60.0).close();
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            canvas.translate(100.0, 100.0);
            canvas.draw_path(&b.build(), &white(Style::Fill));
        }
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .map(|(x, y)| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red())
            .collect::<Vec<_>>()
    };
    let explicit = fill(true);
    assert!(explicit.iter().any(|&v| v > 0));
    assert_eq!(fill(false), explicit);
}

#[test]
fn thin_antialiased_stroke_draws_as_a_coverage_scaled_hairline() {
    // SkDrawTreatAsHairline: an anti-aliased stroke at most 1 device px wide
    // is a hairline with alpha scaled by its width, not an outline too thin
    // to cover any pixel center.
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.5);
    paint.set_anti_alias(true);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(10.0, 50.5),
        Point::new(150.0, 50.5),
        &paint,
    );

    let row: Vec<u8> = (20..140)
        .map(|x| buffer.get_pixel(x, 50).unwrap_or(Color::BLACK).red())
        .collect();
    assert!(row.iter().all(|&v| v > 0), "the thin line must be visible");
    let max = *row.iter().max().unwrap();
    assert!(
        (100..=150).contains(&max),
        "a 0.5px line should draw at about half coverage, got {max}"
    );
}

#[test]
fn hairlines_use_pixel_centers_at_half_coordinates() {
    // Pixel row j covers [j, j+1) with its center at j + 0.5 (as in Skia),
    // so a horizontal hairline at y = 50.5 lies entirely on row 50, with or
    // without anti-aliasing.
    use skia_rs_core::Point;

    for anti_alias in [false, true] {
        let mut paint = white(Style::Stroke);
        paint.set_stroke_width(0.0);
        paint.set_anti_alias(anti_alias);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_line(
            Point::new(10.0, 50.5),
            Point::new(150.0, 50.5),
            &paint,
        );
        let red = |y: i32| buffer.get_pixel(80, y).unwrap_or(Color::BLACK).red();
        assert_eq!(red(50), 255, "aa={anti_alias}: row 50 carries the line");
        assert_eq!(red(49), 0, "aa={anti_alias}: row 49 untouched");
        assert_eq!(red(51), 0, "aa={anti_alias}: row 51 untouched");
    }
}

#[test]
fn thin_translucent_stroke_blends_each_pixel_once() {
    // A 1px translucent AA stroke must be one coverage pass: drawing it as
    // a segment-by-segment hairline would blend every joint pixel twice
    // and bead the outline.
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(1.0);
    paint.set_anti_alias(true);
    paint.set_color32(Color::from_argb(128, 255, 255, 255));
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_circle(Point::new(100.0, 100.0), 60.0, &paint);

    let max = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .map(|(x, y)| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red())
        .max()
        .unwrap();
    assert!(max > 0, "the circle must be drawn");
    assert!(
        max <= 130,
        "a pixel was blended twice: red {max} > single 50% coverage"
    );
}

#[test]
fn thin_stroke_keeps_the_paint_shader() {
    // A 1px AA stroke must still be shaded: converting it to a solid-color
    // hairline would drop the gradient.
    use skia_rs_core::{Color4f, Point};
    use skia_rs_paint::{LinearGradient, TileMode};
    use std::sync::Arc;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(1.0);
    paint.set_anti_alias(true);
    paint.set_shader(Some(Arc::new(LinearGradient::new(
        Point::new(10.0, 0.0),
        Point::new(190.0, 0.0),
        vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ],
        None,
        TileMode::Clamp,
    ))));
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(10.0, 50.5),
        Point::new(190.0, 50.5),
        &paint,
    );

    let left = buffer.get_pixel(20, 50).unwrap_or(Color::BLACK);
    let right = buffer.get_pixel(180, 50).unwrap_or(Color::BLACK);
    assert!(
        left.red() > left.blue(),
        "left end follows the gradient: {left:?}"
    );
    assert!(
        right.blue() > right.red(),
        "right end follows the gradient: {right:?}"
    );
}

#[test]
fn open_contours_fill_as_if_closed() {
    // Skia fills every contour as implicitly closed: an unclosed triangle
    // fills like a closed one, and a filled open arc fills its chord
    // segment.
    let fill = |closed: bool| {
        let mut b = PathBuilder::new();
        b.move_to(20.0, 20.0)
            .line_to(180.0, 20.0)
            .line_to(180.0, 180.0);
        if closed {
            b.close();
        }
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_path(&b.build(), &white(Style::Fill));
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .map(|(x, y)| buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red())
            .collect::<Vec<_>>()
    };
    let closed = fill(true);
    assert!(closed.iter().any(|&v| v > 0));
    assert_eq!(
        fill(false),
        closed,
        "an open triangle must fill like a closed one"
    );

    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_arc(
        &Rect::from_xywh(20.0, 20.0, 160.0, 160.0),
        0.0,
        90.0,
        false,
        &white(Style::Fill),
    );
    // Inside the chord segment between (180, 100) and (100, 180): beyond the
    // chord (x + y > 280) and within the 80px radius of (100, 100).
    assert!(
        lit(&buffer, 150, 150),
        "the open arc's chord segment must be filled"
    );
    assert!(
        !lit(&buffer, 120, 120),
        "the center side of the chord stays empty"
    );
}

#[test]
fn hairline_points_land_on_the_containing_pixel() {
    // Pixel i spans [i, i+1) (Skia's bw_pt_hair_proc floors), the same rule
    // as hairline lines: a point at (50.5, 50.5) or (50.9, 50.9) is pixel
    // (50, 50).
    use skia_rs_canvas::PointMode;
    use skia_rs_core::Point;

    for p in [Point::new(50.5, 50.5), Point::new(50.9, 50.9)] {
        let mut paint = white(Style::Stroke);
        paint.set_stroke_width(0.0);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_points(PointMode::Points, &[p], &paint);
        assert!(lit(&buffer, 50, 50), "{p:?} must light pixel (50, 50)");
        assert!(!lit(&buffer, 51, 51), "{p:?} must not light pixel (51, 51)");
    }
}

#[test]
fn antialiased_hairline_in_the_top_row_is_drawn() {
    // After the half-pixel shift, y = 0.25 becomes -0.25; a negative
    // `fract()` would give row 0 negative coverage and drop it.
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_anti_alias(true);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(10.0, 0.25),
        Point::new(150.0, 0.25),
        &paint,
    );
    let red = buffer.get_pixel(80, 0).unwrap_or(Color::BLACK).red();
    assert!(red > 100, "row 0 must carry most of the line, got {red}");
}

#[test]
fn aliased_hairlines_are_half_open() {
    // SkScan::HairLine excludes the last pixel: a hairline from x = 20 to
    // x = 180 covers pixels 20..=179, and a zero-length hairline is empty.
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(20.0, 50.5),
        Point::new(180.0, 50.5),
        &paint,
    );
    assert!(lit(&buffer, 20, 50), "first pixel drawn");
    assert!(lit(&buffer, 179, 50), "second-to-last pixel drawn");
    assert!(!lit(&buffer, 180, 50), "the end pixel is excluded");
}

#[test]
fn capped_hairlines_extend_to_cover_their_end_pixel() {
    // Skia's hair_path extends open contours for non-butt caps
    // (extend_pts), so a square- or round-capped hairline covers the end
    // pixel that the half-open butt-capped hairline leaves out.
    use skia_rs_paint::StrokeCap;

    let draw = |cap: StrokeCap| {
        let mut paint = white(Style::Stroke);
        paint.set_stroke_width(0.0);
        paint.set_stroke_cap(cap);
        let mut b = PathBuilder::new();
        // Major-axis span is [round(x0), round(x1)): butt covers 10..=19;
        // square (+-0.5) and round (+-pi/8) caps reach 20.59+ -> 10..=20.
        b.move_to(10.2, 10.5).line_to(20.2, 10.5);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_path(&b.build(), &paint);
        buffer
    };
    assert!(!lit(&draw(StrokeCap::Butt), 20, 10), "butt: half-open end");
    assert!(
        lit(&draw(StrokeCap::Square), 20, 10),
        "square cap covers the end"
    );
    assert!(
        lit(&draw(StrokeCap::Round), 20, 10),
        "round cap covers the end"
    );
}

#[test]
fn aliased_hairline_rect_is_a_closed_frame() {
    // SkScan::HairRect covers floor(left)..=floor(right) and
    // floor(top)..=floor(bottom), all four corners included.
    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_rect(&Rect::new(10.0, 10.0, 20.0, 20.0), &paint);
    for (x, y) in [(10, 10), (20, 10), (10, 20), (20, 20), (15, 10), (20, 15)] {
        assert!(lit(&buffer, x, y), "frame pixel ({x}, {y})");
    }
    assert!(!lit(&buffer, 15, 15), "interior stays empty");
    assert!(!lit(&buffer, 21, 20), "nothing past the frame");
}

#[test]
fn zero_length_capped_hairline_draws_a_dot() {
    // extend_pts moves a degenerate segment's ends apart horizontally, so a
    // round- or square-capped zero-length hairline still draws a pixel.
    use skia_rs_paint::StrokeCap;

    for cap in [StrokeCap::Round, StrokeCap::Square] {
        let mut paint = white(Style::Stroke);
        paint.set_stroke_width(0.0);
        paint.set_stroke_cap(cap);
        let mut b = PathBuilder::new();
        b.move_to(30.5, 30.5).line_to(30.5, 30.5);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_path(&b.build(), &paint);
        assert!(lit(&buffer, 30, 30), "{cap:?} cap draws a dot");
    }
}

#[test]
fn hairline_caps_extend_in_device_space() {
    // Under scale(10, 1) a vertical square-capped hairline must still reach
    // half a *device* pixel past its end (Skia extends after mapping).
    use skia_rs_paint::StrokeCap;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_stroke_cap(StrokeCap::Square);
    let mut b = PathBuilder::new();
    b.move_to(5.05, 40.2).line_to(5.05, 60.2);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    let mut canvas = Canvas::new_raster(&mut buffer);
    canvas.scale(10.0, 1.0);
    canvas.draw_path(&b.build(), &paint);
    assert!(lit(&buffer, 50, 60), "the square cap covers device row 60");
}

#[test]
fn far_off_hairlines_only_walk_the_device() {
    // A hairline spanning +-1e9 must finish promptly (the walk is clipped to
    // the device) and still draw its visible part.
    use skia_rs_core::Point;

    for anti_alias in [false, true] {
        let mut paint = white(Style::Stroke);
        paint.set_stroke_width(0.0);
        paint.set_anti_alias(anti_alias);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_line(
            Point::new(-1e9, 70.5),
            Point::new(1e9, 70.5),
            &paint,
        );
        assert!(
            lit(&buffer, 100, 70),
            "visible part drawn (aa = {anti_alias})"
        );
    }
}

/// A white paint with a horizontal red-to-blue gradient across the canvas.
fn gradient_hairline(anti_alias: bool) -> Paint {
    use skia_rs_core::{Color4f, Point};
    use skia_rs_paint::{LinearGradient, TileMode};
    use std::sync::Arc;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_anti_alias(anti_alias);
    paint.set_shader(Some(Arc::new(LinearGradient::new(
        Point::new(0.0, 0.0),
        Point::new(200.0, 0.0),
        vec![
            Color4f::new(1.0, 0.0, 0.0, 1.0),
            Color4f::new(0.0, 0.0, 1.0, 1.0),
        ],
        None,
        TileMode::Clamp,
    ))));
    paint
}

/// Whether the most strongly drawn pixel within two columns of `(x, y)`
/// follows the gradient: red-dominant on the left half, blue-dominant on
/// the right (a white, unshaded pixel has equal red and blue).
fn shaded(buffer: &PixelBuffer, x: i32, y: i32) -> bool {
    let strength = |c: Color| u32::from(c.red()) + u32::from(c.blue());
    let pixel = (x - 2..=x + 2)
        .filter_map(|px| buffer.get_pixel(px, y))
        .max_by_key(|c| strength(*c))
        .unwrap_or(Color::BLACK);
    if strength(pixel) == 0 {
        return false;
    }
    let (red, blue) = (pixel.red(), pixel.blue());
    if x < 100 { red > blue } else { blue > red }
}

#[test]
fn hairline_drawers_use_the_paint_shader() {
    use skia_rs_core::Point;

    for anti_alias in [false, true] {
        let paint = gradient_hairline(anti_alias);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.draw_line(Point::new(10.0, 20.5), Point::new(190.0, 20.5), &paint);
        canvas.draw_rect(&Rect::new(10.0, 40.0, 190.0, 60.0), &paint);
        canvas.draw_circle(Point::new(100.5, 130.5), 50.0, &paint);
        for (x, y) in [
            (20, 20),
            (180, 20),
            (20, 40),
            (180, 40),
            (50, 130),
            (150, 130),
        ] {
            assert!(shaded(&buffer, x, y), "aa={anti_alias}: ({x}, {y}) shaded");
        }
    }
}

#[test]
fn hairline_points_use_the_paint_shader() {
    use skia_rs_core::Point;

    let paint = gradient_hairline(false);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    let mut canvas = Canvas::new_raster(&mut buffer);
    canvas.draw_point(Point::new(20.5, 10.5), &paint);
    canvas.draw_point(Point::new(180.5, 10.5), &paint);
    assert!(shaded(&buffer, 20, 10) && shaded(&buffer, 180, 10));
}

#[test]
fn aliased_hairline_circle_respects_the_clip() {
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    let mut canvas = Canvas::new_raster(&mut buffer);
    canvas.clip_rect(
        &Rect::new(0.0, 0.0, 100.0, 200.0),
        skia_rs_canvas::ClipOp::Intersect,
        false,
    );
    canvas.draw_circle(Point::new(100.5, 100.5), 50.0, &paint);
    assert!(lit(&buffer, 50, 100), "left side inside the clip is drawn");
    assert!(
        !lit(&buffer, 150, 100),
        "right side outside the clip is not"
    );
}

#[test]
fn negative_weight_conic_fills_finitely() {
    // conic_to(w = -1) degrades to a line (SkPath::conicTo), so the fill is
    // the triangle (0,0)-(100,100)-(0,100) rather than inf/NaN edges.
    let mut b = PathBuilder::new();
    b.move_to(0.0, 0.0)
        .conic_to(150.0, 0.0, 100.0, 100.0, -1.0)
        .line_to(0.0, 100.0)
        .close();
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_path(&b.build(), &white(Style::Fill));
    assert!(lit(&buffer, 20, 80), "inside the triangle");
    assert!(!lit(&buffer, 80, 20), "outside the triangle");
    assert!(!lit(&buffer, 150, 150), "nothing spills to the device");
}

#[test]
fn antialiased_hairline_with_endpoints_beyond_i32_does_not_overflow() {
    // y = x / 4 through the origin: it crosses the device, but its
    // endpoints' minor coordinates (|y| = 4e9, exact in f32) saturate pixel
    // indices to i32::MIN/MAX. That must not overflow, and the visible
    // part must still land on the line.
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_anti_alias(true);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(-1.6e10, -4e9),
        Point::new(1.6e10, 4e9),
        &paint,
    );
    // Pixel centers (100.5, y + 0.5): the line is at y = 25.125 there.
    assert!(
        lit(&buffer, 100, 24) || lit(&buffer, 100, 25),
        "on the line"
    );
    assert!(!lit(&buffer, 100, 60), "nowhere else");
}
#[test]
fn non_finite_rects_are_rejected() {
    // SkCanvas quick-rejects non-finite rects instead of drawing to the
    // device edge.
    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    let mut canvas = Canvas::new_raster(&mut buffer);
    canvas.draw_rect(&Rect::new(10.0, 10.0, f32::INFINITY, 20.0), &paint);
    canvas.draw_rect(&Rect::new(10.0, f32::NAN, 20.0, 20.0), &white(Style::Fill));
    for x in 0..SIZE {
        for y in 0..30 {
            assert!(!lit(&buffer, x, y), "({x}, {y})");
        }
    }
}

#[test]
fn draw_point_honors_width_and_anti_aliasing() {
    use skia_rs_core::Point;

    // drawPoint is drawPoints(kPoints, 1): a 10-wide square.
    let mut wide = white(Style::Stroke);
    wide.set_stroke_width(10.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_point(Point::new(50.0, 50.0), &wide);
    assert!(lit(&buffer, 46, 46) && lit(&buffer, 53, 53), "square drawn");
    assert!(!lit(&buffer, 56, 50), "nothing past the square");

    // An AA hairline point covers a 1x1 square: four pixels at a quarter.
    let mut aa = white(Style::Stroke);
    aa.set_stroke_width(0.0);
    aa.set_anti_alias(true);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_point(Point::new(10.0, 10.0), &aa);
    for (x, y) in [(9, 9), (10, 9), (9, 10), (10, 10)] {
        let red = buffer.get_pixel(x, y).unwrap_or(Color::BLACK).red();
        assert!(
            (50..=80).contains(&red),
            "({x}, {y}) quarter coverage: {red}"
        );
    }
}

#[test]
fn hairline_draw_line_ignores_caps_like_skia_draw_points() {
    // Skia blits drawLine/drawPoints hairlines with SkScan::HairLine (no
    // cap extension); only hairline paths extend their ends for caps.
    use skia_rs_core::Point;
    use skia_rs_paint::StrokeCap;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_stroke_cap(StrokeCap::Square);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(
        Point::new(10.2, 10.5),
        Point::new(20.2, 10.5),
        &paint,
    );
    assert!(lit(&buffer, 10, 10) && lit(&buffer, 19, 10));
    assert!(!lit(&buffer, 20, 10), "no cap extension for draw_line");
}

#[test]
fn circles_under_perspective_match_their_path() {
    // A circle under a perspective CTM is a projected ellipse: draw_circle
    // must agree with drawing the same circle as a path.
    use skia_rs_core::{Matrix, Point};

    let perspective = Matrix {
        values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.004, 0.0, 1.0],
    };
    let render = |as_path: bool| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.concat(&perspective);
        if as_path {
            let mut b = PathBuilder::new();
            b.add_circle(100.0, 100.0, 40.0);
            canvas.draw_path(&b.build(), &white(Style::Fill));
        } else {
            canvas.draw_circle(Point::new(100.0, 100.0), 40.0, &white(Style::Fill));
        }
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| lit(&buffer, x, y))
            .count()
    };
    assert_eq!(render(false), render(true));
}

#[test]
fn negative_radius_hairline_circles_draw_nothing() {
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_circle(Point::new(50.0, 50.0), -10.0, &paint);
    for y in 0..SIZE {
        for x in 0..SIZE {
            assert!(!lit(&buffer, x, y), "({x}, {y})");
        }
    }
}

#[test]
fn partial_coverage_blends_toward_the_result_for_non_srcover_modes() {
    // Skia: lerp(dst, blend(src, dst), coverage). With Src on opaque white,
    // a partially covered pixel stays opaque (white blended toward red),
    // not a dark, partly transparent coverage-scaled red.
    use skia_rs_core::Point;
    use skia_rs_paint::BlendMode;

    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            buffer.set_pixel(x, y, Color::WHITE);
        }
    }
    let mut paint = Paint::new();
    paint.set_color32(Color::from_argb(255, 255, 0, 0));
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_anti_alias(true);
    paint.set_blend_mode(BlendMode::Src);
    Canvas::new_raster(&mut buffer).draw_point(Point::new(10.3, 10.7), &paint);
    for (x, y) in [(9, 10), (10, 10), (9, 11), (10, 11)] {
        let c = buffer.get_pixel(x, y).unwrap_or(Color::BLACK);
        assert_eq!(c.alpha(), 255, "({x}, {y}) stays opaque: {c:?}");
        assert_eq!(c.red(), 255, "({x}, {y}) red channel: {c:?}");
    }
}

#[test]
fn negative_radius_aa_circles_draw_nothing() {
    // Clamped to 0, the AA circle must not paint a half-coverage dot.
    use skia_rs_core::Point;

    let mut paint = white(Style::Fill);
    paint.set_anti_alias(true);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_circle(Point::new(10.5, 10.5), -5.0, &paint);
    for y in 0..SIZE {
        for x in 0..SIZE {
            assert!(
                buffer.get_pixel(x, y).is_none_or(|c| c.alpha() == 0),
                "({x}, {y})"
            );
        }
    }
}

#[test]
fn solid_spans_under_an_aa_clip_lerp_for_non_srcover_modes() {
    // Same rule as for shaded spans: Src on opaque white stays opaque at
    // partially clipped pixels.
    use skia_rs_paint::BlendMode;

    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            buffer.set_pixel(x, y, Color::WHITE);
        }
    }
    let mut paint = Paint::new();
    paint.set_color32(Color::from_argb(255, 255, 0, 0));
    paint.set_blend_mode(BlendMode::Src);
    let mut canvas = Canvas::new_raster(&mut buffer);
    let mut clip = PathBuilder::new();
    clip.add_circle(100.0, 100.0, 40.3);
    canvas.clip_path(&clip.build(), skia_rs_canvas::ClipOp::Intersect, true);
    canvas.draw_rect(&Rect::new(0.0, 0.0, 200.0, 200.0), &paint);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let c = buffer.get_pixel(x, y).unwrap_or(Color::BLACK);
            assert_eq!(c.alpha(), 255, "({x}, {y}) stays opaque: {c:?}");
        }
    }
}

#[test]
fn zero_radius_capped_aa_hairline_circles_draw_a_dot_like_aliased_ones() {
    use skia_rs_core::Point;
    use skia_rs_paint::StrokeCap;

    for anti_alias in [false, true] {
        let mut paint = white(Style::Stroke);
        paint.set_stroke_width(0.0);
        paint.set_stroke_cap(StrokeCap::Round);
        paint.set_anti_alias(anti_alias);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_circle(Point::new(30.5, 30.5), 0.0, &paint);
        let lit_any =
            (28..33).any(|y| (28..33).any(|x| buffer.get_pixel(x, y).is_some_and(|c| c.red() > 0)));
        assert!(
            lit_any,
            "aa={anti_alias}: a round-capped empty circle is a dot"
        );
    }
}

#[test]
fn sub_pixel_aa_hairlines_cover_their_column_once() {
    // A 0.3-px segment inside one column covers it by 0.3, not 1 + 0.3.
    use skia_rs_core::Point;

    let mut paint = white(Style::Stroke);
    paint.set_stroke_width(0.0);
    paint.set_anti_alias(true);
    let mut buffer = PixelBuffer::new(SIZE, SIZE);
    Canvas::new_raster(&mut buffer).draw_line(Point::new(10.0, 5.5), Point::new(10.3, 5.5), &paint);
    let red = buffer.get_pixel(10, 5).unwrap_or(Color::BLACK).red();
    assert!(red > 0 && red < 128, "coverage ~0.3: {red}");
}

/// Pixel values (red channel) of a whole buffer, for exact comparisons.
fn pixels(buffer: &PixelBuffer) -> Vec<u8> {
    (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .map(|(x, y)| buffer.get_pixel(x, y).map_or(0, |c| c.red()))
        .collect()
}

#[test]
fn aliased_rects_under_perspective_fill_their_quadrilateral() {
    use skia_rs_core::Matrix;

    let perspective = Matrix {
        values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.004, 0.0, 1.0],
    };
    let render = |as_path: bool| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.concat(&perspective);
        let rect = Rect::new(20.0, 20.0, 150.0, 150.0);
        if as_path {
            let mut b = PathBuilder::new();
            b.add_rect(&rect);
            canvas.draw_path(&b.build(), &white(Style::Fill));
        } else {
            canvas.draw_rect(&rect, &white(Style::Fill));
        }
        pixels(&buffer)
    };
    assert_eq!(render(false), render(true));
}

#[test]
fn reversed_oval_rects_are_sorted_for_every_draw() {
    let render = |rect: Rect| {
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        Canvas::new_raster(&mut buffer).draw_oval(&rect, &white(Style::Fill));
        pixels(&buffer)
    };
    let sorted = render(Rect::new(0.0, 0.0, 40.0, 40.0));
    assert!(sorted.iter().any(|&v| v > 0));
    assert_eq!(render(Rect::new(40.0, 40.0, 0.0, 0.0)), sorted);
}

#[test]
fn fills_crossing_the_perspective_horizon_are_clipped_to_it() {
    // w = 1 - x/100: the horizon is at local x = 100. A circle at x 50..150
    // is half behind it. The visible half maps to device x >= 100; the part
    // behind must not smear a flipped wedge into device x < 100.
    use skia_rs_core::{Matrix, Point};

    let perspective = Matrix {
        values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.01, 0.0, 1.0],
    };
    for anti_alias in [false, true] {
        let mut paint = white(Style::Fill);
        paint.set_anti_alias(anti_alias);
        let mut buffer = PixelBuffer::new(SIZE, SIZE);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.concat(&perspective);
        canvas.draw_circle(Point::new(100.0, 100.0), 50.0, &paint);
        for y in 0..SIZE {
            for x in 0..95 {
                assert!(
                    !lit(&buffer, x, y),
                    "aa={anti_alias}: no wedge at ({x}, {y})"
                );
            }
        }
        let visible = (0..SIZE).any(|y| (105..SIZE).any(|x| lit(&buffer, x, y)));
        assert!(visible, "aa={anti_alias}: the front half is drawn");
    }
}
