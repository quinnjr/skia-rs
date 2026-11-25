//! Backend conformance test cases.
//!
//! Shared by `examples/conformance_gen.rs` (which renders them to PNG files)
//! and `tests/conformance.rs` (which checks them against the committed
//! reference images in `conformance/reference/`). The case names and order
//! match the list in `conformance/README.md`.

use skia_rs_canvas::{Canvas, Surface};
use skia_rs_core::{Color, Point, Rect};
use skia_rs_paint::{Paint, Style};
use skia_rs_path::{FillType, PathBuilder};

/// Width and height, in pixels, of every conformance image.
pub const SIZE: i32 = 200;

/// Background every case is drawn over.
const BACKGROUND: Color = Color::from_rgb(255, 255, 255);

/// A named drawing routine.
pub struct Case {
    /// File stem of the generated image (`<name>.png`).
    pub name: &'static str,
    /// Draws the case onto a cleared `SIZE`x`SIZE` canvas.
    pub draw: fn(&mut Canvas<'_>),
}

/// All conformance cases, in `conformance/README.md` order.
pub const CASES: &[Case] = &[
    Case {
        name: "solid_rect",
        draw: solid_rect,
    },
    Case {
        name: "stroked_rect",
        draw: stroked_rect,
    },
    Case {
        name: "circle_fill",
        draw: circle_fill,
    },
    Case {
        name: "circle_stroke",
        draw: circle_stroke,
    },
    Case {
        name: "oval",
        draw: oval,
    },
    Case {
        name: "lines",
        draw: lines,
    },
    Case {
        name: "path_triangle",
        draw: path_triangle,
    },
    Case {
        name: "path_star",
        draw: path_star,
    },
    Case {
        name: "path_bezier",
        draw: path_bezier,
    },
    Case {
        name: "nested_paths",
        draw: nested_paths,
    },
    Case {
        name: "rounded_rect",
        draw: rounded_rect,
    },
    Case {
        name: "overlapping_shapes",
        draw: overlapping_shapes,
    },
    Case {
        name: "alpha_blending",
        draw: alpha_blending,
    },
    Case {
        name: "antialiased_shapes",
        draw: antialiased_shapes,
    },
    Case {
        name: "stroke_widths",
        draw: stroke_widths,
    },
];

/// Render `case` with the raster backend and return tightly packed,
/// premultiplied RGBA8888 pixels (`SIZE * SIZE * 4` bytes).
#[must_use]
pub fn render_raster(case: &Case) -> Vec<u8> {
    let mut surface = Surface::new_raster_n32_premul(SIZE, SIZE).expect("raster surface");
    {
        let mut canvas = surface.raster_canvas();
        canvas.clear(BACKGROUND);
        (case.draw)(&mut canvas);
    }
    surface.pixels().to_vec()
}

fn fill(color: Color) -> Paint {
    let mut paint = Paint::new();
    paint.set_color32(color);
    paint.set_style(Style::Fill);
    paint
}

fn stroke(color: Color, width: f32) -> Paint {
    let mut paint = Paint::new();
    paint.set_color32(color);
    paint.set_style(Style::Stroke);
    paint.set_stroke_width(width);
    paint
}

fn solid_rect(canvas: &mut Canvas<'_>) {
    let paint = fill(Color::from_rgb(220, 40, 40));
    canvas.draw_rect(&Rect::from_xywh(40.0, 50.0, 120.0, 100.0), &paint);
}

fn stroked_rect(canvas: &mut Canvas<'_>) {
    let paint = stroke(Color::from_rgb(40, 90, 220), 6.0);
    canvas.draw_rect(&Rect::from_xywh(40.0, 50.0, 120.0, 100.0), &paint);
}

fn circle_fill(canvas: &mut Canvas<'_>) {
    let paint = fill(Color::from_rgb(40, 170, 80));
    canvas.draw_circle(Point::new(100.0, 100.0), 70.0, &paint);
}

fn circle_stroke(canvas: &mut Canvas<'_>) {
    let paint = stroke(Color::from_rgb(200, 120, 20), 8.0);
    canvas.draw_circle(Point::new(100.0, 100.0), 70.0, &paint);
}

fn oval(canvas: &mut Canvas<'_>) {
    let paint = fill(Color::from_rgb(130, 60, 190));
    canvas.draw_oval(&Rect::from_xywh(20.0, 55.0, 160.0, 90.0), &paint);
}

fn lines(canvas: &mut Canvas<'_>) {
    let colors = [
        Color::from_rgb(220, 40, 40),
        Color::from_rgb(40, 170, 80),
        Color::from_rgb(40, 90, 220),
        Color::from_rgb(200, 120, 20),
        Color::from_rgb(130, 60, 190),
    ];
    let mut y = 30.0;
    for (i, color) in (0u8..).zip(colors) {
        let paint = stroke(color, f32::from(i) + 1.0);
        canvas.draw_line(Point::new(20.0, y), Point::new(180.0, y + 20.0), &paint);
        y += 32.0;
    }
}

fn path_triangle(canvas: &mut Canvas<'_>) {
    let mut builder = PathBuilder::new();
    builder
        .move_to(100.0, 25.0)
        .line_to(175.0, 170.0)
        .line_to(25.0, 170.0)
        .close();
    canvas.draw_path(&builder.build(), &fill(Color::from_rgb(220, 40, 40)));
}

fn path_star(canvas: &mut Canvas<'_>) {
    // Five-pointed star drawn as one self-intersecting pentagram, so the
    // default winding fill also fills the center pentagon.
    let points = [
        (100.0, 20.0),
        (147.0, 165.0),
        (24.0, 75.0),
        (176.0, 75.0),
        (53.0, 165.0),
    ];
    let mut builder = PathBuilder::new();
    builder.move_to(points[0].0, points[0].1);
    for &(x, y) in &points[1..] {
        builder.line_to(x, y);
    }
    builder.close();
    canvas.draw_path(&builder.build(), &fill(Color::from_rgb(230, 180, 20)));
}

fn path_bezier(canvas: &mut Canvas<'_>) {
    let mut builder = PathBuilder::new();
    builder
        .move_to(20.0, 160.0)
        .cubic_to(60.0, 10.0, 140.0, 190.0, 180.0, 40.0);
    canvas.draw_path(&builder.build(), &stroke(Color::from_rgb(40, 90, 220), 5.0));
}

fn nested_paths(canvas: &mut Canvas<'_>) {
    // Outer rect with an inner rect cut out by the even-odd rule.
    let mut builder = PathBuilder::new();
    builder
        .fill_type(FillType::EvenOdd)
        .add_rect(&Rect::from_xywh(30.0, 30.0, 140.0, 140.0))
        .add_rect(&Rect::from_xywh(70.0, 70.0, 60.0, 60.0));
    canvas.draw_path(&builder.build(), &fill(Color::from_rgb(40, 170, 80)));
}

fn rounded_rect(canvas: &mut Canvas<'_>) {
    let paint = fill(Color::from_rgb(40, 90, 220));
    canvas.draw_round_rect(
        &Rect::from_xywh(30.0, 45.0, 140.0, 110.0),
        24.0,
        24.0,
        &paint,
    );
}

fn overlapping_shapes(canvas: &mut Canvas<'_>) {
    canvas.draw_rect(
        &Rect::from_xywh(25.0, 25.0, 100.0, 100.0),
        &fill(Color::from_rgb(220, 40, 40)),
    );
    canvas.draw_circle(
        Point::new(120.0, 120.0),
        55.0,
        &fill(Color::from_rgb(40, 90, 220)),
    );
}

fn alpha_blending(canvas: &mut Canvas<'_>) {
    let centers = [(80.0, 80.0), (120.0, 80.0), (100.0, 120.0)];
    let colors = [
        Color::from_argb(128, 255, 0, 0),
        Color::from_argb(128, 0, 200, 0),
        Color::from_argb(128, 0, 0, 255),
    ];
    for ((x, y), color) in centers.into_iter().zip(colors) {
        canvas.draw_circle(Point::new(x, y), 50.0, &fill(color));
    }
}

fn antialiased_shapes(canvas: &mut Canvas<'_>) {
    let mut paint = fill(Color::from_rgb(30, 30, 30));
    paint.set_anti_alias(true);
    canvas.draw_circle(Point::new(60.0, 60.0), 40.0, &paint);

    let mut builder = PathBuilder::new();
    builder
        .move_to(110.0, 20.0)
        .line_to(185.0, 95.0)
        .line_to(120.0, 100.0)
        .close();
    canvas.draw_path(&builder.build(), &paint);

    let mut line = stroke(Color::from_rgb(30, 30, 30), 3.0);
    line.set_anti_alias(true);
    canvas.draw_line(Point::new(20.0, 180.0), Point::new(180.0, 120.0), &line);
}

fn stroke_widths(canvas: &mut Canvas<'_>) {
    let mut y = 25.0;
    for width in [1.0, 2.0, 4.0, 8.0, 12.0] {
        let paint = stroke(Color::from_rgb(30, 30, 30), width);
        canvas.draw_line(Point::new(20.0, y), Point::new(180.0, y), &paint);
        y += 35.0;
    }
}
