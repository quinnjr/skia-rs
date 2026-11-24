//! Compare skia-rs performance against original Skia.
//!
//! This example runs a set of benchmarks and compares the results
//! against reference timings from the original Skia library.
//!
//! Usage:
//!   cargo run --example `skia_comparison` -p skia-rs-bench --release
#![allow(
    clippy::cast_precision_loss,
    reason = "loop indices are bounded by small fixed ranges (5, 100); precision loss cannot occur in practice"
)]

use skia_rs_bench::skia_comparison::{
    BenchmarkRunner, ComparisonReport, ComparisonResult, reference_timings,
};
use skia_rs_canvas::Surface;
use skia_rs_core::{Color, Matrix, Point, Rect};
use skia_rs_paint::Paint;
use skia_rs_path::PathBuilder;

fn run_core_benchmarks(report: &mut ComparisonReport, runner: &BenchmarkRunner) {
    println!("Running core benchmarks...");

    // Matrix multiply
    let time = runner.run(|| {
        let a = Matrix::translate(10.0, 20.0);
        let b = Matrix::rotate(0.5);
        std::hint::black_box(a.concat(&b));
    });
    report.add(
        ComparisonResult::compare(
            "matrix_multiply",
            time,
            reference_timings::get("matrix_multiply").unwrap_or(time),
        )
        .with_notes("3x3 matrix concatenation"),
    );

    // Matrix invert
    let m = Matrix::translate(10.0, 20.0).concat(&Matrix::scale(2.0, 3.0));
    let time = runner.run(|| {
        std::hint::black_box(m.invert());
    });
    report.add(
        ComparisonResult::compare(
            "matrix_invert",
            time,
            reference_timings::get("matrix_invert").unwrap_or(time),
        )
        .with_notes("3x3 matrix inversion"),
    );

    // Point transform
    let m = Matrix::translate(10.0, 20.0);
    let p = Point::new(100.0, 200.0);
    let time = runner.run(|| {
        std::hint::black_box(m.map_point(p));
    });
    report.add(
        ComparisonResult::compare(
            "point_transform",
            time,
            reference_timings::get("point_transform").unwrap_or(time),
        )
        .with_notes("Single point transform"),
    );
}

fn run_path_benchmarks(report: &mut ComparisonReport, runner: &BenchmarkRunner) {
    println!("Running path benchmarks...");

    // Build a 100-line path
    let mut builder = PathBuilder::new();
    builder.move_to(0.0, 0.0);
    for i in 0..100 {
        builder.line_to(i as f32 * 10.0, (i % 2) as f32 * 50.0);
    }
    builder.close();
    let path = builder.build();

    // Path bounds
    let time = runner.run(|| {
        std::hint::black_box(path.bounds());
    });
    report.add(
        ComparisonResult::compare(
            "path_bounds",
            time,
            reference_timings::get("path_bounds").unwrap_or(time),
        )
        .with_notes("100-segment path bounds"),
    );

    // Path contains
    let query_point = Point::new(500.0, 25.0);
    let time = runner.run(|| {
        std::hint::black_box(path.contains(query_point));
    });
    report.add(
        ComparisonResult::compare(
            "path_contains",
            time,
            reference_timings::get("path_contains").unwrap_or(time),
        )
        .with_notes("Point containment test"),
    );

    // Path construction
    let time = runner.run(|| {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        for i in 0..100 {
            b.line_to(i as f32 * 10.0, (i % 2) as f32 * 50.0);
        }
        b.close();
        std::hint::black_box(b.build());
    });
    report.add(
        ComparisonResult::compare(
            "path_100_lines",
            time,
            reference_timings::get("path_100_lines").unwrap_or(time),
        )
        .with_notes("Build 100-line path"),
    );
}

fn run_surface_benchmarks(report: &mut ComparisonReport, runner: &BenchmarkRunner) {
    println!("Running surface benchmarks...");

    // Surface creation - 256x256
    let time = runner.run(|| {
        std::hint::black_box(Surface::new_raster_n32_premul(256, 256));
    });
    report.add(
        ComparisonResult::compare(
            "surface_create_256",
            time,
            reference_timings::get("surface_create_256").unwrap_or(time),
        )
        .with_notes("256x256 RGBA surface"),
    );

    // Surface creation - 1080p
    let runner_slow = BenchmarkRunner::new().iterations(100).warmup(10);
    let time = runner_slow.run(|| {
        std::hint::black_box(Surface::new_raster_n32_premul(1920, 1080));
    });
    report.add(
        ComparisonResult::compare(
            "surface_create_1080p",
            time,
            reference_timings::get("surface_create_1080p").unwrap_or(time),
        )
        .with_notes("1920x1080 RGBA surface"),
    );

    // Surface clear
    let mut surface = Surface::new_raster_n32_premul(1920, 1080).unwrap();
    let time = runner_slow.run(|| {
        let mut canvas = surface.raster_canvas();
        canvas.clear(Color::WHITE);
    });
    report.add(
        ComparisonResult::compare(
            "surface_clear",
            time,
            reference_timings::get("surface_clear").unwrap_or(time),
        )
        .with_notes("1080p clear"),
    );
}

fn run_drawing_benchmarks(report: &mut ComparisonReport, runner: &BenchmarkRunner) {
    println!("Running drawing benchmarks...");

    let mut surface = Surface::new_raster_n32_premul(1000, 1000).unwrap();
    let fill_paint = Paint::new();
    let rect = Rect::from_xywh(100.0, 100.0, 200.0, 200.0);

    // Draw rect
    let time = runner.run(|| {
        let mut canvas = surface.raster_canvas();
        canvas.draw_rect(&rect, &fill_paint);
    });
    report.add(
        ComparisonResult::compare(
            "draw_rect",
            time,
            reference_timings::get("draw_rect").unwrap_or(time),
        )
        .with_notes("200x200 filled rect"),
    );

    // Draw circle
    let time = runner.run(|| {
        let mut canvas = surface.raster_canvas();
        canvas.draw_circle(Point::new(500.0, 500.0), 100.0, &fill_paint);
    });
    report.add(
        ComparisonResult::compare(
            "draw_circle",
            time,
            reference_timings::get("draw_circle").unwrap_or(time),
        )
        .with_notes("r=100 filled circle"),
    );

    // Draw star path
    let mut builder = PathBuilder::new();
    let cx = 500.0;
    let cy = 500.0;
    let outer: f32 = 100.0;
    let inner: f32 = 40.0;
    for i in 0..5 {
        let angle_o = (i as f32).mul_add(72.0, -90.0).to_radians();
        let angle_i = ((i as f32).mul_add(72.0, 36.0) - 90.0).to_radians();
        // Unfused, as Skia's own drawing of this star computes it.
        #[allow(clippy::suboptimal_flops, reason = "match Skia's unfused rounding")]
        let (ox, oy, ix, iy) = (
            cx + outer * angle_o.cos(),
            cy + outer * angle_o.sin(),
            cx + inner * angle_i.cos(),
            cy + inner * angle_i.sin(),
        );
        if i == 0 {
            builder.move_to(ox, oy);
        } else {
            builder.line_to(ox, oy);
        }
        builder.line_to(ix, iy);
    }
    builder.close();
    let star = builder.build();

    let time = runner.run(|| {
        let mut canvas = surface.raster_canvas();
        canvas.draw_path(&star, &fill_paint);
    });
    report.add(
        ComparisonResult::compare(
            "draw_path_star",
            time,
            reference_timings::get("draw_path_star").unwrap_or(time),
        )
        .with_notes("10-vertex star"),
    );
}

fn main() {
    println!("skia-rs vs Skia Performance Comparison");
    println!("======================================\n");

    let mut report = ComparisonReport::new();

    // Add system info
    report.set_metadata("platform", std::env::consts::OS);
    report.set_metadata("arch", std::env::consts::ARCH);

    let runner = BenchmarkRunner::new().iterations(1000).warmup(100);

    run_core_benchmarks(&mut report, &runner);
    run_path_benchmarks(&mut report, &runner);
    run_surface_benchmarks(&mut report, &runner);
    run_drawing_benchmarks(&mut report, &runner);

    // ==========================================================================
    // Generate Report
    // ==========================================================================
    println!("\n{}", report.format());

    // Save reports
    if let Err(e) = report.save("skia_comparison_report.md") {
        eprintln!("Failed to save markdown report: {e}");
    } else {
        println!("Saved: skia_comparison_report.md");
    }

    if let Err(e) = report.save_json("skia_comparison_report.json") {
        eprintln!("Failed to save JSON report: {e}");
    } else {
        println!("Saved: skia_comparison_report.json");
    }
}
