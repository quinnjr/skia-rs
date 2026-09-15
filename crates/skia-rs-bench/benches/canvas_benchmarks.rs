//! Canvas and surface benchmarks.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use skia_rs_bench::{
    canvas_sizes, create_rng, generate_complex_path, generate_simple_path, random_rects,
};
use skia_rs_canvas::{Canvas, ClipOp, PictureRecorder, SaveLayerRec, Surface};
use skia_rs_core::{Color, ImageInfo, Matrix, Point, Rect};
use skia_rs_paint::Paint;
use skia_rs_path::PathBuilder;
use std::hint::black_box;

fn bench_canvas_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canvas/creation");

    for (name, (w, h)) in [
        ("small", canvas_sizes::SMALL),
        ("medium", canvas_sizes::MEDIUM),
        ("hd", canvas_sizes::HD),
        ("uhd", canvas_sizes::UHD),
    ] {
        group.bench_with_input(BenchmarkId::new("new", name), &(w, h), |b, &(w, h)| {
            b.iter(|| Canvas::new(black_box(w), black_box(h)));
        });
    }

    group.finish();
}

fn bench_canvas_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canvas/queries");

    let canvas = Canvas::new(1920, 1080);

    group.bench_function("width", |b| b.iter(|| black_box(&canvas).width()));

    group.bench_function("height", |b| b.iter(|| black_box(&canvas).height()));

    group.bench_function("save_count", |b| b.iter(|| black_box(&canvas).save_count()));

    group.bench_function("total_matrix", |b| {
        b.iter(|| black_box(&canvas).total_matrix());
    });

    group.bench_function("clip_bounds", |b| {
        b.iter(|| black_box(&canvas).clip_bounds());
    });

    group.finish();
}

fn bench_canvas_save_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canvas/save_restore");

    group.bench_function("save", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.save();
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("restore", |b| {
        b.iter_batched(
            || {
                let mut canvas = Canvas::new(1920, 1080);
                canvas.save();
                canvas
            },
            |mut canvas| {
                canvas.restore();
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("save_layer", |b| {
        let rec = SaveLayerRec::default();
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.save_layer(&rec);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Deep save stack
    for depth in [10, 50, 100, 500] {
        group.bench_with_input(BenchmarkId::new("deep_save", depth), &depth, |b, &depth| {
            b.iter_batched(
                || Canvas::new(1920, 1080),
                |mut canvas| {
                    for _ in 0..depth {
                        canvas.save();
                    }
                    canvas
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(
            BenchmarkId::new("deep_restore", depth),
            &depth,
            |b, &depth| {
                b.iter_batched(
                    || {
                        let mut canvas = Canvas::new(1920, 1080);
                        for _ in 0..depth {
                            canvas.save();
                        }
                        canvas
                    },
                    |mut canvas| {
                        for _ in 0..depth {
                            canvas.restore();
                        }
                        canvas
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("restore_to_count", depth),
            &depth,
            |b, &depth| {
                b.iter_batched(
                    || {
                        let mut canvas = Canvas::new(1920, 1080);
                        for _ in 0..depth {
                            canvas.save();
                        }
                        canvas
                    },
                    |mut canvas| {
                        canvas.restore_to_count(1);
                        canvas
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_canvas_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canvas/transforms");

    group.bench_function("translate", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.translate(black_box(100.0), black_box(200.0));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("scale", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.scale(black_box(2.0), black_box(2.0));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("rotate", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.rotate(black_box(45.0));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("skew", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.skew(black_box(0.5), black_box(0.25));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("concat", |b| {
        let matrix = Matrix::translate(100.0, 200.0).concat(&Matrix::scale(2.0, 2.0));
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.concat(black_box(&matrix));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("set_matrix", |b| {
        let matrix = Matrix::translate(100.0, 200.0).concat(&Matrix::scale(2.0, 2.0));
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.set_matrix(black_box(&matrix));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("reset_matrix", |b| {
        b.iter_batched(
            || {
                let mut canvas = Canvas::new(1920, 1080);
                canvas.translate(100.0, 200.0);
                canvas
            },
            |mut canvas| {
                canvas.reset_matrix();
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Chain transforms
    group.bench_function("chain_transforms", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.translate(100.0, 200.0);
                canvas.scale(2.0, 2.0);
                canvas.rotate(45.0);
                canvas.skew(0.1, 0.1);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

#[allow(clippy::too_many_lines)]
fn bench_canvas_clip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canvas/clip");

    let rect = Rect::from_xywh(100.0, 100.0, 500.0, 300.0);
    let path = generate_simple_path(20);

    group.bench_function("clip_rect/intersect", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.clip_rect(black_box(&rect), ClipOp::Intersect, false);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("clip_rect/intersect_aa", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.clip_rect(black_box(&rect), ClipOp::Intersect, true);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("clip_rect/difference", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.clip_rect(black_box(&rect), ClipOp::Difference, false);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("clip_path/intersect", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.clip_path(black_box(&path), ClipOp::Intersect, false);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Multiple clip operations
    for count in [5, 10, 20] {
        let mut rng = create_rng();
        let bounds = Rect::from_xywh(0.0, 0.0, 1920.0, 1080.0);
        let rects = random_rects(&mut rng, count, &bounds, 200.0);

        group.bench_with_input(
            BenchmarkId::new("multiple_clips", count),
            &rects,
            |b, rects| {
                b.iter_batched(
                    || Canvas::new(1920, 1080),
                    |mut canvas| {
                        for rect in rects {
                            canvas.clip_rect(rect, ClipOp::Intersect, false);
                        }
                        canvas
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Multiple clip_path operations
    for count in [5, 10, 20] {
        let mut rng = create_rng();
        let bounds = Rect::from_xywh(0.0, 0.0, 1920.0, 1080.0);
        let rects = random_rects(&mut rng, count, &bounds, 200.0);

        group.bench_with_input(
            BenchmarkId::new("multiple_clip_paths", count),
            &rects,
            |b, rects| {
                b.iter_batched(
                    || Canvas::new(1920, 1080),
                    |mut canvas| {
                        for rect in rects {
                            let mut rect_path_builder = PathBuilder::new();
                            rect_path_builder.add_rect(rect);
                            let rect_path = rect_path_builder.build();
                            canvas.clip_path(&rect_path, ClipOp::Intersect, false);
                        }
                        canvas
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Multiple anti-aliased clip_path operations
    for count in [5, 10, 20] {
        let mut rng = create_rng();
        let bounds = Rect::from_xywh(0.0, 0.0, 1920.0, 1080.0);
        let rects = random_rects(&mut rng, count, &bounds, 200.0);

        group.bench_with_input(
            BenchmarkId::new("multiple_clip_paths_aa", count),
            &rects,
            |b, rects| {
                b.iter_batched(
                    || Canvas::new(1920, 1080),
                    |mut canvas| {
                        for rect in rects {
                            let mut rect_path_builder = PathBuilder::new();
                            rect_path_builder.add_rect(rect);
                            let rect_path = rect_path_builder.build();
                            canvas.clip_path(&rect_path, ClipOp::Intersect, true);
                        }
                        canvas
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_canvas_drawing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canvas/drawing");

    let paint = Paint::new();
    let rect = Rect::from_xywh(100.0, 100.0, 200.0, 150.0);
    let path = generate_simple_path(50);
    let complex_path = generate_complex_path(100);

    group.bench_function("clear", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.clear(Color::WHITE);
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_rect", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_rect(black_box(&rect), black_box(&paint));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_oval", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_oval(black_box(&rect), black_box(&paint));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_circle", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_circle(
                    Point::new(black_box(500.0), black_box(400.0)),
                    black_box(100.0),
                    black_box(&paint),
                );
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_round_rect", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_round_rect(
                    black_box(&rect),
                    black_box(10.0),
                    black_box(10.0),
                    black_box(&paint),
                );
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_line", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_line(
                    Point::new(black_box(0.0), black_box(0.0)),
                    Point::new(black_box(100.0), black_box(100.0)),
                    black_box(&paint),
                );
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_path/simple", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_path(black_box(&path), black_box(&paint));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("draw_path/complex", |b| {
        b.iter_batched(
            || Canvas::new(1920, 1080),
            |mut canvas| {
                canvas.draw_path(black_box(&complex_path), black_box(&paint));
                canvas
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("Surface");

    for (name, (w, h)) in [
        ("small", canvas_sizes::SMALL),
        ("medium", canvas_sizes::MEDIUM),
        ("hd", canvas_sizes::HD),
    ] {
        let info = ImageInfo::new_n32_premul(w, h).unwrap();

        group.bench_with_input(BenchmarkId::new("new_raster", name), &info, |b, info| {
            b.iter(|| Surface::new_raster(black_box(info), None));
        });
    }

    let info = ImageInfo::new_n32_premul(1920, 1080).unwrap();
    let surface = Surface::new_raster(&info, None).unwrap();

    group.bench_function("info", |b| b.iter(|| black_box(&surface).info()));

    group.bench_function("width", |b| b.iter(|| black_box(&surface).width()));

    group.bench_function("height", |b| b.iter(|| black_box(&surface).height()));

    group.bench_function("row_bytes", |b| b.iter(|| black_box(&surface).row_bytes()));

    group.bench_function("pixels", |b| b.iter(|| black_box(&surface).pixels().len()));

    group.bench_function("canvas", |b| {
        b.iter_batched(
            || Surface::new_raster(&info, None).unwrap(),
            |mut surface| {
                // `Canvas` now borrows from `Surface`, so take the canvas
                // inside the closure and drop it before returning.
                let _ = surface.canvas();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_picture(c: &mut Criterion) {
    let mut group = c.benchmark_group("Picture");

    group.bench_function("recorder_new", |b| b.iter(PictureRecorder::new));

    let bounds = Rect::from_xywh(0.0, 0.0, 1920.0, 1080.0);

    group.bench_function("begin_recording", |b| {
        b.iter_batched(
            PictureRecorder::new,
            |mut recorder| {
                let _ = recorder.begin_recording(black_box(bounds));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("finish_recording", |b| {
        b.iter_batched(
            || {
                let mut recorder = PictureRecorder::new();
                recorder.begin_recording(bounds);
                recorder
            },
            |mut recorder| recorder.finish_recording(),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_canvas_creation,
    bench_canvas_queries,
    bench_canvas_save_restore,
    bench_canvas_transforms,
    bench_canvas_clip,
    bench_canvas_drawing,
    bench_surface,
    bench_picture,
);

criterion_main!(benches);
