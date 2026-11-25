//! Text rendering example for skia-rs
//!
//! This example demonstrates:
//! - Font configuration with Font
//! - Text blob creation and rendering
//! - Drawing text with different sizes and colors
//!
//! Run with: cargo run --example `text_rendering` --features "text,codec"

fn main() {
    #[cfg(all(feature = "text", feature = "codec"))]
    text_rendering_main();

    #[cfg(not(all(feature = "text", feature = "codec")))]
    println!(
        "This example requires the 'text' and 'codec' features. Run with: cargo run --example text_rendering --features \"text,codec\""
    );
}

#[cfg(all(feature = "text", feature = "codec"))]
#[allow(
    clippy::too_many_lines,
    reason = "linear demo walkthrough; splitting into helpers would obscure the example"
)]
fn text_rendering_main() {
    use skia_rs_canvas::Surface;
    use skia_rs_codec::{ImageEncoder, PngEncoder};
    use skia_rs_core::{AlphaType, Color, ColorType, Point, Rect};
    use skia_rs_paint::{Paint, Style};
    use skia_rs_text::{Font, TextBlob, TextBlobBuilder, Typeface};
    use std::fs::File;
    use std::io::BufWriter;
    use std::sync::Arc;
    println!("skia-rs Text Rendering Example");
    println!("===============================\n");

    let width = 800;
    let height = 600;

    // Create a surface
    let mut surface =
        Surface::new_raster_n32_premul(width, height).expect("Failed to create surface");

    let mut canvas = surface.raster_canvas();

    // Clear with a light background
    canvas.clear(Color::from_rgb(250, 250, 252));

    // Load the platform's default sans-serif font. `Typeface::default_typeface()`
    // is a data-less placeholder that measures text but draws no glyphs, so
    // only fall back to it when no system font is installed.
    let typeface = Arc::new(Typeface::system_default().unwrap_or_else(|| {
        eprintln!("warning: no system sans-serif font found; text will be measured but not drawn");
        Typeface::default_typeface()
    }));
    println!("Using typeface: {}", typeface.family_name());

    // Title
    {
        let font = Font::new(Arc::clone(&typeface), 48.0);
        let mut paint = Paint::new();
        paint.set_anti_alias(true);
        paint.set_color32(Color::from_rgb(30, 30, 50));
        paint.set_style(Style::Fill);

        // Create a text blob for the title
        let blob = TextBlob::from_text("skia-rs Text Rendering", &font, Point::zero());
        canvas.draw_text_blob(&blob, 50.0, 80.0, &paint);
        println!("Drew title with 48pt font");
    }

    // Subtitle
    {
        let font = Font::new(Arc::clone(&typeface), 24.0);
        let mut paint = Paint::new();
        paint.set_anti_alias(true);
        paint.set_color32(Color::from_rgb(100, 100, 120));
        paint.set_style(Style::Fill);

        let blob = TextBlob::from_text("Pure Rust 2D Graphics Library", &font, Point::zero());
        canvas.draw_text_blob(&blob, 50.0, 120.0, &paint);
        println!("Drew subtitle with 24pt font");
    }

    // Different font sizes demonstration
    {
        let mut paint = Paint::new();
        paint.set_anti_alias(true);
        paint.set_style(Style::Fill);

        let sizes = [12.0, 16.0, 20.0, 24.0, 32.0];
        let mut y = 180.0;

        for size in sizes {
            let font = Font::new(Arc::clone(&typeface), size);
            paint.set_color32(Color::from_rgb(50, 50, 70));

            let text = format!("Font size: {size:.0}pt - The quick brown fox jumps");
            let blob = TextBlob::from_text(&text, &font, Point::zero());
            canvas.draw_text_blob(&blob, 50.0, y, &paint);

            y += size * 1.5;
        }
        println!("Drew text in {} different sizes", sizes.len());
    }

    // Colored text
    {
        let font = Font::new(Arc::clone(&typeface), 28.0);
        let mut paint = Paint::new();
        paint.set_anti_alias(true);
        paint.set_style(Style::Fill);

        let colors = [
            ("Red", Color::from_rgb(220, 50, 50)),
            ("Green", Color::from_rgb(50, 180, 50)),
            ("Blue", Color::from_rgb(50, 50, 220)),
            ("Orange", Color::from_rgb(255, 140, 0)),
            ("Purple", Color::from_rgb(150, 50, 200)),
        ];

        let mut x = 50.0;
        let y = 400.0;

        for (name, color) in colors {
            paint.set_color32(color);
            let blob = TextBlob::from_text(name, &font, Point::zero());
            canvas.draw_text_blob(&blob, x, y, &paint);
            x += 120.0;
        }
        println!("Drew colored text samples");
    }

    // Bold emulation (using embolden)
    {
        let mut font = Font::new(Arc::clone(&typeface), 24.0);
        font.set_embolden(true);
        let mut paint = Paint::new();
        paint.set_anti_alias(true);
        paint.set_color32(Color::from_rgb(30, 30, 50));
        paint.set_style(Style::Fill);

        let blob = TextBlob::from_text("Emboldened text for emphasis", &font, Point::zero());
        canvas.draw_text_blob(&blob, 50.0, 460.0, &paint);
        println!("Drew bold text");
    }

    // Text with background box
    {
        let font = Font::new(Arc::clone(&typeface), 20.0);

        // Draw background box
        let mut bg_paint = Paint::new();
        bg_paint.set_color32(Color::from_rgb(45, 55, 72));
        bg_paint.set_style(Style::Fill);

        let bg_rect = Rect::from_xywh(50.0, 490.0, 700.0, 40.0);
        canvas.draw_rect(&bg_rect, &bg_paint);

        // Draw text on background
        let mut text_paint = Paint::new();
        text_paint.set_anti_alias(true);
        text_paint.set_color32(Color::from_rgb(255, 255, 255));
        text_paint.set_style(Style::Fill);

        let blob = TextBlob::from_text(
            "White text on a dark background - great for contrast!",
            &font,
            Point::zero(),
        );
        canvas.draw_text_blob(&blob, 60.0, 518.0, &text_paint);
        println!("Drew text with background");
    }

    // Multi-run text blob using TextBlobBuilder
    {
        let mut builder = TextBlobBuilder::new();

        let normal_font = Font::new(Arc::clone(&typeface), 22.0);
        let mut bold_font = Font::new(Arc::clone(&typeface), 22.0);
        bold_font.set_embolden(true);

        // Add runs to the builder
        // Lay the runs out end to end using their measured advances.
        let runs = [
            ("Normal text ", &normal_font),
            ("BOLD ", &bold_font),
            ("inline.", &normal_font),
        ];
        let mut x = 0.0;
        for (text, font) in runs {
            builder.add_text(text, font, Point::new(x, 0.0));
            x += font.measure_text(text);
        }

        if let Some(blob) = builder.build() {
            let mut paint = Paint::new();
            paint.set_anti_alias(true);
            paint.set_color32(Color::from_rgb(80, 80, 100));
            paint.set_style(Style::Fill);

            canvas.draw_text_blob(&blob, 50.0, 560.0, &paint);
            println!("Drew multi-style text blob");
        }
    }

    // Footer
    {
        let small_font = Font::new(Arc::clone(&typeface), 14.0);
        let mut paint = Paint::new();
        paint.set_anti_alias(true);
        paint.set_color32(Color::from_rgb(150, 150, 160));
        paint.set_style(Style::Fill);

        let footer = format!(
            "skia-rs v{} - Pure Rust 2D Graphics",
            env!("CARGO_PKG_VERSION")
        );
        let blob = TextBlob::from_text(&footer, &small_font, Point::zero());
        canvas.draw_text_blob(&blob, 50.0, 590.0, &paint);
    }

    // Save to file
    let pixels = surface.pixels();
    let stride = usize::try_from(width).expect("width is non-negative") * 4;

    // Save using codec
    let output_path = "text_rendering_output.png";
    let file = File::create(output_path).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    // Create image info and encode
    let img_info =
        skia_rs_codec::ImageInfo::new(width, height, ColorType::Rgba8888, AlphaType::Premul);

    if let Some(image) = skia_rs_codec::Image::from_raster_data(&img_info, pixels, stride) {
        let encoder = PngEncoder::new();
        encoder
            .encode(&image, &mut writer)
            .expect("Failed to encode PNG");
        println!("\nSaved output to: {output_path}");
    } else {
        eprintln!("Failed to create image from surface pixels");
    }

    println!("\nExample complete!");
}
