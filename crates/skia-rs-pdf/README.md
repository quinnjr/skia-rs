# skia-rs-pdf

PDF generation for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **PDF documents**: Create multi-page PDFs
- **Drawing**: Same canvas API as raster rendering
- **Text**: Font embedding and text layout
- **Images**: Embed raster images

## Usage

```rust
use skia_rs_pdf::{PdfDocument, PdfPage};

// Create a document
let mut doc = PdfDocument::new();

// Add a page
let mut page = PdfPage::new(612.0, 792.0); // Letter size
let canvas = page.canvas();

// Draw on the page (same API as Surface)
canvas.draw_rect(&rect, &paint);
canvas.draw_string("Hello, PDF!", 100.0, 100.0, &font, &paint);

doc.add_page(page);

// Save to file
doc.save("output.pdf")?;
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
