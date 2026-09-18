# skia-rs-codec

Image encoding and decoding for [skia-rs](https://github.com/quinnjr/skia-rs), a pure Rust implementation of the Skia 2D graphics library.

## Features

- **Image**: Immutable image with pixel access
- **PNG**: Read/write support
- **JPEG**: Read/write support
- **GIF**: Read/write support
- **WebP**: Read/write support
- **Format detection**: Automatic format identification

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `png`   | ✅ | PNG codec |
| `jpeg`  | ✅ | JPEG codec |
| `gif`   | ✅ | GIF codec |
| `webp`  | ❌ | WebP codec (pulls in `libwebp-sys`) |

## Usage

```rust
use skia_rs_codec::{Image, ImageFormat};

// Load an image
let image = Image::from_file("photo.jpg")?;

// Get image info
println!("Size: {}x{}", image.width(), image.height());

// Detect format from bytes
let format = ImageFormat::from_bytes(&data);

// Encode to PNG
let png_data = image.encode_png()?;
```

## License

MIT OR Apache-2.0

See the [main repository](https://github.com/quinnjr/skia-rs) for more information.
