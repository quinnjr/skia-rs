//! Codec trait and format-specific implementations.
//!
//! Codecs handle encoding and decoding of images in various formats.

use crate::Image;
use std::io::{Read, Write};
use thiserror::Error;

/// Errors that can occur during codec operations.
#[derive(Debug, Error)]
pub enum CodecError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid or corrupt data.
    #[error("invalid data: {0}")]
    InvalidData(String),
    /// Unsupported format or feature.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Encoding error.
    #[error("encoding error: {0}")]
    EncodingError(String),
    /// Decoding error.
    #[error("decoding error: {0}")]
    DecodingError(String),
}

/// Result type for codec operations.
pub type CodecResult<T> = Result<T, CodecError>;

/// Largest per-dimension pixel extent accepted from untrusted encoded
/// data, matching `SkCodec`'s guard against absurd sizes.
const MAX_DIM: usize = 1 << 24;
/// Largest total decoded-canvas byte size accepted from untrusted encoded
/// data. A format's individual dimensions can each be within [`MAX_DIM`]
/// yet still multiply out to tens of gigabytes, so this bounds the
/// product as well.
const MAX_CANVAS_BYTES: usize = 1 << 28; // 256 MiB

/// Detected image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// PNG format.
    Png,
    /// JPEG format.
    Jpeg,
    /// GIF format.
    Gif,
    /// WebP format.
    WebP,
    /// BMP format.
    Bmp,
    /// ICO format.
    Ico,
    /// WBMP format (Wireless Bitmap).
    Wbmp,
    /// AVIF format (AV1 Image File Format).
    Avif,
    /// Camera RAW format.
    Raw,
    /// Unknown format.
    Unknown,
}

impl ImageFormat {
    /// Detect format from magic bytes.
    #[must_use]
    pub fn from_magic(data: &[u8]) -> Self {
        if data.len() < 8 {
            return Self::Unknown;
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return Self::Png;
        }

        // JPEG: FF D8 FF
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return Self::Jpeg;
        }

        // GIF: GIF87a or GIF89a
        if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            return Self::Gif;
        }

        // WebP: RIFF....WEBP
        if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
            return Self::WebP;
        }

        // BMP: BM
        if data.starts_with(b"BM") {
            return Self::Bmp;
        }

        // ICO: 00 00 01 00
        if data.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
            return Self::Ico;
        }

        // AVIF: ftyp box with 'avif' or 'avis' brand
        // HEIF/AVIF files start with ftyp box: size (4 bytes) + "ftyp" + brand
        if data.len() >= 12 && &data[4..8] == b"ftyp" {
            // Check for AVIF brands
            if &data[8..12] == b"avif" || &data[8..12] == b"avis" || &data[8..12] == b"mif1" {
                return Self::Avif;
            }
        }

        // WBMP: Type 0, FixHeaderField 0, followed by width/height encoded
        // as WBMP multibyte integers (bit 7 = "more bytes follow"). Both
        // dimensions must be non-zero for a valid WBMP. The previous check
        // only accepted single-byte (< 128) dimensions and rejected any
        // image wider or taller than 127 pixels.
        if data.len() >= 4 && data[0] == 0 && data[1] == 0 {
            let mut offset = 2;
            if let (Some(width), Some(height)) = (
                read_wbmp_int(data, &mut offset),
                read_wbmp_int(data, &mut offset),
            ) {
                if width > 0 && height > 0 {
                    return Self::Wbmp;
                }
            }
        }

        Self::Unknown
    }

    /// Get the typical file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Gif => "gif",
            Self::WebP => "webp",
            Self::Bmp => "bmp",
            Self::Ico => "ico",
            Self::Wbmp => "wbmp",
            Self::Avif => "avif",
            Self::Raw => "raw",
            Self::Unknown => "",
        }
    }

    /// Get the MIME type for this format.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
            Self::Bmp => "image/bmp",
            Self::Ico => "image/x-icon",
            Self::Wbmp => "image/vnd.wap.wbmp",
            Self::Avif => "image/avif",
            Self::Raw => "image/x-raw",
            Self::Unknown => "application/octet-stream",
        }
    }
}

/// A codec that can decode images.
pub trait ImageDecoder: Send + Sync {
    /// Decode an image from a reader.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the stream cannot be read or the data is
    /// not valid for this decoder's format.
    fn decode<R: Read>(&self, reader: R) -> CodecResult<Image>;

    /// Decode an image from bytes.
    ///
    /// # Errors
    /// Returns [`CodecError`] if `data` is not valid for this decoder's
    /// format.
    fn decode_bytes(&self, data: &[u8]) -> CodecResult<Image> {
        self.decode(std::io::Cursor::new(data))
    }

    /// Get the format this decoder handles.
    fn format(&self) -> ImageFormat;

    /// Check if this decoder can handle the given data.
    fn can_decode(&self, data: &[u8]) -> bool {
        ImageFormat::from_magic(data) == self.format()
    }
}

/// A codec that can encode images.
pub trait ImageEncoder: Send + Sync {
    /// Encode an image to a writer.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the image cannot be encoded (e.g.
    /// unsupported color type) or the writer fails.
    fn encode<W: Write>(&self, image: &Image, writer: W) -> CodecResult<()>;

    /// Encode an image to bytes.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the image cannot be encoded.
    fn encode_bytes(&self, image: &Image) -> CodecResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.encode(image, &mut buf)?;
        Ok(buf)
    }

    /// Get the format this encoder produces.
    fn format(&self) -> ImageFormat;
}

/// Quality setting for lossy encoders.
#[derive(Debug, Clone, Copy)]
pub struct EncoderQuality(u8);

impl EncoderQuality {
    /// Create a quality setting (0-100).
    #[must_use]
    pub fn new(quality: u8) -> Self {
        Self(quality.min(100))
    }

    /// Get the quality value.
    #[must_use]
    pub const fn value(&self) -> u8 {
        self.0
    }

    /// Default quality (80).
    pub const DEFAULT: Self = Self(80);

    /// High quality (95).
    pub const HIGH: Self = Self(95);

    /// Low quality (50).
    pub const LOW: Self = Self(50);
}

impl Default for EncoderQuality {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Unpremultiply an interleaved 8-bit-per-channel buffer in place when the
/// source image carries premultiplied pixels.
///
/// Image encoders must store **straight** (unpremultiplied) alpha: the PNG
/// spec (11.2.3), BMP V3+ alpha, and WebP all define pixel samples as
/// non-premultiplied, and JPEG discards alpha entirely so its RGB must be the
/// unpremultiplied color. Canvas-backed images now report
/// [`skia_rs_core::AlphaType::Premul`], so encoders call this before writing.
///
/// Alpha lives at byte index 3 of every 4-byte pixel in both RGBA and BGRA
/// layouts, so this is safe to run on either ordering.
fn unpremultiply_for_encode(image: &Image, buf: &mut [u8]) {
    if image.alpha_type() == skia_rs_core::AlphaType::Premul {
        skia_rs_core::unpremultiply_in_place(buf);
    }
}

// =============================================================================
// Simple PNG Codec (stub - would use png crate for real implementation)
// =============================================================================

/// PNG decoder.
#[derive(Debug, Default)]
pub struct PngDecoder;

impl PngDecoder {
    /// Create a new PNG decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageDecoder for PngDecoder {
    #[cfg(feature = "png")]
    fn decode<R: Read>(&self, reader: R) -> CodecResult<Image> {
        let mut decoder = png::Decoder::new(reader);
        // The `png` crate defaults to `Transformations::IDENTITY`, which
        // hands back raw palette indices for PNG-8 and 16-bit-per-channel
        // samples for deep images — neither of which our RGBA8 pipeline can
        // consume. Enable EXPAND | STRIP_16 (the crate's
        // `normalize_to_color8`) so paletted images expand to RGB(A),
        // sub-8-bit grayscale expands to 8-bit, tRNS chunks expand to an
        // alpha channel, and 16-bit samples are stripped to 8-bit. This
        // matches SkPngCodec, which always decodes to an 8-bit destination.
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut png_reader = decoder
            .read_info()
            .map_err(|e| CodecError::DecodingError(e.to_string()))?;

        // Pull the iCCP profile bytes out of the PNG info before calling
        // next_frame (next_frame borrows the reader mutably). The `png`
        // crate stores the raw profile bytes in `Info::icc_profile`; we
        // feed them to `IccProfile::from_bytes` to build a real
        // ColorSpace, falling back to None if parsing fails.
        let icc_color_space = png_reader
            .info()
            .icc_profile
            .as_ref()
            .and_then(|bytes| skia_rs_core::IccProfile::from_bytes(bytes))
            .map(|profile| profile.color_space().clone());

        let mut buf = vec![0; png_reader.output_buffer_size()];
        let info = png_reader
            .next_frame(&mut buf)
            .map_err(|e| CodecError::DecodingError(e.to_string()))?;

        let width = i32::try_from(info.width)
            .map_err(|_| CodecError::InvalidData("PNG width exceeds i32 range".into()))?;
        let height = i32::try_from(info.height)
            .map_err(|_| CodecError::InvalidData("PNG height exceeds i32 range".into()))?;
        let width_usize = usize::try_from(info.width).unwrap_or(0);
        let height_usize = usize::try_from(info.height).unwrap_or(0);

        // Convert to RGBA if necessary
        let pixels = match info.color_type {
            png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
            png::ColorType::Rgb => {
                let rgb = &buf[..info.buffer_size()];
                let mut rgba = Vec::with_capacity(width_usize * height_usize * 4);
                for chunk in rgb.chunks(3) {
                    rgba.push(chunk[0]);
                    rgba.push(chunk[1]);
                    rgba.push(chunk[2]);
                    rgba.push(255);
                }
                rgba
            }
            png::ColorType::GrayscaleAlpha => {
                let ga = &buf[..info.buffer_size()];
                let mut rgba = Vec::with_capacity(width_usize * height_usize * 4);
                for chunk in ga.chunks(2) {
                    rgba.push(chunk[0]);
                    rgba.push(chunk[0]);
                    rgba.push(chunk[0]);
                    rgba.push(chunk[1]);
                }
                rgba
            }
            png::ColorType::Grayscale => {
                let gray = &buf[..info.buffer_size()];
                let mut rgba = Vec::with_capacity(width_usize * height_usize * 4);
                for &g in gray {
                    rgba.push(g);
                    rgba.push(g);
                    rgba.push(g);
                    rgba.push(255);
                }
                rgba
            }
            png::ColorType::Indexed => {
                return Err(CodecError::Unsupported("Unsupported PNG color type".into()));
            }
        };

        let mut info = crate::ImageInfo::new(
            width,
            height,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Unpremul,
        );
        info.color_space = icc_color_space;

        Image::from_raster_data_owned(info, pixels, width_usize * 4)
            .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
    }

    #[cfg(not(feature = "png"))]
    fn decode<R: Read>(&self, _reader: R) -> CodecResult<Image> {
        Err(CodecError::Unsupported(
            "PNG decoding requires the 'png' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Png
    }
}

/// PNG encoder.
#[derive(Debug, Default)]
pub struct PngEncoder;

impl PngEncoder {
    /// Create a new PNG encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageEncoder for PngEncoder {
    #[cfg(feature = "png")]
    fn encode<W: Write>(&self, image: &Image, writer: W) -> CodecResult<()> {
        let mut encoder = png::Encoder::new(
            writer,
            u32::try_from(image.width()).unwrap_or(0),
            u32::try_from(image.height()).unwrap_or(0),
        );
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);

        let mut png_writer = encoder
            .write_header()
            .map_err(|e| CodecError::EncodingError(e.to_string()))?;

        let pixels = image
            .peek_pixels()
            .ok_or_else(|| CodecError::EncodingError("Cannot access pixels".into()))?;

        // Convert to RGBA if necessary based on color type
        let mut rgba_data = match image.color_type() {
            skia_rs_core::ColorType::Rgba8888 => pixels.to_vec(),
            skia_rs_core::ColorType::Bgra8888 => {
                let mut rgba = Vec::with_capacity(pixels.len());
                for chunk in pixels.chunks(4) {
                    rgba.push(chunk[2]); // R
                    rgba.push(chunk[1]); // G
                    rgba.push(chunk[0]); // B
                    rgba.push(chunk[3]); // A
                }
                rgba
            }
            _ => {
                return Err(CodecError::Unsupported(
                    "Unsupported color type for PNG encoding".into(),
                ));
            }
        };

        // PNG stores straight alpha; unpremultiply premultiplied input.
        unpremultiply_for_encode(image, &mut rgba_data);

        png_writer
            .write_image_data(&rgba_data)
            .map_err(|e| CodecError::EncodingError(e.to_string()))?;

        Ok(())
    }

    #[cfg(not(feature = "png"))]
    fn encode<W: Write>(&self, _image: &Image, _writer: W) -> CodecResult<()> {
        Err(CodecError::Unsupported(
            "PNG encoding requires the 'png' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Png
    }
}

// =============================================================================
// JPEG Codec (stub)
// =============================================================================

/// JPEG decoder.
#[derive(Debug, Default)]
pub struct JpegDecoder;

impl JpegDecoder {
    /// Create a new JPEG decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageDecoder for JpegDecoder {
    #[cfg(feature = "jpeg")]
    fn decode<R: Read>(&self, reader: R) -> CodecResult<Image> {
        let mut decoder = jpeg_decoder::Decoder::new(reader);
        let pixels = decoder
            .decode()
            .map_err(|e| CodecError::DecodingError(e.to_string()))?;
        let info = decoder
            .info()
            .ok_or_else(|| CodecError::DecodingError("No image info".into()))?;

        // JPEG ICC profiles arrive via APP2 markers; jpeg-decoder
        // re-assembles them and exposes a single contiguous byte slice.
        // Parse into a ColorSpace if present and well-formed.
        let icc_color_space = decoder
            .icc_profile()
            .and_then(|bytes| skia_rs_core::IccProfile::from_bytes(&bytes))
            .map(|profile| profile.color_space().clone());

        let width = i32::from(info.width);
        let height = i32::from(info.height);
        let width_usize = usize::from(info.width);
        let height_usize = usize::from(info.height);

        // Convert to RGBA
        let rgba = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => {
                let mut rgba = Vec::with_capacity(width_usize * height_usize * 4);
                for chunk in pixels.chunks(3) {
                    rgba.push(chunk[0]);
                    rgba.push(chunk[1]);
                    rgba.push(chunk[2]);
                    rgba.push(255);
                }
                rgba
            }
            jpeg_decoder::PixelFormat::L8 => {
                let mut rgba = Vec::with_capacity(width_usize * height_usize * 4);
                for &g in &pixels {
                    rgba.push(g);
                    rgba.push(g);
                    rgba.push(g);
                    rgba.push(255);
                }
                rgba
            }
            _ => {
                return Err(CodecError::Unsupported(
                    "Unsupported JPEG pixel format".into(),
                ));
            }
        };

        let mut img_info = crate::ImageInfo::new(
            width,
            height,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Opaque,
        );
        img_info.color_space = icc_color_space;

        Image::from_raster_data_owned(img_info, rgba, width_usize * 4)
            .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
    }

    #[cfg(not(feature = "jpeg"))]
    fn decode<R: Read>(&self, _reader: R) -> CodecResult<Image> {
        Err(CodecError::Unsupported(
            "JPEG decoding requires the 'jpeg' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Jpeg
    }
}

/// JPEG encoder.
#[derive(Debug)]
pub struct JpegEncoder {
    quality: EncoderQuality,
}

impl JpegEncoder {
    /// Create a new JPEG encoder with default quality.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            quality: EncoderQuality::DEFAULT,
        }
    }

    /// Create a JPEG encoder with specified quality.
    #[must_use]
    pub const fn with_quality(quality: EncoderQuality) -> Self {
        Self { quality }
    }

    /// Get the quality setting.
    #[must_use]
    pub const fn quality(&self) -> EncoderQuality {
        self.quality
    }
}

impl Default for JpegEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageEncoder for JpegEncoder {
    #[cfg(feature = "jpeg")]
    fn encode<W: Write>(&self, image: &Image, mut writer: W) -> CodecResult<()> {
        let pixels = image
            .peek_pixels()
            .ok_or_else(|| CodecError::EncodingError("Cannot access pixels".into()))?;

        // JPEG discards alpha, so we must emit the *unpremultiplied* color.
        // Build an RGBA buffer, unpremultiply if the source is premul, then
        // drop the alpha channel.
        let mut rgba = match image.color_type() {
            skia_rs_core::ColorType::Rgba8888 => pixels.to_vec(),
            skia_rs_core::ColorType::Bgra8888 => {
                let mut rgba = Vec::with_capacity(pixels.len());
                for chunk in pixels.chunks(4) {
                    rgba.push(chunk[2]); // R
                    rgba.push(chunk[1]); // G
                    rgba.push(chunk[0]); // B
                    rgba.push(chunk[3]); // A
                }
                rgba
            }
            _ => {
                return Err(CodecError::Unsupported(
                    "Unsupported color type for JPEG encoding".into(),
                ));
            }
        };
        unpremultiply_for_encode(image, &mut rgba);
        let width_usize = usize::try_from(image.width()).unwrap_or(0);
        let height_usize = usize::try_from(image.height()).unwrap_or(0);
        let rgb_capacity = width_usize
            .checked_mul(height_usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| CodecError::EncodingError("image dimensions overflow".into()))?;
        let mut rgb = Vec::with_capacity(rgb_capacity);
        for chunk in rgba.chunks_exact(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }

        let jpeg_width = u16::try_from(image.width())
            .map_err(|_| CodecError::EncodingError("image width exceeds JPEG limit".into()))?;
        let jpeg_height = u16::try_from(image.height())
            .map_err(|_| CodecError::EncodingError("image height exceeds JPEG limit".into()))?;
        let encoder = jpeg_encoder::Encoder::new(&mut writer, self.quality.value());
        encoder
            .encode(&rgb, jpeg_width, jpeg_height, jpeg_encoder::ColorType::Rgb)
            .map_err(|e| CodecError::EncodingError(e.to_string()))?;

        Ok(())
    }

    #[cfg(not(feature = "jpeg"))]
    fn encode<W: Write>(&self, _image: &Image, _writer: W) -> CodecResult<()> {
        Err(CodecError::Unsupported(
            "JPEG encoding requires the 'jpeg' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Jpeg
    }
}

// =============================================================================
// GIF Codec
// =============================================================================

/// GIF decoder.
#[derive(Debug, Default)]
pub struct GifDecoder;

impl GifDecoder {
    /// Create a new GIF decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Decode an animated GIF into all of its frames.
    ///
    /// The single-frame [`ImageDecoder::decode`] path returns only the
    /// first frame — convenient for static GIFs but loses every
    /// subsequent frame of an animation. `decode_animated` walks the
    /// full frame stream, preserving per-frame delay, disposal, and
    /// offset.
    ///
    /// Each frame's image is its own [`Image`] sized to the frame's
    /// local width/height (not the canvas). The `x_offset` / `y_offset`
    /// fields on [`crate::AnimationFrame`] place the frame inside the
    /// animation canvas; callers performing playback composite those
    /// frames onto the canvas according to the frame's
    /// [`crate::DisposalMethod`] and [`crate::BlendMethod`].
    ///
    /// GIF delays arrive in 10ms ticks (`Frame::delay`), which this
    /// converts to [`std::time::Duration`]. A zero delay is preserved
    /// as `Duration::ZERO` — downstream renderers typically clamp these
    /// to a minimum of 20ms to avoid CPU-pegging animations, but that
    /// policy belongs to the renderer, not the codec.
    ///
    /// # Errors
    /// Returns [`CodecError`] if `data` is not a valid GIF, contains no
    /// frames, or a frame image cannot be built.
    #[cfg(feature = "gif")]
    pub fn decode_animated(&self, data: &[u8]) -> CodecResult<crate::AnimatedImage> {
        let mut options = gif::DecodeOptions::new();
        options.set_color_output(gif::ColorOutput::RGBA);
        let mut decoder = options
            .read_info(std::io::Cursor::new(data))
            .map_err(|e| CodecError::DecodingError(e.to_string()))?;

        // Logical-screen (canvas) dimensions. Individual frames may be
        // smaller and are positioned via their left/top offsets.
        let canvas_width = i32::from(decoder.width());
        let canvas_height = i32::from(decoder.height());

        let mut frames = Vec::new();
        while let Some(frame) = decoder
            .read_next_frame()
            .map_err(|e| CodecError::DecodingError(e.to_string()))?
        {
            let width = i32::from(frame.width);
            let height = i32::from(frame.height);
            let info = crate::ImageInfo::new(
                width,
                height,
                skia_rs_core::ColorType::Rgba8888,
                skia_rs_core::AlphaType::Unpremul,
            );
            let image = Image::from_raster_data(&info, &frame.buffer, usize::from(frame.width) * 4)
                .ok_or_else(|| CodecError::DecodingError("Failed to build frame image".into()))?;

            frames.push(crate::AnimationFrame {
                image,
                delay: std::time::Duration::from_millis(u64::from(frame.delay) * 10),
                dispose: frame.dispose.into(),
                // GIF is palette/transparent-index only; it has no
                // "blend" concept, and every renderer treats frames as
                // alpha-blended over the canvas.
                blend: crate::BlendMethod::Over,
                x_offset: i32::from(frame.left),
                y_offset: i32::from(frame.top),
            });
        }

        if frames.is_empty() {
            return Err(CodecError::DecodingError("No frames in GIF".into()));
        }

        Ok(crate::AnimatedImage {
            loop_count: decoder.repeat().into(),
            frames,
            canvas_width,
            canvas_height,
        })
    }

    /// Stub for builds without the `gif` feature.
    #[cfg(not(feature = "gif"))]
    pub fn decode_animated(&self, _data: &[u8]) -> CodecResult<crate::AnimatedImage> {
        Err(CodecError::Unsupported(
            "GIF decoding requires the 'gif' feature".into(),
        ))
    }
}

impl ImageDecoder for GifDecoder {
    #[cfg(feature = "gif")]
    fn decode<R: Read>(&self, reader: R) -> CodecResult<Image> {
        let mut decoder = gif::DecodeOptions::new();
        decoder.set_color_output(gif::ColorOutput::RGBA);
        let mut decoder = decoder
            .read_info(reader)
            .map_err(|e| CodecError::DecodingError(e.to_string()))?;

        // Decode to the logical-screen canvas, not the (possibly smaller)
        // first frame. SkGifCodec always reports the logical-screen size as
        // the image dimensions and composites the first frame at its
        // left/top offset over a transparent canvas.
        let canvas_width = usize::from(decoder.width());
        let canvas_height = usize::from(decoder.height());

        // Reject absurd logical-screen sizes before allocating the canvas
        // buffer. The GIF header's width/height are attacker-controlled u16
        // fields (max 65535 each), so a tiny malicious file can otherwise
        // request a ~17GB allocation, aborting the process instead of
        // returning an error. `MAX_DIM` matches SkCodec's per-dimension
        // guard; `MAX_CANVAS_BYTES` additionally bounds the total
        // allocation, since GIF's u16 dimensions never individually exceed
        // `MAX_DIM` but can still multiply out to tens of gigabytes.
        if canvas_width > MAX_DIM || canvas_height > MAX_DIM {
            return Err(CodecError::InvalidData(
                "GIF logical screen dimensions too large".into(),
            ));
        }
        let canvas_stride = canvas_width
            .checked_mul(4)
            .ok_or_else(|| CodecError::InvalidData("GIF canvas row too large".into()))?;
        let canvas_size = canvas_stride
            .checked_mul(canvas_height)
            .ok_or_else(|| CodecError::InvalidData("GIF canvas too large".into()))?;
        if canvas_size > MAX_CANVAS_BYTES {
            return Err(CodecError::InvalidData("GIF canvas too large".into()));
        }

        let first_frame = decoder
            .read_next_frame()
            .map_err(|e| CodecError::DecodingError(e.to_string()))?
            .ok_or_else(|| CodecError::DecodingError("No frames in GIF".into()))?;

        let frame_width = usize::from(first_frame.width);
        let frame_height = usize::from(first_frame.height);
        let left = usize::from(first_frame.left);
        let top = usize::from(first_frame.top);

        // Transparent canvas; blit the frame into place, clipping any part
        // that falls outside the logical screen.
        let mut pixels = vec![0u8; canvas_size];
        for row in 0..frame_height {
            let dst_y = top + row;
            if dst_y >= canvas_height {
                break;
            }
            let copy_w = frame_width.min(canvas_width.saturating_sub(left));
            if copy_w == 0 {
                continue;
            }
            let src_off = row * frame_width * 4;
            let dst_off = dst_y * canvas_stride + left * 4;
            pixels[dst_off..dst_off + copy_w * 4]
                .copy_from_slice(&first_frame.buffer[src_off..src_off + copy_w * 4]);
        }

        let info = crate::ImageInfo::new(
            i32::try_from(canvas_width).unwrap_or(i32::MAX),
            i32::try_from(canvas_height).unwrap_or(i32::MAX),
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Unpremul,
        );

        Image::from_raster_data_owned(info, pixels, canvas_stride)
            .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
    }

    #[cfg(not(feature = "gif"))]
    fn decode<R: Read>(&self, _reader: R) -> CodecResult<Image> {
        Err(CodecError::Unsupported(
            "GIF decoding requires the 'gif' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Gif
    }
}

// =============================================================================
// WebP Codec
// =============================================================================

/// WebP decoder.
#[derive(Debug, Default)]
pub struct WebpDecoder;

impl WebpDecoder {
    /// Create a new WebP decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageDecoder for WebpDecoder {
    #[cfg(feature = "webp")]
    fn decode<R: Read>(&self, mut reader: R) -> CodecResult<Image> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(CodecError::Io)?;

        let decoder = webp::Decoder::new(&data);
        let webp_image = decoder
            .decode()
            .ok_or_else(|| CodecError::DecodingError("Failed to decode WebP".into()))?;

        let width = i32::try_from(webp_image.width())
            .map_err(|_| CodecError::InvalidData("WebP width exceeds i32 range".into()))?;
        let height = i32::try_from(webp_image.height())
            .map_err(|_| CodecError::InvalidData("WebP height exceeds i32 range".into()))?;
        let width_usize = usize::try_from(webp_image.width()).unwrap_or(0);
        let height_usize = usize::try_from(webp_image.height()).unwrap_or(0);
        let has_alpha = webp_image.is_alpha();

        // WebP returns RGB for images without alpha and RGBA for images
        // with alpha. Normalize to RGBA so the rest of the pipeline sees
        // a predictable 4-byte-per-pixel layout.
        let src = webp_image.to_vec();
        let pixels = if has_alpha {
            src
        } else {
            // Expand RGB -> RGBA with opaque alpha.
            let mut rgba = Vec::with_capacity(width_usize * height_usize * 4);
            for chunk in src.chunks_exact(3) {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(chunk[2]);
                rgba.push(255);
            }
            rgba
        };

        let info = crate::ImageInfo::new(
            width,
            height,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Unpremul,
        );

        Image::from_raster_data_owned(info, pixels, width_usize * 4)
            .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
    }

    #[cfg(not(feature = "webp"))]
    fn decode<R: Read>(&self, _reader: R) -> CodecResult<Image> {
        Err(CodecError::Unsupported(
            "WebP decoding requires the 'webp' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::WebP
    }
}

/// WebP encoder.
#[derive(Debug)]
pub struct WebpEncoder {
    // Only read by the feature-gated encode path; the stub keeps the type
    // constructible so downstream code compiles without the feature.
    #[cfg_attr(not(feature = "webp"), allow(dead_code))]
    quality: EncoderQuality,
    #[cfg_attr(not(feature = "webp"), allow(dead_code))]
    lossless: bool,
}

impl WebpEncoder {
    /// Create a new WebP encoder with default quality.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            quality: EncoderQuality::DEFAULT,
            lossless: false,
        }
    }

    /// Create a WebP encoder with specified quality.
    #[must_use]
    pub const fn with_quality(quality: EncoderQuality) -> Self {
        Self {
            quality,
            lossless: false,
        }
    }

    /// Create a lossless WebP encoder.
    #[must_use]
    pub const fn lossless() -> Self {
        Self {
            quality: EncoderQuality::DEFAULT,
            lossless: true,
        }
    }
}

impl Default for WebpEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageEncoder for WebpEncoder {
    #[cfg(feature = "webp")]
    fn encode<W: Write>(&self, image: &Image, mut writer: W) -> CodecResult<()> {
        let pixels = image
            .peek_pixels()
            .ok_or_else(|| CodecError::EncodingError("Cannot access pixels".into()))?;
        let width = u32::try_from(image.width()).unwrap_or(0);
        let height = u32::try_from(image.height()).unwrap_or(0);

        // Convert to RGBA if needed
        let mut rgba = match image.color_type() {
            skia_rs_core::ColorType::Rgba8888 => pixels.to_vec(),
            skia_rs_core::ColorType::Bgra8888 => {
                let mut rgba = Vec::with_capacity(pixels.len());
                for chunk in pixels.chunks(4) {
                    rgba.push(chunk[2]);
                    rgba.push(chunk[1]);
                    rgba.push(chunk[0]);
                    rgba.push(chunk[3]);
                }
                rgba
            }
            _ => {
                return Err(CodecError::Unsupported(
                    "Unsupported color type for WebP encoding".into(),
                ));
            }
        };

        // WebP stores straight alpha; unpremultiply premultiplied input.
        unpremultiply_for_encode(image, &mut rgba);

        let webp_encoder = webp::Encoder::from_rgba(&rgba, width, height);
        let webp_bytes = if self.lossless {
            webp_encoder.encode_lossless()
        } else {
            webp_encoder.encode(f32::from(self.quality.value()))
        };

        writer.write_all(&webp_bytes).map_err(CodecError::Io)?;
        Ok(())
    }

    #[cfg(not(feature = "webp"))]
    fn encode<W: Write>(&self, _image: &Image, _writer: W) -> CodecResult<()> {
        Err(CodecError::Unsupported(
            "WebP encoding requires the 'webp' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::WebP
    }
}

// =============================================================================
// BMP Codec
// =============================================================================

/// BMP decoder.
#[derive(Debug, Default)]
pub struct BmpDecoder;

impl BmpDecoder {
    /// Create a new BMP decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageDecoder for BmpDecoder {
    fn decode<R: Read>(&self, mut reader: R) -> CodecResult<Image> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(CodecError::Io)?;
        decode_bmp(&data)
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Bmp
    }
}

/// BMP encoder.
#[derive(Debug, Default)]
pub struct BmpEncoder;

impl BmpEncoder {
    /// Create a new BMP encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageEncoder for BmpEncoder {
    fn encode<W: Write>(&self, image: &Image, mut writer: W) -> CodecResult<()> {
        let raw_pixels = image
            .peek_pixels()
            .ok_or_else(|| CodecError::EncodingError("Cannot access pixels".into()))?;

        // BMP V3+ alpha is straight (unpremultiplied); unpremultiply if the
        // source image carries premultiplied pixels. This assumes an
        // RGBA/BGRA 8888 layout (the encoder always writes 32-bit BGRA).
        let mut pixels = raw_pixels.to_vec();
        unpremultiply_for_encode(image, &mut pixels);
        let pixels = &pixels[..];

        let width = u32::try_from(image.width()).unwrap_or(0);
        let height = u32::try_from(image.height()).unwrap_or(0);
        let width_i32 = image.width();
        let height_i32 = image.height();

        // Calculate row padding (each row must be aligned to 4 bytes)
        let row_size = (width * 4 + 3) & !3; // 32-bit BGRA, aligned
        let pixel_data_size = row_size * height;
        let file_size = 14 + 40 + pixel_data_size; // header + DIB header + pixels

        // BITMAPFILEHEADER (14 bytes)
        writer.write_all(b"BM")?; // Signature
        writer.write_all(&file_size.to_le_bytes())?; // File size
        writer.write_all(&[0u8; 4])?; // Reserved
        writer.write_all(&(14u32 + 40).to_le_bytes())?; // Pixel data offset

        // BITMAPINFOHEADER (40 bytes)
        writer.write_all(&40u32.to_le_bytes())?; // Header size
        writer.write_all(&width_i32.to_le_bytes())?; // Width
        writer.write_all(&height_i32.to_le_bytes())?; // Height (positive = bottom-up)
        writer.write_all(&1u16.to_le_bytes())?; // Planes
        writer.write_all(&32u16.to_le_bytes())?; // Bits per pixel (32-bit BGRA)
        writer.write_all(&0u32.to_le_bytes())?; // Compression (BI_RGB = none)
        writer.write_all(&pixel_data_size.to_le_bytes())?; // Image size
        writer.write_all(&2835u32.to_le_bytes())?; // X pixels per meter (~72 DPI)
        writer.write_all(&2835u32.to_le_bytes())?; // Y pixels per meter
        writer.write_all(&0u32.to_le_bytes())?; // Colors used
        writer.write_all(&0u32.to_le_bytes())?; // Important colors

        // Pixel data (bottom-to-top, BGRA)
        let stride = usize::try_from(width).unwrap_or(0) * 4;
        let row_padding = usize::try_from(row_size).unwrap_or(0) - stride;
        let padding = vec![0u8; row_padding];

        for y in (0..usize::try_from(height).unwrap_or(0)).rev() {
            let row_start = y * stride;
            let row = &pixels[row_start..row_start + stride];

            // Convert RGBA to BGRA
            for chunk in row.chunks(4) {
                writer.write_all(&[chunk[2], chunk[1], chunk[0], chunk[3]])?; // BGRA
            }

            if row_padding > 0 {
                writer.write_all(&padding)?;
            }
        }

        Ok(())
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Bmp
    }
}

/// Channel masks for a `BI_BITFIELDS` BMP.
#[derive(Debug, Clone, Copy)]
struct BitfieldMasks {
    red: u32,
    green: u32,
    blue: u32,
    alpha: u32,
}

impl BitfieldMasks {
    /// Extract `mask`'s channel out of `value` and scale it to a full 8-bit
    /// range (`round(raw * 255 / channel_max)`), matching `SkMasks`.
    fn sample(value: u32, mask: u32) -> u8 {
        if mask == 0 {
            return 0;
        }
        let shift = mask.trailing_zeros();
        let bits = (mask >> shift).count_ones();
        let raw = (value & mask) >> shift;
        let max = (1u32 << bits) - 1;
        // `raw <= max` by construction, so `(raw * 255 + max/2) / max` is
        // always in `0..=255`; `unwrap_or(255)` is an unreachable fallback.
        u8::try_from((raw * 255 + max / 2) / max).unwrap_or(255)
    }
}

/// Decode a BMP image from bytes.
#[allow(
    clippy::too_many_lines,
    reason = "BMP's several bit-depth/compression paths (24/16/8-bit, BI_BITFIELDS) read \
    more clearly inlined in one function than split across helpers that would each need \
    the same header/bounds context threaded through"
)]
fn decode_bmp(data: &[u8]) -> CodecResult<Image> {
    if data.len() < 54 {
        return Err(CodecError::InvalidData("BMP too short".into()));
    }

    // Validate signature
    if &data[0..2] != b"BM" {
        return Err(CodecError::InvalidData("Invalid BMP signature".into()));
    }

    // Parse BITMAPFILEHEADER
    let pixel_offset = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;

    // Parse BITMAPINFOHEADER
    let header_size = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);
    if header_size < 40 {
        return Err(CodecError::Unsupported(
            "Only BITMAPINFOHEADER and later supported".into(),
        ));
    }

    let width = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let height = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let bits_per_pixel = u16::from_le_bytes([data[28], data[29]]);
    let compression = u32::from_le_bytes([data[30], data[31], data[32], data[33]]);

    // Reject non-positive / absurd dimensions before allocating. `height`
    // may be `i32::MIN`, whose negation overflows, so use `unsigned_abs`.
    if width <= 0 || height == 0 {
        return Err(CodecError::InvalidData("Invalid BMP dimensions".into()));
    }
    let width = usize::try_from(width).unwrap_or(0);
    let (height, bottom_up) = if height < 0 {
        (usize::try_from(height.unsigned_abs()).unwrap_or(0), false)
    } else {
        (usize::try_from(height).unwrap_or(0), true)
    };
    if width > MAX_DIM || height > MAX_DIM || width.checked_mul(height).is_none() {
        return Err(CodecError::InvalidData("BMP dimensions too large".into()));
    }

    // Only support uncompressed (BI_RGB) and BI_BITFIELDS.
    if compression != 0 && compression != 3 {
        return Err(CodecError::Unsupported(format!(
            "BMP compression {compression} not supported"
        )));
    }

    if pixel_offset >= data.len() {
        return Err(CodecError::InvalidData("Invalid pixel data offset".into()));
    }

    let pixel_data = &data[pixel_offset..];

    // Row stride (padded to a 4-byte boundary) and the total number of
    // source bytes the pixel array must contain. A file that stops short of
    // this is truncated and must be reported, not silently zero-filled.
    let src_row_size = match bits_per_pixel {
        32 => width
            .checked_mul(4)
            .ok_or_else(|| CodecError::InvalidData("BMP row too large".into()))?,
        24 => (width * 3 + 3) & !3,
        16 => (width * 2 + 3) & !3,
        8 => (width + 3) & !3,
        _ => {
            return Err(CodecError::Unsupported(format!(
                "BMP {bits_per_pixel} bits per pixel not supported"
            )));
        }
    };
    let required = src_row_size
        .checked_mul(height)
        .ok_or_else(|| CodecError::InvalidData("BMP pixel array too large".into()))?;
    if pixel_data.len() < required {
        return Err(CodecError::InvalidData("BMP pixel data truncated".into()));
    }

    // BI_BITFIELDS: channel masks live either inside a V4+ header or in the
    // 12 bytes immediately after a 40-byte BITMAPINFOHEADER (absolute offset
    // 54). Both land the red mask at absolute offset 54.
    let masks = if compression == 3 {
        let mask_at = |off: usize| -> u32 {
            if off + 4 <= data.len() {
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            } else {
                0
            }
        };
        let red = mask_at(54);
        let green = mask_at(58);
        let blue = mask_at(62);
        // Alpha mask is only present for V3+ headers (>= 56 bytes) / V4.
        let alpha = if header_size >= 56 { mask_at(66) } else { 0 };
        Some(BitfieldMasks {
            red,
            green,
            blue,
            alpha,
        })
    } else {
        None
    };

    let mut rgba = vec![0u8; width * height * 4];

    match bits_per_pixel {
        24 => {
            // 24-bit BGR
            let row_size = src_row_size;
            for y in 0..height {
                let src_y = if bottom_up { height - 1 - y } else { y };
                let src_row = src_y * row_size;
                let dst_row = y * width * 4;

                for x in 0..width {
                    let src = src_row + x * 3;
                    let dst = dst_row + x * 4;
                    if src + 2 < pixel_data.len() {
                        rgba[dst] = pixel_data[src + 2]; // R
                        rgba[dst + 1] = pixel_data[src + 1]; // G
                        rgba[dst + 2] = pixel_data[src]; // B
                        rgba[dst + 3] = 255; // A
                    }
                }
            }
        }
        16 => {
            // 16-bit is always mask-based. Default to X1R5G5B5 when no
            // explicit bitfields are given.
            let m = masks.unwrap_or(BitfieldMasks {
                red: 0x7C00,
                green: 0x03E0,
                blue: 0x001F,
                alpha: 0,
            });
            let row_size = src_row_size;
            for y in 0..height {
                let src_y = if bottom_up { height - 1 - y } else { y };
                let src_row = src_y * row_size;
                let dst_row = y * width * 4;
                for x in 0..width {
                    let src = src_row + x * 2;
                    let dst = dst_row + x * 4;
                    if src + 1 < pixel_data.len() {
                        let value =
                            u32::from(u16::from_le_bytes([pixel_data[src], pixel_data[src + 1]]));
                        rgba[dst] = BitfieldMasks::sample(value, m.red);
                        rgba[dst + 1] = BitfieldMasks::sample(value, m.green);
                        rgba[dst + 2] = BitfieldMasks::sample(value, m.blue);
                        rgba[dst + 3] = if m.alpha == 0 {
                            255
                        } else {
                            BitfieldMasks::sample(value, m.alpha)
                        };
                    }
                }
            }
        }
        32 => {
            // 32-bit. For BI_RGB the fourth byte is undefined padding
            // (kBGRX): SkBmpCodec treats a standalone 32-bit BI_RGB image as
            // opaque and only honors the alpha byte for BMP-in-ICO. For
            // BI_BITFIELDS we apply the declared channel masks.
            let row_size = src_row_size;
            for y in 0..height {
                let src_y = if bottom_up { height - 1 - y } else { y };
                let src_row = src_y * row_size;
                let dst_row = y * width * 4;

                for x in 0..width {
                    let src = src_row + x * 4;
                    let dst = dst_row + x * 4;
                    if src + 3 < pixel_data.len() {
                        if let Some(m) = masks {
                            let value = u32::from_le_bytes([
                                pixel_data[src],
                                pixel_data[src + 1],
                                pixel_data[src + 2],
                                pixel_data[src + 3],
                            ]);
                            rgba[dst] = BitfieldMasks::sample(value, m.red);
                            rgba[dst + 1] = BitfieldMasks::sample(value, m.green);
                            rgba[dst + 2] = BitfieldMasks::sample(value, m.blue);
                            rgba[dst + 3] = if m.alpha == 0 {
                                255
                            } else {
                                BitfieldMasks::sample(value, m.alpha)
                            };
                        } else {
                            rgba[dst] = pixel_data[src + 2]; // R
                            rgba[dst + 1] = pixel_data[src + 1]; // G
                            rgba[dst + 2] = pixel_data[src]; // B
                            // BI_RGB: 4th byte is padding, image is opaque.
                            rgba[dst + 3] = 255; // A
                        }
                    }
                }
            }
        }
        8 => {
            // 8-bit indexed (need to read color table)
            let colors_used = u32::from_le_bytes([data[46], data[47], data[48], data[49]]) as usize;
            let palette_size = if colors_used == 0 { 256 } else { colors_used };
            let palette_offset = 14 + header_size as usize;

            if palette_offset + palette_size * 4 > pixel_offset {
                return Err(CodecError::InvalidData("Invalid palette".into()));
            }

            let palette = &data[palette_offset..palette_offset + palette_size * 4];
            let row_size = (width + 3) & !3;

            for y in 0..height {
                let src_y = if bottom_up { height - 1 - y } else { y };
                let src_row = src_y * row_size;
                let dst_row = y * width * 4;

                for x in 0..width {
                    let src = src_row + x;
                    let dst = dst_row + x * 4;
                    if src < pixel_data.len() {
                        let index = pixel_data[src] as usize;
                        if index < palette_size {
                            let p = index * 4;
                            rgba[dst] = palette[p + 2]; // R
                            rgba[dst + 1] = palette[p + 1]; // G
                            rgba[dst + 2] = palette[p]; // B
                            rgba[dst + 3] = 255; // A
                        }
                    }
                }
            }
        }
        _ => {
            return Err(CodecError::Unsupported(format!(
                "BMP {bits_per_pixel} bits per pixel not supported"
            )));
        }
    }

    let info = crate::ImageInfo::new(
        i32::try_from(width).unwrap_or(i32::MAX),
        i32::try_from(height).unwrap_or(i32::MAX),
        skia_rs_core::ColorType::Rgba8888,
        skia_rs_core::AlphaType::Unpremul,
    );

    Image::from_raster_data_owned(info, rgba, width * 4)
        .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
}

// =============================================================================
// ICO Codec
// =============================================================================

/// ICO decoder.
#[derive(Debug, Default)]
pub struct IcoDecoder;

impl IcoDecoder {
    /// Create a new ICO decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ImageDecoder for IcoDecoder {
    fn decode<R: Read>(&self, mut reader: R) -> CodecResult<Image> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(CodecError::Io)?;
        decode_ico(&data)
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Ico
    }
}

/// Decode an ICO image from bytes.
/// Returns the largest image in the icon file.
fn decode_ico(data: &[u8]) -> CodecResult<Image> {
    if data.len() < 6 {
        return Err(CodecError::InvalidData("ICO too short".into()));
    }

    // Validate header
    let reserved = u16::from_le_bytes([data[0], data[1]]);
    let image_type = u16::from_le_bytes([data[2], data[3]]);
    let image_count = u16::from_le_bytes([data[4], data[5]]) as usize;

    if reserved != 0 || image_type != 1 {
        return Err(CodecError::InvalidData("Invalid ICO header".into()));
    }

    if image_count == 0 {
        return Err(CodecError::InvalidData("ICO has no images".into()));
    }

    // Find the largest image. Only consider directory entries that are
    // fully present in the buffer — a truncated directory must not cause an
    // out-of-bounds read.
    let mut best_index: Option<usize> = None;
    let mut best_size = 0u32;

    for i in 0..image_count {
        let entry_offset = 6 + i * 16;
        if entry_offset + 16 > data.len() {
            break;
        }

        let width = if data[entry_offset] == 0 {
            256
        } else {
            u32::from(data[entry_offset])
        };
        let height = if data[entry_offset + 1] == 0 {
            256
        } else {
            u32::from(data[entry_offset + 1])
        };

        let size = width * height;
        if best_index.is_none() || size > best_size {
            best_size = size;
            best_index = Some(i);
        }
    }

    // Get the best entry. If no complete entry was found the directory is
    // truncated.
    let best_index =
        best_index.ok_or_else(|| CodecError::InvalidData("ICO directory truncated".into()))?;
    let entry_offset = 6 + best_index * 16;
    let image_offset = u32::from_le_bytes([
        data[entry_offset + 12],
        data[entry_offset + 13],
        data[entry_offset + 14],
        data[entry_offset + 15],
    ]) as usize;
    let image_size = u32::from_le_bytes([
        data[entry_offset + 8],
        data[entry_offset + 9],
        data[entry_offset + 10],
        data[entry_offset + 11],
    ]) as usize;

    // Bounds-check the embedded image slice. `image_offset`/`image_size` are
    // u32-derived, so their sum cannot overflow usize on 64-bit, but use a
    // checked add for portability and reject offsets past the buffer end.
    let end = image_offset
        .checked_add(image_size)
        .ok_or_else(|| CodecError::InvalidData("Invalid ICO image data".into()))?;
    if image_size == 0 || image_offset >= data.len() || end > data.len() {
        return Err(CodecError::InvalidData("Invalid ICO image data".into()));
    }

    let image_data = &data[image_offset..end];

    // Check if it's PNG or BMP
    if image_data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        // PNG encoded
        PngDecoder::new().decode_bytes(image_data)
    } else {
        // BMP encoded (without file header)
        decode_ico_bmp(image_data)
    }
}

/// Decode a BMP from ICO format (no BITMAPFILEHEADER).
#[allow(
    clippy::too_many_lines,
    reason = "ICO-embedded BMP's 32/24-bit pixel + AND-mask paths read more clearly inlined \
    than split across helpers that would each need the same header/bounds context threaded through"
)]
fn decode_ico_bmp(data: &[u8]) -> CodecResult<Image> {
    if data.len() < 40 {
        return Err(CodecError::InvalidData("ICO BMP too short".into()));
    }

    let header_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if header_size < 40 || header_size > data.len() {
        return Err(CodecError::Unsupported("Invalid ICO BMP header".into()));
    }

    let width = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let height = i32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    // Reject non-positive / absurd dimensions before allocating. The stored
    // height counts both the color bitmap and the AND mask, so it is halved.
    if width <= 0 || height <= 0 {
        return Err(CodecError::InvalidData("Invalid ICO BMP dimensions".into()));
    }
    let width = usize::try_from(width).unwrap_or(0);
    let height = usize::try_from(height).unwrap_or(0) / 2; // Height includes the mask
    if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM {
        return Err(CodecError::InvalidData("Invalid ICO BMP dimensions".into()));
    }
    let bits_per_pixel = u16::from_le_bytes([data[14], data[15]]);

    let pixel_offset = header_size;
    let pixel_data = &data[pixel_offset..];

    // Before allocating the (width * height * 4) output buffer, verify the
    // file actually contains the color bitmap it claims: a tiny file with a
    // header claiming huge in-range dimensions must be rejected here, not
    // handed to the allocator.
    let src_row_size = match bits_per_pixel {
        32 => width
            .checked_mul(4)
            .ok_or_else(|| CodecError::InvalidData("ICO BMP row too large".into()))?,
        24 => (width * 3 + 3) & !3,
        _ => {
            return Err(CodecError::Unsupported(format!(
                "ICO {bits_per_pixel} bits per pixel not supported"
            )));
        }
    };
    let required = src_row_size
        .checked_mul(height)
        .ok_or_else(|| CodecError::InvalidData("ICO BMP pixel array too large".into()))?;
    if pixel_data.len() < required {
        return Err(CodecError::InvalidData(
            "ICO BMP pixel data truncated".into(),
        ));
    }

    let mut rgba = vec![0u8; width * height * 4];

    match bits_per_pixel {
        32 => {
            // 32-bit BGRA
            let row_size = width * 4;
            for y in 0..height {
                let src_y = height - 1 - y; // Bottom-up
                let src_row = src_y * row_size;
                let dst_row = y * width * 4;

                for x in 0..width {
                    let src = src_row + x * 4;
                    let dst = dst_row + x * 4;
                    if src + 3 < pixel_data.len() {
                        rgba[dst] = pixel_data[src + 2]; // R
                        rgba[dst + 1] = pixel_data[src + 1]; // G
                        rgba[dst + 2] = pixel_data[src]; // B
                        rgba[dst + 3] = pixel_data[src + 3]; // A
                    }
                }
            }
        }
        24 => {
            // 24-bit BGR with separate alpha mask
            let row_size = (width * 3 + 3) & !3;
            let mask_row_size = width.div_ceil(32) * 4;
            let mask_offset = height * row_size;

            for y in 0..height {
                let src_y = height - 1 - y;
                let src_row = src_y * row_size;
                let dst_row = y * width * 4;
                let mask_row = mask_offset + src_y * mask_row_size;

                for x in 0..width {
                    let src = src_row + x * 3;
                    let dst = dst_row + x * 4;
                    if src + 2 < pixel_data.len() {
                        rgba[dst] = pixel_data[src + 2]; // R
                        rgba[dst + 1] = pixel_data[src + 1]; // G
                        rgba[dst + 2] = pixel_data[src]; // B

                        // Check mask
                        let mask_byte = mask_row + x / 8;
                        let mask_bit = 7 - (x % 8);
                        let alpha = if mask_byte < pixel_data.len() {
                            if (pixel_data[mask_byte] >> mask_bit) & 1 == 0 {
                                255
                            } else {
                                0
                            }
                        } else {
                            255
                        };
                        rgba[dst + 3] = alpha;
                    }
                }
            }
        }
        _ => {
            return Err(CodecError::Unsupported(format!(
                "ICO {bits_per_pixel} bits per pixel not supported"
            )));
        }
    }

    let info = crate::ImageInfo::new(
        i32::try_from(width).unwrap_or(i32::MAX),
        i32::try_from(height).unwrap_or(i32::MAX),
        skia_rs_core::ColorType::Rgba8888,
        skia_rs_core::AlphaType::Unpremul,
    );

    Image::from_raster_data_owned(info, rgba, width * 4)
        .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
}

// =============================================================================
// WBMP Codec (Wireless Bitmap)
// =============================================================================

/// WBMP decoder.
///
/// WBMP is a simple monochrome image format used in WAP applications.
pub struct WbmpDecoder;

impl WbmpDecoder {
    /// Create a new WBMP decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for WbmpDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageDecoder for WbmpDecoder {
    fn decode<R: Read>(&self, mut reader: R) -> CodecResult<Image> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(CodecError::Io)?;
        decode_wbmp(&data)
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Wbmp
    }
}

/// WBMP encoder.
pub struct WbmpEncoder;

impl WbmpEncoder {
    /// Create a new WBMP encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for WbmpEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageEncoder for WbmpEncoder {
    fn encode<W: Write>(&self, image: &Image, mut writer: W) -> CodecResult<()> {
        encode_wbmp(image, &mut writer)
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Wbmp
    }
}

/// Read a WBMP multi-byte integer.
/// Each byte has 7 bits of data, MSB indicates if more bytes follow.
fn read_wbmp_int(data: &[u8], offset: &mut usize) -> Option<u32> {
    let mut value = 0u32;
    loop {
        if *offset >= data.len() {
            return None;
        }
        let byte = data[*offset];
        *offset += 1;
        value = (value << 7) | u32::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            break;
        }
        if value > 0xFFFF {
            return None; // Sanity check
        }
    }
    Some(value)
}

/// Write a WBMP multi-byte integer.
fn write_wbmp_int<W: Write>(writer: &mut W, mut value: u32) -> CodecResult<()> {
    let mut bytes = Vec::new();
    bytes.push(u8::try_from(value & 0x7F).unwrap_or(0));
    value >>= 7;
    while value > 0 {
        bytes.push(0x80 | u8::try_from(value & 0x7F).unwrap_or(0));
        value >>= 7;
    }
    bytes.reverse();
    writer.write_all(&bytes).map_err(CodecError::Io)
}

/// Decode a WBMP image from bytes.
fn decode_wbmp(data: &[u8]) -> CodecResult<Image> {
    if data.len() < 4 {
        return Err(CodecError::InvalidData("WBMP too short".into()));
    }

    let mut offset = 0;

    // Read type (should be 0 for WBMP type 0)
    let type_field = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP type".into()))?;
    if type_field != 0 {
        return Err(CodecError::Unsupported(format!(
            "WBMP type {type_field} not supported"
        )));
    }

    // Read fixed header field (should be 0)
    let fix_header = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP header".into()))?;
    if fix_header != 0 {
        return Err(CodecError::Unsupported(
            "Extended WBMP headers not supported".into(),
        ));
    }

    // Read width
    let width = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP width".into()))?;

    // Read height
    let height = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP height".into()))?;

    if width == 0 || height == 0 || width > 10000 || height > 10000 {
        return Err(CodecError::InvalidData("Invalid WBMP dimensions".into()));
    }

    // Calculate row size (bits rounded up to bytes)
    let width_usize = usize::try_from(width).unwrap_or(0);
    let height_usize = usize::try_from(height).unwrap_or(0);
    let row_bytes = width_usize.div_ceil(8);
    let expected_size = offset + row_bytes * height_usize;

    if data.len() < expected_size {
        return Err(CodecError::InvalidData("WBMP data too short".into()));
    }

    // Decode pixels (1 = white, 0 = black)
    let mut rgba = vec![0u8; width_usize * height_usize * 4];

    for y in 0..height_usize {
        let row_start = offset + y * row_bytes;
        for x in 0..width_usize {
            let byte_idx = x / 8;
            let bit_idx = 7 - (x % 8);
            let bit = (data[row_start + byte_idx] >> bit_idx) & 1;

            let pixel_idx = (y * width_usize + x) * 4;
            let value = if bit == 1 { 255 } else { 0 };
            rgba[pixel_idx] = value; // R
            rgba[pixel_idx + 1] = value; // G
            rgba[pixel_idx + 2] = value; // B
            rgba[pixel_idx + 3] = 255; // A
        }
    }

    let info = crate::ImageInfo::new(
        i32::try_from(width).unwrap_or(i32::MAX),
        i32::try_from(height).unwrap_or(i32::MAX),
        skia_rs_core::ColorType::Rgba8888,
        skia_rs_core::AlphaType::Opaque,
    );

    Image::from_raster_data_owned(info, rgba, width_usize * 4)
        .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
}

/// Encode an image as WBMP.
fn encode_wbmp<W: Write>(image: &Image, writer: &mut W) -> CodecResult<()> {
    let width = u32::try_from(image.width()).unwrap_or(0);
    let height = u32::try_from(image.height()).unwrap_or(0);
    let width_usize = usize::try_from(width).unwrap_or(0);
    let height_usize = usize::try_from(height).unwrap_or(0);
    let pixels = image
        .peek_pixels()
        .ok_or_else(|| CodecError::EncodingError("Cannot access pixels".into()))?;

    // Write type (0)
    write_wbmp_int(writer, 0)?;

    // Write fixed header field (0)
    write_wbmp_int(writer, 0)?;

    // Write width
    write_wbmp_int(writer, width)?;

    // Write height
    write_wbmp_int(writer, height)?;

    // Convert to 1-bit (use luminance threshold of 128)
    let row_bytes = width_usize.div_ceil(8);

    for y in 0..height_usize {
        let mut row_data = vec![0u8; row_bytes];

        for x in 0..width_usize {
            let pixel_idx = (y * width_usize + x) * 4;

            // Calculate luminance
            let r = u32::from(pixels[pixel_idx]);
            let g = u32::from(pixels[pixel_idx + 1]);
            let b = u32::from(pixels[pixel_idx + 2]);
            let luminance = (r * 299 + g * 587 + b * 114) / 1000;

            // Set bit if luminance > 128 (white)
            if luminance > 128 {
                let byte_idx = x / 8;
                let bit_idx = 7 - (x % 8);
                row_data[byte_idx] |= 1 << bit_idx;
            }
        }

        writer.write_all(&row_data).map_err(CodecError::Io)?;
    }

    Ok(())
}

/// Get WBMP dimensions without fully decoding.
fn get_wbmp_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    if data.len() < 4 {
        return Err(CodecError::InvalidData("WBMP too short".into()));
    }

    let mut offset = 0;

    // Skip type and header
    let _ = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP type".into()))?;
    let _ = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP header".into()))?;

    // Read dimensions
    let width = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP width".into()))?;
    let height = read_wbmp_int(data, &mut offset)
        .ok_or_else(|| CodecError::InvalidData("Invalid WBMP height".into()))?;

    Ok((
        i32::try_from(width).unwrap_or(i32::MAX),
        i32::try_from(height).unwrap_or(i32::MAX),
    ))
}

// =============================================================================
// AVIF Codec
// =============================================================================

/// AVIF decoder.
///
/// AVIF is a modern image format based on AV1 video codec, offering
/// excellent compression and quality.
pub struct AvifDecoder;

impl AvifDecoder {
    /// Create a new AVIF decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for AvifDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageDecoder for AvifDecoder {
    #[cfg(feature = "avif")]
    fn decode<R: Read>(&self, mut reader: R) -> CodecResult<Image> {
        use avif_decode::{Decoder, Image as AvifImage};

        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| CodecError::Io(e))?;

        let decoder = Decoder::from_avif(&data)
            .map_err(|e| CodecError::DecodingError(format!("AVIF decode error: {:?}", e)))?;

        let decoded = decoder
            .to_image()
            .map_err(|e| CodecError::DecodingError(format!("AVIF decode error: {:?}", e)))?;

        // Extract dimensions and pixels based on the image variant
        let (width, height, pixels) = match decoded {
            AvifImage::Rgb8(img) => {
                let w = img.width();
                let h = img.height();
                let rgba: Vec<u8> = img.pixels().flat_map(|p| [p.r, p.g, p.b, 255]).collect();
                (w as i32, h as i32, rgba)
            }
            AvifImage::Rgba8(img) => {
                let w = img.width();
                let h = img.height();
                let rgba: Vec<u8> = img.pixels().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
                (w as i32, h as i32, rgba)
            }
            AvifImage::Rgb16(img) => {
                let w = img.width();
                let h = img.height();
                // Convert 16-bit to 8-bit
                let rgba: Vec<u8> = img
                    .pixels()
                    .flat_map(|p| [(p.r >> 8) as u8, (p.g >> 8) as u8, (p.b >> 8) as u8, 255])
                    .collect();
                (w as i32, h as i32, rgba)
            }
            AvifImage::Rgba16(img) => {
                let w = img.width();
                let h = img.height();
                // Convert 16-bit to 8-bit
                let rgba: Vec<u8> = img
                    .pixels()
                    .flat_map(|p| {
                        [
                            (p.r >> 8) as u8,
                            (p.g >> 8) as u8,
                            (p.b >> 8) as u8,
                            (p.a >> 8) as u8,
                        ]
                    })
                    .collect();
                (w as i32, h as i32, rgba)
            }
            AvifImage::Gray8(img) => {
                let w = img.width();
                let h = img.height();
                let rgba: Vec<u8> = img
                    .pixels()
                    .flat_map(|g| {
                        let v = g.value();
                        [v, v, v, 255]
                    })
                    .collect();
                (w as i32, h as i32, rgba)
            }
            AvifImage::Gray16(img) => {
                let w = img.width();
                let h = img.height();
                let rgba: Vec<u8> = img
                    .pixels()
                    .flat_map(|g| {
                        let g8 = (g.value() >> 8) as u8;
                        [g8, g8, g8, 255]
                    })
                    .collect();
                (w as i32, h as i32, rgba)
            }
        };

        let info = crate::ImageInfo::new(
            width,
            height,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Unpremul,
        );

        Image::from_raster_data_owned(info, pixels, width as usize * 4)
            .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
    }

    #[cfg(not(feature = "avif"))]
    fn decode<R: Read>(&self, _reader: R) -> CodecResult<Image> {
        Err(CodecError::Unsupported(
            "AVIF decoding requires the 'avif' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Avif
    }
}

/// AVIF encoder.
///
/// Encodes images to AVIF format using the rav1e AV1 encoder.
pub struct AvifEncoder {
    quality: u8,
    speed: u8,
}

impl AvifEncoder {
    /// Create a new AVIF encoder with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            quality: 80,
            speed: 6, // Balance between speed and quality
        }
    }

    /// Set the quality (0-100, higher is better quality but larger file).
    #[must_use]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(100);
        self
    }

    /// Set the encoding speed (1-10, higher is faster but lower quality).
    #[must_use]
    pub fn with_speed(mut self, speed: u8) -> Self {
        self.speed = speed.clamp(1, 10);
        self
    }
}

impl Default for AvifEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageEncoder for AvifEncoder {
    #[cfg(feature = "avif")]
    fn encode<W: Write>(&self, image: &Image, mut writer: W) -> CodecResult<()> {
        use ravif::RGBA8;
        use ravif::{Encoder, Img};

        let width = image.width() as usize;
        let height = image.height() as usize;

        let pixels = image
            .peek_pixels()
            .ok_or_else(|| CodecError::EncodingError("Failed to access pixels".into()))?;

        // Convert to RGBA pixels for ravif
        let rgba_pixels: Vec<RGBA8> = pixels
            .chunks_exact(4)
            .map(|c| RGBA8::new(c[0], c[1], c[2], c[3]))
            .collect();

        let img = Img::new(rgba_pixels.as_slice(), width, height);

        let encoder = Encoder::new()
            .with_quality(self.quality as f32)
            .with_speed(self.speed);

        let result = encoder
            .encode_rgba(img)
            .map_err(|e| CodecError::EncodingError(format!("AVIF encode error: {:?}", e)))?;

        writer
            .write_all(&result.avif_file)
            .map_err(|e| CodecError::Io(e))?;

        Ok(())
    }

    #[cfg(not(feature = "avif"))]
    fn encode<W: Write>(&self, _image: &Image, _writer: W) -> CodecResult<()> {
        Err(CodecError::Unsupported(
            "AVIF encoding requires the 'avif' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Avif
    }
}

/// Get AVIF dimensions from data.
#[cfg(feature = "avif")]
fn get_avif_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    use avif_decode::{Decoder, Image as AvifImage};
    let decoder = Decoder::from_avif(data)
        .map_err(|e| CodecError::DecodingError(format!("AVIF decode error: {:?}", e)))?;
    let image = decoder
        .to_image()
        .map_err(|e| CodecError::DecodingError(format!("AVIF decode error: {:?}", e)))?;

    let (w, h) = match image {
        AvifImage::Rgb8(img) => (img.width(), img.height()),
        AvifImage::Rgba8(img) => (img.width(), img.height()),
        AvifImage::Rgb16(img) => (img.width(), img.height()),
        AvifImage::Rgba16(img) => (img.width(), img.height()),
        AvifImage::Gray8(img) => (img.width(), img.height()),
        AvifImage::Gray16(img) => (img.width(), img.height()),
    };
    Ok((w as i32, h as i32))
}

#[cfg(not(feature = "avif"))]
fn get_avif_dimensions(_data: &[u8]) -> CodecResult<(i32, i32)> {
    Err(CodecError::Unsupported(
        "AVIF support requires the 'avif' feature".into(),
    ))
}

// =============================================================================
// RAW Codec
// =============================================================================

/// Bayer color-filter-array pattern layout for a RAW sensor image.
///
/// The four variants cover the four phase offsets of the 2x2 Bayer
/// mosaic. Pattern names read top-left → top-right → bottom-left →
/// bottom-right. `Rggb` is by far the most common.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerPattern {
    /// Red, Green, Green, Blue (top-left order).
    Rggb,
    /// Blue, Green, Green, Red.
    Bggr,
    /// Green, Red, Blue, Green.
    Grbg,
    /// Green, Blue, Red, Green.
    Gbrg,
}

/// Pick the channel (R=0, G=1, B=2) at position (x, y) given the
/// pattern's 2x2 top-left orientation.
#[inline]
fn bayer_channel_at(pattern: BayerPattern, x: usize, y: usize) -> u8 {
    let (r, g0, g1, b) = match pattern {
        BayerPattern::Rggb => (0u8, 1u8, 1u8, 2u8),
        BayerPattern::Bggr => (2, 1, 1, 0),
        BayerPattern::Grbg => (1, 0, 2, 1),
        BayerPattern::Gbrg => (1, 2, 0, 1),
    };
    match (x % 2, y % 2) {
        (0, 0) => r,
        (1, 0) => g0,
        (0, 1) => g1,
        (1, 1) => b,
        _ => unreachable!(),
    }
}

/// Convert a pixel coordinate to `isize` for demosaic neighbor lookups,
/// saturating (never panicking) on the practically unreachable case of a
/// coordinate beyond `isize::MAX` — a buffer that large could not exist.
#[inline]
fn bayer_to_isize(v: usize) -> isize {
    isize::try_from(v).unwrap_or(isize::MAX)
}

/// Fetch a sample at `(x, y)`, clamping out-of-range coordinates to the
/// nearest edge pixel.
#[inline]
fn bayer_clamp_fetch(raw: &[u16], x: isize, y: isize, w: usize, h: usize) -> u32 {
    let max_x = bayer_to_isize(w) - 1;
    let max_y = bayer_to_isize(h) - 1;
    let cx = usize::try_from(x.clamp(0, max_x)).unwrap_or(0);
    let cy = usize::try_from(y.clamp(0, max_y)).unwrap_or(0);
    u32::from(raw[cy * w + cx])
}

/// Bilinear demosaic a 16-bit Bayer-pattern single-channel sensor image
/// into an 8-bit RGB buffer.
///
/// Each output pixel holds the sensor's directly-sampled channel at its
/// Bayer position, plus bilinearly-interpolated values for the two
/// missing channels:
///
/// - R/B pixels: green is the 4-neighbor cross average (N/S/E/W); the
///   other-color channel is the 4-neighbor diagonal average (NW/NE/SW/SE).
/// - G pixels: the two missing channels come from the two direct
///   neighbors that carry them — one pair horizontal, one pair vertical
///   (which axis carries which channel depends on the row parity).
///
/// This is a baseline demosaic — far less sophisticated than
/// Malvar-He-Cutler, VNG, or AHD — but correct for uniform patches, well
/// behaved at edges of the frame (clamped neighborhood), and produces a
/// predictable linear-light RGB image. Proprietary RAW file parsing
/// (CR2, NEF, ARW) is outside the scope of this helper; feed already-
/// parsed sensor samples in.
///
/// The output is 8-bit per channel: the high byte of each 16-bit
/// intermediate value. Input values are treated as linear (no white
/// balance or color matrix applied). `width * height` must equal
/// `raw.len()`.
///
/// Returns an empty vector if the input length does not match
/// `width * height` — this keeps the helper safe to call without
/// propagating `Result` through every RAW pipeline.
#[must_use]
#[allow(
    clippy::many_single_char_names,
    reason = "r/g/b/c/v name Bayer color-plane samples; matches the RGB/channel-index \
    convention used throughout the demosaic literature and is clearer than verbose aliases"
)]
pub fn demosaic_bayer(raw: &[u16], width: usize, height: usize, pattern: BayerPattern) -> Vec<u8> {
    if width == 0 || height == 0 || raw.len() != width * height {
        return Vec::new();
    }

    let mut out = vec![0u8; width * height * 3];

    for y in 0..height {
        let yi = bayer_to_isize(y);
        for x in 0..width {
            let xi = bayer_to_isize(x);
            let c = bayer_channel_at(pattern, x, y);
            let v = u32::from(raw[y * width + x]);

            // Cross = direct N/S/E/W neighbors; diag = NW/NE/SW/SE.
            let cross = (bayer_clamp_fetch(raw, xi - 1, yi, width, height)
                + bayer_clamp_fetch(raw, xi + 1, yi, width, height)
                + bayer_clamp_fetch(raw, xi, yi - 1, width, height)
                + bayer_clamp_fetch(raw, xi, yi + 1, width, height))
                / 4;
            let diag = (bayer_clamp_fetch(raw, xi - 1, yi - 1, width, height)
                + bayer_clamp_fetch(raw, xi + 1, yi - 1, width, height)
                + bayer_clamp_fetch(raw, xi - 1, yi + 1, width, height)
                + bayer_clamp_fetch(raw, xi + 1, yi + 1, width, height))
                / 4;
            let horiz = u32::midpoint(
                bayer_clamp_fetch(raw, xi - 1, yi, width, height),
                bayer_clamp_fetch(raw, xi + 1, yi, width, height),
            );
            let vert = u32::midpoint(
                bayer_clamp_fetch(raw, xi, yi - 1, width, height),
                bayer_clamp_fetch(raw, xi, yi + 1, width, height),
            );

            // Determine the green-axis layout: which of the two G
            // neighbor directions carries R vs B. The horizontal and
            // vertical neighbors of a green pixel lie one step in each
            // axis, which always lands at a pixel of the opposite
            // Bayer parity — use pattern phase math (x ^ 1, y ^ 1)
            // directly so this is independent of image bounds (the
            // clamp-to-edge happens inside horiz/vert via clamp_fetch).
            let (r, g, b) = if c == 1 {
                let horiz_ch = bayer_channel_at(pattern, x ^ 1, y);
                let vert_ch = bayer_channel_at(pattern, x, y ^ 1);
                let mut r = 0u32;
                let mut b = 0u32;
                if horiz_ch == 0 {
                    r = horiz;
                } else if horiz_ch == 2 {
                    b = horiz;
                }
                if vert_ch == 0 {
                    r = vert;
                } else if vert_ch == 2 {
                    b = vert;
                }
                (r, v, b)
            } else if c == 0 {
                // Red pixel: G = cross, B = diag, R = direct.
                (v, cross, diag)
            } else {
                // Blue pixel: G = cross, R = diag, B = direct.
                (diag, cross, v)
            };

            // High byte of 16-bit intermediate; u32 arithmetic saturates
            // to 65535 in practice because no neighbor can exceed u16.
            let idx = (y * width + x) * 3;
            out[idx] = u8::try_from(r.min(0xFFFF) >> 8).unwrap_or(0xFF);
            out[idx + 1] = u8::try_from(g.min(0xFFFF) >> 8).unwrap_or(0xFF);
            out[idx + 2] = u8::try_from(b.min(0xFFFF) >> 8).unwrap_or(0xFF);
        }
    }

    out
}

/// Convenience: [`demosaic_bayer`] with the most common [`BayerPattern::Rggb`]
/// pattern.
#[inline]
#[must_use]
pub fn demosaic_bayer_rggb(raw: &[u16], width: usize, height: usize) -> Vec<u8> {
    demosaic_bayer(raw, width, height, BayerPattern::Rggb)
}

/// Camera RAW decoder.
///
/// Decodes camera RAW formats from various manufacturers including
/// Canon, Nikon, Sony, Fuji, and many others.
pub struct RawDecoder;

impl RawDecoder {
    /// Create a new RAW decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for RawDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageDecoder for RawDecoder {
    #[cfg(feature = "raw")]
    fn decode<R: Read>(&self, mut reader: R) -> CodecResult<Image> {
        use std::io::Cursor;

        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| CodecError::Io(e))?;

        let mut cursor = Cursor::new(&data);
        let raw_image = rawloader::decode(&mut cursor)
            .map_err(|e| CodecError::DecodingError(format!("RAW decode error: {:?}", e)))?;

        // Get dimensions
        let width = raw_image.width as i32;
        let height = raw_image.height as i32;

        // Convert RAW data to RGB
        // RAW images are typically in a Bayer pattern, we need to demosaic
        let pixels = demosaic_raw(&raw_image)?;

        let info = crate::ImageInfo::new(
            width,
            height,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Opaque,
        );

        Image::from_raster_data_owned(info, pixels, width as usize * 4)
            .ok_or_else(|| CodecError::DecodingError("Failed to create image".into()))
    }

    #[cfg(not(feature = "raw"))]
    fn decode<R: Read>(&self, _reader: R) -> CodecResult<Image> {
        Err(CodecError::Unsupported(
            "RAW decoding requires the 'raw' feature".into(),
        ))
    }

    fn format(&self) -> ImageFormat {
        ImageFormat::Raw
    }
}

/// Try to detect one of the four canonical Bayer patterns from the
/// `rawloader::CFA` 2x2 layout. Returns `None` for non-Bayer CFAs such
/// as X-Trans or monochrome.
#[cfg(feature = "raw")]
fn detect_bayer_pattern(cfa: &rawloader::CFA) -> Option<BayerPattern> {
    // rawloader::CFA::color_at(row, col) returns 0=R, 1=G, 2=B.
    let tl = cfa.color_at(0, 0);
    let tr = cfa.color_at(0, 1);
    let bl = cfa.color_at(1, 0);
    let br = cfa.color_at(1, 1);
    match (tl, tr, bl, br) {
        (0, 1, 1, 2) => Some(BayerPattern::Rggb),
        (2, 1, 1, 0) => Some(BayerPattern::Bggr),
        (1, 0, 2, 1) => Some(BayerPattern::Grbg),
        (1, 2, 0, 1) => Some(BayerPattern::Gbrg),
        _ => None,
    }
}

/// Demosaic a RAW image to RGBA.
///
/// For Bayer-pattern sensors this routes through the generic
/// [`demosaic_bayer`] helper after black-level subtraction and white-
/// level normalization to 16-bit range. For non-Bayer CFAs (X-Trans,
/// etc.) it falls back to the legacy per-channel neighbor-average path
/// so unusual sensors still decode rather than erroring.
#[cfg(feature = "raw")]
fn demosaic_raw(raw: &rawloader::RawImage) -> CodecResult<Vec<u8>> {
    let width = raw.width;
    let height = raw.height;

    // Get the raw data
    let data = match &raw.data {
        rawloader::RawImageData::Integer(d) => d,
        rawloader::RawImageData::Float(_) => {
            return Err(CodecError::Unsupported(
                "Float RAW data not yet supported".into(),
            ));
        }
    };

    if let Some(pattern) = detect_bayer_pattern(&raw.cfa) {
        // Fast path: bilinear demosaic via the public helper. Produces
        // an RGB buffer we expand to RGBA with opaque alpha. Normalize
        // each sample into the 0..65535 range first so the demosaic
        // helper sees a consistent scale regardless of the sensor's
        // native bit depth.
        let black = raw.blacklevels[0] as f32;
        let white = raw.whitelevels[0] as f32;
        let range = if white > black {
            white - black
        } else {
            65535.0
        };
        let normalized: Vec<u16> = data
            .iter()
            .map(|&v| (((v as f32 - black) / range * 65535.0).clamp(0.0, 65535.0)) as u16)
            .collect();
        let rgb = demosaic_bayer(&normalized, width, height, pattern);
        if rgb.len() != width * height * 3 {
            return Err(CodecError::DecodingError(
                "Bayer demosaic produced unexpected buffer size".into(),
            ));
        }
        let mut output = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            output[i * 4] = rgb[i * 3];
            output[i * 4 + 1] = rgb[i * 3 + 1];
            output[i * 4 + 2] = rgb[i * 3 + 2];
            output[i * 4 + 3] = 255;
        }
        return Ok(output);
    }

    // Fallback: unknown CFA layout. Sample direct channel + 3x3
    // neighbor average for the two missing channels — this is the
    // legacy path that predates the bilinear helper.
    let mut output = vec![0u8; width * height * 4];

    // Simple demosaicing - convert to grayscale for simplicity
    // A full implementation would do proper Bayer demosaicing
    let cfa = &raw.cfa;
    let black = raw.blacklevels[0] as f32;
    let white = raw.whitelevels[0] as f32;
    let range = if white > black {
        white - black
    } else {
        65535.0
    };

    // Color indices: 0=Red, 1=Green, 2=Blue
    const RED: usize = 0;
    const GREEN: usize = 1;
    const BLUE: usize = 2;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let out_idx = idx * 4;

            // Get color index at this position (returns 0, 1, or 2)
            let color = cfa.color_at(y, x);

            // Get raw value normalized to 0-255
            let raw_val = data[idx] as f32;
            let normalized = ((raw_val - black) / range * 255.0).clamp(0.0, 255.0) as u8;

            // Simple nearest-neighbor for demo purposes
            match color {
                RED => {
                    output[out_idx] = normalized;
                    output[out_idx + 1] = get_neighbor_avg_by_color(
                        data, x, y, width, height, cfa, GREEN, black, range,
                    );
                    output[out_idx + 2] = get_neighbor_avg_by_color(
                        data, x, y, width, height, cfa, BLUE, black, range,
                    );
                }
                GREEN => {
                    output[out_idx] = get_neighbor_avg_by_color(
                        data, x, y, width, height, cfa, RED, black, range,
                    );
                    output[out_idx + 1] = normalized;
                    output[out_idx + 2] = get_neighbor_avg_by_color(
                        data, x, y, width, height, cfa, BLUE, black, range,
                    );
                }
                BLUE => {
                    output[out_idx] = get_neighbor_avg_by_color(
                        data, x, y, width, height, cfa, RED, black, range,
                    );
                    output[out_idx + 1] = get_neighbor_avg_by_color(
                        data, x, y, width, height, cfa, GREEN, black, range,
                    );
                    output[out_idx + 2] = normalized;
                }
                _ => {
                    // Unknown CFA pattern, just use grayscale
                    output[out_idx] = normalized;
                    output[out_idx + 1] = normalized;
                    output[out_idx + 2] = normalized;
                }
            }
            output[out_idx + 3] = 255; // Alpha
        }
    }

    Ok(output)
}

/// Get average of neighboring pixels of a specific color index.
#[cfg(feature = "raw")]
fn get_neighbor_avg_by_color(
    data: &[u16],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    cfa: &rawloader::CFA,
    target_color: usize,
    black: f32,
    range: f32,
) -> u8 {
    let mut sum = 0.0;
    let mut count = 0;

    // Check 3x3 neighborhood
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nx = nx as usize;
                let ny = ny as usize;

                if cfa.color_at(ny, nx) == target_color {
                    let idx = ny * width + nx;
                    let raw_val = data[idx] as f32;
                    sum += (raw_val - black) / range * 255.0;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        (sum / count as f32).clamp(0.0, 255.0) as u8
    } else {
        128 // Fallback if no neighbors found
    }
}

/// Get RAW dimensions from data.
#[cfg(feature = "raw")]
fn get_raw_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(data);
    let raw = rawloader::decode(&mut cursor)
        .map_err(|e| CodecError::DecodingError(format!("RAW decode error: {:?}", e)))?;
    Ok((raw.width as i32, raw.height as i32))
}

#[cfg(not(feature = "raw"))]
fn get_raw_dimensions(_data: &[u8]) -> CodecResult<(i32, i32)> {
    Err(CodecError::Unsupported(
        "RAW support requires the 'raw' feature".into(),
    ))
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Decode an image from bytes, auto-detecting the format.
///
/// # Errors
/// Returns [`CodecError::Unsupported`] if the format cannot be detected
/// from `data`, or the detected format's decoder error otherwise.
pub fn decode_image(data: &[u8]) -> CodecResult<Image> {
    let format = ImageFormat::from_magic(data);

    match format {
        ImageFormat::Png => PngDecoder::new().decode_bytes(data),
        ImageFormat::Jpeg => JpegDecoder::new().decode_bytes(data),
        ImageFormat::Gif => GifDecoder::new().decode_bytes(data),
        ImageFormat::WebP => WebpDecoder::new().decode_bytes(data),
        ImageFormat::Bmp => BmpDecoder::new().decode_bytes(data),
        ImageFormat::Ico => IcoDecoder::new().decode_bytes(data),
        ImageFormat::Wbmp => WbmpDecoder::new().decode_bytes(data),
        ImageFormat::Avif => AvifDecoder::new().decode_bytes(data),
        ImageFormat::Raw => RawDecoder::new().decode_bytes(data),
        ImageFormat::Unknown => Err(CodecError::Unsupported(format!(
            "Format {format:?} not supported"
        ))),
    }
}

/// Get the image dimensions without fully decoding.
///
/// # Errors
/// Returns [`CodecError::Unsupported`] if the format cannot be detected
/// or has no dimension probe, or the detected format's probe error
/// otherwise.
pub fn get_image_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    let format = ImageFormat::from_magic(data);

    match format {
        ImageFormat::Png => get_png_dimensions(data),
        ImageFormat::Jpeg => get_jpeg_dimensions(data),
        ImageFormat::Bmp => get_bmp_dimensions(data),
        ImageFormat::Ico => get_ico_dimensions(data),
        ImageFormat::Wbmp => get_wbmp_dimensions(data),
        ImageFormat::Avif => get_avif_dimensions(data),
        ImageFormat::Raw => get_raw_dimensions(data),
        _ => Err(CodecError::Unsupported(format!(
            "Format {format:?} not supported"
        ))),
    }
}

fn get_png_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    // PNG IHDR chunk starts at byte 8, width at offset 16, height at offset 20
    if data.len() < 24 {
        return Err(CodecError::InvalidData("PNG too short".into()));
    }

    let width = i32::try_from(u32::from_be_bytes([data[16], data[17], data[18], data[19]]))
        .map_err(|_| CodecError::InvalidData("PNG width exceeds i32 range".into()))?;
    let height = i32::try_from(u32::from_be_bytes([data[20], data[21], data[22], data[23]]))
        .map_err(|_| CodecError::InvalidData("PNG height exceeds i32 range".into()))?;

    Ok((width, height))
}

/// Returns true if `marker` is a Start-Of-Frame marker that carries frame
/// dimensions. SOF markers span C0-CF *except* C4 (DHT), C8 (JPG), and CC
/// (DAC), which are not frame headers.
const fn is_jpeg_sof_marker(marker: u8) -> bool {
    matches!(marker,
        0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
}

fn get_jpeg_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    // Walk the marker stream looking for any SOF header. Each marker is
    // `0xFF` followed by a marker byte; multiple `0xFF` fill bytes may
    // precede the marker byte.
    let mut i = 2;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        // Consume any run of 0xFF fill bytes.
        let mut marker = data[i + 1];
        let mut j = i + 1;
        while marker == 0xFF && j + 1 < data.len() {
            j += 1;
            marker = data[j];
        }

        // Standalone markers have no length payload: TEM (0x01) and
        // SOI/EOI/RST0-7 (0xD0-0xD9). Also skip padding 0x00.
        if marker == 0x00 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            i = j + 1;
            continue;
        }

        // All remaining markers carry a 2-byte big-endian segment length
        // (which includes the length field itself).
        if j + 3 >= data.len() {
            break;
        }
        let len = u16::from_be_bytes([data[j + 1], data[j + 2]]) as usize;

        if is_jpeg_sof_marker(marker) {
            // Layout from the marker byte at `j`: length(2) at j+1..j+3,
            // then the SOF payload precision(1) height(2) width(2).
            if j + 7 < data.len() {
                let height = i32::from(u16::from_be_bytes([data[j + 4], data[j + 5]]));
                let width = i32::from(u16::from_be_bytes([data[j + 6], data[j + 7]]));
                return Ok((width, height));
            }
            break;
        }

        if len < 2 {
            break;
        }
        i = j + 1 + len;
    }

    Err(CodecError::InvalidData(
        "Could not find JPEG dimensions".into(),
    ))
}

fn get_bmp_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    if data.len() < 26 {
        return Err(CodecError::InvalidData("BMP too short".into()));
    }

    let width = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    // Height is signed (negative = top-down). Use `unsigned_abs` so
    // `i32::MIN` does not panic (`.abs()` would overflow); its magnitude
    // (2^31) doesn't fit back in `i32`, so saturate to `i32::MAX` rather
    // than wrapping back to a negative value.
    let height_abs = i32::from_le_bytes([data[22], data[23], data[24], data[25]]).unsigned_abs();
    let height = i32::try_from(height_abs).unwrap_or(i32::MAX);

    Ok((width, height))
}

fn get_ico_dimensions(data: &[u8]) -> CodecResult<(i32, i32)> {
    if data.len() < 22 {
        return Err(CodecError::InvalidData("ICO too short".into()));
    }

    let image_count = u16::from_le_bytes([data[4], data[5]]) as usize;
    if image_count == 0 {
        return Err(CodecError::InvalidData("ICO has no images".into()));
    }

    // Return the largest image dimensions
    let mut best_width = 0i32;
    let mut best_height = 0i32;

    for i in 0..image_count {
        let entry_offset = 6 + i * 16;
        if entry_offset + 8 > data.len() {
            break;
        }

        let width = if data[entry_offset] == 0 {
            256
        } else {
            i32::from(data[entry_offset])
        };
        let height = if data[entry_offset + 1] == 0 {
            256
        } else {
            i32::from(data[entry_offset + 1])
        };

        if width * height > best_width * best_height {
            best_width = width;
            best_height = height;
        }
    }

    Ok((best_width, best_height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        // PNG magic bytes
        let png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(ImageFormat::from_magic(&png), ImageFormat::Png);

        // JPEG magic bytes (need at least 8 bytes for magic detection)
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(ImageFormat::from_magic(&jpeg), ImageFormat::Jpeg);

        // GIF magic bytes (need at least 8 bytes)
        let gif = b"GIF89a\x00\x00";
        assert_eq!(ImageFormat::from_magic(gif), ImageFormat::Gif);

        // WebP magic bytes
        let webp = b"RIFF\x00\x00\x00\x00WEBP";
        assert_eq!(ImageFormat::from_magic(webp), ImageFormat::WebP);

        // BMP magic bytes
        let bmp = b"BM\x00\x00\x00\x00\x00\x00";
        assert_eq!(ImageFormat::from_magic(bmp), ImageFormat::Bmp);

        // ICO magic bytes
        let ico = [0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x10];
        assert_eq!(ImageFormat::from_magic(&ico), ImageFormat::Ico);

        // Unknown (need at least 8 bytes to test)
        let unknown = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(ImageFormat::from_magic(&unknown), ImageFormat::Unknown);
    }

    #[test]
    fn test_format_extensions() {
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Gif.extension(), "gif");
        assert_eq!(ImageFormat::Bmp.extension(), "bmp");
        assert_eq!(ImageFormat::Ico.extension(), "ico");
    }

    #[test]
    fn test_format_mime_types() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Bmp.mime_type(), "image/bmp");
        assert_eq!(ImageFormat::Ico.mime_type(), "image/x-icon");
    }

    #[test]
    fn test_encoder_quality() {
        let q = EncoderQuality::new(75);
        assert_eq!(q.value(), 75);

        let q_over = EncoderQuality::new(150);
        assert_eq!(q_over.value(), 100);
    }

    #[test]
    fn test_bmp_encode_decode_roundtrip() {
        // Create a simple 2x2 image
        let info = crate::ImageInfo::new(
            2,
            2,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Unpremul,
        );
        let pixels = vec![
            255, 0, 0, 255, // Red
            0, 255, 0, 255, // Green
            0, 0, 255, 255, // Blue
            255, 255, 0, 255, // Yellow
        ];
        let image = Image::from_raster_data_owned(info, pixels, 8).unwrap();

        // Encode to BMP
        let encoder = BmpEncoder::new();
        let bmp_bytes = encoder.encode_bytes(&image).unwrap();

        // Verify format detection
        assert_eq!(ImageFormat::from_magic(&bmp_bytes), ImageFormat::Bmp);

        // Decode back
        let decoder = BmpDecoder::new();
        let round_tripped = decoder.decode_bytes(&bmp_bytes).unwrap();

        // Verify dimensions
        assert_eq!(round_tripped.width(), 2);
        assert_eq!(round_tripped.height(), 2);
    }

    #[test]
    fn test_bmp_dimensions() {
        // Create a simple BMP header for a 100x50 image
        let mut bmp = vec![0u8; 54];
        bmp[0] = b'B';
        bmp[1] = b'M';
        // Width at offset 18-21
        bmp[18..22].copy_from_slice(&100i32.to_le_bytes());
        // Height at offset 22-25
        bmp[22..26].copy_from_slice(&50i32.to_le_bytes());

        let dims = get_bmp_dimensions(&bmp).unwrap();
        assert_eq!(dims, (100, 50));
    }

    /// Build a small checkerboard Image for round-trip tests.
    fn test_checkerboard(w: i32, h: i32) -> Image {
        let info = crate::ImageInfo::new(
            w,
            h,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Unpremul,
        );
        let w_usize = usize::try_from(w).unwrap();
        let h_usize = usize::try_from(h).unwrap();
        let row_bytes = w_usize * 4;
        let mut pixels = vec![0u8; h_usize * row_bytes];
        for y in 0..h_usize {
            for x in 0..w_usize {
                let off = y * row_bytes + x * 4;
                let on = ((x + y) & 1) == 0;
                pixels[off] = if on { 255 } else { 0 };
                pixels[off + 1] = if on { 0 } else { 255 };
                pixels[off + 2] = 0;
                pixels[off + 3] = 255;
            }
        }
        Image::from_raster_data_owned(info, pixels, row_bytes).unwrap()
    }

    #[cfg(feature = "png")]
    #[test]
    fn test_png_encode_decode_roundtrip() {
        let image = test_checkerboard(4, 4);
        let encoded = PngEncoder::new().encode_bytes(&image).unwrap();
        assert_eq!(ImageFormat::from_magic(&encoded), ImageFormat::Png);

        let decoded = PngDecoder::new().decode_bytes(&encoded).unwrap();
        assert_eq!(decoded.dimensions(), (4, 4));

        // PNG is lossless; pixel values should survive exactly. Alpha
        // interpretation may differ, so compare RGB channels per pixel.
        let src_pixels = image.peek_pixels().unwrap();
        let dst_pixels = decoded.peek_pixels().unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let src_off = (y * 4 + x) * 4;
                let dst_off = (y * 4 + x) * 4;
                assert_eq!(
                    src_pixels[src_off..src_off + 3],
                    dst_pixels[dst_off..dst_off + 3],
                    "pixel mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn test_jpeg_encode_decode_roundtrip() {
        // JPEG is lossy, but round-trip should preserve dimensions and
        // yield a reasonable approximation. Use a larger, smoother source
        // so quantization noise stays bounded.
        let image = test_checkerboard(16, 16);
        let encoded = JpegEncoder::new().encode_bytes(&image).unwrap();
        assert_eq!(ImageFormat::from_magic(&encoded), ImageFormat::Jpeg);

        let decoded = JpegDecoder::new().decode_bytes(&encoded).unwrap();
        assert_eq!(decoded.dimensions(), (16, 16));
    }

    #[cfg(feature = "webp")]
    #[test]
    fn test_webp_encode_decode_roundtrip() {
        let image = test_checkerboard(8, 8);
        let encoded = WebpEncoder::lossless().encode_bytes(&image).unwrap();
        assert_eq!(ImageFormat::from_magic(&encoded), ImageFormat::WebP);

        let decoded = WebpDecoder::new().decode_bytes(&encoded).unwrap();
        assert_eq!(decoded.dimensions(), (8, 8));
    }

    #[cfg(feature = "webp")]
    #[test]
    fn test_webp_lossy_roundtrip_dimensions() {
        let image = test_checkerboard(16, 16);
        let encoded = WebpEncoder::new().encode_bytes(&image).unwrap();
        assert_eq!(ImageFormat::from_magic(&encoded), ImageFormat::WebP);

        let decoded = WebpDecoder::new().decode_bytes(&encoded).unwrap();
        assert_eq!(decoded.dimensions(), (16, 16));
    }

    #[test]
    fn test_wbmp_encode_decode_roundtrip() {
        // WBMP only supports 1-bit monochrome; feed a black/white pattern.
        let image = test_checkerboard(4, 4);
        let mut buf = Vec::new();
        WbmpEncoder::new().encode(&image, &mut buf).unwrap();
        let decoded = WbmpDecoder::new().decode_bytes(&buf).unwrap();
        assert_eq!(decoded.dimensions(), (4, 4));
    }

    #[test]
    fn test_demosaic_bayer_uniform_saturated() {
        // All-white Bayer input → every output pixel should be close to
        // white in all three channels after bilinear interpolation.
        let raw = vec![0xFFFFu16; 4 * 4];
        let rgb = demosaic_bayer_rggb(&raw, 4, 4);
        assert_eq!(rgb.len(), 4 * 4 * 3);
        for (i, chunk) in rgb.chunks_exact(3).enumerate() {
            assert!(chunk[0] > 200, "r too dim at pixel {}: {}", i, chunk[0]);
            assert!(chunk[1] > 200, "g too dim at pixel {}: {}", i, chunk[1]);
            assert!(chunk[2] > 200, "b too dim at pixel {}: {}", i, chunk[2]);
        }
    }

    #[test]
    fn test_demosaic_bayer_zero_input() {
        // All-black Bayer input should produce an all-black RGB buffer.
        let raw = vec![0u16; 4 * 4];
        let rgb = demosaic_bayer_rggb(&raw, 4, 4);
        assert!(rgb.iter().all(|&v| v == 0), "zero input should be black");
    }

    #[test]
    fn test_demosaic_bayer_preserves_direct_channel() {
        // RGGB layout on a 6x6 canvas: (even, even)=R, (odd, even)=G,
        // (even, odd)=G, (odd, odd)=B. Put 0xFF00 at every R position,
        // 0 elsewhere. Pick an interior R pixel (2, 2) so clamp-to-edge
        // artifacts at the border do not pollute the assertion.
        let w = 6;
        let h = 6;
        let mut raw = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                if x % 2 == 0 && y % 2 == 0 {
                    raw[y * w + x] = 0xFF00;
                }
            }
        }
        let rgb = demosaic_bayer_rggb(&raw, w, h);

        // Interior R pixel at (2, 2).
        let off = (2 * w + 2) * 3;
        // Direct R sample survives: 0xFF00 >> 8 = 0xFF.
        assert_eq!(rgb[off], 0xFF);
        // At an R pixel, G comes from the four N/S/E/W neighbors — all
        // G positions, which are zero.
        assert_eq!(rgb[off + 1], 0);
        // B comes from the four diagonal neighbors — all B positions,
        // also zero.
        assert_eq!(rgb[off + 2], 0);

        // An interior G pixel at (3, 2): its direct sample is 0 (G
        // positions hold 0). Horizontal neighbors (2, 2) and (4, 2) are
        // R-positions with 0xFF00, vertical neighbors (3, 1) and (3, 3)
        // are B positions with 0. So R should be ~0xFF and B should be 0.
        let goff = (2 * w + 3) * 3;
        assert!(rgb[goff] > 0xF0, "G pixel R should be close to 0xFF");
        assert_eq!(rgb[goff + 1], 0);
        assert_eq!(rgb[goff + 2], 0);
    }

    #[test]
    fn test_demosaic_bayer_mismatched_length_returns_empty() {
        // Mismatched size is a pre-condition violation; return empty
        // rather than panicking.
        let raw = vec![0u16; 10];
        let rgb = demosaic_bayer_rggb(&raw, 4, 4);
        assert!(rgb.is_empty());
    }

    #[cfg(feature = "gif")]
    #[test]
    fn test_gif_decode_animated_returns_multiple_frames() {
        // Build a minimal 2-frame GIF in memory using the gif encoder.
        let mut buf: Vec<u8> = Vec::new();
        {
            use gif::{Encoder, Frame, Repeat};
            let palette = [0, 0, 0, 255, 255, 255]; // black, white
            let mut encoder = Encoder::new(&mut buf, 2, 2, &palette).unwrap();
            encoder.set_repeat(Repeat::Infinite).unwrap();

            // Frame 0: all-black, 50ms.
            let f0 = Frame {
                width: 2,
                height: 2,
                buffer: std::borrow::Cow::Borrowed(&[0u8; 4]),
                delay: 5, // 5 * 10ms = 50ms
                ..Frame::default()
            };
            encoder.write_frame(&f0).unwrap();

            // Frame 1: all-white, 100ms.
            let f1 = Frame {
                width: 2,
                height: 2,
                buffer: std::borrow::Cow::Borrowed(&[1u8; 4]),
                delay: 10,
                ..Frame::default()
            };
            encoder.write_frame(&f1).unwrap();
        }

        let animated = GifDecoder::new().decode_animated(&buf).unwrap();
        assert_eq!(animated.frame_count(), 2);
        assert_eq!(animated.loop_count, crate::LoopCount::Infinite);
        assert_eq!(animated.frames[0].delay.as_millis(), 50);
        assert_eq!(animated.frames[1].delay.as_millis(), 100);
        assert_eq!(animated.total_duration().as_millis(), 150);
    }

    #[cfg(feature = "png")]
    #[test]
    fn test_png_without_iccp_chunk_has_no_color_space() {
        // PngEncoder does not emit an iCCP chunk, so a round-tripped
        // image must have `color_space == None`.
        let image = test_checkerboard(4, 4);
        let encoded = PngEncoder::new().encode_bytes(&image).unwrap();
        let decoded = PngDecoder::new().decode_bytes(&encoded).unwrap();
        assert!(decoded.color_space().is_none());
    }

    #[cfg(feature = "png")]
    #[test]
    fn test_png_with_iccp_chunk_populates_color_space() {
        // Build a valid ICC profile (sRGB, 128-byte header with "acsp"
        // signature) and inject it via the png crate's encoder.
        let mut icc = vec![0u8; 128];
        // Signature at bytes 36..40.
        icc[36..40].copy_from_slice(b"acsp");
        // Profile class 'mntr' (display) at bytes 12..16.
        icc[12..16].copy_from_slice(b"mntr");
        // Color space 'RGB ' at bytes 16..20.
        icc[16..20].copy_from_slice(b"RGB ");
        // PCS 'XYZ ' at bytes 20..24.
        icc[20..24].copy_from_slice(b"XYZ ");

        // Encode a 2x2 PNG with the synthetic iCCP profile. The png
        // crate's encoder does not expose a setter for the iCCP chunk,
        // but the underlying `Info` struct carries a public
        // `icc_profile` field that the encoder writes through. Build an
        // `Info` with the profile attached and hand it to `with_info`.
        let mut out: Vec<u8> = Vec::new();
        {
            let mut info = png::Info::with_size(2, 2);
            info.color_type = png::ColorType::Rgba;
            info.bit_depth = png::BitDepth::Eight;
            info.icc_profile = Some(std::borrow::Cow::Owned(icc));
            let encoder = png::Encoder::with_info(&mut out, info).unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[0u8; 16]).unwrap();
        }

        let decoded = PngDecoder::new().decode_bytes(&out).unwrap();
        // IccProfile::from_bytes falls back to sRGB when the tag table
        // is unparseable (our synthetic profile has no tags), so the
        // ColorSpace itself is present and equals sRGB. The salient
        // check is that `color_space.is_some()` — i.e. the codec did
        // read the chunk and route it through the parser.
        assert!(
            decoded.color_space().is_some(),
            "iCCP chunk should populate ImageInfo.color_space"
        );
    }

    // ---- PNG palette / bit-depth decode (EXPAND | STRIP_16) --------------

    /// Encode a paletted PNG-8 fixture in memory: a 2x2 image whose four
    /// pixels index four distinct palette colors. With the default
    /// `IDENTITY` transform the decoder would hand back raw indices
    /// (single-channel); EXPAND must turn these into RGB.
    #[cfg(feature = "png")]
    #[test]
    fn test_png_palette8_decodes_to_rgb() {
        let palette = vec![
            255, 0, 0, // 0 red
            0, 255, 0, // 1 green
            0, 0, 255, // 2 blue
            255, 255, 0, // 3 yellow
        ];
        let mut out: Vec<u8> = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 2, 2);
            encoder.set_color(png::ColorType::Indexed);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_palette(palette);
            let mut w = encoder.write_header().unwrap();
            // Indices: (0,0)=0 (1,0)=1 / (0,1)=2 (1,1)=3
            w.write_image_data(&[0, 1, 2, 3]).unwrap();
        }

        let decoded = PngDecoder::new().decode_bytes(&out).unwrap();
        assert_eq!(decoded.dimensions(), (2, 2));
        let px = decoded.peek_pixels().unwrap();
        assert_eq!(&px[0..4], &[255, 0, 0, 255], "index 0 -> red");
        assert_eq!(&px[4..8], &[0, 255, 0, 255], "index 1 -> green");
        assert_eq!(&px[8..12], &[0, 0, 255, 255], "index 2 -> blue");
        assert_eq!(&px[12..16], &[255, 255, 0, 255], "index 3 -> yellow");
    }

    /// A 16-bit-per-channel RGB PNG must decode via `STRIP_16` to the high
    /// byte of each sample. Without the transform, the `png` crate yields a
    /// 6-byte-per-pixel buffer our RGBA8 path cannot interpret.
    #[cfg(feature = "png")]
    #[test]
    fn test_png_rgb16_strips_to_high_byte() {
        let mut out: Vec<u8> = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Sixteen);
            let mut w = encoder.write_header().unwrap();
            // One pixel, big-endian 16-bit R=0x1234 G=0x5678 B=0x9ABC.
            w.write_image_data(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC])
                .unwrap();
        }

        let decoded = PngDecoder::new().decode_bytes(&out).unwrap();
        let px = decoded.peek_pixels().unwrap();
        // STRIP_16 keeps the most-significant byte of each channel.
        assert_eq!(&px[0..4], &[0x12, 0x56, 0x9A, 255]);
    }

    /// A 4-bit grayscale PNG must expand to 8-bit gray. The `png` crate
    /// scales sub-8-bit samples by `255 / (2^depth - 1)` (= 17 for 4-bit),
    /// so index 0 -> 0, 15 -> 255, 8 -> 136.
    #[cfg(feature = "png")]
    #[test]
    fn test_png_gray4_expands_to_8bit() {
        let mut out: Vec<u8> = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 2, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Four);
            let mut w = encoder.write_header().unwrap();
            // Two 4-bit samples packed into one byte: 0x0F -> (0, 15).
            w.write_image_data(&[0x0F]).unwrap();
        }

        let decoded = PngDecoder::new().decode_bytes(&out).unwrap();
        assert_eq!(decoded.dimensions(), (2, 1));
        let px = decoded.peek_pixels().unwrap();
        assert_eq!(&px[0..4], &[0, 0, 0, 255], "gray 0 -> black");
        assert_eq!(&px[4..8], &[255, 255, 255, 255], "gray 15 -> white");
    }

    // ---- Encoder unpremultiplication ------------------------------------

    /// Encoders must store straight (unpremultiplied) alpha. A premultiplied
    /// source (50% red stored as r==a==128) must round-trip through PNG back
    /// to unpremultiplied red (255, 0, 0, 128).
    #[cfg(feature = "png")]
    #[test]
    fn test_png_encode_unpremultiplies() {
        // Premultiplied 50% red: r = premul(255, a=128) = 128.
        let info = crate::ImageInfo::new(
            1,
            1,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Premul,
        );
        let image = Image::from_raster_data_owned(info, vec![128, 0, 0, 128], 4).unwrap();

        let encoded = PngEncoder::new().encode_bytes(&image).unwrap();
        let decoded = PngDecoder::new().decode_bytes(&encoded).unwrap();
        let px = decoded.peek_pixels().unwrap();
        // Unpremultiplying 128 with alpha 128 restores ~255.
        assert!(
            px[0] >= 253,
            "R should unpremultiply back to ~255, got {}",
            px[0]
        );
        assert_eq!(px[3], 128, "alpha preserved");
    }

    /// Differential check for the shared premul rounding landed in core:
    /// premul(200, a=130) must round to 102 (`SkMulDiv255Round`), not truncate
    /// to 101.
    #[test]
    fn test_premul_rounding_matches_skia() {
        let mut px = vec![200u8, 200, 200, 130];
        skia_rs_core::premultiply_in_place(&mut px);
        assert_eq!(px[0], 102, "premul(200, 130) rounds to 102");
    }

    // ---- BMP 32-bit alpha / bitfields -----------------------------------

    /// Build a standalone 32-bit `BI_RGB` BMP by hand (BITMAPINFOHEADER). The
    /// stored 4th byte is 0 (padding), but `SkBmpCodec` treats standalone
    /// 32-bit `BI_RGB` as opaque (kBGRX), so decoded alpha must be 255.
    #[test]
    fn test_bmp_32bit_bi_rgb_is_opaque() {
        let mut bmp = build_bmp_header(1, 1, 32, 0);
        // One pixel: B=10 G=20 R=30 X=0.
        bmp.extend_from_slice(&[10, 20, 30, 0]);

        let decoded = decode_bmp(&bmp).unwrap();
        let px = decoded.peek_pixels().unwrap();
        assert_eq!(
            &px[0..4],
            &[30, 20, 10, 255],
            "BI_RGB 4th byte ignored -> opaque"
        );
    }

    /// A 32-bit `BI_BITFIELDS` BMP with explicit RGBA masks must honor them.
    #[test]
    fn test_bmp_32bit_bitfields_applies_masks() {
        // Masks: R=0x00FF0000 G=0x0000FF00 B=0x000000FF A=0xFF000000.
        // Header size 56 (V3) so an alpha mask is recognized. The 16 mask
        // bytes fill the header out to the pixel offset (14 + 56 = 70).
        let mut bmp = build_bmp_header_ex(1, 1, 32, 3, 56);
        bmp.extend_from_slice(&0x00FF_0000_u32.to_le_bytes()); // red mask @54
        bmp.extend_from_slice(&0x0000_FF00_u32.to_le_bytes()); // green mask @58
        bmp.extend_from_slice(&0x0000_00FF_u32.to_le_bytes()); // blue mask @62
        bmp.extend_from_slice(&0xFF00_0000_u32.to_le_bytes()); // alpha mask @66
        // Pixel value 0xAABBCCDD (little-endian bytes DD CC BB AA):
        //   A=0xAA R=0xBB G=0xCC B=0xDD
        bmp.extend_from_slice(&[0xDD, 0xCC, 0xBB, 0xAA]);
        let decoded = decode_bmp(&bmp).unwrap();
        let px = decoded.peek_pixels().unwrap();
        assert_eq!(px[0], 0xBB, "red from mask");
        assert_eq!(px[1], 0xCC, "green from mask");
        assert_eq!(px[2], 0xDD, "blue from mask");
        assert_eq!(px[3], 0xAA, "alpha from mask");
    }

    /// A BMP whose pixel array is shorter than width*height*stride is
    /// truncated and must be reported, not silently zero-filled.
    #[test]
    fn test_bmp_truncated_reports_error() {
        let mut bmp = build_bmp_header(4, 4, 24, 0);
        // Only supply a few bytes instead of 4 rows * padded stride.
        bmp.extend_from_slice(&[0u8; 8]);
        assert!(decode_bmp(&bmp).is_err(), "truncated BMP must error");
    }

    /// Absurd / non-positive dimensions must be rejected before allocation.
    #[test]
    fn test_bmp_absurd_dimensions_rejected() {
        // width = i32::MAX, height = i32::MAX would overflow the allocation.
        let mut bmp = build_bmp_header(i32::MAX, i32::MAX, 32, 0);
        bmp.extend_from_slice(&[0u8; 16]);
        assert!(decode_bmp(&bmp).is_err(), "huge BMP dims must error");

        let zero = build_bmp_header(0, 1, 32, 0);
        assert!(decode_bmp(&zero).is_err(), "zero width must error");
    }

    /// `get_bmp_dimensions` must not panic on a top-down BMP whose height is
    /// `i32::MIN` (`.abs()` would overflow), and must not silently wrap the
    /// magnitude back to a negative value either (`2^31` doesn't fit in
    /// `i32`, so it saturates to `i32::MAX`).
    #[test]
    fn test_bmp_dimensions_i32_min_height() {
        let mut bmp = vec![0u8; 54];
        bmp[0] = b'B';
        bmp[1] = b'M';
        bmp[18..22].copy_from_slice(&100i32.to_le_bytes());
        bmp[22..26].copy_from_slice(&i32::MIN.to_le_bytes());
        let (w, h) = get_bmp_dimensions(&bmp).unwrap();
        assert_eq!(w, 100);
        assert_eq!(h, i32::MAX);
    }

    /// Helper: build a minimal BITMAPFILEHEADER + BITMAPINFOHEADER for a
    /// standalone BMP with the given dims/bpp/compression. Pixel data is
    /// appended by the caller. Uses a 40-byte core header.
    fn build_bmp_header(width: i32, height: i32, bpp: u16, compression: u32) -> Vec<u8> {
        build_bmp_header_ex(width, height, bpp, compression, 40)
    }

    fn build_bmp_header_ex(
        width: i32,
        height: i32,
        bpp: u16,
        compression: u32,
        header_size: u32,
    ) -> Vec<u8> {
        let mut bmp = Vec::new();
        let pixel_offset = 14 + header_size;
        // BITMAPFILEHEADER
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&0u32.to_le_bytes()); // file size (unused)
        bmp.extend_from_slice(&0u32.to_le_bytes()); // reserved
        bmp.extend_from_slice(&pixel_offset.to_le_bytes()); // pixel data offset
        // BITMAPINFOHEADER (always write 40 core bytes)
        bmp.extend_from_slice(&header_size.to_le_bytes());
        bmp.extend_from_slice(&width.to_le_bytes());
        bmp.extend_from_slice(&height.to_le_bytes());
        bmp.extend_from_slice(&1u16.to_le_bytes()); // planes
        bmp.extend_from_slice(&bpp.to_le_bytes());
        bmp.extend_from_slice(&compression.to_le_bytes());
        bmp.extend_from_slice(&0u32.to_le_bytes()); // image size
        bmp.extend_from_slice(&0u32.to_le_bytes()); // x ppm
        bmp.extend_from_slice(&0u32.to_le_bytes()); // y ppm
        bmp.extend_from_slice(&0u32.to_le_bytes()); // colors used
        bmp.extend_from_slice(&0u32.to_le_bytes()); // important colors
        // Pad the header out to `header_size` (for V3/V4 the extra bytes hold
        // the channel masks, which the caller appends).
        bmp
    }

    // ---- Malformed ICO robustness ---------------------------------------

    /// A truncated ICO directory (claims one image, but the entry is cut
    /// off) must error, not panic with an out-of-bounds read.
    #[test]
    fn test_ico_truncated_directory_errors() {
        // reserved=0, type=1, count=1, then only 4 of the 16 entry bytes.
        let data = vec![0, 0, 1, 0, 1, 0, 16, 16, 0, 0];
        assert!(decode_ico(&data).is_err());
    }

    /// An ICO-embedded BMP whose header claims huge (per-dimension in-range)
    /// dimensions but supplies no pixel data must error *before* allocating
    /// the output buffer — width*height*4 for 16M x 8M would be ~0.5 PB.
    #[test]
    fn test_ico_bmp_absurd_dims_rejected_before_alloc() {
        // BITMAPINFOHEADER (40 bytes), no pixel data.
        let mut bmp = vec![0u8; 40];
        bmp[0..4].copy_from_slice(&40u32.to_le_bytes()); // header size
        bmp[4..8].copy_from_slice(&((1i32 << 24) - 2).to_le_bytes()); // width
        // Stored height counts color+mask, so this decodes to height/2.
        bmp[8..12].copy_from_slice(&((1i32 << 24) - 2).to_le_bytes()); // height
        bmp[14..16].copy_from_slice(&32u16.to_le_bytes()); // bpp
        assert!(
            decode_ico_bmp(&bmp).is_err(),
            "absurd ICO BMP dims with no data must error, not allocate"
        );
    }

    /// An ICO entry whose image offset/size point past the buffer must
    /// error.
    #[test]
    fn test_ico_bad_offset_errors() {
        let mut data = vec![0u8; 6 + 16];
        data[2] = 1; // type
        data[4] = 1; // count
        data[6] = 16; // width
        data[7] = 16; // height
        // image_size at +8, image_offset at +12 point way past the buffer.
        data[6 + 8..6 + 12].copy_from_slice(&1000u32.to_le_bytes());
        data[6 + 12..6 + 16].copy_from_slice(&1000u32.to_le_bytes());
        assert!(decode_ico(&data).is_err());
    }

    // ---- WBMP multibyte sniffing ----------------------------------------

    /// WBMP sniffing must accept multibyte width/height integers (dimensions
    /// larger than 127), not just single-byte ones.
    #[test]
    fn test_wbmp_sniff_multibyte_dimensions() {
        // type=0, fix=0, width=200 (multibyte: 0x81 0x48), height=200.
        let data = [0u8, 0, 0x81, 0x48, 0x81, 0x48, 0, 0];
        assert_eq!(ImageFormat::from_magic(&data), ImageFormat::Wbmp);
    }

    // ---- JPEG SOF / RST marker scan -------------------------------------

    /// `get_jpeg_dimensions` must skip standalone RST markers (no length
    /// payload) and read dimensions from a progressive SOF2 header.
    #[test]
    fn test_jpeg_dimensions_sof2_after_rst() {
        // SOI, RST0 (standalone), then SOF2 with height=0x0064 width=0x00C8.
        let data = [
            0xFF, 0xD8, // SOI
            0xFF, 0xD0, // RST0 (standalone, no length)
            0xFF, 0xC2, // SOF2
            0x00, 0x11, // length = 17
            0x08, // precision
            0x00, 0x64, // height = 100
            0x00, 0xC8, // width = 200
            0x03, // component count (padding for the scan)
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let (w, h) = get_jpeg_dimensions(&data).unwrap();
        assert_eq!((w, h), (200, 100));
    }

    // ---- GIF canvas compositing -----------------------------------------

    /// A GIF whose first frame is smaller than the logical screen and is
    /// placed at an offset must decode to the *canvas* size, compositing the
    /// frame at its left/top over a transparent background.
    #[cfg(feature = "gif")]
    #[test]
    fn test_gif_decode_composites_first_frame_on_canvas() {
        let mut buf: Vec<u8> = Vec::new();
        {
            use gif::{Encoder, Frame, Repeat};
            // 4x4 logical screen; global palette black/white.
            let palette = [0, 0, 0, 255, 255, 255];
            let mut encoder = Encoder::new(&mut buf, 4, 4, &palette).unwrap();
            encoder.set_repeat(Repeat::Infinite).unwrap();
            // A 2x2 opaque-white frame placed at (1,1).
            let f = Frame {
                width: 2,
                height: 2,
                left: 1,
                top: 1,
                buffer: std::borrow::Cow::Borrowed(&[1u8; 4]),
                ..Frame::default()
            };
            encoder.write_frame(&f).unwrap();
        }

        let decoded = GifDecoder::new().decode_bytes(&buf).unwrap();
        assert_eq!(decoded.dimensions(), (4, 4), "decodes to logical screen");
        let px = decoded.peek_pixels().unwrap();
        // Pixel (0,0) is outside the frame -> transparent.
        assert_eq!(&px[0..4], &[0, 0, 0, 0], "corner is transparent");
        // Pixel (1,1) is the frame's top-left -> white opaque.
        let off = (4 + 1) * 4;
        assert_eq!(
            &px[off..off + 4],
            &[255, 255, 255, 255],
            "frame composited at offset"
        );
    }

    /// `decode_animated` must report the logical-screen canvas dimensions.
    #[cfg(feature = "gif")]
    #[test]
    fn test_gif_animated_carries_canvas_dimensions() {
        let mut buf: Vec<u8> = Vec::new();
        {
            use gif::{Encoder, Frame};
            let palette = [0, 0, 0, 255, 255, 255];
            let mut encoder = Encoder::new(&mut buf, 6, 5, &palette).unwrap();
            let f = Frame {
                width: 2,
                height: 2,
                buffer: std::borrow::Cow::Borrowed(&[0u8; 4]),
                ..Frame::default()
            };
            encoder.write_frame(&f).unwrap();
        }
        let anim = GifDecoder::new().decode_animated(&buf).unwrap();
        assert_eq!(anim.canvas_dimensions(), (6, 5));
    }

    /// A GIF declaring an absurd logical-screen size (65535x65535, the max
    /// representable in the format's u16 header fields) must be rejected
    /// with an error rather than attempting a ~17GB canvas allocation,
    /// which would abort the process instead of returning `Err`.
    #[cfg(feature = "gif")]
    #[test]
    fn test_gif_decode_rejects_oversized_canvas() {
        let mut buf: Vec<u8> = Vec::new();
        {
            use gif::{Encoder, Frame};
            let palette = [0, 0, 0, 255, 255, 255];
            // Logical screen at the format's maximum (u16::MAX x u16::MAX)
            // would require ~17GB for an RGBA canvas; a tiny 1x1 frame keeps
            // the encoded file itself small.
            let mut encoder = Encoder::new(&mut buf, u16::MAX, u16::MAX, &palette).unwrap();
            let f = Frame {
                width: 1,
                height: 1,
                buffer: std::borrow::Cow::Borrowed(&[0u8]),
                ..Frame::default()
            };
            encoder.write_frame(&f).unwrap();
        }

        let result = GifDecoder::new().decode_bytes(&buf);
        assert!(
            result.is_err(),
            "oversized GIF canvas must be rejected, not allocated"
        );
    }
}
