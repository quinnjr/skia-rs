//! PDF image embedding support.
//!
//! This module provides image embedding for PDF documents, including:
//! - JPEG images (`DCTDecode`, pass-through)
//! - PNG/other images (`FlateDecode` compression)
//! - Image masks and soft masks
//! - Color space handling

use std::io::Write;

/// Image color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfColorSpace {
    /// Grayscale.
    DeviceGray,
    /// RGB.
    DeviceRGB,
    /// CMYK.
    DeviceCMYK,
    /// Indexed (palette).
    Indexed,
}

impl PdfColorSpace {
    /// Get PDF name.
    #[must_use]
    pub const fn pdf_name(&self) -> &'static str {
        match self {
            Self::DeviceGray => "DeviceGray",
            Self::DeviceRGB => "DeviceRGB",
            Self::DeviceCMYK => "DeviceCMYK",
            Self::Indexed => "Indexed",
        }
    }

    /// Get number of components.
    #[must_use]
    pub const fn components(&self) -> u8 {
        match self {
            Self::DeviceGray | Self::Indexed => 1,
            Self::DeviceRGB => 3,
            Self::DeviceCMYK => 4,
        }
    }
}

/// Image filter (compression).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfImageFilter {
    /// No compression.
    None,
    /// Flate (zlib) compression.
    FlateDecode,
    /// DCT (JPEG) compression.
    DCTDecode,
    /// Run-length encoding.
    RunLengthDecode,
    /// ASCII85 encoding.
    ASCII85Decode,
}

impl PdfImageFilter {
    /// Get PDF filter name.
    #[must_use]
    pub const fn pdf_name(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::FlateDecode => Some("FlateDecode"),
            Self::DCTDecode => Some("DCTDecode"),
            Self::RunLengthDecode => Some("RunLengthDecode"),
            Self::ASCII85Decode => Some("ASCII85Decode"),
        }
    }
}

/// A PDF image `XObject`.
#[derive(Debug, Clone)]
pub struct PdfImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bits per component.
    pub bits_per_component: u8,
    /// Color space.
    pub color_space: PdfColorSpace,
    /// Compression filter.
    pub filter: PdfImageFilter,
    /// Image data (raw or compressed).
    pub data: Vec<u8>,
    /// Soft mask (alpha channel) image ID.
    pub soft_mask_id: Option<u32>,
    /// Object ID (assigned when writing).
    pub object_id: Option<u32>,
    /// Whether this is a mask.
    pub is_mask: bool,
    /// Interpolation.
    pub interpolate: bool,
}

impl PdfImage {
    /// Create a new image from raw RGB data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 3`.
    #[must_use]
    pub fn from_rgb(width: u32, height: u32, data: &[u8]) -> Self {
        assert_eq!(data.len(), (width * height * 3) as usize);

        // Compress with flate
        let compressed = compress_flate(data);

        Self {
            width,
            height,
            bits_per_component: 8,
            color_space: PdfColorSpace::DeviceRGB,
            filter: PdfImageFilter::FlateDecode,
            data: compressed,
            soft_mask_id: None,
            object_id: None,
            is_mask: false,
            interpolate: true,
        }
    }

    /// Create a new image from raw RGBA data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 4`.
    #[must_use]
    pub fn from_rgba(width: u32, height: u32, data: &[u8]) -> (Self, Self) {
        assert_eq!(data.len(), (width * height * 4) as usize);

        // Separate RGB and alpha
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        let mut alpha = Vec::with_capacity((width * height) as usize);

        for chunk in data.chunks(4) {
            rgb.extend_from_slice(&chunk[..3]);
            alpha.push(chunk[3]);
        }

        // Compress both
        let rgb_compressed = compress_flate(&rgb);
        let alpha_compressed = compress_flate(&alpha);

        let image = Self {
            width,
            height,
            bits_per_component: 8,
            color_space: PdfColorSpace::DeviceRGB,
            filter: PdfImageFilter::FlateDecode,
            data: rgb_compressed,
            soft_mask_id: None, // Will be set later
            object_id: None,
            is_mask: false,
            interpolate: true,
        };

        let mask = Self {
            width,
            height,
            bits_per_component: 8,
            color_space: PdfColorSpace::DeviceGray,
            filter: PdfImageFilter::FlateDecode,
            data: alpha_compressed,
            soft_mask_id: None,
            object_id: None,
            is_mask: true,
            interpolate: true,
        };

        (image, mask)
    }

    /// Create a new image from grayscale data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height`.
    #[must_use]
    pub fn from_grayscale(width: u32, height: u32, data: &[u8]) -> Self {
        assert_eq!(data.len(), (width * height) as usize);

        let compressed = compress_flate(data);

        Self {
            width,
            height,
            bits_per_component: 8,
            color_space: PdfColorSpace::DeviceGray,
            filter: PdfImageFilter::FlateDecode,
            data: compressed,
            soft_mask_id: None,
            object_id: None,
            is_mask: false,
            interpolate: true,
        }
    }

    /// Create an image from JPEG data (pass-through, no re-encoding).
    #[must_use]
    pub fn from_jpeg(width: u32, height: u32, jpeg_data: Vec<u8>) -> Self {
        // Detect color space from JPEG header
        let color_space = detect_jpeg_color_space(&jpeg_data);

        Self {
            width,
            height,
            bits_per_component: 8,
            color_space,
            filter: PdfImageFilter::DCTDecode,
            data: jpeg_data,
            soft_mask_id: None,
            object_id: None,
            is_mask: false,
            interpolate: true,
        }
    }

    /// Set the soft mask.
    pub const fn set_soft_mask(&mut self, mask_id: u32) {
        self.soft_mask_id = Some(mask_id);
    }

    /// Generate the image `XObject` PDF dictionary.
    #[must_use]
    pub fn to_pdf_xobject(&self, id: u32) -> Vec<u8> {
        let mut output = Vec::new();

        // Object header
        write!(output, "{id} 0 obj\n<<\n").unwrap();
        writeln!(output, "/Type /XObject").unwrap();
        writeln!(output, "/Subtype /Image").unwrap();
        writeln!(output, "/Width {}", self.width).unwrap();
        writeln!(output, "/Height {}", self.height).unwrap();
        writeln!(output, "/BitsPerComponent {}", self.bits_per_component).unwrap();

        if self.is_mask {
            writeln!(output, "/ColorSpace /DeviceGray").unwrap();
        } else {
            writeln!(output, "/ColorSpace /{}", self.color_space.pdf_name()).unwrap();
        }

        if let Some(filter_name) = self.filter.pdf_name() {
            writeln!(output, "/Filter /{filter_name}").unwrap();
        }

        if let Some(mask_id) = self.soft_mask_id {
            writeln!(output, "/SMask {mask_id} 0 R").unwrap();
        }

        if self.interpolate {
            writeln!(output, "/Interpolate true").unwrap();
        }

        writeln!(output, "/Length {}", self.data.len()).unwrap();
        write!(output, ">>\nstream\n").unwrap();
        output.extend_from_slice(&self.data);
        write!(output, "\nendstream\nendobj\n").unwrap();

        output
    }
}

/// Compress data using flate (zlib).
fn compress_flate(data: &[u8]) -> Vec<u8> {
    use std::io::Write;

    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

/// Detect color space from JPEG header.
fn detect_jpeg_color_space(data: &[u8]) -> PdfColorSpace {
    // Simple JPEG header parsing
    // Look for SOF marker to determine components
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];

            // SOF markers (0xC0-0xCF except 0xC4, 0xC8, 0xCC)
            if (0xC0..=0xCF).contains(&marker)
                && marker != 0xC4
                && marker != 0xC8
                && marker != 0xCC
                && i + 9 < data.len()
            {
                let num_components = data[i + 9];
                return match num_components {
                    1 => PdfColorSpace::DeviceGray,
                    4 => PdfColorSpace::DeviceCMYK,
                    _ => PdfColorSpace::DeviceRGB,
                };
            }

            // Skip marker
            if i + 3 < data.len() && marker != 0xD8 && marker != 0xD9 {
                let length = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
                i += 2 + length;
            } else {
                i += 2;
            }
        } else {
            i += 1;
        }
    }

    PdfColorSpace::DeviceRGB
}

/// Image manager for PDF documents.
#[derive(Debug, Default)]
pub struct PdfImageManager {
    /// Registered images.
    images: Vec<PdfImage>,
}

impl PdfImageManager {
    /// Create a new image manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an RGB image.
    pub fn add_rgb(&mut self, width: u32, height: u32, data: &[u8]) -> usize {
        let idx = self.images.len();
        self.images.push(PdfImage::from_rgb(width, height, data));
        idx
    }

    /// Add an RGBA image (returns image index and mask index).
    ///
    /// The returned image has its [`PdfImage::soft_mask_id`] populated with
    /// the mask's manager *index*. [`crate::PdfDocument::write_to`] resolves
    /// this index to the mask's final object id when serialising, so
    /// callers don't need to patch anything up post-hoc.
    pub fn add_rgba(&mut self, width: u32, height: u32, data: &[u8]) -> (usize, usize) {
        let (mut image, mask) = PdfImage::from_rgba(width, height, data);
        let mask_idx = self.images.len();
        self.images.push(mask);
        let image_idx = self.images.len();
        // Store the mask *index* here; the writer translates to object id.
        image.soft_mask_id = Some(u32::try_from(mask_idx).unwrap_or(u32::MAX));
        self.images.push(image);
        (image_idx, mask_idx)
    }

    /// Add a JPEG image.
    pub fn add_jpeg(&mut self, width: u32, height: u32, data: Vec<u8>) -> usize {
        let idx = self.images.len();
        self.images.push(PdfImage::from_jpeg(width, height, data));
        idx
    }

    /// Add a grayscale image.
    pub fn add_grayscale(&mut self, width: u32, height: u32, data: &[u8]) -> usize {
        let idx = self.images.len();
        self.images
            .push(PdfImage::from_grayscale(width, height, data));
        idx
    }

    /// Get image by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&PdfImage> {
        self.images.get(index)
    }

    /// Get mutable image by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut PdfImage> {
        self.images.get_mut(index)
    }

    /// Get all images.
    #[must_use]
    pub fn images(&self) -> &[PdfImage] {
        &self.images
    }

    /// Get mutable images.
    pub fn images_mut(&mut self) -> &mut [PdfImage] {
        &mut self.images
    }

    /// Get number of images.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.images.len()
    }

    /// Check if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_from_rgb() {
        let data = vec![255u8; 100 * 100 * 3]; // 100x100 white image
        let image = PdfImage::from_rgb(100, 100, &data);

        assert_eq!(image.width, 100);
        assert_eq!(image.height, 100);
        assert_eq!(image.color_space, PdfColorSpace::DeviceRGB);
        assert_eq!(image.filter, PdfImageFilter::FlateDecode);
    }

    #[test]
    fn test_image_from_rgba() {
        let data = vec![255u8; 50 * 50 * 4]; // 50x50 white opaque image
        let (image, mask) = PdfImage::from_rgba(50, 50, &data);

        assert_eq!(image.color_space, PdfColorSpace::DeviceRGB);
        assert_eq!(mask.color_space, PdfColorSpace::DeviceGray);
        assert!(mask.is_mask);
    }

    #[test]
    fn test_image_manager() {
        let mut manager = PdfImageManager::new();

        let data = vec![128u8; 10 * 10 * 3];
        let idx = manager.add_rgb(10, 10, &data);

        assert_eq!(idx, 0);
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_color_space_components() {
        assert_eq!(PdfColorSpace::DeviceGray.components(), 1);
        assert_eq!(PdfColorSpace::DeviceRGB.components(), 3);
        assert_eq!(PdfColorSpace::DeviceCMYK.components(), 4);
    }

    #[test]
    fn test_add_rgba_links_soft_mask_index() {
        let mut m = PdfImageManager::new();
        let data = vec![0u8; 4 * 4 * 4];
        let (image_idx, mask_idx) = m.add_rgba(4, 4, &data);

        // The image stores the mask's *index* in soft_mask_id so that the
        // writer can translate it to the emitted object id.
        let image = m.get(image_idx).unwrap();
        assert_eq!(image.soft_mask_id, Some(u32::try_from(mask_idx).unwrap()));
        // Mask is a mask with gray colorspace.
        let mask = m.get(mask_idx).unwrap();
        assert!(mask.is_mask);
        assert_eq!(mask.color_space, PdfColorSpace::DeviceGray);
    }
}
