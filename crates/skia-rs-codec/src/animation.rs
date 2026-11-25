//! Multi-frame animated image support.
//!
//! Animated images (GIF, APNG, animated WebP) are modeled as an ordered
//! sequence of [`AnimationFrame`]s. Each frame carries its own [`Image`],
//! per-frame delay, and disposal/blend metadata. Consumers walk the frame
//! list themselves and composite according to the disposal and blend
//! methods — the codec layer does not maintain a playback state machine.
//!
//! Only GIF is wired up here today (via `GifDecoder::decode_animated`).
//! APNG and animated WebP share the same shape and will plug into the
//! same types once their decoders surface multi-frame data.

use crate::Image;
use std::time::Duration;

/// A decoded animated image.
#[derive(Debug, Clone)]
pub struct AnimatedImage {
    /// Ordered list of frames. At least one element for a valid image.
    pub frames: Vec<AnimationFrame>,
    /// How many times the animation loops.
    pub loop_count: LoopCount,
    /// Width of the animation canvas (the GIF logical-screen width). Frames
    /// are positioned within this canvas via their `x_offset`/`y_offset`
    /// and may be smaller than the canvas.
    pub canvas_width: i32,
    /// Height of the animation canvas (the GIF logical-screen height).
    pub canvas_height: i32,
}

impl AnimatedImage {
    /// Returns the animation canvas size as `(width, height)`.
    #[inline]
    #[must_use]
    pub const fn canvas_dimensions(&self) -> (i32, i32) {
        (self.canvas_width, self.canvas_height)
    }

    /// Returns the number of frames.
    #[inline]
    #[must_use]
    pub const fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the animation has no frames.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Sum of all per-frame delays. Useful for UI "total duration"
    /// displays. Note this is the sum for a single play-through — it does
    /// not account for `loop_count`.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.frames.iter().map(|f| f.delay).sum()
    }

    /// Borrow a frame by index.
    #[inline]
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&AnimationFrame> {
        self.frames.get(index)
    }
}

/// A single frame in an animated image.
#[derive(Debug, Clone)]
pub struct AnimationFrame {
    /// The decoded pixels for this frame, at its offset position within
    /// the canvas.
    pub image: Image,
    /// How long to display this frame before advancing.
    pub delay: Duration,
    /// What to do with the canvas when this frame's display time ends.
    pub dispose: DisposalMethod,
    /// How this frame's pixels are combined with the existing canvas.
    pub blend: BlendMethod,
    /// X offset of this frame's top-left corner inside the animation
    /// canvas. Zero for formats that don't subdivide (default).
    pub x_offset: i32,
    /// Y offset of this frame's top-left corner inside the animation
    /// canvas.
    pub y_offset: i32,
}

/// Loop/repeat behavior for an animated image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopCount {
    /// Animate once and stop.
    Once,
    /// Loop forever.
    Infinite,
    /// Loop a finite number of times (the integer is the repeat count).
    Finite(u32),
}

/// How to clear the canvas after a frame is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisposalMethod {
    /// Leave the canvas as-is; the next frame composites over the
    /// current pixels.
    Keep,
    /// Clear the frame's region to **transparent** before compositing the
    /// next frame.
    ///
    /// Despite the name, this always clears to transparent black, never to
    /// the GIF's logical-screen background color. This matches Skia's
    /// `SkCodecAnimation::DisposalMethod::kRestoreBGColor`, whose GIF
    /// implementation (`SkGifCodec`) fills the disposed region with
    /// `SK_ColorTRANSPARENT`.
    Background,
    /// Restore the canvas area underneath this frame to what it was
    /// before this frame was drawn.
    Previous,
}

/// How a frame's pixels combine with the existing canvas content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMethod {
    /// Replace canvas pixels with frame pixels (no alpha blending).
    Source,
    /// Alpha-blend frame pixels over the canvas (standard "over"
    /// compositing).
    Over,
}

#[cfg(feature = "gif")]
impl From<gif::DisposalMethod> for DisposalMethod {
    fn from(d: gif::DisposalMethod) -> Self {
        match d {
            // Spec: `Any` = decoder's choice. Historically treated as
            // "do not dispose" (same as Keep), which is what most
            // renderers do.
            gif::DisposalMethod::Any | gif::DisposalMethod::Keep => Self::Keep,
            gif::DisposalMethod::Background => Self::Background,
            gif::DisposalMethod::Previous => Self::Previous,
        }
    }
}

#[cfg(feature = "gif")]
impl From<gif::Repeat> for LoopCount {
    fn from(r: gif::Repeat) -> Self {
        match r {
            gif::Repeat::Infinite => Self::Infinite,
            // GIF89a's NETSCAPE2.0 loop extension uses 0 to mean
            // "infinite" at the wire level, but the `gif` crate already
            // surfaces that as `Repeat::Infinite`. A `Finite(0)` here
            // therefore means "play once" — which is what `Once`
            // expresses in this enum.
            gif::Repeat::Finite(0) => Self::Once,
            gif::Repeat::Finite(n) => Self::Finite(u32::from(n)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageInfo, image::Image};
    use skia_rs_core::{AlphaType, ColorType};

    fn mk_frame(delay_ms: u64) -> AnimationFrame {
        let info = ImageInfo::new(1, 1, ColorType::Rgba8888, AlphaType::Premul);
        let image = Image::from_raster_data(&info, &[0, 0, 0, 255], 4).unwrap();
        AnimationFrame {
            image,
            delay: Duration::from_millis(delay_ms),
            dispose: DisposalMethod::Keep,
            blend: BlendMethod::Over,
            x_offset: 0,
            y_offset: 0,
        }
    }

    #[test]
    fn test_total_duration_sums_frame_delays() {
        let anim = AnimatedImage {
            frames: vec![mk_frame(50), mk_frame(100), mk_frame(25)],
            loop_count: LoopCount::Infinite,
            canvas_width: 1,
            canvas_height: 1,
        };
        assert_eq!(anim.frame_count(), 3);
        assert_eq!(anim.total_duration(), Duration::from_millis(175));
    }

    #[test]
    fn test_frame_accessor() {
        let anim = AnimatedImage {
            frames: vec![mk_frame(10)],
            loop_count: LoopCount::Once,
            canvas_width: 1,
            canvas_height: 1,
        };
        assert!(anim.frame(0).is_some());
        assert!(anim.frame(1).is_none());
        assert!(!anim.is_empty());
    }
}
