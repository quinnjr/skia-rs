//! Direct Metal backend implementation for macOS/iOS.
//!
//! This module provides a direct Metal API backend for Apple platforms,
//! offering low-level access beyond what wgpu provides.
//!
//! # Executor
//!
//! The Metal executor ([`MetalContext::new_wgpu_executor`], enabled when the
//! `wgpu-backend` feature is also on) delegates to wgpu configured with
//! [`wgpu::Backends::METAL`]. Metal's `MTLCommandBuffer`, `MTLRenderCommandEncoder`,
//! and argument buffer machinery are driven inside wgpu; our `CommandBuffer`
//! and `RenderPipelineDescriptor` flow through wgpu's naga pipeline to MSL
//! and then to native Metal.
//!
//! A native metal-rs executor that hand-records `MTLCommandBuffer` sequences
//! is not provided. Duplicating wgpu's Metal translator would require
//! re-implementing resource tracking, argument encoding, and sync
//! primitives — a separate crate-sized effort. For callers who need Metal
//! bypass, [`MetalContext::device`] still exposes the raw `metal::Device`
//! and [`MetalContext::command_queue`] the `metal::CommandQueue`.

use crate::{
    GpuAdapterInfo, GpuBackendType, GpuCaps, GpuContext, GpuDeviceType, GpuError, GpuResult,
    TextureFormat,
};

#[cfg(feature = "metal")]
use metal::{Device, DeviceRef, MTLPixelFormat};

/// Metal feature set levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalFeatureSet {
    /// iOS GPU Family 1 v1 (A7).
    IosGpuFamily1V1,
    /// iOS GPU Family 1 v2 (A7-A8).
    IosGpuFamily1V2,
    /// iOS GPU Family 2 v1 (A8).
    IosGpuFamily2V1,
    /// iOS GPU Family 2 v2 (A8-A9).
    IosGpuFamily2V2,
    /// iOS GPU Family 3 v1 (A9-A10).
    IosGpuFamily3V1,
    /// iOS GPU Family 3 v2 (A9-A11).
    IosGpuFamily3V2,
    /// iOS GPU Family 4 v1 (A11).
    IosGpuFamily4V1,
    /// iOS GPU Family 5 v1 (A12).
    IosGpuFamily5V1,
    /// macOS GPU Family 1 v1.
    MacosGpuFamily1V1,
    /// macOS GPU Family 1 v2.
    MacosGpuFamily1V2,
    /// macOS GPU Family 1 v3.
    MacosGpuFamily1V3,
    /// macOS GPU Family 1 v4.
    MacosGpuFamily1V4,
    /// macOS GPU Family 2 v1.
    MacosGpuFamily2V1,
    /// Common family (Apple Silicon).
    CommonFamily1,
    /// Common family 2.
    CommonFamily2,
    /// Common family 3.
    CommonFamily3,
    /// Apple family 1 (A7-A10).
    AppleFamily1,
    /// Apple family 2 (A8-A10).
    AppleFamily2,
    /// Apple family 3 (A9-A10).
    AppleFamily3,
    /// Apple family 4 (A11).
    AppleFamily4,
    /// Apple family 5 (A12).
    AppleFamily5,
    /// Apple family 6 (A13).
    AppleFamily6,
    /// Apple family 7 (M1, A14).
    AppleFamily7,
    /// Apple family 8 (M2, A15).
    AppleFamily8,
    /// Apple family 9 (M3, A17).
    AppleFamily9,
}

impl std::fmt::Display for MetalFeatureSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IosGpuFamily1V1 => write!(f, "iOS GPU Family 1 v1"),
            Self::IosGpuFamily1V2 => write!(f, "iOS GPU Family 1 v2"),
            Self::IosGpuFamily2V1 => write!(f, "iOS GPU Family 2 v1"),
            Self::IosGpuFamily2V2 => write!(f, "iOS GPU Family 2 v2"),
            Self::IosGpuFamily3V1 => write!(f, "iOS GPU Family 3 v1"),
            Self::IosGpuFamily3V2 => write!(f, "iOS GPU Family 3 v2"),
            Self::IosGpuFamily4V1 => write!(f, "iOS GPU Family 4 v1"),
            Self::IosGpuFamily5V1 => write!(f, "iOS GPU Family 5 v1"),
            Self::MacosGpuFamily1V1 => write!(f, "macOS GPU Family 1 v1"),
            Self::MacosGpuFamily1V2 => write!(f, "macOS GPU Family 1 v2"),
            Self::MacosGpuFamily1V3 => write!(f, "macOS GPU Family 1 v3"),
            Self::MacosGpuFamily1V4 => write!(f, "macOS GPU Family 1 v4"),
            Self::MacosGpuFamily2V1 => write!(f, "macOS GPU Family 2 v1"),
            Self::CommonFamily1 => write!(f, "Common Family 1"),
            Self::CommonFamily2 => write!(f, "Common Family 2"),
            Self::CommonFamily3 => write!(f, "Common Family 3"),
            Self::AppleFamily1 => write!(f, "Apple Family 1"),
            Self::AppleFamily2 => write!(f, "Apple Family 2"),
            Self::AppleFamily3 => write!(f, "Apple Family 3"),
            Self::AppleFamily4 => write!(f, "Apple Family 4"),
            Self::AppleFamily5 => write!(f, "Apple Family 5"),
            Self::AppleFamily6 => write!(f, "Apple Family 6"),
            Self::AppleFamily7 => write!(f, "Apple Family 7"),
            Self::AppleFamily8 => write!(f, "Apple Family 8"),
            Self::AppleFamily9 => write!(f, "Apple Family 9"),
        }
    }
}

/// Metal context configuration.
#[derive(Debug, Clone, Default)]
pub struct MetalContextConfig {
    /// Prefer low-power device (integrated GPU on dual-GPU systems).
    pub prefer_low_power: bool,
    /// Prefer headless device (for compute-only workloads).
    pub prefer_headless: bool,
}

/// Metal capabilities and limits.
#[derive(Debug, Clone, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors an independent MTLDevice capability query; keeping them as named fields preserves the 1:1 mapping to Metal"
)]
pub struct MetalCaps {
    /// Base GPU capabilities.
    pub base: GpuCaps,
    /// Highest supported feature set.
    pub feature_set: Option<MetalFeatureSet>,
    /// Is low-power device.
    pub is_low_power: bool,
    /// Is headless device.
    pub is_headless: bool,
    /// Supports Apple Silicon features.
    pub is_apple_silicon: bool,
    /// Maximum buffer length.
    pub max_buffer_length: u64,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: u64,
    /// Maximum total threadgroup memory.
    pub max_threadgroup_memory_length: u64,
    /// Maximum texture dimensions (1D and 2D).
    pub max_texture_size: u32,
    /// Maximum 3D texture dimensions.
    pub max_texture_3d_size: u32,
    /// Maximum cube texture dimensions.
    pub max_texture_cube_size: u32,
    /// Maximum texture array layers.
    pub max_texture_array_layers: u32,
    /// Maximum samplers per shader.
    pub max_samplers_per_stage: u32,
    /// Maximum textures per shader.
    pub max_textures_per_stage: u32,
    /// Maximum buffers per shader.
    pub max_buffers_per_stage: u32,
    /// Supports argument buffers.
    pub argument_buffers: bool,
    /// Supports raster order groups.
    pub raster_order_groups: bool,
    /// Supports 32-bit float filtering.
    pub float32_filtering: bool,
    /// Supports MSAA depth resolve.
    pub msaa_depth_resolve: bool,
    /// Supports sparse textures (residency).
    pub sparse_textures: bool,
    /// Supports function pointers.
    pub function_pointers: bool,
    /// Supports ray tracing.
    pub ray_tracing: bool,
    /// Supports mesh shaders.
    pub mesh_shaders: bool,
    /// Maximum argument buffer tier.
    pub argument_buffer_tier: u32,
    /// Read-write texture tier.
    pub read_write_texture_tier: u32,
}

/// Metal pixel format information.
#[derive(Debug, Clone, Copy)]
pub struct MetalPixelFormatInfo {
    /// Metal pixel format.
    pub format: u32,
    /// Bytes per pixel (0 for compressed).
    pub bytes_per_pixel: u32,
    /// Whether this is a depth format.
    pub is_depth: bool,
    /// Whether this is a stencil format.
    pub is_stencil: bool,
    /// Whether this format is renderable.
    pub renderable: bool,
}

/// Convert `TextureFormat` to Metal pixel format (nominal mapping).
///
/// Note: `Depth24Stencil8` maps to `Depth24Unorm_Stencil8`, which is **not**
/// supported on all Metal devices (Apple-silicon GPUs do not support it).
#[cfg(feature = "metal")]
#[must_use]
pub const fn texture_format_to_metal(format: TextureFormat) -> MTLPixelFormat {
    match format {
        TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        TextureFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        TextureFormat::R8Unorm => MTLPixelFormat::R8Unorm,
        TextureFormat::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        TextureFormat::Depth24Stencil8 => MTLPixelFormat::Depth24Unorm_Stencil8,
        TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
    }
}

/// Convert Metal pixel format to `TextureFormat`.
#[cfg(feature = "metal")]
#[must_use]
pub const fn metal_to_texture_format(format: MTLPixelFormat) -> Option<TextureFormat> {
    match format {
        MTLPixelFormat::RGBA8Unorm => Some(TextureFormat::Rgba8Unorm),
        MTLPixelFormat::RGBA8Unorm_sRGB => Some(TextureFormat::Rgba8UnormSrgb),
        MTLPixelFormat::BGRA8Unorm => Some(TextureFormat::Bgra8Unorm),
        MTLPixelFormat::BGRA8Unorm_sRGB => Some(TextureFormat::Bgra8UnormSrgb),
        MTLPixelFormat::R8Unorm => Some(TextureFormat::R8Unorm),
        MTLPixelFormat::RG8Unorm => Some(TextureFormat::Rg8Unorm),
        MTLPixelFormat::RGBA16Float => Some(TextureFormat::Rgba16Float),
        MTLPixelFormat::RGBA32Float => Some(TextureFormat::Rgba32Float),
        MTLPixelFormat::Depth24Unorm_Stencil8 => Some(TextureFormat::Depth24Stencil8),
        MTLPixelFormat::Depth32Float => Some(TextureFormat::Depth32Float),
        _ => None,
    }
}

/// Metal blend factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalBlendFactor {
    /// 0.
    Zero,
    /// 1.
    #[default]
    One,
    /// Source color.
    SourceColor,
    /// 1 - source color.
    OneMinusSourceColor,
    /// Source alpha.
    SourceAlpha,
    /// 1 - source alpha.
    OneMinusSourceAlpha,
    /// Destination color.
    DestinationColor,
    /// 1 - destination color.
    OneMinusDestinationColor,
    /// Destination alpha.
    DestinationAlpha,
    /// 1 - destination alpha.
    OneMinusDestinationAlpha,
    /// Source alpha saturated.
    SourceAlphaSaturated,
    /// Blend color.
    BlendColor,
    /// 1 - blend color.
    OneMinusBlendColor,
    /// Blend alpha.
    BlendAlpha,
    /// 1 - blend alpha.
    OneMinusBlendAlpha,
}

/// Metal blend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalBlendOperation {
    /// Add.
    #[default]
    Add,
    /// Subtract.
    Subtract,
    /// Reverse subtract.
    ReverseSubtract,
    /// Min.
    Min,
    /// Max.
    Max,
}

/// Metal compare function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalCompareFunction {
    /// Never pass.
    Never,
    /// Pass if less.
    Less,
    /// Pass if equal.
    Equal,
    /// Pass if less or equal.
    LessEqual,
    /// Pass if greater.
    Greater,
    /// Pass if not equal.
    NotEqual,
    /// Pass if greater or equal.
    GreaterEqual,
    /// Always pass.
    #[default]
    Always,
}

/// Metal stencil operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalStencilOperation {
    /// Keep.
    #[default]
    Keep,
    /// Zero.
    Zero,
    /// Replace.
    Replace,
    /// Increment and clamp.
    IncrementClamp,
    /// Decrement and clamp.
    DecrementClamp,
    /// Invert.
    Invert,
    /// Increment and wrap.
    IncrementWrap,
    /// Decrement and wrap.
    DecrementWrap,
}

/// Metal cull mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalCullMode {
    /// No culling.
    #[default]
    None,
    /// Front face culling.
    Front,
    /// Back face culling.
    Back,
}

/// Metal winding order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalWinding {
    /// Clockwise.
    Clockwise,
    /// Counter-clockwise.
    #[default]
    CounterClockwise,
}

/// Metal triangle fill mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalTriangleFillMode {
    /// Fill.
    #[default]
    Fill,
    /// Lines.
    Lines,
}

/// Metal primitive type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalPrimitiveType {
    /// Point.
    Point,
    /// Line.
    Line,
    /// Line strip.
    LineStrip,
    /// Triangle.
    #[default]
    Triangle,
    /// Triangle strip.
    TriangleStrip,
}

/// Metal index type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalIndexType {
    /// 16-bit unsigned integer.
    #[default]
    UInt16,
    /// 32-bit unsigned integer.
    UInt32,
}

/// Metal sampler address mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalSamplerAddressMode {
    /// Clamp to edge.
    #[default]
    ClampToEdge,
    /// Clamp to border color.
    ClampToBorderColor,
    /// Clamp to zero.
    ClampToZero,
    /// Repeat.
    Repeat,
    /// Mirror clamp to edge.
    MirrorClampToEdge,
    /// Mirror repeat.
    MirrorRepeat,
}

/// Metal sampler min/mag filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalSamplerMinMagFilter {
    /// Nearest.
    Nearest,
    /// Linear.
    #[default]
    Linear,
}

/// Metal sampler mip filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalSamplerMipFilter {
    /// Not mipmapped.
    #[default]
    NotMipmapped,
    /// Nearest.
    Nearest,
    /// Linear.
    Linear,
}

/// Metal load action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalLoadAction {
    /// Don't care.
    #[default]
    DontCare,
    /// Load.
    Load,
    /// Clear.
    Clear,
}

/// Metal store action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetalStoreAction {
    /// Don't care.
    DontCare,
    /// Store.
    #[default]
    Store,
    /// Multisample resolve.
    MultisampleResolve,
    /// Store and multisample resolve.
    StoreAndMultisampleResolve,
}

/// Metal-based GPU context.
#[cfg(feature = "metal")]
pub struct MetalContext {
    /// Metal device.
    device: Device,
    /// Default command queue.
    command_queue: metal::CommandQueue,
    /// Adapter info.
    adapter_info: GpuAdapterInfo,
    /// Metal capabilities.
    caps: MetalCaps,
}

/// Whether `device` implements the Objective-C `selector`
/// (`-respondsToSelector:`).
///
/// Newer `MTLDevice` queries (e.g. `supportsFamily:` on macOS 10.15 /
/// iOS 13, `supportsRaytracing` on macOS 11 / iOS 14) raise an
/// unrecognized-selector exception, aborting the process, when sent to a
/// device on an older OS, so they must be probed first.
#[cfg(feature = "metal")]
fn device_responds_to(device: &DeviceRef, selector: &str) -> bool {
    // Sent via `Message::send_message` with runtime-registered selectors:
    // objc 0.2's `msg_send!`/`sel!` macros expand to a `cfg(feature =
    // "cargo-clippy")` check that trips `unexpected_cfgs` in this crate.
    use metal::objc::Message;
    use metal::objc::runtime::{BOOL, NO, Sel};

    let responds_to_selector = Sel::register("respondsToSelector:");
    let selector = Sel::register(selector);
    // SAFETY: `device` is a valid MTLDevice (an NSObject), and
    // `respondsToSelector:` is implemented by every NSObject with the
    // signature `- (BOOL)respondsToSelector:(SEL)aSelector`.
    let responds: BOOL =
        unsafe { device.send_message(responds_to_selector, (selector,)) }.unwrap_or(NO);
    responds != NO
}

/// [`DeviceRef::supports_family`], or `false` on systems that predate the
/// `supportsFamily:` query.
#[cfg(feature = "metal")]
fn supports_family(device: &DeviceRef, family: metal::MTLGPUFamily) -> bool {
    device_responds_to(device, "supportsFamily:") && device.supports_family(family)
}

#[cfg(feature = "metal")]
impl MetalContext {
    /// Create a new Metal context using the system default device.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DeviceCreation`] if the system has no Metal device.
    pub fn new() -> GpuResult<Self> {
        Self::with_config(MetalContextConfig::default())
    }

    /// Create a new Metal context with configuration.
    ///
    /// Falls back to the system default device when no device matches the
    /// low-power / headless preference.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DeviceCreation`] if the system has no Metal device.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "public constructor takes the config by value (builder-style, like Skia's GrContextOptions); changing to a reference would break callers"
    )]
    pub fn with_config(config: MetalContextConfig) -> GpuResult<Self> {
        let device = if config.prefer_low_power {
            Device::all()
                .into_iter()
                .find(|d| d.is_low_power())
                .or_else(Device::system_default)
        } else if config.prefer_headless {
            Device::all()
                .into_iter()
                .find(|d| d.is_headless())
                .or_else(Device::system_default)
        } else {
            Device::system_default()
        };

        let device =
            device.ok_or_else(|| GpuError::DeviceCreation("No Metal device found".into()))?;

        Self::from_device(device)
    }

    /// Create a new Metal context from an existing device.
    ///
    /// # Errors
    ///
    /// Currently never fails: creating the command queue and querying
    /// capabilities are infallible in Metal. The `Result` keeps the signature
    /// aligned with the other backends' constructors.
    pub fn from_device(device: Device) -> GpuResult<Self> {
        let command_queue = device.new_command_queue();

        let caps = Self::query_caps(&device);
        let adapter_info = Self::build_adapter_info(&device, &caps);

        Ok(Self {
            device,
            command_queue,
            adapter_info,
            caps,
        })
    }

    /// Query device capabilities.
    fn query_caps(device: &DeviceRef) -> MetalCaps {
        let is_apple_silicon = device.name().contains("Apple");

        // Query limits
        let max_buffer_length = device.max_buffer_length();
        let max_threads = device.max_threads_per_threadgroup();
        let max_threadgroup_memory = device.max_threadgroup_memory_length();

        // Detect highest feature set
        let feature_set = Self::detect_feature_set(device);

        // Query texture limits based on feature set
        let (max_texture_size, max_3d_size, max_cube_size, max_array_layers) =
            Self::get_texture_limits(feature_set);

        MetalCaps {
            base: GpuCaps {
                max_texture_size,
                max_render_target_size: max_texture_size,
                msaa_support: true,
                max_msaa_samples: 8,
                compute_support: true,
                instancing_support: true,
            },
            feature_set: Some(feature_set),
            is_low_power: device.is_low_power(),
            is_headless: device.is_headless(),
            is_apple_silicon,
            max_buffer_length,
            max_threads_per_threadgroup: max_threads.width * max_threads.height * max_threads.depth,
            max_threadgroup_memory_length: max_threadgroup_memory,
            max_texture_size,
            max_texture_3d_size: max_3d_size,
            max_texture_cube_size: max_cube_size,
            max_texture_array_layers: max_array_layers,
            max_samplers_per_stage: 16,
            max_textures_per_stage: 128,
            max_buffers_per_stage: 31,
            // Both Tier1 and Tier2 support argument buffers; only the tier
            // differs. The previous `!= Tier1` reported Tier1 devices as
            // lacking argument buffers entirely, which is wrong.
            argument_buffers: matches!(
                device.argument_buffers_support(),
                metal::MTLArgumentBuffersTier::Tier1 | metal::MTLArgumentBuffersTier::Tier2
            ),
            raster_order_groups: device.raster_order_groups_supported(),
            float32_filtering: true, // Most Metal devices support this
            msaa_depth_resolve: true,
            // metal-rs exposes no sparse-texture query; sparse textures require the
            // Apple6 GPU family (A13 / Apple silicon) or newer.
            sparse_textures: supports_family(device, metal::MTLGPUFamily::Apple6),
            // macOS 11 / iOS 14 queries: absent (false) on older systems.
            function_pointers: device_responds_to(device, "supportsFunctionPointers")
                && device.supports_function_pointers(),
            ray_tracing: device_responds_to(device, "supportsRaytracing")
                && device.supports_raytracing(),
            mesh_shaders: false, // Requires checking specific feature set
            argument_buffer_tier: match device.argument_buffers_support() {
                metal::MTLArgumentBuffersTier::Tier1 => 1,
                metal::MTLArgumentBuffersTier::Tier2 => 2,
            },
            read_write_texture_tier: match device.read_write_texture_support() {
                metal::MTLReadWriteTextureTier::TierNone => 0,
                metal::MTLReadWriteTextureTier::Tier1 => 1,
                metal::MTLReadWriteTextureTier::Tier2 => 2,
            },
        }
    }

    /// Detect the highest supported feature set.
    fn detect_feature_set(device: &DeviceRef) -> MetalFeatureSet {
        // `supportsFamily:` needs macOS 10.15 / iOS 13; older systems get
        // the lowest common feature set instead of an unrecognized-selector
        // exception.
        if !device_responds_to(device, "supportsFamily:") {
            return MetalFeatureSet::CommonFamily1;
        }

        // Check from highest to lowest
        // Apple Silicon families (newest first)
        if device.supports_family(metal::MTLGPUFamily::Apple9) {
            return MetalFeatureSet::AppleFamily9;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple8) {
            return MetalFeatureSet::AppleFamily8;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple7) {
            return MetalFeatureSet::AppleFamily7;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple6) {
            return MetalFeatureSet::AppleFamily6;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple5) {
            return MetalFeatureSet::AppleFamily5;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple4) {
            return MetalFeatureSet::AppleFamily4;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple3) {
            return MetalFeatureSet::AppleFamily3;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple2) {
            return MetalFeatureSet::AppleFamily2;
        }
        if device.supports_family(metal::MTLGPUFamily::Apple1) {
            return MetalFeatureSet::AppleFamily1;
        }

        // Common families
        if device.supports_family(metal::MTLGPUFamily::Common3) {
            return MetalFeatureSet::CommonFamily3;
        }
        if device.supports_family(metal::MTLGPUFamily::Common2) {
            return MetalFeatureSet::CommonFamily2;
        }
        if device.supports_family(metal::MTLGPUFamily::Common1) {
            return MetalFeatureSet::CommonFamily1;
        }

        // macOS families (legacy)
        if device.supports_family(metal::MTLGPUFamily::Mac2) {
            return MetalFeatureSet::MacosGpuFamily2V1;
        }
        if device.supports_family(metal::MTLGPUFamily::Mac1) {
            return MetalFeatureSet::MacosGpuFamily1V4;
        }

        // Default to lowest common
        MetalFeatureSet::CommonFamily1
    }

    /// Get texture limits for a feature set.
    ///
    /// Apple3+ GPUs (including the legacy iOS GPU family 3+ feature sets)
    /// and every macOS family support 16384-texel 2D/cube textures; older
    /// iOS families (A7/A8) are limited to 8192.
    const fn get_texture_limits(feature_set: MetalFeatureSet) -> (u32, u32, u32, u32) {
        match feature_set {
            MetalFeatureSet::IosGpuFamily3V1
            | MetalFeatureSet::IosGpuFamily3V2
            | MetalFeatureSet::IosGpuFamily4V1
            | MetalFeatureSet::IosGpuFamily5V1
            | MetalFeatureSet::MacosGpuFamily1V1
            | MetalFeatureSet::MacosGpuFamily1V2
            | MetalFeatureSet::MacosGpuFamily1V3
            | MetalFeatureSet::AppleFamily3
            | MetalFeatureSet::AppleFamily4
            | MetalFeatureSet::AppleFamily5
            | MetalFeatureSet::AppleFamily6
            | MetalFeatureSet::AppleFamily7
            | MetalFeatureSet::AppleFamily8
            | MetalFeatureSet::AppleFamily9
            | MetalFeatureSet::MacosGpuFamily1V4
            | MetalFeatureSet::MacosGpuFamily2V1 => (16384, 2048, 16384, 2048),
            _ => (8192, 2048, 8192, 2048),
        }
    }

    /// Build adapter info from device.
    fn build_adapter_info(device: &DeviceRef, caps: &MetalCaps) -> GpuAdapterInfo {
        // Apple Silicon GPUs are integrated even when not flagged low-power.
        let device_type = if device.is_low_power() || caps.is_apple_silicon {
            GpuDeviceType::Integrated
        } else {
            GpuDeviceType::Discrete
        };

        GpuAdapterInfo {
            name: device.name().to_string(),
            vendor: "Apple".to_string(),
            backend: GpuBackendType::Metal,
            device_type,
        }
    }

    /// Get the Metal device.
    #[must_use]
    pub fn device(&self) -> &DeviceRef {
        &self.device
    }

    /// Get the command queue.
    #[must_use]
    pub fn command_queue(&self) -> &metal::CommandQueueRef {
        &self.command_queue
    }

    /// Get Metal capabilities.
    #[must_use]
    pub const fn metal_caps(&self) -> &MetalCaps {
        &self.caps
    }

    /// Create a new command buffer.
    #[must_use]
    pub fn new_command_buffer(&self) -> metal::CommandBuffer {
        self.command_queue.new_command_buffer().to_owned()
    }

    /// Create a new render pass descriptor.
    #[must_use]
    pub fn new_render_pass_descriptor() -> metal::RenderPassDescriptor {
        metal::RenderPassDescriptor::new().to_owned()
    }

    /// Create a new texture descriptor.
    #[must_use]
    pub fn new_texture_descriptor() -> metal::TextureDescriptor {
        metal::TextureDescriptor::new()
    }

    /// Create a new buffer with data.
    #[must_use]
    pub fn new_buffer_with_data(&self, data: &[u8]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a new buffer with size.
    #[must_use]
    pub fn new_buffer(&self, size: u64) -> metal::Buffer {
        self.device
            .new_buffer(size, metal::MTLResourceOptions::StorageModeShared)
    }

    /// Create a new texture.
    #[must_use]
    pub fn new_texture(&self, descriptor: &metal::TextureDescriptorRef) -> metal::Texture {
        self.device.new_texture(descriptor)
    }

    /// Create a new sampler state.
    #[must_use]
    pub fn new_sampler_state(
        &self,
        descriptor: &metal::SamplerDescriptorRef,
    ) -> metal::SamplerState {
        self.device.new_sampler(descriptor)
    }

    /// Create a new depth stencil state.
    #[must_use]
    pub fn new_depth_stencil_state(
        &self,
        descriptor: &metal::DepthStencilDescriptorRef,
    ) -> metal::DepthStencilState {
        self.device.new_depth_stencil_state(descriptor)
    }

    /// Compile a shader library from source.
    ///
    /// # Errors
    ///
    /// Returns the Metal compiler's error message if `source` fails to compile.
    pub fn new_library_with_source(&self, source: &str) -> Result<metal::Library, String> {
        let options = metal::CompileOptions::new();
        self.device.new_library_with_source(source, &options)
    }

    /// Create a render pipeline state.
    ///
    /// # Errors
    ///
    /// Returns Metal's error message if the descriptor is invalid (e.g.
    /// missing functions or incompatible attachment formats).
    pub fn new_render_pipeline_state(
        &self,
        descriptor: &metal::RenderPipelineDescriptorRef,
    ) -> Result<metal::RenderPipelineState, String> {
        self.device.new_render_pipeline_state(descriptor)
    }

    /// Create a compute pipeline state.
    ///
    /// # Errors
    ///
    /// Returns Metal's error message if `function` is not a valid kernel.
    pub fn new_compute_pipeline_state(
        &self,
        function: &metal::FunctionRef,
    ) -> Result<metal::ComputePipelineState, String> {
        self.device
            .new_compute_pipeline_state_with_function(function)
    }
}

#[cfg(feature = "metal")]
impl GpuContext for MetalContext {
    fn backend_type(&self) -> GpuBackendType {
        GpuBackendType::Metal
    }

    fn adapter_info(&self) -> &GpuAdapterInfo {
        &self.adapter_info
    }

    fn flush(&self) {
        // Metal doesn't have an explicit flush
    }

    fn submit_and_wait(&self) {
        // Create a command buffer and wait for it
        let cmd = self.new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// Build a Metal-backed [`crate::wgpu_backend::WgpuContext`].
///
/// Constructs a standalone `wgpu::Instance` restricted to `Backends::METAL`
/// and requests an adapter + device through it. wgpu creates its own
/// `MTLDevice` / `MTLCommandQueue` internally — this does *not* share state
/// with the `metal::Device` inside [`MetalContext`]. Both can coexist; use
/// the native device for bypass work, use the returned context for the
/// command-buffer replay path.
///
/// Returns `Err(GpuError::DeviceCreation)` on non-Apple platforms or any
/// environment where wgpu cannot enumerate a Metal device.
#[cfg(all(feature = "metal", feature = "wgpu-backend"))]
pub fn new_metal_wgpu_context() -> GpuResult<crate::wgpu_backend::WgpuContext> {
    crate::wgpu_backend::WgpuContext::new_with_backends_blocking(wgpu::Backends::METAL)
}

#[cfg(all(feature = "metal", feature = "wgpu-backend"))]
impl MetalContext {
    /// Create a [`crate::wgpu_backend::WgpuExecutor`] routed through wgpu's
    /// Metal backend.
    ///
    /// The returned `(WgpuContext, WgpuExecutor)` is independent of this
    /// `MetalContext`. The existing `metal::Device` / `metal::CommandQueue`
    /// on `self` remain valid for any caller-driven native work.
    pub fn new_wgpu_executor(
        &self,
    ) -> GpuResult<(
        crate::wgpu_backend::WgpuContext,
        crate::wgpu_backend::WgpuExecutor,
    )> {
        let ctx = new_metal_wgpu_context()?;
        let executor = crate::wgpu_backend::WgpuExecutor::new(&ctx);
        Ok((ctx, executor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_feature_set_display() {
        assert_eq!(
            format!("{}", MetalFeatureSet::AppleFamily7),
            "Apple Family 7"
        );
        assert_eq!(
            format!("{}", MetalFeatureSet::CommonFamily1),
            "Common Family 1"
        );
    }

    #[test]
    fn test_metal_config() {
        let config = MetalContextConfig::default();
        assert!(!config.prefer_low_power);
        assert!(!config.prefer_headless);
    }

    #[test]
    fn test_metal_caps_default() {
        let caps = MetalCaps::default();
        assert!(caps.feature_set.is_none());
        assert!(!caps.is_apple_silicon);
    }

    #[test]
    fn test_blend_factor() {
        let factor = MetalBlendFactor::SourceAlpha;
        assert_eq!(factor, MetalBlendFactor::SourceAlpha);
    }

    #[test]
    fn test_compare_function() {
        let func = MetalCompareFunction::Less;
        assert_eq!(func, MetalCompareFunction::Less);
    }

    #[test]
    fn test_primitive_type() {
        let prim = MetalPrimitiveType::Triangle;
        assert_eq!(prim, MetalPrimitiveType::Triangle);
    }

    #[test]
    fn test_sampler_filter() {
        let filter = MetalSamplerMinMagFilter::Linear;
        assert_eq!(filter, MetalSamplerMinMagFilter::Linear);
    }

    /// Smoke test for the wgpu-routed Metal executor. On non-Apple hosts
    /// (which is where CI usually runs) this returns Err cleanly; on
    /// macOS/iOS with Metal support it produces a real Metal-backed
    /// executor. Either outcome is a pass — the regression guard is
    /// against panics or silent fallback to a different backend.
    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn test_metal_wgpu_executor_construction_does_not_panic() {
        match new_metal_wgpu_context() {
            Ok(ctx) => {
                assert_eq!(ctx.backend_type(), GpuBackendType::Metal);
                let _executor = crate::wgpu_backend::WgpuExecutor::new(&ctx);
            }
            Err(GpuError::DeviceCreation(_)) => {
                // Expected on non-Apple platforms.
            }
            Err(other) => panic!("unexpected error kind: {:?}", other),
        }
    }
}
