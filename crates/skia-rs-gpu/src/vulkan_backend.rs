//! Direct Vulkan backend implementation using ash.
//!
//! This module provides a direct Vulkan API backend for cases where
//! low-level Vulkan access is needed beyond what wgpu provides.
//!
//! # Executor
//!
//! The Vulkan executor ([`VulkanContext::new_wgpu_executor`], enabled when
//! the `wgpu-backend` feature is also on) delegates to wgpu configured with
//! [`wgpu::Backends::VULKAN`]. Vulkan's upfront render-pass / framebuffer
//! binding, descriptor pools, and pipeline barriers are handled inside wgpu;
//! our `CommandBuffer` / `RenderPipelineDescriptor` abstractions translate
//! through wgpu's naga pipeline to SPIR-V and then to native Vulkan.
//!
//! A native-ash executor that hand-records `vkCmdDraw` /
//! `vkCmdPipelineBarrier` / `vkCmdBeginRenderPass` is not provided. That
//! would duplicate wgpu's internal command translator (~thousands of lines
//! for barriers, descriptor sync, swapchain transitions) and is out of scope
//! for skia-rs-gpu's rendering needs. Callers who need direct ash control
//! can still use [`VulkanContext::device`] and the other raw accessors to
//! drive ash themselves — the wgpu routing is the default executor path,
//! not the only one.

#[cfg(feature = "vulkan")]
use ash::vk;

use crate::{
    GpuAdapterInfo, GpuBackendType, GpuCaps, GpuContext, GpuDeviceType, GpuError, GpuResult,
    TextureFormat,
};
use std::ffi::CString;

/// Parse a Vulkan fixed-size, NUL-terminated device-name array into an owned
/// `String` **without** taking ownership of the borrowed array memory.
///
/// This replaces a previous `CString::from_raw` over the props' stack array,
/// which would free memory it did not own on drop (undefined behavior). We
/// reinterpret the `c_char` array as bytes and stop at the first NUL via
/// `CStr::from_bytes_until_nul`, then copy out an owned `String`.
fn parse_device_name(name: &[std::os::raw::c_char]) -> String {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(name.as_ptr().cast::<u8>(), name.len()) };
    std::ffi::CStr::from_bytes_until_nul(bytes)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Vulkan instance extensions required by skia-rs.
pub const REQUIRED_INSTANCE_EXTENSIONS: &[&str] = &[
    "VK_KHR_surface",
    #[cfg(target_os = "windows")]
    "VK_KHR_win32_surface",
    #[cfg(target_os = "linux")]
    "VK_KHR_xcb_surface",
    #[cfg(target_os = "linux")]
    "VK_KHR_wayland_surface",
    #[cfg(target_os = "macos")]
    "VK_EXT_metal_surface",
    #[cfg(target_os = "android")]
    "VK_KHR_android_surface",
];

/// Vulkan device extensions required by skia-rs.
pub const REQUIRED_DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

/// Optional Vulkan device extensions for advanced features.
pub const OPTIONAL_DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_maintenance1",
    "VK_KHR_maintenance2",
    "VK_KHR_maintenance3",
    "VK_KHR_bind_memory2",
    "VK_KHR_get_memory_requirements2",
    "VK_KHR_dedicated_allocation",
    "VK_KHR_descriptor_update_template",
    "VK_KHR_sampler_ycbcr_conversion",
    "VK_EXT_descriptor_indexing",
    "VK_KHR_buffer_device_address",
    "VK_EXT_memory_budget",
];

/// Vulkan version requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
}

impl VulkanVersion {
    /// Vulkan 1.0.
    pub const V1_0: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };
    /// Vulkan 1.1.
    pub const V1_1: Self = Self {
        major: 1,
        minor: 1,
        patch: 0,
    };
    /// Vulkan 1.2.
    pub const V1_2: Self = Self {
        major: 1,
        minor: 2,
        patch: 0,
    };
    /// Vulkan 1.3.
    pub const V1_3: Self = Self {
        major: 1,
        minor: 3,
        patch: 0,
    };

    /// Create a new version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Convert to Vulkan API version.
    #[cfg(feature = "vulkan")]
    #[must_use]
    pub const fn to_vk_version(&self) -> u32 {
        vk::make_api_version(0, self.major, self.minor, self.patch)
    }

    /// Create from Vulkan API version.
    #[cfg(feature = "vulkan")]
    #[must_use]
    pub const fn from_vk_version(version: u32) -> Self {
        Self {
            major: vk::api_version_major(version),
            minor: vk::api_version_minor(version),
            patch: vk::api_version_patch(version),
        }
    }
}

impl std::fmt::Display for VulkanVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Vulkan context configuration.
#[derive(Debug, Clone)]
pub struct VulkanContextConfig {
    /// Application name.
    pub app_name: String,
    /// Application version.
    pub app_version: VulkanVersion,
    /// Minimum required Vulkan version.
    pub min_vulkan_version: VulkanVersion,
    /// Enable validation layers.
    pub enable_validation: bool,
    /// Additional instance extensions.
    pub instance_extensions: Vec<String>,
    /// Additional device extensions.
    pub device_extensions: Vec<String>,
    /// Prefer discrete GPU.
    pub prefer_discrete: bool,
}

impl Default for VulkanContextConfig {
    fn default() -> Self {
        Self {
            app_name: "skia-rs".to_string(),
            app_version: VulkanVersion::new(0, 1, 0),
            min_vulkan_version: VulkanVersion::V1_1,
            enable_validation: cfg!(debug_assertions),
            instance_extensions: Vec::new(),
            device_extensions: Vec::new(),
            prefer_discrete: true,
        }
    }
}

/// Vulkan physical device information.
#[derive(Debug, Clone)]
pub struct VulkanPhysicalDeviceInfo {
    /// Device name.
    pub name: String,
    /// Vendor ID.
    pub vendor_id: u32,
    /// Device ID.
    pub device_id: u32,
    /// Device type.
    pub device_type: GpuDeviceType,
    /// API version.
    pub api_version: VulkanVersion,
    /// Driver version.
    pub driver_version: u32,
}

/// Vulkan queue family information.
#[derive(Debug, Clone)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors an independent Vulkan capability/flag bit; keeping them as named fields preserves the 1:1 mapping to the Vulkan structs"
)]
pub struct VulkanQueueFamilyInfo {
    /// Queue family index.
    pub index: u32,
    /// Queue count.
    pub count: u32,
    /// Supports graphics.
    pub graphics: bool,
    /// Supports compute.
    pub compute: bool,
    /// Supports transfer.
    pub transfer: bool,
    /// Supports sparse binding.
    pub sparse_binding: bool,
}

/// Vulkan memory type information.
#[derive(Debug, Clone)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors an independent Vulkan capability/flag bit; keeping them as named fields preserves the 1:1 mapping to the Vulkan structs"
)]
pub struct VulkanMemoryTypeInfo {
    /// Memory type index.
    pub index: u32,
    /// Heap index.
    pub heap_index: u32,
    /// Is device local.
    pub device_local: bool,
    /// Is host visible.
    pub host_visible: bool,
    /// Is host coherent.
    pub host_coherent: bool,
    /// Is host cached.
    pub host_cached: bool,
    /// Is lazily allocated.
    pub lazily_allocated: bool,
}

/// Vulkan memory heap information.
#[derive(Debug, Clone)]
pub struct VulkanMemoryHeapInfo {
    /// Heap index.
    pub index: u32,
    /// Heap size in bytes.
    pub size: u64,
    /// Is device local.
    pub device_local: bool,
}

/// Vulkan capabilities and limits.
#[derive(Debug, Clone, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors an independent Vulkan capability/flag bit; keeping them as named fields preserves the 1:1 mapping to the Vulkan structs"
)]
pub struct VulkanCaps {
    /// Base GPU capabilities.
    pub base: GpuCaps,
    /// Vulkan version.
    pub vulkan_version: Option<VulkanVersion>,
    /// Maximum push constant size.
    pub max_push_constant_size: u32,
    /// Maximum bound descriptor sets.
    pub max_bound_descriptor_sets: u32,
    /// Maximum per-stage descriptor samplers.
    pub max_per_stage_descriptor_samplers: u32,
    /// Maximum per-stage descriptor uniform buffers.
    pub max_per_stage_descriptor_uniform_buffers: u32,
    /// Maximum per-stage descriptor storage buffers.
    pub max_per_stage_descriptor_storage_buffers: u32,
    /// Maximum per-stage descriptor sampled images.
    pub max_per_stage_descriptor_sampled_images: u32,
    /// Maximum per-stage descriptor storage images.
    pub max_per_stage_descriptor_storage_images: u32,
    /// Maximum per-stage resources.
    pub max_per_stage_resources: u32,
    /// Maximum descriptor set samplers.
    pub max_descriptor_set_samplers: u32,
    /// Maximum descriptor set uniform buffers.
    pub max_descriptor_set_uniform_buffers: u32,
    /// Maximum descriptor set storage buffers.
    pub max_descriptor_set_storage_buffers: u32,
    /// Maximum framebuffer width.
    pub max_framebuffer_width: u32,
    /// Maximum framebuffer height.
    pub max_framebuffer_height: u32,
    /// Maximum framebuffer layers.
    pub max_framebuffer_layers: u32,
    /// Maximum viewports.
    pub max_viewports: u32,
    /// Supports geometry shaders.
    pub geometry_shader: bool,
    /// Supports tessellation shaders.
    pub tessellation_shader: bool,
    /// Supports multi-draw indirect.
    pub multi_draw_indirect: bool,
    /// Supports draw indirect first instance.
    pub draw_indirect_first_instance: bool,
    /// Supports depth clamp.
    pub depth_clamp: bool,
    /// Supports depth bias clamp.
    pub depth_bias_clamp: bool,
    /// Supports fill mode non-solid.
    pub fill_mode_non_solid: bool,
    /// Supports wide lines.
    pub wide_lines: bool,
    /// Supports large points.
    pub large_points: bool,
    /// Supports alpha to one.
    pub alpha_to_one: bool,
    /// Supports multi viewport.
    pub multi_viewport: bool,
    /// Supports sampler anisotropy.
    pub sampler_anisotropy: bool,
    /// Supports texture compression ETC2.
    pub texture_compression_etc2: bool,
    /// Supports texture compression ASTC LDR.
    pub texture_compression_astc_ldr: bool,
    /// Supports texture compression BC.
    pub texture_compression_bc: bool,
}

/// Vulkan format support flags.
#[derive(Debug, Clone, Copy, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors an independent Vulkan capability/flag bit; keeping them as named fields preserves the 1:1 mapping to the Vulkan structs"
)]
pub struct VulkanFormatFeatures {
    /// Can be sampled.
    pub sampled: bool,
    /// Can be used as storage image.
    pub storage: bool,
    /// Can be used as color attachment.
    pub color_attachment: bool,
    /// Can be used as color attachment with blending.
    pub color_attachment_blend: bool,
    /// Can be used as depth/stencil attachment.
    pub depth_stencil_attachment: bool,
    /// Can be used for blit source.
    pub blit_src: bool,
    /// Can be used for blit destination.
    pub blit_dst: bool,
    /// Supports linear filtering when sampled.
    pub sampled_filter_linear: bool,
    /// Supports transfer source.
    pub transfer_src: bool,
    /// Supports transfer destination.
    pub transfer_dst: bool,
}

/// Convert `TextureFormat` to Vulkan format.
#[cfg(feature = "vulkan")]
#[must_use]
pub const fn texture_format_to_vk(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::Depth24Stencil8 => vk::Format::D24_UNORM_S8_UINT,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
    }
}

/// Convert Vulkan format to `TextureFormat`.
#[cfg(feature = "vulkan")]
#[must_use]
pub const fn vk_format_to_texture(format: vk::Format) -> Option<TextureFormat> {
    match format {
        vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        vk::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8UnormSrgb),
        vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        vk::Format::B8G8R8A8_SRGB => Some(TextureFormat::Bgra8UnormSrgb),
        vk::Format::R8_UNORM => Some(TextureFormat::R8Unorm),
        vk::Format::R8G8_UNORM => Some(TextureFormat::Rg8Unorm),
        vk::Format::R16G16B16A16_SFLOAT => Some(TextureFormat::Rgba16Float),
        vk::Format::R32G32B32A32_SFLOAT => Some(TextureFormat::Rgba32Float),
        vk::Format::D24_UNORM_S8_UINT => Some(TextureFormat::Depth24Stencil8),
        vk::Format::D32_SFLOAT => Some(TextureFormat::Depth32Float),
        _ => None,
    }
}

/// Vulkan-based GPU context.
// Handle/state fields (e.g. `entry`, queue-family indices) are retained to
// keep Vulkan objects alive or for future use even when not read directly.
#[cfg(feature = "vulkan")]
#[allow(dead_code)]
pub struct VulkanContext {
    /// Vulkan entry point.
    entry: ash::Entry,
    /// Vulkan instance.
    instance: ash::Instance,
    /// Physical device.
    physical_device: vk::PhysicalDevice,
    /// Logical device.
    device: ash::Device,
    /// Graphics queue.
    graphics_queue: vk::Queue,
    /// Graphics queue family index.
    graphics_queue_family: u32,
    /// Transfer queue (may be same as graphics).
    transfer_queue: vk::Queue,
    /// Transfer queue family index.
    transfer_queue_family: u32,
    /// Compute queue (may be same as graphics).
    compute_queue: vk::Queue,
    /// Compute queue family index.
    compute_queue_family: u32,
    /// Command pool for graphics operations.
    graphics_command_pool: vk::CommandPool,
    /// Command pool for transfer operations.
    transfer_command_pool: vk::CommandPool,
    /// Adapter info.
    adapter_info: GpuAdapterInfo,
    /// Vulkan capabilities.
    caps: VulkanCaps,
    /// Debug messenger (if validation enabled).
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Debug utils extension loader.
    debug_utils: Option<ash::ext::debug_utils::Instance>,
}

#[cfg(feature = "vulkan")]
impl VulkanContext {
    /// Create a new Vulkan context with default configuration.
    ///
    /// # Errors
    /// See [`Self::with_config`].
    pub fn new() -> GpuResult<Self> {
        Self::with_config(VulkanContextConfig::default())
    }

    /// Create a new Vulkan context with custom configuration.
    ///
    /// # Errors
    /// Returns [`GpuError::DeviceCreation`] if the Vulkan loader cannot be
    /// loaded, instance or device creation fails, no suitable physical device
    /// or graphics queue exists, or another Vulkan call fails during setup.
    ///
    /// # Panics
    /// Panics if `config.app_name` contains an interior NUL byte.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "public constructor takes the config by value (builder-style, like Skia's GrContextOptions); changing to a reference would break callers"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "Vulkan bring-up is one linear sequence (loader, instance, debug messenger, physical device, queues, device); splitting it would thread many intermediate handles through helpers"
    )]
    pub fn with_config(config: VulkanContextConfig) -> GpuResult<Self> {
        unsafe {
            // Load Vulkan
            let entry = ash::Entry::load()
                .map_err(|e| GpuError::DeviceCreation(format!("Failed to load Vulkan: {e:?}")))?;

            // Create instance
            let app_name = CString::new(config.app_name.as_str()).unwrap();
            let engine_name = CString::new("skia-rs").unwrap();

            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(config.app_version.to_vk_version())
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(config.min_vulkan_version.to_vk_version());

            // Collect extensions
            let mut extensions: Vec<CString> = REQUIRED_INSTANCE_EXTENSIONS
                .iter()
                .map(|s| CString::new(*s).unwrap())
                .collect();

            if config.enable_validation {
                extensions.push(CString::new("VK_EXT_debug_utils").unwrap());
            }

            for ext in &config.instance_extensions {
                extensions.push(CString::new(ext.as_str()).unwrap());
            }

            let extension_ptrs: Vec<*const i8> = extensions.iter().map(|s| s.as_ptr()).collect();

            // Validation layers
            let mut layers: Vec<CString> = Vec::new();
            if config.enable_validation {
                layers.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
            }
            let layer_ptrs: Vec<*const i8> = layers.iter().map(|s| s.as_ptr()).collect();

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extension_ptrs)
                .enabled_layer_names(&layer_ptrs);

            let instance = entry.create_instance(&create_info, None).map_err(|e| {
                GpuError::DeviceCreation(format!("Failed to create Vulkan instance: {e:?}"))
            })?;

            // Set up debug messenger
            let (debug_utils, debug_messenger) = if config.enable_validation {
                let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);

                let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback));

                let messenger = debug_utils
                    .create_debug_utils_messenger(&debug_info, None)
                    .ok();

                (Some(debug_utils), messenger)
            } else {
                (None, None)
            };

            // Select physical device
            let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
                GpuError::DeviceCreation(format!("Failed to enumerate devices: {e:?}"))
            })?;

            if physical_devices.is_empty() {
                return Err(GpuError::DeviceCreation("No Vulkan devices found".into()));
            }

            // Score and select best device
            let mut best_device = None;
            let mut best_score = 0i32;

            for pdevice in physical_devices {
                let props = instance.get_physical_device_properties(pdevice);
                let score = Self::score_device(&instance, pdevice, &props, config.prefer_discrete);

                if score > best_score {
                    best_score = score;
                    best_device = Some(pdevice);
                }
            }

            let physical_device = best_device.ok_or_else(|| {
                GpuError::DeviceCreation("No suitable Vulkan device found".into())
            })?;

            let device_props = instance.get_physical_device_properties(physical_device);
            let device_features = instance.get_physical_device_features(physical_device);

            // Find queue families
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let mut graphics_family = None;
            let mut transfer_family = None;
            let mut compute_family = None;

            // Vulkan reports queue family counts as u32, so indices always fit.
            for (i, family) in (0u32..).zip(queue_families.iter()) {
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    graphics_family = Some(i);
                }
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    transfer_family = Some(i);
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    compute_family = Some(i);
                }
            }

            let graphics_queue_family = graphics_family
                .ok_or_else(|| GpuError::DeviceCreation("No graphics queue family found".into()))?;

            // Fall back to graphics queue for transfer/compute if dedicated not available
            let transfer_queue_family = transfer_family.unwrap_or(graphics_queue_family);
            let compute_queue_family = compute_family.unwrap_or(graphics_queue_family);

            // Create logical device
            let queue_priorities = [1.0f32];
            let mut unique_families = vec![graphics_queue_family];
            if transfer_queue_family != graphics_queue_family {
                unique_families.push(transfer_queue_family);
            }
            if compute_queue_family != graphics_queue_family
                && compute_queue_family != transfer_queue_family
            {
                unique_families.push(compute_queue_family);
            }

            let queue_create_infos: Vec<_> = unique_families
                .iter()
                .map(|&family| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(family)
                        .queue_priorities(&queue_priorities)
                })
                .collect();

            // Collect device extensions
            let mut device_extensions: Vec<CString> = REQUIRED_DEVICE_EXTENSIONS
                .iter()
                .map(|s| CString::new(*s).unwrap())
                .collect();

            for ext in &config.device_extensions {
                device_extensions.push(CString::new(ext.as_str()).unwrap());
            }

            let device_extension_ptrs: Vec<*const i8> =
                device_extensions.iter().map(|s| s.as_ptr()).collect();

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&device_extension_ptrs)
                .enabled_features(&device_features);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| GpuError::DeviceCreation(format!("Failed to create device: {e:?}")))?;

            // Get queues
            let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
            let transfer_queue = device.get_device_queue(transfer_queue_family, 0);
            let compute_queue = device.get_device_queue(compute_queue_family, 0);

            // Create command pools
            let graphics_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(graphics_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let graphics_command_pool = device
                .create_command_pool(&graphics_pool_info, None)
                .map_err(|e| {
                    GpuError::ResourceCreation(format!("Failed to create command pool: {e:?}"))
                })?;

            let transfer_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(transfer_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let transfer_command_pool = device
                .create_command_pool(&transfer_pool_info, None)
                .map_err(|e| {
                    GpuError::ResourceCreation(format!("Failed to create command pool: {e:?}"))
                })?;

            // Build adapter info.
            //
            // `device_props.device_name` is a fixed-size, NUL-terminated
            // `[c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]` array owned by the
            // props struct. We must NOT build a `CString` from it: that would
            // take ownership of borrowed stack memory and free it on drop
            // (undefined behavior). Instead, borrow the bytes up to the first
            // NUL via `CStr::from_bytes_until_nul` — no ownership, no free.
            let device_name = parse_device_name(&device_props.device_name);

            let adapter_info = GpuAdapterInfo {
                name: device_name,
                vendor: format!("{:04x}", device_props.vendor_id),
                backend: GpuBackendType::Vulkan,
                device_type: match device_props.device_type {
                    vk::PhysicalDeviceType::INTEGRATED_GPU => GpuDeviceType::Integrated,
                    vk::PhysicalDeviceType::DISCRETE_GPU => GpuDeviceType::Discrete,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => GpuDeviceType::Virtual,
                    vk::PhysicalDeviceType::CPU => GpuDeviceType::Cpu,
                    _ => GpuDeviceType::Unknown,
                },
            };

            // Build capabilities
            let limits = device_props.limits;
            let caps = VulkanCaps {
                base: GpuCaps {
                    max_texture_size: limits.max_image_dimension2_d,
                    max_render_target_size: limits
                        .max_framebuffer_width
                        .min(limits.max_framebuffer_height),
                    msaa_support: true,
                    max_msaa_samples: Self::get_max_sample_count(
                        limits.framebuffer_color_sample_counts,
                    ),
                    compute_support: true,
                    instancing_support: true,
                },
                vulkan_version: Some(VulkanVersion::from_vk_version(device_props.api_version)),
                max_push_constant_size: limits.max_push_constants_size,
                max_bound_descriptor_sets: limits.max_bound_descriptor_sets,
                max_per_stage_descriptor_samplers: limits.max_per_stage_descriptor_samplers,
                max_per_stage_descriptor_uniform_buffers: limits
                    .max_per_stage_descriptor_uniform_buffers,
                max_per_stage_descriptor_storage_buffers: limits
                    .max_per_stage_descriptor_storage_buffers,
                max_per_stage_descriptor_sampled_images: limits
                    .max_per_stage_descriptor_sampled_images,
                max_per_stage_descriptor_storage_images: limits
                    .max_per_stage_descriptor_storage_images,
                max_per_stage_resources: limits.max_per_stage_resources,
                max_descriptor_set_samplers: limits.max_descriptor_set_samplers,
                max_descriptor_set_uniform_buffers: limits.max_descriptor_set_uniform_buffers,
                max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
                max_framebuffer_width: limits.max_framebuffer_width,
                max_framebuffer_height: limits.max_framebuffer_height,
                max_framebuffer_layers: limits.max_framebuffer_layers,
                max_viewports: limits.max_viewports,
                geometry_shader: device_features.geometry_shader != 0,
                tessellation_shader: device_features.tessellation_shader != 0,
                multi_draw_indirect: device_features.multi_draw_indirect != 0,
                draw_indirect_first_instance: device_features.draw_indirect_first_instance != 0,
                depth_clamp: device_features.depth_clamp != 0,
                depth_bias_clamp: device_features.depth_bias_clamp != 0,
                fill_mode_non_solid: device_features.fill_mode_non_solid != 0,
                wide_lines: device_features.wide_lines != 0,
                large_points: device_features.large_points != 0,
                alpha_to_one: device_features.alpha_to_one != 0,
                multi_viewport: device_features.multi_viewport != 0,
                sampler_anisotropy: device_features.sampler_anisotropy != 0,
                texture_compression_etc2: device_features.texture_compression_etc2 != 0,
                texture_compression_astc_ldr: device_features.texture_compression_astc_ldr != 0,
                texture_compression_bc: device_features.texture_compression_bc != 0,
            };

            Ok(Self {
                entry,
                instance,
                physical_device,
                device,
                graphics_queue,
                graphics_queue_family,
                transfer_queue,
                transfer_queue_family,
                compute_queue,
                compute_queue_family,
                graphics_command_pool,
                transfer_command_pool,
                adapter_info,
                caps,
                debug_messenger,
                debug_utils,
            })
        }
    }

    /// Score a physical device for selection.
    unsafe fn score_device(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        props: &vk::PhysicalDeviceProperties,
        prefer_discrete: bool,
    ) -> i32 {
        let mut score = 0i32;

        // Prefer discrete GPUs
        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                score += if prefer_discrete { 1000 } else { 500 };
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => {
                score += if prefer_discrete { 500 } else { 1000 };
            }
            vk::PhysicalDeviceType::VIRTUAL_GPU => {
                score += 100;
            }
            vk::PhysicalDeviceType::CPU => {
                score += 10;
            }
            _ => {}
        }

        // Check for graphics queue
        // SAFETY: device is valid and instance is valid
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        let has_graphics = queue_families
            .iter()
            .any(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS));

        if !has_graphics {
            return -1; // Not suitable
        }

        // Bonus for higher max texture size. u32::MAX / 1024 always fits in
        // an i32, but saturate anyway so the score can never overflow.
        score = score.saturating_add(
            i32::try_from(props.limits.max_image_dimension2_d / 1024).unwrap_or(i32::MAX),
        );

        score
    }

    /// Get maximum sample count from sample count flags.
    const fn get_max_sample_count(flags: vk::SampleCountFlags) -> u32 {
        if flags.contains(vk::SampleCountFlags::TYPE_64) {
            64
        } else if flags.contains(vk::SampleCountFlags::TYPE_32) {
            32
        } else if flags.contains(vk::SampleCountFlags::TYPE_16) {
            16
        } else if flags.contains(vk::SampleCountFlags::TYPE_8) {
            8
        } else if flags.contains(vk::SampleCountFlags::TYPE_4) {
            4
        } else if flags.contains(vk::SampleCountFlags::TYPE_2) {
            2
        } else {
            1
        }
    }

    /// Get the Vulkan device.
    #[must_use]
    pub const fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the physical device.
    #[must_use]
    pub const fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get the Vulkan instance.
    #[must_use]
    pub const fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Get the graphics queue.
    #[must_use]
    pub const fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Get the graphics queue family index.
    #[must_use]
    pub const fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    /// Get the transfer queue.
    #[must_use]
    pub const fn transfer_queue(&self) -> vk::Queue {
        self.transfer_queue
    }

    /// Get the compute queue.
    #[must_use]
    pub const fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    /// Get the graphics command pool.
    #[must_use]
    pub const fn graphics_command_pool(&self) -> vk::CommandPool {
        self.graphics_command_pool
    }

    /// Get Vulkan-specific capabilities.
    #[must_use]
    pub const fn vulkan_caps(&self) -> &VulkanCaps {
        &self.caps
    }

    /// Query format features.
    #[must_use]
    pub fn query_format_features(&self, format: TextureFormat) -> VulkanFormatFeatures {
        unsafe {
            let vk_format = texture_format_to_vk(format);
            let props = self
                .instance
                .get_physical_device_format_properties(self.physical_device, vk_format);

            let optimal = props.optimal_tiling_features;

            VulkanFormatFeatures {
                sampled: optimal.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
                storage: optimal.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
                color_attachment: optimal.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
                color_attachment_blend: optimal
                    .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
                depth_stencil_attachment: optimal
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
                blit_src: optimal.contains(vk::FormatFeatureFlags::BLIT_SRC),
                blit_dst: optimal.contains(vk::FormatFeatureFlags::BLIT_DST),
                sampled_filter_linear: optimal
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
                transfer_src: optimal.contains(vk::FormatFeatureFlags::TRANSFER_SRC),
                transfer_dst: optimal.contains(vk::FormatFeatureFlags::TRANSFER_DST),
            }
        }
    }

    /// Wait for device idle.
    ///
    /// # Errors
    /// Returns an error if `vkDeviceWaitIdle` fails (e.g. device lost).
    pub fn wait_idle(&self) -> GpuResult<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| GpuError::OperationFailed(format!("Wait idle failed: {e:?}")))
        }
    }
}

#[cfg(feature = "vulkan")]
impl GpuContext for VulkanContext {
    fn backend_type(&self) -> GpuBackendType {
        GpuBackendType::Vulkan
    }

    fn adapter_info(&self) -> &GpuAdapterInfo {
        &self.adapter_info
    }

    fn flush(&self) {
        // Vulkan doesn't have an explicit flush
    }

    fn submit_and_wait(&self) {
        let _ = self.wait_idle();
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// Build a Vulkan-backed [`crate::wgpu_backend::WgpuContext`].
///
/// Constructs a standalone `wgpu::Instance` restricted to `Backends::VULKAN`
/// and requests an adapter + device through it. wgpu creates its own
/// `VkInstance` / `VkDevice` internally — this does *not* share state with
/// the `ash::Instance` inside [`VulkanContext`]. The two Vulkan worlds live
/// side-by-side; use ash for driver-direct work, use the returned context
/// for the command-buffer / pipeline-descriptor replay path.
///
/// # Errors
/// Returns `Err(GpuError::DeviceCreation)` on systems without a usable
/// Vulkan loader (missing ICD, no physical device, etc.).
#[cfg(all(feature = "vulkan", feature = "wgpu-backend"))]
pub fn new_vulkan_wgpu_context() -> GpuResult<crate::wgpu_backend::WgpuContext> {
    crate::wgpu_backend::WgpuContext::new_with_backends_blocking(wgpu::Backends::VULKAN)
}

#[cfg(all(feature = "vulkan", feature = "wgpu-backend"))]
impl VulkanContext {
    /// Create a [`crate::wgpu_backend::WgpuExecutor`] routed through wgpu's
    /// Vulkan backend.
    ///
    /// The returned `(WgpuContext, WgpuExecutor)` pair is independent of
    /// this `VulkanContext` — the caller gets a fresh wgpu-managed Vulkan
    /// device that can consume `CommandBuffer` and `RenderPipelineDescriptor`.
    /// The existing ash handles on `self` remain valid for any caller-driven
    /// native work.
    ///
    /// # Errors
    /// Returns [`GpuError::DeviceCreation`] if wgpu cannot find a Vulkan
    /// adapter (see [`new_vulkan_wgpu_context`]).
    pub fn new_wgpu_executor(
        &self,
    ) -> GpuResult<(
        crate::wgpu_backend::WgpuContext,
        crate::wgpu_backend::WgpuExecutor,
    )> {
        let ctx = new_vulkan_wgpu_context()?;
        let executor = crate::wgpu_backend::WgpuExecutor::new(&ctx);
        Ok((ctx, executor))
    }
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
            self.device
                .destroy_command_pool(self.transfer_command_pool, None);

            self.device.destroy_device(None);

            if let (Some(debug_utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan debug callback.
#[cfg(feature = "vulkan")]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: p_callback_data is guaranteed to be valid by the Vulkan spec
    let callback_data = unsafe { *p_callback_data };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        // SAFETY: p_message is a valid C string when not null
        unsafe { std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "ERROR",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "WARNING",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "INFO",
        _ => "VERBOSE",
    };

    let msg_type = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
        _ => "UNKNOWN",
    };

    eprintln!("[Vulkan {severity} {msg_type}] {message}");

    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device_name_no_ownership_ub() {
        // Regression for the CString::from_raw UB: parsing must read the
        // NUL-terminated array by borrow and never free it. Build a padded
        // fixed-size array like Vulkan's device_name field.
        let mut arr = [0 as std::os::raw::c_char; 256];
        for (i, b) in b"NVIDIA RTX".iter().enumerate() {
            arr[i] = std::os::raw::c_char::from_ne_bytes([*b]);
        }
        // Trailing bytes after the NUL are garbage and must be ignored.
        arr[64] = std::os::raw::c_char::from_ne_bytes(*b"X");
        assert_eq!(parse_device_name(&arr), "NVIDIA RTX");

        // An all-zero array parses to an empty string, not a panic.
        let zeros = [0 as std::os::raw::c_char; 8];
        assert_eq!(parse_device_name(&zeros), "");

        // An array with no NUL at all does not panic (returns default).
        let no_nul = [std::os::raw::c_char::from_ne_bytes(*b"A"); 4];
        assert_eq!(parse_device_name(&no_nul), "");
    }

    #[test]
    fn test_vulkan_version() {
        let v = VulkanVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(format!("{v}"), "1.2.3");
    }

    #[test]
    fn test_vulkan_config() {
        let config = VulkanContextConfig::default();
        assert_eq!(config.app_name, "skia-rs");
        assert_eq!(config.min_vulkan_version, VulkanVersion::V1_1);
    }

    #[test]
    fn test_vulkan_caps_default() {
        let caps = VulkanCaps::default();
        assert_eq!(caps.base.max_texture_size, 0);
    }

    #[test]
    fn test_format_features_default() {
        let features = VulkanFormatFeatures::default();
        assert!(!features.sampled);
        assert!(!features.storage);
    }

    /// Smoke test for the wgpu-routed Vulkan executor. Returns Err rather
    /// than panicking on systems without a Vulkan ICD; the test verifies
    /// that the adapter path reaches a Vulkan wgpu backend when a driver
    /// is present and fails cleanly when it isn't.
    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn test_vulkan_wgpu_executor_construction_does_not_panic() {
        match new_vulkan_wgpu_context() {
            Ok(ctx) => {
                assert_eq!(ctx.backend_type(), GpuBackendType::Vulkan);
                let _executor = crate::wgpu_backend::WgpuExecutor::new(&ctx);
            }
            Err(GpuError::DeviceCreation(_)) => {
                // Expected on headless CI or non-Vulkan hosts.
            }
            Err(other) => panic!("unexpected error kind: {other:?}"),
        }
    }
}
