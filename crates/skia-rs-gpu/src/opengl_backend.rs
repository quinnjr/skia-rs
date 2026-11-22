//! Direct OpenGL backend implementation using glow.
//!
//! This module provides a direct OpenGL API backend for cases where
//! low-level OpenGL access is needed beyond what wgpu provides.
//!
//! # Executor
//!
//! The OpenGL executor ([`OpenGLContext::new_wgpu_executor`], enabled when the
//! `wgpu-backend` feature is also on) delegates to wgpu configured with
//! [`wgpu::Backends::GL`]. This lets GL shaders, state, and resources flow
//! through the same `CommandBuffer` / `RenderPipelineDescriptor` abstractions
//! used by the primary wgpu backend via naga-generated GLSL.
//!
//! A native GL bypass (direct `glCreateProgram` / `glUseProgram` calls against
//! the `glow::Context`) is not provided by the executor — callers needing
//! GL-specific state (extensions, direct-state-access, bindless textures,
//! `NV_command_list`) can still use the raw `glow::Context` exposed via
//! [`OpenGLContext::gl`], but the default command-buffer execution path
//! goes through wgpu's GL backend.

#[cfg(feature = "opengl")]
use glow::HasContext;

use crate::{
    GpuAdapterInfo, GpuBackendType, GpuCaps, GpuContext, GpuDeviceType, GpuError, GpuResult,
    TextureFormat,
};

/// OpenGL version information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenGLVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Is OpenGL ES.
    pub is_es: bool,
}

impl OpenGLVersion {
    /// OpenGL 3.3.
    pub const GL_3_3: Self = Self {
        major: 3,
        minor: 3,
        is_es: false,
    };
    /// OpenGL 4.0.
    pub const GL_4_0: Self = Self {
        major: 4,
        minor: 0,
        is_es: false,
    };
    /// OpenGL 4.1.
    pub const GL_4_1: Self = Self {
        major: 4,
        minor: 1,
        is_es: false,
    };
    /// OpenGL 4.2.
    pub const GL_4_2: Self = Self {
        major: 4,
        minor: 2,
        is_es: false,
    };
    /// OpenGL 4.3.
    pub const GL_4_3: Self = Self {
        major: 4,
        minor: 3,
        is_es: false,
    };
    /// OpenGL 4.4.
    pub const GL_4_4: Self = Self {
        major: 4,
        minor: 4,
        is_es: false,
    };
    /// OpenGL 4.5.
    pub const GL_4_5: Self = Self {
        major: 4,
        minor: 5,
        is_es: false,
    };
    /// OpenGL 4.6.
    pub const GL_4_6: Self = Self {
        major: 4,
        minor: 6,
        is_es: false,
    };
    /// OpenGL ES 2.0.
    pub const GLES_2_0: Self = Self {
        major: 2,
        minor: 0,
        is_es: true,
    };
    /// OpenGL ES 3.0.
    pub const GLES_3_0: Self = Self {
        major: 3,
        minor: 0,
        is_es: true,
    };
    /// OpenGL ES 3.1.
    pub const GLES_3_1: Self = Self {
        major: 3,
        minor: 1,
        is_es: true,
    };
    /// OpenGL ES 3.2.
    pub const GLES_3_2: Self = Self {
        major: 3,
        minor: 2,
        is_es: true,
    };

    /// Create a new version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, is_es: bool) -> Self {
        Self {
            major,
            minor,
            is_es,
        }
    }

    /// Check if version meets minimum requirement.
    #[must_use]
    pub fn meets(&self, min: &Self) -> bool {
        if self.is_es != min.is_es {
            return false;
        }
        (self.major, self.minor) >= (min.major, min.minor)
    }
}

impl std::fmt::Display for OpenGLVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_es {
            write!(f, "OpenGL ES {}.{}", self.major, self.minor)
        } else {
            write!(f, "OpenGL {}.{}", self.major, self.minor)
        }
    }
}

/// OpenGL context configuration.
#[derive(Debug, Clone)]
pub struct OpenGLContextConfig {
    /// Minimum required OpenGL version.
    pub min_version: OpenGLVersion,
    /// Enable debug output.
    pub debug: bool,
    /// Request core profile (desktop only).
    pub core_profile: bool,
    /// Request forward compatible context (desktop only).
    pub forward_compatible: bool,
}

impl Default for OpenGLContextConfig {
    fn default() -> Self {
        Self {
            min_version: OpenGLVersion::GL_3_3,
            debug: cfg!(debug_assertions),
            core_profile: true,
            forward_compatible: true,
        }
    }
}

/// OpenGL capabilities and limits.
#[derive(Debug, Clone, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool mirrors an independent OpenGL feature/extension flag; a bitflags or enum encoding would obscure the 1:1 mapping to GL queries"
)]
pub struct OpenGLCaps {
    /// Base GPU capabilities.
    pub base: GpuCaps,
    /// OpenGL version.
    pub version: Option<OpenGLVersion>,
    /// GLSL version.
    pub glsl_version: Option<u32>,
    /// Maximum texture units.
    pub max_texture_units: u32,
    /// Maximum combined texture image units.
    pub max_combined_texture_image_units: u32,
    /// Maximum vertex attributes.
    pub max_vertex_attribs: u32,
    /// Maximum uniform buffer bindings.
    pub max_uniform_buffer_bindings: u32,
    /// Maximum uniform block size.
    pub max_uniform_block_size: u32,
    /// Maximum vertex uniform components.
    pub max_vertex_uniform_components: u32,
    /// Maximum fragment uniform components.
    pub max_fragment_uniform_components: u32,
    /// Maximum varying components.
    pub max_varying_components: u32,
    /// Maximum color attachments.
    pub max_color_attachments: u32,
    /// Maximum draw buffers.
    pub max_draw_buffers: u32,
    /// Maximum samples.
    pub max_samples: u32,
    /// Maximum renderbuffer size.
    pub max_renderbuffer_size: u32,
    /// Maximum viewport dimensions.
    pub max_viewport_dims: [u32; 2],
    /// Maximum texture LOD bias.
    pub max_texture_lod_bias: f32,
    /// Maximum anisotropy (0 if not supported).
    pub max_anisotropy: f32,
    /// Supports compute shaders.
    pub compute_shaders: bool,
    /// Supports geometry shaders.
    pub geometry_shaders: bool,
    /// Supports tessellation shaders.
    pub tessellation_shaders: bool,
    /// Supports shader storage buffer objects.
    pub shader_storage_buffers: bool,
    /// Supports shader image load/store.
    pub shader_image_load_store: bool,
    /// Supports multi-draw indirect.
    pub multi_draw_indirect: bool,
    /// Supports buffer storage.
    pub buffer_storage: bool,
    /// Supports texture storage.
    pub texture_storage: bool,
    /// Supports direct state access.
    pub direct_state_access: bool,
    /// Supports debug output.
    pub debug_output: bool,
    /// Supports clip control.
    pub clip_control: bool,
    /// Supports seamless cubemap.
    pub seamless_cubemap: bool,
    /// Supports texture filter anisotropic.
    pub texture_filter_anisotropic: bool,
    /// Supports instanced arrays.
    pub instanced_arrays: bool,
    /// Supports vertex array objects.
    pub vertex_array_objects: bool,
    /// Supports framebuffer objects.
    pub framebuffer_objects: bool,
    /// Supports sRGB framebuffers.
    pub srgb_framebuffer: bool,
}

/// OpenGL shader type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GLShaderType {
    /// Vertex shader.
    Vertex,
    /// Fragment shader.
    Fragment,
    /// Geometry shader.
    Geometry,
    /// Tessellation control shader.
    TessControl,
    /// Tessellation evaluation shader.
    TessEvaluation,
    /// Compute shader.
    Compute,
}

#[cfg(feature = "opengl")]
impl GLShaderType {
    /// Convert to glow shader type.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Vertex => glow::VERTEX_SHADER,
            Self::Fragment => glow::FRAGMENT_SHADER,
            Self::Geometry => glow::GEOMETRY_SHADER,
            Self::TessControl => glow::TESS_CONTROL_SHADER,
            Self::TessEvaluation => glow::TESS_EVALUATION_SHADER,
            Self::Compute => glow::COMPUTE_SHADER,
        }
    }
}

/// OpenGL texture format information.
#[derive(Debug, Clone, Copy)]
pub struct GLTextureFormat {
    /// Internal format (e.g., `GL_RGBA8`).
    pub internal_format: u32,
    /// Format (e.g., `GL_RGBA`).
    pub format: u32,
    /// Type (e.g., `GL_UNSIGNED_BYTE`).
    pub data_type: u32,
}

/// Convert `TextureFormat` to OpenGL format.
#[cfg(feature = "opengl")]
#[must_use]
pub const fn texture_format_to_gl(format: TextureFormat) -> GLTextureFormat {
    match format {
        TextureFormat::Rgba8Unorm => GLTextureFormat {
            internal_format: glow::RGBA8,
            format: glow::RGBA,
            data_type: glow::UNSIGNED_BYTE,
        },
        TextureFormat::Rgba8UnormSrgb => GLTextureFormat {
            internal_format: glow::SRGB8_ALPHA8,
            format: glow::RGBA,
            data_type: glow::UNSIGNED_BYTE,
        },
        TextureFormat::Bgra8Unorm => GLTextureFormat {
            // Note: BGRA is not directly supported in core OpenGL, use RGBA and swizzle
            internal_format: glow::RGBA8,
            format: glow::BGRA,
            data_type: glow::UNSIGNED_BYTE,
        },
        TextureFormat::Bgra8UnormSrgb => GLTextureFormat {
            internal_format: glow::SRGB8_ALPHA8,
            format: glow::BGRA,
            data_type: glow::UNSIGNED_BYTE,
        },
        TextureFormat::R8Unorm => GLTextureFormat {
            internal_format: glow::R8,
            format: glow::RED,
            data_type: glow::UNSIGNED_BYTE,
        },
        TextureFormat::Rg8Unorm => GLTextureFormat {
            internal_format: glow::RG8,
            format: glow::RG,
            data_type: glow::UNSIGNED_BYTE,
        },
        TextureFormat::Rgba16Float => GLTextureFormat {
            internal_format: glow::RGBA16F,
            format: glow::RGBA,
            data_type: glow::HALF_FLOAT,
        },
        TextureFormat::Rgba32Float => GLTextureFormat {
            internal_format: glow::RGBA32F,
            format: glow::RGBA,
            data_type: glow::FLOAT,
        },
        TextureFormat::Depth24Stencil8 => GLTextureFormat {
            internal_format: glow::DEPTH24_STENCIL8,
            format: glow::DEPTH_STENCIL,
            data_type: glow::UNSIGNED_INT_24_8,
        },
        TextureFormat::Depth32Float => GLTextureFormat {
            internal_format: glow::DEPTH_COMPONENT32F,
            format: glow::DEPTH_COMPONENT,
            data_type: glow::FLOAT,
        },
    }
}

/// OpenGL blend factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLBlendFactor {
    /// 0.
    Zero,
    /// 1.
    #[default]
    One,
    /// Source color.
    SrcColor,
    /// 1 - source color.
    OneMinusSrcColor,
    /// Destination color.
    DstColor,
    /// 1 - destination color.
    OneMinusDstColor,
    /// Source alpha.
    SrcAlpha,
    /// 1 - source alpha.
    OneMinusSrcAlpha,
    /// Destination alpha.
    DstAlpha,
    /// 1 - destination alpha.
    OneMinusDstAlpha,
    /// Constant color.
    ConstantColor,
    /// 1 - constant color.
    OneMinusConstantColor,
    /// Constant alpha.
    ConstantAlpha,
    /// 1 - constant alpha.
    OneMinusConstantAlpha,
    /// Saturated source alpha.
    SrcAlphaSaturate,
}

#[cfg(feature = "opengl")]
impl GLBlendFactor {
    /// Convert to OpenGL constant.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Zero => glow::ZERO,
            Self::One => glow::ONE,
            Self::SrcColor => glow::SRC_COLOR,
            Self::OneMinusSrcColor => glow::ONE_MINUS_SRC_COLOR,
            Self::DstColor => glow::DST_COLOR,
            Self::OneMinusDstColor => glow::ONE_MINUS_DST_COLOR,
            Self::SrcAlpha => glow::SRC_ALPHA,
            Self::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
            Self::DstAlpha => glow::DST_ALPHA,
            Self::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
            Self::ConstantColor => glow::CONSTANT_COLOR,
            Self::OneMinusConstantColor => glow::ONE_MINUS_CONSTANT_COLOR,
            Self::ConstantAlpha => glow::CONSTANT_ALPHA,
            Self::OneMinusConstantAlpha => glow::ONE_MINUS_CONSTANT_ALPHA,
            Self::SrcAlphaSaturate => glow::SRC_ALPHA_SATURATE,
        }
    }
}

/// OpenGL blend equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLBlendEquation {
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

#[cfg(feature = "opengl")]
impl GLBlendEquation {
    /// Convert to OpenGL constant.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Add => glow::FUNC_ADD,
            Self::Subtract => glow::FUNC_SUBTRACT,
            Self::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
            Self::Min => glow::MIN,
            Self::Max => glow::MAX,
        }
    }
}

/// OpenGL compare function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLCompareFunc {
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

#[cfg(feature = "opengl")]
impl GLCompareFunc {
    /// Convert to OpenGL constant.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Never => glow::NEVER,
            Self::Less => glow::LESS,
            Self::Equal => glow::EQUAL,
            Self::LessEqual => glow::LEQUAL,
            Self::Greater => glow::GREATER,
            Self::NotEqual => glow::NOTEQUAL,
            Self::GreaterEqual => glow::GEQUAL,
            Self::Always => glow::ALWAYS,
        }
    }
}

/// OpenGL stencil operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLStencilOp {
    /// Keep.
    #[default]
    Keep,
    /// Zero.
    Zero,
    /// Replace.
    Replace,
    /// Increment and clamp.
    Incr,
    /// Increment and wrap.
    IncrWrap,
    /// Decrement and clamp.
    Decr,
    /// Decrement and wrap.
    DecrWrap,
    /// Invert.
    Invert,
}

#[cfg(feature = "opengl")]
impl GLStencilOp {
    /// Convert to OpenGL constant.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Keep => glow::KEEP,
            Self::Zero => glow::ZERO,
            Self::Replace => glow::REPLACE,
            Self::Incr => glow::INCR,
            Self::IncrWrap => glow::INCR_WRAP,
            Self::Decr => glow::DECR,
            Self::DecrWrap => glow::DECR_WRAP,
            Self::Invert => glow::INVERT,
        }
    }
}

/// OpenGL cull face mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLCullFace {
    /// No culling.
    #[default]
    None,
    /// Cull front faces.
    Front,
    /// Cull back faces.
    Back,
    /// Cull both faces.
    FrontAndBack,
}

/// OpenGL front face winding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLFrontFace {
    /// Counter-clockwise.
    #[default]
    Ccw,
    /// Clockwise.
    Cw,
}

#[cfg(feature = "opengl")]
impl GLFrontFace {
    /// Convert to OpenGL constant.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Ccw => glow::CCW,
            Self::Cw => glow::CW,
        }
    }
}

/// OpenGL polygon mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GLPolygonMode {
    /// Fill.
    #[default]
    Fill,
    /// Line.
    Line,
    /// Point.
    Point,
}

#[cfg(feature = "opengl")]
impl GLPolygonMode {
    /// Convert to OpenGL constant.
    #[must_use]
    pub const fn to_gl(self) -> u32 {
        match self {
            Self::Fill => glow::FILL,
            Self::Line => glow::LINE,
            Self::Point => glow::POINT,
        }
    }
}

/// OpenGL-based GPU context.
// Cached binding state (`current_vao`/`current_program`/`current_framebuffer`)
// is retained for future state-tracking use even when not read directly.
#[cfg(feature = "opengl")]
#[allow(dead_code)]
pub struct OpenGLContext {
    /// Glow context.
    gl: glow::Context,
    /// Adapter info.
    adapter_info: GpuAdapterInfo,
    /// OpenGL capabilities.
    caps: OpenGLCaps,
    /// Current bound VAO.
    current_vao: Option<glow::VertexArray>,
    /// Current bound program.
    current_program: Option<glow::Program>,
    /// Current bound framebuffer.
    current_framebuffer: Option<glow::Framebuffer>,
}

#[cfg(feature = "opengl")]
impl OpenGLContext {
    /// Create a new OpenGL context from a glow context.
    ///
    /// # Safety
    /// The provided glow context must be valid and current.
    ///
    /// # Errors
    /// Returns [`GpuError::DeviceCreation`] if the `GL_VERSION` string cannot
    /// be parsed or the context is older than the default minimum version.
    pub unsafe fn from_glow(gl: glow::Context) -> GpuResult<Self> {
        // SAFETY: gl is valid per function contract
        unsafe { Self::from_glow_with_config(gl, OpenGLContextConfig::default()) }
    }

    /// Create a new OpenGL context with configuration.
    ///
    /// # Safety
    /// The provided glow context must be valid and current.
    ///
    /// # Errors
    /// Returns [`GpuError::DeviceCreation`] if the `GL_VERSION` string cannot
    /// be parsed or the context is older than `config.min_version`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "public constructor takes the config by value (builder-style, like Skia's GrContextOptions); changing to a reference would break callers"
    )]
    pub unsafe fn from_glow_with_config(
        gl: glow::Context,
        config: OpenGLContextConfig,
    ) -> GpuResult<Self> {
        // SAFETY: gl is valid per function contract
        unsafe {
            // Get version string
            let version_str = gl.get_parameter_string(glow::VERSION);
            let (major, minor, is_es) = Self::parse_version(&version_str)?;
            let version = OpenGLVersion::new(major, minor, is_es);

            if !version.meets(&config.min_version) {
                return Err(GpuError::DeviceCreation(format!(
                    "OpenGL {} required, found {}",
                    config.min_version, version
                )));
            }

            // Get renderer info
            let renderer = gl.get_parameter_string(glow::RENDERER);
            let vendor = gl.get_parameter_string(glow::VENDOR);

            // Parse GLSL version
            let glsl_version_str = gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION);
            let glsl_version = Self::parse_glsl_version(&glsl_version_str);

            // Query capabilities
            let caps = Self::query_caps(&gl, version, glsl_version);

            let adapter_info = GpuAdapterInfo {
                name: renderer.clone(),
                vendor: vendor.clone(),
                backend: GpuBackendType::OpenGL,
                device_type: Self::detect_device_type(&renderer, &vendor),
            };

            // Enable debug output if requested (without callback for simplicity)
            if config.debug && caps.debug_output {
                gl.enable(glow::DEBUG_OUTPUT);
                gl.enable(glow::DEBUG_OUTPUT_SYNCHRONOUS);
            }

            Ok(Self {
                gl,
                adapter_info,
                caps,
                current_vao: None,
                current_program: None,
                current_framebuffer: None,
            })
        }
    }

    /// Parse OpenGL version string.
    fn parse_version(version: &str) -> GpuResult<(u32, u32, bool)> {
        let is_es = version.contains("ES");
        let numbers: Vec<&str> = version
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .collect();

        if numbers.len() >= 2 {
            let major = numbers[0].parse().unwrap_or(0);
            let minor = numbers[1].parse().unwrap_or(0);
            Ok((major, minor, is_es))
        } else {
            Err(GpuError::DeviceCreation(format!(
                "Failed to parse OpenGL version: {version}"
            )))
        }
    }

    /// Parse GLSL version string.
    fn parse_glsl_version(version: &str) -> Option<u32> {
        let numbers: Vec<&str> = version
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .collect();

        if numbers.len() >= 2 {
            let major: u32 = numbers[0].parse().ok()?;
            let minor: u32 = numbers[1].parse().ok()?;
            Some(major * 100 + minor)
        } else {
            None
        }
    }

    /// Detect device type from renderer/vendor strings.
    fn detect_device_type(renderer: &str, vendor: &str) -> GpuDeviceType {
        let renderer_lower = renderer.to_lowercase();
        let vendor_lower = vendor.to_lowercase();

        if renderer_lower.contains("llvmpipe")
            || renderer_lower.contains("software")
            || renderer_lower.contains("swiftshader")
        {
            return GpuDeviceType::Cpu;
        }

        if renderer_lower.contains("intel") || vendor_lower.contains("intel") {
            // Intel GPUs are typically integrated
            if renderer_lower.contains("arc") {
                return GpuDeviceType::Discrete;
            }
            return GpuDeviceType::Integrated;
        }

        if renderer_lower.contains("nvidia")
            || renderer_lower.contains("geforce")
            || renderer_lower.contains("quadro")
            || vendor_lower.contains("nvidia")
        {
            return GpuDeviceType::Discrete;
        }

        if renderer_lower.contains("amd")
            || renderer_lower.contains("radeon")
            || vendor_lower.contains("amd")
            || vendor_lower.contains("ati")
        {
            // AMD APUs would need more detection
            return GpuDeviceType::Discrete;
        }

        GpuDeviceType::Unknown
    }

    /// Query OpenGL capabilities.
    #[allow(
        clippy::too_many_lines,
        reason = "a flat, linear sequence of independent glGet queries; splitting it would only scatter the field-by-field mapping"
    )]
    unsafe fn query_caps(
        gl: &glow::Context,
        version: OpenGLVersion,
        glsl_version: Option<u32>,
    ) -> OpenGLCaps {
        // SAFETY: gl is valid per function contract
        unsafe {
            // GL reports limits as GLint; reinterpret as unsigned exactly as before.
            let get_int = |pname| gl.get_parameter_i32(pname).cast_unsigned();
            let get_float = |pname| gl.get_parameter_f32(pname);

            let max_texture_size = get_int(glow::MAX_TEXTURE_SIZE);
            let max_samples = get_int(glow::MAX_SAMPLES);

            // Check extension support
            let extensions = Self::get_extensions(gl);
            let has_ext = |name: &str| extensions.iter().any(|e| e == name);

            let compute_shaders = version.meets(&OpenGLVersion::GL_4_3)
                || version.meets(&OpenGLVersion::GLES_3_1)
                || has_ext("GL_ARB_compute_shader");

            let geometry_shaders = version.meets(&OpenGLVersion::GL_3_3)
                || has_ext("GL_ARB_geometry_shader4")
                || has_ext("GL_EXT_geometry_shader");

            let tessellation_shaders =
                version.meets(&OpenGLVersion::GL_4_0) || has_ext("GL_ARB_tessellation_shader");

            let shader_storage_buffers = version.meets(&OpenGLVersion::GL_4_3)
                || version.meets(&OpenGLVersion::GLES_3_1)
                || has_ext("GL_ARB_shader_storage_buffer_object");

            let shader_image_load_store = version.meets(&OpenGLVersion::GL_4_2)
                || version.meets(&OpenGLVersion::GLES_3_1)
                || has_ext("GL_ARB_shader_image_load_store");

            let multi_draw_indirect =
                version.meets(&OpenGLVersion::GL_4_3) || has_ext("GL_ARB_multi_draw_indirect");

            let buffer_storage =
                version.meets(&OpenGLVersion::GL_4_4) || has_ext("GL_ARB_buffer_storage");

            let texture_storage = version.meets(&OpenGLVersion::GL_4_2)
                || version.meets(&OpenGLVersion::GLES_3_0)
                || has_ext("GL_ARB_texture_storage");

            let direct_state_access =
                version.meets(&OpenGLVersion::GL_4_5) || has_ext("GL_ARB_direct_state_access");

            let debug_output = version.meets(&OpenGLVersion::GL_4_3)
                || has_ext("GL_KHR_debug")
                || has_ext("GL_ARB_debug_output");

            let clip_control =
                version.meets(&OpenGLVersion::GL_4_5) || has_ext("GL_ARB_clip_control");

            let seamless_cubemap =
                version.meets(&OpenGLVersion::GL_3_3) || has_ext("GL_ARB_seamless_cube_map");

            let texture_filter_anisotropic = has_ext("GL_EXT_texture_filter_anisotropic")
                || has_ext("GL_ARB_texture_filter_anisotropic");

            let max_anisotropy = if texture_filter_anisotropic {
                get_float(0x84FF) // GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
            } else {
                0.0
            };

            let instanced_arrays = version.meets(&OpenGLVersion::GL_3_3)
                || version.meets(&OpenGLVersion::GLES_3_0)
                || has_ext("GL_ARB_instanced_arrays");

            let vertex_array_objects = version.meets(&OpenGLVersion::GL_3_3)
                || version.meets(&OpenGLVersion::GLES_3_0)
                || has_ext("GL_ARB_vertex_array_object");

            let framebuffer_objects = version.meets(&OpenGLVersion::GL_3_3)
                || version.meets(&OpenGLVersion::GLES_2_0)
                || has_ext("GL_ARB_framebuffer_object");

            let srgb_framebuffer = version.meets(&OpenGLVersion::GL_3_3)
                || has_ext("GL_ARB_framebuffer_sRGB")
                || has_ext("GL_EXT_sRGB");

            let max_viewport_dims = {
                let mut dims = [0i32; 2];
                gl.get_parameter_i32_slice(glow::MAX_VIEWPORT_DIMS, &mut dims);
                [dims[0].cast_unsigned(), dims[1].cast_unsigned()]
            };

            OpenGLCaps {
                base: GpuCaps {
                    max_texture_size,
                    max_render_target_size: get_int(glow::MAX_RENDERBUFFER_SIZE),
                    msaa_support: max_samples > 1,
                    max_msaa_samples: max_samples,
                    compute_support: compute_shaders,
                    instancing_support: instanced_arrays,
                },
                version: Some(version),
                glsl_version,
                max_texture_units: get_int(glow::MAX_TEXTURE_IMAGE_UNITS),
                max_combined_texture_image_units: get_int(glow::MAX_COMBINED_TEXTURE_IMAGE_UNITS),
                max_vertex_attribs: get_int(glow::MAX_VERTEX_ATTRIBS),
                max_uniform_buffer_bindings: if version.meets(&OpenGLVersion::GL_3_3) {
                    get_int(glow::MAX_UNIFORM_BUFFER_BINDINGS)
                } else {
                    0
                },
                max_uniform_block_size: if version.meets(&OpenGLVersion::GL_3_3) {
                    get_int(glow::MAX_UNIFORM_BLOCK_SIZE)
                } else {
                    0
                },
                max_vertex_uniform_components: get_int(glow::MAX_VERTEX_UNIFORM_COMPONENTS),
                max_fragment_uniform_components: get_int(glow::MAX_FRAGMENT_UNIFORM_COMPONENTS),
                max_varying_components: if version.meets(&OpenGLVersion::GL_3_3) {
                    get_int(glow::MAX_VARYING_COMPONENTS)
                } else {
                    0
                },
                max_color_attachments: get_int(glow::MAX_COLOR_ATTACHMENTS),
                max_draw_buffers: get_int(glow::MAX_DRAW_BUFFERS),
                max_samples,
                max_renderbuffer_size: get_int(glow::MAX_RENDERBUFFER_SIZE),
                max_viewport_dims,
                max_texture_lod_bias: get_float(glow::MAX_TEXTURE_LOD_BIAS),
                max_anisotropy,
                compute_shaders,
                geometry_shaders,
                tessellation_shaders,
                shader_storage_buffers,
                shader_image_load_store,
                multi_draw_indirect,
                buffer_storage,
                texture_storage,
                direct_state_access,
                debug_output,
                clip_control,
                seamless_cubemap,
                texture_filter_anisotropic,
                instanced_arrays,
                vertex_array_objects,
                framebuffer_objects,
                srgb_framebuffer,
            }
        }
    }

    /// Get list of supported extensions.
    unsafe fn get_extensions(gl: &glow::Context) -> Vec<String> {
        // SAFETY: gl is valid per function contract
        unsafe {
            // A negative count can only come from a broken driver; treat it as none.
            let num_extensions =
                u32::try_from(gl.get_parameter_i32(glow::NUM_EXTENSIONS)).unwrap_or(0);
            (0..num_extensions)
                .map(|i| gl.get_parameter_indexed_string(glow::EXTENSIONS, i))
                .collect()
        }
    }

    /// Get the glow context.
    pub const fn gl(&self) -> &glow::Context {
        &self.gl
    }

    /// Get OpenGL capabilities.
    pub const fn opengl_caps(&self) -> &OpenGLCaps {
        &self.caps
    }

    /// Clear the color buffer.
    pub fn clear_color(&self, r: f32, g: f32, b: f32, a: f32) {
        unsafe {
            self.gl.clear_color(r, g, b, a);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
        }
    }

    /// Clear the depth buffer.
    pub fn clear_depth(&self, depth: f32) {
        unsafe {
            self.gl.clear_depth_f32(depth);
            self.gl.clear(glow::DEPTH_BUFFER_BIT);
        }
    }

    /// Clear the stencil buffer.
    pub fn clear_stencil(&self, stencil: i32) {
        unsafe {
            self.gl.clear_stencil(stencil);
            self.gl.clear(glow::STENCIL_BUFFER_BIT);
        }
    }

    /// Set the viewport.
    pub fn viewport(&self, x: i32, y: i32, width: i32, height: i32) {
        unsafe {
            self.gl.viewport(x, y, width, height);
        }
    }

    /// Set the scissor rect.
    pub fn scissor(&self, x: i32, y: i32, width: i32, height: i32) {
        unsafe {
            self.gl.scissor(x, y, width, height);
        }
    }

    /// Enable scissor test.
    pub fn enable_scissor(&self) {
        unsafe {
            self.gl.enable(glow::SCISSOR_TEST);
        }
    }

    /// Disable scissor test.
    pub fn disable_scissor(&self) {
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
        }
    }

    /// Enable depth test.
    pub fn enable_depth_test(&self) {
        unsafe {
            self.gl.enable(glow::DEPTH_TEST);
        }
    }

    /// Disable depth test.
    pub fn disable_depth_test(&self) {
        unsafe {
            self.gl.disable(glow::DEPTH_TEST);
        }
    }

    /// Set depth function.
    pub fn depth_func(&self, func: GLCompareFunc) {
        unsafe {
            self.gl.depth_func(func.to_gl());
        }
    }

    /// Set depth mask.
    pub fn depth_mask(&self, enabled: bool) {
        unsafe {
            self.gl.depth_mask(enabled);
        }
    }

    /// Enable blending.
    pub fn enable_blend(&self) {
        unsafe {
            self.gl.enable(glow::BLEND);
        }
    }

    /// Disable blending.
    pub fn disable_blend(&self) {
        unsafe {
            self.gl.disable(glow::BLEND);
        }
    }

    /// Set blend function.
    pub fn blend_func(&self, src: GLBlendFactor, dst: GLBlendFactor) {
        unsafe {
            self.gl.blend_func(src.to_gl(), dst.to_gl());
        }
    }

    /// Set blend function separate.
    pub fn blend_func_separate(
        &self,
        src_rgb: GLBlendFactor,
        dst_rgb: GLBlendFactor,
        src_alpha: GLBlendFactor,
        dst_alpha: GLBlendFactor,
    ) {
        unsafe {
            self.gl.blend_func_separate(
                src_rgb.to_gl(),
                dst_rgb.to_gl(),
                src_alpha.to_gl(),
                dst_alpha.to_gl(),
            );
        }
    }

    /// Set blend equation.
    pub fn blend_equation(&self, mode: GLBlendEquation) {
        unsafe {
            self.gl.blend_equation(mode.to_gl());
        }
    }

    /// Set blend color.
    pub fn blend_color(&self, r: f32, g: f32, b: f32, a: f32) {
        unsafe {
            self.gl.blend_color(r, g, b, a);
        }
    }

    /// Enable face culling.
    pub fn enable_cull_face(&self) {
        unsafe {
            self.gl.enable(glow::CULL_FACE);
        }
    }

    /// Disable face culling.
    pub fn disable_cull_face(&self) {
        unsafe {
            self.gl.disable(glow::CULL_FACE);
        }
    }

    /// Set cull face mode.
    pub fn cull_face(&self, mode: GLCullFace) {
        unsafe {
            match mode {
                GLCullFace::None => self.gl.disable(glow::CULL_FACE),
                GLCullFace::Front => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::FRONT);
                }
                GLCullFace::Back => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::BACK);
                }
                GLCullFace::FrontAndBack => {
                    self.gl.enable(glow::CULL_FACE);
                    self.gl.cull_face(glow::FRONT_AND_BACK);
                }
            }
        }
    }

    /// Set front face winding.
    pub fn front_face(&self, winding: GLFrontFace) {
        unsafe {
            self.gl.front_face(winding.to_gl());
        }
    }

    /// Set polygon mode.
    pub fn polygon_mode(&self, mode: GLPolygonMode) {
        unsafe {
            self.gl.polygon_mode(glow::FRONT_AND_BACK, mode.to_gl());
        }
    }

    /// Draw arrays.
    pub fn draw_arrays(&self, mode: u32, first: i32, count: i32) {
        unsafe {
            self.gl.draw_arrays(mode, first, count);
        }
    }

    /// Draw elements.
    pub fn draw_elements(&self, mode: u32, count: i32, element_type: u32, offset: i32) {
        unsafe {
            self.gl.draw_elements(mode, count, element_type, offset);
        }
    }

    /// Draw arrays instanced.
    pub fn draw_arrays_instanced(&self, mode: u32, first: i32, count: i32, instance_count: i32) {
        unsafe {
            self.gl
                .draw_arrays_instanced(mode, first, count, instance_count);
        }
    }

    /// Draw elements instanced.
    pub fn draw_elements_instanced(
        &self,
        mode: u32,
        count: i32,
        element_type: u32,
        offset: i32,
        instance_count: i32,
    ) {
        unsafe {
            self.gl
                .draw_elements_instanced(mode, count, element_type, offset, instance_count);
        }
    }

    /// Finish all pending commands.
    pub fn finish(&self) {
        unsafe {
            self.gl.finish();
        }
    }

    /// Flush pending commands.
    pub fn flush_gl(&self) {
        unsafe {
            self.gl.flush();
        }
    }
}

#[cfg(feature = "opengl")]
impl GpuContext for OpenGLContext {
    fn backend_type(&self) -> GpuBackendType {
        GpuBackendType::OpenGL
    }

    fn adapter_info(&self) -> &GpuAdapterInfo {
        &self.adapter_info
    }

    fn flush(&self) {
        self.flush_gl();
    }

    fn submit_and_wait(&self) {
        self.finish();
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// Build an OpenGL-backed [`crate::wgpu_backend::WgpuContext`].
///
/// Constructs a standalone `wgpu::Instance` restricted to `Backends::GL`
/// and requests an adapter + device through it. The result is a standard
/// `WgpuContext` that [`crate::wgpu_backend::WgpuExecutor`] consumes — same
/// command-buffer semantics, same pipeline-descriptor translation, routed
/// through wgpu's GL backend (which emits GLSL via naga).
///
/// This is not a native-GL executor. Callers wanting direct `glUseProgram` /
/// `glDrawElements` control should skip this and use [`OpenGLContext::gl`]
/// with hand-rolled GL code. See the module-level docs for the rationale.
///
/// # Errors
/// Returns `Err(GpuError::DeviceCreation)` if the host has no GL driver that
/// wgpu can enumerate. This is expected on headless CI without GPU access;
/// callers should degrade gracefully.
#[cfg(all(feature = "opengl", feature = "wgpu-backend"))]
pub fn new_opengl_wgpu_context() -> GpuResult<crate::wgpu_backend::WgpuContext> {
    crate::wgpu_backend::WgpuContext::new_with_backends_blocking(wgpu::Backends::GL)
}

#[cfg(all(feature = "opengl", feature = "wgpu-backend"))]
impl OpenGLContext {
    /// Create a [`crate::wgpu_backend::WgpuExecutor`] routed through wgpu's
    /// OpenGL backend.
    ///
    /// Note this does *not* reuse the existing `glow::Context` — wgpu owns
    /// its own GL context internally. The two live side-by-side: use
    /// [`Self::gl`] for raw GL state, use the returned executor for
    /// command-buffer replay.
    ///
    /// See [`new_opengl_wgpu_context`] for the standalone constructor that
    /// returns the underlying [`crate::wgpu_backend::WgpuContext`] if you
    /// need access to the device/queue for custom surface creation.
    ///
    /// # Errors
    /// Returns [`GpuError::DeviceCreation`] if wgpu cannot find a GL adapter
    /// (see [`new_opengl_wgpu_context`]).
    pub fn new_wgpu_executor(
        &self,
    ) -> GpuResult<(
        crate::wgpu_backend::WgpuContext,
        crate::wgpu_backend::WgpuExecutor,
    )> {
        let ctx = new_opengl_wgpu_context()?;
        let executor = crate::wgpu_backend::WgpuExecutor::new(&ctx);
        Ok((ctx, executor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opengl_version() {
        let v = OpenGLVersion::new(4, 5, false);
        assert_eq!(v.major, 4);
        assert_eq!(v.minor, 5);
        assert!(!v.is_es);
        assert_eq!(format!("{v}"), "OpenGL 4.5");

        let es = OpenGLVersion::GLES_3_0;
        assert!(es.is_es);
        assert_eq!(format!("{es}"), "OpenGL ES 3.0");
    }

    #[test]
    fn test_version_comparison() {
        assert!(OpenGLVersion::GL_4_5.meets(&OpenGLVersion::GL_3_3));
        assert!(OpenGLVersion::GL_4_5.meets(&OpenGLVersion::GL_4_5));
        assert!(!OpenGLVersion::GL_3_3.meets(&OpenGLVersion::GL_4_5));
        assert!(!OpenGLVersion::GL_4_5.meets(&OpenGLVersion::GLES_3_0)); // Different profiles
    }

    #[test]
    fn test_opengl_config() {
        let config = OpenGLContextConfig::default();
        assert_eq!(config.min_version, OpenGLVersion::GL_3_3);
        assert!(config.core_profile);
    }

    #[test]
    fn test_opengl_caps_default() {
        let caps = OpenGLCaps::default();
        assert_eq!(caps.base.max_texture_size, 0);
        assert!(caps.version.is_none());
    }

    #[test]
    fn test_blend_factor() {
        let factor = GLBlendFactor::SrcAlpha;
        assert_eq!(factor, GLBlendFactor::SrcAlpha);
    }

    #[test]
    fn test_compare_func() {
        let func = GLCompareFunc::Less;
        assert_eq!(func, GLCompareFunc::Less);
    }

    /// Smoke test for the wgpu-routed executor. On hosts without a GL
    /// driver this returns Err rather than panicking — the test passes in
    /// either outcome, the point is to catch a regression where the
    /// adapter construction panics or produces a wgpu instance with the
    /// wrong backend mask.
    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn test_opengl_wgpu_executor_construction_does_not_panic() {
        match new_opengl_wgpu_context() {
            Ok(ctx) => {
                // Confirm the resulting adapter is actually the GL backend,
                // not a silent fallback to Vulkan/Metal/DX12. wgpu reports
                // this through `adapter_info().backend`.
                assert_eq!(ctx.backend_type(), GpuBackendType::OpenGL);
                let _executor = crate::wgpu_backend::WgpuExecutor::new(&ctx);
            }
            Err(GpuError::DeviceCreation(_)) => {
                // Expected on headless CI without a GL driver.
            }
            Err(other) => panic!("unexpected error kind: {other:?}"),
        }
    }
}
