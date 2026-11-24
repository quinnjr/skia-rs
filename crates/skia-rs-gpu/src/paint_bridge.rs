//! Bridge from `skia_rs_paint::Paint` to GPU pipeline descriptors + uniforms.
//!
//! This module closes the gap between the paint-level drawing API (colors,
//! shaders, blend modes, fill/stroke style) and the GPU-level pipeline
//! configuration (WGSL source, vertex layout, blend state, uniform layout).
//!
//! The previous state of the crate defined both sides — `Paint` in
//! `skia_rs_paint`, pipelines in `skia_rs_gpu::pipeline` — with no wiring
//! between them. A consumer holding a `Paint::LinearGradient` could not
//! select the GPU gradient shader, and the GPU gradient uniforms could not
//! be populated from the paint.
//!
//! The bridge has two concerns:
//!
//! * **Pipeline selection.** Map `Paint::shader()` + `Paint::blend_mode()`
//!   to a `PipelineSelection` describing which built-in shader and blend
//!   state the pipeline cache should use. Identical inputs yield identical
//!   selections, so the pipeline cache can key on it directly.
//! * **Uniform packing.** Flatten shader parameters (gradient endpoints,
//!   colors, tile mode) into a `Vec<u8>` matching the WGSL uniform block
//!   layout for that shader, so the caller can write it straight into a
//!   uniform buffer.
//!
//! Callers use it like so:
//! ```no_run
//! # use skia_rs_paint::Paint;
//! # use skia_rs_gpu::paint_bridge::paint_to_pipeline;
//! # let paint = Paint::default();
//! let plan = paint_to_pipeline(&paint);
//! // plan.pipeline_key identifies which cached pipeline to use
//! // plan.uniforms is the bytes to upload to the uniform buffer
//! ```

use crate::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, PipelineKey,
    RenderPipelineDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat,
};
use crate::shader::builtin;
use crate::texture::TextureFormat;
use skia_rs_paint::shader::{
    LinearGradient, RadialGradient, Shader, ShaderKind, ShaderRef, SweepGradient,
};
use skia_rs_paint::{BlendMode, Paint, Style};

/// Which built-in GPU shader a paint maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinShader {
    /// Solid color fill (no shader / `ColorShader`).
    SolidColor,
    /// Linear gradient.
    LinearGradient,
    /// Radial gradient.
    RadialGradient,
    /// Sweep gradient (uses radial shader with atan2 in fragment — mapped
    /// to `LinearGradient` WGSL in the current built-in library; callers
    /// that need a distinct sweep shader should add one and override).
    SweepGradient,
    /// Image/texture fill.
    Image,
    /// Fallback for shader kinds the built-in library does not handle yet.
    Unsupported,
}

impl BuiltinShader {
    /// Return the vertex and fragment WGSL source for this built-in.
    #[must_use]
    pub const fn wgsl_sources(self) -> (&'static str, &'static str) {
        match self {
            // `Unsupported` falls back to the solid-color built-in.
            Self::SolidColor | Self::Unsupported => {
                (builtin::SOLID_COLOR_VS, builtin::SOLID_COLOR_FS)
            }
            // No dedicated sweep shader in the built-in library yet; linear
            // is the closest analogue (same vertex layout, same single-color
            // output semantics). Callers can override by registering a
            // custom shader in `ShaderLibrary`.
            Self::LinearGradient | Self::SweepGradient => {
                (builtin::GRADIENT_VS, builtin::LINEAR_GRADIENT_FS)
            }
            Self::RadialGradient => (builtin::GRADIENT_VS, builtin::RADIAL_GRADIENT_FS),
            Self::Image => (builtin::TEXTURED_VS, builtin::TEXTURED_FS),
        }
    }
}

/// Whether the destination should be read back during blending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendClass {
    /// Blend mode decomposes into standard Porter-Duff factors.
    PorterDuff,
    /// Advanced (non-separable) blend mode — requires shader-side handling.
    Advanced,
}

/// Translate a `BlendMode` into a GPU-level `BlendState`.
///
/// Every Porter-Duff mode maps cleanly to source/destination factor pairs.
/// Advanced modes (Overlay, HSL, etc.) cannot be represented as a
/// fixed-function blend and require the shader to read the destination; we
/// still return a representative `BlendState` (alpha-over) plus a
/// `BlendClass::Advanced` tag so the caller knows it needs a read-modify-write
/// strategy (destination texture binding + custom shader) rather than the
/// fixed-function blender.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "exhaustive one-arm-per-BlendMode table; splitting it up would obscure the Porter-Duff mapping"
)]
pub const fn blend_mode_to_state(mode: BlendMode) -> (BlendState, BlendClass) {
    let (color, class) = match mode {
        BlendMode::Clear => (
            BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::Src => (
            BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::Dst => (
            BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::SrcOver => (
            BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::DstOver => (
            BlendComponent {
                src_factor: BlendFactor::OneMinusDstAlpha,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::SrcIn => (
            BlendComponent {
                src_factor: BlendFactor::DstAlpha,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::DstIn => (
            BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::SrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::SrcOut => (
            BlendComponent {
                src_factor: BlendFactor::OneMinusDstAlpha,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::DstOut => (
            BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::SrcATop => (
            BlendComponent {
                src_factor: BlendFactor::DstAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::DstATop => (
            BlendComponent {
                src_factor: BlendFactor::OneMinusDstAlpha,
                dst_factor: BlendFactor::SrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::Xor => (
            BlendComponent {
                src_factor: BlendFactor::OneMinusDstAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::Plus => (
            BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::Modulate => (
            BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::Src,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        BlendMode::Screen => (
            BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrc,
                operation: BlendOperation::Add,
            },
            BlendClass::PorterDuff,
        ),
        // Non-separable and advanced modes cannot be expressed as fixed-
        // function factors. Return SrcOver as a sensible fallback and tag
        // the result so the caller knows to branch into the shader path.
        BlendMode::Overlay
        | BlendMode::Darken
        | BlendMode::Lighten
        | BlendMode::ColorDodge
        | BlendMode::ColorBurn
        | BlendMode::HardLight
        | BlendMode::SoftLight
        | BlendMode::Difference
        | BlendMode::Exclusion
        | BlendMode::Multiply
        | BlendMode::Hue
        | BlendMode::Saturation
        | BlendMode::Color
        | BlendMode::Luminosity => (
            BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            BlendClass::Advanced,
        ),
    };

    // Ganesh drives all four channels with a single fixed-function blend
    // formula (`skia/src/gpu/BlendFormula.cpp` `gBlendTable[0][0]`). On the
    // alpha channel, a color-referencing coefficient (SrcColor / DstColor)
    // resolves to the source/destination *alpha*. wgpu separates the color
    // and alpha components, so we reproduce the single-formula behavior by
    // mapping each color factor to its alpha-channel equivalent. This yields
    // per-mode alpha components rather than a hardwired SrcOver alpha.
    let alpha = BlendComponent {
        src_factor: to_alpha_factor(color.src_factor),
        dst_factor: to_alpha_factor(color.dst_factor),
        operation: color.operation,
    };

    (BlendState { color, alpha }, class)
}

/// Map a color-channel blend factor to the equivalent alpha-channel factor.
///
/// Fixed-function blending on the alpha channel treats a source/destination
/// *color* coefficient as the corresponding source/destination *alpha*
/// (there is no separate "color" for a single-component channel). Factors
/// that already reference alpha (or constants) are unchanged.
const fn to_alpha_factor(f: BlendFactor) -> BlendFactor {
    match f {
        BlendFactor::Src => BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrc => BlendFactor::OneMinusSrcAlpha,
        BlendFactor::Dst => BlendFactor::DstAlpha,
        BlendFactor::OneMinusDst => BlendFactor::OneMinusDstAlpha,
        other => other,
    }
}

/// Selection of pipeline parameters derived from a `Paint`.
#[derive(Debug, Clone)]
pub struct PipelineSelection {
    /// Which built-in shader to use.
    pub shader: BuiltinShader,
    /// Blend state for the colour target.
    pub blend: BlendState,
    /// Whether blending needs shader-side destination sampling.
    pub blend_class: BlendClass,
    /// Stroke vs fill style (affects vertex buffer contents, not shader).
    pub style: Style,
}

/// Flattened uniform bytes for a paint's shader.
///
/// The layout matches the `@group(0) @binding(1)` struct in the chosen
/// built-in WGSL source. Callers copy this directly into a uniform buffer.
#[derive(Debug, Clone)]
pub struct PaintUniforms {
    /// Raw bytes to upload.
    pub bytes: Vec<u8>,
}

impl PaintUniforms {
    /// Number of bytes in the uniform block.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True if no uniform data is set.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Combined pipeline + uniform plan derived from a `Paint`.
#[derive(Debug, Clone)]
pub struct PaintPlan {
    /// Pipeline selection (selects which shader and blend state).
    pub selection: PipelineSelection,
    /// Uniforms to upload for the shader.
    pub uniforms: PaintUniforms,
    /// Cached pipeline key (matches `PipelineKey::from_descriptor`).
    pub pipeline_key: PipelineKey,
}

/// Translate a `Paint` into a `PaintPlan`.
///
/// Examines the paint's shader (if any) and blend mode, picks an appropriate
/// built-in GPU shader, builds the pipeline descriptor and key, and packs
/// the shader parameters into a uniform blob.
#[must_use]
pub fn paint_to_pipeline(paint: &Paint) -> PaintPlan {
    let (blend, blend_class) = blend_mode_to_state(paint.blend_mode());

    let (shader_kind, uniforms) = classify_paint_shader(paint);

    let selection = PipelineSelection {
        shader: shader_kind,
        blend,
        blend_class,
        style: paint.style(),
    };

    let descriptor = build_pipeline_descriptor(&selection);
    let pipeline_key = PipelineKey::from_descriptor(&descriptor);

    PaintPlan {
        selection,
        uniforms,
        pipeline_key,
    }
}

/// Build a `RenderPipelineDescriptor` matching a selection.
///
/// Target format is `Rgba8Unorm` by default; callers that render to sRGB
/// or other formats should call [`build_pipeline_descriptor_with_target`]
/// directly.
#[must_use]
pub fn build_pipeline_descriptor(selection: &PipelineSelection) -> RenderPipelineDescriptor {
    build_pipeline_descriptor_with_target(selection, TextureFormat::Rgba8Unorm)
}

/// Build a `RenderPipelineDescriptor` for a specific render target format.
#[must_use]
pub fn build_pipeline_descriptor_with_target(
    selection: &PipelineSelection,
    target: TextureFormat,
) -> RenderPipelineDescriptor {
    let (vs, fs) = selection.shader.wgsl_sources();

    // Vertex layout matches the `TessVertex` layout in `tessellation.rs`:
    // position (Float32x2) then uv (Float32x2), stride 16 bytes.
    let vertex_layout = VertexBufferLayout {
        stride: 16,
        step_mode: crate::pipeline::VertexStepMode::Vertex,
        attributes: vec![
            VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            },
            VertexAttribute {
                location: 1,
                offset: 8,
                format: VertexFormat::Float32x2,
            },
        ],
    };

    let color_target = ColorTargetState {
        format: target,
        blend: Some(selection.blend),
        write_mask: crate::pipeline::ColorWriteMask::ALL,
    };

    RenderPipelineDescriptor::new(vs, fs)
        .with_vertex_buffer(vertex_layout)
        .with_color_target(color_target)
}

fn classify_paint_shader(paint: &Paint) -> (BuiltinShader, PaintUniforms) {
    paint.shader().map_or_else(
        || {
            // Solid color from `paint.color()`.
            let c = paint.color();
            (
                BuiltinShader::SolidColor,
                pack_solid_color(c.r, c.g, c.b, c.a),
            )
        },
        |shader_ref| classify_shader_ref(shader_ref, paint),
    )
}

fn classify_shader_ref(shader_ref: &ShaderRef, paint: &Paint) -> (BuiltinShader, PaintUniforms) {
    match shader_ref.shader_kind() {
        ShaderKind::Color => {
            let c = paint.color();
            (
                BuiltinShader::SolidColor,
                pack_solid_color(c.r, c.g, c.b, c.a),
            )
        }
        ShaderKind::LinearGradient => {
            // We downcast via `Arc<dyn Shader>` by attempting to get a
            // concrete reference through the shader trait's parameter
            // accessors. `skia_rs_paint::shader::Shader` exposes these on
            // the concrete types only, so we reach for them through a
            // `downcast`-style `as &dyn std::any::Any` — but the trait does
            // not extend Any. Instead we probe by kind and sample a known
            // point to infer parameters.
            //
            // Proper solution: `Shader` trait should expose a
            // `to_gpu_params()` method. Pending that refactor, we fall back
            // to a safe extraction via `shader_kind()` + paint color for
            // degenerate end-colour and leave dynamic gradient parameters
            // to be set by the caller via `PaintUniforms` directly.
            (
                BuiltinShader::LinearGradient,
                pack_linear_gradient_from_shader(shader_ref.as_ref(), paint),
            )
        }
        ShaderKind::RadialGradient => (
            BuiltinShader::RadialGradient,
            pack_radial_gradient_from_shader(shader_ref.as_ref(), paint),
        ),
        ShaderKind::SweepGradient => (
            BuiltinShader::SweepGradient,
            pack_sweep_gradient_from_shader(shader_ref.as_ref(), paint),
        ),
        ShaderKind::Image => {
            // Image shaders are uniform-driven by texture binding, not data;
            // the uniform block is the 4-float tint color (paint color).
            let c = paint.color();
            (BuiltinShader::Image, pack_solid_color(c.r, c.g, c.b, c.a))
        }
        _ => {
            // Fall back to solid color for shader kinds we don't yet map.
            let c = paint.color();
            (
                BuiltinShader::Unsupported,
                pack_solid_color(c.r, c.g, c.b, c.a),
            )
        }
    }
}

/// Pack a solid-color uniform (16 bytes: r, g, b, a as f32).
fn pack_solid_color(r: f32, g: f32, b: f32, a: f32) -> PaintUniforms {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&r.to_le_bytes());
    bytes.extend_from_slice(&g.to_le_bytes());
    bytes.extend_from_slice(&b.to_le_bytes());
    bytes.extend_from_slice(&a.to_le_bytes());
    PaintUniforms { bytes }
}

fn pack_linear_gradient(
    start: [f32; 2],
    end: [f32; 2],
    color0: [f32; 4],
    color1: [f32; 4],
) -> PaintUniforms {
    // WGSL struct layout per GRADIENT_UNIFORMS in shader.rs:
    //   start: vec2<f32>,    // offset 0
    //   end:   vec2<f32>,    // offset 8
    //   color0: vec4<f32>,   // offset 16 (vec4 has 16-byte alignment)
    //   color1: vec4<f32>,   // offset 32
    // Total: 48 bytes.
    let mut bytes = Vec::with_capacity(48);
    bytes.extend_from_slice(&start[0].to_le_bytes());
    bytes.extend_from_slice(&start[1].to_le_bytes());
    bytes.extend_from_slice(&end[0].to_le_bytes());
    bytes.extend_from_slice(&end[1].to_le_bytes());
    for c in color0 {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    for c in color1 {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    PaintUniforms { bytes }
}

fn pack_radial_gradient(
    center: [f32; 2],
    radius: f32,
    color0: [f32; 4],
    color1: [f32; 4],
) -> PaintUniforms {
    // WGSL struct: center(vec2) + radius(f32) + _padding(f32) + color0 + color1.
    // 8 + 4 + 4 + 16 + 16 = 48 bytes.
    let mut bytes = Vec::with_capacity(48);
    bytes.extend_from_slice(&center[0].to_le_bytes());
    bytes.extend_from_slice(&center[1].to_le_bytes());
    bytes.extend_from_slice(&radius.to_le_bytes());
    bytes.extend_from_slice(&0.0f32.to_le_bytes()); // padding
    for c in color0 {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    for c in color1 {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    PaintUniforms { bytes }
}

// ----- Concrete-type gradient extraction -----
//
// `skia_rs_paint::shader::Shader` is a trait object; we can't `downcast` it
// directly. The published extraction path is to read parameters through
// `shader.sample(x, y)` at probe points, reconstructing the underlying
// gradient colors. For two-stop gradients, `sample(start)` gives `color0`
// and `sample(end)` gives `color1`. We probe via known-geometry points
// specific to each gradient kind.
//
// This is deliberately a *correct-if-lossy* bridge: for any registered stop
// layout the sampled colors at the gradient endpoints equal the first/last
// stops, which is the correct input to a two-stop WGSL gradient. Richer
// stop lists require a future LUT upload; we currently collapse to the
// end stops and document the fact.

fn pack_linear_gradient_from_shader(shader: &dyn Shader, paint: &Paint) -> PaintUniforms {
    // Default gradient geometry: x in [0, 1]. If the shader provides a
    // local matrix we sample through it; otherwise we sample at the unit
    // diagonal. The result is correct for the shader's own coordinate
    // conventions since we're just asking what color it returns at two
    // specific points and using those as the endpoint colors.
    //
    // We deliberately try to downcast to `LinearGradient` to read geometry
    // precisely. Concrete downcasting isn't available without `Any`, so
    // we fall back to sampling.
    //
    // Future: add an `Any` supertrait to `Shader` or expose
    // `as_linear_gradient()` helpers. Until then the sample-based path is
    // what keeps the bridge working end-to-end.
    let (start, end, c0, c1) = sample_linear_endpoints(shader);
    let c0 = apply_paint_alpha(c0, paint);
    let c1 = apply_paint_alpha(c1, paint);
    pack_linear_gradient(start, end, c0, c1)
}

fn pack_radial_gradient_from_shader(shader: &dyn Shader, paint: &Paint) -> PaintUniforms {
    let (center, radius, c0, c1) = sample_radial_endpoints(shader);
    let c0 = apply_paint_alpha(c0, paint);
    let c1 = apply_paint_alpha(c1, paint);
    pack_radial_gradient(center, radius, c0, c1)
}

fn pack_sweep_gradient_from_shader(shader: &dyn Shader, paint: &Paint) -> PaintUniforms {
    // Map sweep onto the linear uniform block: we treat the sweep's
    // starting and ending angles as a "start" and "end" vec2 in the linear
    // gradient WGSL. Pragmatic pending a dedicated sweep WGSL shader.
    let (start, end, c0, c1) = sample_sweep_as_linear(shader);
    let c0 = apply_paint_alpha(c0, paint);
    let c1 = apply_paint_alpha(c1, paint);
    pack_linear_gradient(start, end, c0, c1)
}

fn sample_linear_endpoints(shader: &dyn Shader) -> ([f32; 2], [f32; 2], [f32; 4], [f32; 4]) {
    // Try concrete downcast via the accessor trait we add below. If the
    // shader is actually a LinearGradient we can read geometry exactly.
    if let Some(g) = as_linear_gradient(shader) {
        let start = [g.start().x, g.start().y];
        let end = [g.end().x, g.end().y];
        let colors = g.colors();
        let c0 = colors.first().copied().unwrap_or_default();
        let c1 = colors.last().copied().unwrap_or_default();
        return (
            start,
            end,
            [c0.r, c0.g, c0.b, c0.a],
            [c1.r, c1.g, c1.b, c1.a],
        );
    }

    // Fallback path: sample the shader at a unit segment to recover
    // endpoint colors.
    let s = shader.sample(0.0, 0.0);
    let e = shader.sample(1.0, 0.0);
    (
        [0.0, 0.0],
        [1.0, 0.0],
        [s.r, s.g, s.b, s.a],
        [e.r, e.g, e.b, e.a],
    )
}

fn sample_radial_endpoints(shader: &dyn Shader) -> ([f32; 2], f32, [f32; 4], [f32; 4]) {
    if let Some(g) = as_radial_gradient(shader) {
        let center = [g.center().x, g.center().y];
        let radius = g.radius();
        let colors = g.colors();
        let c0 = colors.first().copied().unwrap_or_default();
        let c1 = colors.last().copied().unwrap_or_default();
        return (
            center,
            radius,
            [c0.r, c0.g, c0.b, c0.a],
            [c1.r, c1.g, c1.b, c1.a],
        );
    }

    let s = shader.sample(0.0, 0.0);
    let e = shader.sample(1.0, 0.0);
    ([0.0, 0.0], 1.0, [s.r, s.g, s.b, s.a], [e.r, e.g, e.b, e.a])
}

fn sample_sweep_as_linear(shader: &dyn Shader) -> ([f32; 2], [f32; 2], [f32; 4], [f32; 4]) {
    if let Some(g) = as_sweep_gradient(shader) {
        let center = [g.center().x, g.center().y];
        let end = [center[0] + 1.0, center[1]];
        let colors = g.colors();
        let c0 = colors.first().copied().unwrap_or_default();
        let c1 = colors.last().copied().unwrap_or_default();
        return (
            center,
            end,
            [c0.r, c0.g, c0.b, c0.a],
            [c1.r, c1.g, c1.b, c1.a],
        );
    }

    let s = shader.sample(0.0, 0.0);
    let e = shader.sample(1.0, 0.0);
    (
        [0.0, 0.0],
        [1.0, 0.0],
        [s.r, s.g, s.b, s.a],
        [e.r, e.g, e.b, e.a],
    )
}

/// Apply the paint's alpha to a sampled color.
///
/// The paint's alpha multiplies any shader output in Skia's model; we
/// reflect that here so solid and gradient paints with `set_alpha` produce
/// correctly faded GPU output.
fn apply_paint_alpha(color: [f32; 4], paint: &Paint) -> [f32; 4] {
    let a = paint.alpha();
    [color[0], color[1], color[2], color[3] * a]
}

// Concrete-type extraction via the `Shader::as_any` downcast hook. The
// built-in gradient shaders return `Some(self as &dyn Any)` from `as_any`,
// so we can recover their exact geometry (endpoints, radius, angles, color
// stops) instead of probing via `sample`. A `None` result (foreign shader,
// or a shader kind without an `as_any` override) makes the callers fall
// back to the sampling path.

fn as_linear_gradient(shader: &dyn Shader) -> Option<&LinearGradient> {
    shader.as_any()?.downcast_ref::<LinearGradient>()
}

fn as_radial_gradient(shader: &dyn Shader) -> Option<&RadialGradient> {
    shader.as_any()?.downcast_ref::<RadialGradient>()
}

fn as_sweep_gradient(shader: &dyn Shader) -> Option<&SweepGradient> {
    shader.as_any()?.downcast_ref::<SweepGradient>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use skia_rs_core::Color4f;
    use skia_rs_core::Point;
    use skia_rs_paint::Paint;
    use skia_rs_paint::shader::{
        ColorShader, LinearGradient as LG, RadialGradient as RG, TileMode,
    };
    use std::sync::Arc;

    #[test]
    fn test_solid_color_paint_plan() {
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(1.0, 0.5, 0.25, 1.0));
        paint.set_blend_mode(BlendMode::SrcOver);

        let plan = paint_to_pipeline(&paint);
        assert_eq!(plan.selection.shader, BuiltinShader::SolidColor);
        assert_eq!(plan.selection.blend_class, BlendClass::PorterDuff);
        assert_eq!(plan.uniforms.len(), 16); // vec4<f32>
        // First 4 bytes should be the red component.
        let r = f32::from_le_bytes(plan.uniforms.bytes[0..4].try_into().unwrap());
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_mode_to_state_porterduff() {
        let (state, class) = blend_mode_to_state(BlendMode::SrcOver);
        assert_eq!(class, BlendClass::PorterDuff);
        assert_eq!(state.color.src_factor, BlendFactor::One);
        assert_eq!(state.color.dst_factor, BlendFactor::OneMinusSrcAlpha);

        let (state, class) = blend_mode_to_state(BlendMode::Plus);
        assert_eq!(class, BlendClass::PorterDuff);
        assert_eq!(state.color.src_factor, BlendFactor::One);
        assert_eq!(state.color.dst_factor, BlendFactor::One);

        let (_state, class) = blend_mode_to_state(BlendMode::Clear);
        assert_eq!(class, BlendClass::PorterDuff);
    }

    #[test]
    fn test_blend_mode_alpha_is_per_mode() {
        // Regression: alpha components must be derived per mode from Ganesh's
        // single-formula table, not hardwired to SrcOver's (One, ISA).

        // Clear: color (Zero, Zero) => alpha (Zero, Zero).
        let (s, _) = blend_mode_to_state(BlendMode::Clear);
        assert_eq!(s.alpha.src_factor, BlendFactor::Zero);
        assert_eq!(s.alpha.dst_factor, BlendFactor::Zero);

        // Src: color (One, Zero) => alpha (One, Zero).
        let (s, _) = blend_mode_to_state(BlendMode::Src);
        assert_eq!(s.alpha.src_factor, BlendFactor::One);
        assert_eq!(s.alpha.dst_factor, BlendFactor::Zero);

        // Modulate: color dst = Src(color) => alpha dst = SrcAlpha.
        let (s, _) = blend_mode_to_state(BlendMode::Modulate);
        assert_eq!(s.color.dst_factor, BlendFactor::Src);
        assert_eq!(s.alpha.src_factor, BlendFactor::Zero);
        assert_eq!(s.alpha.dst_factor, BlendFactor::SrcAlpha);

        // Screen: color dst = OneMinusSrc(color) => alpha dst = OneMinusSrcAlpha.
        let (s, _) = blend_mode_to_state(BlendMode::Screen);
        assert_eq!(s.color.dst_factor, BlendFactor::OneMinusSrc);
        assert_eq!(s.alpha.dst_factor, BlendFactor::OneMinusSrcAlpha);

        // Plus: color (One, One) => alpha (One, One).
        let (s, _) = blend_mode_to_state(BlendMode::Plus);
        assert_eq!(s.alpha.src_factor, BlendFactor::One);
        assert_eq!(s.alpha.dst_factor, BlendFactor::One);
    }

    #[test]
    fn test_blend_mode_to_state_advanced() {
        for mode in [
            BlendMode::Overlay,
            BlendMode::Multiply,
            BlendMode::HardLight,
            BlendMode::Hue,
            BlendMode::Luminosity,
        ] {
            let (_state, class) = blend_mode_to_state(mode);
            assert_eq!(class, BlendClass::Advanced, "mode {mode:?}");
        }
    }

    #[test]
    fn test_linear_gradient_paint_plan() {
        let gradient = LG::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            vec![
                Color4f::new(1.0, 0.0, 0.0, 1.0),
                Color4f::new(0.0, 0.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(gradient)));

        let plan = paint_to_pipeline(&paint);
        assert_eq!(plan.selection.shader, BuiltinShader::LinearGradient);
        // Uniform block is 48 bytes for two-stop gradient layout.
        assert_eq!(plan.uniforms.len(), 48);
    }

    #[test]
    fn test_radial_gradient_paint_plan() {
        let gradient = RG::new(
            Point::new(0.5, 0.5),
            1.0,
            vec![
                Color4f::new(1.0, 1.0, 1.0, 1.0),
                Color4f::new(0.0, 0.0, 0.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(gradient)));

        let plan = paint_to_pipeline(&paint);
        assert_eq!(plan.selection.shader, BuiltinShader::RadialGradient);
        assert_eq!(plan.uniforms.len(), 48);
    }

    #[test]
    fn test_linear_gradient_reads_real_geometry() {
        // Regression: gradient uniforms must carry the real endpoints and
        // straight (unpremultiplied) stop colors read from geometry, not the
        // premultiplied sample() probe or the unit-segment fallback.
        let gradient = LG::new(
            Point::new(2.0, 3.0),
            Point::new(7.0, 3.0),
            vec![
                Color4f::new(1.0, 0.0, 0.0, 0.5), // straight red, alpha 0.5
                Color4f::new(0.0, 1.0, 0.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(gradient)));

        let plan = paint_to_pipeline(&paint);
        let b = &plan.uniforms.bytes;
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        // start (offset 0..8), end (8..16).
        assert!((f(0) - 2.0).abs() < 1e-6, "start.x from geometry");
        assert!((f(4) - 3.0).abs() < 1e-6, "start.y from geometry");
        assert!((f(8) - 7.0).abs() < 1e-6, "end.x from geometry");
        // color0 at offset 16: straight red 1.0, not premultiplied 0.5.
        assert!(
            (f(16) - 1.0).abs() < 1e-6,
            "color0.r must be straight (1.0)"
        );
        assert!((f(28) - 0.5).abs() < 1e-6, "color0.a preserved");
    }

    #[test]
    fn test_radial_gradient_reads_real_geometry() {
        let gradient = RG::new(
            Point::new(4.0, 5.0),
            9.0,
            vec![
                Color4f::new(1.0, 1.0, 1.0, 1.0),
                Color4f::new(0.0, 0.0, 0.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(gradient)));
        let plan = paint_to_pipeline(&paint);
        let b = &plan.uniforms.bytes;
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert!((f(0) - 4.0).abs() < 1e-6, "center.x");
        assert!((f(4) - 5.0).abs() < 1e-6, "center.y");
        assert!((f(8) - 9.0).abs() < 1e-6, "radius");
    }

    #[test]
    fn test_pipeline_key_stable_across_calls() {
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(1.0, 0.0, 0.0, 1.0));

        let a = paint_to_pipeline(&paint).pipeline_key;
        let b = paint_to_pipeline(&paint).pipeline_key;
        assert_eq!(
            a, b,
            "identical paints must produce identical pipeline keys"
        );
    }

    #[test]
    fn test_pipeline_key_differs_for_different_shaders() {
        // Solid colour paint.
        let mut solid = Paint::new();
        solid.set_color(Color4f::new(1.0, 0.0, 0.0, 1.0));

        // Linear gradient paint.
        let mut lg = Paint::new();
        let gradient = LG::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            vec![
                Color4f::new(1.0, 0.0, 0.0, 1.0),
                Color4f::new(0.0, 0.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        lg.set_shader(Some(Arc::new(gradient)));

        let key_a = paint_to_pipeline(&solid).pipeline_key;
        let key_b = paint_to_pipeline(&lg).pipeline_key;
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn test_color_shader_maps_to_solid() {
        // An explicit ColorShader should behave identically to a bare paint color.
        let shader = ColorShader::new(Color4f::new(0.25, 0.5, 0.75, 1.0));
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(0.25, 0.5, 0.75, 1.0));
        paint.set_shader(Some(Arc::new(shader)));

        let plan = paint_to_pipeline(&paint);
        assert_eq!(plan.selection.shader, BuiltinShader::SolidColor);
    }

    #[test]
    fn test_pipeline_descriptor_has_correct_layout() {
        let mut paint = Paint::new();
        paint.set_color(Color4f::new(1.0, 0.0, 0.0, 1.0));

        let plan = paint_to_pipeline(&paint);
        let desc = build_pipeline_descriptor(&plan.selection);

        assert_eq!(desc.vertex_buffers.len(), 1);
        assert_eq!(desc.vertex_buffers[0].stride, 16);
        assert_eq!(desc.vertex_buffers[0].attributes.len(), 2);
        assert_eq!(desc.color_targets.len(), 1);
        assert!(desc.color_targets[0].blend.is_some());
    }

    #[test]
    fn test_paint_alpha_affects_uniforms() {
        let gradient = LG::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            vec![
                Color4f::new(1.0, 1.0, 1.0, 1.0),
                Color4f::new(1.0, 1.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let mut paint = Paint::new();
        paint.set_shader(Some(Arc::new(gradient)));
        paint.set_alpha(0.5);

        let plan = paint_to_pipeline(&paint);
        // color0.a is at byte offset 16 + 12 = 28 (vec2 start + vec2 end + r,g,b)
        let alpha_bytes = &plan.uniforms.bytes[28..32];
        let alpha = f32::from_le_bytes(alpha_bytes.try_into().unwrap());
        assert!(
            (alpha - 0.5).abs() < 1e-6,
            "paint alpha should fold into uniform color"
        );
    }

    #[test]
    fn test_stroke_style_preserved() {
        let mut paint = Paint::new();
        paint.set_style(Style::Stroke);
        let plan = paint_to_pipeline(&paint);
        assert_eq!(plan.selection.style, Style::Stroke);
    }
}
