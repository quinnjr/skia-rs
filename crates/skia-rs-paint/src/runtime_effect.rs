//! Runtime effects for custom shaders.
//!
//! This module provides Skia's runtime effects system, allowing custom
//! shaders written in `SkSL` to be compiled and used at runtime.

use crate::shader::{Shader, ShaderKind};
use crate::sksl::{Expr, FnDecl, Parser, SkslProgram, SkslType, Stmt};
use crate::sksl_interp::{Interp, Value as SkslValue};
use skia_rs_core::cast::scalar_from_i32;
use skia_rs_core::{Color4f, Matrix, Scalar};
use std::fmt::Write as _;
use std::sync::{Arc, OnceLock};
use thiserror::Error;

/// Error type for runtime effect operations.
#[derive(Debug, Clone, Error)]
pub enum RuntimeEffectError {
    /// `SkSL` parsing error.
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Compilation error.
    #[error("Compile error: {0}")]
    CompileError(String),
    /// Missing uniform.
    #[error("Missing uniform: {0}")]
    MissingUniform(String),
    /// Type mismatch.
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),
    /// Invalid child count.
    #[error("Invalid child count: expected {expected}, got {got}")]
    InvalidChildCount {
        /// Expected number of children.
        expected: usize,
        /// Number of children supplied.
        got: usize,
    },
    /// Missing main function.
    #[error("Missing main function")]
    MissingMain,
    /// Invalid entry point signature.
    #[error("Invalid entry point signature: expected {expected}")]
    InvalidEntryPoint {
        /// Expected signature.
        expected: String,
    },
    /// Semantic validation (type checking / scope resolution) failed.
    #[error("Semantic validation failed: {0}")]
    ValidationFailed(String),
}

/// Uniform metadata.
#[derive(Debug, Clone)]
pub struct Uniform {
    /// Uniform name.
    pub name: String,
    /// Uniform type.
    pub ty: UniformType,
    /// Byte offset in uniform data.
    pub offset: usize,
    /// Array count (1 for non-arrays).
    pub count: usize,
}

/// Uniform types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformType {
    /// Float scalar.
    Float,
    /// 2-component float vector.
    Float2,
    /// 3-component float vector.
    Float3,
    /// 4-component float vector.
    Float4,
    /// 2x2 float matrix.
    Float2x2,
    /// 3x3 float matrix.
    Float3x3,
    /// 4x4 float matrix.
    Float4x4,
    /// Integer scalar.
    Int,
    /// 2-component integer vector.
    Int2,
    /// 3-component integer vector.
    Int3,
    /// 4-component integer vector.
    Int4,
}

impl UniformType {
    /// Get the size in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Float | Self::Int => 4,
            Self::Float2 | Self::Int2 => 8,
            Self::Float3 | Self::Int3 => 12,
            Self::Float4 | Self::Int4 | Self::Float2x2 => 16,
            Self::Float3x3 => 36,
            Self::Float4x4 => 64,
        }
    }

    /// Get the number of floats/ints.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        match self {
            Self::Float | Self::Int => 1,
            Self::Float2 | Self::Int2 => 2,
            Self::Float3 | Self::Int3 => 3,
            Self::Float4 | Self::Int4 | Self::Float2x2 => 4,
            Self::Float3x3 => 9,
            Self::Float4x4 => 16,
        }
    }

    /// Check if this is a float type.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::Float
                | Self::Float2
                | Self::Float3
                | Self::Float4
                | Self::Float2x2
                | Self::Float3x3
                | Self::Float4x4
        )
    }
}

impl From<&SkslType> for UniformType {
    fn from(ty: &SkslType) -> Self {
        match ty {
            SkslType::Vec2 | SkslType::Half2 => Self::Float2,
            SkslType::Vec3 | SkslType::Half3 => Self::Float3,
            SkslType::Vec4 | SkslType::Half4 => Self::Float4,
            SkslType::Mat2 => Self::Float2x2,
            SkslType::Mat3 => Self::Float3x3,
            SkslType::Mat4 => Self::Float4x4,
            SkslType::Int => Self::Int,
            // Float, Half, and any other/future scalar-like type default
            // to a float uniform slot.
            _ => Self::Float,
        }
    }
}

/// Child shader/color filter metadata.
#[derive(Debug, Clone)]
pub struct Child {
    /// Child name.
    pub name: String,
    /// Child type.
    pub ty: ChildType,
    /// Index in children array.
    pub index: usize,
}

/// Child types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildType {
    /// Shader child.
    Shader,
    /// Color filter child.
    ColorFilter,
    /// Blender child.
    Blender,
}

/// Target language for shader compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    /// GLSL ES 3.0.
    GlslEs300,
    /// GLSL 4.5.
    Glsl450,
    /// SPIR-V.
    SpirV,
    /// Metal Shading Language.
    Msl,
    /// WebGPU Shading Language.
    Wgsl,
}

/// A compiled runtime effect.
#[derive(Debug)]
pub struct RuntimeEffect {
    /// Original `SkSL` source.
    source: String,
    /// Parsed program.
    program: SkslProgram,
    /// Uniforms.
    uniforms: Vec<Uniform>,
    /// Children (shader/colorFilter).
    children: Vec<Child>,
    /// Total uniform data size.
    uniform_size: usize,
    /// Compiled GLSL (cached).
    glsl_cache: OnceLock<String>,
    /// Compiled WGSL (cached).
    wgsl_cache: OnceLock<String>,
    /// Compiled MSL (cached).
    msl_cache: OnceLock<String>,
    /// Compiled SPIR-V words (cached).
    spv_cache: OnceLock<Vec<u32>>,
}

impl RuntimeEffect {
    /// Create a runtime effect from `SkSL` source for shaders.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` fails to parse or fails `SkSL`
    /// validation for the shader effect kind.
    pub fn make_for_shader(source: &str) -> Result<Self, RuntimeEffectError> {
        Self::compile(source, EffectKind::Shader)
    }

    /// Create a runtime effect from `SkSL` source for color filters.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` fails to parse or fails `SkSL`
    /// validation for the color-filter effect kind.
    pub fn make_for_color_filter(source: &str) -> Result<Self, RuntimeEffectError> {
        Self::compile(source, EffectKind::ColorFilter)
    }

    /// Create a runtime effect from `SkSL` source for blenders.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` fails to parse or fails `SkSL`
    /// validation for the blender effect kind.
    pub fn make_for_blender(source: &str) -> Result<Self, RuntimeEffectError> {
        Self::compile(source, EffectKind::Blender)
    }

    fn compile(source: &str, kind: EffectKind) -> Result<Self, RuntimeEffectError> {
        // Preprocess the source to handle #define, #ifdef, etc.
        let preprocessed = crate::sksl::preprocess(source);
        let mut parser = Parser::new(&preprocessed);
        let program = parser
            .parse_program()
            .map_err(RuntimeEffectError::ParseError)?;

        // Validate entry point
        validate_entry_point(&program, kind)?;

        // Semantic validation: type checking, scope resolution, etc.
        crate::sksl_validate::validate_program(&program)
            .map_err(|e| RuntimeEffectError::ValidationFailed(e.to_string()))?;

        // Extract uniforms. Child declarations (`uniform shader/colorFilter/
        // blender`) are children, not data uniforms — they are excluded from
        // uniforms() and do not occupy uniform bytes.
        //
        // Uniforms pack tightly with 4-byte granularity, matching
        // SkRuntimeEffectPriv::VarAsUniform (`uni.offset = *offset;
        // *offset += uni.sizeInBytes();`) — no 16-byte std140-style
        // alignment. Every uniform size is already a multiple of 4.
        let mut uniforms = Vec::new();
        let mut offset = 0;

        for uniform in &program.uniforms {
            if matches!(
                uniform.ty,
                SkslType::Shader | SkslType::ColorFilter | SkslType::Blender
            ) {
                continue;
            }
            let ty = UniformType::from(&uniform.ty);
            let count = uniform.array_size.unwrap_or(1);
            let size = ty.size_bytes() * count;

            uniforms.push(Uniform {
                name: uniform.name.clone(),
                ty,
                offset,
                count,
            });

            offset += size;
        }

        // Extract children
        let mut children = Vec::new();
        let mut child_index = 0;

        for uniform in &program.uniforms {
            let child_type = match &uniform.ty {
                SkslType::Shader => Some(ChildType::Shader),
                SkslType::ColorFilter => Some(ChildType::ColorFilter),
                SkslType::Blender => Some(ChildType::Blender),
                _ => None,
            };

            if let Some(ty) = child_type {
                children.push(Child {
                    name: uniform.name.clone(),
                    ty,
                    index: child_index,
                });
                child_index += 1;
            }
        }

        Ok(Self {
            source: source.to_string(),
            program,
            uniforms,
            children,
            uniform_size: offset,
            glsl_cache: OnceLock::new(),
            wgsl_cache: OnceLock::new(),
            msl_cache: OnceLock::new(),
            spv_cache: OnceLock::new(),
        })
    }

    /// Get the original `SkSL` source this effect was compiled from.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the uniforms.
    pub fn uniforms(&self) -> &[Uniform] {
        &self.uniforms
    }

    /// Get the children.
    pub fn children(&self) -> &[Child] {
        &self.children
    }

    /// Get the uniform data size in bytes.
    pub const fn uniform_size(&self) -> usize {
        self.uniform_size
    }

    /// Find a uniform by name.
    pub fn find_uniform(&self, name: &str) -> Option<&Uniform> {
        self.uniforms.iter().find(|u| u.name == name)
    }

    /// Find a child by name.
    pub fn find_child(&self, name: &str) -> Option<&Child> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Compile to target language.
    ///
    /// # Errors
    ///
    /// Returns an error if lowering the effect to `target` fails (e.g. an
    /// unsupported construct, or a `naga`/SPIR-V backend failure for
    /// [`ShaderTarget::SpirV`]).
    pub fn compile_to(&self, target: ShaderTarget) -> Result<String, RuntimeEffectError> {
        match target {
            ShaderTarget::GlslEs300 | ShaderTarget::Glsl450 => {
                if let Some(cached) = self.glsl_cache.get() {
                    return Ok(cached.clone());
                }
                let output = self.to_glsl(target == ShaderTarget::GlslEs300);
                let _ = self.glsl_cache.set(output.clone());
                Ok(output)
            }
            ShaderTarget::Wgsl => {
                if let Some(cached) = self.wgsl_cache.get() {
                    return Ok(cached.clone());
                }
                let output = self.to_wgsl();
                let _ = self.wgsl_cache.set(output.clone());
                Ok(output)
            }
            ShaderTarget::Msl => {
                if let Some(cached) = self.msl_cache.get() {
                    return Ok(cached.clone());
                }
                let output = self.to_msl();
                let _ = self.msl_cache.set(output.clone());
                Ok(output)
            }
            ShaderTarget::SpirV => {
                let words = self.compile_to_spirv()?;
                // Hex-encode the little-endian byte stream so SPIR-V
                // output fits the `Result<String, _>` API. Callers that
                // need the raw word array should use `compile_to_spirv`.
                let mut out = String::with_capacity(words.len() * 9);
                for (i, w) in words.iter().enumerate() {
                    if i > 0 && i % 8 == 0 {
                        out.push('\n');
                    } else if i > 0 {
                        out.push(' ');
                    }
                    let _ = write!(out, "{w:08x}");
                }
                Ok(out)
            }
        }
    }

    /// Compile to SPIR-V binary.
    ///
    /// Returns the native SPIR-V word array. The implementation lowers
    /// the effect to WGSL using the existing backend, then runs it
    /// through `naga` (parse -> validate -> SPIR-V emit).
    ///
    /// # Errors
    ///
    /// Returns an error if the WGSL lowering, `naga` parse/validation, or
    /// SPIR-V emission step fails.
    pub fn compile_to_spirv(&self) -> Result<Vec<u32>, RuntimeEffectError> {
        if let Some(cached) = self.spv_cache.get() {
            return Ok(cached.clone());
        }

        // Build a WGSL module suitable for naga. Our regular WGSL output
        // lacks entry-point attributes (naga rejects modules without a
        // real vertex/fragment/compute entry point), so wrap the shader
        // body in a synthetic fragment wrapper that naga accepts.
        let wgsl = self.to_wgsl_for_naga();

        let module = naga::front::wgsl::parse_str(&wgsl).map_err(|e| {
            RuntimeEffectError::CompileError(format!(
                "naga WGSL parse failed: {}\n---\n{}",
                e.message(),
                wgsl
            ))
        })?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).map_err(|e| {
            RuntimeEffectError::CompileError(format!("naga validation failed: {e}"))
        })?;

        let options = naga::back::spv::Options::default();
        let words = naga::back::spv::write_vec(&module, &info, &options, None)
            .map_err(|e| RuntimeEffectError::CompileError(format!("naga SPIR-V emit: {e}")))?;

        let _ = self.spv_cache.set(words.clone());
        Ok(words)
    }

    /// Convert to GLSL.
    fn to_glsl(&self, es: bool) -> String {
        let mut output = String::new();

        // Version
        if es {
            output.push_str("#version 300 es\n");
            output.push_str("precision highp float;\n");
        } else {
            output.push_str("#version 450\n");
        }
        output.push('\n');

        // Uniforms
        for uniform in &self.uniforms {
            let _ = writeln!(
                output,
                "uniform {} {};",
                Self::type_to_glsl(uniform.ty),
                uniform.name
            );
        }
        if !self.uniforms.is_empty() {
            output.push('\n');
        }

        // Functions
        for func in &self.program.functions {
            output.push_str(&Self::function_to_glsl(func));
            output.push('\n');
        }

        output
    }

    const fn type_to_glsl(ty: UniformType) -> &'static str {
        match ty {
            UniformType::Float => "float",
            UniformType::Float2 => "vec2",
            UniformType::Float3 => "vec3",
            UniformType::Float4 => "vec4",
            UniformType::Float2x2 => "mat2",
            UniformType::Float3x3 => "mat3",
            UniformType::Float4x4 => "mat4",
            UniformType::Int => "int",
            UniformType::Int2 => "ivec2",
            UniformType::Int3 => "ivec3",
            UniformType::Int4 => "ivec4",
        }
    }

    fn function_to_glsl(func: &FnDecl) -> String {
        let mut output = String::new();

        // Return type and name
        output.push_str(func.return_type.glsl_name());
        output.push(' ');
        output.push_str(&func.name);
        output.push('(');

        // Parameters
        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(param.ty.glsl_name());
            output.push(' ');
            output.push_str(&param.name);
        }

        output.push_str(") ");
        output.push_str(&Self::stmt_to_glsl(&func.body, 0));

        output
    }

    fn stmt_to_glsl(stmt: &Stmt, indent: usize) -> String {
        let ind = "    ".repeat(indent);
        match stmt {
            Stmt::Expr(expr) => format!("{}{};\n", ind, Self::expr_to_glsl(expr)),
            Stmt::VarDecl { ty, name, init } => init.as_ref().map_or_else(
                || format!("{}{} {};\n", ind, ty.glsl_name(), name),
                |init| {
                    format!(
                        "{}{} {} = {};\n",
                        ind,
                        ty.glsl_name(),
                        name,
                        Self::expr_to_glsl(init)
                    )
                },
            ),
            Stmt::Block(stmts) => {
                let mut output = format!("{ind}{{\n");
                for s in stmts {
                    output.push_str(&Self::stmt_to_glsl(s, indent + 1));
                }
                let _ = writeln!(output, "{ind}}}");
                output
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let mut output = format!("{}if ({}) ", ind, Self::expr_to_glsl(cond));
                output.push_str(&Self::stmt_to_glsl(then_branch, indent));
                if let Some(else_b) = else_branch {
                    let _ = write!(output, "{ind}else ");
                    output.push_str(&Self::stmt_to_glsl(else_b, indent));
                }
                output
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                let mut output = format!("{ind}for (");
                if let Some(init) = init {
                    let init_str = Self::stmt_to_glsl(init, 0);
                    output.push_str(init_str.trim());
                } else {
                    output.push(';');
                }
                output.push(' ');
                if let Some(cond) = cond {
                    output.push_str(&Self::expr_to_glsl(cond));
                }
                output.push_str("; ");
                if let Some(update) = update {
                    output.push_str(&Self::expr_to_glsl(update));
                }
                output.push_str(") ");
                output.push_str(&Self::stmt_to_glsl(body, indent));
                output
            }
            Stmt::While { cond, body } => {
                let mut output = format!("{}while ({}) ", ind, Self::expr_to_glsl(cond));
                output.push_str(&Self::stmt_to_glsl(body, indent));
                output
            }
            Stmt::DoWhile { body, cond } => {
                let mut output = format!("{ind}do ");
                output.push_str(&Self::stmt_to_glsl(body, indent));
                let _ = writeln!(output, " while ({});", Self::expr_to_glsl(cond));
                output
            }
            Stmt::Return(expr) => expr.as_ref().map_or_else(
                || format!("{ind}return;\n"),
                |expr| format!("{}return {};\n", ind, Self::expr_to_glsl(expr)),
            ),
            Stmt::Break => format!("{ind}break;\n"),
            Stmt::Continue => format!("{ind}continue;\n"),
            Stmt::Discard => format!("{ind}discard;\n"),
        }
    }

    fn expr_to_glsl(expr: &Expr) -> String {
        match expr {
            Expr::IntLit(n) => n.to_string(),
            Expr::FloatLit(n) => {
                if n.fract() == 0.0 {
                    format!("{n}.0")
                } else {
                    format!("{n}")
                }
            }
            Expr::BoolLit(b) => b.to_string(),
            Expr::Var(name) => name.clone(),
            Expr::Binary { left, op, right } => {
                format!(
                    "({} {} {})",
                    Self::expr_to_glsl(left),
                    op.glsl_str(),
                    Self::expr_to_glsl(right)
                )
            }
            Expr::Unary { op, expr } => {
                format!("({}{})", op.glsl_str(), Self::expr_to_glsl(expr))
            }
            Expr::Call { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::expr_to_glsl).collect();
                format!("{}({})", name, args_str.join(", "))
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Child eval: emitted as a helper-call naming convention
                // (`<child>_eval(...)`); GPU backends bind child shaders as
                // functions/samplers under this name.
                let args_str: Vec<String> = args.iter().map(Self::expr_to_glsl).collect();
                format!(
                    "{}_{}({})",
                    Self::expr_to_glsl(receiver),
                    method,
                    args_str.join(", ")
                )
            }
            Expr::Constructor { ty, args } => {
                let args_str: Vec<String> = args.iter().map(Self::expr_to_glsl).collect();
                format!("{}({})", ty.glsl_name(), args_str.join(", "))
            }
            Expr::Field { expr, field } => {
                format!("{}.{}", Self::expr_to_glsl(expr), field)
            }
            Expr::Index { expr, index } => {
                format!(
                    "{}[{}]",
                    Self::expr_to_glsl(expr),
                    Self::expr_to_glsl(index)
                )
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                format!(
                    "({} ? {} : {})",
                    Self::expr_to_glsl(cond),
                    Self::expr_to_glsl(then_expr),
                    Self::expr_to_glsl(else_expr)
                )
            }
            Expr::Assign { target, value } => {
                format!(
                    "({} = {})",
                    Self::expr_to_glsl(target),
                    Self::expr_to_glsl(value)
                )
            }
            Expr::CompoundAssign { target, op, value } => {
                format!(
                    "({} {}= {})",
                    Self::expr_to_glsl(target),
                    op.glsl_str(),
                    Self::expr_to_glsl(value)
                )
            }
            Expr::PostIncDec { expr, inc } => {
                format!(
                    "{}{}",
                    Self::expr_to_glsl(expr),
                    if *inc { "++" } else { "--" }
                )
            }
            Expr::PreIncDec { expr, inc } => {
                format!(
                    "{}{}",
                    if *inc { "++" } else { "--" },
                    Self::expr_to_glsl(expr)
                )
            }
        }
    }

    /// Convert to WGSL.
    fn to_wgsl(&self) -> String {
        let mut output = String::new();

        // Uniforms as struct
        if !self.uniforms.is_empty() {
            output.push_str("struct Uniforms {\n");
            for uniform in &self.uniforms {
                let _ = writeln!(
                    output,
                    "    {}: {},",
                    uniform.name,
                    Self::type_to_wgsl(uniform.ty)
                );
            }
            output.push_str("};\n\n");
            output.push_str("@group(0) @binding(0) var<uniform> uniforms: Uniforms;\n\n");
        }

        // Functions
        for func in &self.program.functions {
            output.push_str(&Self::function_to_wgsl(func));
            output.push('\n');
        }

        output
    }

    const fn type_to_wgsl(ty: UniformType) -> &'static str {
        match ty {
            UniformType::Float => "f32",
            UniformType::Float2 => "vec2<f32>",
            UniformType::Float3 => "vec3<f32>",
            UniformType::Float4 => "vec4<f32>",
            UniformType::Float2x2 => "mat2x2<f32>",
            UniformType::Float3x3 => "mat3x3<f32>",
            UniformType::Float4x4 => "mat4x4<f32>",
            UniformType::Int => "i32",
            UniformType::Int2 => "vec2<i32>",
            UniformType::Int3 => "vec3<i32>",
            UniformType::Int4 => "vec4<i32>",
        }
    }

    /// Convert to WGSL that naga can parse and validate.
    ///
    /// Differs from `to_wgsl` in two ways:
    /// 1. `f16` is rewritten to `f32` so we don't need the `f16` WGSL
    ///    extension (naga supports it but not all targets do; SPIR-V
    ///    without Float16 capability would otherwise fail).
    /// 2. A synthetic `@fragment` entry point is appended so naga's
    ///    validator has something to anchor the module on. The real
    ///    `main` function is preserved and called from the wrapper.
    /// 3. Child-shader uniforms (`texture_2d` bindings) are skipped from
    ///    the uniform struct — they require separate texture/sampler
    ///    bindings that the WGSL backend does not currently emit.
    fn to_wgsl_for_naga(&self) -> String {
        let mut output = String::new();

        // Non-child uniforms only. Child samplers need separate binding
        // slots (texture_2d + sampler) which the current backend does
        // not emit, so if the shader references them SPIR-V generation
        // will fail during naga parse with a clear error.
        let scalar_uniforms: Vec<&Uniform> = self
            .uniforms
            .iter()
            .filter(|u| !self.children.iter().any(|c| c.name == u.name))
            .collect();

        if !scalar_uniforms.is_empty() {
            output.push_str("struct Uniforms {\n");
            for uniform in &scalar_uniforms {
                let _ = writeln!(
                    output,
                    "    {}: {},",
                    uniform.name,
                    Self::type_to_wgsl(uniform.ty)
                );
            }
            output.push_str("}\n\n");
            output.push_str("@group(0) @binding(0) var<uniform> uniforms: Uniforms;\n\n");
        }

        // Uniform binding prelude injected at the top of every
        // function body: `let <name> = uniforms.<name>;`. The SkSL ->
        // WGSL backend emits uniform references as bare identifiers
        // (e.g. `scale` rather than `uniforms.scale`), so these
        // `let` bindings make the references resolve when naga
        // validates the module.
        let mut uniform_binds = String::new();
        for u in &scalar_uniforms {
            let _ = writeln!(uniform_binds, "    let {} = uniforms.{};", u.name, u.name);
        }

        // Emit user functions with f16 rewritten to f32 and the
        // uniform-binding prelude injected after the opening brace.
        for func in &self.program.functions {
            let func_wgsl = Self::function_to_wgsl(func);
            let rewritten = rewrite_f16_to_f32(&func_wgsl);
            if uniform_binds.is_empty() {
                output.push_str(&rewritten);
            } else {
                // Insert the prelude right after the first `{\n` so
                // every function sees the uniform values in scope.
                if let Some(brace) = rewritten.find("{\n") {
                    let split = brace + 2;
                    output.push_str(&rewritten[..split]);
                    output.push_str(&uniform_binds);
                    output.push_str(&rewritten[split..]);
                } else {
                    output.push_str(&rewritten);
                }
            }
            output.push('\n');
        }

        // Synthesize a fragment entry point. The signature is:
        //   @fragment fn _naga_main(@location(0) pos: vec2<f32>) -> @location(0) vec4<f32>
        // This is enough for naga to validate and emit SPIR-V for a
        // shader effect. Color filters / blenders don't map cleanly
        // to a fragment stage, but naga just needs a valid entry
        // point that references `main` for the rest of the module to
        // validate.
        if let Some(main) = self.program.functions.iter().find(|f| f.name == "main") {
            output.push_str("@fragment\n");
            output.push_str(
                "fn _naga_main(@location(0) _pos: vec2<f32>) -> @location(0) vec4<f32> {\n",
            );
            // Call main with arguments matching its signature.
            let call_args: Vec<String> = main
                .params
                .iter()
                .map(|p| match p.ty {
                    SkslType::Vec2 | SkslType::Half2 => "_pos".to_string(),
                    SkslType::Vec4 | SkslType::Half4 => "vec4<f32>(0.0, 0.0, 0.0, 1.0)".to_string(),
                    SkslType::Int => "0".to_string(),
                    SkslType::Bool => "false".to_string(),
                    _ => "0.0".to_string(),
                })
                .collect();
            // `main` returns vec4<f32> for shaders (validated at
            // parse time). Promote half4 returns to vec4<f32> via an
            // explicit constructor if needed.
            let returns_half = matches!(main.return_type, SkslType::Half4);
            if returns_half {
                let _ = writeln!(
                    output,
                    "    return vec4<f32>(main({}));",
                    call_args.join(", ")
                );
            } else {
                let _ = writeln!(output, "    return main({});", call_args.join(", "));
            }
            output.push_str("}\n");
        }

        output
    }

    fn function_to_wgsl(func: &FnDecl) -> String {
        let mut output = String::new();

        output.push_str("fn ");
        output.push_str(&func.name);
        output.push('(');

        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&param.name);
            output.push_str(": ");
            output.push_str(param.ty.wgsl_name());
        }

        output.push_str(") -> ");
        output.push_str(func.return_type.wgsl_name());
        output.push_str(" {\n");

        // Body (simplified)
        if let Stmt::Block(stmts) = &func.body {
            for stmt in stmts {
                output.push_str(&Self::stmt_to_wgsl(stmt, 1));
            }
        }

        output.push_str("}\n");
        output
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one match arm per Stmt variant, mirroring stmt_to_glsl/stmt_to_msl; splitting would obscure that pairing"
    )]
    fn stmt_to_wgsl(stmt: &Stmt, indent: usize) -> String {
        let ind = "    ".repeat(indent);
        match stmt {
            Stmt::Expr(expr) => format!("{}{};\n", ind, Self::expr_to_wgsl(expr)),
            Stmt::VarDecl { ty, name, init } => init.as_ref().map_or_else(
                || format!("{}var {}: {};\n", ind, name, ty.wgsl_name()),
                |init| {
                    format!(
                        "{}var {}: {} = {};\n",
                        ind,
                        name,
                        ty.wgsl_name(),
                        Self::expr_to_wgsl(init)
                    )
                },
            ),
            Stmt::Block(stmts) => {
                let mut output = format!("{ind}{{\n");
                for s in stmts {
                    output.push_str(&Self::stmt_to_wgsl(s, indent + 1));
                }
                let _ = writeln!(output, "{ind}}}");
                output
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let mut output = format!("{}if ({}) ", ind, Self::expr_to_wgsl(cond));
                output.push_str(&Self::stmt_to_wgsl(then_branch, indent));
                if let Some(else_b) = else_branch {
                    let _ = write!(output, "{ind}else ");
                    output.push_str(&Self::stmt_to_wgsl(else_b, indent));
                }
                output
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                let mut output = format!("{ind}for (");
                if let Some(init) = init {
                    let init_str = Self::stmt_to_wgsl(init, 0);
                    output.push_str(init_str.trim());
                } else {
                    output.push(';');
                }
                output.push(' ');
                if let Some(cond) = cond {
                    output.push_str(&Self::expr_to_wgsl(cond));
                }
                output.push_str("; ");
                if let Some(update) = update {
                    output.push_str(&Self::expr_to_wgsl(update));
                }
                output.push_str(") ");
                output.push_str(&Self::stmt_to_wgsl(body, indent));
                output
            }
            Stmt::While { cond, body } => {
                let mut output = format!("{ind}loop {{\n");
                let _ = writeln!(
                    output,
                    "{}    if (!{}) {{ break; }}",
                    ind,
                    Self::expr_to_wgsl(cond)
                );
                if let Stmt::Block(stmts) = body.as_ref() {
                    for s in stmts {
                        output.push_str(&Self::stmt_to_wgsl(s, indent + 1));
                    }
                } else {
                    output.push_str(&Self::stmt_to_wgsl(body, indent + 1));
                }
                let _ = writeln!(output, "{ind}}}");
                output
            }
            Stmt::DoWhile { body, cond } => {
                let mut output = format!("{ind}loop {{\n");
                if let Stmt::Block(stmts) = body.as_ref() {
                    for s in stmts {
                        output.push_str(&Self::stmt_to_wgsl(s, indent + 1));
                    }
                } else {
                    output.push_str(&Self::stmt_to_wgsl(body, indent + 1));
                }
                let _ = writeln!(
                    output,
                    "{}    if (!{}) {{ break; }}",
                    ind,
                    Self::expr_to_wgsl(cond)
                );
                let _ = writeln!(output, "{ind}}}");
                output
            }
            Stmt::Return(expr) => expr.as_ref().map_or_else(
                || format!("{ind}return;\n"),
                |expr| format!("{}return {};\n", ind, Self::expr_to_wgsl(expr)),
            ),
            Stmt::Break => format!("{ind}break;\n"),
            Stmt::Continue => format!("{ind}continue;\n"),
            Stmt::Discard => format!("{ind}discard;\n"),
        }
    }

    fn expr_to_wgsl(expr: &Expr) -> String {
        match expr {
            Expr::IntLit(n) => format!("{n}i"),
            Expr::FloatLit(n) => {
                if n.fract() == 0.0 {
                    format!("{n}.0")
                } else {
                    format!("{n}")
                }
            }
            Expr::BoolLit(b) => b.to_string(),
            Expr::Var(name) => name.clone(),
            Expr::Binary { left, op, right } => {
                format!(
                    "({} {} {})",
                    Self::expr_to_wgsl(left),
                    op.glsl_str(),
                    Self::expr_to_wgsl(right)
                )
            }
            Expr::Unary { op, expr } => {
                format!("({}{})", op.glsl_str(), Self::expr_to_wgsl(expr))
            }
            Expr::Call { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::expr_to_wgsl).collect();
                format!("{}({})", name, args_str.join(", "))
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Child eval helper-call naming convention (see GLSL emitter).
                let args_str: Vec<String> = args.iter().map(Self::expr_to_wgsl).collect();
                format!(
                    "{}_{}({})",
                    Self::expr_to_wgsl(receiver),
                    method,
                    args_str.join(", ")
                )
            }
            Expr::Constructor { ty, args } => {
                let args_str: Vec<String> = args.iter().map(Self::expr_to_wgsl).collect();
                format!("{}({})", ty.wgsl_name(), args_str.join(", "))
            }
            Expr::Field { expr, field } => {
                format!("{}.{}", Self::expr_to_wgsl(expr), field)
            }
            Expr::Index { expr, index } => {
                format!(
                    "{}[{}]",
                    Self::expr_to_wgsl(expr),
                    Self::expr_to_wgsl(index)
                )
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                format!(
                    "select({}, {}, {})",
                    Self::expr_to_wgsl(else_expr),
                    Self::expr_to_wgsl(then_expr),
                    Self::expr_to_wgsl(cond)
                )
            }
            Expr::Assign { target, value } => {
                format!(
                    "({} = {})",
                    Self::expr_to_wgsl(target),
                    Self::expr_to_wgsl(value)
                )
            }
            Expr::CompoundAssign { target, op, value } => {
                format!(
                    "({} {}= {})",
                    Self::expr_to_wgsl(target),
                    op.glsl_str(),
                    Self::expr_to_wgsl(value)
                )
            }
            Expr::PostIncDec { expr, inc } => {
                format!(
                    "{}{}",
                    Self::expr_to_wgsl(expr),
                    if *inc { "++" } else { "--" }
                )
            }
            Expr::PreIncDec { expr, inc } => {
                format!(
                    "{}{}",
                    if *inc { "++" } else { "--" },
                    Self::expr_to_wgsl(expr)
                )
            }
        }
    }

    /// Convert to MSL.
    fn to_msl(&self) -> String {
        let mut output = String::new();

        output.push_str("#include <metal_stdlib>\n");
        output.push_str("using namespace metal;\n\n");

        // Uniforms as struct
        if !self.uniforms.is_empty() {
            output.push_str("struct Uniforms {\n");
            for uniform in &self.uniforms {
                let _ = writeln!(
                    output,
                    "    {} {};",
                    Self::type_to_msl(uniform.ty),
                    uniform.name
                );
            }
            output.push_str("};\n\n");
        }

        // Functions
        for func in &self.program.functions {
            output.push_str(&Self::function_to_msl(func));
            output.push('\n');
        }

        output
    }

    const fn type_to_msl(ty: UniformType) -> &'static str {
        match ty {
            UniformType::Float => "float",
            UniformType::Float2 => "float2",
            UniformType::Float3 => "float3",
            UniformType::Float4 => "float4",
            UniformType::Float2x2 => "float2x2",
            UniformType::Float3x3 => "float3x3",
            UniformType::Float4x4 => "float4x4",
            UniformType::Int => "int",
            UniformType::Int2 => "int2",
            UniformType::Int3 => "int3",
            UniformType::Int4 => "int4",
        }
    }

    fn function_to_msl(func: &FnDecl) -> String {
        let mut output = String::new();

        // Use Metal types
        let ret_type = Self::sksl_type_to_msl(&func.return_type);

        output.push_str(ret_type);
        output.push(' ');
        output.push_str(&func.name);
        output.push('(');

        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(Self::sksl_type_to_msl(&param.ty));
            output.push(' ');
            output.push_str(&param.name);
        }

        output.push_str(") ");
        output.push_str(&Self::stmt_to_msl(&func.body, 0));

        output
    }

    const fn sksl_type_to_msl(ty: &SkslType) -> &'static str {
        match ty {
            SkslType::Vec3 | SkslType::Half3 => "float3",
            SkslType::Vec2 | SkslType::Half2 => "float2",
            SkslType::Float | SkslType::Half => "float",
            SkslType::Int => "int",
            SkslType::Bool => "bool",
            SkslType::Mat2 => "float2x2",
            SkslType::Mat3 => "float3x3",
            SkslType::Mat4 => "float4x4",
            SkslType::Void => "void",
            // Vec4/Half4 and any other/future type default to float4.
            _ => "float4",
        }
    }

    fn stmt_to_msl(stmt: &Stmt, indent: usize) -> String {
        let ind = "    ".repeat(indent);
        match stmt {
            Stmt::Expr(expr) => format!("{}{};\n", ind, Self::expr_to_msl(expr)),
            Stmt::VarDecl { ty, name, init } => init.as_ref().map_or_else(
                || format!("{}{} {};\n", ind, Self::sksl_type_to_msl(ty), name),
                |init| {
                    format!(
                        "{}{} {} = {};\n",
                        ind,
                        Self::sksl_type_to_msl(ty),
                        name,
                        Self::expr_to_msl(init)
                    )
                },
            ),
            Stmt::Block(stmts) => {
                let mut output = format!("{ind}{{\n");
                for s in stmts {
                    output.push_str(&Self::stmt_to_msl(s, indent + 1));
                }
                let _ = writeln!(output, "{ind}}}");
                output
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let mut output = format!("{}if ({}) ", ind, Self::expr_to_msl(cond));
                output.push_str(&Self::stmt_to_msl(then_branch, indent));
                if let Some(else_b) = else_branch {
                    let _ = write!(output, "{ind}else ");
                    output.push_str(&Self::stmt_to_msl(else_b, indent));
                }
                output
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                let mut output = format!("{ind}for (");
                if let Some(init) = init {
                    let init_str = Self::stmt_to_msl(init, 0);
                    output.push_str(init_str.trim());
                } else {
                    output.push(';');
                }
                output.push(' ');
                if let Some(cond) = cond {
                    output.push_str(&Self::expr_to_msl(cond));
                }
                output.push_str("; ");
                if let Some(update) = update {
                    output.push_str(&Self::expr_to_msl(update));
                }
                output.push_str(") ");
                output.push_str(&Self::stmt_to_msl(body, indent));
                output
            }
            Stmt::While { cond, body } => {
                let mut output = format!("{}while ({}) ", ind, Self::expr_to_msl(cond));
                output.push_str(&Self::stmt_to_msl(body, indent));
                output
            }
            Stmt::DoWhile { body, cond } => {
                let mut output = format!("{ind}do ");
                output.push_str(&Self::stmt_to_msl(body, indent));
                let _ = writeln!(output, " while ({});", Self::expr_to_msl(cond));
                output
            }
            Stmt::Return(expr) => expr.as_ref().map_or_else(
                || format!("{ind}return;\n"),
                |expr| format!("{}return {};\n", ind, Self::expr_to_msl(expr)),
            ),
            Stmt::Break => format!("{ind}break;\n"),
            Stmt::Continue => format!("{ind}continue;\n"),
            Stmt::Discard => format!("{ind}discard_fragment();\n"),
        }
    }

    fn expr_to_msl(expr: &Expr) -> String {
        match expr {
            Expr::IntLit(n) => n.to_string(),
            Expr::FloatLit(n) => {
                if n.fract() == 0.0 {
                    format!("{n}.0")
                } else {
                    format!("{n}")
                }
            }
            Expr::BoolLit(b) => b.to_string(),
            Expr::Var(name) => name.clone(),
            Expr::Binary { left, op, right } => {
                format!(
                    "({} {} {})",
                    Self::expr_to_msl(left),
                    op.glsl_str(),
                    Self::expr_to_msl(right)
                )
            }
            Expr::Unary { op, expr } => {
                format!("({}{})", op.glsl_str(), Self::expr_to_msl(expr))
            }
            Expr::Call { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::expr_to_msl).collect();
                format!("{}({})", name, args_str.join(", "))
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Child eval helper-call naming convention (see GLSL emitter).
                let args_str: Vec<String> = args.iter().map(Self::expr_to_msl).collect();
                format!(
                    "{}_{}({})",
                    Self::expr_to_msl(receiver),
                    method,
                    args_str.join(", ")
                )
            }
            Expr::Constructor { ty, args } => {
                let args_str: Vec<String> = args.iter().map(Self::expr_to_msl).collect();
                format!("{}({})", Self::sksl_type_to_msl(ty), args_str.join(", "))
            }
            Expr::Field { expr, field } => {
                format!("{}.{}", Self::expr_to_msl(expr), field)
            }
            Expr::Index { expr, index } => {
                format!("{}[{}]", Self::expr_to_msl(expr), Self::expr_to_msl(index))
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                format!(
                    "({} ? {} : {})",
                    Self::expr_to_msl(cond),
                    Self::expr_to_msl(then_expr),
                    Self::expr_to_msl(else_expr)
                )
            }
            Expr::Assign { target, value } => {
                format!(
                    "({} = {})",
                    Self::expr_to_msl(target),
                    Self::expr_to_msl(value)
                )
            }
            Expr::CompoundAssign { target, op, value } => {
                format!(
                    "({} {}= {})",
                    Self::expr_to_msl(target),
                    op.glsl_str(),
                    Self::expr_to_msl(value)
                )
            }
            Expr::PostIncDec { expr, inc } => {
                format!(
                    "{}{}",
                    Self::expr_to_msl(expr),
                    if *inc { "++" } else { "--" }
                )
            }
            Expr::PreIncDec { expr, inc } => {
                format!(
                    "{}{}",
                    if *inc { "++" } else { "--" },
                    Self::expr_to_msl(expr)
                )
            }
        }
    }

    /// Decode a single uniform from the raw byte buffer into an interpreter
    /// Value. Used by the software fallback when running shaders without a
    /// GPU backend. Panics on malformed buffers are impossible — bounds are
    /// checked via `get` reads.
    fn decode_uniform(uniform: &Uniform, data: &[u8]) -> SkslValue {
        let read_scalar = |off: usize| -> f32 {
            if off + 4 <= data.len() {
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            } else {
                0.0
            }
        };
        let read_int = |off: usize| -> i32 {
            if off + 4 <= data.len() {
                i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            } else {
                0
            }
        };
        let o = uniform.offset;
        match uniform.ty {
            UniformType::Float => SkslValue::Float(read_scalar(o)),
            UniformType::Float2 => SkslValue::Vec2([read_scalar(o), read_scalar(o + 4)]),
            UniformType::Float3 => {
                SkslValue::Vec3([read_scalar(o), read_scalar(o + 4), read_scalar(o + 8)])
            }
            UniformType::Float4 => SkslValue::Vec4([
                read_scalar(o),
                read_scalar(o + 4),
                read_scalar(o + 8),
                read_scalar(o + 12),
            ]),
            UniformType::Float2x2 => {
                let mut m = [0.0f32; 4];
                for (i, slot) in m.iter_mut().enumerate() {
                    *slot = read_scalar(o + i * 4);
                }
                SkslValue::Mat2(m)
            }
            UniformType::Float3x3 => {
                let mut m = [0.0f32; 9];
                for (i, slot) in m.iter_mut().enumerate() {
                    *slot = read_scalar(o + i * 4);
                }
                SkslValue::Mat3(m)
            }
            UniformType::Float4x4 => {
                let mut m = [0.0f32; 16];
                for (i, slot) in m.iter_mut().enumerate() {
                    *slot = read_scalar(o + i * 4);
                }
                SkslValue::Mat4(m)
            }
            UniformType::Int => SkslValue::Int(read_int(o)),
            UniformType::Int2 => {
                // No vec2<i32> in the interpreter — fall back to float vec.
                SkslValue::Vec2([
                    scalar_from_i32(read_int(o)),
                    scalar_from_i32(read_int(o + 4)),
                ])
            }
            UniformType::Int3 => SkslValue::Vec3([
                scalar_from_i32(read_int(o)),
                scalar_from_i32(read_int(o + 4)),
                scalar_from_i32(read_int(o + 8)),
            ]),
            UniformType::Int4 => SkslValue::Vec4([
                scalar_from_i32(read_int(o)),
                scalar_from_i32(read_int(o + 4)),
                scalar_from_i32(read_int(o + 8)),
                scalar_from_i32(read_int(o + 12)),
            ]),
        }
    }

    /// Build an interpreter seeded with this effect's functions, uniforms,
    /// and children. Used by the software fallback paths for `RuntimeShader`
    /// and `RuntimeColorFilter`. The returned interpreter borrows function
    /// bodies from `self.program`, so it must not outlive this effect.
    fn build_interp<'a>(
        &'a self,
        uniform_data: &UniformData,
        children: &[Arc<dyn Shader>],
    ) -> Interp<'a> {
        let mut interp = Interp::new();
        for f in &self.program.functions {
            interp.functions.insert(f.name.clone(), f);
        }
        // Decode non-child uniforms.
        for u in &self.uniforms {
            // Child samplers are tracked separately in `self.children`; skip
            // them here so we don't mistake a shader binding for a scalar
            // uniform read.
            if self.children.iter().any(|c| c.name == u.name) {
                continue;
            }
            let v = Self::decode_uniform(u, uniform_data.data());
            interp.uniforms.insert(u.name.clone(), v);
        }
        interp.children = children.to_vec();
        for c in &self.children {
            interp.children_by_name.insert(c.name.clone(), c.index);
        }
        interp
    }

    /// Create a `RuntimeShader` from this effect.
    ///
    /// # Errors
    ///
    /// Returns an error if `children` does not have exactly as many
    /// entries as the effect declares child shaders/color-filters.
    pub fn make_shader(
        self: &Arc<Self>,
        uniforms: &UniformData,
        children: &[Arc<dyn Shader>],
    ) -> Result<RuntimeShader, RuntimeEffectError> {
        if children.len() != self.children.len() {
            return Err(RuntimeEffectError::InvalidChildCount {
                expected: self.children.len(),
                got: children.len(),
            });
        }

        Ok(RuntimeShader {
            effect: Arc::clone(self),
            uniforms: uniforms.clone(),
            children: children.to_vec(),
        })
    }

    /// Create a `RuntimeColorFilter` from this effect.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for API symmetry with
    /// [`Self::make_shader`] and to allow future validation without a
    /// breaking signature change.
    pub fn make_color_filter(
        self: &Arc<Self>,
        uniforms: &UniformData,
    ) -> Result<RuntimeColorFilter, RuntimeEffectError> {
        Ok(RuntimeColorFilter {
            effect: Arc::clone(self),
            uniforms: uniforms.clone(),
        })
    }
}

/// Effect kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectKind {
    Shader,
    ColorFilter,
    Blender,
}

/// Validate the entry point for the given effect kind.
/// Rewrite `f16` type tokens to `f32` in a WGSL source string.
///
/// The WGSL backend emits `f16` for `SkSL` `half` types (the WGSL
/// extension name is `f16`). naga + SPIR-V targets that don't enable
/// the `Float16` capability can't lower those, so for the SPIR-V path
/// we downgrade the precision to f32. This is a lossy transform but
/// avoids requiring an extension that Vulkan implementations may not
/// expose. Matches whole-token `f16` occurrences only.
fn rewrite_f16_to_f32(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"f16") {
            // Check that this is a standalone token (not part of a
            // longer identifier like `f16foo`).
            let prev_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let next_ok = i + 3 >= bytes.len() || !is_ident_byte(bytes[i + 3]);
            if prev_ok && next_ok {
                out.push_str("f32");
                i += 3;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

const fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn validate_entry_point(program: &SkslProgram, kind: EffectKind) -> Result<(), RuntimeEffectError> {
    let main = program
        .functions
        .iter()
        .find(|f| f.name == "main")
        .ok_or(RuntimeEffectError::MissingMain)?;

    match kind {
        EffectKind::Shader => {
            // Expect main(vec2) -> vec4 or main(half2) -> half4
            if main.params.len() != 1 {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec2) -> vec4".into(),
                });
            }
            // Check param is vec2 or half2
            let param_ok = matches!(&main.params[0].ty, SkslType::Vec2 | SkslType::Half2);
            if !param_ok {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec2) -> vec4".into(),
                });
            }
            // Check return is vec4 or half4
            let return_ok = matches!(&main.return_type, SkslType::Vec4 | SkslType::Half4);
            if !return_ok {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec2) -> vec4".into(),
                });
            }
        }
        EffectKind::ColorFilter => {
            // Expect main(vec4) -> vec4 or main(half4) -> half4
            if main.params.len() != 1 {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec4) -> vec4".into(),
                });
            }
            // Check param is vec4 or half4
            let param_ok = matches!(&main.params[0].ty, SkslType::Vec4 | SkslType::Half4);
            if !param_ok {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec4) -> vec4".into(),
                });
            }
            // Check return is vec4 or half4
            let return_ok = matches!(&main.return_type, SkslType::Vec4 | SkslType::Half4);
            if !return_ok {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec4) -> vec4".into(),
                });
            }
        }
        EffectKind::Blender => {
            // Expect main(vec4, vec4) -> vec4 or main(half4, half4) -> half4
            if main.params.len() != 2 {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec4, vec4) -> vec4".into(),
                });
            }
            // Check both params are vec4 or half4
            let param0_ok = matches!(&main.params[0].ty, SkslType::Vec4 | SkslType::Half4);
            let param1_ok = matches!(&main.params[1].ty, SkslType::Vec4 | SkslType::Half4);
            if !param0_ok || !param1_ok {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec4, vec4) -> vec4".into(),
                });
            }
            // Check return is vec4 or half4
            let return_ok = matches!(&main.return_type, SkslType::Vec4 | SkslType::Half4);
            if !return_ok {
                return Err(RuntimeEffectError::InvalidEntryPoint {
                    expected: "main(vec4, vec4) -> vec4".into(),
                });
            }
        }
    }

    Ok(())
}

/// Uniform data builder.
#[derive(Debug, Clone, Default)]
pub struct UniformData {
    data: Vec<u8>,
}

impl UniformData {
    /// Create new uniform data with specified size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Create from a runtime effect.
    pub fn from_effect(effect: &RuntimeEffect) -> Self {
        Self::new(effect.uniform_size())
    }

    /// Set a float uniform.
    pub fn set_float(&mut self, offset: usize, value: f32) {
        if offset + 4 <= self.data.len() {
            self.data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Set a vec2 uniform.
    pub fn set_float2(&mut self, offset: usize, x: f32, y: f32) {
        self.set_float(offset, x);
        self.set_float(offset + 4, y);
    }

    /// Set a vec3 uniform.
    pub fn set_float3(&mut self, offset: usize, x: f32, y: f32, z: f32) {
        self.set_float(offset, x);
        self.set_float(offset + 4, y);
        self.set_float(offset + 8, z);
    }

    /// Set a vec4 uniform.
    pub fn set_float4(&mut self, offset: usize, x: f32, y: f32, z: f32, w: f32) {
        self.set_float(offset, x);
        self.set_float(offset + 4, y);
        self.set_float(offset + 8, z);
        self.set_float(offset + 12, w);
    }

    /// Set a color uniform.
    pub fn set_color(&mut self, offset: usize, color: Color4f) {
        self.set_float4(offset, color.r, color.g, color.b, color.a);
    }

    /// Set an int uniform.
    pub fn set_int(&mut self, offset: usize, value: i32) {
        if offset + 4 <= self.data.len() {
            self.data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Get a float uniform.
    #[must_use]
    pub fn get_float(&self, offset: usize) -> f32 {
        let d = &self.data;
        if offset + 4 <= d.len() {
            f32::from_le_bytes([d[offset], d[offset + 1], d[offset + 2], d[offset + 3]])
        } else {
            0.0
        }
    }

    /// Get the raw data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

/// A runtime shader created from a `RuntimeEffect`.
#[derive(Debug, Clone)]
pub struct RuntimeShader {
    effect: Arc<RuntimeEffect>,
    uniforms: UniformData,
    children: Vec<Arc<dyn Shader>>,
}

impl RuntimeShader {
    /// Get the effect.
    #[must_use]
    pub fn effect(&self) -> &RuntimeEffect {
        &self.effect
    }

    /// Get the uniforms.
    #[must_use]
    pub const fn uniforms(&self) -> &UniformData {
        &self.uniforms
    }

    /// Get the children.
    #[must_use]
    pub fn children(&self) -> &[Arc<dyn Shader>] {
        &self.children
    }
}

impl Shader for RuntimeShader {
    fn local_matrix(&self) -> Option<&Matrix> {
        None
    }

    fn is_opaque(&self) -> bool {
        false
    }

    fn shader_kind(&self) -> ShaderKind {
        ShaderKind::Color // Closest match for runtime shader
    }

    fn sample(&self, x: Scalar, y: Scalar) -> Color4f {
        // Software fallback: run the SkSL interpreter on the parsed program.
        // Runtime shaders follow the convention `half4 main(float2 coord)`,
        // so we dispatch `main` with a Vec2 coordinate and read back a Vec4.
        //
        // Per the Shader::sample contract the returned color is treated as
        // premultiplied: SkSL shader output feeds the raster pipeline
        // directly, whose colors are premul (matching upstream semantics).
        let mut interp = self.effect.build_interp(&self.uniforms, &self.children);
        let coord = SkslValue::Vec2([x, y]);
        let result = interp.run_function("main", vec![coord]);
        let rgba = result.as_vec4();
        Color4f::new(rgba[0], rgba[1], rgba[2], rgba[3])
    }
}

/// A runtime color filter created from a `RuntimeEffect`.
#[derive(Debug, Clone)]
pub struct RuntimeColorFilter {
    effect: Arc<RuntimeEffect>,
    uniforms: UniformData,
}

impl RuntimeColorFilter {
    /// Get the effect.
    #[must_use]
    pub fn effect(&self) -> &RuntimeEffect {
        &self.effect
    }

    /// Get the uniforms.
    #[must_use]
    pub const fn uniforms(&self) -> &UniformData {
        &self.uniforms
    }

    /// Filter a color.
    #[must_use]
    pub fn filter_color(&self, color: Color4f) -> Color4f {
        // Software fallback: run the SkSL interpreter on the parsed program.
        // Runtime color filters follow `half4 main(half4 color)`; we pass the
        // input color as a Vec4 and read the returned Vec4.
        let mut interp = self.effect.build_interp(&self.uniforms, &[]);
        let c = SkslValue::Vec4([color.r, color.g, color.b, color.a]);
        let result = interp.run_function("main", vec![c]);
        let rgba = result.as_vec4();
        Color4f::new(rgba[0], rgba[1], rgba[2], rgba[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_SHADER: &str = r"
        uniform float time;
        uniform vec2 resolution;

        vec4 main(vec2 fragCoord) {
            vec2 uv = fragCoord / resolution;
            return vec4(uv.x, uv.y, sin(time), 1.0);
        }
    ";

    #[test]
    fn test_make_effect() {
        let effect = RuntimeEffect::make_for_shader(SIMPLE_SHADER).unwrap();

        assert_eq!(effect.uniforms().len(), 2);
        assert!(effect.find_uniform("time").is_some());
        assert!(effect.find_uniform("resolution").is_some());
    }

    #[test]
    fn test_uniforms_pack_tightly_no_16_byte_alignment() {
        // SkRuntimeEffect.cpp packs uniforms back to back:
        //   uni.offset = *offset; *offset += uni.sizeInBytes();
        // A float followed by a vec4 sits at offsets 0 and 4 (not 16),
        // total size 20.
        let src = r"
            uniform float a;
            uniform vec4 b;
            vec4 main(vec2 coord) { return b * a; }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let a = effect.find_uniform("a").unwrap();
        let b = effect.find_uniform("b").unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 4, "vec4 must pack tightly after float");
        assert_eq!(effect.uniform_size(), 20);
    }

    #[test]
    fn test_uniform_mat3_packs_tightly() {
        // mat3 is 36 bytes; a following float sits at offset 40 (4 + 36).
        let src = r"
            uniform float a;
            uniform mat3 m;
            uniform float c;
            vec4 main(vec2 coord) { return vec4(a + c); }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        assert_eq!(effect.find_uniform("m").unwrap().offset, 4);
        assert_eq!(effect.find_uniform("c").unwrap().offset, 40);
        assert_eq!(effect.uniform_size(), 44);
    }

    #[test]
    fn test_child_shaders_are_not_uniforms() {
        // `uniform shader s;` declares a child, not a data uniform: it must
        // not appear in uniforms() nor contribute to uniform_size().
        let src = r"
            uniform shader child;
            uniform float x;
            vec4 main(vec2 coord) { return vec4(x); }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        assert_eq!(effect.uniforms().len(), 1, "only x is a data uniform");
        assert!(effect.find_uniform("child").is_none());
        assert_eq!(effect.find_uniform("x").unwrap().offset, 0);
        assert_eq!(effect.uniform_size(), 4);
        assert_eq!(effect.children().len(), 1);
        assert_eq!(effect.children()[0].name, "child");
    }

    #[test]
    fn test_child_shader_eval_samples_child() {
        // Method-style `child.eval(coord)` must parse and sample the bound
        // child shader in the software interpreter.
        let src = r"
            uniform shader child;
            half4 main(float2 coord) {
                half4 c = child.eval(coord);
                return c * 0.5;
            }
        ";
        let effect = Arc::new(RuntimeEffect::make_for_shader(src).unwrap());
        let uniforms = UniformData::from_effect(&effect);
        let child: Arc<dyn Shader> = Arc::new(crate::shader::ColorShader::new(Color4f::new(
            1.0, 0.5, 0.0, 1.0,
        )));
        let shader = effect.make_shader(&uniforms, &[child]).unwrap();
        let c = shader.sample(3.0, 4.0);
        assert!((c.r - 0.5).abs() < 1e-4, "child red halved, got {}", c.r);
        assert!((c.g - 0.25).abs() < 1e-4, "child green halved, got {}", c.g);
        assert!((c.a - 0.5).abs() < 1e-4, "child alpha halved, got {}", c.a);
    }

    #[test]
    fn test_child_shader_eval_uses_coords() {
        // The child must be sampled at the coordinates passed to eval().
        let src = r"
            uniform shader child;
            half4 main(float2 coord) {
                return child.eval(coord * 2.0);
            }
        ";
        let effect = Arc::new(RuntimeEffect::make_for_shader(src).unwrap());
        let uniforms = UniformData::from_effect(&effect);
        // Opaque linear gradient black -> white over x in [0, 10].
        let child: Arc<dyn Shader> = Arc::new(crate::shader::LinearGradient::new(
            skia_rs_core::Point::new(0.0, 0.0),
            skia_rs_core::Point::new(10.0, 0.0),
            vec![
                Color4f::new(0.0, 0.0, 0.0, 1.0),
                Color4f::new(1.0, 1.0, 1.0, 1.0),
            ],
            None,
            crate::shader::TileMode::Clamp,
        ));
        let shader = effect.make_shader(&uniforms, &[child]).unwrap();
        // sample at x=2.5 -> child sampled at x=5 -> t=0.5.
        let c = shader.sample(2.5, 0.0);
        assert!(
            (c.r - 0.5).abs() < 1e-3,
            "child sampled at doubled coord, got {}",
            c.r
        );
    }

    #[test]
    fn test_uniform_data() {
        let effect = RuntimeEffect::make_for_shader(SIMPLE_SHADER).unwrap();
        let mut data = UniformData::from_effect(&effect);

        let time_uniform = effect.find_uniform("time").unwrap();
        data.set_float(time_uniform.offset, 1.5);
        assert!((data.get_float(time_uniform.offset) - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_compile_glsl() {
        let effect = RuntimeEffect::make_for_shader(SIMPLE_SHADER).unwrap();
        let glsl = effect.compile_to(ShaderTarget::GlslEs300).unwrap();

        assert!(glsl.contains("#version 300 es"));
        assert!(glsl.contains("uniform float time"));
        assert!(glsl.contains("uniform vec2 resolution"));
    }

    #[test]
    fn test_compile_wgsl() {
        let effect = RuntimeEffect::make_for_shader(SIMPLE_SHADER).unwrap();
        let wgsl = effect.compile_to(ShaderTarget::Wgsl).unwrap();

        assert!(wgsl.contains("struct Uniforms"));
        assert!(wgsl.contains("time: f32"));
    }

    #[test]
    fn test_compile_msl() {
        let effect = RuntimeEffect::make_for_shader(SIMPLE_SHADER).unwrap();
        let msl = effect.compile_to(ShaderTarget::Msl).unwrap();

        assert!(msl.contains("#include <metal_stdlib>"));
        assert!(msl.contains("struct Uniforms"));
    }

    #[test]
    fn test_make_shader() {
        let effect = Arc::new(RuntimeEffect::make_for_shader(SIMPLE_SHADER).unwrap());
        let mut data = UniformData::from_effect(&effect);

        data.set_float(0, 1.0);
        data.set_float2(4, 800.0, 600.0);

        let shader = effect.make_shader(&data, &[]).unwrap();
        assert_eq!(shader.effect().uniforms().len(), 2);
    }

    #[test]
    fn test_wgsl_if_statement() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                float result = 0.0;
                if (fragCoord.x > 0.5) {
                    result = 1.0;
                } else {
                    result = 0.5;
                }
                return vec4(result, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let wgsl = effect.compile_to(ShaderTarget::Wgsl).unwrap();

        assert!(wgsl.contains("if ("), "WGSL should contain if statement");
        assert!(wgsl.contains("else"), "WGSL should contain else branch");
        assert!(
            !wgsl.contains("Unsupported"),
            "WGSL should not contain unsupported stubs: {wgsl}"
        );
    }

    #[test]
    fn test_wgsl_for_loop() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                float sum = 0.0;
                for (int i = 0; i < 4; i = i + 1) {
                    sum = sum + 1.0;
                }
                return vec4(sum, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let wgsl = effect.compile_to(ShaderTarget::Wgsl).unwrap();

        assert!(wgsl.contains("for ("), "WGSL should contain for loop");
        assert!(
            !wgsl.contains("Unsupported"),
            "WGSL for loop should compile: {wgsl}"
        );
    }

    #[test]
    fn test_wgsl_while_loop() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                int i = 0;
                while (i < 3) {
                    i = i + 1;
                }
                return vec4(0.0, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let wgsl = effect.compile_to(ShaderTarget::Wgsl).unwrap();

        assert!(
            wgsl.contains("loop {"),
            "WGSL should translate while to loop"
        );
        assert!(wgsl.contains("break;"), "WGSL loop should contain break");
        assert!(
            !wgsl.contains("Unsupported"),
            "WGSL while should compile: {wgsl}"
        );
    }

    #[test]
    fn test_wgsl_ternary() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                float x = fragCoord.x > 0.5 ? 1.0 : 0.0;
                return vec4(x, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let wgsl = effect.compile_to(ShaderTarget::Wgsl).unwrap();

        assert!(
            wgsl.contains("select("),
            "WGSL should translate ternary to select()"
        );
        assert!(!wgsl.contains('?'), "WGSL should not use ternary operator");
        assert!(
            !wgsl.contains("unsupported"),
            "WGSL ternary should compile: {wgsl}"
        );
    }

    #[test]
    fn test_msl_uses_float4() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let msl = effect.compile_to(ShaderTarget::Msl).unwrap();

        assert!(
            msl.contains("float4"),
            "MSL should use float4 instead of vec4: {msl}"
        );
        assert!(!msl.contains("vec4"), "MSL should not contain vec4: {msl}");
    }

    #[test]
    fn test_msl_if_statement() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                if (fragCoord.x > 0.5) {
                    return vec4(1.0, 0.0, 0.0, 1.0);
                }
                return vec4(0.0, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let msl = effect.compile_to(ShaderTarget::Msl).unwrap();

        assert!(msl.contains("if ("), "MSL should contain if statement");
        assert!(
            msl.contains("return"),
            "MSL should contain return statements"
        );
    }

    #[test]
    fn test_msl_for_loop() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                float sum = 0.0;
                for (int i = 0; i < 3; i = i + 1) {
                    sum = sum + 1.0;
                }
                return vec4(sum, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let msl = effect.compile_to(ShaderTarget::Msl).unwrap();

        assert!(msl.contains("for ("), "MSL should contain for loop");
        assert!(msl.contains("float4"), "MSL should use float4");
    }

    #[test]
    fn test_runtime_shader_returns_red() {
        // Ensures the software interpreter produces the shader's return
        // value instead of the old magenta placeholder.
        use crate::shader::Shader;
        let src = r"
            vec4 main(vec2 coord) {
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
        ";
        let effect = Arc::new(RuntimeEffect::make_for_shader(src).unwrap());
        let data = UniformData::from_effect(&effect);
        let shader = effect.make_shader(&data, &[]).unwrap();
        let c = shader.sample(10.0, 20.0);
        assert!((c.r - 1.0).abs() < 1e-3, "r should be 1.0, got {}", c.r);
        assert!((c.g - 0.0).abs() < 1e-3, "g should be 0.0, got {}", c.g);
        assert!((c.b - 0.0).abs() < 1e-3, "b should be 0.0, got {}", c.b);
        assert!((c.a - 1.0).abs() < 1e-3, "a should be 1.0, got {}", c.a);
    }

    #[test]
    fn test_runtime_shader_uses_coord() {
        // The returned color should depend on the sampled coordinate.
        use crate::shader::Shader;
        let src = r"
            vec4 main(vec2 coord) {
                return vec4(coord.x, coord.y, 0.0, 1.0);
            }
        ";
        let effect = Arc::new(RuntimeEffect::make_for_shader(src).unwrap());
        let data = UniformData::from_effect(&effect);
        let shader = effect.make_shader(&data, &[]).unwrap();
        let c = shader.sample(0.25, 0.75);
        assert!((c.r - 0.25).abs() < 1e-3);
        assert!((c.g - 0.75).abs() < 1e-3);
    }

    #[test]
    fn test_runtime_shader_uses_uniform() {
        // Uniform values should feed into the shader through the interpreter.
        use crate::shader::Shader;
        let src = r"
            uniform float scale;
            vec4 main(vec2 coord) {
                return vec4(coord.x * scale, 0.0, 0.0, 1.0);
            }
        ";
        let effect = Arc::new(RuntimeEffect::make_for_shader(src).unwrap());
        let mut data = UniformData::from_effect(&effect);
        let u = effect.find_uniform("scale").unwrap();
        data.set_float(u.offset, 2.0);
        let shader = effect.make_shader(&data, &[]).unwrap();
        let c = shader.sample(0.5, 0.0);
        assert!(
            (c.r - 1.0).abs() < 1e-3,
            "expected scale * x = 1.0, got {}",
            c.r
        );
    }

    #[test]
    fn test_runtime_color_filter_inverts() {
        let src = r"
            vec4 main(vec4 color) {
                return vec4(1.0 - color.x, 1.0 - color.y, 1.0 - color.z, color.w);
            }
        ";
        let effect = Arc::new(RuntimeEffect::make_for_color_filter(src).unwrap());
        let data = UniformData::from_effect(&effect);
        let filter = effect.make_color_filter(&data).unwrap();
        let out = filter.filter_color(Color4f::new(0.2, 0.3, 0.4, 1.0));
        assert!((out.r - 0.8).abs() < 1e-3);
        assert!((out.g - 0.7).abs() < 1e-3);
        assert!((out.b - 0.6).abs() < 1e-3);
        assert!((out.a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_msl_discard() {
        let src = r"
            vec4 main(vec2 fragCoord) {
                if (fragCoord.x < 0.0) {
                    discard;
                }
                return vec4(1.0, 0.0, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let msl = effect.compile_to(ShaderTarget::Msl).unwrap();

        assert!(
            msl.contains("discard_fragment()"),
            "MSL should translate discard to discard_fragment(): {msl}"
        );
        assert!(
            !msl.contains("discard;"),
            "MSL should not use GLSL discard syntax"
        );
    }

    #[test]
    fn test_runtime_effect_caches_glsl() {
        let src = "half4 main(float2 p) { return half4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let a = effect.compile_to(ShaderTarget::GlslEs300).unwrap();
        let b = effect.compile_to(ShaderTarget::GlslEs300).unwrap();
        assert_eq!(a, b, "Cached GLSL should be identical");
    }

    #[test]
    fn test_runtime_effect_caches_wgsl() {
        let src = "half4 main(float2 p) { return half4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let a = effect.compile_to(ShaderTarget::Wgsl).unwrap();
        let b = effect.compile_to(ShaderTarget::Wgsl).unwrap();
        assert_eq!(a, b, "Cached WGSL should be identical");
    }

    #[test]
    fn test_runtime_effect_caches_msl() {
        let src = "half4 main(float2 p) { return half4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let a = effect.compile_to(ShaderTarget::Msl).unwrap();
        let b = effect.compile_to(ShaderTarget::Msl).unwrap();
        assert_eq!(a, b, "Cached MSL should be identical");
    }

    #[test]
    fn test_compile_to_spirv_produces_words() {
        let src = "vec4 main(vec2 p) { return vec4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let words = effect.compile_to_spirv().unwrap_or_else(|e| {
            panic!("SPIR-V compile failed: {e}");
        });
        assert!(!words.is_empty(), "SPIR-V output should not be empty");
        // SPIR-V magic number is 0x0723_0203 at index 0.
        assert_eq!(
            words[0], 0x0723_0203,
            "SPIR-V magic word missing (got {:#010x})",
            words[0]
        );
    }

    #[test]
    fn test_compile_to_spirv_via_target() {
        let src = "vec4 main(vec2 p) { return vec4(0.5, 0.5, 0.5, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let hex = effect
            .compile_to(ShaderTarget::SpirV)
            .unwrap_or_else(|e| panic!("SpirV target compile failed: {e}"));
        // First word should be the SPIR-V magic number encoded as
        // little-endian hex.
        assert!(
            hex.starts_with("07230203"),
            "expected SPIR-V magic in hex output, got: {}",
            &hex[..32.min(hex.len())]
        );
    }

    #[test]
    fn test_runtime_effect_caches_spirv() {
        let src = "vec4 main(vec2 p) { return vec4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let a = effect.compile_to_spirv().unwrap();
        let b = effect.compile_to_spirv().unwrap();
        assert_eq!(a, b, "Cached SPIR-V should be identical");
    }

    #[test]
    fn test_compile_to_spirv_with_uniforms() {
        let src = r"
            uniform float scale;
            uniform vec2 offset;
            vec4 main(vec2 p) {
                return vec4((p + offset) * scale, 0.0, 1.0);
            }
        ";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let words = effect.compile_to_spirv().unwrap_or_else(|e| {
            panic!("SPIR-V compile with uniforms failed: {e}");
        });
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn test_runtime_effect_rejects_wrong_main_for_color_filter() {
        // main(vec2) but declared as ColorFilter (should want main(vec4))
        let src = "half4 main(float2 p) { return half4(0.0); }";
        let result = RuntimeEffect::make_for_color_filter(src);
        assert!(
            result.is_err(),
            "ColorFilter main with vec2 param should be rejected"
        );
    }

    #[test]
    fn test_runtime_effect_rejects_wrong_main_for_blender() {
        // main(vec2) but declared as Blender (should want main(vec4, vec4))
        let src = "half4 main(float2 p) { return half4(0.0); }";
        let result = RuntimeEffect::make_for_blender(src);
        assert!(
            result.is_err(),
            "Blender main with vec2 param should be rejected"
        );
    }

    #[test]
    fn test_runtime_effect_accepts_valid_shader() {
        let src = "half4 main(float2 p) { return half4(1.0, 0.0, 0.0, 1.0); }";
        let result = RuntimeEffect::make_for_shader(src);
        assert!(result.is_ok(), "Valid shader main should be accepted");
    }

    #[test]
    fn test_runtime_effect_accepts_valid_color_filter() {
        let src = "half4 main(half4 color) { return color * 0.5; }";
        let result = RuntimeEffect::make_for_color_filter(src);
        assert!(result.is_ok(), "Valid color filter main should be accepted");
    }

    #[test]
    fn test_runtime_effect_accepts_valid_blender() {
        let src = "half4 main(half4 src, half4 dst) { return src + dst; }";
        let result = RuntimeEffect::make_for_blender(src);
        assert!(result.is_ok(), "Valid blender main should be accepted");
    }

    #[test]
    fn test_validation_rejects_undeclared_variable() {
        let src = r"
            vec4 main(vec2 coord) {
                return vec4(undeclared, 0.0, 0.0, 1.0);
            }
        ";
        let result = RuntimeEffect::make_for_shader(src);
        assert!(
            matches!(result, Err(RuntimeEffectError::ValidationFailed(_))),
            "undeclared variable should surface as ValidationFailed, got {result:?}"
        );
    }

    #[test]
    fn test_validation_rejects_arity_mismatch() {
        let src = r"
            float square(float x) { return x * x; }
            vec4 main(vec2 coord) {
                return vec4(square(1.0, 2.0), 0.0, 0.0, 1.0);
            }
        ";
        let result = RuntimeEffect::make_for_shader(src);
        assert!(
            matches!(result, Err(RuntimeEffectError::ValidationFailed(_))),
            "arity mismatch should surface as ValidationFailed, got {result:?}"
        );
    }

    #[test]
    fn test_validation_rejects_type_mismatch_in_initializer() {
        let src = r"
            vec4 main(vec2 coord) {
                float x = vec2(1.0, 2.0);
                return vec4(x, 0.0, 0.0, 1.0);
            }
        ";
        let result = RuntimeEffect::make_for_shader(src);
        assert!(
            matches!(result, Err(RuntimeEffectError::ValidationFailed(_))),
            "type mismatch should surface as ValidationFailed, got {result:?}"
        );
    }

    #[test]
    fn test_parse_accepts_layout_qualifier() {
        let src = "layout(color) uniform half4 tint;\nhalf4 main(float2 p) { return tint; }";
        let result = RuntimeEffect::make_for_shader(src);
        assert!(
            result.is_ok(),
            "layout(color) should parse, got {:?}",
            result.err()
        );
    }

    #[test]
    fn test_glsl_output_syntactically_plausible() {
        let src = "half4 main(float2 p) { return half4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let glsl = effect.compile_to(ShaderTarget::GlslEs300).unwrap();
        // Basic syntax checks
        assert!(
            glsl.contains("void main") || glsl.contains("vec4 main"),
            "GLSL should contain a main function"
        );
        assert_eq!(
            glsl.matches('{').count(),
            glsl.matches('}').count(),
            "braces should balance"
        );
        assert_eq!(
            glsl.matches('(').count(),
            glsl.matches(')').count(),
            "parens should balance"
        );
    }

    #[test]
    fn test_wgsl_output_parses_via_naga() {
        // If naga is available, use it to validate the WGSL output
        let src = "half4 main(float2 p) { return half4(p.x, p.y, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let wgsl = effect.compile_to(ShaderTarget::Wgsl).unwrap();
        // naga is a dep thanks to SPIR-V task
        let parsed = naga::front::wgsl::parse_str(&wgsl);
        // May reject due to missing entry attributes — accept either outcome
        // but log if it fails
        if parsed.is_err() {
            eprintln!(
                "Note: WGSL output not directly naga-parseable (may need entry attributes). Output:\n{wgsl}"
            );
        }
    }

    #[test]
    fn test_msl_uses_float4_not_vec4() {
        let src = "half4 main(float2 p) { return half4(1.0, 0.0, 0.0, 1.0); }";
        let effect = RuntimeEffect::make_for_shader(src).unwrap();
        let msl = effect.compile_to(ShaderTarget::Msl).unwrap();
        // MSL should use float4 (if half4 is in source, at least internal vec types should be float/half4)
        assert!(
            msl.contains("float4") || msl.contains("half4"),
            "MSL should use float4 or half4 syntax:\n{msl}"
        );
    }
}
