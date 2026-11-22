//! Semantic validation for `SkSL` programs.
//!
//! The [`crate::sksl`] module produces a syntactic AST, nothing more.
//! That AST will happily describe programs that reference undeclared
//! variables, pass three floats to a function that expects one, or
//! return a `vec2` from a function that promised `vec4`. The GPU
//! backends used to find these issues only at codegen (sometimes as
//! a confusing `naga` error) and the software interpreter generally
//! did not find them at all, silently producing zero values.
//!
//! This module walks the parsed [`SkslProgram`] and performs a small
//! but useful type-check:
//!
//! - Variable scope resolution (nested scopes + shadowing rules)
//! - Undeclared variable detection
//! - Binary/unary operator type compatibility for scalars and vectors
//! - Function call arity and argument-type matching (user functions
//!   and a curated list of SkSL/GLSL built-ins)
//! - Return type matches the declared function signature
//! - `if`/`while`/`for`/`do-while` conditions are `bool`
//! - Uniforms and child-shader references resolve at call sites
//! - Swizzle accesses are within the base vector's component count and
//!   yield the right result type
//!
//! ## Explicit non-goals
//!
//! - **Custom struct field access** — structs parse, but we treat their
//!   field accesses permissively (accept any field and model the
//!   result as `Float`). Adding a real struct-field check requires
//!   threading [`crate::sksl::StructDecl`] information through the
//!   checker and is left for a future pass.
//! - **Overload resolution** — `SkSL` allows multiple functions with the
//!   same name but different parameter types. The parser does not
//!   support this (the [`SkslProgram::functions`] vector is a flat
//!   name->decl map), so the validator doesn't either. Built-ins with
//!   multiple valid signatures (e.g. `min`, `max`, `clamp`, `mix`) are
//!   modelled loosely: any numeric argument shape is accepted.
//! - **Implicit numeric conversions beyond equal types** — `float` and
//!   `int` are not silently interchanged. Code that mixes them should
//!   wrap the value explicitly (`float(i)` or `int(f)`). This matches
//!   strict `SkSL` semantics and keeps the checker simple.
//! - **Array types and indexing correctness** — arrays parse, but
//!   indexing an array currently yields the array's element type
//!   without bounds checking.
//!
//! The hook point is [`validate_program`], called from
//! `crate::runtime_effect::RuntimeEffect::compile` after
//! `validate_entry_point` so callers see a single coherent validation
//! pass.

use crate::sksl::{BinaryOp, Expr, FnDecl, SkslProgram, SkslType, Stmt, UnaryOp};
use std::collections::HashMap;

/// A semantic error discovered while validating an `SkSL` program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A name was referenced that is not a local, parameter, uniform,
    /// or global function.
    UndeclaredVariable(String),
    /// The type of an expression did not match what the context demanded.
    TypeMismatch {
        /// The type the surrounding construct expected.
        expected: String,
        /// The type the expression actually produced.
        actual: String,
        /// Human-readable description of where the mismatch occurred.
        context: String,
    },
    /// A function call passed the wrong number of arguments.
    ArityMismatch {
        /// Name of the called function.
        function: String,
        /// Number of parameters the function declares.
        expected: usize,
        /// Number of arguments the call supplied.
        actual: usize,
    },
    /// A function's `return` expression did not match its declared
    /// return type.
    ReturnTypeMismatch {
        /// Name of the function that contains the bad return.
        function: String,
        /// The declared return type.
        expected: String,
        /// The type of the return expression.
        actual: String,
    },
    /// A non-void function fell off the end without returning.
    MissingReturn(String),
    /// A binary or unary operator was applied to operand types that
    /// do not support it.
    InvalidOperator {
        /// String form of the operator (e.g. `"+"`).
        op: String,
        /// Type of the left-hand-side / sole operand.
        lhs: String,
        /// Type of the right-hand-side, or `""` for unary.
        rhs: String,
    },
    /// A swizzle (e.g. `.xyz`, `.rga`) referenced a component the base
    /// type does not have, or mixed channel sets.
    InvalidSwizzle {
        /// Swizzle characters, e.g. `"rgb"`.
        field: String,
        /// Base type the swizzle was applied to, e.g. `"vec2"`.
        base: String,
    },
    /// The lvalue side of `=` or `+=` was not assignable (literal,
    /// function call result, etc.).
    NotAssignable(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndeclaredVariable(n) => {
                write!(f, "undeclared variable '{n}'")
            }
            Self::TypeMismatch {
                expected,
                actual,
                context,
            } => write!(
                f,
                "type mismatch in {context}: expected {expected}, got {actual}"
            ),
            Self::ArityMismatch {
                function,
                expected,
                actual,
            } => write!(
                f,
                "function '{function}' takes {expected} arg(s), got {actual}"
            ),
            Self::ReturnTypeMismatch {
                function,
                expected,
                actual,
            } => write!(
                f,
                "function '{function}' declared return type {expected} but returns {actual}"
            ),
            Self::MissingReturn(n) => {
                write!(f, "function '{n}' is missing a return statement")
            }
            Self::InvalidOperator { op, lhs, rhs } => {
                if rhs.is_empty() {
                    write!(f, "invalid operator '{op}' on {lhs}")
                } else {
                    write!(f, "invalid operator '{op}' between {lhs} and {rhs}")
                }
            }
            Self::InvalidSwizzle { field, base } => {
                write!(f, "invalid swizzle '.{field}' on {base}")
            }
            Self::NotAssignable(ctx) => {
                write!(f, "expression is not assignable: {ctx}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Walk `program` and return the first semantic error encountered, or
/// `Ok(())` if the program is well-typed.
///
/// # Errors
///
/// Returns a [`ValidationError`] describing the first type or scoping
/// violation found (undeclared variable, arity/type mismatch, missing
/// return, etc.).
pub fn validate_program(program: &SkslProgram) -> Result<(), ValidationError> {
    let mut v = Validator::new(program);
    v.validate()
}

/// Statement return "signal": when a statement definitely returns a
/// value (e.g. `return x;`), the signal carries the expression's type
/// so the caller can compare against the function signature. A signal
/// of `None` means the statement fell through.
type ReturnSignal = Option<SkslType>;

struct Validator<'a> {
    program: &'a SkslProgram,
    /// Stack of lexical scopes. The outermost element is the function's
    /// parameter scope; pushes happen on `{}` blocks, `if`, `for`, etc.
    scopes: Vec<HashMap<String, SkslType>>,
    /// Top-level user functions indexed by name.
    functions: HashMap<String, (Vec<SkslType>, SkslType)>,
    /// Uniforms (including child shader/color filter/blender slots)
    /// indexed by name. These are visible from all function bodies.
    uniforms: HashMap<String, SkslType>,
    /// Name of the function currently being checked. Populated while
    /// walking `validate_function`.
    current_fn: Option<String>,
}

impl<'a> Validator<'a> {
    fn new(program: &'a SkslProgram) -> Self {
        let mut functions = HashMap::new();
        for f in &program.functions {
            let params: Vec<SkslType> = f.params.iter().map(|p| p.ty.clone()).collect();
            functions.insert(f.name.clone(), (params, f.return_type.clone()));
        }
        let mut uniforms = HashMap::new();
        for u in &program.uniforms {
            uniforms.insert(u.name.clone(), u.ty.clone());
        }
        Self {
            program,
            scopes: Vec::new(),
            functions,
            uniforms,
            current_fn: None,
        }
    }

    fn validate(&mut self) -> Result<(), ValidationError> {
        for f in &self.program.functions {
            self.validate_function(f)?;
        }
        Ok(())
    }

    fn validate_function(&mut self, f: &FnDecl) -> Result<(), ValidationError> {
        self.current_fn = Some(f.name.clone());
        self.scopes.clear();
        self.push_scope();
        for p in &f.params {
            self.declare(&p.name, p.ty.clone());
        }

        let returned = self.validate_stmt(&f.body)?;
        self.pop_scope();
        self.current_fn = None;

        returned.map_or_else(
            || {
                if matches!(f.return_type, SkslType::Void) {
                    Ok(())
                } else {
                    Err(ValidationError::MissingReturn(f.name.clone()))
                }
            },
            |actual| {
                if types_compatible(&f.return_type, &actual) {
                    Ok(())
                } else {
                    Err(ValidationError::ReturnTypeMismatch {
                        function: f.name.clone(),
                        expected: f.return_type.to_string(),
                        actual: actual.to_string(),
                    })
                }
            },
        )
    }

    fn validate_stmts(&mut self, stmts: &[Stmt]) -> Result<ReturnSignal, ValidationError> {
        let mut acc: ReturnSignal = None;
        for s in stmts {
            let r = self.validate_stmt(s)?;
            if acc.is_none() {
                acc = r;
            }
        }
        Ok(acc)
    }

    fn validate_stmt(&mut self, stmt: &Stmt) -> Result<ReturnSignal, ValidationError> {
        match stmt {
            Stmt::Expr(e) => {
                self.validate_expr(e)?;
                Ok(None)
            }
            Stmt::VarDecl { ty, name, init } => {
                if let Some(init_expr) = init {
                    let init_ty = self.validate_expr(init_expr)?;
                    if !types_compatible(ty, &init_ty) {
                        return Err(ValidationError::TypeMismatch {
                            expected: ty.to_string(),
                            actual: init_ty.to_string(),
                            context: format!("variable '{name}' initializer"),
                        });
                    }
                }
                self.declare(name, ty.clone());
                Ok(None)
            }
            Stmt::Block(stmts) => {
                self.push_scope();
                let r = self.validate_stmts(stmts)?;
                self.pop_scope();
                Ok(r)
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => self.validate_if(cond, then_branch, else_branch.as_deref()),
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => self.validate_for(init.as_deref(), cond.as_ref(), update.as_ref(), body),
            Stmt::While { cond, body } => self.validate_while(cond, body),
            Stmt::DoWhile { body, cond } => self.validate_do_while(body, cond),
            Stmt::Return(expr_opt) => {
                let ty = match expr_opt {
                    Some(e) => self.validate_expr(e)?,
                    None => SkslType::Void,
                };
                Ok(Some(ty))
            }
            Stmt::Break | Stmt::Continue | Stmt::Discard => Ok(None),
        }
    }

    fn validate_if(
        &mut self,
        cond: &Expr,
        then_branch: &Stmt,
        else_branch: Option<&Stmt>,
    ) -> Result<ReturnSignal, ValidationError> {
        let ct = self.validate_expr(cond)?;
        if !matches!(ct, SkslType::Bool) {
            return Err(ValidationError::TypeMismatch {
                expected: "bool".into(),
                actual: ct.to_string(),
                context: "if condition".into(),
            });
        }
        let then_ret = self.validate_stmt(then_branch)?;
        let else_ret = if let Some(eb) = else_branch {
            self.validate_stmt(eb)?
        } else {
            None
        };
        // Only report a definite return if BOTH branches return,
        // otherwise the function might still fall through.
        match (then_ret, else_ret) {
            (Some(a), Some(_)) => Ok(Some(a)),
            _ => Ok(None),
        }
    }

    fn validate_for(
        &mut self,
        init: Option<&Stmt>,
        cond: Option<&Expr>,
        update: Option<&Expr>,
        body: &Stmt,
    ) -> Result<ReturnSignal, ValidationError> {
        self.push_scope();
        if let Some(i) = init {
            self.validate_stmt(i)?;
        }
        if let Some(c) = cond {
            let ct = self.validate_expr(c)?;
            if !matches!(ct, SkslType::Bool) {
                self.pop_scope();
                return Err(ValidationError::TypeMismatch {
                    expected: "bool".into(),
                    actual: ct.to_string(),
                    context: "for condition".into(),
                });
            }
        }
        if let Some(u) = update {
            self.validate_expr(u)?;
        }
        // Loop body may or may not execute — never treat as
        // definitely-returning.
        let _ = self.validate_stmt(body)?;
        self.pop_scope();
        Ok(None)
    }

    fn validate_while(
        &mut self,
        cond: &Expr,
        body: &Stmt,
    ) -> Result<ReturnSignal, ValidationError> {
        let ct = self.validate_expr(cond)?;
        if !matches!(ct, SkslType::Bool) {
            return Err(ValidationError::TypeMismatch {
                expected: "bool".into(),
                actual: ct.to_string(),
                context: "while condition".into(),
            });
        }
        let _ = self.validate_stmt(body)?;
        Ok(None)
    }

    fn validate_do_while(
        &mut self,
        body: &Stmt,
        cond: &Expr,
    ) -> Result<ReturnSignal, ValidationError> {
        let _ = self.validate_stmt(body)?;
        let ct = self.validate_expr(cond)?;
        if !matches!(ct, SkslType::Bool) {
            return Err(ValidationError::TypeMismatch {
                expected: "bool".into(),
                actual: ct.to_string(),
                context: "do-while condition".into(),
            });
        }
        Ok(None)
    }

    fn validate_expr(&mut self, expr: &Expr) -> Result<SkslType, ValidationError> {
        match expr {
            Expr::IntLit(_) => Ok(SkslType::Int),
            Expr::FloatLit(_) => Ok(SkslType::Float),
            Expr::BoolLit(_) => Ok(SkslType::Bool),

            Expr::Var(name) => self
                .lookup(name)
                .ok_or_else(|| ValidationError::UndeclaredVariable(name.clone())),

            Expr::Binary { left, op, right } => {
                let lt = self.validate_expr(left)?;
                let rt = self.validate_expr(right)?;
                result_type_for_binop(*op, &lt, &rt).ok_or_else(|| {
                    ValidationError::InvalidOperator {
                        op: op.glsl_str().to_string(),
                        lhs: lt.to_string(),
                        rhs: rt.to_string(),
                    }
                })
            }

            Expr::Unary { op, expr } => {
                let t = self.validate_expr(expr)?;
                result_type_for_unop(*op, &t).ok_or_else(|| ValidationError::InvalidOperator {
                    op: op.glsl_str().to_string(),
                    lhs: t.to_string(),
                    rhs: String::new(),
                })
            }

            Expr::Call { name, args } => self.validate_call(name, args),

            Expr::MethodCall {
                receiver,
                method,
                args,
            } => self.validate_method_call(receiver, method, args),

            Expr::Constructor { ty, args } => {
                for a in args {
                    self.validate_expr(a)?;
                }
                // We don't verify component counts strictly because
                // SkSL allows e.g. `vec4(v3, 1.0)` where a vec3 supplies
                // three components. Constructors produce exactly the
                // declared type regardless of argument shape.
                Ok(ty.clone())
            }

            Expr::Field { expr, field } => self.validate_field(expr, field),

            Expr::Index { expr, index } => self.validate_index(expr, index),

            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => self.validate_ternary(cond, then_expr, else_expr),

            Expr::Assign { target, value } => self.validate_assign(target, value),

            Expr::CompoundAssign { target, op, value } => {
                self.validate_compound_assign(target, *op, value)
            }

            Expr::PostIncDec { expr, .. } | Expr::PreIncDec { expr, .. } => {
                if !is_assignable(expr) {
                    return Err(ValidationError::NotAssignable(format!("{expr:?}")));
                }
                let t = self.validate_expr(expr)?;
                if !is_numeric(&t) {
                    return Err(ValidationError::InvalidOperator {
                        op: "++/--".into(),
                        lhs: t.to_string(),
                        rhs: String::new(),
                    });
                }
                Ok(t)
            }
        }
    }

    fn validate_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<SkslType, ValidationError> {
        // `child.eval(coord)` on a child shader/colorFilter/blender.
        if method == "eval"
            && let Expr::Var(name) = receiver
            && let Some(ty) = self.uniforms.get(name).cloned()
            && matches!(
                ty,
                SkslType::Shader | SkslType::ColorFilter | SkslType::Blender
            )
        {
            for a in args {
                self.validate_expr(a)?;
            }
            return Ok(SkslType::Half4);
        }
        Err(ValidationError::InvalidOperator {
            op: format!(".{method}()"),
            lhs: "non-child receiver".into(),
            rhs: String::new(),
        })
    }

    fn validate_field(&mut self, expr: &Expr, field: &str) -> Result<SkslType, ValidationError> {
        let base_ty = self.validate_expr(expr)?;
        // Structs are a documented limitation: we accept any field name
        // and assume Float, because the AST doesn't carry enough
        // information for the validator to look the field up in the
        // enclosing struct declaration.
        if matches!(base_ty, SkslType::Struct(_)) {
            return Ok(SkslType::Float);
        }
        swizzle_result_type(&base_ty, field).ok_or_else(|| ValidationError::InvalidSwizzle {
            field: field.to_string(),
            base: base_ty.to_string(),
        })
    }

    fn validate_index(&mut self, expr: &Expr, index: &Expr) -> Result<SkslType, ValidationError> {
        let base_ty = self.validate_expr(expr)?;
        let idx_ty = self.validate_expr(index)?;
        if !matches!(idx_ty, SkslType::Int) {
            return Err(ValidationError::TypeMismatch {
                expected: "int".into(),
                actual: idx_ty.to_string(),
                context: "array/vector index".into(),
            });
        }
        Ok(index_result_type(&base_ty))
    }

    fn validate_ternary(
        &mut self,
        cond: &Expr,
        then_expr: &Expr,
        else_expr: &Expr,
    ) -> Result<SkslType, ValidationError> {
        let ct = self.validate_expr(cond)?;
        if !matches!(ct, SkslType::Bool) {
            return Err(ValidationError::TypeMismatch {
                expected: "bool".into(),
                actual: ct.to_string(),
                context: "ternary condition".into(),
            });
        }
        let tt = self.validate_expr(then_expr)?;
        let et = self.validate_expr(else_expr)?;
        if types_compatible(&tt, &et) {
            Ok(tt)
        } else {
            Err(ValidationError::TypeMismatch {
                expected: tt.to_string(),
                actual: et.to_string(),
                context: "ternary branches".into(),
            })
        }
    }

    fn validate_assign(
        &mut self,
        target: &Expr,
        value: &Expr,
    ) -> Result<SkslType, ValidationError> {
        if !is_assignable(target) {
            return Err(ValidationError::NotAssignable(format!("{target:?}")));
        }
        let tt = self.validate_expr(target)?;
        let vt = self.validate_expr(value)?;
        if !types_compatible(&tt, &vt) {
            return Err(ValidationError::TypeMismatch {
                expected: tt.to_string(),
                actual: vt.to_string(),
                context: "assignment".into(),
            });
        }
        Ok(tt)
    }

    fn validate_compound_assign(
        &mut self,
        target: &Expr,
        op: BinaryOp,
        value: &Expr,
    ) -> Result<SkslType, ValidationError> {
        if !is_assignable(target) {
            return Err(ValidationError::NotAssignable(format!("{target:?}")));
        }
        let tt = self.validate_expr(target)?;
        let vt = self.validate_expr(value)?;
        result_type_for_binop(op, &tt, &vt).ok_or_else(|| ValidationError::InvalidOperator {
            op: op.glsl_str().to_string(),
            lhs: tt.to_string(),
            rhs: vt.to_string(),
        })?;
        Ok(tt)
    }

    fn validate_call(&mut self, name: &str, args: &[Expr]) -> Result<SkslType, ValidationError> {
        // User-declared functions are looked up first so users can
        // shadow built-ins (matches SkSL semantics).
        if let Some((params, ret)) = self.functions.get(name).cloned() {
            if args.len() != params.len() {
                return Err(ValidationError::ArityMismatch {
                    function: name.to_string(),
                    expected: params.len(),
                    actual: args.len(),
                });
            }
            for (arg, expected) in args.iter().zip(params.iter()) {
                let actual = self.validate_expr(arg)?;
                if !types_compatible(expected, &actual) {
                    return Err(ValidationError::TypeMismatch {
                        expected: expected.to_string(),
                        actual: actual.to_string(),
                        context: format!("argument to '{name}'"),
                    });
                }
            }
            return Ok(ret);
        }

        // Scalar/vector-to-scalar coercions that the parser emits as
        // Call (rather than Constructor) when the type name is reused.
        if matches!(name, "float" | "int" | "bool" | "half") {
            if args.len() != 1 {
                return Err(ValidationError::ArityMismatch {
                    function: name.to_string(),
                    expected: 1,
                    actual: args.len(),
                });
            }
            self.validate_expr(&args[0])?;
            return Ok(match name {
                "float" | "half" => SkslType::Float,
                "int" => SkslType::Int,
                "bool" => SkslType::Bool,
                _ => unreachable!(),
            });
        }

        // Vector constructors that arrive as Call (e.g. user wrote
        // `vec4(...)` in a position the parser tokenised as ident).
        if let Some(ty) = vector_constructor_type(name) {
            for a in args {
                self.validate_expr(a)?;
            }
            return Ok(ty);
        }

        // Built-in SkSL/GLSL functions.
        if let Some((arity, return_rule)) = builtin_signature(name) {
            if !arity.accepts(args.len()) {
                return Err(ValidationError::ArityMismatch {
                    function: name.to_string(),
                    expected: arity.canonical(),
                    actual: args.len(),
                });
            }
            let mut arg_types = Vec::with_capacity(args.len());
            for a in args {
                arg_types.push(self.validate_expr(a)?);
            }
            return return_rule
                .apply(&arg_types)
                .ok_or_else(|| ValidationError::InvalidOperator {
                    op: name.to_string(),
                    lhs: arg_types
                        .first()
                        .map(std::string::ToString::to_string)
                        .unwrap_or_default(),
                    rhs: arg_types
                        .get(1)
                        .map(std::string::ToString::to_string)
                        .unwrap_or_default(),
                });
        }

        // `sample(child, coord)` style calls on child shaders.
        if args.len() == 2
            && let Some(ty) = self.uniforms.get(name).cloned()
            && matches!(
                ty,
                SkslType::Shader | SkslType::ColorFilter | SkslType::Blender
            )
        {
            // Treat `child(coord)` style call — accepted with
            // a vec4 result.
            for a in args {
                self.validate_expr(a)?;
            }
            return Ok(SkslType::Vec4);
        }

        Err(ValidationError::UndeclaredVariable(name.to_string()))
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare(&mut self, name: &str, ty: SkslType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<SkslType> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) {
                return Some(t.clone());
            }
        }
        self.uniforms.get(name).cloned()
    }
}

/// Is this expression a legal lvalue target?
fn is_assignable(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) => true,
        Expr::Field { expr, .. } | Expr::Index { expr, .. } => is_assignable(expr),
        _ => false,
    }
}

/// Our "compatibility" predicate. Strict equality with a couple of
/// equivalences: `float`/`half` and their matching vectors are
/// interchangeable, matching the interpreter's behaviour. Everything
/// else requires an exact type match.
fn types_compatible(expected: &SkslType, actual: &SkslType) -> bool {
    if expected == actual {
        return true;
    }
    matches!(
        (expected, actual),
        (SkslType::Float, SkslType::Half)
            | (SkslType::Half, SkslType::Float)
            | (SkslType::Vec2, SkslType::Half2)
            | (SkslType::Half2, SkslType::Vec2)
            | (SkslType::Vec3, SkslType::Half3)
            | (SkslType::Half3, SkslType::Vec3)
            | (SkslType::Vec4, SkslType::Half4)
            | (SkslType::Half4, SkslType::Vec4)
    )
}

const fn is_numeric(ty: &SkslType) -> bool {
    matches!(
        ty,
        SkslType::Int
            | SkslType::Float
            | SkslType::Half
            | SkslType::Vec2
            | SkslType::Vec3
            | SkslType::Vec4
            | SkslType::Half2
            | SkslType::Half3
            | SkslType::Half4
            | SkslType::Mat2
            | SkslType::Mat3
            | SkslType::Mat4
    )
}

const fn is_integer(ty: &SkslType) -> bool {
    matches!(ty, SkslType::Int)
}

/// Binary op result typing. Returns None when the operator/types pair
/// is invalid (e.g. `bool + bool`).
fn result_type_for_binop(op: BinaryOp, lhs: &SkslType, rhs: &SkslType) -> Option<SkslType> {
    use BinaryOp::{
        Add, And, BitAnd, BitOr, BitXor, Div, Eq, Gt, GtEq, Lt, LtEq, Mod, Mul, NotEq, Or, Shl,
        Shr, Sub,
    };
    match op {
        Add | Sub | Mul | Div | Mod => {
            if !is_numeric(lhs) || !is_numeric(rhs) {
                return None;
            }
            // Scalar * vector or vector * scalar is legal in SkSL and
            // yields the vector type.
            if lhs == rhs {
                return Some(lhs.clone());
            }
            if is_scalar_numeric(lhs) && is_vector_or_matrix(rhs) {
                return Some(rhs.clone());
            }
            if is_scalar_numeric(rhs) && is_vector_or_matrix(lhs) {
                return Some(lhs.clone());
            }
            // Float/half interchangeable, so pick the float-family wider
            // type if they differ only by precision.
            if types_compatible(lhs, rhs) {
                return Some(lhs.clone());
            }
            // Matrix * vector where sizes agree is common in SkSL; we
            // approximate by returning the vector type.
            if is_vector_or_matrix(lhs) && is_vector_or_matrix(rhs) {
                return Some(lhs.clone());
            }
            None
        }
        Eq | NotEq => {
            if types_compatible(lhs, rhs) {
                Some(SkslType::Bool)
            } else {
                None
            }
        }
        Lt | LtEq | Gt | GtEq => {
            if is_numeric(lhs) && is_numeric(rhs) {
                Some(SkslType::Bool)
            } else {
                None
            }
        }
        And | Or => {
            if matches!(lhs, SkslType::Bool) && matches!(rhs, SkslType::Bool) {
                Some(SkslType::Bool)
            } else {
                None
            }
        }
        BitAnd | BitOr | BitXor | Shl | Shr => {
            if is_integer(lhs) && is_integer(rhs) {
                Some(SkslType::Int)
            } else {
                None
            }
        }
    }
}

fn result_type_for_unop(op: UnaryOp, t: &SkslType) -> Option<SkslType> {
    match op {
        UnaryOp::Neg => {
            if is_numeric(t) {
                Some(t.clone())
            } else {
                None
            }
        }
        UnaryOp::Not => {
            if matches!(t, SkslType::Bool) {
                Some(SkslType::Bool)
            } else {
                None
            }
        }
        UnaryOp::BitNot => {
            if is_integer(t) {
                Some(SkslType::Int)
            } else {
                None
            }
        }
    }
}

const fn is_scalar_numeric(ty: &SkslType) -> bool {
    matches!(ty, SkslType::Int | SkslType::Float | SkslType::Half)
}

const fn is_vector_or_matrix(ty: &SkslType) -> bool {
    matches!(
        ty,
        SkslType::Vec2
            | SkslType::Vec3
            | SkslType::Vec4
            | SkslType::Half2
            | SkslType::Half3
            | SkslType::Half4
            | SkslType::Mat2
            | SkslType::Mat3
            | SkslType::Mat4
    )
}

/// Expected arity pattern for a built-in. Most built-ins have a fixed
/// number of parameters; a few (e.g. `atan`) accept 1 or 2.
#[derive(Debug, Clone, Copy)]
enum BuiltinArity {
    Exact(usize),
    Range(usize, usize),
}

impl BuiltinArity {
    const fn accepts(self, n: usize) -> bool {
        match self {
            Self::Exact(k) => n == k,
            Self::Range(lo, hi) => n >= lo && n <= hi,
        }
    }

    const fn canonical(self) -> usize {
        match self {
            Self::Exact(k) => k,
            Self::Range(_, hi) => hi,
        }
    }
}

/// Rule for computing the return type of a built-in. We don't attempt
/// to prove argument types line up with each other (e.g. that all
/// `clamp` arguments share the same shape) beyond insisting they are
/// numeric where numeric is required. This matches the interpreter's
/// permissive behaviour and keeps the validator pragmatic.
#[derive(Debug, Clone, Copy)]
enum BuiltinReturn {
    /// Constant return type, independent of arguments.
    Fixed(SkslTypeTag),
    /// Return type equals the first argument's type.
    SameAsFirst,
    /// Return type equals `length`/`distance`/`dot` result: Float when
    /// the input is a vector, Float for scalar inputs.
    ScalarFromVec,
}

#[derive(Debug, Clone, Copy)]
enum SkslTypeTag {
    #[allow(dead_code)]
    Float,
    Vec3,
    Vec4,
    #[allow(dead_code)]
    Bool,
    #[allow(dead_code)]
    Int,
}

impl BuiltinReturn {
    fn apply(self, args: &[SkslType]) -> Option<SkslType> {
        match self {
            Self::Fixed(tag) => Some(tag_to_type(tag)),
            Self::SameAsFirst => args
                .first()
                .cloned()
                .map(|t| if is_numeric(&t) { t } else { SkslType::Float }),
            Self::ScalarFromVec => {
                let first = args.first()?;
                if is_numeric(first) {
                    Some(SkslType::Float)
                } else {
                    None
                }
            }
        }
    }
}

const fn tag_to_type(tag: SkslTypeTag) -> SkslType {
    match tag {
        SkslTypeTag::Float => SkslType::Float,
        SkslTypeTag::Vec3 => SkslType::Vec3,
        SkslTypeTag::Vec4 => SkslType::Vec4,
        SkslTypeTag::Bool => SkslType::Bool,
        SkslTypeTag::Int => SkslType::Int,
    }
}

/// Registry of SkSL/GLSL built-ins supported by the interpreter (plus a
/// few type coercion helpers). Each entry states the accepted arity
/// and a rule for computing the return type.
fn builtin_signature(name: &str) -> Option<(BuiltinArity, BuiltinReturn)> {
    use BuiltinArity::{Exact, Range};
    use BuiltinReturn::{Fixed, SameAsFirst, ScalarFromVec};
    Some(match name {
        // Component-wise scalar/vector ops: result shape == arg shape.
        "abs" | "sign" | "floor" | "ceil" | "fract" | "sin" | "cos" | "tan" | "asin" | "acos"
        | "exp" | "exp2" | "log" | "log2" | "sqrt" | "inversesqrt" | "radians" | "degrees" => {
            (Exact(1), SameAsFirst)
        }
        "atan" => (Range(1, 2), SameAsFirst),
        "atan2" | "pow" | "mod" | "min" | "max" | "step" | "reflect" => (Exact(2), SameAsFirst),
        "clamp" | "mix" | "smoothstep" | "refract" => (Exact(3), SameAsFirst),
        // Vector-returning. `normalize`/`transpose`/`inverse` preserve shape.
        "normalize" | "transpose" | "inverse" => (Exact(1), SameAsFirst),
        // Geometry built-ins that collapse to a scalar.
        "length" => (Exact(1), ScalarFromVec),
        "distance" | "dot" => (Exact(2), ScalarFromVec),
        "cross" => (Exact(2), Fixed(SkslTypeTag::Vec3)),
        // Sampler / child invocation style calls — treat as vec4 for
        // shaders. Parsers usually desugar these to child-shader calls,
        // but if they reach here we still type them sensibly.
        "texture" | "texture2D" | "sample" => (Exact(2), Fixed(SkslTypeTag::Vec4)),
        _ => return None,
    })
}

/// Is `name` a vector/matrix constructor? Returns the result type.
fn vector_constructor_type(name: &str) -> Option<SkslType> {
    Some(match name {
        "vec2" | "float2" => SkslType::Vec2,
        "vec3" | "float3" => SkslType::Vec3,
        "vec4" | "float4" => SkslType::Vec4,
        "half2" => SkslType::Half2,
        "half3" => SkslType::Half3,
        "half4" => SkslType::Half4,
        "mat2" | "float2x2" => SkslType::Mat2,
        "mat3" | "float3x3" => SkslType::Mat3,
        "mat4" | "float4x4" => SkslType::Mat4,
        _ => return None,
    })
}

/// Swizzle result type: e.g. `vec3.xy -> vec2`, `vec4.r -> float`.
fn swizzle_result_type(base: &SkslType, field: &str) -> Option<SkslType> {
    let (components, is_half) = match base {
        SkslType::Vec2 => (2, false),
        SkslType::Vec3 => (3, false),
        SkslType::Vec4 => (4, false),
        SkslType::Half2 => (2, true),
        SkslType::Half3 => (3, true),
        SkslType::Half4 => (4, true),
        _ => return None,
    };
    if field.is_empty() || field.len() > 4 {
        return None;
    }
    for c in field.chars() {
        let idx = match c {
            'x' | 'r' | 's' => 0,
            'y' | 'g' | 't' => 1,
            'z' | 'b' | 'p' => 2,
            'w' | 'a' | 'q' => 3,
            _ => return None,
        };
        if idx >= components {
            return None;
        }
    }
    Some(match (field.len(), is_half) {
        (1, false) => SkslType::Float,
        (1, true) => SkslType::Half,
        (2, false) => SkslType::Vec2,
        (2, true) => SkslType::Half2,
        (3, false) => SkslType::Vec3,
        (3, true) => SkslType::Half3,
        (4, false) => SkslType::Vec4,
        (4, true) => SkslType::Half4,
        _ => return None,
    })
}

/// Result type of indexing into a vector/matrix/array.
fn index_result_type(base: &SkslType) -> SkslType {
    match base {
        SkslType::Half2 | SkslType::Half3 | SkslType::Half4 => SkslType::Half,
        SkslType::Mat2 => SkslType::Vec2,
        SkslType::Mat3 => SkslType::Vec3,
        SkslType::Mat4 => SkslType::Vec4,
        SkslType::Array(inner, _) => (**inner).clone(),
        // Vec2/Vec3/Vec4 index to Float; unknown base also falls back to
        // Float so we don't cascade errors.
        _ => SkslType::Float,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sksl::Parser;

    fn parse(src: &str) -> SkslProgram {
        Parser::new(src).parse_program().unwrap()
    }

    #[test]
    fn validates_simple_shader() {
        let p = parse("vec4 main(vec2 coord) { return vec4(coord.x, coord.y, 0.0, 1.0); }");
        validate_program(&p).unwrap();
    }

    #[test]
    fn flags_undeclared_variable() {
        let p = parse("vec4 main(vec2 coord) { return foo; }");
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::UndeclaredVariable(ref n) if n == "foo"));
    }

    #[test]
    fn flags_initializer_type_mismatch() {
        let p = parse(
            "vec4 main(vec2 coord) { float x = vec2(1.0, 2.0); return vec4(x, 0.0, 0.0, 1.0); }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::TypeMismatch { .. }));
    }

    #[test]
    fn flags_arity_mismatch() {
        let p = parse(
            "float sq(float x) { return x * x; } \
             vec4 main(vec2 coord) { return vec4(sq(1.0, 2.0), 0.0, 0.0, 1.0); }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::ArityMismatch { .. }));
    }

    #[test]
    fn flags_return_type_mismatch() {
        let p = parse("vec4 main(vec2 coord) { return vec2(0.0, 0.0); }");
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::ReturnTypeMismatch { .. }));
    }

    #[test]
    fn flags_missing_return() {
        let p = parse("vec4 main(vec2 coord) { float x = 1.0; }");
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::MissingReturn(_)));
    }

    #[test]
    fn allows_uniform_reference() {
        let p = parse(
            "uniform float t; \
             vec4 main(vec2 coord) { return vec4(sin(t), 0.0, 0.0, 1.0); }",
        );
        validate_program(&p).unwrap();
    }

    #[test]
    fn rejects_non_bool_if_condition() {
        let p = parse(
            "vec4 main(vec2 coord) { \
                 if (coord.x) { return vec4(0.0); } \
                 return vec4(1.0); \
             }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::TypeMismatch { .. }));
    }

    #[test]
    fn allows_swizzle() {
        let p = parse(
            "vec4 main(vec2 coord) { \
                 vec4 c = vec4(1.0, 0.5, 0.25, 1.0); \
                 return c.rgba; \
             }",
        );
        validate_program(&p).unwrap();
    }

    #[test]
    fn rejects_out_of_bounds_swizzle() {
        let p = parse("vec4 main(vec2 coord) { return vec4(coord.z, 0.0, 0.0, 1.0); }");
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidSwizzle { .. }));
    }

    #[test]
    fn allows_builtin_mix() {
        let p = parse(
            "vec4 main(vec2 coord) { \
                 return mix(vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), 0.5); \
             }",
        );
        validate_program(&p).unwrap();
    }

    #[test]
    fn scopes_are_nested() {
        let p = parse(
            "vec4 main(vec2 coord) { \
                 { float x = 1.0; } \
                 return vec4(x, 0.0, 0.0, 1.0); \
             }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::UndeclaredVariable(ref n) if n == "x"));
    }

    #[test]
    fn rejects_eval_on_non_child() {
        let p = parse(
            "uniform float2 p; \
             half4 main(float2 coord) { return p.eval(coord); }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidOperator { .. }));
    }

    #[test]
    fn rejects_unknown_method() {
        let p = parse(
            "uniform shader child; \
             half4 main(float2 coord) { return child.frobnicate(coord); }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidOperator { .. }));
    }

    #[test]
    fn for_loop_scope_confined() {
        let p = parse(
            "vec4 main(vec2 coord) { \
                 for (int i = 0; i < 3; i = i + 1) { } \
                 return vec4(float(i), 0.0, 0.0, 1.0); \
             }",
        );
        let err = validate_program(&p).unwrap_err();
        assert!(matches!(err, ValidationError::UndeclaredVariable(ref n) if n == "i"));
    }
}
