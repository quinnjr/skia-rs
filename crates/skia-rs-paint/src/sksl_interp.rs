//! Tree-walking interpreter for `SkSL` AST.
//!
//! Evaluates the subset of `SkSL` needed for runtime shader sampling and
//! runtime color filter application. This is the software fallback used
//! when a GPU backend is not available (or not yet wired).
//!
//! Scope:
//! - Scalar, vector (vec2/vec3/vec4), and matrix (mat2/mat3/mat4) arithmetic
//! - Control flow: if/else, for, while, do-while, break/continue/return/discard
//! - Function calls (user-defined and built-ins)
//! - Constructors (vec2/3/4, mat2/3/4, scalar)
//! - Swizzles (arbitrary length up to 4)
//! - Uniform lookup and child shader sampling via `child.eval(coord)`
//!
//! Limitations (documented, not silent):
//! - No custom structs or arrays of complex types
//! - No matrix inverse beyond what's exposed as a builtin
//! - Integer arithmetic collapses into float for most operations (`SkSL` is a
//!   high-level shader language; most meaningful uses involve floats)
//! - Loops are capped at `10_000` iterations to prevent runaway evaluation

use crate::shader::Shader;
use crate::sksl::{BinaryOp, Expr, FnDecl, SkslType, Stmt, UnaryOp};
use skia_rs_core::cast::{saturate_to_i32, scalar_from_i32};
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum loop iterations. Prevents runaway evaluation if a shader
/// contains an unbounded loop or a mistake in the update expression.
const LOOP_LIMIT: usize = 10_000;

/// Runtime value during `SkSL` evaluation.
#[derive(Debug, Clone)]
pub enum Value {
    /// Absence of a value (from void function returns or uninitialized vars).
    Void,
    /// Boolean scalar.
    Bool(bool),
    /// Signed 32-bit integer scalar.
    Int(i32),
    /// 32-bit float scalar.
    Float(f32),
    /// 2-component float vector.
    Vec2([f32; 2]),
    /// 3-component float vector.
    Vec3([f32; 3]),
    /// 4-component float vector.
    Vec4([f32; 4]),
    /// 2x2 float matrix, column-major.
    Mat2([f32; 4]),
    /// 3x3 float matrix, column-major.
    Mat3([f32; 9]),
    /// 4x4 float matrix, column-major.
    Mat4([f32; 16]),
}

impl Value {
    /// Coerce to a float scalar. Vectors and matrices collapse to 0.
    pub fn as_f32(&self) -> f32 {
        match self {
            Self::Float(f) => *f,
            Self::Int(i) => scalar_from_i32(*i),
            Self::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            _ => {
                debug_assert!(
                    false,
                    "cannot coerce {self:?} to f32 — validator should have caught this"
                );
                0.0
            }
        }
    }

    pub fn as_i32(&self) -> i32 {
        match self {
            Self::Int(i) => *i,
            Self::Float(f) => saturate_to_i32(*f),
            Self::Bool(b) => i32::from(*b),
            _ => {
                debug_assert!(
                    false,
                    "cannot coerce {self:?} to i32 — validator should have caught this"
                );
                0
            }
        }
    }

    #[allow(dead_code)]
    pub fn as_vec2(&self) -> [f32; 2] {
        match self {
            Self::Vec2(v) => *v,
            Self::Vec3(v) => [v[0], v[1]],
            Self::Vec4(v) => [v[0], v[1]],
            Self::Float(f) => [*f, *f],
            Self::Int(i) => [scalar_from_i32(*i), scalar_from_i32(*i)],
            _ => {
                debug_assert!(
                    false,
                    "cannot coerce {self:?} to vec2 — validator should have caught this"
                );
                [0.0, 0.0]
            }
        }
    }

    pub fn as_vec3(&self) -> [f32; 3] {
        match self {
            Self::Vec3(v) => *v,
            Self::Vec4(v) => [v[0], v[1], v[2]],
            Self::Vec2(v) => [v[0], v[1], 0.0],
            Self::Float(f) => [*f, *f, *f],
            Self::Int(i) => {
                let x = scalar_from_i32(*i);
                [x, x, x]
            }
            _ => {
                debug_assert!(
                    false,
                    "cannot coerce {self:?} to vec3 — validator should have caught this"
                );
                [0.0, 0.0, 0.0]
            }
        }
    }

    pub fn as_vec4(&self) -> [f32; 4] {
        match self {
            Self::Vec4(v) => *v,
            Self::Vec3(v) => [v[0], v[1], v[2], 1.0],
            Self::Vec2(v) => [v[0], v[1], 0.0, 1.0],
            Self::Float(f) => [*f, *f, *f, *f],
            Self::Int(i) => {
                let x = scalar_from_i32(*i);
                [x, x, x, x]
            }
            _ => {
                debug_assert!(
                    false,
                    "cannot coerce {self:?} to vec4 — validator should have caught this"
                );
                [0.0, 0.0, 0.0, 0.0]
            }
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            _ => {
                debug_assert!(
                    false,
                    "cannot coerce {self:?} to bool — validator should have caught this"
                );
                false
            }
        }
    }

    /// Number of scalar components this value contains. Scalars are 1.
    pub const fn component_count(&self) -> usize {
        match self {
            Self::Float(_) | Self::Int(_) | Self::Bool(_) => 1,
            Self::Vec2(_) => 2,
            Self::Vec3(_) => 3,
            Self::Vec4(_) | Self::Mat2(_) => 4,
            Self::Mat3(_) => 9,
            Self::Mat4(_) => 16,
            Self::Void => 0,
        }
    }

    /// Get the nth scalar component.
    pub fn component(&self, i: usize) -> f32 {
        match self {
            Self::Float(f) => {
                if i == 0 {
                    *f
                } else {
                    0.0
                }
            }
            Self::Int(v) => {
                if i == 0 {
                    scalar_from_i32(*v)
                } else {
                    0.0
                }
            }
            Self::Bool(b) => {
                if i == 0 {
                    if *b { 1.0 } else { 0.0 }
                } else {
                    0.0
                }
            }
            Self::Vec2(v) => *v.get(i).unwrap_or(&0.0),
            Self::Vec3(v) => *v.get(i).unwrap_or(&0.0),
            Self::Vec4(v) => *v.get(i).unwrap_or(&0.0),
            Self::Mat2(m) => *m.get(i).unwrap_or(&0.0),
            Self::Mat3(m) => *m.get(i).unwrap_or(&0.0),
            Self::Mat4(m) => *m.get(i).unwrap_or(&0.0),
            Self::Void => 0.0,
        }
    }

    /// Build a vector from N scalar components.
    pub fn from_components(components: &[f32]) -> Self {
        match components.len() {
            0 => Self::Void,
            2 => Self::Vec2([components[0], components[1]]),
            3 => Self::Vec3([components[0], components[1], components[2]]),
            4 => Self::Vec4([components[0], components[1], components[2], components[3]]),
            _ => Self::Float(components[0]),
        }
    }
}

/// Control-flow signal propagated through statement interpretation.
enum ControlFlow {
    /// Continue to the next statement normally.
    Normal,
    /// Return from the current function with a value.
    Return(Value),
    /// Break out of the innermost loop.
    Break,
    /// Continue the innermost loop.
    Continue,
    /// Fragment-shader `discard` — treated as an early return of
    /// transparent black for sampling purposes.
    Discard,
}

/// Interpreter environment: variable scopes, uniforms, functions, children.
pub struct Interp<'a> {
    /// User-defined functions available for invocation by name.
    pub functions: HashMap<String, &'a FnDecl>,
    /// Uniforms, keyed by name.
    pub uniforms: HashMap<String, Value>,
    /// Child shaders in declaration order.
    pub children: Vec<Arc<dyn Shader>>,
    /// Maps child name to its index in `children`.
    pub children_by_name: HashMap<String, usize>,
    /// Scope stack, innermost last.
    scopes: Vec<HashMap<String, Value>>,
}

impl Interp<'_> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            uniforms: HashMap::new(),
            children: Vec::new(),
            children_by_name: HashMap::new(),
            scopes: vec![HashMap::new()],
        }
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        self.uniforms.get(name).cloned()
    }

    fn assign(&mut self, name: &str, value: Value) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return;
            }
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    fn declare(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Run a named function with the given argument values.
    pub fn run_function(&mut self, name: &str, args: Vec<Value>) -> Value {
        let func = match self.functions.get(name) {
            Some(f) => *f,
            None => return Value::Void,
        };
        self.push_scope();
        for (param, val) in func.params.iter().zip(args) {
            self.declare(&param.name, val);
        }
        let cf = self.run_stmt(&func.body);
        self.pop_scope();
        match cf {
            ControlFlow::Return(v) => v,
            ControlFlow::Discard => Value::Vec4([0.0, 0.0, 0.0, 0.0]),
            _ => Value::Void,
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one match arm per Stmt variant; splitting would obscure the 1:1 correspondence with the AST"
    )]
    fn run_stmt(&mut self, stmt: &Stmt) -> ControlFlow {
        match stmt {
            Stmt::Block(stmts) => {
                self.push_scope();
                let mut cf = ControlFlow::Normal;
                for s in stmts {
                    cf = self.run_stmt(s);
                    if !matches!(cf, ControlFlow::Normal) {
                        break;
                    }
                }
                self.pop_scope();
                cf
            }
            Stmt::Expr(e) => {
                self.compute_expr(e);
                ControlFlow::Normal
            }
            Stmt::VarDecl { name, init, .. } => {
                let v = init.as_ref().map_or(Value::Void, |e| self.compute_expr(e));
                self.declare(name, v);
                ControlFlow::Normal
            }
            Stmt::Return(expr) => {
                let v = expr.as_ref().map_or(Value::Void, |e| self.compute_expr(e));
                ControlFlow::Return(v)
            }
            Stmt::Break => ControlFlow::Break,
            Stmt::Continue => ControlFlow::Continue,
            Stmt::Discard => ControlFlow::Discard,
            Stmt::If {
                cond,
                then_branch,
                else_branch,
            } => {
                if self.compute_expr(cond).as_bool() {
                    self.run_stmt(then_branch)
                } else if let Some(eb) = else_branch {
                    self.run_stmt(eb)
                } else {
                    ControlFlow::Normal
                }
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
            } => {
                self.push_scope();
                if let Some(init) = init {
                    let cf = self.run_stmt(init);
                    if let ControlFlow::Return(_) | ControlFlow::Discard = cf {
                        self.pop_scope();
                        return cf;
                    }
                }
                let mut result = ControlFlow::Normal;
                for _ in 0..LOOP_LIMIT {
                    let cond_true = cond.as_ref().is_none_or(|c| self.compute_expr(c).as_bool());
                    if !cond_true {
                        break;
                    }
                    match self.run_stmt(body) {
                        ControlFlow::Normal | ControlFlow::Continue => {}
                        ControlFlow::Break => break,
                        cf @ (ControlFlow::Return(_) | ControlFlow::Discard) => {
                            result = cf;
                            break;
                        }
                    }
                    if let Some(u) = update {
                        self.compute_expr(u);
                    }
                }
                self.pop_scope();
                result
            }
            Stmt::While { cond, body } => {
                let mut result = ControlFlow::Normal;
                for _ in 0..LOOP_LIMIT {
                    if !self.compute_expr(cond).as_bool() {
                        break;
                    }
                    match self.run_stmt(body) {
                        ControlFlow::Normal | ControlFlow::Continue => {}
                        ControlFlow::Break => break,
                        cf @ (ControlFlow::Return(_) | ControlFlow::Discard) => {
                            result = cf;
                            break;
                        }
                    }
                }
                result
            }
            Stmt::DoWhile { body, cond } => {
                let mut result = ControlFlow::Normal;
                for _ in 0..LOOP_LIMIT {
                    match self.run_stmt(body) {
                        ControlFlow::Normal | ControlFlow::Continue => {}
                        ControlFlow::Break => break,
                        cf @ (ControlFlow::Return(_) | ControlFlow::Discard) => {
                            result = cf;
                            break;
                        }
                    }
                    if !self.compute_expr(cond).as_bool() {
                        break;
                    }
                }
                result
            }
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one match arm per Expr variant; splitting would obscure the 1:1 correspondence with the AST"
    )]
    fn compute_expr(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::IntLit(n) => Value::Int(*n),
            Expr::FloatLit(n) => Value::Float(*n),
            Expr::BoolLit(b) => Value::Bool(*b),
            Expr::Var(name) => self.lookup(name).unwrap_or(Value::Void),
            Expr::Binary { left, op, right } => {
                // Short-circuit logical ops.
                if matches!(op, BinaryOp::And) {
                    let l = self.compute_expr(left);
                    if !l.as_bool() {
                        return Value::Bool(false);
                    }
                    return Value::Bool(self.compute_expr(right).as_bool());
                }
                if matches!(op, BinaryOp::Or) {
                    let l = self.compute_expr(left);
                    if l.as_bool() {
                        return Value::Bool(true);
                    }
                    return Value::Bool(self.compute_expr(right).as_bool());
                }
                let l = self.compute_expr(left);
                let r = self.compute_expr(right);
                apply_binary(*op, &l, &r)
            }
            Expr::Unary { op, expr } => {
                let v = self.compute_expr(expr);
                apply_unary(*op, &v)
            }
            Expr::Call { name, args } => {
                let arg_vals: Vec<Value> = args.iter().map(|a| self.compute_expr(a)).collect();
                // Built-ins first (matches SkSL semantics).
                if let Some(v) = call_builtin(name, &arg_vals) {
                    return v;
                }
                if self.functions.contains_key(name) {
                    return self.run_function(name, arg_vals);
                }
                Value::Void
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                // `child.eval(coord)` samples a bound child shader. The
                // returned color follows the Shader::sample premul contract.
                if method == "eval" {
                    if let Expr::Var(name) = receiver.as_ref() {
                        if let Some(&idx) = self.children_by_name.get(name.as_str()) {
                            let coord = args
                                .first()
                                .map_or(Value::Vec2([0.0, 0.0]), |a| self.compute_expr(a));
                            let xy = coord.as_vec2();
                            if let Some(child) = self.children.get(idx) {
                                let c = child.sample(xy[0], xy[1]);
                                return Value::Vec4([c.r, c.g, c.b, c.a]);
                            }
                            // Unbound child: transparent black.
                            return Value::Vec4([0.0, 0.0, 0.0, 0.0]);
                        }
                    }
                }
                Value::Void
            }
            Expr::Constructor { ty, args } => {
                let vals: Vec<Value> = args.iter().map(|a| self.compute_expr(a)).collect();
                construct(ty, &vals)
            }
            Expr::Field { expr, field } => {
                let base = self.compute_expr(expr);
                swizzle(&base, field)
            }
            Expr::Index { expr, index } => {
                let base = self.compute_expr(expr);
                let idx = self.compute_expr(index).as_i32();
                index_value(&base, idx)
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                if self.compute_expr(cond).as_bool() {
                    self.compute_expr(then_expr)
                } else {
                    self.compute_expr(else_expr)
                }
            }
            Expr::Assign { target, value } => {
                let v = self.compute_expr(value);
                self.assign_to(target, v.clone());
                v
            }
            Expr::CompoundAssign { target, op, value } => {
                let old = self.compute_expr(target);
                let rhs = self.compute_expr(value);
                let new = apply_binary(*op, &old, &rhs);
                self.assign_to(target, new.clone());
                new
            }
            Expr::PostIncDec { expr, inc } => {
                let old = self.compute_expr(expr);
                let one = match &old {
                    Value::Int(_) => Value::Int(1),
                    _ => Value::Float(1.0),
                };
                let op = if *inc { BinaryOp::Add } else { BinaryOp::Sub };
                let new = apply_binary(op, &old, &one);
                self.assign_to(expr, new);
                old
            }
            Expr::PreIncDec { expr, inc } => {
                let old = self.compute_expr(expr);
                let one = match &old {
                    Value::Int(_) => Value::Int(1),
                    _ => Value::Float(1.0),
                };
                let op = if *inc { BinaryOp::Add } else { BinaryOp::Sub };
                let new = apply_binary(op, &old, &one);
                self.assign_to(expr, new.clone());
                new
            }
        }
    }

    /// Write `value` into an lvalue expression. Supports Var targets, swizzle
    /// writes of any component count (e.g. `v.x = 0.5`, `c.rgb *= 0.5`,
    /// `c.ba = vec2(...)`), and integer index writes.
    fn assign_to(&mut self, target: &Expr, value: Value) {
        match target {
            Expr::Var(name) => self.assign(name, value),
            Expr::Field { expr, field } => {
                let base = self.compute_expr(expr);
                let mut comps: Vec<f32> = (0..base.component_count())
                    .map(|i| base.component(i))
                    .collect();
                // Scatter each source component into the slot named by the
                // corresponding swizzle character.
                for (src_i, ch) in field.chars().enumerate() {
                    if let Some(idx) = swizzle_index(ch) {
                        if idx < comps.len() {
                            comps[idx] = value.component(src_i);
                        }
                    } else {
                        return; // not a swizzle lvalue
                    }
                }
                let new = match base.component_count() {
                    2 => Value::Vec2([comps[0], comps[1]]),
                    3 => Value::Vec3([comps[0], comps[1], comps[2]]),
                    4 => Value::Vec4([comps[0], comps[1], comps[2], comps[3]]),
                    _ => Value::Float(value.as_f32()),
                };
                self.assign_to(expr, new);
            }
            Expr::Index { expr, index } => {
                let idx_i32 = self.compute_expr(index).as_i32();
                let idx = usize::try_from(idx_i32).unwrap_or(usize::MAX);
                let base = self.compute_expr(expr);
                let mut comps: Vec<f32> = (0..base.component_count())
                    .map(|i| base.component(i))
                    .collect();
                if idx < comps.len() {
                    comps[idx] = value.as_f32();
                }
                let new = match &base {
                    Value::Vec2(_) => Value::Vec2([comps[0], comps[1]]),
                    Value::Vec3(_) => Value::Vec3([comps[0], comps[1], comps[2]]),
                    Value::Vec4(_) => Value::Vec4([comps[0], comps[1], comps[2], comps[3]]),
                    _ => value,
                };
                self.assign_to(expr, new);
            }
            _ => {
                // Unsupported lvalue form.
            }
        }
    }
}

fn apply_unary(op: UnaryOp, v: &Value) -> Value {
    match op {
        UnaryOp::Neg => match v {
            Value::Int(i) => Value::Int(-*i),
            Value::Float(f) => Value::Float(-*f),
            Value::Vec2(a) => Value::Vec2([-a[0], -a[1]]),
            Value::Vec3(a) => Value::Vec3([-a[0], -a[1], -a[2]]),
            Value::Vec4(a) => Value::Vec4([-a[0], -a[1], -a[2], -a[3]]),
            Value::Mat2(m) => Value::Mat2([-m[0], -m[1], -m[2], -m[3]]),
            Value::Mat3(m) => {
                let mut out = [0.0f32; 9];
                for i in 0..9 {
                    out[i] = -m[i];
                }
                Value::Mat3(out)
            }
            Value::Mat4(m) => {
                let mut out = [0.0f32; 16];
                for i in 0..16 {
                    out[i] = -m[i];
                }
                Value::Mat4(out)
            }
            _ => v.clone(),
        },
        UnaryOp::Not => Value::Bool(!v.as_bool()),
        UnaryOp::BitNot => match v {
            Value::Int(i) => Value::Int(!*i),
            _ => v.clone(),
        },
    }
}

/// Apply a binary op with `SkSL`'s broadcast rules.
fn apply_binary(op: BinaryOp, l: &Value, r: &Value) -> Value {
    if matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    ) {
        return apply_cmp(op, l, r);
    }

    // Integer fast path for int-int arithmetic.
    if let (Value::Int(a), Value::Int(b)) = (l, r) {
        return match op {
            BinaryOp::Add => Value::Int(a.wrapping_add(*b)),
            BinaryOp::Sub => Value::Int(a.wrapping_sub(*b)),
            BinaryOp::Mul => Value::Int(a.wrapping_mul(*b)),
            BinaryOp::Div => {
                if *b == 0 {
                    Value::Int(0)
                } else {
                    Value::Int(a / b)
                }
            }
            BinaryOp::Mod => {
                if *b == 0 {
                    Value::Int(0)
                } else {
                    Value::Int(a % b)
                }
            }
            BinaryOp::BitAnd => Value::Int(a & b),
            BinaryOp::BitOr => Value::Int(a | b),
            BinaryOp::BitXor => Value::Int(a ^ b),
            #[allow(
                clippy::cast_sign_loss,
                reason = "GLSL/SkSL shift-amount is taken mod the bit width of the operand; reinterpreting the i32 shift count as u32 bits and letting wrapping_shl/shr mask it matches that semantics"
            )]
            BinaryOp::Shl => Value::Int(a.wrapping_shl(*b as u32)),
            #[allow(
                clippy::cast_sign_loss,
                reason = "GLSL/SkSL shift-amount is taken mod the bit width of the operand; reinterpreting the i32 shift count as u32 bits and letting wrapping_shl/shr mask it matches that semantics"
            )]
            BinaryOp::Shr => Value::Int(a.wrapping_shr(*b as u32)),
            _ => Value::Int(0),
        };
    }

    if matches!(op, BinaryOp::Mul) {
        if let Some(v) = mat_mul(l, r) {
            return v;
        }
    }

    // Matrix element-wise ops: mat±mat, mat·scalar, scalar·mat, mat/scalar…
    if let Some(v) = matrix_elementwise_op(op, l, r) {
        return v;
    }

    let lw = vec_width(l);
    let rw = vec_width(r);
    let width = lw.max(rw);

    if width == 0 {
        let a = l.as_f32();
        let b = r.as_f32();
        return scalar_op(op, a, b);
    }

    let lc = broadcast(l, width);
    let rc = broadcast(r, width);
    let mut out = [0.0f32; 4];
    for i in 0..width {
        out[i] = match scalar_op(op, lc[i], rc[i]) {
            Value::Float(f) => f,
            _ => 0.0,
        };
    }
    match width {
        2 => Value::Vec2([out[0], out[1]]),
        3 => Value::Vec3([out[0], out[1], out[2]]),
        4 => Value::Vec4([out[0], out[1], out[2], out[3]]),
        _ => Value::Float(out[0]),
    }
}

const fn vec_width(v: &Value) -> usize {
    match v {
        Value::Vec2(_) => 2,
        Value::Vec3(_) => 3,
        Value::Vec4(_) => 4,
        _ => 0,
    }
}

fn broadcast(v: &Value, width: usize) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    match v {
        Value::Vec2(a) => {
            out[0] = a[0];
            out[1] = a[1];
        }
        Value::Vec3(a) => {
            out[..3].copy_from_slice(a);
        }
        Value::Vec4(a) => {
            out.copy_from_slice(a);
        }
        _ => {
            let x = v.as_f32();
            for slot in out.iter_mut().take(width) {
                *slot = x;
            }
        }
    }
    out
}

fn scalar_op(op: BinaryOp, a: f32, b: f32) -> Value {
    // Float division and modulo by zero follow IEEE 754 (±inf / NaN),
    // matching GPU and SkSL semantics — no special-casing to 0.
    match op {
        BinaryOp::Add => Value::Float(a + b),
        BinaryOp::Sub => Value::Float(a - b),
        BinaryOp::Mul => Value::Float(a * b),
        BinaryOp::Div => Value::Float(a / b),
        BinaryOp::Mod => Value::Float((a / b).floor().mul_add(-b, a)),
        _ => Value::Float(0.0),
    }
}

fn apply_cmp(op: BinaryOp, l: &Value, r: &Value) -> Value {
    if matches!(op, BinaryOp::Eq | BinaryOp::NotEq) {
        let eq = values_equal(l, r);
        return Value::Bool(if matches!(op, BinaryOp::Eq) { eq } else { !eq });
    }
    let a = l.as_f32();
    let b = r.as_f32();
    let res = match op {
        BinaryOp::Lt => a < b,
        BinaryOp::LtEq => a <= b,
        BinaryOp::Gt => a > b,
        BinaryOp::GtEq => a >= b,
        _ => false,
    };
    Value::Bool(res)
}

#[allow(
    clippy::float_cmp,
    reason = "implements SkSL's `==` operator, which is exact equality on GLSL vector/scalar types, not a tolerance comparison"
)]
fn values_equal(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Vec2(a), Value::Vec2(b)) => a == b,
        (Value::Vec3(a), Value::Vec3(b)) => a == b,
        (Value::Vec4(a), Value::Vec4(b)) => a == b,
        _ => (l.as_f32() - r.as_f32()).abs() < f32::EPSILON,
    }
}

/// View a matrix value as (elements, dimension).
fn mat_slice(v: &Value) -> Option<(&[f32], usize)> {
    match v {
        Value::Mat2(m) => Some((&m[..], 2)),
        Value::Mat3(m) => Some((&m[..], 3)),
        Value::Mat4(m) => Some((&m[..], 4)),
        _ => None,
    }
}

/// Build a matrix value of dimension `dim` from elements.
fn mat_from_elems(dim: usize, elems: &[f32]) -> Value {
    match dim {
        2 => {
            let mut out = [0.0f32; 4];
            out.copy_from_slice(&elems[..4]);
            Value::Mat2(out)
        }
        3 => {
            let mut out = [0.0f32; 9];
            out.copy_from_slice(&elems[..9]);
            Value::Mat3(out)
        }
        _ => {
            let mut out = [0.0f32; 16];
            out.copy_from_slice(&elems[..16]);
            Value::Mat4(out)
        }
    }
}

/// Element-wise matrix arithmetic: mat±mat and mat/mat (component-wise per
/// GLSL), plus mat op scalar and scalar op mat for + - * /.
/// Linear-algebra multiplication (mat·mat, mat·vec, vec·mat) lives in
/// `mat_mul`.
#[allow(
    clippy::many_single_char_names,
    reason = "faithful linear-algebra code; l/r/a/b/n/m/s follow standard matrix-math notation"
)]
fn matrix_elementwise_op(op: BinaryOp, l: &Value, r: &Value) -> Option<Value> {
    if !matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    ) {
        return None;
    }
    let scalar_of = |v: &Value| -> Option<f32> {
        match v {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(scalar_from_i32(*i)),
            _ => None,
        }
    };
    let apply = |a: f32, b: f32| -> f32 {
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            _ => a / b,
        }
    };
    match (mat_slice(l), mat_slice(r)) {
        (Some((a, n)), Some((b, m))) if n == m => {
            // mat·mat linear-algebra multiply is handled by mat_mul before
            // we get here; +, -, / are component-wise.
            if matches!(op, BinaryOp::Mul) {
                return None;
            }
            let elems: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| apply(x, y)).collect();
            Some(mat_from_elems(n, &elems))
        }
        (Some((a, n)), None) => {
            let s = scalar_of(r)?;
            let elems: Vec<f32> = a.iter().map(|&x| apply(x, s)).collect();
            Some(mat_from_elems(n, &elems))
        }
        (None, Some((b, m))) => {
            let s = scalar_of(l)?;
            let elems: Vec<f32> = b.iter().map(|&x| apply(s, x)).collect();
            Some(mat_from_elems(m, &elems))
        }
        _ => None,
    }
}

/// Matrix multiplication: mat * mat, mat * vec, vec * mat.
#[allow(
    clippy::many_single_char_names,
    reason = "faithful linear-algebra code; l/r/v/m/a/b/x/y/z/w/s/j/k follow standard matrix-math notation"
)]
fn mat_mul(l: &Value, r: &Value) -> Option<Value> {
    match (l, r) {
        (Value::Vec2(v), Value::Mat2(m)) => {
            // Row-vector times matrix: out[j] = dot(v, column_j).
            let x = v[0].mul_add(m[0], v[1] * m[1]);
            let y = v[0].mul_add(m[2], v[1] * m[3]);
            Some(Value::Vec2([x, y]))
        }
        (Value::Vec3(v), Value::Mat3(m)) => {
            let mut out = [0.0f32; 3];
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = v[2].mul_add(m[j * 3 + 2], v[0].mul_add(m[j * 3], v[1] * m[j * 3 + 1]));
            }
            Some(Value::Vec3(out))
        }
        (Value::Vec4(v), Value::Mat4(m)) => {
            let mut out = [0.0f32; 4];
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = v[3].mul_add(
                    m[j * 4 + 3],
                    v[2].mul_add(m[j * 4 + 2], v[0].mul_add(m[j * 4], v[1] * m[j * 4 + 1])),
                );
            }
            Some(Value::Vec4(out))
        }
        (Value::Mat2(m), Value::Vec2(v)) => {
            let x = m[0].mul_add(v[0], m[2] * v[1]);
            let y = m[1].mul_add(v[0], m[3] * v[1]);
            Some(Value::Vec2([x, y]))
        }
        (Value::Mat3(m), Value::Vec3(v)) => {
            let x = m[6].mul_add(v[2], m[0].mul_add(v[0], m[3] * v[1]));
            let y = m[7].mul_add(v[2], m[1].mul_add(v[0], m[4] * v[1]));
            let z = m[8].mul_add(v[2], m[2].mul_add(v[0], m[5] * v[1]));
            Some(Value::Vec3([x, y, z]))
        }
        (Value::Mat4(m), Value::Vec4(v)) => {
            let x = m[12].mul_add(v[3], m[8].mul_add(v[2], m[0].mul_add(v[0], m[4] * v[1])));
            let y = m[13].mul_add(v[3], m[9].mul_add(v[2], m[1].mul_add(v[0], m[5] * v[1])));
            let z = m[14].mul_add(v[3], m[10].mul_add(v[2], m[2].mul_add(v[0], m[6] * v[1])));
            let w = m[15].mul_add(v[3], m[11].mul_add(v[2], m[3].mul_add(v[0], m[7] * v[1])));
            Some(Value::Vec4([x, y, z, w]))
        }
        (Value::Mat2(a), Value::Mat2(b)) => {
            let mut out = [0.0f32; 4];
            for col in 0..2 {
                for row in 0..2 {
                    let mut s = 0.0;
                    for k in 0..2 {
                        s = a[k * 2 + row].mul_add(b[col * 2 + k], s);
                    }
                    out[col * 2 + row] = s;
                }
            }
            Some(Value::Mat2(out))
        }
        (Value::Mat3(a), Value::Mat3(b)) => {
            let mut out = [0.0f32; 9];
            for col in 0..3 {
                for row in 0..3 {
                    let mut s = 0.0;
                    for k in 0..3 {
                        s = a[k * 3 + row].mul_add(b[col * 3 + k], s);
                    }
                    out[col * 3 + row] = s;
                }
            }
            Some(Value::Mat3(out))
        }
        (Value::Mat4(a), Value::Mat4(b)) => {
            let mut out = [0.0f32; 16];
            for col in 0..4 {
                for row in 0..4 {
                    let mut s = 0.0;
                    for k in 0..4 {
                        s = a[k * 4 + row].mul_add(b[col * 4 + k], s);
                    }
                    out[col * 4 + row] = s;
                }
            }
            Some(Value::Mat4(out))
        }
        _ => None,
    }
}

fn index_value(v: &Value, i: i32) -> Value {
    let idx = usize::try_from(i).unwrap_or(usize::MAX);
    match v {
        Value::Vec2(a) => Value::Float(*a.get(idx).unwrap_or(&0.0)),
        Value::Vec3(a) => Value::Float(*a.get(idx).unwrap_or(&0.0)),
        Value::Vec4(a) => Value::Float(*a.get(idx).unwrap_or(&0.0)),
        Value::Mat2(m) => {
            if idx < 2 {
                Value::Vec2([m[idx * 2], m[idx * 2 + 1]])
            } else {
                Value::Vec2([0.0, 0.0])
            }
        }
        Value::Mat3(m) => {
            if idx < 3 {
                Value::Vec3([m[idx * 3], m[idx * 3 + 1], m[idx * 3 + 2]])
            } else {
                Value::Vec3([0.0, 0.0, 0.0])
            }
        }
        Value::Mat4(m) => {
            if idx < 4 {
                Value::Vec4([m[idx * 4], m[idx * 4 + 1], m[idx * 4 + 2], m[idx * 4 + 3]])
            } else {
                Value::Vec4([0.0, 0.0, 0.0, 0.0])
            }
        }
        _ => Value::Void,
    }
}

const fn swizzle_index(c: char) -> Option<usize> {
    match c {
        'x' | 'r' | 's' => Some(0),
        'y' | 'g' | 't' => Some(1),
        'z' | 'b' | 'p' => Some(2),
        'w' | 'a' | 'q' => Some(3),
        _ => None,
    }
}

fn swizzle(base: &Value, field: &str) -> Value {
    let chars: Vec<char> = field.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    for c in &chars {
        if let Some(idx) = swizzle_index(*c) {
            out.push(base.component(idx));
        } else {
            return Value::Void;
        }
    }
    Value::from_components(&out)
}

fn flatten_components(vals: &[Value]) -> Vec<f32> {
    let mut comps = Vec::new();
    for v in vals {
        for i in 0..v.component_count() {
            comps.push(v.component(i));
        }
    }
    comps
}

fn construct(ty: &SkslType, args: &[Value]) -> Value {
    let want = match ty {
        SkslType::Float | SkslType::Half => 1,
        SkslType::Int => return Value::Int(args.first().map_or(0, Value::as_i32)),
        SkslType::Bool => return Value::Bool(args.first().is_some_and(Value::as_bool)),
        SkslType::Vec2 | SkslType::Half2 => 2,
        SkslType::Vec3 | SkslType::Half3 => 3,
        SkslType::Vec4 | SkslType::Half4 | SkslType::Mat2 => 4,
        SkslType::Mat3 => 9,
        SkslType::Mat4 => 16,
        _ => return args.first().cloned().unwrap_or(Value::Void),
    };

    // Splat a single scalar across all N components.
    if args.len() == 1 && args[0].component_count() == 1 && want >= 2 {
        let x = args[0].as_f32();
        return match ty {
            SkslType::Vec2 | SkslType::Half2 => Value::Vec2([x, x]),
            SkslType::Vec3 | SkslType::Half3 => Value::Vec3([x, x, x]),
            SkslType::Vec4 | SkslType::Half4 => Value::Vec4([x, x, x, x]),
            SkslType::Mat2 => Value::Mat2([x, 0.0, 0.0, x]),
            SkslType::Mat3 => Value::Mat3([x, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, x]),
            SkslType::Mat4 => Value::Mat4([
                x, 0.0, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, 0.0, x,
            ]),
            _ => Value::Float(x),
        };
    }

    let mut comps = flatten_components(args);
    // vec4(vec3) defaults the alpha slot to 1.0 by convention.
    if want == 4 && comps.len() == 3 {
        comps.push(1.0);
    }
    while comps.len() < want {
        comps.push(0.0);
    }

    match ty {
        SkslType::Float | SkslType::Half => Value::Float(comps[0]),
        SkslType::Vec2 | SkslType::Half2 => Value::Vec2([comps[0], comps[1]]),
        SkslType::Vec3 | SkslType::Half3 => Value::Vec3([comps[0], comps[1], comps[2]]),
        SkslType::Vec4 | SkslType::Half4 => Value::Vec4([comps[0], comps[1], comps[2], comps[3]]),
        SkslType::Mat2 => {
            let mut out = [0.0f32; 4];
            out.copy_from_slice(&comps[..4]);
            Value::Mat2(out)
        }
        SkslType::Mat3 => {
            let mut out = [0.0f32; 9];
            out.copy_from_slice(&comps[..9]);
            Value::Mat3(out)
        }
        SkslType::Mat4 => {
            let mut out = [0.0f32; 16];
            out.copy_from_slice(&comps[..16]);
            Value::Mat4(out)
        }
        _ => Value::Void,
    }
}

/// Apply a unary f32 -> f32 per component.
fn map1(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    match v {
        Value::Float(x) => Value::Float(f(*x)),
        Value::Int(i) => Value::Float(f(scalar_from_i32(*i))),
        Value::Vec2(a) => Value::Vec2([f(a[0]), f(a[1])]),
        Value::Vec3(a) => Value::Vec3([f(a[0]), f(a[1]), f(a[2])]),
        Value::Vec4(a) => Value::Vec4([f(a[0]), f(a[1]), f(a[2]), f(a[3])]),
        _ => v.clone(),
    }
}

/// Apply a binary f32 x f32 -> f32 per component with scalar broadcast.
fn map2(a: &Value, b: &Value, f: impl Fn(f32, f32) -> f32) -> Value {
    let width = vec_width(a).max(vec_width(b));
    if width == 0 {
        return Value::Float(f(a.as_f32(), b.as_f32()));
    }
    let aa = broadcast(a, width);
    let bb = broadcast(b, width);
    let mut out = [0.0f32; 4];
    for i in 0..width {
        out[i] = f(aa[i], bb[i]);
    }
    match width {
        2 => Value::Vec2([out[0], out[1]]),
        3 => Value::Vec3([out[0], out[1], out[2]]),
        4 => Value::Vec4([out[0], out[1], out[2], out[3]]),
        _ => Value::Float(out[0]),
    }
}

/// Apply a ternary function per component with scalar broadcast.
fn map3(a: &Value, b: &Value, c: &Value, f: impl Fn(f32, f32, f32) -> f32) -> Value {
    let width = vec_width(a).max(vec_width(b)).max(vec_width(c));
    if width == 0 {
        return Value::Float(f(a.as_f32(), b.as_f32(), c.as_f32()));
    }
    let aa = broadcast(a, width);
    let bb = broadcast(b, width);
    let cc = broadcast(c, width);
    let mut out = [0.0f32; 4];
    for i in 0..width {
        out[i] = f(aa[i], bb[i], cc[i]);
    }
    match width {
        2 => Value::Vec2([out[0], out[1]]),
        3 => Value::Vec3([out[0], out[1], out[2]]),
        4 => Value::Vec4([out[0], out[1], out[2], out[3]]),
        _ => Value::Float(out[0]),
    }
}

/// Dispatch to a built-in function. Returns None if the name isn't a known
/// builtin, allowing the caller to fall through to user-defined functions.
#[allow(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "one match arm per SkSL builtin function name; splitting would obscure the exhaustive builtin dispatch table. Bindings i/n/d/k/etc. follow GLSL's own builtin parameter names (e.g. reflect(I, N), refract(I, N, eta))"
)]
fn call_builtin(name: &str, args: &[Value]) -> Option<Value> {
    match name {
        "abs" => args.first().map(|v| map1(v, f32::abs)),
        "sign" => args.first().map(|v| {
            map1(v, |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
        }),
        "floor" => args.first().map(|v| map1(v, f32::floor)),
        "ceil" => args.first().map(|v| map1(v, f32::ceil)),
        "fract" => args.first().map(|v| map1(v, |x| x - x.floor())),
        "sin" => args.first().map(|v| map1(v, f32::sin)),
        "cos" => args.first().map(|v| map1(v, f32::cos)),
        "tan" => args.first().map(|v| map1(v, f32::tan)),
        "asin" => args.first().map(|v| map1(v, f32::asin)),
        "acos" => args.first().map(|v| map1(v, f32::acos)),
        "atan" => {
            if args.len() == 2 {
                Some(map2(&args[0], &args[1], f32::atan2))
            } else {
                args.first().map(|v| map1(v, f32::atan))
            }
        }
        "atan2" => {
            if args.len() == 2 {
                Some(map2(&args[0], &args[1], f32::atan2))
            } else {
                None
            }
        }
        "exp" => args.first().map(|v| map1(v, f32::exp)),
        "exp2" => args.first().map(|v| map1(v, f32::exp2)),
        "log" => args.first().map(|v| map1(v, f32::ln)),
        "log2" => args.first().map(|v| map1(v, f32::log2)),
        "sqrt" => args.first().map(|v| map1(v, f32::sqrt)),
        "inversesqrt" => args.first().map(|v| map1(v, |x| 1.0 / x.sqrt())),
        "pow" => {
            if args.len() == 2 {
                Some(map2(&args[0], &args[1], f32::powf))
            } else {
                None
            }
        }
        "mod" => {
            if args.len() == 2 {
                Some(map2(&args[0], &args[1], |a, b| {
                    if b == 0.0 {
                        0.0
                    } else {
                        (a / b).floor().mul_add(-b, a)
                    }
                }))
            } else {
                None
            }
        }
        "min" => {
            if args.len() == 2 {
                Some(map2(&args[0], &args[1], f32::min))
            } else {
                None
            }
        }
        "max" => {
            if args.len() == 2 {
                Some(map2(&args[0], &args[1], f32::max))
            } else {
                None
            }
        }
        "clamp" => {
            if args.len() == 3 {
                Some(map3(&args[0], &args[1], &args[2], |x, lo, hi| {
                    x.max(lo).min(hi)
                }))
            } else {
                None
            }
        }
        "mix" => {
            if args.len() == 3 {
                Some(map3(&args[0], &args[1], &args[2], |a, b, t| {
                    t.mul_add(b - a, a)
                }))
            } else {
                None
            }
        }
        "step" => {
            if args.len() == 2 {
                Some(map2(
                    &args[0],
                    &args[1],
                    |edge, x| {
                        if x < edge { 0.0 } else { 1.0 }
                    },
                ))
            } else {
                None
            }
        }
        "smoothstep" => {
            if args.len() == 3 {
                Some(map3(&args[0], &args[1], &args[2], |e0, e1, x| {
                    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
                    t * t * 2.0f32.mul_add(-t, 3.0)
                }))
            } else {
                None
            }
        }
        "length" => args.first().map(|v| {
            let w = vec_width(v);
            if w == 0 {
                Value::Float(v.as_f32().abs())
            } else {
                let a = broadcast(v, w);
                let s: f32 = a.iter().take(w).map(|&x| x * x).sum();
                Value::Float(s.sqrt())
            }
        }),
        "distance" => {
            if args.len() == 2 {
                let diff = apply_binary(BinaryOp::Sub, &args[0], &args[1]);
                call_builtin("length", &[diff])
            } else {
                None
            }
        }
        "dot" => {
            if args.len() == 2 {
                let w = vec_width(&args[0]).max(vec_width(&args[1]));
                if w == 0 {
                    Some(Value::Float(args[0].as_f32() * args[1].as_f32()))
                } else {
                    let a = broadcast(&args[0], w);
                    let b = broadcast(&args[1], w);
                    let mut s = 0.0;
                    for i in 0..w {
                        s = a[i].mul_add(b[i], s);
                    }
                    Some(Value::Float(s))
                }
            } else {
                None
            }
        }
        "cross" => {
            if args.len() == 2 {
                let a = args[0].as_vec3();
                let b = args[1].as_vec3();
                Some(Value::Vec3([
                    a[1].mul_add(b[2], -(a[2] * b[1])),
                    a[2].mul_add(b[0], -(a[0] * b[2])),
                    a[0].mul_add(b[1], -(a[1] * b[0])),
                ]))
            } else {
                None
            }
        }
        "normalize" => args.first().map(|v| {
            let w = vec_width(v);
            if w == 0 {
                let x = v.as_f32();
                Value::Float(if x == 0.0 {
                    0.0
                } else if x > 0.0 {
                    1.0
                } else {
                    -1.0
                })
            } else {
                let a = broadcast(v, w);
                let s: f32 = a.iter().take(w).map(|&x| x * x).sum();
                let len = s.sqrt();
                let mut out = [0.0f32; 4];
                if len > 0.0 {
                    for (slot, x) in out.iter_mut().zip(a.iter()).take(w) {
                        *slot = x / len;
                    }
                }
                match w {
                    2 => Value::Vec2([out[0], out[1]]),
                    3 => Value::Vec3([out[0], out[1], out[2]]),
                    4 => Value::Vec4([out[0], out[1], out[2], out[3]]),
                    _ => Value::Float(out[0]),
                }
            }
        }),
        "reflect" => {
            if args.len() == 2 {
                let i = &args[0];
                let n = &args[1];
                let d = call_builtin("dot", &[n.clone(), i.clone()])?;
                let k = apply_binary(BinaryOp::Mul, &Value::Float(2.0), &d);
                let kn = apply_binary(BinaryOp::Mul, &k, n);
                Some(apply_binary(BinaryOp::Sub, i, &kn))
            } else {
                None
            }
        }
        "refract" => {
            if args.len() == 3 {
                let i = &args[0];
                let n = &args[1];
                let eta = args[2].as_f32();
                let d = call_builtin("dot", &[n.clone(), i.clone()])?.as_f32();
                let k = (eta * eta).mul_add(-d.mul_add(-d, 1.0), 1.0);
                if k < 0.0 {
                    let w = vec_width(i);
                    return Some(match w {
                        2 => Value::Vec2([0.0, 0.0]),
                        3 => Value::Vec3([0.0, 0.0, 0.0]),
                        4 => Value::Vec4([0.0, 0.0, 0.0, 0.0]),
                        _ => Value::Float(0.0),
                    });
                }
                let eta_i = apply_binary(BinaryOp::Mul, &Value::Float(eta), i);
                let scale = Value::Float(eta.mul_add(d, k.sqrt()));
                let scaled_n = apply_binary(BinaryOp::Mul, &scale, n);
                Some(apply_binary(BinaryOp::Sub, &eta_i, &scaled_n))
            } else {
                None
            }
        }
        "radians" => args.first().map(|v| map1(v, f32::to_radians)),
        "degrees" => args.first().map(|v| map1(v, f32::to_degrees)),
        // Free-standing type coercion helpers. These appear as calls rather
        // than Constructor nodes when the source names are treated as idents
        // (e.g. `float(x)` inside an expression that reused a variable name).
        "float" => args.first().map(|v| Value::Float(v.as_f32())),
        "int" => args.first().map(|v| Value::Int(v.as_i32())),
        "bool" => args.first().map(|v| Value::Bool(v.as_bool())),
        _ => None,
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert exact expected interpreter output values, not tolerance comparisons"
)]
mod tests {
    use super::*;
    use crate::sksl::Parser;

    fn run_main(source: &str, arg: Value) -> Value {
        let mut parser = Parser::new(source);
        let program = parser.parse_program().unwrap();
        let mut interp = Interp::new();
        for f in &program.functions {
            interp.functions.insert(f.name.clone(), f);
        }
        interp.run_function("main", vec![arg])
    }

    // --- Conformance regression tests (Task 3) ---

    #[test]
    fn multi_component_swizzle_assignment() {
        // `c.rgb *= 0.5` must work for any swizzle lvalue.
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                vec4 c = vec4(1.0, 0.8, 0.6, 1.0);
                c.rgb *= 0.5;
                return c;
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 0.5).abs() < 1e-5, "r halved, got {}", out[0]);
        assert!((out[1] - 0.4).abs() < 1e-5, "g halved, got {}", out[1]);
        assert!((out[2] - 0.3).abs() < 1e-5, "b halved, got {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-5, "a untouched, got {}", out[3]);
    }

    #[test]
    fn multi_component_swizzle_assignment_shuffled() {
        // Swizzle writes must land in the right slots even out of order.
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                vec4 c = vec4(0.0);
                c.ba = vec2(0.25, 0.75);
                return c;
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!((out[2] - 0.25).abs() < 1e-5, "b, got {}", out[2]);
        assert!((out[3] - 0.75).abs() < 1e-5, "a, got {}", out[3]);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn matrix_plus_matrix() {
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                mat2 a = mat2(1.0, 2.0, 3.0, 4.0);
                mat2 b = mat2(0.5, 0.5, 0.5, 0.5);
                mat2 c = a + b;
                return vec4(c[0].x, c[0].y, c[1].x, c[1].y);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        assert_eq!(v.as_vec4(), [1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn matrix_minus_matrix() {
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                mat2 a = mat2(1.0, 2.0, 3.0, 4.0);
                mat2 c = a - mat2(1.0, 1.0, 1.0, 1.0);
                return vec4(c[0].x, c[0].y, c[1].x, c[1].y);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        assert_eq!(v.as_vec4(), [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn matrix_times_scalar_both_orders() {
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                mat2 a = mat2(1.0, 2.0, 3.0, 4.0);
                mat2 b = a * 2.0;
                mat2 c = 0.5 * a;
                return vec4(b[0].x, b[1].y, c[0].x, c[1].y);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        assert_eq!(v.as_vec4(), [2.0, 8.0, 0.5, 2.0]);
    }

    #[test]
    fn vector_times_matrix() {
        // Row-vector times matrix: (v * M)[i] = dot(v, M[i]) (column i).
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                mat2 m = mat2(1.0, 2.0, 3.0, 4.0);
                vec2 v = vec2(1.0, 1.0) * m;
                return vec4(v.x, v.y, 0.0, 1.0);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        // columns are (1,2) and (3,4): dot((1,1),(1,2)) = 3, dot((1,1),(3,4)) = 7
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 7.0);
    }

    #[test]
    fn matrix_times_vector() {
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                mat2 m = mat2(1.0, 2.0, 3.0, 4.0);
                vec2 v = m * vec2(1.0, 1.0);
                return vec4(v.x, v.y, 0.0, 1.0);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        // M*v: rows dotted with v: (1*1 + 3*1, 2*1 + 4*1) = (4, 6)
        assert_eq!(out[0], 4.0);
        assert_eq!(out[1], 6.0);
    }

    #[test]
    fn float_division_by_zero_is_ieee() {
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                float pos = 1.0 / 0.0;
                float neg = -1.0 / 0.0;
                float nan = 0.0 / 0.0;
                return vec4(pos, neg, nan, 1.0);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!(
            out[0].is_infinite() && out[0] > 0.0,
            "1/0 = +inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] < 0.0,
            "-1/0 = -inf, got {}",
            out[1]
        );
        assert!(out[2].is_nan(), "0/0 = NaN, got {}", out[2]);
    }

    #[test]
    fn float_mod_by_zero_is_nan() {
        let v = run_main(
            "vec4 main(vec2 c) { float m = mod(1.0, 0.0); float n = 1.0 % 0.0; return vec4(m, n, 0.0, 1.0); }",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!(out[1].is_nan(), "1.0 % 0.0 must be NaN, got {}", out[1]);
    }

    #[test]
    fn bitwise_ops_parse_with_glsl_precedence() {
        let v = run_main(
            r"
            vec4 main(vec2 coord) {
                int a = 6 & 3;        // 2
                int b = 6 ^ 3;        // 5
                int c = 1 << 2 | 1;   // shifts bind tighter than |: 5
                int d = 4 >> 1 + 1;   // shifts below additive: 4 >> 2 = 1
                return vec4(float(a), float(b), float(c), float(d));
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        assert_eq!(v.as_vec4(), [2.0, 5.0, 5.0, 1.0]);
    }

    #[test]
    fn bitwise_chain_precedence() {
        // & above ^ above |: 6 & 3 ^ 5 | 8 == ((6 & 3) ^ 5) | 8 == 15
        let v = run_main(
            "vec4 main(vec2 c) { int x = 6 & 3 ^ 5 | 8; return vec4(float(x), 0.0, 0.0, 1.0); }",
            Value::Vec2([0.0, 0.0]),
        );
        assert_eq!(v.as_vec4()[0], 15.0);
    }

    #[test]
    fn half_scalar_constructor() {
        let v = run_main(
            "vec4 main(vec2 c) { return vec4(half(0.5), 0.0, 0.0, 1.0); }",
            Value::Vec2([0.0, 0.0]),
        );
        assert!((v.as_vec4()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn constant_return() {
        let v = run_main(
            "vec4 main(vec2 c) { return vec4(1.0, 0.5, 0.25, 1.0); }",
            Value::Vec2([0.0, 0.0]),
        );
        assert_eq!(v.as_vec4(), [1.0, 0.5, 0.25, 1.0]);
    }

    #[test]
    fn arithmetic_and_swizzle() {
        let v = run_main(
            "vec4 main(vec2 c) { vec2 uv = c * 2.0 + vec2(0.1, 0.2); return vec4(uv.x, uv.y, 0.0, 1.0); }",
            Value::Vec2([0.5, 0.25]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 1.1).abs() < 1e-5);
        assert!((out[1] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn if_statement() {
        let v = run_main(
            "vec4 main(vec2 c) { if (c.x > 0.5) { return vec4(1.0, 0.0, 0.0, 1.0); } return vec4(0.0, 1.0, 0.0, 1.0); }",
            Value::Vec2([0.8, 0.0]),
        );
        assert_eq!(v.as_vec4(), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn for_loop_sum() {
        let v = run_main(
            r"
            vec4 main(vec2 c) {
                float sum = 0.0;
                for (int i = 0; i < 4; i = i + 1) {
                    sum = sum + 0.25;
                }
                return vec4(sum, 0.0, 0.0, 1.0);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn builtin_mix() {
        let v = run_main(
            "vec4 main(vec2 c) { return mix(vec4(0.0), vec4(1.0), 0.25); }",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 0.25).abs() < 1e-5);
        assert!((out[3] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn builtin_length() {
        let v = run_main(
            "vec4 main(vec2 c) { float l = length(vec2(3.0, 4.0)); return vec4(l, 0.0, 0.0, 1.0); }",
            Value::Vec2([0.0, 0.0]),
        );
        assert!((v.as_vec4()[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn filter_color_inverts() {
        let v = run_main(
            "vec4 main(vec4 color) { return vec4(1.0 - color.x, 1.0 - color.y, 1.0 - color.z, color.w); }",
            Value::Vec4([0.2, 0.3, 0.4, 1.0]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 0.8).abs() < 1e-5);
        assert!((out[1] - 0.7).abs() < 1e-5);
        assert!((out[2] - 0.6).abs() < 1e-5);
        assert!((out[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn compound_assign_and_swizzle_write() {
        let v = run_main(
            r"
            vec4 main(vec2 c) {
                vec4 col = vec4(0.0, 0.0, 0.0, 1.0);
                col.x = 0.5;
                col.y += 0.25;
                col.z = col.x * 2.0;
                return col;
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.25).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
        assert!((out[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ternary_and_clamp() {
        let v = run_main(
            r"
            vec4 main(vec2 c) {
                float x = c.x > 0.5 ? 1.0 : 0.0;
                float y = clamp(c.y, 0.2, 0.8);
                return vec4(x, y, 0.0, 1.0);
            }
            ",
            Value::Vec2([0.7, 0.1]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn mat2_times_vec2() {
        let v = run_main(
            r"
            vec4 main(vec2 c) {
                mat2 m = mat2(1.0, 0.0, 0.0, 1.0);
                vec2 r = m * vec2(2.0, 3.0);
                return vec4(r.x, r.y, 0.0, 1.0);
            }
            ",
            Value::Vec2([0.0, 0.0]),
        );
        let out = v.as_vec4();
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn user_function_call() {
        let v = run_main(
            r"
            float square(float x) { return x * x; }
            vec4 main(vec2 c) {
                float s = square(c.x);
                return vec4(s, 0.0, 0.0, 1.0);
            }
            ",
            Value::Vec2([0.5, 0.0]),
        );
        assert!((v.as_vec4()[0] - 0.25).abs() < 1e-5);
    }
}
