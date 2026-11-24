//! Expression evaluation for Lottie animations.
//!
//! This module provides a subset of Lottie expression support:
//! - Basic math operations
//! - Property access
//! - Time-based functions
//! - Simple conditionals

use skia_rs_core::Scalar;
use std::collections::HashMap;

/// Expression value types.
#[derive(Debug, Clone, Default)]
pub enum Value {
    /// Numeric value.
    Number(Scalar),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
    /// Array/vector value.
    Array(Vec<Scalar>),
    /// Null/undefined.
    #[default]
    Null,
}

impl Value {
    /// Get as number.
    #[must_use]
    pub const fn as_number(&self) -> Option<Scalar> {
        match self {
            Self::Number(n) => Some(*n),
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Get as bool.
    #[must_use]
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Number(n) => *n != 0.0,
            Self::String(s) => !s.is_empty(),
            Self::Array(a) => !a.is_empty(),
            Self::Null => false,
        }
    }

    /// Get as array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Scalar]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// Expression evaluation context.
#[derive(Debug, Default)]
pub struct ExpressionContext {
    /// Current time (seconds).
    pub time: Scalar,
    /// Current frame.
    pub frame: Scalar,
    /// Frame rate.
    pub fps: Scalar,
    /// Composition width.
    pub width: Scalar,
    /// Composition height.
    pub height: Scalar,
    /// Variables.
    pub variables: HashMap<String, Value>,
    /// Property cache.
    property_cache: HashMap<String, Value>,
}

impl ExpressionContext {
    /// Create a new expression context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the time.
    pub fn set_time(&mut self, time: Scalar, fps: Scalar) {
        self.time = time;
        self.frame = time * fps;
        self.fps = fps;
    }

    /// Set the composition size.
    pub const fn set_size(&mut self, width: Scalar, height: Scalar) {
        self.width = width;
        self.height = height;
    }

    /// Set a variable.
    pub fn set_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable.
    #[must_use]
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    /// Cache a property value.
    pub fn cache_property(&mut self, path: &str, value: Value) {
        self.property_cache.insert(path.to_string(), value);
    }

    /// Get a cached property.
    #[must_use]
    pub fn get_cached_property(&self, path: &str) -> Option<&Value> {
        self.property_cache.get(path)
    }
}

/// A simple expression evaluator.
#[derive(Debug)]
pub struct ExpressionEvaluator {
    /// Expression source.
    source: String,
}

impl ExpressionEvaluator {
    /// Create a new evaluator.
    #[must_use]
    pub fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
        }
    }

    /// Evaluate the expression.
    #[must_use]
    pub fn evaluate(&self, ctx: &ExpressionContext) -> Value {
        // Simple expression parser
        let source = self.source.trim();

        // Check for simple built-in functions
        if source == "time" {
            return Value::Number(ctx.time);
        }
        if source == "frame" {
            return Value::Number(ctx.frame);
        }

        // Check for math operations
        if let Some(result) = Self::evaluate_math(source, ctx) {
            return result;
        }

        // Check for function calls
        if let Some(result) = Self::evaluate_function(source, ctx) {
            return result;
        }

        // Try to parse as number
        if let Ok(n) = source.parse::<Scalar>() {
            return Value::Number(n);
        }

        // Try to parse as array
        if source.starts_with('[') && source.ends_with(']') {
            let inner = &source[1..source.len() - 1];
            let values: Vec<Scalar> = inner
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            return Value::Array(values);
        }

        // Variable lookup
        if let Some(value) = ctx.get_variable(source) {
            return value.clone();
        }

        Value::Null
    }

    /// Evaluate math operations.
    fn evaluate_math(source: &str, ctx: &ExpressionContext) -> Option<Value> {
        // Very simple math parsing
        // Format: "a + b", "a - b", "a * b", "a / b"

        for op in &[" + ", " - ", " * ", " / ", " % "] {
            if let Some(pos) = source.find(op) {
                let left = source[..pos].trim();
                let right = source[pos + op.len()..].trim();

                let left_val = Self::new(left).evaluate(ctx);
                let right_val = Self::new(right).evaluate(ctx);

                if let (Some(a), Some(b)) = (left_val.as_number(), right_val.as_number()) {
                    let result = match *op {
                        " + " => a + b,
                        " - " => a - b,
                        " * " => a * b,
                        " / " => {
                            if b == 0.0 {
                                0.0
                            } else {
                                a / b
                            }
                        }
                        " % " => {
                            if b == 0.0 {
                                0.0
                            } else {
                                a % b
                            }
                        }
                        _ => return None,
                    };
                    return Some(Value::Number(result));
                }
            }
        }

        None
    }

    /// Evaluate function calls.
    fn evaluate_function(source: &str, ctx: &ExpressionContext) -> Option<Value> {
        // Parse function call: name(args)
        if let Some(paren_pos) = source.find('(')
            && source.ends_with(')')
        {
            let name = &source[..paren_pos];
            let args_str = &source[paren_pos + 1..source.len() - 1];
            let args: Vec<&str> = args_str.split(',').map(str::trim).collect();

            return Self::call_function(name, &args, ctx);
        }

        None
    }

    /// Call a built-in function.
    #[allow(
        clippy::too_many_lines,
        reason = "single dispatch table over AE expression built-in functions; splitting would fragment a single logical match mirroring upstream skottie expression evaluation"
    )]
    fn call_function(name: &str, args: &[&str], ctx: &ExpressionContext) -> Option<Value> {
        match name {
            // Math functions
            "Math.sin" | "sin" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.sin()))
            }
            "Math.cos" | "cos" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.cos()))
            }
            "Math.tan" | "tan" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.tan()))
            }
            "Math.abs" | "abs" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.abs()))
            }
            "Math.floor" | "floor" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.floor()))
            }
            "Math.ceil" | "ceil" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.ceil()))
            }
            "Math.round" | "round" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.round()))
            }
            "Math.sqrt" | "sqrt" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                Some(Value::Number(arg.as_number()?.sqrt()))
            }
            "Math.pow" | "pow" => {
                if args.len() >= 2 {
                    let base = Self::new(args[0]).evaluate(ctx);
                    let exp = Self::new(args[1]).evaluate(ctx);
                    Some(Value::Number(base.as_number()?.powf(exp.as_number()?)))
                } else {
                    None
                }
            }
            "Math.min" | "min" => {
                if args.len() >= 2 {
                    let a = Self::new(args[0]).evaluate(ctx);
                    let b = Self::new(args[1]).evaluate(ctx);
                    Some(Value::Number(a.as_number()?.min(b.as_number()?)))
                } else {
                    None
                }
            }
            "Math.max" | "max" => {
                if args.len() >= 2 {
                    let a = Self::new(args[0]).evaluate(ctx);
                    let b = Self::new(args[1]).evaluate(ctx);
                    Some(Value::Number(a.as_number()?.max(b.as_number()?)))
                } else {
                    None
                }
            }

            // Lottie-specific functions
            "linear" => {
                if args.len() >= 5 {
                    let t = Self::new(args[0]).evaluate(ctx).as_number()?;
                    let t_min = Self::new(args[1]).evaluate(ctx).as_number()?;
                    let t_max = Self::new(args[2]).evaluate(ctx).as_number()?;
                    let v_min = Self::new(args[3]).evaluate(ctx).as_number()?;
                    let v_max = Self::new(args[4]).evaluate(ctx).as_number()?;

                    let normalized = (t - t_min) / (t_max - t_min);
                    let clamped = normalized.clamp(0.0, 1.0);
                    {
                        #[allow(
                            clippy::suboptimal_flops,
                            reason = "the interpolation as originally written (unfused)"
                        )]
                        let value = v_min + clamped * (v_max - v_min);
                        Some(Value::Number(value))
                    }
                } else {
                    None
                }
            }
            "ease" | "easeIn" | "easeOut" | "easeInOut" => {
                if args.len() >= 5 {
                    let t = Self::new(args[0]).evaluate(ctx).as_number()?;
                    let t_min = Self::new(args[1]).evaluate(ctx).as_number()?;
                    let t_max = Self::new(args[2]).evaluate(ctx).as_number()?;
                    let v_min = Self::new(args[3]).evaluate(ctx).as_number()?;
                    let v_max = Self::new(args[4]).evaluate(ctx).as_number()?;

                    let normalized = ((t - t_min) / (t_max - t_min)).clamp(0.0, 1.0);
                    let eased = match name {
                        "easeIn" => normalized * normalized,
                        "easeOut" => (1.0 - normalized).mul_add(-(1.0 - normalized), 1.0),
                        "easeInOut" => {
                            if normalized < 0.5 {
                                2.0 * normalized * normalized
                            } else {
                                1.0 - (-2.0f32).mul_add(normalized, 2.0).powi(2) / 2.0
                            }
                        }
                        _ => normalized, // Default to linear
                    };
                    {
                        #[allow(
                            clippy::suboptimal_flops,
                            reason = "the interpolation as originally written (unfused)"
                        )]
                        let value = v_min + eased * (v_max - v_min);
                        Some(Value::Number(value))
                    }
                } else {
                    None
                }
            }
            "clamp" => {
                if args.len() >= 3 {
                    let value = Self::new(args[0]).evaluate(ctx).as_number()?;
                    let min = Self::new(args[1]).evaluate(ctx).as_number()?;
                    let max = Self::new(args[2]).evaluate(ctx).as_number()?;
                    Some(Value::Number(value.clamp(min, max)))
                } else {
                    None
                }
            }
            "random" => {
                // Simple pseudo-random based on time
                let seed = if args.is_empty() {
                    ctx.time
                } else {
                    Self::new(args[0]).evaluate(ctx).as_number().unwrap_or(0.0)
                };
                Some(Value::Number(pseudo_random(seed)))
            }
            "wiggle" => {
                if args.len() >= 2 {
                    let freq = Self::new(args[0]).evaluate(ctx).as_number()?;
                    let amp = Self::new(args[1]).evaluate(ctx).as_number()?;

                    // Simple wiggle approximation
                    let t = ctx.time * freq;
                    let noise = (t * 5.7)
                        .sin()
                        .mul_add(0.2, t.sin().mul_add(0.5, (t * 2.3).cos() * 0.3));
                    Some(Value::Number(noise * amp))
                } else {
                    None
                }
            }
            "loopOut" | "loopIn" => {
                // Simplified loop - just return time
                Some(Value::Number(ctx.time))
            }

            // Vector functions
            "length" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                arg.as_array().map_or_else(
                    || arg.as_number().map(|n| Value::Number(n.abs())),
                    |arr| {
                        let sum: Scalar = arr.iter().map(|x| x * x).sum();
                        Some(Value::Number(sum.sqrt()))
                    },
                )
            }
            "normalize" => {
                let arg = Self::new(args.first()?).evaluate(ctx);
                arg.as_array().map_or(Some(Value::Number(1.0)), |arr| {
                    let len: Scalar = arr.iter().map(|x| x * x).sum::<Scalar>().sqrt();
                    if len > 0.0 {
                        let normalized: Vec<Scalar> = arr.iter().map(|x| x / len).collect();
                        Some(Value::Array(normalized))
                    } else {
                        Some(arg.clone())
                    }
                })
            }

            _ => None,
        }
    }
}

/// Simple pseudo-random function.
fn pseudo_random(seed: Scalar) -> Scalar {
    let x = seed.mul_add(12.9898, 78.233).sin() * 43_758.547;
    x - x.floor()
}

/// Expression compiler (for performance optimization).
#[derive(Debug, Default)]
pub struct ExpressionCompiler {
    /// Cached compiled expressions.
    cache: HashMap<String, CompiledExpression>,
}

impl ExpressionCompiler {
    /// Create a new compiler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compile an expression.
    pub fn compile(&mut self, source: &str) -> &CompiledExpression {
        self.cache
            .entry(source.to_string())
            .or_insert_with(|| CompiledExpression::new(source))
    }

    /// Evaluate a cached expression.
    pub fn evaluate(&mut self, source: &str, ctx: &ExpressionContext) -> Value {
        self.compile(source).evaluate(ctx)
    }
}

/// A compiled expression.
#[derive(Debug)]
pub struct CompiledExpression {
    /// The evaluator.
    evaluator: ExpressionEvaluator,
}

impl CompiledExpression {
    /// Create a new compiled expression.
    #[must_use]
    pub fn new(source: &str) -> Self {
        Self {
            evaluator: ExpressionEvaluator::new(source),
        }
    }

    /// Evaluate the expression.
    #[must_use]
    pub fn evaluate(&self, ctx: &ExpressionContext) -> Value {
        self.evaluator.evaluate(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_number() {
        let ctx = ExpressionContext::new();
        let eval = ExpressionEvaluator::new("42");

        let result = eval.evaluate(&ctx);
        assert_eq!(result.as_number(), Some(42.0));
    }

    #[test]
    fn test_time_variable() {
        let mut ctx = ExpressionContext::new();
        ctx.set_time(2.5, 30.0);

        let eval = ExpressionEvaluator::new("time");
        let result = eval.evaluate(&ctx);
        assert_eq!(result.as_number(), Some(2.5));
    }

    #[test]
    fn test_math_operations() {
        let ctx = ExpressionContext::new();

        let eval = ExpressionEvaluator::new("10 + 5");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(15.0));

        let eval = ExpressionEvaluator::new("10 - 5");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(5.0));

        let eval = ExpressionEvaluator::new("10 * 5");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(50.0));

        let eval = ExpressionEvaluator::new("10 / 5");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(2.0));
    }

    #[test]
    fn test_math_functions() {
        let ctx = ExpressionContext::new();

        let eval = ExpressionEvaluator::new("abs(-5)");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(5.0));

        let eval = ExpressionEvaluator::new("floor(3.7)");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(3.0));

        let eval = ExpressionEvaluator::new("ceil(3.2)");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(4.0));
    }

    #[test]
    fn test_linear_interpolation() {
        let ctx = ExpressionContext::new();

        let eval = ExpressionEvaluator::new("linear(0.5, 0, 1, 0, 100)");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(50.0));
    }

    #[test]
    fn test_clamp() {
        let ctx = ExpressionContext::new();

        let eval = ExpressionEvaluator::new("clamp(150, 0, 100)");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(100.0));

        let eval = ExpressionEvaluator::new("clamp(-50, 0, 100)");
        assert_eq!(eval.evaluate(&ctx).as_number(), Some(0.0));
    }

    #[test]
    fn test_array_value() {
        let ctx = ExpressionContext::new();

        let eval = ExpressionEvaluator::new("[1, 2, 3]");
        let result = eval.evaluate(&ctx);
        assert_eq!(result.as_array(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn test_expression_compiler() {
        let mut compiler = ExpressionCompiler::new();
        let ctx = ExpressionContext::new();

        let result = compiler.evaluate("10 + 5", &ctx);
        assert_eq!(result.as_number(), Some(15.0));

        // Second call should use cache
        let result = compiler.evaluate("10 + 5", &ctx);
        assert_eq!(result.as_number(), Some(15.0));
    }
}
