//! SVG path data parsing.
//!
//! Parses SVG path `d` attribute strings into a `Path`.

use crate::{Path, PathBuilder};
use skia_rs_core::Scalar;

/// Parse an SVG path data string.
///
/// # Example
/// ```
/// use skia_rs_path::parse_svg_path;
///
/// let path = parse_svg_path("M 10 10 L 100 100 Z").unwrap();
/// assert!(!path.is_empty());
/// ```
///
/// # Errors
///
/// Returns [`SvgPathError`] if `d` is not well-formed SVG path data (e.g. an
/// unknown command, a malformed number, or a missing initial `moveto`).
pub fn parse_svg_path(d: &str) -> Result<Path, SvgPathError> {
    let parser = SvgPathParser::new(d);
    parser.parse()
}

/// Error type for SVG path parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvgPathError {
    /// Unexpected end of input.
    UnexpectedEnd,
    /// Invalid number format.
    InvalidNumber(String),
    /// Unknown command.
    UnknownCommand(char),
    /// Expected a number.
    ExpectedNumber,
    /// Missing move command at start.
    MissingMoveTo,
    /// Numeric data appeared where a command was expected (e.g. directly after `Z`).
    UnexpectedNumber,
}

impl std::fmt::Display for SvgPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of path data"),
            Self::InvalidNumber(s) => write!(f, "invalid number: {s}"),
            Self::UnknownCommand(c) => write!(f, "unknown command: {c}"),
            Self::ExpectedNumber => write!(f, "expected a number"),
            Self::MissingMoveTo => write!(f, "path must start with moveto"),
            Self::UnexpectedNumber => {
                write!(f, "unexpected number where a command was expected")
            }
        }
    }
}

impl std::error::Error for SvgPathError {}

struct SvgPathParser<'a> {
    input: &'a str,
    pos: usize,
    builder: PathBuilder,
    last_control: Option<(Scalar, Scalar)>,
    has_move: bool,
    /// The uppercased previous command letter, used to gate S/T reflection and
    /// to reject numeric data directly after `Z` (like `SkParsePath`'s
    /// `previousOp`). `'\0'` means no command has executed yet.
    prev_op: char,
}

impl<'a> SvgPathParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            builder: PathBuilder::new(),
            last_control: None,
            has_move: false,
            prev_op: '\0',
        }
    }

    fn parse(mut self) -> Result<Path, SvgPathError> {
        self.skip_whitespace();

        while !self.is_end() {
            let cmd = self.parse_command()?;
            self.execute_command(cmd)?;
            self.skip_whitespace();
        }

        Ok(self.builder.build())
    }

    const fn is_end(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() || c == ',' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse_command(&mut self) -> Result<char, SvgPathError> {
        self.skip_whitespace();
        let cmd = self.peek().ok_or(SvgPathError::UnexpectedEnd)?;

        if cmd.is_ascii_alphabetic() {
            self.advance();
            Ok(cmd)
        } else if !self.has_move {
            Err(SvgPathError::MissingMoveTo)
        } else if self.prev_op == 'Z' {
            // Numeric data directly after a close is a parse error (upstream
            // rejects the path rather than starting an implicit lineto).
            Err(SvgPathError::UnexpectedNumber)
        } else {
            // Implicit lineto
            Ok('L')
        }
    }

    fn parse_number(&mut self) -> Result<Scalar, SvgPathError> {
        self.skip_whitespace();

        let start = self.pos;
        let mut has_dot = false;
        let mut has_exp = false;

        // Handle sign
        if let Some(c) = self.peek()
            && (c == '+' || c == '-')
        {
            self.advance();
        }

        // Parse digits and decimal point
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else if c == '.' && !has_dot && !has_exp {
                has_dot = true;
                self.advance();
            } else if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                self.advance();
                // Handle exponent sign
                if let Some(next) = self.peek()
                    && (next == '+' || next == '-')
                {
                    self.advance();
                }
            } else {
                break;
            }
        }

        if start == self.pos {
            return Err(SvgPathError::ExpectedNumber);
        }

        let num_str = &self.input[start..self.pos];
        num_str
            .parse()
            .map_err(|_| SvgPathError::InvalidNumber(num_str.to_string()))
    }

    fn parse_flag(&mut self) -> Result<bool, SvgPathError> {
        self.skip_whitespace();
        match self.peek() {
            Some('0') => {
                self.advance();
                Ok(false)
            }
            Some('1') => {
                self.advance();
                Ok(true)
            }
            _ => Err(SvgPathError::ExpectedNumber),
        }
    }

    fn current_point(&self) -> (Scalar, Scalar) {
        let p = self.builder.current_point();
        (p.x, p.y)
    }

    fn execute_command(&mut self, cmd: char) -> Result<(), SvgPathError> {
        let is_relative = cmd.is_ascii_lowercase();
        let cmd_upper = cmd.to_ascii_uppercase();
        let prev_op = self.prev_op;

        let result = match cmd_upper {
            'M' => self.parse_moveto(is_relative),
            'L' => self.parse_lineto(is_relative),
            'H' => self.parse_horizontal_lineto(is_relative),
            'V' => self.parse_vertical_lineto(is_relative),
            'C' => self.parse_curveto(is_relative),
            'S' => self.parse_smooth_curveto(is_relative, prev_op),
            'Q' => self.parse_quadto(is_relative),
            'T' => self.parse_smooth_quadto(is_relative, prev_op),
            'A' => self.parse_arcto(is_relative),
            'Z' => {
                self.builder.close();
                self.last_control = None;
                Ok(())
            }
            _ => Err(SvgPathError::UnknownCommand(cmd)),
        };
        result?;
        self.prev_op = cmd_upper;
        Ok(())
    }

    fn parse_moveto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        let mut first = true;
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (x, y) = if is_relative && self.has_move {
                let (cx, cy) = self.current_point();
                (cx + x, cy + y)
            } else {
                (x, y)
            };

            if first {
                self.builder.move_to(x, y);
                self.has_move = true;
                first = false;
            } else {
                self.builder.line_to(x, y);
            }
        }
        self.last_control = None;
        Ok(())
    }

    fn parse_lineto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (x, y) = if is_relative {
                let (cx, cy) = self.current_point();
                (cx + x, cy + y)
            } else {
                (x, y)
            };

            self.builder.line_to(x, y);
        }
        self.last_control = None;
        Ok(())
    }

    fn parse_horizontal_lineto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x = self.parse_number()?;
            let (cx, cy) = self.current_point();

            let x = if is_relative { cx + x } else { x };
            self.builder.line_to(x, cy);
        }
        self.last_control = None;
        Ok(())
    }

    fn parse_vertical_lineto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let y = self.parse_number()?;
            let (cx, cy) = self.current_point();

            let y = if is_relative { cy + y } else { y };
            self.builder.line_to(cx, y);
        }
        self.last_control = None;
        Ok(())
    }

    fn parse_curveto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x1 = self.parse_number()?;
            let y1 = self.parse_number()?;
            let x2 = self.parse_number()?;
            let y2 = self.parse_number()?;
            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (x1, y1, x2, y2, x, y) = if is_relative {
                let (cx, cy) = self.current_point();
                (cx + x1, cy + y1, cx + x2, cy + y2, cx + x, cy + y)
            } else {
                (x1, y1, x2, y2, x, y)
            };

            self.builder.cubic_to(x1, y1, x2, y2, x, y);
            self.last_control = Some((x2, y2));
        }
        Ok(())
    }

    fn parse_smooth_curveto(
        &mut self,
        is_relative: bool,
        prev_op: char,
    ) -> Result<(), SvgPathError> {
        let mut iteration = 0;
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x2 = self.parse_number()?;
            let y2 = self.parse_number()?;
            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (cx, cy) = self.current_point();
            // S reflects the previous control point only when the previous
            // command was a cubic (C or S); within a run, subsequent segments
            // reflect off the prior S. Otherwise the control point is the
            // current point (SVG 1.1 / SkParsePath).
            let reflect = iteration > 0 || prev_op == 'C' || prev_op == 'S';
            let (x1, y1) = match (reflect, self.last_control) {
                (true, Some((lx, ly))) => (2.0f32.mul_add(cx, -lx), 2.0f32.mul_add(cy, -ly)),
                _ => (cx, cy),
            };
            iteration += 1;

            let (x2, y2, x, y) = if is_relative {
                (cx + x2, cy + y2, cx + x, cy + y)
            } else {
                (x2, y2, x, y)
            };

            self.builder.cubic_to(x1, y1, x2, y2, x, y);
            self.last_control = Some((x2, y2));
        }
        Ok(())
    }

    fn parse_quadto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x1 = self.parse_number()?;
            let y1 = self.parse_number()?;
            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (x1, y1, x, y) = if is_relative {
                let (cx, cy) = self.current_point();
                (cx + x1, cy + y1, cx + x, cy + y)
            } else {
                (x1, y1, x, y)
            };

            self.builder.quad_to(x1, y1, x, y);
            self.last_control = Some((x1, y1));
        }
        Ok(())
    }

    fn parse_smooth_quadto(
        &mut self,
        is_relative: bool,
        prev_op: char,
    ) -> Result<(), SvgPathError> {
        let mut iteration = 0;
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (cx, cy) = self.current_point();
            // T reflects the previous control point only when the previous
            // command was a quadratic (Q or T); otherwise control == current.
            let reflect = iteration > 0 || prev_op == 'Q' || prev_op == 'T';
            let (x1, y1) = match (reflect, self.last_control) {
                (true, Some((lx, ly))) => (2.0f32.mul_add(cx, -lx), 2.0f32.mul_add(cy, -ly)),
                _ => (cx, cy),
            };
            iteration += 1;

            let (x, y) = if is_relative {
                (cx + x, cy + y)
            } else {
                (x, y)
            };

            self.builder.quad_to(x1, y1, x, y);
            self.last_control = Some((x1, y1));
        }
        Ok(())
    }

    fn parse_arcto(&mut self, is_relative: bool) -> Result<(), SvgPathError> {
        loop {
            self.skip_whitespace();
            if self.is_end() || self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                break;
            }

            let rx = self.parse_number()?;
            let ry = self.parse_number()?;
            let x_rotation = self.parse_number()?;
            let large_arc = self.parse_flag()?;
            let sweep = self.parse_flag()?;
            let x = self.parse_number()?;
            let y = self.parse_number()?;

            let (x, y) = if is_relative {
                let (cx, cy) = self.current_point();
                (cx + x, cy + y)
            } else {
                (x, y)
            };

            self.builder
                .arc_to(rx, ry, x_rotation, large_arc, sweep, x, y);
        }
        self.last_control = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_path() {
        let path = parse_svg_path("M 10 20 L 30 40 Z").unwrap();
        assert_eq!(path.verb_count(), 3); // Move, Line, Close
    }

    #[test]
    fn test_parse_relative_commands() {
        let path = parse_svg_path("M 10 20 l 20 20 z").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_curves() {
        let path = parse_svg_path("M 0 0 C 10 20 30 40 50 60").unwrap();
        assert_eq!(path.verb_count(), 2); // Move, Cubic
    }

    #[test]
    fn test_parse_arc() {
        let path = parse_svg_path("M 0 0 A 50 50 0 0 1 100 0").unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_parse_horizontal_vertical() {
        let path = parse_svg_path("M 0 0 H 100 V 100 H 0 Z").unwrap();
        assert_eq!(path.verb_count(), 5);
    }

    #[test]
    fn test_number_after_z_is_error() {
        // Numeric data directly after Z must be rejected, not treated as lineto.
        assert!(parse_svg_path("M0 0 L10 10 Z 5 5").is_err());
    }

    #[test]
    fn test_smooth_cubic_reflection_gated_on_previous() {
        use crate::PathElement;
        // S after a line: the first control point equals the current point
        // (no reflection), so the cubic is C(cur, x2y2, xy).
        let path = parse_svg_path("M0 0 L10 0 S 20 10 30 0").unwrap();
        let cubic = path.iter().find_map(|e| match e {
            PathElement::Cubic(c1, c2, end) => Some((c1, c2, end)),
            _ => None,
        });
        let (c1, _c2, _end) = cubic.expect("expected a cubic");
        // current point at S is (10,0); with no reflection c1 == current point.
        assert!(
            (c1.x - 10.0).abs() < 1e-3 && (c1.y - 0.0).abs() < 1e-3,
            "S after L must not reflect: c1={c1:?}"
        );
    }

    #[test]
    fn test_smooth_cubic_reflection_after_cubic() {
        use crate::PathElement;
        // S after C: reflect the previous control point about the current point.
        // C: control2 = (10,10); current after C = (20,0). Reflected c1 = (30,-10).
        let path = parse_svg_path("M0 0 C 0 10 10 10 20 0 S 40 10 50 0").unwrap();
        let cubics: Vec<_> = path
            .iter()
            .filter_map(|e| match e {
                PathElement::Cubic(c1, c2, end) => Some((c1, c2, end)),
                _ => None,
            })
            .collect();
        assert_eq!(cubics.len(), 2);
        let c1 = cubics[1].0;
        assert!(
            (c1.x - 30.0).abs() < 1e-3 && (c1.y + 10.0).abs() < 1e-3,
            "S after C must reflect: c1={c1:?}"
        );
    }

    #[test]
    fn test_smooth_quad_reflection_gated() {
        use crate::PathElement;
        // T after a line: control == current point (no reflection).
        let path = parse_svg_path("M0 0 L10 0 T 30 0").unwrap();
        let quad = path.iter().find_map(|e| match e {
            PathElement::Quad(c, end) => Some((c, end)),
            _ => None,
        });
        let (c, _end) = quad.expect("expected a quad");
        assert!(
            (c.x - 10.0).abs() < 1e-3 && (c.y - 0.0).abs() < 1e-3,
            "T after L must not reflect: c={c:?}"
        );
    }

    #[test]
    fn test_smooth_cubic_not_reflected_after_quad() {
        use crate::PathElement;
        // S must reflect only after C/S. After Q, control = current point.
        let path = parse_svg_path("M0 0 Q 10 10 20 0 S 40 10 50 0").unwrap();
        let c1 = path
            .iter()
            .find_map(|e| match e {
                PathElement::Cubic(c1, _, _) => Some(c1),
                _ => None,
            })
            .expect("expected cubic from S");
        assert!(
            (c1.x - 20.0).abs() < 1e-3 && (c1.y - 0.0).abs() < 1e-3,
            "S after Q must not reflect: c1={c1:?}"
        );
    }

    #[test]
    fn test_smooth_quad_not_reflected_after_cubic() {
        use crate::PathElement;
        // T must reflect only after Q/T. After C, control = current point.
        let path = parse_svg_path("M0 0 C 0 10 10 10 20 0 T 40 0").unwrap();
        let c = path
            .iter()
            .find_map(|e| match e {
                PathElement::Quad(c, _) => Some(c),
                _ => None,
            })
            .expect("expected quad from T");
        assert!(
            (c.x - 20.0).abs() < 1e-3 && (c.y - 0.0).abs() < 1e-3,
            "T after C must not reflect: c={c:?}"
        );
    }

    #[test]
    fn test_relative_after_close_uses_subpath_start() {
        use crate::PathElement;
        // After Z the current point returns to the subpath start (5,5), so the
        // relative lineto 'l 10 0' goes to (15,5).
        let path = parse_svg_path("M5 5 L20 5 L20 20 Z l 10 0").unwrap();
        // Find the last line endpoint.
        let last_line = path
            .iter()
            .filter_map(|e| match e {
                PathElement::Line(p) => Some(p),
                _ => None,
            })
            .last()
            .unwrap();
        assert!(
            (last_line.x - 15.0).abs() < 1e-3 && (last_line.y - 5.0).abs() < 1e-3,
            "relative line after Z should start at subpath start: {last_line:?}"
        );
    }
}
