//! SVG export functionality.
//!
//! This module provides functionality to convert an `SvgDom` back to SVG markup,
//! enabling round-trip editing and programmatic SVG generation.

use crate::dom::{
    GradientStop, GradientUnits, SpreadMethod, SvgDom, SvgNode, SvgNodeKind, SvgPaint, TextAnchor,
};
use skia_rs_core::{Color, Matrix, Scalar};
use std::fmt::Write;

/// Options for SVG export.
#[derive(Debug, Clone)]
pub struct SvgExportOptions {
    /// Indent string (default: 2 spaces).
    pub indent: String,
    /// Include XML declaration.
    pub xml_declaration: bool,
    /// Pretty print with indentation.
    pub pretty_print: bool,
    /// Precision for floating point numbers.
    pub precision: usize,
    /// Include default attributes.
    pub include_defaults: bool,
}

impl Default for SvgExportOptions {
    fn default() -> Self {
        Self {
            indent: "  ".to_string(),
            xml_declaration: true,
            pretty_print: true,
            precision: 3,
            include_defaults: false,
        }
    }
}

impl SvgExportOptions {
    /// Create options for minified output.
    #[must_use]
    pub const fn minified() -> Self {
        Self {
            indent: String::new(),
            xml_declaration: false,
            pretty_print: false,
            precision: 2,
            include_defaults: false,
        }
    }
}

/// Export an SVG DOM to a string.
#[must_use]
pub fn export_svg(dom: &SvgDom) -> String {
    export_svg_with_options(dom, &SvgExportOptions::default())
}

/// Export an SVG DOM to a string with custom options.
#[must_use]
pub fn export_svg_with_options(dom: &SvgDom, options: &SvgExportOptions) -> String {
    let mut output = String::new();

    if options.xml_declaration {
        output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    }

    // Start SVG element
    write!(
        output,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\"",
        format_scalar(dom.width, options.precision),
        format_scalar(dom.height, options.precision)
    )
    .unwrap();

    if let Some(vb) = &dom.view_box {
        write!(
            output,
            " viewBox=\"{} {} {} {}\"",
            format_scalar(vb.left, options.precision),
            format_scalar(vb.top, options.precision),
            format_scalar(vb.width(), options.precision),
            format_scalar(vb.height(), options.precision)
        )
        .unwrap();
    }

    output.push('>');

    if options.pretty_print {
        output.push('\n');
    }

    // Export children
    for child in &dom.root.children {
        export_node(&mut output, child, options, 1);
    }

    output.push_str("</svg>");

    if options.pretty_print {
        output.push('\n');
    }

    output
}

#[allow(
    clippy::too_many_lines,
    reason = "one match arm per SVG element kind; splitting would scatter closely related serialization logic and harm readability"
)]
fn export_node(output: &mut String, node: &SvgNode, options: &SvgExportOptions, depth: usize) {
    if !node.visible && !options.include_defaults {
        return;
    }

    let indent = if options.pretty_print {
        options.indent.repeat(depth)
    } else {
        String::new()
    };

    let newline = if options.pretty_print { "\n" } else { "" };

    match &node.kind {
        SvgNodeKind::Svg => {
            // Already handled at top level
            for child in &node.children {
                export_node(output, child, options, depth);
            }
        }
        SvgNodeKind::Group => {
            output.push_str(&indent);
            output.push_str("<g");
            export_common_attrs(output, node, options);

            if node.children.is_empty() {
                output.push_str("/>");
            } else {
                output.push('>');
                output.push_str(newline);

                for child in &node.children {
                    export_node(output, child, options, depth + 1);
                }

                output.push_str(&indent);
                output.push_str("</g>");
            }
            output.push_str(newline);
        }
        SvgNodeKind::Rect(rect) => {
            output.push_str(&indent);
            output.push_str("<rect");

            write!(
                output,
                " x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
                format_scalar(rect.x, options.precision),
                format_scalar(rect.y, options.precision),
                format_scalar(rect.width, options.precision),
                format_scalar(rect.height, options.precision)
            )
            .unwrap();

            if rect.rx > 0.0 {
                write!(
                    output,
                    " rx=\"{}\"",
                    format_scalar(rect.rx, options.precision)
                )
                .unwrap();
            }
            if rect.ry > 0.0 {
                write!(
                    output,
                    " ry=\"{}\"",
                    format_scalar(rect.ry, options.precision)
                )
                .unwrap();
            }

            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Circle(circle) => {
            output.push_str(&indent);
            output.push_str("<circle");

            write!(
                output,
                " cx=\"{}\" cy=\"{}\" r=\"{}\"",
                format_scalar(circle.cx, options.precision),
                format_scalar(circle.cy, options.precision),
                format_scalar(circle.r, options.precision)
            )
            .unwrap();

            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Ellipse(ellipse) => {
            output.push_str(&indent);
            output.push_str("<ellipse");

            write!(
                output,
                " cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\"",
                format_scalar(ellipse.cx, options.precision),
                format_scalar(ellipse.cy, options.precision),
                format_scalar(ellipse.rx, options.precision),
                format_scalar(ellipse.ry, options.precision)
            )
            .unwrap();

            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Line(line) => {
            output.push_str(&indent);
            output.push_str("<line");

            write!(
                output,
                " x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"",
                format_scalar(line.x1, options.precision),
                format_scalar(line.y1, options.precision),
                format_scalar(line.x2, options.precision),
                format_scalar(line.y2, options.precision)
            )
            .unwrap();

            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Polyline(points) => {
            output.push_str(&indent);
            output.push_str("<polyline");
            export_points_attr(output, points, options);
            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Polygon(points) => {
            output.push_str(&indent);
            output.push_str("<polygon");
            export_points_attr(output, points, options);
            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Path(path) => {
            output.push_str(&indent);
            output.push_str("<path");

            let path_data = export_path_data(path, options);
            write!(output, " d=\"{path_data}\"").unwrap();

            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Text(text) => {
            output.push_str(&indent);
            output.push_str("<text");

            write!(
                output,
                " x=\"{}\" y=\"{}\"",
                format_scalar(text.x, options.precision),
                format_scalar(text.y, options.precision)
            )
            .unwrap();

            if let Some(ref family) = text.font_family {
                write!(output, " font-family=\"{}\"", escape_xml(family)).unwrap();
            }

            write!(
                output,
                " font-size=\"{}\"",
                format_scalar(text.font_size, options.precision)
            )
            .unwrap();

            if text.font_weight != 400 {
                write!(output, " font-weight=\"{}\"", text.font_weight).unwrap();
            }

            match text.text_anchor {
                TextAnchor::Middle => output.push_str(" text-anchor=\"middle\""),
                TextAnchor::End => output.push_str(" text-anchor=\"end\""),
                TextAnchor::Start => {}
            }

            export_common_attrs(output, node, options);
            output.push('>');
            output.push_str(&escape_xml(&text.content));
            output.push_str("</text>");
            output.push_str(newline);
        }
        SvgNodeKind::Image(image) => {
            output.push_str(&indent);
            output.push_str("<image");

            write!(
                output,
                " x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
                format_scalar(image.x, options.precision),
                format_scalar(image.y, options.precision),
                format_scalar(image.width, options.precision),
                format_scalar(image.height, options.precision)
            )
            .unwrap();

            write!(output, " href=\"{}\"", escape_xml(&image.href)).unwrap();

            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Use(href) => {
            output.push_str(&indent);
            output.push_str("<use");
            write!(output, " href=\"{}\"", escape_xml(href)).unwrap();
            export_common_attrs(output, node, options);
            output.push_str("/>");
            output.push_str(newline);
        }
        SvgNodeKind::Defs => {
            output.push_str(&indent);
            output.push_str("<defs>");
            output.push_str(newline);

            for child in &node.children {
                export_node(output, child, options, depth + 1);
            }

            output.push_str(&indent);
            output.push_str("</defs>");
            output.push_str(newline);
        }
        SvgNodeKind::LinearGradient(grad) => {
            output.push_str(&indent);
            output.push_str("<linearGradient");

            if let Some(ref id) = node.id {
                write!(output, " id=\"{}\"", escape_xml(id)).unwrap();
            }

            write!(
                output,
                " x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"",
                format_scalar(grad.x1, options.precision),
                format_scalar(grad.y1, options.precision),
                format_scalar(grad.x2, options.precision),
                format_scalar(grad.y2, options.precision)
            )
            .unwrap();

            export_gradient_attrs(output, grad.spread, grad.units);

            if !grad.transform.is_identity() {
                export_gradient_transform_attr(output, &grad.transform, options);
            }

            output.push('>');
            output.push_str(newline);

            for stop in &grad.stops {
                export_gradient_stop(output, stop, options, depth + 1);
            }

            output.push_str(&indent);
            output.push_str("</linearGradient>");
            output.push_str(newline);
        }
        SvgNodeKind::RadialGradient(grad) => {
            output.push_str(&indent);
            output.push_str("<radialGradient");

            if let Some(ref id) = node.id {
                write!(output, " id=\"{}\"", escape_xml(id)).unwrap();
            }

            write!(
                output,
                " cx=\"{}\" cy=\"{}\" r=\"{}\"",
                format_scalar(grad.cx, options.precision),
                format_scalar(grad.cy, options.precision),
                format_scalar(grad.r, options.precision)
            )
            .unwrap();

            if (grad.fx - grad.cx).abs() > 0.001 || (grad.fy - grad.cy).abs() > 0.001 {
                write!(
                    output,
                    " fx=\"{}\" fy=\"{}\"",
                    format_scalar(grad.fx, options.precision),
                    format_scalar(grad.fy, options.precision)
                )
                .unwrap();
            }

            export_gradient_attrs(output, grad.spread, grad.units);

            if !grad.transform.is_identity() {
                export_gradient_transform_attr(output, &grad.transform, options);
            }

            output.push('>');
            output.push_str(newline);

            for stop in &grad.stops {
                export_gradient_stop(output, stop, options, depth + 1);
            }

            output.push_str(&indent);
            output.push_str("</radialGradient>");
            output.push_str(newline);
        }
        SvgNodeKind::ClipPath(id) => {
            output.push_str(&indent);
            write!(output, "<clipPath id=\"{}\">", escape_xml(id)).unwrap();
            output.push_str(newline);

            for child in &node.children {
                export_node(output, child, options, depth + 1);
            }

            output.push_str(&indent);
            output.push_str("</clipPath>");
            output.push_str(newline);
        }
        SvgNodeKind::Unknown(tag) => {
            output.push_str(&indent);
            write!(output, "<{tag}").unwrap();
            export_common_attrs(output, node, options);

            // `<style>` nodes carry their CSS source in the
            // `__text_content` attribute (populated by the parser).
            // Emit it inside a CDATA block so downstream consumers see
            // the same CSS we originally parsed.
            let text_content = node.attributes.get("__text_content");

            if node.children.is_empty() && text_content.is_none() {
                output.push_str("/>");
            } else {
                output.push('>');
                output.push_str(newline);

                if let Some(css) = text_content
                    && !css.is_empty()
                {
                    if options.pretty_print {
                        output.push_str(&options.indent.repeat(depth + 1));
                    }
                    write!(output, "<![CDATA[{css}]]>").unwrap();
                    output.push_str(newline);
                }

                for child in &node.children {
                    export_node(output, child, options, depth + 1);
                }

                output.push_str(&indent);
                write!(output, "</{tag}>").unwrap();
            }
            output.push_str(newline);
        }
    }
}

fn export_common_attrs(output: &mut String, node: &SvgNode, options: &SvgExportOptions) {
    // ID (but not for gradients, which handle it specially)
    if let Some(ref id) = node.id
        && !matches!(
            node.kind,
            SvgNodeKind::LinearGradient(_) | SvgNodeKind::RadialGradient(_)
        )
    {
        write!(output, " id=\"{}\"", escape_xml(id)).unwrap();
    }

    // Classes
    if !node.classes.is_empty() {
        write!(output, " class=\"{}\"", node.classes.join(" ")).unwrap();
    }

    // Transform
    if !node.transform.is_identity() {
        export_transform_attr(output, &node.transform, options);
    }

    // Fill. `None` means the property is unspecified (inherited), so emit
    // nothing; an explicit `SvgPaint::None` serializes as `fill="none"`.
    if let Some(ref fill) = node.fill {
        let fill_str = format_paint(fill);
        if fill_str != "black" || options.include_defaults {
            write!(output, " fill=\"{fill_str}\"").unwrap();
        }
    }

    // Stroke
    if let Some(ref stroke) = node.stroke {
        write!(output, " stroke=\"{}\"", format_paint(stroke)).unwrap();

        match node.stroke_width {
            #[allow(
                clippy::float_cmp,
                reason = "exact match against the SVG default stroke-width of 1.0, not an approximate comparison"
            )]
            Some(sw) if sw != 1.0 || options.include_defaults => {
                write!(
                    output,
                    " stroke-width=\"{}\"",
                    format_scalar(sw, options.precision)
                )
                .unwrap();
            }
            None if options.include_defaults => {
                output.push_str(" stroke-width=\"1\"");
            }
            _ => {}
        }
    }

    // Opacity
    if (node.opacity - 1.0).abs() > 0.001 {
        write!(
            output,
            " opacity=\"{}\"",
            format_scalar(node.opacity, options.precision)
        )
        .unwrap();
    }

    // Visibility
    if !node.visible {
        output.push_str(" visibility=\"hidden\"");
    }

    // Custom attributes (except internal ones)
    for (key, value) in &node.attributes {
        if !key.starts_with("__") && !is_standard_attr(key) {
            write!(output, " {}=\"{}\"", key, escape_xml(value)).unwrap();
        }
    }
}

fn is_standard_attr(key: &str) -> bool {
    matches!(
        key,
        "id" | "class"
            | "transform"
            | "fill"
            | "stroke"
            | "stroke-width"
            | "opacity"
            | "visibility"
    )
}

/// Emit a gradient transform attribute (`gradientTransform=...`) — same
/// matrix encoding as `export_transform_attr`, different attribute name
/// per SVG 1.1 §13.2.3.
fn export_gradient_transform_attr(
    output: &mut String,
    matrix: &Matrix,
    options: &SvgExportOptions,
) {
    let mut buf = String::new();
    export_transform_attr(&mut buf, matrix, options);
    // export_transform_attr writes ` transform="..."` — rewrite the
    // attribute name in place so we don't duplicate its formatting logic.
    let replaced = buf.replacen(" transform=", " gradientTransform=", 1);
    output.push_str(&replaced);
}

fn export_transform_attr(output: &mut String, matrix: &Matrix, options: &SvgExportOptions) {
    let v = &matrix.values;

    // Check for special cases
    let is_translate = (v[0] - 1.0).abs() < 0.001
        && v[1].abs() < 0.001
        && v[3].abs() < 0.001
        && (v[4] - 1.0).abs() < 0.001;

    let is_scale =
        v[1].abs() < 0.001 && v[3].abs() < 0.001 && v[2].abs() < 0.001 && v[5].abs() < 0.001;

    if is_translate && (v[2].abs() > 0.001 || v[5].abs() > 0.001) {
        write!(
            output,
            " transform=\"translate({}, {})\"",
            format_scalar(v[2], options.precision),
            format_scalar(v[5], options.precision)
        )
        .unwrap();
    } else if is_scale && ((v[0] - 1.0).abs() > 0.001 || (v[4] - 1.0).abs() > 0.001) {
        if (v[0] - v[4]).abs() < 0.001 {
            write!(
                output,
                " transform=\"scale({})\"",
                format_scalar(v[0], options.precision)
            )
            .unwrap();
        } else {
            write!(
                output,
                " transform=\"scale({}, {})\"",
                format_scalar(v[0], options.precision),
                format_scalar(v[4], options.precision)
            )
            .unwrap();
        }
    } else {
        // Use matrix form
        write!(
            output,
            " transform=\"matrix({}, {}, {}, {}, {}, {})\"",
            format_scalar(v[0], options.precision),
            format_scalar(v[3], options.precision),
            format_scalar(v[1], options.precision),
            format_scalar(v[4], options.precision),
            format_scalar(v[2], options.precision),
            format_scalar(v[5], options.precision)
        )
        .unwrap();
    }
}

fn export_points_attr(
    output: &mut String,
    points: &[skia_rs_core::Point],
    options: &SvgExportOptions,
) {
    output.push_str(" points=\"");
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            output.push(' ');
        }
        write!(
            output,
            "{},{}",
            format_scalar(p.x, options.precision),
            format_scalar(p.y, options.precision)
        )
        .unwrap();
    }
    output.push('"');
}

fn export_path_data(path: &skia_rs_path::Path, options: &SvgExportOptions) -> String {
    use skia_rs_core::Point;
    use skia_rs_path::PathElement;

    let mut data = String::new();
    // Track the current point so conics can be subdivided from their true
    // start; also remember the last move for `Z` handling.
    let mut current = Point::new(0.0, 0.0);
    let mut subpath_start = Point::new(0.0, 0.0);

    for elem in path {
        match elem {
            PathElement::Move(p) => {
                write!(
                    data,
                    "M{} {}",
                    format_scalar(p.x, options.precision),
                    format_scalar(p.y, options.precision)
                )
                .unwrap();
                current = p;
                subpath_start = p;
            }
            PathElement::Line(p) => {
                write!(
                    data,
                    "L{} {}",
                    format_scalar(p.x, options.precision),
                    format_scalar(p.y, options.precision)
                )
                .unwrap();
                current = p;
            }
            PathElement::Quad(p1, p2) => {
                write!(
                    data,
                    "Q{} {} {} {}",
                    format_scalar(p1.x, options.precision),
                    format_scalar(p1.y, options.precision),
                    format_scalar(p2.x, options.precision),
                    format_scalar(p2.y, options.precision)
                )
                .unwrap();
                current = p2;
            }
            PathElement::Conic(p1, p2, w) => {
                // SVG has no conic primitive. Subdivide into a quad spline
                // via SkConic::chopIntoQuadsPOW2-equivalent math (see
                // `conic_to_quads`) instead of dropping the weight with a
                // single naive quad, which distorts the curve.
                for (ctrl, end) in conic_to_quads(current, p1, p2, w) {
                    write!(
                        data,
                        "Q{} {} {} {}",
                        format_scalar(ctrl.x, options.precision),
                        format_scalar(ctrl.y, options.precision),
                        format_scalar(end.x, options.precision),
                        format_scalar(end.y, options.precision)
                    )
                    .unwrap();
                }
                current = p2;
            }
            PathElement::Cubic(p1, p2, p3) => {
                write!(
                    data,
                    "C{} {} {} {} {} {}",
                    format_scalar(p1.x, options.precision),
                    format_scalar(p1.y, options.precision),
                    format_scalar(p2.x, options.precision),
                    format_scalar(p2.y, options.precision),
                    format_scalar(p3.x, options.precision),
                    format_scalar(p3.y, options.precision)
                )
                .unwrap();
                current = p3;
            }
            PathElement::Close => {
                data.push('Z');
                current = subpath_start;
            }
        }
    }

    data
}

/// Subdivide a single conic (rational quadratic) into a spline of ordinary
/// quads, returning each quad's `(control, end)` pair. Port of
/// `SkConic::computeQuadPOW2` + `chopIntoQuadsPOW2` (SkGeometry.cpp) at a
/// fixed tolerance of 0.25, matching Skia's default conic→quad conversion.
fn conic_to_quads(
    start: skia_rs_core::Point,
    ctrl: skia_rs_core::Point,
    end: skia_rs_core::Point,
    w: f32,
) -> Vec<(skia_rs_core::Point, skia_rs_core::Point)> {
    use skia_rs_core::Point;

    // Recursive chop to 2^pow2 quads, emitting (ctrl, end) for each.
    fn subdivide(
        p0: skia_rs_core::Point,
        p1: skia_rs_core::Point,
        p2: skia_rs_core::Point,
        w: f32,
        level: i32,
        out: &mut Vec<(skia_rs_core::Point, skia_rs_core::Point)>,
    ) {
        if level == 0 {
            out.push((p1, p2));
            return;
        }
        // SkConic::chop: split at the parametric midpoint.
        let scale = 1.0 / (1.0 + w);
        let t0 = Point::new(p0.x * scale, p0.y * scale);
        let t1 = Point::new(p1.x * (w * scale), p1.y * (w * scale));
        let t2 = Point::new(p2.x * scale, p2.y * scale);
        let cp1 = Point::new(t0.x + t1.x, t0.y + t1.y);
        let cp3 = Point::new(t1.x + t2.x, t1.y + t2.y);
        let cp2 = Point::new(
            0.5f32.mul_add(t2.x, 0.5f32.mul_add(t0.x, t1.x)),
            0.5f32.mul_add(t2.y, 0.5f32.mul_add(t0.y, t1.y)),
        );
        let new_w = w.mul_add(0.5, 0.5).sqrt();
        subdivide(p0, cp1, cp2, new_w, level - 1, out);
        subdivide(cp2, cp3, p2, new_w, level - 1, out);
    }

    // computeQuadPOW2 with tol = 0.25.
    const TOL: f32 = 0.25;
    const MAX_POW2: i32 = 5;

    // Degenerate/invalid weight: a single quad through the control point.
    if !w.is_finite() || w <= 0.0 {
        return vec![(ctrl, end)];
    }

    let a = w - 1.0;
    let k = a / (4.0 * (2.0 + a));
    let ex = k * (2.0f32.mul_add(-ctrl.x, start.x) + end.x);
    let ey = k * (2.0f32.mul_add(-ctrl.y, start.y) + end.y);
    let mut error = ex.hypot(ey);
    let mut pow2 = 0;
    while pow2 < MAX_POW2 {
        if error <= TOL {
            break;
        }
        error *= 0.25;
        pow2 += 1;
    }

    let mut out = Vec::with_capacity(1 << pow2);
    subdivide(start, ctrl, end, w, pow2, &mut out);
    out
}

fn export_gradient_attrs(output: &mut String, spread: SpreadMethod, units: GradientUnits) {
    match spread {
        SpreadMethod::Reflect => output.push_str(" spreadMethod=\"reflect\""),
        SpreadMethod::Repeat => output.push_str(" spreadMethod=\"repeat\""),
        SpreadMethod::Pad => {} // default
    }

    match units {
        GradientUnits::UserSpaceOnUse => output.push_str(" gradientUnits=\"userSpaceOnUse\""),
        GradientUnits::ObjectBoundingBox => {} // default
    }
}

fn export_gradient_stop(
    output: &mut String,
    stop: &GradientStop,
    options: &SvgExportOptions,
    depth: usize,
) {
    let indent = if options.pretty_print {
        options.indent.repeat(depth)
    } else {
        String::new()
    };
    let newline = if options.pretty_print { "\n" } else { "" };

    output.push_str(&indent);
    write!(
        output,
        "<stop offset=\"{}\" stop-color=\"{}\"",
        format_scalar(stop.offset, options.precision),
        format_color(stop.color)
    )
    .unwrap();

    if (stop.opacity - 1.0).abs() > 0.001 {
        write!(
            output,
            " stop-opacity=\"{}\"",
            format_scalar(stop.opacity, options.precision)
        )
        .unwrap();
    }

    output.push_str("/>");
    output.push_str(newline);
}

fn format_paint(paint: &SvgPaint) -> String {
    match paint {
        SvgPaint::Color(color) => format_color(*color),
        SvgPaint::CurrentColor => "currentColor".to_string(),
        SvgPaint::Url(url, fallback) => fallback.map_or_else(
            || format!("url({url})"),
            |color| format!("url({url}) {}", format_color(color)),
        ),
        SvgPaint::None => "none".to_string(),
    }
}

fn format_color(color: Color) -> String {
    if color.alpha() == 255 {
        format!(
            "#{:02x}{:02x}{:02x}",
            color.red(),
            color.green(),
            color.blue()
        )
    } else {
        format!(
            "rgba({}, {}, {}, {})",
            color.red(),
            color.green(),
            color.blue(),
            f32::from(color.alpha()) / 255.0
        )
    }
}

fn format_scalar(value: Scalar, precision: usize) -> String {
    let formatted = format!("{value:.precision$}");
    // Remove trailing zeros and decimal point if unnecessary
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::{SvgLinearGradient, SvgRect};

    #[test]
    fn test_export_simple_svg() {
        let mut dom = SvgDom::new();
        dom.width = 100.0;
        dom.height = 100.0;

        let mut rect_node = SvgNode::new(SvgNodeKind::Rect(SvgRect {
            x: 10.0,
            y: 10.0,
            width: 80.0,
            height: 80.0,
            rx: 0.0,
            ry: 0.0,
        }));
        rect_node.fill = Some(SvgPaint::Color(Color::from_rgb(255, 0, 0)));

        dom.root.add_child(rect_node);

        let svg = export_svg(&dom);
        assert!(svg.contains("<rect"));
        assert!(svg.contains("fill=\"#ff0000\""));
    }

    #[test]
    fn test_conic_to_quads_subdivides_high_weight() {
        use skia_rs_core::Point;
        // A high-weight conic exceeds the single-quad error tolerance, so
        // computeQuadPOW2 subdivides it into a power-of-two quad spline. A
        // naive single-quad export would instead drop the weight entirely.
        let start = Point::new(0.0, 0.0);
        let ctrl = Point::new(1.0, 2.0);
        let end = Point::new(2.0, 0.0);
        let quads = conic_to_quads(start, ctrl, end, 8.0);
        assert!(
            quads.len() >= 2,
            "high-weight conic subdivided, got {} quad(s)",
            quads.len()
        );
        // The spline is a power-of-two count and preserves the endpoints.
        assert!(quads.len().is_power_of_two());
        let last_end = quads.last().unwrap().1;
        assert!((last_end.x - end.x).abs() < 1e-4 && (last_end.y - end.y).abs() < 1e-4);
    }

    #[test]
    fn test_conic_to_quads_low_weight_single_quad() {
        use skia_rs_core::Point;
        // A near-quadratic conic (weight ~ 1) is within tolerance and
        // collapses to a single quad through the control point — matching
        // SkConic::computeQuadPOW2 returning pow2 = 0.
        let quads = conic_to_quads(
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 0.0),
            1.0,
        );
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].0, Point::new(1.0, 1.0));
        assert_eq!(quads[0].1, Point::new(2.0, 0.0));
    }

    #[test]
    fn test_export_conic_emits_quad_spline() {
        use skia_rs_path::PathBuilder;
        // Build a path with a conic and export it; the data must contain a
        // Q command (subdivided), not silently drop the curve.
        let mut b = PathBuilder::new();
        b.move_to(1.0, 0.0);
        b.conic_to(1.0, 1.0, 0.0, 1.0, std::f32::consts::FRAC_1_SQRT_2);
        let path = b.build();
        let data = export_path_data(&path, &SvgExportOptions::default());
        assert!(data.starts_with('M'));
        assert!(data.contains('Q'), "conic exported as quad(s): {data}");
    }

    #[test]
    fn test_export_minified() {
        let mut dom = SvgDom::new();
        dom.width = 100.0;
        dom.height = 100.0;

        let svg = export_svg_with_options(&dom, &SvgExportOptions::minified());
        assert!(!svg.contains('\n'));
        assert!(!svg.contains("<?xml"));
    }

    #[test]
    fn test_format_scalar() {
        assert_eq!(format_scalar(10.0, 3), "10");
        assert_eq!(format_scalar(10.5, 3), "10.5");
        assert_eq!(format_scalar(10.123_456, 2), "10.12");
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("<test>"), "&lt;test&gt;");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
    }

    #[test]
    fn test_format_color() {
        assert_eq!(format_color(Color::from_rgb(255, 0, 0)), "#ff0000");
        assert_eq!(
            format_color(Color::from_argb(128, 255, 0, 0)),
            "rgba(255, 0, 0, 0.5019608)"
        );
    }

    #[test]
    fn test_export_gradient_transform_uses_gradient_attribute() {
        // A gradient with a non-identity transform must emit
        // `gradientTransform=`, never `transform=`, per SVG 1.1 §13.2.3.
        let mut dom = SvgDom::new();
        dom.width = 10.0;
        dom.height = 10.0;
        let mut grad = SvgNode::new(SvgNodeKind::LinearGradient(SvgLinearGradient {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 0.0,
            stops: vec![GradientStop {
                offset: 0.0,
                color: Color::BLACK,
                opacity: 1.0,
            }],
            spread: SpreadMethod::Pad,
            units: GradientUnits::ObjectBoundingBox,
            transform: Matrix::translate(5.0, 0.0),
        }));
        grad.id = Some("g".to_string());

        let mut defs = SvgNode::new(SvgNodeKind::Defs);
        defs.add_child(grad);
        dom.root.add_child(defs);

        let svg = export_svg(&dom);
        assert!(
            svg.contains("gradientTransform="),
            "expected gradientTransform attribute, got:\n{svg}"
        );
        assert!(
            !svg.contains("<linearGradient")
                || !svg[svg.find("<linearGradient").unwrap()..]
                    .split('>')
                    .next()
                    .unwrap()
                    .contains(" transform="),
            "linearGradient element must not use `transform=`, got:\n{svg}"
        );
    }

    #[test]
    fn test_export_style_element_round_trips() {
        // When a <style> block is parsed, its contents end up in the
        // Unknown("style") node's __text_content attribute. Exporting
        // such a node must re-emit the CSS in CDATA so a subsequent
        // parse recovers the rule.
        use crate::parser::parse_svg;

        let svg_in = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
          <style>rect { fill: red; }</style>
          <rect x="0" y="0" width="10" height="10"/>
        </svg>"#;
        let dom = parse_svg(svg_in).unwrap();
        assert_eq!(dom.stylesheet.rules.len(), 1);

        let exported = export_svg(&dom);
        assert!(
            exported.contains("<style") && exported.contains("fill: red"),
            "style block should round-trip, got:\n{exported}"
        );

        let reparsed = parse_svg(&exported).unwrap();
        assert_eq!(
            reparsed.stylesheet.rules.len(),
            1,
            "exported style element should survive a re-parse"
        );
    }
}
