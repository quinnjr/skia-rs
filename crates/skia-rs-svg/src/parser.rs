//! SVG parsing.
//!
//! Uses roxmltree for correct XML handling (entities, CDATA, namespaces,
//! mixed text/element content) and walks the tree to build an `SvgDom`.

use crate::css::Stylesheet;
use crate::dom::{
    AlignX, AlignY, GradientStop, GradientUnits, MeetOrSlice, PreserveAspectRatio, SpreadMethod,
    SvgCircle, SvgDom, SvgEllipse, SvgImage, SvgLine, SvgLinearGradient, SvgNode, SvgNodeKind,
    SvgPaint, SvgRadialGradient, SvgRect, SvgText, TextAnchor,
};
use skia_rs_core::{Color, Matrix, Point, Rect, Scalar};
use skia_rs_path::parse_svg_path;
use std::collections::HashMap;
use thiserror::Error;

/// SVG parsing error.
#[derive(Debug, Error)]
pub enum SvgError {
    /// XML parsing error.
    #[error("XML parsing error: {0}")]
    XmlError(String),
    /// Invalid attribute.
    #[error("Invalid attribute {0}: {1}")]
    InvalidAttribute(String, String),
    /// Unsupported feature.
    #[error("Unsupported SVG feature: {0}")]
    Unsupported(String),
}

/// Parse an SVG document from a string.
///
/// Uses `roxmltree` to walk the document and construct an `SvgDom`. Text
/// content between tags is captured (including inside `<text>` elements),
/// `<stop>` children of gradients populate their `stops` vector, inline
/// `style` attributes are preserved on each node, and the contents of
/// `<style>` elements are parsed into `dom.stylesheet` for later
/// application during rendering.
///
/// # Errors
///
/// Returns [`SvgError::XmlError`] if `svg` is not well-formed XML.
pub fn parse_svg(svg: &str) -> Result<SvgDom, SvgError> {
    let doc = roxmltree::Document::parse(svg).map_err(|e| SvgError::XmlError(e.to_string()))?;

    let root = doc.root_element();
    let mut dom = SvgDom::new();

    // Parse the root <svg> element's own attributes.
    dom.width = parse_length(root.attribute("width").unwrap_or("100"));
    dom.height = parse_length(root.attribute("height").unwrap_or("100"));
    if let Some(vb) = root.attribute("viewBox") {
        dom.view_box = parse_viewbox(vb);
    }
    if let Some(par) = root.attribute("preserveAspectRatio") {
        dom.preserve_aspect_ratio = parse_preserve_aspect_ratio(par);
    }

    // Percentage lengths resolve against the establishing viewport: the
    // viewBox size when present, else the intrinsic width/height
    // (SkSVGSVG establishes fViewport from the viewBox). Fall back to the
    // width/height when a dimension is degenerate.
    let lctx = match dom.view_box {
        Some(vb) if vb.width() > 0.0 && vb.height() > 0.0 => LengthContext {
            vw: vb.width(),
            vh: vb.height(),
        },
        _ => LengthContext {
            vw: dom.width,
            vh: dom.height,
        },
    };

    // Build an attribute HashMap from the root for common-attribute
    // application (id/transform/fill/stroke/opacity), then recursively
    // build children.
    let mut root_node = SvgNode::new(SvgNodeKind::Svg);
    apply_common_attrs(&mut root_node, &attrs_of(root));

    for child in root.children() {
        if let Some(node) = build_node(child, &mut dom.stylesheet, lctx)? {
            root_node.add_child(node);
        }
    }

    dom.root = root_node;
    Ok(dom)
}

/// Collect all attributes of a roxmltree node into a `HashMap`.
///
/// Namespaced attributes are stored under both their local name and their
/// `prefix:local` form so that common lookups like `xlink:href` still work.
fn attrs_of(node: roxmltree::Node) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for attr in node.attributes() {
        let name = attr.name();
        let value = attr.value().to_string();
        // Always store under local name.
        map.insert(name.to_string(), value.clone());
        // Also under namespaced form when the attribute carries a namespace.
        if let Some(ns) = attr.namespace() {
            // Convert common namespaces to a prefix.
            let prefix = ns_to_prefix(ns);
            if !prefix.is_empty() {
                map.insert(format!("{prefix}:{name}"), value);
            }
        }
    }
    map
}

fn ns_to_prefix(ns: &str) -> &'static str {
    match ns {
        "http://www.w3.org/1999/xlink" => "xlink",
        "http://www.w3.org/XML/1998/namespace" => "xml",
        _ => "",
    }
}

/// Accumulate all text content in a subtree (like DOM `textContent`).
fn collect_text(node: roxmltree::Node) -> String {
    let mut out = String::new();
    for desc in node.descendants() {
        if desc.is_text()
            && let Some(t) = desc.text()
        {
            out.push_str(t);
        }
    }
    out
}

/// Build an `SvgNode` from a roxmltree element. Returns `None` for text or
/// non-element nodes at the position being iterated. Stop elements are
/// handled by their parent gradient builder and are skipped here.
#[allow(
    clippy::too_many_lines,
    reason = "one match arm per SVG element kind; splitting would scatter closely related parsing logic and harm readability"
)]
#[allow(
    clippy::similar_names,
    reason = "fx/fy mirror the SVG `fx`/`fy` attribute names; renaming would obscure the correspondence"
)]
fn build_node(
    xml_node: roxmltree::Node,
    stylesheet: &mut Stylesheet,
    lctx: LengthContext,
) -> Result<Option<SvgNode>, SvgError> {
    if !xml_node.is_element() {
        return Ok(None);
    }

    let tag = xml_node.tag_name().name();
    let attrs = attrs_of(xml_node);

    // Resolve a length attribute against the viewport for the given axis,
    // defaulting to `dflt` when the attribute is absent.
    let hlen = |k: &str, dflt: &str| {
        lctx.resolve(
            attrs.get(k).map_or(dflt, String::as_str),
            LengthType::Horizontal,
        )
    };
    let vlen = |k: &str, dflt: &str| {
        lctx.resolve(
            attrs.get(k).map_or(dflt, String::as_str),
            LengthType::Vertical,
        )
    };
    let olen = |k: &str, dflt: &str| {
        lctx.resolve(attrs.get(k).map_or(dflt, String::as_str), LengthType::Other)
    };

    // <style> elements contribute to the document stylesheet rather than
    // the rendered DOM. We still emit an Unknown("style") node so that
    // `extract_stylesheets` continues to work on already-parsed DOMs.
    if tag == "style" {
        let css_text = collect_text(xml_node);
        let parsed = Stylesheet::parse(&css_text);
        stylesheet.rules.extend(parsed.rules);

        let mut node = SvgNode::new(SvgNodeKind::Unknown("style".to_string()));
        apply_common_attrs(&mut node, &attrs);
        node.attributes
            .insert("__text_content".to_string(), css_text);
        return Ok(Some(node));
    }

    let mut node = match tag {
        "svg" => SvgNode::new(SvgNodeKind::Svg),
        "g" => SvgNode::new(SvgNodeKind::Group),
        "rect" => {
            // SVG: a missing (or negative) rx/ry is "auto", taking the
            // other radius (SkSVGRect::resolve).
            let rx = attrs
                .contains_key("rx")
                .then(|| hlen("rx", "0"))
                .filter(|r| *r >= 0.0);
            let ry = attrs
                .contains_key("ry")
                .then(|| vlen("ry", "0"))
                .filter(|r| *r >= 0.0);
            let (rx, ry) = match (rx, ry) {
                (Some(rx), Some(ry)) => (rx, ry),
                (Some(r), None) | (None, Some(r)) => (r, r),
                (None, None) => (0.0, 0.0),
            };
            SvgNode::new(SvgNodeKind::Rect(SvgRect {
                x: hlen("x", "0"),
                y: vlen("y", "0"),
                width: hlen("width", "0"),
                height: vlen("height", "0"),
                rx,
                ry,
            }))
        }
        "circle" => SvgNode::new(SvgNodeKind::Circle(SvgCircle {
            cx: hlen("cx", "0"),
            cy: vlen("cy", "0"),
            r: olen("r", "0"),
        })),
        "ellipse" => SvgNode::new(SvgNodeKind::Ellipse(SvgEllipse {
            cx: hlen("cx", "0"),
            cy: vlen("cy", "0"),
            rx: hlen("rx", "0"),
            ry: vlen("ry", "0"),
        })),
        "line" => SvgNode::new(SvgNodeKind::Line(SvgLine {
            x1: hlen("x1", "0"),
            y1: vlen("y1", "0"),
            x2: hlen("x2", "0"),
            y2: vlen("y2", "0"),
        })),
        "polyline" => {
            let points = parse_points(attrs.get("points").map_or("", String::as_str));
            SvgNode::new(SvgNodeKind::Polyline(points))
        }
        "polygon" => {
            let points = parse_points(attrs.get("points").map_or("", String::as_str));
            SvgNode::new(SvgNodeKind::Polygon(points))
        }
        "path" => {
            let d = attrs.get("d").map_or("", String::as_str);
            let path = parse_svg_path(d).unwrap_or_default();
            SvgNode::new(SvgNodeKind::Path(Box::new(path)))
        }
        "text" => {
            let content = collect_text(xml_node);
            SvgNode::new(SvgNodeKind::Text(SvgText {
                x: hlen("x", "0"),
                y: vlen("y", "0"),
                content,
                font_family: attrs.get("font-family").cloned(),
                font_size: olen("font-size", "12"),
                font_weight: attrs
                    .get("font-weight")
                    .and_then(|w| w.parse().ok())
                    .unwrap_or(400),
                text_anchor: match attrs.get("text-anchor").map(String::as_str) {
                    Some("middle") => TextAnchor::Middle,
                    Some("end") => TextAnchor::End,
                    _ => TextAnchor::Start,
                },
            }))
        }
        "image" => {
            let href = attrs
                .get("href")
                .or_else(|| attrs.get("xlink:href"))
                .cloned()
                .unwrap_or_default();
            SvgNode::new(SvgNodeKind::Image(SvgImage {
                x: hlen("x", "0"),
                y: vlen("y", "0"),
                width: hlen("width", "0"),
                height: vlen("height", "0"),
                href,
            }))
        }
        "defs" => SvgNode::new(SvgNodeKind::Defs),
        "clipPath" => {
            let id = attrs.get("id").cloned().unwrap_or_default();
            SvgNode::new(SvgNodeKind::ClipPath(id))
        }
        "linearGradient" => {
            let gradient = SvgLinearGradient {
                x1: parse_length(attrs.get("x1").map_or("0", String::as_str)),
                y1: parse_length(attrs.get("y1").map_or("0", String::as_str)),
                x2: parse_length(attrs.get("x2").map_or("100%", String::as_str)),
                y2: parse_length(attrs.get("y2").map_or("0", String::as_str)),
                stops: collect_stops(xml_node),
                spread: parse_spread(attrs.get("spreadMethod").map(String::as_str)),
                units: parse_gradient_units(attrs.get("gradientUnits").map(String::as_str)),
                transform: attrs
                    .get("gradientTransform")
                    .map_or(Matrix::IDENTITY, |s| parse_transform_str(s)),
            };
            SvgNode::new(SvgNodeKind::LinearGradient(gradient))
        }
        "radialGradient" => {
            let cx = parse_length(attrs.get("cx").map_or("50%", String::as_str));
            let cy = parse_length(attrs.get("cy").map_or("50%", String::as_str));
            let focal_x_str = attrs.get("fx").cloned();
            let focal_y_str = attrs.get("fy").cloned();
            let gradient = SvgRadialGradient {
                cx,
                cy,
                r: parse_length(attrs.get("r").map_or("50%", String::as_str)),
                fx: focal_x_str.map_or(cx, |s| parse_length(&s)),
                fy: focal_y_str.map_or(cy, |s| parse_length(&s)),
                stops: collect_stops(xml_node),
                spread: parse_spread(attrs.get("spreadMethod").map(String::as_str)),
                units: parse_gradient_units(attrs.get("gradientUnits").map(String::as_str)),
                transform: attrs
                    .get("gradientTransform")
                    .map_or(Matrix::IDENTITY, |s| parse_transform_str(s)),
            };
            SvgNode::new(SvgNodeKind::RadialGradient(gradient))
        }
        "use" => {
            let href = attrs
                .get("href")
                .or_else(|| attrs.get("xlink:href"))
                .cloned()
                .unwrap_or_default();
            SvgNode::new(SvgNodeKind::Use(href))
        }
        // Stops are consumed by their parent gradient; drop them here so
        // they don't appear as Unknown nodes in the rendered tree.
        "stop" => return Ok(None),
        _ => SvgNode::new(SvgNodeKind::Unknown(tag.to_string())),
    };

    apply_common_attrs(&mut node, &attrs);

    // `<use x y>` applies an additional translate after the element's own
    // transform, before rendering the referenced content (SkSVGUse).
    if let SvgNodeKind::Use(_) = node.kind {
        let ux = hlen("x", "0");
        let uy = vlen("y", "0");
        if ux != 0.0 || uy != 0.0 {
            node.transform = node.transform.concat(&Matrix::translate(ux, uy));
        }
    }

    // Recurse into children.
    for child in xml_node.children() {
        if let Some(child_node) = build_node(child, stylesheet, lctx)? {
            node.add_child(child_node);
        }
    }

    Ok(Some(node))
}

/// Apply common presentation attributes to a node.
fn apply_common_attrs(node: &mut SvgNode, attrs: &HashMap<String, String>) {
    node.id = attrs.get("id").cloned();

    if let Some(class) = attrs.get("class") {
        node.classes = class
            .split_whitespace()
            .map(std::string::ToString::to_string)
            .collect();
    }

    if let Some(transform) = attrs.get("transform") {
        node.transform = parse_transform_str(transform);
    }

    if let Some(fill) = attrs.get("fill") {
        node.fill = parse_paint(fill);
    }

    if let Some(stroke) = attrs.get("stroke") {
        node.stroke = parse_paint(stroke);
    }

    if let Some(sw) = attrs.get("stroke-width") {
        node.stroke_width = Some(parse_length(sw));
    }

    if let Some(color) = attrs.get("color") {
        // The `color` property feeds `currentColor`. `inherit` leaves it
        // unset so it flows from the parent presentation context.
        if !color.trim().eq_ignore_ascii_case("inherit") {
            node.color = parse_color(color);
        }
    }

    if let Some(fo) = attrs.get("fill-opacity") {
        node.fill_opacity = parse_opacity(fo);
    }

    if let Some(so) = attrs.get("stroke-opacity") {
        node.stroke_opacity = parse_opacity(so);
    }

    if let Some(opacity) = attrs.get("opacity") {
        node.opacity = parse_opacity(opacity).unwrap_or(1.0);
    }

    if let Some(visibility) = attrs.get("visibility") {
        node.visible = visibility != "hidden";
    }

    if let Some(display) = attrs.get("display")
        && display == "none"
    {
        node.visible = false;
    }

    // Preserve `style="..."` so css::apply_stylesheet can reapply inline
    // declarations after stylesheet rules.
    if let Some(style) = attrs.get("style") {
        node.attributes.insert("style".to_string(), style.clone());
    }

    // Preserve a handful of non-standard attributes that downstream code
    // reads directly. The CSS module stores the same keys on its own when
    // applying stroke-linecap / stroke-linejoin / etc. from stylesheets,
    // but inline attribute form needs a parser-side hook too.
    for key in [
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-dasharray",
        "stroke-dashoffset",
        "fill-rule",
        "clip-path",
    ] {
        if let Some(v) = attrs.get(key) {
            node.attributes.insert(key.to_string(), v.clone());
        }
    }
}

/// Collect `<stop>` children of a gradient element into `GradientStops`.
fn collect_stops(gradient: roxmltree::Node) -> Vec<GradientStop> {
    let mut stops = Vec::new();
    for child in gradient.children() {
        if !child.is_element() {
            continue;
        }
        if child.tag_name().name() != "stop" {
            continue;
        }
        let attrs = attrs_of(child);

        // Offset may be a plain number (0..1) or a percentage.
        let offset_str = attrs.get("offset").map_or("0", String::as_str);
        let offset = parse_length(offset_str).clamp(0.0, 1.0);

        // stop-color can live on the element or in a style attribute.
        let (stop_color, stop_opacity) = resolve_stop_style(&attrs);

        stops.push(GradientStop {
            offset,
            color: stop_color.unwrap_or(Color::BLACK),
            opacity: stop_opacity.unwrap_or(1.0),
        });
    }
    stops
}

fn resolve_stop_style(attrs: &HashMap<String, String>) -> (Option<Color>, Option<Scalar>) {
    let mut color = attrs.get("stop-color").and_then(|s| parse_color(s));
    let mut opacity = attrs
        .get("stop-opacity")
        .and_then(|s| s.parse::<Scalar>().ok());

    if let Some(style) = attrs.get("style") {
        for decl in style.split(';') {
            let decl = decl.trim();
            if let Some((prop, val)) = decl.split_once(':') {
                let prop = prop.trim();
                let val = val.trim();
                match prop {
                    "stop-color" if color.is_none() => {
                        color = parse_color(val);
                    }
                    "stop-opacity" if opacity.is_none() => {
                        opacity = val.parse().ok();
                    }
                    _ => {}
                }
            }
        }
    }

    (color, opacity)
}

fn parse_spread(s: Option<&str>) -> SpreadMethod {
    match s {
        Some("reflect") => SpreadMethod::Reflect,
        Some("repeat") => SpreadMethod::Repeat,
        _ => SpreadMethod::Pad,
    }
}

fn parse_gradient_units(s: Option<&str>) -> GradientUnits {
    match s {
        Some("userSpaceOnUse") => GradientUnits::UserSpaceOnUse,
        _ => GradientUnits::ObjectBoundingBox,
    }
}

/// Parse an SVG length value.
///
/// Recognises the unit suffixes defined by CSS Values 4. Viewport-relative
/// units (`vw`, `vh`, `vmin`, `vmax`) resolve against a 1000-unit-square
/// default viewport because the parser has no viewport context; the
/// renderer applies the real viewport transform separately via the
/// viewBox. Font-relative units (`em`, `rem`, `ex`, `ch`) assume a 16px
/// default font size — this matches browsers when no parent font is set.
/// Physical units (`cm`, `mm`, `in`, `pt`, `pc`) convert to CSS pixels at
/// 96dpi.
#[must_use]
pub fn parse_length(s: &str) -> Scalar {
    // Strip optional trailing unit; anything else is treated as a plain
    // number in user units. Ordered longest-first so that `1rem` matches
    // `rem` before falling through to `em`.
    const UNITS: &[&str] = &[
        "vmin", "vmax", "rem", "px", "pt", "pc", "em", "ex", "ch", "vw", "vh", "cm", "mm", "in",
    ];

    let s = s.trim();
    if s.is_empty() {
        return 0.0;
    }

    // Percentage is a different beast: return as a fraction in [0,1].
    if let Some(stripped) = s.strip_suffix('%') {
        return stripped.parse::<Scalar>().unwrap_or(0.0) / 100.0;
    }

    for unit in UNITS {
        if let Some(num) = s.strip_suffix(unit) {
            let n: Scalar = num.parse().unwrap_or(0.0);
            return match *unit {
                "pt" => n * 96.0 / 72.0,
                "pc" | "em" | "rem" => n * 16.0, // 1pc = 12pt = 16px
                "ex" | "ch" => n * 8.0,          // rough x-height/0-character-width approximation
                "vw" | "vh" | "vmin" | "vmax" => n * 10.0, // 1% of a 1000-unit viewport
                "cm" => n * 96.0 / 2.54,
                "mm" => n * 96.0 / 25.4,
                "in" => n * 96.0,
                _ => n, // "px" and unrecognized units pass through as-is
            };
        }
    }

    s.parse().unwrap_or(0.0)
}

/// Parse an `<opacity-value>` (CSS Color 3 §3.2): a number or a percentage,
/// clamped to `[0, 1]`. Returns `None` for unparseable input.
fn parse_opacity(s: &str) -> Option<Scalar> {
    let s = s.trim();
    let v = if let Some(pct) = s.strip_suffix('%') {
        pct.trim().parse::<Scalar>().ok()? / 100.0
    } else {
        s.parse::<Scalar>().ok()?
    };
    Some(v.clamp(0.0, 1.0))
}

/// Which viewport dimension a percentage length resolves against
/// (`SkSVGLengthContext::LengthType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LengthType {
    /// Resolves against the viewport width (x, width, cx, rx, x1…).
    Horizontal,
    /// Resolves against the viewport height (y, height, cy, ry, y1…).
    Vertical,
    /// Resolves against the normalized diagonal
    /// `sqrt(w² + h²) / sqrt(2)` (r, stroke-width, font-size…).
    Other,
}

/// The viewport a length context resolves percentages against. Mirrors
/// `SkSVGLengthContext::fViewport` — the nearest establishing viewport,
/// which is the `viewBox` size when present, else the intrinsic width/height.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LengthContext {
    /// Viewport width.
    pub vw: Scalar,
    /// Viewport height.
    pub vh: Scalar,
}

impl LengthContext {
    fn size_for_type(self, t: LengthType) -> Scalar {
        match t {
            LengthType::Horizontal => self.vw,
            LengthType::Vertical => self.vh,
            // https://www.w3.org/TR/SVG11/coords.html#Units_viewport_percentage
            LengthType::Other => (1.0 / std::f32::consts::SQRT_2) * self.vw.hypot(self.vh),
        }
    }

    /// Resolve a length string to user units. Percentages resolve against
    /// the viewport per `t`; all other units go through [`parse_length`].
    pub(crate) fn resolve(self, s: &str, t: LengthType) -> Scalar {
        let s = s.trim();
        if let Some(stripped) = s.strip_suffix('%') {
            let pct = stripped.trim().parse::<Scalar>().unwrap_or(0.0) / 100.0;
            return pct * self.size_for_type(t);
        }
        parse_length(s)
    }
}

/// Parse a `preserveAspectRatio` value (SVG 1.1 §7.8). Grammar:
/// `[defer] <align> [<meetOrSlice>]`. Unknown/malformed input falls back to
/// the initial `xMidYMid meet`.
fn parse_preserve_aspect_ratio(s: &str) -> PreserveAspectRatio {
    let mut tokens = s.split_whitespace().peekable();
    // Optional `defer` keyword (only meaningful on <image>, ignored here).
    if tokens.peek() == Some(&"defer") {
        tokens.next();
    }
    let align_tok = tokens.next().unwrap_or("xMidYMid");
    let align = match align_tok {
        "none" => None,
        "xMinYMin" => Some((AlignX::Min, AlignY::Min)),
        "xMidYMin" => Some((AlignX::Mid, AlignY::Min)),
        "xMaxYMin" => Some((AlignX::Max, AlignY::Min)),
        "xMinYMid" => Some((AlignX::Min, AlignY::Mid)),
        "xMaxYMid" => Some((AlignX::Max, AlignY::Mid)),
        "xMinYMax" => Some((AlignX::Min, AlignY::Max)),
        "xMidYMax" => Some((AlignX::Mid, AlignY::Max)),
        "xMaxYMax" => Some((AlignX::Max, AlignY::Max)),
        // "xMidYMid" and any unrecognized token fall back to the SVG default.
        _ => Some((AlignX::Mid, AlignY::Mid)),
    };
    let meet_or_slice = match tokens.next() {
        Some("slice") => MeetOrSlice::Slice,
        _ => MeetOrSlice::Meet,
    };
    PreserveAspectRatio {
        align,
        meet_or_slice,
    }
}

/// Parse a viewBox attribute.
fn parse_viewbox(s: &str) -> Option<Rect> {
    let parts: Vec<Scalar> = s
        .split_whitespace()
        .flat_map(|p| p.split(','))
        .filter_map(|p| p.parse().ok())
        .collect();

    if parts.len() == 4 {
        Some(Rect::from_xywh(parts[0], parts[1], parts[2], parts[3]))
    } else {
        None
    }
}

/// Parse points attribute (for polyline/polygon).
fn parse_points(s: &str) -> Vec<Point> {
    let nums: Vec<Scalar> = s
        .split_whitespace()
        .flat_map(|p| p.split(','))
        .filter_map(|p| p.parse().ok())
        .collect();

    nums.chunks(2)
        .filter_map(|c| {
            if c.len() == 2 {
                Some(Point::new(c[0], c[1]))
            } else {
                None
            }
        })
        .collect()
}

/// Parse a paint value.
///
/// Implements the SVG `<paint>` grammar (SVG 1.1 §11.2): `none`,
/// `currentColor`, a `<color>`, or `url(#id) [<color> | none]` where the
/// trailing token is a fallback used when the referenced paint server is
/// missing. Returns `None` only for unparseable input so the caller leaves
/// the property inherited.
pub(crate) fn parse_paint(s: &str) -> Option<SvgPaint> {
    let s = s.trim();
    if s == "none" {
        Some(SvgPaint::None)
    } else if s.eq_ignore_ascii_case("currentcolor") {
        Some(SvgPaint::CurrentColor)
    } else if let Some(url_body) = s.strip_prefix("url(") {
        // Split the funciri from an optional fallback: `url(#id) red`.
        let close = url_body.find(')')?;
        let url = url_body[..close].trim_matches(|c| c == '"' || c == '\'');
        let fallback_str = url_body[close + 1..].trim();
        let fallback = if fallback_str.is_empty() || fallback_str == "none" {
            None
        } else if fallback_str.eq_ignore_ascii_case("currentcolor") {
            // Represent a currentColor fallback as black — a rare case; the
            // reference is normally present so this only matters when it is
            // not. Full currentColor-fallback resolution is out of scope.
            Some(Color::BLACK)
        } else {
            parse_color(fallback_str)
        };
        Some(SvgPaint::Url(url.to_string(), fallback))
    } else {
        parse_color(s).map(SvgPaint::Color)
    }
}

/// Parse a color value.
///
/// Supports `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, `rgb()`, `rgba()`,
/// `hsl()`, `hsla()`, and the full X11 / CSS Color Level 3 named color
/// table. `currentColor` is *not* handled here (it is a paint keyword, see
/// [`parse_paint`]); this returns `None` for it.
pub(crate) fn parse_color(s: &str) -> Option<Color> {
    if s.trim().eq_ignore_ascii_case("currentcolor") {
        return None;
    }
    Color::from_css(s)
}

/// Parse a transform string (public for CSS module).
///
/// # Panics
///
/// Never panics in practice: the internal `unwrap()` only fires on a
/// character just confirmed present via `peek()`.
#[must_use]
pub fn parse_transform_str(s: &str) -> Matrix {
    let mut result = Matrix::IDENTITY;
    let s = s.trim();

    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_alphabetic() {
            let mut name = String::from(c);
            while let Some(&ch) = chars.peek() {
                if ch == '(' {
                    break;
                }
                name.push(chars.next().unwrap());
            }

            if chars.peek() == Some(&'(') {
                chars.next();
            }

            let mut args = String::new();
            while let Some(&ch) = chars.peek() {
                if ch == ')' {
                    chars.next();
                    break;
                }
                args.push(chars.next().unwrap());
            }

            let nums: Vec<Scalar> = args
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|p| p.parse().ok())
                .collect();

            let transform = match name.as_str() {
                "translate" => {
                    let dx = nums.first().copied().unwrap_or(0.0);
                    let dy = nums.get(1).copied().unwrap_or(0.0);
                    Matrix::translate(dx, dy)
                }
                "scale" => {
                    let sx = nums.first().copied().unwrap_or(1.0);
                    let sy = nums.get(1).copied().unwrap_or(sx);
                    Matrix::scale(sx, sy)
                }
                "rotate" => {
                    let angle = nums.first().copied().unwrap_or(0.0);
                    let radians = angle.to_radians();
                    if nums.len() >= 3 {
                        let cx = nums[1];
                        let cy = nums[2];
                        Matrix::translate(cx, cy)
                            .concat(&Matrix::rotate(radians))
                            .concat(&Matrix::translate(-cx, -cy))
                    } else {
                        Matrix::rotate(radians)
                    }
                }
                "skewX" => {
                    let angle = nums.first().copied().unwrap_or(0.0);
                    Matrix::skew(angle.to_radians().tan(), 0.0)
                }
                "skewY" => {
                    let angle = nums.first().copied().unwrap_or(0.0);
                    Matrix::skew(0.0, angle.to_radians().tan())
                }
                "matrix" if nums.len() >= 6 => Matrix {
                    values: [
                        nums[0], nums[2], nums[4], nums[1], nums[3], nums[5], 0.0, 0.0, 1.0,
                    ],
                },
                _ => Matrix::IDENTITY,
            };

            result = result.concat(&transform);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test assertions against literals from unit-less/px-suffixed inputs"
    )]
    fn test_parse_length() {
        assert_eq!(parse_length("100"), 100.0);
        assert_eq!(parse_length("50px"), 50.0);
        assert!((parse_length("50%") - 0.5).abs() < 0.01);
        assert!((parse_length("1in") - 96.0).abs() < 0.01);
        assert!((parse_length("1cm") - 96.0 / 2.54).abs() < 0.01);
        assert!((parse_length("1em") - 16.0).abs() < 0.01);
        assert!((parse_length("1rem") - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_color() {
        assert_eq!(parse_color("#ff0000"), Some(Color::from_rgb(255, 0, 0)));
        assert_eq!(parse_color("#f00"), Some(Color::from_rgb(255, 0, 0)));
        assert_eq!(
            parse_color("rgb(255, 0, 0)"),
            Some(Color::from_rgb(255, 0, 0))
        );
        assert_eq!(parse_color("red"), Some(Color::from_rgb(255, 0, 0)));
        // Extended named colors.
        assert_eq!(
            parse_color("cornflowerblue"),
            Some(Color::from_rgb(100, 149, 237))
        );
        assert_eq!(
            parse_color("rebeccapurple"),
            Some(Color::from_rgb(102, 51, 153))
        );
        // 8-digit hex.
        assert_eq!(
            parse_color("#ff000080"),
            Some(Color::from_argb(128, 255, 0, 0))
        );
        // 4-digit hex.
        assert_eq!(parse_color("#f008"), Some(Color::from_argb(136, 255, 0, 0)));
        // HSL.
        let hsl_red = parse_color("hsl(0, 100%, 50%)").unwrap();
        assert_eq!(hsl_red.red(), 255);
        assert!(hsl_red.green() < 5);
        assert!(hsl_red.blue() < 5);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test assertion against a literal parsed from fixed SVG text"
    )]
    fn test_parse_simple_svg() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" fill="red"/>
        </svg>"#;

        let dom = parse_svg(svg).unwrap();
        assert_eq!(dom.width, 100.0);
        assert_eq!(dom.height, 100.0);
        assert_eq!(dom.root.children.len(), 1);
    }

    #[test]
    fn test_parse_transform() {
        let m = parse_transform_str("translate(10, 20)");
        assert!((m.values[2] - 10.0).abs() < 0.01);
        assert!((m.values[5] - 20.0).abs() < 0.01);

        let m = parse_transform_str("scale(2)");
        assert!((m.values[0] - 2.0).abs() < 0.01);
        assert!((m.values[4] - 2.0).abs() < 0.01);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test assertion against a literal parsed from a fixed font-size attribute"
    )]
    fn test_text_content_captured() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <text x="10" y="20" font-size="14">Hello, world!</text>
        </svg>"#;

        let dom = parse_svg(svg).unwrap();
        let text = &dom.root.children[0];
        if let SvgNodeKind::Text(t) = &text.kind {
            assert_eq!(t.content, "Hello, world!");
            assert_eq!(t.font_size, 14.0);
        } else {
            panic!("expected text node");
        }
    }

    #[test]
    fn test_xml_entities_decoded() {
        // roxmltree decodes entities for us; verify it reaches the text
        // field.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
            <text x="0" y="20">AT&amp;T &lt;3</text>
        </svg>"#;

        let dom = parse_svg(svg).unwrap();
        if let SvgNodeKind::Text(t) = &dom.root.children[0].kind {
            assert_eq!(t.content, "AT&T <3");
        } else {
            panic!("expected text node");
        }
    }

    #[test]
    fn test_gradient_stops_parsed() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g">
              <stop offset="0" stop-color="red"/>
              <stop offset="0.5" stop-color="green" stop-opacity="0.5"/>
              <stop offset="100%" style="stop-color: blue"/>
            </linearGradient>
          </defs>
        </svg>"#;

        let dom = parse_svg(svg).unwrap();
        let gradient = dom.root.find_by_id("g").expect("gradient present");
        if let SvgNodeKind::LinearGradient(g) = &gradient.kind {
            assert_eq!(g.stops.len(), 3);
            assert_eq!(g.stops[0].color, Color::from_rgb(255, 0, 0));
            assert!((g.stops[1].opacity - 0.5).abs() < 0.001);
            assert_eq!(g.stops[2].color, Color::from_rgb(0, 0, 255));
            assert!((g.stops[2].offset - 1.0).abs() < 0.001);
        } else {
            panic!("expected linear gradient");
        }
    }

    #[test]
    fn test_style_element_populates_stylesheet() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
          <style>rect { fill: red; }</style>
          <rect x="0" y="0" width="10" height="10"/>
        </svg>"#;

        let dom = parse_svg(svg).unwrap();
        assert_eq!(dom.stylesheet.rules.len(), 1);
    }

    #[test]
    fn test_inline_style_preserved() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="10" height="10" style="fill: blue"/>
        </svg>"#;

        let dom = parse_svg(svg).unwrap();
        let rect = &dom.root.children[0];
        assert_eq!(
            rect.attributes.get("style").map(String::as_str),
            Some("fill: blue")
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test assertions against literals parsed from fixed SVG attributes"
    )]
    fn test_image_parsed() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
          <image x="5" y="6" width="100" height="200" href="data:image/png;base64,AAAA"/>
        </svg>"#;
        let dom = parse_svg(svg).unwrap();
        if let SvgNodeKind::Image(img) = &dom.root.children[0].kind {
            assert_eq!(img.x, 5.0);
            assert_eq!(img.y, 6.0);
            assert_eq!(img.width, 100.0);
            assert_eq!(img.height, 200.0);
            assert!(img.href.starts_with("data:image/png;base64,"));
        } else {
            panic!("expected image node");
        }
    }

    #[test]
    fn test_percentage_length_resolves_against_viewport() {
        // width% resolves against viewport width, height% against height,
        // per SkSVGLengthContext (previously a raw 0..1 fraction).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
            <rect x="0" y="0" width="50%" height="50%"/>
        </svg>"#;
        let dom = parse_svg(svg).unwrap();
        if let SvgNodeKind::Rect(r) = &dom.root.children[0].kind {
            assert!(
                (r.width - 100.0).abs() < 0.01,
                "50% of 200, got {}",
                r.width
            );
            assert!(
                (r.height - 50.0).abs() < 0.01,
                "50% of 100, got {}",
                r.height
            );
        } else {
            panic!("expected rect");
        }
    }

    #[test]
    fn test_percentage_r_resolves_against_diagonal() {
        // circle r% resolves against the normalized diagonal
        // sqrt(w^2 + h^2)/sqrt(2). For 100x100 that is 100.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <circle cx="0" cy="0" r="50%"/>
        </svg>"#;
        let dom = parse_svg(svg).unwrap();
        if let SvgNodeKind::Circle(c) = &dom.root.children[0].kind {
            assert!(
                (c.r - 50.0).abs() < 0.01,
                "50% of 100 diagonal, got {}",
                c.r
            );
        } else {
            panic!("expected circle");
        }
    }

    #[test]
    fn test_length_context_diagonal() {
        // Non-square viewport: diagonal = sqrt(3^2+4^2)/sqrt(2) = 5/sqrt2.
        let ctx = LengthContext { vw: 3.0, vh: 4.0 };
        let expected = 5.0 / std::f32::consts::SQRT_2;
        assert!((ctx.resolve("100%", LengthType::Other) - expected).abs() < 0.001);
        assert!((ctx.resolve("100%", LengthType::Horizontal) - 3.0).abs() < 0.001);
        assert!((ctx.resolve("100%", LengthType::Vertical) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_paint_currentcolor() {
        assert!(matches!(
            parse_paint("currentColor"),
            Some(SvgPaint::CurrentColor)
        ));
        assert!(matches!(
            parse_paint("currentcolor"),
            Some(SvgPaint::CurrentColor)
        ));
    }

    #[test]
    fn test_parse_paint_url_fallback() {
        // url(#id) <color> — fallback used when the reference is missing.
        match parse_paint("url(#grad) red") {
            Some(SvgPaint::Url(id, Some(color))) => {
                assert_eq!(id, "#grad");
                assert_eq!(color, Color::from_rgb(255, 0, 0));
            }
            other => panic!("expected url with fallback, got {other:?}"),
        }
        // No fallback.
        match parse_paint("url(#grad)") {
            Some(SvgPaint::Url(id, None)) => assert_eq!(id, "#grad"),
            other => panic!("expected url without fallback, got {other:?}"),
        }
        // Explicit `none` fallback.
        assert!(matches!(
            parse_paint("url(#grad) none"),
            Some(SvgPaint::Url(_, None))
        ));
    }

    #[test]
    fn test_parse_preserve_aspect_ratio() {
        assert_eq!(
            parse_preserve_aspect_ratio("none"),
            PreserveAspectRatio {
                align: None,
                meet_or_slice: MeetOrSlice::Meet
            }
        );
        assert_eq!(
            parse_preserve_aspect_ratio("xMinYMax slice"),
            PreserveAspectRatio {
                align: Some((AlignX::Min, AlignY::Max)),
                meet_or_slice: MeetOrSlice::Slice
            }
        );
        // Default when omitted.
        assert_eq!(
            PreserveAspectRatio::default(),
            PreserveAspectRatio {
                align: Some((AlignX::Mid, AlignY::Mid)),
                meet_or_slice: MeetOrSlice::Meet
            }
        );
        // Parsed from the root element.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" preserveAspectRatio="xMaxYMin slice"/>"#;
        let dom = parse_svg(svg).unwrap();
        assert_eq!(
            dom.preserve_aspect_ratio,
            PreserveAspectRatio {
                align: Some((AlignX::Max, AlignY::Min)),
                meet_or_slice: MeetOrSlice::Slice
            }
        );
    }

    #[test]
    fn test_use_x_y_translation() {
        // <use x y> bakes a translate into the node transform.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg">
            <use href="#r" x="10" y="20"/>
        </svg>"##;
        let dom = parse_svg(svg).unwrap();
        let use_node = &dom.root.children[0];
        assert!((use_node.transform.values[2] - 10.0).abs() < 0.01);
        assert!((use_node.transform.values[5] - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_presentation_attrs_unset_by_default() {
        // A bare rect specifies no presentation properties: they must be
        // None so inheritance can supply them (the black-fill default lives
        // at the render root, not on the node).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="10" height="10"/>
        </svg>"#;
        let dom = parse_svg(svg).unwrap();
        let rect = &dom.root.children[0];
        assert!(rect.fill.is_none());
        assert!(rect.stroke.is_none());
        assert!(rect.stroke_width.is_none());
    }

    #[test]
    fn test_fill_opacity_parsed_as_property() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="10" height="10" fill="red" fill-opacity="0.5"/>
        </svg>"#;
        let dom = parse_svg(svg).unwrap();
        let rect = &dom.root.children[0];
        assert!((rect.fill_opacity.unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_radial_gradient_stops() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="rg" cx="50%" cy="50%" r="50%">
              <stop offset="0" stop-color="white"/>
              <stop offset="1" stop-color="black"/>
            </radialGradient>
          </defs>
        </svg>"#;
        let dom = parse_svg(svg).unwrap();
        let grad = dom.root.find_by_id("rg").unwrap();
        if let SvgNodeKind::RadialGradient(g) = &grad.kind {
            assert_eq!(g.stops.len(), 2);
        } else {
            panic!("expected radial gradient");
        }
    }
}
