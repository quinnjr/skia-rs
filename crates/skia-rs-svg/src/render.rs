//! SVG rendering to canvas.
//!
//! Walks the `SvgDom`, resolves paint references (gradients), applies the
//! parsed CSS stylesheet, renders text via skia-rs-text glyph paths, and
//! decodes `<image>` data URIs through skia-rs-codec. Clip paths are
//! applied via `Canvas::clip_path` using the union of geometry inside
//! the referenced `<clipPath>` element.

use crate::css::apply_stylesheet;
use crate::dom::{
    AlignX, AlignY, GradientStop, GradientUnits, MeetOrSlice, PreserveAspectRatio, SpreadMethod,
    SvgDom, SvgImage, SvgNode, SvgNodeKind, SvgPaint, SvgText, TextAnchor,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use skia_rs_canvas::{Canvas, ClipOp, SaveLayerFlags, SaveLayerRec, Surface};
use skia_rs_codec::decode_image;
use skia_rs_core::{Color, Color4f, Matrix, Point, Rect, Scalar};
use skia_rs_paint::{
    LinearGradient as PaintLinearGradient, Paint, RadialGradient as PaintRadialGradient, ShaderRef,
    StrokeCap, StrokeJoin, Style, TileMode,
};
use skia_rs_path::{DashEffect, FillType, Path, PathBuilder, PathEffectRef};
use skia_rs_text::Font;
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum `<use>` reference depth before the render walk bails out. Guards
/// against malformed documents whose `<use>` elements form a cycle (which
/// would otherwise recurse until the stack overflows). SVG has no legal use
/// for nesting this deep.
const MAX_USE_DEPTH: u32 = 64;

/// Resolved, inheritable presentation state threaded down the render walk.
///
/// Mirrors the inherited half of `SkSVGPresentationContext` — every field
/// here is a CSS-inherited property, initialised to the values from
/// `SkSVGPresentationAttributes::MakeInitial`. `opacity` is deliberately
/// absent: it composites the element as a unit and is handled per node.
#[derive(Clone)]
struct PresentationState {
    /// Fill paint selector (initial: black).
    fill: SvgPaint,
    /// Stroke paint selector (initial: none).
    stroke: SvgPaint,
    /// Stroke width in user units (initial: 1).
    stroke_width: Scalar,
    /// Fill opacity (initial: 1).
    fill_opacity: Scalar,
    /// Stroke opacity (initial: 1).
    stroke_opacity: Scalar,
    /// `color` property, resolves `currentColor` (initial: black).
    color: Color,
    /// Fill rule (initial: nonzero winding).
    fill_rule: FillType,
    /// Stroke line cap (initial: butt).
    stroke_cap: StrokeCap,
    /// Stroke line join (initial: miter).
    stroke_join: StrokeJoin,
    /// Dash intervals in user units (initial: none).
    dash_array: Option<Vec<Scalar>>,
    /// Dash phase in user units (initial: 0).
    dash_offset: Scalar,
}

impl PresentationState {
    /// The document-root initial presentation state
    /// (`SkSVGPresentationAttributes::MakeInitial`).
    const fn initial() -> Self {
        Self {
            fill: SvgPaint::Color(Color::BLACK),
            stroke: SvgPaint::None,
            stroke_width: 1.0,
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            color: Color::BLACK,
            fill_rule: FillType::Winding,
            stroke_cap: StrokeCap::Butt,
            stroke_join: StrokeJoin::Miter,
            dash_array: None,
            dash_offset: 0.0,
        }
    }

    /// Derive the child state for `node`: start from the parent (inheritance)
    /// and override with any properties the element specifies.
    fn resolve(&self, node: &SvgNode) -> Self {
        let mut s = self.clone();
        // `color` is resolved first so that a `fill="currentColor"` on the
        // same element sees this element's `color`.
        if let Some(c) = node.color {
            s.color = c;
        }
        if let Some(f) = &node.fill {
            s.fill = f.clone();
        }
        if let Some(st) = &node.stroke {
            s.stroke = st.clone();
        }
        if let Some(w) = node.stroke_width {
            s.stroke_width = w;
        }
        if let Some(o) = node.fill_opacity {
            s.fill_opacity = o;
        }
        if let Some(o) = node.stroke_opacity {
            s.stroke_opacity = o;
        }
        if let Some(v) = node.attributes.get("fill-rule") {
            s.fill_rule = match v.trim() {
                "evenodd" => FillType::EvenOdd,
                _ => FillType::Winding,
            };
        }
        if let Some(v) = node.attributes.get("stroke-linecap") {
            s.stroke_cap = match v.trim() {
                "round" => StrokeCap::Round,
                "square" => StrokeCap::Square,
                _ => StrokeCap::Butt,
            };
        }
        if let Some(v) = node.attributes.get("stroke-linejoin") {
            s.stroke_join = match v.trim() {
                "round" => StrokeJoin::Round,
                "bevel" => StrokeJoin::Bevel,
                _ => StrokeJoin::Miter,
            };
        }
        if let Some(v) = node.attributes.get("stroke-dasharray") {
            s.dash_array = parse_dash_array(v);
        }
        if let Some(v) = node.attributes.get("stroke-dashoffset") {
            s.dash_offset = crate::parser::parse_length(v);
        }
        s
    }
}

/// Parse a `stroke-dasharray` value into user-unit intervals. Returns `None`
/// for `none`, empty, or all-zero patterns (which disable dashing).
fn parse_dash_array(s: &str) -> Option<Vec<Scalar>> {
    let s = s.trim();
    if s.is_empty() || s == "none" {
        return None;
    }
    let intervals: Vec<Scalar> = s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(crate::parser::parse_length)
        .collect();
    if intervals.is_empty() || intervals.iter().all(|v| *v <= 0.0) {
        None
    } else {
        Some(intervals)
    }
}

/// Resolved context passed through the render walk.
///
/// Built once at the top of `render_svg` so that every call to
/// `create_paint_from_svg_paint` has O(1) access to gradient/clipPath
/// lookup instead of re-walking the DOM for each element.
struct RenderContext<'a> {
    /// id -> `SvgNode` for all nodes in the DOM that carry an id.
    defs: HashMap<String, &'a SvgNode>,
}

impl<'a> RenderContext<'a> {
    fn new(dom: &'a SvgDom) -> Self {
        let mut defs = HashMap::new();
        collect_defs(&dom.root, &mut defs);
        Self { defs }
    }

    fn lookup(&self, href: &str) -> Option<&'a SvgNode> {
        let id = href.trim_start_matches('#');
        self.defs.get(id).copied()
    }
}

fn collect_defs<'a>(node: &'a SvgNode, defs: &mut HashMap<String, &'a SvgNode>) {
    if let Some(ref id) = node.id {
        defs.insert(id.clone(), node);
    }
    for child in &node.children {
        collect_defs(child, defs);
    }
}

/// Render an SVG DOM to a surface.
pub fn render_svg_to_surface(dom: &SvgDom, surface: &mut Surface) {
    let mut canvas = surface.canvas();
    render_svg(dom, &mut canvas);
}

/// Render an SVG DOM to a raster canvas.
///
/// Applies the DOM's parsed stylesheet onto a working clone of the DOM
/// before walking so that CSS-styled attributes (fill/stroke/opacity/etc.)
/// take effect without mutating the caller's DOM.
pub fn render_svg(dom: &SvgDom, canvas: &mut Canvas<'_>) {
    // Apply stylesheet onto a cloned DOM so the caller's is untouched.
    let mut working = dom.clone();
    let sheet = working.stylesheet.clone();
    if !sheet.rules.is_empty() {
        apply_stylesheet(&mut working, &sheet);
    } else if dom_has_inline_styles(&working.root) {
        // Even with no <style> block, inline style= attributes may still
        // need to take effect. apply_stylesheet always walks inline
        // styles, so call it with an empty stylesheet whenever any node
        // has a style attribute.
        apply_stylesheet(&mut working, &crate::css::Stylesheet::new());
    }

    // Map the viewBox into the device viewport honouring preserveAspectRatio
    // (SkSVGSVG::onPrepareToRender + ComputeViewboxMatrix).
    let view_box = working.get_view_box();
    #[allow(
        clippy::cast_precision_loss,
        reason = "canvas dimensions are pixel counts far below 2^24; exact in f32"
    )]
    let viewport = Rect::from_xywh(
        0.0,
        0.0,
        canvas.width() as Scalar,
        canvas.height() as Scalar,
    );

    render_dom_into_viewport(&working, canvas, viewport, view_box);
}

/// Shared render prologue for [`render_svg`] and [`render_svg_in_container`]:
/// builds the render context, maps `view_box` into `viewport` honouring
/// `preserveAspectRatio`, and walks the DOM.
///
/// `working` must already have its stylesheet applied. No-ops when
/// `view_box` is degenerate.
fn render_dom_into_viewport(
    working: &SvgDom,
    canvas: &mut Canvas<'_>,
    viewport: Rect,
    view_box: Rect,
) {
    if view_box.width() <= 0.0 || view_box.height() <= 0.0 {
        return;
    }

    let ctx = RenderContext::new(working);
    let content_matrix =
        compute_viewbox_matrix(&view_box, &viewport, working.preserve_aspect_ratio);

    canvas.save();
    canvas.concat(&content_matrix);

    render_node(
        &working.root,
        canvas,
        &ctx,
        &PresentationState::initial(),
        0,
    );

    canvas.restore();
}

/// Compute the viewBox→viewport matrix for a `preserveAspectRatio`
/// (`SkSVGNode::ComputeViewboxMatrix`).
fn compute_viewbox_matrix(view_box: &Rect, viewport: &Rect, par: PreserveAspectRatio) -> Matrix {
    if view_box.width() <= 0.0
        || view_box.height() <= 0.0
        || viewport.width() <= 0.0
        || viewport.height() <= 0.0
    {
        return Matrix::scale(0.0, 0.0);
    }

    let sx = viewport.width() / view_box.width();
    let sy = viewport.height() / view_box.height();

    // "none" -> anisotropic scaling regardless of meet/slice.
    let (scale_x, scale_y) = if par.align.is_none() {
        (sx, sy)
    } else {
        let s = match par.meet_or_slice {
            MeetOrSlice::Meet => sx.min(sy),
            MeetOrSlice::Slice => sx.max(sy),
        };
        (s, s)
    };

    let (cx, cy) = match par.align {
        None => (0.0, 0.0),
        Some((ax, ay)) => (
            match ax {
                AlignX::Min => 0.0,
                AlignX::Mid => 0.5,
                AlignX::Max => 1.0,
            },
            match ay {
                AlignY::Min => 0.0,
                AlignY::Mid => 0.5,
                AlignY::Max => 1.0,
            },
        ),
    };

    let tx = (-view_box.left).mul_add(
        scale_x,
        view_box.width().mul_add(-scale_x, viewport.width()) * cx,
    );
    let ty = (-view_box.top).mul_add(
        scale_y,
        view_box.height().mul_add(-scale_y, viewport.height()) * cy,
    );

    Matrix::translate(viewport.left + tx, viewport.top + ty)
        .concat(&Matrix::scale(scale_x, scale_y))
}

fn dom_has_inline_styles(node: &SvgNode) -> bool {
    if node.attributes.contains_key("style") {
        return true;
    }
    node.children.iter().any(dom_has_inline_styles)
}

/// True when a node draws a single atomic shape and has no descendants —
/// the condition under which `opacity` can be applied as paint alpha rather
/// than an offscreen layer (`SkSVGRenderContext::applyOpacity`, kLeaf).
const fn is_leaf_shape(node: &SvgNode) -> bool {
    node.children.is_empty()
        && matches!(
            node.kind,
            SvgNodeKind::Rect(_)
                | SvgNodeKind::Circle(_)
                | SvgNodeKind::Ellipse(_)
                | SvgNodeKind::Line(_)
                | SvgNodeKind::Polyline(_)
                | SvgNodeKind::Polygon(_)
                | SvgNodeKind::Path(_)
        )
}

/// Render a single SVG node under the inherited presentation `parent`.
///
/// `depth` bounds `<use>` recursion so cyclic references cannot overflow the
/// stack.
#[allow(
    clippy::too_many_lines,
    reason = "one match arm per SVG element kind; splitting would scatter closely related rendering logic and harm readability"
)]
fn render_node(
    node: &SvgNode,
    canvas: &mut Canvas<'_>,
    ctx: &RenderContext<'_>,
    parent: &PresentationState,
    depth: u32,
) {
    if !node.visible {
        return;
    }

    // Resolve this element's presentation state against the inherited one.
    let state = parent.resolve(node);

    canvas.save();
    canvas.concat(&node.transform);

    // clip-path reference applied before drawing the node contents.
    let clip_ref = node.attributes.get("clip-path").cloned();
    if let Some(href) = clip_ref.as_deref()
        && let Some(id) = extract_url_id(href)
        && let Some(clip_node) = ctx.lookup(id)
        && let SvgNodeKind::ClipPath(_) = &clip_node.kind
    {
        let path = build_clip_path(clip_node);
        if !path.is_empty() {
            canvas.clip_path(&path, ClipOp::Intersect, true);
        }
    }

    // Group/element opacity (SkSVGRenderContext::applyOpacity). We can fold
    // opacity into paint alpha only when it affects a single atomic draw:
    // a leaf shape with exactly one of fill/stroke. Otherwise we composite
    // through an offscreen layer so overlapping coverage isn't double-hit.
    let has_fill = !matches!(state.fill, SvgPaint::None);
    let has_stroke = !matches!(state.stroke, SvgPaint::None);
    let opacity = node.opacity.clamp(0.0, 1.0);
    let can_defer = opacity < 1.0 && is_leaf_shape(node) && (has_fill ^ has_stroke);
    let deferred = if can_defer { opacity } else { 1.0 };
    let use_layer = opacity < 1.0 && !can_defer && !matches!(node.kind, SvgNodeKind::Image(_));
    let layer_open = if use_layer {
        let mut layer_paint = Paint::new();
        layer_paint.set_alpha(opacity);
        canvas.save_layer(&SaveLayerRec {
            bounds: None,
            paint: Some(&layer_paint),
            flags: SaveLayerFlags::default(),
        });
        true
    } else {
        false
    };

    let node_bounds_for_gradient = node.bounds();

    let fill_paint = if has_fill {
        build_paint(
            &state.fill,
            Style::Fill,
            state.fill_opacity,
            deferred,
            &state,
            ctx,
            node_bounds_for_gradient,
        )
    } else {
        None
    };

    let stroke_paint = if has_stroke {
        build_paint(
            &state.stroke,
            Style::Stroke,
            state.stroke_opacity,
            deferred,
            &state,
            ctx,
            node_bounds_for_gradient,
        )
        .map(|mut paint| {
            paint.set_stroke_width(state.stroke_width);
            paint.set_stroke_cap(state.stroke_cap);
            paint.set_stroke_join(state.stroke_join);
            if let Some(effect) = build_dash_effect(&state) {
                paint.set_path_effect(Some(effect));
            }
            paint
        })
    } else {
        None
    };

    match &node.kind {
        // SVG does not render a rect with a non-positive width or height.
        SvgNodeKind::Rect(rect) if rect.width > 0.0 && rect.height > 0.0 => {
            let r = Rect::from_xywh(rect.x, rect.y, rect.width, rect.height);
            let (rx, ry) = rect_radii(rect);
            // SkRRect::MakeRectXY: a radius <= 0 gives square corners.
            if rx > 0.0 && ry > 0.0 {
                if let Some(paint) = &fill_paint {
                    canvas.draw_round_rect(&r, rx, ry, paint);
                }
                if let Some(paint) = &stroke_paint {
                    canvas.draw_round_rect(&r, rx, ry, paint);
                }
            } else {
                if let Some(paint) = &fill_paint {
                    canvas.draw_rect(&r, paint);
                }
                if let Some(paint) = &stroke_paint {
                    canvas.draw_rect(&r, paint);
                }
            }
        }
        SvgNodeKind::Circle(circle) => {
            let center = Point::new(circle.cx, circle.cy);
            if let Some(paint) = &fill_paint {
                canvas.draw_circle(center, circle.r, paint);
            }
            if let Some(paint) = &stroke_paint {
                canvas.draw_circle(center, circle.r, paint);
            }
        }
        SvgNodeKind::Ellipse(ellipse) => {
            let oval = Rect::from_xywh(
                ellipse.cx - ellipse.rx,
                ellipse.cy - ellipse.ry,
                ellipse.rx * 2.0,
                ellipse.ry * 2.0,
            );
            if let Some(paint) = &fill_paint {
                canvas.draw_oval(&oval, paint);
            }
            if let Some(paint) = &stroke_paint {
                canvas.draw_oval(&oval, paint);
            }
        }
        SvgNodeKind::Line(line) => {
            if let Some(paint) = &stroke_paint {
                // `draw_line` applies the paint's path effect, so
                // stroke-dasharray needs no special handling here.
                canvas.draw_line(
                    Point::new(line.x1, line.y1),
                    Point::new(line.x2, line.y2),
                    paint,
                );
            }
        }
        SvgNodeKind::Polyline(points) => {
            // A polyline is an open path: per SVG it is *filled* (with the
            // contour implicitly closed) and *stroked* (left open).
            if points.len() >= 2 {
                if let Some(paint) = &fill_paint {
                    let mut b = PathBuilder::new();
                    b.move_to(points[0].x, points[0].y);
                    for p in &points[1..] {
                        b.line_to(p.x, p.y);
                    }
                    b.close();
                    let mut path = b.build();
                    path.set_fill_type(state.fill_rule);
                    canvas.draw_path(&path, paint);
                }
                if let Some(paint) = &stroke_paint {
                    let mut b = PathBuilder::new();
                    b.move_to(points[0].x, points[0].y);
                    for p in &points[1..] {
                        b.line_to(p.x, p.y);
                    }
                    canvas.draw_path(&b.build(), paint);
                }
            }
        }
        SvgNodeKind::Polygon(points) => {
            // Two points still make a closed polygon (SkSVGPoly strokes it
            // out and back); its fill covers no area.
            if points.len() >= 2 {
                let mut builder = PathBuilder::new();
                builder.move_to(points[0].x, points[0].y);
                for p in &points[1..] {
                    builder.line_to(p.x, p.y);
                }
                builder.close();
                let mut path = builder.build();
                path.set_fill_type(state.fill_rule);
                if let Some(paint) = &fill_paint {
                    canvas.draw_path(&path, paint);
                }
                if let Some(paint) = &stroke_paint {
                    canvas.draw_path(&path, paint);
                }
            }
        }
        SvgNodeKind::Path(path) => {
            // Honour the resolved fill-rule for the fill pass.
            let mut filled = path.clone();
            filled.set_fill_type(state.fill_rule);
            if let Some(paint) = &fill_paint {
                canvas.draw_path(&filled, paint);
            }
            if let Some(paint) = &stroke_paint {
                canvas.draw_path(path, paint);
            }
        }
        SvgNodeKind::Text(text) => {
            render_text(text, canvas, fill_paint.as_ref(), stroke_paint.as_ref());
        }
        SvgNodeKind::Image(img) => {
            render_image(img, canvas, opacity);
        }
        SvgNodeKind::Use(href) => {
            if depth < MAX_USE_DEPTH
                && let Some(referenced) = ctx.lookup(href)
            {
                render_node(referenced, canvas, ctx, &state, depth + 1);
            }
        }
        SvgNodeKind::Group | SvgNodeKind::Svg => {
            for child in &node.children {
                render_node(child, canvas, ctx, &state, depth);
            }
        }
        // (A rect reaching here has a non-positive size: nothing to draw.)
        SvgNodeKind::Rect(_)
        | SvgNodeKind::Defs
        | SvgNodeKind::ClipPath(_)
        | SvgNodeKind::LinearGradient(_)
        | SvgNodeKind::RadialGradient(_) => {
            // Definition-only elements — not part of the rendered tree.
        }
        SvgNodeKind::Unknown(_) => {
            // Render children so that unknown wrappers (e.g. <switch>,
            // <a>) don't hide their contents. <style> elements are handled
            // separately during parse and don't reach render.
            for child in &node.children {
                render_node(child, canvas, ctx, &state, depth);
            }
        }
    }

    if layer_open {
        canvas.restore();
    }
    canvas.restore();
}

/// Build a dash [`PathEffectRef`] from the resolved presentation state, or
/// `None` when dashing is disabled.
fn build_dash_effect(state: &PresentationState) -> Option<PathEffectRef> {
    let intervals = state.dash_array.as_ref()?;
    // SVG repeats an odd stroke-dasharray to make it even (as SkSVG does
    // before SkDashPathEffect::Make, which rejects odd counts).
    let mut intervals = intervals.clone();
    if intervals.len() % 2 == 1 {
        intervals.extend_from_within(..);
    }
    let dash = DashEffect::new(intervals, state.dash_offset)?;
    Some(Arc::new(dash) as PathEffectRef)
}

/// Extract the id from an `url(#id)` reference; returns None if `s` is
/// anything else (including a bare id without the `url()` wrapper — those
/// are handled by the caller when appropriate).
#[allow(
    clippy::option_if_let_else,
    reason = "three-way url()/#id/none dispatch reads more clearly as if/else-if than nested map_or_else calls"
)]
fn extract_url_id(s: &str) -> Option<&str> {
    let s = s.trim();
    if let Some(stripped) = s.strip_prefix("url(") {
        let inner = stripped.trim_end_matches(')');
        let inner = inner.trim_matches(|c| c == '"' || c == '\'');
        Some(inner.trim_start_matches('#'))
    } else if let Some(stripped) = s.strip_prefix('#') {
        Some(stripped)
    } else {
        None
    }
}

/// Build a combined Path representing the geometry inside a `<clipPath>`
/// element. Walks direct children and unions their geometry, honouring each
/// child's `transform` (and any nested group transforms) so the clip region
/// matches how the shapes would render (`SkSVGClipPath`).
fn build_clip_path(clip_node: &SvgNode) -> Path {
    let mut builder = PathBuilder::new();
    for child in &clip_node.children {
        add_node_geometry(child, &mut builder, &Matrix::IDENTITY);
    }
    builder.build()
}

/// Accumulate `node`'s geometry into `builder`, transformed by `parent`
/// (the composed ancestor transform) pre-concatenated with the node's own
/// transform.
fn add_node_geometry(node: &SvgNode, builder: &mut PathBuilder, parent: &Matrix) {
    let m = parent.concat(&node.transform);
    let shape: Option<Path> = match &node.kind {
        // A rect with a non-positive size adds nothing; otherwise the same
        // rounded corners as when it is drawn.
        SvgNodeKind::Rect(rect) => (rect.width > 0.0 && rect.height > 0.0).then(|| {
            let r = Rect::from_xywh(rect.x, rect.y, rect.width, rect.height);
            let (rx, ry) = rect_radii(rect);
            let mut b = PathBuilder::new();
            if rx > 0.0 && ry > 0.0 {
                b.add_round_rect(&r, rx, ry);
            } else {
                b.add_rect(&r);
            }
            b.build()
        }),
        SvgNodeKind::Circle(c) => {
            let mut b = PathBuilder::new();
            b.add_oval(&Rect::from_xywh(
                c.cx - c.r,
                c.cy - c.r,
                c.r * 2.0,
                c.r * 2.0,
            ));
            Some(b.build())
        }
        SvgNodeKind::Ellipse(e) => {
            let mut b = PathBuilder::new();
            b.add_oval(&Rect::from_xywh(
                e.cx - e.rx,
                e.cy - e.ry,
                e.rx * 2.0,
                e.ry * 2.0,
            ));
            Some(b.build())
        }
        SvgNodeKind::Path(p) => Some((**p).clone()),
        SvgNodeKind::Polygon(points) if points.len() >= 3 => {
            let mut b = PathBuilder::new();
            b.move_to(points[0].x, points[0].y);
            for p in &points[1..] {
                b.line_to(p.x, p.y);
            }
            b.close();
            Some(b.build())
        }
        SvgNodeKind::Polyline(points) if points.len() >= 2 => {
            let mut b = PathBuilder::new();
            b.move_to(points[0].x, points[0].y);
            for p in &points[1..] {
                b.line_to(p.x, p.y);
            }
            Some(b.build())
        }
        _ => None,
    };

    if let Some(path) = shape {
        builder.add_path(&path.transformed(&m));
    } else {
        // Recurse so groups/use wrappers inside clipPath contribute their
        // geometry under the accumulated transform.
        for child in &node.children {
            add_node_geometry(child, builder, &m);
        }
    }
}

/// Render a `<text>` element by converting each glyph to a path and
/// filling/stroking it. Uses skia-rs-text for glyph-to-path conversion.
fn render_text(
    text: &SvgText,
    canvas: &mut Canvas<'_>,
    fill_paint: Option<&Paint>,
    stroke_paint: Option<&Paint>,
) {
    if text.content.is_empty() {
        return;
    }

    let font = Font::from_size(text.font_size.max(1.0));
    let width = font.measure_text(&text.content);

    let anchor_offset = match text.text_anchor {
        TextAnchor::Start => 0.0,
        TextAnchor::Middle => -width / 2.0,
        TextAnchor::End => -width,
    };

    let mut x_offset = text.x + anchor_offset;
    let y_baseline = text.y;

    for ch in text.content.chars() {
        let glyph = font.char_to_glyph(ch);
        let advance = font.glyph_advance(glyph);

        if let Some(glyph_path) = font.glyph_path(glyph) {
            let translated = glyph_path.transformed(&Matrix::translate(x_offset, y_baseline));
            if let Some(paint) = fill_paint {
                canvas.draw_path(&translated, paint);
            }
            if let Some(paint) = stroke_paint {
                canvas.draw_path(&translated, paint);
            }
        }

        x_offset += advance;
    }
}

/// Render an `<image>` element. Supports `data:` URIs; falls back silently
/// for other schemes since network fetching is out of scope for this
/// crate.
fn render_image(img: &SvgImage, canvas: &mut Canvas<'_>, opacity: Scalar) {
    let Some(data) = decode_image_href(&img.href) else {
        return;
    };

    let Ok(image) = decode_image(&data) else {
        return;
    };

    #[allow(
        clippy::cast_precision_loss,
        reason = "decoded image dimensions are pixel counts far below 2^24; exact in f32"
    )]
    let target_w = if img.width > 0.0 {
        img.width
    } else {
        image.width() as Scalar
    };
    #[allow(
        clippy::cast_precision_loss,
        reason = "decoded image dimensions are pixel counts far below 2^24; exact in f32"
    )]
    let target_h = if img.height > 0.0 {
        img.height
    } else {
        image.height() as Scalar
    };
    let dst = Rect::from_xywh(img.x, img.y, target_w, target_h);

    let mut paint = Paint::new();
    paint.set_alpha(opacity);
    canvas.draw_image_rect(&image, None, &dst, Some(&paint));
}

/// Decode a `data:` URI into raw bytes. Returns None for non-data-URI
/// hrefs or malformed input.
fn decode_image_href(href: &str) -> Option<Vec<u8>> {
    let rest = href.strip_prefix("data:")?;
    // mime/params,payload
    let (meta, payload) = rest.split_once(',')?;
    let is_base64 = meta.contains(";base64");
    if is_base64 {
        decode_base64(payload.trim())
    } else {
        // URL-encoded text — fall back to raw bytes since SVG data URIs
        // typically use base64 for binary formats. Caller gets a decode
        // failure if the format doesn't match.
        Some(payload.as_bytes().to_vec())
    }
}

/// Decode a base64-encoded string (standard alphabet, ignores whitespace
/// and padding).
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    // Filter ASCII whitespace directly on bytes — no UTF-8 round-trip needed
    // since base64 data is always ASCII.
    let cleaned: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    STANDARD.decode(&cleaned).ok()
}

/// Build a [`Paint`] for a resolved fill/stroke paint selector.
///
/// `opacity_prop` is the fill-opacity or stroke-opacity; `deferred` is the
/// element `opacity` folded in for leaf shapes. The final paint alpha is
/// `base_alpha * opacity_prop * deferred`, clamped — matching
/// `SkSVGRenderContext::commonPaint`'s three opacity components.
fn build_paint(
    svg_paint: &SvgPaint,
    style: Style,
    opacity_prop: Scalar,
    deferred: Scalar,
    state: &PresentationState,
    ctx: &RenderContext<'_>,
    node_bounds: Rect,
) -> Option<Paint> {
    let mut paint = Paint::new();
    paint.set_style(style);
    paint.set_anti_alias(true);

    let mut base_alpha: Scalar = 1.0;
    match svg_paint {
        SvgPaint::None => return None,
        SvgPaint::Color(color) => {
            paint.set_color32(*color);
            base_alpha = Scalar::from(color.alpha()) / 255.0;
        }
        SvgPaint::CurrentColor => {
            paint.set_color32(state.color);
            base_alpha = Scalar::from(state.color.alpha()) / 255.0;
        }
        SvgPaint::Url(url, fallback) => {
            let id = url.trim_start_matches('#');
            let shader = ctx
                .defs
                .get(id)
                .copied()
                .and_then(|r| build_gradient_shader(r, node_bounds));
            if let Some(shader) = shader {
                paint.set_shader(Some(shader));
            } else {
                // Unresolvable reference: use the grammar's fallback color if
                // present, else Skia's fallback (black).
                let fb = fallback.unwrap_or(Color::BLACK);
                paint.set_color32(fb);
                base_alpha = Scalar::from(fb.alpha()) / 255.0;
            }
        }
    }

    paint.set_alpha((base_alpha * opacity_prop * deferred).clamp(0.0, 1.0));
    Some(paint)
}

/// Convert an SVG gradient node into a skia-rs-paint shader.
///
/// For `objectBoundingBox` units the gradient is defined in the unit square
/// and a local matrix maps it through the object's bounding box (composed as
/// `bbox × gradientTransform`), which yields the correct elliptical mapping
/// for non-square bounds. For `userSpaceOnUse` the coordinates are already in
/// user space and only `gradientTransform` applies.
fn build_gradient_shader(node: &SvgNode, object_bounds: Rect) -> Option<ShaderRef> {
    match &node.kind {
        SvgNodeKind::LinearGradient(grad) => {
            let (colors, positions) = stops_to_colors(&grad.stops);
            if colors.is_empty() {
                return None;
            }
            let (start, end, local) = match grad.units {
                GradientUnits::ObjectBoundingBox => (
                    Point::new(grad.x1, grad.y1),
                    Point::new(grad.x2, grad.y2),
                    obb_matrix(object_bounds).concat(&grad.transform),
                ),
                GradientUnits::UserSpaceOnUse => (
                    Point::new(grad.x1, grad.y1),
                    Point::new(grad.x2, grad.y2),
                    grad.transform,
                ),
            };
            let tile_mode = spread_to_tile_mode(grad.spread);
            let mut gradient = PaintLinearGradient::new(start, end, colors, positions, tile_mode);
            if local != Matrix::IDENTITY {
                gradient = gradient.with_local_matrix(local);
            }
            Some(Arc::new(gradient) as ShaderRef)
        }
        SvgNodeKind::RadialGradient(grad) => {
            let (colors, positions) = stops_to_colors(&grad.stops);
            if colors.is_empty() {
                return None;
            }
            let (center, radius, local) = match grad.units {
                GradientUnits::ObjectBoundingBox => (
                    // Unit-square coords; the OBB matrix turns the unit circle
                    // into the (possibly elliptical) object-space region. A
                    // percentage radius resolves against the unit diagonal,
                    // which is 1, so the stored fraction is used directly.
                    Point::new(grad.cx, grad.cy),
                    grad.r,
                    obb_matrix(object_bounds).concat(&grad.transform),
                ),
                GradientUnits::UserSpaceOnUse => {
                    (Point::new(grad.cx, grad.cy), grad.r, grad.transform)
                }
            };
            let tile_mode = spread_to_tile_mode(grad.spread);
            let mut gradient =
                PaintRadialGradient::new(center, radius, colors, positions, tile_mode);
            if local != Matrix::IDENTITY {
                gradient = gradient.with_local_matrix(local);
            }
            Some(Arc::new(gradient) as ShaderRef)
        }
        _ => None,
    }
}

/// The unit-square → object-bounding-box matrix
/// (`translate(x, y) · scale(w, h)`), i.e. `transformForCurrentOBB`.
fn obb_matrix(bounds: Rect) -> Matrix {
    Matrix::translate(bounds.left, bounds.top)
        .concat(&Matrix::scale(bounds.width(), bounds.height()))
}

fn stops_to_colors(stops: &[GradientStop]) -> (Vec<Color4f>, Option<Vec<Scalar>>) {
    let colors: Vec<Color4f> = stops
        .iter()
        .map(|s| {
            let mut c = Color4f::from_color(s.color);
            c.a *= s.opacity;
            c
        })
        .collect();
    let positions: Vec<Scalar> = stops.iter().map(|s| s.offset).collect();
    let positions_opt = if positions.is_empty() {
        None
    } else {
        Some(positions)
    };
    (colors, positions_opt)
}

const fn spread_to_tile_mode(spread: SpreadMethod) -> TileMode {
    match spread {
        SpreadMethod::Pad => TileMode::Clamp,
        SpreadMethod::Reflect => TileMode::Mirror,
        SpreadMethod::Repeat => TileMode::Repeat,
    }
}

/// Render an SVG DOM into a container of `container_w` × `container_h`.
///
/// Renders under the canvas's current transform, mapping the document's
/// viewBox into that container via `preserveAspectRatio` — without
/// stretching to the canvas pixel dimensions.
///
/// This is the entry point used for SVG-in-OpenType glyphs: the caller sets
/// the CTM to `translate(origin) · scale(ppem/upem)` and passes the em box
/// (`upem × upem`) as the container so the glyph renders in font units at the
/// requested pixel size (`SkSVGOpenTypeSVGDecoder::render` +
/// `setContainerSize`).
pub fn render_svg_in_container(
    dom: &SvgDom,
    canvas: &mut Canvas<'_>,
    container_w: Scalar,
    container_h: Scalar,
) {
    let mut working = dom.clone();
    let sheet = working.stylesheet.clone();
    if !sheet.rules.is_empty() {
        apply_stylesheet(&mut working, &sheet);
    } else if dom_has_inline_styles(&working.root) {
        apply_stylesheet(&mut working, &crate::css::Stylesheet::new());
    }

    // The document viewport is the container; a viewBox (when present) is
    // mapped into it, otherwise the mapping is identity.
    let viewport = Rect::from_xywh(0.0, 0.0, container_w, container_h);
    let view_box = working.view_box.unwrap_or(viewport);

    render_dom_into_viewport(&working, canvas, viewport, view_box);
}

/// Render an SVG string to a new surface.
#[must_use]
pub fn render_svg_string(svg: &str, width: i32, height: i32) -> Option<Surface> {
    let dom = crate::parse_svg(svg).ok()?;
    let mut surface = Surface::new_raster_n32_premul(width, height)?;

    {
        let mut canvas = surface.canvas();
        canvas.clear(Color::WHITE);
    }

    render_svg_to_surface(&dom, &mut surface);
    Some(surface)
}

/// A rect's corner radii as drawn: each clamped to half the matching side,
/// as SVG specifies.
fn rect_radii(rect: &crate::dom::SvgRect) -> (Scalar, Scalar) {
    (
        rect.rx.min(rect.width / 2.0),
        rect.ry.min(rect.height / 2.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample a pixel from the surface backing buffer.
    ///
    /// The raster surface stores pixels as RGBA8 row-major with 4 bytes
    /// per pixel.
    fn pixel_at(surface: &Surface, x: i32, y: i32) -> Option<Color> {
        let pixels = surface.pixels();
        let width = surface.width();
        let row_bytes = usize::try_from(width).ok()? * 4;
        let off = usize::try_from(y).ok()? * row_bytes + usize::try_from(x).ok()? * 4;
        if off + 4 > pixels.len() {
            return None;
        }
        Some(Color::from_argb(
            pixels[off + 3],
            pixels[off],
            pixels[off + 1],
            pixels[off + 2],
        ))
    }

    #[test]
    fn rect_with_only_rx_rounds_both_radii() {
        // ry missing is "auto": it takes rx, so the corner is rounded and
        // the corner pixel keeps the white background.
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" rx="20" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = |x: i32, y: i32| pixel_at(&surface, x, y).map_or(0, |c| c.green());
        assert!(green(11, 11) > 200, "rounded corner left unfilled");
        assert!(green(50, 50) < 50, "interior filled");
    }

    #[test]
    fn rect_radius_auto_keyword_counts_as_missing() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" rx="auto" ry="20" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = pixel_at(&surface, 11, 11).map_or(0, |c| c.green());
        assert!(green > 200, "rx=auto takes ry: the corner is rounded");
    }

    #[test]
    fn rect_radius_with_a_spaced_unit_counts_as_missing() {
        // "5 px" is not a length (it would read as 0): rx takes ry.
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" rx="5 px" ry="20" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = pixel_at(&surface, 11, 11).map_or(0, |c| c.green());
        assert!(green > 200, "the corner is rounded");
    }

    #[test]
    fn rect_radius_with_a_spaced_percent_is_a_length() {
        // "5 %" reads as 5% (as width="50 %" does), so rx is given: 5% of
        // 100 = 5 against ry 20, a corner much less round than auto (20).
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" rx="5 %" ry="20" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = pixel_at(&surface, 13, 13).map_or(0, |c| c.green());
        assert!(green < 50, "rx = 5: (13, 13) is inside the shallow corner");
    }

    #[test]
    fn auto_radius_takes_the_other_resolved_radius() {
        // SkSVGRect's ResolveOptionalRadii: rx="10%" in a 200x100 viewport
        // resolves on the width (20), and the auto ry takes that value.
        let svg = r#"<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="150" height="80" rx="10%"/>
        </svg>"#;
        let doc = crate::parser::parse_svg(svg).unwrap();
        let rect = doc
            .root
            .children
            .iter()
            .find_map(|n| match &n.kind {
                crate::dom::SvgNodeKind::Rect(r) => Some(*r),
                _ => None,
            })
            .expect("a rect");
        assert_eq!((rect.rx, rect.ry), (20.0, 20.0));
    }

    #[test]
    fn a_negative_rect_radius_gives_square_corners() {
        // A given negative ry is kept, not auto: SkRRect::MakeRectXY then
        // draws square corners, so the corner pixel is filled.
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" rx="10" ry="-5" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = pixel_at(&surface, 11, 11).map_or(255, |c| c.green());
        assert!(green < 50, "square corner: (11, 11) is filled");
    }

    #[test]
    fn rects_with_negative_size_are_not_drawn() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="50" y="10" width="-40" height="40" rx="5" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = pixel_at(&surface, 30, 30).map_or(0, |c| c.green());
        assert!(green > 200, "nothing drawn");
    }

    #[test]
    fn rounded_clip_rects_clip_with_their_corners() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <clipPath id="c"><rect width="100" height="100" rx="50"/></clipPath>
            <rect width="100" height="100" fill="red" clip-path="url(#c)"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = |x: i32, y: i32| pixel_at(&surface, x, y).map_or(0, |c| c.green());
        assert!(green(3, 3) > 200, "the corner is clipped away");
        assert!(green(50, 50) < 50, "the middle is drawn");
    }

    #[test]
    fn two_point_polygons_are_stroked() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <polygon points="10,10 90,90" fill="none" stroke="red" stroke-width="6"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        let green = pixel_at(&surface, 50, 50).map_or(255, |c| c.green());
        assert!(green < 50, "the diagonal is stroked");
    }

    #[test]
    fn odd_stroke_dasharray_repeats_like_svg() {
        // stroke-dasharray="10" is [10, 10] in SVG: dash 0..10, gap 10..20.
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <line x1="0" y1="50" x2="100" y2="50" stroke="red" stroke-width="4"
                  stroke-dasharray="10"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).expect("renders");
        // The background is white: a red dash has no green, a gap keeps it.
        let green = |x: i32| pixel_at(&surface, x, 50).map_or(0, |c| c.green());
        assert!(green(5) < 50, "dash at x = 5");
        assert!(green(15) > 200, "gap at x = 15");
        assert!(green(25) < 50, "dash at x = 25");
    }

    #[test]
    fn test_render_simple_svg() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" fill="red"/>
        </svg>"#;

        let surface = render_svg_string(svg, 100, 100);
        assert!(surface.is_some());

        let surface = surface.unwrap();
        assert_eq!(surface.width(), 100);
        assert_eq!(surface.height(), 100);

        // Interior of the rect (50,50) should be red.
        let c = pixel_at(&surface, 50, 50).unwrap();
        assert!(c.red() > 200, "expected mostly red, got {c:?}");
        assert!(c.green() < 60);
        assert!(c.blue() < 60);
    }

    #[test]
    fn test_render_circle() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="40" fill="blue"/>
        </svg>"#;

        let surface = render_svg_string(svg, 100, 100).unwrap();
        // Center of the circle should be blue.
        let c = pixel_at(&surface, 50, 50).unwrap();
        assert!(c.blue() > 200, "center should be blue, got {c:?}");
        assert!(c.red() < 60);
    }

    #[test]
    fn test_render_path() {
        let svg = r#"<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 10 L90 10 L90 90 L10 90 Z" fill="green"/>
        </svg>"#;

        let surface = render_svg_string(svg, 100, 100).unwrap();
        let c = pixel_at(&surface, 50, 50).unwrap();
        assert!(c.green() > 100, "center should be green-ish, got {c:?}");
    }

    #[test]
    fn test_render_css_styled() {
        // The stylesheet says fill:red; rect tag has no inline fill.
        // Expect the rect to render red.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <style>rect { fill: red; }</style>
            <rect x="0" y="0" width="20" height="20"/>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(c.red() > 200, "css fill should apply, got {c:?}");
        assert!(c.green() < 60);
    }

    #[test]
    fn test_render_inline_style() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <rect x="0" y="0" width="20" height="20" style="fill: blue"/>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(c.blue() > 200, "inline style should apply, got {c:?}");
    }

    #[test]
    fn test_render_linear_gradient() {
        // Horizontal gradient from red at x=0 to blue at x=100.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="20">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="red"/>
              <stop offset="1" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="100" height="20" fill="url(#g)"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 20).unwrap();

        let left = pixel_at(&surface, 5, 10).unwrap();
        let right = pixel_at(&surface, 95, 10).unwrap();
        assert!(left.red() > 200, "left edge should be red, got {left:?}");
        assert!(
            right.blue() > 200,
            "right edge should be blue, got {right:?}"
        );
    }

    #[test]
    fn test_render_clip_path() {
        // Without clipPath the rect would cover the entire surface.
        // With clipPath=url(#tinyClip) only a tiny 5x5 area in the
        // top-left remains filled — the rest of the canvas stays the
        // white background from render_svg_string's clear().
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">
          <defs>
            <clipPath id="tinyClip">
              <rect x="0" y="0" width="5" height="5"/>
            </clipPath>
          </defs>
          <rect x="0" y="0" width="50" height="50" fill="red" clip-path="url(#tinyClip)"/>
        </svg>"#;
        let surface = render_svg_string(svg, 50, 50).unwrap();
        let inside = pixel_at(&surface, 2, 2).unwrap();
        let outside = pixel_at(&surface, 30, 30).unwrap();
        assert!(
            inside.red() > 200 && inside.green() < 60,
            "inside clip should be red, got {inside:?}"
        );
        assert!(
            outside.red() > 240 && outside.green() > 240 && outside.blue() > 240,
            "outside clip should remain the white background, got {outside:?}"
        );
    }

    #[test]
    fn test_url_id_extraction() {
        assert_eq!(extract_url_id("url(#foo)"), Some("foo"));
        assert_eq!(extract_url_id("url(\"#bar\")"), Some("bar"));
        assert_eq!(extract_url_id("#baz"), Some("baz"));
        assert_eq!(extract_url_id("plain"), None);
    }

    #[test]
    fn test_base64_decode() {
        assert_eq!(decode_base64("SGVsbG8=").unwrap(), b"Hello");
        assert_eq!(decode_base64("SGVsbG8gV29ybGQ=").unwrap(), b"Hello World");
        // Whitespace tolerated.
        assert_eq!(decode_base64("SGVs\nbG8=").unwrap(), b"Hello");
    }

    #[test]
    fn test_fill_inherits_from_group() {
        // The rect specifies no fill; it must inherit red from the group,
        // not fall back to a per-node black default.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <g fill="red"><rect x="0" y="0" width="20" height="20"/></g>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(c.red() > 200 && c.green() < 60, "inherited red, got {c:?}");
    }

    #[test]
    fn test_default_fill_is_black_at_root() {
        // No fill anywhere -> initial value black (from the render root).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <rect x="0" y="0" width="20" height="20"/>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(
            c.red() < 30 && c.green() < 30 && c.blue() < 30,
            "black, got {c:?}"
        );
    }

    #[test]
    fn test_current_color_resolves_inherited_color() {
        // fill="currentColor" resolves against the inherited `color`.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <g color="rgb(0,0,255)">
                <rect x="0" y="0" width="20" height="20" fill="currentColor"/>
            </g>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(
            c.blue() > 200 && c.red() < 60,
            "currentColor blue, got {c:?}"
        );
    }

    #[test]
    fn test_group_opacity_composites_group() {
        // A half-opaque red group over white -> ~ (255, 128, 128).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <g opacity="0.5"><rect x="0" y="0" width="20" height="20" fill="red"/></g>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(c.red() > 240, "red stays high, got {c:?}");
        assert!(
            (i32::from(c.green()) - 128).abs() < 20 && (i32::from(c.blue()) - 128).abs() < 20,
            "half-blended toward white, got {c:?}"
        );
    }

    #[test]
    fn test_group_opacity_no_double_composite_on_overlap() {
        // Two overlapping opaque rects inside a half-opaque group. If the
        // group composited via a layer (correct), the overlap is blended once
        // (~128). If opacity were wrongly applied per-leaf, the overlap would
        // be darker. Red on red -> overlap red channel ~255 either way, so
        // check the *green* channel stays ~128 (single 50% composite).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <g opacity="0.5">
                <rect x="0" y="0" width="15" height="20" fill="red"/>
                <rect x="5" y="0" width="15" height="20" fill="red"/>
            </g>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        // Sample the overlap region (x in [5,15)).
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(
            (i32::from(c.green()) - 128).abs() < 25,
            "overlap composited once, got green {}",
            c.green()
        );
    }

    #[test]
    fn test_fill_opacity_multiplies_paint_alpha() {
        // fill-opacity 0.5 on an opaque red rect over white -> ~(255,128,128).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <rect x="0" y="0" width="20" height="20" fill="red" fill-opacity="0.5"/>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(c.red() > 240, "red high, got {c:?}");
        assert!(
            (i32::from(c.green()) - 128).abs() < 20,
            "fill-opacity blended halfway, got green {}",
            c.green()
        );
    }

    #[test]
    fn test_leaf_opacity_applied_as_paint_alpha() {
        // A single fill-only leaf shape with opacity 0.5: Skia folds opacity
        // into the paint alpha (no layer). Visual result matches a 50%
        // composite over white.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <rect x="0" y="0" width="20" height="20" fill="red" opacity="0.5"/>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(
            c.red() > 240 && (i32::from(c.green()) - 128).abs() < 20,
            "leaf opacity, got {c:?}"
        );
    }

    #[test]
    fn test_stroke_linecap_inherited_and_applied() {
        // stroke-linecap="square" (inherited from the group) extends the
        // stroke past the endpoint by half the stroke width; butt would not.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="30" height="20">
            <g stroke-linecap="square">
                <path d="M10 10 L20 10" stroke="black" stroke-width="8" fill="none"/>
            </g>
        </svg>"#;
        let surface = render_svg_string(svg, 30, 20).unwrap();
        // Just past the x=20 endpoint (square cap extends ~4px to x≈24).
        let past = pixel_at(&surface, 22, 10).unwrap();
        assert!(
            past.red() < 80,
            "square cap extends past endpoint, got {past:?}"
        );
        // Butt cap (default) must NOT extend there.
        let svg_butt = r#"<svg xmlns="http://www.w3.org/2000/svg" width="30" height="20">
            <path d="M10 10 L20 10" stroke="black" stroke-width="8" fill="none"/>
        </svg>"#;
        let surface2 = render_svg_string(svg_butt, 30, 20).unwrap();
        let past2 = pixel_at(&surface2, 22, 10).unwrap();
        assert!(past2.red() > 200, "butt cap does not extend, got {past2:?}");
    }

    #[test]
    fn test_polyline_is_filled_by_default() {
        // A polyline forming a triangle fills (default black) even without an
        // explicit fill; previously only stroke rendered.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
            <polyline points="20,2 38,38 2,38"/>
        </svg>"#;
        let surface = render_svg_string(svg, 40, 40).unwrap();
        let c = pixel_at(&surface, 20, 28).unwrap();
        assert!(
            c.red() < 40 && c.green() < 40 && c.blue() < 40,
            "polyline interior filled black, got {c:?}"
        );
    }

    #[test]
    fn test_polygon_fill_and_stroke() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
            <polygon points="20,2 38,38 2,38" fill="green"/>
        </svg>"#;
        let surface = render_svg_string(svg, 40, 40).unwrap();
        let c = pixel_at(&surface, 20, 28).unwrap();
        assert!(
            c.green() > 100 && c.red() < 60,
            "polygon filled green, got {c:?}"
        );
    }

    #[test]
    fn test_preserve_aspect_ratio_meet_centers() {
        // viewBox 100x50 into a 100x100 canvas, xMidYMid meet -> uniform
        // scale 1, vertically centered by (100-50)/2 = 25px. A band at the
        // top of the viewBox lands around y=25..35, leaving y<20 white.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 50">
            <rect x="0" y="0" width="100" height="10" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).unwrap();
        let top = pixel_at(&surface, 50, 5).unwrap();
        assert!(
            top.red() > 240 && top.green() > 240,
            "top stays white, got {top:?}"
        );
        let band = pixel_at(&surface, 50, 30).unwrap();
        assert!(
            band.red() > 200 && band.green() < 60,
            "band centered, got {band:?}"
        );
    }

    #[test]
    fn test_preserve_aspect_ratio_none_stretches() {
        // "none" stretches non-uniformly to fill the whole canvas.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 50" preserveAspectRatio="none">
            <rect x="0" y="0" width="100" height="50" fill="red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 100).unwrap();
        // Bottom row is covered because the rect stretches to full height.
        let bottom = pixel_at(&surface, 50, 95).unwrap();
        assert!(
            bottom.red() > 200 && bottom.green() < 60,
            "stretched to fill, got {bottom:?}"
        );
    }

    #[test]
    fn test_fill_rule_evenodd_creates_hole() {
        // Two nested rectangles in one path with even-odd -> the inner region
        // is a hole (background shows through).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
            <path d="M2 2 H38 V38 H2 Z M12 12 H28 V28 H12 Z" fill="black" fill-rule="evenodd"/>
        </svg>"#;
        let surface = render_svg_string(svg, 40, 40).unwrap();
        let hole = pixel_at(&surface, 20, 20).unwrap();
        assert!(
            hole.red() > 240 && hole.green() > 240,
            "even-odd hole white, got {hole:?}"
        );
        let ring = pixel_at(&surface, 5, 20).unwrap();
        assert!(ring.red() < 40, "ring filled black, got {ring:?}");
    }

    #[test]
    fn test_fill_rule_inherits() {
        // fill-rule on a group is inherited by the path.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
            <g fill-rule="evenodd">
                <path d="M2 2 H38 V38 H2 Z M12 12 H28 V28 H12 Z" fill="black"/>
            </g>
        </svg>"#;
        let surface = render_svg_string(svg, 40, 40).unwrap();
        let hole = pixel_at(&surface, 20, 20).unwrap();
        assert!(hole.red() > 240, "inherited even-odd hole, got {hole:?}");
    }

    #[test]
    fn test_stroke_dasharray_applied() {
        // A dashed horizontal stroke leaves gaps: along the line some pixels
        // are the stroke color and some remain the white background.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="10">
            <line x1="0" y1="5" x2="40" y2="5" stroke="black" stroke-width="4" stroke-dasharray="4 4"/>
        </svg>"#;
        let surface = render_svg_string(svg, 40, 10).unwrap();
        let mut saw_dark = false;
        let mut saw_light = false;
        for x in 0..40 {
            let c = pixel_at(&surface, x, 5).unwrap();
            if c.red() < 80 {
                saw_dark = true;
            }
            if c.red() > 220 {
                saw_light = true;
            }
        }
        assert!(
            saw_dark && saw_light,
            "dashes produce gaps: dark={saw_dark} light={saw_light}"
        );
    }

    #[test]
    fn test_paint_url_fallback_color_renders() {
        // fill references a missing gradient with a red fallback -> red.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
            <rect x="0" y="0" width="20" height="20" fill="url(#missing) red"/>
        </svg>"#;
        let surface = render_svg_string(svg, 20, 20).unwrap();
        let c = pixel_at(&surface, 10, 10).unwrap();
        assert!(c.red() > 200 && c.green() < 60, "fallback red, got {c:?}");
    }

    #[test]
    fn test_clip_path_honors_child_transform() {
        // The clip rect is translated by (25,25); only that region shows.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">
          <defs>
            <clipPath id="c">
              <rect x="0" y="0" width="5" height="5" transform="translate(25,25)"/>
            </clipPath>
          </defs>
          <rect x="0" y="0" width="50" height="50" fill="red" clip-path="url(#c)"/>
        </svg>"#;
        let surface = render_svg_string(svg, 50, 50).unwrap();
        // Inside the translated clip.
        let inside = pixel_at(&surface, 27, 27).unwrap();
        assert!(
            inside.red() > 200 && inside.green() < 60,
            "inside translated clip, got {inside:?}"
        );
        // The original (untranslated) location must now be clipped out.
        let outside = pixel_at(&surface, 2, 2).unwrap();
        assert!(
            outside.red() > 240 && outside.green() > 240,
            "old spot clipped out, got {outside:?}"
        );
    }

    #[test]
    fn test_use_applies_x_y_offset() {
        // <use x="20"> shifts the referenced rect right by 20px.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">
          <defs><rect id="r" x="0" y="0" width="10" height="20" fill="red"/></defs>
          <use href="#r" x="20" y="0"/>
        </svg>"##;
        let surface = render_svg_string(svg, 40, 20).unwrap();
        let shifted = pixel_at(&surface, 25, 10).unwrap();
        assert!(shifted.red() > 200, "use shifted right, got {shifted:?}");
        let origin = pixel_at(&surface, 5, 10).unwrap();
        assert!(
            origin.red() > 240 && origin.green() > 240,
            "origin empty, got {origin:?}"
        );
    }

    #[test]
    fn test_use_cycle_does_not_stack_overflow() {
        // Two <use> elements referencing each other must terminate via the
        // depth guard rather than recursing until the stack overflows.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">
          <g id="a"><use href="#b"/></g>
          <g id="b"><use href="#a"/></g>
        </svg>"##;
        // Rendering must simply return.
        let surface = render_svg_string(svg, 20, 20);
        assert!(surface.is_some());
    }

    #[test]
    fn test_obb_radial_gradient_is_elliptical() {
        // objectBoundingBox radial on a wide (2:1) rect. The gradient maps
        // the unit circle through the bbox, so at the same fractional offset
        // it reaches the horizontal edge later (in pixels) than the vertical
        // edge. Verify the center is the inner stop color and the far
        // horizontal edge is the outer color.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40">
          <defs>
            <radialGradient id="g" cx="50%" cy="50%" r="50%">
              <stop offset="0" stop-color="white"/>
              <stop offset="1" stop-color="black"/>
            </radialGradient>
          </defs>
          <rect x="0" y="0" width="80" height="40" fill="url(#g)"/>
        </svg>"#;
        let surface = render_svg_string(svg, 80, 40).unwrap();
        let center = pixel_at(&surface, 40, 20).unwrap();
        assert!(center.red() > 200, "center near white, got {center:?}");
        let h_edge = pixel_at(&surface, 78, 20).unwrap();
        assert!(
            h_edge.red() < 80,
            "horizontal edge near black (ellipse reaches it), got {h_edge:?}"
        );
    }

    #[test]
    fn test_obb_gradient_transform_composes() {
        // A gradientTransform rotating 90deg turns a horizontal OBB linear
        // gradient into a vertical one: top red, bottom blue.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="0"
                gradientTransform="rotate(90, 0.5, 0.5)">
              <stop offset="0" stop-color="red"/>
              <stop offset="1" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="40" height="40" fill="url(#g)"/>
        </svg>"#;
        let surface = render_svg_string(svg, 40, 40).unwrap();
        let top = pixel_at(&surface, 20, 3).unwrap();
        let bottom = pixel_at(&surface, 20, 37).unwrap();
        assert!(top.red() > 180, "top red after rotate, got {top:?}");
        assert!(
            bottom.blue() > 180,
            "bottom blue after rotate, got {bottom:?}"
        );
    }

    #[test]
    fn test_render_svg_in_container_scales() {
        // A 100x100 document rendered into a 100x100 container under a 0.2
        // scale must land in a 20x20 region, not stretch to the canvas.
        use skia_rs_canvas::raster::PixelBuffer;
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
            <rect x="0" y="0" width="100" height="100" fill="red"/>
        </svg>"#;
        let dom = crate::parse_svg(svg).unwrap();
        let mut buffer = PixelBuffer::new(100, 100);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            canvas.clear(Color::WHITE);
            canvas.scale(0.2, 0.2);
            render_svg_in_container(&dom, &mut canvas, 100.0, 100.0);
        }
        let surface_pixels = &buffer.pixels;
        let at = |x: i32, y: i32| -> Color {
            let off = usize::try_from(y).unwrap() * 100 * 4 + usize::try_from(x).unwrap() * 4;
            Color::from_argb(
                surface_pixels[off + 3],
                surface_pixels[off],
                surface_pixels[off + 1],
                surface_pixels[off + 2],
            )
        };
        // Inside the 20x20 scaled region.
        assert!(
            at(5, 5).red() > 200,
            "scaled rect present at (5,5): {:?}",
            at(5, 5)
        );
        // Outside it (would be covered if it stretched to full canvas).
        assert!(
            at(50, 50).red() > 240 && at(50, 50).green() > 240,
            "no stretch to canvas: {:?}",
            at(50, 50)
        );
    }

    #[test]
    fn test_render_text_does_not_panic() {
        // The default typeface has no glyph data so nothing visible will
        // be drawn, but the render path must complete without panicking
        // when encountering text content (regression test for the prior
        // "Text rendering is a no-op" behaviour).
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40">
            <text x="5" y="25" font-size="14" fill="black">hello</text>
        </svg>"#;
        let surface = render_svg_string(svg, 100, 40);
        assert!(surface.is_some());
    }
}
