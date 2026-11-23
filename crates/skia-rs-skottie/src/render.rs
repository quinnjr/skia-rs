//! Canvas rendering for Lottie animations.
//!
//! This module provides the rendering context and methods for
//! drawing Lottie animations to a canvas.

use crate::animation::{Asset, PrecompAsset};
use crate::layers::{Layer, LayerContent};
use crate::mask;
use crate::shapes::{GradientFillShape, GradientStrokeShape, Shape, StrokeShape, TrimPathShape};
use skia_rs_core::{Color4f, Matrix, Point, Rect, Scalar};
use skia_rs_paint::{LinearGradient, Paint, Style, TileMode, TwoPointConicalGradient};
use skia_rs_path::{Path, PathMeasure};
use std::collections::HashMap;
use std::sync::Arc;

/// Render context for drawing animations.
pub struct RenderContext<'a> {
    /// Canvas to draw on.
    canvas: &'a mut dyn Canvas,
    /// Transform stack.
    transform_stack: Vec<Matrix>,
    /// Opacity stack.
    opacity_stack: Vec<Scalar>,
    /// Current transform.
    current_transform: Matrix,
    /// Current opacity.
    current_opacity: Scalar,
    /// Frame rate (fps), used to resolve precomp `tm` time remapping
    /// (which is expressed in seconds).
    frame_rate: Scalar,
    /// Bounds of the current composition, used to resolve inverted masks.
    bounds: Rect,
    /// Current precomp nesting depth, guarding against self-referencing or
    /// cyclic `refId` chains (see [`MAX_PRECOMP_DEPTH`]).
    precomp_depth: u32,
}

/// Maximum precomp nesting depth. A cyclic or self-referencing `refId`
/// chain is bailed out past this cap rather than recursing until the
/// thread stack overflows (mirrors `skia-rs-svg`'s `MAX_USE_DEPTH`
/// convention for `<use>` cycles).
const MAX_PRECOMP_DEPTH: u32 = 64;

/// Canvas trait for rendering.
pub trait Canvas {
    /// Save the current state.
    fn save(&mut self);
    /// Restore the previous state.
    fn restore(&mut self);
    /// Apply a transform.
    fn concat(&mut self, matrix: &Matrix);
    /// Draw a path with a paint.
    fn draw_path(&mut self, path: &Path, paint: &Paint);
    /// Draw a rect with a paint.
    fn draw_rect(&mut self, rect: &Rect, paint: &Paint);
    /// Set clip to a path.
    fn clip_path(&mut self, path: &Path);
    /// Set clip to a rect.
    fn clip_rect(&mut self, rect: &Rect);
    /// Get the current transform.
    fn get_transform(&self) -> Matrix;
    /// Set the transform.
    fn set_transform(&mut self, matrix: &Matrix);
    /// Push an offscreen layer. All draws until the matching [`Canvas::restore`]
    /// go into this layer; on restore it composites onto its parent using
    /// `paint`'s alpha, blend mode, and color filter (`None` = opaque
    /// `SrcOver`, matching `SkCanvas::saveLayer`).
    ///
    /// Default implementation degrades to a plain [`Canvas::save`] (no
    /// offscreen buffer) so existing `Canvas` implementations keep
    /// compiling; conformant compositing (used by track mattes) requires
    /// overriding this.
    fn save_layer(&mut self, paint: Option<&Paint>) {
        let _ = paint;
        self.save();
    }
}

impl<'a> RenderContext<'a> {
    /// Create a new render context.
    pub fn new(canvas: &'a mut dyn Canvas) -> Self {
        Self {
            canvas,
            transform_stack: Vec::new(),
            opacity_stack: Vec::new(),
            current_transform: Matrix::IDENTITY,
            current_opacity: 1.0,
            frame_rate: 30.0,
            bounds: Rect::from_xywh(0.0, 0.0, 100_000.0, 100_000.0),
            precomp_depth: 0,
        }
    }

    /// Set the frame rate (fps) used for precomp `tm` remapping.
    pub const fn set_frame_rate(&mut self, fps: Scalar) {
        self.frame_rate = fps;
    }

    /// Set the bounds of the current composition (used to resolve inverted
    /// masks to a finite region).
    pub const fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }

    /// Save the current state.
    pub fn save(&mut self) {
        self.transform_stack.push(self.current_transform);
        self.opacity_stack.push(self.current_opacity);
        self.canvas.save();
    }

    /// Restore the previous state.
    pub fn restore(&mut self) {
        if let Some(transform) = self.transform_stack.pop() {
            self.current_transform = transform;
        }
        if let Some(opacity) = self.opacity_stack.pop() {
            self.current_opacity = opacity;
        }
        self.canvas.restore();
    }

    /// Push an offscreen compositing layer directly on the underlying
    /// canvas, bypassing [`RenderContext`]'s own transform/opacity stack
    /// (matrix/opacity state is unaffected by a layer push in Skia's
    /// model). Pair with [`RenderContext::restore_layer`]. Used by track
    /// matte compositing.
    fn save_layer(&mut self, paint: Option<&Paint>) {
        self.canvas.save_layer(paint);
    }

    /// Pop a layer pushed via [`RenderContext::save_layer`], compositing it
    /// onto its parent.
    fn restore_layer(&mut self) {
        self.canvas.restore();
    }

    /// Concatenate a transform.
    pub fn concat(&mut self, matrix: &Matrix) {
        self.current_transform = self.current_transform.concat(matrix);
        self.canvas.concat(matrix);
    }

    /// Multiply opacity.
    pub fn multiply_opacity(&mut self, opacity: Scalar) {
        self.current_opacity *= opacity;
    }

    /// Get current opacity.
    #[must_use]
    pub const fn current_opacity(&self) -> Scalar {
        self.current_opacity
    }

    /// Draw a path.
    pub fn draw_path(&mut self, path: &Path, paint: &Paint) {
        self.canvas.draw_path(path, paint);
    }

    /// Draw a rect.
    pub fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
        self.canvas.draw_rect(rect, paint);
    }

    /// Clip to a path.
    pub fn clip_path(&mut self, path: &Path) {
        self.canvas.clip_path(path);
    }

    /// Clip to a rect.
    pub fn clip_rect(&mut self, rect: &Rect) {
        self.canvas.clip_rect(rect);
    }

    /// Render a layer within a composition (`siblings` is the layer list
    /// of the enclosing composition, needed to resolve `parent` chains).
    ///
    /// Per upstream `Layer.cpp`/`PrecompLayer.cpp`, a layer's own
    /// transform/opacity/masks evaluate at the *unadjusted* composition
    /// frame; only a precomp layer's nested content gets the `st`/`sr`/`tm`
    /// time remap (applied in the `Precomp` branch below).
    pub fn render_layer(
        &mut self,
        layer: &Layer,
        frame: Scalar,
        assets: &HashMap<String, Asset>,
        siblings: &[Layer],
    ) {
        // A legacy `td`-hidden matte source layer is excluded from the main
        // render list (it still renders when pulled in as a matte input by
        // `render_matte_composite`, below).
        if !layer.is_visible_at(frame) || layer.matte_source_hidden {
            return;
        }

        if let Some(mode) = layer
            .matte_mode
            .filter(|m| *m != crate::layers::MatteMode::None)
            && let Some(source) = resolve_matte_source(layer, siblings)
        {
            // Always composite through the matte, even when the source
            // is not visible at this frame: per sksg::MaskEffect, an
            // empty mask layer means zero coverage — the consumer is
            // masked out entirely (or fully shown, for inverted modes)
            // rather than rendered unmasked.
            self.render_matte_composite(layer, source, mode, frame, assets, siblings);
            return;
        }

        self.render_layer_content(layer, frame, assets, siblings);
    }

    /// Render a layer's own transform/opacity/masks/content, with no matte
    /// handling. Used both for normal layers and (directly, bypassing the
    /// `matte_source_hidden` gate) for matte source/consumer content inside
    /// [`RenderContext::render_matte_composite`].
    fn render_layer_content(
        &mut self,
        layer: &Layer,
        frame: Scalar,
        assets: &HashMap<String, Asset>,
        siblings: &[Layer],
    ) {
        let opacity = layer.opacity_at(frame);

        self.save();

        // Apply layer transform, composed with the parent chain (guarded
        // against cycles).
        let matrix = compose_ancestor_matrix(layer, frame, siblings);
        self.concat(&matrix);
        self.multiply_opacity(opacity);

        // Apply masks: Add unions, Subtract subtracts, Intersect
        // intersects (see `mask::build_clip`).
        if layer.has_masks()
            && let Some(clip) = mask::build_clip(&layer.masks, frame, self.bounds)
        {
            self.clip_path(&clip);
        }

        // Render content
        match &layer.content {
            LayerContent::Shape(content) => {
                self.render_shapes(&content.shapes, frame);
            }
            LayerContent::Solid(content) => {
                let rect = Rect::from_xywh(0.0, 0.0, content.width, content.height);
                let mut paint = Paint::new();
                paint.set_color32(content.color);
                paint.set_style(Style::Fill);

                let color = paint.color();
                let adjusted_color =
                    Color4f::new(color.r, color.g, color.b, color.a * self.current_opacity);
                paint.set_color(adjusted_color);

                self.draw_rect(&rect, &paint);
            }
            LayerContent::Precomp(content) => {
                if let Some(Asset::Precomp(precomp)) = assets.get(&content.ref_id) {
                    let content_frame = layer.precomp_content_frame(frame, self.frame_rate);
                    self.render_precomp(precomp, content_frame, assets);
                }
            }
            LayerContent::Image(_content) => {
                // Image rendering would require image loading support
            }
            LayerContent::Text(_content) => {
                // Text rendering would require font support
            }
            LayerContent::None => {}
        }

        self.restore();
    }

    /// Composite `layer` through a track matte sourced from `source`,
    /// matching upstream `sksg::MaskEffect::onRender`
    /// (`modules/sksg/src/SkSGMaskEffect.cpp`):
    ///
    /// 1. Push an outer layer.
    /// 2. Render the matte source's content into it (an inner layer with a
    ///    luma color filter, for luma modes, converts its composited RGB
    ///    into mask coverage stored in alpha).
    /// 3. Push an inner layer with blend mode `SrcIn` (normal) or `SrcOut`
    ///    (inverted) and render the consumer's own content into it — on
    ///    restore, this masks the content by the coverage accumulated in
    ///    the outer layer.
    /// 4. Restore the outer layer onto the real backdrop (`SrcOver`).
    #[allow(clippy::too_many_arguments)]
    fn render_matte_composite(
        &mut self,
        layer: &Layer,
        source: &Layer,
        mode: crate::layers::MatteMode,
        frame: Scalar,
        assets: &HashMap<String, Asset>,
        siblings: &[Layer],
    ) {
        use crate::layers::MatteMode;

        let inverted = matches!(mode, MatteMode::AlphaInverted | MatteMode::LumaInverted);
        let luma = matches!(mode, MatteMode::Luma | MatteMode::LumaInverted);

        self.save_layer(None); // outer: accumulates mask coverage, then masked content.

        if luma {
            let mut mask_paint = Paint::new();
            mask_paint.set_color_filter(Some(Arc::new(luma_color_filter())));
            self.save_layer(Some(&mask_paint));
        } else {
            self.save_layer(None);
        }
        // A source outside its in/out range contributes nothing — zero
        // coverage. The SrcIn/SrcOut composite below then masks the
        // consumer out entirely (or shows it fully, when inverted).
        if source.is_visible_at(frame) {
            self.render_layer_content(source, frame, assets, siblings);
        }
        self.restore_layer(); // composite mask coverage into the outer layer.

        let mut content_paint = Paint::new();
        content_paint.set_blend_mode(if inverted {
            skia_rs_paint::BlendMode::SrcOut
        } else {
            skia_rs_paint::BlendMode::SrcIn
        });
        self.save_layer(Some(&content_paint));
        self.render_layer_content(layer, frame, assets, siblings);
        self.restore_layer(); // composite masked content into the outer layer.

        self.restore_layer(); // composite the outer layer onto the real backdrop.
    }

    /// Render shapes.
    #[allow(
        clippy::too_many_lines,
        reason = "mirrors upstream skottie shape-group rendering (fills/strokes/gradients/trim collected per group before compositing); splitting would obscure the single-pass structure"
    )]
    fn render_shapes(&mut self, shapes: &[Shape], frame: Scalar) {
        // Collect geometry and style
        let mut paths: Vec<Path> = Vec::new();
        let mut fills: Vec<&crate::shapes::FillShape> = Vec::new();
        let mut strokes: Vec<&StrokeShape> = Vec::new();
        let mut gradient_fills: Vec<&GradientFillShape> = Vec::new();
        let mut gradient_strokes: Vec<&GradientStrokeShape> = Vec::new();
        let mut trim: Option<&TrimPathShape> = None;

        for shape in shapes {
            match shape {
                Shape::Group(group) => {
                    self.save();

                    // Apply group transform; group opacity multiplies into
                    // the accumulated opacity.
                    if let Some(ref transform) = group.transform {
                        let matrix = transform.matrix_at(frame);
                        self.concat(&matrix);
                        self.multiply_opacity(transform.opacity_at(frame));
                    }

                    self.render_shapes(&group.shapes, frame);

                    self.restore();
                }
                Shape::Rectangle(rect) => {
                    if let Some(path) = rect.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Ellipse(ellipse) => {
                    if let Some(path) = ellipse.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Path(path_shape) => {
                    if let Some(path) = path_shape.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Polystar(star) => {
                    if let Some(path) = star.to_path(frame) {
                        paths.push(path);
                    }
                }
                Shape::Fill(fill) => {
                    fills.push(fill);
                }
                Shape::Stroke(stroke) => {
                    strokes.push(stroke);
                }
                Shape::GradientFill(gf) => {
                    gradient_fills.push(gf);
                }
                Shape::GradientStroke(gs) => {
                    gradient_strokes.push(gs);
                }
                Shape::TrimPath(tp) => {
                    trim = Some(tp);
                }
                Shape::Transform(st) => {
                    let matrix = st.transform.matrix_at(frame);
                    self.concat(&matrix);
                    self.multiply_opacity(st.transform.opacity_at(frame));
                }
                _ => {}
            }
        }

        // Apply trim (path measure based) if present.
        let final_paths: Vec<Path> = if let Some(trim_shape) = trim {
            apply_trim(paths, trim_shape, frame)
        } else {
            paths
        };

        // Draw fills
        for fill in &fills {
            let mut paint = Paint::new();
            let color = fill.color_at(frame);
            paint.set_color(Color4f::new(
                color.r,
                color.g,
                color.b,
                color.a * self.current_opacity,
            ));
            paint.set_style(Style::Fill);

            for path in &final_paths {
                let mut p = path.clone();
                p.set_fill_type(fill.path_fill_type());
                self.draw_path(&p, &paint);
            }
        }

        // Draw gradient fills
        for gf in &gradient_fills {
            let mut paint = Paint::new();
            paint.set_style(Style::Fill);

            let opacity = gf.opacity.value_at(frame).as_scalar().unwrap_or(100.0) / 100.0;
            paint.set_color(Color4f::new(1.0, 1.0, 1.0, opacity * self.current_opacity));

            if let Some(shader) = build_gradient_shader(
                gf.gradient_type,
                gf.start_point
                    .value_at(frame)
                    .as_vec2()
                    .unwrap_or([0.0, 0.0]),
                gf.end_point.value_at(frame).as_vec2().unwrap_or([0.0, 0.0]),
                &gf.stops_at(frame),
                gf.highlight_length
                    .value_at(frame)
                    .as_scalar()
                    .unwrap_or(0.0),
                gf.highlight_angle
                    .value_at(frame)
                    .as_scalar()
                    .unwrap_or(0.0),
            ) {
                paint.set_shader(Some(shader));
            }

            for path in &final_paths {
                self.draw_path(path, &paint);
            }
        }

        // Draw strokes
        for stroke in &strokes {
            let mut paint = Paint::new();
            let color = stroke.color_at(frame);
            paint.set_color(Color4f::new(
                color.r,
                color.g,
                color.b,
                color.a * self.current_opacity,
            ));
            paint.set_style(Style::Stroke);
            paint.set_stroke_width(stroke.width_at(frame));
            paint.set_stroke_cap(stroke.line_cap);
            paint.set_stroke_join(stroke.line_join);
            paint.set_stroke_miter(stroke.miter_limit);

            for path in &final_paths {
                let dashed = apply_dash(path, stroke, frame);
                self.draw_path(&dashed, &paint);
            }
        }

        // Draw gradient strokes
        for gs in &gradient_strokes {
            let mut paint = Paint::new();
            paint.set_style(Style::Stroke);
            paint.set_stroke_width(gs.width.value_at(frame).as_scalar().unwrap_or(1.0));
            paint.set_stroke_cap(gs.line_cap);
            paint.set_stroke_join(gs.line_join);

            let opacity = gs.opacity.value_at(frame).as_scalar().unwrap_or(100.0) / 100.0;
            paint.set_color(Color4f::new(1.0, 1.0, 1.0, opacity * self.current_opacity));

            if let Some(shader) = build_gradient_shader(
                gs.gradient_type,
                gs.start_point
                    .value_at(frame)
                    .as_vec2()
                    .unwrap_or([0.0, 0.0]),
                gs.end_point.value_at(frame).as_vec2().unwrap_or([0.0, 0.0]),
                &gs.stops_at(frame),
                gs.highlight_length
                    .value_at(frame)
                    .as_scalar()
                    .unwrap_or(0.0),
                gs.highlight_angle
                    .value_at(frame)
                    .as_scalar()
                    .unwrap_or(0.0),
            ) {
                paint.set_shader(Some(shader));
            }

            for path in &final_paths {
                self.draw_path(path, &paint);
            }
        }
    }

    /// Render a precomposition.
    fn render_precomp(
        &mut self,
        precomp: &PrecompAsset,
        frame: Scalar,
        assets: &HashMap<String, Asset>,
    ) {
        // Guard against a self-referencing or cyclic `refId` chain: without
        // this cap, a precomp asset that (directly or transitively)
        // references itself would recurse via `render_layer` ->
        // `render_layer_content` -> `render_precomp` until the thread stack
        // overflows and aborts (uncatchable).
        if self.precomp_depth >= MAX_PRECOMP_DEPTH {
            return;
        }
        self.precomp_depth += 1;
        for layer in precomp.layers.iter().rev() {
            if layer.is_visible_at(frame) {
                self.render_layer(layer, frame, assets, &precomp.layers);
            }
        }
        self.precomp_depth -= 1;
    }
}

/// Compose a layer's own transform with its ancestor chain (`parent`
/// index references within `siblings`), guarding against cycles.
fn compose_ancestor_matrix(layer: &Layer, frame: Scalar, siblings: &[Layer]) -> Matrix {
    let mut result = layer.matrix_at(frame);
    let mut current_parent = layer.parent;
    let mut visited = std::collections::HashSet::new();
    visited.insert(layer.index);

    while let Some(parent_index) = current_parent {
        if !visited.insert(parent_index) {
            break; // cycle guard
        }
        let Some(parent) = siblings.iter().find(|l| l.index == parent_index) else {
            break;
        };
        // Parent transforms are applied "outside" the child's own
        // transform: total = parent_matrix * ... * child_matrix.
        result = parent.matrix_at(frame).concat(&result);
        current_parent = parent.parent;
    }

    result
}

/// Resolve a layer's track matte source, matching upstream
/// `LayerBuilder::buildRenderTree` (`Layer.cpp`): the explicit `tp` index
/// if present, otherwise the layer immediately preceding this one in the
/// layers array (`siblings`) — *not* by `ind` value, by array position,
/// matching upstream's `prev_layer_index`.
fn resolve_matte_source<'s>(layer: &Layer, siblings: &'s [Layer]) -> Option<&'s Layer> {
    if let Some(explicit) = layer.matte_layer {
        return siblings.iter().find(|l| l.index == explicit);
    }
    let pos = siblings.iter().position(|l| l.index == layer.index)?;
    pos.checked_sub(1).map(|prev| &siblings[prev])
}

/// `SkLumaColorFilter`: converts RGB to a single alpha channel via
/// BT.709 luma weights, zeroing RGB. Applied at layer-composite time (see
/// [`RenderContext::render_matte_composite`]) so the accumulated mask
/// layer's RGB becomes mask *coverage* stored in alpha.
const fn luma_color_filter() -> skia_rs_paint::ColorMatrixFilter {
    #[rustfmt::skip]
    let m: [Scalar; 20] = [
        0.0,    0.0,    0.0,    0.0, 0.0,
        0.0,    0.0,    0.0,    0.0, 0.0,
        0.0,    0.0,    0.0,    0.0, 0.0,
        0.2126, 0.7152, 0.0722, 0.0, 0.0,
    ];
    skia_rs_paint::ColorMatrixFilter::new(m)
}

/// Apply a trim-path shape to the collected geometry, matching upstream
/// `ShapeBuilder::AttachTrimGeometryEffect` (`TrimPaths.cpp`):
///
/// - `m:1` ("Trim Multiple Shapes: Simultaneously", upstream `kParallel`,
///   the default) — each geometry is trimmed independently with the same
///   start/stop.
/// - `m:2` ("Trim Multiple Shapes: Individually", upstream `kSerial`) —
///   all geometry is merged into a single path and ONE trim is applied
///   across the combined length (`SkTrimPE` parameterizes by total length
///   across contours, which [`PathMeasure::get_segment`] matches, so the
///   trim span crosses path boundaries).
fn apply_trim(paths: Vec<Path>, trim: &TrimPathShape, frame: Scalar) -> Vec<Path> {
    let (start_t, stop_t, inverted) = trim.resolved_at(frame);

    if trim.mode == 2 && paths.len() > 1 {
        let mut builder = skia_rs_path::PathBuilder::new();
        for p in &paths {
            builder.add_path(p);
        }
        return vec![trim_path(&builder.build(), start_t, stop_t, inverted)];
    }

    paths
        .into_iter()
        .map(|p| trim_path(&p, start_t, stop_t, inverted))
        .collect()
}

/// Trim a path to the `[start, stop]` fraction of its length (both in
/// `0..=1`) using path measure, matching `SkTrimPathEffect` semantics. When
/// `inverted`, the *complement* interval (`[0,start] U [stop,1]`) is kept
/// instead (`SkTrimPathEffect::Mode::kInverted`).
fn trim_path(path: &Path, start: Scalar, stop: Scalar, inverted: bool) -> Path {
    if !inverted && start <= 0.0 && stop >= 1.0 {
        return path.clone();
    }
    if !inverted && start >= stop {
        return Path::default();
    }

    let measure = PathMeasure::new(path);
    let len = measure.length();
    if len <= 0.0 {
        return path.clone();
    }

    if inverted {
        // SkTrimPathEffect's inverted mode (tail first, closed contours kept
        // continuous), as Skottie uses it.
        use skia_rs_path::PathEffect;
        skia_rs_path::TrimEffect::new(start, stop, skia_rs_path::TrimMode::Inverted)
            .and_then(|trim| trim.apply(path))
            .unwrap_or_else(|| path.clone())
    } else {
        measure
            .get_segment(start * len, stop * len)
            .unwrap_or_default()
    }
}

/// Apply a stroke's dash pattern (if any) to a path, splitting it into
/// dash segments via Task 2's `DashEffect`.
fn apply_dash(path: &Path, stroke: &StrokeShape, frame: Scalar) -> Path {
    let Some(mut intervals) = stroke.dash_intervals_at(frame) else {
        return path.clone();
    };
    // Lottie allows an odd dash count; Skottie repeats it to an even one
    // before SkDashPathEffect::Make, which rejects odd counts.
    if intervals.len() % 2 == 1 {
        intervals.extend_from_within(..);
    }
    let phase = stroke.dash_offset_at(frame);
    skia_rs_path::DashEffect::new(intervals, phase).map_or_else(
        || path.clone(),
        |dash| {
            use skia_rs_path::PathEffect;
            dash.apply(path).unwrap_or_else(|| path.clone())
        },
    )
}

/// Build a gradient shader from resolved Lottie gradient parameters.
///
/// For radial gradients, matches upstream `GradientAdapter::onSync`
/// (`Gradient.cpp`): `highlight_length`/`highlight_angle` define a focal
/// point, producing a two-point conical gradient.
#[allow(clippy::too_many_arguments)]
fn build_gradient_shader(
    gradient_type: i32,
    start: [Scalar; 2],
    end: [Scalar; 2],
    stops: &[(Scalar, Color4f)],
    highlight_length: Scalar,
    highlight_angle: Scalar,
) -> Option<Arc<dyn skia_rs_paint::Shader>> {
    if stops.is_empty() {
        return None;
    }

    let colors: Vec<Color4f> = stops.iter().map(|(_, c)| *c).collect();
    let positions: Vec<Scalar> = stops.iter().map(|(t, _)| *t).collect();

    let s = Point::new(start[0], start[1]);
    let e = Point::new(end[0], end[1]);

    if gradient_type == 2 {
        // Radial.
        let angle_rad = highlight_angle.to_radians();
        let (sin, cos) = angle_rad.sin_cos();
        // Rotate `e` around `s` by `highlight_angle` degrees.
        let ex = e.x - s.x;
        let ey = e.y - s.y;
        let rotated_e = Point::new(
            ey.mul_add(-sin, ex.mul_add(cos, s.x)),
            ey.mul_add(cos, ex.mul_add(sin, s.y)),
        );

        let eps = 1e-4;
        let h_len = (highlight_length * 0.01).clamp(-1.0 + eps, 1.0 - eps);
        let focal = Point::new(
            (rotated_e.x - s.x).mul_add(h_len, s.x),
            (rotated_e.y - s.y).mul_add(h_len, s.y),
        );
        let end_radius = (rotated_e.x - s.x).hypot(rotated_e.y - s.y);

        Some(Arc::new(TwoPointConicalGradient::new(
            focal,
            0.0,
            s,
            end_radius,
            colors,
            Some(positions),
            TileMode::Clamp,
        )))
    } else {
        Some(Arc::new(LinearGradient::new(
            s,
            e,
            colors,
            Some(positions),
            TileMode::Clamp,
        )))
    }
}

/// Simple canvas implementation using skia-rs-canvas.
pub struct SkiaCanvas<'c, 'a> {
    inner: &'c mut skia_rs_canvas::Canvas<'a>,
}

impl<'c, 'a> SkiaCanvas<'c, 'a> {
    /// Create a new Skia canvas wrapper.
    pub const fn new(canvas: &'c mut skia_rs_canvas::Canvas<'a>) -> Self {
        Self { inner: canvas }
    }
}

impl Canvas for SkiaCanvas<'_, '_> {
    fn save(&mut self) {
        self.inner.save();
    }

    fn restore(&mut self) {
        self.inner.restore();
    }

    fn concat(&mut self, matrix: &Matrix) {
        self.inner.concat(matrix);
    }

    fn draw_path(&mut self, path: &Path, paint: &Paint) {
        self.inner.draw_path(path, paint);
    }

    fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
        self.inner.draw_rect(rect, paint);
    }

    fn clip_path(&mut self, path: &Path) {
        self.inner
            .clip_path(path, skia_rs_canvas::ClipOp::Intersect, true);
    }

    fn clip_rect(&mut self, rect: &Rect) {
        self.inner
            .clip_rect(rect, skia_rs_canvas::ClipOp::Intersect, true);
    }

    fn get_transform(&self) -> Matrix {
        *self.inner.total_matrix()
    }

    fn set_transform(&mut self, matrix: &Matrix) {
        self.inner.set_matrix(matrix);
    }

    fn save_layer(&mut self, paint: Option<&Paint>) {
        let rec = skia_rs_canvas::SaveLayerRec {
            bounds: None,
            paint,
            flags: skia_rs_canvas::SaveLayerFlags::NONE,
        };
        self.inner.save_layer(&rec);
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert exact keyframe/interpolation output values, not tolerances"
)]
mod tests {
    use super::*;

    struct MockCanvas {
        save_count: usize,
        draw_count: usize,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self {
                save_count: 0,
                draw_count: 0,
            }
        }
    }

    impl Canvas for MockCanvas {
        fn save(&mut self) {
            self.save_count += 1;
        }

        fn restore(&mut self) {
            if self.save_count > 0 {
                self.save_count -= 1;
            }
        }

        fn concat(&mut self, _matrix: &Matrix) {}

        fn draw_path(&mut self, _path: &Path, _paint: &Paint) {
            self.draw_count += 1;
        }

        fn draw_rect(&mut self, _rect: &Rect, _paint: &Paint) {
            self.draw_count += 1;
        }

        fn clip_path(&mut self, _path: &Path) {}

        fn clip_rect(&mut self, _rect: &Rect) {}

        fn get_transform(&self) -> Matrix {
            Matrix::IDENTITY
        }

        fn set_transform(&mut self, _matrix: &Matrix) {}
    }

    #[test]
    fn test_render_context() {
        let mut canvas = MockCanvas::new();
        let mut ctx = RenderContext::new(&mut canvas);

        ctx.save();
        ctx.multiply_opacity(0.5);
        assert_eq!(ctx.current_opacity(), 0.5);
        ctx.restore();
        assert_eq!(ctx.current_opacity(), 1.0);
    }

    #[test]
    fn test_opacity_stack() {
        let mut canvas = MockCanvas::new();
        let mut ctx = RenderContext::new(&mut canvas);

        ctx.multiply_opacity(0.5);
        ctx.save();
        ctx.multiply_opacity(0.5);
        assert_eq!(ctx.current_opacity(), 0.25);
        ctx.restore();
        assert_eq!(ctx.current_opacity(), 0.5);
    }

    fn layer_at(index: i32, parent: Option<i32>, x: Scalar, y: Scalar) -> Layer {
        use crate::model::LayerModel;
        let json = format!(
            r#"{{"ty":4,"nm":"l","ind":{index},"ip":0,"op":100,
                "ks":{{"p":{{"a":0,"k":[{x},{y}]}}}} }}"#
        );
        let mut model: LayerModel = serde_json::from_str(&json).unwrap();
        model.parent = parent;
        Layer::from_lottie(&model)
    }

    #[test]
    fn test_parenting_composes_ancestor_translation() {
        let parent = layer_at(1, None, 100.0, 100.0);
        let child = layer_at(2, Some(1), 10.0, 10.0);
        let siblings = vec![parent, child.clone()];

        let m = compose_ancestor_matrix(&child, 0.0, &siblings);
        let origin = m.map_point(skia_rs_core::Point::new(0.0, 0.0));
        assert_eq!(origin.x, 110.0);
        assert_eq!(origin.y, 110.0);
    }

    #[test]
    fn test_parenting_cycle_guard_terminates() {
        // Two layers that (incorrectly) reference each other as parents.
        let a = layer_at(1, Some(2), 5.0, 0.0);
        let b = layer_at(2, Some(1), 0.0, 5.0);
        let siblings = vec![a.clone(), b];

        // Must terminate rather than infinite-loop/stack-overflow.
        let _ = compose_ancestor_matrix(&a, 0.0, &siblings);
    }

    #[test]
    fn test_trim_path_extracts_partial_segment() {
        let mut builder = skia_rs_path::PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(100.0, 0.0);
        let path = builder.build();

        let trimmed = trim_path(&path, 0.0, 0.5, false);
        let measure = PathMeasure::new(&trimmed);
        assert!((measure.length() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_own_transform_ignores_precomp_remap_end_to_end() {
        use crate::animation::Animation;

        // A precomp layer with st/sr set; its own position must NOT be
        // affected, but its child content time should be remapped.
        let json = r#"{
            "v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,
            "assets":[{"id":"comp_0","layers":[
                {"ty":4,"nm":"inner","ind":1,"ip":0,"op":100,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[20,20]}},
                    {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
                 ]}
            ]}],
            "layers":[
                {"ty":0,"nm":"precomp","ind":1,"ip":0,"op":60,"st":10,"sr":2,
                 "refId":"comp_0","w":100,"h":100,
                 "ks":{"p":{"a":0,"k":[5,5]}}}
            ]
        }"#;

        let anim = Animation::from_json(json).unwrap();
        let mut canvas = MockCanvas::new();
        let mut ctx = RenderContext::new(&mut canvas);
        // Should render without panicking across the remapped range.
        anim.render_frame(&mut ctx, 0.0);
        anim.render_frame(&mut ctx, 30.0);
        anim.render_frame(&mut ctx, 59.0);
    }

    #[test]
    fn test_self_referencing_precomp_does_not_overflow_stack() {
        use crate::animation::Animation;

        // A precomp asset "c" whose only layer is itself a comp layer
        // referencing "c" — a self-referencing `refId` cycle. Rendering
        // must bail out past MAX_PRECOMP_DEPTH rather than recursing until
        // the thread stack overflows.
        let json = r#"{
            "v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,
            "assets":[{"id":"c","layers":[
                {"ty":0,"nm":"self","ind":1,"ip":0,"op":100,"refId":"c",
                 "w":100,"h":100}
            ]}],
            "layers":[
                {"ty":0,"nm":"root","ind":1,"ip":0,"op":60,"refId":"c",
                 "w":100,"h":100}
            ]
        }"#;

        let anim = Animation::from_json(json).unwrap();
        let mut canvas = MockCanvas::new();
        let mut ctx = RenderContext::new(&mut canvas);
        // Must complete and return rather than overflowing the stack.
        anim.render_frame(&mut ctx, 0.0);
    }

    #[test]
    fn test_resolve_matte_source_prefers_explicit_tp() {
        use crate::model::LayerModel;

        let mk = |ind: i32, extra: &str| -> Layer {
            let json =
                format!(r#"{{"ty":4,"nm":"l","ind":{ind},"ip":0,"op":100,"shapes":[]{extra}}}"#);
            let model: LayerModel = serde_json::from_str(&json).unwrap();
            Layer::from_lottie(&model)
        };

        let a = mk(1, "");
        let b = mk(2, "");
        let consumer = mk(3, r#","tt":1,"tp":1"#);
        let siblings = vec![a, b, consumer.clone()];

        let source = resolve_matte_source(&consumer, &siblings).unwrap();
        assert_eq!(
            source.index, 1,
            "explicit tp should win over array position"
        );
    }

    #[test]
    fn test_resolve_matte_source_falls_back_to_previous_array_element() {
        use crate::model::LayerModel;

        let mk = |ind: i32, extra: &str| -> Layer {
            let json =
                format!(r#"{{"ty":4,"nm":"l","ind":{ind},"ip":0,"op":100,"shapes":[]{extra}}}"#);
            let model: LayerModel = serde_json::from_str(&json).unwrap();
            Layer::from_lottie(&model)
        };

        let source = mk(1, r#","td":true"#);
        let consumer = mk(2, r#","tt":1"#);
        let siblings = vec![source, consumer.clone()];

        let resolved = resolve_matte_source(&consumer, &siblings).unwrap();
        assert_eq!(resolved.index, 1);
        assert!(resolved.matte_source_hidden);
    }

    /// End-to-end alpha track matte: a full-canvas red fill (`tt`:1) masked
    /// by a left-half-only white shape (`td`:true, hidden as a standalone
    /// layer). Per upstream `AttachMatteLayer`, the matte source is the
    /// layer immediately preceding the consumer in the JSON `layers` array.
    #[test]
    fn test_alpha_track_matte_composites_end_to_end() {
        use crate::animation::Animation;
        use skia_rs_canvas::Canvas as RasterCanvas;
        use skia_rs_canvas::PixelBuffer;

        let json = r#"{
            "v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,
            "layers":[
                {"ty":4,"nm":"source","ind":1,"ip":0,"op":60,"td":true,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[25,50]},"s":{"a":0,"k":[50,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,1,1,1]},"o":{"a":0,"k":100}}
                 ]},
                {"ty":4,"nm":"consumer","ind":2,"ip":0,"op":60,"tt":1,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
                 ]}
            ]
        }"#;

        let anim = Animation::from_json(json).unwrap();
        let mut buffer = PixelBuffer::new(100, 100);
        let mut raster = RasterCanvas::new_raster(&mut buffer);
        let mut canvas = SkiaCanvas::new(&mut raster);
        let mut ctx = RenderContext::new(&mut canvas);

        anim.render_frame(&mut ctx, 0.0);

        // Under the matte (left half): red, fully opaque.
        let inside = buffer.get_pixel(10, 50).unwrap();
        assert!(
            inside.alpha() > 200,
            "expected opaque pixel under the matte, got {inside:?}"
        );
        assert!(
            inside.red() > 200,
            "expected red under the matte, got {inside:?}"
        );

        // Outside the matte (right half): fully masked out (transparent).
        let outside = buffer.get_pixel(75, 50).unwrap();
        assert_eq!(
            outside.alpha(),
            0,
            "expected transparent pixel outside the matte, got {outside:?}"
        );

        // The (legacy-hidden) source itself must not be rendered standalone
        // — no white pixels anywhere.
        let source_area = buffer.get_pixel(10, 10).unwrap();
        assert!(
            !(source_area.red() > 200 && source_area.green() > 200 && source_area.blue() > 200),
            "matte source should not render standalone, got {source_area:?}"
        );
    }

    fn trim_shape(mode: i32, start_pct: Scalar, end_pct: Scalar) -> TrimPathShape {
        use crate::keyframe::{AnimatedProperty, KeyframeValue};
        TrimPathShape {
            name: String::new(),
            start: AnimatedProperty::static_value(KeyframeValue::Scalar(start_pct)),
            end: AnimatedProperty::static_value(KeyframeValue::Scalar(end_pct)),
            offset: AnimatedProperty::static_value(KeyframeValue::Scalar(0.0)),
            mode,
        }
    }

    fn horizontal_line(y: Scalar) -> Path {
        let mut b = skia_rs_path::PathBuilder::new();
        b.move_to(0.0, y);
        b.line_to(100.0, y);
        b.build()
    }

    /// `m:2` ("Individually", upstream kSerial in TrimPaths.cpp): all
    /// geometry merges into one path and a single trim spans the combined
    /// length — end=50% of two 100-unit lines keeps the FIRST line fully
    /// and leaves the second untouched.
    #[test]
    fn test_trim_mode_individual_spans_across_path_boundaries() {
        let paths = vec![horizontal_line(0.0), horizontal_line(10.0)];
        let trimmed = apply_trim(paths, &trim_shape(2, 0.0, 50.0), 0.0);

        assert_eq!(trimmed.len(), 1, "m:2 merges geometry into a single path");
        let measure = PathMeasure::new(&trimmed[0]);
        assert!(
            (measure.length() - 100.0).abs() < 0.01,
            "expected half of the combined 200-unit length, got {}",
            measure.length()
        );
        // Only the first line (y = 0) survives; the second (y = 10) is untouched.
        let bounds = trimmed[0].bounds();
        assert!(
            bounds.bottom < 5.0,
            "expected only the first (y=0) line to remain, got bounds {bounds:?}"
        );
    }

    /// `m:1` ("Simultaneously", upstream kParallel — the default): each
    /// geometry is trimmed independently with the same start/end.
    #[test]
    fn test_trim_mode_simultaneous_trims_each_path_independently() {
        let paths = vec![horizontal_line(0.0), horizontal_line(10.0)];
        let trimmed = apply_trim(paths, &trim_shape(1, 0.0, 50.0), 0.0);

        assert_eq!(trimmed.len(), 2, "m:1 keeps geometry separate");
        for p in &trimmed {
            let measure = PathMeasure::new(p);
            assert!(
                (measure.length() - 50.0).abs() < 0.01,
                "expected each line trimmed to its own half, got {}",
                measure.length()
            );
        }
    }

    /// A matte source that is not visible at the frame contributes ZERO
    /// coverage (per `sksg::MaskEffect`: content composited `SrcIn` against
    /// nothing) — the consumer must render nothing, not render unmasked.
    #[test]
    fn test_alpha_matte_with_invisible_source_masks_everything_out() {
        use crate::animation::Animation;
        use skia_rs_canvas::Canvas as RasterCanvas;
        use skia_rs_canvas::PixelBuffer;

        // Source is only visible on frames [30, 60); we render frame 0.
        let json = r#"{
            "v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,
            "layers":[
                {"ty":4,"nm":"source","ind":1,"ip":30,"op":60,"td":true,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,1,1,1]},"o":{"a":0,"k":100}}
                 ]},
                {"ty":4,"nm":"consumer","ind":2,"ip":0,"op":60,"tt":1,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
                 ]}
            ]
        }"#;

        let anim = Animation::from_json(json).unwrap();
        let mut buffer = PixelBuffer::new(100, 100);
        let mut raster = RasterCanvas::new_raster(&mut buffer);
        let mut canvas = SkiaCanvas::new(&mut raster);
        let mut ctx = RenderContext::new(&mut canvas);

        anim.render_frame(&mut ctx, 0.0);

        let pixel = buffer.get_pixel(50, 50).unwrap();
        assert_eq!(
            pixel.alpha(),
            0,
            "zero matte coverage must mask the consumer out entirely, got {pixel:?}"
        );
    }

    /// Inverted alpha matte (`tt`:2) with an invisible source: zero
    /// coverage inverts to FULL coverage — the consumer renders fully.
    #[test]
    fn test_inverted_alpha_matte_with_invisible_source_shows_consumer_fully() {
        use crate::animation::Animation;
        use skia_rs_canvas::Canvas as RasterCanvas;
        use skia_rs_canvas::PixelBuffer;

        let json = r#"{
            "v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,
            "layers":[
                {"ty":4,"nm":"source","ind":1,"ip":30,"op":60,"td":true,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,1,1,1]},"o":{"a":0,"k":100}}
                 ]},
                {"ty":4,"nm":"consumer","ind":2,"ip":0,"op":60,"tt":2,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
                 ]}
            ]
        }"#;

        let anim = Animation::from_json(json).unwrap();
        let mut buffer = PixelBuffer::new(100, 100);
        let mut raster = RasterCanvas::new_raster(&mut buffer);
        let mut canvas = SkiaCanvas::new(&mut raster);
        let mut ctx = RenderContext::new(&mut canvas);

        anim.render_frame(&mut ctx, 0.0);

        let pixel = buffer.get_pixel(50, 50).unwrap();
        assert!(
            pixel.alpha() > 200 && pixel.red() > 200,
            "inverted matte with zero coverage must show the consumer fully, got {pixel:?}"
        );
    }

    /// End-to-end luma track matte (`tt`:3): a 50%-gray source produces
    /// ~50% mask coverage (BT.709 luma of an equal-channel gray is that
    /// channel's value, independent of the exact weights).
    #[test]
    fn test_luma_track_matte_composites_end_to_end() {
        use crate::animation::Animation;
        use skia_rs_canvas::Canvas as RasterCanvas;
        use skia_rs_canvas::PixelBuffer;

        let json = r#"{
            "v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,
            "layers":[
                {"ty":4,"nm":"source","ind":1,"ip":0,"op":60,"td":true,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[0.5,0.5,0.5,1]},"o":{"a":0,"k":100}}
                 ]},
                {"ty":4,"nm":"consumer","ind":2,"ip":0,"op":60,"tt":3,
                 "shapes":[
                    {"ty":"rc","p":{"a":0,"k":[50,50]},"s":{"a":0,"k":[100,100]}},
                    {"ty":"fl","c":{"a":0,"k":[1,0,0,1]},"o":{"a":0,"k":100}}
                 ]}
            ]
        }"#;

        let anim = Animation::from_json(json).unwrap();
        let mut buffer = PixelBuffer::new(100, 100);
        let mut raster = RasterCanvas::new_raster(&mut buffer);
        let mut canvas = SkiaCanvas::new(&mut raster);
        let mut ctx = RenderContext::new(&mut canvas);

        anim.render_frame(&mut ctx, 0.0);

        let pixel = buffer.get_pixel(50, 50).unwrap();
        // ~50% coverage -> alpha roughly in [96, 160] (127 +/- slop for
        // sRGB-vs-linear luma weighting/AA differences).
        assert!(
            (96..=160).contains(&i32::from(pixel.alpha())),
            "expected ~50% alpha from luma matte, got {pixel:?}"
        );
    }
}
