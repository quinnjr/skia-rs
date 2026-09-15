//! Canvas drawing interface.
//!
//! The [`Canvas`] type is the single public drawing surface. It carries a
//! [`Backing`] enum that selects where draws actually go:
//!
//! - [`Backing::Raster`] — CPU rasterizer against a borrowed [`PixelBuffer`].
//! - [`Backing::Recording`] — appends [`DrawCommand`]s to a borrowed buffer,
//!   used by [`PictureRecorder`] to build a replayable [`Picture`].
//! - [`Backing::Null`] — no-op. Useful for matrix/clip-only benchmarks and
//!   quick-reject probes.
//!
//! The enum is open to future variants (GPU, PDF, SVG) without breaking the
//! public method surface.
//!
//! [`PixelBuffer`]: crate::raster::PixelBuffer
//! [`DrawCommand`]: crate::picture::DrawCommand
//! [`PictureRecorder`]: crate::picture::PictureRecorder
//! [`Picture`]: crate::picture::Picture

use crate::clip::ClipStack;
use crate::picture::DrawCommand;
use crate::raster::{PixelBuffer, Rasterizer};
use skia_rs_core::{Color, IRect, Matrix, Point, Rect, Scalar};
use skia_rs_paint::Paint;
use skia_rs_path::Path;

/// Clip operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ClipOp {
    /// Intersect with clip.
    #[default]
    Intersect = 0,
    /// Difference from clip.
    Difference,
}

/// Save layer flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SaveLayerFlags(u32);

impl SaveLayerFlags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// Preserve LCD text.
    pub const PRESERVE_LCD_TEXT: Self = Self(1 << 1);
    /// Initialize with previous layer.
    pub const INIT_WITH_PREVIOUS: Self = Self(1 << 2);
}

/// Save layer record.
#[derive(Debug, Clone, Default)]
pub struct SaveLayerRec<'a> {
    /// Bounds for the layer.
    pub bounds: Option<&'a Rect>,
    /// Paint for the layer.
    pub paint: Option<&'a Paint>,
    /// Flags.
    pub flags: SaveLayerFlags,
}

/// The rendering backend a [`Canvas`] is attached to.
///
/// New variants (GPU, PDF, SVG, …) can be added without rewriting the public
/// draw method surface — each method just adds another match arm.
pub enum Backing<'a> {
    /// Rasterize directly into the borrowed pixel buffer.
    Raster(&'a mut PixelBuffer),
    /// Append draw commands into the borrowed command buffer.
    ///
    /// This is how [`crate::picture::PictureRecorder`] captures a picture.
    Recording(&'a mut Vec<DrawCommand>),
    /// Discard every draw. Matrix and clip stack state is still maintained so
    /// that [`Canvas::quick_reject`] and friends can be used against a live
    /// canvas without a backing store.
    Null,
}

impl std::fmt::Debug for Backing<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backing::Raster(_) => f.debug_tuple("Raster").field(&"<PixelBuffer>").finish(),
            Backing::Recording(cmds) => f
                .debug_tuple("Recording")
                .field(&format_args!("<{} cmds>", cmds.len()))
                .finish(),
            Backing::Null => f.write_str("Null"),
        }
    }
}

/// An offscreen layer pushed by [`Canvas::save_layer`].
///
/// While this layer is on top of the stack, all draws are routed into
/// `buffer` instead of the base backing. On [`Canvas::restore`] the buffer
/// is composited back onto the enclosing target (the next layer down or the
/// base backing) using `paint`'s alpha and blend mode.
struct Layer {
    /// Offscreen pixel buffer holding this layer's contents.
    buffer: PixelBuffer,
    /// Rounded **device-space** origin of the buffer: buffer pixel (0, 0)
    /// sits at this device pixel. The same origin is used while drawing
    /// into the layer and when compositing it back, so draw and composite
    /// share one convention.
    device_origin: (i32, i32),
    /// Paint controlling how the layer composites back onto the parent.
    /// `None` means use a default [`Paint`] (opaque [`BlendMode::SrcOver`]).
    paint: Option<Paint>,
    /// Canvas save count at the moment this layer was pushed. Used by
    /// [`Canvas::restore`] to decide whether the current restore should
    /// pop and composite this layer.
    save_count_when_pushed: usize,
}

/// The main drawing interface.
///
/// `Canvas` is the single public canvas type. Its [`Backing`] determines
/// whether draws go to a pixel buffer, into a recorded picture, or are
/// discarded.
pub struct Canvas<'a> {
    /// Where draws go.
    backing: Backing<'a>,
    /// Current transformation matrix stack.
    matrix_stack: Vec<Matrix>,
    /// Clip stack with full region, path, and AA support.
    clip_stack: ClipStack,
    /// Save count.
    save_count: usize,
    /// Width of the logical canvas.
    width: i32,
    /// Height of the logical canvas.
    height: i32,
    /// Stack of offscreen layers pushed by [`Canvas::save_layer`]. The top
    /// entry (if any) receives all rasterized draws until its matching
    /// [`Canvas::restore`], at which point it is composited back onto the
    /// enclosing target.
    layer_stack: Vec<Layer>,
}

impl<'a> Canvas<'a> {
    /// Create a no-op canvas with the given logical dimensions.
    ///
    /// This is equivalent to `Canvas::new_null(width, height)` and is kept as
    /// `new` for backwards compatibility. Draws are discarded; matrix and clip
    /// state is tracked.
    #[must_use]
    pub fn new(width: i32, height: i32) -> Self {
        Self::new_null(width, height)
    }

    /// Create a raster canvas that draws into the given pixel buffer.
    pub fn new_raster(buffer: &'a mut PixelBuffer) -> Self {
        use skia_rs_core::cast::scalar_from_i32;

        let width = buffer.width;
        let height = buffer.height;
        let bounds = Rect::from_xywh(0.0, 0.0, scalar_from_i32(width), scalar_from_i32(height));
        Self {
            backing: Backing::Raster(buffer),
            matrix_stack: vec![Matrix::IDENTITY],
            clip_stack: ClipStack::new(&bounds),
            save_count: 1,
            width,
            height,
            layer_stack: Vec::new(),
        }
    }

    /// Create a recording canvas that appends draw commands into `commands`.
    ///
    /// `width`/`height` are only used to seed the initial clip bounds and to
    /// answer [`width`](Self::width)/[`height`](Self::height) queries.
    pub fn new_recording(commands: &'a mut Vec<DrawCommand>, width: i32, height: i32) -> Self {
        use skia_rs_core::cast::scalar_from_i32;

        let bounds = Rect::from_xywh(0.0, 0.0, scalar_from_i32(width), scalar_from_i32(height));
        Self {
            backing: Backing::Recording(commands),
            matrix_stack: vec![Matrix::IDENTITY],
            clip_stack: ClipStack::new(&bounds),
            save_count: 1,
            width,
            height,
            layer_stack: Vec::new(),
        }
    }

    /// Create a no-op canvas with the given dimensions.
    #[must_use]
    pub fn new_null(width: i32, height: i32) -> Self {
        use skia_rs_core::cast::scalar_from_i32;

        let bounds = Rect::from_xywh(0.0, 0.0, scalar_from_i32(width), scalar_from_i32(height));
        Self {
            backing: Backing::Null,
            matrix_stack: vec![Matrix::IDENTITY],
            clip_stack: ClipStack::new(&bounds),
            save_count: 1,
            width,
            height,
            layer_stack: Vec::new(),
        }
    }

    /// Get the width.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> i32 {
        self.width
    }

    /// Get the height.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.height
    }

    /// Get the current save count.
    #[inline]
    #[must_use]
    pub const fn save_count(&self) -> usize {
        self.save_count
    }

    /// Get the current transformation matrix.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the matrix stack always has at least one
    /// entry (the identity matrix pushed at construction).
    #[inline]
    #[must_use]
    pub fn total_matrix(&self) -> &Matrix {
        self.matrix_stack.last().unwrap()
    }

    /// Get the current clip bounds.
    #[inline]
    #[must_use]
    pub fn clip_bounds(&self) -> Rect {
        self.clip_stack.bounds()
    }

    /// Inspect the backing (primarily for diagnostics and tests).
    #[inline]
    #[must_use]
    pub const fn backing(&self) -> &Backing<'a> {
        &self.backing
    }

    /// Save the current state.
    ///
    /// Returns the save count **before** this save (`SkCanvas::save` returns
    /// `getSaveCount() - 1`; a fresh canvas has save count 1). Pass the
    /// returned value to [`restore_to_count`](Self::restore_to_count) to undo
    /// this save.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the matrix stack always has at least one
    /// entry.
    pub fn save(&mut self) -> usize {
        let matrix = *self.matrix_stack.last().unwrap();
        self.matrix_stack.push(matrix);
        self.clip_stack.save();
        self.save_count += 1;
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Save);
        }
        self.save_count - 1
    }

    /// Save the current state with a layer.
    ///
    /// For raster backings this allocates an offscreen [`PixelBuffer`] sized
    /// to `rec.bounds` (or the full canvas if no bounds are given) and
    /// redirects all subsequent draws into it until the matching
    /// [`restore`](Self::restore). On restore the buffer is composited back
    /// onto the enclosing target using `rec.paint`'s alpha and blend mode.
    ///
    /// Recording backings simply emit a [`DrawCommand::SaveLayer`] and rely
    /// on playback (against a raster canvas) to do the actual composition.
    /// Null backings behave like plain [`save`](Self::save) — no layer is
    /// allocated — but the save count is incremented so the matching
    /// restore still balances.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the matrix stack always has at least one
    /// entry.
    pub fn save_layer(&mut self, rec: &SaveLayerRec<'_>) -> usize {
        use skia_rs_core::cast::scalar_from_i32;

        let bounds = rec.bounds.copied();
        let paint = rec.paint.cloned();

        // Record intent for Recording backings before mutating anything else.
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::SaveLayer {
                bounds,
                paint: paint.clone(),
            });
        }

        // Maintain matrix/clip/save-count stack exactly like save().
        let matrix = *self.matrix_stack.last().unwrap();
        self.matrix_stack.push(matrix);
        self.clip_stack.save();
        self.save_count += 1;

        // Only raster backings allocate an offscreen layer — Recording and
        // Null have no pixels to composite.
        if matches!(self.backing, Backing::Raster(_)) {
            // `rec.bounds` is a CONTENT HINT in **local** space
            // (SkCanvas::saveLayer): map it through the CTM to device space,
            // intersect with the current clip and the device bounds, round,
            // and size the layer buffer from that. No bounds -> the clip.
            let ctm = *self.matrix_stack.last().unwrap();
            let device_full = Rect::from_xywh(
                0.0,
                0.0,
                scalar_from_i32(self.width),
                scalar_from_i32(self.height),
            );
            let clip_bounds = self.clip_stack.bounds();
            let hinted = bounds.map_or(Some(clip_bounds), |b| {
                ctm.map_rect(&b).intersect(&clip_bounds)
            });
            let device_rect = hinted
                .and_then(|r| r.intersect(&device_full))
                .map_or_else(|| IRect::new(0, 0, 0, 0), |r| r.round());

            let (w, h) = (device_rect.width().max(0), device_rect.height().max(0));
            let buffer = PixelBuffer::new(w, h);
            // INIT_WITH_PREVIOUS is left unimplemented here — the buffer is
            // left fully transparent. Pre-seeding from the parent is left
            // as a follow-up because it requires reading back and mapping
            // bounds between coordinate spaces.
            self.layer_stack.push(Layer {
                buffer,
                device_origin: (device_rect.left, device_rect.top),
                paint,
                save_count_when_pushed: self.save_count,
            });
        }

        // Like save(): return the count BEFORE the save
        // (SkCanvas::saveLayer returns getSaveCount() - 1).
        self.save_count - 1
    }

    /// Restore to the previous state.
    ///
    /// If the top entry of the layer stack was pushed at the current save
    /// count, the layer is popped and composited onto its parent target
    /// (the next layer below, or the base backing) before the matrix/clip
    /// stacks are popped.
    ///
    /// # Panics
    ///
    /// Never panics in practice: `should_pop_layer` is only true when
    /// `layer_stack` is non-empty.
    pub fn restore(&mut self) {
        if self.save_count <= 1 {
            return;
        }

        // If the top layer was pushed at this save count, composite it
        // back onto its parent before popping save state.
        let should_pop_layer = self
            .layer_stack
            .last()
            .is_some_and(|l| l.save_count_when_pushed == self.save_count);
        if should_pop_layer {
            let layer = self.layer_stack.pop().unwrap();
            self.composite_layer(layer);
        }

        self.matrix_stack.pop();
        self.clip_stack.restore();
        self.save_count -= 1;

        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Restore);
        }
    }

    /// Composite a popped [`Layer`] back onto the enclosing target (the
    /// next layer below, or the base [`Backing::Raster`] buffer).
    ///
    /// The composite goes **through the current clip** (per-pixel coverage)
    /// and applies the layer paint's alpha, blend mode, and color filter.
    /// (An image filter, if present, is not yet applied here.)
    fn composite_layer(&mut self, layer: Layer) {
        let paint = layer.paint.unwrap_or_default();
        let blend_mode = paint.blend_mode();
        // Layer paint alpha scales the (premultiplied) source per-pixel.
        let layer_alpha = paint.alpha().clamp(0.0, 1.0);
        let color_filter = paint.color_filter().cloned();
        let (offset_x, offset_y) = layer.device_origin;

        // The clip stack lives in a disjoint field, so we can query it while
        // holding the mutable target borrow below.
        let clip_stack = &self.clip_stack;

        // Borrow the target buffer: either the next layer below (parent
        // layer) or the base raster backing. Recording/Null backings have
        // no pixels to composite into. Parent layers have their own device
        // origin to subtract when writing.
        let (target, (tox, toy)): (&mut PixelBuffer, (i32, i32)) =
            if let Some(parent) = self.layer_stack.last_mut() {
                let origin = parent.device_origin;
                (&mut parent.buffer, origin)
            } else {
                match &mut self.backing {
                    Backing::Raster(buf) => (*buf, (0, 0)),
                    _ => return,
                }
            };

        let src_buf = &layer.buffer;
        let src_w = src_buf.width;
        let src_h = src_buf.height;

        for y in 0..src_h {
            let dy = offset_y + y;
            for x in 0..src_w {
                let dx = offset_x + x;
                let src_offset = usize::try_from(y).unwrap_or(0) * src_buf.stride
                    + usize::try_from(x).unwrap_or(0) * 4;
                let sr = src_buf.pixels[src_offset];
                let sg = src_buf.pixels[src_offset + 1];
                let sb = src_buf.pixels[src_offset + 2];
                let sa_raw = src_buf.pixels[src_offset + 3];
                if sa_raw == 0
                    && matches!(
                        blend_mode,
                        skia_rs_paint::BlendMode::SrcOver | skia_rs_paint::BlendMode::DstOver
                    )
                    && color_filter.is_none()
                {
                    // Fully transparent pixels contribute nothing under
                    // over-style blends; skip to avoid pointless work.
                    continue;
                }

                // Composite through the current clip.
                let clip_cov = clip_stack.get_coverage(dx, dy);
                if clip_cov == 0 {
                    continue;
                }

                // Layer pixels are PREMULTIPLIED.
                let mut src = Color::from_argb(sa_raw, sr, sg, sb);

                // Color filters operate on unpremultiplied color.
                if let Some(cf) = &color_filter {
                    let unpremul = skia_rs_core::unpremultiply_color(src);
                    let filtered = cf.filter_color(skia_rs_core::Color4f::from_color(unpremul));
                    let clamped = skia_rs_core::Color4f::new(
                        filtered.r.clamp(0.0, 1.0),
                        filtered.g.clamp(0.0, 1.0),
                        filtered.b.clamp(0.0, 1.0),
                        filtered.a.clamp(0.0, 1.0),
                    );
                    src = skia_rs_core::premultiply_color(clamped.to_color());
                }

                // Scale all four premultiplied channels by paint alpha and
                // the clip coverage.
                let scale = layer_alpha * (f32::from(clip_cov) / 255.0);
                let s = |v: u8| skia_rs_core::cast::f32_to_u8_sat(f32::from(v) * scale);
                let src =
                    Color::from_argb(s(src.alpha()), s(src.red()), s(src.green()), s(src.blue()));
                target.blend_pixel(dx - tox, dy - toy, src, blend_mode);
            }
        }
    }

    /// Restore to a specific save count.
    ///
    /// Like `SkCanvas::restoreToCount`, `count` is clamped to 1 (the initial
    /// save count), so passing 0 restores everything instead of looping
    /// forever.
    pub fn restore_to_count(&mut self, count: usize) {
        let count = count.max(1);
        while self.save_count > count {
            self.restore();
        }
    }

    /// Translate the canvas.
    pub fn translate(&mut self, dx: Scalar, dy: Scalar) {
        let matrix = Matrix::translate(dx, dy);
        self.concat_internal(&matrix);
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Translate { dx, dy });
        }
    }

    /// Scale the canvas.
    pub fn scale(&mut self, sx: Scalar, sy: Scalar) {
        let matrix = Matrix::scale(sx, sy);
        self.concat_internal(&matrix);
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Scale { sx, sy });
        }
    }

    /// Rotate the canvas (angle in degrees).
    pub fn rotate(&mut self, degrees: Scalar) {
        let radians = degrees.to_radians();
        let matrix = Matrix::rotate(radians);
        self.concat_internal(&matrix);
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Rotate { degrees });
        }
    }

    /// Skew the canvas.
    pub fn skew(&mut self, sx: Scalar, sy: Scalar) {
        let matrix = Matrix::skew(sx, sy);
        self.concat_internal(&matrix);
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Skew { sx, sy });
        }
    }

    /// Concatenate a matrix.
    pub fn concat(&mut self, matrix: &Matrix) {
        self.concat_internal(matrix);
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::Concat { matrix: *matrix });
        }
    }

    /// Internal concat that only mutates the matrix stack, no recording.
    fn concat_internal(&mut self, matrix: &Matrix) {
        if let Some(current) = self.matrix_stack.last_mut() {
            *current = current.concat(matrix);
        }
    }

    /// Set the matrix.
    pub fn set_matrix(&mut self, matrix: &Matrix) {
        if let Some(current) = self.matrix_stack.last_mut() {
            *current = *matrix;
        }
        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::SetMatrix { matrix: *matrix });
        }
    }

    /// Reset the matrix to identity.
    pub fn reset_matrix(&mut self) {
        self.set_matrix(&Matrix::IDENTITY);
    }

    /// Clip to a rectangle.
    ///
    /// Under a rotated/skewed CTM the mapped rect is a quad, so the clip is
    /// applied as a path; `map_rect` (which would clip to the quad's
    /// bounding box) is only used for axis-aligned CTMs.
    pub fn clip_rect(&mut self, rect: &Rect, op: ClipOp, do_anti_alias: bool) {
        let matrix = *self.total_matrix();
        let device_bounds = IRect::new(0, 0, self.width, self.height);

        if matrix.skew_x() == 0.0 && matrix.skew_y() == 0.0 {
            let transformed = matrix.map_rect(rect);
            self.clip_stack
                .clip_rect_with_op(&transformed, op, &device_bounds, do_anti_alias);
        } else {
            let mut b = skia_rs_path::PathBuilder::new();
            b.add_rect(rect);
            let mut quad = b.build();
            quad.transform(&matrix);
            self.clip_stack
                .clip_path_with_op(&quad, &device_bounds, op, do_anti_alias);
        }

        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::ClipRect {
                rect: *rect,
                op,
                anti_alias: do_anti_alias,
            });
        }
    }

    /// Clip to a path.
    pub fn clip_path(&mut self, path: &Path, op: ClipOp, do_anti_alias: bool) {
        // Transform the path to device coordinates
        let matrix = self.total_matrix();
        let mut transformed_path = path.clone();
        transformed_path.transform(matrix);

        let device_bounds = IRect::new(0, 0, self.width, self.height);
        self.clip_stack
            .clip_path_with_op(&transformed_path, &device_bounds, op, do_anti_alias);

        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::ClipPath {
                path: path.clone(),
                op,
                anti_alias: do_anti_alias,
            });
        }
    }

    /// Clip to a rounded rectangle.
    pub fn clip_round_rect(
        &mut self,
        rect: &Rect,
        rx: Scalar,
        ry: Scalar,
        op: ClipOp,
        do_anti_alias: bool,
    ) {
        let matrix = *self.total_matrix();
        let mut path_builder = skia_rs_path::PathBuilder::new();
        path_builder.add_round_rect(rect, rx, ry);
        let mut path = path_builder.build();
        path.transform(&matrix);

        let device_bounds = IRect::new(0, 0, self.width, self.height);
        self.clip_stack
            .clip_path_with_op(&path, &device_bounds, op, do_anti_alias);

        if let Backing::Recording(commands) = &mut self.backing {
            commands.push(DrawCommand::ClipRoundRect {
                rect: *rect,
                rx,
                ry,
                op,
                anti_alias: do_anti_alias,
            });
        }
    }

    // =========================================================================
    // Raster-backing helper
    // =========================================================================

    /// Build a configured [`Rasterizer`] from the current matrix and clip.
    ///
    /// If the canvas has an active layer (pushed by
    /// [`save_layer`](Self::save_layer)), the rasterizer targets the top
    /// layer's offscreen buffer; otherwise it targets the base
    /// [`Backing::Raster`] buffer. For [`Backing::Recording`] and
    /// [`Backing::Null`] canvases without an active layer, returns `None`.
    ///
    /// The closure runs with `&mut Rasterizer`; its return value is
    /// propagated.
    #[inline]
    fn with_rasterizer<R>(&mut self, f: impl FnOnce(&mut Rasterizer<'_>) -> R) -> Option<R> {
        use skia_rs_core::cast::scalar_from_i32;

        let matrix = *self.matrix_stack.last().unwrap();
        let clip_stack = self.clip_stack.clone();

        // An active layer takes priority over the base backing. The layer
        // buffer is positioned at its device origin, so we pre-compose a
        // translation mapping device coordinates to buffer coordinates and
        // tell the rasterizer the origin so device-space clip queries and
        // device-rect fills stay aligned.
        if let Some(layer) = self.layer_stack.last_mut() {
            let (ox, oy) = layer.device_origin;
            let adjusted = if (ox, oy) == (0, 0) {
                matrix
            } else {
                Matrix::translate(-scalar_from_i32(ox), -scalar_from_i32(oy)).concat(&matrix)
            };
            let mut raster = Rasterizer::new(&mut layer.buffer);
            raster.set_matrix(&adjusted);
            raster.set_clip_stack(clip_stack);
            raster.set_origin(ox, oy);
            return Some(f(&mut raster));
        }

        if let Backing::Raster(buffer) = &mut self.backing {
            let mut raster = Rasterizer::new(buffer);
            raster.set_matrix(&matrix);
            raster.set_clip_stack(clip_stack);
            Some(f(&mut raster))
        } else {
            None
        }
    }

    // =========================================================================
    // Draw methods
    // =========================================================================

    /// Fill the current **device clip** with `paint`, regardless of the CTM
    /// (`SkDraw::drawPaint`). Shaders still sample through the CTM. Pixels
    /// outside the clip (including complex/AA clips) are untouched via the
    /// blitter's per-pixel clip coverage.
    fn fill_device_clip(&mut self, paint: &Paint) {
        let clip_rect = self.clip_stack.bounds();
        if clip_rect.is_empty() {
            return;
        }
        self.with_rasterizer(|raster| raster.fill_device_rect(&clip_rect, paint));
    }

    /// Clear the canvas with a color.
    ///
    /// `SkCanvas::clear` is `drawColor(color, SkBlendMode::kSrc)`: it only
    /// affects the **device clip**. If a layer is active (pushed by
    /// [`save_layer`](Self::save_layer)), the clear goes to the layer.
    pub fn clear(&mut self, color: Color) {
        match &mut self.backing {
            Backing::Recording(commands) => {
                commands.push(DrawCommand::Clear { color });
                return;
            }
            Backing::Null => return,
            Backing::Raster(_) => {}
        }

        // Fast path: full-device rect clip and no layer — bulk clear.
        if self.layer_stack.is_empty() {
            use skia_rs_core::cast::scalar_from_i32;
            let full = Rect::from_xywh(
                0.0,
                0.0,
                scalar_from_i32(self.width),
                scalar_from_i32(self.height),
            );
            if let crate::clip::ClipState::Rect(r) = self.clip_stack.current() {
                if r.left <= full.left
                    && r.top <= full.top
                    && r.right >= full.right
                    && r.bottom >= full.bottom
                {
                    if let Backing::Raster(buffer) = &mut self.backing {
                        buffer.clear(color);
                        return;
                    }
                }
            }
        }

        let mut paint = Paint::new();
        paint.set_color32(color);
        paint.set_blend_mode(skia_rs_paint::BlendMode::Src);
        self.fill_device_clip(&paint);
    }

    /// Draw a color over the device clip, regardless of the CTM.
    pub fn draw_color(&mut self, color: Color, blend_mode: skia_rs_paint::BlendMode) {
        match &mut self.backing {
            Backing::Raster(_) => {
                let mut paint = Paint::new();
                paint.set_color32(color);
                paint.set_blend_mode(blend_mode);
                self.fill_device_clip(&paint);
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawColor { color, blend_mode });
            }
            Backing::Null => {}
        }
    }

    /// Fill the device clip with the full paint (shader, blend mode, alpha),
    /// like `SkCanvas::drawPaint`. The clip is honored per pixel; the CTM
    /// only affects shader sampling, not the filled geometry.
    pub fn draw_paint(&mut self, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => self.fill_device_clip(paint),
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawPaint {
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw a point.
    pub fn draw_point(&mut self, pt: Point, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => {
                self.with_rasterizer(|raster| raster.draw_point(pt, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawPoint {
                    point: pt,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw points in the specified mode.
    ///
    /// Empty input produces no output. Odd-count input in `Lines` mode
    /// drops the orphan trailing point.
    pub fn draw_points(&mut self, mode: PointMode, points: &[Point], paint: &Paint) {
        if points.is_empty() {
            return;
        }
        match mode {
            PointMode::Points => {
                // SkDraw::drawPoints: with a positive stroke width, each
                // point is a width-sized square (butt/square cap) or circle
                // (round cap). Width 0 stays a hairline point.
                let width = paint.stroke_width();
                if width <= 0.0 {
                    for &p in points {
                        self.draw_point(p, paint);
                    }
                } else {
                    let half = width / 2.0;
                    let mut fill = paint.clone();
                    fill.set_style(skia_rs_paint::Style::Fill);
                    if paint.stroke_cap() == skia_rs_paint::StrokeCap::Round {
                        for &p in points {
                            self.draw_circle(p, half, &fill);
                        }
                    } else {
                        for &p in points {
                            let r = Rect::from_xywh(p.x - half, p.y - half, width, width);
                            self.draw_rect(&r, &fill);
                        }
                    }
                }
            }
            PointMode::Lines => {
                // Consecutive pairs form lines
                for chunk in points.chunks_exact(2) {
                    self.draw_line(chunk[0], chunk[1], paint);
                }
            }
            PointMode::Polygon => {
                // Consecutive points form a polyline
                for pair in points.windows(2) {
                    self.draw_line(pair[0], pair[1], paint);
                }
            }
        }
    }

    /// Draw a line.
    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => {
                self.with_rasterizer(|raster| raster.draw_line(p0, p1, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawLine {
                    p0,
                    p1,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw a rectangle.
    pub fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => {
                self.with_rasterizer(|raster| raster.draw_rect(rect, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawRect {
                    rect: *rect,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw an oval.
    pub fn draw_oval(&mut self, rect: &Rect, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => {
                self.with_rasterizer(|raster| raster.draw_oval(rect, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawOval {
                    rect: *rect,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw a circle.
    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => {
                self.with_rasterizer(|raster| raster.draw_circle(center, radius, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawCircle {
                    center,
                    radius,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw an arc.
    pub fn draw_arc(
        &mut self,
        oval: &Rect,
        start_angle: Scalar,
        sweep_angle: Scalar,
        use_center: bool,
        paint: &Paint,
    ) {
        match &mut self.backing {
            Backing::Raster(_) => {
                // Build the arc path here so recording stays authoritative
                // and the raster path matches the former RasterCanvas
                // implementation.
                let path = build_arc_path(oval, start_angle, sweep_angle, use_center);
                self.with_rasterizer(|raster| raster.draw_path(&path, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawArc {
                    oval: *oval,
                    start_angle,
                    sweep_angle,
                    use_center,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw a rounded rectangle.
    pub fn draw_round_rect(&mut self, rect: &Rect, rx: Scalar, ry: Scalar, paint: &Paint) {
        match &mut self.backing {
            Backing::Raster(_) => {
                let path = build_round_rect_path(rect, rx, ry);
                self.with_rasterizer(|raster| raster.draw_path(&path, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawRoundRect {
                    rect: *rect,
                    rx,
                    ry,
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw a path.
    ///
    /// If the [`Paint`] has a [`PathEffect`] attached (via
    /// [`Paint::set_path_effect`]), the effect is applied to the path before
    /// it is rasterized / recorded. Effects that produce an empty path
    /// (e.g. [`TrimEffect`] with a zero interval) are silently skipped.
    ///
    /// [`PathEffect`]: skia_rs_path::PathEffect
    /// [`TrimEffect`]: skia_rs_path::TrimEffect
    pub fn draw_path(&mut self, path: &Path, paint: &Paint) {
        // If the paint carries a path effect, apply it first and drop through
        // the `Path` slot with the effected path. Fall back to the input path
        // if the effect produces nothing.
        let effected = paint.path_effect().and_then(|e| e.apply(path));
        let final_path: &Path = effected.as_ref().unwrap_or(path);

        match &mut self.backing {
            Backing::Raster(_) => {
                self.with_rasterizer(|raster| raster.draw_path(final_path, paint));
            }
            Backing::Recording(commands) => {
                commands.push(DrawCommand::DrawPath {
                    path: final_path.clone(),
                    paint: paint.clone(),
                });
            }
            Backing::Null => {}
        }
    }

    /// Draw a picture.
    pub fn draw_picture(
        &mut self,
        picture: &crate::Picture,
        matrix: Option<&Matrix>,
        paint: Option<&Paint>,
    ) {
        // SkCanvas::onDrawPicture: a non-null paint wraps the playback in an
        // implicit saveLayer(bounds=cullRect, paint); otherwise a plain save.
        let restore_count = if let Some(p) = paint {
            let bounds = picture.cull_rect();
            let rec = SaveLayerRec {
                bounds: if bounds.is_empty() {
                    None
                } else {
                    Some(&bounds)
                },
                paint: Some(p),
                flags: SaveLayerFlags::NONE,
            };
            self.save_layer(&rec)
        } else {
            self.save()
        };
        if let Some(m) = matrix {
            self.concat(m);
        }
        picture.playback(self);
        self.restore_to_count(restore_count);
    }

    // =========================================================================
    // Quick Reject
    // =========================================================================

    /// Check if a rect would be fully clipped (quick reject).
    ///
    /// Returns true if drawing to this rect would have no visible effect.
    #[inline]
    #[must_use]
    pub fn quick_reject(&self, rect: &Rect) -> bool {
        let transformed = self.total_matrix().map_rect(rect);
        self.clip_stack.quick_reject(&transformed)
    }

    /// Check if a path would be fully clipped.
    #[inline]
    pub fn quick_reject_path(&self, path: &Path) -> bool {
        self.quick_reject(&path.bounds())
    }

    // =========================================================================
    // Image Drawing
    // =========================================================================

    /// Draw an image lattice (nine-patch style stretching).
    ///
    /// This API is designed for future texture-atlas or GPU backends. The
    /// current software-only Canvas implementation has no image-atlas
    /// infrastructure, so this method is a no-op. Callers with an actual image
    /// should use [`draw_image_nine`](Self::draw_image_nine) instead (requires
    /// `codec` feature).
    ///
    /// When image-atlas support is added, `image_bounds` will identify a
    /// sub-region of a bound texture atlas, `lattice` will define the
    /// stretch/fixed cells, and the result will be drawn into `dst`.
    #[cfg(feature = "codec")]
    #[allow(
        clippy::similar_names,
        reason = "src_x_edges/src_y_edges and dst_x_edges/dst_y_edges are paired coordinate arrays; the x/y naming is the clearest option"
    )]
    pub fn draw_image_lattice(
        &mut self,
        image: &skia_rs_codec::Image,
        lattice: &ImageLattice,
        dst: &Rect,
        _filter_mode: FilterMode,
        paint: Option<&Paint>,
    ) {
        use skia_rs_core::cast::{floor_to_i32, scalar_from_i32};

        let src_w = scalar_from_i32(image.width());
        let src_h = scalar_from_i32(image.height());
        if src_w <= 0.0 || src_h <= 0.0 {
            return;
        }

        // Build source edges: [0, x_divs..., src_w]
        let mut src_x_edges = vec![0.0];
        src_x_edges.extend(
            lattice
                .x_divs
                .iter()
                .map(|&x| scalar_from_i32(x).clamp(0.0, src_w)),
        );
        src_x_edges.push(src_w);

        let mut src_y_edges = vec![0.0];
        src_y_edges.extend(
            lattice
                .y_divs
                .iter()
                .map(|&y| scalar_from_i32(y).clamp(0.0, src_h)),
        );
        src_y_edges.push(src_h);

        // Build destination edges using proportional scaling.
        // Full 9-patch semantics (fixed corners, stretch middles) are a
        // future refinement; proportional baseline is functionally correct
        // when divs are chosen proportionally to dst size.
        let dst_scale_x = dst.width() / src_w;
        let dst_scale_y = dst.height() / src_h;
        let dst_x_edges: Vec<Scalar> = src_x_edges
            .iter()
            .map(|&x| x.mul_add(dst_scale_x, dst.left))
            .collect();
        let dst_y_edges: Vec<Scalar> = src_y_edges
            .iter()
            .map(|&y| y.mul_add(dst_scale_y, dst.top))
            .collect();

        // Iterate cells and call draw_image_rect for each non-empty cell
        for j in 0..(src_y_edges.len() - 1) {
            for i in 0..(src_x_edges.len() - 1) {
                // Check if this cell should be skipped (Transparent)
                if let Some(ref rect_types) = lattice.rect_types {
                    let cell_idx = j * (src_x_edges.len() - 1) + i;
                    if rect_types.get(cell_idx) == Some(&LatticeRectType::Transparent) {
                        continue;
                    }
                }

                let src_cell = skia_rs_core::IRect::new(
                    floor_to_i32(src_x_edges[i]),
                    floor_to_i32(src_y_edges[j]),
                    floor_to_i32(src_x_edges[i + 1]),
                    floor_to_i32(src_y_edges[j + 1]),
                );
                let dst_cell = Rect::new(
                    dst_x_edges[i],
                    dst_y_edges[j],
                    dst_x_edges[i + 1],
                    dst_y_edges[j + 1],
                );
                if src_cell.width() > 0
                    && src_cell.height() > 0
                    && dst_cell.width() > 0.0
                    && dst_cell.height() > 0.0
                {
                    self.draw_image_rect(image, Some(&src_cell), &dst_cell, paint);
                }
            }
        }
    }

    /// Draw multiple images from an atlas.
    ///
    /// Draws sprites from a single atlas image. Each sprite is defined by a
    /// source rectangle in `sprites` and transformed by the corresponding
    /// `RSXform` in `xforms`. Optional per-sprite tint colors can be provided.
    ///
    /// Each sprite is drawn using `save`/`concat`/`draw_image_rect`/`restore` to
    /// handle rotation, scale, and translation. Tint colors are accepted but
    /// not applied in this baseline implementation (requires color-filter-on-draw
    /// hookup tracked separately).
    #[cfg(feature = "codec")]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors SkCanvas::drawAtlas's parameter shape (atlas, xforms, tex rects, colors, blend mode, sampling, paint); bundling would break parity with the upstream API"
    )]
    pub fn draw_atlas(
        &mut self,
        atlas: &skia_rs_codec::Image,
        xforms: &[RSXform],
        sprites: &[Rect],
        _colors: Option<&[Color]>,
        _blend_mode: skia_rs_paint::BlendMode,
        _sampling: FilterMode,
        paint: Option<&Paint>,
    ) {
        use skia_rs_core::cast::saturate_to_i32;
        // Truncate-toward-zero-then-saturate, matching the previous `as i32`
        // cast exactly for all finite inputs (only differs from `as i32` on
        // NaN, which cannot arise from finite sprite-rect coordinates here).
        let trunc_to_i32 = |x: Scalar| saturate_to_i32(x.trunc());

        let count = xforms.len().min(sprites.len());
        for i in 0..count {
            let xform = &xforms[i];
            let tex = &sprites[i];
            let matrix = xform.to_matrix();

            // Draw the sprite: save current transform, concat the RSXform,
            // draw the atlas sub-rect to (0, 0) .. (tex.width, tex.height)
            // in local space, then restore.
            self.save();
            self.concat(&matrix);
            let dst_local = Rect::from_xywh(0.0, 0.0, tex.width(), tex.height());
            let src_rect = skia_rs_core::IRect::new(
                trunc_to_i32(tex.left),
                trunc_to_i32(tex.top),
                trunc_to_i32(tex.right),
                trunc_to_i32(tex.bottom),
            );
            self.draw_image_rect(atlas, Some(&src_rect), &dst_local, paint);
            self.restore();
        }
    }

    /// Draw a Coons patch.
    ///
    /// A Coons patch is a bicubic surface defined by 12 control points forming
    /// four boundary cubic Bezier curves. The patch is subdivided into an 8×8
    /// grid and rasterized as triangles. Corner colors (if provided) are
    /// bilinearly interpolated across the patch. Texture mapping
    /// (`tex_coords`) is not yet supported and is silently ignored.
    ///
    /// The 12 control points define the patch boundaries in this order:
    /// - Points 0-3: top edge (left to right)
    /// - Points 3-6: right edge (top to bottom)
    /// - Points 9-6: bottom edge (left to right, reversed)
    /// - Points 0, 11, 10, 9: left edge (top to bottom)
    ///
    /// # Panics
    ///
    /// Never panics in practice: the internal vertex-grid indices are always
    /// in bounds by construction (built from the same fixed `N`).
    #[allow(
        clippy::too_many_lines,
        reason = "Coons-patch subdivision, per-vertex color interpolation, and triangle emission form one cohesive algorithm mirroring SkCanvas::drawPatch; splitting would fragment the math"
    )]
    #[allow(
        clippy::similar_names,
        clippy::many_single_char_names,
        reason = "u/v (patch parameters) and r/g/b/a (color channels) plus idx_tl/idx_tr/idx_bl/idx_br (quad corner indices) are the standard names for Coons-patch and per-pixel color math"
    )]
    pub fn draw_patch(
        &mut self,
        cubic_points: &[Point; 12],
        colors: Option<&[Color; 4]>,
        _tex_coords: Option<&[Point; 4]>,
        _blend_mode: skia_rs_paint::BlendMode,
        paint: &Paint,
    ) {
        // Subdivide the patch into an N×N grid
        const N: usize = 8;

        // Evaluate a cubic Bezier curve at parameter t
        fn cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: Scalar) -> Point {
            let mt = 1.0 - t;
            let mt2 = mt * mt;
            let t2 = t * t;
            let mt3 = mt2 * mt;
            let t3 = t2 * t;
            Point::new(
                p3.x.mul_add(
                    t3,
                    (3.0 * p2.x * mt).mul_add(t2, p0.x.mul_add(mt3, 3.0 * p1.x * mt2 * t)),
                ),
                p3.y.mul_add(
                    t3,
                    (3.0 * p2.y * mt).mul_add(t2, p0.y.mul_add(mt3, 3.0 * p1.y * mt2 * t)),
                ),
            )
        }

        // Evaluate the Coons patch at (u, v) ∈ [0, 1]²
        fn eval_patch(c: &[Point; 12], u: Scalar, v: Scalar) -> Point {
            // Four boundary curves:
            // top:    c[0..=3]
            // right:  c[3..=6]
            // bottom: c[9..=6] (reversed)
            // left:   c[0], c[11], c[10], c[9]
            let top = cubic_bezier(c[0], c[1], c[2], c[3], u);
            let bottom = cubic_bezier(c[9], c[8], c[7], c[6], u);
            let left = cubic_bezier(c[0], c[11], c[10], c[9], v);
            let right = cubic_bezier(c[3], c[4], c[5], c[6], v);

            // Coons patch formula: bilinear blend of opposite edges, minus
            // the bilinear interpolation of corners (to avoid double-counting).
            let lc_x = v.mul_add(bottom.x, (1.0 - v) * top.x);
            let lc_y = v.mul_add(bottom.y, (1.0 - v) * top.y);

            let lr_x = u.mul_add(right.x, (1.0 - u) * left.x);
            let lr_y = u.mul_add(right.y, (1.0 - u) * left.y);

            let bl_x = (u * v).mul_add(
                c[6].x,
                ((1.0 - u) * v).mul_add(
                    c[9].x,
                    (u * (1.0 - v)).mul_add(c[3].x, (1.0 - u) * (1.0 - v) * c[0].x),
                ),
            );
            let bl_y = (u * v).mul_add(
                c[6].y,
                ((1.0 - u) * v).mul_add(
                    c[9].y,
                    (u * (1.0 - v)).mul_add(c[3].y, (1.0 - u) * (1.0 - v) * c[0].y),
                ),
            );

            Point::new(lc_x + lr_x - bl_x, lc_y + lr_y - bl_y)
        }

        use skia_rs_core::cast::{f32_to_u8_sat, scalar_from_i32};

        // Build a grid of vertices
        let n_scalar = scalar_from_i32(i32::try_from(N).unwrap_or(i32::MAX));
        let mut vertices = Vec::with_capacity((N + 1) * (N + 1));
        for j in 0..=N {
            for i in 0..=N {
                let u = scalar_from_i32(i32::try_from(i).unwrap_or(i32::MAX)) / n_scalar;
                let v = scalar_from_i32(i32::try_from(j).unwrap_or(i32::MAX)) / n_scalar;
                vertices.push(eval_patch(cubic_points, u, v));
            }
        }

        // Bilinearly interpolate corner colors at grid point (i, j)
        let get_color = |i: usize, j: usize| -> Option<Color> {
            let colors = colors?;
            let u = scalar_from_i32(i32::try_from(i).unwrap_or(i32::MAX)) / n_scalar;
            let v = scalar_from_i32(i32::try_from(j).unwrap_or(i32::MAX)) / n_scalar;

            // Corner colors: [top-left, top-right, bottom-right, bottom-left]
            let c = colors;
            let r = (u * v).mul_add(
                Scalar::from(c[2].red()),
                (u * (1.0 - v)).mul_add(
                    Scalar::from(c[1].red()),
                    ((1.0 - u) * v).mul_add(
                        Scalar::from(c[3].red()),
                        (1.0 - u) * (1.0 - v) * Scalar::from(c[0].red()),
                    ),
                ),
            );
            let g = (u * v).mul_add(
                Scalar::from(c[2].green()),
                (u * (1.0 - v)).mul_add(
                    Scalar::from(c[1].green()),
                    ((1.0 - u) * v).mul_add(
                        Scalar::from(c[3].green()),
                        (1.0 - u) * (1.0 - v) * Scalar::from(c[0].green()),
                    ),
                ),
            );
            let b = (u * v).mul_add(
                Scalar::from(c[2].blue()),
                (u * (1.0 - v)).mul_add(
                    Scalar::from(c[1].blue()),
                    ((1.0 - u) * v).mul_add(
                        Scalar::from(c[3].blue()),
                        (1.0 - u) * (1.0 - v) * Scalar::from(c[0].blue()),
                    ),
                ),
            );
            let a = (u * v).mul_add(
                Scalar::from(c[2].alpha()),
                (u * (1.0 - v)).mul_add(
                    Scalar::from(c[1].alpha()),
                    ((1.0 - u) * v).mul_add(
                        Scalar::from(c[3].alpha()),
                        (1.0 - u) * (1.0 - v) * Scalar::from(c[0].alpha()),
                    ),
                ),
            );

            Some(Color::from_argb(
                f32_to_u8_sat(a),
                f32_to_u8_sat(r),
                f32_to_u8_sat(g),
                f32_to_u8_sat(b),
            ))
        };

        // Build triangles: for each cell, two triangles (TL-TR-BR, TL-BR-BL)
        let mut triangle_verts = Vec::new();
        let mut triangle_colors = Vec::new();
        for j in 0..N {
            for i in 0..N {
                let idx_tl = j * (N + 1) + i;
                let idx_tr = idx_tl + 1;
                let idx_bl = idx_tl + (N + 1);
                let idx_br = idx_bl + 1;

                // Triangle 1: TL-TR-BR
                triangle_verts.push(vertices[idx_tl]);
                triangle_verts.push(vertices[idx_tr]);
                triangle_verts.push(vertices[idx_br]);

                // Triangle 2: TL-BR-BL
                triangle_verts.push(vertices[idx_tl]);
                triangle_verts.push(vertices[idx_br]);
                triangle_verts.push(vertices[idx_bl]);

                if colors.is_some() {
                    // Colors for triangle 1
                    triangle_colors.push(get_color(i, j).unwrap());
                    triangle_colors.push(get_color(i + 1, j).unwrap());
                    triangle_colors.push(get_color(i + 1, j + 1).unwrap());
                    // Colors for triangle 2
                    triangle_colors.push(get_color(i, j).unwrap());
                    triangle_colors.push(get_color(i + 1, j + 1).unwrap());
                    triangle_colors.push(get_color(i, j + 1).unwrap());
                }
            }
        }

        let vertex_colors = if colors.is_some() {
            Some(triangle_colors.as_slice())
        } else {
            None
        };

        self.draw_vertices(VertexMode::Triangles, &triangle_verts, vertex_colors, paint);
    }

    /// Draw an annotation.
    ///
    /// Annotations attach metadata to a rectangle for consumption by
    /// structured output formats (PDF links, SVG  metadata, etc.). They have
    /// no visual effect on raster or null backings. Recording backings would
    /// capture the annotation for downstream consumers, but this is not yet
    /// implemented — the annotation is currently dropped in all cases.
    pub const fn draw_annotation(&mut self, _rect: &Rect, _key: &str, _value: &[u8]) {
        // Annotations are metadata-only. Raster and null backings simply drop
        // them. Recording backings would emit DrawCommand::Annotation for
        // PDF/SVG export, but that variant doesn't exist yet (tracked by P5-9).
    }

    // =========================================================================
    // Text Drawing
    // =========================================================================

    /// Draw glyphs at specified positions.
    ///
    /// Each glyph's outline is extracted from `font` and drawn as a path at
    /// `origin + positions[i]` (baseline position for glyph `i`). Glyphs
    /// with no outline (e.g. space, .notdef without data) are skipped.
    ///
    /// Colored glyphs (COLR v0/v1, SVG-in-OpenType) are rendered via
    /// [`Canvas::draw_color_glyph`]; the `paint` is honoured for alpha,
    /// blend mode, and filters, but the glyph's own fill data overrides
    /// the color/shader of `paint`.
    #[cfg(feature = "text")]
    pub fn draw_glyphs(
        &mut self,
        glyph_ids: &[u16],
        positions: &[Point],
        origin: Point,
        font: &skia_rs_text::Font,
        paint: &Paint,
    ) {
        for (i, &glyph) in glyph_ids.iter().enumerate() {
            let pos = positions.get(i).copied().unwrap_or(Point::zero());
            let at = Point::new(origin.x + pos.x, origin.y + pos.y);
            if self.draw_color_glyph(glyph, at, font, paint) {
                continue;
            }
            if let Some(path) = font.glyph_path(glyph) {
                let m = skia_rs_core::Matrix::translate(at.x, at.y);
                let transformed = path.transformed(&m);
                self.draw_path(&transformed, paint);
            }
        }
    }

    /// Draw positioned text with alignment.
    #[cfg(feature = "text")]
    pub fn draw_text_aligned(
        &mut self,
        text: &str,
        x: Scalar,
        y: Scalar,
        align: TextAlign,
        font: &skia_rs_text::Font,
        paint: &Paint,
    ) {
        let text_width = font.measure_text(text);
        let adjusted_x = match align {
            TextAlign::Left => x,
            TextAlign::Center => x - text_width / 2.0,
            TextAlign::Right => x - text_width,
        };
        self.draw_string(text, adjusted_x, y, font, paint);
    }

    /// Draw a string with its baseline at `(x, y)`.
    ///
    /// The text is first shaped (when font data is available) by rustybuzz
    /// into positioned glyphs; when no font data is present we fall back to
    /// a simple char-to-glyph mapping with per-glyph advances from
    /// [`skia_rs_text::Font::glyph_advance`]. For each shaped glyph the
    /// outline is extracted via [`skia_rs_text::Font::glyph_path`] and drawn
    /// as a path using the current paint, matrix, and clip stack. Glyphs
    /// with no outline (space, non-printing, dataless default typeface) are
    /// silently skipped.
    #[cfg(feature = "text")]
    pub fn draw_string(
        &mut self,
        text: &str,
        x: Scalar,
        y: Scalar,
        font: &skia_rs_text::Font,
        paint: &Paint,
    ) {
        // Prefer rustybuzz shaping when the typeface has real font data.
        // It handles ligatures, kerning, complex scripts correctly.
        let shaper = skia_rs_text::Shaper::new();
        if let Some(runs) = shaper.shape_auto(text, font) {
            let mut pen_x = x;
            let mut pen_y = y;
            for run in runs {
                for g in run.glyphs {
                    let glyph_x = pen_x + g.x_offset;
                    let glyph_y = pen_y + g.y_offset;
                    let at = Point::new(glyph_x, glyph_y);
                    if !self.draw_color_glyph(g.glyph_id.0, at, font, paint) {
                        if let Some(path) = font.glyph_path(g.glyph_id.0) {
                            let m = skia_rs_core::Matrix::translate(glyph_x, glyph_y);
                            let transformed = path.transformed(&m);
                            self.draw_path(&transformed, paint);
                        }
                    }
                    pen_x += g.x_advance;
                    pen_y += g.y_advance;
                }
            }
            return;
        }

        // Fallback path: no shape output (dataless typeface). Walk the
        // characters, advancing by font.glyph_advance per glyph.
        let mut pen_x = x;
        for c in text.chars() {
            let glyph = font.char_to_glyph(c);
            let at = Point::new(pen_x, y);
            if !self.draw_color_glyph(glyph, at, font, paint) {
                if let Some(path) = font.glyph_path(glyph) {
                    let m = skia_rs_core::Matrix::translate(pen_x, y);
                    let transformed = path.transformed(&m);
                    self.draw_path(&transformed, paint);
                }
            }
            pen_x += font.glyph_advance(glyph);
        }
    }

    /// Draw a color glyph if the font carries color data for it.
    ///
    /// Supports COLR v0 (solid-fill layers), COLR v1 (gradients, transforms,
    /// clips, and composites via `skia_rs_text::GlyphPaint`), and
    /// SVG-in-OpenType when the `svg` feature is enabled.
    ///
    /// Returns `true` when color-glyph rendering happened, `false` when the
    /// glyph is a plain outline (caller should fall back to `draw_path`).
    ///
    /// The base `paint` supplies alpha scaling, blend mode, anti-aliasing,
    /// and filters. The glyph's own paint data (colors, gradient stops,
    /// composite modes, transforms) overrides the base paint's color and
    /// shader.
    #[cfg(feature = "text")]
    pub fn draw_color_glyph(
        &mut self,
        glyph: u16,
        origin: Point,
        font: &skia_rs_text::Font,
        paint: &Paint,
    ) -> bool {
        if !font.glyph_is_color(glyph) {
            return false;
        }

        // COLR v0/v1 layered color glyph. Use palette 0 (default) and
        // treat the paint's current color as the "foreground" used for
        // layers that reference the `foreground` palette index — this is
        // how COLR v1 spec-wise opts into the draw-site color.
        //
        // SVG-in-OpenType glyphs are *not* handled here because the SVG
        // renderer lives in `skia-rs-svg`, which depends on canvas — pulling
        // it in from here would create a dependency cycle. The umbrella
        // `skia-rs` crate wires SVG glyphs through a thin adapter instead.
        let foreground: u32 = paint.color32().into();
        if let Some(layers) = font.glyph_color_layers(glyph, 0, foreground) {
            for layer in layers {
                self.draw_colr_layer(&layer, origin, font, paint);
            }
            return true;
        }

        // CBDT/CBLC/sbix raster glyphs and SVG-in-OpenType fall through —
        // the caller either handles them via skia-rs-svg or draws the
        // plain outline as a fallback.
        false
    }

    #[cfg(feature = "text")]
    fn draw_colr_layer(
        &mut self,
        layer: &skia_rs_text::ColorGlyphLayer,
        origin: Point,
        font: &skia_rs_text::Font,
        base_paint: &Paint,
    ) {
        // Layer outline in glyph space.
        let Some(mut outline) = font.glyph_path(layer.glyph) else {
            return;
        };
        // Apply the layer's in-glyph transform, then translate to origin.
        outline = outline.transformed(&layer.transform);
        let place = skia_rs_core::Matrix::translate(origin.x, origin.y);
        outline = outline.transformed(&place);

        // Clip the outline by clip_glyph's outline if set.
        if let Some(clip_id) = layer.clip_glyph {
            if let Some(clip_path) = font.glyph_path(clip_id) {
                let clip = clip_path.transformed(&layer.transform).transformed(&place);
                self.save();
                self.clip_path(&clip, ClipOp::Intersect, base_paint.is_anti_alias());
                self.fill_colr_layer(&outline, &layer.paint, layer.composite_mode, base_paint);
                self.restore();
                return;
            }
        }

        self.fill_colr_layer(&outline, &layer.paint, layer.composite_mode, base_paint);
    }

    #[cfg(feature = "text")]
    fn fill_colr_layer(
        &mut self,
        outline: &Path,
        glyph_paint: &skia_rs_text::GlyphPaint,
        composite: skia_rs_text::CompositeMode,
        base: &Paint,
    ) {
        use skia_rs_paint::{LinearGradient, RadialGradient, SweepGradient};
        use skia_rs_text::GlyphPaint;
        use std::sync::Arc;

        let blend = colr_composite_to_blend(composite);

        let mut p = base.clone();
        p.set_blend_mode(blend);

        match glyph_paint {
            GlyphPaint::Solid(argb) => {
                p.set_color32(skia_rs_core::Color::from(*argb));
                p.set_shader(None);
            }
            GlyphPaint::LinearGradient {
                p0,
                p1,
                stops,
                extend,
                ..
            } => {
                let (colors, positions) = stops_to_color4f(stops);
                let tile = extend_to_tile(*extend);
                let g = LinearGradient::new(*p0, *p1, colors, Some(positions), tile);
                p.set_shader(Some(Arc::new(g) as _));
            }
            GlyphPaint::RadialGradient {
                c0,
                r0,
                c1,
                r1,
                stops,
                extend,
            } => {
                let (colors, positions) = stops_to_color4f(stops);
                let tile = extend_to_tile(*extend);
                // skia-rs-paint's RadialGradient is single-circle; for
                // two-circle gradients we approximate with the end circle
                // (c1, r1) — the exact two-circle radial matches Skia's
                // SkGradientShader::MakeTwoPointConical, which is not yet
                // surfaced. For r0==0 the approximation is exact.
                let _ = c0;
                let radius = if *r1 > 0.0 { *r1 } else { r0.max(1.0) };
                let g = RadialGradient::new(*c1, radius, colors, Some(positions), tile);
                p.set_shader(Some(Arc::new(g) as _));
            }
            GlyphPaint::SweepGradient {
                center,
                start_angle,
                end_angle,
                stops,
                extend,
            } => {
                let (colors, positions) = stops_to_color4f(stops);
                let tile = extend_to_tile(*extend);
                // `start_angle`/`end_angle` are already in degrees (matching
                // SkShaders::SweepGradient), so pass them straight through.
                let g = SweepGradient::new(
                    *center,
                    *start_angle,
                    *end_angle,
                    colors,
                    Some(positions),
                    tile,
                );
                p.set_shader(Some(Arc::new(g) as _));
            }
        }

        self.draw_path(outline, &p);
    }

    /// Flush any pending drawing operations.
    ///
    /// For raster canvases, drawing is synchronous — every draw method
    /// mutates the backing pixel buffer immediately — so `flush()` is a
    /// no-op. For recording canvases, `flush()` does not force playback
    /// (that's handled separately by Picture consumers). For null
    /// canvases, `flush()` is trivially a no-op.
    ///
    /// When GPU and deferred backings land, this method will route to
    /// the backing's flush primitive.
    pub const fn flush(&mut self) {
        // Currently a no-op for all supported backings. GPU/deferred
        // routing will be added when those backings are introduced.
    }
}

// =============================================================================
// Auxiliary raster-only methods.
//
// These mirror the surface previously exposed through `RasterCanvas` but are
// not part of the unified `DrawCommand` set yet. They no-op on Recording /
// Null backings; adding recording variants is tracked by P5-9 / P5-6.
// =============================================================================

impl Canvas<'_> {
    /// Draw an image at the specified position.
    #[cfg(feature = "codec")]
    pub fn draw_image(
        &mut self,
        image: &skia_rs_codec::Image,
        left: Scalar,
        top: Scalar,
        paint: Option<&Paint>,
    ) {
        use skia_rs_core::cast::scalar_from_i32;

        let src_rect = skia_rs_core::IRect::new(0, 0, image.width(), image.height());
        let dst_rect = Rect::from_xywh(
            left,
            top,
            scalar_from_i32(image.width()),
            scalar_from_i32(image.height()),
        );
        self.draw_image_rect(image, Some(&src_rect), &dst_rect, paint);
    }

    /// Draw an image with source and destination rectangles.
    #[cfg(feature = "codec")]
    #[allow(
        clippy::too_many_lines,
        reason = "per-pixel image-rect blit loop mirrors SkDraw's image-rect drawing; splitting the hot loop would obscure the pixel pipeline"
    )]
    #[allow(
        clippy::similar_names,
        reason = "dst_x_start/dst_y_start/dst_x_end/dst_y_end and src_x/src_y are the natural names for a 2D pixel-rect blit loop"
    )]
    pub fn draw_image_rect(
        &mut self,
        image: &skia_rs_codec::Image,
        src: Option<&skia_rs_core::IRect>,
        dst: &Rect,
        paint: Option<&Paint>,
    ) {
        use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, saturate_to_i32, scalar_from_i32};

        let src_rect = src
            .copied()
            .unwrap_or_else(|| skia_rs_core::IRect::new(0, 0, image.width(), image.height()));
        if dst.is_empty() || src_rect.width() <= 0 || src_rect.height() <= 0 {
            return;
        }

        let matrix = *self.total_matrix();
        let Some(inverse) = matrix.invert() else {
            return;
        };
        let clip = self.clip_bounds();
        // The mapped dst under rotation is a quad; iterate its device AABB
        // and test membership by mapping each pixel center back to local
        // space, so rotation and complex clips both work.
        let transformed_dst = matrix.map_rect(dst);
        let Some(visible_dst) = transformed_dst.intersect(&clip) else {
            return;
        };

        let scale_x = scalar_from_i32(src_rect.width()) / dst.width();
        let scale_y = scalar_from_i32(src_rect.height()) / dst.height();

        let blend_mode = paint.map_or(skia_rs_paint::BlendMode::SrcOver, Paint::blend_mode);
        // Paint alpha is applied exactly ONCE (to all four premultiplied
        // channels).
        let alpha = paint.map_or(1.0, |p| p.alpha().clamp(0.0, 1.0));
        let image_is_premul = image.alpha_type() == skia_rs_core::AlphaType::Premul;

        let dst_x_start = floor_to_i32(visible_dst.left);
        let dst_x_end = ceil_to_i32(visible_dst.right);
        let dst_y_start = floor_to_i32(visible_dst.top);
        let dst_y_end = ceil_to_i32(visible_dst.bottom);

        // The clip stack lives in a disjoint field; query it while holding
        // the mutable buffer borrow below.
        let clip_stack = &self.clip_stack;

        // Route pixel writes to the top layer buffer if one is active, else
        // to the base raster backing. Layer buffers sit at their device
        // origin.
        let (buffer, buf_offset_x, buf_offset_y): (&mut PixelBuffer, i32, i32) =
            if let Some(layer) = self.layer_stack.last_mut() {
                let (ox, oy) = layer.device_origin;
                (&mut layer.buffer, ox, oy)
            } else {
                match &mut self.backing {
                    Backing::Raster(b) => (*b, 0, 0),
                    _ => return,
                }
            };

        for dst_y in dst_y_start..dst_y_end {
            for dst_x in dst_x_start..dst_x_end {
                // Per-pixel clip (handles path/AA/difference clips).
                let clip_cov = clip_stack.get_coverage(dst_x, dst_y);
                if clip_cov == 0 {
                    continue;
                }

                // Map the device pixel center back to local space and check
                // quad membership there (exact under rotation).
                let local = inverse.map_point(Point::new(
                    scalar_from_i32(dst_x) + 0.5,
                    scalar_from_i32(dst_y) + 0.5,
                ));
                if local.x < dst.left
                    || local.x >= dst.right
                    || local.y < dst.top
                    || local.y >= dst.bottom
                {
                    continue;
                }

                // Truncate-toward-zero-then-saturate, matching the previous
                // `as i32` cast exactly for all finite inputs.
                let src_x = saturate_to_i32(
                    (local.x - dst.left)
                        .mul_add(scale_x, scalar_from_i32(src_rect.left))
                        .trunc(),
                );
                let src_y = saturate_to_i32(
                    (local.y - dst.top)
                        .mul_add(scale_y, scalar_from_i32(src_rect.top))
                        .trunc(),
                );
                if src_x < 0 || src_x >= image.width() || src_y < 0 || src_y >= image.height() {
                    continue;
                }

                if let Some(src_color) = image.read_pixel(src_x, src_y) {
                    // Convert to premultiplied bytes for the premul pipeline.
                    let premul4 = if image_is_premul {
                        src_color
                    } else {
                        src_color.premul()
                    };
                    // Apply paint alpha once and the clip coverage, both on
                    // the premultiplied color.
                    let scale = alpha * (f32::from(clip_cov) / 255.0);
                    let to_byte = |v: f32| skia_rs_core::cast::f32_to_u8_sat(v * scale * 255.0);
                    let color = Color::from_argb(
                        to_byte(premul4.a),
                        to_byte(premul4.r),
                        to_byte(premul4.g),
                        to_byte(premul4.b),
                    );

                    buffer.blend_pixel(
                        dst_x - buf_offset_x,
                        dst_y - buf_offset_y,
                        color,
                        blend_mode,
                    );
                }
            }
        }
    }

    /// Draw an image with nine-patch stretching.
    #[cfg(feature = "codec")]
    #[allow(
        clippy::too_many_lines,
        reason = "nine-patch stretching is nine sequential draw_image_rect calls (corners/edges/center); splitting would fragment one cohesive layout"
    )]
    pub fn draw_image_nine(
        &mut self,
        image: &skia_rs_codec::Image,
        center: &skia_rs_core::IRect,
        dst: &Rect,
        paint: Option<&Paint>,
    ) {
        use skia_rs_core::cast::scalar_from_i32;

        let img_w = image.width();
        let img_h = image.height();

        let left_w = scalar_from_i32(center.left);
        let right_w = scalar_from_i32(img_w - center.right);
        let top_h = scalar_from_i32(center.top);
        let bottom_h = scalar_from_i32(img_h - center.bottom);

        let center_w = dst.width() - left_w - right_w;
        let center_h = dst.height() - top_h - bottom_h;

        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(0, 0, center.left, center.top)),
            &Rect::from_xywh(dst.left, dst.top, left_w, top_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                center.left,
                0,
                center.right,
                center.top,
            )),
            &Rect::from_xywh(dst.left + left_w, dst.top, center_w, top_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                center.right,
                0,
                img_w,
                center.top,
            )),
            &Rect::from_xywh(dst.right - right_w, dst.top, right_w, top_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                0,
                center.top,
                center.left,
                center.bottom,
            )),
            &Rect::from_xywh(dst.left, dst.top + top_h, left_w, center_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                center.left,
                center.top,
                center.right,
                center.bottom,
            )),
            &Rect::from_xywh(dst.left + left_w, dst.top + top_h, center_w, center_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                center.right,
                center.top,
                img_w,
                center.bottom,
            )),
            &Rect::from_xywh(dst.right - right_w, dst.top + top_h, right_w, center_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                0,
                center.bottom,
                center.left,
                img_h,
            )),
            &Rect::from_xywh(dst.left, dst.bottom - bottom_h, left_w, bottom_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                center.left,
                center.bottom,
                center.right,
                img_h,
            )),
            &Rect::from_xywh(dst.left + left_w, dst.bottom - bottom_h, center_w, bottom_h),
            paint,
        );
        self.draw_image_rect(
            image,
            Some(&skia_rs_core::IRect::new(
                center.right,
                center.bottom,
                img_w,
                img_h,
            )),
            &Rect::from_xywh(
                dst.right - right_w,
                dst.bottom - bottom_h,
                right_w,
                bottom_h,
            ),
            paint,
        );
    }

    /// Draw a region.
    pub fn draw_region(&mut self, region: &skia_rs_core::Region, paint: &Paint) {
        for rect in region.iter() {
            let rect_f = rect.to_rect();
            self.draw_rect(&rect_f, paint);
        }
    }

    /// Draw vertices (triangles).
    pub fn draw_vertices(
        &mut self,
        mode: VertexMode,
        positions: &[Point],
        colors: Option<&[Color]>,
        paint: &Paint,
    ) {
        if positions.len() < 3 {
            return;
        }

        let matrix = *self.total_matrix();

        match mode {
            VertexMode::Triangles => {
                for (tri_idx, chunk) in positions.chunks(3).enumerate() {
                    if chunk.len() == 3 {
                        let base_idx = tri_idx * 3;
                        let c0 = colors.and_then(|c| c.get(base_idx).copied());
                        let c1 = colors.and_then(|c| c.get(base_idx + 1).copied());
                        let c2 = colors.and_then(|c| c.get(base_idx + 2).copied());
                        self.draw_triangle(
                            matrix.map_point(chunk[0]),
                            matrix.map_point(chunk[1]),
                            matrix.map_point(chunk[2]),
                            c0,
                            c1,
                            c2,
                            paint,
                        );
                    }
                }
            }
            VertexMode::TriangleStrip => {
                for i in 0..positions.len().saturating_sub(2) {
                    let (p0, p1, p2, c0, c1, c2) = if i % 2 == 0 {
                        (
                            positions[i],
                            positions[i + 1],
                            positions[i + 2],
                            colors.and_then(|c| c.get(i).copied()),
                            colors.and_then(|c| c.get(i + 1).copied()),
                            colors.and_then(|c| c.get(i + 2).copied()),
                        )
                    } else {
                        (
                            positions[i + 1],
                            positions[i],
                            positions[i + 2],
                            colors.and_then(|c| c.get(i + 1).copied()),
                            colors.and_then(|c| c.get(i).copied()),
                            colors.and_then(|c| c.get(i + 2).copied()),
                        )
                    };
                    self.draw_triangle(
                        matrix.map_point(p0),
                        matrix.map_point(p1),
                        matrix.map_point(p2),
                        c0,
                        c1,
                        c2,
                        paint,
                    );
                }
            }
            VertexMode::TriangleFan => {
                let center = positions[0];
                for i in 1..positions.len().saturating_sub(1) {
                    let c0 = colors.and_then(|c| c.first().copied());
                    let c1 = colors.and_then(|c| c.get(i).copied());
                    let c2 = colors.and_then(|c| c.get(i + 1).copied());
                    self.draw_triangle(
                        matrix.map_point(center),
                        matrix.map_point(positions[i]),
                        matrix.map_point(positions[i + 1]),
                        c0,
                        c1,
                        c2,
                        paint,
                    );
                }
            }
        }
    }

    /// Draw a single filled triangle with per-vertex color interpolation.
    ///
    /// Both flat and interpolated triangles use the same coverage rule
    /// (pixel-center barycentric test), the clip is applied per pixel, paint
    /// alpha modulates the vertex colors, and the blended color is
    /// premultiplied for the premul pixel pipeline.
    #[allow(
        clippy::too_many_arguments,
        reason = "one vertex-color pair per triangle corner (p0/c0, p1/c1, p2/c2) plus paint mirrors the SkDraw triangle-rasterizer parameter shape; bundling would obscure the correspondence"
    )]
    fn draw_triangle(
        &mut self,
        p0: Point,
        p1: Point,
        p2: Point,
        c0: Option<Color>,
        c1: Option<Color>,
        c2: Option<Color>,
        paint: &Paint,
    ) {
        use skia_rs_core::cast::{ceil_to_i32, floor_to_i32, scalar_from_i32};
        use skia_rs_core::mul_div_255_round;

        // The clip stack lives in a disjoint field; query it while holding
        // the mutable buffer borrow below.
        let clip_stack = &self.clip_stack;

        let (buffer, buf_offset_x, buf_offset_y): (&mut PixelBuffer, i32, i32) =
            if let Some(layer) = self.layer_stack.last_mut() {
                let (ox, oy) = layer.device_origin;
                (&mut layer.buffer, ox, oy)
            } else {
                match &mut self.backing {
                    Backing::Raster(b) => (*b, 0, 0),
                    _ => return,
                }
            };

        let blend_mode = paint.blend_mode();
        let paint_alpha = paint.alpha().clamp(0.0, 1.0);

        // Vertex colors (or the paint color), modulated by paint alpha.
        let default_color = paint.color32().to_color4f();
        let modulate =
            |c: skia_rs_core::Color4f| skia_rs_core::Color4f::new(c.r, c.g, c.b, c.a * paint_alpha);
        let col0 = modulate(c0.map_or(default_color, |c| c.to_color4f()));
        let col1 = modulate(c1.map_or(default_color, |c| c.to_color4f()));
        let col2 = modulate(c2.map_or(default_color, |c| c.to_color4f()));
        let uniform = c0 == c1 && c1 == c2;
        let uniform_color = skia_rs_core::premultiply_color(col0.to_color());

        // Bounding box.
        let min_x = floor_to_i32(p0.x.min(p1.x).min(p2.x));
        let max_x = ceil_to_i32(p0.x.max(p1.x).max(p2.x));
        let min_y = floor_to_i32(p0.y.min(p1.y).min(p2.y));
        let max_y = ceil_to_i32(p0.y.max(p1.y).max(p2.y));

        let denom = (p1.y - p2.y).mul_add(p0.x - p2.x, (p2.x - p1.x) * (p0.y - p2.y));
        if denom.abs() < 1e-7 {
            return; // degenerate triangle
        }

        for y in min_y..max_y {
            for x in min_x..max_x {
                let px = scalar_from_i32(x) + 0.5;
                let py = scalar_from_i32(y) + 0.5;

                // Barycentric coordinates at the pixel center.
                let w0 = (p1.y - p2.y).mul_add(px - p2.x, (p2.x - p1.x) * (py - p2.y)) / denom;
                let w1 = (p2.y - p0.y).mul_add(px - p2.x, (p0.x - p2.x) * (py - p2.y)) / denom;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }

                // Per-pixel clip.
                let clip_cov = clip_stack.get_coverage(x, y);
                if clip_cov == 0 {
                    continue;
                }

                let mut color = if uniform {
                    uniform_color
                } else {
                    let r = col0.r.mul_add(w0, col1.r.mul_add(w1, col2.r * w2));
                    let g = col0.g.mul_add(w0, col1.g.mul_add(w1, col2.g * w2));
                    let b = col0.b.mul_add(w0, col1.b.mul_add(w1, col2.b * w2));
                    let a = col0.a.mul_add(w0, col1.a.mul_add(w1, col2.a * w2));
                    skia_rs_core::premultiply_color(
                        skia_rs_core::Color4f::new(
                            r.clamp(0.0, 1.0),
                            g.clamp(0.0, 1.0),
                            b.clamp(0.0, 1.0),
                            a.clamp(0.0, 1.0),
                        )
                        .to_color(),
                    )
                };

                if clip_cov < 255 {
                    let s = |v: u8| mul_div_255_round(u32::from(v), u32::from(clip_cov));
                    color = Color::from_argb(
                        s(color.alpha()),
                        s(color.red()),
                        s(color.green()),
                        s(color.blue()),
                    );
                }

                buffer.blend_pixel(x - buf_offset_x, y - buf_offset_y, color, blend_mode);
            }
        }
    }

    /// Draw a text blob at `(x, y)`.
    ///
    /// Iterates over the blob's pre-shaped runs and draws each glyph's
    /// outline as a path. The blob already stores positioned glyphs, so no
    /// reshaping happens here — this is the fast path for repeatedly
    /// drawing the same string.
    ///
    /// Glyphs with no outline (space, missing glyphs) are silently skipped.
    #[cfg(feature = "text")]
    pub fn draw_text_blob(
        &mut self,
        blob: &skia_rs_text::TextBlob,
        x: Scalar,
        y: Scalar,
        paint: &Paint,
    ) {
        for run in blob.runs() {
            let font = &run.font;
            for (i, &glyph) in run.glyphs.iter().enumerate() {
                // `i` is a glyph index; the precision loss of the usize->f32
                // cast is irrelevant for this fallback position estimate (only
                // matters beyond 2^24 glyphs in a single run).
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "glyph index for a fallback position estimate"
                )]
                let pos = run
                    .positions
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| Point::new(i as Scalar * font.size() * 0.5, 0.0));

                if let Some(path) = font.glyph_path(glyph) {
                    let gx = x + run.origin.x + pos.x;
                    let gy = y + run.origin.y + pos.y;
                    let m = skia_rs_core::Matrix::translate(gx, gy);
                    let transformed = path.transformed(&m);
                    self.draw_path(&transformed, paint);
                }
            }
        }
    }
}

/// Vertex drawing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum VertexMode {
    /// Separate triangles (every 3 vertices).
    #[default]
    Triangles = 0,
    /// Triangle strip (shared edges).
    TriangleStrip,
    /// Triangle fan (shared center vertex).
    TriangleFan,
}

// =============================================================================
// Arc / Round rect path construction shared with PictureRecorder playback.
// =============================================================================

fn build_round_rect_path(rect: &Rect, rx: Scalar, ry: Scalar) -> Path {
    use skia_rs_path::PathBuilder;

    // Quarter-circle conics of weight sqrt(2)/2 — the exact circular-arc
    // geometry Skia uses for round rect corners.
    const W: Scalar = std::f32::consts::FRAC_1_SQRT_2;

    let mut builder = PathBuilder::new();
    if rx <= 0.0 || ry <= 0.0 || rect.is_empty() {
        builder.add_rect(rect);
        return builder.build();
    }

    // SkRRect::setRectXY: when the radii exceed half the rect dimensions,
    // BOTH are scaled down by the same factor so the corner arcs still fit.
    let scale = (rect.width() / (2.0 * rx))
        .min(rect.height() / (2.0 * ry))
        .min(1.0);
    let rx = rx * scale;
    let ry = ry * scale;

    builder.move_to(rect.left + rx, rect.top);
    builder.line_to(rect.right - rx, rect.top);
    builder.conic_to(rect.right, rect.top, rect.right, rect.top + ry, W);
    builder.line_to(rect.right, rect.bottom - ry);
    builder.conic_to(rect.right, rect.bottom, rect.right - rx, rect.bottom, W);
    builder.line_to(rect.left + rx, rect.bottom);
    builder.conic_to(rect.left, rect.bottom, rect.left, rect.bottom - ry, W);
    builder.line_to(rect.left, rect.top + ry);
    builder.conic_to(rect.left, rect.top, rect.left + rx, rect.top, W);
    builder.close();
    builder.build()
}

fn build_arc_path(oval: &Rect, start_angle: Scalar, sweep_angle: Scalar, use_center: bool) -> Path {
    use skia_rs_path::{PathBuilder, PathElement};

    let mut builder = PathBuilder::new();
    if oval.is_empty() || sweep_angle == 0.0 {
        return builder.build();
    }

    // SkPathPriv::CreateDrawArcPath: |sweep| >= 360 draws the complete oval
    // regardless of the start angle.
    if sweep_angle.abs() >= 360.0 {
        builder.add_oval(oval);
        return builder.build();
    }

    // Curve-segment arc geometry from the path crate (Task 2), not
    // polylines.
    let mut arc_builder = PathBuilder::new();
    arc_builder.add_arc(oval, start_angle, sweep_angle);
    let arc_path = arc_builder.build();

    if !use_center {
        return arc_path;
    }

    // Wedge: center -> arc start -> arc -> close.
    let center = oval.center();
    builder.move_to(center.x, center.y);
    for element in &arc_path {
        match element {
            PathElement::Move(p) | PathElement::Line(p) => {
                builder.line_to(p.x, p.y);
            }
            PathElement::Quad(c, p) => {
                builder.quad_to(c.x, c.y, p.x, p.y);
            }
            PathElement::Conic(c, p, w) => {
                builder.conic_to(c.x, c.y, p.x, p.y, w);
            }
            PathElement::Cubic(c1, c2, p) => {
                builder.cubic_to(c1.x, c1.y, c2.x, c2.y, p.x, p.y);
            }
            PathElement::Close => {}
        }
    }
    builder.close();
    builder.build()
}

// =============================================================================
// Supporting Types
// =============================================================================

/// Text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TextAlign {
    /// Left-aligned text.
    #[default]
    Left = 0,
    /// Center-aligned text.
    Center,
    /// Right-aligned text.
    Right,
}

/// Image lattice for nine-patch style drawing.
#[derive(Debug, Clone)]
pub struct ImageLattice {
    /// X division points.
    pub x_divs: Vec<i32>,
    /// Y division points.
    pub y_divs: Vec<i32>,
    /// Rectangle flags (which cells are fixed vs. scalable).
    pub rect_types: Option<Vec<LatticeRectType>>,
    /// Bounds within the source image.
    pub bounds: Option<skia_rs_core::IRect>,
}

impl ImageLattice {
    /// Create a new image lattice.
    #[must_use]
    pub const fn new(x_divs: Vec<i32>, y_divs: Vec<i32>) -> Self {
        Self {
            x_divs,
            y_divs,
            rect_types: None,
            bounds: None,
        }
    }
}

/// Lattice rectangle type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum LatticeRectType {
    /// Default - draw the cell.
    #[default]
    Default = 0,
    /// Transparent - don't draw this cell.
    Transparent,
    /// Fixed color - fill with a solid color.
    FixedColor,
}

/// Rotation-scale transformation for atlas drawing.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct RSXform {
    /// Scale * cos(rotation).
    pub scos: Scalar,
    /// Scale * sin(rotation).
    pub ssin: Scalar,
    /// Translation X.
    pub tx: Scalar,
    /// Translation Y.
    pub ty: Scalar,
}

impl RSXform {
    /// Create from rotation and scale.
    #[must_use]
    pub fn from_radians(
        scale: Scalar,
        radians: Scalar,
        tx: Scalar,
        ty: Scalar,
        ax: Scalar,
        ay: Scalar,
    ) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            scos: scale * cos,
            ssin: scale * sin,
            tx: (-scale).mul_add(ax.mul_add(cos, -(ay * sin)), tx),
            ty: (-scale).mul_add(ax.mul_add(sin, ay * cos), ty),
        }
    }

    /// Create a simple translation + scale.
    #[must_use]
    pub const fn from_scale_translate(scale: Scalar, tx: Scalar, ty: Scalar) -> Self {
        Self {
            scos: scale,
            ssin: 0.0,
            tx,
            ty,
        }
    }

    /// Convert to a matrix.
    ///
    /// `SkMatrix::setRSXform` layout — tx/ty go in the translation slots
    /// (applied **last**):
    ///
    /// ```text
    /// [ scos  -ssin  tx ]
    /// [ ssin   scos  ty ]
    /// [    0      0   1 ]
    /// ```
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        Matrix {
            values: [
                self.scos, -self.ssin, self.tx, //
                self.ssin, self.scos, self.ty, //
                0.0, 0.0, 1.0,
            ],
        }
    }
}

/// Filter mode for image sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum FilterMode {
    /// Nearest neighbor sampling.
    #[default]
    Nearest = 0,
    /// Bilinear filtering.
    Linear,
}

/// Point drawing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PointMode {
    /// Draw each point.
    Points = 0,
    /// Draw lines between pairs.
    Lines,
    /// Draw connected line strip.
    Polygon,
}

// -----------------------------------------------------------------------------
// COLR v1 helpers: convert skia-rs-text color-glyph data into skia-rs-paint
// shader / blend inputs. These are free functions so the Canvas impl can
// keep the dispatching simple.
// -----------------------------------------------------------------------------

#[cfg(feature = "text")]
fn stops_to_color4f(
    stops: &[skia_rs_text::GradientStop],
) -> (Vec<skia_rs_core::Color4f>, Vec<Scalar>) {
    let mut sorted: Vec<_> = stops.to_vec();
    sorted.sort_by(|a, b| {
        a.offset_f32()
            .partial_cmp(&b.offset_f32())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut colors = Vec::with_capacity(sorted.len().max(1));
    let mut positions = Vec::with_capacity(sorted.len().max(1));
    for s in &sorted {
        let c = skia_rs_core::Color::from(s.color);
        colors.push(skia_rs_core::Color4f::from_color(c));
        positions.push(s.offset_f32().clamp(0.0, 1.0));
    }
    if colors.is_empty() {
        colors.push(skia_rs_core::Color4f::transparent());
        positions.push(0.0);
    }
    (colors, positions)
}

#[cfg(feature = "text")]
const fn extend_to_tile(extend: skia_rs_text::GradientExtend) -> skia_rs_paint::TileMode {
    use skia_rs_paint::TileMode;
    match extend {
        skia_rs_text::GradientExtend::Pad => TileMode::Clamp,
        skia_rs_text::GradientExtend::Repeat => TileMode::Repeat,
        skia_rs_text::GradientExtend::Reflect => TileMode::Mirror,
    }
}

#[cfg(feature = "text")]
const fn colr_composite_to_blend(
    mode: skia_rs_text::CompositeMode,
) -> skia_rs_paint::blend::BlendMode {
    use skia_rs_paint::blend::BlendMode;
    use skia_rs_text::CompositeMode as Cm;
    // Every COLR composite mode maps to a BlendMode; SrcOver is the common case.
    match mode {
        Cm::Clear => BlendMode::Clear,
        Cm::Source => BlendMode::Src,
        Cm::Destination => BlendMode::Dst,
        Cm::SourceOver => BlendMode::SrcOver,
        Cm::DestinationOver => BlendMode::DstOver,
        Cm::SourceIn => BlendMode::SrcIn,
        Cm::DestinationIn => BlendMode::DstIn,
        Cm::SourceOut => BlendMode::SrcOut,
        Cm::DestinationOut => BlendMode::DstOut,
        Cm::SourceAtop => BlendMode::SrcATop,
        Cm::DestinationAtop => BlendMode::DstATop,
        Cm::Xor => BlendMode::Xor,
        Cm::Plus => BlendMode::Plus,
        Cm::Screen => BlendMode::Screen,
        Cm::Overlay => BlendMode::Overlay,
        Cm::Darken => BlendMode::Darken,
        Cm::Lighten => BlendMode::Lighten,
        Cm::ColorDodge => BlendMode::ColorDodge,
        Cm::ColorBurn => BlendMode::ColorBurn,
        Cm::HardLight => BlendMode::HardLight,
        Cm::SoftLight => BlendMode::SoftLight,
        Cm::Difference => BlendMode::Difference,
        Cm::Exclusion => BlendMode::Exclusion,
        Cm::Multiply => BlendMode::Multiply,
        Cm::Hue => BlendMode::Hue,
        Cm::Saturation => BlendMode::Saturation,
        Cm::Color => BlendMode::Color,
        Cm::Luminosity => BlendMode::Luminosity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skia_rs_core::{Color, Rect};
    use skia_rs_paint::Style;

    #[test]
    fn test_null_canvas_discards_draws() {
        // Null canvas must still maintain matrix/clip state and answer
        // quick-reject queries, but should never crash on draw calls.
        let mut c = Canvas::new_null(100, 100);
        c.translate(10.0, 10.0);
        c.save();
        c.clear(Color::from_argb(255, 255, 0, 0));
        let paint = Paint::new();
        c.draw_rect(&Rect::from_xywh(0.0, 0.0, 50.0, 50.0), &paint);
        c.restore();
        assert_eq!(c.save_count(), 1);
    }

    #[test]
    fn test_recording_backing_captures_commands() {
        let mut cmds: Vec<DrawCommand> = Vec::new();
        {
            let mut c = Canvas::new_recording(&mut cmds, 100, 100);
            c.save();
            c.translate(5.0, 5.0);
            let paint = Paint::new();
            c.draw_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0), &paint);
            c.restore();
        }
        // save, translate, draw_rect, restore
        assert_eq!(cmds.len(), 4);
        assert!(matches!(cmds[0], DrawCommand::Save));
        assert!(matches!(cmds[1], DrawCommand::Translate { .. }));
        assert!(matches!(cmds[2], DrawCommand::DrawRect { .. }));
        assert!(matches!(cmds[3], DrawCommand::Restore));
    }

    #[test]
    fn test_quick_reject_respects_clip_and_matrix() {
        let mut c = Canvas::new_null(100, 100);
        assert!(!c.quick_reject(&Rect::from_xywh(10.0, 10.0, 20.0, 20.0)));
        c.translate(200.0, 200.0);
        assert!(c.quick_reject(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0)));
    }

    #[test]
    fn test_clip_path_does_not_collapse_to_bounds() {
        // Verify that clip_path stores the path geometry, not just its bounds.
        // A triangle clip should produce tighter clip bounds than a rectangle
        // of the same overall bounds would.
        use skia_rs_path::PathBuilder;

        let mut c = Canvas::new_null(100, 100);
        let mut builder = PathBuilder::new();
        builder.move_to(50.0, 10.0);
        builder.line_to(90.0, 90.0);
        builder.line_to(10.0, 90.0);
        builder.close();
        let triangle = builder.build();

        // Clip to the triangle path
        c.clip_path(&triangle, ClipOp::Intersect, false);

        // The clip bounds should closely match the triangle's bounds. Scan
        // conversion samples pixel centers, so each edge may be up to ~1.5px
        // inside the analytic bounds (the apex row has a sub-pixel span).
        let clip_bounds = c.clip_bounds();
        let path_bounds = triangle.bounds();
        assert!((clip_bounds.left - path_bounds.left).abs() <= 1.5);
        assert!((clip_bounds.top - path_bounds.top).abs() <= 1.5);
        assert!((clip_bounds.right - path_bounds.right).abs() <= 1.5);
        assert!((clip_bounds.bottom - path_bounds.bottom).abs() <= 1.5);

        // The clip must hold the actual path geometry: points inside the
        // triangle's bounds but outside the triangle are rejected.
        assert!(!c.quick_reject(&Rect::from_xywh(45.0, 50.0, 2.0, 2.0)));
        // quick_reject only tests bounds, so probe the clip stack directly.
        assert!(
            !c.clip_stack.contains(81, 13),
            "corner inside bounds but outside triangle must be clipped"
        );
    }

    #[test]
    fn test_clip_round_rect_clips_corner() {
        let mut c = Canvas::new_null(100, 100);
        c.clip_round_rect(
            &Rect::from_xywh(20.0, 20.0, 60.0, 60.0),
            10.0,
            10.0,
            ClipOp::Intersect,
            false,
        );

        assert!(c.clip_stack.contains(50, 50));
        assert!(!c.clip_stack.contains(20, 20));
    }

    #[test]
    fn test_clip_rect_difference() {
        // Verify that ClipOp::Difference creates a hole in the clip.
        let mut buffer = PixelBuffer::new(100, 100);
        let mut c = Canvas::new_raster(&mut buffer);

        // Apply a difference clip to create a hole
        let hole = Rect::from_xywh(40.0, 40.0, 20.0, 20.0);
        c.clip_rect(&hole, ClipOp::Difference, false);

        // Draw a white rectangle over the entire canvas
        let mut paint = Paint::new();
        paint.set_color32(Color::WHITE);
        c.draw_rect(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0), &paint);

        // Pixel at (50, 50) should be unchanged (black/transparent, initial state)
        // because it's in the difference-clipped hole
        let center_pixel = buffer.get_pixel(50, 50).unwrap_or(Color::TRANSPARENT);
        assert_ne!(
            center_pixel,
            Color::WHITE,
            "Center pixel should not be white due to difference clip"
        );

        // Pixel at (10, 10) should be white (outside the hole)
        let corner_pixel = buffer.get_pixel(10, 10).unwrap_or(Color::TRANSPARENT);
        assert_eq!(
            corner_pixel,
            Color::WHITE,
            "Corner pixel should be white (outside difference clip)"
        );
    }

    #[test]
    fn test_clip_rect_anti_alias() {
        // Verify that anti_alias flag is propagated to ClipStack.
        let mut buffer = PixelBuffer::new(100, 100);
        let mut c = Canvas::new_raster(&mut buffer);

        // Apply an AA clip
        let clip_rect = Rect::from_xywh(10.5, 10.5, 50.0, 50.0);
        c.clip_rect(&clip_rect, ClipOp::Intersect, true);

        // The clip stack should now be AA (though we can't easily verify the
        // exact coverage values without reaching into internals, we can at
        // least verify the canvas doesn't crash and maintains state correctly).
        let clip_bounds = c.clip_bounds();
        assert!(clip_bounds.contains(Point::new(30.0, 30.0)));
    }

    #[test]
    fn test_clip_path_with_transform() {
        // Verify that clip_path respects the current matrix.
        use skia_rs_path::PathBuilder;

        let mut c = Canvas::new_null(200, 200);
        c.translate(50.0, 50.0);

        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0);
        builder.line_to(50.0, 0.0);
        builder.line_to(50.0, 50.0);
        builder.line_to(0.0, 50.0);
        builder.close();
        let rect_path = builder.build();

        c.clip_path(&rect_path, ClipOp::Intersect, false);

        // The clip bounds should be translated
        let clip_bounds = c.clip_bounds();
        assert!(clip_bounds.left >= 49.0 && clip_bounds.left <= 51.0);
        assert!(clip_bounds.top >= 49.0 && clip_bounds.top <= 51.0);
    }

    #[test]
    fn test_clip_stack_save_restore() {
        // Verify that save/restore properly manages the clip stack.
        let mut c = Canvas::new_null(100, 100);

        let initial_bounds = c.clip_bounds();

        c.save();
        c.clip_rect(
            &Rect::from_xywh(10.0, 10.0, 50.0, 50.0),
            ClipOp::Intersect,
            false,
        );
        let clipped_bounds = c.clip_bounds();
        assert!(clipped_bounds.width() < initial_bounds.width());

        c.restore();
        let restored_bounds = c.clip_bounds();
        assert_eq!(restored_bounds, initial_bounds);
    }

    #[test]
    fn test_draw_points_points_mode_draws_each() {
        let mut buffer = PixelBuffer::new(100, 100);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::WHITE);
        paint.set_stroke_width(3.0);
        canvas.draw_points(
            PointMode::Points,
            &[
                Point::new(10.0, 10.0),
                Point::new(50.0, 50.0),
                Point::new(90.0, 90.0),
            ],
            &paint,
        );
        // Three distinct bright pixels should exist
        drop(canvas);
        assert_ne!(
            buffer.get_pixel(10, 10).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
        assert_ne!(
            buffer.get_pixel(50, 50).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
        assert_ne!(
            buffer.get_pixel(90, 90).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn test_draw_points_lines_mode_connects_pairs() {
        let mut buffer = PixelBuffer::new(100, 100);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::WHITE);
        // Two pairs: horizontal and vertical line
        canvas.draw_points(
            PointMode::Lines,
            &[
                Point::new(0.0, 50.0),
                Point::new(100.0, 50.0), // pair 1: horizontal line
                Point::new(50.0, 0.0),
                Point::new(50.0, 100.0), // pair 2: vertical line
            ],
            &paint,
        );
        // Verify by sampling midpoints of both lines
        drop(canvas);
        assert_ne!(
            buffer.get_pixel(50, 50).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn test_draw_points_polygon_mode_connects_consecutive() {
        let mut buffer = PixelBuffer::new(100, 100);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::WHITE);
        // Triangle perimeter: three lines should be drawn
        canvas.draw_points(
            PointMode::Polygon,
            &[
                Point::new(50.0, 10.0),
                Point::new(90.0, 90.0),
                Point::new(10.0, 90.0),
            ],
            &paint,
        );
        drop(canvas);
        // Lines connect 50,10->90,90 and 90,90->10,90
        // Sample a point on the second line
        assert_ne!(
            buffer.get_pixel(50, 90).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn test_draw_points_empty_slice() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let paint = Paint::new();
        canvas.draw_points(PointMode::Points, &[], &paint); // should not panic
        // Verify no pixels were drawn (buffer should be all transparent)
        drop(canvas);
        assert_eq!(
            buffer.get_pixel(5, 5).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn test_draw_points_lines_mode_odd_count_drops_last() {
        // Lines mode requires pairs; odd count drops the last point.
        let mut buffer = PixelBuffer::new(100, 100);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::WHITE);
        canvas.draw_points(
            PointMode::Lines,
            &[
                Point::new(0.0, 50.0),
                Point::new(100.0, 50.0),
                Point::new(50.0, 50.0), // orphan, should be ignored
            ],
            &paint,
        );
        // Should not panic; one line should be drawn
        drop(canvas);
        assert_ne!(
            buffer.get_pixel(50, 50).unwrap_or(Color::TRANSPARENT),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn test_draw_points_recording_backs_up_all_modes() {
        // Verify recording captures all PointMode variants properly.
        let mut cmds: Vec<DrawCommand> = Vec::new();
        {
            let mut c = Canvas::new_recording(&mut cmds, 100, 100);
            let paint = Paint::new();
            c.draw_points(
                PointMode::Points,
                &[Point::new(10.0, 10.0), Point::new(20.0, 20.0)],
                &paint,
            );
        }
        // Two DrawPoint commands should be recorded
        assert_eq!(
            cmds.iter()
                .filter(|cmd| matches!(cmd, DrawCommand::DrawPoint { .. }))
                .count(),
            2
        );

        let mut cmds = Vec::new();
        {
            let mut c = Canvas::new_recording(&mut cmds, 100, 100);
            let paint = Paint::new();
            c.draw_points(
                PointMode::Lines,
                &[
                    Point::new(0.0, 0.0),
                    Point::new(10.0, 10.0),
                    Point::new(20.0, 20.0),
                    Point::new(30.0, 30.0),
                ],
                &paint,
            );
        }
        // Two DrawLine commands should be recorded (one per pair)
        assert_eq!(
            cmds.iter()
                .filter(|cmd| matches!(cmd, DrawCommand::DrawLine { .. }))
                .count(),
            2
        );

        let mut cmds = Vec::new();
        {
            let mut c = Canvas::new_recording(&mut cmds, 100, 100);
            let paint = Paint::new();
            c.draw_points(
                PointMode::Polygon,
                &[
                    Point::new(0.0, 0.0),
                    Point::new(10.0, 10.0),
                    Point::new(20.0, 0.0),
                ],
                &paint,
            );
        }
        // Two DrawLine commands should be recorded (n-1 for polygon)
        assert_eq!(
            cmds.iter()
                .filter(|cmd| matches!(cmd, DrawCommand::DrawLine { .. }))
                .count(),
            2
        );
    }

    // =========================================================================
    // save_layer / restore layer-composition tests
    // =========================================================================

    #[test]
    fn test_save_layer_increments_save_count_and_restore_balances() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let before = canvas.save_count();
        let returned = canvas.save_layer(&SaveLayerRec::default());
        let during = canvas.save_count();
        assert_eq!(during, before + 1);
        // SkCanvas::saveLayer returns getSaveCount() - 1 (the pre-save count).
        assert_eq!(returned, before);
        canvas.restore();
        assert_eq!(canvas.save_count(), before);
    }

    #[test]
    fn test_save_layer_full_alpha_paint_is_pixel_identical_to_direct_draw() {
        // A save_layer with a default (opaque SrcOver) paint should composite
        // the layer unchanged onto the base buffer, matching a direct draw.
        let mut direct = PixelBuffer::new(8, 8);
        {
            let mut c = Canvas::new_raster(&mut direct);
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(255, 200, 100, 50));
            c.draw_rect(&Rect::from_xywh(1.0, 1.0, 6.0, 6.0), &paint);
        }

        let mut layered = PixelBuffer::new(8, 8);
        {
            let mut c = Canvas::new_raster(&mut layered);
            c.save_layer(&SaveLayerRec::default());
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(255, 200, 100, 50));
            c.draw_rect(&Rect::from_xywh(1.0, 1.0, 6.0, 6.0), &paint);
            c.restore();
        }

        for y in 0..8 {
            for x in 0..8 {
                let a = direct.get_pixel(x, y).unwrap();
                let b = layered.get_pixel(x, y).unwrap();
                assert_eq!(
                    a.as_u32(),
                    b.as_u32(),
                    "mismatch at ({x},{y}): direct={:08x} layered={:08x}",
                    a.as_u32(),
                    b.as_u32()
                );
            }
        }
    }

    #[test]
    fn test_save_layer_half_alpha_paint_halves_source_contribution() {
        // Draw an opaque white rect inside a save_layer whose paint has
        // ~50% alpha. The base buffer starts black; after composite the
        // result should be roughly mid-gray (~128), not 255.
        let mut buffer = PixelBuffer::new(4, 4);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);

        let mut layer_paint = Paint::new();
        layer_paint.set_color32(Color::from_argb(128, 255, 255, 255));
        let rec = SaveLayerRec {
            bounds: None,
            paint: Some(&layer_paint),
            flags: SaveLayerFlags::NONE,
        };
        canvas.save_layer(&rec);

        let mut white = Paint::new();
        white.set_color32(Color::WHITE);
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &white);

        canvas.restore();
        drop(canvas);

        let c = buffer.get_pixel(2, 2).unwrap();
        // With 50% paint alpha over solid black, expect ~128 on each channel.
        assert!(
            c.red() > 100 && c.red() < 160,
            "expected ~128, got r={}",
            c.red()
        );
        assert!(
            c.green() > 100 && c.green() < 160,
            "expected ~128, got g={}",
            c.green()
        );
        assert!(
            c.blue() > 100 && c.blue() < 160,
            "expected ~128, got b={}",
            c.blue()
        );
    }

    #[test]
    fn test_save_layer_bounds_restrict_composite_area() {
        // With bounds [0,0,4,4], a full-canvas draw inside the layer must
        // only affect pixels inside those bounds after restore.
        let mut buffer = PixelBuffer::new(10, 10);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);

        let layer_bounds = Rect::from_xywh(0.0, 0.0, 4.0, 4.0);
        let rec = SaveLayerRec {
            bounds: Some(&layer_bounds),
            paint: None,
            flags: SaveLayerFlags::NONE,
        };
        canvas.save_layer(&rec);

        let mut red = Paint::new();
        red.set_color32(Color::from_argb(255, 255, 0, 0));
        // Fill the whole canvas — most of it falls outside the layer's
        // 4x4 buffer and should simply be clipped away on composite.
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0), &red);
        canvas.restore();
        drop(canvas);

        // Inside bounds: red.
        let inside = buffer.get_pixel(1, 1).unwrap();
        assert_eq!(inside.red(), 255);
        // Outside bounds: still the original black — layer composition
        // must not have touched it.
        let outside = buffer.get_pixel(8, 8).unwrap();
        assert_eq!(outside.red(), 0);
        assert_eq!(outside.alpha(), 255);
    }

    #[test]
    fn test_save_layer_nested_lifo_composition() {
        // Two nested layers, each with 50% alpha paint. After both restores
        // the innermost draw should have been multiplied by both alphas
        // (~0.5 * 0.5 = 0.25), producing roughly 64 on the base.
        let mut buffer = PixelBuffer::new(4, 4);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);

        let mut p1 = Paint::new();
        p1.set_color32(Color::from_argb(128, 255, 255, 255));
        let rec1 = SaveLayerRec {
            bounds: None,
            paint: Some(&p1),
            flags: SaveLayerFlags::NONE,
        };
        canvas.save_layer(&rec1);

        let mut p2 = Paint::new();
        p2.set_color32(Color::from_argb(128, 255, 255, 255));
        let rec2 = SaveLayerRec {
            bounds: None,
            paint: Some(&p2),
            flags: SaveLayerFlags::NONE,
        };
        canvas.save_layer(&rec2);

        let mut white = Paint::new();
        white.set_color32(Color::WHITE);
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &white);

        canvas.restore();
        canvas.restore();
        drop(canvas);

        let c = buffer.get_pixel(2, 2).unwrap();
        // Two nested 50% alphas over solid black -> roughly 0.25 * 255 = 64.
        assert!(
            c.red() >= 32 && c.red() <= 96,
            "expected ~64, got r={}",
            c.red()
        );
    }

    #[test]
    fn test_save_layer_restore_to_count_composites_all_open_layers() {
        // restore_to_count must cascade through restore(), which in turn
        // composites each popped layer back onto its parent.
        let mut buffer = PixelBuffer::new(4, 4);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);
        let base = canvas.save_count();

        canvas.save_layer(&SaveLayerRec::default());
        canvas.save_layer(&SaveLayerRec::default());

        let mut white = Paint::new();
        white.set_color32(Color::WHITE);
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &white);

        canvas.restore_to_count(base);
        assert_eq!(canvas.save_count(), base);
        drop(canvas);

        // Both layers composited with default (opaque) paint — pixel should
        // be white on black, i.e. fully opaque white.
        let c = buffer.get_pixel(2, 2).unwrap();
        assert_eq!(c.red(), 255);
        assert_eq!(c.green(), 255);
        assert_eq!(c.blue(), 255);
    }

    #[test]
    fn test_save_layer_draws_do_not_leak_to_base_until_restore() {
        // Drawing inside save_layer must not touch the base buffer until
        // the matching restore composites the layer back.
        let mut buffer = PixelBuffer::new(4, 4);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);

        canvas.save_layer(&SaveLayerRec::default());
        let mut white = Paint::new();
        white.set_color32(Color::WHITE);
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &white);

        // Before restore we expect the base buffer to still be untouched.
        // We can't borrow the base buffer while the canvas holds it, so
        // drop the canvas first — dropping ends the &mut borrow without
        // running composition (composition only runs in restore()).
        drop(canvas);

        let c = buffer.get_pixel(2, 2).unwrap();
        assert_eq!(c.red(), 0, "base buffer should be untouched before restore");
        assert_eq!(c.green(), 0);
        assert_eq!(c.blue(), 0);
    }

    #[test]
    fn test_flush_is_safe_on_all_backings() {
        // Flush must be safe and non-panicking on all backing types.
        // It's a no-op for raster (synchronous), recording (does not force
        // playback), and null (trivially no-op).
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.flush();

        let mut commands = Vec::new();
        let mut canvas = Canvas::new_recording(&mut commands, 10, 10);
        canvas.flush();

        let mut canvas = Canvas::new_null(10, 10);
        canvas.flush();
    }

    #[test]
    fn test_draw_vertices_triangles_per_vertex_color() {
        // Triangle with red, green, blue at vertices should show color
        // interpolation: pixels near each vertex are dominated by that
        // vertex's color.
        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);
        let paint = Paint::new();

        canvas.draw_vertices(
            VertexMode::Triangles,
            &[
                Point::new(50.0, 10.0),
                Point::new(90.0, 90.0),
                Point::new(10.0, 90.0),
            ],
            Some(&[
                Color::from_rgb(255, 0, 0), // red at top
                Color::from_rgb(0, 255, 0), // green at bottom-right
                Color::from_rgb(0, 0, 255), // blue at bottom-left
            ]),
            &paint,
        );
        drop(canvas);

        // Near vertex 0 (50, 10) should be mostly red
        let near_v0 = buffer.get_pixel(50, 15).unwrap();
        assert!(
            near_v0.red() > 150,
            "near v0 should be red-dominated, got r={}",
            near_v0.red()
        );

        // Near vertex 1 (90, 90) should be mostly green
        let near_v1 = buffer.get_pixel(88, 88).unwrap();
        assert!(
            near_v1.green() > 150,
            "near v1 should be green-dominated, got g={}",
            near_v1.green()
        );

        // Near vertex 2 (10, 90) should be mostly blue
        let near_v2 = buffer.get_pixel(12, 88).unwrap();
        assert!(
            near_v2.blue() > 150,
            "near v2 should be blue-dominated, got b={}",
            near_v2.blue()
        );
    }

    #[test]
    fn test_draw_vertices_flat_when_no_colors() {
        // Without colors, should use paint's color for entire triangle
        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 128, 64));

        canvas.draw_vertices(
            VertexMode::Triangles,
            &[
                Point::new(10.0, 10.0),
                Point::new(90.0, 10.0),
                Point::new(50.0, 90.0),
            ],
            None,
            &paint,
        );
        drop(canvas);

        let c = buffer.get_pixel(50, 50).unwrap();
        assert!(
            (i32::from(c.red()) - 255).abs() < 10,
            "Expected ~255 red, got {}",
            c.red()
        );
        assert!(
            (i32::from(c.green()) - 128).abs() < 10,
            "Expected ~128 green, got {}",
            c.green()
        );
        assert!(
            (i32::from(c.blue()) - 64).abs() < 10,
            "Expected ~64 blue, got {}",
            c.blue()
        );
    }

    #[test]
    fn test_draw_vertices_triangle_strip() {
        // TriangleStrip with alternating red/green should interpolate
        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);
        let paint = Paint::new();

        // Strip: vertices 0,1,2 form first triangle; 1,2,3 form second
        canvas.draw_vertices(
            VertexMode::TriangleStrip,
            &[
                Point::new(10.0, 10.0),
                Point::new(50.0, 10.0),
                Point::new(10.0, 50.0),
                Point::new(50.0, 50.0),
            ],
            Some(&[
                Color::from_rgb(255, 0, 0),
                Color::from_rgb(0, 255, 0),
                Color::from_rgb(0, 0, 255),
                Color::from_rgb(255, 255, 0),
            ]),
            &paint,
        );
        drop(canvas);

        // First triangle (0,1,2): red, green, blue
        let c0 = buffer.get_pixel(15, 15).unwrap();
        assert!(
            c0.red() > 100,
            "Expected red contribution in first triangle"
        );

        // Second triangle (1,2,3): green, blue, yellow
        let c1 = buffer.get_pixel(45, 45).unwrap();
        assert!(
            c1.green() > 100 || c1.red() > 100,
            "Expected yellow-ish in second triangle"
        );
    }

    #[test]
    fn test_draw_vertices_triangle_fan() {
        // TriangleFan shares vertex 0 (center)
        let mut buffer = PixelBuffer::new(100, 100);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);
        let paint = Paint::new();

        canvas.draw_vertices(
            VertexMode::TriangleFan,
            &[
                Point::new(50.0, 50.0), // center - white
                Point::new(10.0, 10.0), // top-left - red
                Point::new(90.0, 10.0), // top-right - green
                Point::new(90.0, 90.0), // bottom-right - blue
            ],
            Some(&[
                Color::from_rgb(255, 255, 255),
                Color::from_rgb(255, 0, 0),
                Color::from_rgb(0, 255, 0),
                Color::from_rgb(0, 0, 255),
            ]),
            &paint,
        );
        drop(canvas);

        // Center should be bright (white vertex)
        let center = buffer.get_pixel(50, 50).unwrap();
        assert!(
            center.red() > 200 && center.green() > 200 && center.blue() > 200,
            "Center should be bright white"
        );
    }

    #[test]
    fn test_draw_patch_flat_square() {
        // 12 control points forming a flat 100x100 square patch
        let mut buffer = PixelBuffer::new(120, 120);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);

        let cubics = [
            // Top edge: left to right
            Point::new(10.0, 10.0),
            Point::new(40.0, 10.0),
            Point::new(70.0, 10.0),
            Point::new(100.0, 10.0),
            // Right edge: top to bottom
            Point::new(100.0, 40.0),
            Point::new(100.0, 70.0),
            Point::new(100.0, 100.0),
            // Bottom edge: right to left (reversed)
            Point::new(70.0, 100.0),
            Point::new(40.0, 100.0),
            Point::new(10.0, 100.0),
            // Left edge: bottom to top
            Point::new(10.0, 70.0),
            Point::new(10.0, 40.0),
        ];

        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 0, 0));

        canvas.draw_patch(
            &cubics,
            None,
            None,
            skia_rs_paint::BlendMode::SrcOver,
            &paint,
        );
        drop(canvas);

        // Center of the patch should be red
        let center = buffer.get_pixel(55, 55).unwrap();
        assert!(center.red() > 200, "Center should be red, got {center:?}");
    }

    #[test]
    fn test_draw_patch_with_corner_colors() {
        // Patch with interpolated corner colors
        let mut buffer = PixelBuffer::new(120, 120);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        let mut canvas = Canvas::new_raster(&mut buffer);

        let cubics = [
            Point::new(10.0, 10.0),
            Point::new(40.0, 10.0),
            Point::new(70.0, 10.0),
            Point::new(100.0, 10.0),
            Point::new(100.0, 40.0),
            Point::new(100.0, 70.0),
            Point::new(100.0, 100.0),
            Point::new(70.0, 100.0),
            Point::new(40.0, 100.0),
            Point::new(10.0, 100.0),
            Point::new(10.0, 70.0),
            Point::new(10.0, 40.0),
        ];

        // Corner colors: red, green, blue, yellow
        let colors = [
            Color::from_rgb(255, 0, 0),   // top-left: red
            Color::from_rgb(0, 255, 0),   // top-right: green
            Color::from_rgb(0, 0, 255),   // bottom-right: blue
            Color::from_rgb(255, 255, 0), // bottom-left: yellow
        ];

        let paint = Paint::new();
        canvas.draw_patch(
            &cubics,
            Some(&colors),
            None,
            skia_rs_paint::BlendMode::SrcOver,
            &paint,
        );
        drop(canvas);

        // Center should have a mix of all four colors
        let center = buffer.get_pixel(55, 55).unwrap();
        assert!(
            center.red() > 50 && center.green() > 50,
            "Center should blend all colors, got {center:?}"
        );
    }

    #[test]
    fn test_draw_annotation_noop_on_raster() {
        // Annotations should not panic or affect pixels
        let mut buffer = PixelBuffer::new(10, 10);
        buffer.clear(Color::from_argb(255, 100, 100, 100));
        let mut canvas = Canvas::new_raster(&mut buffer);

        canvas.draw_annotation(
            &Rect::from_xywh(0.0, 0.0, 5.0, 5.0),
            "url",
            b"https://example.com",
        );
        drop(canvas);

        // Pixels should be unchanged
        let pixel = buffer.get_pixel(1, 1).unwrap();
        assert_eq!(pixel.red(), 100);
        assert_eq!(pixel.green(), 100);
        assert_eq!(pixel.blue(), 100);
    }

    #[test]
    #[cfg(feature = "codec")]
    fn test_draw_image_lattice_single_cell() {
        // Lattice with one division in each axis creates a 2x2 grid of cells
        let mut buffer = PixelBuffer::new(40, 40);
        let mut canvas = Canvas::new_raster(&mut buffer);

        let image = skia_rs_codec::Image::from_color(10, 10, 0xFF_FF0000).unwrap();
        let lattice = ImageLattice::new(vec![5], vec![5]);
        canvas.draw_image_lattice(
            &image,
            &lattice,
            &Rect::from_xywh(0.0, 0.0, 40.0, 40.0),
            FilterMode::Nearest,
            None,
        );

        // Should draw something without panicking
    }

    #[test]
    #[cfg(feature = "codec")]
    fn test_draw_image_lattice_empty_divs() {
        // Lattice with no divs = single cell = full image
        let mut buffer = PixelBuffer::new(40, 40);
        let mut canvas = Canvas::new_raster(&mut buffer);

        let image = skia_rs_codec::Image::from_color(10, 10, 0xFF_00FF00).unwrap();
        let lattice = ImageLattice::new(vec![], vec![]);
        canvas.draw_image_lattice(
            &image,
            &lattice,
            &Rect::from_xywh(0.0, 0.0, 40.0, 40.0),
            FilterMode::Nearest,
            None,
        );

        // Should behave like draw_image_rect
    }

    #[test]
    #[cfg(feature = "codec")]
    fn test_draw_atlas_empty() {
        // Empty xforms/sprites should be a no-op
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);

        let image = skia_rs_codec::Image::from_color(100, 100, 0xFF_0000FF).unwrap();
        canvas.draw_atlas(
            &image,
            &[],
            &[],
            None,
            skia_rs_paint::BlendMode::SrcOver,
            FilterMode::Nearest,
            None,
        );

        // Should not panic
    }

    #[test]
    #[cfg(feature = "codec")]
    fn test_draw_atlas_single_sprite() {
        // Single sprite with translation
        let mut buffer = PixelBuffer::new(100, 100);
        let mut canvas = Canvas::new_raster(&mut buffer);

        let image = skia_rs_codec::Image::from_color(100, 100, 0xFF_FFFF00).unwrap();
        let xforms = [RSXform::from_scale_translate(1.0, 10.0, 10.0)];
        let sprites = [Rect::from_xywh(0.0, 0.0, 10.0, 10.0)];

        canvas.draw_atlas(
            &image,
            &xforms,
            &sprites,
            None,
            skia_rs_paint::BlendMode::SrcOver,
            FilterMode::Nearest,
            None,
        );

        // Should draw sprite at (10, 10) without panicking
    }

    // =========================================================================
    // T-1: Matrix stack operation tests
    // =========================================================================

    // =========================================================================
    // Conformance-audit regression tests (Task 4, canvas-level)
    // =========================================================================

    #[test]
    fn test_rsxform_to_matrix_translation_applied_last() {
        // SkMatrix::setRSXform layout: [scos, -ssin, tx; ssin, scos, ty].
        // tx/ty are the translation slots applied LAST: a 90-degree rotation
        // (scos=0, ssin=1) with translation (10, 20) maps (1, 0) to
        // (0*1 - 1*0 + 10, 1*1 + 0*0 + 20) = (10, 21).
        let xform = RSXform {
            scos: 0.0,
            ssin: 1.0,
            tx: 10.0,
            ty: 20.0,
        };
        let m = xform.to_matrix();
        let p = m.map_point(Point::new(1.0, 0.0));
        assert!((p.x - 10.0).abs() < 1e-4, "x: {}", p.x);
        assert!((p.y - 21.0).abs() < 1e-4, "y: {}", p.y);
    }

    #[test]
    fn test_draw_arc_full_sweep_draws_full_oval() {
        // |sweep| >= 360 fills the complete oval regardless of start angle
        // (SkPathPriv::CreateDrawArcPath).
        let mut buffer = PixelBuffer::new(60, 60);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        canvas.draw_arc(
            &Rect::from_xywh(10.0, 10.0, 40.0, 40.0),
            45.0,
            360.0,
            false,
            &paint,
        );
        drop(canvas);
        // All four oval extremes must be covered (a partial arc from 45 deg
        // would leave some of them empty).
        assert_eq!(buffer.get_pixel(30, 12).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(30, 47).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(12, 30).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(47, 30).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(30, 30).unwrap().red(), 255);
    }

    #[test]
    fn test_draw_round_rect_clamps_radii() {
        // SkRRect::setRectXY: radii exceeding half the rect dimensions are
        // scaled down uniformly. rx=ry=50 on a 20x20 rect clamps to 10:
        // the result is a circle inscribed in the rect.
        let mut buffer = PixelBuffer::new(60, 60);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 0, 255, 0));
        canvas.draw_round_rect(&Rect::from_xywh(20.0, 20.0, 20.0, 20.0), 50.0, 50.0, &paint);
        drop(canvas);
        // Center covered.
        assert_eq!(buffer.get_pixel(30, 30).unwrap().green(), 255);
        // Rect corner NOT covered (rounded away with radius 10).
        assert_eq!(buffer.get_pixel(21, 21).unwrap().alpha(), 0);
        // Edge midpoints covered.
        assert_eq!(buffer.get_pixel(30, 21).unwrap().green(), 255);
        assert_eq!(buffer.get_pixel(21, 30).unwrap().green(), 255);
    }

    #[test]
    fn test_clear_respects_device_clip() {
        // SkCanvas::clear is drawColor(color, kSrc): it only affects the
        // device clip.
        let mut buffer = PixelBuffer::new(20, 20);
        buffer.clear(Color::from_argb(255, 0, 0, 255)); // blue background
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.clip_rect(
            &Rect::from_xywh(0.0, 0.0, 10.0, 20.0),
            ClipOp::Intersect,
            false,
        );
        canvas.clear(Color::from_argb(255, 255, 0, 0)); // red
        drop(canvas);
        // Left half (inside clip): red.
        assert_eq!(buffer.get_pixel(5, 10).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(5, 10).unwrap().blue(), 0);
        // Right half (outside clip): still blue.
        assert_eq!(buffer.get_pixel(15, 10).unwrap().blue(), 255);
        assert_eq!(buffer.get_pixel(15, 10).unwrap().red(), 0);
    }

    #[test]
    fn test_draw_color_fills_clip_regardless_of_ctm() {
        // draw_color fills the device clip even under a wild CTM.
        let mut buffer = PixelBuffer::new(20, 20);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.rotate(37.0);
        canvas.translate(1000.0, -500.0);
        canvas.draw_color(
            Color::from_argb(255, 0, 255, 0),
            skia_rs_paint::BlendMode::SrcOver,
        );
        drop(canvas);
        for &(x, y) in &[(0, 0), (19, 19), (10, 10), (19, 0)] {
            assert_eq!(
                buffer.get_pixel(x, y).unwrap().green(),
                255,
                "({x}, {y}) must be filled regardless of the CTM"
            );
        }
    }

    #[test]
    fn test_draw_paint_fills_clip_with_shader_through_ctm() {
        use skia_rs_paint::shader::{TileMode, shaders};

        // draw_paint fills the clip; its shader is still transformed by the
        // CTM (local space). Gradient black->white over local x [0, 10],
        // CTM translate(10, 0): device x=10 is the black end.
        let shader = shaders::linear_gradient(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            vec![
                skia_rs_core::Color4f::new(0.0, 0.0, 0.0, 1.0),
                skia_rs_core::Color4f::new(1.0, 1.0, 1.0, 1.0),
            ],
            None,
            TileMode::Clamp,
        );
        let mut buffer = PixelBuffer::new(30, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.translate(10.0, 0.0);
        let mut paint = Paint::new();
        paint.set_shader(Some(shader));
        canvas.draw_paint(&paint);
        drop(canvas);
        // Whole clip is filled (even x < 10, where local x is negative and
        // the clamped gradient is black).
        assert_eq!(buffer.get_pixel(2, 5).unwrap().alpha(), 255);
        assert!(buffer.get_pixel(2, 5).unwrap().red() < 30);
        // Device x=10 -> local 0 (black); device x=19.5 -> local ~9.5 (bright).
        assert!(buffer.get_pixel(10, 5).unwrap().red() < 40);
        assert!(buffer.get_pixel(19, 5).unwrap().red() > 200);
    }

    #[test]
    fn test_clip_rect_rotated_ctm_clips_actual_quad() {
        // clip_rect under a rotated CTM must clip to the rotated quad, not
        // its AABB.
        let mut buffer = PixelBuffer::new(100, 100);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.translate(50.0, 50.0);
        canvas.rotate(45.0);
        canvas.translate(-50.0, -50.0);
        canvas.clip_rect(
            &Rect::from_xywh(20.0, 20.0, 60.0, 60.0),
            ClipOp::Intersect,
            false,
        );
        // Draw a full-canvas rect in device space.
        canvas.reset_matrix();
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0), &paint);
        drop(canvas);
        // Center of the rotated clip: filled.
        assert_eq!(buffer.get_pixel(50, 50).unwrap().red(), 255);
        // AABB corner outside the rotated quad: clipped.
        assert_eq!(
            buffer.get_pixel(25, 25).unwrap().alpha(),
            0,
            "AABB corner outside the rotated clip quad must stay clipped"
        );
        // Quad vertex direction: inside.
        assert_eq!(buffer.get_pixel(50, 15).unwrap().red(), 255);
    }

    #[test]
    fn test_draw_points_points_mode_draws_width_sized_squares() {
        // SkDraw::drawPoints: Points mode with stroke width w draws a w x w
        // square (butt/square cap) centered on each point.
        let mut buffer = PixelBuffer::new(40, 40);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        paint.set_style(Style::Stroke);
        paint.set_stroke_width(10.0);
        canvas.draw_points(PointMode::Points, &[Point::new(20.0, 20.0)], &paint);
        drop(canvas);
        // 10x10 square centered at (20, 20): [15, 25).
        assert_eq!(buffer.get_pixel(16, 16).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(24, 24).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(20, 20).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(13, 20).unwrap().alpha(), 0);
        assert_eq!(buffer.get_pixel(27, 20).unwrap().alpha(), 0);
    }

    #[test]
    fn test_draw_points_points_mode_round_cap_draws_circles() {
        let mut buffer = PixelBuffer::new(40, 40);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 255, 0, 0));
        paint.set_style(Style::Stroke);
        paint.set_stroke_width(10.0);
        paint.set_stroke_cap(skia_rs_paint::StrokeCap::Round);
        canvas.draw_points(PointMode::Points, &[Point::new(20.0, 20.0)], &paint);
        drop(canvas);
        // Circle radius 5 centered at (20, 20): center and axis points in.
        assert_eq!(buffer.get_pixel(20, 20).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(24, 20).unwrap().red(), 255);
        // The square's corner pixel (15, 15) has center distance ~6.4 from
        // (20, 20): inside the 10x10 square but outside the circle.
        assert_eq!(
            buffer.get_pixel(15, 15).unwrap().alpha(),
            0,
            "round cap points are circles, not squares"
        );
    }

    #[test]
    fn test_save_layer_bounds_are_local_space_hint() {
        // SkCanvas::saveLayer bounds are a content hint in LOCAL space:
        // with CTM translate(10, 10) and bounds (0,0,4,4), content drawn at
        // local (0..4) lands at device (10..14) after restore.
        let mut buffer = PixelBuffer::new(20, 20);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.translate(10.0, 10.0);
        let bounds = Rect::from_xywh(0.0, 0.0, 4.0, 4.0);
        let rec = SaveLayerRec {
            bounds: Some(&bounds),
            paint: None,
            flags: SaveLayerFlags::NONE,
        };
        canvas.save_layer(&rec);
        let mut red = Paint::new();
        red.set_color32(Color::from_argb(255, 255, 0, 0));
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &red);
        canvas.restore();
        drop(canvas);

        // Device (10..14): red.
        assert_eq!(buffer.get_pixel(11, 11).unwrap().red(), 255);
        assert_eq!(buffer.get_pixel(13, 13).unwrap().red(), 255);
        // Device (0..4): untouched — the bounds are local, not device.
        assert_eq!(
            buffer.get_pixel(1, 1).unwrap().alpha(),
            0,
            "layer bounds must be mapped through the CTM"
        );
    }

    #[test]
    fn test_composite_layer_through_clip() {
        // A clip narrower than the layer must clip the composite: clearing
        // inside a layer touches the whole layer buffer, but only the
        // clipped part reaches the base on restore.
        let mut buffer = PixelBuffer::new(20, 20);
        buffer.clear(Color::from_argb(255, 0, 0, 255)); // blue base
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.clip_rect(
            &Rect::from_xywh(0.0, 0.0, 10.0, 20.0),
            ClipOp::Intersect,
            false,
        );
        canvas.save_layer(&SaveLayerRec::default());
        canvas.clear(Color::from_argb(255, 255, 0, 0)); // red into the layer
        canvas.restore();
        drop(canvas);

        // Inside the clip: red composed onto the base.
        assert_eq!(buffer.get_pixel(5, 10).unwrap().red(), 255);
        // Outside the clip: base untouched.
        assert_eq!(buffer.get_pixel(15, 10).unwrap().blue(), 255);
        assert_eq!(buffer.get_pixel(15, 10).unwrap().red(), 0);
    }

    #[test]
    fn test_composite_layer_applies_color_filter() {
        use skia_rs_paint::filter::ColorMatrixFilter;
        use std::sync::Arc;

        // A layer paint with an inverting color filter must filter the
        // layer contents during composite: white -> black.
        let mut buffer = PixelBuffer::new(8, 8);
        buffer.clear(Color::from_argb(255, 0, 0, 255)); // blue base
        let mut canvas = Canvas::new_raster(&mut buffer);
        let mut layer_paint = Paint::new();
        layer_paint.set_color_filter(Some(Arc::new(ColorMatrixFilter::invert())));
        let rec = SaveLayerRec {
            bounds: None,
            paint: Some(&layer_paint),
            flags: SaveLayerFlags::NONE,
        };
        canvas.save_layer(&rec);
        let mut white = Paint::new();
        white.set_color32(Color::from_argb(255, 255, 255, 255));
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 8.0, 8.0), &white);
        canvas.restore();
        drop(canvas);

        let c = buffer.get_pixel(4, 4).unwrap();
        assert_eq!(
            (c.red(), c.green(), c.blue()),
            (0, 0, 0),
            "invert color filter must run at composite: {c:?}"
        );
        assert_eq!(c.alpha(), 255);
    }

    #[test]
    fn test_draw_vertices_respects_clip_and_paint_alpha() {
        // draw_vertices must clip per pixel and apply paint alpha to vertex
        // colors.
        let mut buffer = PixelBuffer::new(20, 20);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.clip_rect(
            &Rect::from_xywh(0.0, 0.0, 10.0, 20.0),
            ClipOp::Intersect,
            false,
        );
        let mut paint = Paint::new();
        paint.set_alpha(0.5);
        let positions = [
            Point::new(0.0, 0.0),
            Point::new(20.0, 0.0),
            Point::new(0.0, 20.0),
        ];
        let colors = [Color::WHITE, Color::WHITE, Color::WHITE];
        canvas.draw_vertices(VertexMode::Triangles, &positions, Some(&colors), &paint);
        drop(canvas);

        // Inside triangle and clip: white at 50% alpha, premul => ~128.
        let inside = buffer.get_pixel(3, 3).unwrap();
        assert!(
            inside.alpha() > 120 && inside.alpha() < 136,
            "paint alpha must modulate vertex colors, got {inside:?}"
        );
        assert_eq!(inside.red(), inside.alpha(), "premultiplied white");
        // Inside triangle but outside the clip: untouched.
        assert_eq!(
            buffer.get_pixel(14, 2).unwrap().alpha(),
            0,
            "vertices must be clipped per pixel"
        );
    }

    #[cfg(feature = "codec")]
    #[test]
    fn test_draw_image_rect_applies_alpha_once_and_clips() {
        // A 50%-alpha paint on an opaque white image must produce ~50%
        // contribution (128), not 25% (64, alpha applied twice), and the
        // draw must respect the clip.
        let info = skia_rs_codec::ImageInfo::new(
            4,
            4,
            skia_rs_core::ColorType::Rgba8888,
            skia_rs_core::AlphaType::Premul,
        );
        let pixels = vec![255u8; 4 * 4 * 4]; // opaque white
        let image = skia_rs_codec::Image::from_raster_data_owned(info, pixels, 16).unwrap();

        let mut buffer = PixelBuffer::new(20, 20);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.clip_rect(
            &Rect::from_xywh(0.0, 0.0, 10.0, 20.0),
            ClipOp::Intersect,
            false,
        );
        let mut paint = Paint::new();
        paint.set_alpha(0.5);
        canvas.draw_image_rect(
            &image,
            None,
            &Rect::from_xywh(0.0, 0.0, 16.0, 16.0),
            Some(&paint),
        );
        drop(canvas);

        let c = buffer.get_pixel(5, 5).unwrap();
        assert!(
            c.alpha() > 120 && c.alpha() < 136,
            "paint alpha must be applied exactly once, got {c:?}"
        );
        // Outside the clip: untouched.
        assert_eq!(buffer.get_pixel(12, 5).unwrap().alpha(), 0);
    }

    #[test]
    fn test_save_count_increments_and_restore_decrements() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let initial = canvas.save_count();
        assert_eq!(initial, 1);
        canvas.save();
        assert_eq!(canvas.save_count(), initial + 1);
        canvas.save();
        assert_eq!(canvas.save_count(), initial + 2);
        canvas.restore();
        assert_eq!(canvas.save_count(), initial + 1);
        canvas.restore();
        assert_eq!(canvas.save_count(), initial);
    }

    #[test]
    fn test_restore_to_count_pops_multiple_levels() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        let base = canvas.save_count();
        canvas.save();
        canvas.save();
        canvas.save();
        assert_eq!(canvas.save_count(), base + 3);
        canvas.restore_to_count(base);
        assert_eq!(canvas.save_count(), base);
    }

    #[test]
    fn test_restore_to_count_clamps_below_one() {
        // SkCanvas::restoreToCount clamps count < 1 to 1 — must terminate and
        // leave the canvas at save count 1 (previously infinite-looped).
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.save();
        canvas.save();
        canvas.restore_to_count(0);
        assert_eq!(canvas.save_count(), 1);
        // And again with an already-minimal stack: still terminates.
        canvas.restore_to_count(0);
        assert_eq!(canvas.save_count(), 1);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test: identity-matrix mapping of (0,0) after undoing a translate is exactly 0.0"
    )]
    fn test_save_returns_previous_save_count() {
        // SkCanvas::save returns getSaveCount() - 1, i.e. the count BEFORE
        // the save. Fresh canvas: getSaveCount() == 1, save() returns 1 and
        // the count becomes 2. restore_to_count(returned) undoes that save.
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        assert_eq!(canvas.save_count(), 1);
        let ret = canvas.save();
        assert_eq!(ret, 1, "save() must return the pre-save count");
        assert_eq!(canvas.save_count(), 2);
        canvas.translate(3.0, 0.0);
        canvas.restore_to_count(ret);
        assert_eq!(canvas.save_count(), 1);
        // The translate was undone by restoring to the returned count.
        let p = canvas.total_matrix().map_point(Point::new(0.0, 0.0));
        assert_eq!(p.x, 0.0);
    }

    #[test]
    fn test_translate_accumulates() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.translate(5.0, 10.0);
        canvas.translate(3.0, 2.0);
        let m = canvas.total_matrix();
        let p = m.map_point(Point::new(0.0, 0.0));
        assert!((p.x - 8.0).abs() < 1e-4, "expected tx=8, got {}", p.x);
        assert!((p.y - 12.0).abs() < 1e-4, "expected ty=12, got {}", p.y);
    }

    #[test]
    fn test_save_then_translate_then_restore_undoes() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.save();
        canvas.translate(10.0, 10.0);
        canvas.restore();
        let p = canvas.total_matrix().map_point(Point::new(0.0, 0.0));
        assert!(
            p.x.abs() < 1e-4 && p.y.abs() < 1e-4,
            "restore should undo translate"
        );
    }

    #[test]
    fn test_scale_accumulates() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.scale(2.0, 3.0);
        let p = canvas.total_matrix().map_point(Point::new(5.0, 5.0));
        assert!((p.x - 10.0).abs() < 1e-4, "expected x scaled by 2");
        assert!((p.y - 15.0).abs() < 1e-4, "expected y scaled by 3");
    }

    #[test]
    fn test_rotate_90_degrees() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.rotate(90.0);
        let p = canvas.total_matrix().map_point(Point::new(1.0, 0.0));
        // (1, 0) rotated 90° CCW → (0, 1)
        assert!(p.x.abs() < 1e-3, "x should be ~0, got {}", p.x);
        assert!((p.y - 1.0).abs() < 1e-3, "y should be ~1, got {}", p.y);
    }

    #[test]
    fn test_reset_matrix_clears_all_transforms() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.translate(100.0, 100.0);
        canvas.scale(2.0, 2.0);
        canvas.reset_matrix();
        let m = canvas.total_matrix();
        assert!(m.is_identity(), "reset_matrix should produce identity");
        let p = m.map_point(Point::new(5.0, 5.0));
        assert!((p.x - 5.0).abs() < 1e-4);
        assert!((p.y - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_concat_composes_matrices() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.concat(&Matrix::translate(3.0, 4.0));
        let p = canvas.total_matrix().map_point(Point::new(1.0, 1.0));
        assert!((p.x - 4.0).abs() < 1e-4, "expected p.x = 1+3 = 4");
        assert!((p.y - 5.0).abs() < 1e-4, "expected p.y = 1+4 = 5");
    }

    #[test]
    fn test_set_matrix_replaces_ctm() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        canvas.translate(100.0, 100.0);
        canvas.set_matrix(&Matrix::scale(2.0, 2.0));
        let p = canvas.total_matrix().map_point(Point::new(5.0, 5.0));
        // translate was discarded, only scale remains
        assert!((p.x - 10.0).abs() < 1e-4);
        assert!((p.y - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_total_matrix_reflects_current_transform() {
        let mut buffer = PixelBuffer::new(10, 10);
        let mut canvas = Canvas::new_raster(&mut buffer);
        assert!(canvas.total_matrix().is_identity());
        canvas.translate(5.0, 10.0);
        let m = canvas.total_matrix();
        let p = m.map_point(Point::new(0.0, 0.0));
        assert!((p.x - 5.0).abs() < 1e-4);
        assert!((p.y - 10.0).abs() < 1e-4);
    }

    // =========================================================================
    // T-2: SaveLayerRec and SaveLayerFlags tests
    // =========================================================================

    #[test]
    fn test_save_layer_rec_default_has_none_fields() {
        let rec = SaveLayerRec::default();
        assert!(rec.bounds.is_none());
        assert!(rec.paint.is_none());
    }

    #[test]
    fn test_save_layer_flags_none() {
        let flags = SaveLayerFlags::NONE;
        assert_eq!(flags.0, 0);
    }

    #[test]
    fn test_save_layer_flags_preserve_lcd_text() {
        let flags = SaveLayerFlags::PRESERVE_LCD_TEXT;
        assert_ne!(flags.0, 0);
    }

    #[test]
    fn test_save_layer_flags_init_with_previous() {
        let flags = SaveLayerFlags::INIT_WITH_PREVIOUS;
        assert_ne!(flags.0, 0);
    }

    // =========================================================================
    // T-3: Supporting types tests
    // =========================================================================

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact test: from_scale_translate(1.0, 0.0, 0.0) produces exact field values"
    )]
    fn test_rsxform_identity() {
        let xform = RSXform::from_scale_translate(1.0, 0.0, 0.0);
        assert_eq!(xform.scos, 1.0);
        assert_eq!(xform.ssin, 0.0);
        assert_eq!(xform.tx, 0.0);
        assert_eq!(xform.ty, 0.0);
    }

    #[test]
    fn test_rsxform_translation() {
        let xform = RSXform::from_scale_translate(1.0, 10.0, 20.0);
        let m = xform.to_matrix();
        let p = m.map_point(Point::new(0.0, 0.0));
        assert!((p.x - 10.0).abs() < 1e-4, "expected tx=10");
        assert!((p.y - 20.0).abs() < 1e-4, "expected ty=20");
    }

    #[test]
    fn test_rsxform_from_radians_identity() {
        let xform = RSXform::from_radians(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let m = xform.to_matrix();
        let p = m.map_point(Point::new(5.0, 5.0));
        assert!((p.x - 5.0).abs() < 1e-4, "identity should not transform");
        assert!((p.y - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_rsxform_from_radians_90_degrees() {
        use std::f32::consts::FRAC_PI_2;
        let xform = RSXform::from_radians(1.0, FRAC_PI_2, 0.0, 0.0, 0.0, 0.0);
        let m = xform.to_matrix();
        let p = m.map_point(Point::new(1.0, 0.0));
        // (1, 0) rotated 90° → (0, 1)
        assert!(p.x.abs() < 1e-3, "x should be ~0");
        assert!((p.y - 1.0).abs() < 1e-3, "y should be ~1");
    }

    #[test]
    fn test_rsxform_with_scale() {
        let xform = RSXform::from_scale_translate(2.0, 0.0, 0.0);
        let m = xform.to_matrix();
        let p = m.map_point(Point::new(5.0, 5.0));
        assert!((p.x - 10.0).abs() < 1e-4, "x should be scaled by 2");
        assert!((p.y - 10.0).abs() < 1e-4, "y should be scaled by 2");
    }

    #[test]
    fn test_filter_mode_variants() {
        let _ = FilterMode::Nearest;
        let _ = FilterMode::Linear;
        assert_eq!(FilterMode::default(), FilterMode::Nearest);
    }

    #[test]
    fn test_point_mode_variants() {
        let _ = PointMode::Points;
        let _ = PointMode::Lines;
        let _ = PointMode::Polygon;
    }
}
