//! Picture recording and playback.
//!
//! Pictures are display lists that record drawing commands for later playback.
//! This is useful for caching complex drawings, serialization, and deferred rendering.

use crate::Canvas;
use skia_rs_core::{Color, Matrix, Point, Rect, Scalar};
use skia_rs_paint::{BlendMode, Paint};
use skia_rs_path::Path;
use std::sync::Arc;

/// A recorded picture that can be played back to a canvas.
///
/// Corresponds to Skia's `SkPicture`.
#[derive(Debug, Clone)]
pub struct Picture {
    /// The recorded drawing commands.
    commands: Vec<DrawCommand>,
    /// Bounding box of the picture.
    cull_rect: Rect,
}

impl Picture {
    /// Create a new picture from recorded commands.
    pub(crate) const fn new(commands: Vec<DrawCommand>, cull_rect: Rect) -> Self {
        Self {
            commands,
            cull_rect,
        }
    }

    /// Create a new picture from a caller-supplied command stream.
    ///
    /// Used by callers (notably the FFI layer) that compose their own
    /// recording pipeline rather than going through [`PictureRecorder`].
    /// The returned picture can be [`Self::playback`]-ed like any other.
    #[must_use]
    pub const fn from_parts(commands: Vec<DrawCommand>, cull_rect: Rect) -> Self {
        Self::new(commands, cull_rect)
    }

    /// Get the cull rect (bounding box).
    #[inline]
    #[must_use]
    pub const fn cull_rect(&self) -> Rect {
        self.cull_rect
    }

    /// Play the picture back to a canvas.
    ///
    /// The canvas's CTM at the start of playback is captured so that recorded
    /// `SetMatrix` commands compose with it (`setMatrix(initialCTM * recorded)`,
    /// per `SkCanvasPriv`/`SkPicturePlayback`) instead of clobbering it.
    pub fn playback(&self, canvas: &mut Canvas) {
        let initial = *canvas.total_matrix();
        for command in &self.commands {
            command.execute_with_base(canvas, &initial);
        }
    }

    /// Get the approximate byte size of this picture.
    #[must_use]
    pub fn approximate_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>() + self.commands.len() * std::mem::size_of::<DrawCommand>()
    }

    /// Get the number of operations in this picture.
    #[must_use]
    pub fn approximate_op_count(&self) -> usize {
        self.commands.len()
    }
}

/// A picture reference (shared ownership).
pub type PictureRef = Arc<Picture>;

/// A recorded drawing command.
#[derive(Debug, Clone)]
pub enum DrawCommand {
    /// Save the canvas state.
    Save,
    /// Restore the canvas state.
    Restore,
    /// Save with a layer.
    SaveLayer {
        /// Bounds for the layer.
        bounds: Option<Rect>,
        /// Paint for the layer.
        paint: Option<Paint>,
    },
    /// Translate the canvas.
    Translate {
        /// X translation.
        dx: Scalar,
        /// Y translation.
        dy: Scalar,
    },
    /// Scale the canvas.
    Scale {
        /// X scale.
        sx: Scalar,
        /// Y scale.
        sy: Scalar,
    },
    /// Rotate the canvas.
    Rotate {
        /// Rotation in degrees.
        degrees: Scalar,
    },
    /// Skew the canvas.
    Skew {
        /// X skew.
        sx: Scalar,
        /// Y skew.
        sy: Scalar,
    },
    /// Concatenate a matrix.
    Concat {
        /// The matrix to concatenate.
        matrix: Matrix,
    },
    /// Set the matrix.
    SetMatrix {
        /// The matrix to set.
        matrix: Matrix,
    },
    /// Clip to a rectangle.
    ClipRect {
        /// The rectangle to clip to.
        rect: Rect,
        /// The clip operation (Intersect or Difference).
        op: crate::ClipOp,
        /// Whether to antialias.
        anti_alias: bool,
    },
    /// Clip to a path.
    ClipPath {
        /// The path to clip to.
        path: Path,
        /// The clip operation (Intersect or Difference).
        op: crate::ClipOp,
        /// Whether to antialias.
        anti_alias: bool,
    },
    /// Clip to a rounded rectangle.
    ClipRoundRect {
        /// The rounded rectangle bounds.
        rect: Rect,
        /// The x radius.
        rx: Scalar,
        /// The y radius.
        ry: Scalar,
        /// The clip operation.
        op: crate::ClipOp,
        /// Whether to antialias.
        anti_alias: bool,
    },
    /// Clear the canvas.
    Clear {
        /// The color to clear with.
        color: Color,
    },
    /// Draw a color.
    DrawColor {
        /// The color to draw.
        color: Color,
        /// The blend mode.
        blend_mode: BlendMode,
    },
    /// Fill the device clip with a full paint (`SkCanvas::drawPaint`).
    DrawPaint {
        /// The paint to fill with.
        paint: Paint,
    },
    /// Draw a point.
    DrawPoint {
        /// The point to draw.
        point: Point,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw a line.
    DrawLine {
        /// Start point.
        p0: Point,
        /// End point.
        p1: Point,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw a rectangle.
    DrawRect {
        /// The rectangle to draw.
        rect: Rect,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw an oval.
    DrawOval {
        /// The bounding rectangle of the oval.
        rect: Rect,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw a circle.
    DrawCircle {
        /// Center of the circle.
        center: Point,
        /// Radius of the circle.
        radius: Scalar,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw an arc.
    DrawArc {
        /// The bounding oval.
        oval: Rect,
        /// Start angle in degrees.
        start_angle: Scalar,
        /// Sweep angle in degrees.
        sweep_angle: Scalar,
        /// Whether to draw lines to center.
        use_center: bool,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw a rounded rectangle.
    DrawRoundRect {
        /// The rectangle.
        rect: Rect,
        /// X radius.
        rx: Scalar,
        /// Y radius.
        ry: Scalar,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw a path.
    DrawPath {
        /// The path to draw.
        path: Path,
        /// The paint to use.
        paint: Paint,
    },
    /// Draw another picture.
    DrawPicture {
        /// The picture to draw.
        picture: PictureRef,
        /// Optional matrix to apply.
        matrix: Option<Matrix>,
        /// Optional paint to apply.
        paint: Option<Paint>,
    },
}

impl DrawCommand {
    /// Execute this command on a canvas.
    ///
    /// For a single-command replay the canvas's current CTM is the playback
    /// start, so recorded `SetMatrix` composes with it. Multi-command playback
    /// ([`Picture::playback`]) captures the CTM once up front instead.
    pub fn execute(&self, canvas: &mut Canvas) {
        let base = *canvas.total_matrix();
        self.execute_with_base(canvas, &base);
    }

    /// Execute this command with `base` as the CTM captured at playback start.
    #[allow(
        clippy::too_many_lines,
        reason = "single exhaustive dispatch match mirroring SkPicturePlayback::draw; splitting would obscure the 1:1 command mapping"
    )]
    fn execute_with_base(&self, canvas: &mut Canvas, base: &Matrix) {
        match self {
            Self::Save => {
                canvas.save();
            }
            Self::Restore => {
                canvas.restore();
            }
            Self::SaveLayer { bounds, paint } => {
                let rec = crate::SaveLayerRec {
                    bounds: bounds.as_ref(),
                    paint: paint.as_ref(),
                    flags: crate::SaveLayerFlags::NONE,
                };
                canvas.save_layer(&rec);
            }
            Self::Translate { dx, dy } => {
                canvas.translate(*dx, *dy);
            }
            Self::Scale { sx, sy } => {
                canvas.scale(*sx, *sy);
            }
            Self::Rotate { degrees } => {
                canvas.rotate(*degrees);
            }
            Self::Skew { sx, sy } => {
                canvas.skew(*sx, *sy);
            }
            Self::Concat { matrix } => {
                canvas.concat(matrix);
            }
            Self::SetMatrix { matrix } => {
                // Compose with the CTM at playback start:
                // setMatrix(initialCTM * recorded).
                canvas.set_matrix(&base.concat(matrix));
            }
            Self::ClipRect {
                rect,
                op,
                anti_alias,
            } => {
                canvas.clip_rect(rect, *op, *anti_alias);
            }
            Self::ClipPath {
                path,
                op,
                anti_alias,
            } => {
                canvas.clip_path(path, *op, *anti_alias);
            }
            Self::ClipRoundRect {
                rect,
                rx,
                ry,
                op,
                anti_alias,
            } => {
                canvas.clip_round_rect(rect, *rx, *ry, *op, *anti_alias);
            }
            Self::Clear { color } => {
                canvas.clear(*color);
            }
            Self::DrawColor { color, blend_mode } => {
                canvas.draw_color(*color, *blend_mode);
            }
            Self::DrawPaint { paint } => {
                canvas.draw_paint(paint);
            }
            Self::DrawPoint { point, paint } => {
                canvas.draw_point(*point, paint);
            }
            Self::DrawLine { p0, p1, paint } => {
                canvas.draw_line(*p0, *p1, paint);
            }
            Self::DrawRect { rect, paint } => {
                canvas.draw_rect(rect, paint);
            }
            Self::DrawOval { rect, paint } => {
                canvas.draw_oval(rect, paint);
            }
            Self::DrawCircle {
                center,
                radius,
                paint,
            } => {
                canvas.draw_circle(*center, *radius, paint);
            }
            Self::DrawArc {
                oval,
                start_angle,
                sweep_angle,
                use_center,
                paint,
            } => {
                canvas.draw_arc(oval, *start_angle, *sweep_angle, *use_center, paint);
            }
            Self::DrawRoundRect {
                rect,
                rx,
                ry,
                paint,
            } => {
                canvas.draw_round_rect(rect, *rx, *ry, paint);
            }
            Self::DrawPath { path, paint } => {
                canvas.draw_path(path, paint);
            }
            Self::DrawPicture {
                picture,
                matrix,
                paint,
            } => {
                canvas.draw_picture(picture, matrix.as_ref(), paint.as_ref());
            }
        }
    }
}

/// A recorder that captures drawing commands into a Picture.
///
/// Corresponds to Skia's `SkPictureRecorder`.
pub struct PictureRecorder {
    commands: Vec<DrawCommand>,
    cull_rect: Rect,
    is_recording: bool,
}

impl Default for PictureRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl PictureRecorder {
    /// Create a new picture recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            commands: Vec::new(),
            cull_rect: Rect::EMPTY,
            is_recording: false,
        }
    }

    /// Begin recording with a cull rect.
    pub fn begin_recording(&mut self, cull_rect: Rect) -> &mut RecordingCanvas {
        self.commands.clear();
        self.cull_rect = cull_rect;
        self.is_recording = true;
        // Safety: we're returning a reference that allows recording
        unsafe { &mut *std::ptr::from_mut(self).cast::<RecordingCanvas>() }
    }

    /// Finish recording and return the picture.
    #[must_use]
    pub fn finish_recording(&mut self) -> Option<PictureRef> {
        if !self.is_recording {
            return None;
        }
        self.is_recording = false;
        let commands = std::mem::take(&mut self.commands);
        Some(Arc::new(Picture::new(commands, self.cull_rect)))
    }

    /// Check if currently recording.
    #[inline]
    #[must_use]
    pub const fn is_recording(&self) -> bool {
        self.is_recording
    }
}

/// A canvas that records drawing commands.
///
/// This is actually a [`PictureRecorder`] with a canvas-like interface.
#[repr(transparent)]
pub struct RecordingCanvas {
    inner: PictureRecorder,
}

impl RecordingCanvas {
    /// Current save count implied by the recorded command stream.
    ///
    /// Mirrors `SkCanvas::getSaveCount`: starts at 1, +1 per Save/SaveLayer,
    /// -1 per Restore (never below 1).
    fn save_count(&self) -> usize {
        let mut count = 1usize;
        for cmd in &self.inner.commands {
            match cmd {
                DrawCommand::Save | DrawCommand::SaveLayer { .. } => count += 1,
                DrawCommand::Restore => count = count.saturating_sub(1).max(1),
                _ => {}
            }
        }
        count
    }

    /// Record a save command.
    ///
    /// Returns the save count **before** the save, mirroring the fixed
    /// [`Canvas::save`](crate::Canvas::save) / `SkCanvas::save` semantics.
    pub fn save(&mut self) -> usize {
        let prev = self.save_count();
        self.inner.commands.push(DrawCommand::Save);
        prev
    }

    /// Record a restore command.
    pub fn restore(&mut self) {
        self.inner.commands.push(DrawCommand::Restore);
    }

    /// Record a save layer command.
    pub fn save_layer(&mut self, bounds: Option<Rect>, paint: Option<&Paint>) {
        self.inner.commands.push(DrawCommand::SaveLayer {
            bounds,
            paint: paint.cloned(),
        });
    }

    /// Record a translate command.
    pub fn translate(&mut self, dx: Scalar, dy: Scalar) {
        self.inner.commands.push(DrawCommand::Translate { dx, dy });
    }

    /// Record a scale command.
    pub fn scale(&mut self, sx: Scalar, sy: Scalar) {
        self.inner.commands.push(DrawCommand::Scale { sx, sy });
    }

    /// Record a rotate command.
    pub fn rotate(&mut self, degrees: Scalar) {
        self.inner.commands.push(DrawCommand::Rotate { degrees });
    }

    /// Record a skew command.
    pub fn skew(&mut self, sx: Scalar, sy: Scalar) {
        self.inner.commands.push(DrawCommand::Skew { sx, sy });
    }

    /// Record a concat command.
    pub fn concat(&mut self, matrix: &Matrix) {
        self.inner
            .commands
            .push(DrawCommand::Concat { matrix: *matrix });
    }

    /// Record a set matrix command.
    pub fn set_matrix(&mut self, matrix: &Matrix) {
        self.inner
            .commands
            .push(DrawCommand::SetMatrix { matrix: *matrix });
    }

    /// Record a clip rect command (Intersect).
    pub fn clip_rect(&mut self, rect: &Rect, anti_alias: bool) {
        self.clip_rect_op(rect, crate::ClipOp::Intersect, anti_alias);
    }

    /// Record a clip rect command with an explicit clip op.
    pub fn clip_rect_op(&mut self, rect: &Rect, op: crate::ClipOp, anti_alias: bool) {
        self.inner.commands.push(DrawCommand::ClipRect {
            rect: *rect,
            op,
            anti_alias,
        });
    }

    /// Record a clip path command (Intersect).
    pub fn clip_path(&mut self, path: &Path, anti_alias: bool) {
        self.clip_path_op(path, crate::ClipOp::Intersect, anti_alias);
    }

    /// Record a clip path command with an explicit clip op.
    pub fn clip_path_op(&mut self, path: &Path, op: crate::ClipOp, anti_alias: bool) {
        self.inner.commands.push(DrawCommand::ClipPath {
            path: path.clone(),
            op,
            anti_alias,
        });
    }

    /// Record a rounded rectangle clip command.
    pub fn clip_round_rect(&mut self, rect: &Rect, rx: Scalar, ry: Scalar, anti_alias: bool) {
        self.clip_round_rect_op(rect, rx, ry, crate::ClipOp::Intersect, anti_alias);
    }

    /// Record a rounded rectangle clip command with an explicit clip op.
    pub fn clip_round_rect_op(
        &mut self,
        rect: &Rect,
        rx: Scalar,
        ry: Scalar,
        op: crate::ClipOp,
        anti_alias: bool,
    ) {
        self.inner.commands.push(DrawCommand::ClipRoundRect {
            rect: *rect,
            rx,
            ry,
            op,
            anti_alias,
        });
    }

    /// Record a clear command.
    pub fn clear(&mut self, color: Color) {
        self.inner.commands.push(DrawCommand::Clear { color });
    }

    /// Record a draw color command.
    pub fn draw_color(&mut self, color: Color, blend_mode: BlendMode) {
        self.inner
            .commands
            .push(DrawCommand::DrawColor { color, blend_mode });
    }

    /// Record a draw point command.
    pub fn draw_point(&mut self, pt: Point, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawPoint {
            point: pt,
            paint: paint.clone(),
        });
    }

    /// Record a draw line command.
    pub fn draw_line(&mut self, p0: Point, p1: Point, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawLine {
            p0,
            p1,
            paint: paint.clone(),
        });
    }

    /// Record a draw rect command.
    pub fn draw_rect(&mut self, rect: &Rect, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawRect {
            rect: *rect,
            paint: paint.clone(),
        });
    }

    /// Record a draw oval command.
    pub fn draw_oval(&mut self, rect: &Rect, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawOval {
            rect: *rect,
            paint: paint.clone(),
        });
    }

    /// Record a draw circle command.
    pub fn draw_circle(&mut self, center: Point, radius: Scalar, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawCircle {
            center,
            radius,
            paint: paint.clone(),
        });
    }

    /// Record a draw arc command.
    pub fn draw_arc(
        &mut self,
        oval: &Rect,
        start_angle: Scalar,
        sweep_angle: Scalar,
        use_center: bool,
        paint: &Paint,
    ) {
        self.inner.commands.push(DrawCommand::DrawArc {
            oval: *oval,
            start_angle,
            sweep_angle,
            use_center,
            paint: paint.clone(),
        });
    }

    /// Record a draw round rect command.
    pub fn draw_round_rect(&mut self, rect: &Rect, rx: Scalar, ry: Scalar, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawRoundRect {
            rect: *rect,
            rx,
            ry,
            paint: paint.clone(),
        });
    }

    /// Record a draw path command.
    pub fn draw_path(&mut self, path: &Path, paint: &Paint) {
        self.inner.commands.push(DrawCommand::DrawPath {
            path: path.clone(),
            paint: paint.clone(),
        });
    }

    /// Record a draw picture command.
    pub fn draw_picture(
        &mut self,
        picture: &PictureRef,
        matrix: Option<&Matrix>,
        paint: Option<&Paint>,
    ) {
        self.inner.commands.push(DrawCommand::DrawPicture {
            picture: picture.clone(),
            matrix: matrix.copied(),
            paint: paint.cloned(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::PixelBuffer;

    #[test]
    fn test_clip_op_recorded_and_replayed() {
        // A Difference clip recorded through Canvas must replay as Difference,
        // not silently as Intersect.
        let mut commands = Vec::new();
        {
            let mut rec = Canvas::new_recording(&mut commands, 10, 10);
            rec.clip_rect(
                &Rect::from_xywh(2.0, 2.0, 6.0, 6.0),
                crate::ClipOp::Difference,
                false,
            );
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(255, 255, 0, 0));
            rec.draw_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0), &paint);
        }
        let picture = Picture::new(commands, Rect::from_xywh(0.0, 0.0, 10.0, 10.0));

        let mut buffer = PixelBuffer::new(10, 10);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            picture.playback(&mut canvas);
        }
        // Inside the difference rect: untouched (transparent).
        assert_eq!(buffer.get_pixel(5, 5).unwrap().alpha(), 0);
        // Outside the difference rect: red.
        assert_eq!(buffer.get_pixel(0, 0).unwrap().red(), 255);
    }

    #[test]
    fn test_playback_set_matrix_composes_with_initial_ctm() {
        // SkPicturePlayback: setMatrix during playback is
        // setMatrix(initialCTM * recorded), not an absolute replacement.
        let mut commands = Vec::new();
        {
            let mut rec = Canvas::new_recording(&mut commands, 20, 20);
            rec.set_matrix(&Matrix::translate(5.0, 0.0));
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(255, 0, 255, 0));
            rec.draw_rect(&Rect::from_xywh(0.0, 0.0, 2.0, 2.0), &paint);
        }
        let picture = Picture::new(commands, Rect::from_xywh(0.0, 0.0, 20.0, 20.0));

        let mut buffer = PixelBuffer::new(20, 20);
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            canvas.translate(10.0, 0.0);
            picture.playback(&mut canvas);
        }
        // Composed: initial translate(10,0) * recorded translate(5,0) => x=15.
        assert_eq!(buffer.get_pixel(16, 1).unwrap().green(), 255);
        // Absolute set_matrix would have drawn at x=5..7 instead.
        assert_eq!(buffer.get_pixel(6, 1).unwrap().green(), 0);
    }

    #[test]
    fn test_recording_canvas_save_returns_previous_count() {
        // RecordingCanvas::save mirrors SkCanvas semantics: initial save
        // count 1, save() returns the pre-save value.
        let mut recorder = PictureRecorder::new();
        let canvas = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 10.0, 10.0));
        assert_eq!(canvas.save(), 1);
        assert_eq!(canvas.save(), 2);
        canvas.restore();
        assert_eq!(canvas.save(), 2);
    }

    #[test]
    fn test_draw_picture_paint_applies_implicit_layer() {
        // drawPicture(picture, matrix, paint) wraps playback in an implicit
        // saveLayer(paint): a half-alpha paint halves the picture's opaque
        // white over black to ~mid-gray.
        let mut inner_cmds = Vec::new();
        {
            let mut rec = Canvas::new_recording(&mut inner_cmds, 4, 4);
            let mut white = Paint::new();
            white.set_color32(Color::from_argb(255, 255, 255, 255));
            rec.draw_rect(&Rect::from_xywh(0.0, 0.0, 4.0, 4.0), &white);
        }
        let inner = Arc::new(Picture::new(
            inner_cmds,
            Rect::from_xywh(0.0, 0.0, 4.0, 4.0),
        ));

        let mut buffer = PixelBuffer::new(4, 4);
        buffer.clear(Color::from_argb(255, 0, 0, 0));
        {
            let mut canvas = Canvas::new_raster(&mut buffer);
            let mut layer_paint = Paint::new();
            layer_paint.set_alpha(0.5);
            canvas.draw_picture(&inner, None, Some(&layer_paint));
        }
        let c = buffer.get_pixel(2, 2).unwrap();
        assert!(
            c.red() > 100 && c.red() < 160,
            "expected ~128 via implicit save_layer, got {}",
            c.red()
        );
    }

    #[test]
    fn test_picture_recorder() {
        let mut recorder = PictureRecorder::new();
        assert!(!recorder.is_recording());

        let canvas = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        canvas.save();
        canvas.translate(10.0, 20.0);
        let paint = Paint::new();
        canvas.draw_rect(&Rect::from_xywh(0.0, 0.0, 50.0, 50.0), &paint);
        canvas.restore();

        let picture = recorder.finish_recording().unwrap();
        assert!(!recorder.is_recording());
        assert_eq!(picture.approximate_op_count(), 4); // save, translate, draw_rect, restore
    }

    #[test]
    fn test_picture_playback() {
        let mut recorder = PictureRecorder::new();
        let canvas = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        canvas.translate(10.0, 20.0);
        let picture = recorder.finish_recording().unwrap();

        let mut canvas = Canvas::new(100, 100);
        picture.playback(&mut canvas);
        // Verify canvas was modified
        let matrix = canvas.total_matrix();
        assert!(!matrix.is_identity());
    }

    #[test]
    fn test_picture_playback_draws_pixels() {
        // Before the Canvas/Backing unification, Picture::playback received a
        // Canvas whose draw methods were TODO stubs, so replay produced no
        // pixels. This test pins the new end-to-end behavior: record a clear
        // + filled rect, replay to a raster surface, and verify the pixels.
        use crate::Surface;
        use skia_rs_core::Color;
        use skia_rs_paint::Style;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 20.0, 20.0));
        rc.clear(Color::from_argb(255, 255, 255, 255));
        let mut paint = Paint::new();
        paint.set_color32(Color::from_argb(255, 0, 0, 255));
        paint.set_style(Style::Fill);
        rc.draw_rect(&Rect::from_xywh(5.0, 5.0, 10.0, 10.0), &paint);
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(20, 20).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        let pixel = surface.pixel_buffer().get_pixel(10, 10).unwrap();
        assert_eq!(pixel.blue(), 255);
        assert_eq!(pixel.red(), 0);

        // A pixel outside the rect should be the clear color (white).
        let bg = surface.pixel_buffer().get_pixel(1, 1).unwrap();
        assert_eq!(bg.red(), 255);
        assert_eq!(bg.green(), 255);
        assert_eq!(bg.blue(), 255);
    }

    #[test]
    fn test_nested_pictures() {
        // Create inner picture
        let mut recorder = PictureRecorder::new();
        let canvas = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 50.0, 50.0));
        canvas.translate(5.0, 5.0);
        let inner = recorder.finish_recording().unwrap();

        // Create outer picture that contains inner
        let mut recorder2 = PictureRecorder::new();
        let canvas = recorder2.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        canvas.draw_picture(&inner, None, None);
        let outer = recorder2.finish_recording().unwrap();

        assert_eq!(outer.approximate_op_count(), 1);
    }

    // =========================================================================
    // T-5: Picture playback round-trip tests for DrawCommand variants
    // =========================================================================

    #[test]
    fn test_picture_playback_circle() {
        // Record draw_circle, replay, and verify pixels match direct draw.
        use crate::Surface;
        use skia_rs_core::Color;

        let center = Point::new(50.0, 50.0);
        let radius = 20.0;
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 0, 0));

        // Record
        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        rc.draw_circle(center, radius, &paint);
        let picture = recorder.finish_recording().unwrap();

        // Playback
        let mut surface_playback = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface_playback.canvas();
            picture.playback(&mut canvas);
        }

        // Direct
        let mut surface_direct = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface_direct.canvas();
            canvas.draw_circle(center, radius, &paint);
        }

        // Compare center pixel (should be red in both)
        let pixel_playback = surface_playback.pixel_buffer().get_pixel(50, 50).unwrap();
        let pixel_direct = surface_direct.pixel_buffer().get_pixel(50, 50).unwrap();
        assert_eq!(pixel_playback.red(), pixel_direct.red());
        assert_eq!(pixel_playback.green(), pixel_direct.green());
        assert_eq!(pixel_playback.blue(), pixel_direct.blue());
    }

    #[test]
    fn test_picture_playback_path() {
        // Record draw_path, replay, and verify output.
        use crate::Surface;
        use skia_rs_core::Color;
        use skia_rs_path::PathBuilder;

        let mut builder = PathBuilder::new();
        builder.move_to(10.0, 10.0);
        builder.line_to(90.0, 10.0);
        builder.line_to(50.0, 90.0);
        builder.close();
        let path = builder.build();

        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(0, 255, 0));

        // Record
        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        rc.draw_path(&path, &paint);
        let picture = recorder.finish_recording().unwrap();

        // Playback
        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Verify center pixel has green contribution
        let pixel = surface.pixel_buffer().get_pixel(50, 50).unwrap();
        assert!(
            pixel.green() > 100,
            "center should have green from triangle"
        );
    }

    #[test]
    fn test_picture_playback_with_transforms() {
        // Record: save, translate, draw_rect, restore. Verify rect appears
        // at translated position.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        rc.save();
        rc.translate(25.0, 25.0);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 255, 0));
        rc.draw_rect(&Rect::from_xywh(0.0, 0.0, 10.0, 10.0), &paint);
        rc.restore();
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Pixel at (30, 30) should be yellow (inside translated rect)
        let pixel = surface.pixel_buffer().get_pixel(30, 30).unwrap();
        assert!(
            pixel.red() > 200 && pixel.green() > 200,
            "pixel should be yellow"
        );

        // Pixel at (5, 5) should be background (not in translated rect)
        let bg = surface.pixel_buffer().get_pixel(5, 5).unwrap();
        assert_eq!(
            bg,
            Color::TRANSPARENT,
            "pixel outside rect should be transparent"
        );
    }

    #[test]
    fn test_picture_playback_oval() {
        // Record draw_oval, verify playback draws pixels.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(0, 0, 255));
        rc.draw_oval(&Rect::from_xywh(25.0, 25.0, 50.0, 50.0), &paint);
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Center of oval should be blue
        let pixel = surface.pixel_buffer().get_pixel(50, 50).unwrap();
        assert!(pixel.blue() > 200, "oval center should be blue");
    }

    #[test]
    fn test_picture_playback_arc() {
        // Record draw_arc, verify playback doesn't panic.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(128, 0, 128));
        rc.draw_arc(
            &Rect::from_xywh(10.0, 10.0, 80.0, 80.0),
            0.0,
            90.0,
            true,
            &paint,
        );
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Arc playback should complete without panicking. Exact pixel verification
        // depends on arc implementation details (whether it fills, strokes, etc).
        // For now, verify the command was recorded and executed.
        assert_eq!(picture.approximate_op_count(), 1);
    }

    #[test]
    fn test_picture_playback_round_rect() {
        // Record draw_round_rect, verify playback.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 128, 0));
        rc.draw_round_rect(&Rect::from_xywh(20.0, 20.0, 60.0, 60.0), 10.0, 10.0, &paint);
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Center should be orange
        let pixel = surface.pixel_buffer().get_pixel(50, 50).unwrap();
        assert!(
            pixel.red() > 200 && pixel.green() > 50,
            "round rect center should be orange"
        );
    }

    #[test]
    fn test_picture_playback_line() {
        // Record draw_line, verify playback.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(255, 255, 255));
        paint.set_stroke_width(2.0);
        rc.draw_line(Point::new(10.0, 10.0), Point::new(90.0, 90.0), &paint);
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Pixel near the line midpoint should be white
        let pixel = surface.pixel_buffer().get_pixel(50, 50).unwrap();
        assert!(pixel.red() > 200, "line midpoint should be white");
    }

    #[test]
    fn test_picture_playback_clip_rect() {
        // Record: clip_rect, then draw outside the clip. Verify clipping worked.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        rc.clip_rect(&Rect::from_xywh(25.0, 25.0, 50.0, 50.0), false);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(200, 200, 200));
        rc.draw_rect(&Rect::from_xywh(0.0, 0.0, 100.0, 100.0), &paint);
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // Pixel at (50, 50) inside clip should be gray
        let inside = surface.pixel_buffer().get_pixel(50, 50).unwrap();
        assert!(inside.red() > 150, "inside clip should be drawn");

        // Pixel at (10, 10) outside clip should be transparent
        let outside = surface.pixel_buffer().get_pixel(10, 10).unwrap();
        assert_eq!(
            outside,
            Color::TRANSPARENT,
            "outside clip should not be drawn"
        );
    }

    #[test]
    fn test_picture_playback_scale() {
        // Record: scale, draw_rect. Verify scaling is applied.
        use crate::Surface;
        use skia_rs_core::Color;

        let mut recorder = PictureRecorder::new();
        let rc = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, 100.0, 100.0));
        rc.scale(2.0, 2.0);
        let mut paint = Paint::new();
        paint.set_color32(Color::from_rgb(100, 100, 255));
        rc.draw_rect(&Rect::from_xywh(10.0, 10.0, 10.0, 10.0), &paint);
        let picture = recorder.finish_recording().unwrap();

        let mut surface = Surface::new_raster_n32_premul(100, 100).unwrap();
        {
            let mut canvas = surface.canvas();
            picture.playback(&mut canvas);
        }

        // With scale(2, 2), rect at (10, 10) size 10x10 → (20, 20) size 20x20
        let pixel = surface.pixel_buffer().get_pixel(30, 30).unwrap();
        assert!(
            pixel.blue() > 200,
            "scaled rect should draw at scaled position"
        );
    }
}
