//! Command buffer recording and submission.
//!
//! This module provides abstractions for recording GPU commands and managing
//! command buffer submission.

use crate::cast_util::u32_from_scalar_sat;
use skia_rs_core::cast::f32_to_u8_sat;
use skia_rs_core::{Color, Rect};

/// Draw command for batching.
#[derive(Debug, Clone)]
pub enum DrawCommand {
    /// Clear the render target.
    Clear {
        /// Clear color.
        color: Color,
    },
    /// Draw primitives.
    Draw {
        /// Vertex count.
        vertex_count: u32,
        /// Instance count.
        instance_count: u32,
        /// First vertex.
        first_vertex: u32,
        /// First instance.
        first_instance: u32,
    },
    /// Draw indexed primitives.
    DrawIndexed {
        /// Index count.
        index_count: u32,
        /// Instance count.
        instance_count: u32,
        /// First index.
        first_index: u32,
        /// Base vertex.
        base_vertex: i32,
        /// First instance.
        first_instance: u32,
    },
    /// Dispatch a compute workgroup grid. Distinct from `Draw` so the
    /// executor can route it to a compute pass (or reject it) rather than
    /// mis-replaying it as a draw call.
    DispatchCompute {
        /// Workgroups in X.
        x: u32,
        /// Workgroups in Y.
        y: u32,
        /// Workgroups in Z.
        z: u32,
    },
    /// Set scissor rect.
    SetScissor {
        /// Scissor rectangle.
        rect: ScissorRect,
    },
    /// Set viewport.
    SetViewport {
        /// Viewport configuration.
        viewport: Viewport,
    },
    /// Set blend constant.
    SetBlendConstant {
        /// Blend constant color.
        color: [f32; 4],
    },
    /// Set stencil reference.
    SetStencilReference {
        /// Reference value.
        reference: u32,
    },
    /// Push debug group.
    PushDebugGroup {
        /// Label.
        label: String,
    },
    /// Pop debug group.
    PopDebugGroup,
    /// Insert debug marker.
    InsertDebugMarker {
        /// Label.
        label: String,
    },
    /// Set pipeline.
    SetPipeline {
        /// Pipeline ID.
        pipeline_id: u64,
    },
    /// Set bind group.
    SetBindGroup {
        /// Group index.
        index: u32,
        /// Bind group ID.
        bind_group_id: u64,
        /// Dynamic offsets.
        dynamic_offsets: Vec<u32>,
    },
    /// Set vertex buffer.
    SetVertexBuffer {
        /// Slot index.
        slot: u32,
        /// Buffer ID.
        buffer_id: u64,
        /// Offset.
        offset: u64,
        /// Size (None for entire buffer).
        size: Option<u64>,
    },
    /// Set index buffer.
    SetIndexBuffer {
        /// Buffer ID.
        buffer_id: u64,
        /// Index format.
        format: IndexFormat,
        /// Offset.
        offset: u64,
        /// Size (None for entire buffer).
        size: Option<u64>,
    },
    /// Copy buffer to buffer.
    CopyBufferToBuffer {
        /// Source buffer ID.
        src: u64,
        /// Source offset.
        src_offset: u64,
        /// Destination buffer ID.
        dst: u64,
        /// Destination offset.
        dst_offset: u64,
        /// Size.
        size: u64,
    },
    /// Copy buffer to texture.
    CopyBufferToTexture {
        /// Source buffer ID.
        src_buffer: u64,
        /// Source layout.
        src_layout: ImageDataLayout,
        /// Destination texture ID.
        dst_texture: u64,
        /// Destination origin.
        dst_origin: [u32; 3],
        /// Copy size.
        size: [u32; 3],
    },
    /// Copy texture to buffer.
    CopyTextureToBuffer {
        /// Source texture ID.
        src_texture: u64,
        /// Source origin.
        src_origin: [u32; 3],
        /// Destination buffer ID.
        dst_buffer: u64,
        /// Destination layout.
        dst_layout: ImageDataLayout,
        /// Copy size.
        size: [u32; 3],
    },
    /// Copy texture to texture.
    CopyTextureToTexture {
        /// Source texture ID.
        src_texture: u64,
        /// Source origin.
        src_origin: [u32; 3],
        /// Destination texture ID.
        dst_texture: u64,
        /// Destination origin.
        dst_origin: [u32; 3],
        /// Copy size.
        size: [u32; 3],
    },
}

// Re-use IndexFormat from pipeline module
pub use crate::pipeline::IndexFormat;

/// Image data layout for copy operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageDataLayout {
    /// Offset into the buffer.
    pub offset: u64,
    /// Bytes per row.
    pub bytes_per_row: Option<u32>,
    /// Rows per image (for 3D textures).
    pub rows_per_image: Option<u32>,
}

/// Scissor rectangle.
#[derive(Debug, Clone, Copy)]
pub struct ScissorRect {
    /// X position.
    pub x: u32,
    /// Y position.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl ScissorRect {
    /// Create a new scissor rect.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create from a rect, clamping the box into the non-negative quadrant.
    ///
    /// When `left`/`top` are negative the origin is clamped to 0 **and** the
    /// width/height shrink by the clipped-off amount, so the box's right/
    /// bottom edges stay put (rather than sliding right as they would if the
    /// original width were kept). Fully off-screen boxes collapse to zero.
    #[must_use]
    pub fn from_rect(rect: &Rect) -> Self {
        let left = rect.left.max(0.0);
        let top = rect.top.max(0.0);
        let right = rect.right.max(left);
        let bottom = rect.bottom.max(top);
        Self {
            x: u32_from_scalar_sat(left),
            y: u32_from_scalar_sat(top),
            width: u32_from_scalar_sat(right - left),
            height: u32_from_scalar_sat(bottom - top),
        }
    }

    /// Clamp the scissor box to a framebuffer of `fb_width` x `fb_height`,
    /// shrinking width/height so the box never extends past the edges.
    #[must_use]
    pub fn clamp_to_framebuffer(&self, fb_width: u32, fb_height: u32) -> Self {
        let x = self.x.min(fb_width);
        let y = self.y.min(fb_height);
        Self {
            x,
            y,
            width: self.width.min(fb_width - x),
            height: self.height.min(fb_height - y),
        }
    }
}

/// Viewport configuration.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
    /// Minimum depth.
    pub min_depth: f32,
    /// Maximum depth.
    pub max_depth: f32,
}

impl Viewport {
    /// Create a new viewport.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Set depth range.
    #[must_use]
    pub const fn with_depth(mut self, min: f32, max: f32) -> Self {
        self.min_depth = min;
        self.max_depth = max;
        self
    }
}

/// Command buffer for recording GPU commands.
#[derive(Debug, Default)]
pub struct CommandBuffer {
    /// Recorded commands.
    commands: Vec<DrawCommand>,
    /// Current debug group depth.
    debug_depth: u32,
}

impl CommandBuffer {
    /// Create a new empty command buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            commands: Vec::new(),
            debug_depth: 0,
        }
    }

    /// Create with capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            debug_depth: 0,
        }
    }

    /// Record a command.
    pub fn record(&mut self, command: DrawCommand) {
        match &command {
            DrawCommand::PushDebugGroup { .. } => self.debug_depth += 1,
            DrawCommand::PopDebugGroup => {
                self.debug_depth = self.debug_depth.saturating_sub(1);
            }
            _ => {}
        }
        self.commands.push(command);
    }

    /// Clear the render target.
    pub fn clear(&mut self, color: Color) {
        self.record(DrawCommand::Clear { color });
    }

    /// Draw primitives.
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.record(DrawCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex: 0,
            first_instance: 0,
        });
    }

    /// Draw primitives with offsets.
    pub fn draw_with_offsets(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.record(DrawCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    /// Draw indexed primitives.
    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32) {
        self.record(DrawCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        });
    }

    /// Draw indexed primitives with offsets.
    pub fn draw_indexed_with_offsets(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        self.record(DrawCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }

    /// Record a compute dispatch of `(x, y, z)` workgroups.
    pub fn dispatch_compute(&mut self, x: u32, y: u32, z: u32) {
        self.record(DrawCommand::DispatchCompute { x, y, z });
    }

    /// Set scissor rect.
    pub fn set_scissor(&mut self, rect: ScissorRect) {
        self.record(DrawCommand::SetScissor { rect });
    }

    /// Set viewport.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.record(DrawCommand::SetViewport { viewport });
    }

    /// Set blend constant.
    pub fn set_blend_constant(&mut self, color: [f32; 4]) {
        self.record(DrawCommand::SetBlendConstant { color });
    }

    /// Set stencil reference.
    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.record(DrawCommand::SetStencilReference { reference });
    }

    /// Push a debug group.
    pub fn push_debug_group(&mut self, label: impl Into<String>) {
        self.record(DrawCommand::PushDebugGroup {
            label: label.into(),
        });
    }

    /// Pop a debug group.
    pub fn pop_debug_group(&mut self) {
        self.record(DrawCommand::PopDebugGroup);
    }

    /// Insert a debug marker.
    pub fn insert_debug_marker(&mut self, label: impl Into<String>) {
        self.record(DrawCommand::InsertDebugMarker {
            label: label.into(),
        });
    }

    /// Set the current pipeline.
    pub fn set_pipeline(&mut self, pipeline_id: u64) {
        self.record(DrawCommand::SetPipeline { pipeline_id });
    }

    /// Set a bind group.
    pub fn set_bind_group(&mut self, index: u32, bind_group_id: u64, dynamic_offsets: &[u32]) {
        self.record(DrawCommand::SetBindGroup {
            index,
            bind_group_id,
            dynamic_offsets: dynamic_offsets.to_vec(),
        });
    }

    /// Set a vertex buffer.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_id: u64, offset: u64, size: Option<u64>) {
        self.record(DrawCommand::SetVertexBuffer {
            slot,
            buffer_id,
            offset,
            size,
        });
    }

    /// Set the index buffer.
    pub fn set_index_buffer(
        &mut self,
        buffer_id: u64,
        format: IndexFormat,
        offset: u64,
        size: Option<u64>,
    ) {
        self.record(DrawCommand::SetIndexBuffer {
            buffer_id,
            format,
            offset,
            size,
        });
    }

    /// Copy buffer to buffer.
    pub fn copy_buffer_to_buffer(
        &mut self,
        src: u64,
        src_offset: u64,
        dst: u64,
        dst_offset: u64,
        size: u64,
    ) {
        self.record(DrawCommand::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
    }

    /// Get the recorded commands.
    #[must_use]
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Take ownership of the commands.
    pub fn take_commands(&mut self) -> Vec<DrawCommand> {
        std::mem::take(&mut self.commands)
    }

    /// Clear all recorded commands.
    pub fn reset(&mut self) {
        self.commands.clear();
        self.debug_depth = 0;
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Get the number of commands.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.commands.len()
    }
}

/// Command encoder for recording render and compute passes.
pub struct CommandEncoder {
    /// Command buffer being built.
    buffer: CommandBuffer,
    /// Label for debugging.
    label: Option<String>,
}

impl CommandEncoder {
    /// Create a new command encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: CommandBuffer::new(),
            label: None,
        }
    }

    /// Create with a label.
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            buffer: CommandBuffer::new(),
            label: Some(label.into()),
        }
    }

    /// Get the debug label, if any.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Begin a render pass.
    pub fn begin_render_pass(&mut self, desc: &RenderPassDescriptor) -> RenderPassEncoder<'_> {
        if let Some(color) = desc.clear_color {
            self.buffer.clear(Color::from_argb(
                f32_to_u8_sat(color[3] * 255.0),
                f32_to_u8_sat(color[0] * 255.0),
                f32_to_u8_sat(color[1] * 255.0),
                f32_to_u8_sat(color[2] * 255.0),
            ));
        }
        RenderPassEncoder { encoder: self }
    }

    /// Begin a compute pass.
    pub const fn begin_compute_pass(&mut self) -> ComputePassEncoder<'_> {
        ComputePassEncoder { encoder: self }
    }

    /// Copy buffer to buffer.
    pub fn copy_buffer_to_buffer(
        &mut self,
        src: u64,
        src_offset: u64,
        dst: u64,
        dst_offset: u64,
        size: u64,
    ) {
        self.buffer
            .copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
    }

    /// Copy buffer to texture.
    pub fn copy_buffer_to_texture(
        &mut self,
        src_buffer: u64,
        src_layout: ImageDataLayout,
        dst_texture: u64,
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) {
        self.buffer.record(DrawCommand::CopyBufferToTexture {
            src_buffer,
            src_layout,
            dst_texture,
            dst_origin,
            size,
        });
    }

    /// Copy texture to buffer.
    pub fn copy_texture_to_buffer(
        &mut self,
        src_texture: u64,
        src_origin: [u32; 3],
        dst_buffer: u64,
        dst_layout: ImageDataLayout,
        size: [u32; 3],
    ) {
        self.buffer.record(DrawCommand::CopyTextureToBuffer {
            src_texture,
            src_origin,
            dst_buffer,
            dst_layout,
            size,
        });
    }

    /// Copy texture to texture.
    pub fn copy_texture_to_texture(
        &mut self,
        src_texture: u64,
        src_origin: [u32; 3],
        dst_texture: u64,
        dst_origin: [u32; 3],
        size: [u32; 3],
    ) {
        self.buffer.record(DrawCommand::CopyTextureToTexture {
            src_texture,
            src_origin,
            dst_texture,
            dst_origin,
            size,
        });
    }

    /// Push a debug group.
    pub fn push_debug_group(&mut self, label: &str) {
        self.buffer.push_debug_group(label);
    }

    /// Pop a debug group.
    pub fn pop_debug_group(&mut self) {
        self.buffer.pop_debug_group();
    }

    /// Insert a debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.buffer.insert_debug_marker(label);
    }

    /// Finish recording and return the command buffer.
    #[must_use]
    pub fn finish(self) -> CommandBuffer {
        self.buffer
    }
}

impl Default for CommandEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// Re-use RenderPassDescriptor from surface module
pub use crate::surface::RenderPassDescriptor;

/// Render pass encoder.
pub struct RenderPassEncoder<'a> {
    encoder: &'a mut CommandEncoder,
}

impl RenderPassEncoder<'_> {
    /// Set the current pipeline.
    pub fn set_pipeline(&mut self, pipeline_id: u64) {
        self.encoder.buffer.set_pipeline(pipeline_id);
    }

    /// Set a bind group.
    pub fn set_bind_group(&mut self, index: u32, bind_group_id: u64, dynamic_offsets: &[u32]) {
        self.encoder
            .buffer
            .set_bind_group(index, bind_group_id, dynamic_offsets);
    }

    /// Set a vertex buffer.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_id: u64, offset: u64, size: Option<u64>) {
        self.encoder
            .buffer
            .set_vertex_buffer(slot, buffer_id, offset, size);
    }

    /// Set the index buffer.
    pub fn set_index_buffer(
        &mut self,
        buffer_id: u64,
        format: IndexFormat,
        offset: u64,
        size: Option<u64>,
    ) {
        self.encoder
            .buffer
            .set_index_buffer(buffer_id, format, offset, size);
    }

    /// Set scissor rect.
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.encoder
            .buffer
            .set_scissor(ScissorRect::new(x, y, width, height));
    }

    /// Set viewport.
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.encoder.buffer.set_viewport(Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        });
    }

    /// Set blend constant.
    pub fn set_blend_constant(&mut self, color: [f32; 4]) {
        self.encoder.buffer.set_blend_constant(color);
    }

    /// Set stencil reference.
    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.encoder.buffer.set_stencil_reference(reference);
    }

    /// Draw primitives.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.encoder.buffer.draw_with_offsets(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    /// Draw indexed primitives.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        self.encoder.buffer.draw_indexed_with_offsets(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        );
    }

    /// Push a debug group.
    pub fn push_debug_group(&mut self, label: &str) {
        self.encoder.buffer.push_debug_group(label);
    }

    /// Pop a debug group.
    pub fn pop_debug_group(&mut self) {
        self.encoder.buffer.pop_debug_group();
    }

    /// Insert a debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.encoder.buffer.insert_debug_marker(label);
    }
}

/// Compute pass encoder.
pub struct ComputePassEncoder<'a> {
    encoder: &'a mut CommandEncoder,
}

impl ComputePassEncoder<'_> {
    /// Set the current pipeline.
    pub fn set_pipeline(&mut self, pipeline_id: u64) {
        self.encoder.buffer.set_pipeline(pipeline_id);
    }

    /// Set a bind group.
    pub fn set_bind_group(&mut self, index: u32, bind_group_id: u64, dynamic_offsets: &[u32]) {
        self.encoder
            .buffer
            .set_bind_group(index, bind_group_id, dynamic_offsets);
    }

    /// Dispatch compute work.
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        // Record a real compute dispatch. Executors that do not support
        // compute may reject or skip it, but it must never be replayed as a
        // draw call (which would issue a bogus vertex draw).
        self.encoder.buffer.dispatch_compute(x, y, z);
    }

    /// Push a debug group.
    pub fn push_debug_group(&mut self, label: &str) {
        self.encoder.buffer.push_debug_group(label);
    }

    /// Pop a debug group.
    pub fn pop_debug_group(&mut self) {
        self.encoder.buffer.pop_debug_group();
    }

    /// Insert a debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.encoder.buffer.insert_debug_marker(label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_buffer() {
        let mut buffer = CommandBuffer::new();
        buffer.clear(Color::RED);
        buffer.draw(6, 1);

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_command_encoder() {
        let mut encoder = CommandEncoder::new();

        {
            let mut pass =
                encoder.begin_render_pass(&RenderPassDescriptor::color_clear(0.0, 0.0, 0.0, 1.0));
            pass.set_pipeline(1);
            pass.draw(6, 1, 0, 0);
        }

        let buffer = encoder.finish();
        assert!(buffer.len() >= 2);
    }

    #[test]
    fn test_scissor_rect() {
        let rect = Rect::from_xywh(10.0, 20.0, 100.0, 200.0);
        let scissor = ScissorRect::from_rect(&rect);

        assert_eq!(scissor.x, 10);
        assert_eq!(scissor.y, 20);
        assert_eq!(scissor.width, 100);
        assert_eq!(scissor.height, 200);
    }

    #[test]
    fn test_scissor_rect_negative_origin_shrinks_box() {
        // Regression: a box with negative left/top must clamp origin to 0 and
        // shrink width/height so the right/bottom edges stay put.
        let rect = Rect::new(-10.0, -20.0, 90.0, 80.0); // left,top,right,bottom
        let scissor = ScissorRect::from_rect(&rect);
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 90, "right edge preserved: 90 - 0");
        assert_eq!(scissor.height, 80, "bottom edge preserved: 80 - 0");
    }

    #[test]
    fn test_scissor_clamp_to_framebuffer() {
        let s = ScissorRect::new(90, 90, 100, 100);
        let c = s.clamp_to_framebuffer(128, 128);
        assert_eq!(c.x, 90);
        assert_eq!(c.width, 38, "clamped to 128 - 90");
        assert_eq!(c.height, 38);
        // Origin beyond the framebuffer collapses to zero extent.
        let off = ScissorRect::new(200, 200, 10, 10).clamp_to_framebuffer(128, 128);
        assert_eq!(off.width, 0);
        assert_eq!(off.height, 0);
    }

    #[test]
    fn test_dispatch_compute_records_distinct_command() {
        // Regression: a compute dispatch must record a DispatchCompute, not a
        // Draw (which would replay as a bogus vertex draw).
        let mut buffer = CommandBuffer::new();
        buffer.dispatch_compute(8, 4, 1);
        assert_eq!(buffer.len(), 1);
        match &buffer.commands()[0] {
            DrawCommand::DispatchCompute { x, y, z } => {
                assert_eq!((*x, *y, *z), (8, 4, 1));
            }
            other => panic!("expected DispatchCompute, got {other:?}"),
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_viewport() {
        let viewport = Viewport::new(0.0, 0.0, 800.0, 600.0).with_depth(0.0, 1.0);

        assert_eq!(viewport.width, 800.0);
        assert_eq!(viewport.height, 600.0);
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn test_debug_groups() {
        let mut buffer = CommandBuffer::new();

        buffer.push_debug_group("outer");
        buffer.push_debug_group("inner");
        buffer.pop_debug_group();
        buffer.pop_debug_group();

        assert_eq!(buffer.debug_depth, 0);
    }
}
