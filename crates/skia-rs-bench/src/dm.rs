//! DM-Style Testing Infrastructure
//!
//! This module provides a testing infrastructure similar to Skia's DM (Drawing Manager).
//! It enables systematic testing of rendering operations across multiple backends
//! and configurations.
//!
//! # Architecture
//!
//! The DM system consists of:
//! - **Renderers**: Implementations that can render test content
//! - **Sources**: Test content generators (GMs, tests, etc.)
//! - **Sinks**: Output handlers (PNG files, comparison tools, etc.)
//! - **Runner**: Orchestrates test execution
//!
//! # Example
//!
//! ```ignore
//! use skia_rs_bench::dm::{DmRunner, GmSource, PngSink};
//!
//! let mut runner = DmRunner::new();
//! runner.add_source(GmSource::all());
//! runner.add_sink(PngSink::new("output/"));
//! runner.run();
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use skia_rs_canvas::Surface;
use skia_rs_core::{Color, Point, Rect};
use skia_rs_paint::Paint;
use skia_rs_path::PathBuilder;

// =============================================================================
// Core Traits
// =============================================================================

/// A source of test content (similar to Skia's GM)
pub trait Source: Send + Sync {
    /// Name of this test source
    fn name(&self) -> &str;

    /// Size of the output canvas
    fn size(&self) -> (i32, i32);

    /// Draw the test content
    fn draw(&self, surface: &mut Surface);

    /// Tags for filtering
    fn tags(&self) -> Vec<&str> {
        vec![]
    }
}

/// A sink for test output
pub trait Sink: Send + Sync {
    /// Name of this sink
    fn name(&self) -> &str;

    /// Process a test result
    ///
    /// # Errors
    ///
    /// Returns [`SinkError`] if the sink fails to process the result (e.g. I/O failure).
    fn process(&self, result: &TestResult) -> Result<(), SinkError>;
}

/// A renderer backend
pub trait Renderer: Send + Sync {
    /// Name of this renderer
    fn name(&self) -> &str;

    /// Create a surface for rendering
    fn create_surface(&self, width: i32, height: i32) -> Option<Surface>;

    /// Tags for filtering
    fn tags(&self) -> Vec<&str> {
        vec![]
    }
}

// =============================================================================
// Test Results
// =============================================================================

/// Result of a single test run
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Source name
    pub source: String,
    /// Renderer name
    pub renderer: String,
    /// Test outcome
    pub outcome: TestOutcome,
    /// Render duration
    pub duration: Duration,
    /// Output pixels (RGBA)
    pub pixels: Option<Vec<u8>>,
    /// Output dimensions
    pub width: i32,
    /// Output dimensions
    pub height: i32,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Outcome of a test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestOutcome {
    /// Test passed
    Pass,
    /// Test failed
    Fail,
    /// Test was skipped
    Skip,
    /// Test crashed/panicked
    Crash,
}

/// Error from a sink
#[derive(Debug)]
pub struct SinkError {
    /// Error message
    pub message: String,
}

impl std::fmt::Display for SinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SinkError {}

// =============================================================================
// Built-in Renderers
// =============================================================================

/// Software (CPU) raster renderer
pub struct RasterRenderer;

impl Default for RasterRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl RasterRenderer {
    /// Create a new raster renderer
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Renderer for RasterRenderer {
    fn name(&self) -> &'static str {
        "raster"
    }

    fn create_surface(&self, width: i32, height: i32) -> Option<Surface> {
        Surface::new_raster_n32_premul(width, height)
    }

    fn tags(&self) -> Vec<&str> {
        vec!["cpu", "software"]
    }
}

// =============================================================================
// Built-in Sources (GMs)
// =============================================================================

/// A GM (golden master) test
pub struct Gm {
    name: String,
    width: i32,
    height: i32,
    draw_fn: Box<dyn Fn(&mut Surface) + Send + Sync>,
    tags: Vec<String>,
}

impl Gm {
    /// Create a new GM
    pub fn new<F>(name: &str, width: i32, height: i32, draw_fn: F) -> Self
    where
        F: Fn(&mut Surface) + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            width,
            height,
            draw_fn: Box::new(draw_fn),
            tags: Vec::new(),
        }
    }

    /// Add a tag
    #[must_use]
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }
}

impl Source for Gm {
    fn name(&self) -> &str {
        &self.name
    }

    fn size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    fn draw(&self, surface: &mut Surface) {
        (self.draw_fn)(surface);
    }

    fn tags(&self) -> Vec<&str> {
        self.tags.iter().map(std::string::String::as_str).collect()
    }
}

/// Collection of standard GMs
pub struct StandardGms;

impl StandardGms {
    /// Get all standard GMs
    #[must_use]
    pub fn all() -> Vec<Arc<dyn Source>> {
        vec![
            Arc::new(Self::simple_rect()),
            Arc::new(Self::circles()),
            Arc::new(Self::paths()),
            Arc::new(Self::gradients()),
            Arc::new(Self::transforms()),
            Arc::new(Self::clipping()),
            Arc::new(Self::blend_modes()),
            Arc::new(Self::stroke_styles()),
            Arc::new(Self::antialiasing()),
            Arc::new(Self::alpha_blending()),
        ]
    }

    /// Simple rectangle GM
    #[must_use]
    pub fn simple_rect() -> Gm {
        Gm::new("simple_rect", 200, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            let mut paint = Paint::new();
            paint.set_color32(Color::RED);
            canvas.draw_rect(&Rect::from_xywh(50.0, 50.0, 100.0, 100.0), &paint);
        })
        .with_tag("basic")
    }

    /// Circles GM
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "loop index is bounded by a 3-element array; precision loss cannot occur in practice"
    )]
    pub fn circles() -> Gm {
        Gm::new("circles", 300, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            let colors = [Color::RED, Color::GREEN, Color::BLUE];
            for (i, color) in colors.iter().enumerate() {
                let mut paint = Paint::new();
                paint.set_color32(*color);
                paint.set_anti_alias(true);

                let cx = (i as f32).mul_add(100.0, 50.0);
                canvas.draw_circle(Point::new(cx, 100.0), 40.0, &paint);
            }
        })
        .with_tag("basic")
    }

    /// Paths GM
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "loop index is bounded by a fixed 5-point star; precision loss cannot occur in practice"
    )]
    pub fn paths() -> Gm {
        Gm::new("paths", 200, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            let mut paint = Paint::new();
            paint.set_color32(Color::from_rgb(100, 100, 200));
            paint.set_anti_alias(true);

            // Star path
            let mut builder = PathBuilder::new();
            let cx = 100.0;
            let cy = 100.0;
            let outer: f32 = 80.0;
            let inner: f32 = 30.0;

            for i in 0..5 {
                let angle_outer =
                    std::f32::consts::PI * 2.0 * i as f32 / 5.0 - std::f32::consts::PI / 2.0;
                let angle_inner = angle_outer + std::f32::consts::PI / 5.0;

                // Unfused, as Skia's own drawing of this star computes it.
                #[allow(clippy::suboptimal_flops, reason = "match Skia's unfused rounding")]
                let (px, py, ix, iy) = (
                    cx + outer * angle_outer.cos(),
                    cy + outer * angle_outer.sin(),
                    cx + inner * angle_inner.cos(),
                    cy + inner * angle_inner.sin(),
                );

                if i == 0 {
                    builder.move_to(px, py);
                } else {
                    builder.line_to(px, py);
                }
                builder.line_to(ix, iy);
            }
            builder.close();

            canvas.draw_path(&builder.build(), &paint);
        })
        .with_tag("path")
    }

    /// Gradients GM
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "loop index is bounded by small fixed ranges (100, 40); precision loss cannot occur in practice"
    )]
    pub fn gradients() -> Gm {
        Gm::new("gradients", 300, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            // Simple colored rectangles as gradient approximation
            let mut paint = Paint::new();

            // "Linear gradient" approximation
            for i in 0..100 {
                let t = i as f32 / 100.0;
                let r = skia_rs_core::cast::f32_to_u8_sat(255.0 * (1.0 - t));
                let b = skia_rs_core::cast::f32_to_u8_sat(255.0 * t);
                paint.set_color32(Color::from_argb(255, r, 0, b));
                canvas.draw_rect(
                    &Rect::from_xywh((i as f32).mul_add(1.3, 10.0), 20.0, 2.0, 60.0),
                    &paint,
                );
            }

            // "Radial gradient" approximation
            for i in (0..40).rev() {
                let t = i as f32 / 40.0;
                let g = skia_rs_core::cast::f32_to_u8_sat(255.0 * t);
                paint.set_color32(Color::from_argb(255, 255, g, 0));
                canvas.draw_circle(Point::new(200.0, 140.0), i as f32, &paint);
            }
        })
        .with_tag("paint")
    }

    /// Transforms GM
    #[must_use]
    pub fn transforms() -> Gm {
        Gm::new("transforms", 300, 300, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            let mut paint = Paint::new();
            paint.set_color32(Color::from_rgb(200, 100, 50));

            // Draw rectangles in different positions
            // (actual transforms would need canvas.save/restore/transform)
            let positions = [(50.0, 50.0), (150.0, 50.0), (50.0, 150.0), (150.0, 150.0)];

            for (x, y) in positions {
                canvas.draw_rect(&Rect::from_xywh(x, y, 80.0, 80.0), &paint);
            }
        })
        .with_tag("transform")
    }

    /// Clipping GM
    #[must_use]
    pub fn clipping() -> Gm {
        Gm::new("clipping", 200, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            // Draw a large shape that would be clipped
            let mut paint = Paint::new();
            paint.set_color32(Color::from_rgb(100, 150, 200));
            canvas.draw_rect(&Rect::from_xywh(20.0, 20.0, 160.0, 160.0), &paint);

            // Inner rectangle to show "clipping" effect visually
            paint.set_color32(Color::from_rgb(200, 150, 100));
            canvas.draw_rect(&Rect::from_xywh(40.0, 40.0, 120.0, 120.0), &paint);
        })
        .with_tag("clip")
    }

    /// Blend modes GM
    #[must_use]
    pub fn blend_modes() -> Gm {
        Gm::new("blend_modes", 300, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            // Background circles
            let mut paint = Paint::new();
            paint.set_color32(Color::from_argb(180, 255, 0, 0));
            canvas.draw_circle(Point::new(80.0, 80.0), 60.0, &paint);

            paint.set_color32(Color::from_argb(180, 0, 255, 0));
            canvas.draw_circle(Point::new(120.0, 80.0), 60.0, &paint);

            paint.set_color32(Color::from_argb(180, 0, 0, 255));
            canvas.draw_circle(Point::new(100.0, 120.0), 60.0, &paint);

            // Second set with different alpha
            paint.set_color32(Color::from_argb(128, 255, 255, 0));
            canvas.draw_circle(Point::new(220.0, 100.0), 50.0, &paint);
        })
        .with_tag("blend")
    }

    /// Stroke styles GM
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "loop index is bounded by a 4-element array; precision loss cannot occur in practice"
    )]
    pub fn stroke_styles() -> Gm {
        Gm::new("stroke_styles", 300, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            let mut paint = Paint::new();
            paint.set_color32(Color::BLACK);
            paint.set_style(skia_rs_paint::Style::Stroke);

            // Different stroke widths
            for (i, width) in [1.0, 2.0, 4.0, 8.0].iter().enumerate() {
                paint.set_stroke_width(*width);
                let y = (i as f32).mul_add(40.0, 30.0);
                canvas.draw_rect(&Rect::from_xywh(20.0, y, 260.0, 0.0), &paint);
            }
        })
        .with_tag("stroke")
    }

    /// Antialiasing GM
    #[must_use]
    pub fn antialiasing() -> Gm {
        Gm::new("antialiasing", 200, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            // Without AA
            let mut paint = Paint::new();
            paint.set_color32(Color::BLACK);
            paint.set_anti_alias(false);

            let mut builder = PathBuilder::new();
            builder.move_to(20.0, 100.0);
            builder.line_to(80.0, 20.0);
            builder.line_to(80.0, 180.0);
            builder.close();
            canvas.draw_path(&builder.build(), &paint);

            // With AA
            paint.set_anti_alias(true);

            let mut builder = PathBuilder::new();
            builder.move_to(120.0, 100.0);
            builder.line_to(180.0, 20.0);
            builder.line_to(180.0, 180.0);
            builder.close();
            canvas.draw_path(&builder.build(), &paint);
        })
        .with_tag("aa")
    }

    /// Alpha blending GM
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "loop index is bounded by a 5-element array; precision loss cannot occur in practice"
    )]
    pub fn alpha_blending() -> Gm {
        Gm::new("alpha_blending", 200, 200, |surface| {
            let mut canvas = surface.raster_canvas();
            canvas.clear(Color::WHITE);

            // Overlapping semi-transparent rectangles
            let mut paint = Paint::new();

            let alphas = [255, 200, 150, 100, 50];
            for (i, alpha) in alphas.iter().enumerate() {
                paint.set_color32(Color::from_argb(*alpha, 100, 100, 200));
                let offset = i as f32 * 25.0;
                canvas.draw_rect(
                    &Rect::from_xywh(20.0 + offset, 20.0 + offset, 100.0, 100.0),
                    &paint,
                );
            }
        })
        .with_tag("alpha")
    }
}

// =============================================================================
// Built-in Sinks
// =============================================================================

/// PNG file output sink
pub struct PngSink {
    output_dir: PathBuf,
}

impl PngSink {
    /// Create a new PNG sink
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }
}

impl Sink for PngSink {
    fn name(&self) -> &'static str {
        "png"
    }

    fn process(&self, result: &TestResult) -> Result<(), SinkError> {
        if result.outcome != TestOutcome::Pass {
            return Ok(());
        }

        let Some(pixels) = &result.pixels else {
            return Ok(());
        };

        // Create output directory
        std::fs::create_dir_all(&self.output_dir).map_err(|e| SinkError {
            message: format!("Failed to create output directory: {e}"),
        })?;

        let filename = format!("{}_{}.png", result.source, result.renderer);
        let path = self.output_dir.join(&filename);

        // Write PNG (simplified - would use actual PNG encoder)
        std::fs::write(&path, pixels).map_err(|e| SinkError {
            message: format!("Failed to write PNG: {e}"),
        })?;

        Ok(())
    }
}

/// Comparison sink that compares against reference images
pub struct ComparisonSink {
    reference_dir: PathBuf,
    tolerance: f32,
    results: std::sync::Mutex<Vec<ComparisonResult>>,
}

/// Result of image comparison
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Test name
    pub name: String,
    /// Whether images match
    pub matches: bool,
    /// Difference percentage (0.0 to 1.0)
    pub difference: f32,
    /// Number of differing pixels
    pub diff_pixels: u32,
}

impl ComparisonSink {
    /// Create a new comparison sink
    pub fn new(reference_dir: impl AsRef<Path>, tolerance: f32) -> Self {
        Self {
            reference_dir: reference_dir.as_ref().to_path_buf(),
            tolerance,
            results: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Get comparison results
    ///
    /// # Panics
    ///
    /// Panics if the internal results mutex is poisoned (i.e. a prior lock holder panicked).
    #[must_use]
    pub fn results(&self) -> Vec<ComparisonResult> {
        self.results.lock().unwrap().clone()
    }
}

impl Sink for ComparisonSink {
    fn name(&self) -> &'static str {
        "comparison"
    }

    /// # Errors
    ///
    /// This sink never returns an error; the signature matches [`Sink::process`].
    #[allow(
        clippy::cast_precision_loss,
        reason = "pixel diff counts are bounded by benchmark image sizes; precision loss is immaterial for a reported ratio"
    )]
    fn process(&self, result: &TestResult) -> Result<(), SinkError> {
        if result.outcome != TestOutcome::Pass {
            return Ok(());
        }

        let Some(pixels) = &result.pixels else {
            return Ok(());
        };

        let filename = format!("{}_{}.png", result.source, result.renderer);
        let ref_path = self.reference_dir.join(&filename);

        // Load reference image
        let reference = std::fs::read(&ref_path).ok();

        let threshold = u32::from(skia_rs_core::cast::f32_to_u8_sat(self.tolerance * 255.0));

        let comparison = reference.map_or_else(
            || ComparisonResult {
                // No reference - consider it a new test
                name: format!("{}_{}", result.source, result.renderer),
                matches: true, // New tests pass by default
                difference: 0.0,
                diff_pixels: 0,
            },
            |ref_pixels| {
                // Compare pixels
                let mut diff_count = 0u32;
                let total = pixels.len().min(ref_pixels.len());

                for i in 0..total {
                    let diff = (i32::from(pixels[i]) - i32::from(ref_pixels[i])).unsigned_abs();
                    if diff > threshold {
                        diff_count += 1;
                    }
                }

                let diff_ratio = diff_count as f32 / total as f32;

                ComparisonResult {
                    name: format!("{}_{}", result.source, result.renderer),
                    matches: diff_ratio <= self.tolerance,
                    difference: diff_ratio,
                    diff_pixels: diff_count,
                }
            },
        );

        self.results.lock().unwrap().push(comparison);
        Ok(())
    }
}

// =============================================================================
// Test Runner (DM)
// =============================================================================

/// DM-style test runner
pub struct DmRunner {
    sources: Vec<Arc<dyn Source>>,
    renderers: Vec<Arc<dyn Renderer>>,
    sinks: Vec<Arc<dyn Sink>>,
    filter: Option<String>,
    parallel: bool,
}

impl Default for DmRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl DmRunner {
    /// Create a new DM runner
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            renderers: vec![Arc::new(RasterRenderer::new())],
            sinks: Vec::new(),
            filter: None,
            parallel: false,
        }
    }

    /// Add a source
    pub fn add_source(&mut self, source: Arc<dyn Source>) {
        self.sources.push(source);
    }

    /// Add multiple sources
    pub fn add_sources(&mut self, sources: impl IntoIterator<Item = Arc<dyn Source>>) {
        self.sources.extend(sources);
    }

    /// Add a renderer
    pub fn add_renderer(&mut self, renderer: Arc<dyn Renderer>) {
        self.renderers.push(renderer);
    }

    /// Add a sink
    pub fn add_sink(&mut self, sink: Arc<dyn Sink>) {
        self.sinks.push(sink);
    }

    /// Set name filter
    pub fn set_filter(&mut self, filter: &str) {
        self.filter = Some(filter.to_string());
    }

    /// Enable parallel execution
    pub const fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Run all tests
    #[must_use]
    pub fn run(&self) -> DmReport {
        let mut report = DmReport::new();
        let start = Instant::now();

        for source in &self.sources {
            // Apply filter
            if let Some(filter) = &self.filter
                && !source.name().contains(filter)
            {
                continue;
            }

            for renderer in &self.renderers {
                let result = Self::run_single(source.as_ref(), renderer.as_ref());

                // Process through sinks
                for sink in &self.sinks {
                    if let Err(e) = sink.process(&result) {
                        eprintln!("Sink {} error: {}", sink.name(), e);
                    }
                }

                report.add_result(result);
            }
        }

        report.total_duration = start.elapsed();
        report
    }

    fn run_single(source: &dyn Source, renderer: &dyn Renderer) -> TestResult {
        let (width, height) = source.size();
        let start = Instant::now();

        // Create surface
        let surface = renderer.create_surface(width, height);

        let (outcome, pixels, error) = surface.map_or_else(
            || {
                (
                    TestOutcome::Skip,
                    None,
                    Some("Could not create surface".to_string()),
                )
            },
            |mut surface| {
                // Catch panics
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    source.draw(&mut surface);
                }));

                match result {
                    Ok(()) => {
                        let pixels = surface.pixels().to_vec();
                        (TestOutcome::Pass, Some(pixels), None)
                    }
                    Err(_) => (
                        TestOutcome::Crash,
                        None,
                        Some("Panic during draw".to_string()),
                    ),
                }
            },
        );

        TestResult {
            source: source.name().to_string(),
            renderer: renderer.name().to_string(),
            outcome,
            duration: start.elapsed(),
            pixels,
            width,
            height,
            error,
        }
    }
}

/// Report of a DM run
#[derive(Debug)]
pub struct DmReport {
    /// All test results
    pub results: Vec<TestResult>,
    /// Total duration
    pub total_duration: Duration,
    /// Stats
    pub stats: DmStats,
}

/// Statistics from a DM run
#[derive(Debug, Default)]
pub struct DmStats {
    /// Total tests
    pub total: u32,
    /// Passed tests
    pub passed: u32,
    /// Failed tests
    pub failed: u32,
    /// Skipped tests
    pub skipped: u32,
    /// Crashed tests
    pub crashed: u32,
}

impl DmReport {
    fn new() -> Self {
        Self {
            results: Vec::new(),
            total_duration: Duration::ZERO,
            stats: DmStats::default(),
        }
    }

    fn add_result(&mut self, result: TestResult) {
        self.stats.total += 1;
        match result.outcome {
            TestOutcome::Pass => self.stats.passed += 1,
            TestOutcome::Fail => self.stats.failed += 1,
            TestOutcome::Skip => self.stats.skipped += 1,
            TestOutcome::Crash => self.stats.crashed += 1,
        }
        self.results.push(result);
    }

    /// Generate a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "DM Report: {} total, {} passed, {} failed, {} skipped, {} crashed in {:?}",
            self.stats.total,
            self.stats.passed,
            self.stats.failed,
            self.stats.skipped,
            self.stats.crashed,
            self.total_duration
        )
    }

    /// Check if all tests passed
    #[must_use]
    pub const fn all_passed(&self) -> bool {
        self.stats.failed == 0 && self.stats.crashed == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dm_runner() {
        let mut runner = DmRunner::new();
        runner.add_sources(StandardGms::all());

        let report = runner.run();
        println!("{}", report.summary());

        assert!(report.stats.total > 0);
        assert!(report.stats.passed > 0);
    }

    #[test]
    fn test_standard_gms() {
        let gms = StandardGms::all();
        assert!(!gms.is_empty());

        for gm in gms {
            let (w, h) = gm.size();
            assert!(w > 0 && h > 0);
            assert_ne!(gm.name(), "");
        }
    }

    #[test]
    fn test_single_gm() {
        let gm = StandardGms::simple_rect();
        let renderer = RasterRenderer::new();

        let mut surface = renderer.create_surface(200, 200).unwrap();
        gm.draw(&mut surface);

        let pixels = surface.pixels();
        assert_ne!(pixels, []);
    }
}
