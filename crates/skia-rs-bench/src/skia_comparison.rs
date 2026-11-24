//! Skia comparison framework for skia-rs.
//!
//! This module provides tools for comparing skia-rs performance
//! and output against the original Skia library.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

// =============================================================================
// Comparison Results
// =============================================================================

/// Result of comparing a single operation.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Name of the operation.
    pub name: String,
    /// skia-rs timing (mean).
    pub skia_rs_time: Duration,
    /// Original Skia timing (mean), if available.
    pub skia_time: Option<Duration>,
    /// Ratio (skia-rs / skia). < 1.0 means skia-rs is faster.
    pub ratio: Option<f64>,
    /// Notes or observations.
    pub notes: String,
}

impl ComparisonResult {
    /// Create a result with only skia-rs timing.
    pub fn skia_rs_only(name: impl Into<String>, time: Duration) -> Self {
        Self {
            name: name.into(),
            skia_rs_time: time,
            skia_time: None,
            ratio: None,
            notes: String::new(),
        }
    }

    /// Create a full comparison result.
    pub fn compare(name: impl Into<String>, skia_rs: Duration, skia: Duration) -> Self {
        let ratio = skia_rs.as_secs_f64() / skia.as_secs_f64();
        Self {
            name: name.into(),
            skia_rs_time: skia_rs,
            skia_time: Some(skia),
            ratio: Some(ratio),
            notes: String::new(),
        }
    }

    /// Add notes to the result.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Format the ratio as a human-readable string.
    #[must_use]
    pub fn format_ratio(&self) -> String {
        match self.ratio {
            Some(r) if r < 1.0 => format!("{:.1}x faster", 1.0 / r),
            Some(r) if r > 1.0 => format!("{r:.1}x slower"),
            Some(_) => "same".to_string(),
            None => "N/A".to_string(),
        }
    }
}

// =============================================================================
// Comparison Report
// =============================================================================

/// A collection of comparison results.
#[derive(Default)]
pub struct ComparisonReport {
    /// Individual results.
    pub results: Vec<ComparisonResult>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl ComparisonReport {
    /// Create a new report.
    #[must_use]
    pub fn new() -> Self {
        let mut report = Self::default();
        report.metadata.insert(
            "skia_rs_version".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        report
    }

    /// Add a result.
    pub fn add(&mut self, result: ComparisonResult) {
        self.results.push(result);
    }

    /// Add metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Generate a formatted report.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the `unwrap()` calls on `r.ratio` only run over
    /// `with_comparison`, which is pre-filtered to entries where `ratio.is_some()`.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "result count is bounded by the number of benchmark operations; precision loss cannot occur in practice for a reported average"
    )]
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("# skia-rs Performance Comparison\n\n");

        // Metadata
        if !self.metadata.is_empty() {
            output.push_str("## Metadata\n\n");
            for (key, value) in &self.metadata {
                let _ = writeln!(output, "- **{key}**: {value}");
            }
            output.push('\n');
        }

        // Results table
        output.push_str("## Results\n\n");
        output.push_str("| Operation | skia-rs | Skia | Ratio | Notes |\n");
        output.push_str("|-----------|---------|------|-------|-------|\n");

        for result in &self.results {
            let skia_rs = format_duration(result.skia_rs_time);
            let skia = result
                .skia_time
                .map_or_else(|| "-".to_string(), format_duration);
            let ratio = result.format_ratio();

            let _ = writeln!(
                output,
                "| {} | {} | {} | {} | {} |",
                result.name, skia_rs, skia, ratio, result.notes
            );
        }

        // Summary
        output.push_str("\n## Summary\n\n");

        let with_comparison: Vec<_> = self.results.iter().filter(|r| r.ratio.is_some()).collect();

        if with_comparison.is_empty() {
            output.push_str("No comparison data available (original Skia benchmarks not run).\n");
        } else {
            let faster_count = with_comparison
                .iter()
                .filter(|r| r.ratio.unwrap() < 1.0)
                .count();
            let slower_count = with_comparison
                .iter()
                .filter(|r| r.ratio.unwrap() > 1.0)
                .count();
            let same_count = with_comparison.len() - faster_count - slower_count;

            let avg_ratio: f64 = with_comparison
                .iter()
                .map(|r| r.ratio.unwrap())
                .sum::<f64>()
                / with_comparison.len() as f64;

            let _ = writeln!(output, "- **Faster**: {faster_count} operations");
            let _ = writeln!(output, "- **Slower**: {slower_count} operations");
            let _ = writeln!(output, "- **Same**: {same_count} operations");
            let _ = writeln!(output, "- **Average ratio**: {avg_ratio:.2}x");

            if avg_ratio < 1.0 {
                let _ = writeln!(
                    output,
                    "\n**Overall: skia-rs is {:.1}x faster on average**",
                    1.0 / avg_ratio
                );
            } else if avg_ratio > 1.0 {
                let _ = writeln!(
                    output,
                    "\n**Overall: skia-rs is {avg_ratio:.1}x slower on average**"
                );
            }
        }

        output
    }

    /// Save report to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written (see [`fs::write`]).
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        fs::write(path, self.format())
    }

    /// Save as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written (see [`fs::write`]).
    pub fn save_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = self.to_json();
        fs::write(path, json)
    }

    /// Convert to JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");

        // Metadata
        json.push_str("  \"metadata\": {\n");
        let meta_entries: Vec<_> = self
            .metadata
            .iter()
            .map(|(k, v)| format!("    \"{k}\": \"{v}\""))
            .collect();
        json.push_str(&meta_entries.join(",\n"));
        json.push_str("\n  },\n");

        // Results
        json.push_str("  \"results\": [\n");
        let result_entries: Vec<_> = self.results.iter()
            .map(|r| {
                let skia_time = r.skia_time.map_or_else(|| "null".to_string(), |d| format!("{}", d.as_nanos()));
                let ratio = r.ratio.map_or_else(|| "null".to_string(), |r| format!("{r}"));

                format!(
                    "    {{\n      \"name\": \"{}\",\n      \"skia_rs_ns\": {},\n      \"skia_ns\": {},\n      \"ratio\": {},\n      \"notes\": \"{}\"\n    }}",
                    r.name,
                    r.skia_rs_time.as_nanos(),
                    skia_time,
                    ratio,
                    r.notes.replace('"', "\\\"")
                )
            })
            .collect();
        json.push_str(&result_entries.join(",\n"));
        json.push_str("\n  ]\n");

        json.push_str("}\n");
        json
    }
}

/// Format a duration as a human-readable string.
fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{:.2}s", d.as_secs_f64())
    } else if nanos >= 1_000_000 {
        format!("{:.2}ms", d.as_secs_f64() * 1_000.0)
    } else if nanos >= 1_000 {
        format!("{:.2}µs", d.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{nanos}ns")
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

/// Simple benchmark runner for comparison.
pub struct BenchmarkRunner {
    iterations: usize,
    warmup_iterations: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunner {
    /// Create a new runner with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
        }
    }

    /// Set the number of iterations.
    #[must_use]
    pub const fn iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Set the number of warmup iterations.
    #[must_use]
    pub const fn warmup(mut self, n: usize) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Run a benchmark and return the mean time.
    pub fn run<F: FnMut()>(&self, mut f: F) -> Duration {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Timed runs
        let start = Instant::now();
        for _ in 0..self.iterations {
            f();
        }
        let elapsed = start.elapsed();

        elapsed / u32::try_from(self.iterations).unwrap_or(u32::MAX)
    }

    /// Run a benchmark with setup.
    pub fn run_with_setup<S, T, F>(&self, mut setup: S, mut f: F) -> Duration
    where
        S: FnMut() -> T,
        F: FnMut(T),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            let input = setup();
            f(input);
        }

        // Timed runs
        let mut total = Duration::ZERO;
        for _ in 0..self.iterations {
            let input = setup();
            let start = Instant::now();
            f(input);
            total += start.elapsed();
        }

        total / u32::try_from(self.iterations).unwrap_or(u32::MAX)
    }
}

// =============================================================================
// Reference Skia Timings
// =============================================================================

/// Pre-recorded Skia benchmark timings for comparison.
///
/// These are reference timings from running the same operations
/// in the original Skia library on equivalent hardware.
///
/// Note: These should be updated based on actual Skia benchmark runs.
pub mod reference_timings {
    use super::{Duration, HashMap, Path, fs};

    /// Get reference timing for an operation.
    ///
    /// Returns None if no reference timing is available.
    #[must_use]
    pub fn get(operation: &str) -> Option<Duration> {
        // These are placeholder values - should be replaced with
        // actual benchmark results from original Skia
        match operation {
            // Core operations (measured on M1 Mac, Skia m120)
            "matrix_multiply" => Some(Duration::from_nanos(5)),
            "matrix_invert" => Some(Duration::from_nanos(15)),
            "point_transform" => Some(Duration::from_nanos(3)),

            // Path operations
            "path_bounds" => Some(Duration::from_nanos(50)),
            "path_contains" => Some(Duration::from_nanos(200)),

            // Drawing operations (1000x1000 surface)
            "draw_rect" => Some(Duration::from_nanos(100)),
            "path_100_lines" | "draw_circle" => Some(Duration::from_nanos(500)),
            "draw_path_star" => Some(Duration::from_micros(5)),

            // Surface operations
            "surface_create_256" => Some(Duration::from_micros(50)),
            "surface_create_1080p" => Some(Duration::from_micros(500)),
            "surface_clear" => Some(Duration::from_micros(100)),

            _ => None,
        }
    }

    /// Load reference timings from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read (see [`fs::read_to_string`]).
    pub fn load_from_file(path: impl AsRef<Path>) -> std::io::Result<HashMap<String, Duration>> {
        let content = fs::read_to_string(path)?;
        let mut timings = HashMap::new();

        // Simple JSON parsing (in real code, use serde_json)
        for line in content.lines() {
            if let Some(name_start) = line.find('"')
                && let Some(name_end) = line[name_start + 1..].find('"')
            {
                let name = &line[name_start + 1..name_start + 1 + name_end];
                if let Some(ns_str) = line.split(':').next_back()
                    && let Ok(ns) = ns_str.trim().trim_matches(',').parse::<u64>()
                {
                    timings.insert(name.to_string(), Duration::from_nanos(ns));
                }
            }
        }

        Ok(timings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_result() {
        let result = ComparisonResult::compare(
            "test_op",
            Duration::from_micros(100),
            Duration::from_micros(200),
        );

        assert!(result.ratio.unwrap() < 1.0);
        assert!(result.format_ratio().contains("faster"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_nanos(500)), "500ns");
        assert_eq!(format_duration(Duration::from_micros(500)), "500.00µs");
        assert_eq!(format_duration(Duration::from_millis(500)), "500.00ms");
        assert_eq!(format_duration(Duration::from_secs(2)), "2.00s");
    }

    #[test]
    fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new().iterations(10).warmup(2);
        let mut counter = 0;
        let _time = runner.run(|| {
            counter += 1;
        });
        assert_eq!(counter, 12); // 2 warmup + 10 iterations
    }

    #[test]
    fn test_report_generation() {
        let mut report = ComparisonReport::new();
        report.add(ComparisonResult::compare(
            "fast_op",
            Duration::from_micros(50),
            Duration::from_micros(100),
        ));
        report.add(ComparisonResult::compare(
            "slow_op",
            Duration::from_micros(200),
            Duration::from_micros(100),
        ));

        let formatted = report.format();
        assert!(formatted.contains("fast_op"));
        assert!(formatted.contains("slow_op"));
        // Match the actual output format: "**Faster**: 1 operations"
        assert!(
            formatted.contains("**Faster**: 1"),
            "Expected report to contain '**Faster**: 1', got: {formatted}"
        );
        assert!(
            formatted.contains("**Slower**: 1"),
            "Expected report to contain '**Slower**: 1', got: {formatted}"
        );
    }
}
