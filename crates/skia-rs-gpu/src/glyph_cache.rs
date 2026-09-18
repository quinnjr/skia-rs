//! Glyph cache for GPU text rendering.
//!
//! This module provides a cache for rasterized glyphs, managing their
//! storage in texture atlases for efficient GPU rendering.

use crate::atlas::{AtlasAllocResult, AtlasConfig, AtlasEntryId, AtlasRegion, TextureAtlas};
use crate::cast_util::{f64_from_u64, scalar_from_u32, u8_from_scalar_sat, u32_from_scalar_sat};
use skia_rs_core::{Point, Rect, Scalar};
use std::collections::HashMap;
use std::hash::Hash;

/// A unique key for identifying a glyph in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Font ID.
    pub font_id: u32,
    /// Glyph ID within the font.
    pub glyph_id: u32,
    /// Font size in pixels (quantized).
    pub size_px: u32,
    /// Sub-pixel position (0-3 for 1/4 pixel precision).
    pub sub_pixel_x: u8,
    /// Sub-pixel position (0-3 for 1/4 pixel precision).
    pub sub_pixel_y: u8,
    /// Additional flags (bold, italic, etc.).
    pub flags: u8,
}

impl GlyphKey {
    /// Create a new glyph key.
    #[must_use]
    pub fn new(font_id: u32, glyph_id: u32, size: f32, sub_pixel: Point) -> Self {
        Self {
            font_id,
            glyph_id,
            size_px: u32_from_scalar_sat(size * 4.0), // Quarter pixel precision
            sub_pixel_x: u8_from_scalar_sat(sub_pixel.x.fract() * 4.0).min(3),
            sub_pixel_y: u8_from_scalar_sat(sub_pixel.y.fract() * 4.0).min(3),
            flags: 0,
        }
    }

    /// Create with flags.
    #[must_use]
    pub const fn with_flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }
}

/// Cached glyph data.
#[derive(Debug, Clone)]
pub struct CachedGlyph {
    /// Atlas entry id backing `region`, used to free the atlas slot when the
    /// glyph is evicted so the space is actually reclaimed (not merely
    /// dropped from the lookup map).
    pub entry_id: AtlasEntryId,
    /// Atlas region containing the glyph.
    pub region: AtlasRegion,
    /// Glyph metrics: offset from baseline.
    pub offset: Point,
    /// Glyph advance width.
    pub advance: Scalar,
    /// Glyph bounding box (local).
    pub bounds: Rect,
}

/// Glyph cache statistics.
#[derive(Debug, Clone, Default)]
pub struct GlyphCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Current number of cached glyphs.
    pub cached_count: usize,
}

impl GlyphCacheStats {
    /// Calculate hit rate.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            f64_from_u64(self.hits) / f64_from_u64(total)
        }
    }
}

/// Glyph cache configuration.
#[derive(Debug, Clone)]
pub struct GlyphCacheConfig {
    /// Maximum number of cached glyphs.
    pub max_glyphs: usize,
    /// Atlas configuration.
    pub atlas_config: AtlasConfig,
    /// Enable sub-pixel rendering.
    pub sub_pixel_rendering: bool,
}

impl Default for GlyphCacheConfig {
    fn default() -> Self {
        Self {
            max_glyphs: 4096,
            atlas_config: AtlasConfig {
                width: 1024,
                height: 1024,
                max_layers: 4,
                padding: 1,
                allow_resize: true,
            },
            sub_pixel_rendering: true,
        }
    }
}

/// A glyph cache for GPU rendering.
pub struct GlyphCache {
    /// Configuration.
    config: GlyphCacheConfig,
    /// Glyph atlas.
    atlas: TextureAtlas,
    /// Cached glyphs by key.
    cache: HashMap<GlyphKey, CachedGlyph>,
    /// LRU order (front = most recently used).
    lru_order: Vec<GlyphKey>,
    /// Statistics.
    stats: GlyphCacheStats,
}

impl GlyphCache {
    /// Create a new glyph cache.
    #[must_use]
    pub fn new(config: GlyphCacheConfig) -> Self {
        let atlas = TextureAtlas::new(config.atlas_config.clone());
        Self {
            config,
            atlas,
            cache: HashMap::new(),
            lru_order: Vec::new(),
            stats: GlyphCacheStats::default(),
        }
    }

    /// Get cache configuration.
    #[must_use]
    pub const fn config(&self) -> &GlyphCacheConfig {
        &self.config
    }

    /// Get cache statistics.
    #[must_use]
    pub const fn stats(&self) -> &GlyphCacheStats {
        &self.stats
    }

    /// Get the glyph atlas.
    #[must_use]
    pub const fn atlas(&self) -> &TextureAtlas {
        &self.atlas
    }

    /// Look up a glyph in the cache.
    pub fn lookup(&mut self, key: &GlyphKey) -> Option<&CachedGlyph> {
        if let Some(glyph) = self.cache.get(key) {
            // Update LRU
            if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
                let key = self.lru_order.remove(pos);
                self.lru_order.insert(0, key);
            }
            self.stats.hits += 1;
            Some(glyph)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check if a glyph is cached without updating LRU.
    #[must_use]
    pub fn contains(&self, key: &GlyphKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Insert a glyph into the cache.
    ///
    /// Returns the atlas region where the glyph data should be uploaded.
    pub fn insert(
        &mut self,
        key: GlyphKey,
        width: u32,
        height: u32,
        offset: Point,
        advance: Scalar,
    ) -> Option<AtlasRegion> {
        // Check if already cached
        if self.cache.contains_key(&key) {
            return self.cache.get(&key).map(|g| g.region);
        }

        // Evict if at capacity
        while self.cache.len() >= self.config.max_glyphs {
            if !self.evict_lru() {
                break;
            }
        }

        // Allocate in atlas. The loop is guaranteed to terminate: every
        // iteration either succeeds/returns, evicts+compacts (reclaiming
        // space), or performs a full reset. A glyph that can never fit
        // (padding included) returns TooLarge on the first call.
        let (entry_id, region) = loop {
            match self.atlas.allocate(width, height) {
                AtlasAllocResult::Success(id, region) => break (id, region),
                AtlasAllocResult::TooLarge => {
                    // Glyph too large for atlas — do not spin.
                    return None;
                }
                AtlasAllocResult::Full => {
                    if self.evict_lru() {
                        // Evicting marked the region freed; compact to
                        // actually reclaim the space, then retry.
                        self.compact_atlas();
                    } else {
                        // Nothing left to evict: hard-reset the atlas. This
                        // bumps the atlas generation, invalidating any
                        // outstanding GlyphBatch (which must revalidate).
                        self.reset();
                    }
                }
            }
        };

        let bounds = Rect::from_xywh(0.0, 0.0, scalar_from_u32(width), scalar_from_u32(height));

        let glyph = CachedGlyph {
            entry_id,
            region,
            offset,
            advance,
            bounds,
        };

        self.cache.insert(key, glyph);
        self.lru_order.insert(0, key);
        self.stats.cached_count = self.cache.len();

        Some(region)
    }

    /// Evict the least recently used glyph, freeing its atlas region so the
    /// space can be reclaimed by the next `compact()`.
    fn evict_lru(&mut self) -> bool {
        if let Some(key) = self.lru_order.pop() {
            if let Some(glyph) = self.cache.remove(&key) {
                self.atlas.free(glyph.entry_id);
            }
            self.stats.evictions += 1;
            self.stats.cached_count = self.cache.len();
            true
        } else {
            false
        }
    }

    /// Compact the atlas, reclaiming freed regions, and reconcile the cache:
    /// remapped entries have their cached `region` updated; entries dropped by
    /// the compaction (freed or unplaceable) are removed from the cache.
    ///
    /// The atlas generation is bumped by `compact()`, so any previously
    /// emitted `GlyphBatch` becomes stale and must be revalidated against
    /// [`TextureAtlas::generation`] before being drawn.
    fn compact_atlas(&mut self) {
        let result = self.atlas.compact();

        if !result.remapped.is_empty() {
            let moved: HashMap<AtlasEntryId, AtlasRegion> = result
                .remapped
                .iter()
                .map(|(id, _old, new)| (*id, *new))
                .collect();
            for glyph in self.cache.values_mut() {
                if let Some(new_region) = moved.get(&glyph.entry_id) {
                    glyph.region = *new_region;
                }
            }
        }

        if !result.removed.is_empty() {
            let removed: std::collections::HashSet<AtlasEntryId> =
                result.removed.iter().copied().collect();
            let drop_keys: Vec<GlyphKey> = self
                .cache
                .iter()
                .filter(|(_, g)| removed.contains(&g.entry_id))
                .map(|(k, _)| *k)
                .collect();
            for k in drop_keys {
                self.cache.remove(&k);
                if let Some(pos) = self.lru_order.iter().position(|x| x == &k) {
                    self.lru_order.remove(pos);
                }
            }
            self.stats.cached_count = self.cache.len();
        }
    }

    /// Current atlas generation. Emit it into a [`GlyphBatch`] and revalidate
    /// batches against it before drawing (a reset or compaction bumps it,
    /// invalidating outstanding UV coordinates).
    #[must_use]
    pub const fn atlas_generation(&self) -> u64 {
        self.atlas.generation()
    }

    /// Reset the cache, clearing all entries.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.atlas.reset();
        self.stats.cached_count = 0;
    }

    /// Get number of cached glyphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new(GlyphCacheConfig::default())
    }
}

/// Validation state of a [`GlyphBatch`] against the current atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchValidity {
    /// The batch's UV coordinates still address the current atlas contents.
    Current,
    /// The atlas has been reset or compacted since the batch was built; its
    /// UVs are stale and it must be rebuilt before drawing.
    Stale,
}

/// A batch of glyphs to render.
#[derive(Debug, Clone)]
pub struct GlyphBatch {
    /// Atlas generation this batch was created for.
    pub atlas_generation: u64,
    /// Glyph instances.
    pub instances: Vec<GlyphInstance>,
}

/// A single glyph instance for rendering.
#[derive(Debug, Clone, Copy)]
pub struct GlyphInstance {
    /// Position on screen.
    pub position: Point,
    /// Atlas region UV coordinates [u0, v0, u1, v1].
    pub uv: [f32; 4],
    /// Glyph size.
    pub size: [f32; 2],
    /// Color (RGBA).
    pub color: [f32; 4],
    /// Atlas layer.
    pub layer: u32,
}

impl GlyphBatch {
    /// Create a new empty batch.
    #[must_use]
    pub const fn new(atlas_generation: u64) -> Self {
        Self {
            atlas_generation,
            instances: Vec::new(),
        }
    }

    /// Add a glyph instance.
    pub fn add_glyph(
        &mut self,
        glyph: &CachedGlyph,
        position: Point,
        color: [f32; 4],
        atlas_size: (u32, u32),
    ) {
        let uv = glyph.region.uv_rect(atlas_size.0, atlas_size.1);

        self.instances.push(GlyphInstance {
            position: Point::new(position.x + glyph.offset.x, position.y + glyph.offset.y),
            uv,
            size: [
                scalar_from_u32(glyph.region.width),
                scalar_from_u32(glyph.region.height),
            ],
            color,
            layer: glyph.region.layer,
        });
    }

    /// Check if batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Get number of glyphs in batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Clear the batch.
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    /// Validate the batch against the current atlas generation.
    ///
    /// Consumers MUST call this before drawing: if the atlas was reset or
    /// compacted (its generation bumped) since the batch was built, the
    /// batch's UV coordinates point at texels that may now belong to other
    /// glyphs. A [`BatchValidity::Stale`] result means the batch must be
    /// rebuilt from the (repopulated) cache.
    #[must_use]
    pub const fn validate(&self, current_generation: u64) -> BatchValidity {
        if self.atlas_generation == current_generation {
            BatchValidity::Current
        } else {
            BatchValidity::Stale
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glyph_key() {
        let key = GlyphKey::new(1, 65, 16.0, Point::new(0.25, 0.5));
        assert_eq!(key.font_id, 1);
        assert_eq!(key.glyph_id, 65);
        assert_eq!(key.size_px, 64); // 16 * 4
        assert_eq!(key.sub_pixel_x, 1); // 0.25 * 4
        assert_eq!(key.sub_pixel_y, 2); // 0.5 * 4
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_glyph_cache_insert_lookup() {
        let mut cache = GlyphCache::default();

        let key = GlyphKey::new(1, 65, 16.0, Point::zero());
        let region = cache.insert(key, 16, 20, Point::new(0.0, -15.0), 10.0);

        assert!(region.is_some());
        assert_eq!(cache.len(), 1);

        let glyph = cache.lookup(&key);
        assert!(glyph.is_some());
        assert_eq!(glyph.unwrap().advance, 10.0);
    }

    #[test]
    fn test_glyph_cache_eviction() {
        let config = GlyphCacheConfig {
            max_glyphs: 3,
            ..Default::default()
        };
        let mut cache = GlyphCache::new(config);

        // Insert 4 glyphs, should evict first
        for i in 0..4 {
            let key = GlyphKey::new(1, i, 16.0, Point::zero());
            cache.insert(key, 16, 16, Point::zero(), 10.0);
        }

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_glyph_cache_stats() {
        let mut cache = GlyphCache::default();

        let key = GlyphKey::new(1, 65, 16.0, Point::zero());
        cache.insert(key, 16, 16, Point::zero(), 10.0);

        // Miss
        let miss_key = GlyphKey::new(1, 66, 16.0, Point::zero());
        cache.lookup(&miss_key);

        // Hit
        cache.lookup(&key);

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact literal values, no accumulated error"
    )]
    fn test_glyph_batch() {
        let mut batch = GlyphBatch::new(0);
        assert!(batch.is_empty());

        let glyph = CachedGlyph {
            entry_id: AtlasEntryId::new(0),
            region: AtlasRegion {
                x: 0,
                y: 0,
                width: 16,
                height: 20,
                layer: 0,
            },
            offset: Point::new(0.0, -15.0),
            advance: 10.0,
            bounds: Rect::from_xywh(0.0, 0.0, 16.0, 20.0),
        };

        batch.add_glyph(
            &glyph,
            Point::new(100.0, 100.0),
            [1.0, 1.0, 1.0, 1.0],
            (1024, 1024),
        );

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.instances[0].position.x, 100.0);
        assert_eq!(batch.instances[0].position.y, 85.0); // 100 + (-15)
    }

    #[test]
    fn test_oversized_glyph_returns_none_no_infinite_loop() {
        // Regression: a glyph larger than the atlas (padding included) must
        // return None immediately instead of spinning the retry loop.
        let config = GlyphCacheConfig {
            max_glyphs: 16,
            atlas_config: AtlasConfig {
                width: 64,
                height: 64,
                max_layers: 1,
                padding: 1,
                allow_resize: false,
            },
            sub_pixel_rendering: false,
        };
        let mut cache = GlyphCache::new(config);
        let key = GlyphKey::new(1, 1, 16.0, Point::zero());
        // 64 + 2*1 = 66 > 64 -> TooLarge -> None.
        assert!(cache.insert(key, 64, 10, Point::zero(), 5.0).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_atlas_full_reclaims_via_eviction() {
        // Regression: when the atlas fills, eviction must free atlas regions
        // (not just drop cache entries) so new glyphs keep being placeable.
        let config = GlyphCacheConfig {
            max_glyphs: 100_000, // do not cap on cache size; force atlas pressure
            atlas_config: AtlasConfig {
                width: 128,
                height: 128,
                max_layers: 1,
                padding: 1,
                allow_resize: false,
            },
            sub_pixel_rendering: false,
        };
        let mut cache = GlyphCache::new(config);

        // Insert many 30x30 glyphs; a 128x128 atlas holds ~16 at a time, so
        // this forces repeated eviction/compaction. Every insert must place.
        for i in 0..200u32 {
            let key = GlyphKey::new(1, i, 16.0, Point::zero());
            let r = cache.insert(key, 30, 30, Point::zero(), 5.0);
            assert!(r.is_some(), "insert {i} failed to place a fitting glyph");
        }
        // The cache is bounded by atlas capacity via eviction, not unbounded.
        assert!(
            cache.len() <= 32,
            "cache should be bounded by atlas capacity"
        );
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_batch_validation_detects_stale_generation() {
        // Regression: GlyphBatch must be validatable against the atlas
        // generation; a bumped generation marks the batch stale.
        let generation = 7;
        let batch = GlyphBatch::new(generation);
        assert_eq!(batch.validate(generation), BatchValidity::Current);
        assert_eq!(batch.validate(generation), BatchValidity::Current);
        assert_eq!(batch.validate(generation + 1), BatchValidity::Stale);
        assert!(!(batch.validate(generation + 1) == BatchValidity::Current));
    }

    #[test]
    fn test_reset_bumps_atlas_generation() {
        let mut cache = GlyphCache::default();
        let g0 = cache.atlas_generation();
        cache.insert(
            GlyphKey::new(1, 1, 16.0, Point::zero()),
            16,
            16,
            Point::zero(),
            10.0,
        );
        cache.reset();
        assert!(
            cache.atlas_generation() > g0,
            "reset must bump atlas generation"
        );
    }

    #[test]
    fn test_glyph_cache_reset() {
        let mut cache = GlyphCache::default();

        let key = GlyphKey::new(1, 65, 16.0, Point::zero());
        cache.insert(key, 16, 16, Point::zero(), 10.0);
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert!(cache.is_empty());
    }
}
