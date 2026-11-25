//! Atlas management for GPU rendering.
//!
//! This module provides texture atlas management for efficiently batching
//! small paths, glyphs, and other small elements into larger textures.

use crate::cast_util::{scalar_from_u32, u32_from_usize};
use skia_rs_core::Rect;
use std::collections::HashMap;

/// Unique identifier for an atlas entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasEntryId(u64);

impl AtlasEntryId {
    /// Create a new entry ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

/// A region within an atlas.
#[derive(Debug, Clone, Copy)]
pub struct AtlasRegion {
    /// X position in atlas.
    pub x: u32,
    /// Y position in atlas.
    pub y: u32,
    /// Width of region.
    pub width: u32,
    /// Height of region.
    pub height: u32,
    /// Atlas layer (for array textures).
    pub layer: u32,
}

impl AtlasRegion {
    /// Get UV coordinates for this region within an atlas of given size.
    ///
    /// The rect is inset by half a texel on every edge so that bilinear
    /// sampling never reaches into a neighbouring entry's texels (the classic
    /// atlas-bleed artifact). This matches Skia's convention of sampling at
    /// texel centers for atlased content.
    #[must_use]
    pub fn uv_rect(&self, atlas_width: u32, atlas_height: u32) -> [f32; 4] {
        let half_w = 0.5 / scalar_from_u32(atlas_width);
        let half_h = 0.5 / scalar_from_u32(atlas_height);
        [
            scalar_from_u32(self.x) / scalar_from_u32(atlas_width) + half_w,
            scalar_from_u32(self.y) / scalar_from_u32(atlas_height) + half_h,
            scalar_from_u32(self.x + self.width) / scalar_from_u32(atlas_width) - half_w,
            scalar_from_u32(self.y + self.height) / scalar_from_u32(atlas_height) - half_h,
        ]
    }

    /// Convert to a rect.
    #[must_use]
    pub const fn to_rect(&self) -> Rect {
        Rect::from_xywh(
            scalar_from_u32(self.x),
            scalar_from_u32(self.y),
            scalar_from_u32(self.width),
            scalar_from_u32(self.height),
        )
    }
}

/// Atlas configuration.
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Atlas width in pixels.
    pub width: u32,
    /// Atlas height in pixels.
    pub height: u32,
    /// Maximum number of layers (for array textures).
    pub max_layers: u32,
    /// Padding between entries.
    pub padding: u32,
    /// Allow resizing when full.
    pub allow_resize: bool,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            max_layers: 4,
            padding: 1,
            allow_resize: true,
        }
    }
}

/// Atlas allocation result.
#[derive(Debug)]
pub enum AtlasAllocResult {
    /// Successfully allocated. Carries the entry id (so callers can later
    /// [`TextureAtlas::free`] the region) and the placed region.
    Success(AtlasEntryId, AtlasRegion),
    /// Atlas is full, need to flush and reset.
    Full,
    /// Request too large for atlas (even with padding, in an empty atlas).
    TooLarge,
}

/// A single atlas layer using shelf-based allocation.
#[derive(Debug)]
struct AtlasLayer {
    /// Current shelf Y position.
    current_y: u32,
    /// Current shelf height.
    current_shelf_height: u32,
    /// Current X position in shelf.
    current_x: u32,
    /// Layer dimensions.
    width: u32,
    height: u32,
}

impl AtlasLayer {
    const fn new(width: u32, height: u32) -> Self {
        Self {
            current_y: 0,
            current_shelf_height: 0,
            current_x: 0,
            width,
            height,
        }
    }

    fn allocate(&mut self, width: u32, height: u32, padding: u32) -> Option<(u32, u32)> {
        let padded_width = width + padding * 2;
        let padded_height = height + padding * 2;

        // Check if it fits in current shelf
        if self.current_x + padded_width <= self.width
            && self.current_y + padded_height.max(self.current_shelf_height) <= self.height
        {
            let x = self.current_x + padding;
            let y = self.current_y + padding;

            self.current_x += padded_width;
            self.current_shelf_height = self.current_shelf_height.max(padded_height);

            return Some((x, y));
        }

        // Try next shelf
        if self.current_y + self.current_shelf_height + padded_height <= self.height {
            self.current_y += self.current_shelf_height;
            self.current_x = 0;
            self.current_shelf_height = padded_height;

            if self.current_x + padded_width <= self.width {
                let x = self.current_x + padding;
                let y = self.current_y + padding;

                self.current_x += padded_width;

                return Some((x, y));
            }
        }

        None
    }

    const fn reset(&mut self) {
        self.current_y = 0;
        self.current_shelf_height = 0;
        self.current_x = 0;
    }
}

/// A texture atlas for batching small elements.
#[derive(Debug)]
pub struct TextureAtlas {
    /// Configuration.
    config: AtlasConfig,
    /// Atlas layers.
    layers: Vec<AtlasLayer>,
    /// Entry lookup by ID.
    entries: HashMap<AtlasEntryId, AtlasRegion>,
    /// IDs of entries that have been freed and are eligible for removal on
    /// the next `compact()`. Kept as a set for O(1) membership checks
    /// during compaction.
    freed: std::collections::HashSet<AtlasEntryId>,
    /// Next entry ID.
    next_id: u64,
    /// Generation counter (incremented on reset).
    generation: u64,
}

impl TextureAtlas {
    /// Create a new texture atlas.
    #[must_use]
    pub fn new(config: AtlasConfig) -> Self {
        let mut layers = Vec::with_capacity(config.max_layers as usize);
        layers.push(AtlasLayer::new(config.width, config.height));

        Self {
            config,
            layers,
            entries: HashMap::new(),
            freed: std::collections::HashSet::new(),
            next_id: 0,
            generation: 0,
        }
    }

    /// Get atlas configuration.
    #[must_use]
    pub const fn config(&self) -> &AtlasConfig {
        &self.config
    }

    /// Get current generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Get number of active layers.
    #[must_use]
    pub fn layer_count(&self) -> u32 {
        u32_from_usize(self.layers.len())
    }

    /// Get number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Look up an existing entry.
    #[must_use]
    pub fn lookup(&self, id: AtlasEntryId) -> Option<&AtlasRegion> {
        self.entries.get(&id)
    }

    /// Allocate a region in the atlas.
    pub fn allocate(&mut self, width: u32, height: u32) -> AtlasAllocResult {
        // Too-large check must account for the padding the allocator adds on
        // every side: an entry needs `width + 2*padding` texels to fit. A
        // request that can never fit even in a fresh, empty atlas is
        // `TooLarge` (retrying/evicting would loop forever), distinct from
        // `Full` (would fit in an empty atlas but not right now).
        let pad = self.config.padding.saturating_mul(2);
        if width.saturating_add(pad) > self.config.width
            || height.saturating_add(pad) > self.config.height
        {
            return AtlasAllocResult::TooLarge;
        }

        // Try to allocate in existing layers
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if let Some((x, y)) = layer.allocate(width, height, self.config.padding) {
                let id = AtlasEntryId::new(self.next_id);
                self.next_id += 1;

                let region = AtlasRegion {
                    x,
                    y,
                    width,
                    height,
                    layer: u32_from_usize(layer_idx),
                };

                self.entries.insert(id, region);
                return AtlasAllocResult::Success(id, region);
            }
        }

        // Try to add a new layer
        if self.layers.len() < self.config.max_layers as usize {
            let mut new_layer = AtlasLayer::new(self.config.width, self.config.height);
            if let Some((x, y)) = new_layer.allocate(width, height, self.config.padding) {
                let layer_idx = self.layers.len();
                self.layers.push(new_layer);

                let id = AtlasEntryId::new(self.next_id);
                self.next_id += 1;

                let region = AtlasRegion {
                    x,
                    y,
                    width,
                    height,
                    layer: u32_from_usize(layer_idx),
                };

                self.entries.insert(id, region);
                return AtlasAllocResult::Success(id, region);
            }
        }

        AtlasAllocResult::Full
    }

    /// Allocate and return the entry ID.
    pub fn allocate_with_id(
        &mut self,
        width: u32,
        height: u32,
    ) -> Option<(AtlasEntryId, AtlasRegion)> {
        match self.allocate(width, height) {
            AtlasAllocResult::Success(id, region) => Some((id, region)),
            _ => None,
        }
    }

    /// Reset the atlas, clearing all entries.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.entries.clear();
        self.freed.clear();
        self.generation += 1;
        // Keep only first layer
        self.layers.truncate(1);
    }

    /// Mark an entry as freed. The slot remains occupied until the next
    /// `compact()` call; this lets callers drop cached data they no longer
    /// need without immediately forcing a repack.
    pub fn free(&mut self, id: AtlasEntryId) -> bool {
        if self.entries.contains_key(&id) {
            self.freed.insert(id);
            true
        } else {
            false
        }
    }

    /// Number of entries currently marked as freed but not yet repacked.
    #[must_use]
    pub fn freed_count(&self) -> usize {
        self.freed.len()
    }

    /// Result of a compaction pass.
    ///
    /// Callers get back the set of entries whose atlas region changed and
    /// therefore need their GPU texture data re-uploaded to the new slots.
    /// Entries whose region did not change are not included.
    ///
    /// `removed` lists entries that were freed and are no longer in the atlas.
    pub fn compact(&mut self) -> CompactResult {
        // Gather current entries into a sorted list (largest first gives
        // best shelf-pack behaviour for new placements). Drop any that
        // were previously marked as freed.
        let old_entries: Vec<(AtlasEntryId, AtlasRegion)> = self
            .entries
            .iter()
            .filter(|(id, _)| !self.freed.contains(id))
            .map(|(id, region)| (*id, *region))
            .collect();

        let mut removed: Vec<AtlasEntryId> = self.freed.iter().copied().collect();

        // Clear the layers — we're about to replay all surviving entries
        // into a fresh packing.
        for layer in &mut self.layers {
            layer.reset();
        }
        // Keep just one layer; we may grow it back during re-insertion.
        self.layers.truncate(1);
        self.entries.clear();
        self.freed.clear();
        self.generation += 1;

        // Sort by height descending for shelf-pack stability. Shelf-pack
        // works best when the tallest rectangles go down first so later
        // rectangles fit under existing shelves.
        let mut sorted = old_entries;
        sorted.sort_by(|a, b| b.1.height.cmp(&a.1.height).then(b.1.width.cmp(&a.1.width)));

        let mut remapped: Vec<(AtlasEntryId, AtlasRegion, AtlasRegion)> = Vec::new();

        for (id, old_region) in sorted {
            let mut placed = false;
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                if let Some((x, y)) =
                    layer.allocate(old_region.width, old_region.height, self.config.padding)
                {
                    let new_region = AtlasRegion {
                        x,
                        y,
                        width: old_region.width,
                        height: old_region.height,
                        layer: u32_from_usize(layer_idx),
                    };
                    self.entries.insert(id, new_region);
                    if (old_region.x, old_region.y, old_region.layer)
                        != (new_region.x, new_region.y, new_region.layer)
                    {
                        remapped.push((id, old_region, new_region));
                    }
                    placed = true;
                    break;
                }
            }

            if !placed && self.layers.len() < self.config.max_layers as usize {
                let mut new_layer = AtlasLayer::new(self.config.width, self.config.height);
                if let Some((x, y)) =
                    new_layer.allocate(old_region.width, old_region.height, self.config.padding)
                {
                    let layer_idx = self.layers.len();
                    self.layers.push(new_layer);
                    let new_region = AtlasRegion {
                        x,
                        y,
                        width: old_region.width,
                        height: old_region.height,
                        layer: u32_from_usize(layer_idx),
                    };
                    self.entries.insert(id, new_region);
                    if (old_region.x, old_region.y, old_region.layer)
                        != (new_region.x, new_region.y, new_region.layer)
                    {
                        remapped.push((id, old_region, new_region));
                    }
                    placed = true;
                }
            }

            if !placed {
                // Unable to re-pack — this only happens if the atlas has
                // pathological fragmentation across max_layers. We must NOT
                // re-insert at the stale original slot: after the layers were
                // reset that coordinate is no longer reserved, so another
                // entry may be packed on top of it, producing overlapping
                // regions and corrupt sampling. Drop the entry and report it
                // as evicted so the caller re-rasterizes it on demand.
                removed.push(id);
            }
        }

        CompactResult { remapped, removed }
    }
}

/// Result of a `TextureAtlas::compact()` call.
#[derive(Debug, Clone, Default)]
pub struct CompactResult {
    /// Entries whose region moved during repacking. Each tuple is
    /// (id, `old_region`, `new_region`).
    pub remapped: Vec<(AtlasEntryId, AtlasRegion, AtlasRegion)>,
    /// Entries that were freed and dropped from the atlas.
    pub removed: Vec<AtlasEntryId>,
}

/// Atlas manager for multiple atlases by type.
#[derive(Debug)]
pub struct AtlasManager {
    /// Path atlas.
    path: TextureAtlas,
    /// Glyph atlas (alpha).
    glyph: TextureAtlas,
    /// Color atlas (RGBA).
    color: TextureAtlas,
}

impl AtlasManager {
    /// Create a new atlas manager with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            path: TextureAtlas::new(AtlasConfig {
                width: 2048,
                height: 2048,
                max_layers: 4,
                padding: 2,
                allow_resize: true,
            }),
            glyph: TextureAtlas::new(AtlasConfig {
                width: 1024,
                height: 1024,
                max_layers: 2,
                padding: 1,
                allow_resize: true,
            }),
            color: TextureAtlas::new(AtlasConfig {
                width: 1024,
                height: 1024,
                max_layers: 2,
                padding: 1,
                allow_resize: true,
            }),
        }
    }

    /// Get path atlas.
    #[must_use]
    pub const fn path_atlas(&self) -> &TextureAtlas {
        &self.path
    }

    /// Get mutable path atlas.
    pub const fn path_atlas_mut(&mut self) -> &mut TextureAtlas {
        &mut self.path
    }

    /// Get glyph atlas.
    #[must_use]
    pub const fn glyph_atlas(&self) -> &TextureAtlas {
        &self.glyph
    }

    /// Get mutable glyph atlas.
    pub const fn glyph_atlas_mut(&mut self) -> &mut TextureAtlas {
        &mut self.glyph
    }

    /// Get color atlas.
    #[must_use]
    pub const fn color_atlas(&self) -> &TextureAtlas {
        &self.color
    }

    /// Get mutable color atlas.
    pub const fn color_atlas_mut(&mut self) -> &mut TextureAtlas {
        &mut self.color
    }

    /// Reset all atlases.
    pub fn reset_all(&mut self) {
        self.path.reset();
        self.glyph.reset();
        self.color.reset();
    }
}

impl Default for AtlasManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_region_uv() {
        let region = AtlasRegion {
            x: 100,
            y: 200,
            width: 50,
            height: 75,
            layer: 0,
        };

        // uv_rect insets by half a texel on every edge to avoid atlas bleed.
        let uv = region.uv_rect(1000, 1000);
        let half = 0.5 / 1000.0;
        assert!((uv[0] - (0.1 + half)).abs() < 1e-6);
        assert!((uv[1] - (0.2 + half)).abs() < 1e-6);
        assert!((uv[2] - (0.15 - half)).abs() < 1e-6);
        assert!((uv[3] - (0.275 - half)).abs() < 1e-6);
        // Inset shrinks the rect, so u0<u1 and v0<v1 still hold.
        assert!(uv[0] < uv[2] && uv[1] < uv[3]);
    }

    #[test]
    fn test_atlas_too_large_includes_padding() {
        // Regression: a request whose footprint + padding cannot fit even in
        // an empty atlas must be TooLarge, not Full (which would make the
        // glyph-cache retry loop spin forever).
        let config = AtlasConfig {
            width: 256,
            height: 256,
            max_layers: 1,
            padding: 4,
            allow_resize: false,
        };
        let mut atlas = TextureAtlas::new(config);
        // 250 + 2*4 = 258 > 256 -> TooLarge even though 250 < 256.
        match atlas.allocate(250, 10) {
            AtlasAllocResult::TooLarge => {}
            other => panic!("expected TooLarge, got {other:?}"),
        }
        // Exactly-fits case (248 + 8 = 256) still succeeds.
        match atlas.allocate(248, 10) {
            AtlasAllocResult::Success(_, _) => {}
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn test_compact_survivors_do_not_overlap() {
        // Regression: compact() must never re-place an entry at a stale slot
        // that another entry now occupies. Verify all surviving regions are
        // pairwise non-overlapping after a compaction that frees some.
        let mut atlas = TextureAtlas::new(AtlasConfig {
            width: 256,
            height: 256,
            max_layers: 2,
            padding: 1,
            allow_resize: false,
        });
        let mut ids = Vec::new();
        for _ in 0..20 {
            if let Some((id, _)) = atlas.allocate_with_id(40, 30) {
                ids.push(id);
            }
        }
        // Free every third entry, then compact.
        for (i, id) in ids.iter().enumerate() {
            if i % 3 == 0 {
                atlas.free(*id);
            }
        }
        atlas.compact();

        let regions: Vec<AtlasRegion> = atlas.entries.values().copied().collect();
        for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                let a = &regions[i];
                let b = &regions[j];
                if a.layer != b.layer {
                    continue;
                }
                let disjoint = a.x + a.width <= b.x
                    || b.x + b.width <= a.x
                    || a.y + a.height <= b.y
                    || b.y + b.height <= a.y;
                assert!(
                    disjoint,
                    "compact produced overlapping regions {a:?} / {b:?}"
                );
            }
        }
    }

    #[test]
    fn test_atlas_allocate() {
        let config = AtlasConfig {
            width: 256,
            height: 256,
            max_layers: 1,
            padding: 0,
            allow_resize: false,
        };

        let mut atlas = TextureAtlas::new(config);

        // First allocation should succeed
        match atlas.allocate(64, 64) {
            AtlasAllocResult::Success(_, region) => {
                assert_eq!(region.width, 64);
                assert_eq!(region.height, 64);
            }
            _ => panic!("Expected success"),
        }

        // Too large allocation should fail
        match atlas.allocate(512, 512) {
            AtlasAllocResult::TooLarge => {}
            _ => panic!("Expected TooLarge"),
        }
    }

    #[test]
    fn test_atlas_multiple_allocations() {
        let config = AtlasConfig {
            width: 256,
            height: 256,
            max_layers: 1,
            padding: 0,
            allow_resize: false,
        };

        let mut atlas = TextureAtlas::new(config);

        // Allocate multiple small regions
        for _ in 0..16 {
            match atlas.allocate(32, 32) {
                AtlasAllocResult::Success(..) => {}
                _ => panic!("Expected success"),
            }
        }

        assert_eq!(atlas.entry_count(), 16);
    }

    #[test]
    fn test_atlas_reset() {
        let mut atlas = TextureAtlas::new(AtlasConfig::default());

        atlas.allocate(100, 100);
        atlas.allocate(100, 100);
        assert_eq!(atlas.entry_count(), 2);

        let gen_before = atlas.generation();
        atlas.reset();
        assert_eq!(atlas.entry_count(), 0);
        assert_eq!(atlas.generation(), gen_before + 1);
    }

    #[test]
    fn test_atlas_manager() {
        let mut manager = AtlasManager::new();

        // Allocate in different atlases
        manager.path_atlas_mut().allocate(64, 64);
        manager.glyph_atlas_mut().allocate(32, 32);
        manager.color_atlas_mut().allocate(48, 48);

        assert_eq!(manager.path_atlas().entry_count(), 1);
        assert_eq!(manager.glyph_atlas().entry_count(), 1);
        assert_eq!(manager.color_atlas().entry_count(), 1);

        manager.reset_all();
        assert_eq!(manager.path_atlas().entry_count(), 0);
    }

    #[test]
    fn test_atlas_entry_id() {
        let id = AtlasEntryId::new(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_atlas_lookup() {
        let mut atlas = TextureAtlas::new(AtlasConfig::default());

        if let Some((id, region)) = atlas.allocate_with_id(100, 100) {
            let looked_up = atlas.lookup(id);
            assert!(looked_up.is_some());
            assert_eq!(looked_up.unwrap().width, region.width);
        }
    }

    #[test]
    fn test_atlas_free_and_compact_removes_entries() {
        let mut atlas = TextureAtlas::new(AtlasConfig {
            width: 512,
            height: 512,
            max_layers: 1,
            padding: 0,
            allow_resize: false,
        });

        let mut ids = Vec::new();
        for _ in 0..4 {
            if let Some((id, _)) = atlas.allocate_with_id(64, 64) {
                ids.push(id);
            }
        }
        assert_eq!(atlas.entry_count(), 4);

        // Free two entries.
        assert!(atlas.free(ids[1]));
        assert!(atlas.free(ids[2]));
        assert_eq!(atlas.freed_count(), 2);

        let result = atlas.compact();
        assert_eq!(result.removed.len(), 2);
        assert_eq!(atlas.entry_count(), 2);
        assert_eq!(atlas.freed_count(), 0);

        // Surviving ids still resolve.
        assert!(atlas.lookup(ids[0]).is_some());
        assert!(atlas.lookup(ids[3]).is_some());
        // Removed ids are gone.
        assert!(atlas.lookup(ids[1]).is_none());
        assert!(atlas.lookup(ids[2]).is_none());
    }

    #[test]
    fn test_atlas_compact_reclaims_space() {
        // After compacting away freed entries we should be able to allocate
        // new rectangles that previously hit "Full".
        let mut atlas = TextureAtlas::new(AtlasConfig {
            width: 128,
            height: 128,
            max_layers: 1,
            padding: 0,
            allow_resize: false,
        });

        // Fill most of a 128-wide shelf with 60x60 rectangles.
        let a = atlas.allocate_with_id(60, 60).unwrap().0;
        let b = atlas.allocate_with_id(60, 60).unwrap().0;
        let c = atlas.allocate_with_id(60, 60).unwrap().0;
        assert_eq!(atlas.entry_count(), 3);

        // Trying to fit a 60x60 on top should still succeed once we have a
        // second shelf, but a 100x100 should not fit until we free & compact.
        match atlas.allocate(100, 100) {
            AtlasAllocResult::Full => {}
            other => panic!("expected Full, got {other:?}"),
        }

        // Free all three entries and compact.
        atlas.free(a);
        atlas.free(b);
        atlas.free(c);
        atlas.compact();

        // Now the 100x100 should fit.
        match atlas.allocate(100, 100) {
            AtlasAllocResult::Success(..) => {}
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn test_atlas_free_unknown_returns_false() {
        let mut atlas = TextureAtlas::new(AtlasConfig::default());
        assert!(!atlas.free(AtlasEntryId::new(9999)));
    }

    #[test]
    fn test_atlas_compact_without_frees_is_noop_ish() {
        let mut atlas = TextureAtlas::new(AtlasConfig {
            width: 256,
            height: 256,
            max_layers: 1,
            padding: 0,
            allow_resize: false,
        });

        let (id_a, _) = atlas.allocate_with_id(50, 50).unwrap();
        let (id_b, _) = atlas.allocate_with_id(50, 50).unwrap();

        let before_a = *atlas.lookup(id_a).unwrap();
        let before_b = *atlas.lookup(id_b).unwrap();

        let result = atlas.compact();
        assert_eq!(result.removed, []);

        // Both entries still resolve (possibly with the same regions,
        // possibly remapped).
        let after_a = *atlas.lookup(id_a).unwrap();
        let after_b = *atlas.lookup(id_b).unwrap();
        assert_eq!(before_a.width, after_a.width);
        assert_eq!(before_b.width, after_b.width);
    }
}
