# Migration Guide: Skia C++ to skia-rs

This guide helps developers familiar with Google's Skia C++ library transition to skia-rs, a 100% Rust implementation with API compatibility.

## Overview

skia-rs aims for API compatibility with Skia, meaning most concepts and operations map directly. The main differences are:
- Rust naming conventions (snake_case vs camelCase)
- Rust ownership/borrowing instead of reference counting
- Result/Option for error handling instead of return codes
- Builder patterns where Skia uses constructors

## Quick Reference

### Type Mapping

| Skia C++ | skia-rs | Notes |
|----------|---------|-------|
| `SkScalar` | `Scalar` (f32) | Type alias |
| `SkPoint` | `Point` | Same layout |
| `SkIPoint` | `IPoint` | Integer point |
| `SkRect` | `Rect` | Same layout |
| `SkIRect` | `IRect` | Integer rect |
| `SkSize` | `Size` | Same layout |
| `SkMatrix` | `Matrix` | 3x3 matrix |
| `SkMatrix44` | `Matrix44` | 4x4 matrix |
| `SkColor` | `Color` | ARGB u32 |
| `SkColor4f` | `Color4f` | RGBA floats |
| `SkPath` | `Path` | Immutable path |
| `SkPathBuilder` | `PathBuilder` | Path construction |
| `SkPaint` | `Paint` | Drawing style |
| `SkCanvas` | `Canvas` trait | Drawing surface |
| `SkSurface` | `Surface` | Pixel backing |
| `SkImage` | `Image` | Immutable image |
| `SkFont` | `Font` | Text styling |
| `SkTypeface` | `Typeface` | Font data |

### Method Naming

| Skia C++ | skia-rs |
|----------|---------|
| `getX()` | `x()` |
| `setX(val)` | `set_x(val)` |
| `makeXY()` | `new()` or `from_xy()` |
| `isValid()` | `is_valid()` |
| `isEmpty()` | `is_empty()` |

## Geometry Types

### Point

```cpp
// Skia C++
SkPoint pt = SkPoint::Make(10, 20);
SkScalar len = pt.length();
pt.normalize();
SkPoint sum = pt + SkPoint::Make(5, 5);
```

```rust
// skia-rs
let pt = Point::new(10.0, 20.0);
let len = pt.length();
let normalized = pt.normalize();
let sum = pt + Point::new(5.0, 5.0);
```

### Rect

```cpp
// Skia C++
SkRect rect = SkRect::MakeXYWH(10, 20, 100, 50);
SkRect rect2 = SkRect::MakeLTRB(10, 20, 110, 70);
bool empty = rect.isEmpty();
SkScalar w = rect.width();
rect.outset(5, 5);
```

```rust
// skia-rs
let rect = Rect::from_xywh(10.0, 20.0, 100.0, 50.0);
let rect2 = Rect::new(10.0, 20.0, 110.0, 70.0);
let empty = rect.is_empty();
let w = rect.width();
let outset = rect.outset(5.0, 5.0);
```

### Matrix

```cpp
// Skia C++
SkMatrix matrix;
matrix.setTranslate(100, 200);
matrix.preScale(2, 2);
matrix.postRotate(45);

SkPoint dst;
matrix.mapPoint(&dst, SkPoint::Make(10, 10));

SkMatrix inverse;
if (matrix.invert(&inverse)) { ... }
```

```rust
// skia-rs
let matrix = Matrix::translate(100.0, 200.0)
    .pre_scale(2.0, 2.0)
    .post_rotate(45.0_f32.to_radians());

let dst = matrix.map_point(Point::new(10.0, 10.0));

if let Some(inverse) = matrix.invert() { ... }
```

## Path

### Building Paths

```cpp
// Skia C++
SkPath path;
path.moveTo(0, 0);
path.lineTo(100, 0);
path.quadTo(150, 50, 100, 100);
path.cubicTo(50, 150, 0, 150, 0, 100);
path.close();

// Or with builder (Skia modern)
SkPathBuilder builder;
builder.moveTo(0, 0)
       .lineTo(100, 0)
       .close();
SkPath path = builder.detach();
```

```rust
// skia-rs (always use builder)
let mut builder = PathBuilder::new();
builder.move_to(0.0, 0.0);
builder.line_to(100.0, 0.0);
builder.quad_to(150.0, 50.0, 100.0, 100.0);
builder.cubic_to(50.0, 150.0, 0.0, 150.0, 0.0, 100.0);
builder.close();
let path = builder.build();

// Fluent style (methods return &mut Self)
let path = PathBuilder::new()
    .move_to(0.0, 0.0)
    .line_to(100.0, 0.0)
    .close()
    .build();
```

### Path Shapes

```cpp
// Skia C++
SkPath path;
path.addRect(SkRect::MakeWH(100, 100));
path.addOval(SkRect::MakeWH(100, 50));
path.addCircle(50, 50, 25);
path.addRoundRect(rect, 10, 10);
```

```rust
// skia-rs
let mut builder = PathBuilder::new();
builder.add_rect(&Rect::from_wh(100.0, 100.0));
builder.add_oval(&Rect::from_wh(100.0, 50.0));
builder.add_circle(50.0, 50.0, 25.0);
builder.add_round_rect(&rect, 10.0, 10.0);
let path = builder.build();
```

### Path Operations

```cpp
// Skia C++
SkRect bounds = path.getBounds();
bool contains = path.contains(50, 50);
SkPath::FillType fill = path.getFillType();
path.setFillType(SkPathFillType::kEvenOdd);
```

```rust
// skia-rs
let bounds = path.bounds();
let contains = path.contains(Point::new(50.0, 50.0));
let fill = path.fill_type();
// Note: Path is immutable, create new with different fill type
let path = path.with_fill_type(FillType::EvenOdd);
```

## Paint

```cpp
// Skia C++
SkPaint paint;
paint.setColor(SK_ColorRED);
paint.setStyle(SkPaint::kStroke_Style);
paint.setStrokeWidth(2.0f);
paint.setAntiAlias(true);
paint.setStrokeCap(SkPaint::kRound_Cap);
paint.setStrokeJoin(SkPaint::kMiter_Join);
paint.setBlendMode(SkBlendMode::kMultiply);
```

```rust
// skia-rs
let mut paint = Paint::new();
paint.set_color32(Color::RED);
paint.set_style(Style::Stroke);
paint.set_stroke_width(2.0);
paint.set_anti_alias(true);
paint.set_stroke_cap(StrokeCap::Round);
paint.set_stroke_join(StrokeJoin::Miter);
paint.set_blend_mode(BlendMode::Multiply);

// Or with method chaining
let paint = Paint::new()
    .with_color(Color4f::RED)
    .with_style(Style::Stroke)
    .with_stroke_width(2.0);
```

## Canvas & Drawing

### Creating a Surface

```cpp
// Skia C++
sk_sp<SkSurface> surface = SkSurfaces::Raster(
    SkImageInfo::MakeN32Premul(800, 600)
);
SkCanvas* canvas = surface->getCanvas();
```

```rust
// skia-rs
let mut surface = Surface::new_raster_n32_premul(800, 600)
    .expect("Failed to create surface");
let mut canvas = surface.raster_canvas();
```

### Drawing Primitives

```cpp
// Skia C++
canvas->clear(SK_ColorWHITE);
canvas->drawRect(rect, paint);
canvas->drawCircle(100, 100, 50, paint);
canvas->drawOval(rect, paint);
canvas->drawLine(0, 0, 100, 100, paint);
canvas->drawPath(path, paint);
canvas->drawPoint(50, 50, paint);
```

```rust
// skia-rs
canvas.clear(Color::WHITE);
canvas.draw_rect(&rect, &paint);
canvas.draw_circle(Point::new(100.0, 100.0), 50.0, &paint);
canvas.draw_oval(&rect, &paint);
canvas.draw_line(Point::new(0.0, 0.0), Point::new(100.0, 100.0), &paint);
canvas.draw_path(&path, &paint);
canvas.draw_point(Point::new(50.0, 50.0), &paint);
```

### Save/Restore

```cpp
// Skia C++
canvas->save();
canvas->translate(100, 100);
canvas->rotate(45);
canvas->scale(2, 2);
// draw...
canvas->restore();

// Or with auto-restore
{
    SkAutoCanvasRestore acr(canvas, true);
    canvas->translate(100, 100);
    // draw...
} // auto-restore here
```

```rust
// skia-rs
canvas.save();
canvas.translate(100.0, 100.0);
canvas.rotate(45.0_f32.to_radians());
canvas.scale(2.0, 2.0);
// draw...
canvas.restore();

// RAII guard (if available)
{
    let _guard = canvas.save_guard();
    canvas.translate(100.0, 100.0);
    // draw...
} // auto-restore here
```

### Clipping

```cpp
// Skia C++
canvas->clipRect(rect);
canvas->clipPath(path);
canvas->clipRect(rect, SkClipOp::kDifference);
```

```rust
// skia-rs
canvas.clip_rect(&rect, ClipOp::Intersect);
canvas.clip_path(&path, ClipOp::Intersect);
canvas.clip_rect(&rect, ClipOp::Difference);
```

## Text

```cpp
// Skia C++
sk_sp<SkTypeface> typeface = SkTypeface::MakeFromName("Arial", SkFontStyle::Normal());
SkFont font(typeface, 24);
canvas->drawString("Hello", 100, 100, font, paint);

// Or with TextBlob
SkTextBlobBuilder builder;
const SkTextBlobBuilder::RunBuffer& run = builder.allocRun(font, 5, 100, 100);
// fill run.glyphs...
sk_sp<SkTextBlob> blob = builder.make();
canvas->drawTextBlob(blob, 0, 0, paint);
```

```rust
// skia-rs
let typeface = Typeface::from_name("Arial", FontStyle::normal())
    .unwrap_or_else(Typeface::default);
let font = Font::new(typeface, 24.0);
canvas.draw_string("Hello", Point::new(100.0, 100.0), &font, &paint);

// Or with TextBlob
let blob = TextBlobBuilder::new()
    .add_run(&font, &glyphs, Point::new(100.0, 100.0))
    .build();
canvas.draw_text_blob(&blob, Point::ZERO, &paint);
```

## Image Loading

```cpp
// Skia C++
sk_sp<SkData> data = SkData::MakeFromFileName("image.png");
sk_sp<SkImage> image = SkImages::DeferredFromEncodedData(data);
canvas->drawImage(image, 0, 0);
```

```rust
// skia-rs
let data = std::fs::read("image.png")?;
let image = Image::decode(&data)?;
canvas.draw_image(&image, Point::ZERO, &paint);
```

## Shaders

```cpp
// Skia C++
SkPoint pts[] = { {0, 0}, {100, 0} };
SkColor colors[] = { SK_ColorRED, SK_ColorBLUE };
sk_sp<SkShader> shader = SkGradientShader::MakeLinear(
    pts, colors, nullptr, 2, SkTileMode::kClamp
);
paint.setShader(shader);
```

```rust
// skia-rs
let shader = LinearGradient::new(
    Point::new(0.0, 0.0),
    Point::new(100.0, 0.0),
    &[Color4f::RED, Color4f::BLUE],
    None, // positions
    TileMode::Clamp,
);
paint.set_shader(Some(Box::new(shader)));
```

## Error Handling

```cpp
// Skia C++ - check return values or null pointers
SkMatrix inverse;
if (!matrix.invert(&inverse)) {
    // handle error
}

sk_sp<SkImage> image = SkImages::DeferredFromEncodedData(data);
if (!image) {
    // handle error
}
```

```rust
// skia-rs - use Option/Result
let inverse = matrix.invert()
    .ok_or("Matrix not invertible")?;

let image = Image::decode(&data)
    .map_err(|e| format!("Decode error: {}", e))?;

// Or with pattern matching
match matrix.invert() {
    Some(inv) => { /* use inv */ }
    None => { /* handle error */ }
}
```

## Memory Management

```cpp
// Skia C++ - reference counting with sk_sp
sk_sp<SkSurface> surface = SkSurfaces::Raster(...);
sk_sp<SkImage> image = surface->makeImageSnapshot();
// Objects freed when last sk_sp goes out of scope
```

```rust
// skia-rs - Rust ownership
let surface = Surface::new_raster_n32_premul(800, 600)?;
let image = surface.make_image_snapshot();
// Objects freed when they go out of scope

// For shared ownership, use Arc
let shared_image: Arc<Image> = Arc::new(image);
```

## Common Patterns

### Drawing a Rounded Rectangle

```cpp
// Skia C++
SkRRect rrect = SkRRect::MakeRectXY(rect, 10, 10);
canvas->drawRRect(rrect, paint);
```

```rust
// skia-rs
let mut builder = PathBuilder::new();
builder.add_round_rect(&rect, 10.0, 10.0);
canvas.draw_path(&builder.build(), &paint);
```

### Saving to PNG

```cpp
// Skia C++
sk_sp<SkImage> image = surface->makeImageSnapshot();
sk_sp<SkData> png = SkPngEncoder::Encode(nullptr, image.get(), {});
SkFILEWStream file("output.png");
file.write(png->data(), png->size());
```

```rust
// skia-rs
let image = surface.make_image_snapshot();
let png_data = image.encode_png()?;
std::fs::write("output.png", png_data)?;
```

## Feature Comparison

| Feature | Skia C++ | skia-rs | Notes |
|---------|----------|---------|-------|
| Raster backend | ✅ | ✅ | Full support |
| GPU (Vulkan) | ✅ | ✅ | Via feature flag |
| GPU (Metal) | ✅ | ✅ | macOS only |
| GPU (OpenGL) | ✅ | ✅ | Via feature flag |
| GPU (WebGPU) | ✅ | ✅ | Via wgpu |
| Text shaping | ✅ | ✅ | Via rustybuzz |
| SVG | ✅ | ✅ | Parse & render |
| PDF | ✅ | ✅ | Basic support |
| Lottie | ✅ | ✅ | Via skottie |
| SkSL shaders | ✅ | ✅ | Runtime effects |
| Path ops | ✅ | ✅ | Boolean operations |
| Image codecs | ✅ | ✅ | PNG, JPEG, WebP, etc. |

## Getting Help

- [API Documentation](https://docs.rs/skia-rs)
- [GitHub Issues](https://github.com/quinnjr/skia-rs/issues)
- [Examples](https://github.com/quinnjr/skia-rs/tree/main/examples)
