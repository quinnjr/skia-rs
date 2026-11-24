//! CSS parsing and styling support for SVG.
//!
//! This module provides CSS parsing and style application for SVG documents,
//! including support for:
//! - Inline styles (`style` attribute)
//! - Embedded stylesheets (`<style>` elements)
//! - Selectors: element, class, ID, descendant, multiple
//! - Cascading and specificity

use crate::dom::{SvgDom, SvgNode, SvgPaint};
use skia_rs_core::{Matrix, Scalar};

/// A CSS stylesheet containing multiple rules.
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    /// CSS rules in order.
    pub rules: Vec<CssRule>,
}

impl Stylesheet {
    /// Create an empty stylesheet.
    #[must_use]
    pub const fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Parse a CSS stylesheet from a string.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the internal `unwrap()` only fires on a
    /// character just confirmed present via `peek()`.
    #[must_use]
    pub fn parse(css: &str) -> Self {
        let mut stylesheet = Self::new();
        let css = css.trim();

        // Simple CSS parser
        let mut chars = css.chars().peekable();

        while chars.peek().is_some() {
            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            if chars.peek().is_none() {
                break;
            }

            // Skip comments
            if chars.peek() == Some(&'/') {
                chars.next();
                if chars.peek() == Some(&'*') {
                    chars.next();
                    // Skip until */
                    loop {
                        match chars.next() {
                            Some('*') if chars.peek() == Some(&'/') => {
                                chars.next();
                                break;
                            }
                            None => break,
                            _ => {}
                        }
                    }
                    continue;
                }
            }

            // Read selector until {
            let mut selector = String::new();
            while let Some(&c) = chars.peek() {
                if c == '{' {
                    chars.next();
                    break;
                }
                selector.push(chars.next().unwrap());
            }

            let selector = selector.trim().to_string();
            if selector.is_empty() {
                break;
            }

            // Read declarations until }
            let mut declarations_str = String::new();
            let mut brace_depth = 1;
            for c in chars.by_ref() {
                if c == '{' {
                    brace_depth += 1;
                } else if c == '}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        break;
                    }
                }
                declarations_str.push(c);
            }

            // Parse declarations
            let declarations = parse_declarations(&declarations_str);

            // Parse selector into multiple selectors (comma-separated)
            for sel in selector.split(',') {
                let sel = sel.trim();
                if !sel.is_empty() {
                    stylesheet.rules.push(CssRule {
                        selector: CssSelector::parse(sel),
                        declarations: declarations.clone(),
                    });
                }
            }
        }

        stylesheet
    }

    /// Add a rule to the stylesheet.
    pub fn add_rule(&mut self, rule: CssRule) {
        self.rules.push(rule);
    }
}

/// A CSS rule with selector and declarations.
#[derive(Debug, Clone)]
pub struct CssRule {
    /// The selector for this rule.
    pub selector: CssSelector,
    /// Property declarations.
    pub declarations: StyleDeclarations,
}

/// A CSS selector.
#[derive(Debug, Clone)]
pub enum CssSelector {
    /// Universal selector (*).
    Universal,
    /// Element selector (e.g., rect, circle).
    Element(String),
    /// Class selector (e.g., .classname).
    Class(String),
    /// ID selector (e.g., #id).
    Id(String),
    /// Descendant selector (e.g., g rect).
    Descendant(Box<Self>, Box<Self>),
    /// Child selector (e.g., g > rect).
    Child(Box<Self>, Box<Self>),
    /// Multiple conditions (e.g., rect.classname).
    And(Vec<Self>),
}

impl CssSelector {
    /// Parse a selector string.
    ///
    /// # Panics
    ///
    /// Never panics in practice: the internal `expect()` only fires when
    /// exactly one simple selector was collected, in which case `pop()`
    /// always succeeds.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        let s = s.trim();

        // Check for descendant/child selectors
        if s.contains(" > ") {
            let parts: Vec<&str> = s.splitn(2, " > ").collect();
            if parts.len() == 2 {
                return Self::Child(
                    Box::new(Self::parse(parts[0])),
                    Box::new(Self::parse(parts[1])),
                );
            }
        }

        if s.contains(' ') {
            let parts: Vec<&str> = s.splitn(2, ' ').collect();
            if parts.len() == 2 && !parts[1].is_empty() {
                return Self::Descendant(
                    Box::new(Self::parse(parts[0])),
                    Box::new(Self::parse(parts[1])),
                );
            }
        }

        // Check for combined selectors (e.g., rect.classname#id)
        let mut selectors = Vec::new();
        let mut current = String::new();
        let chars = s.chars();

        for c in chars {
            match c {
                '.' | '#' => {
                    if !current.is_empty() {
                        selectors.push(parse_simple_selector(&current));
                        current.clear();
                    }
                    current.push(c);
                }
                _ => current.push(c),
            }
        }

        if !current.is_empty() {
            selectors.push(parse_simple_selector(&current));
        }

        if selectors.len() == 1 {
            selectors
                .pop()
                .expect("selectors.len() == 1 guarantees element")
        } else if selectors.is_empty() {
            Self::Universal
        } else {
            Self::And(selectors)
        }
    }

    /// Calculate specificity (ID, class, element counts).
    #[must_use]
    pub fn specificity(&self) -> (u32, u32, u32) {
        match self {
            Self::Universal => (0, 0, 0),
            Self::Element(_) => (0, 0, 1),
            Self::Class(_) => (0, 1, 0),
            Self::Id(_) => (1, 0, 0),
            Self::Descendant(a, b) | Self::Child(a, b) => {
                let (id_a, class_a, elem_a) = a.specificity();
                let (id_b, class_b, elem_b) = b.specificity();
                (id_a + id_b, class_a + class_b, elem_a + elem_b)
            }
            Self::And(selectors) => {
                let mut id = 0;
                let mut class = 0;
                let mut elem = 0;
                for sel in selectors {
                    let (i, c, e) = sel.specificity();
                    id += i;
                    class += c;
                    elem += e;
                }
                (id, class, elem)
            }
        }
    }

    /// Check if this selector matches a node.
    #[must_use]
    pub fn matches(&self, node: &SvgNode, ancestors: &[&SvgNode]) -> bool {
        match self {
            Self::Universal => true,
            Self::Element(tag) => node_tag_name(node) == tag,
            Self::Class(class) => node.classes.contains(class),
            Self::Id(id) => node.id.as_deref() == Some(id.as_str()),
            Self::Descendant(ancestor_sel, child_sel) => {
                if !child_sel.matches(node, ancestors) {
                    return false;
                }
                // Check if any ancestor matches
                for ancestor in ancestors {
                    if ancestor_sel.matches(ancestor, &[]) {
                        return true;
                    }
                }
                false
            }
            Self::Child(parent_sel, child_sel) => {
                if !child_sel.matches(node, ancestors) {
                    return false;
                }
                // Check immediate parent
                ancestors.last().is_some_and(|parent| {
                    parent_sel.matches(parent, &ancestors[..ancestors.len().saturating_sub(1)])
                })
            }
            Self::And(selectors) => selectors.iter().all(|s| s.matches(node, ancestors)),
        }
    }
}

fn parse_simple_selector(s: &str) -> CssSelector {
    if s == "*" {
        CssSelector::Universal
    } else if let Some(class) = s.strip_prefix('.') {
        CssSelector::Class(class.to_string())
    } else if let Some(id) = s.strip_prefix('#') {
        CssSelector::Id(id.to_string())
    } else {
        CssSelector::Element(s.to_string())
    }
}

fn node_tag_name(node: &SvgNode) -> &str {
    use crate::dom::SvgNodeKind;
    match &node.kind {
        SvgNodeKind::Svg => "svg",
        SvgNodeKind::Group => "g",
        SvgNodeKind::Rect(_) => "rect",
        SvgNodeKind::Circle(_) => "circle",
        SvgNodeKind::Ellipse(_) => "ellipse",
        SvgNodeKind::Line(_) => "line",
        SvgNodeKind::Polyline(_) => "polyline",
        SvgNodeKind::Polygon(_) => "polygon",
        SvgNodeKind::Path(_) => "path",
        SvgNodeKind::Text(_) => "text",
        SvgNodeKind::Image(_) => "image",
        SvgNodeKind::Use(_) => "use",
        SvgNodeKind::Defs => "defs",
        SvgNodeKind::LinearGradient(_) => "linearGradient",
        SvgNodeKind::RadialGradient(_) => "radialGradient",
        SvgNodeKind::ClipPath(_) => "clipPath",
        SvgNodeKind::Unknown(name) => name,
    }
}

/// Style declarations (property-value pairs).
///
/// Declarations are kept in **document order** rather than in a `HashMap`,
/// because CSS cascade within a block is order-dependent (a later
/// declaration of the same property wins, and dependent properties such as
/// `fill` / `fill-opacity` must apply in source order). A repeated property
/// updates in place so "last wins" holds without duplicating entries.
#[derive(Debug, Clone, Default)]
pub struct StyleDeclarations {
    decls: Vec<(String, String)>,
}

impl StyleDeclarations {
    /// Create an empty declaration list.
    #[must_use]
    pub const fn new() -> Self {
        Self { decls: Vec::new() }
    }

    /// Insert or overwrite a property, preserving first-seen order.
    pub fn insert(&mut self, property: String, value: String) {
        if let Some(entry) = self.decls.iter_mut().find(|(p, _)| *p == property) {
            entry.1 = value;
        } else {
            self.decls.push((property, value));
        }
    }

    /// Look up a property's value.
    #[must_use]
    pub fn get(&self, property: &str) -> Option<&String> {
        self.decls
            .iter()
            .find(|(p, _)| p == property)
            .map(|(_, v)| v)
    }

    /// Iterate declarations in document order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.decls.iter().map(|(p, v)| (p, v))
    }

    /// Number of declarations.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.decls.len()
    }

    /// Whether there are no declarations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }
}

/// Parse CSS declarations from a string, preserving document order.
#[must_use]
pub fn parse_declarations(s: &str) -> StyleDeclarations {
    let mut declarations = StyleDeclarations::new();

    for decl in s.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }

        if let Some((property, value)) = decl.split_once(':') {
            let property = property.trim().to_string();
            let value = value.trim().to_string();
            declarations.insert(property, value);
        }
    }

    declarations
}

/// Parse an inline style attribute.
#[must_use]
pub fn parse_inline_style(style: &str) -> StyleDeclarations {
    parse_declarations(style)
}

/// Apply a stylesheet to an SVG DOM.
pub fn apply_stylesheet(dom: &mut SvgDom, stylesheet: &Stylesheet) {
    apply_stylesheet_to_node(&mut dom.root, stylesheet, &[]);
}

fn apply_stylesheet_to_node(node: &mut SvgNode, stylesheet: &Stylesheet, ancestors: &[&SvgNode]) {
    // Collect matching rules with their specificity
    let mut matching_rules: Vec<(&CssRule, (u32, u32, u32))> = stylesheet
        .rules
        .iter()
        .filter(|rule| rule.selector.matches(node, ancestors))
        .map(|rule| (rule, rule.selector.specificity()))
        .collect();

    // Sort by specificity (lower first, so later rules override)
    matching_rules.sort_by_key(|(_, spec)| *spec);

    // Apply declarations in order
    for (rule, _) in matching_rules {
        apply_declarations_to_node(node, &rule.declarations);
    }

    // Also parse and apply inline style attribute if present
    if let Some(style) = node.attributes.get("style").cloned() {
        let inline_decls = parse_inline_style(&style);
        apply_declarations_to_node(node, &inline_decls);
    }

    // Recursively apply to children
    // We need to create a new ancestors list that includes this node
    let node_ptr = std::ptr::from_ref::<SvgNode>(node);
    let mut new_ancestors: Vec<&SvgNode> = ancestors.to_vec();
    // Safety: we're not modifying ancestors while iterating
    new_ancestors.push(unsafe { &*node_ptr });

    for child in &mut node.children {
        apply_stylesheet_to_node(child, stylesheet, &new_ancestors);
    }
}

fn apply_declarations_to_node(node: &mut SvgNode, declarations: &StyleDeclarations) {
    for (property, value) in declarations.iter() {
        apply_style_property(node, property, value);
    }
}

/// Apply a single style property to a node.
pub fn apply_style_property(node: &mut SvgNode, property: &str, value: &str) {
    match property {
        "fill" => {
            node.fill = parse_css_paint(value);
        }
        "stroke" => {
            node.stroke = parse_css_paint(value);
        }
        "stroke-width" => {
            node.stroke_width = Some(parse_css_length(value));
        }
        "color" => {
            if !value.trim().eq_ignore_ascii_case("inherit") {
                node.color = crate::parser::parse_color(value);
            }
        }
        "opacity" => {
            node.opacity = parse_opacity_value(value).unwrap_or(1.0);
        }
        "fill-opacity" => {
            // Fill opacity is an independent inherited property that
            // multiplies into the fill paint alpha at render time — it does
            // not mutate the fill color.
            node.fill_opacity = parse_opacity_value(value);
        }
        "stroke-opacity" => {
            node.stroke_opacity = parse_opacity_value(value);
        }
        "visibility" => {
            node.visible = value != "hidden";
        }
        "display" => {
            if value == "none" {
                node.visible = false;
            }
        }
        "transform" => {
            node.transform = parse_css_transform(value);
        }
        "font-family" => {
            if let crate::dom::SvgNodeKind::Text(ref mut text) = node.kind {
                text.font_family = Some(value.trim_matches('"').trim_matches('\'').to_string());
            }
        }
        "font-size" => {
            if let crate::dom::SvgNodeKind::Text(ref mut text) = node.kind {
                text.font_size = parse_css_length(value);
            }
        }
        "font-weight" => {
            if let crate::dom::SvgNodeKind::Text(ref mut text) = node.kind {
                text.font_weight = match value {
                    "normal" => 400,
                    "bold" => 700,
                    "lighter" => 300,
                    "bolder" => 800,
                    _ => value.parse().unwrap_or(400),
                };
            }
        }
        "text-anchor" => {
            if let crate::dom::SvgNodeKind::Text(ref mut text) = node.kind {
                text.text_anchor = match value {
                    "middle" => crate::dom::TextAnchor::Middle,
                    "end" => crate::dom::TextAnchor::End,
                    _ => crate::dom::TextAnchor::Start,
                };
            }
        }
        "stroke-linecap" | "stroke-linejoin" | "stroke-dasharray" | "stroke-dashoffset" => {
            // Store in attributes for later use
            node.attributes
                .insert(property.to_string(), value.to_string());
        }
        _ => {
            // Store unknown properties in attributes
            node.attributes
                .insert(property.to_string(), value.to_string());
        }
    }
}

fn parse_css_paint(s: &str) -> Option<SvgPaint> {
    let s = s.trim();
    if s == "transparent" {
        return Some(SvgPaint::None);
    }
    // Reuse the SVG `<paint>` grammar (none / currentColor / color /
    // url(#id) fallback) so CSS and presentation attributes agree.
    crate::parser::parse_paint(s)
}

/// Parse an `<opacity-value>` (number or percentage) clamped to `[0, 1]`.
fn parse_opacity_value(s: &str) -> Option<Scalar> {
    let s = s.trim();
    let v = if let Some(pct) = s.strip_suffix('%') {
        pct.trim().parse::<Scalar>().ok()? / 100.0
    } else {
        s.parse::<Scalar>().ok()?
    };
    Some(v.clamp(0.0, 1.0))
}

#[allow(
    clippy::option_if_let_else,
    reason = "five-way unit suffix dispatch reads far more clearly as an if/else-if chain than nested map_or_else calls"
)]
fn parse_css_length(s: &str) -> Scalar {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix("px") {
        stripped.parse().unwrap_or(0.0)
    } else if let Some(stripped) = s.strip_suffix("pt") {
        stripped.parse::<Scalar>().unwrap_or(0.0) * 1.333
    } else if let Some(stripped) = s.strip_suffix("rem") {
        stripped.parse::<Scalar>().unwrap_or(0.0) * 16.0
    } else if let Some(stripped) = s.strip_suffix("em") {
        stripped.parse::<Scalar>().unwrap_or(0.0) * 16.0
    } else if let Some(stripped) = s.strip_suffix('%') {
        // Percentage - context dependent, return as fraction
        stripped.parse::<Scalar>().unwrap_or(0.0) / 100.0
    } else {
        s.parse().unwrap_or(0.0)
    }
}

fn parse_css_transform(s: &str) -> Matrix {
    // Reuse the transform parser from the parser module
    crate::parser::parse_transform_str(s)
}

/// Extract embedded stylesheets from an SVG DOM.
#[must_use]
pub fn extract_stylesheets(dom: &SvgDom) -> Stylesheet {
    let mut stylesheet = Stylesheet::new();
    extract_stylesheets_from_node(&dom.root, &mut stylesheet);
    stylesheet
}

fn extract_stylesheets_from_node(node: &SvgNode, stylesheet: &mut Stylesheet) {
    // Check for style element content
    if let crate::dom::SvgNodeKind::Unknown(tag) = &node.kind
        && tag == "style"
        && let Some(content) = node.attributes.get("__text_content")
    {
        let parsed = Stylesheet::parse(content);
        stylesheet.rules.extend(parsed.rules);
    }

    for child in &node.children {
        extract_stylesheets_from_node(child, stylesheet);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_color as parse_css_color;
    use skia_rs_core::Color;

    #[test]
    fn test_parse_css_color() {
        assert_eq!(parse_css_color("#ff0000"), Some(Color::from_rgb(255, 0, 0)));
        assert_eq!(parse_css_color("#f00"), Some(Color::from_rgb(255, 0, 0)));
        assert_eq!(parse_css_color("red"), Some(Color::from_rgb(255, 0, 0)));
        assert_eq!(
            parse_css_color("rgb(255, 128, 0)"),
            Some(Color::from_rgb(255, 128, 0))
        );
    }

    #[test]
    fn test_parse_declarations() {
        let decls = parse_declarations("fill: red; stroke: blue; stroke-width: 2px");
        assert_eq!(decls.get("fill"), Some(&"red".to_string()));
        assert_eq!(decls.get("stroke"), Some(&"blue".to_string()));
        assert_eq!(decls.get("stroke-width"), Some(&"2px".to_string()));
    }

    #[test]
    fn test_declarations_preserve_document_order() {
        // Order must be deterministic (document order), not HashMap iteration
        // order. A repeated property keeps its first position but takes the
        // last value ("last wins").
        let decls = parse_declarations("fill: red; stroke: blue; fill: green");
        let order: Vec<&str> = decls.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(order, vec!["fill", "stroke"]);
        assert_eq!(decls.get("fill"), Some(&"green".to_string()));
    }

    #[test]
    fn test_fill_opacity_is_independent_property() {
        // fill-opacity sets the node property and does NOT mutate the fill
        // color; the two are combined at render time.
        use crate::dom::{SvgNode, SvgNodeKind, SvgRect};
        let mut node = SvgNode::new(SvgNodeKind::Rect(SvgRect::default()));
        apply_style_property(&mut node, "fill", "red");
        apply_style_property(&mut node, "fill-opacity", "0.25");
        assert!(matches!(
            node.fill,
            Some(SvgPaint::Color(c)) if c == Color::from_rgb(255, 0, 0)
        ));
        assert!((node.fill_opacity.unwrap() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_css_currentcolor_paint() {
        use crate::dom::{SvgNode, SvgNodeKind, SvgRect};
        let mut node = SvgNode::new(SvgNodeKind::Rect(SvgRect::default()));
        apply_style_property(&mut node, "fill", "currentColor");
        assert!(matches!(node.fill, Some(SvgPaint::CurrentColor)));
    }

    #[test]
    fn test_parse_stylesheet() {
        let css = r"
            rect { fill: red; }
            .highlight { stroke: yellow; }
            #main { opacity: 0.5; }
        ";

        let stylesheet = Stylesheet::parse(css);
        assert_eq!(stylesheet.rules.len(), 3);
    }

    #[test]
    fn test_selector_parse() {
        assert!(matches!(
            CssSelector::parse("rect"),
            CssSelector::Element(_)
        ));
        assert!(matches!(
            CssSelector::parse(".class"),
            CssSelector::Class(_)
        ));
        assert!(matches!(CssSelector::parse("#id"), CssSelector::Id(_)));
    }

    #[test]
    fn test_selector_specificity() {
        let elem = CssSelector::parse("rect");
        let class = CssSelector::parse(".highlight");
        let id = CssSelector::parse("#main");

        assert_eq!(elem.specificity(), (0, 0, 1));
        assert_eq!(class.specificity(), (0, 1, 0));
        assert_eq!(id.specificity(), (1, 0, 0));
    }

    #[test]
    fn test_descendant_selector() {
        let sel = CssSelector::parse("g rect");
        assert!(matches!(sel, CssSelector::Descendant(_, _)));
    }

    #[test]
    fn test_hsl_color() {
        let color = parse_css_color("hsl(0, 100%, 50%)").unwrap();
        // Should be red
        assert_eq!(color.red(), 255);
        assert!(color.green() < 10);
        assert!(color.blue() < 10);
    }

    #[test]
    fn test_apply_stylesheet_to_dom() {
        // Verify that apply_stylesheet actually modifies a DOM node's
        // resolved style based on a matching class selector.
        use crate::dom::{SvgDom, SvgNode, SvgNodeKind, SvgPaint, SvgRect};

        let mut dom = SvgDom::new();
        dom.width = 50.0;
        dom.height = 50.0;

        let mut rect = SvgNode::new(SvgNodeKind::Rect(SvgRect {
            x: 0.0,
            y: 0.0,
            width: 50.0,
            height: 50.0,
            rx: 0.0,
            ry: 0.0,
        }));
        rect.classes.push("accent".to_string());
        // Default fill is black; the stylesheet should overwrite it to
        // red.
        rect.fill = Some(SvgPaint::Color(Color::BLACK));
        dom.root.add_child(rect);

        let sheet = Stylesheet::parse(".accent { fill: #ff0000; opacity: 0.5; }");
        apply_stylesheet(&mut dom, &sheet);

        let styled = &dom.root.children[0];
        assert!(matches!(
            styled.fill,
            Some(SvgPaint::Color(c)) if c == Color::from_rgb(255, 0, 0)
        ));
        assert!((styled.opacity - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_apply_stylesheet_inline_overrides_rule() {
        // Inline styles are applied after selector-matched rules, so
        // they win even over a more specific id selector in the sheet.
        use crate::dom::{SvgDom, SvgNode, SvgNodeKind, SvgPaint, SvgRect};

        let mut dom = SvgDom::new();
        let mut rect = SvgNode::new(SvgNodeKind::Rect(SvgRect::default()));
        rect.id = Some("hero".to_string());
        rect.fill = Some(SvgPaint::Color(Color::BLACK));
        rect.attributes
            .insert("style".to_string(), "fill: green".to_string());
        dom.root.add_child(rect);

        let sheet = Stylesheet::parse("#hero { fill: red }");
        apply_stylesheet(&mut dom, &sheet);

        let styled = &dom.root.children[0];
        assert!(matches!(
            styled.fill,
            Some(SvgPaint::Color(c)) if c == Color::from_rgb(0, 128, 0)
        ));
    }
}
