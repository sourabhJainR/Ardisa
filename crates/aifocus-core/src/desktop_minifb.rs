//! Concrete cross-platform desktop host backed by minifb.
//!
//! This is a lightweight pixel-rendered host. It presents accessible names as
//! visible labels, but minifb does not expose native accessibility-tree APIs;
//! applications requiring screen-reader integration need a platform UI backend.
use std::collections::VecDeque;

use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

use crate::application::{
    AppCapability, AppError, AppPolicy, DesktopHost, UiAction, UiEvent, UiEventKind,
    UiNode, UiNodeKind, UiTree,
};

const WIDTH: usize = 800;
const HEIGHT: usize = 600;
const MAX_TEXT_BYTES: usize = 4_096;
const ROW_HEIGHT: usize = 52;
const LEFT: usize = 24;
const TEXT_TOP: usize = 16;

pub struct MinifbDesktopHost {
    window: Window,
    pixels: Vec<u32>,
    tree: Option<UiTree>,
    events: VecDeque<UiEvent>,
    mouse_was_down: bool,
    focused_input: Option<u32>,
    closed: bool,
}

impl MinifbDesktopHost {
    /// Open a native window only when the host policy grants desktop access.
    pub fn new(policy: &AppPolicy, title: &str) -> Result<Self, AppError> {
        policy.require(AppCapability::DesktopWindow)?;
        if title.trim().is_empty() || title.len() > 256 {
            return Err(AppError::InvalidRequest("desktop window title must contain 1..=256 bytes".into()));
        }
        let mut window = Window::new(title, WIDTH, HEIGHT, WindowOptions::default())
            .map_err(|error| AppError::Handler(format!("desktop window creation failed: {error}")))?;
        window.set_target_fps(30);
        Ok(Self {
            window,
            pixels: vec![0x00_F4_F6_F8; WIDTH * HEIGHT],
            tree: None,
            events: VecDeque::new(),
            mouse_was_down: false,
            focused_input: None,
            closed: false,
        })
    }

    pub fn is_open(&self) -> bool {
        !self.closed && self.window.is_open()
    }

    fn draw_tree(&mut self, tree: &UiTree) -> Result<(), AppError> {
        self.pixels.fill(0x00_F4_F6_F8);
        let mut row = 0usize;
        for node in tree.nodes().values() {
            if node.kind == UiNodeKind::Window { continue; }
            let y = TEXT_TOP + row * ROW_HEIGHT;
            if y + 40 >= HEIGHT { break; }
            let (background, foreground) = match node.kind {
                UiNodeKind::Button if node.enabled => (0x00_245A9A, 0x00_FFFFFF),
                UiNodeKind::Button => (0x00_888888, 0x00_FFFFFF),
                UiNodeKind::TextInput => (0x00_FFFFFF, 0x00_202020),
                _ => (0x00_F4_F6_F8, 0x00_202020),
            };
            if matches!(node.kind, UiNodeKind::Button | UiNodeKind::TextInput) {
                fill_rect(&mut self.pixels, LEFT, y, WIDTH - LEFT * 2, 36, background);
            }
            let text = if node.kind == UiNodeKind::TextInput {
                format!("{}: {}", node.accessible_name, node.text)
            } else if node.text.is_empty() {
                node.accessible_name.clone()
            } else {
                format!("{}: {}", node.accessible_name, node.text)
            };
            draw_text(&mut self.pixels, LEFT + 10, y + 12, &text, foreground);
            row += 1;
        }
        self.window.update_with_buffer(&self.pixels, WIDTH, HEIGHT)
            .map_err(|error| AppError::Handler(format!("desktop frame presentation failed: {error}")))
    }

    fn poll_native_events(&mut self) -> Result<(), AppError> {
        if !self.window.is_open() {
            if !self.closed {
                self.events.push_back(UiEvent { node_id: None, kind: UiEventKind::WindowClosed });
                self.closed = true;
            }
            return Ok(());
        }

        let pressed = self.window.get_keys_pressed(KeyRepeat::No);
        if let Some(node_id) = self.focused_input {
            let mut changed = None;
            if let Some(tree) = self.tree.as_mut() {
                if let Some(node) = tree.nodes().get(&node_id).filter(|node| node.kind == UiNodeKind::TextInput && node.enabled) {
                    let mut value = node.text.clone();
                    for key in pressed {
                        match key {
                            Key::Backspace => { value.pop(); }
                            Key::Space => value.push(' '),
                            Key::Enter => {}
                            other => {
                                let label = format!("{other:?}");
                                if label.len() == 1 && label.as_bytes()[0].is_ascii_alphabetic() {
                                    value.push_str(&label.to_ascii_lowercase());
                                } else if let Some(digit) = label.strip_prefix("Key") {
                                    if digit.len() == 1 && digit.as_bytes()[0].is_ascii_digit() {
                                        value.push_str(digit);
                                    }
                                }
                            }
                        }
                        while value.len() > MAX_TEXT_BYTES { value.pop(); }
                    }
                    if value != node.text { changed = Some(value); }
                }
                if let Some(value) = changed {
                    self.events.push_back(UiEvent {
                        node_id: Some(node_id),
                        kind: UiEventKind::TextChanged(value),
                    });
                }
            }
        }

        let mouse_down = self.window.get_mouse_down(MouseButton::Left);
        if mouse_down && !self.mouse_was_down {
            if let Some((x, y)) = self.window.get_mouse_pos(MouseMode::Discard) {
                let x = x as usize;
                let y = y as usize;
                let row = y.saturating_sub(TEXT_TOP) / ROW_HEIGHT;
                let inside_control = x >= LEFT && x < WIDTH - LEFT
                    && y >= TEXT_TOP + row * ROW_HEIGHT
                    && y < TEXT_TOP + row * ROW_HEIGHT + 36;
                let target = inside_control.then(|| self.tree.as_ref()).flatten().and_then(|tree| {
                    tree.nodes().values()
                        .filter(|node| node.kind != UiNodeKind::Window)
                        .nth(row)
                        .map(|node| (node.id, node.kind, node.enabled))
                });
                if let Some((id, kind, enabled)) = target {
                    if enabled {
                        match kind {
                            UiNodeKind::Button => self.events.push_back(UiEvent {
                                node_id: Some(id), kind: UiEventKind::Clicked,
                            }),
                            UiNodeKind::TextInput => self.focused_input = Some(id),
                            _ => self.focused_input = None,
                        }
                    }
                }
            }
        }
        self.mouse_was_down = mouse_down;
        Ok(())
    }
}

impl DesktopHost for MinifbDesktopHost {
    fn present(&mut self, tree: &UiTree) -> Result<(), AppError> {
        if self.closed { return Err(AppError::Handler("desktop window is closed".into())); }
        self.tree = Some(tree.clone());
        self.draw_tree(tree)
    }

    fn next_event(&mut self) -> Result<Option<UiEvent>, AppError> {
        self.poll_native_events()?;
        Ok(self.events.pop_front())
    }

    fn apply(&mut self, action: &UiAction) -> Result<(), AppError> {
        let tree = self.tree.as_mut()
            .ok_or_else(|| AppError::InvalidUiTree("present a UI tree before applying actions".into()))?;
        if tree.apply_action(action)? {
            self.window.set_title("Ardisa application (closing)");
            self.closed = true;
            return Ok(());
        }
        let snapshot = tree.clone();
        self.draw_tree(&snapshot)
    }
}

fn fill_rect(buffer: &mut [u32], x: usize, y: usize, width: usize, height: usize, color: u32) {
    let x_end = x.saturating_add(width).min(WIDTH);
    let y_end = y.saturating_add(height).min(HEIGHT);
    for py in y..y_end {
        let start = py * WIDTH + x.min(WIDTH);
        let end = py * WIDTH + x_end;
        if start < end { buffer[start..end].fill(color); }
    }
}

fn draw_text(buffer: &mut [u32], x: usize, y: usize, text: &str, color: u32) {
    let mut cursor = x;
    for character in text.chars().take(96) {
        let glyph = BASIC_FONTS.get(character).or_else(|| BASIC_FONTS.get('?'));
        if let Some(glyph) = glyph {
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in 0..8 {
                    if bits & (1 << gx) != 0 {
                        let px = cursor + gx;
                        let py = y + gy;
                        if px < WIDTH && py < HEIGHT { buffer[py * WIDTH + px] = color; }
                    }
                }
            }
        }
        cursor += 9;
        if cursor + 8 >= WIDTH { break; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_requires_explicit_desktop_capability() {
        let result = MinifbDesktopHost::new(&AppPolicy::default(), "Denied");
        assert!(matches!(result, Err(AppError::CapabilityDenied(AppCapability::DesktopWindow))));
    }

    #[test]
    fn pixel_primitives_respect_framebuffer_bounds() {
        let mut pixels = vec![0; WIDTH * HEIGHT];
        fill_rect(&mut pixels, WIDTH - 2, HEIGHT - 2, 20, 20, 7);
        assert_eq!(pixels[HEIGHT * WIDTH - 1], 7);
        draw_text(&mut pixels, WIDTH - 2, HEIGHT - 2, "overflow-safe", 9);
    }
}
