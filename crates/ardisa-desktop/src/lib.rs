//! Cross-platform native desktop host using minifb.
//!
//! The renderer deliberately supports the bounded UiTree controls in the
//! current application ABI. Native accessibility APIs and advanced text
//! shaping are not claimed by this first backend.
use std::collections::VecDeque;
use ardisa_core::application::{AppError, DesktopHost, UiAction, UiEvent, UiEventKind, UiNode, UiNodeKind, UiTree};
use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

const DEFAULT_WIDTH: usize = 800;
const DEFAULT_HEIGHT: usize = 600;
const MAX_DIMENSION: usize = 4096;
const MAX_PIXELS: usize = 16_777_216;
const BACKGROUND: u32 = 0x00F5F6F8;
const FOREGROUND: u32 = 0x001D2433;
const BORDER: u32 = 0x00778391;
const ACCENT: u32 = 0x002D6CDF;
const DISABLED: u32 = 0x00A0A6AE;

#[derive(Debug, Clone, Copy)]
pub struct DesktopConfig {
    pub width: usize,
    pub height: usize,
    pub title: &'static str,
    pub max_nodes: usize,
}
impl Default for DesktopConfig {
    fn default() -> Self { Self { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT, title: "Ardisa Application", max_nodes: 512 } }
}

pub struct MinifbDesktopHost {
    window: Window,
    config: DesktopConfig,
    buffer: Vec<u32>,
    events: VecDeque<UiEvent>,
    focused_input: Option<u32>,
    input_values: std::collections::BTreeMap<u32, String>,
    previous_mouse_down: bool,
    close_emitted: bool,
    close_requested: bool,
    enabled_overrides: std::collections::BTreeMap<u32, bool>,
}
impl MinifbDesktopHost {
    pub fn new(config: DesktopConfig) -> Result<Self, AppError> {
        if config.width == 0 || config.height == 0 || config.width > MAX_DIMENSION || config.height > MAX_DIMENSION
            || config.width.checked_mul(config.height).filter(|pixels| *pixels <= MAX_PIXELS).is_none()
            || config.max_nodes == 0 || config.max_nodes > 4096 {
            return Err(AppError::InvalidUiTree("desktop window dimensions or node limit are outside configured bounds".into()));
        }
        let window = Window::new(config.title, config.width, config.height, WindowOptions::default())
            .map_err(|error| AppError::Handler(format!("could not create native window: {error}")))?;
        Ok(Self { window, config, buffer: vec![BACKGROUND; config.width * config.height],
            events: VecDeque::new(), focused_input: None, input_values: Default::default(),
            previous_mouse_down: false, close_emitted: false, close_requested: false, enabled_overrides: Default::default() })
    }

    fn draw_text(&mut self, x: usize, y: usize, value: &str, color: u32) {
        let mut cursor = x;
        for character in value.chars().take(96) {
            let glyph = BASIC_FONTS.get(character).or_else(|| BASIC_FONTS.get('?')).expect("ASCII fallback font");
            for (row, bits) in glyph.iter().enumerate() {
                for column in 0..8 {
                    if bits & (1 << column) != 0 {
                        let px = cursor + column;
                        let py = y + row;
                        if px < self.config.width && py < self.config.height {
                            self.buffer[py * self.config.width + px] = color;
                        }
                    }
                }
            }
            cursor = cursor.saturating_add(9);
            if cursor >= self.config.width { break; }
        }
    }
    fn rect(&mut self, x: usize, y: usize, width: usize, height: usize, color: u32) {
        let right = x.saturating_add(width).min(self.config.width);
        let bottom = y.saturating_add(height).min(self.config.height);
        for py in y.min(self.config.height)..bottom {
            for px in x.min(self.config.width)..right { self.buffer[py * self.config.width + px] = color; }
        }
    }
    fn bounds(&self, index: usize) -> (usize, usize, usize, usize) {
        let row_height = 48usize;
        (24, 24 + index.saturating_mul(row_height), self.config.width.saturating_sub(48), 36)
    }
    fn hit_test<'a>(&self, tree: &'a UiTree, x: usize, y: usize) -> Option<&'a UiNode> {
        tree.nodes().values().filter(|node| node.kind != UiNodeKind::Window && node.kind != UiNodeKind::Container)
            .enumerate().find_map(|(index, node)| {
                let (left, top, width, height) = self.bounds(index);
                (x >= left && x < left.saturating_add(width) && y >= top && y < top.saturating_add(height)).then_some(node)
            })
    }
    fn draw_tree(&mut self, tree: &UiTree) {
        self.buffer.fill(BACKGROUND);
        for (index, node) in tree.nodes().values().filter(|node| node.kind != UiNodeKind::Window && node.kind != UiNodeKind::Container).cloned().enumerate() {
            let (x, y, width, height) = self.bounds(index);
            if y >= self.config.height { break; }
            let enabled = self.enabled_overrides.get(&node.id).copied().unwrap_or(node.enabled);
            let text = self.input_values.get(&node.id).cloned().unwrap_or_else(|| node.text.clone());
            match node.kind {
                UiNodeKind::Label => {
                    self.draw_text(x + 4, y + 12, &text, if enabled { FOREGROUND } else { DISABLED });
                }
                UiNodeKind::Button => {
                    self.rect(x, y, width, height, if enabled { ACCENT } else { DISABLED });
                    self.draw_text(x + 12, y + 13, &text, 0x00FFFFFF);
                }
                UiNodeKind::TextInput => {
                    self.rect(x, y, width, height, 0x00FFFFFF);
                    self.rect(x, y, width, 1, if self.focused_input == Some(node.id) { ACCENT } else { BORDER });
                    self.rect(x, y + height - 1, width, 1, BORDER);
                    self.rect(x, y, 1, height, BORDER);
                    self.rect(x + width - 1, y, 1, height, BORDER);
                    let value = self.input_values.get(&node.id).cloned().unwrap_or_else(|| node.text.clone());
                    self.draw_text(x + 8, y + 13, &value, if enabled { FOREGROUND } else { DISABLED });
                }
                _ => {}
            }
            if !node.accessible_name.is_empty() && node.kind == UiNodeKind::TextInput {
                self.draw_text(x, y.saturating_sub(12), &node.accessible_name, FOREGROUND);
            }
        }
    }
    fn pump(&mut self, tree: &UiTree) -> Result<(), AppError> {
        if self.close_requested || !self.window.is_open() {
            if !self.close_emitted { self.events.push_back(UiEvent { node_id: None, kind: UiEventKind::WindowClosed }); self.close_emitted = true; }
            return Ok(());
        }
        self.draw_tree(tree);
        self.window.update_with_buffer(&self.buffer, self.config.width, self.config.height)
            .map_err(|error| AppError::Handler(format!("native window update failed: {error}")))?;
        let mouse_down = self.window.get_mouse_down(MouseButton::Left);
        if mouse_down && !self.previous_mouse_down {
            if let Some((mx, my)) = self.window.get_mouse_pos(MouseMode::Clamp) {
                let selected = self.hit_test(tree, mx as usize, my as usize)
                    .map(|node| (node.id, node.kind, node.enabled, node.text.clone()));
                if let Some((node_id, kind, node_enabled, initial_text)) = selected {
                    let enabled = self.enabled_overrides.get(&node_id).copied().unwrap_or(node_enabled);
                    match kind {
                        UiNodeKind::Button if enabled => self.events.push_back(UiEvent { node_id: Some(node_id), kind: UiEventKind::Clicked }),
                        UiNodeKind::TextInput if enabled => {
                            self.input_values.entry(node_id).or_insert(initial_text);
                            self.focused_input = Some(node_id);
                        }
                        _ => self.focused_input = None,
                    }
                } else { self.focused_input = None; }
            }
        }
        self.previous_mouse_down = mouse_down;
        if let Some(id) = self.focused_input {
            let initial = tree.nodes().get(&id).map(|node| node.text.clone()).unwrap_or_default();
            let mut value = self.input_values.get(&id).cloned().unwrap_or(initial);
            let mut changed = false;
            for key in self.window.get_keys_pressed(KeyRepeat::Yes) {
                match key {
                    Key::Backspace => { value.pop(); changed = true; }
                    Key::Space if value.len() < 4096 => { value.push(' '); changed = true; }
                    Key::Enter => { self.events.push_back(UiEvent { node_id: Some(id), kind: UiEventKind::TextChanged(value.clone()) }); }
                    _ => if let Some(character) = key_to_char(key) {
                        if value.len() < 4096 { value.push(character); changed = true; }
                    }
                }
            }
            if changed {
                self.input_values.insert(id, value.clone());
                self.events.push_back(UiEvent { node_id: Some(id), kind: UiEventKind::TextChanged(value) });
            }
        }
        Ok(())
    }
}
impl DesktopHost for MinifbDesktopHost {
    fn present(&mut self, tree: &UiTree) -> Result<(), AppError> {
        if tree.nodes().len() > self.config.max_nodes { return Err(AppError::InvalidUiTree("desktop UI node limit exceeded".into())); }
        self.pump(tree)
    }
    fn next_event(&mut self) -> Result<Option<UiEvent>, AppError> { Ok(self.events.pop_front()) }
    fn apply(&mut self, action: &UiAction) -> Result<(), AppError> {
        match action {
            UiAction::CloseWindow => { self.close_requested = true; },
            UiAction::SetText { node_id, text } if text.len() <= 4096 => { self.input_values.insert(*node_id, text.clone()); }
            UiAction::SetText { .. } => return Err(AppError::InvalidUiTree("desktop text exceeds byte limit".into())),
            UiAction::SetEnabled { node_id, enabled } => { self.enabled_overrides.insert(*node_id, *enabled); }
        }
        Ok(())
    }
}
fn key_to_char(key: Key) -> Option<char> {
    match key {
        Key::A=>Some('a'),Key::B=>Some('b'),Key::C=>Some('c'),Key::D=>Some('d'),Key::E=>Some('e'),Key::F=>Some('f'),Key::G=>Some('g'),Key::H=>Some('h'),Key::I=>Some('i'),Key::J=>Some('j'),Key::K=>Some('k'),Key::L=>Some('l'),Key::M=>Some('m'),Key::N=>Some('n'),Key::O=>Some('o'),Key::P=>Some('p'),Key::Q=>Some('q'),Key::R=>Some('r'),Key::S=>Some('s'),Key::T=>Some('t'),Key::U=>Some('u'),Key::V=>Some('v'),Key::W=>Some('w'),Key::X=>Some('x'),Key::Y=>Some('y'),Key::Z=>Some('z'),
        Key::Key0=>Some('0'),Key::Key1=>Some('1'),Key::Key2=>Some('2'),Key::Key3=>Some('3'),Key::Key4=>Some('4'),Key::Key5=>Some('5'),Key::Key6=>Some('6'),Key::Key7=>Some('7'),Key::Key8=>Some('8'),Key::Key9=>Some('9'),Key::Period=>Some('.'),Key::Comma=>Some(','),Key::Minus=>Some('-'),Key::Apostrophe=>Some('\''),_=>None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn desktop_dimensions_are_bounded_before_window_creation() {
        let result = MinifbDesktopHost::new(DesktopConfig { width: usize::MAX, ..DesktopConfig::default() });
        assert!(matches!(result, Err(AppError::InvalidUiTree(_))));
    }
    #[test]
    fn key_mapping_is_ascii_and_deterministic() {
        assert_eq!(key_to_char(Key::A), Some('a'));
        assert_eq!(key_to_char(Key::Key9), Some('9'));
        assert_eq!(key_to_char(Key::Escape), None);
    }
}
