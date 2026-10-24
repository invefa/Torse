//! Client UI foundation: asset skin, HUD, modal focus, and a clearly labeled
//! machine preview. No UI control has access to authoritative mutation.
mod canvas;
mod gpu;

use canvas::{Canvas, Rect, Sprite, Text};
use gpu::Overlay;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Game,
    Pause,
    MachinePreview,
}
impl Mode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "hud" => Ok(Self::Game),
            "pause" => Ok(Self::Pause),
            "machine" => Ok(Self::MachinePreview),
            _ => Err("UI preview must be hud, pause or machine".into()),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Resume,
    Preview,
    Quit,
}

#[derive(Default)]
pub struct State {
    pub mode: Mode,
    pub cursor: [f64; 2],
    armed: Option<Action>,
}
impl State {
    pub fn modal(&self) -> bool {
        self.mode != Mode::Game
    }
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.armed = None;
    }
    pub fn pointer(&mut self, down: bool, size: [u32; 2]) -> Option<Action> {
        let hit = Layout::new(size).hit(self.mode, self.cursor);
        if down {
            self.armed = hit;
            None
        } else {
            self.armed.take().filter(|a| Some(*a) == hit)
        }
    }
}

struct Layout {
    scale: u32,
    width: u32,
    height: u32,
}
impl Layout {
    fn new(size: [u32; 2]) -> Self {
        let scale = (size[0] / 480).min(size[1] / 270).clamp(1, 4);
        Self {
            scale,
            width: size[0].max(1).div_ceil(scale),
            height: size[1].max(1).div_ceil(scale),
        }
    }
    fn panel(&self, mode: Mode) -> Rect {
        let (w, h) = if mode == Mode::Pause {
            (244, 168)
        } else {
            (316, 186)
        };
        let w = w.min(self.width.saturating_sub(8));
        let h = h.min(self.height.saturating_sub(8));
        Rect::new(
            (self.width - w) as i32 / 2,
            (self.height - h) as i32 / 2,
            w,
            h,
        )
    }
    fn buttons(&self, mode: Mode) -> Vec<(Action, Rect, &'static str)> {
        let p = self.panel(mode);
        match mode {
            Mode::Game => Vec::new(),
            Mode::Pause => vec![
                (
                    Action::Resume,
                    Rect::new(p.x + 24, p.y + 44, p.w.saturating_sub(48), 24),
                    "menu_resume",
                ),
                (
                    Action::Preview,
                    Rect::new(p.x + 24, p.y + 74, p.w.saturating_sub(48), 24),
                    "menu_preview",
                ),
                (
                    Action::Quit,
                    Rect::new(p.x + 24, p.y + 104, p.w.saturating_sub(48), 24),
                    "menu_quit",
                ),
            ],
            Mode::MachinePreview => vec![(
                Action::Resume,
                Rect::new(p.x + p.w as i32 / 2 - 48, p.y + 150, 96, 24),
                "close",
            )],
        }
    }
    fn close(&self, mode: Mode) -> Rect {
        let p = self.panel(mode);
        Rect::new(p.x + p.w as i32 - 26, p.y + 6, 20, 20)
    }
    fn hit(&self, mode: Mode, cursor: [f64; 2]) -> Option<Action> {
        if mode == Mode::Game {
            return None;
        }
        let p = [
            (cursor[0] / f64::from(self.scale)) as i32,
            (cursor[1] / f64::from(self.scale)) as i32,
        ];
        if self.close(mode).contains(p) {
            return Some(Action::Resume);
        }
        self.buttons(mode)
            .into_iter()
            .find(|(_, rect, _)| rect.contains(p))
            .map(|(a, _, _)| a)
    }
}

struct Skin {
    sprites: Vec<Sprite>,
}
impl Skin {
    fn load() -> Result<Self, String> {
        macro_rules! asset {
            ($name:literal) => {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../games/base/assets/",
                    $name
                )) as &[u8]
            };
        }
        let sources = [
            (asset!("ui/panel_window.png"), (24, 24)),
            (asset!("ui/panel_tooltip.png"), (16, 16)),
            (asset!("ui/button_normal.png"), (24, 20)),
            (asset!("ui/button_hover.png"), (24, 20)),
            (asset!("ui/button_pressed.png"), (24, 20)),
            (asset!("ui/slot_normal.png"), (24, 24)),
            (asset!("ui/slot_selected.png"), (24, 24)),
            (asset!("ui/progress_bg.png"), (64, 8)),
            (asset!("ui/progress_fill.png"), (60, 4)),
            (asset!("ui/crosshair.png"), (9, 9)),
            (asset!("ui/icon_close.png"), (12, 12)),
            (asset!("icons/items/stone.png"), (16, 16)),
            (asset!("icons/items/glass_slab.png"), (16, 16)),
            (asset!("icons/items/plant.png"), (16, 16)),
        ];
        Ok(Self {
            sprites: sources
                .into_iter()
                .map(|(bytes, size)| Sprite::decode(bytes, size))
                .collect::<Result<_, _>>()?,
        })
    }
    fn draw(&self, canvas: &mut Canvas, index: usize, rect: Rect, border: u32) {
        canvas.sprite(&self.sprites[index], rect, border);
    }
}

#[derive(PartialEq, Eq)]
struct View {
    size: [u32; 2],
    mode: Mode,
    material: usize,
    network: bool,
    ready: bool,
    result: String,
    hover: Option<Action>,
    pressed: Option<Action>,
}

pub struct Layer {
    overlay: Overlay,
    skin: Skin,
    text: Text,
    labels: BTreeMap<&'static str, &'static str>,
    last: Option<View>,
}
pub struct Hud<'a> {
    pub material: usize,
    pub network: bool,
    pub ready: bool,
    pub result: &'a str,
}
const DARK: [u8; 4] = [34, 40, 48, 255];
const MUTED: [u8; 4] = [78, 86, 96, 255];
impl Layer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Result<Self, String> {
        let text = Text::load()?;
        let resource = if text.chinese {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../games/base/assets/locales/zh-Hans.txt"
            ))
        } else {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../games/base/assets/locales/en.txt"
            ))
        };
        Ok(Self {
            overlay: Overlay::new(device, format),
            skin: Skin::load()?,
            text,
            labels: resource
                .lines()
                .filter_map(|line| line.split_once('='))
                .collect(),
            last: None,
        })
    }
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: [u32; 2],
        state: &State,
        hud: Hud<'_>,
    ) {
        let layout = Layout::new(size);
        let view = View {
            size,
            mode: state.mode,
            material: hud.material.min(2),
            network: hud.network,
            ready: hud.ready,
            result: hud.result.chars().take(120).collect(),
            hover: layout.hit(state.mode, state.cursor),
            pressed: state.armed,
        };
        if self.last.as_ref() == Some(&view) {
            return;
        }
        let mut canvas = Canvas::new(layout.width, layout.height);
        self.paint(&mut canvas, &layout, &view);
        self.overlay.upload(device, queue, &canvas, layout.scale);
        self.last = Some(view);
    }
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.overlay.draw(encoder, target);
    }
    fn label(&mut self, canvas: &mut Canvas, key: &str, rect: Rect, size: u16, color: [u8; 4]) {
        let value = self.labels.get(key).copied().unwrap_or("?");
        self.text.draw(canvas, value, rect, size, color);
    }
    fn shadow_label(&mut self, canvas: &mut Canvas, key: &str, rect: Rect, size: u16) {
        self.label(
            canvas,
            key,
            Rect::new(rect.x + 1, rect.y + 1, rect.w, rect.h),
            size,
            [15, 19, 25, 240],
        );
        self.label(canvas, key, rect, size, [245, 247, 250, 255]);
    }
    fn paint(&mut self, c: &mut Canvas, l: &Layout, v: &View) {
        let w = l.width as i32;
        let h = l.height as i32;
        let slot_x = w / 2 - 38;
        let slot_y = h - 36;
        for i in 0..3 {
            let x = slot_x + i as i32 * 26;
            self.skin.draw(
                c,
                if i == v.material { 6 } else { 5 },
                Rect::new(x, slot_y, 24, 24),
                0,
            );
            self.skin
                .draw(c, 11 + i, Rect::new(x + 4, slot_y + 4, 16, 16), 0);
            self.text.draw(
                c,
                &(i + 1).to_string(),
                Rect::new(x + 2, slot_y + 1, 10, 10),
                9,
                DARK,
            );
        }
        let selected = ["item_stone", "item_glass", "item_plant"][v.material];
        let width = self.text.measure(self.labels[selected], 12);
        self.shadow_label(
            c,
            selected,
            Rect::new(w / 2 - width as i32 / 2, slot_y - 18, width, 16),
            12,
        );
        let info = if !v.network {
            "hud_local"
        } else if v.ready {
            "hud_online"
        } else {
            "hud_connecting"
        };
        let info_width = (self.text.measure(self.labels[info], 12) + 20).max(72);
        self.skin.draw(c, 1, Rect::new(8, 8, info_width, 22), 3);
        // Tooltip centres are intentionally transparent in the supplied skin.
        c.fill(Rect::new(12, 12, info_width - 8, 14), [232, 236, 241, 255]);
        self.label(c, info, Rect::new(16, 12, info_width - 16, 16), 12, DARK);
        self.shadow_label(
            c,
            "hud_help",
            Rect::new(10, h - 15, l.width.saturating_sub(20), 14),
            10,
        );
        if !v.result.is_empty() {
            c.fill(
                Rect::new(8, 34, l.width.saturating_sub(16), 20),
                [20, 24, 30, 210],
            );
            self.text.draw(
                c,
                &v.result,
                Rect::new(13, 37, l.width.saturating_sub(26), 15),
                11,
                [245, 247, 250, 255],
            );
        }
        if v.mode == Mode::Game {
            self.skin
                .draw(c, 9, Rect::new(w / 2 - 4, h / 2 - 4, 9, 9), 0);
            return;
        }
        c.fill(Rect::new(0, 0, l.width, l.height), [0, 0, 0, 120]);
        let p = l.panel(v.mode);
        c.fill(
            Rect::new(
                p.x + 3,
                p.y + 3,
                p.w.saturating_sub(6),
                p.h.saturating_sub(6),
            ),
            [232, 236, 241, 255],
        );
        self.skin.draw(c, 0, p, 4);
        let title = if v.mode == Mode::Pause {
            "menu_title"
        } else {
            "machine_title"
        };
        self.label(
            c,
            title,
            Rect::new(p.x + 16, p.y + 12, p.w.saturating_sub(48), 22),
            16,
            DARK,
        );
        let close = l.close(v.mode);
        self.skin
            .draw(c, 10, Rect::new(close.x + 4, close.y + 4, 12, 12), 0);
        if v.mode == Mode::Pause {
            self.label(
                c,
                "menu_note",
                Rect::new(p.x + 16, p.y + 140, p.w.saturating_sub(32), 15),
                10,
                MUTED,
            );
        } else {
            self.label(
                c,
                "machine_notice",
                Rect::new(p.x + 16, p.y + 38, p.w.saturating_sub(32), 17),
                11,
                MUTED,
            );
            self.label(
                c,
                "machine_input",
                Rect::new(p.x + 22, p.y + 63, 54, 17),
                12,
                DARK,
            );
            self.label(
                c,
                "machine_output",
                Rect::new(p.x + 244, p.y + 63, 54, 17),
                12,
                DARK,
            );
            self.skin
                .draw(c, 5, Rect::new(p.x + 26, p.y + 83, 24, 24), 0);
            self.skin
                .draw(c, 11, Rect::new(p.x + 30, p.y + 87, 16, 16), 0);
            self.text
                .draw(c, "1", Rect::new(p.x + 42, p.y + 98, 8, 10), 9, DARK);
            self.skin
                .draw(c, 5, Rect::new(p.x + 252, p.y + 83, 24, 24), 0);
            self.label(
                c,
                "empty",
                Rect::new(p.x + 252, p.y + 111, 38, 15),
                10,
                MUTED,
            );
            self.label(
                c,
                "machine_progress",
                Rect::new(p.x + 106, p.y + 63, 100, 17),
                12,
                DARK,
            );
            self.skin
                .draw(c, 7, Rect::new(p.x + 114, p.y + 87, 64, 8), 0);
            c.fill(Rect::new(p.x + 116, p.y + 89, 60, 4), [197, 204, 213, 255]);
            c.progress(
                &self.skin.sprites[8],
                p.x + 116,
                p.y + 89,
                0.75,
                [83, 138, 164, 255],
            );
            self.text.draw(
                c,
                "3 / 4",
                Rect::new(p.x + 134, p.y + 103, 64, 15),
                11,
                DARK,
            );
            self.label(
                c,
                "machine_status",
                Rect::new(p.x + 16, p.y + 126, 140, 17),
                12,
                DARK,
            );
        }
        for (action, rect, key) in l.buttons(v.mode) {
            let index = if v.pressed == Some(action) && v.hover == Some(action) {
                4
            } else if v.hover == Some(action) {
                3
            } else {
                2
            };
            self.skin.draw(c, index, rect, 4);
            let label = self.labels.get(key).copied().unwrap_or("?");
            let width = self.text.measure(label, 12).min(rect.w.saturating_sub(16));
            self.text.draw(
                c,
                label,
                Rect::new(rect.x + (rect.w - width) as i32 / 2, rect.y + 5, width, 18),
                12,
                DARK,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supplied_assets_decode_at_the_declared_dimensions() {
        assert_eq!(Skin::load().unwrap().sprites.len(), 14);
    }
    #[test]
    fn modal_buttons_require_press_and_release_on_the_same_target() {
        let mut state = State::default();
        state.set_mode(Mode::Pause);
        let l = Layout::new([960, 540]);
        let r = l.buttons(Mode::Pause)[0].1;
        state.cursor = [f64::from((r.x + 5) * 2), f64::from((r.y + 5) * 2)];
        assert_eq!(state.pointer(true, [960, 540]), None);
        state.cursor = [0.0, 0.0];
        assert_eq!(state.pointer(false, [960, 540]), None);
        state.cursor = [f64::from((r.x + 5) * 2), f64::from((r.y + 5) * 2)];
        state.pointer(true, [960, 540]);
        assert_eq!(state.pointer(false, [960, 540]), Some(Action::Resume));
    }
    #[test]
    fn scaled_hit_testing_agrees_with_small_and_large_layouts() {
        for size in [[640, 360], [960, 540], [1920, 1080]] {
            let l = Layout::new(size);
            let r = l.close(Mode::MachinePreview);
            let cursor = [
                f64::from((r.x + 10) * l.scale as i32),
                f64::from((r.y + 10) * l.scale as i32),
            ];
            assert_eq!(l.hit(Mode::MachinePreview, cursor), Some(Action::Resume));
            assert_eq!(l.hit(Mode::Game, cursor), None);
        }
    }
}
