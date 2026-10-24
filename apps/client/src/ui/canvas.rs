//! Small retained UI rasterizer. Source sprites remain at their original size;
//! the GPU expands the resulting logical pixels by an integer scale.
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;

use ab_glyph::{Font, FontVec, ScaleFont, point};

pub struct Sprite {
    pub width: u32,
    pub height: u32,
    pixels: Vec<u8>,
}
impl Sprite {
    pub fn decode(bytes: &[u8], expected: (u32, u32)) -> Result<Self, String> {
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
        let mut data = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut data).map_err(|e| e.to_string())?;
        if (info.width, info.height) != expected {
            return Err(format!(
                "UI sprite must be {}x{}, got {}x{}",
                expected.0, expected.1, info.width, info.height
            ));
        }
        let channels = info.color_type.samples();
        let mut pixels = Vec::with_capacity((info.width * info.height * 4) as usize);
        for p in data[..info.buffer_size()].chunks_exact(channels) {
            pixels.extend_from_slice(&match info.color_type {
                png::ColorType::Rgba => [p[0], p[1], p[2], p[3]],
                png::ColorType::Rgb => [p[0], p[1], p[2], 255],
                png::ColorType::Grayscale => [p[0], p[0], p[0], 255],
                png::ColorType::GrayscaleAlpha => [p[0], p[0], p[0], p[1]],
                png::ColorType::Indexed => return Err("unexpanded UI palette".into()),
            });
        }
        Ok(Self {
            width: info.width,
            height: info.height,
            pixels,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}
impl Rect {
    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
    pub fn contains(self, p: [i32; 2]) -> bool {
        p[0] >= self.x
            && p[1] >= self.y
            && p[0] < self.x + self.w as i32
            && p[1] < self.y + self.h as i32
    }
}
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize],
        }
    }
    fn blend(&mut self, x: i32, y: i32, source: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 || source[3] == 0 {
            return;
        }
        let i = ((y as u32 * self.width + x as u32) * 4) as usize;
        let dst = &mut self.pixels[i..i + 4];
        let sa = u32::from(source[3]);
        let da = u32::from(dst[3]);
        let a = sa * 255 + da * (255 - sa);
        for c in 0..3 {
            dst[c] =
                ((u32::from(source[c]) * sa * 255 + u32::from(dst[c]) * da * (255 - sa)) / a) as u8;
        }
        dst[3] = a.div_ceil(255) as u8;
    }
    pub fn fill(&mut self, rect: Rect, color: [u8; 4]) {
        for y in rect.y.max(0)..(rect.y + rect.h as i32).min(self.height as i32) {
            for x in rect.x.max(0)..(rect.x + rect.w as i32).min(self.width as i32) {
                self.blend(x, y, color);
            }
        }
    }
    pub fn sprite(&mut self, sprite: &Sprite, rect: Rect, border: u32) {
        // Nine-slice mappings preserve corners and stretch only the middle.
        let map = |p: u32, dst: u32, src: u32| {
            let b = border.min(src / 2).min(dst / 2);
            if p < b {
                p
            } else if p >= dst - b {
                src - (dst - p)
            } else {
                b + (p - b) * (src - 2 * b) / (dst - 2 * b).max(1)
            }
        };
        for y in 0..rect.h {
            let sy = map(y, rect.h, sprite.height);
            for x in 0..rect.w {
                let sx = map(x, rect.w, sprite.width);
                let i = ((sy * sprite.width + sx) * 4) as usize;
                self.blend(
                    rect.x + x as i32,
                    rect.y + y as i32,
                    sprite.pixels[i..i + 4].try_into().unwrap(),
                );
            }
        }
    }
    pub fn progress(&mut self, sprite: &Sprite, x: i32, y: i32, amount: f32, tint: [u8; 4]) {
        let width = (sprite.width as f32 * amount.clamp(0.0, 1.0)).round() as u32;
        for py in 0..sprite.height {
            for px in 0..width {
                let i = ((py * sprite.width + px) * 4) as usize;
                let mut pixel = [0; 4];
                for c in 0..4 {
                    pixel[c] = (u16::from(sprite.pixels[i + c]) * u16::from(tint[c]) / 255) as u8;
                }
                self.blend(x + px as i32, y + py as i32, pixel);
            }
        }
    }
}

struct Raster {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    advance: i32,
}
pub struct Text {
    font: FontVec,
    glyphs: BTreeMap<(char, u16), Raster>,
    pub chinese: bool,
}
impl Text {
    pub fn load() -> Result<Self, String> {
        let paths = if let Some(path) = std::env::var_os("TORSE_UI_FONT") {
            vec![PathBuf::from(path)]
        } else {
            let mut paths = Vec::new();
            if let Some(windir) = std::env::var_os("WINDIR") {
                paths.push(PathBuf::from(windir).join("Fonts/msyh.ttc"));
            }
            paths.extend(
                [
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/System/Library/Fonts/PingFang.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                ]
                .map(PathBuf::from),
            );
            paths
        };
        for path in paths {
            if let Ok(bytes) = std::fs::read(path)
                && let Ok(font) = FontVec::try_from_vec(bytes)
            {
                return Ok(Self {
                    chinese: font.glyph_id('\u{52a0}').0 != 0,
                    font,
                    glyphs: BTreeMap::new(),
                });
            }
        }
        Err("UI font unavailable; set TORSE_UI_FONT to a TTF, OTF or TTC file".into())
    }
    fn raster(&mut self, c: char, size: u16) -> &Raster {
        if self.glyphs.len() >= 1024 {
            self.glyphs.clear();
        }
        self.glyphs.entry((c, size)).or_insert_with(|| {
            let scaled = self.font.as_scaled(f32::from(size));
            let id = self.font.glyph_id(c);
            let advance = scaled.h_advance(id).ceil() as i32;
            let glyph =
                id.with_scale_and_position(f32::from(size), point(0.0, scaled.ascent().ceil()));
            if let Some(outline) = self.font.outline_glyph(glyph) {
                let bounds = outline.px_bounds();
                let (width, height) = (bounds.width() as u32, bounds.height() as u32);
                let mut pixels = vec![0; (width * height) as usize];
                outline.draw(|x, y, coverage| {
                    pixels[(y * width + x) as usize] = (coverage * 255.0) as u8
                });
                Raster {
                    pixels,
                    width,
                    height,
                    x: bounds.min.x as i32,
                    y: bounds.min.y as i32,
                    advance,
                }
            } else {
                Raster {
                    pixels: Vec::new(),
                    width: 0,
                    height: 0,
                    x: 0,
                    y: 0,
                    advance,
                }
            }
        })
    }
    pub fn measure(&mut self, value: &str, size: u16) -> u32 {
        value
            .chars()
            .take(160)
            .map(|c| self.raster(c, size).advance.max(0) as u32)
            .sum()
    }
    pub fn draw(
        &mut self,
        canvas: &mut Canvas,
        value: &str,
        rect: Rect,
        size: u16,
        color: [u8; 4],
    ) {
        let mut x = rect.x;
        for c in value.chars().take(160) {
            let glyph = self.raster(c, size);
            if x + glyph.advance > rect.x + rect.w as i32 {
                break;
            }
            for gy in 0..glyph.height {
                for gx in 0..glyph.width {
                    let y = rect.y + glyph.y + gy as i32;
                    if y < rect.y || y >= rect.y + rect.h as i32 {
                        continue;
                    }
                    let mut ink = color;
                    ink[3] = (u16::from(color[3])
                        * u16::from(glyph.pixels[(gy * glyph.width + gx) as usize])
                        / 255) as u8;
                    canvas.blend(x + glyph.x + gx as i32, y, ink);
                }
            }
            x += glyph.advance;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nine_slice_preserves_corners_and_transparency() {
        let sprite = Sprite {
            width: 3,
            height: 3,
            pixels: [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [0, 0, 0, 0],
                [40, 40, 40, 255],
                [0, 0, 0, 0],
                [255, 255, 0, 255],
                [0, 255, 255, 255],
                [255, 0, 255, 255],
            ]
            .concat(),
        };
        let mut canvas = Canvas::new(9, 9);
        canvas.sprite(&sprite, Rect::new(0, 0, 9, 9), 1);
        assert_eq!(&canvas.pixels[..4], &[255, 0, 0, 255]);
        assert_eq!(&canvas.pixels[32..36], &[0, 0, 255, 255]);
        assert_eq!(&canvas.pixels[160..164], &[40, 40, 40, 255]);
        assert_eq!(&canvas.pixels[36..40], &[0, 0, 0, 0]);
    }
    #[test]
    fn clipped_progress_does_not_stretch_the_fill() {
        let sprite = Sprite {
            width: 4,
            height: 1,
            pixels: [[255, 255, 255, 255]; 4].concat(),
        };
        let mut canvas = Canvas::new(4, 1);
        canvas.progress(&sprite, 0, 0, 0.5, [255; 4]);
        assert_eq!(canvas.pixels.iter().filter(|p| **p == 255).count(), 8);
    }
}
