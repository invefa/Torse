//! Torse client.
//!
//! Two modes share the scene, mesher, and pipeline. `--capture` renders one
//! frame offscreen into a PNG without a window; `--window` opens an interactive
//! winit window with a fly camera, streaming sections, and continuous
//! rendering.

mod camera;
mod input;
mod net;
mod pick;
mod renderer;
mod scene;
mod ui;
mod window;

use std::process::ExitCode;
use std::time::Instant;

use camera::Camera;
use renderer::Renderer;
use scene::Scene;
use window::{WindowConfig, run_window};

const USAGE: &str = "usage: torse-client --capture <png> [--width N] [--height N] [--radius N] [--view overview|player]\n\
       torse-client --window [--width N] [--height N] [--radius N] [--frames N]\n\
                        [--screenshot <png>] [--autoplay] [--ui-preview hud|pause|machine]";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut capture: Option<String> = None;
    let mut window = false;
    let mut width = 960u32;
    let mut height = 540u32;
    let mut radius = 2i64;
    let mut radius_set = false;
    let mut view = String::from("overview");
    let mut frames: Option<u32> = None;
    let mut screenshot: Option<String> = None;
    let mut autoplay = false;
    let mut walk = false;
    let mut connect: Option<String> = None;
    let mut probe_outside = false;
    let mut ui_preview = None;
    let mut index = 0;
    while index < arguments.len() {
        let argument = arguments[index].clone();
        let mut value = || -> Option<String> {
            index += 1;
            arguments.get(index).cloned()
        };
        match argument.as_str() {
            "--capture" => capture = value(),
            "--window" => window = true,
            "--width" => {
                if let Some(parsed) = value().and_then(|v| v.parse().ok()) {
                    width = parsed;
                }
            }
            "--height" => {
                if let Some(parsed) = value().and_then(|v| v.parse().ok()) {
                    height = parsed;
                }
            }
            "--radius" => {
                if let Some(parsed) = value().and_then(|v| v.parse().ok()) {
                    radius = parsed;
                    radius_set = true;
                }
            }
            "--view" => view = value().unwrap_or(view),
            "--frames" => {
                if let Some(parsed) = value().and_then(|v| v.parse().ok()) {
                    frames = Some(parsed);
                }
            }
            "--screenshot" => screenshot = value(),
            "--autoplay" => autoplay = true,
            "--walk" => walk = true,
            "--connect" => connect = value(),
            "--probe-outside" => probe_outside = true,
            "--ui-preview" => match value().as_deref().map(ui::Mode::parse) {
                Some(Ok(mode)) => ui_preview = Some(mode),
                _ => {
                    eprintln!("--ui-preview requires hud, pause or machine");
                    return ExitCode::FAILURE;
                }
            },
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
        index += 1;
    }

    if window {
        let config = WindowConfig {
            radius: if radius_set { radius } else { 4 },
            width,
            height,
            frames,
            screenshot,
            autoplay,
            walk,
            connect,
            probe_outside,
            ui_mode: ui_preview.unwrap_or_default(),
        };
        match run_window(config) {
            Ok(summary) => {
                println!(
                    "adapter={} backend={} format={} frames={} streams={} rebuilds={} \
                     uploads={} screenshots={} actions_ok={} actions_rejected={} walk_ticks={} \
                     walker=({:.1},{:.1},{:.1}) session={} batches={} requests_sent={} requests_rejected={} baseline_sections={} \
                     sections={} meshes={} vertices={} faces={} elapsed_ms={}",
                    summary.adapter,
                    summary.backend,
                    summary.format,
                    summary.frames,
                    summary.streams,
                    summary.rebuilds,
                    summary.uploads,
                    summary.screenshots,
                    summary.actions_ok,
                    summary.actions_rejected,
                    summary.walk_ticks,
                    summary.walker.map_or(0.0, |p| p[0]),
                    summary.walker.map_or(0.0, |p| p[1]),
                    summary.walker.map_or(0.0, |p| p[2]),
                    summary.session.map_or(0, |session| session),
                    summary.batches,
                    summary.requests_sent,
                    summary.requests_rejected,
                    summary.baseline_sections,
                    summary.sections,
                    summary.meshes,
                    summary.vertices,
                    summary.faces,
                    summary.elapsed_ms,
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("client failed: {error}");
                ExitCode::FAILURE
            }
        }
    } else if let Some(path) = capture {
        match run(&path, width, height, radius, &view, ui_preview) {
            Ok(summary) => {
                println!("{summary}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("client failed: {error}");
                ExitCode::FAILURE
            }
        }
    } else {
        println!("{USAGE}");
        ExitCode::SUCCESS
    }
}

fn run(
    path: &str,
    width: u32,
    height: u32,
    radius: i64,
    view: &str,
    ui_preview: Option<ui::Mode>,
) -> Result<String, String> {
    let start = Instant::now();
    let mut scene = Scene::new(radius)?;
    scene.refresh_meshes();
    let (vertices, indices, faces) = scene.geometry();
    if indices.is_empty() {
        return Err("scene produced no geometry".to_owned());
    }
    let vertex_bytes = Scene::vertex_bytes(&vertices);

    let mut renderer = Renderer::new(width, height)?;
    renderer.upload(&vertex_bytes, &indices);

    let (eye, target) = match view {
        "player" => {
            let eye = scene.eye();
            (eye, [eye[0], eye[1] - 1.5, eye[2] - 12.0])
        }
        _ => ([40.0, 18.0, 40.0], [0.0, 6.0, 0.0]),
    };
    let camera = Camera::new(eye, target);
    renderer.render(camera.view_projection(width as f32 / height as f32));
    if let Some(mode) = ui_preview {
        renderer.render_ui(mode)?;
    }
    renderer.save_png(path)?;

    let elapsed = start.elapsed();
    Ok(format!(
        "sections={} meshes={} entities={} seq={} vertices={} indices={} faces={} eye=({:.1},{:.1},{:.1}) elapsed_ms={}",
        scene.sections().len(),
        scene.mesh_count(),
        scene.authority().entities().len(),
        scene.authority().seq().0,
        vertices.len(),
        indices.len(),
        faces,
        eye[0],
        eye[1],
        eye[2],
        elapsed.as_millis()
    ))
}
