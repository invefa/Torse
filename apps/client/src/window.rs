//! Interactive windowed client.
//!
//! One winit window driven by [`ApplicationHandler`]. Each frame integrates the
//! fly camera from aggregated input, streams scene sections around the camera,
//! rebuilds dirty meshes, and draws through the shared [`GpuContext`] into the
//! window surface. Sections outside the radius unload, so sustained flight
//! keeps the resident set bounded.

use std::sync::Arc;
use std::time::Instant;

use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, CurrentSurfaceTexture, Device,
    DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, PollType, Queue,
    RequestAdapterOptions, Surface, SurfaceConfiguration, SurfaceTexture, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::camera::FlyCamera;
use crate::input::InputState;
use crate::pick::{PickHit, PickOutcome};
use crate::renderer::GpuContext;
use crate::scene::{Scene, Vertex};
use torse_types::{BlockPos, ContentKey, SectionPos};
use torse_world::Position;

/// How far the pick ray reaches, in blocks.
const PICK_RANGE: f32 = 64.0;
/// Autoplay fires one demonstration action every this many frames.
const AUTOPLAY_PERIOD: u32 = 24;
/// Camera eye height above the walker's body centre.
const WALKER_EYE_OFFSET: f64 = 0.7;

/// Startup parameters for the windowed run.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    /// Section loading radius around the camera.
    pub radius: i64,
    /// Initial window size in logical pixels.
    pub width: u32,
    pub height: u32,
    /// Exit automatically after this many presented frames.
    pub frames: Option<u32>,
    /// Save the final presented frame to this path before exiting.
    pub screenshot: Option<String>,
    /// Fly forward automatically instead of waiting for keys.
    pub autoplay: bool,
    /// Start in walk mode with a player entity on the ground.
    pub walk: bool,
    /// Connect to a standalone server and mirror its authorized projection.
    pub connect: Option<String>,
    /// Send one edit request far outside the approved scope, for acceptance
    /// evidence that the server rejects out-of-scope targets.
    pub probe_outside: bool,
    /// Initial UI screen; machine mode is explicitly a sample-data preview.
    pub ui_mode: crate::ui::Mode,
}

/// Measurements reported when the window closes.
#[derive(Clone, Debug, Default)]
pub struct WindowSummary {
    /// Adapter name reported by the backend.
    pub adapter: String,
    /// Backend selected for the surface.
    pub backend: String,
    /// Surface colour format the pipeline was built for.
    pub format: String,
    /// Presented frames.
    pub frames: u32,
    /// Times the section window moved with the camera.
    pub streams: usize,
    /// Section meshes rebuilt.
    pub rebuilds: usize,
    /// Geometry uploads to the GPU.
    pub uploads: usize,
    /// Screenshots written.
    pub screenshots: usize,
    /// Actions planned and committed through the action service.
    pub actions_ok: u32,
    /// Actions rejected by the action service or the commit.
    pub actions_rejected: u32,
    /// Walk ticks committed while in walk mode.
    pub walk_ticks: u32,
    /// Final walker position, when walk mode was entered.
    pub walker: Option<[f64; 3]>,
    /// Server session, when a connection was established.
    pub session: Option<u128>,
    /// Replica increments applied.
    pub batches: u32,
    /// Edit requests sent over the connection.
    pub requests_sent: u32,
    /// Requests the server rejected at admission.
    pub requests_rejected: u32,
    /// Sections received in the baseline.
    pub baseline_sections: usize,
    /// Resident sections at exit.
    pub sections: usize,
    /// Resident meshes at exit.
    pub meshes: usize,
    /// Final vertex and face counts.
    pub vertices: usize,
    pub faces: usize,
    /// Wall time from resume to exit.
    pub elapsed_ms: u128,
}

struct Gpu {
    ui: crate::ui::Layer,
    context: GpuContext,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    depth: Texture,
    depth_view: TextureView,
    window: Arc<Window>,
    backend_name: String,
}

struct WindowApp {
    ui: crate::ui::State,
    config: WindowConfig,
    scene: Option<Scene>,
    camera: Option<FlyCamera>,
    input: InputState,
    gpu: Option<Gpu>,
    materials: Vec<ContentKey>,
    material: usize,
    center: Option<SectionPos>,
    uploaded: bool,
    frames: u32,
    streams: usize,
    rebuilds: usize,
    uploads: usize,
    screenshots: usize,
    actions_ok: u32,
    actions_rejected: u32,
    autoplay_break: bool,
    net: Option<crate::net::Connection<torse_net::TcpTransport>>,
    session: Option<u128>,
    probe_sent: bool,
    batches: u32,
    requests_sent: u32,
    requests_rejected: u32,
    walk: bool,
    tick_accumulator: f32,
    walk_ticks: u32,
    /// Autoplay sets this when the walker stopped against a terrain step.
    walk_blocked: bool,
    last_result: String,
    adapter: String,
    started: Option<Instant>,
    last_frame: Option<Instant>,
    fps_origin: Instant,
    fps_frames: u32,
    pending_shot: Option<String>,
    summary: Option<WindowSummary>,
    failure: Option<String>,
}

impl WindowApp {
    fn new(config: WindowConfig) -> Self {
        let materials = ["base:stone", "base:glass_slab", "base:plant"]
            .into_iter()
            .map(|name| ContentKey::parse(name).expect("static material key"))
            .collect();
        Self {
            ui: {
                let mut state = crate::ui::State::default();
                state.set_mode(config.ui_mode);
                state
            },
            config,
            scene: None,
            camera: None,
            input: InputState::new(),
            gpu: None,
            materials,
            material: 0,
            center: None,
            uploaded: false,
            frames: 0,
            streams: 0,
            rebuilds: 0,
            uploads: 0,
            screenshots: 0,
            actions_ok: 0,
            actions_rejected: 0,
            autoplay_break: false,
            net: None,
            session: None,
            probe_sent: false,
            batches: 0,
            requests_sent: 0,
            requests_rejected: 0,
            walk: false,
            tick_accumulator: 0.0,
            walk_ticks: 0,
            walk_blocked: false,
            last_result: String::new(),
            adapter: String::new(),
            started: None,
            last_frame: None,
            fps_origin: Instant::now(),
            fps_frames: 0,
            pending_shot: None,
            summary: None,
            failure: None,
        }
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        // First frame assumes 60 fps; afterwards integrate the real interval,
        // clamped so a pause (breakpoint, window drag) cannot teleport the camera.
        let step = self
            .last_frame
            .map_or(1.0 / 60.0, |last| now.duration_since(last).as_secs_f32())
            .clamp(0.0001, 0.25);
        self.last_frame = Some(now);

        // Drain the connection first: the baseline creates the scene and
        // increments refresh it before this frame's meshes are captured.
        let events = match self.net.as_mut().map(crate::net::Connection::poll) {
            Some(Ok(events)) => events,
            Some(Err(error)) => {
                self.failure = Some(format!("connection failed: {error}"));
                event_loop.exit();
                return;
            }
            None => Vec::new(),
        };
        for event in events {
            if let Err(error) = self.apply_net_event(event, event_loop) {
                self.failure = Some(error);
                event_loop.exit();
                return;
            }
        }

        let Some(camera) = self.camera.as_mut() else {
            return;
        };
        if self.scene.is_none() {
            // Connected but no baseline yet: keep presenting cleared frames
            // so polling continues and the window does not sit on an
            // uninitialized surface.
            if let Some(gpu) = self.gpu.as_mut() {
                match gpu.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(frame)
                    | CurrentSurfaceTexture::Suboptimal(frame) => {
                        let view = frame.texture.create_view(&TextureViewDescriptor::default());
                        let (device, queue) = gpu.context.parts();
                        let mut encoder =
                            device.create_command_encoder(&CommandEncoderDescriptor {
                                label: Some("client idle frame"),
                            });
                        gpu.context
                            .encode_draw(&mut encoder, &view, &gpu.depth_view);
                        gpu.ui.prepare(
                            device,
                            queue,
                            [gpu.config.width, gpu.config.height],
                            &self.ui,
                            crate::ui::Hud {
                                material: self.material,
                                network: self.net.is_some(),
                                ready: false,
                                result: &self.last_result,
                            },
                        );
                        gpu.ui.draw(&mut encoder, &view);
                        queue.submit([encoder.finish()]);
                        queue.present(frame);
                    }
                    _ => {}
                }
                gpu.window.request_redraw();
            }
            return;
        }
        let (mut motion, speed) = self.input.take_input(camera.speed);
        camera.speed = speed;
        if self.config.autoplay && !self.ui.modal() {
            motion.forward = 1.0;
        }
        if self.ui.modal() {
            self.tick_accumulator = 0.0;
        } else if self.walk {
            // Walk mode: look still follows the drag, but the position comes
            // from the authoritative walker entity driven at a fixed tick.
            camera.look(motion.yaw_delta, motion.pitch_delta);
            self.tick_accumulator = (self.tick_accumulator + step).min(0.4);
            const TICK: f32 = 1.0 / 20.0;
            let mut ticks = 0;
            while self.tick_accumulator >= TICK && ticks < 8 {
                self.tick_accumulator -= TICK;
                ticks += 1;
                self.walk_ticks += 1;
                let yaw = camera.yaw as f64;
                let forward = [-yaw.sin(), 0.0, -yaw.cos()];
                let right = [yaw.cos(), 0.0, -yaw.sin()];
                let mut wish = [0.0f64; 3];
                for axis in 0..3 {
                    wish[axis] = forward[axis] * f64::from(motion.forward)
                        + right[axis] * f64::from(motion.strafe);
                }
                let length = (wish[0] * wish[0] + wish[1] * wish[1] + wish[2] * wish[2]).sqrt();
                if length > 0.0 {
                    for value in &mut wish {
                        *value /= length;
                    }
                }
                // Autoplay jumps when the previous tick ran into a step.
                let jump = self.input.jump_held() || (self.config.autoplay && self.walk_blocked);
                // camera.eye still holds the previous walker position here.
                let before = (f64::from(camera.eye[0]), f64::from(camera.eye[2]));
                let Some(scene) = self.scene.as_mut() else {
                    return;
                };
                match scene.step_walker(wish, jump, TICK) {
                    Ok((position, _)) => {
                        let wanted = wish[0].abs() + wish[2].abs() > 1e-9;
                        let drifted =
                            (position.x - before.0).abs() + (position.z - before.1).abs() > 0.02;
                        self.walk_blocked = wanted && !drifted;
                        camera.eye = [
                            position.x as f32,
                            (position.y + WALKER_EYE_OFFSET) as f32,
                            position.z as f32,
                        ];
                    }
                    Err(reason) => {
                        self.failure = Some(reason);
                        event_loop.exit();
                        return;
                    }
                }
            }
        } else {
            camera.advance(motion, step);
        }

        let center = self
            .scene
            .as_ref()
            .expect("camera implies scene")
            .center_section(camera.eye);
        if self.center != Some(center) {
            self.center = Some(center);
            // Network scenes keep the subscribed sections; streaming would
            // fabricate terrain the server does not have.
            if !self
                .scene
                .as_ref()
                .expect("camera implies scene")
                .is_network()
            {
                let Some(scene) = self.scene.as_mut() else {
                    return;
                };
                if let Err(error) = scene.load_around(center) {
                    self.failure = Some(format!("streaming failed: {error}"));
                    event_loop.exit();
                    return;
                }
                self.streams += 1;
            }
        }

        // Resolve the aim from the fresh camera, translate clicks and the
        // autoplay demo into place or break intents, and commit them before
        // this frame's meshes are captured.
        let aim = {
            let Some(scene) = self.scene.as_ref() else {
                return;
            };
            let Some(camera) = self.camera.as_ref() else {
                return;
            };
            scene.pick(camera.eye, camera.forward(), PICK_RANGE)
        };
        let mut intent = self
            .input
            .take_click()
            .and_then(|button| action_for(&aim, button));
        // The walk demo is about walking; firing place or break at the aim
        // while walking would block the walker's own path.
        let network = self.scene.as_ref().is_some_and(Scene::is_network);
        if network
            && self.config.probe_outside
            && !self.probe_sent
            && self.frames > 12
            && let Some(net) = self.net.as_mut()
        {
            // One deliberate out-of-scope target; the server should reject
            // it and neither world nor mirror changes.
            self.probe_sent = true;
            let call = crate::net::edit_call(
                false,
                &self.materials[self.material],
                BlockPos::new(200, 1, 1),
            );
            match net.request(call) {
                Ok(seq) => {
                    self.requests_sent += 1;
                    self.last_result = format!("outside probe seq {} sent", seq.0);
                }
                Err(error) => self.last_result = format!("probe failed: {error}"),
            }
        }
        if network
            && self.config.autoplay
            && !self.ui.modal()
            && self.frames.is_multiple_of(AUTOPLAY_PERIOD)
        {
            // Over a connection the demo exercises the request path: the
            // guest session should see its edit rejected by admission.
            let aim = {
                let Some(scene) = self.scene.as_ref() else {
                    return;
                };
                let Some(camera) = self.camera.as_ref() else {
                    return;
                };
                scene.pick(camera.eye, camera.forward(), PICK_RANGE)
            };
            if let Some((place, target)) = action_for(&aim, MouseButton::Left) {
                let call = crate::net::edit_call(place, &self.materials[self.material], target);
                if let Some(Ok(seq)) = self.net.as_mut().map(|net| net.request(call)) {
                    self.requests_sent += 1;
                    self.last_result = format!("seq {} sent", seq.0);
                }
            }
        }
        if intent.is_none()
            && self.config.autoplay
            && !self.ui.modal()
            && !self.walk
            && !network
            && self.frames.is_multiple_of(AUTOPLAY_PERIOD)
        {
            let button = if self.autoplay_break {
                MouseButton::Left
            } else {
                MouseButton::Right
            };
            intent = action_for(&aim, button);
            self.autoplay_break = !self.autoplay_break;
        }
        if let Some((place, target)) = intent {
            if self.scene.as_ref().is_some_and(Scene::is_network) {
                // Edits go to the server, not the local mirror. The guest
                // session has no edit grants, so the admission reply carries
                // the rejection; the mirror never forks.
                let call = crate::net::edit_call(place, &self.materials[self.material], target);
                match self.net.as_mut().map(|net| net.request(call)) {
                    Some(Ok(seq)) => self.last_result = format!("seq {} sent", seq.0),
                    Some(Err(error)) => self.last_result = format!("send failed: {error}"),
                    None => self.last_result = "offline".to_owned(),
                }
            } else {
                let material = self.materials[self.material].clone();
                let result = match self.scene.as_mut() {
                    Some(scene) => scene.apply_edit(place, &material, target),
                    None => return,
                };
                match result {
                    Ok(()) => {
                        self.actions_ok += 1;
                        self.last_result.clear();
                    }
                    Err(reason) => {
                        self.actions_rejected += 1;
                        self.last_result = reason;
                    }
                }
            }
        }

        // Stage the selection outline from the post-edit aim so it matches
        // what the next frame will show.
        let outline = {
            let Some(scene) = self.scene.as_ref() else {
                return;
            };
            let Some(camera) = self.camera.as_ref() else {
                return;
            };
            match scene.pick(camera.eye, camera.forward(), PICK_RANGE) {
                PickOutcome::Hit(hit) => outline_bytes(&hit),
                PickOutcome::Miss => Vec::new(),
            }
        };
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.context.write_lines(&outline);
        }

        let rebuilt = self
            .scene
            .as_mut()
            .expect("camera implies scene")
            .refresh_meshes();
        self.rebuilds += rebuilt;
        if rebuilt > 0 || !self.uploaded {
            let (vertices, indices, _) = self
                .scene
                .as_ref()
                .expect("camera implies scene")
                .geometry();
            let bytes = Scene::vertex_bytes(&vertices);
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.context.upload(&bytes, &indices);
            }
            self.uploaded = true;
            self.uploads += 1;
        }

        if self.input.take_screenshot() {
            self.pending_shot = Some(format!("torse-shot-{}.png", self.screenshots + 1));
        }

        let frame = match self.gpu.as_mut() {
            Some(gpu) => match gpu.surface.get_current_texture() {
                CurrentSurfaceTexture::Success(frame)
                | CurrentSurfaceTexture::Suboptimal(frame) => frame,
                CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                    gpu.window.request_redraw();
                    return;
                }
                CurrentSurfaceTexture::Outdated => {
                    let (device, _) = gpu.context.parts();
                    gpu.surface.configure(device, &gpu.config);
                    gpu.window.request_redraw();
                    return;
                }
                CurrentSurfaceTexture::Lost | CurrentSurfaceTexture::Validation => {
                    self.failure = Some("surface lost or failed validation".to_owned());
                    event_loop.exit();
                    return;
                }
            },
            None => return,
        };

        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
        let (device, queue) = gpu.context.parts();
        gpu.context.write_camera(
            self.camera
                .as_ref()
                .expect("gpu implies camera")
                .view_projection(aspect),
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("client window frame"),
        });
        gpu.context
            .encode_draw(&mut encoder, &view, &gpu.depth_view);
        gpu.ui.prepare(
            device,
            queue,
            [gpu.config.width, gpu.config.height],
            &self.ui,
            crate::ui::Hud {
                material: self.material,
                network: self.net.is_some(),
                ready: self.scene.is_some(),
                result: &self.last_result,
            },
        );
        gpu.ui.draw(&mut encoder, &view);
        queue.submit([encoder.finish()]);
        if let Some(path) = self.pending_shot.take() {
            // The copy must be queued after the draw pass, otherwise it reads
            // the presented texture before this frame has been drawn into it.
            match capture_frame(
                device,
                queue,
                &frame,
                gpu.config.width,
                gpu.config.height,
                gpu.config.format,
                &path,
            ) {
                Ok(()) => self.screenshots += 1,
                Err(error) => {
                    self.failure = Some(error);
                }
            }
        }
        queue.present(frame);

        self.frames += 1;
        self.fps_frames += 1;
        let title_span = self.fps_origin.elapsed().as_secs_f32();
        if title_span >= 0.5 {
            let fps = self.fps_frames as f32 / title_span;
            if let Some(scene) = self.scene.as_ref() {
                let material = self.materials[self.material].path();
                let status = if self.last_result.is_empty() {
                    "ok".to_owned()
                } else {
                    self.last_result.clone()
                };
                gpu.window.set_title(&format!(
                    "Torse client | {fps:.0} fps | sections {} | meshes {} | {material} | {status}",
                    scene.sections().len(),
                    scene.mesh_count(),
                ));
            }
            self.fps_origin = Instant::now();
            self.fps_frames = 0;
        }

        if let Some(limit) = self.config.frames
            && self.frames >= limit
        {
            if let Some(path) = self.config.screenshot.clone() {
                self.pending_shot = Some(path);
                // Present once more so the screenshot captures a real frame,
                // then stop on the following one.
                if self.frames == limit {
                    gpu.window.request_redraw();
                    return;
                }
            }
            self.finish(event_loop);
            return;
        }
        gpu.window.request_redraw();
    }

    /// Apply one decoded server event; errors stop the run.
    fn apply_net_event(
        &mut self,
        event: crate::net::NetEvent,
        _event_loop: &ActiveEventLoop,
    ) -> Result<(), String> {
        use crate::net::NetEvent;
        match event {
            NetEvent::Welcomed { session } => {
                self.session = Some(session.0);
            }
            NetEvent::Baseline(snapshot) => {
                // The mirror installs the y = 0 section layer; the eye must
                // stay inside it or the pick ray stops at the first unloaded
                // cell above the terrain.
                let loaded_top = (snapshot.side as i64).min(snapshot.bounds.max_y);
                let scene = Scene::from_snapshot(&snapshot)?;
                if let Some(camera) = self.camera.as_mut() {
                    camera.eye = [8.0, (loaded_top as f32 - 2.0).max(4.0), 8.0];
                    // Look nearly straight down: the scope is 32 by 16 blocks,
                    // so a shallow view drifts out of the loaded area before
                    // the ray reaches the floor.
                    camera.pitch = -1.5;
                    camera.yaw = 0.6;
                }
                self.scene = Some(scene);
                self.center = None;
                self.uploaded = false;
            }
            NetEvent::Increment(changes) => {
                self.batches += 1;
                let Some(scene) = self.scene.as_mut() else {
                    return Err("increment before baseline".to_owned());
                };
                scene.apply_changes(&changes)?;
            }
            NetEvent::Admitted { seq, rejected } => {
                if rejected.is_some() {
                    self.requests_rejected += 1;
                }
                self.last_result = match rejected {
                    Some(code) => format!("seq {} rejected {code:?}", seq.0),
                    None => format!("seq {} admitted", seq.0),
                };
            }
            NetEvent::Finished { seq, committed } => {
                if !committed {
                    self.requests_rejected += 1;
                }
                self.last_result = format!(
                    "seq {} {}",
                    seq.0,
                    if committed { "committed" } else { "denied" }
                );
            }
            NetEvent::Closed(reason) => return Err(reason),
        }
        Ok(())
    }

    /// Switch to walk mode by spawning the player entity under the camera.
    ///
    /// `Scene::eye` sits 1.4 blocks above the surface cell's top, and the
    /// walker's centre must stand 0.9 above that top for the collider to
    /// clear the ground, hence the 0.5 drop. Movement preparation rejects any
    /// entity that already overlaps terrain, so a low spawn point freezes the
    /// walker until it is toggled again from higher up.
    fn enter_walk(&mut self, eye: [f32; 3]) {
        let under = Position::new(
            f64::from(eye[0]),
            f64::from(eye[1]) - 0.5,
            f64::from(eye[2]),
        );
        match self.scene.as_mut() {
            Some(scene) => {
                if let Err(reason) = scene.spawn_walker(under) {
                    self.last_result = reason;
                    return;
                }
            }
            None => return,
        }
        self.walk = true;
        self.tick_accumulator = 0.0;
    }

    /// F5 flips between fly and walk; the walker entity stays in the world.
    fn toggle_walk(&mut self) {
        if self.walk {
            self.walk = false;
            return;
        }
        if let Some(camera) = self.camera.as_ref() {
            self.enter_walk(camera.eye);
        }
    }

    fn finish(&mut self, event_loop: &ActiveEventLoop) {
        if let Some((scene, gpu)) = self.scene.as_ref().zip(self.gpu.as_ref()) {
            let (vertices, _, faces) = scene.geometry();
            self.summary = Some(WindowSummary {
                adapter: self.adapter.clone(),
                backend: gpu.backend_name.clone(),
                format: format!("{:?}", gpu.config.format),
                frames: self.frames,
                streams: self.streams,
                rebuilds: self.rebuilds,
                uploads: self.uploads,
                screenshots: self.screenshots,
                actions_ok: self.actions_ok,
                actions_rejected: self.actions_rejected,
                session: self.session,
                batches: self.batches,
                requests_sent: self.requests_sent,
                requests_rejected: self.requests_rejected,
                baseline_sections: self
                    .scene
                    .as_ref()
                    .map_or(0, |scene| scene.sections().len()),
                walk_ticks: self.walk_ticks,
                walker: self
                    .scene
                    .as_ref()
                    .and_then(Scene::walker_position)
                    .map(|position| [position.x, position.y, position.z]),
                sections: scene.sections().len(),
                meshes: scene.mesh_count(),
                vertices: vertices.len(),
                faces,
                elapsed_ms: self
                    .started
                    .map(|start| start.elapsed().as_millis())
                    .unwrap_or_default(),
            });
        }
        event_loop.exit();
    }
}

impl ApplicationHandler for WindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = Window::default_attributes()
            .with_title("Torse client")
            .with_min_inner_size(LogicalSize::new(640, 360))
            .with_inner_size(LogicalSize::new(self.config.width, self.config.height));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.failure = Some(format!("window creation failed: {error}"));
                event_loop.exit();
                return;
            }
        };

        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let surface = match instance.create_surface(window.clone()) {
            Ok(surface) => surface,
            Err(error) => {
                self.failure = Some(format!("surface creation failed: {error}"));
                event_loop.exit();
                return;
            }
        };
        let options = RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..RequestAdapterOptions::default()
        };
        let adapter = match pollster::block_on(instance.request_adapter(&options)) {
            Ok(adapter) => adapter,
            Err(error) => {
                self.failure = Some(format!("adapter unavailable: {error}"));
                event_loop.exit();
                return;
            }
        };
        let info = adapter.get_info();
        let (device, queue) =
            match pollster::block_on(adapter.request_device(&DeviceDescriptor::default())) {
                Ok(pair) => pair,
                Err(error) => {
                    self.failure = Some(format!("device unavailable: {error}"));
                    event_loop.exit();
                    return;
                }
            };

        let size = window.inner_size();
        let mut config =
            match surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)) {
                Some(config) => config,
                None => {
                    self.failure = Some("surface has no supported configuration".to_owned());
                    event_loop.exit();
                    return;
                }
            };
        // Screenshots read back from the presented texture.
        config.usage |= TextureUsages::COPY_SRC;
        surface.configure(&device, &config);
        let depth = create_depth(&device, config.width, config.height);
        let depth_view = depth.create_view(&TextureViewDescriptor::default());
        let context = GpuContext::new(device, queue, config.format);

        let scene = if self.config.connect.is_some() {
            // Network scenes arrive with the first baseline; the window stays
            // empty until then.
            None
        } else {
            match Scene::new(self.config.radius) {
                Ok(scene) => Some(scene),
                Err(error) => {
                    self.failure = Some(error);
                    event_loop.exit();
                    return;
                }
            }
        };
        let mut camera = FlyCamera::new([8.0, 26.0, 8.0]);
        if let Some(scene) = scene.as_ref() {
            camera = FlyCamera::new(scene.eye());
            // Only the y = 0 section layer is loaded, and picking terminates at
            // the first unloaded cell. Keep the starting eye inside that layer
            // so the aim works no matter how high the spawn surface is.
            camera.eye[1] = camera.eye[1].min(14.0);
        } else {
            // Overlook the subscribed scope until the baseline lands.
            camera.pitch = -0.7;
            camera.yaw = 0.6;
        }
        if let Some(address) = self.config.connect.clone() {
            let compatibility = match Scene::compatibility() {
                Ok(value) => value,
                Err(error) => {
                    self.failure = Some(error);
                    event_loop.exit();
                    return;
                }
            };
            let mut connection: crate::net::Connection<torse_net::TcpTransport> =
                match crate::net::Connection::connect(&address) {
                    Ok(connection) => connection,
                    Err(error) => {
                        self.failure = Some(format!("connect failed: {error}"));
                        event_loop.exit();
                        return;
                    }
                };
            let handshake = connection
                .hello(compatibility)
                .and_then(|()| connection.subscribe(demo_scope().to_vec()));
            if let Err(error) = handshake {
                self.failure = Some(format!("handshake failed: {error}"));
                event_loop.exit();
                return;
            }
            self.net = Some(connection);
        }
        self.adapter = info.name;
        let ui = {
            let (device, _) = context.parts();
            match crate::ui::Layer::new(device, config.format) {
                Ok(ui) => ui,
                Err(error) => {
                    self.failure = Some(error);
                    event_loop.exit();
                    return;
                }
            }
        };
        self.gpu = Some(Gpu {
            ui,
            backend_name: format!("{:?}", info.backend),
            context,
            surface,
            config,
            depth,
            depth_view,
            window,
        });
        self.scene = scene;
        self.camera = Some(camera);
        // Enter walk mode only after the scene exists; enter_walk spawns the
        // walker through it.
        if self.config.walk {
            let eye = self.camera.as_ref().expect("camera was just installed").eye;
            self.enter_walk(eye);
        }
        self.started = Some(Instant::now());
        self.fps_origin = Instant::now();
        // Prime one frame immediately instead of waiting for the first event.
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.finish(event_loop);
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut()
                    && size.width > 0
                    && size.height > 0
                {
                    gpu.config.width = size.width;
                    gpu.config.height = size.height;
                    let (device, _) = gpu.context.parts();
                    gpu.surface.configure(device, &gpu.config);
                    gpu.depth = create_depth(device, size.width, size.height);
                    gpu.depth_view = gpu.depth.create_view(&TextureViewDescriptor::default());
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::KeyCode;
                if event.state == ElementState::Pressed
                    && let winit::keyboard::PhysicalKey::Code(code) = event.physical_key
                {
                    match code {
                        KeyCode::Escape if !event.repeat => {
                            let mode = if self.ui.modal() {
                                crate::ui::Mode::Game
                            } else {
                                crate::ui::Mode::Pause
                            };
                            self.ui.set_mode(mode);
                            self.input.reset();
                            return;
                        }
                        KeyCode::KeyE if !event.repeat => {
                            let mode = if self.ui.modal() {
                                crate::ui::Mode::Game
                            } else {
                                crate::ui::Mode::MachinePreview
                            };
                            self.ui.set_mode(mode);
                            self.input.reset();
                            return;
                        }
                        KeyCode::F12 => {
                            self.input.key(event.physical_key, event.state);
                            return;
                        }
                        _ if self.ui.modal() => return,
                        KeyCode::Digit1 => self.material = 0,
                        KeyCode::Digit2 => self.material = 1,
                        KeyCode::Digit3 => self.material = 2,
                        KeyCode::F5 => self.toggle_walk(),
                        _ => {}
                    }
                }
                if !self.ui.modal() {
                    self.input.key(event.physical_key, event.state);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if self.ui.modal() {
                    if button == MouseButton::Left {
                        let size = self
                            .gpu
                            .as_ref()
                            .map_or([960, 540], |g| [g.config.width, g.config.height]);
                        if let Some(action) = self.ui.pointer(state == ElementState::Pressed, size)
                        {
                            match action {
                                crate::ui::Action::Quit => self.finish(event_loop),
                                crate::ui::Action::Resume => {
                                    self.ui.set_mode(crate::ui::Mode::Game)
                                }
                                crate::ui::Action::Preview => {
                                    self.ui.set_mode(crate::ui::Mode::MachinePreview)
                                }
                            }
                            self.input.reset();
                        }
                    }
                } else {
                    self.input.mouse(button, state);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.ui.cursor = [position.x, position.y];
                if !self.ui.modal() {
                    self.input.cursor(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !self.ui.modal() {
                    self.input.wheel(delta);
                }
            }
            WindowEvent::Focused(false) => {
                self.input.reset();
                self.ui.set_mode(self.ui.mode);
            }
            WindowEvent::RedrawRequested => {
                if self.gpu.is_some() && self.camera.is_some() {
                    self.frame(event_loop);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window.request_redraw();
        }
    }
}

/// The demo server's host-approved scope; subscriptions must be a subset.
const fn demo_scope() -> [SectionPos; 2] {
    [SectionPos::new(0, 0, 0), SectionPos::new(1, 0, 0)]
}

/// Translate one aim and mouse button into a place or break intent.
///
/// Left breaks the hit cell; right places the selected material against the
/// entered face. A hit inside a solid cell carries no face, so placement is
/// dropped rather than guessed.
fn action_for(outcome: &PickOutcome, button: MouseButton) -> Option<(bool, BlockPos)> {
    let PickOutcome::Hit(hit) = outcome else {
        return None;
    };
    match button {
        MouseButton::Left => Some((false, hit.block)),
        MouseButton::Right => {
            let (face_x, face_y, face_z) = (hit.face[0], hit.face[1], hit.face[2]);
            if (face_x, face_y, face_z) == (0, 0, 0) {
                return None;
            }
            Some((
                true,
                BlockPos::new(
                    hit.block.x + i64::from(face_x),
                    hit.block.y + i64::from(face_y),
                    hit.block.z + i64::from(face_z),
                ),
            ))
        }
        _ => None,
    }
}

/// Build the 12 outline edges of the targeted cell, inflated outward slightly
/// so the wireframe does not z-fight with the block faces it sits on.
fn outline_bytes(hit: &PickHit) -> Vec<u8> {
    const INFLATE: f32 = 0.004;
    let min = [
        hit.block.x as f32 - INFLATE,
        hit.block.y as f32 - INFLATE,
        hit.block.z as f32 - INFLATE,
    ];
    let max = [
        hit.block.x as f32 + 1.0 + INFLATE,
        hit.block.y as f32 + 1.0 + INFLATE,
        hit.block.z as f32 + 1.0 + INFLATE,
    ];
    // Corner bits: x, y, z select the min or max end per axis.
    let corner = |bits: u32| {
        [
            if bits & 1 != 0 { max[0] } else { min[0] },
            if bits & 2 != 0 { max[1] } else { min[1] },
            if bits & 4 != 0 { max[2] } else { min[2] },
        ]
    };
    // 12 edges of the cube as corner-index pairs.
    const EDGES: [(u32, u32); 12] = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let mut vertices = Vec::with_capacity(EDGES.len() * 2);
    for (from, to) in EDGES {
        for position in [corner(from), corner(to)] {
            vertices.push(Vertex {
                position,
                normal: [0.0, 1.0, 0.0],
                color: [0.05, 0.05, 0.05],
            });
        }
    }
    Scene::vertex_bytes(&vertices)
}

fn create_depth(device: &Device, width: u32, height: u32) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("client window depth"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

/// Read one presented frame back and write it as a PNG.
fn capture_frame(
    device: &Device,
    queue: &Queue,
    frame: &SurfaceTexture,
    width: u32,
    height: u32,
    format: TextureFormat,
    path: &str,
) -> Result<(), String> {
    let bytes_per_pixel = 4;
    let padded_row = (width * bytes_per_pixel).next_multiple_of(256);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("client window readback"),
        size: u64::from(padded_row) * u64::from(height),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("client window readback"),
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &frame.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .map_err(|error| format!("device poll failed: {error}"))?;
    let view = slice
        .get_mapped_range()
        .map_err(|error| format!("map failed: {error}"))?;
    let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
    for row in 0..height {
        let start = (row * padded_row) as usize;
        let end = start + (width * bytes_per_pixel) as usize;
        pixels.extend_from_slice(&view[start..end]);
    }
    drop(view);
    readback.unmap();

    if matches!(
        format,
        TextureFormat::Bgra8UnormSrgb | TextureFormat::Bgra8Unorm
    ) {
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    } else if !matches!(
        format,
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm
    ) {
        return Err(format!("unsupported screenshot format {format:?}"));
    }
    crate::renderer::write_png(path, width, height, &pixels)
}

/// Open the interactive window and block until it closes.
///
/// Returns the run summary, or the first failure that stopped the loop.
pub fn run_window(config: WindowConfig) -> Result<WindowSummary, String> {
    let event_loop = EventLoop::new().map_err(|error| format!("event loop failed: {error}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WindowApp::new(config);
    event_loop
        .run_app(&mut app)
        .map_err(|error| format!("event loop failed: {error}"))?;
    if let Some(failure) = app.failure {
        return Err(failure);
    }
    app.summary
        .ok_or_else(|| "window closed before the first frame".to_owned())
}
