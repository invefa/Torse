//! wgpu rendering for the client.
//!
//! [`GpuContext`] owns the device-facing half: pipeline, camera uniform, and
//! terrain buffers. Both targets reuse it: [`Renderer`] draws into an offscreen
//! texture so a frame can be captured without a window, and the windowed app
//! draws into a surface view. The pass is identical, only the target differs.

use std::collections::BTreeMap;
use std::fs::File;

use wgpu::{
    BindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BlendState, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoder, CommandEncoderDescriptor, Device, DeviceDescriptor, Extent3d,
    FragmentState, Instance, InstanceDescriptor, LoadOp, MapMode, MultisampleState, Operations,
    Origin3d, PollType, PrimitiveState, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StoreOp, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo,
    Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor, VertexState,
};

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = camera.view_proj * vec4<f32>(input.position, 1.0);
    out.normal = input.normal;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.35, 0.9, 0.25));
    let diffuse = max(dot(normalize(input.normal), light), 0.0);
    let shade = 0.38 + 0.62 * diffuse;
    return vec4<f32>(input.color * shade, 1.0);
}
"#;

/// Sky colour shared by both targets.
pub const SKY: Color = Color {
    r: 0.66,
    g: 0.74,
    b: 0.80,
    a: 1.0,
};

/// Selection outline capacity: 12 cube edges of 2 vertices each.
const MAX_LINE_VERTICES: usize = 24;

/// Device, queue, pipeline, camera uniform, and terrain buffers.
pub struct GpuContext {
    device: Device,
    queue: Queue,
    pipeline: RenderPipeline,
    line_pipeline: RenderPipeline,
    uniform: Buffer,
    bind_group: BindGroup,
    #[allow(dead_code)]
    layout: BindGroupLayout,
    /// One vertex/index buffer pair per resident section; edits rewrite only
    /// the changed sections instead of re-uploading the whole scene.
    terrain: BTreeMap<torse_types::SectionPos, SectionBuffers>,
    line_buffer: Buffer,
    line_count: u32,
}

/// GPU buffers for one section's mesh.
struct SectionBuffers {
    vertices: Buffer,
    indices: Buffer,
    index_count: u32,
}

impl GpuContext {
    /// Build the pipeline and camera binding on an already connected device.
    ///
    /// `color_format` is the format of the views the pass will draw into; the
    /// offscreen target picks it itself while a surface inherits it from the
    /// adapter.
    pub fn new(device: Device, queue: Queue, color_format: TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("client camera layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("client camera uniform"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("client camera bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("client terrain shader"),
            source: ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("client pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 2,
            },
        ];
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("client terrain pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 36,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        // The outline reuses the terrain shader with a line topology; it only
        // reads depth, so the outline never occludes the terrain it marks.
        let line_descriptor = RenderPipelineDescriptor {
            label: Some("client outline pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 36,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: color_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..PrimitiveState::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        };
        let line_pipeline = device.create_render_pipeline(&line_descriptor);
        let line_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("client outline vertices"),
            size: (MAX_LINE_VERTICES * 36) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            line_pipeline,
            uniform,
            bind_group,
            layout,
            terrain: BTreeMap::new(),
            line_buffer,
            line_count: 0,
        }
    }

    /// Connected device and queue.
    pub fn parts(&self) -> (&Device, &Queue) {
        (&self.device, &self.queue)
    }

    /// Replace one section's terrain buffers, keeping other sections as-is.
    pub fn upload_section(
        &mut self,
        section: torse_types::SectionPos,
        vertex_bytes: &[u8],
        indices: &[u32],
    ) {
        use wgpu::util::DeviceExt;
        let buffers = SectionBuffers {
            vertices: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("client terrain vertices"),
                    contents: vertex_bytes,
                    usage: BufferUsages::VERTEX,
                }),
            indices: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("client terrain indices"),
                    contents: &by_index_bytes(indices),
                    usage: BufferUsages::INDEX,
                }),
            index_count: indices.len() as u32,
        };
        self.terrain.insert(section, buffers);
    }

    /// Drop the buffers of unloaded sections.
    pub fn retain_sections(&mut self, keep: &[torse_types::SectionPos]) {
        self.terrain.retain(|pos, _| keep.contains(pos));
    }

    /// Stage the camera matrix for the next frame.
    pub fn write_camera(&self, view_projection: [f32; 16]) {
        self.queue
            .write_buffer(&self.uniform, 0, &matrix_bytes(view_projection));
    }

    /// Replace the selection outline. `vertex_bytes` uses the terrain layout
    /// and is truncated to the outline capacity.
    pub fn write_lines(&mut self, vertex_bytes: &[u8]) {
        let stride = 36;
        let limit = vertex_bytes.len().min(MAX_LINE_VERTICES * stride);
        self.queue
            .write_buffer(&self.line_buffer, 0, &vertex_bytes[..limit]);
        self.line_count = (limit / stride) as u32;
    }

    /// Encode the one terrain pass onto any colour and depth view pair.
    pub fn encode_draw(
        &self,
        encoder: &mut CommandEncoder,
        color: &TextureView,
        depth: &TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("client terrain pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(SKY),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for section in self.terrain.values() {
            pass.set_vertex_buffer(0, section.vertices.slice(..));
            pass.set_index_buffer(section.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..section.index_count, 0, 0..1);
        }
        if self.line_count > 0 {
            pass.set_pipeline(&self.line_pipeline);
            pass.set_vertex_buffer(0, self.line_buffer.slice(..));
            pass.draw(0..self.line_count, 0..1);
        }
    }
}

/// Offscreen renderer.
pub struct Renderer {
    context: GpuContext,
    color: Texture,
    color_view: TextureView,
    depth_view: TextureView,
    width: u32,
    height: u32,
}

impl Renderer {
    /// Create a headless renderer.
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default()))
                .map_err(|error| format!("adapter unavailable: {error}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&DeviceDescriptor::default()))
                .map_err(|error| format!("device unavailable: {error}"))?;

        let color = device.create_texture(&TextureDescriptor {
            label: Some("client color target"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&TextureViewDescriptor::default());
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("client depth target"),
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
        });
        let depth_view = depth.create_view(&TextureViewDescriptor::default());

        let context = GpuContext::new(device, queue, TextureFormat::Rgba8UnormSrgb);
        Ok(Self {
            context,
            color,
            color_view,
            depth_view,
            width,
            height,
        })
    }

    /// Replace one section's terrain buffers.
    pub fn upload_section(
        &mut self,
        section: torse_types::SectionPos,
        vertex_bytes: &[u8],
        indices: &[u32],
    ) {
        self.context.upload_section(section, vertex_bytes, indices);
    }

    /// Render one frame into the offscreen target.
    pub fn render(&mut self, view_projection: [f32; 16]) {
        self.context.write_camera(view_projection);
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("client frame"),
            });
        self.context
            .encode_draw(&mut encoder, &self.color_view, &self.depth_view);
        self.context.queue.submit([encoder.finish()]);
    }

    /// Draw the same UI used by the window into an offscreen capture.
    pub fn render_ui(&self, mode: crate::ui::Mode) -> Result<(), String> {
        let (device, queue) = self.context.parts();
        let mut layer = crate::ui::Layer::new(device, TextureFormat::Rgba8UnormSrgb)?;
        let mut state = crate::ui::State::default();
        state.set_mode(mode);
        layer.prepare(
            device,
            queue,
            [self.width, self.height],
            &state,
            crate::ui::Hud {
                material: 0,
                network: false,
                ready: true,
                result: "",
            },
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("UI capture"),
        });
        layer.draw(&mut encoder, &self.color_view);
        queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Copy the rendered frame into a PNG file.
    pub fn save_png(&self, path: &str) -> Result<(), String> {
        let padded_row = align_256(self.width * 4);
        let buffer_size = u64::from(padded_row) * u64::from(self.height);
        let readback = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("client readback"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("client readback"),
            });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.context.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.context
            .device
            .poll(PollType::wait_indefinitely())
            .map_err(|error| format!("device poll failed: {error}"))?;
        let view = slice
            .get_mapped_range()
            .map_err(|error| format!("map failed: {error}"))?;
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * padded_row) as usize;
            let end = start + (self.width * 4) as usize;
            pixels.extend_from_slice(&view[start..end]);
        }
        drop(view);
        readback.unmap();

        write_png(path, self.width, self.height, &pixels)
    }
}

/// Encode tight RGBA rows into a PNG file.
pub fn write_png(path: &str, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("create failed: {error}"))?;
    let mut writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(&mut writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder
        .write_header()
        .map_err(|error| format!("png header failed: {error}"))?;
    png_writer
        .write_image_data(pixels)
        .map_err(|error| format!("png write failed: {error}"))?;
    Ok(())
}

fn matrix_bytes(matrix: [f32; 16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for value in matrix {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn by_index_bytes(indices: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(indices.len() * 4);
    for value in indices {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

const fn align_256(value: u32) -> u32 {
    (value + 255) & !255
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_png_reports_file_errors() {
        // A path with a missing directory must surface the OS error, not panic.
        let result = write_png("no-such-dir/frame.png", 1, 1, &[0, 0, 0, 255]);
        assert!(result.is_err());
    }
}
