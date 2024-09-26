pub mod gui;
pub use egui;
pub use wgpu;
pub use winit;
pub use winit_input_helper;

use std::thread;

use bytemuck::{Pod, Zeroable};

use crate::gui::GuiHandler;
use ultraviolet::Vec2;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::*,
    event_loop::{ControlFlow, EventLoopBuilder},
    platform::windows::EventLoopBuilderExtWindows,
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;
use winit_input_helper::WinitInputHelper;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct View {
    position: Vec2,
    scale: f32,
    x: u16,
    y: u16,
}

unsafe impl Pod for View {}
unsafe impl Zeroable for View {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
    pub color: [u8; 4],
}

unsafe impl Pod for Rect {}
unsafe impl Zeroable for Rect {}

impl Rect {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Unorm8x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Rect>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: Vec2,
    pub color: [u8; 4],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Unorm8x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub position: Vec2,
    pub radius: f32,
    pub color: [u8; 4],
}

unsafe impl Pod for Instance {}
unsafe impl Zeroable for Instance {}

impl Instance {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Unorm8x4];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct State {
    window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    rects: u32,
    rect_buffer: wgpu::Buffer,
    vertices: u32,
    vertex_buffer: wgpu::Buffer,
    instances: u32,
    instance_buffer: wgpu::Buffer,
    rect_render_pipeline: wgpu::RenderPipeline,
    line_render_pipeline: wgpu::RenderPipeline,
    circle_render_pipeline: wgpu::RenderPipeline,
    view: View,
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,

    gui: GuiHandler,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::CONSERVATIVE_RASTERIZATION,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync, // Could be surface_caps.present_modes[0] but Intel Arc A770 go brrr.
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let gui = GuiHandler::new(&window, config.format, &device);

        let circle_shader = device.create_shader_module(wgpu::include_wgsl!("circle_shader.wgsl"));
        let line_shader = device.create_shader_module(wgpu::include_wgsl!("line_shader.wgsl"));
        let rect_shader = device.create_shader_module(wgpu::include_wgsl!("rect_shader.wgsl"));

        let view = View {
            position: Vec2::zero(),
            scale: 1.0,
            x: config.width as u16,
            y: config.height as u16,
        };

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Buffer"),
            contents: bytemuck::cast_slice(&[view]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let view_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("View Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Bind Group"),
            layout: &view_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        });

        let rect_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let rect_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&rect_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rect_shader,
                entry_point: "vs_main",
                buffers: &[Rect::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &rect_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                conservative: true,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let line_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&line_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let circle_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&view_bind_group_layout],
                push_constant_ranges: &[],
            });

        let circle_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&circle_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &circle_shader,
                    entry_point: "vs_main",
                    buffers: &[Instance::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &circle_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: true,
                },
                depth_stencil: None, // 1.
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let rects = 0;

        let rect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rect Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 28,
            mapped_at_creation: false,
        });

        let vertices = 0;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 28,
            mapped_at_creation: false,
        });

        let instances = 0;

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 28,
            mapped_at_creation: false,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            rects,
            rect_buffer,
            vertices,
            vertex_buffer,
            instances,
            instance_buffer,
            rect_render_pipeline,
            line_render_pipeline,
            circle_render_pipeline,
            view,
            view_buffer,
            view_bind_group,
            gui,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.view.x = new_size.width as u16;
            self.view.y = new_size.height as u16;
        }
    }

    fn set_rects(&mut self, rects: &[Rect]) {
        self.rects = rects.len() as u32;
        self.queue
            .write_buffer(&self.rect_buffer, 0, bytemuck::cast_slice(rects));
    }

    fn set_vertices(&mut self, vertices: &[Vertex]) {
        self.vertices = vertices.len() as u32;
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    fn set_instances(&mut self, instances: &[Instance]) {
        self.instances = instances.len() as u32;
        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    fn input(&mut self, event: &Event<()>) -> bool {
        // If gui doesn't want exclusive access and it's time to update
        !self.gui.handle_event(event)
    }

    fn render(&mut self, gui: &mut dyn FnMut(&egui::Context)) -> Result<(), wgpu::SurfaceError> {
        self.queue
            .write_buffer(&self.view_buffer, 0, bytemuck::cast_slice(&[self.view]));

        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let (clipped_primitives, screen_descriptor) =
            self.gui
                .render(&self.device, &self.queue, &self.window, &mut encoder, gui);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.rect_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.rect_buffer.slice(..));
            render_pass.draw(0..4, 0..self.rects);

            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.vertices, 0..1);

            render_pass.set_pipeline(&self.circle_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            render_pass.draw(0..3, 0..self.instances);

            self.gui
                .renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum WindowMode {
    Windowed(u32, u32),
    Fullscreen,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub window_mode: WindowMode,
}

pub struct RenderContext {
    pos: Vec2,
    scale: f32,
    circles: Vec<Instance>,
    lines: Vec<Vertex>,
    rects: Vec<Rect>,
}

impl RenderContext {
    fn new() -> Self {
        Self {
            pos: Vec2::zero(),
            scale: 1.0,
            circles: Vec::new(),
            lines: Vec::new(),
            rects: Vec::new(),
        }
    }

    pub fn set_view_pos(&mut self, pos: Vec2) {
        self.pos = pos;
    }

    pub fn set_view_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    pub fn clear_rects(&mut self) {
        self.rects.clear();
    }

    pub fn clear_lines(&mut self) {
        self.lines.clear();
    }

    pub fn clear_circles(&mut self) {
        self.circles.clear();
    }

    pub fn draw_circle(&mut self, position: Vec2, radius: f32, color: [u8; 4]) {
        self.circles.push(Instance {
            position,
            radius,
            color,
        });
    }

    pub fn draw_line(&mut self, src: Vec2, dst: Vec2, color: [u8; 4]) {
        self.lines.push(Vertex { pos: src, color });
        self.lines.push(Vertex { pos: dst, color });
    }

    pub fn draw_rect(&mut self, min: Vec2, max: Vec2, color: [u8; 4]) {
        self.rects.push(Rect { min, max, color });
    }
}

pub trait Renderer {
    fn new() -> Self;
    fn input(&mut self, input: &WinitInputHelper, width: u16, height: u16);
    fn render(&mut self, ctx: &mut RenderContext);
    fn gui(&mut self, ctx: &egui::Context);
}

pub fn run<R>(config: Config)
where
    R: Renderer + 'static,
{
    thread::spawn(move || {
        let event_loop = EventLoopBuilder::new().with_any_thread(true).build();

        let mut builder = WindowBuilder::new().with_title("Quarkstrom");

        match config.window_mode {
            WindowMode::Windowed(width, height) => {
                let monitor = event_loop.primary_monitor().unwrap();
                let size = monitor.size();
                let position = PhysicalPosition::new(
                    (size.width - width) as i32 / 2,
                    (size.height - height) as i32 / 2,
                );
                builder = builder
                    .with_inner_size(PhysicalSize::new(width, height))
                    .with_position(position);
            }
            WindowMode::Fullscreen => {
                builder =
                    builder.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
        }

        let window = builder.build(&event_loop).unwrap();

        let mut state = pollster::block_on(State::new(window));
        let mut input = WinitInputHelper::new();
        let mut renderer = R::new();
        let mut render_ctx = RenderContext::new();

        event_loop.run(move |event, _, control_flow| {
            if state.input(&event) {
                input.update(&event);
            }

            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window().id() => match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                },
                Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                    renderer.input(&input, state.view.x, state.view.y);
                    renderer.render(&mut render_ctx);
                    state.view.position = render_ctx.pos;
                    state.view.scale = render_ctx.scale;
                    state.set_instances(&render_ctx.circles);
                    state.set_vertices(&render_ctx.lines);
                    state.set_rects(&render_ctx.rects);

                    match state.render(&mut |ctx| renderer.gui(ctx)) {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                Event::MainEventsCleared => {
                    // RedrawRequested will only trigger once, unless we manually
                    // request it.
                    state.window().request_redraw();
                }
                _ => {}
            }
        });
    });
}
