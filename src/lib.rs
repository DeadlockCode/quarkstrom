use std::{
    thread,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};

pub use gui::Gui;
use gui::GuiHandler;
use ultraviolet::Vec2;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoopBuilder},
    platform::windows::EventLoopBuilderExtWindows,
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;
use winit_input_helper::WinitInputHelper;

pub mod gui;
pub use egui;
pub use wgpu;
pub use winit;
pub use winit_input_helper;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct View {
    position: Vec2,
    scale: f32,
    aspect: f32,
}

unsafe impl Pod for View {}
unsafe impl Zeroable for View {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: Vec2,
    pub color: u32,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32];

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
    pub color: u32,
}

unsafe impl Pod for Instance {}
unsafe impl Zeroable for Instance {}

impl Instance {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Uint32];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct State<GUI: Gui> {
    window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    input: WinitInputHelper,
    vertices: Vec<Vertex>,
    vertex_buffer: wgpu::Buffer,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    line_render_pipeline: wgpu::RenderPipeline,
    circle_render_pipeline: wgpu::RenderPipeline,
    view: View,
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,

    gui: GuiHandler<GUI>,
}

impl<GUI: Gui> State<GUI> {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window, config: Config) -> Self {
        let in_config = config;

        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
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
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let gui = GuiHandler::new(&window, config.format, &device);

        let circle_shader = device.create_shader_module(wgpu::include_wgsl!("circle_shader.wgsl"));
        let line_shader = device.create_shader_module(wgpu::include_wgsl!("line_shader.wgsl"));

        let view = View {
            position: Vec2::zero(),
            scale: in_config.view_size,
            aspect: config.height as f32 / config.width as f32,
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

        let circle_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    blend: Some(wgpu::BlendState::REPLACE),
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

        let vertices = Vec::new();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 20,
            mapped_at_creation: false,
        });

        let instances = Vec::new();

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            size: 1 << 20,
            mapped_at_creation: false,
        });

        let input = WinitInputHelper::new();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            input,
            vertices,
            vertex_buffer,
            instances,
            instance_buffer,
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

            self.view.aspect = new_size.height as f32 / new_size.width as f32;
        }
    }

    fn set_vertices(&mut self, vertices: Vec<Vertex>) {
        self.vertices = vertices;
        self.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&self.vertices),
        );
    }

    fn set_instances(&mut self, instances: Vec<Instance>) {
        self.instances = instances;
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instances),
        );
    }

    fn input(&mut self, event: &Event<()>) {
        // If gui doesn't want exclusive access and it's time to update
        if !self.gui.handle_event(event) && self.input.update(event) {
            self.gui.gui.input(&self.input);
            // Zoom
            if let Some((mx, my)) = self.input.mouse() {
                // Scroll steps to double/halve the scale
                let steps = 5.0;

                // Modify input
                let zoom = (-self.input.scroll_diff() / steps).exp2();

                // Screen space -> view space
                let target = Vec2::new(
                    mx * 2.0 - self.size.width as f32,
                    self.size.height as f32 - my * 2.0,
                ) / self.size.height as f32;

                // Move view position based on target
                self.view.position += target * self.view.scale * (1.0 - zoom);

                // Zoom
                self.view.scale *= zoom;
            }

            // Grab
            if self.input.mouse_held(2) {
                let (mdx, mdy) = self.input.mouse_diff();
                self.view.position.x -= mdx / self.size.height as f32 * self.view.scale * 2.0;
                self.view.position.y += mdy / self.size.height as f32 * self.view.scale * 2.0;
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
                .render(&self.device, &self.queue, &self.window, &mut encoder);

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

            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.vertices.len() as u32, 0..1);

            render_pass.set_pipeline(&self.circle_render_pipeline);
            render_pass.set_bind_group(0, &self.view_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            render_pass.draw(0..3, 0..self.instances.len() as u32);

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

pub trait Simulation<GUI>
where
    GUI: Gui,
{
    fn new() -> Self;
    fn update(&mut self) {}
    fn convert(&mut self) {}
}

#[derive(Clone, Copy)]
pub struct Config {
    pub tps_cap: Option<u32>,
    pub view_size: f32,
    pub window_size: PhysicalSize<u32>,
}

pub fn run<SIM, GUI>(config: Config)
where
    SIM: Simulation<GUI>,
    GUI: Gui + 'static,
{
    env_logger::init();

    thread::spawn(move || {
        let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
        let window = WindowBuilder::new()
            .with_title("Quarkstrom")
            .with_inner_size(config.window_size)
            .build(&event_loop)
            .unwrap();

        let mut state = pollster::block_on(State::<GUI>::new(window, config));

        event_loop.run(move |event, _, control_flow| {
            state.input(&event);

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
                    let (instances, vertices) = state.gui.gui.render();
                    if let Some(i) = instances {
                        state.set_instances(i);
                    }
                    if let Some(v) = vertices {
                        state.set_vertices(v);
                    }

                    match state.render() {
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

    let desired_frame_time = config
        .tps_cap
        .map(|tps| Duration::from_secs_f64(1.0 / tps as f64));

    // Init
    let mut simulation = SIM::new();

    loop {
        let frame_timer = Instant::now();

        simulation.update();
        simulation.convert();

        // Cap tps
        if let Some(desired_frame_time) = desired_frame_time {
            // Could've been the following line of code,
            // "thread::sleep(desired_frame_time - frame_timer.elapsed());"
            // but sleeping on windows (at least) is extremely unpredictable.
            // Therefore we have to waste resources.  
            while frame_timer.elapsed() < desired_frame_time {}
        }
    }
}
